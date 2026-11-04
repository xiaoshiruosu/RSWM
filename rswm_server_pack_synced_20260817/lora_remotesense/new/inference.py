#!/usr/bin/env python3
"""
LingBot-World 遥感 LoRA 推理测试脚本

使用微调后的 LoRA checkpoint 生成视频，验证训练效果。
支持单张图片输入 + 可选相机轨迹。
"""

import argparse
import logging
import os
import sys
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from peft import PeftModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS
from wan.modules.model import WanModel
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.animate.animate_utils import get_loraconfig
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from wan.utils.utils import save_video


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_lora_model(
    checkpoint_dir: str,
    lora_path: str,
    device: torch.device,
    config,
    t5_cpu: bool = True,
    inactive_model_cpu: bool = True,
):
    """加载基础模型并注入 LoRA 权重

    80GB A100 显存预算（默认 inactive_model_cpu=True, t5_cpu=True）:
      - 1 份 14B DiT (bf16) on GPU      ~28 GB
      - 另一份在 CPU                     ~0
      - VAE                              ~0.3 GB
      - T5 推理时短暂上 GPU             ~10 GB (transient)
      - 采样 activations + buffers       ~5-10 GB
      合计运行时峰值                     ~43 GB
    """

    # T5
    logger.info("加载 T5 编码器...")
    t5_device = torch.device("cpu") if t5_cpu else device
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=t5_device,
        checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
    )
    text_encoder.model.eval().requires_grad_(False)
    logger.info("T5 编码器加载完成" + ("（t5_cpu 模式）" if t5_cpu else ""))

    # VAE
    logger.info("加载 VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        device=device,
    )
    logger.info("VAE 加载完成")

    # DiT low_noise + high_noise
    logger.info("加载 DiT 模型...")
    low_noise_model = WanModel.from_pretrained(
        checkpoint_dir,
        subfolder=config.low_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    high_noise_model = WanModel.from_pretrained(
        checkpoint_dir,
        subfolder=config.high_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )

    # 注入 LoRA
    lora_dir = Path(lora_path)
    low_lora = lora_dir / "low_noise"
    high_lora = lora_dir / "high_noise"

    # [v2.7.1 F2/P0 修复] 旧行为:目录缺失只 warning、继续用【基模型】——
    # "LoRA 结果"可能整段是基模型,甚至一个专家有 LoRA 另一个没有的混合体,
    # 对比实验直接失效。evaluate.py / generate_samples.py 都走本函数,一处
    # 封死同时保住评测与出样两条路。基模型推理请走 evaluate_baseline.py。
    if not low_lora.exists() or not high_lora.exists():
        raise FileNotFoundError(
            f"LoRA 权重缺失: low_noise={low_lora} (exists={low_lora.exists()}), "
            f"high_noise={high_lora} (exists={high_lora.exists()}) —— 两个专家"
            f"目录必须同时存在(--lora_path 应指向含 low_noise/ 与 high_noise/ "
            f"的 checkpoint 目录)。拒绝静默退化为基模型。")

    logger.info(f"注入 LoRA (low_noise): {low_lora}")
    low_noise_model = PeftModel.from_pretrained(low_noise_model, str(low_lora))
    low_noise_model = low_noise_model.merge_and_unload()

    logger.info(f"注入 LoRA (high_noise): {high_lora}")
    high_noise_model = PeftModel.from_pretrained(high_noise_model, str(high_lora))
    high_noise_model = high_noise_model.merge_and_unload()

    low_noise_model.eval().requires_grad_(False)
    high_noise_model.eval().requires_grad_(False)
    if inactive_model_cpu:
        low_noise_model.cpu()
        high_noise_model.cpu()
        logger.info("DiT 模型加载完成（LoRA 已合并；两份均驻 CPU，采样时按 timestep 切换）")
    else:
        low_noise_model.to(device)
        high_noise_model.to(device)
        logger.info("DiT 模型加载完成（LoRA 已合并；两份均驻 GPU）")

    return text_encoder, vae, low_noise_model, high_noise_model


def _resize_decoded_video(
    video: torch.Tensor,
    target_size=(480, 832),
) -> torch.Tensor:
    """Resize a decoded ``(C, T, H, W)`` video without changing its layout."""
    if video.ndim != 4:
        raise ValueError(
            f"decoded video must have shape (C,T,H,W), got {tuple(video.shape)}")
    if tuple(video.shape[-2:]) == tuple(target_size):
        return video

    original_dtype = video.dtype
    # Wan VAE latent stride alignment causes 464 height for the nominal
    # 480x832 contract.  Treat T as the interpolation batch and restore CTHW.
    resized = torch.nn.functional.interpolate(
        video.permute(1, 0, 2, 3),
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).permute(1, 0, 2, 3).contiguous()
    if resized.dtype != original_dtype:
        resized = resized.to(dtype=original_dtype)
    return resized


def generate_video(
    text_encoder,
    vae,
    low_noise_model,
    high_noise_model,
    config,
    prompt: str,
    image: Image.Image,
    device: torch.device,
    frame_num: int = 33,
    size: str = "480*832",
    action_path: str = None,
    seed: int = 42,
    sampling_steps: int = 70,
    shift: float = 10.0,
    guide_scale=(5.0, 5.0),
    t5_cpu: bool = True,
    framewise: bool = True,
    inactive_model_cpu: bool = True,
):
    """使用 LoRA 模型生成视频

    inactive_model_cpu=True (80GB 默认): 采样循环按 timestep 在 low/high 之间切换，
    仅 1 份 14B 驻 GPU。boundary 在去噪序列里只跨越 1 次，所以整次采样最多 swap 1 次，
    几乎零开销。
    """

    boundary = config.boundary * config.num_train_timesteps
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    neg_prompt = config.sample_neg_prompt

    # guide_scale 兼容 int/float/tuple
    if isinstance(guide_scale, (int, float)):
        guide_scale = (float(guide_scale), float(guide_scale))

    # 相机轨迹
    c2ws = None
    if action_path is not None:
        c2ws = np.load(os.path.join(action_path, "poses.npy"))
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

    # 预处理图像
    img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(device)
    F_total = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    max_area = MAX_AREA_CONFIGS.get(size, 480 * 832)

    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2]
    )
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    lat_f = (F_total - 1) // vae_stride[0] + 1
    max_seq_len = lat_f * lat_h * lat_w // (patch_size[1] * patch_size[2])

    # 随机噪声
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16, lat_f, lat_h, lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=device,
    )

    # 掩码
    msk = torch.ones(1, F_total, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    # 文本编码
    logger.info("编码文本...")
    if not t5_cpu:
        text_encoder.model.to(device)
        context = text_encoder([prompt], device)
        context_null = text_encoder([neg_prompt], device)
    else:
        context = text_encoder([prompt], torch.device("cpu"))
        context_null = text_encoder([neg_prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]
        context_null = [t.to(device) for t in context_null]

    # 相机条件
    # [v2.8 ④] pose_mode=drop 时整条位姿通路不传(真正 no-pose/text-only);
    # identity 由数据侧(中和 H5)决定,推理这里与 native 同路 —— 区别在于
    # 传进来的 poses.npy 内容,不在代码分支。
    dit_cond_dict = None
    _pose_mode = os.environ.get("RSWM_POSE_MODE", "native")
    if _pose_mode not in ("native", "identity", "drop"):
        raise ValueError(f"RSWM_POSE_MODE 非法: {_pose_mode!r}")
    if _pose_mode == "drop":
        # [v2.8.2 N6] **顺序敏感**:frame_num 的 4n+1 钳制必须发生在这之前。
        # 它按 poses.npy 的长度对齐 T,从而保证 drop 臂与 native/identity 臂
        # 帧数一致(FVD/FID 才可比)。若把这段挪到钳制之前,drop 臂的 T 会与
        # 另外两臂不同,而症状完全静默 —— 数字照出,只是不可比。
        action_path = None      # 位姿通路整体不激活
    if action_path is not None and c2ws is not None:
        Ks = torch.from_numpy(
            np.load(os.path.join(action_path, "intrinsics.npy"))
        ).float()
        Ks = get_Ks_transformed(
            Ks,
            height_org=480,
            width_org=832,
            height_resize=h,
            width_resize=w,
            height_final=h,
            width_final=w,
        )
        Ks = Ks[0]

        len_c2ws_cam = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws_cam - 1, len_c2ws_cam),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws_cam - 1, lat_f),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=framewise)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(device)
        Ks = Ks.to(device)
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            "b (f h w) c -> b c f h w",
            f=lat_f,
            h=lat_h,
            w=lat_w,
        ).to(config.param_dtype)
        dit_cond_dict = {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

    # 图像条件 y
    y = vae.encode([
        torch.cat([
            torch.nn.functional.interpolate(
                img[None].cpu(), size=(h, w), mode="bicubic"
            ).transpose(0, 1),
            torch.zeros(3, F_total - 1, h, w),
        ], dim=1).to(device)
    ])[0]
    y = torch.cat([msk, y])

    # 采样
    logger.info(f"开始采样 ({sampling_steps} 步, {F_total} 帧)...")
    with torch.amp.autocast("cuda", dtype=config.param_dtype), torch.no_grad():
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps

        latent = noise
        arg_c = {
            "context": [context[0]],
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }
        arg_null = {
            "context": context_null,
            "seq_len": max_seq_len,
            "y": [y],
            "dit_cond_dict": dit_cond_dict,
        }

        torch.cuda.empty_cache()

        current_model_tag = None  # 跟踪当前哪个 14B 在 GPU 上
        for t in tqdm(timesteps, desc="Sampling"):
            latent_input = [latent.to(device)]
            t_batch = torch.stack([t]).to(device)

            if t.item() >= boundary:
                model = high_noise_model
                other = low_noise_model
                tag = "high"
                gs = guide_scale[1]
            else:
                model = low_noise_model
                other = high_noise_model
                tag = "low"
                gs = guide_scale[0]

            # 按需 swap：boundary 在去噪序列里只跨越 1 次，整次采样最多 1 次 swap
            if inactive_model_cpu and tag != current_model_tag:
                other.cpu()
                torch.cuda.empty_cache()
                model.to(device)
                current_model_tag = tag

            pred_cond = model(latent_input, t=t_batch, **arg_c)[0]
            torch.cuda.empty_cache()
            pred_uncond = model(latent_input, t=t_batch, **arg_null)[0]
            torch.cuda.empty_cache()

            noise_pred = pred_uncond + gs * (pred_cond - pred_uncond)
            temp_x0 = scheduler.step(
                noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)

        # VAE 解码
        logger.info("VAE 解码...")
        videos = vae.decode([latent])
        videos = [_resize_decoded_video(video) for video in videos]

    return videos[0]


def main():
    parser = argparse.ArgumentParser(description="LingBot 遥感 LoRA 推理测试")
    parser.add_argument("--checkpoint_dir", required=True, help="基础模型目录")
    parser.add_argument("--lora_path", required=True, help="LoRA checkpoint 目录")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--prompt", type=str, default=None, help="文本 prompt")
    parser.add_argument("--action_path", type=str, default=None, help="相机轨迹目录")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--frame_num", type=int, default=33)
    parser.add_argument("--sampling_steps", type=int, default=70)
    parser.add_argument("--shift", type=float, default=10.0)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="逻辑 CUDA 设备；物理卡由 CUDA_VISIBLE_DEVICES 选择")
    parser.add_argument("--t5_cpu", action="store_true", default=True,
                        help="T5 在 CPU 上跑（80GB 默认 ON，省 ~10GB）")
    parser.add_argument("--no_t5_cpu", dest="t5_cpu", action="store_false")
    parser.add_argument("--inactive_model_cpu", action="store_true", default=True,
                        help="按 timestep 切换 14B（80GB 默认 ON，省 ~28GB）")
    parser.add_argument("--no_inactive_model_cpu", dest="inactive_model_cpu",
                        action="store_false",
                        help="两份 14B 都驻 GPU（需要 >=96GB）")
    parser.add_argument("--no_framewise", action="store_true",
                        help="Use cumulative poses instead of framewise for Plücker embeddings")

    args = parser.parse_args()
    device = torch.device(args.device)
    config = WAN_CONFIGS["i2v-A14B"]

    # 默认 prompt
    if args.prompt is None:
        args.prompt = "A video sequence of a remote sensing scene, showing terrain from a satellite view."

    # 默认输出路径
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"rs_lora_test_{ts}.mp4"

    # 加载模型
    text_encoder, vae, low_model, high_model = load_lora_model(
        checkpoint_dir=args.checkpoint_dir,
        lora_path=args.lora_path,
        device=device,
        config=config,
        t5_cpu=args.t5_cpu,
        inactive_model_cpu=args.inactive_model_cpu,
    )

    # 加载图片
    img = Image.open(args.image).convert("RGB")
    logger.info(f"输入图片: {args.image} ({img.size[0]}x{img.size[1]})")

    # 生成
    video = generate_video(
        text_encoder=text_encoder,
        vae=vae,
        low_noise_model=low_model,
        high_noise_model=high_model,
        config=config,
        prompt=args.prompt,
        image=img,
        device=device,
        frame_num=args.frame_num,
        size=args.size,
        action_path=args.action_path,
        seed=args.seed,
        sampling_steps=args.sampling_steps,
        shift=args.shift,
        guide_scale=(args.guide_scale, args.guide_scale),
        t5_cpu=args.t5_cpu,
        framewise=not args.no_framewise,
        inactive_model_cpu=args.inactive_model_cpu,
    )

    # 保存
    logger.info(f"保存视频: {args.output}")
    save_video(
        tensor=video[None],
        save_file=args.output,
        fps=config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )

    logger.info("推理完成!")

    # 清理
    del text_encoder, vae, low_model, high_model, video
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
