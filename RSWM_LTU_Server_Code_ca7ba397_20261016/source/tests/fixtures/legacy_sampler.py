"""Exact supplied legacy generate_video fixture, executed only with test doubles.
Original licensing bytes: docs/legacy-source-LICENSE.md and
 docs/legacy-source-NOTICE.md. This fixture is not a generation backend.
Source: rswm_server_pack_synced_20260817/lora_remotesense/new/inference.py
Source ZIP SHA256: 2d079e768bdf66a06cc7cb8891f408c7adbcb1985424b801fa962da97400ce73
Original function text SHA256: 74c905c7ec6ddc76aceb6386d4beb79d47d5fe6189d119799d4aa00d1a553453
"""

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

    return videos[0]

# Exact original inference.py load_lora_model function text SHA256: 58b916823bfbb4e20e296e8df67d9202de0d5453413e9c6bec0a0f59eb4d81ef

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

# Exact original evaluate_baseline.py load_base_model function text SHA256: 130edb8f8982627be151b41942c0afc1cb2cc425a159372a289b3c3158fbca13

def load_base_model(
    checkpoint_dir: str,
    device: torch.device,
    config,
    t5_cpu: bool = True,
    inactive_model_cpu: bool = True,
):
    """加载原始预训练模型，不注入任何 LoRA 权重

    80GB A100 默认开 inactive_model_cpu + t5_cpu，预算与 load_lora_model 相同。
    """
    from wan.modules.model import WanModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.modules.t5 import T5EncoderModel

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

    # VAE
    logger.info("加载 VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        device=device,
    )

    # DiT（原始权重，无 LoRA）
    logger.info("加载 DiT 模型（原始权重，无 LoRA）...")
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

    low_noise_model.eval().requires_grad_(False)
    high_noise_model.eval().requires_grad_(False)
    if inactive_model_cpu:
        low_noise_model.cpu()
        high_noise_model.cpu()
        logger.info("DiT 模型加载完成（原始权重；两份均驻 CPU，采样时按 timestep 切换）")
    else:
        low_noise_model.to(device)
        high_noise_model.to(device)
        logger.info("DiT 模型加载完成（原始权重；两份均驻 GPU）")

    return text_encoder, vae, low_noise_model, high_noise_model
