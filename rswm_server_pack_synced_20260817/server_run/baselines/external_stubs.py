#!/usr/bin/env python3
# =============================================================================
# external_stubs.py — 外部 **生成式** baseline 的适配器 STUB
#
# 铁律:这里只登记 **仓库里真实存在钩子** 的候选。凭空捏一个模型/权重/数字进来,
# 比留白危害大得多。截至本文件写就:
#   - 仓库里有 MVSplat360 的完整评估代码 + 安装脚本 → 登记(status=unavailable:
#     代码在,依赖不在)。
#   - 仓库里有 image_outpainting / rswise_recursive 两个 NotImplementedError 函数
#     桩(diagnostic_baselines.py:449-513)→ 登记(status=stub:方法本身没写)。
#   - CamI2V / 独立 SVD 视频生成 / 其它 world model:全仓 grep 无任何代码或权重
#     引用 → **不登记**(见 README_BASELINES.md「查过但不存在的候选」)。
#
# 每个 stub 的 run() 一律抛 NotImplementedError,并 **点名** 到底缺哪个依赖
# (逐个探测本地路径后报,不是笼统说「缺依赖」)。
#
# 本文件只用 stdlib(os/pathlib),没有 numpy 也能 import + 列表。
# =============================================================================
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# [v2.6] 与 registry.py 的 '__main__' 别名同一招:直接 `python3 external_stubs.py`
# (跑桥接自测)时,本模块名是 '__main__';下面 import registry 会触发
# registry._register_external() 再 import 'external_stubs' —— 若不先把自己
# 别名成 'external_stubs',该 import 会把本文件 **再执行一遍**,底部的
# register() 撞「重复注册」。registry.py:492-496 修过完全同款的坑。
sys.modules.setdefault("external_stubs", sys.modules[__name__])

try:
    from .registry import (ADAPTER_VERSION, ROOT, BaselineAdapter,
                           SampleUnavailable,
                           _import_diagnostic_baselines, _require,
                           _source_fingerprint, direction_word, register,
                           validate_record)
except ImportError:  # 直接 `python3 registry.py` / 脚本式 import 时的回退
    from registry import (ADAPTER_VERSION, ROOT, BaselineAdapter,
                          SampleUnavailable,
                          _import_diagnostic_baselines, _require,
                          _source_fingerprint, direction_word, register,
                          validate_record)


class ExternalStub(BaselineAdapter):
    """外部生成式基线的通用 stub。

    子类实现 missing_dependencies() -> list[str](缺什么就列什么,空列表=齐了)。
    run() 永远抛 NotImplementedError:即使依赖齐了,把外部模型的输出桥接成本仓
    record 契约的代码 **还没写**,不能假装能跑。
    """

    kind = "external_generative"
    requires_gpu = True

    # 子类填:给人看的依赖清单
    needs_repo: str = ""
    needs_checkpoints: List[str] = []
    needs_license: str = ""
    needs_env: str = ""
    approx_gpu_hours: str = ""

    def missing_dependencies(self) -> List[str]:
        raise NotImplementedError(f"{self.name}: 子类必须实现 missing_dependencies()")

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        missing = self.missing_dependencies()
        detail = "\n  - ".join(missing) if missing else "(本地依赖看起来齐了)"
        raise NotImplementedError(
            f"{self.name}: 未实现,无法产出 record。\n"
            f"缺失依赖:\n  - {detail}\n"
            f"接线说明: {self.wiring_note()}"
        )

    def wiring_note(self) -> str:
        return "需要先把外部模型的输出桥接成 registry.RECORD_KEYS 契约(该桥接代码尚未编写)。"


# =============================================================================
# 1. MVSplat360 —— 仓库里 **有** 完整评估代码,缺的是 repo/权重/env
# =============================================================================

class MVSplat360Stub(ExternalStub):
    """通用稀疏视角 NVS(MVSplat360)在遥感 rollout 上的 domain-gap 对照。

    仓库既有资产(都已核实存在):
      - lora_remotesense/evaluate_mvsplat360.py   推理 + 指标 pipeline
      - lora_remotesense/setup_mvsplat360_autodl.sh  装 repo + 拉权重
      - server_run/p6_mvsplat.sh                  驱动脚本(含黑帧自检)

    ⚠ 公平性提醒(必须写进论文脚注):MVSplat360 拿到的是 **5 个 context views +
    相机位姿**(p6_mvsplat.sh 推理段 --num_context 5,现 :103),信息量 **多于** RSWM 的
    单图 + 文本。所以它不是「更弱的对手」,而是「给了更多信息仍有 domain gap」的
    论证。record 契约的 condition_type 三选一
    ({'text_only','text_plucker','none'})表达不了「多视角+位姿」,登记为 'none'
    (无文本条件),差异靠这条注释和 README 说明。

    ⚠ 黑帧陷阱:evaluate_mvsplat360.py 单样本失败会用黑帧填充继续算指标;
    p6_mvsplat.sh 第 4 步黑帧自检(现 :108-124)强制核查 generation_success >= 80%,否则判失败。任何
    绕过该检查得到的数字都不可信,禁止写进论文。

    ⚠ lora_remotesense/mvsplat360_baseline.pbs 是旧作业脚本,project/storage 指向
    **已退役的 mv44**,不可复用。请用 server_run/pbs/baseline_eval_template.pbs。
    """

    name = "mvsplat360"
    status = "unavailable"      # 代码在仓库里,但 repo/权重/env 不在 → 跑不了
    description = ("MVSplat360(通用稀疏视角 NVS)domain-gap 对照。仓库已有评估代码, "
                   "缺 repo/权重/独立 env。")

    needs_repo = "https://github.com/donydchen/mvsplat360.git (setup_mvsplat360_autodl.sh:42)"
    needs_checkpoints = [
        "re10k.ckpt (~48MB, HF donydchen/mvsplat, 无需认证)",
        "dl3dv_480p.ckpt (MVSplat360 主模型, HF donydchen/mvsplat360, 无需认证)",
        "svd.safetensors (~9.5GB, HF stabilityai/stable-video-diffusion-img2vid, **gated**)",
    ]
    needs_license = ("SVD 权重是 HF gated:须先在 HF 网页 'Agree and access repository' "
                     "再 huggingface-cli login(setup_mvsplat360_autodl.sh:109-117)。"
                     "具体条款以 HF 页面为准 —— 本文件不复述许可证内容。")
    needs_env = ("独立 conda env(python=3.10 + torch==2.2.2/torchvision==0.17.2, "
                 "setup_mvsplat360_autodl.sh:57)。与主训练环境(torch 2.4/2.6 + "
                 "flash-attn)**不兼容**;p6_mvsplat.sh 环境守卫段(现 :53-57)检测到 "
                 "flash_attn 可 import 会直接拒跑。")
    approx_gpu_hours = ("~4-8 GPU-hours / 20 样本 + 调试缓冲(p6_mvsplat.sh 头部自述; "
                        "旧 pbs walltime=04:00:00)。本仓从未实测,数字仅为脚本作者估计。")

    # [v2.6] 桥接层黑帧判据:target 槽位里「近黑帧」(帧均值 < BLACK_FRAME_MEAN)
    # 占比超过 BLACK_FRAME_MAX_FRACTION 即判该样本不可信 —— 兜住
    # generation_success=True 但输出实际塌黑的情形(flag 只反映"没抛异常")。
    BLACK_FRAME_MEAN = 2.0
    BLACK_FRAME_MAX_FRACTION = 0.2

    def install_dir(self) -> Path:
        """MVSplat360 安装目录:MVSPLAT_DIR 优先(p6_mvsplat.sh:22 默认 ~/mvsplat360;
        offline_prefetch MVSPLAT=1 的产物在 $STAGE/mvsplat360,拷入后指向它)。"""
        return Path(os.environ.get("MVSPLAT_DIR", str(Path.home() / "mvsplat360")))

    def missing_dependencies(self) -> List[str]:
        missing: List[str] = []
        d = self.install_dir()
        if not d.is_dir():
            missing.append(f"MVSplat360 repo 未克隆: {d} 不存在 "
                           f"(设 MVSPLAT_DIR 或跑 lora_remotesense/setup_mvsplat360_autodl.sh;"
                           f"离线预取: MVSPLAT=1 bash server_run/offline_prefetch.sh)")
        for ck in ("re10k.ckpt", "dl3dv_480p.ckpt", "svd.safetensors"):
            p = d / "checkpoints" / ck
            if not p.is_file():
                missing.append(f"缺权重: {p}")
        ev = ROOT / "lora_remotesense" / "evaluate_mvsplat360.py"
        if not ev.is_file():
            missing.append(f"缺评估脚本: {ev}")
        try:
            import flash_attn  # noqa: F401
            missing.append("当前是主训练环境(flash_attn 可 import)。MVSplat360 需要 "
                           "torch==2.2.2 的独立 env;p6_mvsplat.sh 环境守卫段(现 :53-57)会拒跑。")
        except ImportError:
            pass
        return missing

    def wiring_note(self) -> str:
        return ("[v2.6] 桥接层已写好,两段式:①在 mvsplat env 里跑 "
                "lora_remotesense/evaluate_mvsplat360.py --no_metrics 产出 eval 目录;"
                "②export RSWM_MVSPLAT_EVAL_DIR=<该目录> 后走 run_baseline.py "
                "--baseline mvsplat360 --skip-failed --min-success 0.8 —— run() 会把"
                "每个样本的产物回收成 RECORD_KEYS 契约(黑帧/失败样本抛 "
                "SampleUnavailable,由 --skip-failed 显式计数,绝不静默平均)。"
                "冒烟入口: bash server_run/mvsplat_smoke.sh(2-5 样本 + 80% 门)。")

    # ------------------------------------------------------------------
    # [v2.6] 桥接:把 evaluate_mvsplat360.py 的产物回收成 record 契约
    # ------------------------------------------------------------------
    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        eval_dir = os.environ.get("RSWM_MVSPLAT_EVAL_DIR")
        missing = self.missing_dependencies()
        if missing and not eval_dir:
            # 依赖不齐且没有现成产物目录 → 维持 stub 语义(smoke 规则依赖这条路)
            detail = "\n  - ".join(missing)
            raise NotImplementedError(
                f"{self.name}: 无法产出 record。\n缺失依赖:\n  - {detail}\n"
                f"接线说明: {self.wiring_note()}")
        if not eval_dir:
            raise NotImplementedError(
                f"{self.name}: 依赖已就绪,但未指定产物目录。先在 mvsplat env 跑 "
                f"evaluate_mvsplat360.py --no_metrics,再 "
                f"export RSWM_MVSPLAT_EVAL_DIR=<output_dir>。\n"
                f"接线说明: {self.wiring_note()}")

        import time
        import numpy as np

        ed = Path(eval_dir)
        sid = str(_require(sample, "sample_id", self.name))
        prompt = str(_require(sample, "prompt", self.name))
        waypoints = _require(sample, "waypoints", self.name)
        segment_angles = _require(sample, "segment_angles", self.name)
        gt_frames = np.asarray(_require(sample, "gt_frames", self.name))

        npy = ed / "generated" / f"{sid}.npy"
        mjs = ed / "meta" / f"{sid}.json"
        if not npy.is_file() or not mjs.is_file():
            raise SampleUnavailable(
                f"{self.name}[{sid}]: {eval_dir} 里缺 generated/{sid}.npy 或 "
                f"meta/{sid}.json —— 该样本没跑过(检查 evaluate_mvsplat360.py 日志)。")
        import json as _json
        meta = _json.loads(mjs.read_text())

        # ---- 产物一致性守卫(目录传错/新旧混用当场暴露) ----
        if str(meta.get("prompt", "")) != prompt:
            raise ValueError(
                f"{self.name}[{sid}]: eval 目录 meta 的 prompt 与 manifest H5 不一致 "
                f"—— RSWM_MVSPLAT_EVAL_DIR 指向了别的数据集的产物?")
        for k in ("gen_seed", "infer_seconds"):
            if k not in meta:
                raise ValueError(
                    f"{self.name}[{sid}]: meta 缺 {k!r} —— 产物由 v2.6 之前的 "
                    f"evaluate_mvsplat360.py 产出(无种子控制/无耗时),请重跑。")

        if not meta.get("generation_success", False):
            raise SampleUnavailable(
                f"{self.name}[{sid}]: generation_success=false"
                f"(error={meta.get('error')!r})—— 黑帧填充产物,拒绝打分。")

        video = np.load(str(npy))
        if video.dtype != np.uint8:
            video = np.clip(video, 0, 255).astype(np.uint8)
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"{self.name}[{sid}]: 生成视频形状非法 {video.shape}")

        # ---- 黑帧兜底:只查 target 槽位(context 槽位按协议填 GT 帧) ----
        tgt_idx = [int(t) for t in meta.get("target_indices", range(len(video)))
                   if int(t) < len(video)]
        if tgt_idx:
            means = video[tgt_idx].reshape(len(tgt_idx), -1).mean(axis=1)
            frac_black = float((means < self.BLACK_FRAME_MEAN).mean())
            if frac_black > self.BLACK_FRAME_MAX_FRACTION:
                raise SampleUnavailable(
                    f"{self.name}[{sid}]: {frac_black:.0%} 的 target 帧近黑"
                    f"(阈值 {self.BLACK_FRAME_MAX_FRACTION:.0%})—— "
                    f"generation_success 旗标失真,产物不可信。")

        hw = (int(video.shape[1]), int(video.shape[2]))
        DB = _import_diagnostic_baselines()
        target = DB._resize_video(gt_frames, hw)
        src_frame = sample.get("source_frame")
        src_frame = gt_frames[0] if src_frame is None else np.asarray(src_frame)
        source = DB._resize_video(np.asarray(src_frame)[None, ...], hw)

        wp_list = np.asarray(waypoints, dtype=float).tolist()
        seg_list = np.asarray(segment_angles, dtype=float).ravel().tolist()
        record = {
            "sample_id": sid,
            "generated_frames": video,
            "source_frames": source,
            "target_frames": target,
            "route_metadata": {
                "prompt": prompt,
                "waypoints": wp_list,
                "segment_angles": seg_list,
                "num_segments": len(seg_list),
                "direction_word": direction_word(prompt, wp_list),
                # record 枚举表达不了「5 context views + poses」;登记 'none' +
                # method_version 注明。⚠️ 论文脚注必须写:它拿到的条件多于 RSWM。
                "condition_type": "none",
            },
            "runtime_metadata": {
                "wall_seconds": float(meta["infer_seconds"]),
                "gen_seed": int(meta["gen_seed"]),
                "resolution": [hw[0], hw[1]],
                "num_frames": int(video.shape[0]),
                "device": "cuda",
                "method_version": (
                    f"{ADAPTER_VERSION}/mvsplat360-bridge-1"
                    f"(ctx{meta.get('num_context', '?')};context 槽位=GT 帧,"
                    f"5views+poses 条件多于 RSWM)"),
            },
        }
        return validate_record(record)


register(MVSplat360Stub())


# =============================================================================
# 2. image_outpainting —— 仓库里只有函数桩,没有模型
# =============================================================================

class ImageOutpaintingAdapter(BaselineAdapter):
    """[v2.5 已实现] 平移 + 扩散 outpaint 的 construction-aware 负对照。

    每帧沿【内容运动方向】平移窗口,前沿新暴露条带用 inpainter 补全 ——
    方向由构造免费拿对、内容靠单图外推伪造。用来回答"为什么不直接平移+
    outpaint",并检验方向指标能否被非世界模型的纯空间生成器钻空子:
    预期 Track A/B 的方向项接近合规、而时序一致性(FVD/跨帧 SSIM 衰减)崩。

    与 deterministic_translation 的区别:那条读【整幅静态场景】裁真窗口
    (场景查表,上界);这条只有首帧,新内容是生成的 —— 是实践者真会先试
    的现成管线,故 kind=external_generative。

    方向约定:传入 route 的是【相机】方向角,底层函数内部 +180 得到内容
    流向 —— 调用侧不得提前翻转(翻两次就反了)。

    模型:diffusers StableDiffusionInpaintPipeline,权重经
    `BASELINE_A=1 bash server_run/offline_prefetch.sh` 预取
    (stabilityai/stable-diffusion-2-inpainting,非 gated,~5GB);
    节点侧 HF_HUB_OFFLINE=1 直读。env 变量 RSWM_INPAINT_MODEL 可覆盖。
    """

    name = "image_outpainting"
    kind = "external_generative"
    requires_gpu = True
    status = "implemented"
    upper_bound = False
    description = ("平移+扩散 outpaint 的平凡管线:方向由构造正确、内容单图外推。"
                   "测纯空间生成器能否骗过方向指标却在时序一致性上崩。")

    _pipe = None    # 进程内复用(一个 PBS 作业跑一个 baseline,加载一次)

    def _load(self, device: str):
        if ImageOutpaintingAdapter._pipe is not None:
            return ImageOutpaintingAdapter._pipe
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        model_id = os.environ.get("RSWM_INPAINT_MODEL",
                                  "stabilityai/stable-diffusion-2-inpainting")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        ImageOutpaintingAdapter._pipe = pipe
        return pipe

    def missing_dependencies(self) -> List[str]:
        missing = []
        try:
            import diffusers  # noqa: F401
        except Exception:
            missing.append("diffusers 未安装(主 env requirements 内应有)")
        hook = ROOT / "lora_remotesense" / "new" / "diagnostic_baselines.py"
        if not hook.is_file():
            missing.append(f"缺函数实现: {hook}")
        return missing

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        import time
        import numpy as np
        # (符号已在文件头 import,含脚本式导入的回退;run() 内再做相对导入
        #  在 run_baseline.py 的 sys.path 方式下会 ImportError —— v2.5 审查抓到)
        DB = _import_diagnostic_baselines()
        sid = _require(sample, "sample_id", self.name)
        prompt = _require(sample, "prompt", self.name)
        waypoints = _require(sample, "waypoints", self.name)
        segment_angles = _require(sample, "segment_angles", self.name)
        gt_frames = np.asarray(_require(sample, "gt_frames", self.name))
        gen_seed = _require(sample, "gen_seed", self.name)
        device = sample.get("device", "cuda")
        T = int(sample.get("T", sample.get("num_frames")) or len(gt_frames))

        src_frame = sample.get("source_frame")
        src_frame = gt_frames[0] if src_frame is None else np.asarray(src_frame)

        # 相机净方向角(度):由 GT waypoints 端点位移量化到四基向 —— 与建库
        # prompt 的量化同口径(不读 prompt,避免 direction-stripped 数据失配)。
        wp = np.asarray(waypoints, dtype=float)
        ang = float(np.degrees(np.arctan2(-(wp[-1, 0] - wp[0, 0]),
                                          wp[-1, 1] - wp[0, 1]))) % 360.0
        cam_deg = (0.0 if (ang < 45 or ang >= 315) else 90.0 if ang < 135
                   else 180.0 if ang < 225 else 270.0)

        import torch
        pipe = self._load(device)
        # generator 必须显式传进 pipeline 调用(diffusers 不读 pipeline 上的
        # 自定义属性);同 gen_seed 重跑必须逐位复现。
        gen = torch.Generator(device=device).manual_seed(int(gen_seed))
        steps = int(os.environ.get("RSWM_INPAINT_STEPS", "25"))

        t0 = time.perf_counter()
        video = DB.image_outpainting_baseline(
            src_frame, direction_angle=cam_deg, T=T, model=pipe, device=device,
            inpaint_prompt="aerial satellite imagery, seamless continuation",
            generator=gen, num_inference_steps=steps)
        wall = float(time.perf_counter() - t0)

        hw = (int(video.shape[1]), int(video.shape[2]))
        target = DB._resize_video(gt_frames, hw)
        source = DB._resize_video(np.asarray(src_frame)[None, ...], hw)
        wp_list = wp.tolist()
        seg_list = np.asarray(segment_angles, dtype=float).ravel().tolist()

        record = {
            "sample_id": str(sid),
            "generated_frames": video,
            "source_frames": source,
            "target_frames": target,
            "route_metadata": {
                "prompt": str(prompt),
                "waypoints": wp_list,
                "segment_angles": seg_list,
                "num_segments": len(seg_list),
                "direction_word": direction_word(str(prompt), wp_list),
                # 方向来自 GT 轨迹(构造性),不是文本/位姿条件生成
                "condition_type": "none",
            },
            "runtime_metadata": {
                "wall_seconds": wall,
                "gen_seed": int(gen_seed),
                "resolution": [hw[0], hw[1]],
                "num_frames": int(video.shape[0]),
                "device": str(device),
                "method_version": (f"{ADAPTER_VERSION}/outpaint-steps{steps}@"
                                   f"{_source_fingerprint()}"),
            },
        }
        return validate_record(record)


register(ImageOutpaintingAdapter())


# =============================================================================
# 3. rswise_recursive —— 仓库里有的是 **benchmark 构建器**,不是模型
# =============================================================================

class RSWISERecursiveStub(ExternalStub):
    """递归/自回归 world model rollout(RSWISE 式)的长程漂移对照。

    仓库钩子:diagnostic_baselines.rswise_recursive_baseline
    (diagnostic_baselines.py:483-513)—— NotImplementedError 函数桩,签名要求外部
    传入已加载的 `model`。

    ⚠ 关键澄清(别被名字骗了):仓库里的
    wan/remotesensebenchmark/build_rsvideowise_benchmark.py 是一个
    **benchmark 数据构建器**("extends RSWISE from RemoteBAGEL to video sequences",
    该文件:5-6),**不是** 可推理的 RSWISE 模型。仓库里没有 RemoteBAGEL/RSWISE 的
    模型代码,也没有任何权重。所以这条不是「配好环境就能跑」,而是「模型压根不在」。
    """

    name = "rswise_recursive"
    status = "stub"
    description = ("RSWISE 式递归 world model rollout,测误差累积/长程一致性。"
                   "仓库里只有 RSVideoWise **benchmark 构建器** 和函数桩,没有模型。")

    needs_repo = ("无可用模型 repo。仓库里的 build_rsvideowise_benchmark.py 是 benchmark "
                  "构建器,不是模型(该文件:5-6)。RemoteBAGEL/RSWISE 模型代码不在本仓。")
    needs_checkpoints = ["无(仓库中没有任何递归 world model 权重)"]
    needs_license = "未定 —— 取决于最终选用的递归 world model。"
    needs_env = "未定(模型未选定)。"
    approx_gpu_hours = "无法估计 —— 模型未选定。不编数字。"

    def missing_dependencies(self) -> List[str]:
        missing = [
            "没有递归 world model 的实现或权重:仓库里的 RSVideoWise 资产是 benchmark "
            "构建器(wan/remotesensebenchmark/build_rsvideowise_benchmark.py:5-6),不是模型",
            "diagnostic_baselines.rswise_recursive_baseline() 是 NotImplementedError 桩 "
            "(diagnostic_baselines.py:510-513),需要传入已加载的 `model` 并实现递归采样循环",
        ]
        hook = ROOT / "lora_remotesense" / "new" / "diagnostic_baselines.py"
        if not hook.is_file():
            missing.append(f"连函数桩都找不到: {hook}")
        return missing


register(RSWISERecursiveStub())


# =============================================================================
# [v2.6] MVSplat360 桥接层自测:`python3 external_stubs.py`
# 假产物端到端(镜像 outpaint 假 pipe 模式):真实文件 + 真实 numpy,
# 只有「模型推理」被假产物替代;真模型契约由 mvsplat_smoke.sh 首跑裁决。
# =============================================================================

def _bridge_self_test() -> int:
    import json
    import shutil
    import uuid

    import numpy as np

    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            n_fail += 1

    adapter = None
    for a in [MVSplat360Stub()]:   # 独立实例,不动注册表
        adapter = a

    T, H, W = 18, 48, 80   # 缩小分辨率,契约与真尺寸无关
    rng = np.random.default_rng(0)

    def _sample(prompt):
        return dict(
            sample_id="s00", prompt=prompt,
            waypoints=[[10.0, 10.0], [10.0, 60.0]], segment_angles=[0.0],
            gt_frames=(rng.random((T, H, W, 3)) * 255).astype(np.uint8),
            gen_seed=42, window_size=(H, W), T=T,
            source_frame=(rng.random((H, W, 3)) * 255).astype(np.uint8),
        )

    PROMPT = ("A satellite video of an urban area, following a 1-segment "
              "trajectory with net displacement toward the east.")

    # Python's TemporaryDirectory chmod/ACL normalization can make the resulting
    # folder inaccessible to native extensions in managed Windows sandboxes.  Use an
    # explicit, narrowly scoped project-local fixture and clean exactly that path.
    td = ROOT / "tests" / f"_external_bridge_{uuid.uuid4().hex}"
    td.mkdir()
    try:
        ed = td / "eval_mv"
        (ed / "generated").mkdir(parents=True)
        (ed / "meta").mkdir(parents=True)
        vid = (rng.random((T, H, W, 3)) * 200 + 30).astype(np.uint8)
        np.save(str(ed / "generated" / "s00.npy"), vid)
        meta = {"key": "s00", "prompt": PROMPT, "generation_success": True,
                "error": None, "gen_seed": 42, "infer_seconds": 3.21,
                "num_context": 5, "context_indices": [0, 4, 9, 13, 17],
                "target_indices": [t for t in range(T) if t not in (0, 4, 9, 13, 17)]}
        (ed / "meta" / "s00.json").write_text(json.dumps(meta))

        os.environ["RSWM_MVSPLAT_EVAL_DIR"] = str(ed)
        try:
            print("[1/6] 成功样本 → 合规 record")
            rec = adapter.run(_sample(PROMPT))
            check("record 契约通过(validate_record 已在 run 内跑)",
                  rec["sample_id"] == "s00"
                  and rec["runtime_metadata"]["num_frames"] == T
                  and rec["runtime_metadata"]["gen_seed"] == 42
                  and rec["route_metadata"]["direction_word"] == "east")
            check("wall_seconds 来自 meta.infer_seconds",
                  rec["runtime_metadata"]["wall_seconds"] == 3.21)

            print("[2/6] generation_success=false → SampleUnavailable")
            meta_bad = dict(meta, generation_success=False, error="boom")
            (ed / "meta" / "s00.json").write_text(json.dumps(meta_bad))
            try:
                adapter.run(_sample(PROMPT))
                check("应抛 SampleUnavailable", False)
            except SampleUnavailable:
                check("抛 SampleUnavailable", True)
            (ed / "meta" / "s00.json").write_text(json.dumps(meta))

            print("[3/6] target 槽黑帧 → SampleUnavailable(旗标失真兜底)")
            black = vid.copy()
            for t in meta["target_indices"]:
                black[t] = 0
            np.save(str(ed / "generated" / "s00.npy"), black)
            try:
                adapter.run(_sample(PROMPT))
                check("应抛 SampleUnavailable", False)
            except SampleUnavailable as e:
                check(f"抛 SampleUnavailable({e})", "近黑" in str(e))
            np.save(str(ed / "generated" / "s00.npy"), vid)

            print("[4/6] 缺产物 npy → SampleUnavailable")
            try:
                adapter.run(dict(_sample(PROMPT), sample_id="s99"))
                check("应抛 SampleUnavailable", False)
            except SampleUnavailable:
                check("抛 SampleUnavailable", True)

            print("[5/6] prompt 与产物不一致 → ValueError(目录传错当场暴露)")
            try:
                adapter.run(_sample("A totally different prompt toward the east."))
                check("应抛 ValueError", False)
            except ValueError:
                check("抛 ValueError", True)

            print("[6/6] 旧版产物(meta 无 gen_seed)→ ValueError")
            m2 = {k: v for k, v in meta.items() if k != "gen_seed"}
            (ed / "meta" / "s00.json").write_text(json.dumps(m2))
            try:
                adapter.run(_sample(PROMPT))
                check("应抛 ValueError", False)
            except ValueError:
                check("抛 ValueError", True)
        finally:
            os.environ.pop("RSWM_MVSPLAT_EVAL_DIR", None)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print(f"\n{'All checks passed.' if n_fail == 0 else f'{n_fail} check(s) FAILED.'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(_bridge_self_test())
