#!/usr/bin/env python3
# =============================================================================
# recompute_fvd_i3d.py — 标准 I3D 口径的 FVD 离线重算(v2.6 新增)
#
# 为什么存在(REVIEWER_STYLE §3-O3 / §8-B;审稿人小组 2026-07-25 必做项):
#   仓内 metrics.py 的 "FVD" 实际用 torchvision r3d_18 代理提取器(512-d,
#   16 帧 112×112),与文献的标准 I3D-FVD(Kinetics-400 I3D,400-d logits)
#   **不在同一特征空间,数值不可跨论文比较**。本脚本对已有 eval 目录
#   (generated/*.npy + gt/*.npy)离线补算标准口径 FVD_I3D:
#     * 提取器 = StyleGAN-V 发行的 i3d_torchscript.pt(事实标准实现,
#       VideoGPT / TATS / DIGAN / common_metrics_on_video_quality 同源),
#       调用契约 model(video, rescale=True, resize=True, return_features=True)
#       → (N,400) 特征;video 为 (N,3,T,H,W) float32,取值 0–255。
#     * 帧策略 = 每条视频取【前 --num-frames 帧】(默认 16,即文献 FVD-16
#       口径;本仓 clip 最短 17 帧,恰好可行)。gen 与 GT 两侧同一策略。
#     * 分辨率 = 原样喂入(gen 464×832 / GT 480×832),由 torchscript 的
#       resize=True 内部到 224 —— 与 metrics.py 代理版"两侧各自 resize、
#       不做交叉裁剪"的分布级口径一致。
#     * FVD = Fréchet 距离 |μ1−μ2|² + tr(Σ1+Σ2−2(Σ1Σ2)^{1/2}),scipy sqrtm。
#
# 诚实声明(必须随数字一起报告,论文措辞见 REVIEWER_STYLE §8-B):
#   n=100 vs 400-d 同样是秩亏估计(文献通用协议亦如此);输出 JSON 会
#   把 n / 特征维 / 帧策略 / 提取器 sha256 全部落盘。FVD_I3D 可与文献
#   同口径对照;metrics.py 的代理 FVD 从此只作【内部可比】的辅助读数。
#
# 用法(登录节点,rswm 环境;纯离线,权重用 offline_prefetch.sh I3D=1 预取):
#   python server_run/recompute_fvd_i3d.py \
#       --i3d $ASSET_ROOT/i3d/i3d_torchscript.pt \
#       --eval-dirs textonly=$W/eval_r16_sem_s42 base=$W/eval_base \
#       -o $W/fvd_i3d.json [--device cuda] [--patch]
#   --patch: 把 FVD_I3D 写回各 eval 目录 eval_results.json 的 overall 块
#            (先备份 eval_results.json.bak_prei3d;默认不写,只出汇总 JSON)。
#
# 自测(纯 numpy,无需 torch;沙箱可跑):
#   python server_run/recompute_fvd_i3d.py --self-test
#   覆盖:闭式高斯 Fréchet 对照 / 同分布≈0 / 均值漂移单调 / NaN 特征硬错 /
#         配对缺失硬错 / 短 clip 硬错 / 假提取器端到端(镜像 outpaint 假 pipe
#         自测模式)。⚠️ 真 torchscript 的调用契约在【首次真跑】时由
#         _load_extractor 的 dummy 前向断言验证(shape=(N,400) 且有限),
#         契约不符即 FATAL —— 不会静默算出垃圾。
# =============================================================================
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_CONTRACT_DIR = Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))
from submission_contract import (CONDITION_FRAMES,
                                 MIN_SAMPLES_FOR_DISTRIBUTION)

FEATURE_DIM = 400          # I3D logits 维度(运行时 dummy 前向再次断言)
DEFAULT_NUM_FRAMES = 16    # 文献 FVD-16 口径;本仓 clip 最短 17 帧


# =============================================================================
# [v2.9.4] headline 资格门 —— 本脚本产出的是【会进论文】的数字。
# collect_results 有三档分类,但这些独立出口此前完全没有门:一个 legacy/
# identity/来路不明的 eval 目录照样能被聚合进多种子均值或 FVD 重算表。
# 规则与 collect_results.classify 同源:必须有 run_manifest.json、其
# headline_eligible 必须显式为 true、且不得带 manifest_override 痕迹。
# 逃生口 RSWM_SKIP_HEADLINE_GATE=1(产物不得作为 headline 数字)。
# =============================================================================
def _assert_headline_eligible(eval_dir, *, what="该目录"):
    import json as _j, os as _o
    from pathlib import Path as _P
    if _o.environ.get("RSWM_SKIP_HEADLINE_GATE", "0").strip().lower() not in (
            "0", "false", "no", "off", ""):
        print(f"  ⚠️ RSWM_SKIP_HEADLINE_GATE=1,跳过 {what} 的资格校验")
        return
    mp = _P(eval_dir) / "run_manifest.json"
    if not mp.is_file():
        raise SystemExit(f"[headline] {what} {eval_dir} 没有 run_manifest.json —— "
                         f"无法证明口径,拒绝把它的数字聚合进论文表。")
    m = _j.loads(mp.read_text())
    if m.get("manifest_override"):
        raise SystemExit(f"[headline] {what} {eval_dir} 带 manifest_override 痕迹"
                         f"(用过 RSWM_ALLOW_MANIFEST_MISMATCH),不得引用。")
    if m.get("headline_eligible") is not True:
        raise SystemExit(
            f"[headline] {what} {eval_dir} 不合 headline 资格:"
            f"{m.get('headline_ineligible_reason') or '未记录理由'} —— "
            f"诊断臂/判门/legacy 的数字不得进论文主表。")


def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# 纯 numpy 数学(自测直接打这里;与 torch 完全解耦)
# ---------------------------------------------------------------------------

def gaussian_stats(feats: np.ndarray):
    """(N,D) 特征 → (μ, Σ)。含 fail-loud:非有限值即错(NaN 会静默传染 sqrtm)。"""
    f = np.asarray(feats, dtype=np.float64)
    if f.ndim != 2 or len(f) < 2:
        raise ValueError(f"gaussian_stats: 需要 (N>=2, D) 特征,得到 shape={f.shape}")
    if not np.isfinite(f).all():
        raise ValueError("gaussian_stats: 特征含 NaN/Inf —— 提取器输出损坏,拒绝继续"
                         "(静默吞掉会像 metrics.py 旧 except→NaN 一样产出假数)。")
    mu = f.mean(axis=0)
    sigma = np.cov(f, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2) -> dict:
    """两高斯的 Fréchet 距离。返回 {'fvd', 'regularized'}。

    sqrtm 在近奇异积上可能给出带虚部/非有限结果:先按标准实现取实部
    (虚部相对量级 < 1e-3 视为数值噪声);仍非有限 → 加 1e-6·I 正则重试一次
    并把 regularized=True 落盘;再失败即硬错 —— 绝不像旧实现那样 except→NaN。
    """
    from scipy import linalg

    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    diff = mu1 - mu2
    regularized = False

    def _sqrtm_real(a, b):
        # [v2.6.1 审查修复 F5] 不传 disp:该 kwarg 在 scipy 1.18 移除(1.17 已
        # DeprecationWarning),且 disp=False 返回 tuple 而默认返回 array ——
        # 兼容两代:统一取 array。旧 scipy 失败时打印告警属响亮行为,
    # [v2.7.1 更正] metrics.py 曾有 4 处 disp=False,v2.7.1 已修(过期断言曾在此误导审查)
        cov = linalg.sqrtm(a @ b)
        if isinstance(cov, tuple):   # 极老版本防御
            cov = cov[0]
        if np.iscomplexobj(cov):
            imax = float(np.abs(cov.imag).max())
            rmax = float(max(np.abs(cov.real).max(), 1e-12))
            if imax / rmax > 1e-3:
                return None
            cov = cov.real
        if not np.isfinite(cov).all():
            return None
        return cov

    covmean = _sqrtm_real(sigma1, sigma2)
    if covmean is None:
        regularized = True
        eps = 1e-6 * np.eye(len(mu1))
        covmean = _sqrtm_real(sigma1 + eps, sigma2 + eps)
        if covmean is None:
            raise FloatingPointError(
                "frechet_distance: sqrtm 在正则化后仍不可用 —— 特征分布病态"
                "(检查 n 是否过小 / 提取器输出是否常数),拒绝产出数字。")
    fvd = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2)
                - 2.0 * np.trace(covmean))
    return {"fvd": fvd, "regularized": regularized}


def clip_first_frames(video: np.ndarray, num_frames: int, name: str) -> np.ndarray:
    """(T,H,W,3) uint8 → 排除条件帧后取前 num_frames 帧。

    不足即硬错(不许 pad/循环补齐 ——
    补出来的静止尾巴会系统性拉低时间维方差,FVD 静默失真)。"""
    v = np.asarray(video)
    if v.ndim != 4 or v.shape[-1] != 3:
        raise ValueError(f"{name}: 期望 (T,H,W,3),得到 {v.shape}")
    needed = CONDITION_FRAMES + num_frames
    if len(v) < needed:
        raise ValueError(
            f"{name}: 只有 {len(v)} 帧 < 条件帧 {CONDITION_FRAMES} + "
            f"--num-frames {num_frames}。canonical clip 最短 17 帧，"
            f"排除首帧后恰有 16 个预测帧；不自动 pad/降级。")
    return v[CONDITION_FRAMES:needed]


def collect_pairs(eval_dir: Path):
    """eval 目录 → [(key, gen_npy, gt_npy)]。gen 有而 gt 无 → 硬错(配对断裂);
    目录为空 → 硬错。"""
    _assert_headline_eligible(eval_dir, what="被重算 FVD 的 eval 目录")
    gen_dir, gt_dir = eval_dir / "generated", eval_dir / "gt"
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"{gen_dir} 不存在 —— 不是 evaluate.py 产出的 eval 目录")
    gen_files = sorted(gen_dir.glob("*.npy"))
    if not gen_files:
        raise FileNotFoundError(f"{gen_dir} 里没有 *.npy")
    pairs = []
    missing = []
    for gf in gen_files:
        gt = gt_dir / gf.name
        if not gt.exists():
            missing.append(gf.stem)
            continue
        pairs.append((gf.stem, gf, gt))
    if missing:
        raise FileNotFoundError(
            f"{eval_dir}: {len(missing)} 个样本缺 gt/ 配对(如 {missing[:3]})——"
            f"gen/gt 分布必须来自同一批样本,缺配对会静默改变 GT 分布,拒绝继续。")
    return pairs


def extract_features_streaming(npy_paths, extractor, num_frames, batch_size, tag):
    """流式读 npy → 批量过提取器 → (N,D) float64。内存只驻留一个 batch。"""
    feats = []
    batch = []
    for p in npy_paths:
        v = clip_first_frames(np.load(str(p)), num_frames, str(p))
        batch.append(v)
        if len(batch) >= batch_size:
            feats.append(extractor(np.stack(batch)))
            batch = []
    if batch:
        feats.append(extractor(np.stack(batch)))
    out = np.concatenate(feats, axis=0).astype(np.float64)
    if out.shape[1] != FEATURE_DIM:
        raise RuntimeError(
            f"{tag}: 提取器输出维度 {out.shape[1]} != {FEATURE_DIM}(I3D logits)——"
            f"torchscript 契约与预期不符,数字不可信,拒绝继续。")
    return out


def fvd_for_dir(eval_dir: Path, extractor, num_frames: int, batch_size: int) -> dict:
    pairs = collect_pairs(eval_dir)
    if len(pairs) < MIN_SAMPLES_FOR_DISTRIBUTION:
        raise ValueError(
            f"{eval_dir}: n={len(pairs)} < {MIN_SAMPLES_FOR_DISTRIBUTION}；"
            f"分布型指标样本量不足，不产出 FVD_I3D。")
    gen_feats = extract_features_streaming(
        [p[1] for p in pairs], extractor, num_frames, batch_size, f"{eval_dir}/generated")
    gt_feats = extract_features_streaming(
        [p[2] for p in pairs], extractor, num_frames, batch_size, f"{eval_dir}/gt")
    fd = frechet_distance(*gaussian_stats(gen_feats), *gaussian_stats(gt_feats))
    return {
        "FVD_I3D": fd["fvd"],
        "sqrtm_regularized": fd["regularized"],
        "n": len(pairs),
        "feature_dim": FEATURE_DIM,
        "num_frames": num_frames,
        "condition_frames_excluded": CONDITION_FRAMES,
        "frame_selection": "first_predicted_frames_after_condition",
        # n < feature_dim → 协方差秩亏(文献同协议亦然);随数字一起公开。
        "rank_deficient": len(pairs) < FEATURE_DIM,
    }


# ---------------------------------------------------------------------------
# torch 侧(只在真跑时 import;自测不需要)
# ---------------------------------------------------------------------------

def _load_extractor(i3d_path: str, device: str, batch_log: bool = True):
    """加载 torchscript I3D 并用 dummy 前向断言契约(shape/(有限性))。"""
    import torch

    p = Path(i3d_path)
    if not p.is_file():
        _fatal(f"--i3d {i3d_path} 不存在。先在联网机跑 "
               f"I3D=1 bash server_run/offline_prefetch.sh 并拷入节点。")
    model = torch.jit.load(str(p), map_location="cpu").eval().to(device)

    def extractor(batch_uint8: np.ndarray) -> np.ndarray:
        # (B,T,H,W,3) uint8 → (B,3,T,H,W) float32 0–255;rescale/resize 交给
        # torchscript(StyleGAN-V 官方调用方式)。
        x = torch.from_numpy(np.ascontiguousarray(batch_uint8)) \
                 .permute(0, 4, 1, 2, 3).contiguous().float().to(device)
        with torch.no_grad():
            f = model(x, rescale=True, resize=True, return_features=True)
        return f.detach().cpu().numpy()

    # 契约自检:2 条 dummy 视频。契约不符(kwargs 不认 / 输出非 (2,400))会在
    # 这里当场爆,而不是烧完 100 条后产出错数。
    dummy = (np.random.RandomState(0).rand(2, DEFAULT_NUM_FRAMES, 64, 64, 3)
             * 255).astype(np.uint8)
    try:
        out = extractor(dummy)
    except Exception as e:
        _fatal(f"i3d torchscript 契约自检失败({type(e).__name__}: {e})——"
               f"该权重不是 StyleGAN-V 版 i3d_torchscript.pt?")
    if out.shape != (2, FEATURE_DIM) or not np.isfinite(out).all():
        _fatal(f"i3d torchscript 契约自检失败:dummy 前向输出 shape={out.shape}"
               f"(期望 (2,{FEATURE_DIM}))或含非有限值。")
    # [v3.1] 分块 —— i3d torchscript 权重几十到上百 MB,整文件读进内存没必要
    # (同 make_variant_h5 的教训,只是量级小)。
    _h = hashlib.sha256()
    with open(p, "rb") as _f:
        while True:
            _b = _f.read(1 << 22)
            if not _b:
                break
            _h.update(_b)
    sha = _h.hexdigest()
    if batch_log:
        print(f"[i3d] 契约自检通过  sha256={sha[:16]}…  size={p.stat().st_size}B")
    return extractor, sha


# ---------------------------------------------------------------------------
# --patch:把 FVD_I3D 写回 eval_results.json(先备份)
# ---------------------------------------------------------------------------

def patch_eval_results(eval_dir: Path, entry: dict) -> None:
    jp = eval_dir / "eval_results.json"
    if not jp.is_file():
        _fatal(f"--patch: {jp} 不存在(先跑 evaluate.py --metrics_only)")
    bak = eval_dir / "eval_results.json.bak_prei3d"
    if not bak.exists():
        shutil.copyfile(jp, bak)
    with open(jp) as f:
        results = json.load(f)
    if "overall" not in results:
        _fatal(f"--patch: {jp} 无 overall 块,格式不认识,不动它")
    results["overall"]["FVD_I3D"] = entry["FVD_I3D"]
    results["overall"]["FVD_I3D_Meta"] = {
        k: entry[k] for k in ("n", "num_frames", "feature_dim",
                              "rank_deficient", "sqrtm_regularized")}
    with open(jp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[patch] {jp} ← FVD_I3D={entry['FVD_I3D']:.2f}(备份: {bak.name})")


# ---------------------------------------------------------------------------
# 自测(纯 numpy;镜像 outpaint 假 pipe 模式:假提取器跑通全管线 + 故障注入)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    import tempfile

    rng = np.random.default_rng(0)
    n_fail = 0

    def check(name, cond):
        nonlocal n_fail
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            n_fail += 1

    print("[1/4] 闭式高斯对照(数学正确性)")
    # 同均值同协方差 → 0
    d = 8
    mu = np.zeros(d)
    a = rng.normal(size=(5000, d))
    m1, s1 = gaussian_stats(a)
    r0 = frechet_distance(m1, s1, m1, s1)
    check(f"同分布 FVD≈0 (got {r0['fvd']:.2e})", abs(r0["fvd"]) < 1e-6)
    # 均值平移 δ、协方差同 → 解析值 |δ|²(用同一份样本消采样噪声)
    delta = np.full(d, 2.0)
    r1 = frechet_distance(m1, s1, m1 + delta, s1)
    check(f"均值漂移解析值 |δ|²={delta @ delta:.1f} (got {r1['fvd']:.3f})",
          abs(r1["fvd"] - float(delta @ delta)) < 1e-6)
    # 各向同性方差 1 vs 4(独立同构高斯):解析 tr 项 = d·(1+4−2·2)=d
    b = rng.normal(scale=2.0, size=(200000, d))
    m2, s2 = gaussian_stats(b)
    r2 = frechet_distance(m1, s1, m2, s2)
    check(f"方差 1→4 解析≈{d}(采样估计 got {r2['fvd']:.2f})",
          abs(r2["fvd"] - d) < 0.8)
    check("漂移单调: shift(2δ) > shift(δ)",
          frechet_distance(m1, s1, m1 + 2 * delta, s1)["fvd"] > r1["fvd"])

    print("[2/4] 故障注入(fail-loud 行为)")
    bad = a.copy()
    bad[3, 2] = np.nan
    try:
        gaussian_stats(bad)
        check("NaN 特征应硬错", False)
    except ValueError:
        check("NaN 特征硬错", True)
    try:
        clip_first_frames(np.zeros((8, 4, 4, 3), np.uint8), 16, "short")
        check("短 clip 应硬错", False)
    except ValueError:
        check("短 clip 硬错", True)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "eval_x"
        (root / "generated").mkdir(parents=True)
        (root / "gt").mkdir(parents=True)
        T, H, W = 17, 24, 40

        def _mk(dirname, key, seed, drift=0.0):
            r = np.random.default_rng(seed)
            base = r.random((H, W, 3)) * 255
            v = np.stack([np.clip(base + t * drift, 0, 255) for t in range(T)])
            np.save(str(root / dirname / f"{key}.npy"), v.astype(np.uint8))

        # 分布指标统一执行 n>=32 的硬门。自检也必须走同一条生产路径，不能
        # 为了夹具方便偷偷绕过样本量契约。
        for i in range(MIN_SAMPLES_FOR_DISTRIBUTION):
            _mk("generated", f"s{i:02d}", seed=i, drift=3.0)
            _mk("gt", f"s{i:02d}", seed=100 + i, drift=3.0)

        # [v3.1] 夹具必须带 run_manifest.json。
        # 这个自检此前**整体失败**:v2.9.4 给本脚本加了 headline 资格门
        # (collect_pairs → _assert_headline_eligible),但夹具没跟着更新,
        # 于是自己的门把自己的自检拦死在 [3/4]。
        # 顺手把它从"补个文件让测试过"升级成**真验门** —— 见下面三条故障注入。
        import json as _json

        def _write_manifest(**over):
            m = {"schema": "rswm.run_manifest/1", "profile": "official",
                 "sem_arm_mode": "drop", "pose_mode": "native",
                 "role": "main", "headline_eligible": True}
            m.update(over)
            (root / "run_manifest.json").write_text(_json.dumps(m))

        _write_manifest()
        print("[3/4] 配对收集(真实文件)+ headline 资格门故障注入")

        def _gate_blocks(label, mutate):
            mutate()
            try:
                collect_pairs(root)
            except SystemExit as e:
                check(f"{label} → 拒跑", "[headline]" in str(e))
            else:
                check(f"{label} → 拒跑(门没拦住)", False)
            _write_manifest()          # 恢复成合规状态

        _gate_blocks("缺 run_manifest.json",
                     lambda: (root / "run_manifest.json").unlink())
        _gate_blocks("headline_eligible=false",
                     lambda: _write_manifest(headline_eligible=False))
        _gate_blocks("带 manifest_override 痕迹",
                     lambda: _write_manifest(manifest_override="用过逃生口"))
        check("合规 manifest 下门放行",
              len(collect_pairs(root)) == MIN_SAMPLES_FOR_DISTRIBUTION)
        check("32 对配对", len(collect_pairs(root)) == MIN_SAMPLES_FOR_DISTRIBUTION)
        (root / "gt" / "s31.npy").unlink()
        try:
            collect_pairs(root)
            check("缺 gt 配对应硬错", False)
        except FileNotFoundError:
            check("缺 gt 配对硬错", True)
        _mk("gt", "s31", seed=131, drift=3.0)   # 恢复

        print("[4/4] 假提取器端到端(管线正确性;真 I3D 契约由首跑 dummy 断言)")

        def fake_extractor(batch):  # (B,T,H,W,3) → 确定性 (B,400) 特征
            b = batch.astype(np.float64)
            per_frame = b.mean(axis=(2, 3, 4))            # (B,T)
            f = np.concatenate([per_frame,
                                np.diff(per_frame, axis=1)], axis=1)  # (B,2T-1)
            reps = int(np.ceil(FEATURE_DIM / f.shape[1]))
            return np.tile(f, (1, reps))[:, :FEATURE_DIM]

        r = fvd_for_dir(root, fake_extractor, num_frames=16, batch_size=4)
        check(f"端到端出数 (FVD_I3D={r['FVD_I3D']:.3f}, n={r['n']})",
              np.isfinite(r["FVD_I3D"]) and
              r["n"] == MIN_SAMPLES_FOR_DISTRIBUTION)
        check("n<400 → rank_deficient=True 被公开", r["rank_deficient"] is True)
        r_same = fvd_for_dir(root, fake_extractor, 16, 4)
        check("确定性: 重跑逐位一致", r_same["FVD_I3D"] == r["FVD_I3D"])

    print(f"\n{'All checks passed.' if n_fail == 0 else f'{n_fail} check(s) FAILED.'}")
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i3d", help="i3d_torchscript.pt 路径(offline_prefetch I3D=1 预取)")
    ap.add_argument("--eval-dirs", nargs="*", default=[],
                    help="name=dir(evaluate.py 产出,含 generated/ 与 gt/)")
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                    help=f"每 clip 取前 N 帧(默认 {DEFAULT_NUM_FRAMES}=文献 FVD-16 口径)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda",
                    help="cuda|cpu(登录节点无 GPU 用 cpu,~200 次前向可接受)")
    ap.add_argument("-o", "--output", default="fvd_i3d.json")
    ap.add_argument("--patch", action="store_true",
                    help="把 FVD_I3D 写回各 eval 目录 eval_results.json(先备份)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    if not args.i3d or not args.eval_dirs:
        ap.error("--i3d 与 --eval-dirs 必填(或用 --self-test)")

    extractor, sha = _load_extractor(args.i3d, args.device)
    out = {
        "extractor": "i3d_torchscript(StyleGAN-V 发行;Kinetics-400 I3D,400-d logits)",
        "i3d_sha256": sha,
        "call_contract": "model(x[N,3,T,H,W] float 0-255, rescale=True, resize=True, "
                         "return_features=True)",
        "frames_policy": f"first {args.num_frames} frames(两侧同策略)",
        "note": ("FVD_I3D 与文献 I3D-FVD 同口径可对照;metrics.py 的 'FVD'(r3d_18 "
                 "代理)仅内部可比。n<400 时协方差秩亏 —— 文献同协议,如实并报。"),
        "configs": {},
    }
    for spec in args.eval_dirs:
        if "=" not in spec:
            _fatal(f"--eval-dirs 需要 name=dir 形式,得到 {spec!r}")
        name, d = spec.split("=", 1)
        ed = Path(d)
        print(f"[{name}] {ed}")
        entry = fvd_for_dir(ed, extractor, args.num_frames, args.batch_size)
        entry["eval_dir"] = str(ed)
        out["configs"][name] = entry
        print(f"[{name}] FVD_I3D = {entry['FVD_I3D']:.2f} "
              f"(n={entry['n']}, frames={entry['num_frames']}, "
              f"regularized={entry['sqrtm_regularized']})")
        if args.patch:
            patch_eval_results(ed, entry)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] → {args.output}")


if __name__ == "__main__":
    main()
