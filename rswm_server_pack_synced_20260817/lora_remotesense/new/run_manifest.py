#!/usr/bin/env python3
# =============================================================================
# run_manifest.py — 一次运行的完整身份(v3.2.0)
#
# 为什么需要它:v2.9.1 的外部审查发现三条"把错误产物洗成合格 headline"的路径,
# 根因都一样 —— **每处各自校验各自的一个字段**,于是:
#   * 续训只比 pose_mode(而 native 臂在 official/legacy 下都是 native)
#     → 跨 profile 续训被放行,权重随后被重新标成 official;
#   * 评测只要 meta 文件在就跳过 → identity+legacy 的旧结果在 drop+official
#     环境里被原样复用,再被当前环境写上 official/headline 标记;
#   * 汇总只拒显式 False → 没有资格字段的旧产物(fail-open)照进主表。
#
# 单一 manifest 把"这次运行是谁"一次性钉死,训练恢复 / 评测复用 / 汇总三处
# 共同校验同一份字段。**任何字段缺失一律 fail-closed** —— 缺失恰恰是旧产物
# 和被篡改产物的共同特征,把它当"未知所以放行"就等于没有门。
#
# 用法:
#   m = build_manifest(role="headline_text", data_h5=..., split_file=..., ...)
#   write_manifest(out_dir, m)
#   assert_compatible(read_manifest(prev_dir), m, ctx="评测复用")
# =============================================================================
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

MANIFEST_VERSION = "3.2.0"
MANIFEST_NAME = "run_manifest.json"

# 这些字段一旦不同,前一次的产物就不能拿来当这一次的结果用。
IDENTITY_FIELDS = (
    "rswm_profile",
    "sem_arm_mode",
    "pose_mode",
    "role",
    "data_h5_sha256",
    "split_sha256",
    "split_name",
    "checkpoint_id",
    "dataset_tier",
    "prompt_schema",
    "frame_num",
    "gen_seed",
    "train_seed",
    "code_sha256",
    # [v2.9.4] 训练身份的真正区分项(旧版全靠恒 None 的 split_file/gen_seed,
    # 等于三个字段在空转)。缺这些的旧 manifest 会因"<缺失>"不等而被拒 —— 对。
    "split_seed",
    "lora_rank",
    "lora_alpha",
    "lora_targets",
    "max_train_samples",
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    # [RSRoll-Bench P0#1] 换骨干/换适配方式/换轨迹估计器/换重叠档,产物都不可互换。
    # 缺了它们,一次 registration 跑出来的 Track B 会和 flow 跑出来的在同一张表里
    # 静默混用 —— 而两者的仪器地板差一个数量级(50% 重叠下 111 px vs 0.09 px)。
    "backbone",
    "backbone_adaptation",
    "path_estimator",
    "step_overlap_band",
)

# 判门/诊断类角色永远不进 headline;identity 口径同理(它是静止相机诊断臂)。
_NEVER_HEADLINE_ROLES = frozenset({
    "diagnostic_stationary", "probe_gate", "legacy", "variant_attribution",
    "trivial_baseline", "metric_calibration", "submission_scoring",
    # [v3.1] 难度阶梯:它跑真数据真指标、论文要独立报告,但 K / 角度粒度 /
    # overlap 三个旋钮都被**故意改过**,数字进主表会让 benchmark 看起来比它
    # 实际定义的容易,而主表里没有任何字段提示"这几行来自另一个数据分布"。
    # 与 probe_gate 同一功能类别:报告出来,但不入 headline。
    "ladder",
})


def _sha256_file(p: Optional[str | Path], chunk: int = 1 << 20) -> Optional[str]:
    if not p:
        return None
    p = Path(p)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# [v3.1] 对外公开的分块哈希 —— 其它脚本别再各写一遍整文件 read_bytes()。
# 实测教训:对 canonical H5 用 `hashlib.sha256(p.read_bytes())` 会把整个库读进
# 内存,12 个样本(带 full_scene)就 1.3 GB,4000 个样本会 OOM 被 kill。
sha256_file = _sha256_file


def code_sha256(files: Optional[Iterable[str | Path]] = None) -> str:
    """对决定"跑出来的数字长什么样"的核心代码取指纹。

    只覆盖会改变数值/语义的文件;改注释也会变指纹是可以接受的代价 ——
    宁可多要一次重跑,也不要"代码换了但产物看起来还合格"。
    """
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    if files is None:
        files = [
            here / "train.py",
            here / "evaluate.py",
            here / "evaluate_baseline.py",
            here / "inference.py",
            here / "metrics.py",
            here / "dataset_loader.py",
            root / "lora_remotesense" / "build_dataset_from_benchmark.py",
            root / "wan" / "remotesensebenchmark"
                 / "build_multi_direction_benchmark_v5.py",
            # [RSRoll-Bench P0#1] 这三个文件同样决定"跑出来的数字长什么样":
            # backbone.py 决定用哪个模型跑,trajectory_metrics.py 决定 Track B 怎么算,
            # geom_registration.py 决定轨迹从哪来。缺了它们,换估计器不会改变指纹。
            here / "backbone.py",
            here / "trajectory_metrics.py",
            here / "geom_registration.py",
            here / "submission_contract.py",
            root / "server_run" / "common.sh",
            root / "server_run" / "score_submission.py",
        ]
    h = hashlib.sha256()
    for f in sorted(Path(x) for x in files):
        h.update(f.name.encode())
        s = _sha256_file(f)
        h.update((s or "MISSING").encode())
    return h.hexdigest()


def headline_eligible(profile: str, sem_arm_mode: str, pose_mode: str,
                      role: str, default_sem_arm: Optional[str] = None,
                      prompt_schema: Optional[str] = None,
                      path_estimator: Optional[str] = None):
    """→ (eligible: bool, reason: str|None)。

    规则(全部必须成立):
      * profile 必须是 official;
      * SEM_ARM_MODE 未被覆盖(等于该 profile 的官方值);
      * pose_mode 不是 identity —— identity 按拍板定义是**静止相机命令诊断臂**,
        永远不是 headline(36 号与 identity baseline 都归此类);
      * role 不在诊断/判门/legacy 名单里。
    """
    default_sem_arm = default_sem_arm or ("drop" if profile == "official"
                                          else "identity")
    if profile != "official":
        return False, f"RSWM_PROFILE={profile}(非 official 链)"
    if sem_arm_mode != default_sem_arm:
        return False, (f"SEM_ARM_MODE={sem_arm_mode} 覆盖了 official 官方值 "
                       f"{default_sem_arm}")
    if profile == "official" and prompt_schema != "net_continuous":
        return False, (f"official 主通路要求 prompt_schema=net_continuous，"
                       f"实际为 {prompt_schema!r}")
    if profile == "official" and path_estimator != "registration":
        return False, (f"official 主通路要求 path_estimator=registration，"
                       f"实际为 {path_estimator!r}")
    if pose_mode == "identity":
        return False, ("pose_mode=identity 是静止相机命令诊断臂,按拍板定义"
                       "永远不入 headline")
    if role in _NEVER_HEADLINE_ROLES:
        return False, f"role={role} 属诊断/判门/legacy,不入 headline"
    return True, None


def build_manifest(*, role: str,
                   pose_mode: Optional[str] = None,
                   data_h5: Optional[str | Path] = None,
                   split_file: Optional[str | Path] = None,
                   split_name: Optional[str] = None,
                   checkpoint_id: Optional[str] = None,
                   prompt_schema: Optional[str] = None,
                   frame_num: Optional[int] = None,
                   gen_seed: Optional[int] = None,
                   train_seed: Optional[int] = None,
                   split_seed: Optional[int] = None,
                   dataset_tier: Optional[str] = None,
                   lora_rank: Optional[int] = None,
                   lora_alpha: Optional[int] = None,
                   lora_targets: Optional[str] = None,
                   max_train_samples: Optional[int] = None,
                   num_epochs: Optional[int] = None,
                   batch_size: Optional[int] = None,
                   gradient_accumulation_steps: Optional[int] = None,
                   learning_rate: Optional[float] = None,
                   backbone: Optional[str] = None,
                   backbone_adaptation: Optional[str] = None,
                   path_estimator: Optional[str] = None,
                   step_overlap_band: Optional[str] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = os.environ.get("RSWM_PROFILE", "unset")
    sem_arm = os.environ.get("SEM_ARM_MODE", "unset")
    pm = (pose_mode or os.environ.get("RSWM_POSE_MODE")
          or os.environ.get("POSE_MODE") or "native")
    resolved_prompt = (prompt_schema if prompt_schema is not None
                       else os.environ.get("RSWM_PROMPT_SCHEMA"))
    resolved_estimator = (path_estimator if path_estimator is not None
                          else os.environ.get("RSWM_PATH_ESTIMATOR"))
    elig, why = headline_eligible(
        profile, sem_arm, pm, role,
        prompt_schema=resolved_prompt, path_estimator=resolved_estimator)
    m = {
        "manifest_version": MANIFEST_VERSION,
        "rswm_profile": profile,
        "sem_arm_mode": sem_arm,
        "pose_mode": pm,
        "role": role,
        "data_h5": str(data_h5) if data_h5 else None,
        "data_h5_sha256": _sha256_file(data_h5),
        "split_file": str(split_file) if split_file else None,
        "split_sha256": _sha256_file(split_file),
        "split_name": split_name,
        "checkpoint_id": checkpoint_id,
        "dataset_tier": (dataset_tier if dataset_tier is not None
                         else os.environ.get("DATASET_TIER")),
        "prompt_schema": resolved_prompt,
        "frame_num": int(frame_num) if frame_num is not None else None,
        "gen_seed": int(gen_seed) if gen_seed is not None else None,
        "train_seed": int(train_seed) if train_seed is not None else None,
        "split_seed": int(split_seed) if split_seed is not None else None,
        "lora_rank": int(lora_rank) if lora_rank is not None else None,
        "lora_alpha": int(lora_alpha) if lora_alpha is not None else None,
        "lora_targets": lora_targets,
        "max_train_samples": (int(max_train_samples)
                              if max_train_samples is not None else None),
        "num_epochs": int(num_epochs) if num_epochs is not None else None,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "gradient_accumulation_steps": (
            int(gradient_accumulation_steps)
            if gradient_accumulation_steps is not None else None),
        "learning_rate": (float(learning_rate)
                          if learning_rate is not None else None),
        "code_sha256": code_sha256(),
        # [RSRoll-Bench P0#1] 身份字段。None 时回落到环境变量，
        # 使未改造的调用方也能写出正确的身份（而不是全 None 让 assert_compatible 形同虚设）。
        "backbone": backbone or os.environ.get("RSWM_BACKBONE", "wan-inline"),
        "backbone_adaptation": backbone_adaptation,
        "path_estimator": resolved_estimator or "flow",
        "step_overlap_band": step_overlap_band or os.environ.get("RSWM_STEP_OVERLAP_BAND"),
        "headline_eligible": bool(elig),
        "headline_ineligible_reason": why,
    }
    if extra:
        m.update(extra)
    return m


def write_manifest(out_dir: str | Path, manifest: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / MANIFEST_NAME
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return p


def read_manifest(d: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(d) / MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def assert_compatible(prev: Optional[Dict[str, Any]],
                      cur: Dict[str, Any],
                      *, ctx: str,
                      fields: Iterable[str] = IDENTITY_FIELDS,
                      allow_env: str = "RSWM_ALLOW_MANIFEST_MISMATCH") -> None:
    """prev 与 cur 的身份必须一致,否则拒绝复用/续跑。fail-closed。"""
    escape = os.environ.get(allow_env, "0").strip().lower() not in (
        "0", "false", "no", "off", "")
    if prev is None:
        msg = (f"[manifest] {ctx}:目标目录没有 {MANIFEST_NAME} —— 它要么是 "
               f"v2.9.2 之前的产物,要么被手工改过。无法证明它与本次运行同源,"
               f"因此【不予复用】。删除该目录重跑,或确认无误后设 "
               f"{allow_env}=1(产物不得作为 headline)。")
        if escape:
            print("  ⚠️ " + msg)
            cur.setdefault("manifest_override", []).append(
                {"ctx": ctx, "kind": "missing_prev_manifest", "detail": msg[:400]})
            cur["headline_eligible"] = False
            cur["headline_ineligible_reason"] = (
                f"{allow_env}=1 强行复用了无 manifest 的产物({ctx})")
            return
        raise RuntimeError(msg)
    diffs = []
    for f in fields:
        a, b = prev.get(f, "<缺失>"), cur.get(f, "<缺失>")
        if a != b:
            diffs.append(f"  {f}: 已有={a!r}  本次={b!r}")
    if diffs:
        msg = (f"[manifest] {ctx}:已有产物与本次运行不同源,拒绝复用。\n"
               + "\n".join(diffs)
               + f"\n  删除该目录重跑;确要强行复用设 {allow_env}=1"
                 f"(产物不得作为 headline)。")
        if escape:
            # [v2.9.4] 逃生口必须在**产物本身**留痕。旧写法只打印一行警告,
            # 而调用方紧接着 write_manifest 会用当前身份把旧 manifest 整个覆写
            # —— 用过之后磁盘上一点痕迹都没有,事后无法归因,注释里承诺的
            # "产物不得作为 headline"没有任何机制去执行。
            print("  ⚠️ " + msg)
            cur.setdefault("manifest_override", []).append(
                {"ctx": ctx, "kind": "identity_mismatch",
                 "diffs": [d.strip() for d in diffs]})
            cur["headline_eligible"] = False
            cur["headline_ineligible_reason"] = (
                f"{allow_env}=1 强行复用了不同源产物({ctx}):"
                + "; ".join(d.strip() for d in diffs)[:300])
            return
        raise RuntimeError(msg)


def assert_metrics_only_ok(output_dir: str | Path, *, ctx: str = "--metrics_only",
                           extra_fields: Iterable[str] = ()) -> Dict[str, Any]:
    """[v2.9.3] `--metrics_only` 重算前的身份门。**两个评估入口共调**。

    重算不产生新的生成结果,但 compute_metrics 会用【当前环境】给 overall 盖上
    rswm_profile / pose_mode / headline_eligible —— 于是"在 official/drop 环境里
    对一个 legacy/identity 的旧目录重算指标"就能把旧结果洗成合格 headline。

    抽成函数是因为 v2.9.2 只给 evaluate.py 加了这道门、evaluate_baseline.py 漏了
    —— 而 baseline 是所有 Δ 的分母。孪生入口各写一份必然漂移。
    """
    m = read_manifest(output_dir)
    if m is None:
        raise SystemExit(
            f"[manifest] {ctx} 拒绝执行:{output_dir} 没有 {MANIFEST_NAME} —— "
            f"无法证明这批产物的口径,重算出来的指标会被盖上当前环境的 "
            f"profile/pose_mode/headline 标记。\n"
            f"  该目录若是 v2.9.2 之前的产物,请重跑评估而不是重算指标。")
    cur = {
        "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
        "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
        "pose_mode": os.environ.get("RSWM_POSE_MODE", "native"),
        "code_sha256": code_sha256(),
    }
    for f in extra_fields:
        cur[f] = os.environ.get(f.upper())
    bad = [f"{k}: 产物={m.get(k)!r} 当前环境={v!r}"
           for k, v in cur.items() if m.get(k) != v]
    # [v2.9.4] 也要看产物自己怎么说 —— 一个 manifest 明写 headline_eligible=false
    # 的目录,重算后 overall 会被盖上当前环境的 true,等于用重算洗白。
    if m.get("headline_eligible") is False and os.environ.get(
            "RSWM_HEADLINE_ELIGIBLE", "0") == "1":
        bad.append(f"headline_eligible: 产物=False 当前环境=True —— "
                   f"产物自己声明不合格({m.get('headline_ineligible_reason')})")
    if bad:
        raise SystemExit(
            f"[manifest] {ctx} 拒绝执行:当前环境与产物口径不符 —— "
            f"重算会把旧结果重新标注成当前口径。\n  " + "\n  ".join(bad)
            + f"\n  正确做法:设成产物自己的口径再重算"
              f"(RSWM_PROFILE={m.get('rswm_profile')} "
              f"SEM_ARM_MODE={m.get('sem_arm_mode')} "
              f"RSWM_POSE_MODE={m.get('pose_mode')});"
              f"代码指纹不同说明核心代码变了,应重跑而不是重算。")
    return m


def summarize(m: Optional[Dict[str, Any]]) -> str:
    if not m:
        return "(无 manifest)"
    return (f"profile={m.get('rswm_profile')} pose_mode={m.get('pose_mode')} "
            f"role={m.get('role')} headline={m.get('headline_eligible')} "
            f"data={str(m.get('data_h5_sha256'))[:12]} "
            f"code={str(m.get('code_sha256'))[:12]}")
