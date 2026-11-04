# RSWM 跑代码指南(总入口)

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FlashAttention-2→SDPA）；Legacy/reference = **H200 141GB**
>（sm90、可选 FA3 路径保留，本轮未做 H200 实机回归）。

> 更新:2026-08-14(v3.0 + A800 80GB profile)| 适用:单卡 A800 80GB / H200 | 包:`rswm_server_pack`
>
> **正式矩阵见 `DEPLOY_D4.md` 顶部**:20-22=drop(headline text-only)/ 23-25=native
> (headline Plücker)/ 36=identity(静止相机**诊断**臂,非 text-only)。
> 判门是 **34a + 34b 两个**,都 PASS 才放行主矩阵。
>
> **v2.7.6 P0 热修**:兼容 Transformers v5 已投影的
> `get_*_features().pooler_output`——512 维直通、768 维仅投影一次,修复真实
> CLIP 校准 `9×512 @ 768×512` 崩溃;当前提交入口仍只用
> `qsub_corrected_rerun.sh`。
>
> **v2.6 一句话**:五项代码补齐 —— Track A well-posed 双栏、Track B 报表键补全
> (O9)、commanded 路由判定加固(O11,`--eval-dirs` 支持 `:text/:geo` 后缀)、
> 标准 I3D-FVD 离线重算(`server_run/recompute_fvd_i3d.py`)、MVSplat360 桥接层
> +冒烟门(`server_run/mvsplat_smoke.sh`)。详见 `CHANGELOG_vs_0710.md` v2.6 节。
>
> **一句话**:旧 AID 实验已全部完成且**不可触碰**;新的"档2 修正-prompt 重跑"写进**独立 WORK**,
> 跑之前**必须先过第 0 步 CPU 校验**,不绿就别烧 GPU。

## 服务器 official / legacy 运行约定

两个 profile 的用途必须分开，产物不得混入同一 headline 汇总：

| Profile | 入口与用途 | Prompt / 路径估计 | 论文资格 |
|---|---|---|---|
| `legacy` | `server_run/p0_smoke.sh`、`server_run/p1_main.sh` 明确固定；仅用于历史复现、调试和 smoke | `net` / `flow` | 不进入 headline |
| `official` | ICLR 正式流水线与正式矩阵 | `net_continuous` / `registration` | 通过全部数据、训练和评估门后才可进入 headline |

正式 ICLR 运行前显式记录以下环境值；不要依赖交互 shell 的残留变量：

```bash
export RSWM_PROFILE=official
export RSWM_PROMPT_SCHEMA=net_continuous
export RSWM_PATH_ESTIMATOR=registration

# AID 必须来自本地 Hugging Face cache。服务器曾出现 429 Too Many Requests，
# 正式实验不允许在建库时联网，也不允许回退到 synthetic。
export HF_HOME=/nix/rsroll_25828/cache/huggingface
export HF_DATASETS_CACHE=/nix/rsroll_25828/cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
```

上述路径不包含、也不需要硬编码 Hugging Face token。`imageio`、
`imageio-ffmpeg`、`scikit-image` 与 `lpips` 已列入正式 requirements；LPIPS
初始化失败会记录警告并跳过该指标，不阻塞其余 evaluation，但正式环境仍应安装并预取权重。

A800 正式 profile 是**单进程单卡 + CPU expert offload**。`train.py` 的 DDP
实现保留，但 `inactive_model_cpu=True` 与经典 DDP 不兼容，代码会在加载 28B 权重前
直接拒绝该组合。多卡运行必须关闭 offload；若显存条件不满足，则需先完成并验证
FSDP/model-parallel 方案，不能绕过现有保护。

**文档导航**(本文件是总入口,细节分散在):

| 想知道 | 看 |
|---|---|
| **怎么跑(从零到出表)** | 👈 **本文件** |
| **A800 80GB 显存与提交配置** | `server_run/A800_80G.md` |
| PBS 作业依赖图 / 单个作业细节 / 断点续跑 | `server_run/pbs/README_PBS.md` |
| 代码是否真满足师兄 9 条意见 + 5 个已修 bug | `REVIEW_GAP_REPORT.md` |
| 审稿人会怎么打 + 哪些实验必须补 | `REVIEWER_STYLE_EXPERIMENT_REVIEW.md` |
| 第二数据集(RESISC45)细节 | `server_run/README_RESISC45.md` |
| 基线框架 / 怎么加一个新方法 | `server_run/baselines/README_BASELINES.md` |

---

## 0. 目录

1. [已完成的工作(历史,只读)](#1-已完成的工作历史只读)
2. [本轮新变动](#2-本轮新变动20260711-0715)
3. [环境与一次性准备](#3-环境与一次性准备)
4. [🚦 第 0 步:CPU 校验闸门(必做)](#4--第-0-步cpu-校验闸门必做)
5. [第 1 步:档2 全量重跑](#5-第-1-步档2-全量重跑)
6. [第 2 步:分析与出表](#6-第-2-步分析与出表)
7. [第 3 步:可选实验](#7-第-3-步可选实验)
8. [文件地图](#8-文件地图)
9. [安全红线与排障](#9-安全红线与排障)

---

## 1. 已完成的工作(历史,只读)

### 1.1 旧 AID 实验(**全部完成、数字已确认、严禁覆盖**)

在 Gadi H200 上跑完的完整流水线,产物在 **`server_run/work/`**(旧 WORK):

| 阶段 | 内容 | 结果 |
|---|---|---|
| 01 | canonical 建库(AID,500 样本,窗口 128×222 / steps 9 / 480×832 / seed 42) | `train_geo.h5` + `train_semantic.h5` |
| 02–04 | 主训练 r128 语义 + baseline 评估 + LoRA 评估 | **FVD 77.1→31.2(−60%)**、Urban SSIM 0.345→0.550 |
| 05–10 | rank 扫描 r16/r32/r64(全 @100 重评过) | FVD 33.6/37.6/31.1 ≈ r128 的 31.2 → **低 rank 已饱和** |
| 11–12 | 几何通道(Plücker)训练 + 评估 | 方向 ≈ 随机 |
| 13–17 | overlap 四档 + 重分箱 | ΔFVD 单调,**Ω≈0.3 处 crossover**(低重叠才是有意义的测试区间) |
| 18 | 定性 mp4 / 帧 | 24 帧已入论文 |
| 19 | 对齐汇总 | `work/summary.md` + `aligned_main/comparison.json`(**论文数字的权威源**) |

> ⚠️ 这些结果 **只读**。所有新脚本都强制独立 WORK,指向旧 `server_run/work` 会 **FATAL 拒绝**。

### 1.2 历史上已修复的关键坑(别踩回去)

| 坑 | 现状 |
|---|---|
| **错构建器**(96×96/eurosat,与论文不可比) | `new/build_from_remotesense.py` 已加 `[REFUSED]` 硬守卫 |
| 帧数指纹 | canonical=**17/25/33**(steps 9 段界去重,4n+1);v2.8 前=18/27/36;错的=16/24/32(steps 8) |
| **offload + 8-bit AdamW** 在 CPU 张量上崩 | A800 正式默认 `OFFLOAD=auto`（解析为 1）+ `torch.AdamW`，并同步迁移 LoRA grad/state；H200 常驻路径可保留 8-bit |
| P5 quals 崩(缺 `source_scene_category`) | builder 已写该 attr |
| 每 epoch checkpoint 吃 ~160GB | **本轮已在作业内自动清理**(见 §2) |
| **角误差符号反转** | 已修正:目标 = 相机角 **+180°**(内容流与相机相反) |

---

## 2. 本轮新变动(2026-07-11 ~ 07-15)

### 2.1 实验设定修正

| 变动 | 说明 |
|---|---|
| **prompt 修正** | 旧:`"...moving eastward with {K} direction changes."`(K 段却说 K 次转向,差一)<br>新:`"A satellite video of {desc}, following a {K}-segment trajectory with net displacement toward the {DIR}."` |
| **GT 轨迹持久化** | H5 每样本新增 `waypoints[K+1,2]` / `segment_angles[K]` / `num_segments` —— **Track B 的 GT 来源**。<br>⚠️ **旧 AID H5 没有这些字段**,Track B 必须用新 builder 重建的数据集。 |
| **默认 rank 改 16** | r16 在 SSIM/PSNR/LPIPS/FID 上 **≥ r128**,参数只有 1/8。`T_MAIN` 已改指 `train_r16_sem_s42` |
| **`--split_seed`** | train/val 划分种子与训练种子**解耦**。多种子必须 `SPLIT_SEED=42` 固定,否则每个种子换划分 → **验证集泄漏、种子不可比** |
| **`--gen_seed`** | 扩散采样种子(baseline 无训练随机性,只换它) |
| **`--lora_targets`** | `{all, attn_cam, attn_only}`;`attn_cam` 去掉 FFN LoRA。0 匹配 **fail-loud** |

### 2.2 新增能力

| 模块 | 作用 |
|---|---|
| `new/trajectory_metrics.py` | **Track B 完整轨迹控制**:净方向 / 逐段角误差 / 分段成功率 / 转弯 recall+precision+F1+时机 / waypoint 误差 + 终点误差 / Fréchet + DTW / 流幅度 / 有效性门。**12 指标**,8 场景 63 断言自测 |
| `new/diagnostic_baselines.py` | 诊断基线:persistence / deterministic-translation / random-direction / GT-oracle |
| `server_run/baselines/` | **统一 registry/adapter**:同格式记录(`sample_id/generated_frames/source_frames/target_frames/route_metadata/runtime_metadata`)、同 split/manifest/分辨率/帧数/gen_seed |
| `server_run/aggregate_seeds.py` | 多种子统计:mean±std / median / paired diff / bootstrap CI / 置换检验。**缺 seed 直接 fail-loud** |
| `server_run/parameter_efficiency_summary.py` | 参数效率表:可训参数 / 峰值显存 / 训练+推理耗时 / FPS / 各指标 |
| `server_run/check_resisc45.py` | 第二数据集 CPU 冒烟(源/H5/离线/split + `--self-test`) |
| `server_run/validate_direction_metric.py` | **方向指标校准**(GT 地板 / 8 方向人工平移 / 流幅度) |
| `server_run/run_trajectory_eval.py` | 串联 Track B + 诊断基线 |

### 2.3 本轮自查修掉的 **5 个真 bug**

| # | Bug | 危害 |
|---|---|---|
| B1 | `evaluate.py` 内联方向解析只认 `"eastward"` | 新 prompt 下 `dir_E/N/W/S` **静默全丢** |
| B2 | 4 处传"全量 videos + 过滤后 angles",`zip` 截断 | **视频与角度错位、角误差全错且不报错**(既存缺陷) |
| B3 | 逐样本指标被丢弃 | 逐轨迹 bootstrap **无数据** |
| B4 | 13 个融合作业不清理 epoch checkpoint | **撑爆 qk73**(历史 ~160GB) |
| B5 | `train.py` 无显存/推理耗时打点 | 参数效率表 peak GPU、FPS **恒 null** |

> B1/B2 是**静默出错**类,最危险。详见 `REVIEW_GAP_REPORT.md` §10。

---

## 3. 环境与一次性准备

```bash
# 1) 解压(覆盖旧包不影响 work/ 产物)
cd /scratch/qk73/$USER/RSWM
tar xzf rswm_server_pack.tar.gz && cd rswm_server_pack

# 2) 环境(_pbs_env.sh 已填好实测值,换机器改这里)
#   conda:  /scratch/qk73/zz1230/miniconda3        env: /scratch/rx40/zz1230/envs/rswm
#   资产:   ASSET_ROOT=/scratch/qk73/zz1230/RSWM   (基模 / hf_cache / torch_cache)
#   计算节点断网 → HF_*_OFFLINE=1 已默认开
conda activate /scratch/rx40/zz1230/envs/rswm

# 3) 若尚未预取(登录节点,有网)
bash server_run/offline_prefetch.sh          # 基模 56G + AID 3G + FID/FVD/LPIPS 权重 0.5G
# TIER2=1 会额外拉 RESISC45 + PatternNet(只跑 AID 不需要;PatternNet 默认不用,见 README_RESISC45.md)
```

**PBS 头**(Gadi 扁平格式,已填好):
```
#PBS -P rx40 -q gpuhopper
#PBS -l ncpus=12,ngpus=1,mem=128GB,jobfs=100GB,walltime=...
#PBS -l storage=scratch/qk73+scratch/rx40      ← 必须,否则计算节点看不到 /scratch
#PBS -l wd
```

---

## 4. 🚦 第 0 步:CPU 校验闸门(必做)

**全部在登录节点跑,纯 CPU、几分钟、不烧 GPU。任何一条不绿 → 停,先修指标,别投训练。**

```bash
# (a) Track B 数值自测 —— 63 条断言。⚠️ 这是它第一次真正运行(开发沙箱无 numpy/cv2)
python lora_remotesense/new/trajectory_metrics.py     # 期望:All checks passed. 退出 0
python lora_remotesense/new/diagnostic_baselines.py   # 期望:打印各基线 shape + 流幅度

# (b) RESISC45 逻辑自测(只在要跑第二数据集时需要)
python server_run/check_resisc45.py --self-test       # 期望:9 PASS / 0 FAIL

# (c) 语义门桩回归(不需要 CLIP 权重)
python server_run/test_semantic_gate.py
# 判据:退出 0、0 FAIL、结果全绿;不要把检查项数量写死
```

桩回归不能替代真实 Transformers/CLIP 集成校准。提交 01/P0 前必须严格执行
`DEPLOY_D4.md §3b`，确认 `SEMANTIC_CALIBRATION_RC=0`、校准 JSON 非空,并
人工判读 00009/00011 分离。

**(d) 方向指标校准**(需要一份已有的 eval 输出;首轮可先只看 GT 地板/人工平移):
```bash
python server_run/validate_direction_metric.py \
    --gt-h5   $WORK/data/train_semantic.h5 \
    --split   $WORK/train_r16_sem_s42/train_val_split.json \
    --eval-dirs base=$WORK/eval_base \
    -o $WORK/direction_validation.json
```

**判停条件(四条全绿才继续)**:

| 检查 | 期望 | 不过意味着 |
|---|---|---|
| GT 地板(vs 内容角) | **≈ 22.5°**(四方位量化地板) | 符号约定又拧了 / 光流不可靠 |
| GT 地板(vs 相机角,对照) | ≈ 157.5° | 同上 |
| 人工平移 8 方向恢复 | **≈ 0°** | 估计器本身有问题 |
| 各场景流幅度 | **> 0**(GT ≈ 34 px/帧) | 该场景光流测不到(低纹理) |
| text-only 的 103° | 不是少数异常轨迹拉的(看 `top5_outliers`) | 结论不稳 |

---

## 5. 第 1 步:档2 全量重跑

**一键提交**(01 建库 → 13 训练+评估 → baseline):

```bash
# 先干跑,看清楚要提交什么、写到哪
DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh

# 真提交(会自动做容量/inode 预检)
bash server_run/pbs/qsub_corrected_rerun.sh

# 自定独立 WORK(推荐显式指定)
WORK_CORRECTED=/scratch/rx40/$USER/rswm_corrected bash server_run/pbs/qsub_corrected_rerun.sh
```

**作业矩阵(13 个 train+eval 全并行 + 1 个 baseline)**:

| 作业 | 配置 | 用途 |
|---|---|---|
| 20–22 | r16 **text-only** × seed 42/43/44 | 主结果 + 种子稳定性 |
| 23–25 | r16 **text+Plücker** × seed 42/43/44 | 路由对比 + 种子稳定性 |
| 26–27 | r128 sem / geo × s42 | 容量对照(误接地是否只在高 rank 出现) |
| 28–29 | r16 sem / geo **attn_cam**(去 FFN LoRA) | 参数效率 |
| 30 | r8 sem | 更小 rank 探针 |
| 31 | baseline(zero-shot) | 参照 |
| 32–33 | shuffled_pose / direction_stripped | 归因实验 |

- 每个作业固定 `SPLIT_SEED=42`;eval 读**自己训练目录的 split**(确定性 → 各配置 split 逐字节相同 → 全并行,互不等待)。
- **训练+评估都成功后自动清理 `checkpoint-epoch-*`**,只留 `best_checkpoint`(省上百 GB)。
- 墙钟 40h/作业(训 ~10h + 评 ~10h);超时重投会自动 `--resume_from` 续训。

---

## 6. 第 2 步:分析与出表

**全部纯 CPU,在登录节点跑**(命令也会在编排脚本末尾打印):

```bash
W=/scratch/rx40/$USER/rswm_corrected      # 你的独立 WORK

# (1) 方向指标校准(完整版,含各配置对比)
python server_run/validate_direction_metric.py \
    --gt-h5 $W/data/train_semantic.h5 --split $W/train_r16_sem_s42/train_val_split.json \
    --eval-dirs textonly=$W/eval_r16_sem_s42 textpluecker=$W/eval_r16_geo_s42 base=$W/eval_base \
    -o $W/direction_validation.json

# (2a) 【Track B v2 前置】回填 trajectory/full_scene(若 01 是旧转换器建的;新建库自带,可跳过)
DRYRUN=1 python server_run/augment_h5_trajectory.py \
    --src $W/tmp_bench_main/*.h5 --dst $W/data/train_geo.h5     # 先看计划
python server_run/augment_h5_trajectory.py --src ... --dst $W/data/train_geo.h5

# (2b) 【Track B v2 冒烟】5 样本 + 校准闸门;必须打印 TRACK B V2 CALIBRATION: PASS 才跑全量
python server_run/run_trajectory_eval.py --gt-h5 $W/data/train_geo.h5 \
    --split $W/train_r16_sem_s42/train_val_split.json \
    --eval-dirs textonly=$W/eval_r16_sem_s42 --max-samples 5 --with-baselines -o "$TMPDIR/smoke.json"

# (2) Track B 完整轨迹控制 + 诊断基线(GT 轨迹从 GEO_H5 读)
#     [v2.6] :text/:geo 后缀 = commanded 路由的显式声明。v2.6 evaluate 产出的
#     目录带 meta pose_conditioning 标记可省略;旧目录/归因目录(shufpose、
#     geostrip)必须显式写,歧义或与 meta 冲突会 FATAL(不再按目录名猜)。
python server_run/run_trajectory_eval.py \
    --gt-h5 $W/data/train_geo.h5 --split $W/train_r16_sem_s42/train_val_split.json \
    --eval-dirs textonly=$W/eval_r16_sem_s42:text textpluecker=$W/eval_r16_geo_s42:geo \
                geostrip=$W/eval_r16_geostrip_s42:geo \
    --with-baselines -o $W/trajectory_eval.json

# (3) 多种子统计(缺 seed 会 fail-loud;--per-traj 加逐轨迹 bootstrap)
#     [v2.6] 默认指标含 Angular_Error_WellPosed_Mean;聚合 v2.6 之前的旧 eval
#     目录时用 --metrics 显式去掉它(旧目录没有该键,会按约定 fail-loud)。
python server_run/aggregate_seeds.py --seeds 42 43 44 \
    --eval-dirs r16_sem=$W/eval_r16_sem_s42,$W/eval_r16_sem_s43,$W/eval_r16_sem_s44 \
                r16_geo=$W/eval_r16_geo_s42,$W/eval_r16_geo_s43,$W/eval_r16_geo_s44 \
    --pair r16_sem r16_geo --per-traj --out-prefix $W/seed_summary
# → seed_summary.{csv,json,md,tex}

# (4) 参数效率表
python server_run/parameter_efficiency_summary.py \
    --exp-dirs $W/train_r16_* $W/train_r8_* $W/train_r128_* \
    --out-prefix $W/param_efficiency
# → param_efficiency.{csv,json,md,tex}

# (5) [v2.6] 标准 I3D 口径 FVD 离线重算(与文献可比;metrics.py 的 FVD 是
#     r3d_18 代理、仅内部可比。权重先在联网机 I3D=1 bash server_run/offline_prefetch.sh)
python server_run/recompute_fvd_i3d.py --i3d $ASSET_ROOT/i3d/i3d_torchscript.pt \
    --eval-dirs textonly=$W/eval_r16_sem_s42 textpluecker=$W/eval_r16_geo_s42 \
                base=$W/eval_base -o $W/fvd_i3d.json --device cpu
# → fvd_i3d.json(FVD_I3D 每配置;--patch 可写回各 eval_results.json,先备份)

# (6) 统一结果总表(corrected 目录优先,run_all 目录回退)
python server_run/collect_results.py $W
# → summary.md + summary.json
```

### [v2.6] Track A 角误差双栏(报表口径)

eval_results.json 每块新增 `Angular_Error_WellPosed_*` 与 `WellPosed_N`:
well-posed = GT 命令轨迹净位移比 ≥ 0.2(与 Track B 净方向门控**同一真源、
同一阈值**;近闭合轨迹的净方向问题无定义)。**正式读数以 well-posed 栏为准,
两栏并报、各印 n**;scores.json 逐样本行含 `gt_net_displacement_ratio` 与
`well_posed`,逐轨迹 bootstrap 可分栏做。

**回传给我这四份 + 各 `eval_*/eval_results.json`,我据此填论文表格。**

### 报告口径(别混)

| 不确定性 | 怎么算 | 回答什么 |
|---|---|---|
| **训练种子方差** | 3 个训练种子的 mean±std | 换个初始化,结论还在吗 |
| **测试轨迹 bootstrap** | 对 per-sample 重采样 | 换一批测试轨迹,这个数还稳吗 |

> 两者是**不同**的不确定性,**必须都报,不能互相替代**。
> n=3 时置换检验的最小双侧 p = **0.25** —— 别过度解读显著性。

### Track B 的适用范围(审稿人会问)

| 配置 | 拿到的条件 | 可评什么 |
|---|---|---|
| **text-only** | 只有净方向词 + 段数 | **只有**净方向 + 流幅度是"达标量";逐段/转弯/waypoint/路径**标 `commanded=False`,仅作诊断,不能算它的失败** |
| **text+Plücker** | 完整逐帧轨迹 | **全部 Track B 指标都成立** |

---

## 7. 第 3 步:可选实验

### 7.1 第二数据集泛化(RESISC45,TEST-ONLY)

**不在 RESISC45 上训练** —— 拿已训好的 AID checkpoint 去评它,掉点 = 泛化间隙。最便宜也最站得住。

```bash
DRYRUN=1 bash server_run/pbs/qsub_resisc45.sh --lora-ckpt $W/train_r16_sem_s42/best_checkpoint
bash server_run/pbs/qsub_resisc45.sh --lora-ckpt $W/train_r16_sem_s42/best_checkpoint
```
默认 **RESISC45-only,不下载 PatternNet**。细节见 `server_run/README_RESISC45.md`。

### 7.2 诊断基线 / 外部基线

```bash
python server_run/baselines/run_baseline.py --help
```
- **诊断基线**(persistence / deterministic-translation / random-direction / gt-oracle):无需模型,证明 benchmark 能区分四类失败(视觉好但方向错 / 方向对但质量差 / 单帧好但不连续 / 几乎没动)。
- **外部生成基线**:目前**只有 stub,没有任何一个跑出过数**(`baseline_status.json` 全部 `tested=false / result_available=false`)。
  ⚠️ **论文不得声称已完成外部方法比较。** 要落地见 `server_run/baselines/README_BASELINES.md`。

---

## 8. 文件地图

```
lora_remotesense/
  build_dataset_from_benchmark.py   建库(修正 prompt + waypoints 持久化)★改
  new/
    train.py                        训练(--split_seed / --lora_targets / 参数审计 / 显存打点)★改
    evaluate.py                     评估(--gen_seed / 方向解析单一真源 / scores.json)★改
    metrics.py                      基础指标(逐样本出参)★改
    trajectory_metrics.py           Track B 12 指标 + 63 断言自测 ★新
    diagnostic_baselines.py         诊断基线 ★新
    wan/modules/animate/animate_utils.py   LoRA target 过滤 ★改
server_run/
  common.sh                         封装 + 旋钮(SEED/SPLIT_SEED/GEN_SEED/LORA_TARGETS/WORK)★改
  validate_direction_metric.py      方向校准 ★新
  run_trajectory_eval.py            Track B + 基线驱动(v2.6: 报表键补全 + 路由加固)★改
  aggregate_seeds.py                多种子统计(v2.6: 默认含 well-posed 栏)★改
  parameter_efficiency_summary.py   参数效率表 ★新
  check_resisc45.py                 第二数据集冒烟 ★新
  recompute_fvd_i3d.py              标准 I3D 口径 FVD 离线重算(文献可比)★v2.6 新
  mvsplat_smoke.sh                  MVSplat360 档B 2-5 样本冒烟门 ★v2.6 新
  baselines/                        统一 registry / wrapper / status(v2.6: mvsplat360
                                    桥接层 + SampleUnavailable + --skip-failed 协议)★改
  pbs/
    01-19_*.pbs                     旧 AID 流水线(历史,已跑完)
    20-31_*.pbs                     档2 修正-prompt 重跑 ★新
    40_build_resisc45.pbs           第二数据集 ★新
    qsub_corrected_rerun.sh         档2 编排(独立 WORK + 预检)★新
    qsub_resisc45.sh                RESISC45 编排 ★新
REVIEW_GAP_REPORT.md                代码事实审计 + 5 个 bug ★新
REVIEWER_STYLE_EXPERIMENT_REVIEW.md 审稿式实验缺口(8 节)★新
```

---

## 9. 安全红线与排障

### 红线(脚本已强制)

| 红线 | 强制方式 |
|---|---|
| **旧 AID 结果不可覆盖** | 所有新编排器的 `WORK` 指向旧 `server_run/work` → **FATAL 拒绝**;不再删除任何旧 H5/stamp |
| **无 mv44** | 包内零可执行 mv44 路径(仅有"勿用 mv44"的告诫语;v2.4.3 已把 `pbs_train_96g.sh` 剔出打包白名单、`evaluate_mvsplat360.py` 默认值改必填、各用法示例改占位符)。⚠️ 源码树历史遗留(根目录 `eval.pbs` 等)不入包,**新流程一个都不调用**。mv44 已确认关闭(2026-07-24 用户再确认) |
| **qk73 容量 / rx40 inode** | 提交前 `df -Pk/-Pi` 预检(< 300GiB 或 < 200k inode 拒跑);分析脚本只写**聚合文件**;作业自动清 epoch checkpoint |
| **DRYRUN** | 所有编排器支持 `DRYRUN=1`,只打印不提交 |

### 常见问题

| 症状 | 原因 / 处理 |
|---|---|
| `FATAL: 找不到 _pbs_env.sh` | 必须**从包根** qsub(缺 `-l wd` 或提交目录不对) |
| `Track B: GT 里没有 waypoints` | 用的是**旧数据集**。Track B 必须用新 builder 重建的 H5(第 1 步的 01) |
| 角误差 ≈ 180° 而非 ≈ 22.5° | 符号约定被改动了。目标应 = 方向词角 **+180°** |
| `BACKBONE PARAMS ARE UNEXPECTEDLY UNFROZEN` | 主干泄漏进优化器,参数效率结论会失效。**别绕过**,查 LoRA 注入 |
| `peak_gpu_mem_gb=null` | 该 run 是**加显存打点之前**跑的。重跑才有真值(不臆造) |
| `TRACK B V2 CALIBRATION: FAIL` | gt_oracle 没过闸门(净方向<15°/逐段<15°/turn_recall>0.8/valid>95%)。查:H5 是否回填了 trajectory/full_scene;knots_source 是否 uniform_fallback |
| Track B 基线 valid_pct=0% | 传了首帧 `image` 当场景(≠full_scene)。v2 已加 1.5× 守卫直接 ValueError |
| bootstrap 报"不可用" | 该 eval 目录没有 `scores.json`(加打点前跑的)。重跑 eval 即可 |
| 8-bit AdamW 崩 `is_on_gpu` | 不要绕过守卫；A800 用 `HARDWARE_PROFILE=a800_80g OFFLOAD=auto`（smoke 固定 1），训练器会选 torch.AdamW |

### 关键约定速查

| 项 | 值 |
|---|---|
| 帧数指纹 | canonical **17/25/33**(v2.8 段界去重);v2.8 前 18/27/36;错的 16/24/32 |
| 生成 / GT 分辨率 | 生成 **464**×832(VAE 取整),GT 480×832;配对指标取共同 **top-left 464×832** |
| 角误差约定 | 目标 = 相机方向 **+180°**(内容流与相机相反);**90°=随机**,**≈22.5°=完美**(四方位量化地板),**≈180°=跟着词漂移** |
| K 段 vs 转向 | K 段 = **K−1** 次转向 |
| FVD | R3D-18 代理提取器 → **仅内部可比**,不可与 I3D-FVD 文献比 |
| FID | 单中间帧、n=25/100 → **远低于稳定区间**,别过度解读 |
