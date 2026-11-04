# REVIEW_GAP_REPORT — RSWM 档二代码 vs 师兄意见 事实审计

> 生成时间:2026-07-11 | 审计方式:**逐行读代码 + py_compile + bash -n + 纯 Python 逻辑追踪**
> ⚠️ **审计环境无 numpy / cv2 / pip / conda** → 所有依赖 numpy/cv2 的**数值行为均未运行验证**,
> 只做了静态审读。凡标「未运行时自测」的,必须在服务器(rswm 环境)跑一次才能确认。
> 本报告只陈述代码事实,不改论文正文。

---

## 0 结论速览

| # | 师兄意见 | 当前代码是否支持 | 是否经过真实自测 | 缺口 |
|---|---|---|---|---|
| 1 | 多段控制不能只测最终位置/净方向 | ✅ Track B 模块(12 指标全覆盖,§2 三缺口已补齐) | ❌ **从未运行**(沙箱无 numpy/cv2) | **服务器必须先跑 `python lora_remotesense/new/trajectory_metrics.py` 自测(63 断言)** |
| 2 | 判断每一段是否执行正确 | ✅ `segment_wise_angular_error` + **`segment_success_rate`(已补)** | ❌ 待服务器自测 | — |
| 3 | 判断是否在正确时刻转弯 | ✅ `turn_detection` + recall/timing + **precision/F1(已补)** | ❌ 待服务器自测 | — (S3/S4 提前/延后场景已加入自测) |
| 4 | 每段角度 / waypoint / 路径形状 | ✅ 逐段角 / `waypoint_error` / **`waypoint_endpoint_error`(已补)** / Fréchet+DTW | ❌ 待服务器自测 | — |
| 5 | 第二数据集泛化 | ✅ builder + **`40_build_resisc45.pbs` / `qsub_resisc45.sh`(TEST-ONLY,独立 WORK)** + **`check_resisc45.py`** + `README_RESISC45.md` | ✅ `--self-test` 9/9 实跑;DRYRUN + 3 重 fail-loud 实证 | 真实 RESISC45 缓存上的 `--check-source` 待服务器跑 |
| 6 | 约 4 个其他生成方法 | 🟡 **统一 registry/adapter 已建**(4 诊断 + 3 外部 **stub**)+ `README_BASELINES.md` + 统一评估模板 | ✅ registry smoke 实跑(坏记录/未知名 fail-loud) | **仍无任何外部生成基线真跑出数** → 论文不得声称已完成外部比较 |
| 7 | 附录多 seed | ✅ 训练侧 + **`server_run/aggregate_seeds.py`(已补)** | ✅ DRYRUN 与缺-seed fail-loud 已实跑 | — |
| 8 | 参数效率 | ✅ `--lora_targets` + r8 + **`_audit_trainable_params`(逐类分解+主干解冻 RuntimeError)** + **峰值显存打点** + **`parameter_efficiency_summary.py`** | ✅ DRYRUN 已实跑 | — |
| 9 | 审稿式实验缺口 | ✅ `REVIEWER_STYLE_EXPERIMENT_REVIEW.md`(553 行,8 节) | — | 仅建议,未改论文正文 |

图例:✅ 已正确实现 / 🟡 部分 / ❌ 缺失

---

## 1 已核实为**正确**的部分(不要重复实现)

| 项 | 证据 | 结论 |
|---|---|---|
| **waypoints 坐标约定** | `build_multi_direction_benchmark_v5.py:447` `return (cy, cx)`;`:486` `return (new_cy, new_cx)`;`camera_trajectory.py:69-70` `dy=wp[...,0]`(cy)/`dx=wp[...,1]`(cx) | H5 里 `waypoints` **确为 `[K+1,2] = (cy,cx)` 相机路径**。`trajectory_metrics._camera_net_angle` (:187-195) 与之逐字一致 ✅ |
| **相机角约定** | `camera_trajectory.py:17-21` `θ=atan2(-(cy_end-cy_start), cx_end-cx_start)`,east=0/north=90/west=180/south=270;`build_dataset_from_benchmark.py:580-590` 同式 | 一致 ✅ |
| **内容流 = −相机** 的 180° 换算 | `trajectory_metrics.py:21-30`(模块 docstring)、`:272`(`gt_content_ang=(seg_ang+180)%360`)、`:516`(`gt_net_content=(camera_net+180)%360`);`evaluate.py:76-102` `parse_direction_angle_from_prompt` 返回 `(camera_angle+180)%360` | **已在代码与文档中明确说明** ✅(师兄要求的"必须明确说明"已满足) |
| 修正 prompt | `build_dataset_from_benchmark.py:602-610` `"A satellite video of {desc}, following a {K}-segment trajectory with net displacement toward the {DIR}."` + `:605` 注释「K 段有 K−1 转向」 | ✅ |
| GT 轨迹持久化 | `build_dataset_from_benchmark.py:805-807` `waypoints`/`segment_angles`/`num_segments` | ✅(**旧 AID H5 无此字段,Track B 必须用新 builder 重建的数据集**) |
| prompt 解析兼容新旧 | `evaluate.py:87-102`(基数词子串,`east` 同时命中 `eastward` 与 `toward the east`) | ✅ |
| LoRA target 过滤 | `animate_utils.py:17-46`,substrings `self_attn`/`cross_attn`(attn)、`cam_injector`/`cam_scale`/`cam_shift`(cam)、排除 `ffn`;**0 匹配 fail-loud** `:47-52` | ✅ |
| `--lora_targets` 贯通 | `train.py:1013`(argparse)、`:238`/`:252`(两个 noise expert 都传)、`:281`(日志);`common.sh:132` `--lora_targets "${LORA_TARGETS:-all}"` | ✅ |
| `--split_seed` 防泄漏 | `train.py:997-1000`(argparse)、`:311-319`(`split_seed` 与训练种子解耦);`common.sh` 传 `${SPLIT_SEED:-42}` | ✅ **这是真陷阱,已修**:不固定则每个训练种子换 train/val 划分 → 验证集泄漏、种子不可比 |
| `--gen_seed` | `evaluate.py:149/216/(argparse)`、`evaluate_baseline.py` 同;`common.sh` 传 `${GEN_SEED:-42}` | ✅ |
| **无 mv44** | 全 `server_run/` grep `mv44` → **0 命中** | ✅ 合规 |
| WORK 可隔离 | `common.sh:26` `WORK="${WORK:-$ROOT/server_run/work}"` | ✅ 可 env 覆盖(但**新脚本需加"拒绝指向旧 AID work"的硬守卫**,见 §5) |
| DRYRUN | `qsub_corrected_rerun.sh`、`qsub_all.sh` | ✅(新脚本须一并支持) |

---

## 2 Track B 指标覆盖(师兄意见 1–4)

文件:`lora_remotesense/new/trajectory_metrics.py`(743 行)

| 必需指标 | 状态 | 证据 |
|---|---|---|
| net direction angular error | ✅ | `:442-451` `net_direction_error`(薄封装 `metrics.estimate_motion_direction`,目标=content 角) |
| per-segment angular error | ✅ | `:264-296` `segment_wise_angular_error`(逐段净位移方向 vs `(seg_ang+180)%360`) |
| **segment success rate** | ❌ **缺** | — |
| turn recall | ✅ | `:340-372` `turn_recall_and_timing`(±tol 贪心一对一) |
| **turn precision** | ❌ **缺** | 只有 recall |
| turn timing error | ✅ | 同上,`mean_timing_error` |
| **waypoint endpoint error** | ❌ **缺** | 只有全结点均值 |
| normalized waypoint error | ✅ | `:375-396` `waypoint_error`(两路径各自单位弧长归一化后取结点距离均值) |
| trajectory Fréchet | ✅ | `:399-435` `path_shape_error`(Eiter–Mannila DP) |
| trajectory DTW | ✅ | 同上 |
| optical-flow magnitude | ✅ | `:454-462` `flow_magnitude` |
| valid trajectory % | 🟡 | `:465-474` `is_valid`(单样本布尔);**聚合口径的 valid%** 在驱动 `run_trajectory_eval.py:46-49` 里算 |

### text-only vs text+Plücker 的区分 —— ✅ 已正确实现
- `score_trajectory(..., text_only=True)` → 输出 `commanded=False`(`:521`),分段/拐点/结点/形状仍计算但**标注为诊断量而非达标量**;模块 docstring `:32-41` 明确写了「text-only 只承诺净方向与段数」。
- **符合师兄要求**:text-only 不得声称精确 waypoint 控制。

### 已发现的**真问题**
1. **Farneback 被重复计算 3 次**(`:541` `flow_magnitude` → `_compute_flows`;`:551` `net_direction_error` → `metrics.estimate_motion_direction` 内部再算;`:554` `recover_content_path` → 又一次 `_compute_flows`)。100 条 × 多配置下是 3× 光流开销。**已列入修复**(net_direction 必须保留 Track A 原式以保持跨表一致,只合并另外两次)。
2. **8 个必需合成测试只覆盖 3 个**:已有 (1) 正确分段平移、(2) 反向≈180、(7) persistence 静止;**缺** (3) 转弯提前、(4) 转弯延后、(5) 形状对但整体偏移、(6) 终点方向对但中途错、(8) random direction。
   - 其中 **(6) 是最关键的一条**:它正是"净方向测不出、Track B 能测出"的存在理由,必须有。
3. `num_frames` 形参实际被覆盖(`:557` `T = len(rec_path)`)。**不是 bug**(GT 路径按生成帧数重采样,两者同轴可比),但形参语义具误导性,应在 docstring 说明。

### 自测状态 —— ❌ **从未运行过**
`__main__`(`:741-743`)结构正确(`sys.exit(_self_test())`,失败返回 1,符合"失败非零退出"要求),但:
- 本审计环境无 numpy → **连纯几何测试都跑不了**;
- `[2/3]` 段的端到端光流测试需 cv2,沙箱缺失时会 SKIP。
**结论:Track B 的一切数值(DTW/Fréchet/路径恢复/拐点检测)目前是"审读通过、运行未验证"。**

---

## 3 基线框架(师兄意见 6)

文件:`lora_remotesense/new/diagnostic_baselines.py`(586 行)、`server_run/run_trajectory_eval.py`

| 项 | 状态 |
|---|---|
| persistence / deterministic translation / random-direction / GT oracle | ✅ 已实现(诊断基线) |
| outpainting / RSWISE-递归 | 🟡 仅 `NotImplementedError` 桩 |
| **统一 adapter/registry(统一记录格式)** | ❌ **缺** |
| **统一协议**(同 split / 同 manifest / 同分辨率 / 同帧数 / 同 gen_seed / 同评估脚本) | ❌ 未成文、未强制 |
| **baseline_status.json** | ❌ 缺 |
| **README_BASELINES.md** | ❌ 缺 |

**风险点(必须写进论文口径)**:现有 4 个全部是**诊断基线**,**不能冒充"4 个其他生成方法"**。
GT oracle 只是**上界**,不是可部署方法。目前**没有任何外部生成基线被运行过** → 论文**不得**写"已完成与外部方法的比较"。

一处语义说明:`run_trajectory_eval.py` 给 `deterministic_translation` 喂的 `scene_image` 是 H5 里的
`image` 字段,而该字段是**首帧(源窗口)**(`build_dataset_from_benchmark.py` `image = _resize_frame(frames[0], target_size)`),
非全 512 场景 → 它实际是"**平移源窗口 + 反射填充**"语义(方向准、质量差),正是要的对照;若要全场景版需另存全景图。

---

## 4 第二数据集 RESISC45(师兄意见 5)

| 项 | 状态 | 证据 |
|---|---|---|
| builder tier 支持 | ✅ | `build_dataset_from_benchmark.py:21-22`(tier 说明)、`:81` `RESISC45_SCENE_GROUPS`、`:107` `HD_DATASET_CONFIG_RESISC45` |
| common.sh 文档 | ✅ | `common.sh:45-46`(`DATASET_TIER=resisc45` + `WORK=.../work_resisc45` 隔离) |
| 离线预取 | ✅ | `offline_prefetch.sh` `TIER2=1` → RESISC45 + PatternNet |
| **PBS 作业** | ❌ **完全没有**(仅 README 提到) |
| **smoke test / H5 完整性 / 类别分布 / split 检查** | ❌ 缺 |
| **独立 WORK 硬守卫**(拒绝指向旧 AID work) | ❌ 缺 |
| **README_RESISC45.md** | ❌ 缺 |
| RESISC45-only vs +PatternNet 区分 | 🟡 仅 `TIER2=1` 一把梭下载两个 | 需拆开、默认不下 PatternNet |

---

## 5 多种子与统计(师兄意见 7)

| 项 | 状态 |
|---|---|
| 训练侧 seed 42/43/44 | ✅ `common.sh` `SEED` 可覆盖;PBS 20-25 已是 r16 sem/geo × 3 种子 |
| `SPLIT_SEED=42` 固定 | ✅ `train.py --split_seed` + PBS 全部 `export SPLIT_SEED=42` |
| 同 val manifest | 🟡 靠"split 确定性"隐式保证(同 H5 + 同 SPLIT_SEED → 逐字节相同 split),**未做显式校验** |
| 同 GEN_SEED | ✅ `common.sh` 传 `${GEN_SEED:-42}` |
| **聚合脚本**(每 seed 原始值 / mean±std / median / paired / bootstrap CI / permutation) | ❌ **完全缺失** |
| **缺 seed 时 fail-loud** | ❌ 缺 |
| CSV/JSON/Markdown/LaTeX 输出 | ❌ 缺 |

---

## 6 参数效率(师兄意见 8)

| 项 | 状态 | 证据 |
|---|---|---|
| `--lora_targets {all,attn_cam,attn_only}` | ✅ | `animate_utils.py:20-46`、`train.py:238/252/1013` |
| r8 / 更小 rank 可传 | ✅ | `--lora_rank` 是普通数值,无需改码 |
| 训练启动打印 total/trainable/ratio | 🟡 | `train.py:277` 有 `n_trainable/n_total` 日志,但**未落盘**、无 ratio 之外的分解 |
| **每类模块可训参数分解** | ❌ 缺 |
| **主干意外解冻检测 + 报错** | ❌ **缺**(这是参数效率声明的命门:若主干被解冻,整个"0.55% 参数"结论作废) |
| **param_audit.json 落盘** | ❌ 缺 |
| **parameter_efficiency_summary.py**(rank×targets 表 + 峰值显存 + walltime + 指标) | ❌ 缺 |
| 已实测参数量 | ✅(旧 AID 跑,记忆/日志):r16=102.9M/0.55%、r32=205.8M/1.10%、r64=411.6M/2.17%、r128=823.1M/4.25%(每 expert) |

---

## 7 PBS / 目录安全(第八条)

| 要求 | 状态 |
|---|---|
| DRYRUN=1 | 🟡 `qsub_corrected_rerun.sh` / `qsub_all.sh` 有;**新脚本需一并支持** |
| 独立 WORK | 🟡 机制有(`WORK` env),**缺"拒绝指向旧 AID work"的硬守卫** |
| 不删旧 AID 输出 | ⚠️ **注意**:`qsub_corrected_rerun.sh` 会 `rm -f $STAMPS/p1_build.done $GEO_H5 $SEM_H5` —— 这是**为修正 prompt 重建而设计的**,但它**删的正是旧 AID 数据集 H5**。在"旧 AID 结果不得覆盖"的新约束下,**此脚本必须改为写入独立 WORK,不得删除旧 H5**(见"未解决"清单) |
| 无 mv44 | ✅ 0 命中 |
| 磁盘/inode 预检 | ❌ 缺(qk73 容量紧、rx40 inode 紧) |
| 依赖/H5/checkpoint 存在性检查 | 🟡 `_pbs_env.sh` 有存在性守卫;数据/权重级检查缺 |
| 失败非零退出 | 🟡 部分(`set -euo pipefail` 在编排脚本有) |

---

## 8 建议修改(按优先级)

**P0(不做完不能信任任何 Track B 数字)**
1. 服务器上跑 `trajectory_metrics.py` / `diagnostic_baselines.py` 自测 —— 这是它们**第一次真正运行**。
2. 补齐 Track B 三个缺失指标 + 5 个缺失合成测试(尤其 **(6) 终点对但中途错**)。
3. 合并重复的 Farneback 计算。
4. **改掉 `qsub_corrected_rerun.sh` 删除旧 AID H5 的行为**,改为独立 WORK。

**P1(师兄意见直接对应)**
5. baseline registry/adapter + status.json + README_BASELINES(并明确"外部生成基线尚未运行")。
6. RESISC45:PBS + check 脚本 + 独立 WORK 守卫 + README。
7. `aggregate_seeds.py`(缺 seed fail-loud + 两种不确定性分列)。
8. 参数效率:`param_audit.json` + 主干解冻报错 + summary 表。

**P2**
9. 磁盘/inode 预检写进所有新 PBS。
10. `run_trajectory_eval.py` 的 valid% 与 Track B 聚合并入统一报表。

---

## 9 对旧 AID 结果的影响声明

- 本次审计**未修改任何**旧 AID 的 H5 / checkpoint / eval 输出 / `summary.md`。
- ⚠️ **唯一风险点**:既有的 `server_run/pbs/qsub_corrected_rerun.sh` 设计上会删除 `$GEO_H5`/`$SEM_H5`
  与 `p1_build` stamp(为重建修正-prompt 数据集)。在新约束下**该脚本不应被原样执行**,
  必须先改为独立 WORK 根目录。**在改好之前,请勿运行该脚本。**

---

## 10 本轮自查中发现并修复的**真 bug**(非需求,是既存/新引入的缺陷)

| # | Bug | 影响 | 证据 | 修复 |
|---|---|---|---|---|
| B1 | `evaluate.py` 内联方向解析只认旧措辞 `"eastward"` | **修正 prompt(`"toward the east"`)下所有 `dir_E/N/W/S` 分方向统计静默变 None、整块消失** | 已实证:新 prompt 喂内联解析器 → `None` | 抽出单一真源 `parse_direction_word_from_prompt`,两处共用;新旧措辞均验证通过 |
| B2 | 4 处 `compute_all_metrics` 传**全量 videos + 过滤后 angles** | `zip` 按短的截断 → **视频与目标角整体错位、角误差全错且不报错**;`valid_angle_videos`/`s_valid_vids` 算了从不用(死代码) | `evaluate.py` overall/scene/dir/dir×scene 全部命中 | 改为传**与视频对齐的全量角度**(None 占位),`compute_batch_angular_error` 跳过 None 且**长度不等即硬失败** |
| B3 | 逐样本指标被丢弃 → **逐轨迹 bootstrap 无数据** | 师兄要求的"测试轨迹 bootstrap CI"对主配置不可用 | `compute_batch_angular_error` 内部算出 `errors` 却只返回 mean/std | `compute_all_metrics` 加**可选出参** `per_sample_out`(不破坏任何既有调用方);`evaluate.py` 落**单个** `scores.json`;已实证能被 `aggregate_seeds --per-traj` 消费 |
| B4 | 11 个融合训练作业**不清理 epoch checkpoint** | 历史上该项吃 ~160GB,档2 是 11 训练 → **撑爆 qk73** | 作业 20-30 无清理逻辑 | 全部加清理:**双 stamp 完成后**、**只删本作业独立 `$OUT` 内**、保留 `best_checkpoint`、不触碰旧 AID |
| B5 | `train.py` 无显存打点 | 参数效率表要的 peak GPU memory **恒为 null** | 全文件 0 处 `max_memory_allocated` | `train()` 开头 `reset_peak_memory_stats`,收尾写 `peak_gpu_mem_allocated/reserved_gb`;汇总脚本改为真读 |

> B1/B2 会**静默产出错误数字**(不报错),是最危险的一类;B2 更是既存缺陷,与本次 prompt 修改无关。
