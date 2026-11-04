# REVIEWER_STYLE_EXPERIMENT_REVIEW — RSWM 实验缺口的审稿人视角复核

> 对应师兄意见第 9 条。生成时间:2026-07-15
> **审计方式**:逐行读代码 + `python3 -m py_compile` + `bash -n` + 静态逻辑追踪。
> **本沙箱无 numpy / cv2 / h5py / torch / pip**(已实测:`import numpy` → `ModuleNotFoundError`),
> 因此**任何依赖数值运行的结论都未经运行验证**,凡属推断处均已显式标注。
> **本文件不修改论文正文**,只陈述"代码/产物事实"与"审稿人会问什么"。
>
> **实际执行过的检查(仅此四项)**:
> 1. `py_compile`:`trajectory_metrics.py`、`run_trajectory_eval.py`、`validate_direction_metric.py`、
>    `aggregate_seeds.py`、`parameter_efficiency_summary.py` → 全部通过。
> 2. `bash -n`:`common.sh`、`qsub_corrected_rerun.sh`、`server_run/pbs/2*.pbs`、`3*.pbs` → 全部通过。
> 3. 产物清点:`find` 全仓搜 `eval_results.json` / `summary.md` / `*.h5` / `work*` 目录。
> 4. `grep` 事实核对(mv44、α、rank、prompt 约定)。
>
> **未执行**:`trajectory_metrics._self_test()`、`diagnostic_baselines._demo()`、任何训练/评估/GPU 作业、任何 qsub。

---

## 0 一句话结论

论文正文里**每一个数字**都来自"修正前"的管线(旧 prompt 模板 + 旧角度约定 + 旧 builder),
而这些产物**不在本仓库内**;仓库里唯一存在的 `eval_results.json` 是一份 **n=5 的冒烟产物,
FID/FVD 均为 NaN**。修正版重跑**一次都没跑过**,且其超参(α=rank)与论文声明(α=64)**不一致**,
rank 覆盖(8/16/128)也**无法重建论文的 r∈{16,32,64,128} 消融表**。
当前状态下,**论文的主表、消融表、附录角度误差全部处于"无仓内证据"状态**。

---

## 1 论文当前的主要 claim(逐条标注证据配置/指标)

以下 claim 逐条摘自 `experiments.tex`(未改动该文件)。

| # | Claim(论文原意) | 出处 | 该 claim 需要的证据配置 / 指标 |
|---|---|---|---|
| **C1** | LoRA(r=128)相对 Baseline "在**每一个指标、每一个场景**上一致提升" | `experiments.tex:25`(`tab:main_results` 引文段) | `eval_r128_sem` vs `eval_base`,SSIM/PSNR/LPIPS/FID/FVD × {overall, flood, general, rural, urban} |
| **C2** | FVD 79.0→31.3(−60%)= "显著提升空间连贯 rollout 能力" | `experiments.tex:60` | FVD(overall),`eval_r128_sem` vs `eval_base` |
| **C3** | LPIPS 0.623→0.550(−12%)= 感知质量一致提升 | `experiments.tex:60` | LPIPS(overall) |
| **C4** | urban 增益最大(SSIM +51%、FVD −53%),归因于"域差最大" | `experiments.tex:64` | 逐场景 SSIM/LPIPS/FVD;**归因**需要额外证据(见 §3) |
| **C5** | rank 与性能**非单调**;r=16 ≈ r=128 且均优于 r=32/r=64;r=16 "仅加 **~0.3%** 可训参数即捕获大部分领域知识" | `experiments.tex:94` | `tab:ablation_rank` 四行 + **参数量审计** |
| **C6** | 选 r=128 作主实验,因其"整体指标平衡最好" | `experiments.tex:94` | 同上 |
| **C7** | 增益在**生成质量**而非**方向可控性**;角度误差在所有配置下"维持 ~80–90°",故 prompt 式方向编码**不产生有意义的控制** | `experiments.tex:108`(Discussion)+ `Appendix:app:angular_error` | `Angular_Error_Mean` × 全配置 |
| **C8** | RSWM v5 benchmark = 扩展 RSWISE:(1) 连续 0–360° 方向条件、(2) 视频级 rollout、(3) 多段轨迹 overlap<10%;GT 由 **96×96 窗口**在 512×512 上滑动生成 | `experiments.tex:17` | benchmark 构建器参数 |
| **C9** | 数据量消融(100/200/500) | `experiments.tex:98-101` | **表格不存在**(源码里是 `% TODO: Insert Table 3`) |
| **C10** | 验证集 100 样本按场景/方向分层;baseline 因 15 例推理失败只评了 85;"**在共同的 85 例上做对齐比较结论一致**" | `experiments.tex:20`(脚注) | 对齐比较产物(`eval_aligned.py` 输出) |

---

## 2 逐 claim:仓库里**此刻**有没有 DIRECT 实验证据?

**先说判据**:本节严格区分 **"代码存在"** 与 **"结果存在"**。审稿人不接受前者当后者。

### 2.0 全仓产物清点(已实跑 `find`)

| 查找目标 | 结果 |
|---|---|
| `server_run/work/`(旧 AID 结果根) | **不存在于本仓库**(旧结果在服务器上,不在此 checkout 内) |
| `eval_results.json` | **仅 1 份**:`eval_output_165035647.gadi-pbs/test_5/eval_results.json` |
| `summary.md` | **0 份** |
| `trajectory_eval*.json` / `direction_validation*.json` / `seed_summary*` / `param_audit*.json` / `param_efficiency*` | **全部 0 份** |
| `*.h5`(数据集) | **0 份** |

**唯一存在的那份结果**(`eval_output_165035647.gadi-pbs/test_5/eval_results.json`)的实际内容:

- `num_samples: 5`、`Angular_Error_Count: 5`
- `"FID": NaN`、`"FVD": NaN`(**两个分布级指标在 n=5 下直接失效**)
- 无 `LPIPS` 键
- 同级 `full/generated/` 只有 **85** 个 `.npy`

且其 meta(`test_5/meta/sample_00036.json`)为:
```json
{"prompt": "A satellite video of a general remote sensing scene, moving northward with 2 direction changes.",
 "direction_angle": 90.0, "num_frames_gt": 18, "num_frames_gen": 17}
```
**两处致命的"陈旧"标记**:
1. **旧 prompt 模板**("moving northward with 2 direction changes"),而修正版模板是
   `build_dataset_from_benchmark.py:602-610` 的
   `"A satellite video of {desc}, following a {K}-segment trajectory with net displacement toward the {DIR}."`
2. **旧角度约定**:`direction_angle: 90.0` 对应 "northward" = **相机角**。而当前
   `evaluate.py:76-101` 的 `parse_direction_angle_from_prompt` 返回 `(camera_angle + 180) % 360`
   → 现版本对 "northward" 应返回 **270.0**。

**结论:这份产物由修正前的 `evaluate.py` 产生,在当前约定下已作废,不能作为任何 claim 的证据。**

### 2.1 逐 claim 判定

| # | 直接证据? | 事实依据 |
|---|---|---|
| **C1** | ❌ **无** | 主表数字不在仓内。**且论文自身在 `experiments.tex:62` 承认 FID 在 flood(230.5→263.1)与 general(213.2→225.8)**上升**** → C1 的"每一个指标每一个场景一致提升"**与同页表格自相矛盾**(见 §3-O1) |
| **C2** | ❌ **无** | 无 `eval_results.json`;且 FVD 的提取器是非标准代理(§8-B) |
| **C3** | ❌ **无** | 同上;唯一那份产物连 LPIPS 键都没有 |
| **C4** | ❌ **无**(数字)/ ❌ **无**(归因) | 逐场景数字不在仓内;"域差最大"是**解释**,仓内**无任何域差度量**(无特征距离、无 pretrain 分布分析) |
| **C5** | ❌ **无** | ①`tab:ablation_rank` 数字不在仓内;②修正重跑**没有 r=32 / r=64 作业**(见 §2.2)→ **该表在修正管线下不可重建**;③"~0.3%" 与本仓 `REVIEW_GAP_REPORT.md:147` 记录的实测 **r16 = 102.9M / 0.55%** **对不上**,且 `param_audit.json` **不存在**,无从裁决 |
| **C6** | ❌ **无** | 依赖 C5 的表 |
| **C7** | ❌ **无**(数字);⚠️ **结论方向可能反而更糟** | 详见 §7-S4。"~80–90°" 是**旧约定**下的数;`evaluate.py:84-86` 明确记载两种约定满足 **e → 180 − e 的精确映射** → 旧约定 80–90° 在修正约定下**恒等映射为 90–100°**,即**从"≈随机"变成"≈随机或略差于随机"** |
| **C8** | ❌ **与代码冲突** | 论文写 **96×96** 窗口;canonical builder 是 `build_dataset_from_benchmark.py:205-206` `DEFAULT_WINDOW_HEIGHT=128` / `DEFAULT_WINDOW_WIDTH=222`,target `(480,832)`(`:717`)。`common.sh:7-8` **明文裁决**:canonical = `build_dataset_from_benchmark.py`(window 128×222),"**绝不使用 `new/build_from_remotesense.py`(96×96/eurosat,非论文管道)**" → **论文的 setup 段描述的是被明令禁用的那条管线** |
| **C9** | ❌ **表不存在** | `experiments.tex:101` 就是一行 `% TODO: Insert Table 3 when 100/200 sample results are available.`;**且修正重跑没有任何 100/200 样本量作业** |
| **C10** | ❌ **无** | "aligned comparisons ... yield consistent conclusions" 在仓内**无任何产物支撑**;`common.sh:172-178` 有 `aligned()` 封装,但**没有任何 PBS 调用它**(已 grep 2*/3* 全部 pbs:只有 `train_lora` / `eval_lora` / `eval_base`) |

### 2.2 修正重跑:**作业存在,结果为零**

`qsub_corrected_rerun.sh` 的 `TRAIN` 数组(已读)= 11 个配置:

```
20_r16_sem_s42  21_r16_sem_s43  22_r16_sem_s44
23_r16_geo_s42  24_r16_geo_s43  25_r16_geo_s44
26_r128_sem_s42 27_r128_geo_s42
28_r16_sem_attncam 29_r16_geo_attncam 30_r8_sem_s42
```
依赖链:`01_build_data` → (afterok) 11 训练+评估 → `31_eval_base`(afterok:20,取其 split)。

**已实测的三处 claim-blocking 事实**(grep `train_lora` 全部 pbs):

1. **α = rank,不是 α = 64**:`20:17` `train_lora r16_sem_s42 **16 16**`、
   `26:17` `train_lora r128_sem_s42 **128 128**`、`30:17` `train_lora r8_sem_s42 **8 8**`,
   与 `common.sh:17` "`alpha=rank(=各 ablation pbs)`" 一致。
   而 `experiments.tex:11` 与 `tab:ablation_rank` caption(`:76`)均声明 **α=64 固定**。
   → **修正重跑不会复现论文声明的超参**。二者必须择一改口径(改论文正文不在本文件职责内,仅提示)。
2. **rank 覆盖 = {8, 16, 128}**,**无 r=32 / r=64**。
   `06_train_r32.pbs` / `07_train_r64.pbs` / `09` / `10` 存在于 `pbs/`,但**不在** `qsub_corrected_rerun.sh` 的
   `TRAIN` 列表里 → 属旧管线遗留 → **`tab:ablation_rank` 的 r=32/r=64 两行在修正管线下无对应作业**。
3. **多种子只覆盖 r16 sem/geo**(seed 42/43/44),r128 / r8 / attn_cam 均**只有 seed 42** → 只能对 r16 报 mean±std。

**Track B / 基线 / 第二数据集的执行状态**:

| 项 | 代码 | 结果 |
|---|---|---|
| `trajectory_metrics._self_test()`(`:1241-1545`,8 组场景 S1–S8 全在) | ✅ 存在且 `py_compile` 通过 | ❌ **从未运行**(本沙箱无 numpy;E2E 段还需 cv2,`:1460-1466` 会 SKIP) |
| `run_trajectory_eval.py` | ✅ | ❌ 无输出(`trajectory_eval.json` 0 份) |
| `validate_direction_metric.py`(GT 地板校准) | ✅ | ❌ 无输出 → **§8 里所有"地板/门控"数值都是预期值,不是实测值** |
| 4 个**诊断**基线 | ✅ | ❌ `baseline_status.json` 中 **每一条** `"tested": false, "result_available": false` |
| 3 个**外部生成**基线(mvsplat360 / image_outpainting / rswise_recursive) | ❌ `implemented: false` | ❌ 无 |
| RESISC45 第二数据集 | 🟡 builder 贯通 | ❌ `qsub_corrected_rerun.sh` 里**无任何 RESISC45 作业** |

> `baseline_status.json:4` 自己就写了:
> *"Do NOT claim a completed baseline comparison in the paper on the basis of this file."*

---

## 3 审稿人**大概率**会提的反对意见(具体、技术性)

**O1 — 主表结论与主表数据自相矛盾。**
`experiments.tex:25` 写 "consistent improvements on **every metric and every scene type**",
但同节 `:62` 自认 FID 在 flood 与 general **上升**。审稿人一眼就能抓。
现有的辩解("per-scene 20–30 样本使 FID 噪声大")**反而更致命**:它等于承认**表里 FID 那两列不可信**,
那么 overall FID 179.4→168.0 凭什么可信?**同一个理由不能只用来解释不利数据。**

**O2 — FID/FVD 在 n=25/100 下不在指标的稳定区间(可证伪的维度论证)。**
- FID:`metrics.py:130-131` 用 `inception_v3` + `model.fc = Identity()` → **2048 维**特征;
  `metrics.py:360-364` 每条视频**只取中间帧一帧** → overall n=100、per-scene n≈20–30。
  `metrics.py:153-154` 用 `np.cov` 估 2048×2048 协方差,而样本数 100 → **协方差秩 ≤ 99 ≪ 2048**,
  per-scene 秩 ≤ 29。`:157` `sqrtm(sigma_gen @ sigma_gt)` 作用在**严重奇异**的矩阵积上。
- FVD:`metrics.py:171-174` 用 `r3d_18` + `fc = Identity()` → **512 维**;n=100 → 秩 ≤ 99 ≪ 512。
- **仓内实证**:唯一那份 `eval_results.json` 在 n=5 下 `FID: NaN`、`FVD: NaN`
  ——`metrics.py:366`/`:373` 的 `except` 把 `sqrtm` 的失败吞成 NaN。
  这直接演示了小 n 下该实现会退化。**n=5 崩成 NaN、n=25 恰好"不崩"** 不等于 n=25 可信。
> 审稿人的措辞会是:"作者报告的 FID/FVD 绝对值在此样本量下是**有偏估计**,论文却据其做了 −60% 的定量声明。"

**O3 — FVD 用的不是 I3D,却叫 FVD,并与文献并列。**
`metrics.py:168` 函数名 `_load_i3d_model`、docstring 写 "加载 **I3D** 模型",`:171-174` 实际加载
**`torchvision.models.video.r3d_18`**(Kinetics-400 预训练 ResNet3D-18),返回 `name = "r3d_18"`;
`:198-200` 把 clip 重采样到 **16×112×112**。标准 FVD 用 I3D(Kinetics-400,400-logits/1024-d pool),
输入 224×224。**两者的特征空间不同 → 数值与文献 FVD 不可比。**
FVD **31.3** 这个量级本身就会让熟悉 FVD 的审稿人起疑(标准 I3D-FVD 在此类任务通常量级更大),
从而怀疑整套 metric 实现。

**O4 — 角度误差 ≈ 90° 意味着"无控制",而 −60% FVD 与之并置会被读成"刷了保真度、丢了任务"。**
论文 `:108` 已诚实指出这点,但**主表把 FVD/LPIPS 的提升放在最前面**,而 benchmark 的**存在理由**
(方向可控性)是失败的。审稿人会问:**"如果方向完全不可控,那么这个方法解决了它自称要解决的问题吗?
一个不受方向控制的 remote-sensing video LoRA,与一篇普通的 domain-adaptation LoRA 报告有何区别?"**
这是**最可能致命**的一条,且它是**科学问题,不是代码问题** —— 补跑任何实验都不会自动解决。

**O5 — 没有任何外部方法比较。**
论文 `:117` 的 Limitations 已承认。但审稿人会追问:仓库里**有** `evaluate_mvsplat360.py` + 安装器 +
`server_run/p6_mvsplat.sh`(见 `baseline_status.json:49`),为什么不跑?
而 `baseline_status.json:48` 给出的**真实**阻塞是:repo 未 clone、三个 ckpt 全缺、`svd.safetensors` 需 HF 授权、
需要 torch==2.2.2 的独立 env(与主环境 torch 2.4/2.6 + flash-attn 冲突)、record 契约桥接未写。
**额外雷点**:`evaluate_mvsplat360.py:11-14` 与 `:478` 仍**硬编码已退役的 mv44 路径**
(`/g/data/mv44/...`、`:14` 甚至是 `/scratch/mv44/...`)→ 该脚本**当前不可执行**。
(注:`REVIEW_GAP_REPORT.md:42` 声称"全 server_run grep mv44 → 0 命中"。**实测 6 命中**,
但**全部是"警告不要用 mv44"的说明文字**,非活路径 → `server_run/` 本身合规;
**活的 mv44 路径在 `lora_remotesense/evaluate_mvsplat360.py`**。)
另注:MVSplat360 拿 **5 个 context views + poses**,条件量**多于** RSWM(`baseline_status.json:49`)
→ 即便跑出来,**也不是公平对照**,必须在正文里点明,否则审稿人会说"你挑了个占便宜的对手/输给了拿更多信息的方法"。

**O6 — "aligned comparisons on the common 85 samples yield consistent conclusions" 是无据之言。**
`experiments.tex:20` 脚注的这句话,仓内**无任何产物**。`common.sh:172-178` 有 `aligned()`,
但**没有任何 PBS 调用它**。审稿人会直接要这张表。
更糟的是:**baseline 恰好在 15 例上推理失败**,这**不是随机缺失** —— 失败样本很可能是最难的那批,
**baseline 的 85 例均值因此被系统性高估**(幸存者偏差)。论文却用它当对照的分母。

**O7 — setup 段描述的构建器不是实际用的构建器。**
论文 `:17`:96×96 窗口 / 512×512 / `RandomPositionPathGenerator`。
代码:`build_dataset_from_benchmark.py:205-206` = **128×222**,target `(480,832)`(`:717`),
`common.sh:7-8` **明令禁用** 96×96 那条路径。审稿人若照 setup 段复现,**必然复现不出**。

**O8 — 论文声明的 α=64 与将要跑的 α=rank 冲突;rank 消融表无法重建。**
见 §2.2-1/2。审稿人若索要修正后的 `tab:ablation_rank`,**r=32/r=64 两行拿不出来**。

**O9 — Track B 聚合器把新补的指标全丢了(实测代码事实)。**
`run_trajectory_eval.py:49-50`:
```python
TRACKB_KEYS = ["net_direction_error", "segment_angular_mean", "turn_recall",
               "turn_timing_error", "waypoint_error", "frechet", "dtw", "flow_magnitude"]
```
而 `score_trajectory` 实际输出(`trajectory_metrics.py:1082-1107`)还包含
`segment_success_rate_30/45`、`turn_precision`、`turn_f1`、`waypoint_endpoint_error`、
`num_gt_turns`、`num_recovered_turns`。
`_flatten`(`:53-58`)**只挑 TRACKB_KEYS** → **这 6 个键在聚合报表里被静默丢弃**。
即:师兄意见 2/3/4 要求补的 `segment_success_rate` / `turn_precision` / `waypoint_endpoint_error`
**在指标层已实现,但在报表层拿不到**。**这是当前最容易被忽视、又最容易修的一条。**
(`REVIEW_GAP_REPORT.md:56/58/60` 仍把这三个标为 ❌ 缺失,与 `trajectory_metrics.py:444`/`:647`/`:709`
的实际实现**不符** → 该审计报告的 §2 表已陈旧,勿再据其判断。)

**O10 — Track B 的均值是"条件于 valid"的,存在幸存者偏差。**
`run_trajectory_eval.py:40-41`:`vals = [d[k] for d in dicts if d.get(k) is not None]`。
静止/塌缩样本在 `score_trajectory` 里 `valid=False` → 所有方向/路径字段为 `None`
(`trajectory_metrics.py:1119-1120`)→ **被排除出分母**。
后果:**一个在难样本上直接塌成静止的模型,其 `segment_angular_mean` 反而更漂亮**。
`valid_pct`(`:43-45`)虽被报告,但**它与误差均值必须联读**,否则排名可被"塌缩"策略刷。
**任何 Track B 表格都必须把 `valid_pct` 和 n 印在同一行。**

**O11 — text-only 的判定靠目录名字符串匹配。**
`run_trajectory_eval.py:115`:`text_only = ("textonly" in name.lower() or "sem" in name.lower())`。
**`commanded` 标志(决定一个数是"达标量"还是"诊断量")由命令行传的目录别名决定**,
而非由数据/条件类型决定。别名拼错 → 整张表的语义标注翻转。审稿人不会看到这行,
但**内部必须视为高危**:它决定了"我们没声称 text-only 能跟折线"这一防线是否成立。

**O12 — 诊断基线的坐标系可能不自洽(静态推断,未运行,须首跑验证)。**
`run_trajectory_eval.py:136-138` 传 `scene_image=g["image"]`。
而 `build_dataset_from_benchmark.py:770` `image = _resize_frame(frames[0], target_size)`
→ **`image` 是被 resize 到 (480,832) 的首帧窗口,不是 512×512 全景**。
但同时传的 `waypoints`(`:805`,原始场景像素 (cy,cx))与 `window_size`
(`:751` `pose_window_size = (win_h, win_w)` = (128,222))**都在 512-场景坐标系里**。
→ `deterministic_translation` 会在一张 480×832 的**首帧**上、按 **512-场景坐标**的 waypoints
裁 128×222 窗口。
*方向*大体仍对(128:222 ≈ 480:832,`:206` 注释明写该比例是为匹配 480:832 而选,故 resize 近似等比),
但**位移幅度的单位是错的**。
影响面:形状/分段/结点类指标因 `_length_normalize` 归一化(`trajectory_metrics.py:159-169`)而**尺度不变**,
基本免疫;但 **`flow_magnitude`(px/frame)与依赖它的 `min_mag=5.0` 有效性门控
(`trajectory_metrics.py:1116`)不是尺度不变的** → `deterministic_translation` 的 valid% 可能失真。
`REVIEW_GAP_REPORT.md:101-103` 已注意到"平移源窗口"语义,但**未指出坐标系单位不匹配**。
**⚠️ 这是静态读码推断,本沙箱无 numpy/h5py 无法验证 —— 列为首跑必查项,不得当既定结论。**

**O13 — Track B 的样本覆盖可能静默 < 100。**
`build_dataset_from_benchmark.py:804` 仅在 `waypoints is not None and len(waypoints) >= 2` 时持久化轨迹;
`run_trajectory_eval.py:76-77` 对无 `waypoints` 的 key **静默 `continue`**。
→ Track B 的 n **可能小于主表的 n**,且**不 fail-loud**。两张表的 n 不同却并排放 = 审稿人必抓。
**Track B 表必须显式印 n,并与主表 n 对账。**

---

## 4 **必须**进正文的实验

前提:全部基于**独立 WORK**(`qsub_corrected_rerun.sh` 的 `WORK_CORRECTED`,已有拒绝指向旧 AID work 的硬守卫),
**不得触碰 `server_run/work/`**。

| # | 实验 | 理由(对应哪条审稿人反对) | 依赖 |
|---|---|---|---|
| **M1** | **方向指标校准表**:`validate_direction_metric.py` 的 A(GT 地板 vs content 角 / vs camera 角)+ A2(人工平移恢复) | **一切方向数字的前置合法性**。若 GT 地板不≈22.5°、A2 恢复误差不≈0°,则**角度指标本身不成立**,C7 连"失败"都不能声称。**必须先于任何 GPU 作业在登录节点跑** | 新 H5(01) |
| **M2** | **修正 prompt + 修正约定下的主表**:`eval_r128_sem` vs `eval_base` | C1/C2/C3 的全部数字**必须重出**;旧数字在新约定下作废 | 26, 31 |
| **M3** | **Track B 全轨迹可控性表**(text-only vs text+Plücker),含 `valid_pct` 与 n | 师兄意见 1–4 的核心;也是 O4 的**唯一**正面回应:即便净方向失败,**逐段/拐点**能否分离出信号,决定论文还有没有故事 | 20, 23,+ **修 O9** |
| **M4** | **4 个诊断基线行**(persistence / det-translation / random-direction / gt-oracle),**显式标注为诊断基线与上界** | 证明 benchmark **能区分**四类失败模式。**没有它,Track B 的任何数字都无参照系** | M3 同批 |
| **M5** | **多种子 mean±std**(r16 sem/geo × seed 42/43/44)+ paired test | 师兄意见 7;C5 的"非单调"若在种子噪声内则**该 claim 直接撤销** | 20-25 + `aggregate_seeds.py` |
| **M6** | **参数效率表** + `param_audit.json`(逐类模块分解 + **主干解冻检测**) | C5 的 "~0.3%" **当前与本仓记录的 0.55% 冲突**且无产物可裁决;主干若被解冻则整个参数效率结论作废 | 20/26/30 + `parameter_efficiency_summary.py` |
| **M7** | **对齐比较表(共同 85 例)** | `experiments.tex:20` 脚注已**声称**结论一致 → **必须兑现**,否则删掉那句话。另需报告 15 例失败的**场景分布**以正面回应 O6 的幸存者偏差 | `aligned()`,**目前无 PBS 调用** |

> **M1 是 gate,不是可选项。** `qsub_corrected_rerun.sh` 末尾的提示也把它列为第 1 步,
> 且脚本注释(`:20`)明写"**前置:先在登录节点跑 第0步 校准 + 两个模块自测,全绿再提交**"。
> 现状:自测**从未跑过**。**在自测全绿前提交 11 个 40 小时 GPU 作业 = 拿 GPU 时长赌一个未验证的指标实现。**

---

## 5 属于**附录**的实验

| # | 实验 | 为何是附录 |
|---|---|---|
| **A1** | rank 消融(现实覆盖 r ∈ {8, 16, 128}) | 非单调性在 M5 的种子噪声界定前**不是可靠发现**;且 r=32/64 无修正作业(§2.2-2)→ 表不完整,正文放会被追问 |
| **A2** | `lora_targets` 消融(all vs attn_cam;28/29) | 参数效率的支撑证据,非主线 claim |
| **A3** | 逐场景分解(flood/general/rural/urban) | **per-scene n≈20–30,FID/FVD 在该 n 下不可信(O2)**。附录可留 SSIM/PSNR/LPIPS 的逐场景分解,**FID/FVD 逐场景列建议整列删除**而非"解释噪声大" |
| **A4** | Track B 的次级指标(`turn_timing_error`、`dtw`、`frechet` 全分布) | 正文放 `segment_success_rate` + `turn_f1` + `net_direction_error` 足够;DTW **只在同 T 下可比**(`trajectory_metrics.py:788-789`),放附录并注明 |
| **A5** | 合成自测 S1–S8 的通过表(`trajectory_metrics._self_test()`) | **指标可信度的证据**,审稿人会喜欢,但占正文空间。**注意:必须真跑过才能放** |
| **A6** | RESISC45 第二数据集泛化 | 师兄意见 5。**当前无任何作业**;若时间不够,**正文一个字都不能提泛化** |
| **A7** | 定性图 | 常规 |

---

## 6 低价值 / 可以砍的实验(附理由)

| # | 可砍项 | 理由 |
|---|---|---|
| **K1** | **逐场景 FID / FVD** | n≈20–30 下 2048-d / 512-d 协方差秩 ≤ 29(O2)→ **该数不携带信息**。论文现在用它的"噪声大"来解释不利结果,**砍掉整列比解释更诚实、也更省版面** |
| **K2** | **数据量消融(100/200/500,C9)** | 表本就不存在(`:101` TODO),且**无对应作业**。要补需 2 个新训练(各 ~40h GPU)。在 **M1 校准都没跑**、方向可控性根本失败的前提下,"数据量→质量"的曲线**回答不了任何审稿人关心的问题**。**建议直接删掉 §4.2.2 整节**,而不是补跑 |
| **K3** | **r=32 / r=64 补跑** | 2 × 40h GPU,只为填 `tab:ablation_rank` 两行。若 M5 显示 r16 与 r128 的差落在种子 std 内,**整个"非单调"叙事本就要撤**,补这两行等于给一个要撤的 claim 花 80 GPU 小时 |
| **K4** | **`image_outpainting` / `rswise_recursive` 外部基线** | `baseline_status.json:57`/`:66`:**没有选定模型、没有权重、只有 `NotImplementedError` 桩**(`diagnostic_baselines.py:449-480` / `:483-513`)。从零选模型+接管线+调参 ≫ 本轮预算。**明确写进 Limitations,不要假装在做** |
| **K5** | **MVSplat360 基线** | 阻塞项 5 条(`baseline_status.json:48`)+ 脚本仍指向已退役 mv44(`evaluate_mvsplat360.py:11-14`, `:478`)+ **它拿 5 views + poses,条件多于 RSWM,本就不是公平对照**(`:49`)。**性价比最低的一项**。若一定要一个外部对照,`gt_oracle` + `deterministic_translation` 作为**上界**已能撑起"benchmark 可区分"的论证 |
| **K6** | **PatternNet** | `offline_prefetch.sh` 的 `TIER2=1` 一把梭同时下 RESISC45 + PatternNet。RESISC45 本身已足够支撑"第二数据集"论证;**多一个数据集 = 多一份 qk73 空间 + 多一批 inode**,收益为零 |

> **总计可省约 160 GPU 小时(K2 + K3)**,且**不损失任何审稿人真正关心的 claim**。

---

## 7 **不得**夸大的结论(附:必须避免的确切句子)

### S1 — 不得声称"完成了与其他生成方法的比较"

- ❌ **禁写**:"We compare against four baselines."
- ❌ **禁写**:"RSWM outperforms existing remote sensing generation methods."
- ✅ **可写**:"We report four **diagnostic** baselines (persistence, deterministic translation,
  random direction, GT oracle) that probe distinct failure modes. **No external generative
  baseline has been run**; `gt_oracle` and `deterministic_translation` are **upper bounds**,
  not deployable methods."
- **依据**:`baseline_status.json:4`(*"Do NOT claim a completed baseline comparison..."*)、
  `:13`(gt_oracle = *"UPPER BOUND ..., NOT a deployable method"*)、
  `:22`(det-translation = *"a scene lookup, not generation"*)。

### S2 — 不得把 `gt_oracle` / `deterministic_translation` 与真方法并排排名

- ❌ **禁写**:任何把 gt_oracle 与 LoRA 放进同一个"Best in **bold**"竞争排序的表。
- ✅ **可写**:上界行单独分区、加水平线、注明 "upper bound (sees the full static scene)"。
- **依据**:`baseline_status.json:13`、`:22`(*"Needs the full static scene, which a real world
  model never sees."*)。

### S3 — 不得把 FVD 与文献 FVD 并列

- ❌ **禁写**:"our FVD of 31.3 is lower than that reported by [X]"
- ❌ **禁写**:任何把本文 FVD 放进跨论文对比表的写法。
- ✅ **可写**:"FVD is computed with an **R3D-18 proxy extractor** (`metrics.py:171`), not the
  standard I3D. Values are **comparable only across configurations within this paper** and
  **must not be compared to I3D-based FVD in the literature**."

### S4 — 不得沿用"角度误差 ~80–90°"这个数

- ❌ **禁写**:"angular error remains $\sim$80--90$^{\circ}$ for all configurations"
  (`experiments.tex:108` 现文)。
- **理由**:该数产自**旧约定**(目标=相机角)。`evaluate.py:84-86` 记载两约定满足 **e → 180 − e**
  的**精确**映射(折叠误差对称)。故旧的 80–90° 在**修正约定**下**恒等于 90–100°**。
  ⚠️ **这是对论文自报区间做代数变换,不是新测量;修正管线的真实数字必须重跑 M2 才知道。**
- **含义变化(必须让读者看到)**:90° = 随机;>90° = **系统性地朝反方向漂**,即
  **模型在逐字跟随方向词做内容漂移**(见 §8-C)。
  → 结论从"没有控制"可能变成"**有控制,但符号是反的**"。**后者是一个更强、也更有救的发现**
  (它意味着信号进去了、只是约定拧了),但**在 M2 跑出来之前,两种说法都不能写**。
- ✅ **临时可写**:`\TBD`,或 "under the corrected convention the angular error is re-measured;
  numbers pending."

### S5 — 不得声称"每个指标每个场景一致提升"

- ❌ **禁写**:"LoRA adaptation yields consistent improvements on every metric and every scene
  type."(`experiments.tex:25` 现文)
- **理由**:与同节 `:68` 自认的 FID 上升(flood 230.5→263.1、general 213.2→225.8)**直接矛盾**。
- ✅ **可写**:"LoRA improves SSIM/PSNR/LPIPS/FVD across all scenes; FID is **not reported
  per-scene** because at n≈20–30 the estimator is outside its stable regime (§8-A)."

### S6 — 不得把 text-only 的 Track B 分数说成它的失败

见 §8-E。
- ❌ **禁写**:"the text-only model fails to follow the per-segment trajectory"
- ❌ **禁写**:任何把 text-only 的 `waypoint_error` / `frechet` / `turn_recall` 加粗参与
  "谁更好"排序的表。
- ✅ **可写**:"For text-only conditioning these quantities are **diagnostic only**
  (`commanded=False`, `trajectory_metrics.py:1085`): the polyline was **never given** to the model."

### S7 — 不得声称"对齐比较结论一致"

- ❌ **禁写**:"aligned comparisons on the common 85 samples yield consistent conclusions."
  (`experiments.tex:20` 脚注现文)
- **理由**:仓内**无任何产物**;`aligned()`(`common.sh:172-178`)**无 PBS 调用**。
  **要么跑出来(M7),要么删掉这句。**

### S8 — 不得沿用 setup 段的 benchmark 描述

- ❌ **禁写**:"sliding a $96{\times}96$ window across $512{\times}512$ ..."(`experiments.tex:17` 现文)
- **理由**:canonical builder 是 **128×222** → **(480,832)**
  (`build_dataset_from_benchmark.py:205-206`, `:717`);`common.sh:8` 明令禁用 96×96 那条管线。

### S9 — 不得沿用 "α=64" 与 "~0.3% 参数"

- ❌ **禁写**:"$\alpha{=}64$"(`experiments.tex:11`, `:76`)—— 修正重跑全部是 **α=rank**(§2.2-1)。
- ❌ **禁写**:"adding only $\sim$0.3\% trainable parameters"(`experiments.tex:94`)
  —— 与 `REVIEW_GAP_REPORT.md:147` 记录的 **r16 = 0.55%** 冲突,且 `param_audit.json` 不存在。
  **等 M6 的 `param_audit.json` 落盘后照抄,不要凭记忆写。**

---

## 8 显式局限声明(FID / FVD / net-direction / Track B)

> 以下为**建议写进论文 Limitations / Metrics 小节**的措辞骨架,全部锚定代码行。

### A. FID — n=25/100 单中间帧,低于该指标的稳定区间

- **实现**:`metrics.py:130-131` `inception_v3` + `fc=Identity()` → **2048-d**;
  `metrics.py:360-361` **每条视频只取中间帧一帧**(`v[len(v)//2]`);
  `:153-154` `np.cov` 估 2048×2048 协方差;`:157` `sqrtm(sigma_gen @ sigma_gt)`。
- **样本量**:overall **n=100**(`experiments.tex:20`),per-scene **n≈20–30**(`:62`;`:117` 记为 20–25)。
- **数学后果**:样本协方差的**秩 ≤ n−1**(100 → ≤99;25 → ≤24)**远小于 2048** → 两个协方差都**奇异**,
  Fréchet 距离的 trace 项**系统性有偏**。FID 的偏差在小 n 下**随 n 单调、不随机**
  → **"多跑几个种子取平均"消不掉它**。
- **仓内实证**:唯一那份产物在 n=5 下 `FID: NaN`(`eval_output_165035647.gadi-pbs/test_5/eval_results.json`),
  由 `metrics.py:365-367` 的 `except → NaN` 产生。
- **建议措辞**:
  > "FID is computed from a **single middle frame per video** over **n=100** (overall) and
  > **n≈20–30** (per scene), against a **2048-dimensional** Inception feature. At these sample
  > sizes the empirical covariance is **rank-deficient** and the Fréchet estimator is
  > **strongly biased**; we therefore report FID **for reference only**, do **not** report it
  > per scene, and draw **no** conclusion from FID differences smaller than the estimator's bias."
- ⚠️ **未运行验证**:偏差的**具体量级**本沙箱无法测(无 numpy/torch)。**不要写具体偏差数字。**

### B. FVD — 非标准 R3D-18 代理提取器,仅内部可比

- **实现**:`metrics.py:168` 函数名 `_load_i3d_model`、docstring 写 "加载 **I3D** 模型",
  但 `:171-174` 实际加载 **`torchvision.models.video.r3d_18`**(Kinetics-400 预训练 ResNet3D-18),
  `fc=Identity()` → **512-d**,返回 `name="r3d_18"`;`:189-200` 采样 **16 帧 → 112×112**(trilinear)。
- **与标准 FVD 的差异**:标准 FVD 用 **I3D**(Kinetics-400)、输入 224×224、取 1024-d pool 特征。
  **特征空间不同 → 数值不可跨实现比较。**
- **样本量**:n=100 vs 512-d → 协方差秩 ≤ 99 < 512 → **同 A 的秩亏问题**。
- **建议措辞**:
  > "Our FVD uses an **R3D-18 proxy extractor** (`metrics.py:171`; 16 frames at 112$\times$112),
  > **not** the I3D network used by the standard FVD protocol. Our FVD values are therefore
  > **internally comparable across the configurations in this paper only**, and are
  > **not comparable to I3D-based FVD reported in the literature**. We do not claim any
  > cross-paper FVD comparison."
- **附带动作**:`_load_i3d_model` 这个**函数名与 docstring 本身就在误导**读者与未来的自己。
  (本文件只读不改 `metrics.py`;**建议**改名为 `_load_video_feature_model` 并修 docstring。)

### C. Net-direction 角度误差 — 修正约定 + 22.5° 量化地板

- **约定**:目标 = **相机方向 + 180°**,因为**内容流与相机运动相反**(窗口东移 → 画面内容西移)。
  代码三处一致:`trajectory_metrics.py:21-30`(模块 docstring)、`:1080`
  (`gt_net_content = (_camera_net_angle(wp) + 180) % 360`)、`:437`
  (`gt_content_ang = (seg_ang + 180) % 360`)、`evaluate.py:76-101`
  (`parse_direction_angle_from_prompt` 返回 `(camera_angle + 180) % 360`)。
- **量化地板 ≈ 22.5°**:prompt 只有 **4 个基数方向词**(`evaluate.py:93-100`:east/north/west/south),
  即命令被量化到 **90° 的格**;而 GT 轨迹的真实净角**连续**。真实角在格内近似均匀分布时,
  |偏移| 的期望 = **22.5°**。→ **即使模型完美跟随命令词,角度误差也不会低于 ≈22.5°。**
  `validate_direction_metric.py:8` 与 `:158` 明写该预期。
- **读数刻度**:
  | 数值 | 含义 |
  |---|---|
  | **≈22.5°** | 完美跟随(量化地板,**不可能更低**) |
  | **90°** | **随机**(与命令无关) |
  | **≈180°** | **逐字跟随方向词做内容漂移** —— 即模型把方向词当成了**内容**该走的方向,而非相机 |
- **⚠️ 地板从未实测**:`validate_direction_metric.py` 的 A 段(GT 地板)与 A2 段(人工平移恢复)
  **一次都没跑过**(仓内无 `direction_validation.json`)。22.5° / 157.5° 是**代码注释里的预期值**
  (`:8`, `:158`),**不是测量值**。
- **建议措辞**:
  > "Direction is commanded by one of **four cardinal words**, i.e. the command is **quantized to
  > a 90$^{\circ}$ grid** while ground-truth trajectories are continuous. The angular error
  > therefore has a **quantization floor of $\approx$22.5$^{\circ}$**: 22.5$^{\circ}$ indicates
  > perfect compliance, **90$^{\circ}$ indicates chance**, and $\approx$180$^{\circ}$ indicates
  > that the model drifts **content** along the direction word (the opposite of the commanded
  > camera motion). The target angle follows the corrected convention
  > (target $=$ camera direction $+$ 180$^{\circ}$), because **content flow is opposite to camera
  > motion**. **The GT floor is measured and reported in Appendix~\ref{...}** [— 必须真跑 M1 才能这样写]."
- **额外局限(Track B 存在的理由,应一并写)**:net-direction 把整段视频**折叠成一个角度**,
  只反映首尾净位移 → **绕行 / 漏拐 / 走过头都可能拿 ≈0°**
  (`trajectory_metrics.py:871-873` docstring 已明写此盲区)。**必须与 frechet/dtw/segment_* 联读。**

### D. Track B — 依赖光流积分路径,且有效性门控的 GT 地板从未测量

- **核心假设**:Track B 的**全部**路径类指标都建立在
  "**逐帧 Farneback 光流的幅度加权平均可以积分出可靠的内容路径**"之上。
  链条:`_compute_flows`(`:84-105`,step=1)→ `_step_vector`(`:108-123`,幅度加权平均)
  → `_path_from_flows`(`:126-136`,前缀和)。
  **积分会累积误差**;单步噪声在 T 帧后线性累积到终点。该假设**在真实生成视频上从未验证过**。
- **有效性门控**:`is_valid`(`:932-971`)/ `score_trajectory`(`:1116`)用
  **`min_mag = 5.0` px/frame**(`:983` 默认值)判定样本是否"动了";不过关 → `valid=False` →
  **所有方向/路径字段置 None 并立即返回**(`:1119-1120`)。
- **⚠️ 该门控的 GT 地板从未被测量**:
  1. 仓内无 `direction_validation.json` → `validate_direction_metric.py` 的 **B 段(GT 流幅度)从未跑过**。
  2. `trajectory_metrics.py:913-914` 写 "生成分辨率下 GT 约 **34 px/frame**",但**紧跟着的依据是
     「见任务约定」—— 即这是一个约定值,不是测量值**。
  3. **更麻烦**:两处 `flow_magnitude` **不是同一个估计量**——
     `trajectory_metrics.py:139-144` 用 **step=1** 的逐帧流取均值;
     `validate_direction_metric.py:44-53` 用 **step = len(frames)//8** 的跨帧流**再除以 step**(`:52`)。
     → **`min_mag=5.0` 这个阈值所在的刻度,与那个"34 px/frame"的刻度并不是同一把尺子。**
  → **`min_mag=5.0` 目前是一个未经校准的魔数。** 它同时决定了 `valid_pct` 和**哪些样本进入误差均值的分母**
    (O10 的幸存者偏差入口)。
- **建议措辞**:
  > "All Track B path metrics assume that the **flow-integrated content path** recovered by
  > dense Farneback optical flow is a reliable proxy for the generated camera trajectory.
  > Samples are gated by a **minimum mean flow magnitude** (`min_mag`, default 5.0 px/frame);
  > **samples below the gate are excluded from the error means**, so Track B means must always
  > be read together with the reported **valid\%** and **n**. The GT floor of this gate is
  > reported in Appendix~\ref{...} [— 必须真跑 M1/B 段才能这样写];
  > **until measured, the gate value is a convention, not a calibrated threshold**."
- **自测状态**:`_self_test()`(`:1241-1545`)结构正确、S1–S8 **八组场景全在**
  (S1 分段平移 `:1251`、S2 反向 `:1291`、S3/S4 转早/转晚 `:1348`、S5 整体平移 `:1396`、
  **S6 首尾对但中途绕远** `:1414`、S7 静止/persistence `:1519`、S8 随机方向 `:1440`),
  `__main__` 以 `sys.exit(_self_test())` 失败非零退出(`:1548-1550`)。
  **但从未运行**(本沙箱无 numpy;E2E 段还需 cv2,`:1460-1466` 会 SKIP)。
  → **"审读通过、运行未验证"。**
  (注:`REVIEW_GAP_REPORT.md:73` 称"8 个必需合成测试只覆盖 3 个" —— **该说法已过时**,现为 8/8。)

### E. text-only 路由 — 逐段/拐点/结点/路径指标**未被命令**,只能作诊断量

- **事实**:text-only 条件下模型**只收到**"**净方向词 + 段数**"。
  修正版 prompt(`build_dataset_from_benchmark.py:602-610`):
  `"A satellite video of {desc}, following a {K}-segment trajectory with net displacement toward the {DIR}."`
  → 里面**只有** `{K}`(段数)和 `{DIR}`(净方向,4 选 1)。
  **折线本身(waypoints / 逐段角度)从未进入 text-only 的条件。**
- **代码如何标注**:`score_trajectory(..., text_only=True)` → `commanded=False`
  (`trajectory_metrics.py:1085`:`"commanded": (not text_only)`);
  模块 docstring `:32-41` 明写 "text-only 只承诺**净方向与段数**"。
- **因此**:`segment_angular_*`、`segment_success_rate_*`、`turn_*`、`waypoint_*`、`frechet`、`dtw`
  对 text-only **不是达标量**。它们**照常被计算并报告**,但**比对的 GT 折线从未作为条件给过模型**。
- **必须的写法**:
  > "Under **text-only** conditioning the model receives **only the net direction word and the
  > number of segments**; the polyline is **never** provided. Consequently
  > **only `net_direction_error` and `flow_magnitude` are commanded quantities**. Per-segment,
  > turn, waypoint and path-shape metrics are reported for text-only **as diagnostics only**
  > (`commanded=False`) and **must never be read as a failure of that configuration** —
  > they measure compliance with an instruction that was never issued."
- ❌ **禁止**:任何在 text-only 与 text+Plücker 之间对 `waypoint_error` / `frechet` / `turn_recall`
  **加粗比较、宣布赢家**的表。若要并排,**必须**给 text-only 那些格子加
  "n/c (not commanded)" 标记,而不是数字加粗。
- ⚠️ **实现脆弱性(§3-O11)**:`commanded` 由 `run_trajectory_eval.py:115` 的**目录名字符串匹配**
  (`"textonly" in name or "sem" in name`)决定,**不是**由数据/条件类型决定。
  → **这条防线的正确性依赖于命令行别名拼写。** 出表前必须人工核对每个 `name` 的 `text_only` 判定。

---

## 附:本次复核发现的、与 `REVIEW_GAP_REPORT.md` **不一致**之处(供交叉校对)

| `REVIEW_GAP_REPORT.md` 的说法 | 本次实测 |
|---|---|
| `:56`/`:58`/`:60` `segment_success_rate` / `turn_precision` / `waypoint_endpoint_error` **❌ 缺** | **已实现**:`trajectory_metrics.py:444` / `:647` / `:709`(§0 表里标"已补",§2 表未同步)。**真正的缺口在报表层**(§3-O9) |
| `:73` "8 个必需合成测试只覆盖 3 个" | **8/8 全在**(`:1251`/`:1291`/`:1348`/`:1396`/`:1414`/`:1440`/`:1519`) |
| `:72` "Farneback 被重复计算 3 次" | **已优化为 2 次**(`trajectory_metrics.py:1062-1068` 效率节 + `:1111` 单次 `_compute_flows`) |
| `:42` "全 `server_run/` grep mv44 → **0 命中**" | **6 命中**,但**全部是"警告勿用 mv44"的说明文字** → `server_run/` 实质合规。**活的 mv44 路径在 `lora_remotesense/evaluate_mvsplat360.py:11-14`, `:478`** |
| `:130` 聚合脚本 **❌ 完全缺失** | `server_run/aggregate_seeds.py` 与 `server_run/parameter_efficiency_summary.py` **均存在且 `py_compile` 通过** |
| `:157` `qsub_corrected_rerun.sh` 会删旧 AID H5 | **已修**:现为独立 `WORK_CORRECTED` + 拒绝指向旧 work 的硬守卫 + 磁盘/inode 预检 + DRYRUN |

> **`REVIEW_GAP_REPORT.md` 的 §2/§3/§5/§6/§7 表格已陈旧于代码。** 判断实现状态请以本文件与源码为准。

---

## 附:立即可做、零 GPU 成本的三件事

1. **在服务器(rswm 环境)跑两个自测** —— 这将是它们的**第一次真正运行**:
   `python lora_remotesense/new/trajectory_metrics.py` 与
   `python lora_remotesense/new/diagnostic_baselines.py`。
   **全绿之前不要提交 11 个 40 小时的 GPU 作业**(`qsub_corrected_rerun.sh:20` 自己就这么写的)。
2. **修 §3-O9**:把 `run_trajectory_eval.py:49-50` 的 `TRACKB_KEYS` 补上
   `segment_success_rate_30/45`、`turn_precision`、`turn_f1`、`waypoint_endpoint_error`。
   **一行改动,直接决定师兄意见 2/3/4 的产出能否落到报表上。**
   (本次任务只授权新建本文件,故**未改动** `run_trajectory_eval.py`。)
3. **决策 §7-S9 的口径冲突**:α=64(论文)vs α=rank(全部 PBS)。
   **必须在提交作业前定**,否则 11 个作业跑完得到的是一组与论文声明不符的超参。

---

*本文件只陈述代码与产物事实,未修改论文、未修改任何其他文件、未提交任何作业。
所有"未运行验证"的推断已逐条标注。*
