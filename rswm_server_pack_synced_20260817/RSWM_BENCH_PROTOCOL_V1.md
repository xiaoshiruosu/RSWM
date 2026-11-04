# RSWM-Bench Protocol V1

**Protocol ID:** `rswm.benchmark_protocol/v1`  
**状态:** `FROZEN`  
**日期:** 2026-08-22  
**批准记录:** 2026-08-23；项目负责人已授权进入 Phase 1，并进一步确认 Phase 1 P1-01 至 P1-15、允许进入 Phase 2 M1-M3。  
**拟议论文标题:** *RSWM-Bench: A Benchmark for Continuous View Extrapolation in Remote Sensing World Models*  
**依据:** `RSWM_BENCH_CURRENT_STATUS.md` 与 `RSWM_BENCH_UPGRADE_PLAN.md`  
**生效记录:** Phase 0 协议按本文推荐值冻结；后续实现不得静默改变本文定义。

> 本文冻结的是 benchmark 的科学与评测合同，不是代码实现。本文不修改模型、LoRA、训练超参数、现有 ICLR 数据或结果，也不代表任何尚未运行的实验已经完成。

## 0. 规范性用语与证据边界

- **必须 / 禁止:** 正式数据、提交、评测和 leaderboard 必须遵守；不满足时不得标记为 official。
- **应:** 原则上要求遵守；偏离时必须在 result card 中说明理由和影响。
- **可:** 可选扩展，不构成 RSWM-Bench V1 发布的前置条件。
- **已观察:** 当前仓库已有直接代码或文档证据。
- **计划:** Phase 1 以后拟实现的抽象层，不得写成当前已完成能力。
- **尚未验证:** 当前本地包缺少正式 HDF5、模型权重或结果产物，不能据此宣称通过。

当前 `rsroll.benchmark_manifest/v2`、现有 HDF5、official split 和 ICLR 实验保持原义。V1 通过只读兼容层引用它们，不回写、不改字节、不追溯性重标已有结果。

本协议草案对应的输入文档版本为：

| 输入文档 | SHA-256 |
|---|---|
| `RSWM_BENCH_CURRENT_STATUS.md` | `08E4C9E7BA41C3537F072CA02CB3F8988F9CA7B3C3E421990BFF3B668E319BCA` |
| `RSWM_BENCH_UPGRADE_PLAN.md` | `E19174F76A8EC3E1B6BBC5A9D32F3880D2C53D2CC65C95D83D51E2692903C4A4` |

## 1. Final benchmark claim

### 1.1 允许作为论文摘要或主页首句的英文表述

> **RSWM-Bench is a benchmark for trajectory-conditioned continuous spatial view extrapolation from a single remote-sensing image. Its initial controlled domain constructs camera/viewport rollouts by moving a fixed window over real remote-sensing imagery, and evaluates visual fidelity, temporal consistency, trajectory-conditioned planar spatial consistency, and semantic preservation across direction, overlap, and rollout-horizon cohorts.**

### 1.2 对应中文表述

> **RSWM-Bench 是一个面向遥感世界模型的轨迹条件连续空间视角外推 benchmark。其首发受控域以真实遥感图像为像素来源，通过固定窗口在静态场景上的连续移动构造相机/视口 rollout，并从视觉保真、时序一致性、轨迹条件二维空间一致性和语义保持四个维度，在方向、重叠率与 rollout 长度等受控条件下进行评测。**

### 1.3 论文定位

- benchmark 的贡献是任务合同、受控难度轴、source-disjoint 数据协议、多维评测、baseline 公平性和可复现提交机制。
- RS-Roll / LoRA 的角色是 **camera-conditioned remote-sensing rollout 的 parameter-efficient reference adaptation baseline**。
- LoRA 结果可以作为 benchmark 的代表性基线和消融证据，但不是 RSWM-Bench 唯一或必要的核心创新。
- “world model” 在本文中限定为根据观测图像和显式控制条件生成连续空间观测序列的系统类别，不等同于已验证的通用物理世界模拟器。

## 2. Task definition

### 2.1 共同输入输出合同

所有任务共享以下运行时合同：

```text
input:
  one remote-sensing condition image at t=0
  + task-specific command exposed by the selected conditioning track
  + frozen public metadata

output:
  one complete rollout with shape (T,H,W,3), dtype uint8
  T ∈ {17,25,33}, H=480, W=832
```

- `t=0` 是提供给模型的 condition frame，不是需要预测的未来帧。
- 输出中的 `t=0` 必须与给定 condition frame 一致；视觉、分布和输出语义保真仅评分 `t=1...T-1`。
- 轨迹恢复指标可使用 `t=0` 作为位移原点。
- 不允许把单张输出静默复制成视频；图像方法必须声明递归、逐 waypoint 或 outpainting adaptation。
- 每个方法必须对同一冻结 manifest 的全部 sample ID 生成结果，不允许方法专属子集。

### 2.2 T1：Trajectory-conditioned Single-view Spatial Rollout

**Task ID:** `T1_spatial_rollout`

给定一张遥感 condition image 和所属 Track 可见的轨迹命令，生成连续空间视角 rollout。

T1 是基础任务，回答：模型能否在保持场景视觉与语义一致性的同时，生成与命令相符的连续空间观测序列。它不表示现实时间上的 future forecasting。

**必报维度:**

- future-frame visual fidelity；
- clip-level video quality；
- temporal consistency；
- Track 允许范围内的 trajectory/control consistency；
- semantic preservation；
- generation success rate 与 metric valid rate。

### 2.3 T2：Arbitrary-direction Spatial Extrapolation

**Task ID:** `T2_arbitrary_direction`

在与 T1 相同的输入输出合同下，使用覆盖完整连续角空间、并具有八方向审计分桶的测试 manifest，评价模型对不同移动方向的泛化和控制一致性。

- 连续角是任务真值；八方向标签只用于 coverage、split audit 和结果 breakdown。
- T2 不能被简化为八分类生成任务。
- 主方向 cohort 使用起点到终点的净相机/视口方向；多段路径还必须报告逐段方向覆盖。
- 净位移不适定的近闭合轨迹不进入净方向均值，但仍可进入 Track B 的逐段和形状指标。

### 2.4 T3：Overlap-controlled Spatial Extrapolation

**Task ID:** `T3_overlap_ladder`

在与 T1 相同的输入输出合同下，按冻结的 condition-to-endpoint overlap 区间构造或分配难度 cohort，评价模型随已见区域减少而产生的性能变化。

- T3 必须分别报告四个 overlap level，禁止只汇报混合平均值。
- 同一 source 可在同一 split 内构造成对的难度样本；统计分析必须以 source 为 cluster，不能把同源样本视为独立样本。
- 当前 ICLR `<10%` AID 数据只映射到 `L4_overlap_lt10`，不因 V1 协议而重建或改变。

### 2.5 Task 之间的关系

T1 是基础生成合同；T2 和 T3 是对方向与 overlap 两个难度轴的正式受控任务。一个样本可被多个 task manifest 引用，但各 task 必须拥有独立的 manifest checksum、样本数和报告，不得把三个任务的分数直接混合。

## 3. Track definition

Track 描述参赛方法实际获得的控制信息，而不是模型架构类别。不同 Track 不直接排名。

| Track ID | 正式名称 | 模型可见输入 | 被命令的控制量 | 正式控制指标范围 |
|---|---|---|---|---|
| `A_text` | Text-commanded rollout | condition image + scene prompt + segment count + continuous net camera bearing | 净方向与段数 | 净方向及通用生成质量；完整轨迹指标仅作 `commanded=false` 诊断 |
| `B_camera` | Camera-conditioned rollout | Track A 输入 + 逐帧 Plücker poses / intrinsics 附件 | 完整逐帧轨迹 | 净方向、逐段、转弯、waypoint、路径形状与通用生成质量 |
| `C_multiview` | Multi-view conditioned rollout | 多张 condition views + 相机信息 | 多视图条件下的目标轨迹 | 可选未来扩展，独立榜单，不是 V1 首发必需项 |

### 3.1 Track A prompt freeze

- V1 official prompt schema 建议冻结为 `net_continuous`：prompt 给出连续净相机方位角和段数。
- `net` 四方向词式保留为 legacy/diagnostic compatibility arm，不得伪装成 V1 official continuous-direction 结果。
- `per_segment_continuous` 属于更强控制输入，如启用必须建立独立 capability partition，不能并入 `A_text`。
- `stripped` 是无方向归因对照，不是提交 Track。
- 已有 HDF5 或结果必须使用其实际记录的 `prompt_schema`；禁止把 `net` 结果追溯性重标为 `net_continuous`。

### 3.2 Track B camera metadata boundary

- 当前 AID rollout 的 poses 是由二维窗口轨迹导出的代理表示；intrinsics 可能是 placeholder。
- camera metadata 必须携带 `calibration_status`，只允许如实标为 `derived_planar` 或 `placeholder`，不得标为 `measured`。
- Track B 的几何/轨迹结果只能支持二维视口轨迹一致性 claim，不支持真实相机标定或 3D 重建 claim。

### 3.3 非正式 Track

`pose_mode=identity`、GT oracle、privileged translation、persistence 和 random-direction 等仅是诊断臂或校准参照，不构成正式提交 Track，也不参与排行榜排名。

## 4. Dataset scope

### 4.1 V1 必需域

| Domain ID | 像素来源 | Rollout 构造 | V1 地位 |
|---|---|---|---|
| `synthetic_controlled` | 真实遥感图像 | 静态场景上的受控二维滑动窗口 | 首发必需域 |
| `generalization` | 其他遥感数据源 | 与主域兼容的受控 rollout | 可选扩展；只有冻结 manifest、许可和泄漏审计完成后才能报告 |
| `real_world` | 真实多视图、真实序列或带真实标定的数据 | 由数据真实采集过程决定 | 可选未来域；V1 可标记 `NOT_AVAILABLE`，不是首发阻断条件 |

`synthetic_controlled` 表示轨迹和视角序列由程序受控构造，不表示像素是合成图像。AID 的正确 metadata 是：

```yaml
pixel_source: real_remote_sensing_image
rollout_construction: planar_sliding_window
camera_physics: controlled_2d_proxy
```

### 4.2 与现有 ICLR 数据的关系

- 现有 ICLR 主协议保持 AID、N=2000、window 128×222、target 480×832、steps=9、T=17/25/33、endpoint overlap `[0,0.1)`、SemanticGate 开启。
- 现有 2000 样本数据若通过其原始验证，只映射为 `synthetic_controlled / T3 / L4` 的 legacy-compatible artifact；V1 不要求修改或重建它。
- 新的 L1–L3 cohort 可继续使用现有 AID 来源构造，不要求增加新的真实数据集。
- 每个正式 release 必须冻结 source-disjoint train/val/test、sample order、manifest checksum、源身份与重复检查结果。
- 同一 source image 产生的全部窗口、轨迹和难度版本必须落入同一 split。

### 4.3 数据许可与发布

- 在 AID 再分发权未被可核验证据确认前，V1 默认采用 **build-from-source** 发布：发布构建脚本、来源说明、annotations、checksums 和验证工具，不直接承诺重新分发像素。
- `generalization` 中的每个数据源必须单独记录许可证、访问条件、版本和来源 checksum。
- 数据卡必须区分“真实像素来源”和“合成相机轨迹”，不得把 AID 滑窗写成真实视频或真实多视图采集。
- sample 数量由冻结 manifest 决定并随结果报告；不得在生成前凭计划值宣称正式规模已经完成。

## 5. Unsupported claims

除非未来新增独立数据域和相应证据，RSWM-Bench V1 **不支持**以下表述：

1. 真实时间事件、天气、季节、灾害演化或动态目标的 future forecasting；
2. 真实 6DoF novel-view synthesis、真实视差、深度恢复或相机标定；
3. 显式 3D world reconstruction、物理世界模拟或通用 world dynamics 学习；
4. 遮挡显露、尺度变化、相机旋转和真实成像过程的完整建模；
5. AID 滑窗属于真实视频、真实多视图或实测相机轨迹；
6. 在未运行正式跨数据集实验前声称具备跨传感器、跨区域或开放世界泛化；
7. 将 SemanticGate 的数据筛选结果写成生成视频语义正确性的直接证据；
8. 将二维 registration、optical flow、Plücker condition 或 DINO 相似度写成真实 3D geometry 证明；
9. 将 `FVD_R3D_PROXY` 与文献标准 I3D-FVD 混称为同一指标；
10. 在许可证未确认前声称 AID 像素可以公开再分发；
11. 将仅有 adapter/card、但没有 checkpoint、正式结果和日志的方法称为“已复现 baseline”；
12. 将 RSWM Score 或任何未校准 composite score 作为 V1 主指标或唯一排行榜依据；
13. 声称 benchmark 升级已经完成训练、评测或论文实验。

## 6. Overlap definition

### 6.1 几何定义

Overlap 在原始静态场景的窗口坐标中、resize 到 480×832 **之前**计算。令 `W_0` 为 condition window，`W_t` 为第 `t` 帧对应的轴对齐窗口，则：

```text
O_t = area(W_0 ∩ W_t) / area(W_0)
```

- 该定义是“condition window 被未来窗口覆盖的面积比例”，不是 IoU。
- canonical 样本使用相同大小的完整窗口，因此数值关于两窗口对称；被边界裁断或窗口尺寸不一致的样本不得进入 headline cohort。
- `O_0 = 1`。
- 所有 overlap 数值必须位于 `[0,1]`，并记录算法版本、scene size 与 window size。

### 6.2 三种 overlap 不得混用

- **主难度:** `O_endpoint = O(condition t=0, endpoint)`；
- **辅助曲线:** `O_t = O(condition t=0, frame t)`；
- **连续性审计:** `A_t = O(frame t-1, frame t)`。

T3 分档只使用 `O_endpoint`。`mean_future_overlap`、`minimum_future_overlap` 和 adjacent overlap 必须单独报告，不得替代 endpoint 难度标签。

### 6.3 V1 overlap ladder

| Level ID | 名称 | `O_endpoint` 区间 | 排名用途 |
|---|---|---:|---|
| `L1_overlap_80` | 80% anchor | `[0.70,0.90)` | T3 必报 cohort |
| `L2_overlap_50` | 50% anchor | `[0.40,0.60)` | T3 必报 cohort |
| `L3_overlap_20` | 20% anchor | `[0.15,0.25)` | T3 必报 cohort |
| `L4_overlap_lt10` | below 10% | `[0.00,0.10)` | 现有 ICLR 主协议映射 cohort |

- 区间采用左闭右开；边界值按表中区间唯一归属。
- 区间之间的 buffer 是有意设计，用于降低边界抖动和相邻难度混淆。
- 落在 buffer 区域的样本可用于诊断，但不得进入 T3 V1 headline cohort。
- 任何语义筛选 fallback 都不得放宽 overlap、轨迹合法性、source identity 或 duplicate 约束。

## 7. Direction convention

### 7.1 坐标与角度

轨迹点存储为 scene 像素中心坐标 `(cy,cx)`，其中 `cy` 向图像下方增大、`cx` 向右增大。对从 `p_0=(cy_0,cx_0)` 到 `p_1=(cy_1,cx_1)` 的相机/视口移动：

```text
theta_camera = degrees(atan2(-(cy_1-cy_0), cx_1-cx_0)) mod 360
theta_content = (theta_camera + 180) mod 360
```

- `0° = E/right`；`90° = N/up`；`180° = W/left`；`270° = S/down`。
- 角度按数学方向逆时针增加。
- prompt 和 camera trajectory 使用 `theta_camera`；光流/registration 观察到的画面内容运动使用 `theta_content`。
- 报告必须显式写明角度类型，禁止把 camera angle 与 content-flow angle 直接比较。

### 7.2 八方向报告分桶

| Direction ID | `theta_camera` 半开区间 | 中心角 |
|---|---:|---:|
| `E` | `[337.5°,360°) ∪ [0°,22.5°)` | 0° |
| `NE` | `[22.5°,67.5°)` | 45° |
| `N` | `[67.5°,112.5°)` | 90° |
| `NW` | `[112.5°,157.5°)` | 135° |
| `W` | `[157.5°,202.5°)` | 180° |
| `SW` | `[202.5°,247.5°)` | 225° |
| `S` | `[247.5°,292.5°)` | 270° |
| `SE` | `[292.5°,337.5°)` | 315° |

方向 bin 只承担覆盖与 breakdown；Track A 的 V1 official continuous bearing 和 Track B 的连续轨迹均不得先量化为八方向再评分。

### 7.3 净方向与逐段方向

- 净方向由起点到终点计算；逐段方向由相邻 waypoints 计算。
- 定义 `net_ratio = ||p_end-p_start|| / sum_k ||p_(k+1)-p_k||`。
- 当 `net_ratio < 0.2` 时，净方向被标记为 `not_well_posed`，不进入净方向均值；样本数和排除原因必须公开。
- Track A 只把净方向和段数视为 commanded；Track B 可评价逐段角、turn、waypoint、Fréchet、DTW 与 endpoint/path consistency。

## 8. Metric policy

### 8.1 总原则

1. V1 采用 **多指标面板**，不采用单一总分作为主结论。
2. 所有视觉、分布与输出语义指标只使用 future frames `t=1...T-1`。
3. 每个指标必须冻结名称、版本、方向、预处理、输入尺度、checkpoint hash、最小样本量、invalid policy 和适用 task/track。
4. 同一比较中的所有方法必须使用相同 manifest checksum、evaluator version 和 metric checkpoint。
5. 缺失或无效指标写 `null` 并报告原因；禁止写 0、静默删除失败样本或改用方法专属样本集。
6. 2D trajectory/flow/registration 指标必须附带二维 proxy claim guard。

### 8.2 V1 指标层级

| 层级 | 指标 | 政策 |
|---|---|---|
| Headline visual | SSIM ↑、PSNR ↑、LPIPS ↓ | future-only；LPIPS 固定 backbone/weights |
| Headline distribution | FID ↓、`FVD_I3D` ↓ | 至少 32 个完整 clips；固定 extractor/checkpoint；报告 N |
| Headline control | well-posed net direction error ↓；Track B segment/turn/waypoint/path metrics | 必须通过 GT oracle / instrument-floor calibration gate |
| Required audit | success rate、metric valid rate、per-frame overlap、adjacent overlap、horizon/direction/scene breakdown | 不得省略 |
| Diagnostic only | `FVD_R3D_PROXY`、persistence/wrong-direction response、未校准的新指标 | 不得与标准 FVD 或 headline 指标混称 |
| Candidate experimental | temporal perceptual-change、occlusion-aware warping、DINO correspondence、输出语义 drift | known-answer、版本和校准完成前不参与正式排名 |

SemanticGate 是数据构建质量门，不是输出级 semantic metric。输出语义指标必须独立定义和验证。

### 8.3 统计与失败政策

- 训练型 reference baseline 应报告预注册 seeds；现有 ICLR RS-Roll 保持 seeds 42/43/44，不新增或修改实验矩阵。
- 同一 source 的 paired comparison 使用 source-cluster bootstrap 95% CI。
- 所有 breakdown 同时报告 `N`；样本量不足的格子只作描述，不给显著性结论。
- 正式提交必须覆盖全部 required sample IDs，并满足形状、dtype、有限值和重复检查；不完整提交不进入 official leaderboard。
- metric estimator 自身 valid rate 不足时，该指标标记 ineligible，而不是把该方法判为 0 分。
- evaluator 或 calibration gate 失败意味着指标失效，不得解释为模型失败。

### 8.4 RSWM Score policy

- V1 **不定义 RSWM Score 权重、归一化或主排名公式**。
- 如果 Phase 3 产生 composite score，只能以独立版本 `experimental` 发布，并同时保留全部原始指标。
- composite score 成为 headline 前，必须另行完成人类/专家校准、anchor 冻结、权重敏感性、排名稳定性和缺失值政策审查，并再次获得人工批准。

## 9. Baseline admission policy

### 9.1 正式准入条件

一个方法只有同时满足以下条件，才能称为 admitted baseline：

1. 有真实可运行实现或可验证的外部结果桥接，不以 stub/card 代替运行；
2. 有明确 checkpoint/version/license/hash 和安装环境；
3. 声明 native input/output、trajectory support、adaptation 类型和训练数据；
4. 声明是否使用外部数据，并通过 test-source overlap 核验；
5. 对冻结 manifest 的全部样本输出符合 `(T,H,W,3) uint8` 合同的结果；
6. 记录 seed、precision、compute、成功率、失败原因和 artifact checksum；
7. 使用共享 evaluator，不接受手工填写或覆盖 metric；
8. 只进入与其实际可见输入一致的 task/track/capability partition。

“存在 adapter”“配置文件存在”或“论文声称支持类似任务”均不等于已复现。

### 9.2 Baseline 分类

- **Diagnostic:** persistence、random-direction、identity-camera；不排名。
- **Upper bound / privileged:** GT oracle、privileged translation；只校准指标，不排名。
- **Image-generation adapted:** SD/SDXL、DiffusionSat 等通过显式递归/outpainting adapter 运行。
- **Generic video adapted:** frozen I2V、VideoCrafter、AnimateDiff、Open-Sora 类方法；必须如实声明是否原生支持 camera trajectory。
- **World-model candidate:** Genie、Cosmos 等只有在输入接口、权重、许可和适配均可复核后准入。
- **Remote-sensing representation model:** SatMAE 可作为 evaluator/encoder candidate，不作为 rollout 生成 baseline。
- **Multi-view method:** MVSplat 类方法只进入 `C_multiview`，不与单图 Track 排名。
- **Reference adaptation:** RS-Roll LoRA 作为 V1 reference baseline；其模型结构和现有 ICLR 配置保持不变。

### 9.3 公平性与披露

- `native`、`prompt_adapter`、`recursive_rollout`、`finetuned` 和 `multiview` 必须显式标记。
- 输入信息不同的方法不得在同一 capability partition 中给出单一名次。
- 允许公开预训练权重和外部遥感语料，但必须披露；任何 test source 或其派生数据参与训练都会导致拒收。
- 首发论文主表只纳入有完整 artifact 的方法；候选列表不能替代实验结果。

## 10. Leaderboard policy

### 10.1 排行榜分区

最小结果键为：

```text
(benchmark_version, task_id, conditioning_track, data_domain, capability_partition)
```

- `A_text`、`B_camera`、`C_multiview` 分榜。
- native/adapted/multiview capability 不混成单一总榜。
- overlap、direction、trajectory type、scene 和 17/25/33 horizon 是必报 breakdown；样本量足够时可形成子榜。
- 诊断方法和 privileged upper bound 展示为参照线，不参与名次。

### 10.2 排名与展示

- V1 不发布跨指标单一 global rank，也不宣称唯一 overall winner。
- leaderboard 提供按各冻结 headline metric 排序的视图，并并列展示全部关键指标、CI、N、success rate 和 valid rate。
- 不允许用未校准 group score 或 RSWM Score 决定主名次。
- 缺失 required metric、样本不全、合同失败或 test-source 泄漏的提交标记为 `ineligible`，不得通过零填充进入排序。

### 10.3 提交与验证状态

每条 entry 必须标记：

- `unverified`: 仅由提交者提供，尚未独立复核；
- `reproduced`: 在独立环境或组织者环境成功复跑；
- `official`: 通过 hidden test、合同、来源、许可与 artifact 审计。

Entry 至少记录 benchmark/evaluator/metric version、manifest checksum、method/card/checkpoint/container identity、原始 metrics、CI、N、成功率、外部数据声明、时间戳和结果签名。

### 10.4 隐藏测试、泄漏与版本

- train/val/test 必须 source-disjoint；public dev 与 hidden test 分离。
- Track A 不发布 hidden trajectory/poses；Track B 的 camera command 通过独立附件提供。
- 因公开遥感源图可能被 condition-frame 检索，参赛者必须提交外部来源声明；组织者保留源身份核验和违规撤榜机制。
- scorer 更新必须提升 evaluator/leaderboard version；旧榜保留，不得静默覆盖。
- 只有保留原始 rollout 和完整 provenance 的 entry 才能被新 evaluator 重算。
- 排行榜必须公布提交频率、超时、撤回、重算、维护和 deprecation 政策。

## 11. 兼容边界与实施门

以下内容在 V1 协议下保持不变：

- Video Diffusion / Video DiT 与 LingBot-Wan I2V 模型结构；
- LoRA 注入、rank、target、optimizer、学习率、batch 和训练步数；
- Primary `native_attn_cam_r16` 与 seeds 42/43/44；
- 当前 AID 2000 样本 ICLR protocol 及已产生结果的原始解释；
- 现有 HDF5 `(T,H,W,C) uint8` 表示；
- `rsroll.benchmark_manifest/v2` 和当前 official split 的可读性。

协议冻结后的持续约束：

- `benchmark/` 代码只能按获批 Phase 范围增量实现；
- 不修改 builder、训练器、evaluator 或配置；
- 不重建 HDF5；
- 不启动训练、evaluation 或 baseline GPU 作业；
- 任何协议变更必须重新进入人工审查并形成新版本身份。

Phase 1 只能新增兼容层、schema 和 benchmark card，并首先证明旧 HDF5 无需改字节即可读取。若实现发现协议与现有 artifact 不兼容，必须回到人工协议审查，不能在代码中静默改变定义。

## 12. 最终协议决策表

以下条目已按“建议冻结值”冻结。批准仅确认协议，不代表未运行的模型、数据或 evaluation 已完成。

| ID | 决策项 | 建议冻结值 | 若不确认的影响 | 状态 |
|---|---|---|---|---|
| `P0-01` | Final claim boundary | 采用“真实遥感像素 + 受控二维连续空间 rollout”；不宣称真实时间、6DoF 或通用 3D world dynamics | 无法确定论文标题、摘要与 claim guard | 已确认（2026-08-23） |
| `P0-02` | Task IDs | 冻结 `T1_spatial_rollout`、`T2_arbitrary_direction`、`T3_overlap_ladder`；T2/T3 继承 T1 输出合同但独立 manifest/报告 | Phase 1 无法定义 task schema | 已确认（2026-08-23） |
| `P0-03` | Conditioning Tracks | 冻结 `A_text`、`B_camera`；`C_multiview` 仅为可选独立扩展 | submission schema 与分榜规则不确定 | 已确认（2026-08-23） |
| `P0-04` | Official prompt schema | V1 official Track A 使用 `net_continuous`；legacy `net` 只作兼容/诊断，禁止追溯性重标 | continuous-direction claim 仍存在口径分叉 | 已确认（2026-08-23） |
| `P0-05` | Required dataset domain | 首发只要求 AID-derived `synthetic_controlled`；`real_world` 可为 `NOT_AVAILABLE`，不作为首发 blocker | 若要求真实多视图，将扩大数据、许可和实验范围 | 已确认（2026-08-23） |
| `P0-06` | Existing ICLR preservation | AID N=2000 `<10%` 数据原样保留并映射到 T3/L4；不重建、不改已有结果含义 | 可能破坏现有 ICLR 证据链 | 已确认（2026-08-23） |
| `P0-07` | AID release mode | 许可证未核验前采用 build-from-source；不承诺直接再分发像素 | 公开发布存在许可风险 | 已确认（2026-08-23） |
| `P0-08` | Overlap definition | 冻结 `intersection / condition-window area`、resize 前计算；T3 使用 endpoint overlap | 数据构建、分桶和评测可能使用不同口径 | 已确认（2026-08-23） |
| `P0-09` | Overlap ladder | 冻结 L1 `[0.70,0.90)`、L2 `[0.40,0.60)`、L3 `[0.15,0.25)`、L4 `[0,0.10)`；buffer 样本不进 headline | Phase 2 无法生成稳定 task split | 已确认（2026-08-23） |
| `P0-10` | Direction convention | `(cy,cx)`、y 向下；camera 角 E=0/N=90，逆时针；content 角加 180°；八方向使用 ±22.5° 半开分桶 | 方向标签和光流评测可能相差 180° | 已确认（2026-08-23） |
| `P0-11` | Net-direction validity | 冻结 `net_ratio < 0.2` 为净方向不适定；排除出净方向均值但保留计数和其他可用指标 | 近闭合轨迹会污染净方向结论 | 已确认（2026-08-23） |
| `P0-12` | Metric headline policy | 使用多指标面板；标准 I3D-FVD 与 R3D proxy 分名；future-only；缺失为 null；校准门失败不解释模型 | 无法形成可审计的主表 | 已确认（2026-08-23） |
| `P0-13` | RSWM Score | V1 不定义权重、不作主指标、不作总榜；未来 composite 必须另行校准和审批 | 提前合成分数可能导致任意排名 | 已确认（2026-08-23） |
| `P0-14` | Baseline admission | 仅实际运行且有 checkpoint/result/log/provenance 的方法准入；diagnostic/upper bound 不排名；多视图分榜 | “有 adapter”会被误报为“已复现” | 已确认（2026-08-23） |
| `P0-15` | Minimum paper baseline evidence | 发布主表前至少有 5 个可运行生成方法，加诊断下界和校准上界；具体外部模型按资源另行批准 | baseline 覆盖不足以支持 benchmark 论文 | 已确认（2026-08-23） |
| `P0-16` | Leaderboard ranking | 按 task × track × domain × capability 分区；展示逐指标排序，不发布单一 global rank | 不同输入条件和量纲会被不公平混排 | 已确认（2026-08-23） |
| `P0-17` | Official submission completeness | 必须覆盖冻结 manifest 的全部 required sample IDs；不完整、泄漏或合同失败均为 ineligible | 方法专属子集会造成不可比结果 | 已确认（2026-08-23） |
| `P0-18` | Phase 1 authorization boundary | 仅在全部关键项确认后开发 schema、只读 adapter 和 benchmark card；不先跑数据/训练/GPU 实验 | 实现可能先于协议并再次产生口径漂移 | 已确认（2026-08-23） |

### 批准与变更控制

本表是 V1 的冻结决策记录。后续若需调整任一 P0 条目，必须生成新的协议版本、release identity 和迁移说明；不得原地静默改写已发布结果的含义。
