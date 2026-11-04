# RSWM-Bench 当前状态审查

**拟议论文定位：** *RSWM-Bench: A Benchmark for Continuous View Extrapolation in Remote Sensing World Models*  
**审查日期：** 2026-08-22  
**审查性质：** 只读科研代码审计；未修改实现、模型、LoRA 配置或既有实验协议  
**结论：** 当前仓库已经具备一个较成熟的“内部 benchmark/evaluation 平台”，但尚不具备可直接发布为 ICLR/CVPR 风格公开 benchmark 的完整证据链。最准确的当前定位是：**基于真实遥感图像构造的、受控二维空间滚动预测 benchmark 原型**，而不是已完成的通用遥感 world model benchmark。

## 1. 证据边界与主版本确认

### 1.1 审查对象

主运行包确认为仓库顶层目录，而不是内部同名嵌套副本。依据如下：

- `AGENTS.md` 明确指定“当前顶层同步目录是主运行包”，内部同名目录为冗余副本；
- `SYNC_REPORT.md` 说明该包由本地基线结合用户提供的服务器运行记录同步而来，并非服务器目录的直接下载副本；
- 当前 Git 仓库无 commit、无 remote，不能用 Git revision 证明版本身份，因此本次以来源记录和文件 SHA-256 为准；
- 排除内部冗余副本、`work/` 历史材料和 `__pycache__` 后，共清点 474 个活动文件；251 个 Python 文件通过 AST 解析，31 个 JSON 和 27 个 YAML 文件通过解析。

关键审查文件的当前 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `reports/BENCHMARK_PROTOCOL.md` | `873C78BCE139C50BD4F116A84D2727526F9C8EC17856BFD17278E8A6C161DD78` |
| `DATASHEET.md` | `96333F66FD28301FD9A923E7A9732F29F6E88A76970D0AD6924EA75496846177` |
| `SUBMISSION.md` | `8495546770C6236EB9BD0C8451E6CCDA77FECE4EDA3267F60A2F5DABC6CD9FC0` |
| `server_run/make_official_splits.py` | `F231886EE2895AC7F0E01E9C8A31E3CCDB23D0F91E6F94347131C613A519E2F2` |
| `server_run/score_submission.py` | `5AC547F4C9CEA3A8082084747182009779EC16996935C71CFD177F6081ADE782` |
| `lora_remotesense/new/metrics.py` | `C8A4225D0FA31C2B3F4F49B877F8324630D30292062E8FA12452E8BDBEEEE4AE` |

### 1.2 状态标签

- **已观察：** 直接从当前源码、配置、状态文件或本次实际测试得到。
- **合理推断：** 由代码路径推导，但没有真实数据、模型或服务器运行产物验证。
- **尚未验证：** 当前源代码包没有所需数据、权重、日志或正式结果，不能声称通过。

本地源码包不含正式 HDF5、official split、benchmark manifest、模型权重或正式 baseline/result artifact。因此本文只评价工程能力与协议一致性，不声称 2000 样本数据、训练、evaluation 或论文结果已经完成。

## 2. 当前 benchmark 已经具备什么

### 2.1 数据构建与质量门

**已观察：**

- 唯一正式入口为 `server_run/build_correct.sh`，调用 canonical builder `lora_remotesense/build_dataset_from_benchmark.py`；
- 正式 full pipeline 固定 AID、2000 样本、窗口 128×222、目标 480×832、每段 9 steps、17/25/33 帧和终点 overlap `[0,0.1)`；
- 构建使用 `.partial` 文件，验证通过后原子发布；损坏或不完整 HDF5 会被隔离，不会静默复用；
- AID 与 CLIP 都要求本地 snapshot，Hugging Face/Transformers 设为 offline；SemanticGate 强制开启；
- `check_dataset.py` 对样本数、HDF5 结构、帧数、分辨率、轨迹、overlap、SemanticGate 证据和像素统计执行 fail-closed 验证；
- `check_benchmark_duplicates.py`、source identity 和内容哈希提供重复及泄漏审计；
- 每条样本包含 `image`、`video_frames`、`trajectory`、`waypoints`、`segment_angles`、`poses`、`intrinsics`、`full_scene` 及来源/语义门属性。

这是当前工程最扎实的部分。它已经超出普通训练脚本，形成了数据生成、验证、溯源、重复检查和失败隔离链。

### 2.2 冻结 manifest 与可比性契约

`server_run/baselines/build_benchmark_manifest.py` 和 `server_run/baselines/protocol.py` 已实现 `rsroll.benchmark_manifest/v2`：

- 冻结 sample ID 和顺序；
- 记录数据集、split 和 manifest checksum；
- 记录 source image、trajectory ID/type/checksum；
- 记录 condition、GT rollout 和整条样本 checksum；
- 记录逐帧 condition-to-future overlap 与相邻帧 overlap；
- 记录 17/25/33 帧及实际分辨率；
- 对子集、跳过 H5 hash、轨迹插值等诊断模式标记 `headline_eligible=false`；
- 结果适配器严格检查 ID、帧数、分辨率、随机种子和条件类型，禁止静默截断或自动补齐。

这已经具备公开 benchmark 所需的“同一测试样本、同一协议、同一输出契约”的核心骨架。

### 2.3 source-disjoint 划分和隐藏测试

`server_run/make_official_splits.py` 已具备：

- 70/10/20 的 train/val/test 三划分；
- 按来源 ID、路径、scene ID 和 `full_scene` 内容 SHA1 合并同源样本；
- 同源组整体进入同一 split，防止同一原图裁窗跨 train/test；
- 默认按场景类别分层并做类别平衡自检；
- 输出确定性 split checksum；
- 可生成去除 GT、轨迹和来源标识的公开 test HDF5；
- Track B 位姿附件与 Track A 输入物理隔离。

`SUBMISSION.md` 和 `server_run/score_submission.py` 进一步提供提交 schema、压缩包安全检查、完整 ID 集、零失败容忍、形状和非有限值检查、隐藏 GT 评分及机器可读拒收原因。该部分已经接近可复用的 challenge backend。

### 2.4 统一 baseline 平台

当前存在 16 张 baseline card、注册表、运行器、result adapter、标准结果目录和 paper table exporter。统一协议要求：

```text
sample + frozen manifest
  -> baseline adapter / external result bridge
  -> (T,H,W,3) uint8 rollout
  -> future-only evaluator
  -> aggregate_metrics.json + sample_metrics.jsonl
  -> paper CSV tables
```

平台能够区分：

- 诊断上界：GT oracle、privileged deterministic translation；
- 诊断下界：persistence、random-direction translation；
- 外部/适配方法：image outpainting、RSWISE bridge、frozen I2V、no-camera、MVSplat bridge；
- RS-Roll/LoRA matched arms 与 rank/target ablation。

缺失值被保留为 `null`，不会以 0 填充；card 记录 native output、trajectory support、adaptation 和是否使用 future GT。这些设计适合作为后续 benchmark abstraction layer 的兼容底座。

### 2.5 评估与统计基础设施

当前实现包括：

| 维度 | 已有实现 | 当前边界 |
|---|---|---|
| 视觉 | SSIM、PSNR、LPIPS、FID | 条件帧从全部保真指标排除；LPIPS/FID 为可选依赖 |
| 视频分布 | R3D-18 proxy、标准 I3D-FVD 重算入口 | 两者明确分名；真实 I3D extractor 未在本地验证 |
| 净方向 | 光流方向误差、well-posed gate | Track A 只评价被文本命令的信息 |
| 完整轨迹 | segment angle、turn precision/recall/F1、turn timing、waypoint、endpoint、Fréchet、DTW | 属于二维 registration/flow proxy，不是 3D 几何恢复 |
| overlap/horizon | 逐帧 overlap 曲线、17/25/33 horizon、2D flow vector consistency | 当前是结果分桶/审计，不是四档数据生成协议 |
| 统计 | 三 seed mean/std、paired comparison、seed/traj bootstrap 工具 | 尚未接入统一 leaderboard result schema |

指标层已经认真处理了 condition-frame 泄漏、最小分布样本量、静止视频拒答、方向角约定和仪器地板，比普通实验脚本更规范。

## 3. 当前 task definition 的准确含义

### 3.1 当前只有一个主任务、两个 conditioning track

当前 manifest 的正式任务是：

> single remote-sensing image + continuous requested trajectory → video rollout

当前 Track A/Track B 的差别是条件可见性，而不是两个独立任务：

- **Track A / pose drop：** 单图 + 文本；只把净方向/段数视为被命令信息；
- **Track B / pose native：** 单图 + 文本 + Plücker poses/intrinsics；完整逐帧轨迹被视为命令。

这与拟议的 Task 1/2/3 不同。当前没有机器可读 `task_id`、任务专属 input/output schema 或独立 leaderboard。

### 3.2 它是空间视角滚动，不是时间预测

当前 `video_frames` 来自同一静态场景上的移动窗口裁剪，再上采样到 480×832。代码和协议正确地声明 `not_temporal_forecasting=true`，但拟议标题中的 “future view” 容易被理解为真实未来或动态世界预测。

当前 GT 不包含：

- 真实时间变化；
- 天气、光照和动态目标；
- 遮挡显露的真实成像过程；
- 相机旋转、尺度变化或 6DoF；
- 真实视差、深度和多视图几何。

`poses` 的旋转为 identity，平移由二维窗口中心构造；`intrinsics` 是 placeholder。因此当前最多支持“已知二维平移轨迹下的受控空间外推”结论，不能据此声称学习了通用 3D world dynamics。

## 4. Dataset 现状与缺口

### 4.1 当前数据域

- 主集：AID；
- 泛化代码路径：WHU-RS19、RESISC45、PatternNet；
- full pipeline 当前明确只评价 AID，跨数据集需要另行 staging；
- AID/WHU 是真实遥感图像，但 rollout 轨迹和帧序列是合成构造；
- RESISC45/PatternNet 的 2×2 拼接模式并不等价于真实连续地理区域。

因此不能把当前 AID rollout 简单归为 `real_world/`，也不应称为完全 synthetic。更准确的类别是：**real imagery + controlled synthetic camera rollout**。

### 4.2 数据文档不一致

**已观察：**

- 当前 full pipeline 固定 N=2000；
- `DATASHEET.md` 和部分复现文档仍写 N_MAIN=4000；
- `DATASHEET.md` 记录 AID taxonomy 未覆盖 BaseballField 类的 220 张图；
- 小图拼接的 provenance 不能完整表示全部 4 张来源图；
- 场景标签来自整图，裁剪后不保证主体仍存在；SemanticGate 可降低风险，但现有人审样本很小且没有多标注者一致性；
- `benchmark_card.yaml`、`camera.json`、`trajectory.json`、`overlap.json` 均不存在。

### 4.3 许可与发布是阻断项

仓库的 `REPRODUCIBILITY.md` 将 AID、RESISC45、PatternNet 的许可记为 unknown。当前 Hugging Face 卡片也仍显示 [AID license unknown](https://huggingface.co/datasets/blanchon/AID)；[RESISC45](https://huggingface.co/datasets/blanchon/RESISC45/tree/main) 和 [PatternNet](https://huggingface.co/datasets/blanchon/PatternNet/tree/main) 同样标为 unknown，而 WHU-RS19 卡片标为 [CC BY 4.0](https://huggingface.co/datasets/jonathan-roberts1/WHU-RS19/tree/main)。

在完成上游条款核验之前，不能承诺重新分发像素数据。公开 benchmark 可以先发布构建脚本、哈希、annotations 和受控下载说明，但这不能替代正式许可确认。

### 4.4 overlap 与方向 split 尚未形成

- 正式 builder 只构建 endpoint overlap `[0,0.1)`；
- `evaluate_rollout_protocol.py` 有 80/50/30/10/5 的 condition-to-frame 分桶，但它是评估分桶，不是独立数据难度 split；
- manifest 的 Dense/Medium/Sparse/Extreme 是按当前样本分布的经验四分位数，不是固定 80/50/20/<10 语义；
- split 仅对 source 和 scene category 强约束，方向、轨迹类型、overlap、horizon 只做分布审计，没有联合分层和最低覆盖门；
- trajectory 已有连续角和逐段 `segment_angles`，但没有冻结的 left/right/up/down/diagonal label、坐标系说明和方向覆盖报告。

## 5. Metric 现状与缺口

### 5.1 已有指标值得保留

当前视觉、分布和二维轨迹指标应作为 v3 的兼容核心，而不是重写。尤其应保留：

- future-only 评分；
- exact shape/ID contract；
- FVD R3D proxy 与标准 I3D-FVD 分名；
- 方向/轨迹的 valid rate 和 instrument-floor gate；
- condition-to-future overlap 与 adjacent overlap 的定义隔离；
- 缺失指标为 null；
- 逐样本结果和 bootstrap 支持。

标准 FVD 本身用于同时衡量视觉质量与时间动态，原始工作也通过人类研究验证其相关性；当前仓库把非标准 R3D proxy 与 I3D 口径分开是正确做法。[FVD paper](https://arxiv.org/abs/1812.01717)

### 5.2 拟议四类 metric 尚不完整

| 拟议类别 | 当前完成度 | 缺失项 |
|---|---|---|
| Visual | 高 | 需要统一 extractor/checkpoint/version 和正式样本量研究 |
| Temporal | 低—中 | 无独立 temporal LPIPS；无 occlusion-aware warping error；仅有 FVD 与 2D flow vector proxy |
| Spatial geometry | 中（二维） | 无 feature correspondence、DINO similarity、真实相机 reprojection；当前 registration 指标不能称为 3D geometry |
| Semantic | 低 | SemanticGate 是数据过滤，不是生成结果语义保持指标；没有输出语义分数和人类校准 |
| Unified score | 无 | exporter 明确不合成 composite score；没有 normalization、权重、缺失值或排名稳定性协议 |

DINOv2 可作为固定特征工具候选，但其“通用视觉特征”能力并不自动证明对遥感视角变化有效，必须先在 GT/扰动/失败样本上校准。[DINOv2 paper](https://arxiv.org/abs/2304.07193) 同理，LPIPS 的优势来自人类感知相似度数据，不能把它当作几何正确性的替代品。[LPIPS paper](https://arxiv.org/abs/1801.03924)

### 5.3 RSWM Score 不能先定权重再找证据

当前没有 RSWM Score 是一个缺口，但也是一种安全边界。直接把 FID、PSNR、trajectory error 和 semantic score 线性相加会遇到量纲、方向、样本量、缺失值和排名翻转问题。

建议在第一版 leaderboard 中以四个分组和关键子指标为主；综合分仅作为 `experimental`，必须经过：

1. 固定 normalization anchor；
2. 人类 pairwise preference/错误标签校准；
3. 权重敏感性和排名稳定性分析；
4. 版本化配置和旧结果重算机制。

VBench 的重要经验不是“指标越多越好”，而是把视频质量拆成可解释维度，并用人类偏好逐维校验自动指标。[VBench](https://arxiv.org/abs/2311.17982) WorldModelBench 也把 world-model claim 与一般视频质量分开，并用大规模人类标签校准评判器。[WorldModelBench](https://arxiv.org/abs/2502.20694)

## 6. Baseline 现状与缺口

### 6.1 实际状态

`server_run/baselines/baseline_status.json` 的所有 16 个条目均为 `result_available=false`；只有 persistence 标为 synthetic CPU smoke tested。源码包中没有正式 `scores.json`、`aggregate_metrics.json` 或 baseline checkpoint。

因此：

- **已实现 adapter 不等于复现了方法；**
- **配置存在不等于通过真实 GPU inference；**
- **bridge 存在不等于有公平、可入论文的数值。**

当前 baseline framework 的工程完整度较高，但实验完成度很低。这是 benchmark 论文当前最直接的证据缺口之一。

### 6.2 用户给出的候选不能直接全部列入主表

| 候选 | 原生能力 | 在 RSWM-Bench 中的合理位置 |
|---|---|---|
| Stable Diffusion / SDXL | 图像生成/修补；SDXL 是 text-to-image | image outpainting adapted track，不是原生连续视频或相机轨迹方法。[SDXL](https://arxiv.org/abs/2307.01952) |
| VideoCrafter2 | text-to-video | text-only video baseline；无证据表明原生接收逐帧遥感相机轨迹。[VideoCrafter2](https://arxiv.org/abs/2401.09047) |
| AnimateDiff | 给 T2I 模型插入 motion module | adapted animation baseline；不能标成原生 camera-conditioned world model。[AnimateDiff](https://arxiv.org/abs/2307.04725) |
| Genie | latent-action interactive environment | 任务、域和动作接口差异大；没有可验证适配前只列为候选，不进入主表。[Genie](https://arxiv.org/abs/2402.15391) |
| Cosmos | 可定制 world foundation model 平台 | 若有许可允许的 checkpoint 和明确 image/action/trajectory adapter，可进入 world-model track。[Cosmos](https://arxiv.org/abs/2501.03575) |
| Open-Sora | 支持 text/image-to-video | 可作为 generic I2V；没有显式相机轨迹时进入 text-only/adapted track。[Open-Sora](https://arxiv.org/abs/2412.20404) |
| DiffusionSat | 遥感图像生成、时序生成、超分和 inpainting | remote-sensing image/adapted rollout baseline；需明确任务适配。[DiffusionSat](https://arxiv.org/abs/2312.03606) |
| SatMAE | 遥感 masked-autoencoder 表征学习 | 不是生成模型，不能作为 rollout 生成 baseline；可作为特征/语义 evaluator 或 encoder ablation。[SatMAE](https://arxiv.org/abs/2207.08051) |
| MVSplat | 稀疏多视图输入的 3D Gaussian 方法 | 输入条件更强，必须放在 separate multi-view track，不能与单图方法直接排名。[MVSplat](https://arxiv.org/abs/2403.14627) |

一个可信的 benchmark 主表需要同时报告 native input、适配方式、是否训练、是否见过测试来源、成功率、checkpoint hash 和许可证，而不是仅按论文所属类别分组。

## 7. Leaderboard 现状与缺口

### 7.1 已有能力

- 隐藏测试生成；
- Track A/B 提交元数据；
- 严格提交校验和拒收码；
- official scorer；
- 标准结果目录；
- CSV paper table export；
- result/card/manifest 交叉校验；
- external-data declaration 与 source leakage 检查框架。

### 7.2 尚不存在的 leaderboard 层

仓库当前没有顶层 `benchmark/`、`leaderboard/`、`dataset/`、`benchmark_card.yaml` 或根级 `evaluate.py`。现有 paper table exporter 不是 leaderboard，原因是它没有：

- task × track × domain × difficulty 的排名键；
- metric direction、tie-break、缺失值和失败率排名规则；
- leaderboard version 与 evaluator version；
- submission lifecycle、审核状态、撤回/重算和历史版本；
- 结果签名、容器/环境身份和公开 audit record；
- 防 test-overfitting 的提交频率/团队策略；
- 人类评价或 composite-score 校准记录；
- 可长期访问的发布平台和维护政策。

另一个已记录风险是：AID 是公开源数据，参赛者可能从公开 condition frame 检索原图并重建隐藏 GT。当前主要依赖声明和组织者核验，而非密码学隔离。正式 leaderboard 必须把这一点作为威胁模型和违规政策公开。

## 8. 缺陷列表（按阻断级别）

### P0：发布或 headline claim 阻断

1. **official prompt schema 存在分叉。** `common.sh` 的 official profile 要求 `net_continuous`，但 canonical builder 在环境变量未设置时默认 `net`；`build_correct.sh`、full pipeline 和自动启动器没有显式冻结该变量，`check_dataset.py` 也不验证 `prompt_schema`。新机器上直接跑 full pipeline 可能构建出四方向文本库，却被当作 continuous-direction 主库。
2. **主数据许可未澄清。** AID 是 headline 数据源，但其发布卡片仍为 unknown license；公开像素或衍生数据能否重新分发尚未得到证据支持。
3. **没有正式 baseline 结果。** 所有 card 的 `result_available` 均为 false，不能形成可信 benchmark 主表。
4. **当前数据不支持宽泛 world-model/3D claim。** GT 是静态图像二维滑窗，pose/intrinsics 不是实测相机标定；必须收窄 claim，或增加真正多视图/真实序列子集。
5. **本地包没有正式数据和结果 artifact。** 2000 样本、官方 split、I3D-FVD、三 seed 和跨数据集结论均尚未验证。

### P1：benchmark 完整性阻断

1. 三个拟议任务尚未形成独立、机器可读、可排名的 task contract；当前只是一个任务加两个 conditioning track。
2. 没有 80/50/20/<10 四档构建 split；现有分桶与拟议档位定义不同。
3. 没有 8-direction/diagonal annotation、方向平衡门或方向专属 split。
4. 没有真正 `real_world` 多视图/视频域；generalization 路径也未接入 full pipeline。
5. 缺 temporal LPIPS、warping error、feature correspondence、DINO、输出语义指标。
6. 标准 I3D-FVD、LPIPS 权重和外部 metric checkpoints 尚未在正式环境核验。
7. manifest 的 `source_dataset`/`source_split` 主要来自 CLI 全局参数，而不是逐样本 provenance；这不适合未来混合域 release。
8. official split 没有对方向、overlap、trajectory type 和 horizon 做联合覆盖硬门。
9. RSWM Score 无定义、无校准、无版本化；当前不能用于主排名。
10. 没有正式 leaderboard 数据模型、版本策略、维护策略和公开平台。

### P2：可维护性与文档问题

1. `DATASHEET.md` 的 N=4000 与 canonical full pipeline 的 N=2000 不一致。
2. 根目录没有统一 `README.md`，运行、复现、提交和历史 README 分散。
3. 顶层包存在冗余嵌套副本，Wan 代码也有两份同步副本；虽有漂移测试，仍增加维护风险。
4. `camera.json`/`trajectory.json`/`overlap.json` 和 machine-readable benchmark card 缺失。
5. SemanticGate 的小规模人审、单标注者和类别偏置不足以支持强语义一致性结论。
6. RESISC45/PatternNet 拼接 provenance 不完整，AID taxonomy 排除了一个类别。

## 9. 与高质量 benchmark 论文的差距

这里的“标准”不是某一条 ICLR/CVPR 强制清单，而是代表性 benchmark 工作和公开数据规范的共同要求。

| 方面 | 当前状态 | 目标状态 |
|---|---|---|
| 科学问题 | 有明确低 overlap + trajectory 问题，但仍与 LoRA 主实验强绑定 | task 本身独立于 RS-Roll；LoRA 只是一个 reference baseline |
| 数据代表性 | 真实图像上的二维合成轨迹 | 明确 controlled 与 real-world 子集，且 claim 与数据物理含义一致 |
| 任务覆盖 | 单任务 + Track A/B | 3 task × conditioning track 的正交矩阵 |
| 难度控制 | endpoint <10%，经验分桶 | 固定、版本化 overlap ladder 和方向/horizon 平衡 |
| 指标有效性 | 多个技术指标和 instrument floor | 四类指标、失败模式、人类/专家校准、权重敏感性 |
| baseline | 平台齐全，正式结果为空 | 多类别、真实运行、统一资源/输入/适配披露 |
| 统计 | 工具存在 | 每张主表自动输出 seed 与 source-cluster uncertainty |
| 发布 | datasheet、submission 文档、隐藏测试代码 | 明确许可证、持久链接/DOI、机器可读卡、维护计划 |
| leaderboard | scorer + table exporter | 版本化 leaderboard、审计、重算、排名政策 |
| 复现 | checksum/manifest 很强 | 再补 evaluator/container/checkpoint 固定和正式 artifact |

Datasheets 强调记录数据动机、组成、采集、推荐用途和限制；当前 `DATASHEET.md` 已有很好的基础。[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) 但面向公开发布，还需要许可证、持久托管和机器可读 metadata。Hugging Face dataset card 也把 license、size、task 和 bias/usage context 作为核心信息。[Dataset Cards](https://huggingface.co/docs/hub/datasets-cards) NeurIPS 现行数据指南进一步要求长期保存、持久标识、许可/访问限制与机器可读元数据；这些可作为 benchmark release 的上限参考。[NeurIPS Data Guidelines](https://dev.neurips.cc/Conferences/2026/MainTrackHandbook)

## 10. 本次验证结果

本次只运行无 GPU、无正式数据、无网络下载的低成本检查：

| 检查 | 实际结果 |
|---|---|
| 活动 Python/JSON/YAML 静态解析 | PASS：251 Python、31 JSON、27 YAML |
| `tests.test_baseline_platform` | PASS：13/13 |
| `evaluate_rollout_protocol.py --self-test` | PASS：10/10 |
| `test_metrics_known_answer.py` | PASS：68，FAIL：0，SKIP：1（本地缺 `lpips`） |
| `test_submission_contract.py` | 未完成：T1–T11 已执行部分通过，随后因 Windows 临时目录 ACL 报 `PermissionError`；不能记为全套 PASS，也没有证据表明是算法失败 |
| 正式 HDF5、GPU、CLIP、I3D、模型 inference、三 seed | 尚未验证 |

## 11. 审查结论

### 当前完成度

- 数据构建与审计：**约 75%**；
- frozen manifest / submission contract：**约 80%**；
- 三任务 benchmark 设计：**约 30%**；
- 四类 metrics：**约 45%**；
- baseline 工程：**约 70%**，baseline 实验：**低于 10%**；
- leaderboard/release：**约 20%**；
- 整体公开 benchmark readiness：**约 40%**。

这些百分比是工程审查用的相对成熟度估计，不是实验测量值。

### Readiness verdict

**内部 benchmark 开发：CONDITIONALLY READY。** 可以在不改模型和训练协议的前提下增加 benchmark abstraction layer。  
**公开 benchmark 发布/benchmark paper headline：NOT READY。** 必须先解决 prompt schema、许可、任务冻结、四档 split、真实 baseline 结果和指标校准。  
**昂贵正式实验：暂不建议因“benchmark 升级”而立即重跑。** 应先冻结 v3 schema 和兼容映射，否则新结果可能再次失去可比性。

