# RSWM-Bench 升级方案

**目标定位：** *RSWM-Bench: A Benchmark for Continuous View Extrapolation in Remote Sensing World Models*  
**方案版本：** Draft v0.1（等待确认，尚未实施）  
**日期：** 2026-08-22  
**约束：** 不修改模型结构，不删除 LoRA，不改变已冻结论文实验协议，不重写现有训练流水线；通过兼容层把已有能力升级为独立 benchmark 工程。

## 1. 总体结论

建议采用“**冻结 v2、旁路新增 v3**”的升级方式：

```text
现有训练/数据/评估链（保持）
              │
              ├── legacy H5 / official split / v2 manifest
              │
              ▼
      benchmark compatibility adapter
              │
              ├── task schema + annotations + release card
              ├── unified evaluator + metric registry
              ├── baseline capability contract
              └── leaderboard result contract
```

不应移动或改写 `lora_remotesense/`、`lora_training/`、`wan/` 的核心实现。新 `benchmark/` 层读取已有 HDF5、manifest 和 rollout 结果，统一任务、指标、baseline 和 leaderboard 语义。只有发现 v2 无法表达的 metadata 时，才对旧入口做向后兼容的最小修改。

首要工作不是增加更多模型，而是先冻结：

1. benchmark claim；
2. task × track 的正交定义；
3. overlap/direction/horizon split；
4. annotation 与 evaluator schema；
5. metric 的有效性和排名规则。

## 2. 不可改变的兼容边界

### 2.1 保持不变

- Video Diffusion / Video DiT 模型结构；
- LingBot-Wan I2V backbone；
- LoRA 注入和现有 rank/target 配置；
- Primary `native_attn_cam_r16` 与 seeds 42/43/44；
- 当前 AID 2000 样本 ICLR protocol；
- canonical builder 的窗口 128×222、目标 480×832、steps=9、T=17/25/33；
- 当前 official split 和已产生结果的解释；
- `server_run/run_iclr_full_pipeline.sh` 的 checkpoint/resume 语义；
- 现有 HDF5 runtime 表示 `(T,H,W,C)` `uint8`。

### 2.2 新增而不替换

- `rsroll.benchmark_manifest/v2` 继续可读；
- 新增 `rswm.benchmark_manifest/v3` 或 v2 extension schema；
- 现有 Track A/B 保留为 conditioning track；
- 新 Task 1/2/3 与 Track A/B 做笛卡尔组合；
- 当前低 overlap 主实验映射到 Task 3 / Level 4；
- LoRA 从“主要创新”调整为 benchmark 的 reference adaptation baseline，但不删除任何实验。

### 2.3 Claim guard

在没有真实多视图/真实时间序列数据前，论文必须把当前 AID 部分称为：

> controlled continuous spatial rollout derived from real remote-sensing imagery

不应称为真实 6DoF novel-view synthesis、动态未来预测或完整 3D world simulation。`real_world/` 子集只有在数据确实包含真实多视图、相机/地理标定或真实序列时才能启用。

## 3. 正式 benchmark 设计

### 3.1 四个正交轴

RSWM-Bench 不应再把 task、条件方式和难度混在一个配置名中。建议冻结四个轴：

| 轴 | 枚举 |
|---|---|
| Task | `T1_future_view`、`T2_arbitrary_direction`、`T3_overlap_ladder` |
| Conditioning track | `A_text`、`B_camera`；可选 `C_multiview` 单独排名 |
| Data domain | `synthetic_controlled`、`real_world`、`generalization` |
| Cohort | overlap level、direction bin、trajectory type、17/25/33 horizon |

Leaderboard 的最小排名键为：

```text
(benchmark_version, task_id, conditioning_track, data_domain)
```

overlap、direction、horizon 和 trajectory type 首先作为该排名下的必报 breakdown；只有样本量足够时才建立独立子榜。

### 3.2 Task 1：Single View Future View Prediction

**建议正式名称：** `Trajectory-conditioned Single-view Spatial Rollout`。保留用户给出的论文友好名称，但在协议中避免把 “future” 误读为现实时间预测。

**Input：**

- 一张遥感 condition image；
- Track A：文本化路径摘要；
- Track B：逐帧 camera/trajectory condition；
- task、frame count、分辨率和 coordinate-system metadata。

**Output：**

- 完整 `(T,H,W,3)` rollout；
- `t=0` 为 condition frame；
- 仅 `t=1...T-1` 进入视觉/语义保真评分；
- 轨迹恢复可使用 `t=0` 作为位移原点。

**Primary dimensions：** visual、temporal、semantic、trajectory-conditioned spatial consistency。

### 3.3 Task 2：Arbitrary Direction View Extrapolation

当前连续角轨迹已经是基础能力，但需增加可复核的方向标注和覆盖规则。

**坐标约定：**

- trajectory 继续使用 `(cy,cx)`；
- `0°=right/east`，`90°=up/north`，角度逆时针；
- 同时记录 camera-motion angle 与 content-flow angle，二者相差 180°；
- 所有标注必须带 `coordinate_system_version`。

**报告分桶：**

| direction_id | 角度中心 | 人类可读标签 |
|---|---:|---|
| `E` | 0° | right/east |
| `NE` | 45° | up-right/northeast |
| `N` | 90° | up/north |
| `NW` | 135° | up-left/northwest |
| `W` | 180° | left/west |
| `SW` | 225° | down-left/southwest |
| `S` | 270° | down/south |
| `SE` | 315° | down-right/southeast |

方向 bin 只用于 coverage、split 和 breakdown；Track B 的真正目标仍是连续角，不应把连续任务重新量化成 8 分类。

**自动 split 要求：**

- 每个 split 和每个 horizon 对 8 个方向设最低样本量；
- train/val/test 仍按 source group 分配；
- 可增加 `heldout_diagonal` diagnostic split，但不能把它与完整 arbitrary-direction 主榜混为一谈；
- 多段路径同时报告首段、逐段和净方向覆盖。

### 3.4 Task 3：Low-overlap Spatial Extrapolation

现有 `<10%` 主集映射到 Level 4。建议为新 v3 数据冻结互不重叠的目标带，而不是要求浮点 overlap 精确等于单点：

| Level | 名称 | condition-to-endpoint overlap 目标带 | 说明 |
|---|---|---|---|
| L1 | `overlap_80` | `[0.70,0.90)` | 以 80% 为 anchor |
| L2 | `overlap_50` | `[0.40,0.60)` | 以 50% 为 anchor |
| L3 | `overlap_20` | `[0.15,0.25)` | 以 20% 为 anchor |
| L4 | `overlap_lt10` | `[0.00,0.10)` | 现有 headline protocol |

区间之间故意留出 buffer，降低边界抖动和难度混淆。最终区间可在 Phase 0 通过可行性统计微调，但一经发布必须冻结并版本化。

**重要定义：**

- 主难度：`O(condition t=0, endpoint)`；
- 辅助曲线：`O(condition t=0, frame t)`；
- 连续性审计：`O(frame t-1, frame t)`；
- 三者不能互换。

建议同一 source 在同一 split 内生成成对的 overlap ladder，以做 paired comparison；统计时按 source cluster bootstrap，不能把同源四条轨迹当作四个独立样本。

## 4. Dataset 升级方案

### 4.1 目录语义

```text
dataset/
  benchmark_card.yaml
  README.md
  mlcroissant.json
  real_world/
    README.md
    manifests/
  synthetic_controlled/
    README.md
    manifests/
    annotations/
  generalization/
    README.md
    manifests/
    annotations/
```

`synthetic_controlled/` 的含义是“轨迹和视角序列受控构造”，不是像素一定为合成图像。AID 应标记：

```yaml
pixel_source: real_remote_sensing_image
rollout_construction: planar_sliding_window
camera_physics: controlled_2d_proxy
```

`real_world/` 第一版可以只有 schema 和 `NOT_AVAILABLE` 状态，不能用 AID 滑窗冒充。正式启用前需要完成数据源、真实采集属性、许可、坐标/相机可用性和 train/test 泄漏审计。

### 4.2 Annotation schema

每条样本建议导出一个原子 annotation record；`camera.json`、`trajectory.json`、`overlap.json` 可以作为分表视图，但必须共享 `sample_id`、schema version 和 H5 checksum，避免双重真相。

#### `camera.json`

- `sample_id`；
- `condition_track`；
- `poses` 与 pose convention；
- `intrinsics`；
- `calibration_status`: `measured | derived_planar | placeholder`；
- `units` 与 scale；
- `camera_checksum`。

当前 AID 必须写 `derived_planar`/`placeholder`，不得写 `measured`。

#### `trajectory.json`

- per-frame `(cy,cx)`；
- waypoints、segment angles、segment boundaries；
- path length、net displacement、viewpoint distance；
- trajectory type；
- continuous angle 与 8-bin report label；
- camera/content direction convention；
- checksum。

#### `overlap.json`

- condition-to-frame overlap curve；
- adjacent overlap curve；
- endpoint/mean/min/max；
- target level、target interval、realized value；
- window/scene size和 overlap algorithm version。

### 4.3 `benchmark_card.yaml`

至少包含：

- benchmark/version/schema；
- intended tasks 和不支持的 claims；
- domain、sensor/source、spatial/temporal含义；
- 样本、source、scene、direction、trajectory、horizon、overlap 统计；
- train/val/test 策略与 checksum；
- annotation schema；
- metric/evaluator version；
- data license、code license、access restriction；
- bias、SemanticGate、人审与已知失败；
- maintenance、contact、deprecation 和 citation；
- persistent release ID/DOI（发布时）。

公开数据规范普遍要求 license、size、task、使用边界和机器可读 metadata；Hugging Face 的 [Dataset Cards](https://huggingface.co/docs/hub/datasets-cards) 和 NeurIPS 的 [Data Hosting Guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines) 可作为实现参考。

### 4.4 Split 生成器

v3 split 需继续保留当前并查集 source-disjoint 逻辑，并增加：

```text
source group
  -> assign train/val/test
  -> within each split audit/construct cohorts
       scene × direction × overlap × trajectory_type × horizon
  -> enforce minimum coverage
  -> emit split checksum + imbalance report
```

不要为了平衡 cohort 而把同源样本拆到不同 split。若联合分层无法满足，应 fail loudly 并报告缺口，不做 silent oversampling。

## 5. Metric 升级方案

### 5.1 Metric registry

每个 metric 在配置中固定：

- name/version；
- input scope（frame、clip、sample、dataset）；
- higher/lower is better；
- conditioning frame policy；
- preprocessing、resolution、frame sampling；
- model/checkpoint hash；
- minimum N；
- invalid/missing policy；
- uncertainty method；
- eligible tasks/tracks/domains；
- claim guard。

### 5.2 Visual Quality

保留已有实现并统一 provenance：

- SSIM、PSNR：future frames only；
- LPIPS：future frames only，固定 backbone/weights；
- FID：全部 predicted frame pool，报告 clip 数和 frame 数；
- FVD：`FVD_I3D` 为 headline，`FVD_R3D_PROXY` 只作内部/诊断。

分布指标不对很小的 direction × scene 单元强行计算。需要给出 N 收敛曲线或重采样稳定性。

### 5.3 Temporal Consistency

新增但必须给出严格定义：

1. **Temporal perceptual-change error**：比较生成与 GT 的相邻帧 LPIPS 变化，而不是简单把低变化当作好，避免 persistence 获益；
2. **Flow consistency**：保留当前 2D flow-vector proxy，并增加 per-step curve；
3. **Occlusion-aware warping error**：使用已知二维轨迹/有效共同区域，或冻结 RAFT 与 forward-backward mask；
4. **Temporal flicker residual**：在 trajectory-aligned 后计算高频残差。

若采用 RAFT，必须固定 checkpoint、输入尺度和 occlusion policy。RAFT 是可选的高质量光流基础，但对遥感域仍需 GT/合成已知运动校准。[RAFT](https://arxiv.org/abs/2003.12039)

### 5.4 Spatial Geometry Consistency

分为两个层次，防止 2D proxy 被误写成 3D：

**Controlled planar domain：**

- registration path error；
- segment angular error；
- turn F1/timing；
- waypoint/endpoint error；
- Fréchet/DTW；
- known-translation reprojection/warping error；
- feature correspondence in valid common area；
- DINOv2 patch correspondence/similarity。

**Real multi-view domain（有真实标定时）：**

- pose-conditioned reprojection error；
- epipolar/correspondence consistency；
- depth/geometry consistency；
- camera-conditioned endpoint/path error。

二者必须在 result schema 中使用不同 metric names，不能合并成一个“geometry”数字后掩盖物理语义。

### 5.5 Semantic Consistency

新增输出级 evaluator，而不是复用构建时 SemanticGate 判定：

- scene-category preservation；
- CLIP/remote-sensing encoder similarity；
- source-to-future semantic drift curve；
- 关键类别（water/urban/rural/general）分层结果；
- 小规模专家 pairwise/hallucination annotation。

SatMAE 是表征模型而非生成 baseline，可作为 remote-sensing feature evaluator 候选；是否优于 DINO/CLIP 必须通过校准集决定。[SatMAE](https://arxiv.org/abs/2207.08051)

### 5.6 RSWM Score

建议分两步：

**v0（首发）：**

- 不作为主排名；
- 输出 Visual、Temporal、Geometry、Semantic 四个 0–100 group score；
- 配置文件写权重，但标记 `experimental=true`；
- 主表同时保留原始指标。

**v1（校准后）：**

1. 以 GT oracle、persistence、wrong-direction、已知扰动建立 metric anchors；
2. 用专家/人类 pairwise judgment 校准 normalization 与权重；
3. 对权重做敏感性分析；
4. 只有不同合理权重下排名稳定，才作为 headline；
5. 用加权几何均值或调和均值，防止一个维度极差被其他维度完全抵消；
6. metric 缺失、valid rate 不足或 success rate 不足时不计算综合分，不做零填充。

配置示意：

```yaml
score_version: rswm_score/v0
experimental: true
groups:
  visual: 0.25
  temporal: 0.25
  geometry: 0.35
  semantic: 0.15
normalization: frozen_anchor_v1
missing_policy: ineligible
```

上述数值只是 schema 示例，不是本方案预先决定的论文权重。

### 5.7 统计协议

- 训练方法：至少报告 seeds 42/43/44 的 mean、std 和全部单 seed；
- 同一测试 source 的方法比较：paired source-cluster bootstrap 95% CI；
- overlap ladder：同源配对分析；
- 分布指标：重算 feature statistics 的 bootstrap/样本量研究；
- 多重 breakdown：报告样本数，不对极小格子作显著性结论；
- 所有方法报告生成成功率、valid rate 和失败原因分布；
- 失败样本不能静默从均值中消失。

## 6. Baseline 扩展方案

### 6.1 先按能力分轨，再按模型类别展示

每个 baseline card 必须声明：

- native input/output；
- trajectory support：`none | text_summary | continuous_camera | multiview`；
- adaptation：`none | prompt_adapter | recursive_rollout | finetuned`；
- 是否使用 GT/full scene/future frames；
- training data 与 test-source overlap declaration；
- checkpoint/version/license/hash；
- compute、precision、seed、成功率；
- eligible task/track；
- `native` 或 `adapted` 标签。

输入条件不同的方法不能仅靠 resize 到同一输出后直接排名。

### 6.2 建议的首发 baseline 最小集

| 角色 | 方法 | 目的 |
|---|---|---|
| Diagnostic | persistence | 静止下界 |
| Diagnostic | random-direction translation | 方向指标负对照 |
| Upper bound | GT oracle / privileged translation | 指标校准，不进入排名 |
| Image diffusion | SDXL inpainting/outpainting adapted rollout | 单图生成类代表 |
| Generic video | frozen I2V backbone | 无遥感适配基线 |
| Camera ablation | same backbone, no camera | 证明轨迹条件贡献 |
| Reference adaptation | RS-Roll LoRA | 当前方法作为 reference baseline |
| Remote sensing | DiffusionSat adapted outpainting/rollout | 遥感生成类代表 |
| External video | 一个实际可运行的 Open-Sora/VideoCrafter 类 I2V | 通用视频模型代表 |

只有上述最小集有真实 checkpoint、完整 test manifest 结果和可审计日志后，再扩展 Genie/Cosmos/MVSplat 等高成本或异构输入方法。

### 6.3 候选方法的准入结论

- SD/SDXL、DiffusionSat：image/adapted track；
- VideoCrafter、AnimateDiff、Open-Sora：text-only 或 I2V adapted track，不能假称 camera-native；
- Genie：没有等价 action/trajectory adapter 和可复现权重前不准入；
- Cosmos：通过 capability probe 后进入 world-model track；
- SatMAE：不作为生成 baseline；
- MVSplat：进入 separate multi-view track。

## 7. Evaluation 与 leaderboard

### 7.1 用户入口

零参数入口必须可用：

```bash
python evaluate.py
```

其默认读取 `benchmark/default_eval.yaml`，从约定的 `submission/` 目录读取提交，并写入
`results/default/`。显式参数只覆盖默认配置，便于服务器流水线和第三方提交复用同一入口：

```bash
python evaluate.py \
  --task T3_overlap_ladder \
  --track B_camera \
  --submission /path/to/submission \
  --output results/my_method
```

输出：

```text
results/my_method/
  result.json
  sample_metrics.jsonl
  audit.json
  environment.json
  baseline_card.yaml
  paper_tables/
    main.csv
    by_overlap.csv
    by_direction.csv
    by_horizon.csv
```

根入口只负责编排和 schema validation；底层优先复用现有 scorer、trajectory metrics、baseline evaluator 和 table exporter。

### 7.2 Leaderboard 数据模型

每条 entry 至少包含：

- benchmark/task/track/metric/evaluator version；
- dataset/split/manifest checksum；
- method/card/checkpoint/container identity；
- raw metrics、group metrics、CI、N、success/valid rate；
- per-cohort breakdown；
- external data declaration；
- verification status：`unverified | reproduced | official`；
- submitted/recomputed/withdrawn timestamp；
- result signature。

### 7.3 排名规则

- 诊断基线和 upper bound 不参与排序；
- native 与 adapted 方法显式标注；
- multiview 单独排名；
- 第一版按关键 metric 或四 group score 展示，不强制单一总榜；
- success rate 或有效样本率低于冻结门槛时标记 ineligible；
- tie-break、缺失值、超时、失败和重提交规则写入 `leaderboard_v1.yaml`；
- evaluator 版本变化时保留旧榜，并对有原始 rollout 的 entry 执行可审计重算。

### 7.4 防泄漏与维护

- 保留 hidden GT 和 source-disjoint；
- public dev 与 hidden test 分离；
- 对公开 AID 的 source retrieval 风险设外部数据声明、组织者审计和违规政策；
- 设提交频率限制，记录 benchmark version；
- 公开 scorer 的 dev 版本，hidden scorer 使用相同 schema；
- 发布 maintenance/deprecation policy 和长期托管位置。

## 8. 建议工程结构

```text
benchmark/
  __init__.py
  default_eval.yaml
  cli/
    evaluate.py
  specs/
    tasks_v3.yaml
    metrics_v1.yaml
    score_v0.yaml
    leaderboard_v1.yaml
  schemas/
    benchmark_card.schema.json
    task.schema.json
    annotation.schema.json
    submission.schema.json
    result.schema.json
    leaderboard.schema.json
  datasets/
    legacy_h5_adapter.py
    export_annotations.py
    build_task_splits.py
    validate_release.py
    statistics.py
  evaluation/
    runner.py
    aggregation.py
    provenance.py
    submission.py
  metrics/
    registry.py
    visual.py
    temporal.py
    geometry.py
    semantic.py
    composite.py
  baselines/
    registry.py
    adapter.py
    capability.py
    cards/
  leaderboard/
    schema.py
    rank.py
    export.py
    verify.py
dataset/
  benchmark_card.yaml
  README.md
  mlcroissant.json
  real_world/
  synthetic_controlled/
  generalization/
tests/
  benchmark/
evaluate.py
```

`benchmark/` 是新公共 API；`server_run/` 仍是现有 ICLR pipeline 的运行层。不要把两者互相替换。

## 9. 预计新增文件

### 9.1 最小可发布抽象层

预计新增 **30–36 个文件**：

- 顶层入口：`evaluate.py`；
- `benchmark/` package、默认评测配置与四份 spec；
- 六份 JSON schema；
- legacy H5/manifest adapter；
- annotations 和 task split 工具；
- metric registry 与四类 metric wrapper；
- baseline capability contract；
- leaderboard schema/rank/export；
- benchmark card、Croissant metadata 和 release README；
- 8–10 个 benchmark unit/integration tests。

### 9.2 完整用户目标

包含多个真实外部 baseline adapter、metric checkpoints、human-eval schema 和 release tooling 后，预计新增 **44–58 个文件**。外部项目源码不应复制进主仓，只保存 adapter、版本、安装说明和 checkpoint identity。

## 10. 预计修改文件

以下是计划范围，不代表已经修改：

| 文件 | 预计最小修改 |
|---|---|
| `server_run/build_correct.sh` | 显式冻结 official prompt/profile metadata，消除 `net`/`net_continuous` 分叉 |
| `server_run/check_dataset.py` | 验证 prompt schema、annotation/export identity 和 benchmark version |
| `server_run/run_iclr_full_pipeline.sh` | 记录 task/spec/metric version；可选调用新 evaluator，默认 v2 行为不变 |
| `server_run/make_official_splits.py` | 增加显式 v3 输出模式和联合 cohort audit；v1 默认输出保持 |
| `server_run/score_submission.py` | 通过 adapter 输出新 result schema；旧 score report 保留 |
| `server_run/evaluate_rollout_protocol.py` | 读取统一 overlap spec，补固定 80/50/20/<10 breakdown |
| `server_run/evaluate_baseline_results.py` | 接 metric registry 和统一 provenance，不重写已有 metric |
| `server_run/baselines/protocol.py` | 增加 task/track/domain/capability/version 字段，保留 v2 reader |
| `server_run/baselines/result_adapters.py` | 适配新 benchmark sample/result contract |
| `server_run/baselines/cards.py` | 验证新的 capability/license/checkpoint 字段 |
| `lora_remotesense/build_dataset_from_benchmark.py` | 只补 machine-readable metadata/sidecar identity；不改像素或轨迹生成逻辑 |
| `tools/export_paper_tables.py` | 增加 task/track/cohort tables 和 CI 列；不自动伪造综合分 |
| `DATASHEET.md` | 统一 N=2000 与数据边界，链接 benchmark card |
| `SUBMISSION.md` | 增加三任务、track 和 leaderboard policy |
| `REPRODUCIBILITY.md` | 增加 benchmark version、容器/checkpoint/metric identity |

预计修改 **13–17 个现有文件**。优先通过 wrapper 复用 `lora_remotesense/new/metrics.py` 和 `trajectory_metrics.py`；除非测试证明必须，否则不直接改这两个数值核心。

`wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py` 已支持连续方向和 overlap 区间。第一阶段应由新 split/annotation 层复用，**不计划修改**；只有在冻结 v3 后确认必须持久化 `target_overlap_level` 或稳定 sample ID，才做向后兼容的 metadata 增补。

明确不计划修改：

- 模型模块；
- LoRA 注入逻辑；
- optimizer、学习率、batch、训练步数；
- `configs/iclr_main/` 的实验超参数；
- 既有 v2 结果解释。

## 11. 测试计划

### 11.1 Schema 与数据

- v2 manifest → v3 adapter 的 golden test；
- camera/trajectory/overlap sidecar 与 H5 checksum 一致；
- source-disjoint 与四档/八方向/horizon 最低覆盖；
- 坐标系、camera/content 180° 约定；
- placeholder intrinsics 不能冒充 measured；
- duplicate/source leakage fail-closed；
- split 重跑逐位一致。

### 11.2 Metric known-answer

- identical、known translation、wrong direction、static、flicker、occlusion；
- condition frame 不进入保真；
- temporal metric 不奖励 persistence；
- registration/DINO/flow 在已知平移上的单调性；
- FID/FVD 最小 N；
- missing/invalid 不零填充；
- score normalization 和权重敏感性。

### 11.3 Baseline/evaluation

- 每种 capability 的输入拒收测试；
- native/adapted/multiview 不串榜；
- producer identity、checkpoint hash 和 seed 不一致时失败；
- partial failure 与 success-rate gate；
- `python evaluate.py` 端到端 synthetic fixture；
- JSON、CSV 和 leaderboard entry 同源。

### 11.4 Release gate

- 数据/代码/权重 license 完整；
- benchmark card schema；
- Croissant/metadata validation；
- clean environment offline reproduction；
- hidden/public test redaction；
- 至少一个外部团队或独立环境复跑；
- 不执行任何正式 GPU job，除非用户另行明确授权。

## 12. 开发顺序与退出条件

### Phase 0：协议冻结（先做）

1. 确认 task 名称和 claim boundary；
2. 确认 v2 → v3 映射；
3. 冻结 prompt schema；
4. 冻结四档 overlap 区间和 8-direction 定义；
5. 决定 real-world 子集是首发必需还是后续扩展；
6. 明确 AID/其他数据许可和发布方式。

**退出条件：** `tasks_v3.yaml`、annotation schema 和 license decision 获得审查确认。此阶段完成前不重建数据、不跑训练。

### Phase 1：兼容层与 benchmark card

1. 创建 `benchmark/` 骨架；
2. 实现 v2 H5/manifest read-only adapter；
3. 导出 sidecar annotations；
4. 生成 statistics/benchmark card；
5. 修正 N=2000 和 prompt schema 文档分叉。

**退出条件：** 旧 H5 不改字节即可通过 v3 validation。

### Phase 2：Task split

1. source-group split；
2. direction/overlap/horizon cohort；
3. 四档构建可行性 dry-run；
4. overlap ladder 小样本 rehearsal；
5. 任务 manifest/checksum。

**退出条件：** 100 样本/小规模 fixture 覆盖全部 task/cohort，且 source-disjoint、overlap 和轨迹硬约束通过。

### Phase 3：Metric registry

1. 包装现有视觉/轨迹指标；
2. 新增 temporal/warping/DINO/semantic；
3. known-answer 与故障注入；
4. metric sample-size study；
5. 专家小规模校准；
6. RSWM Score 保持 experimental。

**退出条件：** 每个 headline metric 有版本、checkpoint hash、known-answer、有效范围和失败语义。

### Phase 4：Baseline 最小集

1. 跑通诊断基线；
2. frozen I2V 与 no-camera；
3. RS-Roll LoRA；
4. SDXL/DiffusionSat adapted；
5. 一个外部 generic I2V；
6. 统一 resource/result cards。

**退出条件：** 至少 5 个可部署/适配方法加诊断基线在同一 hidden manifest 上有完整结果，且 `result_available=true` 有实际 artifact 支持。

### Phase 5：Leaderboard 与发布

1. 统一 `evaluate.py`；
2. leaderboard schema/ranking/version；
3. public dev + hidden test；
4. release card、license、DOI/持久托管；
5. 独立复跑；
6. paper tables 与 benchmark website/静态榜。

**退出条件：** 第三方能从发布说明生成提交、复现评分，并理解每个数字支持和不支持的 claim。

## 13. Benchmark 论文可成立的贡献点

下列是**计划中的贡献**，不是当前已经验证的论文结论：

1. **新问题定义：** 遥感单视图在给定连续二维/相机轨迹下的空间视角外推，而非一般 text-to-video；
2. **可控难度轴：** arbitrary direction、multi-segment trajectory、17/25/33 长时域和 80/50/20/<10 overlap ladder；
3. **多维评估：** 视觉、时序、空间轨迹/几何和语义分解，并明确 2D proxy 与真实 3D metric 的边界；
4. **公平比较协议：** source-disjoint hidden test、冻结 manifest、严格提交契约、native/adapted/multiview capability track；
5. **遥感域分析：** AID 主域、跨数据集 generalization、类别和 SemanticGate 偏置审计；
6. **系统性 baseline study：** image diffusion、video diffusion、world model、remote-sensing model 统一输入/输出/评估；
7. **可复现 benchmark artifact：** machine-readable cards、checksums、版本化 evaluator、leaderboard 和 paper-table export。

LoRA 的论文角色调整为：

> a parameter-efficient reference adaptation baseline for camera-conditioned remote-sensing rollout

而不是 benchmark 论文的唯一核心贡献。

## 14. 需要确认的决策

在修改代码前需要项目负责人确认以下事项：

1. 新 paper claim 是否接受“controlled 2D spatial rollout”边界，还是首发必须加入真实多视图数据；
2. official prompt 是否统一冻结为 `net_continuous`；
3. overlap 四档是否采用本文建议的目标带；
4. Track A/B 是否继续命名，还是改为 `text-only`/`camera-conditioned`；
5. RSWM Score 是否同意首发仅作 experimental；
6. AID 像素是否有可核验的再分发许可，若无，是否采用 build-from-source release；
7. 首发外部 baseline 最小集及可承担的 GPU 预算；
8. `real_world/` 是首发 blocker，还是明确标记为后续扩展。

## 15. 当前行动建议

在获得确认之前不改代码。下一步最安全的动作是召开一次协议审查，只确认 Phase 0 的八项决策；确认后先实现 schema、adapter 和 synthetic fixture，不先重建 2000 样本，也不先跑昂贵 baseline。
