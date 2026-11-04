# RSWM-Bench Phase 1 Benchmark Abstraction Layer 设计

> 文档状态：设计已确认；Phase 2 M1-M3 与 Phase 2.5 Model Identity Schema Alignment 已实现，M4 及以后未授权  
> 日期：2026-08-22；确认、M1-M3 与 Phase 2.5 实施日期：2026-08-23  
> 上位协议：`RSWM_BENCH_PROTOCOL_V1.md`  
> 配套审查：`RSWM_PHASE1_ARCHITECTURE_REVIEW.md`  
> 本阶段产物：本设计本身保持为架构规范；获批后的 M1-M3 仅新增 core types、schema、mock world model、只读 `LegacyH5Adapter`、task contract 与 CPU fixture tests。未修改模型、LoRA、ICLR pipeline、既有 HDF5，未执行训练、正式 evaluation 或 GPU 作业。

## 1. 目标与非目标

### 1.1 目标

在不改变现有 RSWM/RS-Roll 模型与 ICLR 实验的前提下，增加一个通用、可审计的 benchmark abstraction layer，使以下对象彼此解耦：

- world model 的具体实现；
- T1/T2/T3 task 定义；
- HDF5/manifest 数据载体；
- trajectory、direction 与 overlap 语义；
- visual/temporal/spatial/semantic metric；
- native/adapted/multiview baseline 能力；
- submission、leaderboard 和 paper artifact。

完成后的 benchmark 应通过同一个公共合同接入现有 LingBot-World reference model（Wan I2V A14B 架构实现），并为 VideoCrafter、Open-Sora、Cosmos、DiffusionSat 等未来 adapter 留出明确扩展位；本阶段不实现、下载、运行或宣称复现这些外部模型。

### 1.2 非目标

本设计不授权以下行为：

- 修改 Wan/Video DiT、VAE、T5、camera injection 或 LoRA 结构；
- 修改 optimizer、训练超参数、primary seeds、checkpoint 解释或已有实验身份；
- 修改 canonical builder、轨迹生成核心、HDF5 格式或正式数据；
- 重建 AID 2000 样本 HDF5；
- 改写现有 metric 数值定义或生成 composite RSWM Score；
- 把真实数据设为 V1 的强制发布条件；
- 将历史 ICLR 结果追溯标记为完整 V1 T1/T2/T3 结果；
- 在 Phase 1 启动 GPU、training、evaluation 或 external baseline。

## 2. 设计原则

1. **协议高于实现。** task、track、direction、overlap 和 metric policy 来自 frozen protocol；adapter 不得自行解释。
2. **兼容优先。** 现有 HDF5、manifest v2、checkpoint 和结果保持原样；新增语义用带 checksum 的 sidecar 表达。
3. **单一真源。** Wan 生成复用现有 `VideoBackbone/WanBackbone`；metric 和 trajectory 数值复用现有函数，不复制算法。
4. **最小可见输入。** task 只把所属 track 允许的信息投影给模型；GT、full scene 和 hidden trajectory 留在 organizer 侧。
5. **失败显式。** shape、dtype、ID、checksum、能力不匹配、缺 metric 与无效 estimator 均不得静默修补或零填充。
6. **future-only 一次。** t=0 只作为 condition/连续性审计，不进入生成质量评分；裁剪只能发生一次并被记录。
7. **能力分榜。** `A_text`、`B_camera`、`C_multiview` 与 native/adapted/multiview capability 不混排。
8. **轻量核心。** domain types、协议验证与 task contract 不导入 torch、Wan 或 h5py；重依赖限定在具体 adapter。
9. **旧入口不变。** 新 runner 初期为 opt-in，现有 full pipeline、训练和 `wan-inline` 默认路径保持可回退。
10. **证据不升级。** stub、adapter 或 card 的存在不等于模型已运行、结果已得到或 baseline 已 admitted。

## 3. 目标架构图

```text
                         Frozen protocol / benchmark spec
                task + track + domain + capability + manifest checksum
                                         |
                                         v
        +------------------------- Benchmark Task Layer -------------------------+
        | BaseTask                                                               |
        |  T1SpatialRolloutTask  T2ArbitraryDirectionTask  T3OverlapLadderTask   |
        |  prepare_input()       validate_output()          evaluate()            |
        +-------------+---------------------+----------------------+--------------+
                      |                     |                      |
              public ModelInput      canonical Rollout      EvaluationView
                      |                     |                      |
                      v                     |                      v
        +---------- World Model --------+   |      +---------- Evaluation --------+
        | WorldModel protocol           |   |      | EvaluationRunner             |
        | WorldModelCapabilities        |   |      | Metric registry              |
        |                               |   |      | visual / temporal / spatial  |
        | LingBotWorldModelAdapter -----+---+      | semantic (future extension)  |
        | Future adapter placeholders   |          +---------------+--------------+
        +---------------+---------------+                          |
                        |                                          v
              existing VideoBackbone                     MetricResult / audit
                        |                                          |
              existing WanBackbone                                  v
                        |                              Result bundle / leaderboard
              existing inference.py                         / paper exporter
                        |
              LingBot-World base-cam checkpoint

        +-------------------------- Dataset Layer -------------------------------+
        | BenchmarkDataset protocol                                               |
        | LegacyH5Adapter (read-only)                                              |
        | manifest v2 validator + benchmark spec sidecar                          |
        | PublicSample / OrganizerSample / GroundTruthBundle                      |
        +-----------------------------+--------------------------------------------+
                                      |
                                      v
                            existing HDF5 + manifest v2

        +------------------------- Trajectory Layer ------------------------------+
        | TrajectoryCondition / TrajectoryPlan / Direction / OverlapProfile       |
        | TrajectoryGenerator protocol + LegacyTrajectoryAdapter                  |
        +-----------------------------+--------------------------------------------+
                                      |
                                      v
                       existing v5/camera/overlap trajectory code

        +--------------------------- Baseline Layer ------------------------------+
        | BaselineCapabilities / BaselineCardV2 / admission validator             |
        | LegacyBaselineAdapter -> existing server_run/baselines registry          |
        | native / adapted / multiview capability partitions                      |
        +--------------------------------------------------------------------------+
```

### 3.1 依赖方向

允许的依赖方向为：

```text
CLI/orchestrator
    -> task
        -> core types
        -> dataset adapter
        -> world model protocol
        -> evaluation runner
    -> baseline registry

LingBotWorldModelAdapter -> existing backbone.py -> existing inference.py -> wan/*
LegacyH5Adapter           -> manifest v2 validator + h5py
Metric wrappers           -> existing metrics.py / trajectory_metrics.py
LegacyBaselineAdapter     -> existing server_run/baselines/*
```

禁止的反向依赖：

- `wan/**` 不导入 `benchmark/**`；
- `train.py` 不导入 task/evaluation；
- task 不导入 Wan、LoRA 或具体 baseline；
- metric 不调用模型；
- model adapter 不读取 GT 或 leaderboard；
- core types 不导入 torch/h5py；
- 新 package 不通过目录名或 prompt 字符串猜测 track。

## 4. 建议新增目录结构

以下是 **Phase 2 实现候选结构**，本次未创建这些代码文件：

```text
models/
  __init__.py
  world_model.py
  registry.py
  adapters/
    __init__.py
    wan_adapter.py
    future_video_model_adapter.py

benchmark/
  __init__.py
  core/
    __init__.py
    types.py
    contracts.py
    errors.py
  spec/
    __init__.py
    schema.py
    loader.py
  tasks/
    __init__.py
    base.py
    registry.py
    spatial_rollout.py
    arbitrary_direction.py
    overlap_ladder.py
  datasets/
    __init__.py
    base.py
    legacy_h5.py
    manifest_v2.py
  trajectory/
    __init__.py
    types.py
    conventions.py
    trajectory_generator.py
    legacy_adapter.py
  runner.py
  cli.py

evaluation/
  __init__.py
  metric.py
  registry.py
  runner.py
  visual/
    __init__.py
    legacy_metrics.py
  temporal/
    __init__.py
    legacy_metrics.py
  spatial/
    __init__.py
    legacy_metrics.py
  semantic/
    __init__.py
    unavailable.py

baselines/
  __init__.py
  capability.py
  card.py
  registry.py
  admission.py
  legacy_adapter.py
  future_external.py

tests/benchmark/
  test_world_model_contract.py
  test_wan_adapter_mock.py
  test_task_input_visibility.py
  test_task_contracts.py
  test_legacy_h5_adapter.py
  test_manifest_sidecar.py
  test_trajectory_conventions.py
  test_overlap_levels.py
  test_future_only.py
  test_metric_wrappers.py
  test_baseline_capabilities.py
  test_legacy_parity.py
  fixtures/
    manifest_v2_minimal.json
    benchmark_spec_v1_minimal.json
```

目录名 `models/`、`benchmark/`、`evaluation/`、`baselines/` 与用户指定目标一致。为了避免名字冲突，根 `baselines/` 只提供 benchmark 公共能力和兼容包装；现有执行实现仍留在 `server_run/baselines/`。

## 5. 核心数据对象

### 5.1 `BenchmarkSample`

用户要求的 sample 结构保留，但应使用不可变、带类型的对象而不是任意 dict：

| 字段 | 类型/语义 | 来源 |
|---|---|---|
| `id` | 稳定 sample ID | manifest v2 `sample_id` |
| `image` | condition image，canonical `H×W×3 uint8` | HDF5 `image` |
| `trajectory` | `TrajectoryPlan`；organizer 完整视图 | HDF5 `trajectory/poses/intrinsics` + manifest checksum |
| `overlap` | `OverlapProfile` | manifest v2 frozen profile |
| `direction` | `DirectionTarget`，同时含 camera/content bearing | 从冻结 trajectory 按唯一 convention 计算 |
| `horizon` | 总帧数 T，V1 为 17/25/33 | manifest `frame_count` |
| `metadata` | 只读 provenance、scene、task eligibility、checksum | HDF5 attrs + manifest |

为防止信息泄漏，实际 API 分为三种视图：

1. `OrganizerSample`：内部包含完整 `BenchmarkSample` 和 `GroundTruthBundle`；只存在于可信 runner；
2. `PublicSample`：可用于公开 manifest/提交说明，不含 GT 像素和 hidden control；
3. `ModelInput`：由 task 按 track 白名单生成，是唯一允许传给 `WorldModel.generate()` 的对象。

### 5.2 `GroundTruthBundle`

```text
GroundTruthBundle
  rollout              (T,H,W,3) uint8，含 t=0
  rollout_checksum
  full_scene?           organizer-only，非所有 metric 必需
  commanded_trajectory  evaluator-only；Track A 不传给模型
  source_reference      不向 submission 暴露的 HDF5 locator
```

GT 不放进 `metadata`，避免 adapter 通过通用 metadata 意外读取。对进程外 submission，建议生成剥离 GT 的 request bundle，而不是在同一 JSON 中用命名约定隐藏字段。

### 5.3 `TrajectoryCondition`

```text
TrajectoryCondition
  track_id: A_text | B_camera | C_multiview
  prompt_schema
  scene_prompt
  segment_count
  net_camera_bearing_deg?
  per_segment_camera_bearings_deg?
  poses?
  intrinsics?
  waypoints?             evaluator/Track B 可见性由 task 决定
  frame_count
  convention_version
```

约束：

- `A_text` official 只暴露 `net_continuous` 的净 camera bearing 和段数，不暴露逐帧 poses、intrinsics、hidden waypoints 或 full trajectory；
- `B_camera` 暴露 Track A 信息以及完整逐帧 camera condition；
- `C_multiview` 在 V1 是可选扩展，必须独立定义 condition views，不可假装成单图 Track；
- 不使用无语义字段名 `angle`；所有角都带 `camera` 或 `content` 前缀。

### 5.4 `OverlapProfile`

```text
OverlapProfile
  condition_to_frame: tuple[float, ...]
  adjacent: tuple[float, ...]
  endpoint: float
  mean_future: float
  minimum_future: float
  maximum_future: float
  level: L1 | L2 | L3 | L4 | None
  definition_version
```

`level` 只由 endpoint overlap 决定：

- L1 `[0.70,0.90)`；
- L2 `[0.40,0.60)`；
- L3 `[0.15,0.25)`；
- L4 `[0.00,0.10)`。

位于空档区间的样本不应被就近分配，`level=None` 并由相应 T3 manifest 拒绝。所有值在 resize 前、以交集面积除以 condition window 面积计算。

### 5.5 `DirectionTarget`

```text
DirectionTarget
  camera_bearing_deg
  content_bearing_deg
  segment_camera_bearings_deg
  segment_content_bearings_deg
  eight_way_bin
  net_displacement_ratio
  net_direction_well_posed
  convention_version
```

冻结 convention：轨迹点为 `(cy,cx)`、图像 y 向下，camera bearing 使用 `atan2(-dy, dx)`，东 0°、北 90°、逆时针增加；content bearing = camera bearing + 180°（mod 360）。八方向仅用于 coverage/breakdown，不用于量化连续角评分。`net_displacement_ratio < 0.2` 时净方向不进入均值。

## 6. World Model Interface

### 6.1 抽象职责

`WorldModel` 表达“给定 benchmark 合法输入，生成 canonical rollout”的能力，不表达训练、优化器、checkpoint 保存或评测。建议设计签名为：

```python
class WorldModel(Protocol):
    name: str
    capabilities: WorldModelCapabilities

    def generate(
        self,
        condition_image,
        trajectory_condition,
        metadata,
    ) -> Rollout:
        ...
```

实际实现应使用 keyword-only 参数，但对外概念保持用户指定的三个输入。`metadata` 必须是经过 task 裁剪的 `GenerationMetadata`，而不是原始 HDF5 attrs。

### 6.2 `GenerationMetadata`

允许字段：

- benchmark/version、task ID、track ID；
- sample ID；
- horizon、期望输出尺寸、dtype/layout；
- generation seed；
- prompt schema 与公开 scene metadata；
- adapter/runtime identity；
- condition checksum。

禁止字段：GT rollout、GT checksum locator、full scene、hidden trajectory、test source path、未来帧统计或任何能泄漏答案的信息。

### 6.3 `Rollout`

```text
Rollout
  frames: (T,H,W,3) uint8
  sample_id
  model_id / adapter_version
  generation_seed
  runtime_metadata
  warnings
```

V1 输出必须含 t=0 condition frame，总帧数严格等于 manifest 的 17/25/33，尺寸严格等于合同值，数值有限。adapter boundary 可以把模型原生 layout/dtype 转为 canonical 格式，但必须记录转换；task/evaluator 不再二次 resize、转置或截断。

### 6.4 `WorldModelCapabilities`

最低字段：

| 字段 | 说明 |
|---|---|
| `input_types` | single image、multi-view 等 |
| `supported_tracks` | A_text/B_camera/C_multiview |
| `prompt_schemas` | `net_continuous`、legacy net 等 |
| `trajectory_support` | none/net direction/per-segment/full per-frame |
| `camera_support` | none/pose/intrinsics/Plücker 等 |
| `supported_horizons` | 17/25/33 或动态 |
| `output_contract` | native layout/dtype 与 canonicalization 方法 |
| `stochastic` | 是否需要 seed/repeats |
| `adaptation_type` | frozen/LoRA/full fine-tune/bridge/none |
| `condition_view_count` | 1、固定多视图或可变 |

能力校验发生在任何模型加载之前。若 task/track 不在能力声明中，runner 返回 `INELIGIBLE_CAPABILITY`，不能把缺少的 camera control 静默转换成文本后仍列为 native。

### 6.5 与现有 `VideoBackbone` 的关系

两层职责必须保持不同：

| 层 | 负责 | 不负责 |
|---|---|---|
| `WorldModel` | benchmark task/track 输入、能力分区、canonical rollout、身份 | Wan tensor、权重加载细节 |
| `VideoBackbone` | 单首帧 + prompt + 可选 `PoseCond` 的视频推理合同 | task、manifest、GT、metric、leaderboard |

未来具体类 `LingBotWorldModelAdapter` 组合一个现有 `WanBackbone` 实例：

1. 校验 task/track capability；
2. 将 `TrajectoryCondition` 转为现有 prompt 和 `PoseCond`；
3. 调用 `WanBackbone.generate()`；
4. 使用现有 `check_output_contract()` 校验 canonical output；
5. 返回带身份的 `Rollout`。

它不得复制 `load_lora_model()`、`generate_video()`、VAE resize 或 LoRA 合并逻辑。

#### 6.5.1 Model identity 分层

Backbone Identity Audit 后，模型发布名、架构名、adapter 名和执行路线不得继续共用一个 `backbone`/`model` 字符串。冻结以下层次：

```text
WorldModel
  |
  v
LingBotWorldModelAdapter          # 未来具体 Python 类；本阶段不实现
  |  adapter_identity = LingBotWorldAdapter
  v
Wan I2V A14B backbone             # architecture_identity
  |  architecture alias = wan
  v
LingBot-World base-cam checkpoint # release_identity / checkpoint_identity

current official execution route = wan-inline
future benchmark route           = benchmark-adapter
```

- **LingBot** 是 checkpoint/model release 身份：`LingBot-World-base-cam`。
- **Wan** 是架构与实现身份：`Wan-I2V-A14B`、`WanModel`；历史名称 `wan` 继续作为 architecture alias。
- **`wan-inline`** 是现有 official ICLR pipeline 的 execution route，不是 checkpoint 发布名，也不因新增 benchmark schema 而删除或改名。
- **`LingBotWorldAdapter`** 是稳定的 `adapter_identity` 示例；未来 Python 类名为 `LingBotWorldModelAdapter`。前者是身份记录，后者是实现类标识，两者不得与 execution route 混用。
- 原设计名 `WanWorldModelAdapter` 仅是 Phase 1 的架构族占位名；Phase 2.5 后，当前 LingBot 具体 adapter 使用上述新名称。现有低层 `WanBackbone` 保持不变。

机器可读合同使用 `rswm.model_identity/v1`，至少分开记录：

| 层 | 字段 | 当前 LingBot 声明示例 |
|---|---|---|
| 模型族 | `model_family` | `Wan-I2V-A14B` |
| 发布 | `release_identity` | `LingBot-World-base-cam` |
| adapter | `adapter_identity` | `LingBotWorldAdapter` |
| checkpoint | `checkpoint_identity` | repository、revision、SHA-256 manifest、config hash、status |
| 架构 | `architecture_identity` | transformer、VAE、text encoder、camera conditioning、`wan` alias |
| 相机条件 | `camera_condition_identity` | `plucker_ray_v1`、pose injection、injector type |
| LoRA | `lora_identity` | target modules、rank、alpha；无 LoRA 时为 `null` |
| 执行 | `execution_identity` | `wan-inline` 或未来 `benchmark-adapter`；可保留 legacy aliases |

`BenchmarkReleaseIdentity` 与 `ArtifactLineage` 可选绑定整个 `ModelIdentity` 及其 checksum；`WorldModelCapabilities` 可选携带同一对象。缺少该扩展的旧 v1 文档继续有效，且省略字段时旧 checksum 序列化保持不变。Benchmark spec 仍然定义任务/数据/评测协议，保持 model-agnostic，不写死 LingBot。

### 6.6 Future model adapter

`future_video_model_adapter.py` 只定义模板/抽象基类或明确抛出 `NotImplementedError` 的 placeholder：

- 不 import VideoCrafter/Open-Sora/Cosmos/DiffusionSat；
- 不注册为 available/admitted baseline；
- 不提供虚构 checkpoint、license 或结果；
- 文档只说明实现者需完成 native input 到 `TrajectoryCondition` 的公开映射，并声明 adaptation。

## 7. Benchmark Task Layer

### 7.1 `BaseTask`

建议接口：

```python
class BaseTask(ABC):
    task_id: str
    protocol_version: str

    def prepare_input(self, sample, track, model_capabilities) -> ModelInput:
        ...

    def validate_output(self, rollout, sample, track) -> ValidationReport:
        ...

    def evaluate(self, rollout, ground_truth, evaluator) -> EvaluationResult:
        ...
```

职责边界：

- `prepare_input()` 只做可见字段白名单、能力检查和 domain object 组装；不加载模型、不生成 trajectory、不读 GT pixels；
- `validate_output()` 检查 ID、T/H/W/C、dtype、有限值、t=0/condition 合同和重复/失败状态；不计算论文指标；
- `evaluate()` 选择 frozen metric panel、cohort 和 evaluation view，实际数值委托 `EvaluationRunner`；
- task 不持有 Wan、checkpoint 或 optimizer；
- task 不以文件夹名称、环境变量或 prompt 文本猜测 track。

### 7.2 T1：`T1_spatial_rollout`

实现类建议命名 `T1SpatialRolloutTask`。它定义共同的 single-view rollout 输出合同：

- 输入：一张 condition image + 所属 track 允许的命令；
- 输出：含 condition frame 的连续 rollout；
- horizon：17/25/33；
- 评价：future-only visual、temporal、spatial/control、success/valid audit；
- task manifest 独立冻结，不能借用 T2/T3 混合平均值。

### 7.3 T2：`T2_arbitrary_direction`

`T2ArbitraryDirectionTask` 继承 T1 输出合同，但增加：

- 连续净 camera bearing coverage；
- 八方向 coverage/breakdown；
- 多段路径逐段方向 coverage；
- `net_ratio < 0.2` 的净方向不适定标记；
- Track A 只评价被命令的净方向与段数；Track B 才把逐段/turn/waypoint/path 作为 commanded 指标。

T2 不在 runtime 临时采样新方向；它消费独立冻结的 T2 manifest。这样可保证所有模型收到相同 sample IDs 与命令。

### 7.4 T3：`T3_overlap_ladder`

`T3OverlapLadderTask` 继承 T1 输出合同，并要求：

- manifest 中每个样本存在冻结 `OverlapProfile`；
- L1/L2/L3/L4 各自独立报告 N、指标、CI、success/valid rate；
- 只以 endpoint overlap 归档 level；
- 现有 ICLR `<10%` 数据只能映射到 L4 legacy-compatible cohort；
- 缺任何 ladder level 时报告 `NOT_AVAILABLE`，不能合并或用插值样本补齐。

### 7.5 Task registry

registry 以固定 task ID 查找 task factory，并校验：

- task ID 和 implementation version；
- benchmark spec 与 manifest checksum；
- track/capability；
- dataset domain；
- metric panel version。

禁止动态 `eval()`、隐式 fallback 到 T1 或将未知 task 当作 legacy。

## 8. Dataset Abstraction

### 8.1 `BenchmarkDataset`

建议协议：

```text
ids() -> ordered sequence[str]
public_sample(id) -> PublicSample
organizer_sample(id) -> OrganizerSample
ground_truth(id) -> GroundTruthBundle       # trusted evaluator only
dataset_identity -> checksums/provenance
close()
```

dataset adapter 不做训练归一化、不返回 torch tensor、不 resize、不改变帧数、不生成 prompt、不插值 trajectory。

### 8.2 `LegacyH5Adapter`

`LegacyH5Adapter` 只读接入：

- canonical HDF5；
- official split；
- `rsroll.benchmark_manifest/v2`；
- 可选 benchmark spec sidecar。

打开流程：

1. 使用现有 `validate_manifest_document()` 校验 schema 与 manifest checksum；
2. official 模式验证 HDF5、split 文件及其 SHA-256；
3. 以 manifest `sample_keys` 顺序为唯一顺序；
4. 逐样本核对 ID、frame count、resolution、condition/GT/trajectory checksum；
5. 使用 HDF5 只读模式，lazy 读取大数组；
6. 将 HDF5/manifest 字段映射到 domain object；
7. 关闭文件句柄，不写 attrs、不创建 cache、不回填字段。

### 8.3 字段映射

| Domain 字段 | manifest v2 | HDF5 | 规则 |
|---|---|---|---|
| `id` | `sample_id` | group key | 三者必须一致 |
| `image` | checksum/size | `image` | 返回 uint8，不做训练归一化 |
| `trajectory` | ID/source/checksum | `trajectory`, `poses`, `intrinsics`, `waypoints` | official 缺必需字段则 fail closed |
| `overlap` | frozen profiles/summary | attrs 仅作审计 | manifest 为正式评测真源 |
| `direction` | 由 trajectory checksum 锚定 | trajectory/waypoints | 用统一 convention 计算 |
| `horizon` | `frame_count` | `num_frames`/video T | 必须一致且为支持值 |
| `metadata` | source/category/difficulty | attrs/metadata attrs | 仅公开白名单进入模型 |
| GT | checksum/locator | `video_frames` | 仅 `GroundTruthBundle` 可见 |

manifest v2 允许 waypoint interpolation 的 diagnostic manifest 不能进入 official/headline。若 `trajectory` 缺失，adapter 不自行从 waypoints 插值后冒充 canonical trajectory。

### 8.4 Benchmark spec sidecar

为避免修改 manifest v2，新增独立 sidecar，概念字段为：

```text
schema: rswm.benchmark_spec/v1
benchmark_version
task_id
conditioning_track
data_domain
capability_partition
manifest_path
manifest_checksum
metric_panel_version
direction_convention_version
overlap_definition_version
sample_visibility_policy
headline_eligible
spec_checksum
```

sidecar 只引用 manifest checksum，不复制样本列表。任何改动都产生新 `spec_checksum`。旧 ICLR artifact 的 sidecar 必须标明 `legacy_compatible` 和实际覆盖范围，不能追溯性提升为 full V1。

## 9. Trajectory Abstraction

### 9.1 目标

集中表达 arbitrary angle、continuous trajectory、multi-step path 和 overlap-controlled generation 的语义，同时保留现有 v5 builder/camera utilities 作为实现真源。

### 9.2 核心类型

```text
TrajectorySpec
  frame_count
  num_segments
  requested_camera_bearings
  overlap_level or endpoint_interval
  window_size
  scene_size
  seed

TrajectoryPlan
  per_frame_centers (cy,cx)
  waypoints (cy,cx)
  segment_camera_bearings_deg
  segment_content_bearings_deg
  net_camera_bearing_deg
  net_content_bearing_deg
  net_displacement_ratio
  overlap_profile
  source_algorithm/version
```

### 9.3 `TrajectoryGenerator`

建议协议：

```python
class TrajectoryGenerator(Protocol):
    def generate(self, spec, rng) -> TrajectoryPlan:
        ...
```

Phase 2 初始实现只提供 `LegacyTrajectoryAdapter`：

- 调用现有 v5 `RandomPositionPathGenerator`/相关函数；
- 复用现有 camera pose、continuous direction 和 overlap 工具；
- 不复制随机采样、retry、边界 clamp 或 overlap 算法；
- 记录原始实现版本和参数；
- 对旧 HDF5 只做解析，不重新生成 trajectory；
- 仅在未来经批准的数据构建流程中调用 generation。

因此，“支持”表示接口能够表达并适配现有能力，不表示 Phase 1 会重建数据。

### 9.4 Convention service

`conventions.py` 是无随机、无 I/O 的纯函数层，负责：

- `(cy,cx)` delta 到 camera bearing；
- camera bearing 到 content bearing；
- 连续角归一化；
- 八方向半开区间分桶；
- net displacement ratio 与 well-posed 判定；
- endpoint overlap 到 L1-L4；
- convention/version 字符串。

它必须以现有已知答案测试和 frozen protocol 为准；任何改变都需要新 convention version，不能静默更新旧榜。

## 10. Evaluation Framework

### 10.1 `Metric` 接口

用户指定的统一形式保留：

```python
class Metric(Protocol):
    spec: MetricSpec

    def compute(self, prediction, ground_truth, metadata) -> MetricResult:
        ...
```

`MetricSpec` 至少声明：

- metric ID、显示名、version、方向（higher/lower）；
- category：visual/temporal/spatial/semantic；
- scope：sample 或 dataset；
- applicable tasks/tracks；
- input view：full rollout、future-only、per-frame、features；
- preprocessing 与 resolution；
- checkpoint ID/hash（如 LPIPS/I3D）；
- minimum N；
- invalid/missing policy；
-是否为 headline、diagnostic 或 audit。

`MetricResult` 包含 value、valid、reason、N、per-sample/aggregate scope、warnings、metric/checkpoint identity，不允许用 0 表示缺失。

### 10.2 Dataset-level metric

FID/FVD 需要样本集合。它们仍实现统一 `compute()` 概念，但由 runner 通过 accumulator 管理：

```text
start(context)
update(prediction, ground_truth, metadata)
finalize() -> MetricResult
```

这不是第二套 metric API，而是 `scope=dataset` 的生命周期。禁止先计算“每样本 FID/FVD”再取平均。

### 10.3 Future-only policy

`EvaluationRunner` 构造两个显式 view：

- `FullRolloutView = frames[0:T]`：只供 output contract、首帧连续性和特定 audit；
- `FutureOnlyView = frames[1:T]`：供视觉、时序和分布生成质量指标。

wrapper 必须声明旧函数是否已在内部 future-only：

- 若旧 `compute_all_metrics()` 负责裁剪，runner 传 full rollout 并记录 `future_only_owner=legacy_wrapper`；
- 若包装拆到更细函数，runner 传 future-only view 并记录 `future_only_owner=evaluation_runner`；
- 同一次评测只能出现一个 owner；测试用 t=0 污染帧证明结果不受其影响，并验证 t=1 未被误删。

### 10.4 分类目录与现有复用

| 新分类 | V1 初始 wrapper | 现有真源 | 状态 |
|---|---|---|---|
| `visual/` | SSIM、PSNR、LPIPS、FID | `lora_remotesense/new/metrics.py` | 包装，不改数值 |
| `temporal/` | 标准 I3D-FVD、R3D proxy、时间连续性 audit | metrics + I3D runner | I3D 与 proxy 必须分名 |
| `spatial/` | net direction、segment、turn、waypoint、Fréchet/DTW、endpoint/path consistency | `trajectory_metrics.py`、`run_trajectory_eval.py` | 必须保留 calibration gate |
| `semantic/` | 输出级 semantic preservation placeholder | 当前无冻结实现 | `NOT_AVAILABLE`，不进 headline |

SemanticGate 仍是数据构建质量门，不可被包装为输出级 semantic metric。

### 10.5 `EvaluationRunner`

runner 流程：

1. 校验 benchmark spec、manifest、submission/model identity；
2. 校验 required ID set 和每个 rollout contract；
3. 构造唯一 future-only view；
4. 按 task × track 选择 metric panel；
5. 运行 sample metrics 与 dataset accumulators；
6. 执行 GT oracle / instrument-floor calibration gate；
7. 生成 per-sample、aggregate、invalid reason、N、CI 与 breakdown；
8. 写带 checksum 的标准结果 bundle；
9. 不推断论文结论，不生成 RSWM Score。

## 11. Baseline Capability Interface

### 11.1 `BaselineCapabilities`

每个 baseline 必须声明：

| 必需字段 | 示例语义 |
|---|---|
| `input_type` | single_image / multiview / video_prefix |
| `trajectory_support` | none / net_direction / segment_direction / full_path |
| `camera_support` | none / extrinsics / intrinsics / Plücker |
| `adaptation_type` | frozen / prompt_bridge / recursive / LoRA / full_finetune |
| `checkpoint` | 旧 card v1 自由文本字段继续保留；v2 overlay 引用结构化 `ModelIdentity.checkpoint_identity` |
| `license` | code、weights、dataset 使用许可；未知即 UNKNOWN |
| `output_format` | native layout/dtype/frame policy 与 canonicalizer |
| `condition_views` | 数量与选择方式 |
| `tasks` / `tracks` | 实际 eligible 范围 |
| `external_data` | 训练/微调/检索数据披露 |
| `runtime` | framework/version/container identity |

### 11.2 capability partition

- **native**：模型原生接收该 track 要求的控制信息；例如原生 full camera path 才能进入 `B_camera/native`；
- **adapted**：通过 prompt、递归 outpainting、trajectory-to-text、bridge 或微调把 benchmark 输入转换为模型输入；必须披露适配器并进入 adapted partition；
- **multiview**：需要多于一张 condition view；只进入 `C_multiview`，不与单图方法直接排名。

`native/adapted/multiview` 不是“方法优劣标签”，而是公平比较条件。一个方法可因不同 adapter 产生多个独立 entry，但每个 entry 必须有独立身份和结果。

### 11.3 `BaselineCardV2` overlay

现有 card v1 保留不动。新 card v2/overlay 引用旧 card hash，并补充 capability、license、checkpoint hash、task/track、external data、output schema、adaptation code hash 与 result availability。状态机建议为：

其中模型相关字段必须引用 `rswm.model_identity/v1`，不得再把 `source`、`checkpoint`、`adaptation` 或 runtime route 拼成一个非结构化“model name”。本阶段不修改 `server_run/baselines/cards/*.json`，也不提前实现 M5 baseline capability overlay。

```text
DECLARED -> ADAPTER_AVAILABLE -> CHECKPOINT_VERIFIED
         -> RUNNABLE -> EVALUATED -> ADMITTED
```

任何缺失阶段不能跳级。外部 placeholder 最高只能是 `DECLARED`；没有 checkpoint、正式日志和标准结果时不得称为“已复现”。

### 11.4 旧 baseline 兼容

`LegacyBaselineAdapter` 包装 `server_run/baselines/registry.py`：

- 保留现有 diagnostic/external/result adapter；
- 将旧 card/record 映射到新能力对象；
- 无法确定的字段置 `UNKNOWN` 或 `NOT_AVAILABLE`；
- 旧四方向 prompt 不自动升级为 `net_continuous`；
- 旧 `text_plucker` 只有在 camera metadata 与 command visibility 验证后才能映射到 `B_camera`；
- 现有 result availability 必须从真实标准化产物确认。

## 12. Runner 与 artifact 合同

### 12.1 建议新入口

未来实现一个独立、默认 dry-run 的入口，例如：

```text
python -m benchmark.cli plan --spec <benchmark_spec>
python -m benchmark.cli validate-data --spec <benchmark_spec>
python -m benchmark.cli run --spec <benchmark_spec> --model <model_card>
python -m benchmark.cli evaluate --spec <benchmark_spec> --submission <dir>
```

Phase 2 初始只实现 CPU 可验证的 plan/validate-contract 路径。任何真实生成/evaluation 仍需单独授权和现有服务器门禁。

### 12.2 标准结果 identity

每个结果至少绑定：

```text
benchmark_version
task_id
track_id
data_domain
capability_partition
manifest_checksum
benchmark_spec_checksum
model_family / release_identity / architecture_identity
checkpoint_identity / adapter_identity / camera_condition_identity
lora_identity / execution_identity / model_identity_checksum
evaluator/metric/checkpoint identity
sample IDs and generation seeds
success/valid counts
raw metric values and CI
result checksum/signature
```

旧论文表导出器继续消费旧标准结果；新 schema 在验证 parity 前不替换旧 exporter。

## 13. 现有代码复用关系

| 新组件 | 复用对象 | 复用方式 |
|---|---|---|
| `LingBotWorldModelAdapter` | `lora_remotesense/new/backbone.py` | 未来组合 `WanBackbone`、复用 output validator；Phase 2.5 只冻结身份，不创建类 |
| `LegacyH5Adapter` | `server_run/baselines/protocol.py` | 调用 manifest v2 validator/checksum helpers |
| trajectory adapter | v5 builder、`camera_trajectory.py`、`continuous_direction.py`、`step_overlap.py` | 只调用，不复制生成算法 |
| visual/temporal wrapper | `lora_remotesense/new/metrics.py` | 包装函数与版本身份 |
| spatial wrapper | `trajectory_metrics.py`、`run_trajectory_eval.py` | 复用评分与聚合/标定逻辑 |
| baseline bridge | `server_run/baselines/*` | 旧 card/adapter/result 到新能力的保守映射 |
| result export | `tools/export_paper_tables.py` | parity 期继续使用，缺失值政策不变 |
| submission validation | `submission_contract.py` | 复用 required IDs、shape、failure policy |

## 14. 文件变更计划

### 14.1 Phase 1 设计当时实际新增（历史记录）

Phase 1 设计当时只新增：

- `RSWM_PHASE1_ARCHITECTURE_REVIEW.md`；
- `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md`。

当时没有修改任何 `.py`、`.sh`、`.yaml`、`.json`、HDF5、checkpoint 或结果文件。Phase 2 M1-M3 与 Phase 2.5 的后续授权变更分别由实施报告和本文第 21 节记录，不改写这条历史事实。

### 14.2 经确认后 Phase 2 建议新增

新增前述 `models/`、`benchmark/`、`evaluation/`、`baselines/` 与 `tests/benchmark/` 文件。实现优先顺序为 core types/contract、dataset adapter、mock world model、task、metric wrapper、baseline capability，最后才是 Wan adapter 的真实接线。

### 14.3 Phase 2 初始不需要修改的现有文件

采用独立 runner + compatibility adapter 后，第一版实现可以不修改现有生产文件：

- `lora_remotesense/new/backbone.py`；
- `lora_remotesense/new/evaluate.py`；
- `lora_remotesense/new/inference.py`；
- `server_run/baselines/*`；
- `server_run/run_iclr_full_pipeline.sh`；
- `tools/export_paper_tables.py`。

这使新层可以通过 CPU fixture 与 mock 验证，而不影响旧 ICLR 路径。

### 14.4 parity 通过后的可选最小修改

只有经过单独确认，才考虑：

- 在 `lora_remotesense/new/evaluate.py` 增加显式 opt-in 新 runner 路由，默认 `wan-inline` 不变；
- 在 `server_run/baselines/cards.py` 增加读取 card v2 overlay 的兼容入口，card v1 保持有效；
- 在 `tools/export_paper_tables.py` 增加新 result schema reader，旧 schema 输出保持逐值一致；
- 在自动化脚本中增加只读 plan/contract gate，不改变训练/评测步骤。

这些都不是本次或 Phase 2 第一小步的必要修改。

## 15. 明确不修改文件列表

以下文件/目录在本设计下保持冻结：

- `wan/**`；
- `lora_remotesense/new/wan/**`；
- `lora_remotesense/new/train.py`；
- `lora_training/**`；
- `configs/iclr_main/**`；
- `lora_remotesense/build_dataset_from_benchmark.py`；
- `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py`；
- `server_run/build_correct.sh`；
- `server_run/check_dataset.py`；
- `server_run/make_official_splits.py`；
- 现有 HDF5、official split、manifest v2 与 checksum；
- 现有 checkpoints、evaluation outputs、paper tables 和 `work/` 历史证据；
- `RSWM_BENCH_PROTOCOL_V1.md` 的科学定义，除非人工批准单独修正文档状态字段。

## 16. 迁移步骤与门禁

### M0：确认设计

- 审批本文第 19 节决策；
- 形成 Phase 0 protocol 冻结的可审计批准记录；
- 不运行任何实验。

### M1：core types 与 schema

- 实现无 torch/h5py 依赖的 domain objects、errors、capability 和 spec validator；
- 用纯 CPU 单元测试冻结方向、overlap、visibility 与 output contract；
- gate：无现有文件 diff。

### M2：LegacyH5Adapter

- 以最小 synthetic fixture 只读映射 HDF5 + manifest v2；
- 校验 checksum、顺序、GT 隔离与缺字段 fail-closed；
- 不读取正式大 HDF5，不重建数据。

### M3：Task layer

- 实现 T1/T2/T3 的输入投影、cohort 校验和 metric panel 选择；
- 用 spy model 证明 Track A 看不到 pose/GT，Track B 只收到冻结 camera command。

### M4：Metric wrappers

- 包装旧 metric 真源；
- 已知答案与旧 fixture parity；
- 验证 future-only 只执行一次、null/invalid/min-N 政策不变。

### M5：Baseline capability

- conservative 映射现有 card v1/registry；
- 未知 license/checkpoint/result 不得自动通过；
- 验证 native/adapted/multiview 分区。

### M6：Wan mock adapter

- 只用 mock 检查 prompt、pose、seed、size、T 与 output 转发；
- 对照现有 `test_backbone_protocol.py`；
- 不加载真实权重。

### M7：CPU 端到端 dry-run

- tiny fixture + deterministic mock world model；
- data -> task -> model -> output validation -> metric wrapper -> result bundle；
- 与旧 fixture 结果逐字段比对。

### M8：受控真实接线（需另行授权）

- 先 plan/preflight；
- 再单样本真实 Wan parity；
- 只有输出与旧 `wan-inline` 在同 seed/输入下满足预注册等价条件，才扩大到 smoke；
- 正式 pipeline 默认入口切换需再次审批。

任何阶段失败都回退到旧入口；不删除旧代码、不覆盖旧 artifact。

## 17. 测试计划

### 17.1 Contract 单元测试

- `WorldModel.generate()` 输入/输出 shape、dtype、T、seed 与 ID；
- unsupported task/track/capability 在模型加载前失败；
- duplicate/extra/missing sample IDs 失败；
- canonical output 不接受单图、TCHW、float 或错误尺寸，除非明确由 model adapter 在边界转换并记录。

### 17.2 Track 可见性测试

- Track A spy adapter 断言收不到 poses、intrinsics、waypoints、full trajectory、GT 或 source locator；
- Track B 收到逐帧 pose/intrinsics，长度与 T 一致；
- C_multiview 不得用一张图伪装；
- legacy `net` 不映射为 official `net_continuous`。

### 17.3 Dataset 测试

- manifest checksum、H5 checksum、sample checksum、顺序与 split 一致；
- HDF5 以只读方式打开，测试前后 hash 不变；
- 缺 trajectory/pose window/resolution 的 official fixture fail closed；
- diagnostic interpolation 必须显式且 `headline_eligible=false`；
- organizer/public/model 三种视图没有字段泄漏。

### 17.4 Trajectory 已知答案

- 东/北/西/南与四个对角方向；
- camera/content 恰差 180°；
- 0/360° wrap 与 ±22.5° 半开 bin 边界；
- `net_ratio=0.2` 边界；
- L1-L4 上下界为左闭右开；
- endpoint/mean/adjacent overlap 不能互换。

### 17.5 Future-only 与 metric 测试

- 只污染 t=0 时 headline 生成质量指标不变；
- 只污染 t=1 时指标必须变化，以防双重裁剪；
- FID/FVD 走 dataset accumulator，不产生 per-sample 假分数；
- 小于 minimum N 返回 ineligible/null；
- I3D-FVD 与 R3D proxy 名称、checkpoint 和列完全分离；
- GT oracle/instrument-floor gate 失败时只判 estimator，不反向判模型为 0。

### 17.6 Legacy parity

- 同一 tiny fixture 下，新 wrapper 与现有 metric 输出逐值一致；
- Wan adapter mock 与现有 `WanBackbone` 参数转发一致；
- 旧 manifest/card/result 经过 adapter 后 hash/原字段不变；
- 新 exporter 接入前，旧 `tools/export_paper_tables.py` 结果逐字节或逐单元格一致。

### 17.7 后续真实运行门

真实 Wan/GPU、正式 HDF5、baseline 和 evaluation 不属于 Phase 1。若未来执行，必须记录服务器状态、权重/checkpoint hash、manifest/spec hash、命令、日志和真实结果；不能把 mock/CPU test 写成正式实验通过。

## 18. 验收标准

Phase 2 abstraction 实现只有同时满足以下条件才可称为完成：

1. 所有新 core/test 可在 CPU 最小环境运行；
2. 现有生产代码与配置 hash 未改变，或每个获批改动均有独立审查；
3. 旧 HDF5/manifest 无写入、无重建；
4. Track A visibility 泄漏测试通过；
5. direction/overlap/future-only 已知答案通过；
6. metric wrapper 与旧 fixture parity 通过；
7. baseline capability 不把 placeholder 标为 admitted；
8. 新 runner 默认 plan/dry-run，不自动启动 GPU；
9. 未生成、排序或宣传 RSWM Score；
10. 文档清楚区分“已观察、合理推断、尚未验证”。

## 19. 已确认的 Phase 1 设计决策表

| ID | 决策 | 推荐冻结项 | 不确认的影响 | 状态 |
|---|---|---|---|---|
| `P1-01` | 两层模型抽象 | `WorldModel` 是 benchmark 门面；现有 `VideoBackbone` 是推理桥 | 会产生重复协议与输出漂移 | 已确认（2026-08-23） |
| `P1-02` | LingBot/Wan 接入方式 | 未来 `LingBotWorldModelAdapter` 组合 `WanBackbone`，不复制 inference；发布身份与架构身份分离 | 可能改变 checkpoint/采样解释 | 已确认；Phase 2.5 名称对齐（2026-08-23） |
| `P1-03` | manifest 兼容 | manifest v2 原样保留；task/track/domain 用 checksum sidecar | 原地扩 schema 会使旧 checksum/结果失效 | 已确认（2026-08-23） |
| `P1-04` | 样本安全边界 | `OrganizerSample`、`PublicSample`、`ModelInput`、`GroundTruthBundle` 分离 | Track A/外部 submission 可能泄漏答案 | 已确认（2026-08-23） |
| `P1-05` | 模型输出 | canonical rollout 含 t=0，严格 `(T,H,W,3) uint8` | 与现有 17/25/33 和 future-only 合同分叉 | 已确认（2026-08-23） |
| `P1-06` | future-only owner | 每次 evaluation 仅一个裁剪 owner，并写入结果 identity | 双重裁剪或把 condition 纳入评分 | 已确认（2026-08-23） |
| `P1-07` | dataset metric | FID/FVD 使用 dataset accumulator/finalize | 可能出现无意义的 per-sample FID/FVD | 已确认（2026-08-23） |
| `P1-08` | trajectory adapter | 只适配现有 v5/camera/overlap 实现，不复制生成算法 | trajectory 协议可能与旧数据漂移 | 已确认（2026-08-23） |
| `P1-09` | T2/T3 数据消费 | task 只消费独立 frozen manifest，不在评测时随机造样本 | 模型之间样本与难度不一致 | 已确认（2026-08-23） |
| `P1-10` | semantic metric | 当前保留 `NOT_AVAILABLE` 扩展位，不作为 V1 headline | 未校准指标可能被误作论文证据 | 已确认（2026-08-23） |
| `P1-11` | baseline 分区 | native/adapted/multiview 严格分区；UNKNOWN 不自动准入 | 不同输入资源会被不公平混排 | 已确认（2026-08-23） |
| `P1-12` | legacy evidence | 旧 ICLR `<10%` 仅映射 T3/L4 legacy-compatible，不追溯升级 | 可能夸大完整 benchmark 覆盖 | 已确认（2026-08-23） |
| `P1-13` | 初始集成策略 | 独立 opt-in runner；旧 full pipeline 和 `wan-inline` 默认不变 | 未验证 abstraction 会影响已有实验 | 已确认（2026-08-23） |
| `P1-14` | Phase 2 首批范围 | 只实现 core、fixture、adapter/mock 和 CPU dry-run | 过早进入 GPU/正式数据扩大风险与成本 | 已确认（2026-08-23） |
| `P1-15` | Phase 0 状态记录 | 将协议文档中的“待确认”表与本次“已冻结”指令形成一致批准记录 | frozen 状态缺少单一可审计真源 | 已确认（2026-08-23） |
| `P1-16` | Benchmark Release Identity | 统一绑定 protocol、benchmark spec、schema bundle、evaluator 与 manifest identity；任一组件未验证则 release 不得标记 ready | 结果可能引用彼此不一致的协议、schema 或 evaluator | 已确认（2026-08-23） |
| `P1-17` | Artifact Lineage | 记录 `checkpoint → adapter → benchmark spec → manifest → metric → result` 的有序、带 checksum、可审计链路 | 无法追溯结果由何种模型与评测身份产生 | 已确认（2026-08-23） |

## 20. 批准记录与当前停止点

2026-08-23，项目负责人确认 P1-01 至 P1-15，并新增、确认 P1-16 与 P1-17；同时授权进入 Phase 2 的 M1-M3。

当前授权不包含 M4 及以后工作。继续禁止修改模型、LoRA、ICLR pipeline 与既有 HDF5，禁止 GPU 训练、正式 evaluation 和外部 baseline 运行。进入下一里程碑前必须再次取得明确确认。

## 21. Phase 2.5 Model Identity Schema Alignment 记录

Phase 2.5 只新增/修改 identity schema、identity contract、CPU fixture tests 与本文档：

- 新增 `rswm.model_identity/v1`；
- `BenchmarkReleaseIdentity`、`ArtifactLineage` 和 `WorldModelCapabilities` 以可选字段引用分层模型身份；
- 旧 v1 release/lineage 文档在不提供 `model_identity` 时保持原字段与原 checksum 计算；
- manifest v2、benchmark spec 的科学字段、模型/推理/backbone/LoRA/训练配置和 ICLR pipeline 均未修改；
- 未创建 `LingBotWorldModelAdapter` 代码，未进入 M4 evaluator，未运行 GPU 或加载真实 checkpoint。
