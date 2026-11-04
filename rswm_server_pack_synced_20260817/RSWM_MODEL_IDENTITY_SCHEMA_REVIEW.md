# RSWM-Bench Phase 2.5 Model Identity Schema Review

> 日期：2026-08-23（Asia/Shanghai）  
> 阶段：Phase 2.5 — Model Identity Schema Alignment  
> 状态：identity schema/contract 已对齐；未进入 M4 evaluator  
> 审查对象：当前顶层 canonical 同步包  
> 变更边界：仅文档、schema、identity contract 与对应 CPU contract fixture  
> 明确未触碰：模型、`inference.py`、`backbone.py`、LoRA、训练配置、ICLR pipeline、HDF5、manifest v2、真实 checkpoint 与 GPU

## 1. 结论

Phase 2.5 已将原先可能混用的“模型名”拆成可校验的分层 `ModelIdentity`：

```text
model_family
release_identity
adapter_identity
checkpoint_identity
architecture_identity
camera_condition_identity
lora_identity
execution_identity
```

当前 RSWM 的身份解释冻结为：

| 层 | 当前声明 |
|---|---|
| checkpoint/release | `LingBot-World-base-cam` |
| model family | `Wan-I2V-A14B` |
| architecture | Wan I2V A14B camera-conditioned / `WanModel` |
| future adapter identity | `LingBotWorldAdapter` |
| future concrete class | `LingBotWorldModelAdapter`（本阶段未实现） |
| current official execution route | `wan-inline` |
| legacy architecture/execution alias | `wan`、`wan-inline` 均保留 |

`LingBot` 表示模型/checkpoint 的发布身份；`Wan` 表示架构实现身份；`wan-inline` 表示执行路线。三者不再由同一个 `backbone` 或 `model` 字符串承担。

## 2. 证据边界

### 已检查

- `models/world_model.py`；
- `benchmark/core/identity.py`；
- `benchmark/spec/schema.py` 与三个既有 JSON schema；
- `benchmark/spec/schemas/benchmark_spec_v1.schema.json`；
- `benchmark/core/types.py` 的 `Rollout` 旧身份字段；
- `server_run/baselines/cards.py` 及 representative card v1；
- `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md`；
- `RSWM_BACKBONE_IDENTITY_REPORT.md`；
- Phase 2 M1-M3 CPU fixture tests。

### 未检查或不可验证

- 本地不存在真实 LingBot checkpoint；revision、checkpoint SHA-256 manifest 与 config hash 仍未实测；
- 未连接服务器，不能确认服务器当前目录内容；
- 未实现或加载 `LingBotWorldModelAdapter`；
- 未执行 M4 evaluator、GPU 推理、训练或正式 evaluation；
- baseline capability v2 overlay 尚未进入 M5 实现。

因此，本文的“已验证”只指 schema/CPU contract 行为，不指真实模型或实验结果。

## 3. Step 1：现有 Identity Schema 审查

### 3.1 字段混用矩阵

| 位置 | 审查前状态 | 混用风险 | 判定 |
|---|---|---|---|
| `WorldModel.name` | 单一字符串 | 可能同时被理解为产品名、架构名或运行入口 | **已观察，P2** |
| `WorldModelCapabilities` | 有 `adaptation_type`、`camera_support`、`checkpoint_required`，无结构化模型身份 | 能力可判断，无法追踪具体 release/adapter/checkpoint | **已观察，P1** |
| `Rollout.model_id` / `adapter_version` | 两个自由字符串 | 无法绑定 checkpoint revision、架构、camera/LoRA 与 execution route | **已观察，P2**；旧字段保留 |
| `BenchmarkReleaseIdentity` | 绑定 protocol/spec/schema/evaluator/manifest | 是 benchmark release 身份，不含被测模型身份 | **已观察**；模型扩展应可选 |
| `ArtifactLineage` | 有 `checkpoint`、`adapter` artifact kind，但 `metadata` 无结构 | 可记录文件，却不能证明 release、architecture、adapter 与 route 的分层一致性 | **已观察，P1** |
| `BenchmarkSpec` | task/track/domain/capability/manifest/evaluator | 没有具体模型字段 | **正确设计**；保持 model-agnostic |
| baseline card v1 | `source`、`checkpoint`、`adaptation` 为自由文本 | 如 “Wan/LingBot” 一类字符串会把 release 与 architecture 混在一起；无 revision/hash contract | **已观察，P2** |
| `PLAN_BACKBONE=wan-inline` | execution route 字符串 | 若写入 `model_id` 会被误解为模型发布名 | **已观察**；必须归入 execution identity |

### 3.2 优先级发现

#### P1 — P1-17 lineage 缺少结构化模型身份

- **证据：** 审查前 `ArtifactLineage` 只规定 `checkpoint -> adapter -> benchmark_spec -> manifest -> metric -> result` 顺序；checkpoint/adapter 的语义只靠 `artifact_id` 与通用 metadata。
- **影响：** 两条 lineage 即使使用不同 release、架构或 execution route，也可能缺少机器可判的差异字段。
- **最小修正：** lineage 根部可选绑定 checksum-protected `ModelIdentity`；已完成。
- **信心：** 高。

#### P1 — Capability 声明不能识别具体模型 release

- **证据：** 审查前 `WorldModelCapabilities` 没有 checkpoint/release/architecture/adapter/execution 字段。
- **影响：** 将来 M4/M5 runner 只能知道模型“支持什么”，不能可靠知道“哪个模型以哪条路线执行”。
- **最小修正：** 增加可选 `model_identity`，并提供只读的 release/architecture/adapter/execution accessor；已完成。
- **信心：** 高。

#### P2 — 旧 baseline card 身份为自由文本

- **证据：** `server_run/baselines/cards.py` 的 card v1 必需字段包含 `source/checkpoint/adaptation`，但不要求 revision、hash 或架构分层。
- **影响：** 外部 baseline 的可用性、权重与适配方式可能被一行文字混合表达。
- **最小修正：** 未来 M5 card v2 overlay 引用 `rswm.model_identity/v1`；本阶段只冻结接口，不修改 legacy card 或实现 M5。
- **信心：** 高。

#### P3 — Adapter identity 与 Python 类名存在两个用户给定名称

- **证据：** Phase 2.5 输入同时给出 `LingBotWorldAdapter` 和层级图中的 `LingBotWorldModelAdapter`。
- **对齐：** `adapter_identity = LingBotWorldAdapter`；未来具体 Python 类 = `LingBotWorldModelAdapter`。
- **影响：** 身份 ID 可稳定于代码重构之外，同时类名明确它实现 `WorldModel` 协议。
- **信心：** 高；属于术语对齐，不是模型改动。

## 4. Step 2：分层 Model Identity 设计

### 4.1 顶层合同

新增逻辑 schema：`rswm.model_identity/v1`。JSON Schema 资源使用绝对 URN `urn:rswm:model-identity:v1`，避免跨 schema `$ref` 的相对路径歧义。

| 字段 | 类型 | 含义 |
|---|---|---|
| `model_family` | non-empty string | 模型族，例如 `Wan-I2V-A14B` |
| `release_identity` | non-empty string | 发布/checkpoint 产品身份，例如 `LingBot-World-base-cam` |
| `adapter_identity` | non-empty string | benchmark adapter 的稳定身份，例如 `LingBotWorldAdapter` |
| `checkpoint_identity` | structured object | repository、revision、SHA-256 manifest、config hash、status |
| `architecture_identity` | structured object | transformer、VAE、text encoder、camera conditioning、implementation aliases |
| `camera_condition_identity` | structured object | encoding、pose injection、injector type、version |
| `lora_identity` | object or null | target modules、rank、alpha；base/frozen 模型可为 null |
| `execution_identity` | structured object | route、entrypoint、legacy aliases |
| `status` | enum | `declared` / `verified` / `not_available` |
| `identity_checksum` | SHA-256 | canonical JSON 的身份校验和 |

### 4.2 Checkpoint verification policy

`checkpoint_identity.status=verified` 时，以下字段全部必需且必须合法：

- `revision`；
- `sha256_manifest`（64 位小写 SHA-256）；
- `config_hash`（64 位小写 SHA-256）。

当前本地没有 checkpoint，因此只能构造 `declared` 身份，不能把 revision/hash 留空后标为 `verified`。

### 4.3 Architecture 与 execution 分离

- `architecture_identity.implementation_aliases` 可以保存 `wan`；
- `execution_identity.route` 可以保存 `wan-inline` 或未来的 `benchmark-adapter`；
- schema 不以 enum 限制未来 route，但所有值必须非空；
- `wan`、`wan-inline` 没有删除，也没有被改写成 LingBot release 名。

### 4.4 Camera 与 LoRA 身份

Camera identity 单独记录表示与注入方式，不把“使用相机”压缩为布尔值。正式 LingBot 声明可以记录：

```text
encoding       = plucker_ray_v1
pose_injection = global projection + per-transformer-block modulation
injector_type  = camera injector + scale/shift
```

LoRA identity 只描述适配配置：target modules、rank、alpha。实际 adapter 权重文件仍由 P1-17 的 `adapter` artifact SHA-256 绑定；schema 不加载、修改或推断 LoRA 权重。

## 5. Step 3：Schema/Contract 修改

### 5.1 `BenchmarkReleaseIdentity`

新增可选 `model_identity`：

- 缺省时，旧字段、旧序列化文档与旧 checksum 计算路径不变；
- 提供时，整个 `ModelIdentity` 文档及 checksum 进入 release checksum；
- 新调用方可使用 `require_model_identity=True` fail closed；
- 提供但未 verified 的 model identity 不得使 result-bound release 标为 release-ready；
- 原 positional constructor 字段顺序保持不变，新字段追加在末尾。

### 5.2 `ArtifactLineage`

新增可选 `model_identity`：

- 缺省时保持旧 artifact/edge/checksum 行为；
- 提供时进入 lineage checksum；
- `require_model_identity=True` 可用于未来正式结果门；
- release 与 lineage 均绑定身份时，`assert_matches_release()` 会比较 model identity checksum；
- 原 `checkpoint -> adapter -> spec -> manifest -> metric(s) -> result` 顺序不变。

### 5.3 `WorldModelCapabilities`

新增可选 `model_identity`，以及只读 accessor：

- `release_identity`；
- `architecture_identity`；
- `adapter_identity`；
- `execution_identity`。

旧模型/fixture 不提供该字段时返回 `None`，原 capability 判断逻辑不变。

### 5.4 Benchmark spec

`benchmark_spec_v1.schema.json` **未修改**。Task、track、domain、capability partition、manifest、protocol 和 evaluator 属于 benchmark 定义，不应绑定具体 LingBot 模型。Model identity 只由 result-bound release/lineage/capabilities 引用。

### 5.5 Baseline capability

现阶段没有独立实现的 `BaselineCapabilities`/card v2 schema；只有 legacy card v1 与 Phase 1 设计。Phase 2.5 没有越权进入 M5。未来 overlay 必须引用同一 `ModelIdentity`，未知值保持 `declared/not_available`，不能自动 admission。

## 6. 修改文件列表

| 文件 | 类型 | 修改内容 |
|---|---|---|
| `benchmark/core/identity.py` | identity contract | 新增分层 identity dataclasses；release/lineage 可选绑定 |
| `benchmark/core/__init__.py` | contract export | 导出新增 identity types |
| `models/world_model.py` | identity/capability contract | `WorldModelCapabilities` 可选携带 identity，并分层访问 |
| `benchmark/spec/schema.py` | schema bundle contract | 将 model identity schema 纳入 checksum-protected bundle |
| `benchmark/spec/schemas/model_identity_v1.schema.json` | 新 schema | 分层 Model Identity JSON Schema |
| `benchmark/spec/schemas/release_identity_v1.schema.json` | 兼容扩展 | 可选引用 Model Identity |
| `benchmark/spec/schemas/artifact_lineage_v1.schema.json` | 兼容扩展 | 可选引用 Model Identity |
| `tests/benchmark/test_identity_and_schema.py` | CPU contract fixture | round-trip、checksum、旧 positional API、schema binding/optional reference 与 identity binding 回归断言 |
| `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md` | 文档 | LingBot release / Wan architecture / adapter / route 分层与新层级图 |
| `RSWM_MODEL_IDENTITY_SCHEMA_REVIEW.md` | 新文档 | 本审查与变更说明 |

以下相关文件保持不变：

- `benchmark/spec/schemas/benchmark_spec_v1.schema.json`；
- `benchmark/datasets/manifest_v2.py`；
- `server_run/baselines/cards.py` 与现有 card v1；
- 所有模型、推理、训练、LoRA、ICLR pipeline 与配置文件。

## 7. 兼容性说明

### 7.1 文档兼容

Release identity 与 artifact lineage 的 `model_identity` 都是**可选扩展**，没有加入旧 JSON schema 的 required 列表。旧 v1 文档可以原样读取。

### 7.2 Checksum 兼容

Python 序列化只在 `model_identity is not None` 时输出新字段。因此旧对象：

- 不产生空的 `model_identity: null`；
- canonical JSON 字段集合不变；
- release/lineage checksum 计算输入不变。

### 7.3 Python API 兼容

- 新 dataclass 字段追加在已有默认字段之后，旧 positional constructor 的参数位置不变；
- `ArtifactLineage.create(artifacts)` 原调用继续有效；
- `BenchmarkSpec.release_identity()` 继续生成 model-agnostic 旧形态；
- `WorldModelCapabilities(...)` 的既有构造调用无需增加参数。

### 7.4 名称兼容

- `wan` 作为 architecture/legacy alias 保留；
- `wan-inline` 作为 current official execution route 保留；
- 没有重命名或删除现有 `WanBackbone`；
- 没有创建 `LingBotWorldModelAdapter` 类。

## 8. 为什么不会影响已有 ICLR 结果

1. 旧 ICLR pipeline 没有 import `benchmark.core.identity` 或 `models.world_model`；本次静态搜索在 `lora_remotesense/`、`lora_training/`、`wan/`、`server_run/`、`configs/`、`tools/` 和 `paper/` 中匹配数为 0。
2. `evaluate.py`、`inference.py`、`backbone.py`、`train.py` 未修改。
3. Primary seed 42/43/44 配置未修改。
4. `run_iclr_full_pipeline.sh`、manifest v2 adapter、HDF5 与 official split 未修改。
5. 当前 official route 仍为 `wan-inline`；新增 schema 没有被接入旧 runner。
6. 没有加载模型或 checkpoint，没有生成 rollout、metric、table 或新实验结果。

所以本阶段只改变未来 benchmark 结果“如何声明并校验模型身份”，不改变任何已有 ICLR 数值的生成路径或数学定义。

## 9. 验证记录

| 检查 | 结果 | 证明范围 |
|---|---|---|
| 改动 Python 文件 `py_compile` | PASS | identity/schema/capability 合同无语法错误 |
| `python -m unittest tests.benchmark.test_identity_and_schema` | PASS，14 tests | 分层 round-trip、旧文档、旧 positional API、release/lineage binding |
| `python -m unittest discover -s tests/benchmark -p 'test_*.py'` | PASS，30 tests | M1-M3 全部 CPU fixture 未回归 |
| 四个 JSON 文件解析 | PASS，4/4 | JSON 语法有效 |
| Draft 2020-12 `check_schema` | PASS，4/4 | JSON Schema 元结构有效 |
| release/lineage 跨 schema `$ref` 验证 | PASS | 绝对 URN 可解析，带 ModelIdentity 的文档通过标准 validator |
| 冻结文件 SHA-256 前后对比 | PASS（见最终复核） | 禁止修改的核心执行文件内容未变 |

这些是本地、CPU、无数据检查；不证明真实 checkpoint、GPU 或正式 evaluator 可运行。

## 10. 尚未验证与下一步门禁

- `checkpoint_identity` 目前只能是 `declared`，待服务器提供固定 revision、checkpoint SHA-256 manifest 与 config hash 后才能升级为 `verified`；
- `LingBotWorldModelAdapter` 未实现，不能声称 benchmark adapter 已可用；
- baseline card v2 overlay 尚未实现；
- M4 evaluator identity、metric wrapper 与 result schema 尚未实现或冻结；
- 正式 release 必须同时绑定 verified evaluator、manifest、model identity 与完整 artifact lineage。

Phase 2.5 到此停止。**未进入 M4 evaluator；等待下一步明确授权。**
