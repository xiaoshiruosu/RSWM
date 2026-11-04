# RSWM-Bench Phase 2 M5 Baseline Capability Framework 实施报告

日期：2026-08-23  
阶段：Phase 2 M5（Baseline Capability Framework）  
状态：**M5 CPU abstraction 已完成；按授权边界停止，未运行任何真实 baseline、GPU 或正式 evaluation。**

## 1. 范围与结论

本阶段在不修改旧 baseline card v1 的前提下，新增了独立、CPU-safe、默认拒绝推断的 baseline 接入层，覆盖：

- 分离 model、adapter、checkpoint、execution 的结构化 `BaselineIdentity`；
- `BaselineCapabilities` 及 license、external data、runtime 披露合同；
- 引用 `ModelIdentity` 和旧 card 哈希的 `BaselineCardV2` overlay；
- 不可跳级的 admission 状态机；
- native、adapted、multiview 三类公平比较分区；
- 对 `server_run/baselines` 的只读 `LegacyBaselineAdapter`；
- identity、能力校验、状态转换、分区和旧卡兼容 CPU 回归测试。

本阶段没有下载或加载外部模型，没有运行 VideoCrafter、Open-Sora、Cosmos、DiffusionSat 或其他真实 baseline，没有调用 GPU，也没有修改 LingBot/Wan、ICLR pipeline、旧 baseline result 或 benchmark evaluator 数学。

## 2. 修改文件

### 2.1 Baseline framework

| 文件 | 作用 |
|---|---|
| `baselines/__init__.py` | 导出 M5 公共 API；导入路径保持 CPU-safe |
| `baselines/errors.py` | identity、capability、admission 和 legacy 映射的显式异常 |
| `baselines/identity.py` | 结构化 `BaselineIdentity` 与 canonical checksum |
| `baselines/capability.py` | 能力字段、披露对象、分区校验和 task/track 准入规则 |
| `baselines/card_v2.py` | 不可变 v2 overlay、旧卡哈希引用和顺序状态机 |
| `baselines/legacy.py` | 对 `server_run/baselines` 的保守、只读、惰性兼容层 |

### 2.2 CPU tests

| 文件 | 覆盖内容 |
|---|---|
| `tests/benchmark/baseline_fixture_factory.py` | 仅供测试使用的结构化 identity、能力和状态证据 fixture |
| `tests/benchmark/test_baseline_identity.py` | 身份分层、round-trip、checkpoint/adapter 一致性 |
| `tests/benchmark/test_baseline_capability.py` | capability 校验、B_camera 门槛、三类分区、禁止 fallback |
| `tests/benchmark/test_baseline_card_v2.py` | 状态不可跳级、证据顺序、admission 前置条件、checksum |
| `tests/benchmark/test_baseline_legacy_adapter.py` | 16 份 v1 卡兼容、UNKNOWN 映射、旧卡哈希引用 |

## 3. M5-1：Baseline Identity

新增 schema ID `rswm.baseline_identity/v1`。`BaselineIdentity` 不再用一个自由字符串表示“模型”，而是绑定以下四层身份：

| 层 | 类型 | 约束 |
|---|---|---|
| model identity | `ModelIdentity` | 保留 model family、release、architecture、camera condition 等分层信息 |
| adapter identity | `ComponentIdentity(kind="baseline_adapter")` | artifact ID 必须与 `ModelIdentity.adapter_identity` 一致 |
| checkpoint identity | `CheckpointIdentity` | 必须与 `ModelIdentity.checkpoint_identity` 完整一致 |
| execution identity | `ExecutionIdentity` | 必须与 `ModelIdentity.execution_identity` 完整一致 |

完整 identity 使用 canonical SHA-256 绑定。序列化/反序列化会重新检查 schema、嵌套类型、层间一致性与 checksum；不能用 `model_name`、checkpoint 文本或 runtime route 替代上述结构。

## 4. M5-2：BaselineCapabilities

`BaselineCapabilities` 实现了本阶段要求的字段：

- `input_type`；
- `trajectory_support`；
- `camera_support`；
- `adaptation_type`；
- `checkpoint_identity`；
- `license`；
- `tasks`；
- `tracks`；
- `external_data`；
- `runtime`。

同时保留 Phase 1 公平性合同所需的 `condition_views`、`uses_future_gt` 与 `output_format`。license 被拆分为 code/weights/data，runtime 记录 framework/version/container/hardware/precision；缺失信息使用显式 `UNKNOWN`，不使用空字符串或猜测值。

关键 fail-closed 规则包括：

- `C_multiview` 必须是多视图输入且 condition view 数量大于 1；
- 多视图 entry 不能声明 A/B 单图 track；
- `B_camera` 必须声明 full-path trajectory，并具备 extrinsics 或 Plücker camera 能力；
- `T2_arbitrary_direction` 不接受 none/UNKNOWN trajectory support；
- `UNKNOWN` 能力不能通过 admission 披露检查。

## 5. M5-3：Baseline Card v2 overlay

新增 schema ID `rswm.baseline_card_overlay/v2`。overlay 是 additive sidecar 合同，不覆盖或重写 card v1，并明确包含：

- 完整 `BaselineIdentity`；
- `BaselineCapabilities`；
- 旧 card 的 `rsroll.baseline_card/v1` schema、相对路径和 SHA-256；
- capability partition；
- admission state 与逐步证据；
- overlay checksum。

状态机固定为：

```text
DECLARED
   |
   | adapter evidence
   v
ADAPTER_AVAILABLE
   |
   | checkpoint evidence
   v
CHECKPOINT_VERIFIED
   |
   | runtime probe evidence
   v
RUNNABLE
   |
   | result evidence
   v
EVALUATED
   |
   | admission decision evidence
   v
ADMITTED
```

每个状态要求历史中恰好包含此前全部阶段，且 evidence kind 与目标阶段匹配。`advance()` 只允许进入唯一下一状态；构造或载入被篡改、缺段或乱序的历史同样会失败。

额外前置条件如下：

- `ADAPTER_AVAILABLE`：adapter identity 必须为 verified；
- `CHECKPOINT_VERIFIED`：checkpoint identity 必须为 verified；
- `RUNNABLE`：partition、tasks、tracks 与完整 runtime identity 必须已解析；
- `ADMITTED`：完整 identity 已验证，license/external data/runtime/capability 已披露，且不得使用 future GT。

## 6. M5-4：Capability Partition

| 分区 | 准入合同 | 明确拒绝 |
|---|---|---|
| native | 单图或 video-prefix 输入；原生、冻结接入；能力足以消费所声明 track | UNKNOWN input/adaptation、multiview、通过 adapter 补足原生能力 |
| adapted | 单图或 video-prefix 输入；明确声明 prompt bridge、recursive、LoRA 或 full fine-tune | frozen/UNKNOWN adaptation、multiview、自动回退到 native |
| multiview | `input_type=multiview`、`condition_views>1`、仅 `C_multiview` | A/B track、单图伪装、多视图自动回退到 adapted/native |

`assert_eligible()` 必须同时接收明确的 task、track 和 partition；校验失败直接抛出 capability error，不搜索其他分区，也不把 camera condition 降级为文本后冒充 native。特别地，无 camera、仅 intrinsics 或 UNKNOWN camera 能力均不能进入 `B_camera/native`。

## 7. M5-5：Legacy Adapter

`LegacyBaselineAdapter` 只读取现有 `server_run/baselines/cards.py`、`registry.py`、`protocol.py` 与 card JSON；不会调用任何 adapter 的 `run()`，也不会写回旧卡或旧结果。

保守映射规则为：

| v1 已知事实 | v2 映射 |
|---|---|
| `number_of_condition_views > 1` | `input_type=multiview`、partition=`multiview` |
| 单 condition view | `input_type=UNKNOWN`，不猜 single-image/video-prefix |
| `uses_camera_pose=false` | `camera_support=none` |
| `uses_camera_pose=true` | `camera_support=UNKNOWN`，不猜 extrinsics/Plücker |
| trajectory 明确为 `none` 或 `ignored` | `trajectory_support=none` |
| 其他自由文本 trajectory 描述 | `trajectory_support=UNKNOWN` |
| v1 checkpoint/source/adaptation/license 自由文本或缺失 | 不升级为已验证结构化事实 |

所有 16 份旧卡生成的 overlay 初始状态均为 `DECLARED`。tasks/tracks、license、external data、output format 等无法从 v1 可靠证明的字段保持 UNKNOWN 或空 eligible 集合；不存在因旧 registry 中有 adapter class 就自动进入 RUNNABLE/EVALUATED/ADMITTED 的行为。

## 8. 验证结果

以下均为本地 CPU 验证结果，不代表真实 baseline、checkpoint、GPU 或正式 benchmark 已运行。

| 验证 | 已观察结果 |
|---|---|
| `python -m compileall -q baselines tests/benchmark` | PASS |
| M5 定向 unit tests | **20 PASS / 0 FAIL** |
| `python -m unittest discover -s tests/benchmark -v` | **68 PASS / 0 FAIL** |
| 载入并生成全部 legacy overlay | **16/16 成功，均为 DECLARED** |
| 全部 legacy overlay 载入后的模块检查 | `torch_loaded=False`，`transformers_loaded=False` |

68 项全量回归包含既有 M1–M4 的 core/schema、LegacyH5Adapter、task contract、trajectory convention、metric identity/registry、future-only、FID/FVD accumulator 和 legacy metric parity 测试。

## 9. 旧文件与禁改范围审计

### 已观察

以下旧 baseline 文件的 SHA-256 与 M5 修改前记录一致：

| 文件 | SHA-256 |
|---|---|
| `server_run/baselines/cards.py` | `81D4493B160E2BB8D3967E6763E15E8C1964B399E4CF9E1AFA7C41A7EA691A6A` |
| `server_run/baselines/registry.py` | `C4191693E047122A88065F715E1FB833DFA28CA82DB2AFA93F7D738315400C4E` |
| `server_run/baselines/protocol.py` | `0AA109DD80FABD74792CB61502E9EC3F73A40B9D566110958DC7EAAE81AC33BA` |
| `server_run/baselines/baseline_status.json` | `B4FF01F025738B97825E541B02281BEA4374264AAD4A70973995DE32A0B2834E` |
| `server_run/baselines/cards/persistence.json` | `4B46733D1EBDA5802A9754CFADC75604A6CBD61DF12169064086304EEE5301F8` |
| `server_run/baselines/cards/frozen_i2v.json` | `F4DE8483408EAFA7CFAD5D2CEFE7233266595BACF125E39AB0BAF94B0AFE539F` |
| `server_run/baselines/cards/mvsplat360.json` | `FB228EA851445AFE7B0ED3210E44E8F82CDAF1818959F248B0C1F0747FBAED3E` |

M5 的实际 patch 目标仅为第 2 节列出的新模块、新测试和本报告。没有编辑：

- `server_run/baselines` 下的 v1 card、registry、protocol、status 或 result；
- LingBot/Wan、`inference.py`、`backbone.py`；
- LoRA、训练配置和 ICLR pipeline；
- HDF5、manifest v2、M4 evaluator 或 metric 数学；
- 任何已有 baseline result。

当前项目目录在上级 Git 工作树中整体显示为未跟踪子树，因此没有将空 `git diff` 作为“旧文件未变化”的证据；本报告使用实际 patch 集、回归测试和修改前后内容哈希进行核验。

## 10. 兼容性与对已有实验的影响

新 `baselines/` 包为 opt-in，现有 ICLR pipeline 和 `server_run/baselines` 不导入它，默认执行路线不变。v2 overlay 通过旧卡哈希建立引用关系，但不会改变 v1 schema 或旧卡内容。因此本阶段：

- 不改变模型结构、LoRA 或生成行为；
- 不改变 benchmark protocol、HDF5、manifest 或 metric contract；
- 不重算、不覆盖、不升级任何旧 baseline 数字；
- 不把 adapter/stub/card 的存在解释为 checkpoint 已验证、模型可运行或 baseline 已复现；
- 不影响已有 ICLR 实验结果。

## 11. 尚未验证与停止点

### 尚未验证

- 任一外部 baseline 的真实模型身份、checkpoint hash、许可与 external-data 声明；
- 任一真实 adapter 的输入输出合同、运行环境或可重复执行性；
- RUNNABLE、EVALUATED 或 ADMITTED 状态的真实证据；
- GPU、真实 checkpoint、正式 dataset 或正式 evaluation；
- v2 overlay 的发布目录、签署/审查流程及 leaderboard 集成。

测试中的完整状态推进仅使用合成 CPU fixture，用于验证状态机合同，不代表任何现有 baseline 获得 admission。

### 停止点

M5 要求的 baseline identity、capability、v2 overlay、严格状态机、能力分区、legacy compatibility 与 CPU tests 已完成。本阶段在此停止；没有实现或运行真实 baseline，没有进入后续模型 adapter、正式 evaluation 或 leaderboard 开发。
