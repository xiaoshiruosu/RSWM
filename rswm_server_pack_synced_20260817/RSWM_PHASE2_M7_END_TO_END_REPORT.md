# RSWM-Bench Phase 2 M7 CPU End-to-End Dry Run 报告

## 1. 结论

RSWM-Bench Phase 2 M7 已完成。当前仓库已经建立并通过 CPU fixture 验证的完整数据流：

```text
LegacyH5Adapter（只读 tiny fixture）
        ↓
Benchmark Task / track-specific projection
        ↓
ModelInput
        ↓
LingBotWorldModelAdapter（Wan mock backend）
        ↓
canonical Rollout + layered model identity
        ↓
EvaluationRunner
        ↓
existing metric wrapper → MetricResult
        ↓
ResultBundle + ArtifactLineage
```

本阶段结论仅为“CPU 端到端合同 dry-run 通过”。本阶段没有加载真实 LingBot checkpoint，没有导入或运行 PyTorch/Transformers，没有使用 GPU/H200，没有训练，没有正式 evaluation，没有运行外部 baseline，也没有修改旧 ICLR pipeline、HDF5 格式或 manifest v2。

M7 结果对象固定标记为：

- `purpose=cpu_end_to_end_dry_run`；
- `admission_eligible=false`；
- mock checkpoint artifact 显式记录 `mock=true`、`weights_loaded=false`。

因此，M7 结果不能进入 leaderboard、headline 或论文结果表。

## 2. 实施范围与边界

### 2.1 本阶段实现

1. CPU-only `EvaluationRunner`；
2. 可序列化、可校验的 `SampleIdentity` 与 `ResultBundle`；
3. `checkpoint → adapter → benchmark spec → manifest → metric → result` 完整 artifact lineage；
4. dataset → task → model → evaluation 的 CPU dry-run runtime；
5. Track A/Track B 可见性、future-only、invalid fail-closed、legacy parity 回归。

### 2.2 本阶段未实施

- 未实现或调用真实 checkpoint loader；
- 未修改 `LingBotWorldModelAdapter` 的 M6 模型接线；
- 未修改 `evaluate.py`、`inference.py`、`backbone.py`；
- 未接入正式 result exporter、leaderboard 或 paper table；
- 未把 M7 sample runner 扩展为正式 FID/FVD dataset job；M4 已有的 `start/update/finalize` accumulator 合同保持不变；
- 未修改模型、LoRA、训练配置、benchmark 科学协议或旧结果。

## 3. 修改文件

| 文件 | 类型 | 作用 |
|---|---|---|
| `evaluation/runner.py` | 新增 | CPU sample `EvaluationRunner`；构造唯一 evaluation view、调现有 wrapper、invalid fail-closed |
| `evaluation/bundle.py` | 新增 | `SampleIdentity`、`ResultBundle`、双层 checksum、完整 artifact lineage |
| `benchmark/runtime.py` | 新增 | 串联 dataset、task、LingBot adapter mock、evaluation 与 result bundle |
| `tests/benchmark/test_end_to_end_runtime.py` | 新增 | M7 CPU fixture 端到端与 tamper/fail-closed/legacy parity 测试 |
| `evaluation/errors.py` | 修改 | 新增携带 invalid `MetricResult` 的 `InvalidMetricResultError` |
| `evaluation/result.py` | 修改 | 增加 `MetricResult.from_document()`，支持 result bundle 安全反序列化 |
| `evaluation/__init__.py` | 修改 | 导出 runner、result bundle、sample identity 与新错误类型 |
| `RSWM_PHASE2_M7_END_TO_END_REPORT.md` | 新增 | 本报告 |

未修改 M6 adapter 逻辑，也未修改任何旧实验入口。

## 4. M7-1：端到端 runtime

### 4.1 输入与 task 投影

`CpuEndToEndDryRun` 要求 dataset 已绑定并校验 `BenchmarkSpec`，并在运行前检查：

- task ID 与 spec 一致；
- benchmark version 与 spec 一致；
- dataset release identity 与 spec 一致；
- sample ID 属于冻结 dataset；
- checkpoint/adapter artifact kind 正确。

runtime 先调用 `dataset.benchmark_sample()`，然后由 task 将其投影为 track-specific `ModelInput`。该路径复用 M1-M3 的 task contract，不重新定义输入协议。

### 4.2 GT 隔离

runtime 在模型生成前不读取 ground truth：

```text
benchmark_sample
  → task.prepare_input
  → WorldModel.generate
  → task.validate_output
  → dataset.ground_truth
  → evaluation
```

因此，GT 只在 rollout 已生成并完成 output contract 校验后进入 evaluator。`DryRunOutcome` 也不携带 `GroundTruthBundle`。

### 4.3 Track A 与 Track B

- Track A：`ModelInput.trajectory_condition.poses=None`、`intrinsics=None`；LingBot adapter 向 mock Wan backend 传 `pose_cond=None`，并使用 `RSWM_POSE_MODE=drop`；
- Track B：传递逐帧 `(T,4,4)` poses 与 `(T,4)` intrinsics；LingBot adapter 构造 `pose_cond={c2w, intrinsics}`，并使用 native pose mode；
- 两条路径都不向 model/backend 传 GT、full scene、source locator 或 organizer-only object。

### 4.4 输出合同

mock backend 返回 THWC `uint8` rollout，t=0 与 condition image 精确一致。输出继续由原有 task/contract 验证：

- frame count；
- `(T,H,W,3)` layout；
- `uint8` dtype；
- sample ID；
- generation seed；
- condition-frame continuity；
- structured rollout identity。

## 5. M7-2：标准 ResultBundle

### 5.1 结果字段

`ResultBundle` 包含：

- `benchmark_identity`：`BenchmarkReleaseIdentity`；
- `model_identity`：完整 layered `ModelIdentity`；
- `adapter_identity`；
- `execution_identity`；
- `sample_identity`；
- 一个或多个带完整 `MetricIdentity` 的 `MetricResult`；
- `artifact_lineage`；
- `result_payload_checksum`；
- `bundle_checksum`；
- `valid` 与显式 `invalid_reasons`；
- dry-run purpose 与 admission 状态。

`SampleIdentity` 绑定：

- benchmark version；
- task/track；
- sample ID；
- manifest checksum；
- condition checksum；
- generation seed；
- horizon；
- sample identity checksum。

结果文档不保存 GT 数组、HDF5 source reference 或 checkpoint 本地路径。

### 5.2 双层 checksum

为避免“result artifact hash 又包含自身 lineage”造成循环依赖，M7 使用两层绑定：

1. `result_payload_checksum`：对 benchmark/model/sample/metric result 等结果 payload 计算；
2. 将该 digest 作为 lineage 最后一个 `result` artifact 的 SHA-256；
3. `bundle_checksum`：再对 payload、result digest 与完整 lineage 一并计算。

任意 metric value、metric identity、sample identity、artifact digest、lineage 顺序或 bundle 内容被修改，反序列化或校验都会失败。

### 5.3 Artifact Lineage

M7 冻结并实测以下顺序：

```text
checkpoint
  → adapter
  → benchmark_spec
  → manifest
  → metric(s)
  → result
```

绑定规则包括：

- checkpoint artifact 必须绑定 `ModelIdentity.checkpoint_identity.repository`；若 checkpoint identity 已有 `sha256_manifest`，digest 必须一致；
- adapter artifact ID 必须等于 `ModelIdentity.adapter_identity`；
- benchmark spec 与 manifest digest 必须匹配 release identity；
- metric artifact 必须逐项等于 `MetricIdentity.artifact_ref()`；
- lineage model identity 必须等于 result bundle model identity；
- result artifact digest 必须等于 `result_payload_checksum`，且必须位于 lineage 末尾。

本阶段 checkpoint 是无权重的 mock artifact，不代表真实 LingBot checkpoint 已验证。adapter artifact 使用实际 `models/adapters/lingbot_adapter.py` 内容哈希；spec、manifest 和 sample 来自临时 CPU fixture。

## 6. M7-3：合同测试

### 6.1 Track A 不可见性

已验证：

- poses/intrinsics 均为 `None`；
- mock Wan backend 收到 `pose_cond=None`；
- `ModelInput` 不存在 ground-truth 字段；
- result bundle 序列化文档不包含 `ground_truth`；
- GT 只在 model generation 完成后加载。

### 6.2 Track B camera condition

已验证：

- backend 收到与 task input 逐元素一致的 `c2w`；
- backend 收到与 task input 逐元素一致的 intrinsics；
- pose mode 为 native；
- sample/result track identity 为 `B_camera`。

### 6.3 Rollout、metric 与 result identity

已验证：

- rollout model/release identity 为 `LingBot-World-base-cam`；
- adapter identity 为 `LingBotWorldAdapter`；
- architecture identity 为 `Wan-I2V-A14B`；
- execution route 为 `benchmark-adapter`；
- result bundle identity 与 rollout identity 一致；
- metric artifact SHA-256 等于嵌入 `MetricIdentity.identity_checksum`；
- result bundle round-trip 后 checksum 与 metric value 不变。

### 6.4 Future-only 单一 owner

`EvaluationRunner` 从所有 metric identity 中解析唯一 future-only owner：

- 同一次 evaluation 出现两个 owner 时直接拒绝；
- runner-owned view 从 full T=17 精确转换为 future T=16；
- probe metric 只被调用一次；
- `source_start_index=1`、`source_frame_count=17`；
- probe 看到的第一帧逐元素等于原 rollout 的 t=1，而不是 t=2。

### 6.5 Invalid fail-closed

显式返回 `MetricResult(valid=false, reason=...)` 的 metric 会触发 `InvalidMetricResultError`。默认 runtime 不会为 invalid metric 生成可用 result bundle，也不会用 0 或空值伪装为成功。

此外，即使显式构造 diagnostic invalid bundle，`validate(require_valid=true)` 仍会拒绝；M7 bundle 无论 valid 与否都不能通过 admission gate。

## 7. M7-4：Legacy parity

本阶段没有重写旧数学定义。测试继续调用：

- `lora_remotesense/new/metrics.py` 的现有 PSNR 实现；
- `lora_remotesense/new/submission_contract.py` 的 shape、dtype 与 condition-frame 规则；
- M4 `legacy_psnr_metric()` wrapper。

同一 rollout/GT fixture 上已验证：

1. 旧 submission shape check 返回空违规列表；
2. 旧 dtype check 返回空违规列表；
3. 旧 contract 将 T=17 精确裁为 future T=16；
4. wrapper 的 `MetricResult.value` 与旧函数直接对 `frames[1:]` 计算的值一致到 12 位小数；
5. 新 result bundle 内的值与该旧值一致；
6. bundle serialization/round-trip 前后 rollout 字节 SHA-256 不变；
7. 修改 metric value 或 lineage metric digest 后，result 读取 fail closed。

上述 parity 仅证明 fixture 上的新编排层没有改变旧函数结果，不等价于正式数据集或真实 checkpoint evaluation 已完成。

## 8. 验证结果

### 8.1 计划与依赖门禁

| 验证 | 结果 |
|---|---|
| `run_iclr_full_pipeline.sh --plan` | PASS；`GPU_ACCESSED=0`、`TRAINING_STARTED=0`、`FINAL_RESULT=PLAN_ONLY` |
| M7 独立进程依赖检查 | PASS；`TORCH_LOADED=False`、`TRANSFORMERS_LOADED=False` |
| M7 新文件静态扫描 | PASS；未发现 torch/transformers import、`torch.load`、`from_pretrained` 或 checkpoint 文件读取 |
| `python -m compileall -q benchmark evaluation models tests/benchmark` | PASS |

### 8.2 测试汇总

| 测试集 | 结果 |
|---|---|
| `tests.benchmark.test_end_to_end_runtime` | 6/6 PASS |
| `tests/benchmark` 全部 CPU tests | 83/83 PASS |
| 旧 full-pipeline/GPU-scheduler/A800 plan tests | 32/32 PASS |
| `server_run/test_submission_contract.py` | 93/93 PASS |

旧 submission contract 首次在受限 Windows 沙箱内运行时因系统临时目录 ACL 被阻断；在沙箱外原命令重跑后为 93 PASS / 0 FAIL。最终通过结果来自完整重跑，不是对中断结果的推断。

### 8.3 冻结文件哈希复核

以下关键文件哈希与 M6 结束时记录一致：

| 冻结对象 | SHA-256 |
|---|---|
| `lora_remotesense/new/evaluate.py` | `BD08CAC22F7C2D2EC835E2103B79982A0BA2B327C40BB63F051654770B2565E0` |
| `lora_remotesense/new/inference.py` | `9703FA88143DF80D146815555DF7D48D4B815BB942E12994D6BF8BFF185D1008` |
| `lora_remotesense/new/backbone.py` | `590A759CBCAB24A21226D052C3B6F8FB18B0F294AF9761E3B4814C97C6570F13` |
| `lora_remotesense/new/train.py` | `DF585FE0EFD083B16484CA1F5DCA1263AB29ADFE04E2228BA9FC5A0118A0DFA4` |
| `server_run/run_iclr_full_pipeline.sh` | `D8073D89FB210DE240737E18ABB2CDC9E218787C1EC200B158C850A43E753342` |
| `benchmark/datasets/manifest_v2.py` | `D035F0A8BD3EB93D948D7FDACFD51BA21E2BBD16178C3A831FD006DADCDF1D1F` |
| primary seed 42 config | `07B82612F86953BA5EE9F8C38D2ECA2143573B79F69D007EACEB4EAED6514BC7` |
| primary seed 43 config | `54409C8DFFCCE3E8F9A1C7D29746BF3DDB53DAFF80ABA6C420AC82B6AA69B87F` |
| primary seed 44 config | `3739D1FABB58280CDFB1FC599A2ECC8EF27D7C3B1CCCB4E873C119675FD161CF` |

该项目目录在父仓库中整体显示为未跟踪目录，不能用空 `git diff` 作为“未修改”的证据。因此本报告使用明确修改文件清单、保护文件内容哈希和行为回归共同审计。

## 9. 科研状态分级

### 9.1 已观察

- tiny legacy HDF5 fixture 经 `LegacyH5Adapter` 只读加载；
- Track A/Track B 的输入可见性符合合同；
- LingBot adapter 通过 mock Wan backend 生成 canonical rollout；
- existing PSNR wrapper 输出 `MetricResult`；
- ResultBundle 可生成、round-trip 并检测 tampering；
- 完整 P1-17 lineage 顺序与 digest binding 通过测试；
- M7 运行未导入 torch/transformers；
- 上述 CPU 测试结果全部通过。

### 9.2 合理推断

由于旧模型/训练/evaluation/pipeline/config 文件内容哈希未变，且 M7 仅新增独立 CPU abstraction/runtime，现有 ICLR 结果的生成逻辑没有被本阶段改写。该判断有代码边界、内容哈希和旧回归测试支持，但不是一次真实 GPU 数值复现实验。

### 9.3 尚未验证

- 真实 LingBot checkpoint 的加载与输出；
- H200/A800/GPU runtime；
- 新 adapter 与旧 `wan-inline` 的真实同 seed 数值 parity；
- 正式 FID/FVD/LPIPS 或 trajectory panel 的端到端运行；
- 正式数据集规模、正式 result exporter、leaderboard admission；
- 外部 baseline；
- evaluator release component 的正式 verified identity。

## 10. 停止点

M7 已在授权范围内完成并停止。下一阶段如需真实 checkpoint/GPU parity、正式 evaluator job 或旧 `wan-inline` 接线，必须另行授权并先执行 plan/preflight。本阶段不进入真实模型、GPU 或正式 evaluation。
