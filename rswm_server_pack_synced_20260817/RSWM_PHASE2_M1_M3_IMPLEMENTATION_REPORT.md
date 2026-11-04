# RSWM-Bench Phase 2 M1-M3 实施报告

> 实施日期：2026-08-23  
> 实施范围：M1-M3（core types、schema、mock world model、`LegacyH5Adapter`、task contract、CPU fixture tests）  
> 结论：M1-M3 已按批准边界完成并通过 CPU 测试；M4 evaluator、正式 evaluation、GPU 工作与外部 baseline 均未进入。  

## 1. 授权与边界

项目负责人已确认 Phase 1 的 P1-01 至 P1-15，并新增、确认：

- `P1-16 Benchmark Release Identity`：统一绑定 protocol、benchmark spec、schema bundle、evaluator 与 manifest identity；
- `P1-17 Artifact Lineage`：记录 `checkpoint → adapter → benchmark spec → manifest → metric → result` 的可审计链路。

本次严格限定于 M1-M3。未修改模型结构、Wan/Video DiT、LoRA、训练配置、ICLR pipeline、旧 benchmark 数据格式或既有 HDF5；未启动 GPU、训练、正式 evaluation 或外部 baseline。

## 2. 已实施内容

### 2.1 Core types 与合同

新增强类型、不可变合同及失败关闭校验，覆盖：

- 冻结 task ID：`T1_spatial_rollout`、`T2_arbitrary_direction`、`T3_overlap_ladder`；
- 冻结 track ID：`A_text`、`B_camera`、保留但未实现的 `C_multiview`；
- data domain、capability partition、prompt schema；
- `BenchmarkSample`、`OrganizerSample`、`PublicSample`、`GroundTruthBundle`、`ModelInput`、`Rollout`；
- 严格 `(T,H,W,3) uint8` rollout 合同，canonical rollout 含 `t=0`，future-only 起点显式记录为索引 1；
- ndarray 防御性复制与只读标记，避免调用方原地改写已校验输入。

### 2.2 P1-16 Benchmark Release Identity

新增 checksum 保护的 `BenchmarkReleaseIdentity`，统一绑定五类组件：

1. protocol；
2. benchmark spec；
3. schema bundle；
4. evaluator；
5. manifest。

每个组件具有 artifact ID、schema、version、SHA-256 与状态。`require_verified=True` 时，任一组件不是 `verified` 都会失败关闭。M1-M3 尚未实现正式 evaluator，因此 fixture 中 evaluator 明确为 `declared`，整个 release identity 也明确为 **not release-ready**，不会把当前合同测试误写成正式 benchmark release。

### 2.3 P1-17 Artifact Lineage

新增有序 artifact lineage：

```text
checkpoint
  → adapter
  → benchmark_spec
  → manifest
  → metric（可有多个）
  → result
```

链路节点均携带 SHA-256；相邻关系由节点顺序确定并纳入 lineage checksum。实现拒绝跳过 adapter/spec/manifest、顺序逆转、重复 singleton、result 后追加节点，以及与 release identity 不一致的 spec/manifest checksum。

### 2.4 Benchmark spec 与机器可读 schema

新增 checksum-protected `rswm.benchmark_spec/v1` sidecar，不回写 manifest v2。sidecar 绑定：

- task、track、domain、capability partition；
- frozen manifest identity 与路径；
- protocol、schema bundle、evaluator 声明；
- metric panel、direction convention、overlap definition；
- sample visibility policy 与 task parameters。

新增三份 JSON Schema：benchmark spec、release identity、artifact lineage。schema bundle 对每份 schema 的规范化 JSON 计算 SHA-256，并作为 release identity 的一个独立组件。

### 2.5 Trajectory、direction 与 overlap 合同

新增只读语义适配，不复制或修改既有轨迹生成算法：

- 坐标采用 `(cy,cx)`，图像 y 轴向下；
- camera bearing 使用 `atan2(-dcy, dcx)`；
- content bearing 为 camera bearing 加 180°；
- 八方向采用 ±22.5° 半开分桶；
- `net_ratio < 0.2` 判为净方向不适定，边界 0.2 有效；
- overlap ladder 严格采用 L1 `[0.70,0.90)`、L2 `[0.40,0.60)`、L3 `[0.15,0.25)`、L4 `[0,0.10)`，buffer/gap 不被静默归类。

### 2.6 WorldModel 与 CPU mock

新增面向 benchmark 的 `WorldModel` protocol 与能力声明。`DeterministicMockWorldModel` 仅使用 NumPy 在 CPU 上生成确定性平移序列，用于验证输入、输出和 task 合同；其元数据明确标记 `mock_only=True`、`scientific_result_eligible=False`，不属于可准入 baseline。

本阶段未接入 Wan adapter，也未导入或调用任何真实模型、权重、torch、CUDA 或 GPU 运行路径。

### 2.7 Task contract

新增 T1/T2/T3 task 与 registry：

- Track A 只向模型提供公开图像、文本和净连续方向，隐藏 pose、intrinsics、trajectory、GT 与 source locator；
- Track B 仅额外公开 poses 与 intrinsics，不在公共 condition 中暴露 waypoints/per-frame centers；
- official 输入必须是 `net_continuous`，legacy `net` 会被拒绝，不能追溯性重标；
- T2 检查方向适定性；
- T3 按 spec 中冻结的 overlap level 进行门禁；
- `evaluate()` 当前只有委托接口，不计算指标，避免越过 M4 授权边界。

### 2.8 LegacyH5Adapter

新增只读兼容层，将 canonical legacy HDF5 与 `rsroll.benchmark_manifest/v2` 映射为 typed samples：

- HDF5 仅以 `h5py.File(..., "r")` 打开；
- 默认验证 HDF5、split、manifest checksum 与样本级 checksum；
- manifest v2 原样保留，task/track/domain 由 checksum sidecar 表达；
- public、organizer 与 ground-truth 视图物理分离；
- GT rollout 仅在 organizer/GT API 被调用时延迟读取；
- 按既有 manifest builder 的 float64 trajectory checksum 合同校验，不改变 HDF5 中原始 dtype 或字节。

## 3. 文件清单

### 3.1 新增实现与 schema

- `benchmark/core/`：`errors.py`、`identity.py`、`types.py`、`contracts.py`；
- `benchmark/spec/`：`schema.py`、`loader.py` 与三份 JSON Schema；
- `benchmark/trajectory/`：`types.py`、`conventions.py`；
- `benchmark/tasks/`：base、T1、T2、T3 与 registry；
- `benchmark/datasets/`：base、manifest v2 wrapper、`LegacyH5Adapter`；
- `models/`：`WorldModel` protocol、capabilities、registry、deterministic mock；
- 上述 package 的 `__init__.py`。

### 3.2 新增 CPU 测试

- `tests/benchmark/fixture_factory.py`；
- `tests/benchmark/test_identity_and_schema.py`；
- `tests/benchmark/test_trajectory_conventions.py`；
- `tests/benchmark/test_legacy_h5_adapter.py`；
- `tests/benchmark/test_task_contracts.py`。

### 3.3 更新规范记录

- `RSWM_BENCH_PROTOCOL_V1.md`：写入冻结状态与 P0 批准记录；科学定义未改；
- `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md`：P1-01 至 P1-15 标记已确认，并增加 P1-16/P1-17；
- `RSWM_PHASE1_ARCHITECTURE_REVIEW.md`：同时保留审查时 hash 与批准记录写入后的 frozen hash，避免覆盖历史证据。

## 4. 测试结果

测试环境：Windows、Python 3.13.5、NumPy 2.1.3、h5py 3.12.1，CPU-only。

已实际执行：

```text
python -m compileall -q benchmark models tests/benchmark
python -m unittest discover -s tests/benchmark -p 'test_*.py' -v
```

最终结果：

```text
Ran 24 tests
OK
```

覆盖内容包括：release identity 完整性、未实现 evaluator 的失败关闭、lineage 顺序与防跳级、schema bundle 绑定、spec 篡改检测、manifest 篡改检测、HDF5 只读与关闭状态、public/GT 隔离、trajectory checksum 兼容、T1/T2/T3 输入输出、task-from-spec registry、mock rollout、direction 边界、net-ratio 边界及 overlap ladder 边界。

测试实现过程中发现并修正两项仅限新 abstraction/fixture 的问题：

1. Windows 托管环境的系统临时目录 ACL 不能被 native HDF5 打开，CPU fixture 改为工作区内 UUID 隔离目录；
2. 既有 manifest builder 在计算 trajectory checksum 前固定转换为 float64，新 adapter 已严格匹配该既有合同，未修改 manifest 或 HDF5。

## 5. 既有生产代码保护证据

以下关键文件的 SHA-256 与 Phase 1 审查基线一致：

| 文件 | SHA-256（前 12 位） |
|---|---|
| `lora_remotesense/new/backbone.py` | `590A759CBCAB` |
| `lora_remotesense/new/inference.py` | `9703FA88143D` |
| `lora_remotesense/new/train.py` | `DF585FE0EFD0` |
| `lora_remotesense/new/evaluate.py` | `BD08CAC22F7C` |
| `lora_remotesense/new/metrics.py` | `C8A4225D0FA3` |
| `lora_remotesense/new/dataset_loader.py` | `125067823D59` |
| `lora_remotesense/build_dataset_from_benchmark.py` | `253636FD9028` |
| `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py` | `B74EDCB04B6E` |
| `server_run/baselines/protocol.py` | `0AA109DD80FA` |
| `server_run/baselines/build_benchmark_manifest.py` | `C88DE2EFD340` |
| `server_run/baselines/registry.py` | `C4191693E047` |
| `server_run/run_iclr_full_pipeline.sh` | `D8073D89FB21` |
| `configs/iclr_main/experiment_plan.json` | `D28AB3B8D7F3` |
| `tools/export_paper_tables.py` | `5327DA94EB8B` |

因此，本次没有改变模型、LoRA、训练超参数、benchmark builder、metric contract、baseline protocol、ICLR 实验计划或 paper table exporter。

## 6. 当前状态与下一门禁

### 已观察

- M1-M3 新增模块可在本地 CPU 环境编译并通过 24 个 fixture tests；
- 旧 HDF5 兼容层的生产实现为只读；测试只创建微型临时 fixture；
- P1-16/P1-17 已具备机器可读结构、checksum 与失败关闭校验；
- 既有生产关键文件 hash 未改变。

### 尚未验证

- 未对服务器正式 HDF5 执行只读兼容验证；
- 未实现或冻结 M4 evaluator identity；
- 未进行真实模型加载、GPU inference、训练或正式 evaluation；
- 未运行任何外部 baseline，不能宣称 benchmark release-ready 或生成 leaderboard 结果。

### 下一步门禁

M1-M3 到此停止。进入 M4 前需要项目负责人再次明确授权，并先冻结 evaluator/future-only ownership、metric accumulator 接口与 result schema。当前 `BenchmarkReleaseIdentity.is_release_ready` 预期为 false，这是正确的保护状态。

## 7. 本地测试副产物说明

受托管 Windows 临时目录 ACL 影响，首次 fixture 尝试在 `temp_benchmark/` 留下 23 个不可访问/未清理目录与 1 个 `h5_probe.h5`，编译过程也产生了 `__pycache__`。这些不是正式数据、结果或 release 文件，后续成功测试创建的 UUID fixture 均已自行清理。按照项目“清理前需明确授权”的规则，本次未删除上述遗留项。
