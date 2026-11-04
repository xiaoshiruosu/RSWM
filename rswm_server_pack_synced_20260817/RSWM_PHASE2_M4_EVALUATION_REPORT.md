# RSWM-Bench Phase 2 M4 Evaluation Framework 实施报告

日期：2026-08-23  
阶段：Phase 2 M4（Evaluation Framework）  
状态：**M4 CPU abstraction 已完成；按授权边界停止，未进入真实模型或正式 evaluation。**

## 1. 范围与结论

本阶段新增了与模型完全解耦的 benchmark evaluation framework，覆盖：

- 可审计的 `MetricIdentity` 与 `MetricResult`；
- sample metric 与 dataset metric 的统一抽象；
- 分版本、按 scope 校验的 metric registry；
- `FullRolloutView`、`FutureOnlyView` 与唯一 future-only owner；
- 对现有 `metrics.py`、`trajectory_metrics.py` 的薄包装；
- FID/FVD dataset accumulator 合同；
- visual、temporal、spatial、semantic 四类目录；
- CPU-only identity、registry、future-only、accumulator 与 legacy parity 回归测试。

本阶段没有修改任何模型、推理、训练、LoRA、ICLR pipeline、HDF5 或 manifest v2 实现；没有加载真实 checkpoint，没有运行 GPU，也没有生成正式 benchmark 数字。

## 2. 新增文件

### 2.1 Evaluation framework

| 文件 | 作用 |
|---|---|
| `evaluation/__init__.py` | 公共 API 导出，保持 CPU-safe |
| `evaluation/errors.py` | evaluator、identity、registry、owner 与 accumulator 的 fail-loud 异常 |
| `evaluation/identity.py` | `MetricIdentity`、枚举、implementation hash 与 lineage binding |
| `evaluation/result.py` | `MetricResult` 及 valid/null/N 合同 |
| `evaluation/views.py` | `FullRolloutView`、`FutureOnlyView`、`EvaluationView` |
| `evaluation/base.py` | `Metric`、`SampleMetric`、`DatasetMetric` |
| `evaluation/registry.py` | `MetricRegistry` 与 18 项 M4 panel 构建函数 |
| `evaluation/_legacy.py` | 旧指标真源的惰性、安全加载桥 |
| `evaluation/_distribution.py` | dataset accumulator 公共生命周期 |
| `evaluation/visual/legacy.py` | SSIM、PSNR、LPIPS、FID 包装 |
| `evaluation/temporal/legacy.py` | 现有 R3D-18 FVD proxy 包装 |
| `evaluation/spatial/legacy.py` | `score_trajectory()` 标量结果包装 |
| `evaluation/semantic/unavailable.py` | 输出级 semantic metric 的显式 `NOT_AVAILABLE` 扩展位 |
| 各分类目录 `__init__.py` | 稳定分类 API |

### 2.2 CPU tests

| 文件 | 覆盖内容 |
|---|---|
| `tests/benchmark/test_evaluation_identity_registry.py` | identity round-trip、lineage binding、scope、重复注册、四分类 panel |
| `tests/benchmark/test_evaluation_views.py` | t=0 污染、t=1 删除、单次裁剪、owner 冲突 |
| `tests/benchmark/test_evaluation_accumulators.py` | FID/FVD 集合累积、生命周期、N≥32、禁止 per-sample 分数 |
| `tests/benchmark/test_evaluation_legacy_parity.py` | PSNR 与轨迹旧实现逐值一致、semantic placeholder、帧数截断拒绝 |

## 3. M4-1：Metric Identity

`MetricIdentity` 使用 schema ID `rswm.metric_identity/v1`，包含并校验以下字段：

| 字段 | 合同 |
|---|---|
| `metric_id` | 稳定、非空 ID |
| `version` | 指标/包装版本 |
| `category` | `visual` / `temporal` / `spatial` / `semantic` |
| `scope` | `sample` / `dataset` |
| `higher_or_lower_better` | `higher` / `lower` |
| `preprocessing` | 冻结的结构化预处理描述 |
| `checkpoint_identity` | 可选 `ComponentIdentity`；未核验权重只能标为 `declared` |
| `implementation_hash` | 包装器与实际旧数学真源的内容哈希 |
| `future_only_owner` | `evaluation_runner` / `legacy_wrapper` / `not_applicable` |
| `identity_checksum` | 对上述身份内容的 canonical SHA-256 |

每个 identity 可生成 P1-17 `ArtifactRef(kind="metric")`，其 SHA-256 等于 `identity_checksum`。`MetricResult` 嵌入完整 metric identity，从而可接入：

```text
checkpoint → adapter → benchmark spec → manifest → metric identity → result
```

无权重指标的 `checkpoint_identity` 为 `None`。LPIPS、Inception FID 与 R3D proxy 的本地权重未在本阶段加载或哈希，因此 checkpoint identity 诚实标记为 `declared`，没有伪装成 `verified`。

## 4. M4-2：Metric Registry 与统一接口

统一抽象为：

```text
Metric
├── SampleMetric.compute(view, metadata) -> MetricResult
└── DatasetMetric
    ├── start(context)
    ├── update(view, metadata)
    └── finalize() -> MetricResult
```

dataset metric 的 `compute()` 明确抛错，防止把 FID/FVD 伪造成逐样本分数后再平均。registry 在注册时同时校验 identity checksum、scope 与实现接口，重复的 `(metric_id, version)` 会失败退出；省略 version 且同 ID 存在多个版本时也不会猜测版本。

`build_m4_metric_registry(path_estimator=..., device=...)` 当前构造 18 个 identity：

- visual：SSIM、PSNR、LPIPS-Alex、InceptionV3 FID；
- temporal：R3D-18 proxy FVD；
- spatial：net direction、segment mean、30°/45° success、turn recall/precision/F1/timing、waypoint、endpoint、Fréchet、DTW，共 12 项；
- semantic：1 个明确不进入 headline 的 `NOT_AVAILABLE` placeholder。

现有 `metrics.py` 中的 FVD 真源实际采用 torchvision R3D-18，不是标准 I3D-FVD。因此新 ID 固定为 `temporal.fvd_r3d18_proxy`，预处理身份中同时记录 `standard_i3d_fvd=false`，没有把 proxy 冒充文献可比的 I3D-FVD。

## 5. M4-3：Future-only ownership

新框架同时保留两个语义明确的 view：

```text
canonical rollout [t=0, t=1, ..., t=T-1]
                |
                +-- FullRolloutView [t=0:T] ----> spatial/control metric
                |
                +-- 唯一 owner: evaluation_runner
                         |
                         +-- FutureOnlyView [t=1:T]
                                  └--> visual / temporal / distribution / semantic
```

当前细粒度 wrapper 调用 `compute_video_ssim()`、`compute_video_psnr()`、`compute_video_lpips()`、`compute_fid()` 和 `compute_fvd()`；这些函数本身不裁剪 t=0，因此 identity 声明 `future_only_owner=evaluation_runner`。框架不通过 `compute_all_metrics()` 再裁剪一次。

空间轨迹指标需要从 condition frame 开始的连续转移，使用 full rollout，并声明 `future_only_owner=not_applicable`。同一 `EvaluationView` 只能选择一个 future-only owner；任何 metric identity 与 evaluation owner 不一致都会 fail closed。

回归测试确认：

- 只污染 t=0，PSNR wrapper 数值不变；
- 删除 t=1，PSNR wrapper 数值发生变化；
- future view 恰好删除 1 帧且从 source index 1 开始；
- `evaluation_runner` 与 `legacy_wrapper` owner 冲突会抛出合同错误。

## 6. M4-4：Legacy metric wrappers

### 6.1 Visual 与 distribution

visual wrapper 直接调用现有函数，不复制公式：

- `compute_video_ssim()`；
- `compute_video_psnr()`；
- `compute_video_lpips()`；
- `compute_fid()`；
- `compute_fvd()`。

FID accumulator 按样本接收完整 future clip，但累积全部 future frame 后一次性调用旧 FID；FVD accumulator 每个样本累积一个完整 future clip，最终一次性调用旧 FVD。两者的正式最小样本数均硬限制为 32，不能通过构造参数降到 31 或更低。

### 6.2 Spatial trajectory

`LegacyTrajectoryMetric` 只调用 `trajectory_metrics.score_trajectory()` 并抽取一个已有标量 key，没有复制光流、registration、方向、turn、waypoint、Fréchet 或 DTW 数学。`path_estimator` 必须在 wrapper 构造时显式选择 `flow` 或 `registration`，并写入 identity；metadata 试图切换 estimator 会返回 invalid。metadata 的 `num_frames` 与 rollout 长度不一致时也会拒绝，避免旧函数内部截断掩盖 contract drift。

### 6.3 Invalid policy

`MetricResult` 固定包含：

- `value`；
- `valid`；
- `reason`；
- `N`；
- 完整 `metric_identity`；
- `warnings`。

有效结果要求有限数值且 `N>0`；无效或不可用结果必须使用 `value=null` 并给出非空 reason。不会用 0、90 或其他数值填充缺失结果。

## 7. 验证结果

以下均为本地 CPU 执行结果，不代表真实模型、GPU 或正式 benchmark 已运行。

| 验证 | 结果 |
|---|---|
| `python -m compileall -q evaluation tests/benchmark` | PASS |
| `python -m unittest discover -s tests/benchmark -v` | **48 PASS / 0 FAIL** |
| `python server_run/test_metrics_known_answer.py` | **68 PASS / 0 FAIL / 1 SKIP** |
| `python server_run/test_submission_contract.py` | **93 PASS / 0 FAIL** |
| 独立进程导入并执行 PSNR wrapper | PASS，`torch_loaded=False` |

known-answer 中唯一 SKIP 为本地未安装 `lpips`，因此没有加载 LPIPS-Alex 权重。wrapper 在此情况下返回 invalid/null，不会阻断其它 metric。FID/FVD 的真实深度特征权重同样没有加载；本阶段只通过 deterministic stub 验证 accumulator 合同与 N 语义。

## 8. 禁改范围与哈希审计

### 已观察

两份旧数学真源在本阶段前后哈希一致：

| 文件 | SHA-256 |
|---|---|
| `lora_remotesense/new/metrics.py` | `C8A4225D0FA31C2B3F4F49B877F8324630D30292062E8FA12452E8BDBEEEE4AE` |
| `lora_remotesense/new/trajectory_metrics.py` | `755A720EFD6C26583CD34600704141F8FCA07DC27FA9B99822BC41EB75D0294C` |

当前生产关键文件哈希与既有 backbone/Phase 2 审计记录一致：

| 文件 | SHA-256 |
|---|---|
| `lora_remotesense/new/evaluate.py` | `BD08CAC22F7C2D2EC835E2103B79982A0BA2B327C40BB63F051654770B2565E0` |
| `lora_remotesense/new/inference.py` | `9703FA88143DF80D146815555DF7D48D4B815BB942E12994D6BF8BFF185D1008` |
| `lora_remotesense/new/backbone.py` | `590A759CBCAB24A21226D052C3B6F8FB18B0F294AF9761E3B4814C97C6570F13` |
| `lora_remotesense/new/train.py` | `DF585FE0EFD083B16484CA1F5DCA1263AB29ADFE04E2228BA9FC5A0118A0DFA4` |
| `server_run/run_iclr_full_pipeline.sh` | `D8073D89FB210DE240737E18ABB2CDC9E218787C1EC200B158C850A43E753342` |
| `configs/iclr_main/primary_native_attn_cam_r16_s42.yaml` | `07B82612F86953BA5EE9F8C38D2ECA2143573B79F69D007EACEB4EAED6514BC7` |
| `configs/iclr_main/primary_native_attn_cam_r16_s43.yaml` | `54409C8DFFCCE3E8F9A1C7D29746BF3DDB53DAFF80ABA6C420AC82B6AA69B87F` |
| `configs/iclr_main/primary_native_attn_cam_r16_s44.yaml` | `3739D1FABB58280CDFB1FC599A2ECC8EF27D7C3B1CCCB4E873C119675FD161CF` |

`benchmark/datasets/manifest_v2.py` 未被本阶段编辑；M4 完成后记录的 SHA-256 为 `D035F0A8BD3EB93D948D7FDACFD51BA21E2BBD16178C3A831FD006DADCDF1D1F`。

本项目目录相对上级 Git 工作树当前整体显示为未跟踪子树，因此本阶段没有把空的 `git diff` 误作“无改动”证据；修改清单由实际 patch 目标、文件时间和上述内容哈希共同审计。

## 9. 兼容性与对既有 ICLR 结果的影响

新 `evaluation/` 包是 opt-in：现有 `evaluate.py`、训练入口、full pipeline 和 paper exporter 不导入它，也没有改变默认执行路线。旧指标文件作为唯一数学真源被惰性调用，原文件内容不变。因此：

- 不改变模型结构或生成行为；
- 不改变 LoRA target、rank、alpha 或训练超参数；
- 不改变 17/25/33 帧、overlap、trajectory 或 metric 数学；
- 不改变 HDF5、manifest v2 或已有 ICLR 结果文件；
- 不追溯重算、覆盖或升级任何既有 headline 数字。

## 10. 尚未验证与下一阶段边界

### 尚未验证

- LPIPS、Inception FID、R3D proxy FVD 的真实 checkpoint identity/hash 与真实数值；
- 标准 I3D-FVD。本阶段只诚实包装现有 R3D proxy，未创建或冒充 I3D 实现；
- registration estimator 在正式 480×832 rollout 上的运行与 calibration gate；
- 真实 LingBot-World checkpoint、GPU、正式 HDF5 或 official evaluation；
- 端到端 `EvaluationRunner` 与 result bundle/paper exporter 集成。

### 停止点

M4 所要求的 abstraction、registry、future-only ownership、legacy wrapper 和 CPU tests 已完成。本阶段在此停止；没有实现 `LingBotWorldModelAdapter`，没有连接真实模型，也没有进入 M5、正式 evaluation 或外部 baseline。
