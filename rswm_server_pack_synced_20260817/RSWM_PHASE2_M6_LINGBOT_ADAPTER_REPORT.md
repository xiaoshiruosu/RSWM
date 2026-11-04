# RSWM-Bench Phase 2 M6 LingBotWorldModelAdapter Integration 实施报告

日期：2026-08-24  
阶段：Phase 2 M6（LingBotWorldModelAdapter Integration）  
状态：**M6 adapter integration 与 CPU mock parity 已完成；按授权边界停止，未加载真实 checkpoint、未运行 GPU、训练或正式 evaluation。**

## 1. 范围与结论

本阶段将已确认的 LingBot-World / Wan 模型身份接入通用 `WorldModel` abstraction，新增一个依赖轻量、惰性构造、可注入 mock backend 的具体适配器：

```text
WorldModel.generate(condition image, trajectory condition, metadata)
                              |
                              v
                 LingBotWorldModelAdapter
                              |
             input/capability/identity validation
                              |
                              v
         existing build_backbone("wan", checkpoint_dir=..., ...)
                              |
                              v
                  existing WanBackbone.generate
                              |
                              v
              canonical THWC uint8 Rollout
                              |
            model/adapter/execution identity binding
```

适配器没有复制或重写 checkpoint loader、LoRA merge、sampling loop、VAE decode、视频 canonicalization 或 Wan camera-conditioning 数学。真实生成仍只通过既有 `VideoBackbone.generate()` / `WanBackbone.generate()` 边界执行。

## 2. 修改文件

| 文件 | 类型 | 作用 |
|---|---|---|
| `models/adapters/lingbot_adapter.py` | 新增 | `LingBotWorldModelAdapter`、runtime config、惰性 Wan factory、参数转发及 identity binding |
| `models/adapters/__init__.py` | 新增 | 导出 LingBot adapter 公共 API |
| `models/__init__.py` | 修改 | 从通用 `models` 包导出 adapter 与 runtime config；不导入 torch/Wan 权重 |
| `benchmark/core/types.py` | 向后兼容扩展 | 为 `Rollout` 增加可选且 all-or-none 的 model/adapter/execution identity 三元组 |
| `tests/benchmark/test_lingbot_adapter.py` | 新增 | CPU Wan mock 参数转发、输出合同、identity lineage 与 fail-closed 回归 |
| `RSWM_PHASE2_M6_LINGBOT_ADAPTER_REPORT.md` | 新增 | 本阶段实施与验证报告 |

没有修改 `models/world_model.py` 的 `WorldModel.generate()` 签名，也没有修改低层 `backbone.py`、`inference.py` 或 `evaluate.py`。

## 3. M6-1：LingBotWorldModelAdapter

`LingBotWorldModelAdapter.generate()` 保持现有 `WorldModel` 合同：

```python
generate(
    condition_image: np.ndarray,
    trajectory_condition: TrajectoryCondition,
    metadata: GenerationMetadata,
) -> Rollout
```

调用顺序如下：

1. 构造 `ModelInput`；
2. 使用既有 `validate_model_input()` 检查 image、track、pose、intrinsics、frame count 和 output size；
3. 使用 `WorldModelCapabilities.assert_eligible()` 检查 task、track、prompt schema 与 17/25/33 horizon；
4. 将 benchmark 输入映射为现有 `VideoBackbone.generate()` 参数；
5. 调用 backend；
6. 封装 `Rollout`；
7. 使用既有 `validate_rollout()` 强制 `(T,H,W,3) uint8`、sample ID、seed 与分辨率合同。

任何错误帧数、错误尺寸或错误 dtype 都不会被 adapter 静默 resize、截断或填充。

## 4. 模型身份绑定

适配器构造的 `ModelIdentity` 固定分离以下命名空间：

| 身份层 | 绑定值 |
|---|---|
| release identity | `LingBot-World-base-cam` |
| model family | `Wan-I2V-A14B` |
| architecture transformer identity | `Wan-I2V-A14B` |
| adapter identity | `LingBotWorldAdapter` |
| execution identity | `benchmark-adapter` |
| architecture aliases | `wan`、`i2v-A14B` |
| execution legacy aliases | `wan`、`wan-inline` |
| camera encoding | `plucker_ray_v1` |

其余 architecture identity 延续 Backbone Identity Audit 的静态事实：

- VAE：`Wan2.1_VAE`；
- text encoder：`umT5-XXL`；
- camera conditioning：Plücker-ray camera conditioning；
- pose injection：global projection + per-transformer-block modulation；
- injector：camera injector with scale/shift。

`CheckpointIdentity` 由调用者显式传入并直接绑定到 `ModelIdentity`，不会从目录名猜测 repository/revision/hash。可选 `LoraIdentity` 同样进入 `ModelIdentity`；只提供 LoRA 路径而无 identity，或只提供 identity 而无路径，都会失败。

为了防止测试替身形成错误科研声明，只要使用自定义 `backbone_factory`，即使传入 verified checkpoint identity，生成的 model identity 也只会保持 `declared`。mock 的存在不能把模型提升为 verified。

## 5. M6-2：不复制 inference 逻辑

### 已观察

对 M6 新增模块的静态扫描未发现以下调用或导入：

- `load_lora_model()`；
- `generate_video()`；
- `vae.decode()`；
- `merge_and_unload()`；
- `torch` import；
- sampling loop。

adapter 的唯一生成调用为：

```text
backbone.generate(
  first_frame=...,
  prompt=...,
  num_frames=...,
  size=...,
  pose_cond=...,
  seed=...,
)
```

默认工厂在第一次需要 backend 时才惰性导入现有 `build_backbone`，并调用：

```text
build_backbone("wan", checkpoint_dir=..., lora_path=..., runtime kwargs...)
```

adapter 构造和 identity 检查本身不会导入 `lora_remotesense.new.backbone`、torch 或 transformers。

## 6. M6-3：Checkpoint 与 runtime 参数链

构造器显式接收：

- `checkpoint_identity: CheckpointIdentity`；
- `checkpoint_path`；
- `lora_identity: Optional[LoraIdentity]`；
- `lora_path`（与 LoRA identity 成对）；
- `runtime_config: LingBotRuntimeConfig`；
- `backbone_factory`（仅作为 CPU mock/test seam）。

`LingBotRuntimeConfig` 固定并校验以下 Wan 参数：

| 参数 | 默认值 |
|---|---:|
| `device` | `cuda`（仅配置值；本阶段未使用 GPU） |
| `config_key` | `i2v-A14B` |
| `sampling_steps` | 70 |
| `shift` | 10.0 |
| `guide_scale` | 5.0 |
| `t5_cpu` | true |
| `inactive_model_cpu` | true |
| `framewise` | true |

`config_key` 不能改为其他 Wan 架构后仍沿用 LingBot/Wan-I2V-A14B identity；不一致会直接失败。

无生成构造检查实际经过默认 registry 路径后得到：

```text
backend_class=WanBackbone
checkpoint_dir=/not-loaded/checkpoint
device=cpu
backend_loaded=False
torch_loaded=False
```

这确认 `checkpoint_dir` 已由 WorldModel adapter 显式传入现有 registry，而不是依赖旧 `evaluate.py` 的无参数 `build_backbone()` 路径。该检查没有调用 `WanBackbone.load()` 或 `generate()`，没有读取路径内容。

## 7. Prompt、pose、seed 与帧数转发

### Track A：`A_text`

- prompt：原样传递 `TrajectoryCondition.scene_prompt`；
- pose：`None`；
- legacy Wan pose guard：调用期间临时设为 `drop`；
- seed：显式使用 `GenerationMetadata.generation_seed`；
- frame count：显式使用 metadata horizon。

### Track B：`B_camera`

- prompt：原样传递；
- pose：映射为现有 `WanBackbone.coerce_pose_cond()` 支持的 `c2w + intrinsics` 形式；
- legacy Wan pose guard：调用期间临时设为 `native`；
- pose 数量必须与 horizon 一致，形状必须为 `(T,4,4)` 和 `(T,4)`；
- seed、frame count 与 output size 均显式传递。

现有 `WanBackbone` 同时通过显式 `pose_cond` 和 `RSWM_POSE_MODE` 做一致性检查。adapter 只在调用期间、在进程锁保护下设置与 track 对应的 mode，并在结束或异常后恢复调用前环境值；未修改旧 wan-inline 路径。

## 8. M6-5：Rollout identity lineage

`Rollout` 新增三个可选字段：

- `model_identity: Optional[ModelIdentity]`；
- `adapter_identity: Optional[str]`；
- `execution_identity: Optional[ExecutionIdentity]`。

扩展保持旧代码兼容：旧 `Rollout` 不提供任何 identity 时行为不变。一旦提供其中任一字段，则三个字段必须全部存在，并满足：

- model identity checksum 有效；
- `model_id == model_identity.release_identity`；
- `adapter_identity == model_identity.adapter_identity`；
- execution identity 与 model identity 内记录完全一致。

`LingBotWorldModelAdapter` 每次输出都会填充这三个字段，并在 runtime metadata 中记录 model identity checksum、公开 checkpoint repository、runtime config、pose mode、backbone 名和是否启用 LoRA。出于可移植性与隐私考虑，本地 checkpoint 路径不写入 rollout；其公开身份由结构化 `CheckpointIdentity` 承担。

## 9. Capability 声明

当前 adapter 声明：

- input：single image；
- tasks：T1/T2/T3；
- tracks：A_text、B_camera；
- prompt schema：`net_continuous`；
- horizons：17/25/33；
- trajectory：net direction + full path；
- camera：`plucker_ray_v1`；
- output：`THWC_uint8`；
- base checkpoint 无 LoRA：native / frozen；
- 显式 LoRA identity + path：adapted / lora。

该分区只声明输入与适配条件，不修改 M5 admission 状态，也不自动把 LingBot 或任何 baseline 标记为 EVALUATED/ADMITTED。

## 10. CPU 验证结果

以下均为本地 CPU/静态验证结果，不代表真实模型或正式 benchmark 已运行。

| 验证 | 已观察结果 |
|---|---|
| `run_iclr_full_pipeline.sh --plan` | PASS；`GPU_ACCESSED=0`、`TRAINING_STARTED=0`、`FINAL_RESULT=PLAN_ONLY` |
| `python -m compileall -q benchmark/core models tests/benchmark` | PASS |
| M6 mock parity 定向测试 | **9 PASS / 0 FAIL** |
| `python -m unittest discover -s tests/benchmark` | **77 PASS / 0 FAIL** |
| 原有 full-pipeline + GPU scheduler CPU tests | **22 PASS / 0 FAIL** |
| `bash -n server_run/*.sh` | PASS |
| `python server_run/test_semantic_gate.py` | **96 PASS / 0 FAIL** |
| adapter import + construction | `torch_loaded=False`、`transformers_loaded=False`、`backbone_module_loaded=False`、`backend_built=False` |
| 默认 registry 无生成构造 | `WanBackbone` 创建成功；checkpoint path 已传入；`backend_loaded=False`、`torch_loaded=False` |

SemanticGate 测试第一次在受限 Windows 临时目录创建测试 HDF5 时因 ACL 返回 `PermissionError`；同一代码在获准的沙箱外临时目录重跑后为 96/96 PASS。该失败属于测试临时目录权限，不是 SemanticGate 逻辑失败；两次真实结果均在此报告中保留，没有把初次失败隐去。

Mock 测试具体覆盖：

- `WorldModel` interface 验证；
- checkpoint path、LoRA path 与全部 runtime kwargs 的工厂转发；
- prompt、pose、seed、frame count、`H*W` size 转发；
- A_text/drop 与 B_camera/native 映射及环境恢复；
- canonical output shape/dtype/read-only；
- identity namespace 与 rollout lineage；
- backend 单例惰性构造；
- 错误 output contract 拒绝；
- LoRA identity/path 不完整拒绝；
- mock factory 不得提升 verified 身份。

## 11. 禁改范围与哈希审计

### 已观察

以下冻结文件哈希与 M6 修改前记录一致：

| 文件 | SHA-256 |
|---|---|
| `lora_remotesense/new/evaluate.py` | `BD08CAC22F7C2D2EC835E2103B79982A0BA2B327C40BB63F051654770B2565E0` |
| `lora_remotesense/new/inference.py` | `9703FA88143DF80D146815555DF7D48D4B815BB942E12994D6BF8BFF185D1008` |
| `lora_remotesense/new/backbone.py` | `590A759CBCAB24A21226D052C3B6F8FB18B0F294AF9761E3B4814C97C6570F13` |
| `lora_remotesense/new/train.py` | `DF585FE0EFD083B16484CA1F5DCA1263AB29ADFE04E2228BA9FC5A0118A0DFA4` |
| `server_run/run_iclr_full_pipeline.sh` | `D8073D89FB210DE240737E18ABB2CDC9E218787C1EC200B158C850A43E753342` |
| `benchmark/datasets/manifest_v2.py` | `D035F0A8BD3EB93D948D7FDACFD51BA21E2BBD16178C3A831FD006DADCDF1D1F` |
| Primary s42 config | `07B82612F86953BA5EE9F8C38D2ECA2143573B79F69D007EACEB4EAED6514BC7` |
| Primary s43 config | `54409C8DFFCCE3E8F9A1C7D29746BF3DDB53DAFF80ABA6C420AC82B6AA69B87F` |
| Primary s44 config | `3739D1FABB58280CDFB1FC599A2ECC8EF27D7C3B1CCCB4E873C119675FD161CF` |

因此本阶段没有修改：

- LingBot/Wan 模型结构；
- inference、VAE decode 或 sampling；
- LoRA 实现、target、rank、alpha 或训练配置；
- ICLR pipeline；
- HDF5 与 manifest v2；
- evaluator 数学、正式结果或旧 baseline result。

当前顶层同步目录在上级 Git 工作树中整体为未跟踪子树，不能以空 `git diff` 证明文件未变化；本报告使用实际 patch 目标、测试与修改前后 SHA-256 审计禁改范围。

## 12. 对已有 ICLR 结果的影响

影响为零：

- adapter 是 opt-in，新 ICLR runner 尚未调用它；
- 当前 official `wan-inline` 执行路线保持不变；
- `Rollout` identity 字段为带默认值的 additive 扩展，旧构造调用保持有效；
- 未重跑、重算、覆盖或升级任何已有结果；
- 未改变模型输入、生成数学、dataset、split 或 metric contract。

## 13. 尚未验证与停止点

### 尚未验证

- 服务器真实 LingBot-World checkpoint 的内容、固定 revision 与 SHA-256 manifest；
- checkpoint path 与传入 `CheckpointIdentity` 在真实服务器上的一致性；
- 可选 LoRA 权重目录与 `LoraIdentity` 的 artifact hash/lineage；
- 真实 `WanBackbone.load()`、torch 权重加载、GPU 显存与生成；
- 真实 480×832、17/25/33 rollout 的数值和性能；
- 正式 evaluation、result bundle、leaderboard 或 paper artifact 集成。

本阶段的默认 registry 检查只构造未加载的 `WanBackbone`；所有 `generate()` 测试均使用 CPU mock。不得把这些结果表述为“真实 LingBot checkpoint 已跑通”或“正式 evaluation 已完成”。

### 停止点

M6 要求的 LingBot WorldModel adapter、显式 checkpoint/runtime 参数链、Wan mock parity、canonical Rollout 与 identity lineage 已完成。本阶段在此停止；没有加载真实 checkpoint，没有运行 GPU、训练或正式 evaluation，也没有进入后续 runner/leaderboard 开发。
