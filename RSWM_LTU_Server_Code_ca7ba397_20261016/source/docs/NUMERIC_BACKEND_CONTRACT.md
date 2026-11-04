# 本地数值后端合同

`numeric-camera` 已由统一 registry 和 suite 的实际 factory 接通。它从显式批准、逐文件 hash 锁定的本地源文件调用 `load_backend(weights_dir=..., device=...)`；不会根据模型名猜包、联网下载权重或退回噪声。原 Wan/LingBot 源码 ZIP 已实际读回并形成可追溯的加载/采样/训练接线，参见 [原后端指南](LEGACY_WAN_BRIDGE.md)。当前没有实际权重和 CUDA 设备；真实生成、训练与模型层消费验收仍为 `NOT_RUN`，不能把源代码和软件接线通过视为真实模型已验收。

模型配置必须明确填写 `model_id`、`adapter="numeric-camera"`、`device`、`backend_manifest`、完整 `camera` 字典及 `training_camera_sha256`；可选 `training_seed`。`prompt`、`diagnostic_noise`、`model_dir`、未来查询和私有世界字段均拒绝。`backend_manifest` 是 JSON 对象，恰好包含以下字段。

| 字段 | 要求 |
| --- | --- |
| source_dir | 本地、无 symlink 的原后端源目录 |
| source_tree_sha256 | `digest(training.locked_tree(source_dir))`；实际逐文件 SHA256 表，不含 `__pycache__` |
| entrypoint | 源目录内的相对 `.py` 路径，不能跳出目录 |
| weights_dir | 本地、非空、无 symlink 的明确权重目录 |
| weights_tree_sha256 | `digest(training.locked_tree(weights_dir))` |
| objective | 实际后端的 `flow_matching` 或 `diffusion`；加载对象的 objective 必须相同 |
| approved | 必须是 JSON `true`；这是明确来源批准记录，不是模型正确性证明 |
| license_record | 非空的原源码和权重使用依据 |

`camera` 恰好包含 `translation_scaling`（`fixed_units` 或 `legacy_maxnorm`）、`framewise`、`camera_channel`（`native`、`identity` 或 `drop`）和 `temporal_stride`（1、2、4、8、16）。必须有

```python
training_camera_sha256 == digest(CameraConfig(**camera).manifest())
```

这个字段绑定调用者声明的训练条件。真实训练检查点和其元数据是否确实采用该条件，还需实际检查点验收；不能从这个等式推出训练或模型层消费已经通过。

后端必须实现 `generate(image, camera, native_frames, seed)`、`snapshot()`、`restore(state)`，可选 `reset(seed=...)` 和资源接口。`generate` 返回 `{"frames": [...], "camera_payload_sha256": ...}`；receipt 为 `numeric.payload_hash(camera)`。相机 payload 仅有 poses、intrinsics、selected event indices、camera manifest 和其 hash，绝不传 `CameraBatch.native_queries_xy`，尤其不能向 `identity/drop` 暗送第二个数值通道。receipt 证明传输对象一致，实际模型层消费仍需真实后端审计。

离线每调用 32 个新槽位，要求原生 33 帧；在线每调用 16 个新槽位，要求原生 17 帧。原生锚帧不计新输出。新帧必须是 480×832 的无损 PIL RGB；返回评分前按 camera manifest 的精确 2×2 整数平均转换到 240×416 uint8 RGB，不经视频压缩。后续调用使用自身上次已提交的规范末帧和该帧的 global q。末帧缺失会终止该流的后续模型生成，保留缺失槽位，不补写首次参考。

snapshot 包括规范 anchor RGB、位置、调用索引、RNG、终止标志和后端可写状态。恢复分支不回退对象生命周期内的实际资源累计数；共享冻结权重无需拷贝。缺少可信后端累计计数或可测状态时，资源完整性必须保留未知，不能由有效 PNG 输出推为完整成本。

suite 的冻结 model manifest 保存 config、实际源/权重逐文件表、backend manifest、camera manifest、推理 camera hash 和声明的训练 camera hash。每次执行重新检查本地实际文件；历史 PNG 重放则检查保留的 manifest 内部绑定，不导入原后端。缺少原文件时新执行失败，已保留 PNG 的独立 replay 不因此重新加载模型。

统一入口是 `python -m rswm_inria.suite plan/run/verify`。批准的模型 JSON 中设置上述 adapter 和字段即可走实际 factory；真实 pilot 仍必须满足计划中固定 dev 2 worlds × 1 origin × 四族 × 1 seed 的 8 包 / 704 新输出，以及来源、权重、设备、状态和实际成本门禁。
