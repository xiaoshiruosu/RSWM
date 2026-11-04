# 模型适配与公共输入合同

适配器执行入口是 `python -m rswm_inria.suite`，模型配置由 `registry.validate_config` 校验，实际 factory 加载相应实现。配置不能记录后忽略：未知字段、无效设备、未提供本地检查点、噪声路径带 `prompt/steps` 等情况会明确拒绝。所有模型与模型文件清单进入冻结 plan；名称相同不代表相同权重或实验身份。

## 公共接口

合作式模型实现 `reset(initial, spec, seed)`、`generate(queries)`、`snapshot()`、`restore(state)`。`initial` 是独立复制的 240×416×3 uint8 RGB I0；`spec` 是公共尺寸、坐标方向、合法范围和协议规范。`generate` 仅获得当前允许的查询：F2/F3 离线 32 个，F5/F7 在线每次 16 个。返回相同数量的规范图像或明确缺失槽位；错误尺寸、dtype、批长度或异常会留下不可补写的失败槽位。

模型不应读取隐藏 world、真实未来 crop、私有标签、完整未来轨迹或其他方法输出。合作式进程内运行依赖模型守约，传参白名单本身不是安全沙箱。Oracle 由 evaluator 单独构造，可以读取隐藏世界，仅用于仪器对照；registry 和公共 worker 拒绝把它当第三方模型。

snapshot 保存逻辑地图/历史、RNG、锚帧和后端可写状态。F7 的两个后缀从同一前缀状态独立恢复；恢复不能回退已经发生的资源成本。可选 `resource_state()` 声明可测动态状态，`resource_counters()` 声明可信累计工作；缺少或失败的测量不会从有效 PNG 推出零成本或完整成本。

## 已接通的 adapter

| adapter | 实际实现与用途 | 额外配置 |
| --- | --- | --- |
| `copy` / `random` | I0 复制 / 当前请求随机 RGB；诊断 | 公共字段 |
| `noise-map` | 固定空间噪声地图；诊断一致性 | 公共字段 |
| `oracle` | evaluator-only 隐藏世界裁剪 | 公共字段；不能公开 worker 化 |
| `lazy-map` / `eager-map` | 查询时补全 / 初始化时补全的 WindowMap | 本地 `model_dir`，或明确 `diagnostic_noise=true` |
| `raster` / `frontier` / `lazy` | 经审计的不同补全顺序和地图读取策略 | 同上；记录 operation trace 与权威累计计数 |
| `wan-i2v` | 本地 Wan 图生视频包装；公共数值查询被忽略 | 必须提供本地 `model_dir`；无噪声 fallback |
| `numeric-camera` | approved 本地数值后端的 LastFrame 包装 | 完整 backend/camera 合同，见下文 |

公共字段为 `model_id`、`adapter`、`device` 和可选 `training_seed`。生成型地图及 Wan 可设置实际消费的 `steps/prompt`。真实检查点目录必须非空且无 symlink；执行不会自动联网取得权重。`diagnostic_noise` 与真实 `model_dir` 互斥。capability 中的 `real_model_validation/real_gpu_validation=NOT_RUN` 是独立验证状态，不能由 factory 可调用改成 PASS。

最小诊断配置示例：

```json
[
  {"model_id":"copy","adapter":"copy","device":"cpu"},
  {"model_id":"frontier-noise","adapter":"frontier","device":"cpu","diagnostic_noise":true}
]
```

此配置没有真实生成模型或权重。查看实际许可、来源、原 backend、权重恢复和真实开发集 pilot 证据后，才可以冻结对应真实实验。正式 test 的执行另受 `experiments.py` 的数据、预算、每模型 pilot、已验证仪器和 rights decision 门禁控制。

## LastFrame / FrameCache / MapAssisted 机制

独立 paired 机制执行器使用 `HistoryCompleterAdapter`，policy 恰好包含：

```json
{"schema":"rswm-history-policy-v1","mechanism":"frame_cache","capacity":3,"protect_initial":true,"protect_history":true}
```

`last_frame` 容量必须为 1，仅保留最近成功提交帧；`frame_cache` 从有界缓存选择视口重叠最大的帧，相同重叠按最近顺序取帧；`map_assisted` 读取自身已生成空间地图。失败帧不会进入缓存。`protect_initial` 硬保护真实 I0 区域；`protect_history` 独立保护被读取的生成历史，不能借此增加真实观测。检索帧位置、容量、命中、读取像素、补全次数、成功提交与淘汰均有实际 trace/counter。

paired plan 的机制 model config 使用 `lazy-map`，model ID 以 `-` 加 policy digest 的前 16 位结尾，`--mechanisms` 文件把该 ID 映射到 policy。同样的骨干配不同读取机制，不代表复现某论文已训练系统。独立 native LR64 使用 `history-completer` 和 config 内 `policy`；其诊断 adapter 为 `copy/random/native-noise-map/oracle`。LR64 的 CPU 诊断配置不能声称 CUDA 执行；其他模型必须明确支持 native 1024 世界。详见 [G7_EXECUTION.md](G7_EXECUTION.md)。

## 数值相机与原后端门禁

`numeric-camera` 不猜测第三方包，不替换成玩具损失或噪声。`backend_manifest` 必须明确记录原源码目录及逐文件 tree SHA、源内 `.py` entrypoint、非空本地权重目录及 tree SHA、真实 `flow_matching/diffusion` objective、`approved=true` 与非空 `license_record`。源目录 entrypoint 必须实际实现 `load_backend(weights_dir=..., device=...)`。对象必须提供 `generate(image, camera, native_frames, seed)`、`snapshot/restore`，以及与 manifest 一致的 objective。新执行重新检查源和权重；历史 PNG replay 不重新导入模型。

`camera` 恰好包含 `translation_scaling`（`fixed_units/legacy_maxnorm`）、`framewise`、`camera_channel`（`native/identity/drop`）和 `temporal_stride`（1/2/4/8/16）。声明的 `training_camera_sha256` 必须等于该完整 CameraConfig manifest 的 digest；这项等式不能代替实际检查点训练元数据验收。

固定单位采用规范视口对角线缩放；legacy max-norm 只计算当前允许块及已有锚位置，在线不得偷读整条未来轨迹取得尺度。采样先于 framewise delta。相机是虚拟平面纯平移 c2w 条件，不是恢复出的 Inria 外参。后端 payload 只含 poses、intrinsics、选中事件和相机 manifest/hash；不传 `CameraBatch.native_queries_xy`。identity/drop 不接受旁路数值 trace。

规范 I0/末帧进入原生 480×832 时使用已记录的 Pillow bilinear 2× 转换；新原生帧必须为无损 PIL RGB。离线 32 新槽位要求 33 原生帧，在线 16 新槽位要求 17 原生帧；原生首锚帧不计新输出。回到规范评分采用精确 2×2 uint16 求和、`(sum+2)//4`，得到 240×416 uint8 RGB，不经有损视频。后续调用只接自身上次提交的规范末帧和 global q；末帧缺失终止该流后续模型调用，保留槽位，不补写参考。

后端返回的 camera payload receipt 验证传输对象身份，不能证明神经网络某层确实消费了条件。真实 VAE 时序步幅、模型层消费、检查点恢复、生成和 GPU 成本仍需实际后端证据。原源码/权重当前缺失时保持 BLOCKED/NOT_RUN。字段完整说明见 [NUMERIC_BACKEND_CONTRACT.md](NUMERIC_BACKEND_CONTRACT.md)。

## Worker 模式

`cooperative_inprocess`、`cooperative_subprocess` 和 `isolated_worker` 是不同 profile。公共 IPC 限定版本、请求 ID、数据长度/shape/dtype 和当前批查询；snapshot token 保留在 worker 内，worker 资源由实际 worker collector 采样。合作式子进程有传输与进程清理检查，但不提供私有文件隔离证明。

目前受限 worker 是需实际 OS 验证的 Linux bubblewrap CPU 诊断 profile；真实模型、numeric backend 和 GPU profile 在前置检查明确 BLOCKED。namespace 不可用时不会退回合作式 profile。实际使用前执行 `python -m rswm_inria.worker_client --probe-isolation`，并检查对应部署的权限证据；当前环境不能把 BLOCKED 探针或 IPC 单测写成安全验收通过。
