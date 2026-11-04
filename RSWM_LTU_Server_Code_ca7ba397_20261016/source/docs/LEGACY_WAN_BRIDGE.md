# 原 Wan / LingBot 后端接入

本接入使用用户交付的原源码 ZIP，SHA256 为 `2d079e768bdf66a06cc7cb8891f408c7adbcb1985424b801fa962da97400ce73`。选定 `lora_remotesense/new/wan`，不把顶层另一份 `wan` 混入同一进程。部署副本保留原 LICENSE / NOTICE、原文件哈希和派生修改映射。原 ZIP 保持不变，模型权重和影像不进入分发包。

准备器排除未被这些模型/采样函数引用的 `wan/remotesensebenchmark/`。该旧工具子目录含 provider 配置，不能随模型部署携带；来源记录只列排除目录、文件数和理由，不保存其中的配置值。原输入字节保持不变。

## 运行输入

部署前确定实际 Python/CUDA 环境、基础权重根、持久源码和输出目录，以及使用依据和推理预算。旧源码的 A800 环境记录不代表新服务器已经验收；CPU Torch 软件测试也不证明 CUDA、FlashAttention 或真实模型兼容。`.[generate]` 是通用依赖集合，不包含原源码的全部运行依赖；目标解释器须根据选定源码实际 imports 安装并锁定 torchvision、tqdm、einops、peft、diffusers、easydict、ftfy、regex、transformers、imageio 等实际需要的版本。CPU 软件控制版本不能直接当成 CUDA 环境推荐。

原 attention 实现动态读取 `RSWM_ATTENTION_BACKEND`。进程启动前冻结其请求值，记录实际 attention kernel、相关版本、precision 与确定性设置，并在目标设备实测；缺库或切换实现不能沿用先前的运行身份。当前训练要求两专家常驻同一设备，不支持凭多卡数量合并显存、optimizer offload 或自动分片。

源树中的 `backend_config.json` 锁定 `mode`、`config_key`、`sampling`、`lora_subdir` 和可选 `training`。sampling 明确记录固定 prompt、negative prompt、steps、shift、low/high guidance、expert boundary、param/T5 dtype 和两个 offload 开关。没有从模型名、环境变量或旧 epoch 数自动批准这些设置。

```bash
# 从同一核验提交或 sdist 获取本脚本；wheel 不包含仓库 scripts 目录。
# 先实际验收原 ZIP 并形成新的部署目录；不复制权重，也不执行模型。
PYTHONPATH=src python scripts/prepare_legacy_wan.py \
  --source-zip inputs/legacy-source.zip --out build/wan-source

# 确定运行设置后，在另一个新目录把明确配置一起锁入源树。
PYTHONPATH=src python scripts/prepare_legacy_wan.py \
  --source-zip inputs/legacy-source.zip --backend-config inputs/wan-runtime.json \
  --out build/wan-source-configured
```

未传配置的输出是 `PREPARED_SOURCE_ONLY`，不能加载模型。配置后的 `backend.py` 是现有 registry 的 entrypoint；使用 `training.locked_tree` 求源树与实际权重根的 SHA，不手填历史 hash。`legacy_wan.inspect_backend(source_dir=..., weights_dir=...)` 可在无 Torch 环境列出逐文件组件、checkpoint index 所需 shards 和缺口。不同部署源树或运行配置在同一进程中拒绝混用，切换时启动新进程；同一 source/config 下的 LegacyNorm / FixedUnits 相机对照保持兼容。

- `mode=base` 调用原 `evaluate_baseline.load_base_model`，加载 T5、tokenizer、VAE、low/high experts，不要求 LoRA。
- 原 `rswm-legacy-wan-backend-v1` 的 `mode=restored_lora` 调用原 `inference.load_lora_model`，另需基础权重根内 `lora_subdir` 的 low_noise / high_noise 适配权重和匹配的训练条件记录。缺任一专家失败。
- 新 LoRA 训练从已批准基础权重初始化；训练配置、参数策略和预算另行冻结，不能把旧恢复检查点当成新训练完成证明。

已有 registry 配置不增加 prompt 或私有查询字段。它仍通过 `training.load_approved_backend` 读取明确源文件的 `load_backend(weights_dir=..., device=...)`。生产者不会生成 `approved=true`；现有 backend manifest 的批准和 license_record 必须来自实际使用决定。

## 相机和状态

RSWM 路径向原 `generate_video` 提供互斥的 prepared-camera 输入。它直接使用已选事件的 pose / intrinsics 构造原模型 Plücker 条件，不再插值、取相对位姿、差分或最大范数归一化。原 action_path 路径保留用于历史比较，两入口不可同时提供。native / identity / drop 使用相同图像、文本和采样设置，drop 不带额外 q。

当前选定 VAE stride 为 `(4,8,8)`，patch 为 `(1,2,2)`。online 要求 17 原生帧、5 个相机事件；offline 要求 33 原生帧、9 个相机事件。事件索引必须为 `0,4,…,16/32`，原生输出必须为 480×832 RGB，不允许截尾或重排转弯事件。已有 stride4 的信息丢失限制仍成立。

prepared 路径直接使用上述整数网格。实际原函数 CPU 接线测试发现旧面积公式的浮点 floor 可把 480 算成 464；派生修改只纠正此新路径，旧 action_path 路径的公式和行为原样保留。加载器另要求实际低/高模型返回完整 loading info；missing、unexpected、mismatched keys 或 error messages 均失败，不接受缺相机权重后随机初始化。恢复 LoRA 在原 merge 前核对保留 checkpoint 的逐键集合、shape、有限性与实际加载值，不能将 PEFT 的缺键警告视作成功。tokenizer 只读本地资产，T5/VAE 沿用原严格 state_dict 加载。

后端快照恢复输出相关可变状态与 Python / NumPy / Torch / CUDA RNG；冻结模型权重不作为 F7 后缀状态复制。模型设备、VAE 可写缓存等状态不能靠共享可写对象伪装独立分支。累计调用和实际工作计数在 restore 后继续增长。合作式运行保留 `security_verified=false`；调用返回后检查墙钟不等于硬抢占。

实际 forward hook 观察进入的 expert、camera projector、首个 block 的 injector / scale / shift 张量，记录固定选值、shape、dtype、有限性和调用数。每次成功或失败调用的观察在生命周期日志中追加，F7 restore 不删除它。suite 将该日志保存在 attempt 内 `backend-evidence.json`，绑定 model/task/source/seed/attempt 并纳入现有 artifact manifest；读回缺失、篡改或错误身份使完整性验收失败。它与 RGB 评分分开，`research_acceptance=NOT_EVALUATED`，不能凭观察日志自签科研 PASS。

## 训练与验收范围

训练桥接保留原目标 `z_t=(1-t)z+t·noise`、`target=noise-z` 和 MSE，根据真实 timestep 选择低/高噪声专家。现有 driver 独占 optimizer、AMP、梯度累积、scheduler、更新预算和 checkpoint 边界。相机处理与推理共享同一函数和 hash。批内逐样本调用避免原模型把不同样本的相机序列拼接到同一 token 轴。

正式训练仍要求当前 core120 审计绑定 train60 / 3840 clips / 126720 view events。两 dev 世界、synthetic fixture 和旧 AID/HDF5 不能替代。真实 dev 验证从已审计 PNG 读取，不能返回无输入 PASS。

默认训练桥接仍使用两个 resident expert。0.4.0 另提供显式 `block_cpu` 训练和 `sampling.block_cpu_offload=true` 推理路径：冻结骨干在 CPU，按模块搬运，LoRA 保持执行设备及优化器对象身份，保存张量的 CPU hooks 支持真实反传及原非重入检查点重算。详见 [单卡 CPU 卸载说明](SINGLE_GPU_CPU_OFFLOAD.md)。冻结参数仍以有界 CPU 内容 hash 验证，同时检查参数身份、冻结策略、活跃专家梯度与实际更新。完整 checkpoint 仍包含冻结权重；真实 RAM、磁盘、I/O、传输与 GPU 峰值需在批准设备及预算下测量。PEFT 导出不替代含 optimizer、scheduler、scaler、RNG 与 cursor 的训练恢复文件。

训练输出目录保留不可变 `training-contract.json` 及 `attempts/attempt-NNN/events.jsonl`。已开始的真实 dev forward 在执行前落盘，失败或未收到完成记录仍占累计验证预算；恢复较旧 checkpoint 不会回退这些成本。loss、optimizer、validation 与 update 的开始/完成记录保留已测墙钟；只有开始的调用标为未完成，不能补为零成本或 GPU 峰值。恢复实际 Wan 必须保留同一训练输出目录和所有 attempts；缺失、损坏或不匹配的台账拒绝恢复。旧三字段 receipt 的通用读取兼容性不意味着缺乏成本台账的旧 Wan 训练目录可以通过此检查。

同一训练输出目录只允许一个 driver 持有 `training-coordinator.lock`。CLI 在加载模型前取得锁，Python driver 在读取尝试成本前取得锁；竞争进程立即拒绝。锁覆盖失败退出，进程崩溃后由操作系统释放，已有尝试日志与未完成调用费用继续保留。不要删除活动锁文件来启动第二个 driver。

checkpoint 与其 receipt 使用排他发表。v2 receipt 绑定配置、数据与当前来源审计、相机、backend、源码、训练 seed、实际 step/cursor、两个专家的完整状态，以及固定的 checkpoint 时刻日志快照。后续日志追加不会改变该快照。已有或部分发表的路径保留，不自动覆盖；任何写入、fsync 或读回异常都不能报告成功。

## 训练到推理交付

`rswm-export-trained-adapter` 只接受原 driver 的完整 checkpoint 和 `rswm-training-checkpoint-v2` 强收据，不能把 `update-XXXXXXXX.pt` 改名当成 PEFT adapter。公共 CLI 继续要求完整 core120/train60 数据、实际 dev、原 source/base weights 和原使用批准。以下路径是操作示例，运行前替换为实际已审计资产与批准配置；训练和导出均受既定设备、时长、磁盘及更新预算约束。

```bash
rswm-train --config configs/wan-train.json \
  --index build/core120/private/index.json --clips build/core120/private/training-clips.json \
  --root build/core120/private/worlds --source-audit runs/core120-audit.json \
  --out runs/wan-train

# 如需恢复，在同一 out 下使用其 checkpoint，并保留全部 attempt 台账。
rswm-train --config configs/wan-train.json \
  --index build/core120/private/index.json --clips build/core120/private/training-clips.json \
  --root build/core120/private/worlds --source-audit runs/core120-audit.json \
  --out runs/wan-train --resume runs/wan-train/update-00000010.pt

rswm-export-trained-adapter --checkpoint runs/wan-train/update-00000010.pt \
  --config configs/wan-train.json \
  --index build/core120/private/index.json --clips build/core120/private/training-clips.json \
  --world-root build/core120/private/worlds --source-audit runs/core120-audit.json \
  --inference-config configs/wan-inference.json \
  --out deliveries/wan-adapter-10 --deployment-out deliveries/wan-inference-10
```

`wan-train.json` 的 backend 源树保留非空 `training`、相同 policy 和原 `new_lora` 或 `restored_lora` 初始化语义。输入 `wan-inference.json` 是明确的 v1 `mode=base`、`lora_subdir=null`、`training=null` 配置，用于锁定新推理采样设置；它不表示导出后退回基础模型。输出源树自动形成显式 v2 `mode=restored_lora`、`training=null`，由独立 `adapter_release` 绑定受训 PEFT 文件。原 v1 base 和已有恢复 LoRA 路径保持原意。

两个输出目录均须全新，彼此及原 source、weights、训练目录互不嵌套。release 保存 `low_noise/`、`high_noise/` 的实际 `adapter_config.json` / `adapter_model.safetensors`，以及原 checkpoint 的身份、receipt、训练配置、来源、seed/step/cursor、完整专家内容身份和逐文件 SHA 的关联。原 checkpoint 及基础权重不复制到 release；原文件字节不改写。新推理源树仅复制已锁定安全源文件并替换 backend 配置，拒绝旧 provider 子树。release 引用使用明确的绝对根路径，迁移后必须据实际位置重新冻结引用和新源树身份，不能沿用不匹配的旧 manifest。

导出严格检查完整键集合、shape、dtype 和有限数；冻结骨干与 buffers 必须匹配实际基础专家。仅支持既定纯 LoRA policy，其他可训练参数、bias、modules_to_save 或不同缩放规则拒绝导出，不能静默丢弃。实际 PEFT 序列化后逐键读回，所有训练参数均须被保留。`release.json` 最后发表；loader 检查其绑定、全部保留文件和实际 tensor 内容，缺失或篡改不能降级到旧恢复路径。加载两个专家时先核对完整未合并状态及实际 PEFT 配置，再执行原 merge。

CLI 成功返回 `EXPORTED_TRAINED_PEFT_SOURCE_LINKED_NOT_MODEL_VALIDATED`，列出新 `source_dir` / `source_tree_sha256`、原 `weights_dir` / `weights_tree_sha256`、entrypoint/objective、训练 seed、camera 与 `adapter_release`。它不生成 `approved=true` 或许可决定。负责人须据实际使用依据为新 backend manifest 明确批准，在 `numeric-camera` 模型配置中使用同一 release 引用、显式 `training_seed`、原 camera 和 `training_camera_sha256`。registry 核对模型、实际新源树、基础权重、release、seed 与相机身份；不得把训练源 manifest 的旧 source hash 直接写入推理配置。

随后在新进程实际加载该配置，冻结新的 suite/experiment，执行适用来源与预算内的真实 pilot。受训模型候选 release 是来源关联记录，不能代替当前源像素审计、目标设备加载、真实层消费、8 包/704 输出或研究验收，也不是第三方签名认证。

## 软件控制与真实运行

`scripts/reproduce_trained_adapter.py` 在 CPU 使用实际 PEFT 小型对象，分别从新 LoRA 和已有 LoRA 开始，执行现有训练 driver、双专家导出、源码目录外新进程 registry 加载、固定输入前向比较，以及一包 synthetic F2 suite/PNG replay。固定比较容差在 round-trip 前保留；测试不放宽生产数据或 CUDA 门。

```bash
# 使用 .github/workflows/tests.yml 中锁定的 CPU Torch/PEFT 控制环境。
PYTHONPATH=src python scripts/reproduce_trained_adapter.py \
  --work runs/tiny-peft-process-cwd --out runs/tiny-peft-control
```

该 release 始终为 `scope=software_control`，registry purpose 为 diagnostic，实际 Wan loader 拒绝它。tiny harness 的专用检查器仅在受控调用中临时绑定并在 finally 恢复；公共训练/导出 CLI 仍要求完整真实数据。小型模型、synthetic 输出和实际 PEFT round-trip 不签收预训练 Wan、train60、真实 GPU 或正式 pilot；真实 pilot 仍为 `NOT_RUN`，不能把这些输出计入 704 分母的完成数。

软件测试覆盖真实原采样函数的参数接线和拒绝路径；昂贵组件 test doubles 的结果只记 software。真实权重加载、实际 camera injector / scale / shift 层消费、GPU 成本、真实生成和恢复重复性各自保留独立验收。payload hash 只证明传输一致。

固定数据联调沿用 `--profile pilot-dev2` 和现有独立审计。真实生成必须继续走 `rswm-suite` / `rswm-experiment`，每方法 8 包、704 计划新槽位，保留首 attempt、所有尝试成本和首次失败分母。没有权重、设备、来源决定或预算时不启动真实生成，也不保存替代噪声输出。
