# RSWM 持续执行的一次性输入与授权表

本表不填写或收集密码、私钥、token。会话授权使用正常服务器/IDE/终端机制。下列空项不是批准，也不能由Codex填成虚假实值。

## A. 最先需要的四组输入

| 内容 | 用户/师兄提供 | Codex在实际机器核对 |
|---|---|---|
| 执行环境 |真实可用会话或执行方式；允许读取根；持久输出卷|解释器、GPU/显存、RAM、磁盘、已有作业、依赖|
| 固定数据 |kitsap31/19路径，或新可读Part03，或固定revision精确下载许可|bytes/SHA、实际TIFF解码和当前物化审计|
| 模型 |实际相机模型权重根及来源/使用决定；base/restored_lora|全部专家、VAE、T5、tokenizer及实际配置所需LoRA|
| 运行范围 |本轮可执行的任务范围、有限墙钟和输出/传输预算|估算成本、冻结技术配置、记录实际用量|

可直接填写：

```text
server_execution_method = <已有可用会话/由师兄执行的终端>
allowed_read_roots = <具体根>
persistent_output_root = <具体持久路径>
fixed_tiff_paths_or_recovery_locator = <实际路径或新入口>
allow_exact_revision_download = <明确是/否及范围>
base_camera_weights_root = <实际路径>
model_mode = <base / restored_lora>
weight_source_and_use_decision = <实际依据和决定>
allowed_initial_milestones = <例如M00-M03，非自动批准后续训练>
max_transfer_bytes = <批准值>
max_total_wall_seconds = <批准值>
max_output_bytes = <批准值>
retry_budget = <批准次数/时长>
```

技术参数可授权Codex在现有设计和指定候选内提出一套配置，再一次性确认。不存在的旧LoRA仅在明确恢复该模型时才是必需；base预训练推理可以独立推进。

## B. 正式训练和测量前补充

实际补全器与其使用依据；正式预训练特征编码器/层/权重/预处理决定；当前core120持久数据；训练初始化、更新数、batch/累积、precision、LoRA策略、验证和checkpoint预算；正式模型/seed矩阵及机制/LEVIR范围。

采用分阶段额度授权，不允许用“直到完成”推定无限算力、磁盘和下载额度。预算耗尽时Codex保留实际成本并申请具体差额，不得删失败或自动延长。

## C. 已知无需再次索取的内容

当前软件实现和N06-E已具备；原安全Wan准备/部署入口存在。DDC证据和分发已提供。不要让用户重新提供能从现有源码读出的函数名、已知hash或测试统计。

尚缺实际会话或输入时，Codex只能保存依赖阻断并继续独立任务，不能把本表填满当作真实执行完成。
