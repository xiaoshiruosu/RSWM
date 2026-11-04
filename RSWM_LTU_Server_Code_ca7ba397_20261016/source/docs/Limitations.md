# 当前验证与解释边界

工程实现、合成行为测试、真实数据、真实模型、GPU、集成和发布分别验收。源码入口存在、安装包能运行、Oracle 误差为零或模拟 backend 通过，都不能把未执行的真实验收提升为 PASS。当前阶段证据和对应 Git/CI 身份以 `evidence/build-closure/` 的报告、命令与 receipt 为准；不能用旧 commit 的绿色 CI 证明当前工作树。

## 真实输入与实验

| 项目 | 当前限制 |
| --- | --- |
| 原始 Inria TIFF / core120 / smoke12 | 本轮环境未提供真实来源文件；真实下载、逐文件物化、完整 Oracle/Copy/StaticNoiseMap 矩阵和正式模型实验仍 NOT_RUN/BLOCKED |
| LEVIR100-LR64 | 三个 pinned test Parquet 实物未提供；1024 runner 的合成通过不能代替 348 行 footer、100 行先选后读与实际 image1 验收 |
| 预训练生成权重 | 未提供；Diffusers/Wan 路径、地图策略和 history 机制有实际执行实现，但真实权重恢复与遥感生成性能 NOT_RUN |
| 原数值 backbone / 训练 | 原 approved source 和 base/checkpoint 权重缺失；训练编排及数值包装不能代替真实 flow/diffusion loss、LoRA 更新、检查点恢复和模型层消费验收 |
| GPU | 当前没有可用实际 GPU；同步/峰值 mock 与 CPU 结果不能成为真实时间、显存、成本或吞吐结论 |
| motion / quality | 真实 calibration、独立 validation、冻结 feature 权重/接收范围缺失时保持诊断状态；正式 test 由完整真实 release receipt 门禁阻止 |
| 公开安全评测 | 当前 Linux namespace/bubblewrap 权限探针 BLOCKED；真实隔离检查尚未执行完成；不开放未验证的真实模型/numeric/GPU profile |

以上阻塞不阻止独立源码、边界测试、CLI、打包和故障证据工作。它们也不能通过换成“看起来真实”的合成图、随机权重或薄 PASS 文件消除。

## 数据与地理

核心的世界构建、地域角色和任务集合冻结，不为通过验收更换源文件、重采样像素或修补失败参考。audit 可以验证来源身份、像素、尺寸、I0、labels 和公共字段，但源 tile/world 身份不等于经坐标证明的地理独立。世界级 bootstrap 也不是独立地区数量证明。

LEVIR 原生 GSD 和有限地图范围是独立补充；来源提到 Texas/Austin 等地区但缺少可靠坐标时，不能断言与 Inria Austin 完全不重叠。不同数据集名称不构成严格 OOD 或全球地理无交叉证明。FAIR1M/OSCD 有限预算探针不属于新增正式基准；header 检查不代表投影/GSD、像素、许可或地理已验收。

静态正射 RGB 序列的索引是空间查询顺序，不是现实时间。未知区域可有多种生成解；模型不需要逐像素复现隐藏未来。I0 保真和首次生成历史保持分不能直接解释为真实未来场景准确率。

## 指标与模型

固定噪声地图可以具有完美历史一致性，却没有高质量遥感生成证据。copy/random/noise/Oracle 仅为诊断；declared `real_generator` capability 是配置用途，不是实景效果验证。

同起点 exact/offset 共享 I0 和 prefix，但 offset 仍改变几何。差值不能单独作为纯记忆因果效果。F7 分支恢复测试验证允许历史与状态属性，不证明某内部表示已学到空间记忆。

相机条件是虚拟平面纯平移、固定 intrinsics 和显式时序采样，不是实际 Inria 外参。temporal stride 可能遗漏块内转向；真实 VAE 对齐必须另验。相机 payload hash receipt 只证明传入对象一致；identity/drop 没有旁路 trace，也仍不能从 receipt 推出真实网络层是否使用相机。

图像运动估计有纹理、周期歧义、低对比、搜索范围与形变限制。不可测为弃权/null；RGB 低误差不自动表示正确运动。校准/验证接收条件不能事后根据 test 修改。

质量结果仅解释为所选冻结特征、几何采样及完整样本下的未知区域分布诊断。first-commit mosaic 含时序拼接缝，不能当作单帧美观评分。MMD-U 可为负；样本不足、缺失覆盖和共同子集缩小必须保留。当前不是 Inception KID，不证明建筑、道路或语义布局正确。

## 测量与信任

资源 v3 的 complete 只覆盖该记录所声明的适用量和范围。GPU 时间为同步 host wall；RSS 为进程生命周期高水位；动态状态为显式支持 owner；磁盘为采样时逻辑长度。未知 backend cache、Tensor/object/memmap 或计数器读取缺口必须保留。不能由提交帧数推断隐藏原生工作，也不能以部分峰值或部分和冒充完整值。

`cooperative_subprocess` 测试已执行真实子进程与有界 IPC、超时/清理、snapshot 和 worker 资源采样；该 profile 不隔离私有文件。`isolated_worker` 不可用时明确 BLOCKED，不能自动 fallback 后仍称隔离。worker 自报资源属于合作式观察，不是 hostile-code attestation。当前 `protocol_security_verified=false` 必须保留；安全发布需要对应部署的真实 OS 权限门和攻击测试。

本地 hash 链、manifest、receipt、exposure ledger 和 approval 字段验证内部绑定与调用者保留的历史，不提供外部可信时间戳、全局未曝光证明或原始来源真实性保证。源码/数据/权重 rights record 也需要负责人真实确定，不能因文件可读就推定权利已清理。

## 复现与发布

测试里的 mock、toy TorchScript、合成纹理和模拟 counter epoch 均应在 artifact 明确标记 diagnostic。可重放 PNG 不等于可从零恢复真实模型。运行环境、依赖版本、源树/权重、模型/指标/seed/retry/预算身份和每次命令须随证据保留。

v2/v3 retry 语义、source fingerprint 或冻结 camera/feature 配置不匹配时应拒绝或显式按旧环境分析，不能为读旧 artifact 关闭验证。全量测试、wheel/sdist 和干净环境安装以实际当前提交证据签收；未完成真实门禁时，完整工程交付仍不能称论文实验或公开评测平台验收完成。代码、数据和权重许可分别处理，参见 [NOTICE.md](../NOTICE.md)；不得随包发布私有源数据、权重、token 或大批生成影像。
