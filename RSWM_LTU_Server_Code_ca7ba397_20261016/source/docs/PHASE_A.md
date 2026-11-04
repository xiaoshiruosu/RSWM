# Phase A: suite、资源与聚合

本模块支持已构建的 Inria F2/F3/F5/F7 合作式进程内评测。Core 与 same-start paired 扩展分别建计划；LR64 仍须独立 runner/scorer 适配，当前拒绝其任务。不会下载数据或模型，不把诊断模型当作真实生成系统。

## 1. 固定模型配置

模型文件是 JSON 列表。诊断示例：

```json
[
  {"model_id":"copy", "adapter":"copy", "device":"cpu", "training_seed":null},
  {"model_id":"oracle", "adapter":"oracle", "device":"cpu", "training_seed":null},
  {"model_id":"lazy-noise", "adapter":"lazy-map", "device":"cpu", "diagnostic_noise":true}
]
```

支持 copy/random/noise-map/oracle/lazy-map/eager-map/wan-i2v。真实地图补全或 Wan 必须提供 `model_dir` 本地权重、`device` 和 `steps`；模型目录中的全部文件进入哈希清单。Oracle单列为评测器诊断，不参加单观测生成排名。训练seed单列，`--seeds`只表示推理seed。

## 2. 冻结计划

对于 hf_materialize.py 导出的合并 private/index.json：

```bash
python -m rswm_inria.suite plan \
  --task-index build/inria/private/index.json \
  --models configs/phasea-models.json \
  --seeds 0 1 2 --role dev \
  --code-commit "$(git rev-parse HEAD)" \
  --out runs/dev-plan.json
```

旧 cli tasks/worlds 的分离索引也支持：额外提供 `--world-index data/worlds/index.json --world-dir data/worlds`。paired 的 index 使用分离world索引，其study保留原名字，不能与core混榜。

计划固定完整expected集合、源manifest哈希、任务、标签、世界文件/像素、I0像素、模型清单、RGB指标配置、源commit及执行模块源码哈希。哈希提供完整性关联，不证明原始数据真实性。输入路径按当前机器解析为绝对路径，因此迁移机器后需要重新冻结计划，不能编辑旧计划后直接resume。

计划内保存几何缺失报告：用原ledger对计划位置计算面积，没有查询真实未来像素或生成输出。模型加载失败、输入文件后来损坏、进程崩溃也能保留完整输出/面积分母。构建计划可能耗时，按任务缓存几何结果后再展开方法和seed。

## 3. 执行、恢复与重试

```bash
python -m rswm_inria.suite run --plan runs/dev-plan.json --out runs/dev-suite
python -m rswm_inria.suite run --plan runs/dev-plan.json --out runs/dev-suite --resume
python -m rswm_inria.suite run --plan runs/dev-plan.json --out runs/dev-suite --resume --retry-failed
```

suite执行入口使用POSIX文件锁，适用于Linux服务器。锁由内核释放，进程突然退出也不会遗留阻塞；禁止手动删除正在使用的锁文件。不同run目录可以独立运行。

新计划使用 `rswm-suite-v3`：默认 `first_attempt_only`，每run最多一次attempt。首轮失败或缺失不会被后来成功替代。恢复必须在计划冻结时设置 `--max-attempts-per-run 2` 等明确预算及 `--allowed-retry-statuses`，之后才可执行 `--resume --retry-failed`；预算及允许状态进入plan hash。完整执行的run不得为了更好的分数重复尝试。

`--resume`保留已有attempt，不凭PNG续接隐藏模型状态。SIGTERM/Ctrl-C记录INTERRUPTED，硬退出恢复记录CRASHED；每次恢复建新的完整attempt，全部实际成本保留。first_attempt与recovery_attempt分别导出，主表取首轮，恢复表仅表示operational recovery。旧v2仍显式保持legacy latest policy，且保持其源码身份检查，不能把历史计划静默重标为v3或跨源码绕过fingerprint。

`run`与`verify`只有执行、独立提交和资源计量范围均完整才退出0。单run执行、资源或提交问题退出1；根配置/plan身份错误退出2。`inspect`实际检查计划及运行状态；`verify`独立重放，而不是仅检查JSON字段。

产物：

- plan.json、expected.jsonl：固定计划与运行表。
- runs/<run_id>/attempt-NNN/submission/：原runner日志、task、I0、原始PNG和summary。
- 每attempt的started.json、output-slots.json、outcome.json、resources.json及成功回放的replayed.json。
- heads/与receipts.jsonl：run目录之外的journal收据；请独立保留副本以防整个suite同时被替换。
- replay-status.jsonl：每个计划run的最新attempt索引。

外部收据没有可信时间戳，本工具不是安全沙箱。未完整运行的几何缺失报告只作为分母，不标记为已成功回放。

## 4. 资源语义

分别记录输入校验、模型加载、初始化、每次generate、输出转换、snapshot、restore和replay。资源采集器与实际操作累计计数位于adapter快照之外。F7恢复逻辑状态不会回退真实开销；eager初始化补全也计入。

- attempt_wall_seconds覆盖collector创建到最终采样，包含输入校验、评分、IO与回放，排除随后写入的资源/收据文件。
- CPU墙钟与GPU同步后的host墙钟分开；后者不是CUDA kernel时间。
- 显存峰值是collector创建后进程/device分配器的高水位，包含共享权重和其他驻留内容。
- RSS转换为bytes；仍是进程生命周期高水位，包含评测器、依赖和先前任务，不能冒充单任务增量RSS。
- 动态状态通过显式resource_state读取NumPy buffer，别名去重，不deepcopy权重；不包含Python对象开销。未支持adapter保留null及原因。
- Wan累计原生帧包括上下文帧，额外帧单列；实际生成计数不随F3 reset/F7 restore回退。
- disk_bytes是采样时run文件大小，排除模型/源数据和随后写入的资源/收据文件。

无GPU和真实权重的测试只验证软件连接，不能算真实GPU成本验证。

## 5. 独立回放与导出

```bash
python -m rswm_inria.export_tables \
  --plan runs/dev-plan.json --suite runs/dev-suite \
  --out runs/dev-tables --bootstrap 2000 --seed 0
```

导出前按冻结策略独立回放主attempt，并单列首轮和受限恢复，检查固定I0及外部run收据。篡改或缺失的提交转为FAILED_REPLAY，完整计划面积保留。派生CSV输出目录必须为空。

输出原CSV/JSON并增加run-completeness、first_attempt/recovery_attempt、seed覆盖、failure/integrity摘要、figure-data和tables/main_control.tex、main_persistence.tex、coverage.tex、resources.tex。LaTeX转义标识、null为--；完整成本和部分计量明确分列。provenance绑定所有attempt结果/资源/收据与派生表hash。按study/model/family/protocol/variant/stream/metric分组：任务、起点和seed先在世界内等权，再城市内世界等权，最后城市等权。F3 small/large分列；F7共同前缀只在main计一次，branch与paired补充视图不重复计输出数。

resources记录主分析attempt；resource_attempts与resource_totals保留全部已启动attempt成本，并保留expected-run coverage。`rswm-resource-v3`每字段记录eligible/measured units、适用性、完整值和部分和/观测峰值：存在未知范围时full值为null，partial单列，不重新解释为完整成本。旧v2标量不能推出计量完整。CPU的GPU字段not_applicable且null。RSS仍是进程生命周期高水位。

NumPy owning数组与bytes/bytearray/memoryview按根owner去重，unsupported storage保留未知。累计计数器读取失败不当0；稳定epoch允许可信累计终值恢复总量，但阶段归属仍披露缺口，F7恢复不回退真实成本。请求槽位和实际model调用分开。

条件误差只平均可测单元；覆盖与缺失敏感性界保留全部适用计划单元。整城无可测值时等城市主误差为null，不删除该城。保留计划/有效像素总数与补充pixel_pooled误差。结构性不适用为null；全缺失RGB界为[0,1]，该界不是置信区间，不适用于MMD。

两方法先匹配同task_id/起点/推理seed的共同可测单元，再形成世界级值进入配对统计，另存每世界的共同单元覆盖，按城市分层bootstrap。区间仅表示给定城市下源瓦片重采样敏感性，地理独立性仍未验证。真实运动校准、未知区域特征质量、真实模型实验、LR64与安全隔离属于后续独立验收。
