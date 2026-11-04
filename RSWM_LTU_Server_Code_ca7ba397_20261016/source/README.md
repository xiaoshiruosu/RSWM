# RSWM-Inria-120

## 服务器代码交付 v0.3.0

本版按“本地完成代码、由用户上传服务器执行实验”的分工交付。代码从已核验的
`ddc1dd1868ac94cec9c3652bfec0f75f26003772` 留存源码继续，包含实际修改与新增入口；
当前没有恢复 Git 历史，构建会如实标记 `source_dirty=true`，以当前源码指纹和差异文件绑定该工程候选。
真实 GPU、模型和科研签收须在服务器执行后判断。

完整上传包包含源码、安装包、固定 Inria 清单、安全原 Wan 源树、配置模板和验证记录。
原图、模型权重和旧 provider 配置不在包内。使用顺序见
[服务器运行指南](docs/SERVER_EXECUTION.md)、[输入准备](docs/SERVER_INPUTS.md)和
[本地预训练特征导出](docs/FEATURE_EXPORT.md)。

```bash
# 先按目标服务器的 CUDA 环境安装匹配的 torch / torchvision，再安装本工程。
python -m pip install './source[dev,data,motion,features,generate]'
rswm-server init --source-root "$PWD/source" --out "$PWD/server-project"
# 编辑 server-project/project.json 与 owner/*.json 中的实际路径、决定和有限额度。
rswm-server check --project server-project/project.json --stage pilot
rswm-server run --project server-project/project.json --stage pilot
rswm-server status --project server-project/project.json
# 有前次保留现场时显式接续；不会删除失败或重置实际成本。
rswm-server run --project server-project/project.json --stage all --resume
```

`init` 只创建未批准模板；`check/plan` 不加载模型。`run` 使用 Linux POSIX 协调锁和独立
子进程串联既有执行器，保留 PID、退出码、日志、来源与阶段预算。新增的
`rswm-server-inputs` 自动派生实际源/权重与相机身份、固定 origin0 八包选择及完整训练输入。
`rswm-feature-export` 支持明确本地 DINOv2 源码与完整预训练权重的 TorchScript 候选导出；
它不会下载或签发正式指标 release。

实验门新增可选的事前 reference/comparison 角色。参考 pilot 仍要求完整资格链；
可信的比较失败保留原分母、资源和分析 `INCOMPLETE`，可作为后续实际研究结果。
正式执行终态和科研发表状态分开；所有数值指标、首次参考与冻结样本保持原合同。
详见 [实验门合同](docs/EXPERIMENT_GATE_CONTRACT.md)。

遥感 RGB 空间查询、首次提交与持久重访基准。连续静态正射影像是有限世界，序列表示查询顺序。模型仅获得 I0、公共规范及当前协议允许的查询和历史。

当前交付为 **可构建的软件工程版本，正式模型实验未验收**。Phase A 资源、retry、容错与聚合修复已完成；真实 Inria smoke12 的 12 源、12 世界、48 I0、192 任务包通过完整独立物化审计，三种诊断系统的 576 runs / 50,688 outputs 已通过完整重放与聚合验收。实景运动仪器在固定受控留出集通过规定阈值，LEVIR100 / 200 LR64 任务的数据审计通过；这些原实景运行保留其9ef源码身份。core120 历史 ac8 完整物化与 audit 已完成；后继源码仍需恢复完整资产链并重新 audit。固定 dev2 的源字节是否可读须按当前存储实际核验，历史备份元数据不能替代读回。用户原 Wan/LingBot 源码已实际 SHA/CRC 读回；原加载、采样、prepared 相机、flow-matching 训练及受训 PEFT 导出见[后端指南](docs/LEGACY_WAN_BRIDGE.md)。真实权重加载、生成/训练、批准特征、GPU 和目标 OS 隔离仍未验收。实际命令、失败记录、不可变身份与各范围见[续建交付报告](evidence/build-closure/RESUMPTION_DELIVERY_2026-09-30.md)和[构建状态](docs/BUILD_STAGE_STATUS.md)。

## 完整运行入口

按以下顺序准备和运行；每一步使用实际已核验提交的源码与安装包，新计划记录实际源码身份。表中的配置、批准、数据和预算由相应负责人提供，软件控制不会生成这些批准。

| 步骤 | 实际入口 | 所需输入与详细说明 |
| --- | --- | --- |
| 安装与软件验收 | `pip install`、`rswm-acceptance` | 下节及[复现指南](docs/ReproductionGuide.md)；标准 CPU、CPU Torch/PEFT 与真实 CUDA 分开记录 |
| 目标环境与源准备 | `scripts/prepare_legacy_wan.py`、`legacy_wan.inspect_backend` | 原 ZIP、明确运行配置、本地权重、使用依据和预算；完整原环境、attention kernel 与设备要求见[后端指南](docs/LEGACY_WAN_BRIDGE.md) |
| 数据恢复与审计 | `rswm-materialize`、`rswm-materialized-audit` | 固定 revision、manifest、源 SHA 和真实字节；见[数据合同](docs/DatasetCard.md)及[G2 指南](docs/G2_DATA_ACCEPTANCE.md) |
| 数值模型配置与真实 pilot | `rswm-suite plan`、`rswm-experiment freeze/run` | 已批准 backend/camera、当前来源审计、8 包/704 槽位和明确预算；见[后端指南](docs/LEGACY_WAN_BRIDGE.md)及[实验门合同](docs/EXPERIMENT_GATE_CONTRACT.md) |
| 正式训练与恢复 | `rswm-train` | 完整 core120/train60、真实 dev、固定更新预算和原 backend；见[后端指南](docs/LEGACY_WAN_BRIDGE.md#训练与验收范围) |
| 受训模型导出与重新加载 | `rswm-export-trained-adapter`、registry 的 `numeric-camera` | driver checkpoint、强收据、相同基础专家、两个新排他目录；见[训练到推理交付](docs/LEGACY_WAN_BRIDGE.md#训练到推理交付) |
| 正式测量 | `rswm-motion-validate`、`rswm-release-features`、`rswm-quality` | 真实仪器、冻结配置、曝光账本和实际模型输出；见[测量协议](docs/MeasurementProtocol.md)与[C05 消费说明](docs/C05_METRIC_CONSUMPTION.md) |
| 出表与最终分发 | `rswm-export`、`scripts/retain_distribution.py` | 同一冻结 suite 的保留结果；最终 HEAD 的 wheel/sdist、下载读回与源码外安装见[C05](docs/C05_METRIC_CONSUMPTION.md)及[C07](docs/C07_DISTRIBUTION_RETENTION.md) |

## 安装和 CPU 验收

```bash
python -m pip install -e '.[dev,data,motion]' build 'setuptools>=68' wheel
python -m pytest -q
python -m build --no-isolation
rswm-acceptance --out runs/cpu-acceptance
```

默认验收实际执行 synthetic Oracle 的 F2/F7，保存 192 个新输出并独立 replay、检查资源、输出 JSON/CSV/LaTeX。它是软件验收，不能代表真实图像或生成质量。输出目录必须为空；重新运行须选新目录。

CI 另有独立 `torch-cpu-controls` 作业：使用锁定的官方 CPU Torch/PEFT，运行完整测试、本地 TorchScript 特征 CLI 正常/故障控制，以及实际小型 PEFT 的训练→导出→新进程 registry 加载→synthetic suite/replay。安装与重现方法见[复现指南](docs/ReproductionGuide.md)和[后端指南](docs/LEGACY_WAN_BRIDGE.md#软件控制与真实运行)。这些控制明确标为 diagnostic / software_control，不代替批准的生成/特征权重、train60 或原 Wan 模型训练。

```bash
rswm-acceptance --profile full-build --out runs/all-gates
```

上述命令执行软件检查并记录五个实际验证门。未提供真实输入时返回非零并保留 BLOCKED，不能用跳过项换成整体 PASS。各 profile 的冻结输入见 [复现指南](docs/ReproductionGuide.md)。

## 冻结的研究合同

原始 5000² uint8 RGB TIFF → 中央 4096² → 整数 2×2 半向上平均 → 2048²；标称 0.6 m/px。视口 H240×W416，四起点不变，两轴查询 ±704。Austin/Chicago/Vienna 各 20 train；Kitsap 12 dev；Bellingham 12 calibration；Bloomington 12 validation；SFO/Innsbruck 各 12 test。

F2 32 离线；F3 small/2×large 各 32；F5 96 在线；F7 32 共同前缀加两个 64 后缀。在线每次 16 新查询。K 对真实 I0，P 对不可回写的首次提交；128²目标像素首次再入只评分一次，首次失败不能补写。F7 前缀只计一次。两座测试城市先分别汇总，再等城市平均。核心设计和冻结文件详见 [BenchmarkCard](docs/BenchmarkCard.md) 与 [BENCHMARK_DESIGN](docs/BENCHMARK_DESIGN.md)。

## 实际数据管线

```bash
rswm-inria freeze --out data/frozen/manifest.json
rswm-inria first-source --manifest data/frozen/manifest.json --inventory data/frozen/source-inventory.json --source-dir data/source --out runs/first-source
rswm-materialize inria --manifest data/frozen/manifest.json --inventory data/frozen/source-inventory.json --profile smoke12 --source-dir data/source --out build/smoke12
rswm-materialized-audit --build build/smoke12 --manifest data/frozen/manifest.json --inventory data/frozen/source-inventory.json --source-dir data/source --out runs/smoke12-audit.json
rswm-smoke12 --build build/smoke12 --manifest data/frozen/manifest.json --inventory data/frozen/source-inventory.json --source-dir data/source --code-commit "$(git rev-parse HEAD)" --out runs/smoke12-acceptance --execute
```

freeze 只冻结指定完整 revision 的 inventory/manifest。物化默认只读本地源，--download 需显式选择并受源字节预算约束；首文件验收不能替代完整 smoke12。真实 smoke12 三诊断要求 576 runs / 50,688 个新输出。新标签存于 private/tasks/labels；旧 ../labels 布局须新目录重建。格式、角色、完整几何与来源审计见 [DatasetCard](docs/DatasetCard.md) 和 [G2 指南](docs/G2_DATA_ACCEPTANCE.md)。

固定两世界开发联调可使用同一物化命令的 `--profile pilot-dev2`，仍提供完整 core120 父 manifest 与 inventory。它只接受原 revision 的 `kitsap31.tif` / `kitsap19.tif` 及其固定 SHA，构建 2 世界、8 I0、32 任务包和零训练 clips；现有独立审计器核对实际源和全部产物。此范围可支持 origin 0 的 8 包 / 704 槽位 pilot 来源检查，不能替代 smoke12、core120 或真实生成验收。没有源字节时不形成数据 PASS。

## 固定计划、回放与出表

```bash
rswm-suite plan --task-index build/smoke12/private/index.json --models configs/smoke12-diagnostics.json --seeds 0 --role dev --code-commit "$(git rev-parse HEAD)" --out runs/dev-plan.json
rswm-suite run --plan runs/dev-plan.json --out runs/dev-suite
rswm-suite verify --plan runs/dev-plan.json --suite runs/dev-suite --out runs/dev-verified.json
rswm-export --plan runs/dev-plan.json --suite runs/dev-suite --out runs/dev-tables
```

v3 默认冻结首 attempt 主分析和单次预算；恢复必须在计划中预注册，并单独出表。旧 v2 保持原口径。单 run 坏元数据不会删 expected 分母；坏根计划拒绝生成分析。字段级完整性、共享存储、计数器 epoch、部分成本和 GPU NA 语义见 [MeasurementProtocol](docs/MeasurementProtocol.md)。

可用 `--motion-result` 与 `--quality-result` 消费绑定同一冻结 suite 的实际评价。消费者重放保留 PNG、重建图块并重算特征统计，运动、质量、RGB 和资源分表；无输入时继续导出 RGB。仪器 PASS 不能替代模型评价，诊断结果不升级正式分数。见 [C05 消费说明](docs/C05_METRIC_CONSUMPTION.md)。最终干净提交的实际 wheel/sdist 留存、远端 ZIP 下载和独立安装验收见 [C07 分发说明](docs/C07_DISTRIBUTION_RETENTION.md)。

## 模型、仪器与独立研究

诊断 Oracle 仅 evaluator 可见，Copy/StaticNoiseMap 不代表生成模型性能。批准的本地真实 backend 必须锁 source/weights/objective/许可及 camera manifest；不自动猜模型或退回噪声。`rswm-train` 执行原 backend 自己的真实 loss，绑定 train60/core120 audit、dev 来源、冻结预算和完整 checkpoint 状态。`rswm-export-trained-adapter` 将通过来源检查的 driver checkpoint 转为低/高噪声专家的实际 PEFT 文件，形成独立推理源树及 release；原训练 checkpoint 和 receipt 保持不变。导出成功只形成受训模型候选，实际重新加载、pilot、测量与研究验收仍各自执行。原模型未提供时真实恢复、训练、GPU 成本不通过。

运动的后续搜索域元数据修订按新的package source身份记录，不追改原9ef full smoke12或G5 validation。实际整数overlap搜索谓词、任务几何与RGB评分不变；后验Bellingham calibration-only工程诊断实际执行336对：96内侧可测、96边界与144域外均弃权，不是expanded-heldout验证或新的release门；不替代生成模型泛化或owner正式批准。具体结果和C clean/D CI身份见续建交付报告。

运动使用 rswm-motion-calibrate / rswm-motion-validate；特征使用 rswm-features / rswm-validate-features / rswm-release-features。calibration、held-out validation、源身份和曝光账本分离。用户冻结的 feature 接受规则不能由代码或 test 数据凭空生成。质量分布诊断与 RGB 一致性分开，MMD-U 可以为负。

rswm-paired 和 rswm-lr64 使用独立 namespace、计划、replay 与 world 统计，不能混入核心 2048² 基准。[AdapterGuide](docs/AdapterGuide.md)、[数值后端合同](docs/NUMERIC_BACKEND_CONTRACT.md) 和 [G7 指南](docs/G7_EXECUTION.md) 给出接口与范围。

worker 的 framed IPC、状态 token、deadline、生命周期资源与进程组清理有实际测试。合作式子进程不是安全沙箱。系统隔离和真实 GPU profile 需各自实际验收，当前限制见 [Limitations](docs/Limitations.md)。

仓库和分发包不含原图、世界、预测 PNG、模型权重或凭据。数据和代码许可尚有负责人待定项，见 [NOTICE](NOTICE.md)。完整执行计划是 [CODEX_FULL_BUILD_PLAN](docs/CODEX_FULL_BUILD_PLAN.md)；历史研究方案保留于 [plan.md](plan.md)。软件可构建、实验验收、论文发布是三个独立状态。
