# 上传服务器后的执行入口

`rswm-server` 串联既有数据、真实模型、训练恢复导出、正式实验和测量入口；project JSON 只描述 RSWM 输入，没有任意 shell/job 字段。实际执行使用 Linux/POSIX flock、独立子进程 argv 和自有进程组。所有科研验收仍由原生产者/审计器决定，driver 不自动签收 `RSWM_V0_1_RESEARCH_PACKAGE_ACCEPTED`。

## 建立一个可填写项目

将完整上传包的 `source/` 保存到持久卷，安装同一版本 wheel 或源码及服务器所需 CUDA/model 依赖。完整源目录必须保留 `scripts/prepare_legacy_wan.py`、安全 `vendor/legacy-original` 与 `vendor/legacy-source-inventory.json`；wheel 和标准 sdist 本身不携带原 vendor，所以要使用完整上传包。原 provider 工具子树被排除，模型权重和 TIFF 单独存放。

```bash
rswm-server init --source-root /srv/rswm/source --out /srv/rswm/project
rswm-server check --project /srv/rswm/project/project.json --stage pilot
rswm-server plan --project /srv/rswm/project/project.json --stage pilot
```

`init` 排他建立 `project.json` 与六份 owner 模板。已有目录拒绝覆盖。使用决定默认 false，采样/训练参数和每阶段预算默认 null；`check/plan` 可在没有模型时运行，显示输入或前置阶段阻断并返回非零，不启动生成或训练。

`code_commit` 自动优先取匹配完整源码目录的真实 Git HEAD；无法使用匹配 Git 时读取包的 `_build_info.source_commit`。后一种记录明确为 `base_commit_plus_dirty_snapshot`，`source_dirty=true`，当前 Python 内容用独立完整 `source_code_sha256` 绑定。它表示保留基线上的修改快照，不伪称新提交或干净发布；无需手抄已核验 DDC hash。installed package 与 `source_root/src/rswm_inria` 指纹不一致时拒绝执行。

按真实服务器实值填写以下分组：

| 分组/文件 | 要填写的内容 |
| --- | --- |
| `data` | 独立的 `pilot_source_dir`/`core_source_dir`；两 profile 各自 download 布尔值与正整数 `*_max_source_bytes`；已有完整物化树可填 `existing_pilot_build`/`existing_core_build` |
| `models` | numeric 原 Wan、无数值 Diffusers Wan I2V、真实补全器三套实际权重路径；视频/补全明确 steps、固定 prompt 和模型 ID |
| `owner/wan-runtime.json` | 推理 source 的明确 mode、sampling、dtype 和 offload；所有 null 都须替换为对应实际值 |
| `owner/camera.json` | 原有四键相机合同；默认是 FixedUnits/native/stride4 的技术配置，不代表实际相机层已验证 |
| 两份 `*use-decision.json` 与 `rights` | 实际原模型/受训 release 使用决定、许可记录及数据/权重使用参考；程序不会填 true |
| `budgets.STAGE` | 正有限 `max_wall_seconds`、正整数 `max_output_bytes`、正整数 `max_attempts`；默认 null 无运行额度 |
| `training` 与 `owner/training-spec.json` | 明确更新/batch/累积/LR/seed/checkpoint/LoRA policy；从冻结 manifest 的 12 个正式 dev 世界中显式选 validation IDs；checkpoint 必须明确指定 |
| `owner/wan-training-runtime.json` | 与推理 runtime 独立；实际 training sampling、resident 两专家及 `inactive_model_cpu=false`；不改已经冻结的 pilot runtime |
| `formal` | 事前冻结 inference seeds、是否含 trained 模型、motion/quality 实际 release 路径、事前配置 hash、真实 smoke12 保留证据目录 |
| `measurement` / `independent` | 正式配置及既有 held-out 输入/曝光账本，或独立 paired/LR64 的已冻结数据、模型和有限统计范围 |

路径相对于 `project.json` 所在目录解析；owner training spec 的 camera 路径相对于该 spec 解析。运行输出应在持久 `run_root`，不能覆盖 source/src、vendor、脚本等保留输入。环境检查报告实际 Python、POSIX 能力、磁盘可用量与 CUDA Torch 可用性，不声称显存足够或原模型已兼容。

## 一条命令推进开发闭环

填好 pilot 所需实值后：

```bash
rswm-server run --project /srv/rswm/project/project.json --stage pilot
rswm-server status --project /srv/rswm/project/project.json
```

第一条自动按依赖完成 `pilot-data → prepare-models → pilot`。数据使用包内冻结 manifest/inventory；已有 build 只重新独立审计，未提供已有树才调用物化。固定 Kitsap dev2 的完整 32 包保持原样，运行索引选择 origin0 的八包；每模型计划 704 新输出。

四系统配置是实际 numeric-camera、无数值 `wan-i2v`、同一真实补全器的 Frontier 和 Lazy。numeric 是 reference，其他三个是 comparison。controller 从当前审计 JSON 内联 source audit，原 source 准备器和 numeric 身份准备均为新进程；模型只获得 I0/公开规范/当前允许查询与历史。它仍是合作式运行，不声称不受信任代码隔离通过。

新的 comparison pilot 可能合法退出 1：若 reference 完整且 comparison 失败已完整保留，依据实际强 receipt 的资格状态继续；缺资源、完整性错误、`INCOMPLETE` 或 `BUDGET_EXHAUSTED` 均停止。driver 不以 exit1 泛化成功，也不放宽图像或模型分数。

## 训练、导出及正式测量

```bash
rswm-server run --project /srv/rswm/project/project.json --stage train
rswm-server run --project /srv/rswm/project/project.json --stage export
rswm-server run --project /srv/rswm/project/project.json --stage trained-pilot
rswm-server run --project /srv/rswm/project/project.json --stage formal
rswm-server run --project /srv/rswm/project/project.json --stage metrics
```

`train` 自动先完成 core 数据、独立 training source 和 training inputs。它保持完整 train60/3840 clips 合同；driver 根据 owner 明选的正式 dev IDs 派生验证记录与 `updates × dev世界数` forward 上限，不自动增加额度。训练的 sampling/resident 设置来自独立训练 runtime，不迫使修改原 pilot 的 source/config。训练参数仍须 owner 明确提供；源树、相机和真实字节 SHA 由 [服务器输入准备器](SERVER_INPUTS.md)计算。

`training.checkpoint` 在实际 checkpoint 产生后填写明确路径，不按时间或名字猜“最新”。恢复使用 `--stage resume`，或在失败 train 的同一项目中 `--stage train --resume`；它使用同一 `run_root/training/wan` 与所有原 attempts。强 checkpoint receipt、冻结合同、snapshot 和旧成本文件字节保持可核对；合法新增 attempt/checkpoint 不使旧 train milestone 失效，删改原成本仍拒绝。

导出调用原双专家 PEFT 入口，随后 trained-pilot 用新的独立 inference source 和 owner 新使用决定，在新进程重新加载。导出不是 GPU/模型评价签收。`formal` 内联真实 pilot/仪器收据及 source audit，保留全部 planned test 槽位与 comparison 失败；完成状态仍是 pending instrument analysis。`metrics` 从同一冻结 suite 重算 motion、首次提交图块、实际特征、quality 与 CSV/JSON/LaTeX/figure-data。缺图块的合法 feature bundle 可退出 1 并继续覆盖/质量分析；无权重导致没有 bundle、坏 schema 或 hash 仍失败，不把缺失当成零或补首次参考。

已有 instrument 入口也可由固定 stage 执行：`motion-calibration`、`motion-validation`、`feature-calibration`、`feature-validation`。feature 输入文件恰为原 release CLI 的 config/acceptance/reference/patch/suite/source-audit/cache 字段，validation 再带 calibration receipt 与 exposure ledger；driver 不生成 reviewed bounds。明确的新曝光账本路径可以由原 producer 首次创建；已有历史必须指定原 ledger，不创建替代空 JSON或清空旧曝光。

`paired` 与 `lr64` 使用现有独立 namespace 的 plan/run/replay/aggregate，并要求独立输入和预算；不混入核心 2048² 统计。不支持的模型原尺寸范围如实拒绝或留失败，不自动缩放成另一个研究。正式 feature 原始权重导出见 [DINOv2 候选工具](FEATURE_EXPORT.md)，候选与仪器 release 分开。

全部所需实值均具备时，可用 `--stage all` 按 `workflow` 连续执行。默认 workflow 只串联核心工程链，不包含自动恢复、仪器校准/验证、paired 或 LR64；这些总计划必需研究要单独运行，或在首次执行前把对应阶段明确列入 workflow 的适当位置。例如：

```bash
rswm-server run --project /srv/rswm/project/project.json --stage motion-calibration
rswm-server run --project /srv/rswm/project/project.json --stage motion-validation
rswm-server run --project /srv/rswm/project/project.json --stage feature-calibration
rswm-server run --project /srv/rswm/project/project.json --stage feature-validation
rswm-server run --project /srv/rswm/project/project.json --stage paired
rswm-server run --project /srv/rswm/project/project.json --stage lr64
```

仪器的 reviewed 配置、实际参考图块/来源和保留曝光历史依旧由对应既有 producer 消费；不能跳过它们而提交 formal，不能把 `all` 当总科研签收。checkpoint 仍需明确选择；driver 不跳过未尝试阶段来宣布完成。

## 中断、失败和接续

每阶段保存 scoped `stage-contract.json`、`attempts/attempt-NNN`、逐命令实际 argv/PID/出生 token/stdout/stderr/exit/成本，以及 `cursor.json` 下一条命令。失败与重试用新 attempt；已经完成阶段先核对 marker 与实际保留字节再跳过。pilot/core 的目录、下载决定和源额度独立冻结，因此以后落实 core 数据不会改写已完成 pilot-data 身份。

```bash
rswm-server run --project /srv/rswm/project/project.json --stage pilot --resume
```

source/config/budget 已开始后发生变化会拒绝复用，保留原现场。恢复不重置累计墙钟、attempt 额度或实验/训练成本；hard exit 未完成时间保守计入 downtime。活的原 PID/自有 group 仍在时先观察，不重启。预算耗尽停止相关阶段，不自动扩额度。

SIGTERM、墙钟或输出超限会记录实际失败并终止本 driver 创建的自有 SID/PGID；leader 已退出也检查并清理仍活的同组后代。出生 token 防止误杀被复用的 PID。锁由操作系统释放，不删除 coordinator 锁文件。退出记录不能把剩余 worker 写成 `live=false`；无法完成受控清理时保持 attention/block。它不处理未授权远程调度器任务，也不构成安全沙箱。

`REQUESTED_CODE_STAGES_EXECUTED` 只表示所请求代码阶段已执行，`goal_completed=false`、`research_acceptance=NOT_EVALUATED` 始终明确。实际模型/数据/仪器结果和总计划第 3.3 节的逐项证据仍须服务器实验形成，不能由 templates、测试统计或 driver marker 代替。
