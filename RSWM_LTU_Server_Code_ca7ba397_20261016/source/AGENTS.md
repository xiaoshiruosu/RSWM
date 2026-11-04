# 合并到现有 AGENTS.md 的执行规则片段

不要用本文件覆盖仓库已有AGENTS.md；仅将以下段落合并，并遵守更高优先级和现有范围约束。

## RSWM completion mission

当用户要求完成RSWM Benchmark时，读取 `docs/RSWM_PROJECT_SUMMARY.md` 与 `docs/RSWM_COMPLETION_MASTER_PLAN.md`，以该总计划的L0/L1/L2和M00–M10验收作为任务终点，不以单测数量、一个commit或一个安装包作为全项目完成。

继续实际最新合法后继；当前已知基线为 `codex/trained-wan-export@ddc1dd1868ac94cec9c3652bfec0f75f26003772`。不得从旧main重写已完成的suite/resources/aggregate、原Wan桥接或N06-E。

在已授权会话、范围和预算内，完成一个里程碑后直接继续下一可执行里程碑，不逐次要求用户回复“继续”。每项工作须实际实现/运行/验证，并记录范围和证据。

优先复用既有状态文件，维护当前执行游标、真实job/PID、输入缺口和下一条可执行命令。会话中断标HANDOFF_REQUIRED，不标COMPLETE；接续先核查已有进程，避免重复启动。

外部阻断只暂停其依赖链，其他可执行工作继续。所有剩余项都被真实访问、资产、批准或预算阻断时，集中返回最小实值请求并保存现场。无条件变化不反复重试502、扫描同一路径或重新做相同mock。

不改变冻结任务/样本/首次参考/主分析/指标阈值来通过验收；不伪造approved、PASS或实际模型记录；不将diagnostic计入真实pilot。模型性能低或实际失败本身是结果，不是删除样本的理由。

代码修改按影响范围回归并在开发分支提交。无代码变化无需强造commit。未经明确授权，不合并main、不发布tag、不公开数据/权重、不扩大预算、不删除唯一原件或活动锁。
