# RSWM 项目规则

## 项目身份

- 当前顶层同步目录是主运行包；内部同名嵌套目录是内容完全相同的冗余副本，不作为第二份真相，也不要擅自删除。
- 先阅读 `README_RUN.md`、`SYNC_REPORT.md` 和 `DEPLOY_D4.md`，并以其中记录的同步来源与验证命令作为来源证据。
- 历史 `server_run/work/` 结果只读。新实验使用独立的 `WORK` 目录，禁止混入或覆盖历史证据。

## 冻结实验契约

- 除非用户明确要求改变研究设计，冻结协议、模型、LoRA 配置、随机种子、数据划分、轨迹和 `SemanticGate`。
- 任何新结果都要保存实际命令、环境、修订、配置、种子、输入清单、日志和失败记录。

## 验证与运行

- 首先运行计划模式：`bash server_run/run_iclr_full_pipeline.sh --plan`。
- 计划模式必须显示 `GPU_ACCESSED=0`、`TRAINING_STARTED=0` 和 `FINAL_RESULT=PLAN_ONLY`。
- 再按 `SYNC_REPORT.md` 执行 shell 语法检查、Python 编译检查、单元测试和 SemanticGate；只报告实际运行的结果。
- 未经明确授权，不提交 PBS/调度器任务，不申请共享 GPU，也不启动正式训练。缺少目标环境或真实数据时结论为“受阻/未验证”，不是“已就绪”。
