# Pilot / formal 实验门禁

`python -m rswm_inria.experiments freeze` 生成冻结实验；`run` 实际调用现有 suite、独立 replay 和出表。诊断运行可以验证工程，不能批准正式模型实验。

每个 pilot 必须是固定 2 个 dev 世界 × 1 origin × 四族 × 1 inference seed；每模型 8 包 / 704 新输出。正式 core matrix 是固定 24 个 test_unseen_city 世界、SFO / Innsbruck 各 12、4 origins、四族，每模型每 seed 384 包 / 33,792 新输出。模型训练 seed 属于模型 manifest，不与 inference seed 混淆。

freeze 的 `--inputs` JSON 可以有以下字段；所有真实材料都必须是实际生产者输出，不能用单独的 `PASS` 字段或自签 checksum 当完成证明。

| 字段 | 合同 |
| --- | --- |
| budget | 恰好 `max_wall_seconds` 与整数 `max_output_bytes`，正且有限 |
| source_audit | 真实 `rswm-materialized-audit-v1`；完整源/world/task/role/city/file/pixel/hash 与 plan 绑定；formal 必须 core120 |
| rights | 真实运行必须有 `data_use_approved=true` 和 `weight_use_approved=true` 的决定 |
| pilot_receipts | formal 必须覆盖每个当前模型 manifest；实际新版 experiment receipt，含保留证据路径 |
| pilot_qualification_roles | 可选 `{model_id: "reference" 或 "comparison"}`，精确覆盖当前 plan 模型，pilot 和 formal 均在执行前冻结；formal 至少一个 reference。省略时保持原来所有模型完整通过的 strict 合同，不增加旧序列化字段或改旧身份 |
| metric_receipts | formal 必须有实际 motion 与 quality release receipt；candidate controls / diagnostic receipt 不可替代 |
| instrument_config_sha256 | formal 必填 `{"motion": "完整SHA256", "quality": "完整SHA256"}`，作为事前冻结的指标配置 |
| smoke12_evidence_root | formal 必须有实际真实来源 smoke12 三诊断的保留目录；pixel-only core120 audit 不替代该门 |

source_audit 验证使用 `materialized.validate_audit_receipt`。motion 使用 `motion_protocol.validate_release` 的完整 Bloomington 留出角色、实际阈值与未重调合同。quality 使用 `features.validate_feature_release(receipt, preregistered_feature_config_sha256, source_manifest_sha256)`，并重核保留的真实配置/权重/mosaic/世界像素/角色与 exposure ledger。模型列表、指标配置、来源或代码变化都会改变实验身份或导致拒绝。

真实 pilot 的新版 receipt 还包含 source/code identity、固定 matrix、verified_outputs、budget_usage、evidence_root、receipt_path、analysis_provenance_path 与 hash。formal 放行时重新读取其实际 experiment / suite plan，独立 replay PNG 和全部尝试资源，再核对出表文件 hashes。缺原保留证据、篡改表格、孤立薄 `ACCEPTED_REAL_PILOT` receipt 或不完整资源均拒绝。保留 hash 是内部身份一致性证据，不是外部时间戳或数据真实性认证。

显式资格角色分别执行以下合同，不能在看过失败之后改角色来放行同一 pilot：

- **reference：**每模型固定 8 包、704 唯一新槽位全部完整，首 attempt 的 PNG、资源和独立回放均可信；默认 strict 旧 pilot 只能作为 reference 使用。
- **comparison：**允许 `INCOMPLETE`、`FAILED_INITIALIZATION`、`FAILED_GENERATE`、`FAILED_SNAPSHOT`、`FAILED_RESTORE` 等真实模型终态，但每个冻结包都必须已有 attempt 000、完整预定查询/输出槽位日志、保留 PNG 或明确缺失槽位、完整资源及独立 replay。所有历史 attempt 的资源和身份仍须完整；未尝试、加载失败、崩溃、回放失败、账目/代码身份错误或预算耗尽不构成资格。
- 若参考链完整且只有 comparison 模型失败，实验 receipt 为 `EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES`。原 suite integrity 和 aggregate/provenance 的 `INCOMPLETE` 保留，不能改写成 `ACCEPTED`。formal gate 按已冻结的 model manifest 与资格角色逐一核验，并从保留 suite 重算 aggregation，核对原 aggregation.json、输入指纹及全部分析文件哈希。

重算比较仅对 `cells`、`first_attempt`、`recovery_attempt` 三张统计表按完整 canonical row 排序：原内存任务与持久化 JSON 的字典键顺序可能不同，统计行次序不属于指标定义。排序不去重，不改变任何值、缺失或分母；原 PNG/journal、输出 slot、attempt 顺序及其他分析字段仍严格比较。

同一 pilot 可包含 reference 和 comparison，也可为某个 comparison 单独执行；正式矩阵必须有至少一个取得完整参考资格的模型。receipt 的 `pilot_qualification_roles` 必须与原冻结 experiment 一致，与正式矩阵对应角色一致。比较失败从未减少计划 8 包/704 槽位分母，后续仍使用 `first_attempt_only`，预算内恢复和全部尝试成本继续另列。这个资格状态表示失败可作为研究数据进入下一阶段，不表示模型、测量或科研总包已经通过。

显式角色的正式矩阵使用相同的完整执行证据规则。参考模型在 pilot 取得完整资格后，其正式输出仍可失败；参考与比较模型的完整真实失败 trace 都是研究结果。所有正式计划包均已实际终态、首 attempt 独立 replay 和全部资源/历史身份可信、预算未耗尽时，存在模型失败则 receipt 为 `EXECUTED_FORMAL_WITH_RETAINED_FAILURES_PENDING_INSTRUMENT_ANALYSIS`；全输出完整时仍为 `EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS`。两者 publication_acceptance 都是 `NOT_READY`，正式 motion/quality 尚需保留仪器结果独立分析；原 aggregate/integrity 的失败状态和全部正式分母不变。未尝试、加载失败、崩溃、回放/账目/源码身份错误或预算失败继续为 `INCOMPLETE`/`BUDGET_EXHAUSTED`。未提供角色映射的旧 formal 保持原 strict 状态语义。

真实 smoke12 门读取其实际 acceptance / plan / input-audit，核对 576 runs / 50,688 新输出、三诊断固定配置、192 task / 12 world / seed 0、同源和同代码，并重新验证保留的 PNG / 资源 / artifact hashes。没有真实矩阵证据时 formal 为 `BLOCKED`，不通过 synthetic subset 推断通过。

预算 ledger 位于执行目录的 budget.json，持久保留累计 session elapsed；普通 resume 延续以前的总预算。hard exit 留下的活动 session 会在 resume 时保守计入停顿，不因重启重置时间。预算在每个下一次模型加载前和终端出表后检查；合作式模式不能抢占正在执行的模型调用，真正的进程 deadline 依赖 D11 worker。最后一个 run 即使 RGB / replay 成功，超预算仍是 `BUDGET_EXHAUSTED`。最终发布预留 receipt / ledger metadata 字节，usage 同时记录已观察 bytes 与 publication reserve。

resume 不覆盖原 receipt / analysis。首次结果位于 receipt.json 和 analysis；后续结果位于 receipts/invocation-NNNNNN.json 与 analyses/invocation-NNNNNN，receipt-head.json 指向最新保留结果。实验 coordinator 和 suite coordinator 分别排他锁定，symlink 路径拒绝。

CLI 的配置、来源或根身份错误退出码为 2；`INCOMPLETE`、`BUDGET_EXHAUSTED`、`EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS`、`EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES`、`EXECUTED_FORMAL_WITH_RETAINED_FAILURES_PENDING_INSTRUMENT_ANALYSIS` 为 1。服务器串联程序应读取显式资格/待分析状态并验证原证据后继续，不能仅凭退出码把它算作完整成功，也不能因此删失败重新筛图。诊断工程成功 `DIAGNOSTIC_ONLY` 可以返回 0；它不能变成真实 pilot。所有当前 receipt 的 publication_acceptance 仍是 `NOT_READY`，未单凭 RGB 出表宣布正式 motion/quality 分析或科研发表通过。

真实数据、weights、GPU、仪器 acceptance config 和批准材料当前缺失；实际 formal 没有运行。`scripts/reproduce_experiment_gate_cli.py` 用明确 synthetic CPU 图片执行实际 plan/freeze/run/resume、无隐式覆盖和预算失败 CLI，工程证据保存于 evidence/build-closure；不会下载真实数据或权重。
