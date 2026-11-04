# 完整构建阶段状态

## 2026-10-01 原 Wan 源码接入后继

从 `5825fde350cea73921e518049c0cac45368b8b47` 合法后继继续。用户提供的原 ZIP 已实际读回，473 文件、251 Python AST 检查和 SHA256 与交接一致。新增接入选定 `lora_remotesense/new/wan`，复用原 base / restored-LoRA 加载器和 `generate_video`；prepared-camera 路径避免再次插值、相对变换和归一化，并连接原 flow-matching 目标到现有训练 driver。源树、实际权重组件、严格加载、可变状态和每次调用观察分别核对。操作与限制见 [原后端接入](LEGACY_WAN_BRIDGE.md)。

固定 `pilot-dev2` 仅允许原 manifest 的 kitsap31 / kitsap19，仍由现有物化器与独立审计器执行；smoke12 / core120 完整数量门不放宽。本轮限定路径查找没有恢复实际 TIFF，没有实际权重或 CUDA 设备。源码连接和 CPU 软件控制不签收真实模型、真实训练、正式测量或 8 包 / 704 输出 pilot；实际有效 pilot 输出为 0。历史诊断、曝光 ledger 和旧收据保持各自原身份。

最终源码冻结后执行本阶段完整标准 CPU / CPU Torch 回归、CLI 及最终 HEAD 分发留存、下载读回与安装。具体命令、计数、实际 hash 和精确 HEAD CI 终态在提交后的独立交付报告中记录，不再追加“记录自身 CI”的提交。

## 2026-10-01 指标与分发后继

本轮从审核集成提交 `0c5f98a26c38d50f6ffd6e32d03815e8d6834ce3` 继续；启动时远端集成分支没有后继。已有 Phase A、suite/resources/aggregate、真实 f1 smoke12 三诊断和历史 ac8 core120 audit 保留，不重做基础设施或机械合并 PR #1。

C05 新增实际保留评价消费：重新核对 suite、模型/权重、source/task/study/seed/attempt/config 身份，重测 PNG、重建图块与参考、重算 MMD；正式仪器还须具备实际保留发布验证输入。RGB、运动、质量、资源按各自单位出 JSON/CSV/LaTeX、figure-data 和 provenance。合成贯通明确为 diagnostic，正式模型分数仍 `NOT_AVAILABLE`。见 [消费合同](C05_METRIC_CONSUMPTION.md)。

C07 新增干净具体提交的实际 wheel/sdist 留存、输入映射、SHA256SUMS、CI artifact ZIP 下载和两种独立安装验收。最终 HEAD 的测试、实际字节 hash 和 CI 终态在提交后报告；不由前一提交包或递归记录自身 CI 的提交签收。见 [分发流程](C07_DISTRIBUTION_RETENTION.md)。

资产清查逐项列出固定 120 源；本轮当前可读源核验为 0/120，不代表其他存储没有剩余文件。恢复的备份 manifest 内 12 源 metadata 与固定清单一致，但原归档 GET 返回 HTTP 502、0 字节，备用路径 403，不能签收影像恢复或当前 core120 audit。没有原 numeric 后端/权重、批准预训练特征及训练/推理预算，也没有 GPU；真实模型/pilot/正式运动质量重验保持 BLOCKED/NOT_RUN。历史运动配置及 ledger、6px 错误原样保留，当前仅完成配置 metadata 兼容检查。C06 现有 CPU 隔离范围仍 fail closed；C08 没有生产去重或 hardlink 重试。

下文及旧收据仍对应各自历史源码；本节不将历史测试数或数据 PASS 转授当前后继。

## 2026-10-01 当前源码续建状态

当前状态仍为 `SOFTWARE_ENGINEERING_DELIVERY_EXPERIMENT_NOT_ACCEPTED`。下文2026-09-30记录保留其原源码和实验身份；本轮当前结论以[续办摘要](../evidence/build-closure/continuation-20261001/summary-current-source.json)及[续办报告](../evidence/build-closure/CONTINUATION_20261001.md)为准。

源码checkpoint为 `63b98201209933689dbd6d0cebf9fd91fe91dd8c`，递归包指纹为 `f1f7edadeba430a44acfb5076d3cc2ea904c34cf41c84cc0dbc2809589fe3096`。五个冻结研究文件的SHA256和Git blob与G0冻结清单一致，核心像素、几何、任务和RGB语义未改变。本轮修复smoke12消费门禁：接受生产者的精确空列表或legacy精确整数0，拒绝缺失、非空、布尔和浮点类型。

| 项目 | 当前实证 | 适用范围与限制 |
| --- | --- | --- |
| G0/G1/G3及组合软件 | 63b自身标准CI为915 passed/4 skipped，CPU Torch为918 passed/1 skipped，各919项；clean安装验收2/2 runs、192/192输出 | 只对应63b；GPU与实际OS权限验收未通过，不替后继提交背书。 |
| G2当前完整smoke12 | 12冻结TIFF、12世界/48 I0/192任务；copy、oracle、static-noise-map各192 runs/16,896输出，合计576/576、50,688/50,688；生产者及独立canonical消费门禁均PASS | 真实来源CPU诊断与Phase A链路，不是实际生成模型、训练或正式test验收。 |
| 原始诊断保留 | 57,634文件、12,175,024,053逻辑字节，保存为49个归档；本地逐成员SHA256与冻结文件集合验证PASS，压缩完整恢复清单实际回读匹配 | 大归档上传身份/大小已确认；远端逐字节回读UNAVAILABLE，不写作PASS。原件与失败记录保留。 |
| G5/G6地理及数据 | 120世界的4,896跨角色元数据对完成比较，包括此前未比较的4,224跨CRS对；历史ac8 core120物化/audit记录保留 | 包围范围工程复核不签收地理真值、近重复、统计独立性或owner批准。当前f1仅定位且哈希验证12/120原源，f1 core120 audit仍NOT_RUN。 |
| G4/G5/G6/G7/G8正式门 | 当前real-model-pilot、metric-validation、core-formal、isolated-release无输入CLI均实际exit2/BLOCKED | 原数值后端、批准生成/学习特征权重、loss与训练预算、完整f1 core120、GPU及可用OS隔离环境仍缺；权限检查全NOT_RUN，publication NOT_READY。 |

块去重的新合成文件探针为FICLONE ENOTSUP、FIDEDUPERANGE -22/EINVAL且deduped0；当前范围不可用，没有生产文件改写、hardlink重试或回收量声明。冻结来源的历史年份/适用条款与owner签收仍未闭合，不把来源论文或镜像说明当作授权。

本次证据和状态后继提交的自身CI在提交存在前为PENDING；提交后须回读其精确HEAD终态，并记录于[Draft PR #3](https://github.com/yedda-tv/rswm-1/pull/3)，不以反复新增“记录自身CI”的提交改变验收对象。未合并main、未创建正式发布tag。

## 2026-09-30 续建实证

总状态：`SOFTWARE_ENGINEERING_DELIVERY_EXPERIMENT_NOT_ACCEPTED`。真实 Inria 三诊断576 runs / 50,688 unique outputs已完成独立回放、资源、聚合与导出验收；运动仪器固定实景留出通过限定范围验证；LEVIR100/200 LR64数据审计通过。core120在原环境实际 `BLOCKED_SPACE`；恢复环境重新核验缓存/空间，core120实际运行/物化/audit状态由本轮报告与handoff签收。真实模型/特征权重、原数值训练后端、GPU、系统安全、owner正式配置及论文实验分别未验收。

完整来源、实际命令/计数/失败保留、D01–D12状态、Git/CI与后继验证收据见[本轮交付报告](../evidence/build-closure/RESUMPTION_DELIVERY_2026-09-30.md)。后继精确提交的CI/clean结果以该报告“软件checkpoint与后继验证”及对应不可变收据为准；本文件不将历史checkpoint的测试数转授给以后提交。

实际full smoke12执行base=`fd7eb6e3ecfec361b4c91d43efc496c9a5e7d4c5`，dirty package source由fingerprint `9ef2a49d194dd224ef54b1254871805f837083620cc67ee819e0d85dc39fd6e7`约束。2fc/d9/A/B的历史package source等价另列；后续搜索域元数据修订改变整个package指纹，不能再把原9ef fullsmoke/G5验证写成新package等价验收。实际整数overlap谓词及冻结几何/RGB不变，旧plan、run_id、validation和dirty provenance不被改写。

| 阶段 | 本轮真实完成范围 | 明确限制与证据 |
| --- | --- | --- |
| G0 | 恢复共享工作树、基线和失败记录；五个冻结文件与Phase A逐字节一致；原现场XML按旧收据保留，恢复初始未找到原XML，实际可用性见本轮报告，不造替代 | [基线](../evidence/build-closure/G0-report.md)、[本轮身份](../evidence/build-closure/RESUMPTION_DELIVERY_2026-09-30.md) |
| G1 | R01/R02/R03、retry、逐run容错、CPU资源和CSV/LaTeX工程验证 | GPU mock与CPU不代表实际GPU；[G1](../evidence/build-closure/G1-report.md) |
| G2 | 12真实TIFF、12世界/48 I0/192包独立audit；Oracle/Copy/StaticNoiseMap各192 runs/16,896 valid outputs，合计576/50,688；八项实景故障控制PASS | 合作式CPU诊断；hard-exit未知资源不补零；source rights/geography未闭合；[原执行收据](../evidence/build-closure/G2-real-full-smoke12/commands.json)、[故障](../evidence/build-closure/G2-real-fault-controls-resumption/summary.json) |
| G3 | Frontier、相机/数学/状态契约、来源探针和registry组合工程回归 | 真实camera训练层消费/模型未验收；[G3](../evidence/build-closure/G3-report.md) |
| G4 | 原numeric公共包装、source/weights门、模型状态/成本及训练driver路径 | 指定原骨干/loss、批准权重、真实恢复/梯度/GPU缺失；已有数据不代替原backend；[G4](../evidence/build-closure/G4-report.md) |
| G5 | 原9ef真实24世界；Bellingham calibration与首次Bloomington 3,072 pairs，2,110 measured、coverage68.6849%，`VALIDATED_IN_STATED_SCOPE`；后续搜索域metadata修订由62 targeted tests验证；Bellingham-only后验工程诊断336对完成（96内侧0px可测、96边界/144域外弃权），与原留出单列 | 原留出、selected config及ledger保持原字节，保留一个6px错误；工程边界控制不替代formal expanded-heldout/owner批准或D06 feature release；[motion v2](../evidence/build-closure/G5-real-resumption/v2-report.md) |
| G6 | core120固定选集/预算/缓存SHA与共享disk预检，训练/编排工程 | 历史13:57预检为36有效缓存/84 missing、available6,391,939,072、required9,014,030,277、缺2,622,091,205 bytes；恢复环境必须重新核验这些动态值，恢复后的实际data运行/审计按本轮报告/handoff签收；正式模型运行NOT_RUN；[最终预检](../evidence/build-closure/G6-core120-real-resumption/final-blocked-space-disk-preflight.json) |
| G7 | 固定3 Parquet，100原生1024世界/100 I0/200 LR64包，真实像素及LR64 audit PASS；paired工程完成 | LR64真实生成0；真实机制/生成/feature科学实验NOT_RUN；[实际摘要](../evidence/build-closure/G7-real-resumption/summary.json) |
| G8 | 历史A精确CI：882+4skip / CPU Torch885+1skip，各886 tests；A clean130 authored inputs/24 scripts、19入口、offline no-Git rebuild、仓库外192输出及33 storage tests PASS | metadata修订后的C clean与最终D CI分别以本轮报告/不可变收据/PR为准，不能借A/B旧9ef绿色结果；actual OS安全BLOCKED、GPU/不可信执行未验收；[A CI](../evidence/build-closure/G8-CI-f7b8f19/receipt.json)、[A clean](../evidence/build-closure/G8-clean-f7b8f19/checkpoint-summary.json) |

后验存储hardlink两次严格guard中止，第二次FD nlink136→135，actor UNKNOWN；每次后原57,634路径和所有SHA独立复核不变。失败ledger/stage/reserve保留，进一步hardlink尝试停止；不声称inode/mtime保持或回收成功。同文件系统新对齐块去重探针 `NOT_SUPPORTED_IN_PROBED_SCOPE`，production raw块去重未运行。见[存储终态](../evidence/build-closure/G8-storage-dedup-resumption/summary.json)、[能力收据](../evidence/build-closure/G8-storage-capability-negative/final-receipt.json)。这些后验操作不重写原诊断时延或logical disk resource。

真实raw TIFF/Parquet/PNG、模型权重和巨型原plan排除Git，原现场保留于scratch；恢复环境的缺失原件和可用缓存另做实读核验，不能由Git收据推定原文件仍可用；[无损text归档](../evidence/build-closure/G2-real-full-smoke12/collected-for-git/TRANSFER_README.md)保留原acceptance完整hash map、aggregation独立replay、576 metadata index与全部派生表。数据准备、文本采集和压缩均不升级科学门禁。正式发布/主榜为NOT_READY，未合main或发布正式tag。

## 保留的历史checkpoint

以下数量只对应其原source/attempt，失败记录不被改为PASS，也不与本轮测试相加：

| 历史记录 | 原范围 | 证据与限制 |
| --- | --- | --- |
| 673a241/fd7及更早交付 | package fingerprint `7d114910…`；800 passed/4 skipped | [804项原XML](../evidence/build-closure/full-build-final-tests.xml)、[旧交付](../evidence/build-closure/full-build-delivery-report.md)；673首条build缺setuptools失败另存，后继workflow修复不能追改旧失败 |
| fd7后继dirty snapshot | source fingerprint `8d721c0e…`；819 collected、818 passed/1 OS skip | [原819项XML](../evidence/build-closure/G8-torch-cpu-continuation/current-source-full.xml)；是当时工作树，不是2fc/d9或本轮后继HEAD |
| d9 checkpoint | source fingerprint9ef；CPU Torch881 passed/1 skip、standard878 passed/4 skip，各882 tests；clean130 inputs/24 scripts | [d9 CI](../evidence/build-closure/G8-CI-d9bc3b9/receipt.json)、[d9 clean](../evidence/build-closure/G8-clean-d9bc3b9/checkpoint-summary.json)；不能替新metadata source背书 |
| 2fc package checkpoint | source fingerprint9ef；CPU Torch836 passed/1 skip，standard833 passed/4 skip | [2fc精确CI](../evidence/build-closure/G8-CI-2fc1c4d/receipt.json)；历史clean104-input及192诊断安装验收不代替d9的130-input归档 |
| 旧实数据/合成/仪器失败 | 旧物化拒绝、串行12-run子集、publisher取证、motion v1校准失败 | [旧G2](../evidence/build-closure/G2-report.md)、[writer仍UNKNOWN](../evidence/build-closure/G2-publisher-forensics-resumption/report.md)、[motion v1](../evidence/build-closure/G5-real-resumption/v1-report.md)；本轮新目录PASS不抹去这些失败 |

D01–D12实现与实证应结合[分维度状态](../evidence/implementation-status.json)及本轮交付报告读取；“数据PASS”“软件PASS”“trained模型/GPU/安全/论文PASS”分别签收。


## 2026-10-02 本地 DDC 接续

匹配 DDC 的留存 wheel/sdist 已恢复，Python 包指纹保持 1a3b49b51bd4f1f8ee9c4c3b3a7566a0cb0a9234ed941181822d5b4ea4413978。此副本不含 Git 历史，不代表恢复了合法后继分支。

当前执行状态与真实 PID、命令、数据审计见 [接续游标](../evidence/implementation-status.json)。历史软件、模型和运动证据不转授为本轮科研签收。总计划见 [完整交付计划](RSWM_COMPLETION_MASTER_PLAN.md)。


### 本轮实际执行终态（2026-10-02）

当前 DDC 专用环境已离线安装留存 wheel，实际包指纹匹配。M01 物化与独立审计均 exit 0：2 世界、8 I0、32 任务、0 训练 clips；origin0 的 8 包子集在审计目录外保存。恢复读回校验完整文件集合及哈希，没有重启已完成物化或审计。

M02 安全原源码准备 exit 0，仍为 PREPARED_SOURCE_ONLY：没有 runtime config、没有权重加载、真实模型调用和训练更新均 0。M05 当前 12/120 原源 bytes/SHA/严格解码已核验，尚缺108幅、8,105,049,513 bytes。M04 现有 metric24 0/24，原曝光 ledger/正式配置和真实 feature 实值未恢复。

当前游标 M02 / EXTERNAL_INPUT_BLOCKED；HANDOFF_REQUIRED 已保存，本地无关联运行 PID。Git 历史尚未恢复，因此无新源码提交。最小输入和下一条安全命令见 execution state；L1/L2 与总目标均未签收。所有旧失败、首次参考、冻结样本和成本口径保留。


### M10 可独立推进材料实际完成

已回读四篇既定近邻最新精确原文版本并核对六个旧稿包的主tex身份；旧性能数字不导入当前研究。Figure2公开／私有边界与Figure3固定dev2首次再入已自动生成，PNG和独立SVG栅格均经视觉检查。Table2 CSV／JSON／LaTeX片段已自动生成并独立读回，明确计划量与当前实测量；LaTeX未编译。收据见 [M10当前交付](../evidence/M10-paper-materials-delivery.json)。

这些是部分论文材料，正式模型表图、真实训练／测量／复现和论文实证结论仍未完成。首个LaTeX导出脚本转义语法错误与修复成本保留，未降低原Benchmark门禁。下一执行节点仍为M02，实值输入未补齐，当前无模型／训练作业，不新增同类wrapper或重复基线测试。


### 连续三轮外部阻断已复核

OWNER_INPUTS 仍为未填写模板，无实际目标会话、真实权重及使用决定、有限额度、原正式仪器输入或 DDC Git 历史。当前本地无关联运行PID；远端作业未提供可核查句柄。已完成的M01与部分M10材料保持，不重跑或重复打包。连续三轮同一外部条件已复核，目标控制状态已确认 blocked，总目标仍未完成。输入恢复后重新开始阻断审计计数并从M02接续，不重置成本或首次参考。游标持久化首次工作目录错误的失败与成本保留，修正后已写回。
