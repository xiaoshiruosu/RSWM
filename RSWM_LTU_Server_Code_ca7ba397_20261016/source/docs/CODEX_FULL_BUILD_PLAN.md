# RSWM-Bench：交给 Codex 的完整代码构建与验收计划

**计划版本：`rswm-codex-full-build-plan-v1`**  
**唯一目标仓库：`yedda-tv/rswm-1`**  
**本次核对的 Phase A 代码基线：`3f733e6880c296c43a40fe9535ab4f69e0025687`**  
**用途：在现有代码上修复、整合和完成整个 Benchmark 代码库，不是重新设计基准或重新写一套平行项目。**

> 给 Codex 的首要要求：先复现并修复现有审核中的 R01、R02、R03，再解决 A01–A03，完成真实 smoke12 的 Phase A 验收；随后逐阶段完成分支整合、测量验证、真实模型、LR64、隔离与复现发布。不得只新增接口、状态字段和 README 就宣布完成。

## 0. 如何使用本计划

将本文件保存为工作仓库中的 `docs/CODEX_FULL_BUILD_PLAN.md`。先不要覆盖现有根目录 `plan.md`；保留原设计与历史实施记录，完成一个阶段后再更新根计划中的**实现状态**。

本文件分清三类内容：

- **已核对事实**：来自本次回读的 PR #1/#2，以及附带的 Phase A 审核和复现结果。
- **冻结研究定义**：来自项目已经确定的 `docs/BENCHMARK_DESIGN.md` 和历史方案，不能为了方便实现而改变。
- **待实施工程要求**：本计划规定的修复策略、版本化字段、模块和验收；不表示这些功能现在已经存在。

后文新增文件名和新 schema 是开发目标。已有等价模块应就地补齐，不为满足文件名而复制一套实现。代码接口、资源定义和统计口径发生变化时要升级软件或指标协议版本；核心查询版本不随普通工程修复改变。

### 0.1 必须先阅读的材料

按顺序读取：

1. 本文件，以及 `RSWM_PhaseA_Completion_Audit_3f733e6.md`。
2. `RSWM_PhaseA_Audit_3f733e6.zip` 内的复现脚本、固定源文件和 `evidence/resource_reproductions.json`。
3. 当前仓库的 `README.md`、`plan.md`、`docs/BENCHMARK_DESIGN.md`、`docs/PHASE_A.md`、`docs/BENCHMARK_PAPER_BLUEPRINT.md`。
4. `suite.py / resources.py / aggregate.py / export_tables.py / runner.py / replay.py / score_rgb.py / stats.py`。
5. `data.py / geometry.py / tasks.py / baselines.py / video.py / paired.py`、`scripts/hf_materialize.py`、全部现有测试。
6. 研究扩展分支中的相机契约、Frontier、数据探针和对应测试，再决定如何整合。

审核包的 `reproduce_resources.py` **固定导入旧版本资源代码，并用断言证明旧问题仍能触发**。它不是修复后的回归测试。先用它复现原行为，再将测试逻辑改为导入当前工作树，断言修复后的正确结果。不要改掉旧证据或把原脚本“仍能复现 bug”写成修复成功。

### 0.2 本轮任务授权与边界

Codex 的任务是修改代码、增加测试、运行可执行验证、更新文档、创建分支和提交。默认在开发分支中完成；向 `main` 合并、改变仓库权限、删除已有实验数据、下载超出预算的大模型或运行大额 GPU 实验，必须有明确授权。

网络、GPU、真实权重或旧源码暂时不可用时，继续可完成的纯工程任务，但把真实验证单独标为未完成；不得用 mock、随机补全或伪造文件补成成功。

---

## 1. 开始时的真实仓库状态

### 1.1 本次回读确认

| 对象 | 回读到的状态 | Codex 的处理方式 |
|---|---|---|
| `main` | PR 当前基线为 `3e22673be4d6656990312665b3a0b6932c82f112` | 开始时再次 fetch 核对，不假定永久不变 |
| PR #2 | `feature/phaseA-suite-resource-aggregation`；HEAD `3f733e6880c296c43a40fe9535ab4f69e0025687`；Open、Draft、未合并 | 这是 Phase A 的实际实现起点，不从旧 main 重写 D03/D04/D07 |
| PR #1 | `research/review-20260918`；HEAD `656ed3fd5b7aa629e6544d4972f0daec996e5456`；Open、未合并 | 含相机条件、Frontier、缺失界与来源探针；稍后审查整合 |
| Phase A 已有验证 | 记录为 181 passed、2 skipped；合成/CPU 测试 | 不等于真实 smoke12、真实生成权重或 GPU 验收 |
| 其他仓库 | 用户报告 `yedda-ct/rswm-benchmark` 有 Phase 1 实数据工作 | 未取得具体源码和资产身份前，不能将报告直接移植成当前仓库的完成状态 |

PR 的 `merge_commit_sha` 可能是 GitHub 的测试合并引用，不能当成 `main` 已合并。两个 PR 各自 CI 通过不证明组合版本通过。181、253、231 等历史数字属于不同代码树和范围，不相加、不当作新版本测试数量。

### 1.2 当前已经存在的能力

D03 已经调用真实 `run_task()`，保存独立 attempt，处理多种模型异常，运行独立 replay，并有 Linux 协调器锁。D04 已经接到 reset、generate、snapshot、restore 和 suite 阶段，不只是数据类。D07 已有 expected 左连接、世界/城市 RGB 汇总、共同可测子集比较及 CSV/JSON 导出。

因此本轮是**补齐和修复现有链路**，不是重新创建几个同名空文件。

### 1.3 必须先关闭的审核事项

| ID | 已有证据 | 问题 | 第一阶段要求 |
|---|---|---|---|
| R01 | 已做模拟 CUDA 的隔离复现 | GPU 仅测到部分阶段，attempt/总表仍可能 `complete=true` | 字段级测量完整性贯穿阶段、attempt 和总表 |
| R02 | 已做 CPU 复现 | 两个 `np.frombuffer` 共享 16 bytes，却记录 32 bytes | 支持外部 buffer owner 去重，或明确计量不完整 |
| R03 | 已做 CPU 复现 | pre-counter 采样失败被当 0，真实累计 12 记成 22 | 未知前值不取 0，累计区间不重复计数 |
| A01 | 已读代码与文档 | 默认最新 attempt，与此前首 attempt 主分析约定不同 | 新计划明确冻结首 attempt 主分析和重试预算；旧版本不得静默重标 |
| A02 | 静态代码发现，尚需集成复现 | 损坏 outcome/resource JSON 可能中断整套导出 | 逐 run 元数据故障隔离；分数与资源状态分离 |
| A03 | 已读导出实现 | CSV/JSON 存在，但 LaTeX 尚未输出 | 自动生成论文表格，保留 null/覆盖/来源 |
| A04 | 已读实现和文档 | suite 当前是 Linux/POSIX；内存与 GPU 字段有特定范围 | 保留范围说明，不能扩大平台和计量完成声明 |
| A05 | 已核对两个 PR | Phase A 与研究扩展仍分散 | 基线修复后整合，再做组合回归 |

---

## 2. 不可改变的 Benchmark 合同

### 2.1 数据与像素

核心版本：`rswm-inria-120-v0.1`。主源：`blanchon/INRIA-Aerial-Image-Labeling`。

指定历史 revision：`ad8315c82c8a3021b93a7d52922d19ea76201f60`。已有正式清单使用另一 revision 时先检查差异，不能覆盖；revision 会参与世界身份和任务种子。

只使用原始 `data/train/images/*.tif` 与 `data/test/images/*.tif`，排除 gt、预览和缩略图。必须验证真正 TIFF 格式，不接受 PNG 改名为 TIFF。不要用 `.convert('RGB')` 掩盖灰度/波段错误。

```python
# 冻结像素转换，必须先检查 source.shape/dtype/mode/format。
center = source[452:4548, 452:4548, :]
sums = center.reshape(2048, 2, 2048, 2, 3).sum(
    axis=(1, 3), dtype=np.uint16
)
world = ((sums + 2) // 4).astype(np.uint8)
```

输入 5000² RGB uint8；世界 2048²；名义 0.6 m/px。不能恢复旧 1 m/px，不能逐帧曝光、旋转、直方图处理或有损重编码。地理标签不存在就记录缺失。

### 2.2 地域和选择

| 角色 | 地域 | 每地数量 | 合计 |
|---|---|---:|---:|
| train | austin/chicago/vienna | 20 | 60 |
| dev | kitsap | 12 | 12 |
| metric_calibration | bellingham | 12 | 12 |
| metric_validation | bloomington | 12 | 12 |
| test_unseen_city | sfo/innsbruck | 12 | 24 |

城市内按 `SHA256('rswm-inria-120-v0.1/' + source_path)` 排序取前 k；保存完整 inventory、选中清单、字节、可用 SHA256、源 split 和 Benchmark role。唯一预置别名 `sanfrancisco -> sfo`。禁止根据图像好看、配准容易或模型表现换图。

### 2.3 视口与任务

```text
world = 2048 × 2048
viewport_hw = [240, 416]
origins = [(960,960),(1088,960),(960,1088),(1088,1088)]
q0 = (0,0)
qx,qy ∈ [-704,704]
x向右，y向下
normalizer = sqrt(416² + 240²) = 480.2665926337163
crop = world[cy-120:cy+120, cx-208:cx+208]
```

| 族 | 固定规则 | 唯一新输出/起点 |
|---|---|---:|
| F2 | 32步，2/3/4段，每段≥6，分量16/24/32，转角±45/90/135°，路长1–2.5个对角线，offline | 32 |
| F3 | 基础分量8/12/16，路长0.60–1.25个对角线，large=2small，整对接受、两流独立reset | 64 |
| F5 | 16/16/32/16/16步，B中心128²目标，6次online16，完整端点集合以冻结设计为准 | 96 |
| F7 | 32共同前缀，父状态分叉，两条各64后缀独立恢复，10次online16 | 160 |

F2/F3 最多 10000 候选；长路径最多 1000，失败停止、无 fallback。所有检查在最终整数坐标上做，相邻交叠≥0.65。D4只变命令、不旋转矩形视口。主版本 exact/offset 与 origin 的绑定不变；同起点对照作为独立研究。

### 2.4 首次提交、评分和权限

`first_event` 是首次**计划**曝光；首次缺失仍写 seen，参考永久无效，后续不得补写。目标逐像素在首次重新进入视口时评分，不只看最后 B。F7 分支分别对共同前缀，双支完整才给成对保持。

`K=A_t∩O` 对 I0；`P=(A_t∩U)\O` 对首次计划曝光槽位的有效提交；`N=A_t\(U∪O)` 没有由 I0 唯一确定的真实像素答案。

模型只获得 I0、公共规范和当次允许查询。online16 不给整个未来列表，不在跨块时改任务原点；允许当前块内联合生成。隐藏 world、绝对 origin、city、目标标签和真实未来不传给普通 adapter。Oracle 是评测器诊断例外，必须单独标记。

前几个阶段不得修改 `geometry.py / tasks.py / score_rgb.py` 的数学或抽样语义。D10 的通用世界适配若必须抽取公共函数，另设兼容性提交，证明核心查询、像素和整数充分统计量不变；不得偷偷更改冻结测试。

---

## 3. 完整构建的定义与阶段顺序

**“代码可 import”“CLI 能显示 help”“单测通过”“真实数据通过”“真实模型通过”“论文结果已完成”是不同层次。**

最终代码需要支持：

```text
来源版本与权限记录
→ raw-file校验
→ world/I0/query/label构建
→ 物化产物审计
→ 固定模型与expected运行计划
→ 受限调用、attempt与首次台账
→ 独立提交回放
→ RGB / motion / first-commit quality / resource
→ 世界与地域汇总
→ CSV/JSON/LaTeX及figure data
→ 完整复现、兼容性和发布验证
```

### 3.1 里程碑

| 阶段 | 对应队列 | 主要交付 | 放行门 |
|---|---|---|---|
| G0 基线审计 | 全局 | 工作树、分支、依赖、冻结文件与旧问题复现 | 基线测试和差异可追溯 |
| G1 Phase A 修复 | D03/D04/D07 | R01–R03、retry、元数据容错、LaTeX | 新负例能在旧版失败、在修复版通过 |
| G2 实数据诊断闭环 | D01/D02 + Phase A | 真实 smoke12 三诊断模型与完整导出 | 每模型192包/16896输出；错误场景分母正确 |
| G3 研究分支整合 | D08基础/M02/来源探针 | Frontier、相机契约、计数器与registry接通 | 组合代码全回归，不按两边测试数相加 |
| G4 真实基线小规模接入 | D08 | 本地真实权重、统一模型协议、开发集pilot | 跑通实际生成和状态，测得真实成本 |
| G5 仪器冻结和验证 | D05/D06 | calibration/validation、特征/参考协议 | 测量有效性有独立实景证据 |
| G6 核心正式运行 | D01/D02/D03–D08 | core120审计、测试集执行和汇总 | 任务/模型/指标/seed预先冻结 |
| G7 机制与小地图 | D09/D10 | 同起点、表示/记忆消融、LR64 | 独立study，Inria兼容回归 |
| G8 隔离与复现发布 | D11/D12 | 受限worker、安装包、复現脚本、文档 | 干净环境重建和受控权限测试 |

D05/D06 的代码和 calibration 数据工作可在 G4 并行准备。正式 test 的生成和排名在指标协议冻结后进行；不能看 test 结果调测量参数。D11 必须在接受不可信第三方代码之前完成，不因内部合作式实验可运行而省略。

### 3.2 状态必须分维度记录

保留 D01–D12 的任务编号，但不要只填单个完成枚举。建议 `evidence/implementation-status.json` 每项包含：

```json
{
  "task": "D04",
  "code_state": "IMPLEMENTED",
  "synthetic_validation": "PASS",
  "real_data_validation": "NOT_RUN",
  "real_model_validation": "NOT_RUN",
  "real_gpu_validation": "NOT_RUN",
  "integration_validation": "NOT_RUN",
  "publication_acceptance": "NOT_READY",
  "scope": "示例字段，不是当前任务的实际结论",
  "evidence_paths": [],
  "blocking_items": []
}
```

这些状态仅在真正运行后更新。GPU mock 通过只影响 synthetic，不影响 real_gpu。真实影像上的 Oracle 不影响 real_model。第三方源码入口存在不等于权重恢复或训练完成。

---

## 4. G0：基线检查和安全分支工作

### 4.1 必做操作

在已有克隆中先执行只读检查：

```bash
git status --short
git remote -v
git rev-parse HEAD
git log -8 --oneline
git fetch origin
```

若有用户未提交改动，先报告并保留，不做 `reset --hard`、`clean -fd` 或自动 stash。工作区干净时，从 Phase A 分支创建修复分支，例如：

```bash
git switch -c codex/phase-a-audit-closure \
  origin/feature/phaseA-suite-resource-aggregation
```

该命令只用于尚未存在的新分支。若远端已在 `3f733e6` 后更新，检查实际 diff；使用新的后继提交，不退回旧 SHA 以匹配文档。保存 `git merge-base` 和来源关系。

### 4.2 环境与证据

按实际 `pyproject.toml` 建隔离环境，先安装已有开发依赖：

```bash
python -m pip install -e '.[dev,data,motion]'
python -m pytest -q --junitxml=evidence/build-closure/baseline-tests.xml
```

保存 Python、操作系统、CPU/GPU可见性、依赖版本、commit、工作树差异。新增 GPU/模型依赖以选定实际后端为准，不无约束升级整个环境。源文件和脚本路径含空格时必须正确引用。

### 4.3 源码审计产物

创建 `docs/BUILD_GAP_AUDIT.md`，逐项写：现有函数、调用关系、已知缺陷、新增设计、测试入口、真实资源是否可用。冻结文件生成 SHA256/Git blob 清单。检查现有测试名称保留情况；不同分支数量下降不能先入为主判断删测，应逐文件比较。

提取旧审核包到独立只读证据目录，复现 R01/R02/R03，结果标 `expected_old_failure=true`。接着为当前工作树建立回归测试，先确认这些正确性断言在未修复版本失败。

**G0 产物：** `baseline.json`、原测试 XML、冻结文件清单、旧缺陷复现、差异报告。没有真实数据也能完成本阶段。

## 5. G1-R01：修复资源完整性从阶段到总表的传播

**目标文件：** `resources.py`、`aggregate.py`、`export_tables.py`，新增 `tests/test_resource_integrity.py` 或补入现有测试模块。

### 5.1 新计量合同

建议新输出使用 `rswm-resource-v3`。不是简单改版本字符串，而是每个字段具有独立测量状态。例如以下是**演示结构，不是实际测量**：

```json
{
  "schema": "rswm-resource-v3",
  "measurements": {
    "gpu_synchronized_model_wall_seconds": {
      "value": null,
      "measured_sum": 0.125,
      "applicability": "applicable",
      "eligible_units": 2,
      "measured_units": 1,
      "complete": false,
      "aggregation": "sum",
      "unit": "seconds",
      "reason_codes": ["GPU_POST_SYNC_FAILED"],
      "scope": "已进入的模型阶段，同步后的host墙钟；不是kernel时间"
    }
  }
}
```

要求：

- 完整值 `value` 只有在该字段所有适用计量单元均可测时给出；不完整时为 null。
- `measured_sum` 或 `observed_peak` 可以保留有效部分，但必须明确是部分和或观测峰值，不能改名为完整总量。
- `eligible_units` 在进入待测阶段时登记，不能只在测量成功后增加。
- CPU 运行的 GPU 字段 `applicability=not_applicable`，所有数值为 null；不生成“GPU=0”的结论。
- 实际运行未进入的阶段与进入后测量失败分开。`scheduled_calls / entered_calls / model_calls / measurement_eligible_calls` 分别记录；不要把初始化失败后守卫返回缺失图的请求计为真实模型推理。
- `measurement_complete` 仅表示其声明范围内计量完整；另保留 `execution_complete`。失败任务的已执行部分完全测到，也不等于任务完整执行。
- 老 `rswm-resource-v2` 只有一个非 null 标量时，不能凭此推断 complete。兼容读取应标 `legacy_completeness_unknown`，不得默认为完整。

### 5.2 时间边界

每个阶段记录唯一 `phase_id`、可选父阶段、stream、call、event 范围、墙钟时间和错误代码。明确区分：

| 时间 | 范围 |
|---|---|
| `attempt_wall_seconds` | attempt 的系统端到端时间，注明是否包含校验、IO、replay和最终文件落盘 |
| `model_load_seconds` | 模型实际构造/权重恢复，不与输入文件哈希检查混为一谈 |
| `initialize_seconds` | model.reset；Eager 提前补全属于这里的实际成本 |
| `generate_seconds` | 模型调用范围；原生输出转换属于模型还是评测器必须明确 |
| `snapshot_seconds` / `restore_seconds` | 实际状态导出/复制/恢复 |
| `output_validation_seconds` / `evaluator_io_seconds` | 规范输出检查、PNG和日志写盘 |
| `replay_seconds` | 独立评分回放，不并入模型推理时间 |

GPU 适用模型阶段在前后同步。记录 host 执行耗时和同步后 host 墙钟时，使用清晰命名；不要把同一个同步后的数值分别说成纯 CPU 时间和 kernel 时间。同步前/后失败均保留普通墙钟，GPU字段为部分或不可测。

父阶段包裹子阶段时，不能把父子时长再次相加。选择互不重叠的原子阶段作总和，或按区间并集计算；明细可同时保存嵌套阶段。一个 GPU `generate` 内部的 64 次 fill 不得在总时长中重复计算。

### 5.3 聚合修复

`resource_totals()` 必须读取每个字段的测量状态，不能再用 `value is not None` 代替完整性。

按模型/研究协议输出：

```text
expected_runs
observed_runs
started_attempts
resource_readable_attempts
eligible_measurement_units
measured_units
complete_attempts
full_total / measured_partial_sum
full_peak / observed_peak
```

求和字段的完整总值需要所有参与 attempt 都具备完整范围；否则 full_total=null，部分和单列。峰值也区分已观测最大值与整个声明范围的可信峰值。进程RSS仍不能变成模型独占内存。

资源未生成的已启动 attempt 仍进入相应分母。完全未开始的 run 不虚构 attempt 文件，但保留 expected-run coverage，不能因为只统计已经创建的目录就声称整个计划资源完整。

### 5.4 必须通过的测试

- 全阶段同步成功，计量完整。
- 先成功后同步失败：部分值保留、full value=null、complete=false。
- 先失败后成功：不能被后续成功覆盖成完整。
- pre-sync失败、post-sync失败、峰值读取失败分别覆盖。
- CPU无GPU：字段null、not_applicable。
- 某stage根本未进入：与已进入但测量失败分开。
- 两个attempt，一个完整一个部分：总表full value=null，测量覆盖正确。
- 旧v2数据读取不被强制升级成完整。
- 嵌套phase不重复累计；资源采集异常不替换模型原始异常。

**验收：原审核R01不再出现，错误完整性不能通过CSV/LaTeX重新变成“完整”。**

---

## 6. G1-R02：修复共享缓冲区去重，限定真实可测范围

### 6.1 不能混用三种内存

分别命名：

1. `logical_array_bytes`：模型显式状态数组的逻辑字节，可按别名重复引用。
2. `unique_supported_storage_bytes`：已支持 owner 类型的唯一底层存储字节。
3. 进程RSS/显存峰值：含分配器、Python、依赖等，不等于前两者。

不能把“当前输出 RGB 图大小”当完整状态；不能通过 deep-copy 全模型测内存；不能把未知 Tensor/缓存忽略后称精确总量。

### 6.2 建议算法

先保证 NumPy owning array、普通 view、`bytearray`、`bytes`、`memoryview` 支持正确。沿 ndarray.base 和 memoryview.obj 追到支持的根 owner，保留引用，按本次采样内的 owner 身份去重。

- 对 owning ndarray + 两个slice，计 owner 存储一次，不只计最大单slice。
- 两个 `np.frombuffer(shared_bytearray)` 指向同一16-byte owner，计16，不能计32。
- 同一 owner 的不重叠view仍计该 owner 一次；不同owner内容相同不能去重。
- 支持多层view、只读buffer、零长度array、负stride和非连续slice；不能依据切片的简单起止区间错误推断真实分配。
- object dtype、未知第三方buffer、无法确证的映射存储，返回不完整原因；可保留已支持部分，但完整状态总值为null。
- 若以后支持Torch，按实际选定版本的storage接口实现独立测试；当前不识别的Tensor不得被忽略。不同device的存储不能合并计数。
- `np.memmap`/同文件不同映射不宣称共享物理内存；要么明示计映射范围，要么留作不支持。
- 不用不安全内存读取去验证owner；无法证明时保守标未知，不造精确值。

保留 `state_scope`：包含哪些显式状态、排除权重/Python对象/不可见后端缓存。适配器自报状态还不是安全证明；仅表示其声明契约。

### 6.3 测试和验收

原16-byte共享外部buffer负例修复；原普通200-byte owning-array/300-byte逻辑视图正对照继续通过。新增交错views、不同owner、嵌套容器、循环引用、unsupported、混合supported/unsupported场景。

状态采样一次失败后，不能保留此前标量并继续标为完整峰值。后续恢复采样只能增加观察证据，不能消除未知时段。F7父状态、分支动态地图和冻结权重分开记录。

---

## 7. G1-R03：累计计数器不得以未知前值为零

### 7.1 读取合同

将 `_counters()` 从“失败返回空字典”改成有类型的读取结果，至少包含 `status / values / counter_epoch / error_code`。区分未实现、尚未初始化、暂时读取失败和正常零值。

原则：`unknown != 0`。前采样无效时，不计算 `after - 0`。累积计数器必须有稳定的生命周期身份；F3 reset 和 F7 restore 不允许悄悄让累计真实成本回退。

### 7.2 不重复计区间的两种合法实现

首选保存最后一个可信累计读数及其边界，区分：

- **phase delta**：前后读数都有效且同一epoch，才可精确归属该阶段。
- **lifetime observed total**：可用同一epoch的可信累计终值和已知初始值计算；跨丢失边界的增量只能标为跨阶段，不能同时再加到两个phase。

对于审核例 `0 → 10 → pre_read失败 → 12`：

- 若契约保证同一单调累计epoch，允许总量12，第二阶段单独归属不明要明确记录；
- 若不能证明epoch稳定，则完整总量null、已测部分10、不完整；
- **任何情况下都不能得到22。**

计数器下降、key临时消失、增加新key、模型替换、读取类型错误均需显式处理。合法新epoch必须由adapter声明，不凭值变小猜测reset。没有已知初值时不能把首个累积数全算为当前阶段。

### 7.3 模型计数与请求计数分开

记录 `planned_output_slots / requests_issued / model_calls_attempted / valid_outputs / native_frames / context_frames / discarded_frames`。调用失败可能消耗实际计算但没有有效图像；守卫因为reset失败直接返回None不等于后端执行了一次生成。

原生帧无法测得时为null。禁止用 `max(0, native-requested)` 掩盖不相容的计数定义；先验证字段属于同一scope，给出精确“未评分原生帧”的定义。模型返回的条件锚帧、内部上下文重复编码和新预测帧不要混成一个数量。

### 7.4 测试和验收

前采样失败、后采样失败、连续失败、key缺失、counter rollback、epoch切换、F3两次reset、F7两次restore、失败的Eager fill、collector重复finish都要测试。重复finish不能再次累计；重新导出不能额外算模型成本。

**验收：R03正确关闭，所有可测累计值与显式后端操作记录一致，不可测部分在资源表中仍可见。**

---

## 8. G1-A01：冻结 retry 策略，保护首轮主分析

### 8.1 运行身份与尝试身份

运行样本身份不包含attempt：

```text
sample_id = digest(study, task_id, model_manifest_hash,
                   inference_seed, protocol, metric_protocol_id)
```

attempt是这个样本的一次执行记录。训练seed属于模型配置；推理seed是重复生成条件；attempt不是新的seed，也不是新的独立测试样本。

### 8.2 新计划默认

建议新schema `rswm-suite-v3` 使用结构化策略：

```json
{
  "retry_policy": {
    "primary_selection": "first_attempt_only",
    "max_attempts_per_run": 1,
    "automatic_retry": false,
    "allowed_retry_statuses": ["CRASHED", "INTERRUPTED", "FAILED_LOAD"],
    "secondary_selection": "latest_attempt_within_budget",
    "cost_scope": "all_started_attempts"
  }
}
```

该对象进入plan hash。默认不重试；事前明确最大次数为2等时才允许执行新attempt。允许重试哪些状态由实验配置冻结；不能根据测试分数临时扩大预算。完整执行但图像差不属于失败重试。

`first_attempt_only` 指 attempt-000；若其目录/结果缺失，不跳到首个“现存且成功”的attempt。主表始终保留首轮表现。后续成功另表为operational recovery，不改善主表历史表现。

需要 `fixed_attempt` 或受限最新attempt时必须是显式的独立分析策略；生成新的分析配置及hash，保留旧表。不得混到同一主排名。

### 8.3 旧v2计划不能被静默改义

旧计划文档明确选择latest。读取旧计划时只有两种路径：

1. 兼容地复算旧口径，输出显眼 `legacy_latest_attempt_policy=true`；
2. 基于不可变旧attempt产物新建独立分析计划，写明首轮规则和映射，并验证身份，不修改原plan/收据。

不要重算旧plan hash后覆盖文件冒充原计划；代码指纹变化也不能随意绕过。全新实验用v3，旧产物可审计地保留。

### 8.4 必须通过的测试

- attempt0失败、attempt1成功：主表仍失败/缺失，恢复表成功，两次成本都计。
- attempt0成功：默认不允人为重跑争取更低误差。
- attempt0缺失但attempt1存在：不以attempt1代替首轮。
- 第二次失败不能覆盖第一次完整结果的主分析。
- 超最大次数拒绝并保存事件，不创建半个新attempt。
- 非COMPLETED不等于可以无限重试；策略字段变更禁止resume旧plan。
- 序号数字排序，不能用字符串顺序在attempt-999/1000之后选错。
- crash / Ctrl-C / SIGTERM / SIGKILL 的记录与下一次恢复可区分；不凭最后PNG恢复模型隐藏状态。

---

## 9. G1-A02：损坏元数据的逐run隔离与可信结果状态

### 9.1 分开三个维度

```text
execution_status   # 模型执行与进程是否正常
submission_status  # 任务、I0、日志、PNG、收据能否独立验证
resource_status    # 资源JSON与测量范围是否完整可信
```

不再让一个 `COMPLETED` 布尔值同时替代三者。状态至少覆盖 NOT_RUN、RUNNING、COMPLETED、INCOMPLETE、FAILED_INPUT、FAILED_LOAD、FAILED_INITIALIZATION、FAILED_GENERATE、FAILED_SNAPSHOT、FAILED_RESTORE、INTERRUPTED、CRASHED、FAILED_REPLAY、METADATA_CORRUPT；可以分字段表达，须有schema和状态转换测试。

### 9.2 信任层级与处理表

| 出错位置 | 处理要求 |
|---|---|
| 根plan/expected损坏或hash不匹配 | 根信任已丢失：停止运行/分析，输出诊断，不伪造计划分母 |
| 单个outcome损坏/必需身份不匹配 | 标记该run评分不可验证，使用冻结几何分母；保留文件；继续其他run |
| 单个resource JSON损坏 | 只将该run资源标未知；若提交和独立收据可验证，RGB分数仍可使用 |
| 旧失败attempt资源损坏 | 资源总量不完整；不能使其他run的RGB导出失败 |
| 当前PNG、I0、任务或日志篡改 | FAILED_REPLAY，完整分母保留，有效评分按冻结故障规则处理 |
| 缺少head/receipt | 不自行用当前日志重建头并宣称独立认证；标记证据不足 |
| 不属于expected的运行目录 | 隔离并显式报unexpected；不纳入分数，也不能无日志忽略 |
| 半写/非法JSON、重复键、NaN/Infinity | 拒绝对应对象，产生稳定结构化错误代码 |

单run的异常捕获不能包住整个根计划验证后直接全部填空；也不能仅用 `except Exception: continue` 删除失败。输入解析、状态验证、资源解析和replay分别定位错误。

若将来实现从独立可信日志恢复坏outcome，要在单独repair流程中写新收据并保持原文件；本阶段默认保守，不自动“修复历史”。

### 9.3 输出与退出码

导出在根计划可信时尽力完成全expected表，同时输出 `integrity-summary.json`、错误条目与未测量范围。CLI 返回码建议：0=请求范围完整且校验通过；1=存在run执行/产物问题但完整分母结果已输出；2=配置/根身份/用法错误，不能形成可信分析。资源缺失是否影响发布门按声明measurement profile决定，不能隐藏。

`ANALYSIS_WRITTEN` 只能表示写出产物；`ACCEPTED` 必须检查要求的执行、提交、资源完整性。不能在有失败时生成无条件“全部通过”的 COMPLETE 标志。

### 9.4 测试

至少对outcome/resource/receipt/started/旧attempt JSON分别注入：截断、非对象、缺字段、错run_id、错attempt、错plan hash、重复键、NaN、外链路径。验证其他样本照常导出、坏样本分母保留、资源故障不误删已验证RGB。

---

## 10. D03 收尾：运行计划、索引、锁和原子性

### 10.1 计划内容

现有 build_plan 保留并补强，而不是丢弃重写。新计划至少绑定：source inventory/manifest、物化index、task与label、world文件/像素、I0、模型配置/权重、推理seed、训练seed、metric config、retry policy、runtime source、依赖环境和任务角色。

code fingerprint 必须递归包含实际执行的 `src/rswm_inria/**/*.py`；PR #1有adapters子目录，当前只glob顶层不能覆盖它。脚本、配置、外部后端checkout另存hash清单。排除运行产物和证据文件，防止生成一个证据就改变代码指纹。

校验plan不仅比较自哈希，还应检查：schema、唯一身份、expected与冻结tasks/models的一致性、合法role/city、32/64/96/160输出数、model配置、source/metric关系和必需文件清单。无密钥哈希只能检测相对已保留身份的变化，不能证明来源真实性。

### 10.2 模型配置不得“记录了但没执行”

补建或完善轻量 `registry.py`，实际调用已有adapter工厂。声明支持的配置字段；未知字段默认报错，不允许hash里有prompt而factory根本没传入。

记录 adapter类型、用途 diagnostic/real_generator、weight manifest、设备、采样步数、文本、数值消费方式、状态支持。普通factory只给公开模型输入；Oracle另设 evaluator-only入口，不给所有第三方factory传完整world再寄望其不读取。

本地checkpoint要求实际文件齐备、可恢复。没有model_dir的地图只能显式diagnostic_noise；不能注册成真实inpainting。模型复用/每任务重载策略也在plan中固定，加载成本和共享成本说明一致。

### 10.3 文件布局

保持现有submission布局，避免破坏replay：

```text
suite-root/
├── plan.json
├── expected.jsonl
├── coordinator.lock
├── runs/<sample_id>/attempt-000/
│   ├── started.json
│   ├── submission/             # 原task/I0/events/PNG/summary
│   ├── output-slots.json
│   ├── outcome.json
│   ├── resources.json
│   ├── replayed.json
│   └── artifact-manifest.json
├── heads/
├── receipts.jsonl
├── run-index.jsonl
└── integrity-summary.json
```

新attempt不能覆盖旧attempt。artifact-manifest绑定资源和结果文件；收据保存在attempt外，并支持导出到独立指定位置。相同磁盘上的另一个目录不是可信时间服务，必须在文档保留这一边界。

### 10.4 并发与中断

锁要在初始化或修改根状态之前正确获得，消除两个进程同时创建plan/expected的竞态；锁文件inode不能运行时删除。导出器读取活动suite时获取共享锁或一致快照，不能把尚在写入的attempt误判为不可恢复的最终失败。

JSON/JSONL使用UTF-8、strict JSON；原子替换或排他创建，必要时fsync文件与父目录。所有未完成写入有明确状态；不把临时文件当合法输入。不可变源文件与可重建派生索引分开。

资源采集/finish或磁盘错误可能再次抛异常，不能盖住原模型异常、导致完全没有结构化状态。尽力写独立故障收据；文件系统不可用时明确stderr/exit非零，后续恢复按未完成attempt处理，不编造完整记录。

### 10.5 路径和跨机器复现

当前计划含绝对路径，旧计划迁机需要重新冻结。新代码可增加根路径映射，但样本内容身份与本机locator分开；映射不可改变任何文件hash。不能为了可移植偷偷编辑旧plan。全部路径做允许根检查；拒绝 `..` 越界和不允许的symlink，不将临时带token的下载URL写日志。

### 10.6 新增CLI的目标，不代表现在已支持

保留 `suite plan / suite run` 和原参数；补充显式 `suite verify`、`suite inspect`、受限retry参数、模型配置验证。命令帮助与文档从实际parser核对，不先写不存在的功能。

完成后每项操作都要有CLI集成测试，覆盖中文/空格路径、空选择、未知model、重复seed、身份变更、活动锁和中断恢复。

## 11. D07 收尾：估计量、完整分母和论文自动表格

### 11.1 先冻结统计口径，不借修复偷偷换公式

当前实现的主误差是：世界内可测的 task/origin/seed 单元等权，然后城市内可测世界等权，最后城市等权；覆盖和缺失界保留所有适用计划单元。保留此定义，并在报告写出精确 `estimand_id`。

增加每seed结果和seed覆盖，但不要把“先各seed均值再等权平均”静默替换现有世界内单元等权；两者在缺失分布不均匀时不等价。需要新的seed-balanced估计量时单列版本与补充表，保留原表和差异测试。

源代码原有 `.stats.paired_city_summary` 可以复用；上游必须已构成每世界一个匹配值。不能把逐帧、起点或推理seed当独立地理样本。均值差方向固定为A-B，并写清低值更优还是高值更优。

### 11.2 分组键

至少包含：benchmark/config、study、role、model manifest、family、protocol、variant、stream、metric protocol。训练/开发/校准/验证/测试不得混为一组；不能只靠world_id碰巧不同防止混榜。

F3 small和large分列；F5 exact/offset分列；F7主流、a、b和paired是不同视图。核心与same-start扩展、LR64、诊断扰动分开。模型输入权限或数值通道不同必须在表中可见。

同一个source id被不同city/role绑定时拒绝；同一个model_id对应不同权重时拒绝或显式分组。不同metric config混入默认拒绝，不能只合并相同列名。

### 11.3 分数、覆盖与缺失界

对每个适用RGB单元保存整数充分统计量：`expected_pixels=e, valid_pixels=v, absolute_error_rgb=a`。

```text
conditional_mae = a / (3 × 255 × v)   仅v>0
coverage        = v/e                仅e>0
lower           = a / (3 × 255 × e)  仅e>0
upper           = lower + (1-v/e)    仅e>0
```

e=0：not_applicable、误差/覆盖/界为null。e>0且v=0：missing、误差null、覆盖0、界[0,1]。不要把这个描述性范围命名为CI或置信区间，也不要用它补写首参。

先任务层算界，再按与主表一致的计划权重聚合；不能用合并像素比替代等世界权重。`pixel_pooled_mae/coverage` 仅作为明确命名的补充量。

整城没有可测主误差时，等城市主误差为null，不把该城权重分给另一个城市。`test_unseen_city`的完整核心报告必须知道计划中应有SFO和Innsbruck；只跑部分城市时明确为subset，不冒充完整主榜。

### 11.4 唯一输出数不能跨metric或F7视图重复累计

新增唯一的 run/stream 完整性表，按 `(sample_id, stream, event)` 去重计数。当前metric细胞中同一 `planned_outputs` 会伴随多个metric出现，不能直接把这些行全部相加。

F2=32；F3 small32+large32；F5=96；F7 prefix32+a64+b64=160。F7的branch/paired metric是评价视图，不是额外生成输出。expected runs、output slots、world/city报告与资源native计数应相互核验。

### 11.5 配对比较

先匹配两方法相同world/task/origin/inference seed且同时可测的单元，再形成每世界值。保存全部计划世界、双方都失败世界、单方可测和共同单元覆盖。

独立world bootstrap按城市分层，匹配关系和seed重复始终随world整体采样。区间只反映固定城市下源瓦片重采样，不称全球地理泛化区间。没有有效配对不计算虚假p值或零差。

随机seed显式传入，stable排序后再采样；重复运行表格数值与hash应一致（排除声明的执行时间记录）。

### 11.6 LaTeX与机器可读导出

补齐但保留已有CSV/JSON：

```text
analysis/<analysis_id>/
├── aggregation.json
├── provenance.json
├── run-completeness.csv
├── cells.csv
├── worlds.csv
├── cities.csv
├── equal_city.csv
├── paired_comparisons.csv
├── first_attempt.csv
├── recovery_attempt.csv
├── resource_attempts.csv
├── resource_totals.csv
├── failure-summary.json
├── integrity-summary.json
├── figure-data/
└── tables/
    ├── main_control.tex
    ├── main_persistence.tex
    ├── coverage.tex
    └── resources.tex
```

`main_control.tex`在D05尚未通过时只能展示适用的命令对应RGB等已实现量，并显式标motion `NOT_AVAILABLE`，不能将RGB交叠MAE称图像运动估计。D06未冻结时质量列同样标待测。

LaTeX特殊字符必须转义，model_id不能注入命令；null渲染“—”并注明未测/不适用，不渲染0。浮点显示精度和舍入固定，原JSON保留原数值。表内覆盖、重试口径、计量部分和标记不得省略。

provenance应引用所有参与attempt的结果/资源/收据内容hash，而不只是最新replay结果；retry/estimation/filter配置也进入analysis identity。新增输出目录必须为空或为显式可验证复用，不覆盖旧表。

### 11.7 测试最低清单

不均衡城市世界数仍等城市权重；全缺失world/城市；seed部分缺失；not_applicable；F7仅一支失败；模型共同可测集合不同；重复/未知run；错world-city绑定；错model/metric hash；多metric输出去重；LaTeX转义；稳定排序；首轮失败后恢复不改变主表；资源半量不渲染为全量；导出存在错误仍有完整expected和非成功退出码。

---

## 12. D04 其余验收：范围、性能开销与后端状态

R01–R03修复后再补以下范围，不能只通过三个负例就宣布完整资源系统完成。

### 12.1 RSS与GPU内存

保留RSS为进程生命周期高水位，不称单任务增量。Linux/其他平台转换方法注明并测试；Windows原生不支持suite时在启动前明确报错，不产生半个运行目录。WSL/Linux可作为部署选择，不冒充原生Windows验证。

GPU峰值需绑定device、测量开始/结束、同步状态及allocator定义。共享权重常驻/每任务重载是不同profile，必须分开。释放模型或 `empty_cache` 不能被当成真实峰值为零的证明。

可增加每任务进程隔离的资源profile，使生命周期更易解释；但不得在没有实现和验证时把同进程高水位重新命名为独立任务峰值。

### 12.2 模型原生工作和显式地图

真实补全器的每次fill、生成核心面积、输入上下文、真实I0/生成历史/未知/padding分别记录。Eager生成却未被query的区域照常计。F7恢复后之前支路的所有实际工作仍计入成本。

相机视频记录实际原生输出帧、提交帧、条件/上下文帧和丢弃帧；模型内部额外运算不能按提交图像数反推。模型实现无法报告时保留null。

同一事件不能既从adapter累计计数器统计、又从Frontier operation_log再次加总。选择一个权威源，另一个作一致性核对，并报告不一致。

### 12.3 磁盘与采集开销

定义目录字节为逻辑文件长度还是物理占用；默认逻辑字节。包含哪些PNG/日志/快照、是否排除权重与源数据、采样是否在最终收据前，都写清。资源文件自身的大小避免递归定义；可保存final inventory并说明排除自身。

采集器关闭和开启时，在确定seed与模型下规范输出、首参台账和RGB统计应一致。observer错误不得改变模型控制流程或随机状态；采集本身的CPU/GPU同步开销明确属于测量profile。

---

## 13. G2：真实数据构建、D01/D02与Phase A实数据验收

### 13.1 先取得实际输入，不转移历史报告的完成等级

优先复用用户已经存在的真实smoke12，但要读取其manifest、source lock、world像素hash和构建审计；路径变化不表示数据内容变化，文件名相同也不证明身份相同。

`yedda-ct/rswm-benchmark` 的D01/D02改动只有在真实读到代码后才能审查移植。不可访问时列明缺失，不把231 passed或某条验收报告复制成当前版本证据。不要因为这项同步受阻就停止当前仓库的纯工程修复。

### 13.2 来源锁和首件步骤

继续沿用现有CLI，并检查当前parser参数。以下是已讨论且基线存在的构建入口：

```bash
python -m rswm_inria.cli freeze \
  --revision ad8315c82c8a3021b93a7d52922d19ea76201f60 \
  --out data/inria-lock/manifest.json

python -m rswm_inria.cli download \
  --manifest data/inria-lock/manifest.json \
  --out data/inria-raw --limit 1
```

首文件必须真正构建为世界并核对像素，不以下载成功代替。固定revision与完整inventory要相互一致；原可变main默认应修为明确锁定或要求参数。图像真实format必须TIFF；完整源SHA/size检查。PNG伪装TIFF、RGBA/灰度、多波段错误、空文件和截断文件分别有负例。

### 13.3 smoke12

```bash
python scripts/hf_materialize.py inria \
  --manifest data/inria-lock/manifest.json \
  --inventory data/inria-lock/source-inventory.json \
  --profile smoke12 --source-dir data/inria-raw \
  --out build/inria-smoke12 --download --max-source-gb 1.0
```

默认先审阅下载计划再授权实际网络操作。已存在源文件校验后复用；预算是选中源payload，不包含世界和输出。缺源、hash错、超预算一律停止，不替换样本。

smoke12使用train/dev四地域各正式排序前3幅，共9train、3dev。期望12世界、48个I0、192任务包。训练索引只含9个train世界，576 clips/19008真实视图事件。

### 13.4 物化审计器

若当前树尚无 `materialized.py / audit_materialized.py`，实现或审查移植；有则扩充，不重复造一个新审计器。

检查源→世界像素、世界→I0真实crop、task查询、private NPZ所有数组与hash、公开字段白名单、索引完整性、训练角色、重复文件、完整文件集合。使用独立慢速几何路径核对首次出现/最后可见/再入，而不只调用同一个标签生成函数自证。

重复构建比较像素和数组语义，不把NPZ容器时间戳不同误判为几何改变。world/source地理关系没有元数据时记录unknown；精确图像去重不等于地理无泄漏。

### 13.5 Phase A全smoke12诊断矩阵

按完整12世界×4起点×4族×1seed运行三种诊断系统：Oracle、Copy、StaticNoiseMap。不要把 `lazy-noise` 当成StaticNoiseMap同一种实现；需要时另作第四组。

| 对象 | 每模型 | 三模型合计 |
|---|---:|---:|
| 世界 | 12 | 相同12世界，不是36独立世界 |
| 任务运行 | 192 | 576 |
| 新输出槽位 | 16896 | 50688 |

三模型所有规范输出未压缩RGB约15.18GB；PNG实际大小另测。先检查磁盘和时间预算，数据和PNG留在批准的本地目录，不上传Git。允许先固定子集开发，但未运行全矩阵前不得把状态写成“完整smoke12通过”。

Oracle适用RGB误差0且覆盖1；Copy应完整返回但不应被硬编码成某个固定误差；StaticNoiseMap可以得到完美保持，这是指标边界，不是错误。每个run自动replay，planned/valid唯一输出计数与聚合一致。

另外用固定少量任务注入load/reset/generate/snapshot/restore失败、first exposure缺失、硬退出和retry。故障对照不混入正常三模型实验总量，不改变正常模型权重。

### 13.6 真实验收证据

保存suite/analysis配置、命令、环境、输入hash、每模型任务与输出数、失败分布、回放计数、资源计量覆盖、世界/城市表、首轮与恢复表、全部派生文件hash。只给stderr末行或一张截图不算完整证据。

`REAL_DATA_VALIDATED`只能写为“smoke12 + 诊断模型 + Phase A链路范围”。真实生成模型、真实GPU与正式test仍未完成。

---

## 14. G3：整合PR #1，不盲目合并两个绿色CI

### 14.1 整合流程

G1/G2放行后，在统一开发分支审查PR #1相对共同祖先的改动，合并或逐项cherry-pick；保留其来源SHA。先合源码与测试，再人工合并README/plan状态，不能直接以另一分支的旧文档覆盖新审计结论。

合并过程中只在工作分支操作，不自动合入main。若要用PR #2作为交付面，说明该PR将包含哪些依赖；否则建立组合PR，说明两个原PR如何处理。不要留下多个同名“最新方案”却不知道代码在哪个分支。

### 14.2 必须核对的具体兼容点

| 模块 | 要验证的接点 |
|---|---|
| Frontier继承WindowMap | Phase A已新增累计计数器；frontier日志不能重复计成本，snapshot不能回退实际计数 |
| AuditedWindowMap.resource_state | 真实/生成掩膜、地图和诊断缓存是否纳入支持范围；不复制冻结completer权重 |
| registry | 实际注册raster/frontier/lazy，不仅单独脚本能跑；诊断噪声与真实权重显式区分 |
| camera_conditioning子目录 | recursive code fingerprint必须包含；配置hash在训练、推理、报告一致 |
| missing_bounds辅助函数 | 与aggregate公式保持一致，避免两套边界处理不同；保留薄包装或合一权威实现 |
| source_probe | 仅候选检查，不被默认dataset构建自动下载或误入主榜 |
| runner/replay | 旧核心query、首次参考、分支统计不变；新计量不会改图像输出 |
| 测试 | 两个分支原有测试均保留，实际重跑后报告总数，不相加历史通过数 |

Frontier真实质量仍需同权重比较；虚拟相机数学parity不是LingBot后端复现。源探针header通过不是全像素/地理/许可验收。

### 14.3 放行证据

组合全pytest、CLI plan/run/verify/export、四族Oracle、Frontier与Lazy诊断、F7累计计数、条件幅度与时间抽样反例、真实smoke12至少受影响路径再验收。source/score frozen文件若变化，必须说明原因和数值兼容结果，不能只写“merge conflict resolved”。

## 15. D05：运动测量代码、校准和留出验证

**依赖：** 可信世界/任务、Phase A提交记录与角色锁。  
**建议文件：** 复用 `motion.py`，增加 `motion_protocol.py`、`scripts/calibrate_motion.py`、`scripts/validate_motion.py` 和对应测试。

### 15.1 输入和输出合同

估计器只接受两张规范图和固定配置，不接收期望q/delta、world、city、真位移或完整任务。外层评测程序在估计完成后用命令计算偏差。尺寸必须在进入估计器前按统一方法规范化，不按模型各自调resize。

每对输出：

```text
pair_id, source_world_id, event_pair
status: measured / ambiguous / insufficient_evidence / submission_failure / out_of_domain
content_translation_xy
viewport_translation_xy
confidence/peak diagnostics
configured_search_domain
boundary_hit
reference_displacement（仅受控真实裁剪验收记录）
command_error（仅外层对模型结果计算）
```

图像内容向左24，对应视口向右24；符号和xy顺序写正对照。不可测输出位移与误差null，不填零。

### 15.2 受控实景图像对

在Bellingham的12世界上构建固定pair manifest：不同方向、幅度、纹理、周期性、边界和低对比。可用每世界256对作为初始工程预算，但这是**待校准前冻结的建议数量**，不是已有实验结果或独立样本数。

先生成合法中心和真位移，再裁两张真实图；受控扰动单独写版本、参数和seed。包含反向、半幅、双幅、域内/边界/域外过冲、静止、重复纹理、缺图、形变和颜色扰动。并非所有扰动都代表真实生成器错误，应保持操作层解释。

只有calibration允许选择阈值/搜索范围；将配置、候选表和选择规则写入hash。不能因为某类场景困难就从源集合删除。模糊/重复纹理应允许弃权。

### 15.3 独立validation

在Bloomington使用冻结参数和未用于选择的pair集合，输出每世界误差与risk-coverage。规范像素容差、错误可信率和最低可测覆盖的接收条件应在验证前登记；不能预先伪造“达到99%可靠率”。

验证不合格时标 `DIAGNOSTIC_ONLY`，保存失败报告；需要重新设计测量时发布新候选版本并说明原validation已被查看，不能反复调同一validation后仍称独立一次验证。

### 15.4 与suite连接

对已验证PNG提交逐事件测量，保留缺失/错误槽位。连续有效边才能累计位置；断链后不使用命令真坐标重新锚定。F3比例仅在双方符合预注册可测条件且分母非零时给出，保留配对可测覆盖。

RGB对应误差与图像估计运动误差分别导出。输出看起来稳定、RGB重叠低误差不自动当成可测正确运动。将估计器置信度/搜索域风险列入结果，而非只给最后一列平均误差。

### 15.5 测试和完成标准

合成平移精确正对照、符号/幅度、静止、周期歧义、低纹理弃权、域外边界、缺失图、路径断链、F3零分母、冻结role和参数篡改测试。真实calibration和validation的执行证据不可由mock替代。

**D05代码完成**需要真实入口、严格配置、pair构建和分析；**D05测量验证完成**还需要独立实景验收报告。二者状态分开。

---

## 16. D06：未知区域首提交图块与冻结特征分布评价

**建议文件：** 复用 `quality.py`，增加 `features.py`、`quality_protocol.py`、`scripts/extract_features.py`、`scripts/evaluate_quality.py`。

### 16.1 先冻结几何选块，后读取模型图像

根据任务计划的N区域并集选择128²世界网格块，排除I0和不满足完整计划覆盖的候选。沿用已确定规则；若需每流上限或hash采样，写明确版本、namespace和稳定排序，所有模型共用一个patch plan。

不能按实际有效输出挑“正好能拼完整”的块；不能在某模型失败后换块。patch计划绑定world、origin、task lineage、stream、像素范围与首次曝光事件。

### 16.2 first-commit mosaic

从独立验证过的提交和原始首次计划台账组装，每个像素仅从first_event的槽位读取一次。first_event缺失就永久留空，不从后续返回帧补齐。

F7共享前缀的同一首提交样本不能当成两个独立新样本；分叉后各支独立首次生成的内容属于各自lineage，不能因为世界坐标相同就强迫RGB相同或跨支拼接。按lineage明确去重键，并写分支边界测试。

组装输出complete/missing mask、计划像素数、有效像素数、首次来源索引和patch hash。拼接缝可能体现不一致，所以mosaic指标不是单帧美观；Oracle走完全相同管线做参照。

### 16.3 特征配置不是留一个类名

实现至少一个真正可加载的冻结特征后端，记录：模型repo/local path、版本、全部必要权重hash、输出层、维数、色彩输入、resize、crop、normalization、dtype、设备和batch策略。具体骨干需要根据实际可用权重与许可确定；本计划不声称DINOv2、RemoteCLIP或Inception已被验证。

无权重时必须在预检查失败，不自动回退随机投影或未训练网络后仍称正式质量分。mock后端只在测试专用路径，输出artifact明确 `diagnostic_features=true`。

特征缓存按patch像素、完整性、特征配置和权重hash寻址，不能仅按文件名复用。缺失patch不编码为黑图当有效样本。

### 16.4 分布统计与真实参考

通用多项式MMD-U继续按数学定义命名；只有实现对应冻结Inception流程时才叫KID。无偏估计可为负，不能裁零美化。样本不足返回明确状态，不填0。

真实参考按照冻结几何采样规则、角色和世界/地理组划分；真实—真实的两组不能由同世界邻接patch伪装独立来源。按世界控制抽样权重，避免有更多完整patch的世界支配结果。原始图块相关性通过世界级重采样处理，不把全部patch数量当独立地理样本。

同时报告全计划覆盖、完整块数、缺失像素、各方法自己的完整样本指标，以及共同计划可评子集的敏感性结果。共同子集属于补充比较，不隐藏其缩小范围。

### 16.5 验证套件

Oracle相同前处理、真实—真实、固定噪声地图、平铺、模糊、颜色扰动、人工拼接缝、样本量曲线和可选第二特征骨干敏感性。不能强制某个特征对所有负例排序都符合预期；失败应促使收窄指标解释或更换候选并重新冻结。

单测覆盖首次缺失不补写、F7去重、缓存失效、dtype/shape、块不完整、特征维数错、NaN、样本不足、负U估计保留和世界抽样可重复。

**发布解释只能是选定特征和采样下的未知区域分布诊断，不是每栋建筑、道路或真实未来布局的正确率。**

---

## 17. D08：真实模型统一接口与开发集pilot

### 17.1 支持四类，不用诊断系统顶替

| 类别 | 必须实际完成 | 不合格的替代 |
|---|---|---|
| 无精确数值视频 | 加载真实权重，I0到实际视频，说明不消费数值q | Copy、随机图、预先保存完整真视频 |
| 数值条件生成 | q/delta真实进入后端条件路径，训练/推理一致，有尺度与时间映射审计 | API收到了q但丢弃；仅prompt写“向右”；用真世界裁图 |
| 整图补全 | 同一I0条件下真实补全有限世界，全部初始化区域计成本 | 噪声地图、Oracle、从服务器获取真实底图 |
| 增量地图 | 同补全器依查询填未知，使用自生成历史 | 每块喂真实上一帧、漏计重新生成和缓存成本 |

真实系统列表由权重可用性、许可证据和适配可行性决定。计划中的LingBot/Wan、Genesis、RemoteBAGEL等是来源/候选名称，不代表通用Diffusers接口已经复现它们。必须逐一建立模型manifest及验收报告。

### 17.2 接口与capability

保留现有reset/generate/snapshot/restore语义，新增可选resource_state/resource_counters/capabilities，不大改核心runner。模型公共输入对象只含允许字段，禁止task完整字典/标签/源城市泄漏。

capability明确offline32、online16、snapshot、图像规范化、输入数值通道、额外帧和状态可测范围。不支持任务族时，在计划阶段显式标能力缺口，不能偷偷换成另一个便宜实现。

模型输出的uint8、HWC、RGB、尺寸和native→canonical转换统一审计。不能经MP4压缩后再取评分帧。转换是方法组成部分，应记录成本；不把模型内部480×832视图当额外真实信息。

### 17.3 先做真实小规模pilot

每方法固定2个dev世界×1个origin×4族×1seed，共8包/704新输出。pilot前锁定task IDs、model配置、seed和资源profile。

记录实际权重加载、后端版本、设备、有效图数、PNG/replay、first-reference规则、F7父状态、全部额外原生工作和峰值范围。不能用曾经跑过的旧模型或旧数据结果替代当前管线pilot。

初始一张图生成失败时先定位输入/权重/设备，不直接启动33k输出完整test。真实GPU不可用时保持mock通过和real_model未验收，继续其他工程。

### 17.4 Eager-Raster / Frontier / Lazy真实比较

同补全器、权重、512²上下文、256²核心、prompt、steps、scheduler、precision和保护策略。记录首fill真实I0上下文和全部fill顺序。Frontier须确定性选择，不因看到生成质量再改顺序。

I0保护和历史保护的修改属于明确ablation；不能在一个方法硬粘贴而另一个没有却不说明。两种地图仍分配完整数组时不宣传稀疏内存。真实质量改善是实验问题，不把几何或计数测试当结果。

---

## 18. 旧相机/LoRA工作的真实接入与数值训练

### 18.1 先锁定可用旧源码和后端

读取实际提供的旧ZIP/仓库、`cam_utils.py`、训练器和推理代码，并核对来源hash。缺少旧源码或权重就列具体依赖，不能凭历史文件名重建并声称原样复现。不要同时vendoring两个不同版本Wan导致import随机选择。

建议有单独、锁版本的backend模块；训练和推理共用PR #1的camera contract或经过验收的等价转换。代码、VAE、LoRA目标层、双噪声专家和权重布局都记录。

### 18.2 LegacyNorm与FixedUnits

保留旧最大范数归一化为明确对照，不在FixedUnits路径里再次隐藏调用旧默认归一化。固定单位使用(q-anchor)/固定对角线的虚拟条件；这是虚拟相机，不是真实卫星外参。

审计链：完整q → 逐帧pose → Plücker → latent时间条件 → 模型层输入。检查xy、c2w/w2c、方向/矩通道顺序、像素中心、dtype、维数和时间stride。

固定单位保留幅度不等于所有事件信息不丢失。继续保留PR #1时间抽样短环反例，补真实后端latent对齐测试。段数和步数不均时不能只用转弯端点均匀插值重排事件。

`native / identity / drop`不是同一配置。drop基线不能从CameraBatch的trace q再获得第二数值通道；prompt中不得泄漏task_id、small/large名称、目标位置或隐藏城市。

### 18.3 Offline与online包装

离线F2/F3可用I0+32查询对应原生33帧，锚槽位不计新输出；F3两流各reset。Online16可采用LastFrame包装：自身上次已提交规范末帧+当次16查询，原生17帧。

下一块不能从隐藏world裁真实图。末帧缺失时使用预先固定策略：例如终止该流后续模型生成并保留缺失槽位；若允许使用更早有效anchor，必须记录其真实global q并完整验证，不静默把它当当前端点。不能重新生成先前缺失图来修补首次参考。

状态包含已提交anchor、global位置、调用索引、RNG、可写特征/缓存以及必要的后端状态。F7两支从同父状态独立恢复。共享冻结权重可以保留，不共享分支可写地图或随机发生器。

33/17的原生槽位数量用于适配与计费说明，实际后端产生更多时按真实计数记录。未消费q的原生视频不能被标为数值条件完成。

### 18.4 训练数据和预算

使用当前train60世界和3840个33视图clip索引按batch裁图。训练不得使用dev以外的模型选参数据；metric_calibration/validation用于仪器，test不可参与训练或挑seed。

完整逐事件q、图像、pose和K来自同一个样本索引。若加入17帧子clip，显式冻结窗口位置、抽样比例、anchor及曝光预算，不能更改核心测试。

实现可执行训练入口、数据loader、基础权重预检、真实flow/diffusion训练路径、可训练层策略、optimizer/scheduler、AMP、梯度累积、checkpoint/save/resume和validation。损失与噪声日程以实际选定骨干为准，不能用合成toy loss冒充完成真实训练。

模型配置应填写真实选择：base checkpoint hash、可训练层、LoRA rank/alpha（如采用）、更新步数、有效batch、学习率、precision、训练seed、评估间隔和checkpoint选择准则。这些参数当前尚未全部确定；先在dev冻结，不由Codex宣称已有最优结果。

LegacyNorm和FixedUnits公平比较用相同预训练初始化、数据、optimizer更新次数与有效曝光；仅相同epoch不保证同预算。同权重改预处理的诊断实验与分别匹配训练的实验分表。

### 18.5 必须验证的训练细节

可训练参数实际有非零梯度且确实更新；冻结参数不更新；双专家/相机分支需要分别检查；save/load张量一致；中断恢复optimizer、scheduler、scaler、RNG和数据进度；错误退出码不能被上层吞掉；验证集role绑定；训练/推理条件hash相同。

真实小训练smoke后再开展正式训练，保存实际GPU证据。mock只能验证控制流，不能标 `REAL_MODEL_VALIDATED`。权重和大型产物不进Git。

---

## 19. G6：core120构建与正式实验执行

### 19.1 构建完整世界之前的门

已通过来源锁/首件/真实smoke12；修复后的Phase A稳定；至少真实基线pilot确认软件和资源预算；原始数据使用条件已记录。完整9GB源数据下载必须是显式批准操作，不随单测触发。

```bash
python scripts/hf_materialize.py inria \
  --manifest data/inria-lock/manifest.json \
  --inventory data/inria-lock/source-inventory.json \
  --profile core120 --source-dir data/inria-raw \
  --out build/inria-core120 --download --max-source-gb 9.5
```

保持世界处理及正式world_id；源失败不换图。审计120世界、480I0、1920包、训练3840clips。正式geometry曾检查通过不等于120真实世界已物化。

### 19.2 正式运行前冻结

实验清单绑定模型权重、训练seed、推理seed、角色、task subset、native转换、在线状态、指标版本、retry预算、资源profile、主比较和统计估计量。不同指标版本产生不同analysis_id，不覆盖旧排名。

测试24世界×4起点每模型每seed33792张新输出；三seed101376张。资源预算来源于实际pilot，不预写未知硬件GPU小时。可先分批调度但expected全集事前固定，任务分片不改变seed或丢掉未执行项。

### 19.3 分布式/多GPU调度的可选实现

若实现并发，按run独占锁或调度数据库保证同attempt单写，model与GPU资源分配固定。一个GPU上的多个并发模型会污染过程峰值和同步边界；没有隔离测量profile时禁止将其作为可比独占成本主表。

不得让多个节点同时追加无锁JSONL；采用每run不可变结果+协调合并或事务存储。集群walltime临近退出产生INTERRUPTED，不当模型错误默默重试；计量全部实际attempt。

并发不是“完整构建”的先决条件；先提供正确的串行Linux链路，后扩展并行。

### 19.4 验收输出

完整expected对账、第一attempt表、恢复表、独立replay、所有资源尝试记录、RGB/motion/quality分项、每城市结果、共同可评子集、完整缺失报告、稳定LaTeX/figure-data。模型/指标没通过实景验证时结果标diagnostic，不能混正式结论。

## 20. D09：同起点配对与机制实验执行器

复用已有 `paired.py` 的 `same-start-return-pair-v0.1`，不改变核心1920包。exact/offset同世界、origin、I0与前80个查询，分别按自己的再入标签评价；偏移仍改变几何，因此不是排除所有因素的纯记忆因果实验。

### 20.1 工程交付

- 从配对index生成独立suite，不混core；pair_id进入运行与分析身份。
- 验证成对task I0/prefix共享关系和各自合法性，不仅检查字符串相同。
- 执行相同模型/seed/预算；两变体独立reset，不将一变体的后缀状态给另一变体。
- 成对缺失、共同可评像素/任务覆盖和资源分别报告；不能只保留两边成功配对。
- 整对bootstrap随world采样；任务数量不增加地理独立性。

### 20.2 历史和保护机制

同骨干LastFrame/FrameCache/MapAssisted应有实际不同且可解释的状态读取规则，不仅更换配置名字。记录检索帧、位置、容量、命中、特征、重算/解码成本。硬保护I0和生成历史分别设置，禁止暗中增加真实观测。

固定单位与轨迹自身归一化消融时，online方法只能使用已允许历史和当前块；不能读取全未来轨迹取得max_norm。未来依赖的版本只能放在明确离线诊断，不能当合法online基线。

最终B与首次再入在同一批不可回写输出上计算，不能分别挑模型样本。新增任何辅助指标必须记录版本，不覆盖原主保持分。

**验收：** paired任务、执行、失败、汇总、成本可追溯；真实机制结果需实际模型运行。全测试同起点扩展额外49152新输出/方法/seed，先固定小子集pilot，不自动扩大预算。

---

## 21. D10：LEVIR100-LR64原尺寸运行与评分

### 21.1 固定来源和构建

源 `blanchon/LEVIR_CDPlus`，指定历史revision `8062826f12e4f3cd5690b84278ad67c1f42acc46`。仅三个test Parquet，既定大小分别439171835、439361709、437594257 bytes，总1316127801；SHA按当前已保存source lock逐项核对。

从footer核对348行，先按冻结的study/shard/row哈希选100行，再只解码image1。原始1024² RGB，不resize、padding或拼接；image2与change mask不用来生成下一空间帧。没有真实Parquet读取时保持真实数据验收未完成。

### 21.2 LR64定义

```text
world_hw = [1024,1024]
viewport_hw = [240,416]
origin = [512,512]
query_bounds_xy = [[-288,288],[-288,288]]
A=(0,0), B=(256,256), C=(-256,256), D=(-256,-256)
end_exact=(256,256)
end_offset=(288,280)
每段16步，64个新查询，4次online16
目标为B中心128×128
```

两变体前48查询一致，独立reset。既定几何示例exact再入59–63/gap34–42，offset再入58–62/gap33–41；代码必须重新由几何算标签，不能把范围常数替代所有像素计算。

### 21.3 适配方式

当前核心Ledger/runner是2048²接口，不能把LR64伪装F5或F7。优先在独立namespace/显式WorldSpec中支持LR64，保持原Inria入口默认不变。

通用WorldSpec至少含world/viewport尺寸、origin、合法范围、task schema、版本和study。若抽取公共裁剪/台账实现，先构造兼容性黄金结果；每个核心task的整数误差和分母必须一致。存在必要源码变化要记录，不再虚称核心文件字节未改。

补齐LR64的task index、world关联、公开session、expected输出、运行、first-reference ledger、replay和aggregation。模型输入公开小地图规格但不公开隐藏source位置；不支持该世界规格的模型显式拒绝，不补边兼容。

### 21.4 验收

原始1024输入从读取到评分无放大；q=±288合法、超界拒绝；每包64新输出，100世界200包/12800输出每seed；Oracle适用误差0、满覆盖；首次缺失不可修补；offset/exact独立；相同世界相关性进入统计说明；核心全回归不变。

LEVIR是不同原生GSD/地图范围的独立补充，不与Inria主表混榜，也不因来源名称不同就宣称无地理交叉。

### 21.5 FAIR1M/OSCD探针边界

保留PR #1有限预算plan/download/inspect能力及其测试。它们不属于本轮正式基准必需样本，不可为“完整构建”自动下载新数据或创建未经批准的研究任务。

真实header检查不等于完整像素、投影、GSD、许可、地理独立性已验收。新的正式来源要单独研究规范和发布版本，不能由Codex在补代码时自行决定。

---

## 22. D11：合作式评测与受限worker明确分级

### 22.1 两种模式都必须诚实命名

- `cooperative_inprocess`：现有内部开发模式；可以复算，但不能认证隐藏文件不可达。
- `isolated_worker`：部署在实际系统隔离边界内，仅给公开输入；必须通过部署测试后才能标相应隔离验证。

不将普通subprocess、换工作目录、删几个环境变量或哈希链称为安全沙箱。

### 22.2 受限模式的最低工程合同

评测器持有private worlds/labels/future schedule/台账；worker仅获得允许的I0、当前查询与模型权重。IPC消息有版本、长度/shape/type上限和请求ID；消息不携带私有路径，异常不回显敏感文件内容。

容器或其他实际OS隔离方案需要非特权用户、只读权重挂载、私有数据不挂载、独立临时目录、网络限制、资源/时间限制、环境变量白名单和进程生命周期清理。GPU设备暴露范围按威胁模型注明；不承诺消除所有侧信道。

worker不能修改评测器保存的提交；图像进入评测器后复制并写不可回写台账。所有真实模型自身的CPU/GPU工作应由正确的进程/设备资源collector观察，不能只量IPC客户端等待时间就称模型资源已测。

F7在worker中建立可恢复状态；snapshot不能依赖跨分支共享可写路径。允许存储状态的位置要计入资源与可达权限。

### 22.3 验收测试

使用专门的无敏感数据测试目录验证：隐藏哨兵文件读取失败、未来查询不可取得、symlink越界失败、环境私有哨兵不可见、未授权网络失败、IPC错误长度/类型拒绝、超时终止、子进程清理、不能修改已提交PNG。两支状态污染测试必须失败而非被误判正常。

测试不具备容器/namespace能力时返回清晰BLOCKED，不自动回退inprocess却标安全。隔离结果只对已测试配置和威胁模型有效，不写“密码学安全”或“绝无泄漏”。

---

## 23. D12：安装构建、复现、一键入口和正式发布

### 23.1 代码包不是只支持editable import

从干净checkout构建sdist/wheel，在新环境安装wheel，离开仓库目录运行已安装CLI。确保配置和必需资源可定位，避免仅因 `PYTHONPATH=src` 通过而发布失败。

可选依赖分组：core/dev、data、motion、features、generate、isolation；没有GPU的环境仍能运行数据/几何/CPU诊断。真实模型依赖按适配器profile锁定，不因一个后端安装失败破坏整个核心包。

构建包不得包含raw影像、生成PNG、权重、token、用户绝对路径、临时缓存。发布文件清单和SHA256，保留source commit、任务/指标协议版本。

### 23.2 README必须能从零到结果

保留一个主README作为执行入口，其他docs作明确引用，不保留互相矛盾的多份“唯一最新版”。README至少说明：

```text
安装/环境
→ 只生成下载计划
→ 首件和smoke12构建
→ 物化审计
→ suite plan/run/verify
→ RGB与资源导出
→ 运动校准/独立验证
→ 特征与质量
→ 真实模型预检/pilot/训练
→ core120正式运行
→ paired/LR64独立运行
→ 论文表图/完整性检查
```

每条命令必须在对应配置下实际跑过或明确标未实测；不能从最初规划复制已经不存在的参数。设置`--help`、路径校验和`--dry-run`。默认一键脚本只执行小型可验证链路，不偷偷下载9GB数据或大型权重。

### 23.3 验收入口

建议新增：

```text
scripts/acceptance_phase_a.py     # Phase A证据和实数据诊断
scripts/acceptance_full_build.py  # 按profile组合全部验收门
```

两个脚本是目标不是当前已存在接口。输入profile、数据根、模型清单、预算和输出根；运行每步并传递返回码。没有输入的真实任务标NOT_RUN/BLOCKED，不因跳过就总PASS。

验收至少区分：

| profile | 可以证明 | 不能证明 |
|---|---|---|
| `cpu-synthetic` | 算法/故障/状态机与包装构建 | 真实遥感、GPU和生成质量 |
| `smoke12-diagnostic` | 真实数据构建与完整Phase A | 真实生成模型成绩 |
| `real-model-pilot` | 指定权重/设备/任务的真实执行 | 全core120性能与所有后端兼容 |
| `metric-validation` | 固定仪器与特征协议的指定验证 | 超出所测域的语义真实 |
| `core-formal` | 固定全集实验执行与可重算表 | 全球地理泛化和录用保证 |
| `isolated-release` | 指定部署威胁模型的权限测试 | 无条件安全承诺 |

### 23.4 文档和科研发布

提供Benchmark Card、Dataset Card、Adapter Guide、Measurement Protocol、Reproduction Guide、Limitations与实验manifest。代码license、第三方代码license、数据/衍生影像使用权和权重license分别核查；没有授权信息不擅自填宽松许可证。

论文表图全部从固定产物导出，不手填数字。正文主张需要实际模型证据；“接口已实现”“CPU测试通过”只能作工程结果。匿名包去除Git身份、绝对路径和可识别元数据，但保留数据原作者正常引用。

另一干净环境重建世界/任务/标签并重放至少代表性结果；像素/任务hash一致，非确定GPU结果的容差和依赖限制单独报告。正式发布tag只能在实际验收报告对应的commit上打，不能借旧CI给后继代码放行。

---

## 24. 建议的模块职责：保留当前包名和成熟实现

以下是目标组织，允许按现有代码规模合并小模块，但禁止重建一个平行 `src/rswm/` 或将 `tasks.py` 变成同名目录制造import歧义。

```text
src/rswm_inria/
  geometry.py               # 冻结坐标/裁剪
  data.py                   # 来源锁与精确转换
  tasks.py                  # 核心采样与标签
  score_rgb.py              # 核心首次台账数学
  replay.py                 # 独立产物回放
  runner.py                 # 执行与观察器钩子
  suite.py                  # expected、attempt、恢复与验证
  resources.py              # 字段完整性、计数与状态存储
  aggregate.py              # 计划分母、世界/城市/配对分析
  export_tables.py          # CSV/JSON/LaTeX/来源
  registry.py               # 明确模型配置和权限能力
  materialized.py           # 若移植/新增：物化审计
  frontier.py               # 合入PR #1的同窗口地图
  motion.py                 # 已有图像位移仪器
  motion_protocol.py        # 校准/验证和批量连接
  quality.py                # 首次图块与统计数学
  features.py               # 冻结真实特征后端
  quality_protocol.py       # patch/reference/feature流水线
  paired.py                 # 独立同起点任务
  lr64.py                   # 独立小地图执行/评分入口或等价模块
  source_probe.py           # 候选小预算来源，不混核心
  adapters/
    camera_conditioning.py
    <chosen_real_backend>.py
  training/
    <shared_data_and_training_entry>.py
  isolation/
    <worker_protocol_and_launcher>.py
```

`registry/features/lr64/training/isolation`中不能只留注册表、TODO或NotImplementedError，再宣称D08/D10/D11完成。每个被README宣传可用的后端都必须有实质执行路径；不支持的能力在预检查明确拒绝。

运行结果统一使用schema版本；schema相关validation集中实现，避免suite、replay、导出器各自容忍不同非法JSON。

---

## 25. 测试矩阵：按行为验收，不靠凑数量

### 25.1 必需场景

| 测试族 | 必测行为 | 范围标记 |
|---|---|---|
| 来源/数据 | revision与inventory不符、大小/SHA错、伪装TIFF、波段/模式错误、子集稳定 | synthetic + real首件 |
| 几何 | 整数溢出、边界、F3两倍、F5/F7首见/再入、160计数、独立慢速核对 | synthetic + 正式ID |
| 首次语义 | 首槽失败永久无效、不能后补、跨调用、双支共同锚、合法噪声保持 | synthetic |
| suite | 重复/未知expected、身份改变、model load/reset/generate/snapshot/restore失败 | synthetic |
| attempt | 首轮/恢复分离、上限、数字序号、并发锁、SIGTERM/SIGKILL、不可覆盖 | Linux集成 |
| 元数据 | 截断outcome/resource/receipt、错身份、重复键、NaN、旧attempt损坏 | 集成负例 |
| R01 | 部分GPU测量、阶段/attempt总表完整性、CPU null、嵌套计时 | mock；GPU另验 |
| R02 | owning array/view、共享bytearray/memoryview、unsupported状态、采样失败 | CPU |
| R03 | pre/post计数失败、epoch/rollback、F3/F7累计不回退、finish幂等 | CPU |
| 统计 | 城市不均衡、整城缺失、配对子集、seed缺失、NA、输出去重、retry口径 | 确定性fixture |
| 导出 | JSON/CSV/LaTeX精度、转义、null、来源hash、半量标记、错误退出 | CLI集成 |
| 分支整合 | Frontier计数不双算、相机子模块锁、配置真实执行、原测试保留 | 组合回归 |
| motion | 符号、方向、过冲域外、歧义、弃权、断链、角色隔离 | synthetic + 实景 |
| quality | 先选块、首参不补、分支去重、权重hash、负U、真实参考独立 | synthetic + 实景 |
| 真实模型 | 真权重恢复、q真实消费、原生帧、状态/RNG、训练更新/恢复 | real weights/GPU |
| LR64 | 1024原尺寸、±288、64输出、独立变体、核心回归 | synthetic + real |
| 隔离 | 隐藏哨兵、未来查询、网络、环境、symlink、IPC、资源进程 | 实际部署 |
| 包构建 | wheel/sdist、新环境安装、离开源码目录CLI、无敏感文件 | 干净环境 |

### 25.2 回归纪律

每次先受影响测试，再全pytest，再集成。重大分支整合后重跑组合测试；不能删旧用例让CI过。可选依赖skip注明理由，missing dependency不是通过。不要以“至少X项测试”替代语义验收。

现有审核负例应变成长期回归；重要JSON/artifact攻击检查完整非零退出和分母，而不只断言异常类型。模板snapshot可以检查字段，不能把耗时、GPU峰值硬编码成跨机器固定数字。

### 25.3 环境矩阵

至少一个Linux CPU环境覆盖完整core软件与POSIX suite；一个实际GPU环境验证被选中的生成/特征后端；Windows原生仅在代码明确支持并测试时宣传。没有GPU的CI必须保留真实GPU验收为未做，不能靠假的cuda对象升格。

依赖锁分别服务重现环境和允许版本范围；记录wheel构建与运行版本。浮点算法在不同依赖版本容差要有依据，不能为通过测试任意放宽。

---

## 26. 提交拆分与每次交付要求

建议按下列逻辑提交，实际名字可调整：

| 提交组 | 内容 | 必附证据 |
|---|---|---|
| C00 | 基线审计与现有问题回归测试 | baseline与旧版失败记录 |
| C01 | R01字段级计量完整性 | 双顺序同步失败、总表传播 |
| C02 | R02共享存储与R03累计计数 | 16-byte、12-not-22、未知范围 |
| C03 | retry v3与旧v2兼容 | 首轮/恢复表、冻结预算 |
| C04 | 元数据隔离与suite原子性 | 坏JSON、多进程/中断、分母 |
| C05 | 分层出表、LaTeX和完整provenance | 确定性表格fixture、转义 |
| C06 | 真实smoke12验收及必要D01/D02迁移 | 输入hash、576run/50688输出三诊断或明确未完成范围 |
| C07 | PR #1整合与registry | 组合pytest、Frontier/条件/资源测试 |
| C08 | 真实生成/数值条件后端与pilot | 真实权重/设备/状态报告 |
| C09 | D05运动协议 | calibration与独立validation |
| C10 | D06首图块/特征/参考 | 真特征、真实—真实、覆盖 |
| C11 | 训练/正式运行编排 | checkpoint、预注册预算、实际运行记录 |
| C12 | D09/D10扩展 | paired/LR64独立验收和核心兼容 |
| C13 | D11隔离与D12发布 | threat model、wheel、干净重现 |

不可用真实资源时，将对应提交的代码和模拟测试与“实际验收证据”分开；不能为保持提交顺序制造假报告。先合代码不意味着批准正式发布。

### 26.1 每个阶段报告模板

```text
A. 起点
repository / branch / base SHA / HEAD / working tree状态

B. 目标与实际修改
审核ID或D编号；文件；原行为；新行为；schema兼容影响

C. 实际验证
command / exit code / tests collected/pass/fail/skip
synthetic或真实数据；模型名称和权重hash；GPU是否真实
证据路径与hash；未覆盖范围

D. 不变量
核心任务/像素/首次提交语义是否改变；逐文件或数值兼容依据

E. 结果身份与统计
expected / executed / completed / failed
planned outputs / valid outputs / replay verified
first attempt与recovery是否分开；资源测量覆盖

F. 状态
D01–D12各维度状态，哪些BLOCKED、原因、已完成纯工程路径

G. Git
commit / pushed / CI身份与状态 / PR状态 / 是否合并

H. 下一阶段
明确要修改的模块和依赖；不将未来动作写成已完成
```

只在确实push并回读远端后写pushed=yes；只有对应commit的CI通过才写当前CI通过。不能用上一个绿色commit证明本次代码。

---

## 27. 总验收清单：代码完成与实验完成分别签收

### 27.1 Phase A修复签收

- [ ] R01部分GPU测量不再标完整，字段状态传播到LaTeX。
- [ ] R02同一16-byte外部buffer不再计32，未知存储明确不完整。
- [ ] R03真实累计12不再成为22，跨阶段与epoch解释清楚。
- [ ] 首attempt主分析与受限恢复分表；旧v2口径不静默改变。
- [ ] 元数据损坏不会无说明地中断整套导出，也不会删分母。
- [ ] CSV/JSON/LaTeX、唯一输出计数、failure summary与provenance完整。
- [ ] 实际Linux全回归通过；真实smoke12 Phase A验收范围有文件证据。

### 27.2 完整代码库签收

- [ ] 单一组合分支/正式提交包含Phase A与已批准研究扩展，组合CI已跑。
- [ ] source→world→I0→tasks/labels→audit可重复，显式预算，无整库误下载。
- [ ] expected→attempt→replay→资源→世界/城市→表格贯通且覆盖失败。
- [ ] D05/D06不只是函数名：校准、验证、特征和参考有真实入口与配置。
- [ ] 四类生成系统有实质统一接口；至少目标实际后端有真实权重验证。
- [ ] 数值条件确实被消费，LegacyNorm/Fixed、时间映射和训练恢复可检查。
- [ ] paired/LR64单独执行/评分/统计，1024不被伪装成2048。
- [ ] 受限worker模式有实际系统隔离；否则发布明确只支持合作式，不声称D11通过。
- [ ] wheel/sdist干净安装、README命令、验收脚本与完整错误码测试通过。
- [ ] 所有未验证字段、许可和硬件限制都显式保留，未用TODO占位宣称完成。

### 27.3 论文实验签收（不能由代码测试代替）

- [ ] core120真实物化与正式test身份冻结。
- [ ] 真实模型/训练预算/seed记录齐全，不拿诊断系统顶替。
- [ ] 运动与质量在冻结规则下完成独立验证，可信范围明确。
- [ ] 完整任务/失败/首轮/资源记录可独立复算。
- [ ] 两城市分表和明确适用的跨骨干/独立扩展证据存在。
- [ ] 机制结论来自实际实验，而不是提前填写预期“提升”。
- [ ] 图表自动生成、干净环境复现、权利/匿名材料检查完成。

代码库可以先达到软件可构建状态，但若真实权重或数据验收尚缺，交付名称必须保留“实验未验收”。不能使用同一个PASS同时代表三个清单。

---

## 28. 尚需由项目负责人确定的少量输入

这些信息在当前材料中没有全部冻结，Codex不能伪造。但它们不阻碍G0/G1纯工程工作。

| 输入 | 需要给出什么 | 未提供时的行为 |
|---|---|---|
| 真实smoke12位置 | manifest、inventory、world/index根目录或允许的下载预算 | 完成代码和synthetic验证，真实验收BLOCKED |
| GPU与调度环境 | 型号/数量/显存、可用时间、Linux或集群约束 | 不编造耗时，不自动运行昂贵任务 |
| 真实模型权重 | 本地路径/可授权获取来源、许可、旧方法检查点 | 不用随机模型伪装实际系统 |
| 旧方法源码 | 经确认的ZIP/commit、条件与训练配置 | 保留数学接口测试，真实backend未验收 |
| 训练资源上限 | 更新数、预算、模型规模、允许数据角色 | 先开发小pilot，不擅自正式训练 |
| 数据与代码发布许可 | 各来源与第三方权利记录、项目代码license决策 | 不自动再分发原图/权重、不编造许可 |
| main合并与正式tag | 审核通过后的明确发布授权 | 在开发分支交付，不擅自合main或删历史 |

模型、特征骨干和校准接收阈值需要在对应开发阶段选定并冻结，不能把本文件中的候选名称当成已批准的最终配置。原数据和任务定义优先级不因缺资源降低。

---

## 29. 可以直接发给Codex的启动指令

```text
请执行 docs/CODEX_FULL_BUILD_PLAN.md，对 yedda-tv/rswm-1 做完整代码构建。

先核对当前GitHub/main/PR #1/#2和本地工作树。以实际最新的Phase A后继提交为基础，不从旧main重写suite/resources/aggregate，不reset用户修改。

先读取附带的Phase A审核报告和资源复现证据。第一轮执行G0和G1：
1. 基线完整测试和冻结文件核对；
2. 将R01/R02/R03转为导入当前工作树的回归测试，先在旧版复现；
3. 修复资源字段完整性、外部共享buffer去重、计数器未知前值；
4. 实现版本化first_attempt_only主分析、受限retry和旧v2兼容；
5. 修复单run元数据损坏隔离，保留完整expected分母；
6. 完成CSV/JSON/LaTeX和全attempt来源链；
7. 跑受影响测试、完整pytest、CLI端到端与故障测试；
8. 更新阶段状态和证据，提交开发分支，不自动合main。

这不是让我再看一份方案，也不是只增加dataclass、CLI help或README。
需要实际修改代码、测试、构建并按计划验收。G1通过后按G2-G8继续，不跳过真实数据/模型验收；资源不可用时继续独立工程部分，并将未做的真实验证明确标BLOCKED或NOT_RUN。

冻结Inria数据处理、地域、F2/F3/F5/F7、首次计划曝光和RGB评分语义不变。
原审核reproduce_resources.py锁定旧文件并断言bug存在，不能直接把它当修复后回归。
不要伪造数据、GPU、权重、CI、push或“全部完成”；每阶段按本计划第26节给出真实报告。
```

第一轮不要同时大改模型训练、LR64和安全隔离；先把现有正式结果最容易出错的资源/失败/统计层固定。该执行顺序不是缩减全项目目标，而是保证后续模块不会建立在不可靠的结果管线上。

---

## 30. 材料依据与交接说明

### 30.1 已核对来源

- 目标仓库 `yedda-tv/rswm-1` 的 PR #2：HEAD `3f733e6880c296c43a40fe9535ab4f69e0025687`，仍Open/Draft/未合并；含Phase A实现与合成验证范围。
- PR #1：HEAD `656ed3fd5b7aa629e6544d4972f0daec996e5456`，仍Open/未合并；含Frontier、固定单位相机、来源探针。
- main ref：`3e22673be4d6656990312665b3a0b6932c82f112`。
- 附件 `RSWM_PhaseA_Completion_Audit_3f733e6.md`，包括R01–R03已复现问题和A01–A05静态/交付缺口。
- 附件 `resource_reproductions.json` 与旧版本复现包，资源原文件Git blob `bf1372433db84b27ff14c63b813c5bfa1ec573fe`，聚合原文件Git blob `8f3e45b322798a1f9b0a7f2b292a61c1f71e03a2`。
- 当前对话与仓库冻结 `docs/BENCHMARK_DESIGN.md` 的研究定义；尚未确定的模型/阈值/资源在本计划第28节保留。

本文件的工程改造要求是交给Codex的**待执行计划**。本次交付仅创建计划与交接文件，未修改目标仓库、未合并PR、未重新运行模型或宣称新测试通过。

### 30.2 最终目标

不是再制造一个“看起来很完整”的目录，而是让另一位研究者能够从固定真实来源构建基准，接入一个实际生成模型，在明确输入权限和失败规则下运行，独立复算所有分数、覆盖与成本，并知道每一项结论的验证范围。
