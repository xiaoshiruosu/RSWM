# 测量、覆盖与结果身份

本协议区分输出执行、提交完整性和资源测量完整性。有效 RGB PNG 不证明原生工作完整可测；计数器完整不证明图像正确；CLI 返回成功不替代真实模型、GPU、地理或测量仪器的独立验证。

## 首次计划曝光与 RGB

真实已知区 K 的参考只有 I0；生成历史 P 的参考固定为每个像素第一次**计划**曝光时对应槽位的不可回写提交。首次曝光失败后该参考永久无效，不能用重访帧、另一个样本或 F7 另一分支补写。缺失当前提交保留原预计分母。逐像素首次再入仅在几何标签指定事件评分一次；辅助最终目标分另冻结版本并复用同批 PNG。

`absolute_error_rgb` 是有效配对像素三通道绝对误差整数和；MAE 为 `absolute_error_rgb/(255*3*valid_pixels)`，coverage 为 `valid_pixels/expected_pixels`。没有有效像素时 MAE 为 null；无适用分母时为 NA。条件 MAE=0 不能掩盖低覆盖。输出 completeness 按唯一生成槽位计数，不能随多个 metric 或 F7 的 branch/paired 视图重复累计。

Core 冻结 F2/F3/F5/F7、地域角色与像素管线不因资源或仪器修复改变。主 RGB 估计量保留等适用 task/origin/seed cells → world → city 的层级；测试 SFO、Innsbruck 分列后等城市平均。分组包括 study、role、model manifest、family/protocol/variant 和 metric identity。世界/城市 bootstrap 表示该固定采样下的敏感性，不证明地理独立。

## 尝试与完整分母

v3 plan 默认 `first_attempt_only`、每 run 一次、无自动 retry。较大有限预算和允许状态必须进入冻结 retry policy；`run --resume --retry-failed` 仅创建预算内新尝试，不能覆盖 attempt 000 或用 PNG 恢复隐藏状态。first-attempt 主表和 within-budget recovery 补充表分开；所有已开始尝试均进入成本。旧 v2 采用明确的历史 latest 语义，不静默转换为 v3 主表。

root plan/expected 身份无效时拒绝整次分析；单 run 元数据损坏保留该 run 的预计分母。资源 JSON 损坏只使资源失去信任，独立验证通过的 RGB 仍可评分；损坏 outcome/不可认证提交不得以目录里存在 PNG 为由宣称成功。保留的 hash/receipt 证明内部一致性，不是可信时间戳或原数据真实性证明。

核心 suite/export 退出码为 0（合同内通过）、1（逐 run 缺失/故障，仍可导出完整分母分析）、2（根计划/参数无效）。具体仪器和 worker CLI 有自己的返回码；例如隔离探针 BLOCKED 返回 3，应检查其 JSON reason，而不能把全项目返回码一概映射成同一含义。

## 资源 v3 字段合同

`rswm-resource-v3` 对每个 quantity 保存 applicability、完整值、部分观测、计量单位、aggregation、eligible/measured 单元数、complete、reason codes 与 scope。字段完整性从阶段传播到 run 和全部尝试总表。

| 状态 | `value` | 部分观测 | 含义 |
| --- | --- | --- | --- |
| 完整适用 | 测得完整值 | 同时保留 measured sum/observed peak | 所有适用单元均测得 |
| 部分或未知 | null | 已知部分保留 `measured_sum` 或 `observed_peak` | 不能包装成完整总量或完整峰值 |
| 不适用 | null | null，明确 NA/reason | 例如 CPU 的 GPU 时间/显存或未进入阶段 |

求和型总量不会把缺失记录当 0；峰值型完整值也不会由部分样本的最大值冒充。额外报告计划 run 覆盖、已开始尝试数、可读资源数和逐字段测量覆盖。LaTeX 用 `--` 表示 null，并显示 PARTIAL/NA 以及测得部分。旧资源文件即使自带 `complete=true` 也不能据此获得 v3 完整性。

## 时间、内存、磁盘与原生工作

| Quantity | 当前可解释范围 |
| --- | --- |
| attempt wall | collector 建立至最终采样，含校验/评分/IO/replay；不含之后 resource/receipt 写入 |
| 阶段 wall | 实际进入的加载、初始化、generate、转换、snapshot、restore、replay 等阶段；嵌套区间避免重复累计 |
| GPU 时间 | 不重叠的 model load/initialize/generate/snapshot/restore 阶段，前后同步的 host wall；不是 CUDA kernel time；IO/replay 不混入 |
| `gpu_time_seconds` | `gpu_synchronized_model_wall_seconds` 的兼容 alias，不能再相加 |
| GPU peaks | 指定设备 torch caching allocator 自 collector 初始化后的高水位，包含共享权重；非整个 GPU 物理显存 |
| process peak RSS | 进程生命周期高水位，包含依赖与之前尝试；不是单任务增量或模型独占内存 |
| dynamic state | adapter 显式声明且支持的 buffer owner 去重后的观测峰值；排除权重、Python 对象开销和不可见 backend cache |
| disk bytes | attempt 目录采样时逻辑文件长度；不含之后 resource/receipt 写入，源数据和权重不属于该目录统计 |

NumPy owning arrays、bytes、bytearray 和可追到这些 owner 的 memoryview/views 按共享根去重。逐 ndarray 引用的 logical bytes 与唯一 owner bytes 分列，切片覆盖区间不代表底层分配大小。object dtype 内容、memmap、Tensor storage 和不透明 owner 保留 unknown/partial，不能随意推为零。采样失败一次后，后续成功不会抹掉未测单元。

累计 `resource_counters()` 支持明确的 `ok/not_initialized/not_implemented/read_failed` 读取状态、整数值和稳定 counter epoch。未知前值不能当零；只加有可信边界、不重叠的增量。同 epoch 可按声明合同衔接生命周期计数；无法识别 epoch、回退、换名或读失败保留未知范围。即使所有读取失败而计数器名字也未知，仍保存 counter-contract eligible/measured 覆盖。adapter 逻辑 reset/snapshot/restore 不回退生命周期 epoch 或累计成本。

计划调用、进入包装器、实际模型调用、计量适用调用、请求新槽位和有效输出分别计数。返回/丢弃/条件帧不等于原生模型内部工作；缺少原后端工作计数时保留 null，不能用 `native-requested` 推出额外工作。Frontier operation log 用于核对权威 adapter 累计计数，不能再加一次。Observer 失败不改变原模型异常、随机数状态、提交或 RGB 台账。

## 图像运动仪器

估计器只接两张规范图和固定配置，外层在估计完成后才读取 command/delta 计算偏差。图像内容向左对应视口向右，xy 按右/下约定。ambiguous、insufficient evidence、submission failure 和 out of domain 的位移/误差为 null，不能填零。搜索边界、置信度、域风险和状态均保留；路径断链不靠命令重新锚定。

实际 calibration 使用 Bellingham 的 `metric_calibration` 12 worlds；独立 validation 使用 Bloomington 的 `metric_validation` 12 worlds。受控几何、扰动、候选配置、选择规则、容差/错误可信率/最低覆盖在验证前冻结，输出每 world 结果与 risk-coverage。重新看过 validation 再选参数时保留 exposure 历史，不能重复称独立留出。真实验证失败只能作为 DIAGNOSTIC_ONLY 仪器。

入口为 `rswm-motion-calibrate build-pairs/calibrate` 和 `rswm-motion-validate validate/evaluate-suite`；suite 测量要求认证 PNG。F3 幅度比例只在双方符合预注册可测条件且分母非零时给出，并报告配对覆盖。RGB 一致性与估计运动正确性是两个指标。

## 未知区域质量

`rswm-quality-plan plan` 先由任务几何冻结 128² patch plan，再由 `mosaics` 读取认证提交或由 `reference` 读取同一几何真实 crop。每个像素只取 first-event 槽位；缺失永久留空。F7 共同前缀按 lineage 去重，后缀 lineage 分开，禁止跨支拼图或失败后换 patch。

`rswm-features` 加载明确本地 TorchScript archive，冻结 repo/revision、完整权重 hash、输出层/selector/维数、128² uint8 RGB 图块到 float32 的输入转换、resize/crop/normalization、device/batch 和运行版本。没有实际权重/torch/CUDA时预检查 BLOCKED；不会回退随机投影。缺失 patch 不编码为有效黑图。缓存键绑定像素、mask/完整性、feature config 和权重，缓存内容还检查 shape、有限数与 hash。

分布量名称为 `polynomial_mmd_u`；无偏 U 估计可为负，保留负数；样本不足为明确 null。当前没有冻结 Inception KID 管线，不能叫 KID。按 world 控制完整样本数并分 role/city 重采样；真实—真实两组要求不同 source worlds，不能把同世界邻接 patch 当独立参照。报告全计划 tile/pixel 覆盖、各方法自身完整样本结果和共同可评子集敏感性，不能只展示成功块。

candidate feature controls 与正式 release 是不同门。正式仪器要提供绑定实际 source audit、真实 calibration/heldout validation、预注册接收范围、真实冻结权重和 exposure ledger 的完整 receipt。未知区域分布诊断不是每栋建筑、道路或真实未来布局正确率，也不替代 RGB 保持分。
