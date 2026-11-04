> **v3.0(2026-07-30,不可作弊评分契约 + ICLR 发布要件)**
>
> 外部审查(基于 v2.9.1 只读复现)动态证明**评测器可被直接作弊**,四种手段全部通过
> 静默截断实现,每种都能拿到 `PSNR=inf`。核实后修完,并在核实过程中发现**一条更严重的**:
> 首帧是 i2v 的输入、`gen[0]==gt[0]` 是结构性必然,而 `compute_video_psnr` 逐帧取平均
> → `mean([inf, ...]) = inf` → **每一份提交的 PSNR 都是 inf**(不需要作弊,自动发生)。
>
> 新增 `lora_remotesense/new/submission_contract.py` 作评分的**单一权威**(两种呈现:
> `check_*` 批量报违规供评分器用 / `assert_*` 立即抛供指标计算用):样本 ID 集精确匹配、
> `(T,H,W,C)` 精确一致、dtype 一致、**零失败容忍**(旧版容忍 20%)、条件帧从**全部**保真
> 指标剔除、分布型指标样本量下限 32(旧版 `n>=2` 就算 FID/FVD)。宽松模式
> `RSWM_LENIENT_METRICS=1` 只为排障保留且**污染结果**(自动置 `headline_eligible=false`)。
> `score_submission.py` 复用同一权威,契约模块导入失败即拒跑。
>
> 其它已核实并修:oracle 门槛 15°→30°(实测仪器地板 ~25°,旧值恒 FAIL);
> 光流默认 `step` 从 `T//8` 改 1(实测 24px/帧对误差 **128.77°**、近乎反向,
> 而旧默认给 T=17→18px / T=33→36px);MDE 从 50% 功效检出阈改 80% 功效正式口径
> (差 1.43 倍,旧口径**低估**所需差异);FID 不再只取中间帧;阶梯的解析仪器地板
> 按段数分(`aq=4` 只保证逐段轴向,K≥2 时净位移不轴向,残差回到 ~20°)。
>
> ICLR 发布要件:LICENSE(Apache-2.0)/ NOTICE / DATASHEET / CITATION / requirements-lock /
> CPU-only CI / REPRODUCIBILITY / SUBMISSION 协议 / 官方三划分 + 脱敏公开版 /
> 11 档难度阶梯 / 仪器地板标定 / outpaint 平凡基线(带预注册读数规则)/ 骨干可插拔接口。

> **v2.9.4(2026-07-29,对抗审查剩余项:逃生口留痕 / 汇总三档 / 四处 fail-open / 其它出口补门)**
> 详见 `README_v30.md` §10。全部来自 5 视角对抗审查的 35 条 confirmed(去重后 6 类)。
>
> **v2.9.3(2026-07-29,对抗审查修复:编排器 $PY 未定义 / baseline --metrics_only 无门 /
> RSWM_RUN_ROLE 泄漏致判门被标 headline / approval_token 算得太早)**
> 详见 `README_v30.md` §10(历史条目)。四条全部是 v2.9.2 自己引入的。
>
> **v2.9.2(2026-07-29,run manifest 统一身份 + headline fail-closed + 34a/34b 双判门)**
>
> 三处"把错误产物洗成合格 headline"的路径被堵死,根因都是【每处各自校验各自的一个字段】:
> ① 续训只比 pose_mode(而 native 臂在 official/legacy 下都是 native),且 `rswm_profile`
>    读错层级(存在 `data_split` 内、读的是顶层)→ 恒为 None → 跨 profile 续训被放行,
>    权重随后被重新标成 official;
> ② 评测只要 meta 文件在就跳过 → identity+legacy 的旧结果在 drop+official 环境里被原样
>    复用,overall 又用当前环境写上 official/headline 标记;
> ③ 汇总只拒显式 `False` → 缺资格字段的旧产物 fail-open 进主表。
>
> 修法 = 单一 **`run_manifest.json`**(`lora_remotesense/new/run_manifest.py`):
> profile / sem_arm_mode / pose_mode / role / data-H5 SHA / split SHA / checkpoint /
> prompt_schema / frame_num / gen_seed / **代码指纹**,一次性钉死。
> **训练续训、评测复用、baseline 复用三处共同校验同一份字段,任何字段缺失一律 fail-closed。**
>
> 其余:`headline_eligible` 改为**逐作业按 pose_mode 重算**(identity 永远为 false,
> 36 号与 identity baseline 都归此类,不再继承 profile 级的 true);baseline 自报 pose_mode
> 与目录口径交叉验证(伪装即不予采用);判门拆 **34a(drop)/34b(native)**,共享 01 建的
> 同一份探针库、并行提交、**两者都 PASS 才放行主矩阵**;探针的四方向位移改为**从 overlap
> 带反解 + 轮转**(此前 [0.50,0.95) 与 128×222 窗口几何互斥,N/S 接受率 0.9-1.3%,判门必 FAIL);
> H5 持久化 `min/max_overlap_ratio`、`angle_quantize`、逐样本 `overlap_ratio`,判门直接
> 从产物自证口径;34a/34b 落 `gate_result.json`;`APPROVED_01_DATA` 绑定 `approval_token`
> (数据重建后旧批准自动失效);`p1_probe_build` 改 sha 绑定 stamp。
>
# 本包说明:要干什么 + 相对 2026-07-10 版本的全部变化

> **v2.8(2026-07-29,P0 时间轴修复 + 三臂重定义 + overlap enforce —— 20-33 全部作废重跑)**
> 外部审查发现训练/推理/评分三条时间轴不一致,核实**五条全属实**,量级实测:
> * **【灾难级】训练 pose 截断**:`train.py` 用 `c2ws_i[:lat_f]` 截断而
>   `inference.py` 用 interpolate 重采样 —— 实测 18/27/36 帧下训练 pose 分别只
>   覆盖轨迹前 **29%/28%/27%**,三种配置**一个转弯点都不含**;推理给 100%。
>   模型训练中从未见过转弯的 pose 信号,却在推理时被喂完整多段轨迹 ——
>   "几何通道零增益""损保真 FVD 33.4→41.5""忽略转弯"全部可由此平凡解释。
>   修:两侧共用 interpolate_camera_poses 同一口径(新 test_pose_timeline.py
>   36 项钉死,裸截断再现即红)。
> * **【严重】尾截断**:源头每段含首尾 → 18/27/36 含 K-1 个段界重复帧,训练
>   4n+1 对齐取【前】17/25/33 = 保留重复帧却丢真实终点。修:源头跳过后续段
>   step=0,直接产 17/25/33(无重复、含终点、恰为 4n+1,训练无需再截断);
>   下游帧数常量(build_correct/check_resisc45/文档/baseline)同步。
>   加 4n+1 契约校验:steps≠9 时大声警告(steps=8 去重后 15/22/29 不合规)。
> * 【中】converter 从 waypoints 均匀插值 pose → 改用真实逐帧 trajectory
>   (`pose_source=plucker_from_trajectory`)。**实测量级远小于报告描述**:
>   最大错位 5.9px(窗口宽 3%)、转弯帧一致 —— 修它是消除"段等长"巧合依赖,
>   **不是** null 结论的解释,不夸大。
> **[拍板] pose_mode 三态**(修"text-only 实为静止命令"的釜底抽薪):
>   native=真实轨迹 Plücker;identity=**静止相机诊断臂**(恒等位姿仍走 Plücker
>   通路,对相机条件骨干等于命令"相机不动",与 prompt 方向词矛盾 —— 不再称
>   text-only);drop=训练与推理都不传 dit_cond_dict(**真** no-pose/text-only)。
>   全链落盘:train argparse + H5 一致性硬校验 + checkpoint config + eval meta +
>   common.sh 透传与 checkpoint 交叉校验(模式冲突直接死)。
> **[拍板] overlap enforce <0.10**(C1 的"硬约束"claim 此前只是计数器):
>   顺序=便宜的 overlap 检查(仅端点)→ 裁帧 → 语义门 → 原子写;超阈轨迹在
>   写入前拒绝并**对同一源图重采**(RSWM_TRAJ_MAX_TRIES=12,坏轨迹不消耗源图),
>   耗尽才换源图并计数;禁止静默放宽阈值。产 overlap_audit.json(总尝试/
>   overlap 拒/语义拒/源图耗尽/各场景接受率/max·p50·p90·p99),**最终断言
>   max(overlap)<0.10 否则构建判废**。
> **新实验**:34 可学习性探针(**判门作业**,单段+四方向+高重叠、独立 profile
>   豁免 overlap、pose_mode=drop;角误差 >45° 即 exit 3 阻断全部后续 GPU ——
>   "范式不迁移"与"配方教不会"必须可区分)、35 逐段富文本臂、36 真 no-pose 臂;
>   编排器改为 01 → **34 判门** → 20-33/35/36(全部 afterok:34)。
>   make_variant_h5 加 rich_text 模式(逐段方向语言化,taxonomy 指纹校验)。
> 测试:新 test_pose_timeline.py **36 项**;全套六件套全绿。
> ⚠️ **20-33 既有结果全部作废**,34 通过前不得提交主矩阵。


> **v2.7.13(2026-07-29,拍板执行:方案 A 冻结 + 受控 P0 验收埋点)**
> 四轮真值校准终局(预注册规则执行,CALIB_RC=9 为"不可完美分离"的设计返回):
> clspres 被 00006=0.09/00016=0.10 vs 坏样本 00009=0.11 的反例否决;事后
> OR 组合按预注册禁止。**冻结规则 = frac >= 0.2777777777777778(全精度,
> 9 帧,2/9 与 3/9 分界中点)**,已写死为代码默认 `FROZEN_P0_THRESHOLD`
> (env 覆盖仅供诊断);0 坏放走/3 误拒(Port/BareLand/Farmland → datasheet
> 记录欠采样偏置);clspres 三键降为诊断证据;第五轮特征迭代禁止;决策留档
> rule_decision_v2712.txt(服务器只读+SHA256)。
> 受控 P0 验收埋点(拍板清单):①包装器建库后落 `semgate_attempts.json`
> (全部尝试含被拒的 scene|sid→class,tmp 清理前由入口脚本抢救到 WORK);
> ②新 `server_run/semgate_build_report.py`:逐 source_class 尝试/拒绝/接受、
> 已知误拒类单列、最大 tries 对照池耗尽、semgate_frac 全分布(p0_smoke 与
> 01_build 自动出报告,信息性不判门);③人检可视化改全部样本(--max 20)+
> 五类视觉重点写入 DEPLOY;④规则两级状态(P0 候选=已冻结批准;生产/全量=
> 未冻结)如实入档。测试 86→**88 项**(冻结值全精度+默认=冻结断言)。


> [v2.7.12 终审补丁,单 agent 对抗审查 5 确认全修+实证可达] C1 rc9 分支弱分母
> 报告漏 clspres 族(该分支唯一用武之地;实跑曾放走 clspres=0.99/类分母
> 0.0007 的坏样本零输出)→ 特征映射铺全,报告可达性已实证;C2 明细 raw 键
> 写死 pres(类分母塌 86 倍显示成 0.61)→ 按主特征取;C3 checker 增独立
> 类分母警示(WEAK_CLS_DENOM_WARN=0.01 暂定 —— 单类质量天然比池质量低
> 1-2 个数量级,0.05 线不可复用);C4 stitched+场景下 cls 三值=None 的
> 守卫测试(故障注入曾 83 全绿放行 0.0 假证据);C5 池可加性断言钉死
> "同 softmax 空间"前提(池质量==Σ单类质量,1e-9;故障注入当场红);
> N3 clspres 覆盖数提示(防"不可分"伪结论)。测试 83→**86 项**全绿。

> **v2.7.12(2026-07-29,四轮校准特征:具体类别保真比 —— 三轮 CALIB_RC=9 后)**
> 三轮实证(semgate_v3,pad 后):工程链全绿、血统 12/12 MAE=0,但
> frac(最优 0.278,0 放走/3 误拒)与 pres 都不可分 —— Port 帧仍有码头和船
> 但池级更像停车场、BareLand 像 Desert 压低 rural 池:场景池保真 ≠ 类别主体
> 保真(pres 对坏样本 0.19/0.34 有效,但 00006=0.12/00016=0.19/00019=0.29
> 同样低)。新特征 **cls_pres_ratio = source_class 单类 softmax 质量的
> 帧/场景之比**(同一相似度矩阵复用,零额外前向):直接问"River/
> StorageTanks/Port/BareLand 这个类还在不在";比 retention 的全库 top-1
> 宽松,比池质量聚焦。"stitched" 哨兵样本自动 None。
> 接线:GateVerdict/checker(表加"类保真"列,JSON 三键)/校准器
> (clspres/clspres+frac/clspres|frac 族、独立扰动步长、弱分母按
> scene_cls_mass、明细按主特征取值);判定规则仍 frac≥阈值不变。
> **四轮预注册决策规则已写入 DEPLOY §3b:clspres 分得开→用它;分不开→
> 落方案 A(frac≥0.278 受控 P0 + datasheet 记录欠采样偏置);不追加第五轮。**
> 测试 78→**83 项**全绿;校准器"frac/pres 不可分但 clspres 可分"形态自测
> 推荐 clspres 族正确。semgate_v3_1(判定语义未变,仅增特征)。


> **[v2.7.11 终审补丁,两轮对抗审查共 11 确认 + 高价值 nitpicks 全落]**
> 门本体:C2【pad 方形】——CLIPProcessor 中心裁剪砍掉矩形帧 42% 宽度而
> full_scene 完整,两轮校准分不开的候选根因,帧/场景同函数 pad(改变每个
> 分数,semgate_v3);C1 retention 对 "stitched" 哨兵 → None、未知类名硬错
> (旧 0.0 假证据曾污染待拍板规则);0帧 NaN/桩 details 长度/归一化图拒收/
> taxonomy 互斥断言进评分进程/阈值三键校验。
> 校准器:_key_dist 空序列崩溃(两条已复现路径)修 default;单类样本 FATAL
> (13 全 valid 时曾推荐 frac>=0.0=不设门还带稳定✓);退化族剔除(pres/ret
> 全空不进候选);pres 独立扰动步长 0.05(借用 1/9 曾系统性压制 pres 族);
> 弱分母报告移至真正可达的 rc9 分支(perfect 分支 fp==0 数学上不可能放行
> 坏样本,旧打印是同义反复假保证);同族阈值区间折叠输出。
> 血统脚本:uint8 硬守卫 vs 全链 float32(合法数据恒 STOP,P0)改值域判定;
> 零样本受检 STOP;exit 码修正(sys.exit(str)→1 的三处);钳制窗口改同链
> 重建不再假 STOP;--keys 不存在键干净 STOP。
> 证据链:semgate_enabled 补进转换器透传(旧漏传使 exit-8 守卫在一切生产
> 产物上不可达 —— 70 项测试全绿是夹具手写 attrs 绕过了转换器,教训:兼容
> 矩阵必须过真转换器);flag 改真实注入态(env+use_hf),--no-hf 不误杀;
> v1 回退+门开 → 硬错;签名解析缓存;stored_pres_ratio_raw 对账;
> bad_attr/enabled_mismatch 进 JSON。测试 70→**78 项**全绿。

> **v2.7.11(2026-07-28,审查方最终清单全落地 —— 分发定稿/血统升级/证据链闭环)**
> 1. **v5 门调用分发定稿**(审查方设计原样采纳):inspect.signature 判定 +
>    VAR_KEYWORD 支持(**kwargs 代理收得到新字段)+ 签名不可解析硬错 +
>    内部 TypeError 原样传播绝不二次调用。专项测试 5 用例(含调用计数断言)。
> 2. **血统验证升级为两级**:①坐标重建优先 —— trajectory 逐帧窗口中心 +
>    pose_window_size 按 v5 _get_bounds 同款取整裁 full_scene,用生成链同款
>    cv2.INTER_LINEAR 放大与存储帧逐像素比(最强一致性证据);②模板匹配
>    回退(帧降采样回窗口)+ 最优/次优唯一性检查。输出 MAE 全分布
>    (min/median/p90/p95/max);阈值 12 标注为暂定、以真实分布重定;措辞
>    统一"血统一致性检查",不称"证明"。dtype/属性异常/定位不稳 → exit 10。
> 3. **建库 pres 证据链闭环**:(a) 拒后重采只落被接受尝试的值(测试钉);
>    (b) metadata 记 semgate_enabled,general/关门时字段【缺失】而非 0.0;
>    (c) checker 分开报 stored_*(建库)与 recomputed_*(重算)+ pres_drift,
>    不许重算值冒充建库证据(模型/taxonomy/TAU 变化都会漂)。
> 4. 弱分母放行报告标签定稿:WEAK_DENOM_ALLOWED_VALID(降低可信度)/
>    WEAK_DENOM_ALLOWED_INVALID(直接阻塞)。
> 5. **旧 H5 兼容矩阵**(4 用例):v261 旧 H5 无 stored 字段 → missing
>    合法、report-only 正常、drift=unavailable(不填 0.0 不判缺陷);
>    新 H5 正常算 drift;semgate_enabled=true 但无建库证据 → exit 8;
>    enabled=false/general 缺字段合法。checker 读 metadata.semgate_enabled
>    (旧 H5 无此 attr = 旧构建,跳过一致性检查)。
> 测试 59→**70 项**全绿。验收门与三分支决策按审查方原文写入 DEPLOY §3b。
> 正式规则未冻结;P0 与全量链禁止,直至真实校准 + 人工规则审批。


> **v2.7.10(2026-07-28,pres_ratio 审查六点回应 —— 三点构造已满足,三点补齐)**
> 审查方对 v2.7.9 设计放行、对实现提六点,逐条核实:
> ②分子分母同口径 —— **构造满足**:场景行拼进与帧同一批,一次
>   _sim_matrix_clip(同 bank/同 TAU/同预处理/同归一);dataset 一致由 R3
>   硬门保证(None 即 raise,不可能帧 AID 场景 union)。
> ③full_scene 只编码一次 —— **构造满足**:整样本一个 batch(9 帧+1 场景)。
> ①分母稳定性 —— 补:_EPS_MASS=1e-6 命名常量;逐帧 pool_mass 与
>   **pres_raw(未截断)** 落盘;verdict 增 pres_ratio_raw(未截断均值)。
> ④min(1,·) 饱和 —— 补:raw 全链落 JSON(帧级+样本级),饱和分析随时可做。
> ⑤共同抵消风险(00009 场景本身草多水少→分母也小→比值可能≈1)—— 补:
>   checker 对 scene_pool_mass<0.05 的样本打「⚠️分母弱」警示(高保真比不可
>   轻信);校准表加 raw/场景质量两列。**该风险最终只能由真实 00009 数据
>   检验 —— 公式不自证,与审查方一致。**
> ⑥完美分离≠可冻结 —— 补:校准器对每条完美规则打印【坏样本距阈值】与
>   【最近有效样本距离】;既有 ±1帧扰动稳定/分离带宽/简单性三重排序保留;
>   输出显式声明仅拟合 15 样本。
> 测试 55→**58 项**(raw 落盘/截断为 1 且 raw>1.5/softmax 未饱和场景构造)。
> SEMGATE_VERSION 维持 semgate_v2_2(评分语义未变,只增记录)。禁 P0 不变。


> **v2.7.9(2026-07-28,二轮校准复盘:换判据,不再调词)**
> 二轮真实校准(v2.1 库):坏样本 00009/00011 correctly 低分,但 4 个好样本
> 更低(BareLand/Farmland=0.00、Port=0.11、Beach=0.33),CALIB_RC=9。逐帧
> 诊断给出病根:①BareLand↔Desert 跨池视觉同义(taxonomy 固有,词库无解);
> ②margin ±0.01 时逐帧硬比是掷硬币(Farmland 输给 square plaza);③00006
> 样张亲验:码头船排在 CLIP 眼里≈停车场的车 —— 零样本文本判据的固有混淆。
> 结论:继续调 prompt 是死路,换判据 ——
> 1. **场景参照保真比 pres_ratio**(新主候选特征):对帧与它自己的
>    full_scene 各算「声明池 softmax 概率质量」(类组模板均值分,TAU=100),
>    pres = mean_f min(1, mass_帧/mass_场景)。关键性质:帧与源场景共享同样的
>    零样本混淆(船排≈停车场、BareLand≈Desert 在分子分母同现)→ 比值把
>    混淆抵消,量的是「源场景的声明语义有多少在裁剪后幸存」。00006 型
>    (帧与场景同混淆)→ 比值≈1 通过;00009 型(场景有河、帧没河)→
>    比值塌 → 拒。此为假设,由三轮校准检验。
> 2. check_frames 增 full_scene 参数(v5 建库门/checker 已接线;CLIP 批里
>    带场景行,一次前向);详情帧加 pool_mass;表格/JSON 加 保真比 列。
> 3. 校准器规则族扩 pres / pres+frac(与)/ pres|frac(放宽,警示);
>    修 _sep 按规则主特征算(旧版对 pres 族错用 frac 维,带宽恒负)。
> 4. 空格拼接指控核实:本包 build_prompt_bank 全部串有空格(repr 逐串验证);
>    部署侧请跑认包命令核对副本。retention 维持诊断轨(二轮实证:好样本也有
>    0.00,不能硬门)。
> 5. 测试 49→**55 项**(保真比数学 5 项:同分布=1/衰减单调/无场景=None 等);
>    校准器自测三形态(可分 rc0 / frac 不可分但 pres 可分 rc0 / 全不可分 rc9)。
> SEMGATE_VERSION → semgate_v2_2_20260728。规则仍待校准拍板,禁 P0 不变。


> **v2.7.8(2026-07-28,外部盲审 5 风险点核实:3 真 2 已备,全部落地)**
> R1【真】短语覆盖不全:69 类中 52 类无 override,库里存在 "showing
>   commercial" 类破句 → 全 69 类短语表补齐(单测钉:新类入 taxonomy 缺表项
>   即红)。R2【真】负池数量偏置:逐条 max 下 E[max] 随 n 增长,AID rural
>   正 11 条 vs 负 50 条,margin 被系统性压负 → 评分改【类级模板均值】聚合
>   (每类 2 条模板取均值为组分,组间再比 max;单模板噪声峰值被抑制;池间
>   类数差异保留 —— 那是语义覆盖不是噪声)。R3【真】dataset 静默回退:
>   checker 缺 source_dataset attr 时曾默默用 union 库(wetland 污染复活路径)
>   → CLIP 路径 dataset=None 直接 raise;checker 缺 attr 判数据缺陷 exit 8。
> R4【已备,增强】保持度同义类风险:details 增 top_classes(top-3 类+分),
>   同义混淆可直接看;保持度维持诊断轨,启用决策点(等价类/top-k/差距)
>   留给校准拍板。R5【已备,增强】小样本过拟合:完美规则按 简单性→±1帧
>   扰动稳定→分离带宽 三重排序,OR(frac|ret)放宽型单独警示默认不推荐,
>   输出显式声明"仅拟合 15 样本,不证明泛化"。
> SEMGATE_VERSION → semgate_v2_1_20260728;测试 48→49 项(R3 回归);
> 明文审查包已导出 /root/ctw/semgate_review_v278/(6 文件+SHA256SUMS)。


> **v2.7.7(2026-07-28,语义门评分器重设计 —— 20 样本人检真值上 v1 分离失败)**
> 校准复盘:崩溃修完后评分器不可靠 —— 7 误拒 vs 真值仅 2 坏;BareLand=0.00、
> Farmland=0.11 比真坏样本(00009=0.22)还低;根因=手写干扰负类
> ("grassland"/"bare land")与 rural 池合法类(Meadow/BareLand)对撞、
> Beach 沙滩帧同型中招;且旧输出只有 9 帧二值票,无法诊断。不存在全局阈值
> 可分(用户已证)。重设计:
> 1. **prompt 库从 scene_taxonomy.json 派生**:正类=本池全部类名成句
>    (池覆盖自动完整),负类=其余三池(含 general)类名成句;手写干扰库
>    删除 —— 四池互斥由 taxonomy 断言保证,冲突按构造消灭。库按数据集构建
>    (AID 样本用 AID 池;union 回退会引入 RESISC45 wetland,校准场景禁用),
>    v5 建库门传 dataset。SEMGATE_VERSION → semgate_v2_20260728。
> 2. **逐帧诊断**:pos_max/neg_max/margin/胜出正类词/胜出负类(池+类)
>    全量落 GateVerdict.details;checker 新增 --dump-frames 落 JSON,
>    表格加 margin/保持度两列。
> 3. **source_class 保持度双轨**:全库 class 条目 top-1 == source_class 的
>    帧占比(StorageTanks 离场型失败靠它抓);判定规则仍是场景轨
>    frac≥阈值 —— 组合规则待真值校准拍板后启用,不擅自改判据。
> 4. **新 calibrate_semantic_gate.py + 真值入包**
>    (calibration_labels_smoke_v261.json,20 样本 2 无效):对表网格搜索
>    frac / frac+margin / frac+ret(与) / frac|ret(或)四族规则,完美分离
>    输出全部候选(放走坏样本代价 10×);找不到 exit 9。自测:v1 失败形态
>    →rc9、期望形态→rc0 复现正确。
> 5. 测试 34→**48 项**(库冲突消灭 7 断言 + 双轨评分纯 numpy 6 项 +
>    扩展桩契约);v5 两拷贝同步;v2.7.6 的 _coerce_features 原样保留。
> ⚠️ 语义门规则拍板前禁止重建;v261 smoke H5 继续只读保留(校准真值集)。


> **v2.7.6(2026-07-28,真实 CLIP 校准 P0 热修)**
> 服务器桩单测虽为 28 PASS / 0 FAIL,真实 cached CLIP 校准仍在
> `SemanticGate._coerce_features()` 失败:`get_image_features()` 返回的
> `pooler_output` 已是投影后的 `9×512`,旧兼容层却再次送入输入维为 768 的
> `visual_projection(768→512)`,触发
> `mat1 and mat2 shapes cannot be multiplied (9x512 and 768x512)`,
> `SEMANTIC_CALIBRATION_RC=1` / PBS=1。根因是 Transformers v5 将
> `get_*_features` 改为返回 ModelOutput,但其 `pooler_output` 仍等价于旧版
> helper 返回的【已投影】特征。
>
> 修复:按 projection 维度 fail-loud 归一——尾维先匹配 `out_features` 就
> 直通,仅匹配 `in_features` 才投影一次,其他维度/3D 硬错;输出维优先同时
> 封死 text `512→512` 方阵的静默二次投影。生产调用显式
> `return_dict=True`,兼容 Tensor、ModelOutput 属性和 mapping 字段;
> `SEMGATE_VERSION` 升为 `semgate_v1_1_20260728` 留存数据溯源。回归新增
> 512 直通、768 单投影、方阵不重投、dict 字段优先、未知维/3D 拒绝;
> 完整桩测试 **34 PASS / 0 FAIL**。本地桩测不能代替服务器真实 cached CLIP
> 校准:仍须确认 RC=0、JSON 非空,再人工判读 00009/00011 分离。


> **v2.7.5(2026-07-28,最终汇总接线与操作入口收口)**
> ① **[P1]** collect_results 的 r128 Rank/方向消融改为 corrected
> `eval_r128_{sem,geo}_s42` 优先、run_all 无后缀目录回退;新增 corrected 30 号
> r8 Rank 行,r32/r64 显式标为 legacy/run_all。② **[P2]** qsub_all 头部与
> PBS/旧流程文档统一为默认拒绝、仅 FORCE_LEGACY=1 可复现;当前唯一推荐入口
> 是 qsub_corrected_rerun。③ corrected 脚本末尾补 geostrip Track B 与
> collect_results 命令,参数效率纳入 r128 上界,DEPLOY/总入口同步;训练矩阵
> 更正为 13 项(含 32/33)。


> **v2.7.4(2026-07-28,汇总层三套目录规范收敛 —— 外部审查四轮唯一遗留项)**
> 决策:冻结范围只含 corrected 20-33 链(D3 拍板既定),legacy qsub_all 走
> 废弃而非兼容。① qsub_all.sh 加 FORCE_LEGACY=1 硬守卫(默认 REFUSED,与
> 错构建器守卫同型);② collect_results 几何行双候选(corrected 的
> eval_r16_geo_s42 优先,run_all 的 eval_r16_geo 回退)—— 修掉"corrected 链
> 汇总查不到几何"的缺口;③ 19_collect 的对齐产物改名
> aligned_r128_legacy / aligned_r128_geo_legacy,不再以 aligned_main 之名
> 冒充 headline。外部验收:v2.7.3 三项修复全部生效、28 个 eval_lora 调用点
> 参数全对、corrected 主链无阻断。


> **v2.7.3(2026-07-28,外部审查三轮 2+1 项全修 —— ① 又是上一轮自己引入的)**
> ① **[P0] p1_main:73 评估命令被注释截断**:v2.7.2 用"前缀替换+行内注释"改
>   该行,注释吞掉了后面 4 个参数 → eval_lora 需 6 参,set -u 下 P1 在训练+
>   baseline 跑完后中止(p1_main/run_all 均不可用)。修:整行重写为续行格式;
>   并全仓扫一遍 eval_lora 调用点参数计数(全部 7 词 ✓)。教训:改带续参的
>   调用行必须整行替换,改完机器验参数数。
> ② **[P1] collect_results/19_collect 仍指旧 r128 路径**:p1→p3 新流程下
>   summary.md 会【静默把 r128 消融当主结果】、r16 几何显示缺失。修:主表/
>   方向行指向 eval_r16_sem_s42 / eval_r16_geo(r128 降为 Rank 与方向消融行);
>   19_collect 属 legacy 链,跟随 02/04 的显式 eval_r128_* 产物。
> ③ **[P3]** DEPLOY_D4 认包文案残留 v2.7.1 → 已统一。
> 二轮修复验收(外部复核确认):baseline 两模式/全失败门/最佳 checkpoint/
> 对齐清理/P3 r16/DDP 提示 —— 全部通过。


> **v2.7.2(2026-07-28,外部审查二轮 5 项全修 —— 其中 ① 是 v2.7.1 F1 修复自己
> 引入的 P0,再次印证"审自己当轮改动")**
> ① **[P0] baseline 评估必然崩**:v2.7.1 的失败门读了 run_baseline_inference
>   的局部变量 failed(以及 metrics_only 分支外的 samples)→ 正常模式跑完
>   昂贵推理后 NameError、--metrics_only 当场 UnboundLocalError,阻断
>   p0_smoke/p1_main/31 号。修:函数显式 `return failed`,main 先
>   `samples, failed = [], 0` 再捕获返回值(AST 静态验证两条路径)。
> ② **[P1] r16/r128 统一收尾**:p3_direction(r128_geo→r16_geo)、
>   p1_main:73 eval tag、p2:37 提示;legacy 链名实相符化 —— 02/11 号改写
>   显式 train_r128_{sem,geo} 目录(不再污染 r16 命名的 T_MAIN/T_GEO),
>   下游 03/04/12/13-16/18 全部跟随显式目录(消费端 8 处,T_MAIN/T_GEO
>   在 PBS 链残留=0)。
> ③ **[P1] 第 10 轮最佳模型复制错**:`sorted(glob("checkpoint-epoch-*"))[-1]`
>   是字符串排序(epoch-9 > epoch-10),默认 10 epochs 正好触发 → 改用
>   save_checkpoint 返回的目录(它本就 return save_dir)。
> ④ **[P2] eval_aligned 残留旧样本**:复制交集前清空目标 generated/gt/meta
>   子目录(同目录重跑更小交集时旧 npy/json 曾静默混入指标)。
> ⑤ **[P3] DDP 守卫提示的假参数**:--inactive_model_cpu 是 store_true 不能接
>   0,提示改为真实开关 --no_inactive_model_cpu(两处)。


> **v2.7.1(2026-07-27,双源修复:Gadi 实炸 1 项 + 外部审查 11 项全 CONFIRMED)**
> 来源A(服务器实证):**CLIP API 漂移** —— 新版 transformers 的
> get_text_features 返回 BaseModelOutputWithPooling,旧代码 `.norm()` 当场
> AttributeError(作业 174857936)。修:`SemanticGate._coerce_features` 归一层
> (Tensor 原样 / CLIPOutput.text_embeds / pooler_output→过对应投影头 / 未知
> 字段硬错)+ 图文特征维度一致性断言 + 5 项 API 漂移回归测试(假对象,免
> torch)。教训:py_compile+桩测试抓不到运行期 API 漂移。
> 来源B(外部审查,3-agent 逐行核实后全部 CONFIRMED):
> **F1[P0] 评估失败误报成功**:推理异常被吞、全失败也 exit 0、空 results 照写
> "成功"、p0_smoke 无条件盖章且 stamp 永久跳过 → 五处门:失败计数+全失败
> raise+20% 下限门(evaluate/evaluate_baseline)、空 results sys.exit(1)、
> eval_aligned 无共同样本 exit 1、p0_smoke 盖章前验 comparison.json 含 overall。
> **F2[P0] LoRA 缺失静默退化基模型**:low_noise//high_noise/ 任一不存在只
> warning 继续 → "LoRA 结果"可能是基模型或单专家混合体。改 FileNotFoundError
> 硬错(evaluate/generate_samples 共用 load_lora_model,一处封死)。
> **F3[P1] 断点续评漏样本**:完成标志从 gen npy(最先写)改 meta json(最后
> 写);半成品 gen/gt 被重推覆盖。**F4[P1] 恢复训练覆盖真最佳**:最佳判定挪到
> save_checkpoint 之前,checkpoint 里的 best_val_loss 不再滞后一轮(旧行为
> epoch1 存 inf,恢复后 0.6 会覆盖真最佳 0.5)。**F5[P1] 语义门 fail-open**:
> 未知 scene_category(拼错/新类未登记)从静默跳过改 ValueError;检查器废除
> default-general(缺失/非法 attr 收集后门模式 exit 8,report-only 仅警告)+
> 空转守卫(0 个语义类受检不许 PASS)。**F6[P3] DDP+CPU offload**:_wrap_ddp
> 入口守卫(可读错误;现役单卡路径不可达;H200 141G 可 --inactive_model_cpu 0)。
> **F7[P2] SciPy sqrtm(disp=False)**:new/metrics.py 与 legacy metrics.py 共
> 4 处修掉(SciPy 1.18 实测 TypeError);⚠️ 更正 v2.6.1-F5 的不实声明 ——
> 当时只修了 recompute_fvd_i3d.py,其"metrics.py 本就无 disp"注释与代码矛盾,
> 已删。**F8[P2] split 溯源**:两处落盘 seed 字段拆为 train_seed/split_seed
> (旧字段在多种子实验里显示 43/44 而实际划分用 42)。**F9[P2]** 每样本
> mkdtemp 加 finally 清理。**F10** p1-p6 链按 D3 拍板统一 r16(旧 p1 训 r128
> 但目录叫 r16 → param_efficiency tag 错配/eval 目录找不到),r128 移入 p2
> rank 环 {32,64,128}。**F11** DEPLOY_D4 标题/解包名/认包判据改 v2.7.1。
> 测试:test_semantic_gate 28 项(+API 漂移 5 +F5 回归 4);全套六件套全绿。


> **v2.7(2026-07-27,P0 人检失败的修复:裁剪后语义保持门)**
> 服务器 P0 复盘:自动四门(池/重复/泄漏/配额)全 PASS、`P0_SMOKE_RC=0`,
> 但人眼可视化门 FAIL —— sample_00009(water/River)12 帧里 10 帧纯草地,
> sample_00011(urban/StorageTanks)4 帧 100% 河水。根因:AID 标签是【整图】
> 属性,轨迹滑窗离开类别主体后,prompt 的场景语义("a water-dominated …")
> 对实际帧成了假监督 —— 文本条件通道正是本 benchmark 的被测对象。修复:
> 1. **新 `lora_remotesense/semantic_gate.py`**:CLIP 零样本逐帧场景评分
>    (正/负 prompt 库,帧同意 ⇔ 对正类最大相似度 > 负类;样本通过 ⇔ 同意帧
>    占比 ≥ 阈值,默认 0.6,`RSWM_SEMGATE_MIN_FRAC[_WATER/_URBAN/_RURAL]`
>    可调)。general 无语义声明不设门。source_class 定位为源图 provenance,
>    门只对 prompt 声明的 scene_category 负责(无像素标注时"不离开类别主体"
>    以场景级逐帧评分为代理)。
> 2. **建库期拒绝采样**(v5 `_process_scene`,两份拷贝同步):不达标 → 确定性
>    游标推进换下一张池内源图重采(每次尝试消耗一个索引,种子链 (场景,索引)
>    不变,池耗尽由 v2.4 无回绕守卫硬错);连续 `RSWM_SEMGATE_MAX_TRIES`(6)
>    张都不过 → 硬错不静默。样本新增 `source_scene_id`(实际消耗的池索引)
>    与 `semgate_frac/threshold/n_scored/tries` attrs,转换器透传最终 H5。
> 3. **registry 对账修复**:`_annotate_source_identity` 优先按样本落盘的
>    `source_scene_id` 查登记表 —— 重采后"第 i 个接受样本"≠"scene_id=i",
>    旧顺序计数会把源身份错配到被拒尝试的源图上(溯源/泄漏检查跟着全错)。
>    旧构建无该 attr 时回退顺序计数(与 v2.6 逐字节一致)。
> 4. **第四道门 `server_run/check_semantic_consistency.py`**:建后独立复核
>    (同评分器重跑;复核阈值 = 建库阈值-0.1 防边界抖动;FAIL=exit 8),
>    `--report-only` 校准模式按占比升序打表。已挂进 p0_smoke / 01_build /
>    p1_main(sha-bound stamp 与前三道同构)。**部署顺序:先在已有 smoke H5
>    上 report-only 校准 —— 人检已定的 00009/00011 必须与其余样本明显分离,
>    分不开就不许带门建库(DEPLOY_D4 §3b)。**
> 5. **门注入**在包装器(use_hf 时;`RSWM_SEMGATE=0` 显式关门须大声警告,
>    关门构建不得用于论文);CLIP 权重进 offline_prefetch 默认清单(~600MB);
>    eager 加载 —— 缺权重建库开工前就炸,不烧完数据才发现。
> 6. `builder_version` 改由 `taxonomy_version` 派生 + semgate 标记(修正
>    "串里 taxonomy_v2.5、实际 water_v1_20260726"的漂移)。
> 7. **新 `server_run/test_semantic_gate.py`(19 项)**:本体(含 00009 帧型
>    必拒)/ v5 重采循环【真跑】(游标跳号、source_scene_id 对账、耗尽硬错、
>    无门=旧行为)/ 检查器端到端(report-only、exit 8、general 跳过)。
> ⚠️ 语义门改变数据集组成(拒绝采样),smoke 与全量都必须全新构建;
> 阈值为暂定,以 §3b 服务器校准为准。


> **v2.6.1+tests(2026-07-26,另一台机器的并行线合并回来时补的测试)**
> 背景:同一 v2.5 起点上有两台机器并行做 v2.6。本包(v2.6.1)是主线、更完整;
> 另一条线的实现已作废归档,只捞回它的测试覆盖(主线原本只有
> test_taxonomy_prompt.py 一个测试,而 v2.6 新增的路由判定/双栏接线恰是
> 「错了不报错、只是数字悄悄变味」的那类逻辑,最需要测试钉住)。
> 1. 新 `server_run/test_route_resolution.py`(15 项):固定
>    _resolve_text_only 的优先级(显式 :text/:geo > meta 的 pose_conditioning
>    > 目录名且必须唯一命中)与四类硬失败(后缀与 meta 矛盾/混版本目录/
>    同目录两种 pose_conditioning/未知取值),以及 base、shufpose、sem_geo
>    这类歧义名必须 FATAL 而不是静默默认。**实跑确认主线行为与规格逐条一致。**
> 2. 新 `server_run/test_wellposed_dualcolumn.py`(17 项):造真 eval 目录后
>    走 compute_metrics 端到端,断言 Angular_Error_WellPosed_*/WellPosed_N/
>    _Min_Net_Ratio/_Ratio_Missing 的键与数值、ratio=null 单独计数、
>    v2.6 前旧目录不产双栏但其余指标照常、混版本目录硬失败、scores.json
>    逐样本行含 gt_net_displacement_ratio 与 well_posed。
> (另一条线的教训:曾把 net_ratios 参数误加到 estimate_motion_direction 上,
>  compute_all_metrics 缺形参却引用裸名 —— py_compile 与「只测单个函数」的
>  自测都抓不到,实机调外层才炸。故本轮两个测试都强制走最外层入口。)


> **v2.6.1(2026-07-26,v2.6 对抗审查 5 项 CONFIRMED 全修 + 1 项自审加固)**
> 独立审查(候选 17,驳回 12 —— 驳回项含 import 遮蔽/键名漂移/掩码错位等,
> 均以实跑或 file:line 证据排除)。修复:
> **F1[P1] 冒烟门被「先写盘后判决」旁路**:run_baseline 刻意先写 scores.json
> 再判 min-success(exit 4,保留失败现场),p6 的前置门却只查文件存在 →
> 部分失败的冒烟(如 2/3=67%)也留下文件、永久放行全量。改为解析
> success_rate>=0.8 且 n_success>=2(夹具三态验证:全过/67% 拦/旧格式拦)。
> **F3[P2] p6 子集旧布尔 stamp + 同名 dst 旁路 SEM→GEO 修复**:v2.5 时期
> stamp/SEM 子集残留 → 不重抽 → identity 守卫反复 FATAL 且病因无人点名。
> 改 GEO_H5 sha 绑定 stamp(v2.4.3 机制)+ dst 更名 eval_subset_20_geo.h5
> 物理隔离;**重抽时同步清下游 eval 产物与 p6_eval stamp**(断点续推只看
> npy 存在,不清会拿旧数据产物给新子集打分);mvsplat_smoke.sh 子集同款
> sha 绑定 + 清 $OUT;identity 守卫报错补陈旧子集排查指引。
> **F2[P2] I3D=1 开关文档-代码漂移**:文档四处写 I3D=1,prefetch 3b 段却
> 无条件执行 → 按本仓「按需开关」约定补上 if 门。
> **F4[P2] p6 插行后 file:line 引用集体失准**:external_stubs/README_BASELINES
> 里 :22/:29-33/:66/:71-85 等锚点刷新为段名+现行行号;顺带修 README_BASELINES
> 头部两处 v2.5 就已过期的断言("三个外部候选全是桩"、--list 快照里
> image_outpainting 仍标 stub)—— 过期结论性文档会主动带偏后续审查。
> **F5[P2] sqrtm(disp=False) 是 scipy 1.18 定时炸弹**(1.17 已 DeprecationWarning,
> 1.18 移除该 kwarg 且返回值形态变化):recompute_fvd_i3d 改无-disp 调用 +
> tuple 防御,requirements 的 scipy 无上界也安全;自测 11 项重跑全绿、告警清零。
> **自审加固**:run_trajectory_eval 零匹配 FATAL(目录在、但 generated/ 无任何
> npy 与 GT val key 对上时,旧行为是该配置无声不出现在输出里 —— 传错 gt-h5/
> split 不同源的最后一个静默洞)。回归全套重跑 8/8 绿(路由 24/well-posed e2e
> 16/桥 6/registry/taxonomy 49/resisc45 9/TM 99/import 链)。

> **v2.6(2026-07-26,五项代码补齐:O9/O11/well-posed/I3D/MVSplat360 桥)**
> 1. **Track B 报表键补全(O9 修复)**:`run_trajectory_eval.py` 的 TRACKB_KEYS
>    补上 `segment_success_rate_30/45`、`turn_precision`、`turn_f1`、
>    `waypoint_endpoint_error`、`num_gt_turns`、`num_recovered_turns` ——
>    指标层(trajectory_metrics)早已实现、报表层此前静默丢弃,师兄意见
>    2/3/4 的产出现在真正落到聚合与 per_sample。v2.3 近共线退化置 None 清单
>    同步扩展(success_rate/precision/f1/endpoint 同属 knot/GT-拐点依赖;
>    num_recovered_turns 只看恢复路径、保留作诊断)。
> 2. **commanded 路由判定加固(O11 修复)**:text_only 不再由目录名字符串猜。
>    优先级 = 显式后缀(`--eval-dirs name=dir:text|:geo`)→ eval 目录 meta 的
>    `pose_conditioning` 标记(v2.6 evaluate 起逐样本落盘,值由【poses 是否全
>    identity】的数据级判定给出,SEM 文件 attr 交叉校验、不符硬错)→ 仅旧目录
>    退目录名启发式且只认显式词根;歧义/冲突/混版本目录一律 FATAL。旧启发式
>    对 `base=$W/eval_base`(31 号在 SEM_H5 上评,是 text 路由)会静默判成
>    geo —— 正是要根治的形态。输出新增 `route_source` 供出表脚本拒收启发式行。
>    eval 目录不存在也改为 FATAL(旧行为静默产出空配置)。
> 3. **Track A well-posed 双栏(审稿人小组必做项)**:`new/evaluate.py` 每块
>    (overall/场景/方向/交叉)新增 `Angular_Error_WellPosed_{Mean,Median,Std,
>    Count}` + `WellPosed_N`;well-posed = GT 命令轨迹 `polyline_net_ratio >=
>    0.2`,阈值经 inspect 直读 `net_direction_from_path` 默认值 —— 与 Track B
>    门控**单一真源**,永不漂移。meta 落 `gt_net_displacement_ratio`,
>    scores.json(schema v2)逐样本落 ratio + well_posed;复用 overall 的逐样本
>    角误差做子集统计,零额外光流。版本卫生:meta 全缺该键(v2.6 前目录)→
>    大声跳过双栏;部分缺 → 硬错(断点续推混版本)。aggregate_seeds 默认指标
>    加 `Angular_Error_WellPosed_Mean`(聚合旧目录用 --metrics 去掉)。
> 4. **标准 I3D-FVD 离线重算**:新 `server_run/recompute_fvd_i3d.py` ——
>    StyleGAN-V 版 i3d_torchscript(400-d logits,FVD-16 帧策略,两侧同策略)
>    对 eval 目录(generated/ + gt/)离线补算文献可比的 `FVD_I3D`;加载后
>    dummy 前向断言契约(shape=(N,400) 且有限),纯 numpy 自测 11 项全绿
>    (闭式高斯对照/均值漂移解析值/方差 tr 项/NaN 硬错/短 clip 硬错/缺配对
>    硬错/假提取器端到端/确定性)。`--patch` 可写回 eval_results.json(先备
>    份)。metrics.py `_load_i3d_model` 更名 `_load_video_feature_model`(留
>    兼容别名)+ docstring 改为如实的 r3d_18 代理声明 + 结果新增
>    `FVD_Extractor="r3d_18_proxy"` 标记。offline_prefetch.sh 新增 I3D 段
>    (⚠️ 两候选 URL 在打包沙箱被代理拦截【未实测 200】,首跑请在联网机确认,
>    失败用 I3D_URL= 覆盖;下载失败只 WARN 不断预取)。
> 5. **MVSplat360 桥接层 + 冒烟门(档B)**:external_stubs 的 mvsplat360
>    run() 从纯桩改为【产物回收桥】—— 在 mvsplat env 跑
>    `evaluate_mvsplat360.py --no_metrics` 后,`RSWM_MVSPLAT_EVAL_DIR=<dir>`
>    走 `run_baseline.py --baseline mvsplat360 --skip-failed --min-success 0.8`
>    即得统一 record(逐样本校验 prompt 一致/成功旗标/target 槽近黑帧兜底,
>    失败抛新语义 `registry.SampleUnavailable`,由 run_baseline 显式计数 +
>    成功率门,scores.json 落 n_failed/failed_keys —— 绝不静默平均黑帧)。
>    新 `server_run/mvsplat_smoke.sh` = 2-5 样本冒烟门(env 守卫/GEO 子集/
>    --no_metrics 推理/80% 黑帧门/桥接打分,PASS 才许全量;p6 加前置门)。
>    连带修复:①**p6_mvsplat.sh 子集源 SEM_H5 → GEO_H5**(既存 bug:给位姿
>    条件 NVS 喂中和成 identity 的位姿,5 个 context 相机重合必退化;
>    evaluate_mvsplat360 再加"全 identity 即 FATAL"守卫);②evaluate_mvsplat360
>    改 import canonical `new/evaluate.py`(旧 import 顶层 legacy 版=量程外
>    角误差口径),meta 补 error/gen_seed/infer_seconds + v2.5/v2.6 口径标记,
>    `--gen_seed` 逐样本 torch.manual_seed(SVD 采样可复现),`--no_metrics`;
>    ③p6 黑帧门 `.get(...,True)`→`False`(缺旗标不再算成功)。
>    ⚠️ 如实声明:`run_mvsplat360_single` 的 ModelWrapper 调用契约按公开接口
>    写、**从未在真 repo 上运行**(GitHub 在打包沙箱被拦,无法核实)——冒烟门
>    存在的意义就是首跑裁决它;bridge 侧 6 块假产物自测全绿。
> **回归(本沙箱有 numpy/cv2/scipy/skimage,无 torch/h5py —— 后两者用 import
> 桩,被测路径全部真实执行)**:trajectory_metrics 99 断言 ✅;taxonomy 49/49 ✅;
> check_resisc45 9/9 ✅;registry 冒烟 ✅(含 v2.6 环境确定性修正:进程内清
> RSWM_MVSPLAT_EVAL_DIR);外部桥 6 块 ✅;I3D 自测 11 ✅;路由/键清单 24 项 ✅;
> **well-posed 端到端夹具 16 项 ✅**(真 SSIM/真光流/真 CSV,4 well-posed +
> 2 ill-posed 反目标构造:全样本栏 60.0° vs well-posed 栏 0.0°,掩码数学吻合;
> 混版本硬错/旧目录跳过/v2.5 口径门回归全过);py_compile 8/8、bash -n 全过。
> **待服务器首跑验证**:evaluate.py 双栏在真 H5/GPU 链路、路由 meta 在 20-33
> 作业产物上的端到端、I3D 权重 URL 与 torchscript 契约、MVSplat360 冒烟。
> 仍待办:Track A well-posed 论文侧双栏表、论文 overlap/legacy 删除(冻结中)。

> **v2.5(2026-07-26,用户拍板五项的硬门 A/B/C 落地)**
> 1. **统一分类学(拍板1)**:flood 更名 water;唯一真源 =
>    `lora_remotesense/scene_taxonomy.json`(taxonomy_version=water_v1,
>    builder/检查器/测试全部从此读);四数据集统一 beach→water、forest→
>    rural、pond/river/port/harbor(+ferry_terminal)→water、swimming_pool→
>    urban、shipping_yard→urban、resort→general;导入期互斥断言升级(组键
>    必须=现行四类、禁 flood);HD tier 配置按 SCENE_CATEGORIES 程序化生成;
>    v5 两份拷贝默认场景表与合成分支同步;metadata 新增 taxonomy_version/
>    taxonomy_file_sha256/prompt_template_version 并透传最终 H5。
> 2. **prompt pt_v3**:四类固定模板(water="a water-dominated remote-sensing
>    scene" 等),方向独立成句("The camera follows a K-segment trajectory
>    with net displacement toward the {DIR}.");灾害语义废除;
>    include_direction=False 支持 direction-stripped 归因派生。
> 3. **Track A 生产口径修复(硬门 B)**:evaluate.py 的 compute_metrics 全部
>    调用点显式 flow_size=窗口分辨率(meta 落盘,缺失回退 (128,222) 大声
>    记录)+ step=1;metrics.py compute_batch_angular_error/compute_all_
>    metrics 增口径参数(默认 None=旧行为,仅 evaluate 显式启用新口径);
>    生成期 meta.json 记录 pose_window_size 与 direction_target_source。
> 4. **方向目标 waypoints 回退**:prompt 无方向词(stripped 归因实验)时从
>    waypoints 端点量化推导(direction_target_from_waypoints,与建库量化
>    0-359 全扫逐位一致);evaluate/validate 共用。
> 5. **场景读侧统一**:evaluate/validate 场景取 attrs 优先+canonical_scene
>    别名归一(flood→water);validate by_scene、check_aid_pools SCENES、
>    check_resisc45(改读 taxonomy JSON+自测 9/9)全部 water 化。
> 6. **新 server_run/test_taxonomy_prompt.py(硬门 C)**:49 项单测全绿
>    (结构/互斥/拍板规则逐条/模板/方向独立句/stripped 零方向词/evaluate
>    一致性/量化 360° 全扫)。硬门 A 的 grep 检查:生产侧零 flood 产出
>    (仅存读侧别名与守卫死代码)。
> 7. **归因实验(拍板④)已落地**:新 `server_run/make_variant_h5.py` 派生
>    两个 variant —— shuffled_pose(poses 样本间确定性无不动点置换,图像/
>    prompt/目标不变)与 direction_stripped(poses 原样,prompt 用
>    include_direction=False 重生成、原 prompt 留档);新 PBS
>    `32_r16_shufpose_s42` / `33_r16_geostrip_s42`(单 seed、AID、主 r16,
>    sha 绑定派生 + afterok:01),编排器 TRAIN 数组已含。派生逻辑沙箱端到端
>    验证(无不动点/集合守恒/零方向词/waypoints 目标可推/poses 不变/留档)。
> 8. **outpaint 外部基线(拍板②档A)已实现**:
>    `diagnostic_baselines.image_outpainting_baseline` 从 NotImplementedError
>    桩改为真 rollout(逐帧沿内容方向平移 + 前沿新暴露条带 inpaint,已知区
>    强制保留防漂移;无模型时硬错、绝不退化成平移);
>    `external_stubs.ImageOutpaintingAdapter` 注册为 status=implemented
>    (SD-2-inpainting,RSWM_INPAINT_MODEL 可覆盖;方向由 GT waypoints 量化
>    而非读 prompt,兼容 direction-stripped 数据)。假 pipe 6/6 自测通过
>    (形状/首帧/位移语义/mask 保留/无模型硬错/四方向暴露边符号)。
> 9. **预取开关(拍板②)**:offline_prefetch.sh 新增 `BASELINE_A=1`
>    (SD-2-inpainting ~5GB)、`MVSPLAT=1`(三权重+代码+torch2.2.2 第二
>    wheelhouse,SVD 为 GATED 须先 login)、`COGVIDEOX=1`(档C,不阻塞冻结)。
> ⚠️ 仍待办:MVSplat360 桥接层与 2-5 样本冒烟;I3D-FVD 离线重算;Track A
> well-posed(net-ratio≥0.2)双栏;论文侧 overlap/legacy 表删除(冻结中)。

> **v2.4.5(2026-07-25,尺寸审计 + 最终 H5 可视化)**
> 1. **多尺寸处理审计结论**:AID/WHU-RS19(600²≥512)走大图直裁分支,每样本
>    1 张唯一源图;RESISC45/PatternNet(256²)走拼接分支,ceil(512/256)=2×2
>    恰好整除、无黑边,每样本消耗 4 张唯一 tile(禁回绕硬错);分支判定按
>    每场景探针尺寸缓存,四数据集尺寸均匀,混尺寸场景会响亮崩溃而非静默走错
>    分支(对抗核验 V1 已证)。窗口(128×222)与目标(480×832)对全部 tier
>    一致 → 跨数据集口径可比。WHU-RS19 rural 池仅 2 类(~106 张),N=200
>    (50/场景)有 2× 余量但是全 tier 容量瓶颈 —— --check-source 会先量化。
>    **2026-08-14 数据更正**:真实 WHU Parquet 为 1001 张 600² + 4 张异形图，
>    其中 726×456 与 600×500 的短边不足 512；现由统一解码器在既有裁剪前
>    等比例 resize-to-cover，并逐样本记录原始/归一化尺寸。上面的“尺寸均匀”
>    仅是 v2.4.5 当时的假设，不再代表现行实现。
> 2. **新 `server_run/visualize_benchmark.py`**:最终 H5 可视化(旧
>    visualize_multi_direction_* 只认中间产物 sliding_frames 格式,而中间
>    产物 v2.4.4 起建完即删)。纯 numpy+h5py+Pillow;样张网格(场景+轨迹
>    折线+waypoint 编号+起终点+起始窗口框 | 首/中/末帧 | 溯源注释,每 8
>    样本一张大图,inode 友好)+ summary.json(每场景/方向/段数分布、
>    净位移比统计含 <0.2 门控占比、源 class 多样性、缺字段计数)。
>    已在沙箱以合成 H5 端到端验证(样张渲染 + summary 断言全过)。
>    冒烟/建库后各跑一次,人眼确认场景-标签相符、轨迹合理,再进训练。

> **v2.4.4(2026-07-25,多数据集 benchmark 路线打通)**
> 目标钉死:代码改正 → 建成【多数据集 benchmark】(AID 主集训+评 +
> RESISC45/PatternNet/WHU-RS19 三个 test-only 泛化集)→ 用 LoRA 探针验证
> benchmark 有效性。本轮把 v2.4 的保障延伸到泛化 tier 构建入口:
> 1. 新 `server_run/pbs/_quota_preflight.sh`:lquota 配额预检共享库;
>    qsub_corrected_rerun(300GiB/20k)、qsub_resisc45 与 40_build(60GiB/
>    20k)三处 df 判决全部替换(df 只余"仅供参考"打印)。
> 2. `40_build_resisc45.pbs` tier 化:旧硬守卫写死 `!=resisc45 即 FATAL`,
>    把 patternnet/whurs19 挡在门外(与编排器 --tier 三选矛盾)→ 改为三
>    tier 白名单;stamp/split/tmp 目录/报告名/eval TAG 全部按
>    $DATASET_TIER 派生;check_resisc45 调用补 --tier。
> 3. 40 号建库后加【重复/源身份门】(--expect-zero --expect-n
>    --allow-missing-paths):泛化 tier 走 HF label 路线源路径可为空,
>    豁免只对非 AID 生效;FAIL 即作业死,神经化/评估不会碰坏数据。

> **v2.4.3(2026-07-24,用户对包实测审查的 2 P0 + 2 P1 + 2 P2 全修)**
> 1. **P0 配额预检**:qsub_corrected_rerun.sh 弃用 df(查的是整个 Lustre
>    4.3PiB,必然假放行)→ `lquota` 查真实项目配额;项目号从 WORK 路径
>    /scratch/<proj>/ 自动解析(PROJECT= 可覆盖);阈值 MIN_QUOTA_GIB=300 /
>    MIN_QUOTA_INODES=20000(env 可覆盖);解析失败硬错。⚠️ 未写死任何
>    mv44 路径 —— 既有红线"mv44 已关闭"未被用户显式解除前不进代码。
> 2. **P0 陈旧 stamp 旁路**:门/neutralize/split 门全部改 **sha256 绑定
>    stamp**(common.sh 新增 h5_sha/sha_stamp_ok/sha_stamp):H5 内容一变
>    必重跑;p1_build/smoke_build 重建前清全部下游 stamp;已用 WORK 里
>    重建会大声警告。三个入口(01_build.pbs/p1_main/p0_smoke)同构。
> 3. **P1 split 泄漏门入自动链**:新 server_run/make_precheck_split.py 在
>    建库后立即按 dataset_loader 同源 RNG(seed42/0.8)预生成确定性 split,
>    当场跑 --split 泄漏检查 —— 门 FAIL 则 01_build 非零退出,20-31 一个
>    都提交不了("第三道绿灯"从此强制);另有 --verify-against 在 job20
>    出真 split 后逐位比对防 RNG 漂移(不一致 exit 6)。
> 4. **P1 缺 source_path**:--expect-zero 默认严格(任何缺路径 FAIL);
>    label 型数据集须显式 --allow-missing-paths,且 AID 永不豁免。
> 5. **P2**:check_aid_pools PASS 文案改为"仅第一道门,三门全过才启用
>    claim";builder -n 硬限 4 的倍数(旧行为 -n 21 静默建 20,与
>    --expect-n 门打架)。

> **v2.4 增量(2026-07-23 深夜,AID 池失效根因 B 的全套修复)**
> 服务器取证坐实:①旧数据四重完整重复(同偏移 i/i+125/i+250/i+375 的
> full_scene/image/trajectory/video_frames/waypoints/segment_angles/poses/
> intrinsics 逐位相同,仅标签不同 → 实际只有 125 组唯一内容);②根因 B =
> blanchon/AID 无 label 字段(10000 JPG + 1 thumbnail.png,类别只在原始路径
> .../data/<CLASS>/<f>.jpg 里),_get_scene_filter_indices 过滤失败静默回退
> 全集 → 四池均 10001 且含非法索引 10000;③check_aid_pools 只查池大小 →
> 假 PASS。修复(全部落包):
> 1. **过滤重写**:AID 用未解码原始路径解析(cast_column(Image(decode=
>    False)) / pyarrow path 子列,绝不用 PIL .filename);只收
>    data/<CLASS>/*.jpg(排 thumbnail/非 data//非 JPG/缺路径);全部数据集
>    归一化【精确】匹配(双向子串废除);已知数据集(AID/RESISC45/
>    PatternNet/WHU-RS19)过滤失败 = SceneSelectionError 硬错,禁止静默
>    回退全量(未知数据集才允许,带大声警告)。
> 2. **池互斥**:AID beach/forest、WHU beach 双归属已去重;导入期
>    _assert_scene_groups_disjoint 自检;运行期同数据集池两两交集非零即
>    硬错。取图 = 每场景确定性排列((random_seed,场景名) 种子,类别混合)
>    + scene_id 越界硬错,【禁止 % 回绕】(大图与拼接分支都改)。
> 3. **异常穿透**:HD 加载失败不再返回 None(旧行为会静默退回合成场景/
>    eurosat 把假地物混进 canonical 数据)——一律抛错中止构建。
> 4. **源图身份持久化(§六)**:每样本写 source_dataset/source_split/
>    source_index/source_path/source_class(拼接样本另有 source_tile_
>    indices);metadata 写 builder_version/builder_script_sha256/
>    converter_script_sha256/random_seed;转换器透传到最终 H5。
> 5. **check_aid_pools.py 重写(§七)**:单池(非空/≥阈值/无重复/全为合法
>    训练图)+ 池间(两两交集 0/不全同/并集=和)+ 容量;exit 0/4;FAIL 时
>    零 claim 文本。
> 6. **check_benchmark_duplicates.py 扩展(§八)**:source_key 跨场景重复、
>    train/val 源身份泄漏(normal 模式也 exit 4)、thumbnail/非 JPG/缺
>    source_path;--expect-zero 门加严:缺 source attrs / 跨场景 /
>    thumbnail / 非 JPG / 混合缺路径 → 全部 FAIL exit 5。
> ⚠️ 旧数据不得原地修补(§九):benchmark.h5/train_geo.h5/train_semantic.h5/
> split 全部重生成;旧文件保留作证据,不得被训练脚本 resume/复用。
>
> **v2.4.1(同日,对抗核验 7 项 CONFIRMED 的修复)**
> 1. 数据集名一律取【实际加载】的 info["name"](mixed tier 下 v5 用"第一个
>    加载成功的",按配置推导会用错场景组并伪造 source_dataset 溯源);
> 2. use_hf 构建下源图登记表为空 = 全合成泄漏 → 硬错(旧行为静默跳过;
>    LOCAL_AID_DIR 残留 + 非 AID tier 时极易触发);
> 3. check_aid_pools 合法图判据按数据集类型分路(本地目录无 data/ 段,
>    旧判据必然全场景误 FAIL;HF label 路线则真空通过);
> 4. 门模式补查 source_key 复用【含同场景】(同源图两个裁剪 → 三类指纹全
>    不同,只有 source_key 能抓)+ train/val 源泄漏;
> 5. full_scene 级 train/val 泄漏也触发 exit 4(旧 H5 无 source attrs 时
>    这是唯一的源级泄漏指纹);
> 6. `--expect-n`:门模式校验样本数(空/截断 H5 曾能"零重复"空过门),
>    01_build 已传 --expect-n $N_MAIN;
> 7. _raw_image_paths 逐 chunk 取 path 子列(绝不 combine_chunks 物化
>    ~2.5GB bytes,登录节点 OOM 风险)+ 按数据集对象记忆化(建池 5 次
>    调用 → 1 次实扫)。
> 另修:建池缓存原子化提交(中途崩不留半初始化状态)、random_seed 从
> config 透传进 metadata(不再写死)。

> **D3 重建拍板(2026-07-23,用户最终决策)**
> 范围:**主线三种子、消融分阶段** —— r16 sem/geo × seed 42/43/44 +
> zero-shot/persistence/det/random 重评;r8/r128/attn_cam 先各 seed 42,
> 趋势影响主结论才补 43/44。Track A 统一 128×222 + step=1;Track B v2.3 口径。
> 21/22 立即 qdel(旧数据 125 模板缺陷,产物保留标记为调试结果,不并入);
> 24/25 继续冻结;泛化 tier(WHU-RS19/RESISC45/PatternNet)全部顺延到新基准
> +新模型(现阶段只许下载/预处理检查/1-2 样本冒烟);论文数字继续冻结。
> **固定执行顺序**:
> ```
> 重复 checker 判决(旧数据,--split 全量取证)
> → builder 已修复(v2.3 #5)
> → 20 样本冒烟集(-n 20)→ checker --expect-zero 零重复
> → 全量重建(01_build 尾部已内嵌重复门,FAIL 则 afterok 挂住下游)
> → 全量 checker → 固定 split(split_seed=42)
> → 训练 canary → sem/geo 主线各三种子(PBS 20-25)
> → 单种子消融(26-30)筛选 → 正式泛化 → 更新论文数字
> ```
> 配套改动:`check_benchmark_duplicates.py` 的 `--split` 改可选 +
> 新增 `--expect-zero` 门模式(任何重复组 exit 5);`01_build_data.pbs`
> 建库后内嵌重复门,PASS 才 stamp。

> **v2.3 增量(2026-07-23 晚,基于 v2.2 服务器实测的三项修改)**
> 实测背景:窗口分辨率+默认 step 地板仍 82.67°(step=3 每对~27px 照样超量
> 程)→ **step=1 定为正式口径**(GT 地板 35.93°/median 24.45°);尸检否定
> "低净位移比尾部"假设(r=−0.092,ratio≥0.5 仍 18.19°);sample_00016 最小
> 折叠转角 13.1° 确认近共线根因。修改:
> 1. `validate_direction_metric.py`:各配置角误差补 **step=1 正式读数**
>    (`angular_by_config_step1` 含 per_scene/top5)+ `C_paired_tests_step1`;
>    默认 step 保留仅作对照。方向表/显著性以 *_step1 为准。
> 2. `run_trajectory_eval.py` + `trajectory_metrics.py`:**近共线统一规则**
>    ——最小折叠转角 <20°(=knot 检测容差 DEGENERATE_TURN_DEG)的样本,
>    4 个 knot 依赖指标置 None、knots_source='degenerate_geometry'(所有
>    此类样本统一处理,不许单样本豁免);净方向/Fréchet/DTW/流幅度保留;
>    标定门放行该标签并公开计数(poses/uniform_fallback/None 仍零容忍)。
> 3. 偏置尸检数据链:`net_direction_from_path`/score 输出新增
>    `recovered_net_angle`(未折叠原始角),per_sample 落
>    `recovered_net_angle`+`gt_net_content_angle`;`autopsy_trackb.py` 新增
>    带符号旋偏检验(圆均值+合成向量长 R:R>0.5=系统性单向旋偏→查光流
>    偏置;R<0.3=对称失效/离群)、误差 vs 比例失配 |rec−gt|/gt 相关
>    (分段长度失真假设)、oracle 分场景分解。
> ⚠️ oracle net 25.11° 尚无修复——v2.3 的任务是把根因二选一(旋偏 vs 离群/
> 场景失效)钉死,凭数据再定门。重跑后门仍可能 FAIL net 项,属预期。
> 4. **新增 `server_run/check_benchmark_duplicates.py`(最高优先,先于一切
>    重跑)**:top-15 里出现跨场景逐位相同的 (净误差,GT ratio,恢复 ratio)
>    组(00390/00015/00140 三重等),索引差全为 125 倍数(500=4 场景×125
>    块内同偏移)→ 疑 builder 逐场景 RNG 重置产生**完整重复样本**(恢复
>    ratio 来自光流,不同影像几乎不可能逐位一致)。脚本对全部样本取
>    轨迹模板/视频内容/源场景图三类指纹,分组并交叉 train_val_split:
>    视频指纹组横跨 train/val = **保真指标泄漏(exit 4 红色警报,一切论文
>    数字冻结)**;val 内重复 = 有效 N<100;仅轨迹指纹重复 = Track B 模板
>    权重问题。判决结果决定后续所有实验的走向。
> 5. **builder 重复样本根因修复(只影响未来重建,不碰现有数据)**:
>    (a) `build_multi_direction_benchmark_v5.py:266`(两份拷贝已同步)——
>    旧 `np.random.seed(random_seed + scene_id)` 里 scene_id 是块内局部序号,
>    四场景块同偏移样本共享同一 RNG 状态 → 轨迹逐位相同;修复 = 种子吃
>    (场景名, 序号) crc32,4×125 个互异确定性流(自测无碰撞);
>    (b) `build_dataset_from_benchmark.py` `_hd_load_scene_from_hf` 取图 ——
>    旧 `valid_indices[scene_id % len]` 在四池=全集(AID 子串 bug)下同偏移
>    取同一张图;修复 = 池按场景切不相交连续区段(AID stride=2500≥125,
>    块内顺序取,source-disjoint 更强,跨段时大声警告)。
>    ⚠️ 修复后重建 = 新基准,与现有一切结果不可比;是否/何时重建重训,
>    等 check_benchmark_duplicates.py 判决 + 用户拍板。

> **v2.2 增量(2026-07-23,针对服务器全量诊断的四项修改)**
> 1. `new/metrics.py` estimate_motion_direction 加可选 `flow_size/step`
>    (不传=行为逐字节不变)。根因:真实 GT 地板在 480x832 实测 82.2° ——
>    GT 位移 ~34px/帧 × step=len//8=3 采样 ≈102px/对,超 Farnebäck 量程
>    (与 Track B v1 gt_oracle 80.67° 同一事故),**不是指标语义问题**。
> 2. `server_run/validate_direction_metric.py` 一切光流(GT 地板/各配置角
>    误差/流幅度)默认降到原生窗口分辨率;`--fullres` 复现旧口径;地板另
>    报 step=1 读数(窗口口径 step=3 每对 ~27px 仍在量程边缘);各配置块
>    加 per_scene;输出全部打 `_flow_resolution` 戳。**修复后判读顺序:
>    先看地板是否≈22.5°,再看 angular_by_config.<cfg>.flow_mag_mean 决定
>    模型角误差是"真随机"还是"根本没动"。**
> 3. `server_run/run_trajectory_eval.py` `_score_v2` 长度相等分支同步去重:
>    GT 轨迹含段边界重复(诊断34:27帧25唯一位置)且**帧内容也在同索引
>    重复**时,frames/trajectory/poses 同 mask 去重(gt_oracle 与生成共享
>    25 帧时间轴);帧不重复的合成基线(det/random)保持原样。⚠️ 此修复
>    **不解决** oracle net 25.10° 门(净方向=端点法,重复帧光流恒为 0,
>    去重前后数学等价)与 uniform_fallback:1(根因=waypoint clamp 后近
>    共线转角 <20° 容差,builder 几何边角料)。输出 JSON 新增逐样本
>    `per_sample` 行(stdout 只打聚合)。
> 4. 新增 `server_run/autopsy_trackb.py`:门失败尸检(oracle 净方向逐样本
>    ×净位移比相关/受限均值、fallback 样本转角检验、覆盖率、mean vs
>    median)。**跑法:v2.2 重跑 Track B 全量 → 尸检 → 再谈门怎么改。**

> 基线 = 7 月 10 号那版包(628KB/250 文件:旧 prompt、PBS 01–19 旧流水线、
> 角误差未修正、无 Track B、无基线框架)。
> 当前 = 832KB/315 文件(2026-07-17,Track B v2.1 + 严格标定门)。

---

## 一、这个包要干什么(执行顺序)

**目标:产出"档2 修正-prompt"的全套论文数字** —— r16 默认 × 3 训练种子的主结果、
r128 容量对照、参数效率、Track B 完整轨迹控制、诊断基线、多种子统计。
旧 AID 结果(`server_run/work/`)只读不动。

```
第 0 步(登录节点,纯 CPU,全绿才继续)
  python lora_remotesense/new/trajectory_metrics.py      # 99 断言 ✅已在服务器通过
  python lora_remotesense/new/diagnostic_baselines.py
  python server_run/check_resisc45.py --self-test        # 9/9 ✅已通过

第 1 步(GPU,一键提交;已跑完 20/23/31 的可跳过重复部分,stamp 会自动跳)
  DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh   # 先看
  bash server_run/pbs/qsub_corrected_rerun.sh            # 01建库→20-30训练+评估→31基线
  # 独立 WORK(默认 work_corrected),指向旧 AID work 会 FATAL 拒绝

第 2 步(登录节点,纯 CPU 分析)
  # 2a. 回填 Track B 的 GT 结构(旧建库产物缺 trajectory/full_scene 时;免重建)
  python server_run/augment_h5_trajectory.py --src $W/tmp_bench_main/<src>.h5 \
      --dst $W/data/train_geo.h5           # DRYRUN=1 先看
  # 2b. Track B 5 样本冒烟 —— 必须出现 TRACK B V2 CALIBRATION: PASS
  python server_run/run_trajectory_eval.py --gt-h5 $W/data/train_geo.h5 \
      --split $W/train_r16_sem_s42/train_val_split.json \
      --eval-dirs textonly=$W/eval_r16_sem_s42 --max-samples 5 --with-baselines
  # 期望:gt_oracle 净方向≈7°、00349 被 GT 门控、knots 严格全 positions、
  #       timeline_mismatch=0 → PASS(FAIL 则 exit 3,不许跑全量)
  # 2c. PASS 后跑全量 Track B(textonly + textpluecker + --with-baselines)
  # 2d. 方向校准 / 多种子统计 / 参数效率(命令见编排脚本末尾打印或 README_RUN.md §6)
  python server_run/validate_direction_metric.py ...
  python server_run/aggregate_seeds.py --seeds 42 43 44 ... --per-traj ...
  python server_run/parameter_efficiency_summary.py ...

回传:trajectory_eval.json / direction_validation.json / seed_summary.* /
      param_efficiency.* / 各 eval_*/eval_results.json → 用于填论文表格。
```

**判停红线**:第 0 步任一自测不绿、或 2b 冒烟 FAIL → 停,先修指标,不烧 GPU、
Track B 数字不进论文。

---

## 二、相对 07-10 版本的全部变化

### A. 实验设定修正(影响数据与训练,须重建/重训)

| # | 变化 | 文件 |
|---|---|---|
| A1 | **角误差符号修正**:目标 = 相机角 **+180°**(滑窗基准里内容流与相机相反;旧版直接用相机角,所有角度数字语义反了 180−x) | `new/evaluate.py` |
| A2 | **prompt 修正**:`"...moving eastward with {K} direction changes"`(K 段却说 K 次转向)→ `"...following a {K}-segment trajectory with net displacement toward the {DIR}"`;解析器兼容新旧措辞 | builder + `new/evaluate.py` |
| A3 | **GT 轨迹持久化**:最终 H5 每样本新增 `waypoints/segment_angles/num_segments/trajectory(逐帧真实位置)/full_scene(512² 场景)`——Track B 与诊断基线的 GT 来源 | builder |
| A4 | **默认 rank 128→16**(r16 保真全面 ≥ r128、参数 1/8);`T_MAIN` 改指 r16 | `common.sh` |
| A5 | **`--split_seed`**:train/val 划分与训练种子解耦(多种子必须固定 42,否则验证集泄漏、种子不可比) | `new/train.py` + `common.sh` |
| A6 | **`--gen_seed`**:扩散采样种子独立旋钮(baseline 多采样种子用) | `new/evaluate*.py` + `common.sh` |
| A7 | **`--lora_targets {all,attn_cam,attn_only}`**:LoRA 目标模块过滤(去 FFN),0 匹配 fail-loud | `animate_utils.py` + `new/train.py` + `common.sh` |

### B. 评估修正(07-10 版的静默 bug)

| # | Bug(旧版行为) | 修复 |
|---|---|---|
| B1 | 内联方向解析只认 "eastward" → 新 prompt 下 dir_E/N/W/S 分方向统计**静默全丢** | 方向词单一真源 `parse_direction_word_from_prompt` |
| B2 | 4 处传"全量视频+过滤后角度",zip 截断 → **视频与角度错位、角误差全错不报错** | 对齐传参 + 长度不等硬失败 |
| B3 | 逐样本指标被丢弃 → 逐轨迹 bootstrap 无数据 | evaluate 落 `scores.json`(单文件) |
| B4 | 训练作业不清理 epoch checkpoint(历史吃 ~160GB) | 20–30 作业双 stamp 后自动清理,只留 best |
| B5 | 无显存/推理耗时打点 → 参数效率表恒 null | train() 峰值显存 + meta `infer_seconds/fps` + eval 聚合 |

### C. 新增模块(07-10 版完全没有)

| 模块 | 干什么 |
|---|---|
| `new/trajectory_metrics.py`(**v2.1**) | Track B 完整轨迹控制 12 指标:端点净方向+**GT 命令轨迹门控**、逐段角误差、分段成功率、转弯 recall/precision/F1/时机(±3帧向量均值+NMS)、waypoint+终点误差、Fréchet+DTW、流幅度、有效性门;**全部光流在原生窗口分辨率 128×222**;真实结点(positions→poses→uniform 回退链,knots_source 标注);99 断言自测(✅服务器已通过) |
| `new/diagnostic_baselines.py` | persistence / deterministic-translation / random-direction / gt-oracle 四个诊断基线(证明 benchmark 区分四类失败);1.5× 场景守卫(传窗口帧当场景直接 ValueError) |
| `server_run/run_trajectory_eval.py`(**v2.1**) | Track B 驱动:时间轴自适配(生成去 K−1 重复边界帧,trajectory+poses 同 mask 去重;对不齐 fail-loud 计 timeline_mismatch)、**标定门 TRACK B V2 CALIBRATION: PASS/FAIL**(gt_oracle 净<15°/逐段<15°/recall>0.8/valid>95%/**knots 严格=={"positions":n}**),FAIL exit 3 |
| `server_run/augment_h5_trajectory.py` | 从 tmp_bench 源把 trajectory/full_scene **免重建回填**进现有 H5(waypoints allclose 校验、旧 AID 拒写、DRYRUN) |
| `server_run/validate_direction_metric.py` | 方向指标校准:GT 地板(预期≈22.5°)/8 方向人工平移(6px/帧,预期≈0°)/分场景流幅度/异常轨迹 |
| `server_run/aggregate_seeds.py` | 多种子统计:mean±std/median/paired diff/bootstrap CI/置换检验;**缺 seed fail-loud**;csv/json/md/tex 四输出;--per-traj 吃 scores.json |
| `server_run/parameter_efficiency_summary.py` | 参数效率表:可训参数(逐模块类)/峰值显存/训练+推理耗时/FPS/各指标;缺失记 null 不臆造 |
| `server_run/check_resisc45.py` | 第二数据集 CPU 冒烟(--check-source/h5/offline/split + --self-test 9/9) |
| `server_run/baselines/`(registry 等 5 文件) | 统一基线 adapter/registry:同格式记录(sample_id/generated_frames/route_metadata/runtime_metadata...)、同 split/分辨率/帧数/gen_seed;4 诊断已实现 + 3 外部 stub(**未跑出过数,论文不得声称外部比较**) |
| `new/train.py` 参数审计 | 逐模块类可训参数分解 + **主干意外解冻立即 RuntimeError** + param_audit.json |

### D. 作业与编排(PBS 01–19 之外全新)

- **PBS 20–31(档2 矩阵)**:r16 sem/geo × seed 42/43/44(20–25)+ r128 sem/geo s42(26–27)+ r16 attn_cam ×2(28–29)+ r8(30)+ baseline(31);train+eval 融合、全并行、每作业读自己的 split
- **`qsub_corrected_rerun.sh`**:一键编排;**独立 WORK**(指向旧 AID work 即 FATAL)、容量/inode 预检(<300GiB 或 <200k inode 拒跑)、DRYRUN、依赖显式打印
- **`40_build_resisc45.pbs` + `qsub_resisc45.sh`**:第二数据集 TEST-ONLY 泛化(不训练,AID checkpoint 直接评 RESISC45;--lora-ckpt 必填且校验存在)

### E. 安全红线(07-10 版没有的强制机制)

- 旧 AID 结果不可覆盖:新编排器 WORK 指向 `server_run/work` → FATAL;不再删任何旧 H5/stamp
- 零 `/scratch/mv44` 真实路径(仅存"勿用 mv44"告诫)
- inode/容量友好:分析只写聚合文件;epoch checkpoint 自动清理
- 一切 fail-loud:缺 seed、时间轴对不齐、主干解冻、v2 参数缺失、标定门失败 → 非零退出,绝不静默

### F. 文档(全新)

`README_RUN.md`(总入口)/ `REVIEW_GAP_REPORT.md`(9 条意见审计+bug 清单)/
`REVIEWER_STYLE_EXPERIMENT_REVIEW.md`(审稿视角 8 节)/ `README_RESISC45.md` /
`baselines/README_BASELINES.md` / `pbs/README_PBS.md`(档2 章节)

### G. 已作废与已确认(数字状态)

- **作废**(未进论文):v2 冒烟的 textonly 94.08° / textpluecker 40.95° / gt_oracle 80.67°(指标失效所致)
- **已确认**(Track A 口径,修正 prompt r16 s42):base SSIM .492/FVD 80.7/角 99.6°;r16 text-only .590/32.6(−60%)/89.3°≈随机;r16 text+Plücker .568/41.9/91.9° —— 保真迁移 + 方向不迁移 + Plücker 无边际
- **待产**:21/22/24–30 作业、Track B 全量(冒烟 PASS 后)、多种子统计、参数效率表
