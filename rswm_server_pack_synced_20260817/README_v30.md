# RSWM-Bench 运行包 v3.0

> **当前部署说明**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FA2→SDPA）；本文件中的 H200 内容属于 legacy/reference，
> sm90/FA3 支持仍保留，但本轮没有 H200 实机回归。

> 2026-07-29 · 包 `rswm_server_pack_v30.tar.gz`
> 313 文件 / 30 目录
>
> 包的 sha256 不写在这里 —— README 在包内,写进去就是循环引用(改 README 会
> 改 sha)。校验值随包分发时单独给出。
>
> **本包尚未被验收放行。** 部署前请先在你自己的环境独立复核。

---

## 0. 三句话

1. **正式矩阵的 headline 是 drop ↔ native**(真·无位姿 ↔ 真实轨迹 Plücker)。`identity`(中和位姿仍走 Plücker 通路)是**静止相机命令诊断臂**,不是 text-only,永远不进 headline。
2. **34a + 34b 两个判门都 PASS 才放行主矩阵**。34a 测文本通道能否学会方向,34b 测 Plücker 通道能否学会方向;两者共享同一批探针数据。
3. **产物身份由 `run_manifest.json` 统一裁决**,训练续训 / 评测复用 / 汇总三处校验同一份字段,**任何字段缺失一律拒绝**。

---

## 1. 正式矩阵(拍板,勿改)

| 作业 | 数据 | pose_mode | 角色 | headline |
|---|---|---|---|---|
| 20–22 | `GEO_H5` | `drop` | **真·text-only 主线**(s42/43/44) | ✅ |
| 23–25 | `GEO_H5` | `native` | **Plücker 主线**(s42/43/44) | ✅ |
| 26 / 28 / 30 | `GEO_H5` | `drop` | Rank / target 消融(跟随主线口径) | ✅ |
| 27 / 29 | `GEO_H5` | `native` | 对应 native 消融 | ✅ |
| 31 | 三份 | `drop` / `native` / `identity` | 每种口径各一份 baseline | 前两者 ✅ |
| 32 / 33 | 变体 | `native` | 归因(shufpose / geostrip) | ✅ |
| **34a** | probe | `drop` | **判门**:文本方向可学习性 | ❌(判门) |
| **34b** | probe | `native` | **判门**:Plücker 方向可学习性 | ❌(判门) |
| 35 | `GEO_H5` | `drop` | 逐段富文本臂 | ✅ |
| 36 | `SEM_H5` | **`identity`** | 静止相机命令**诊断**臂 | ❌(诊断) |

**为什么 20–22 用 `GEO_H5` 而不是 `SEM_H5`**:`drop` 理论上不读位姿,但 `SEM_H5` 的 `identity_neutralized` 元数据会污染下游的 `pose_conditioning` 路由和语义门报告口径。数据跟着口径走。

**Δ 只许减同 pose_mode 的 baseline**。共用一份 base 会把"位姿通路开/关"混进 Δ,而 headline 单变量对照的意义正在于此。

02–12、18 属**已退役的 legacy 链**(`SEM_H5` + `identity`),各自声明 `RSWM_PROFILE=legacy`,只用于复现旧数字,产物 `headline_eligible=false`。

---

## 2. 部署

```bash
# 登录节点
tar -xzf ~/rswm_server_pack_v30.tar.gz
cd rswm_server_pack

# 认包:下面五样都必须在
head -3 CHANGELOG_vs_0710.md                      # 应显示 v3.0
ls lora_remotesense/new/run_manifest.py \
   server_run/pbs/34a_learnability_drop.pbs \
   server_run/pbs/34b_learnability_native.pbs \
   server_run/test_pose_timeline.py \
   server_run/test_semantic_gate.py

# 离线自检(不需要 GPU,约 10 秒)
for t in server_run/test_*.py; do python "$t" | tail -2 | head -1; done
# 期望 8 个文件全 0 FAIL(约 697 条断言)
```

环境与依赖沿用旧版,`DEPLOY_D4.md` 有完整步骤(conda / 离线预取 / Gadi PBS 头格式)。

---

## 3. 跑一遍

```bash
export WORK=/scratch/<proj>/<user>/RSWM_work_v30      # 用全新 WORK,勿在旧 WORK 上重建
bash server_run/pbs/qsub_corrected_rerun.sh            # DRYRUN=1 先干跑
```

编排链:

```
01 建库(主库 + 共享探针库)
      ↓  ← 人工批准闸(见 §4)
   ┌──────────────┬──────────────┐
  34a(drop)     34b(native)      ← 并行,两个都 PASS 才继续
   └──────────────┴──────────────┘
      ↓
20–30 / 32 / 33 / 35 / 36 训练+评估(全并行)
      ↓
31 baseline ×3(drop / native / identity)
      ↓
19 collect → summary.md / summary.json
```

`profile` 由编排自带,**不需要你设环境变量**。正式链固定 `RSWM_PROFILE=official` → `SEM_ARM_MODE=drop`;编排器会硬断言这一点,用 legacy 口径提交主矩阵会被直接拒。

---

## 4. 三道门

### 门一:人工批准(01 之后)

01 跑完会打印一个 `approval_token`,它是 GEO H5 + 探针 H5 + 语义门报告 + overlap 审计 + oracle 标定的**联合摘要**。人工核完数据后:

```bash
cat $WORK/approval_target.json          # 看 token
echo <approval_token> > $WORK/APPROVED_01_DATA
```

**数据一旦重建,token 就变,旧批准自动失效** —— 不是"批准过一次就永远批准"。跳过这道门:`SKIP_DATA_APPROVAL=1`(产物不得作为 headline)。

核对清单:

| 文件 | 看什么 |
|---|---|
| `semgate_attempts.json` | 语义门逐样本判定(report-only 口径) |
| `overlap_audit.json` | overlap 重采审计:拒样数 / 换源图次数 / 接受分布 |
| `probe34/overlap_audit.json` | 探针的高重叠带是否真的落在 [0.50, 0.95) |
| oracle 标定产物 | 光流量程标定,否则角误差地板不可解读 |
| 抽检若干样本帧 | 确认场景语义没被轨迹带跑 |

### 门二:可学习性判门(34a + 34b)

论文 headline 是 null 结论。审稿人第一问是"是范式不迁移,还是你的配方教不会任何东西"。判门用**最简单可能的方向任务**给出正对照:

* 单段直线(K=1,无转弯)
* 四个基本方向(E/N/W/S,由 `--angle-quantize 4` 从 overlap 带反解位移 + 轮转保证均衡)
* 预注册高重叠带 `[0.50, 0.95)`
* 20 条训练集过拟合探针(不是泛化测试 —— 连训练过的 20 条都学不会,泛化无从谈起)

每个判门有**五道检查**:有效率 ≥ 70% / 四方向各 ≥ 2 条 / 逐方向角误差 ≤ 45° / 生成运动能量 ≥ 真值 30% / 总体角误差 ≤ 45°;外加**预注册 overlap 区间**的硬断言(区间越界、`enforce=false`、建库参数与判门口径分叉、artifact 缺失,四种都拦)。

结论落 `gate_result.json`(机器可读),不用去 grep 日志。

**两个都 PASS 才放行。** 只让 34a(drop)PASS 就解锁全部 native 实验,证据链不完整 —— native 主线失败时,审稿人仍可归因"pose 表示学不会"而不是迁移能力失败。

判门 FAIL 时不要急着调参:先查 pose 时间轴、prompt 注入、训练是否真在优化方向相关的项、光流估计的量程标定。

### 门三:产物身份(run_manifest)

每次训练/评估都落一份 `run_manifest.json`:

```
rswm_profile · sem_arm_mode · pose_mode · role
data_h5_sha256 · split_sha256 · checkpoint_id
prompt_schema · frame_num · gen_seed · code_sha256
headline_eligible · headline_ineligible_reason
```

三处共同校验:

| 场景 | 行为 |
|---|---|
| 训练自动续训 | 断点的 manifest 必须与本次逐字段一致,否则拒绝续训 |
| 评估断点续推 | 目录已有产物的 manifest 必须与本次一致,否则拒绝复用 |
| `evaluate.py --metrics_only` | 目录必须有 manifest,且 profile/sem_arm/pose_mode/代码指纹一致 |
| `evaluate_baseline.py --metrics_only` | 同上(两个入口共调 `assert_metrics_only_ok`) |

`role` 也进 manifest 并**参与 headline 资格判定**:`probe_gate`(34a/34b)、
`diagnostic_stationary`(36)、`variant_attribution`(32/33)、`legacy`(02–12)
四类产物永远不入 headline,与 `pose_mode=identity` 同级。

**缺 manifest = 拒绝**,不是"未知所以放行"。缺失恰恰是旧产物和被手工搬进来的产物的共同特征。

⚠️ `code_sha256` 覆盖 8 个核心文件。**改任何一个都会让已有产物失效**,这是刻意的 —— 宁可多一次重跑,也不要"代码换了但产物看起来还合格"。验收时若修改了包内文件,已有中间产物会全部要求重跑。

逃生口 `RSWM_ALLOW_MANIFEST_MISMATCH=1`(带警告,产物不得作为 headline)。

---

## 5. 读结果

```bash
python server_run/collect_results.py $WORK
# → $WORK/summary.md  +  $WORK/summary.json
```

产物分**三档**,不是两档:

| 档 | 去哪 | 谁在这里 |
|---|---|---|
| **headline** | 主表 | 20–30 / 35 / drop+native baseline —— `headline_eligible` 显式为 true |
| **diagnostic** | 🔬 诊断臂节 | 36(identity)、34a/34b(判门)、32/33(归因)、identity baseline —— **设计上就不进 headline,出现在这里是正常的** |
| **rejected** | ⛔ 已拒收节 | 缺 `run_manifest.json`、manifest 与 eval_results 口径不一致、带 `manifest_override` 痕迹、legacy 链、`SEM_ARM_MODE` 被覆盖 —— 这些是**故障**,任何结论都不得引用 |

汇总层会**读 `run_manifest.json` 并与 `eval_results.json` 交叉核对**;两者不一致即拒收(结果文件被重写过的典型特征)。

`aggregate_seeds.py` 与 `recompute_fvd_i3d.py` 这两个独立出口也各有同源的资格门 —— 论文数字不止 `collect_results` 一个出口。

Δ 表读法:

* Δ = LoRA − **同模式** baseline,是**原始差值**,不是"改善量"
* 表头带方向:SSIM/PSNR 是 ↑,LPIPS/FID/FVD/角误差是 ↓
* 每格后缀 ✅改善 / ❌变差 / ➖无变化,**按各自方向判定** —— 不要把正 Δ 一律读成变好
* baseline 自己也要自证:若 `eval_base_drop` 自报 `pose_mode` 不是 `drop`,该 baseline 不予采用,相关 Δ 整行不计算

---

## 6. 数据契约

| 项 | 值 |
|---|---|
| 帧数指纹 | canonical **17/25/33**(steps 9,段界去重,天然 4n+1);探针 **9**(K=1) |
| v2.8 前 | 18/27/36(段界含重复帧) |
| 错构建器 | 16/24/32(steps 8) |
| 窗口 | 128×222,场景 512×512,目标 480×832 |
| overlap | 主库 **[0.00, 0.10)**;探针 **[0.50, 0.95)** —— 都是硬约束,建库即断言 |
| pose | `plucker_from_trajectory`(真实逐帧轨迹,非 waypoint 插值) |
| prompt_schema | `net`(净方向句)/ `rich_persegment`(净方向+逐段)/ `stripped`(无方向词) |

**非 4n+1 与批内混长度一律硬错**,不再静默截尾;帧数截断(`MAX_FRAMES` 降低)也硬错 —— 截断会让视频只走完轨迹的前半程,而 prompt 净方向 / waypoints 真值 / 语义门判定仍是全程口径,症状完全静默。逃生口 `RSWM_ALLOW_FRAME_TRUNCATION=1` / `RSWM_ALLOW_NON_4N1=1`,产物不得用于任何 headline 数字。

方向真值以 **waypoints 为数据级真源**,prompt 只在没有 waypoints 的旧数据上回退。原因:朴素解析器按固定序 east→north→west→south 取第一个命中词,rich prompt 含多个方向词时会返回一个既非净方向也非首段的任意词。net 型 prompt 与 waypoints 不一致会硬错(400 条合成轨迹验证过二者本就逐位一致)。

---

## 7. 环境开关

**正常跑不需要设任何一个。** 逐条都是为特定排障准备的,用了就不该再当 headline。

| 变量 | 默认 | 作用 |
|---|---|---|
| `RSWM_PROFILE` | `official` | `official`→drop 主线;`legacy`→identity(复现旧数字) |
| `SEM_ARM_MODE` | 随 profile | 覆盖语义臂口径,会大声警告并置 `headline_eligible=0` |
| `POSE_MODE` | 从 H5 推断 | 单作业 pose 口径;36 号显式 `identity` |
| `RSWM_RUN_ROLE` | 各作业显式声明 | 产物角色;判门/诊断/归因 三类进诊断节,legacy 进拒收节 |
| `SKIP_DATA_APPROVAL` | `0` | 跳过 01 后的人工批准闸 |
| `RSWM_ALLOW_MANIFEST_MISMATCH` | `0` | 强行复用不同源产物(**会在产物里永久留痕并自动置 headline=false**) |
| `RSWM_SKIP_HEADLINE_GATE` | `0` | 跳过 aggregate_seeds / FVD 重算的资格门 |
| `RSWM_ALLOW_FRAME_TRUNCATION` | `0` | 允许 `MAX_FRAMES` 截断样本 |
| `RSWM_ALLOW_NON_4N1` | `0` | 允许建出非 4n+1 帧数 |
| `RSWM_ENFORCE_OVERLAP` | `1` | 关掉 overlap 硬约束(仅独立 profile) |
| `RSWM_TRAJ_MAX_TRIES` | `12` | 同一源图内重采轨迹次数 |
| `RSWM_SRC_MAX_TRIES` | `6` | 换源图次数(与语义门预算解耦) |
| `PROBE_N` | `20` | 探针样本数 |
| `PROBE34_OV_LO` / `_HI` | `0.50` / `0.95` | 探针预注册 overlap 带 |
| `PROBE34_MAX_ANGLE` | `45` | 判门角误差阈值 |
| `NO_AUTO_RESUME` | `0` | 关掉墙钟超时后的自动续训 |
| `KEEP_TMP_BENCH` | `0` | 保留 `tmp_bench_main` 中间目录 |

---

## 8. 出问题时先看哪里

| 症状 | 大概率原因 |
|---|---|
| 训练启动即 `pose_mode` 冲突 | 数据与口径不匹配。`_infer_pose_mode` 会从 H5 推断,显式设了 `POSE_MODE` 就以你为准 |
| 续训被拒 | 断点与本次不同源(profile / pose_mode / 数据 / 代码变了)。换输出目录或删旧断点 |
| 评估拒绝复用 | 同上。删掉 `meta/` 与 `generated/` 重跑 |
| `--metrics_only` 拒绝执行 | 当前环境口径与产物 manifest 不符。按提示设成产物自己的口径 |
| 建库报 `[overlap]` 耗尽 | 看报错里"偏低/偏高"计数 —— 偏高是窗口相对场景过大,偏低是位移下限与 overlap 下界互斥 |
| 建库报 `[semgate]` 耗尽 | 这个是真的语义门问题,先跑 `check_semantic_consistency.py --report-only` 校准 |
| 主表某行显示"缺" | 该产物被拒收了,去"⛔ 已拒收"节看理由 |
| 判门 FAIL | 读 `gate_result.json` 的 `failures` 数组,五道门分别说明了失败在哪 |

---

## 9. 已知限制

* `code_sha256` 对注释改动也敏感 —— 改任何核心文件都会作废已有中间产物。
* 语义门规则已冻结为 `frac >= 0.2777777777777778`(9 帧,2/9 与 3/9 分界中点),0 个坏样本放行 / 3 个有效样本被拒(Port / BareLand / Farmland),已作为系统性欠采样偏置记录在 datasheet。**禁止第五轮特征迭代。**
* 探针的四方向由位移反解保证,但**样本数少时四方向不会精确均分**(20 条实测 E5/N5/W4/S6),判门只要求各 ≥ 2。
* 内参目前逐帧恒定。训练侧按 linspace 重采样、推理侧复制首帧,今天逐位相同;一旦内参变成逐帧变化会硬错提醒,届时需先统一两侧口径。
* 本包只在 Linux 侧验证过 `bash -n` 与全部离线自检;Windows 上没有可用 bash。

---

## 10. 版本

v3.0 相对 v2.9.4(全部经**行为级故障注入**验证,不是字符串断言):

1. **评测器可被直接作弊 → 不可作弊契约**。外部审查动态证明四种手段(少交帧 / 少交像素 /
   缺样本 / 容忍 20% 失败)都能拿 `PSNR=inf`。新增 `submission_contract.py` 作单一权威,
   7 种作弊全部被拒,合法完整提交照常通过。
2. **发现更严重的一条:PSNR 恒为 inf 且自动发生**。首帧是 i2v 输入、`gen[0]==gt[0]`
   是结构性必然,而 `compute_video_psnr` 逐帧取平均 → `mean([inf,...])=inf`。
   实测:后 16 帧全噪声、只有首帧对,整段 PSNR = inf;剔除首帧后 = 7.77。
   条件帧剔除现在覆盖**全部**保真指标(第一版只覆盖 FID/FVD,漏了最严重的那半)。
3. **oracle 门槛 15° 恒 FAIL**。实测仪器地板 ~25°,一道恒 FAIL 的门等于没有门,
   还会诱人去调数据而不是承认仪器精度有限。改 30°(`RSWM_ORACLE_ANGLE_GATE` 可配)。
4. **光流默认 step 落在混叠灾难区**。实测平滑纹理 ≤16px/帧对误差 ≤5°,
   **24px 处 128.77°、带符号 −113°(报的是近乎反向)**;旧默认 `T//8` 给出
   T=17→18px、T=33→36px。默认改 1,三处调用点补显式传参。
5. **MDE 口径更正**。`1.96√2·s/√n` 是 50% 功效检出阈,不是 80% 功效 MDE(差 1.43 倍)。
   用真实 σ≈54° 标定后:N=100 的 MDE(ρ=0)= 21.4°,**15° 的臂间差在 N=100 上不可分辨** ——
   这是把 headline 臂 N 提到 400 的定量依据。
6. **FID 不再只取中间帧**(单帧 FID 是 N 个点,估不出协方差);分布型指标样本量下限 32
   (旧版 `n>=2` 的方向×场景交叉块也算 FID/FVD,那是纯噪声)。
7. **阶梯解析地板按段数分**。`aq=4` 只保证逐**段**轴向,而 prompt 量化的是**净位移**,
   K≥2 时残差回到 ~20°;原写法给 0°,低估 20°。11 档现与实测吻合到 2.7° 内。
8. 发布要件补齐:LICENSE / NOTICE / DATASHEET / CITATION / requirements-lock /
   CPU-only CI / REPRODUCIBILITY / SUBMISSION 协议 / 官方三划分 + 脱敏公开版
   (`image`+`intrinsics` only,poses 单独成 Track B 输入文件)/ 11 档难度阶梯 /
   仪器地板标定 / outpaint 平凡基线(预注册读数规则)/ 骨干可插拔接口。

### v3.0 尚未验证的部分(部署前请自行确认)

- 阶梯的 fail-closed 预检、CI 在纯 CPU 无网环境的实跑、`score_submission` 拒绝规则的
  故障注入 —— 这三项只做了语法与静态核对,没有端到端跑过。
- `baselines/baseline_status.json` 里六条外部 baseline 仍全部 `tested=false`。
- 语义门仍只有 20 样本单人单遍标注,无标注者一致性数据。
- 三赛道拆分(Full-path / Coarse-text / Stationary 诊断)只做了**数据侧**分离
  (脱敏公开版已分文件),实验矩阵语义未改 —— 那是论文层决定。

---

v2.9.4 相对 v2.9.3 的修复(来自 5 视角对抗审查的 35 条 confirmed,去重后 6 类):

1. **逃生口 `RSWM_ALLOW_MANIFEST_MISMATCH` 会抹掉证据** —— 放行之后调用方紧接着
   `write_manifest` 用当前身份把旧 manifest 整个覆写,用过之后磁盘上一点痕迹都没有,
   注释里承诺的"产物不得作为 headline"没有任何机制执行。现在会在 manifest 里写
   `manifest_override` 并**自动把 `headline_eligible` 置 false**,汇总层见到即拒收。
2. **汇总只有两档,把诊断臂当成了故障** —— 36(identity)和 34a/34b(判门)被
   fail-closed 过滤器和"来路不明的旧产物"一起丢进「已拒收」,读起来像出了错。
   现在三档:headline / diagnostic / rejected。
3. **Δ 表的 baseline 侧完全不过门** —— 主表刚拒收的 baseline,Δ 表照用还打 ✅。
   baseline 是每一个差值的分母。现在 baseline 与臂都过 `classify()`,
   自报 `pose_mode` 为 `None` 也不再放行(删掉 `meta/` 就能让旧门闭嘴)。
4. **汇总层从不读 `run_manifest.json`** —— v2.9.2 建了一整套 manifest 机制,
   而汇总只看 `eval_results.json` 的 overall,那恰恰是最容易被 `--metrics_only`
   之类操作重写的地方。现在两者交叉核对,不一致即拒收。
5. **判门两处 fail-open** —— 某方向拿不到角误差时被静默豁免("没有数字"被当成
   "通过");H5 缺逐样本 `overlap_ratio` 时自证检查直接跳过,而"从产物自证"正是
   这条检查存在的理由。两处都改成判 FAIL。
6. **manifest 有三个字段恒空** —— `train.py` 根本没有 `--split_file` 和 `--gen_seed`,
   `getattr` 静默取到 `None`,两个身份字段一直在空转;而真正决定训练身份的
   `--split_seed` 反倒没进去。现已改用真实属性,并把 `split_seed`/`lora_rank`/
   `lora_alpha` 纳入身份字段。
7. 其余:`aggregate_seeds.py` / `recompute_fvd_i3d.py` 补资格门;
   13–16 / p3 / p4 / 03–12 等 17 个入口补 `RSWM_RUN_ROLE` 声明。

---

v2.9.3 相对 v2.9.2 的修复(均由对抗审查发现,且都是 v2.9.2 自己引入的):

1. **编排器用了未定义的 `$PY`** —— 它不 source `common.sh`,`set -u` 下在提交完 01
   之后立刻死,主矩阵一个作业都提交不出去。而 `DRYRUN=1` 与 `SKIP_DATA_APPROVAL=1`
   都绕过这段,所以干跑全绿、唯一能跑通的真提交路径恰好是把批准闸关掉那条。
2. **`evaluate_baseline.py --metrics_only` 没有身份门** —— v2.9.2 只给 `evaluate.py`
   加了,baseline 这边的门写在 `run_baseline_inference` 内部而 `--metrics_only`
   恰好跳过整个函数。baseline 是所有 Δ 的分母,一次重算就能把 legacy/identity 的旧
   baseline 标成 official + headline,且不留痕迹。现已抽成共用函数,两个入口共调。
3. **`RSWM_RUN_ROLE` 被 export 泄漏** —— `train_lora` 之后的 `eval_lora` 会继承
   `role=train`;更关键的是 34a/34b 从没设过 role,默认落在 `train`/`eval` 上,
   于是 `probe_gate` 这条“判门不入 headline”的规则成了死代码,判门产物会被标成合格
   headline。现改 per-command 传递,22 个作业各自显式声明 role。
4. **`approval_token` 算得太早** —— 它绑定的 `semgate_attempts.json` /
   `overlap_audit.json` 当时还没从 `tmp_bench` 抢救出来,`sha_of()` 全返回 `None`,
   token 根本没绑住语义门报告和 overlap 审计。现已移到抢救之后、清理之前。
5. `--metrics_only` 的比对字段从 2 个扩到 4 个(加 `sem_arm_mode` 与 `code_sha256`)。
