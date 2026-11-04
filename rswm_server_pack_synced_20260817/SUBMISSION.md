# RSWM-Bench 提交协议

面向使用 RSWM-Bench 隐藏 test 集的参赛者与组织者。**本文件只写代码里真实存在
的行为**;每条规则都能指到 `server_run/make_official_splits.py` 或
`server_run/score_submission.py` 的对应实现,指不到的一律不写。

配套文件:`DATASHEET.md`(数据集本身)、`REPRODUCIBILITY.md`(复现与地板)、
`README_RUN.md`(内部实验编排)。

---

## 0. 三句话版本

1. **公开 train + val,隐藏 test。** 划分是 **source-disjoint** 的:同一张源图
   裁出的样本绝不跨划分,所以 test 上的保真指标测的是泛化,不是记忆。
2. **交 `generated/<sample_id>.npy`**,每条 `(T, 480, 832, 3)` uint8,`T` 逐样本
   给定。样本集合、帧数、分辨率任一不符 → **整份拒收,不出分**。
3. **报分必须连着三条地板一起报**(量化 22.5° / 随机 90° / 仪器实测 ≈25°),
   否则 30° 的角误差看起来像失败,其实已经贴着仪器噪声。

---

## 1. 官方划分

由 `server_run/make_official_splits.py` 生成,产物 `official_splits.json`。

| 划分 | 默认占比 | 标签 | 用途 |
|---|---|---|---|
| `train` | 0.70 | 全公开 | 训练 |
| `val` | 0.10 | 全公开 | 调参、早停、自报数字 |
| `test` | 0.20 | **隐藏**(只发输入) | 官方排行榜 |

### 1.1 source-disjoint 是怎么保证的

同一张源遥感影像可能被裁成多个样本;它们的视觉内容高度相关,分居 train/test
就等于把答案抄进了训练集。划分器先做**源身份并查集**,再把整组投进同一划分。
源身份用四路信号,任一相同即判同源:

| 信号 | 来源 | 备注 |
|---|---|---|
| `(source_dataset, source_split, source_index)` | HF 建库 | 与 `check_benchmark_duplicates.py` 的 `source_key` 同口径 |
| `source_path` | 本地 AID 目录建库 | |
| `(source_scene_category, source_scene_id)` | 建库登记表键 | `--no-hf` 合成库里唯一可用 |
| `full_scene` 字节 SHA1 | 像素内容 | attrs 可以缺、可以被改,像素不会说谎 |

一个信号都没有的样本 → 划分器 **fail-closed 退出**(`exit 2`),不会悄悄假设
它们互不同源。

> 读数提示:canonical 主库过 `check_benchmark_duplicates --expect-zero` 之后,
> **每张源图恰好一个样本** → 组数 == 样本数,分组机制此时是**守卫**而非约束。
> `official_splits.json` 的 `self_check.checks.source_disjoint.max_group_size`
> 会如实告诉你实际情况。

### 1.2 确定性

划分只依赖 `(H5 内容, seed, ratios, 分层开关)`。同 seed 重跑,
`params` / `splits` / `groups` 三块**逐位相同**,`splits_sha256` 也相同
(该 SHA 覆盖这三块,不含时间戳与路径 —— 换机器、换目录重算必须仍然一致,
否则它就没法用来证明"我们说的 test 和你手上的 test 是同一个")。

```bash
python3 server_run/make_official_splits.py --gt-h5 $WORK/data/train_geo.h5 \
    -o "$TMPDIR/check.json" --verify-against "$WORK/official_splits.json"
# 不一致 → exit 6
```

### 1.3 自校验(生成时自动跑,写进 `self_check`)

- 三划分是全集的一个划分(不重不漏);
- **从写出的划分反查**每个源图组是否只落在一个划分里(不信任分配过程本身);
- 类别分布卡方拟合优度(H0: 各划分的类别构成 = 全库构成),`p < 0.01` 判 FAIL,
  另附逐类占比表与期望频数(期望 <5 的格子会被点名 —— 那时卡方近似本来就不成立);
- 帧数契约 `T = 9 + (K-1)*8` 且 `T ≡ 1 (mod 4)`;
- 帧数/段数/overlap 的逐划分分布(不判 pass/fail,用来一眼看出
  "test 恰好全是 K=4 的长样本"这类隐性偏斜)。

任一 FAIL → `exit 5`,且**拒绝产出公开 test**。

---

## 2. 两个 Track

与代码里的 `pose_mode` 三态严格对应(`RSWM_POSE_MODE`):

| Track | pose_mode | 模型拿到什么 | 什么是"被命令的" |
|---|---|---|---|
| **A** | `drop` | 首帧 + prompt | **只有**净方向词与段数 |
| **B** | `native` | 首帧 + prompt + Plücker 位姿 + 内参 | 完整逐帧轨迹 |

> `pose_mode=identity`(位姿中和成 identity 但仍走 Plücker 通路)是**静止相机
> 诊断臂**,不是 text-only,也不是一个提交 Track —— 它在内部实验里永不进
> headline(`run_manifest.py` 的 `_NEVER_HEADLINE_ROLES` / `headline_eligible`)。

**为什么位姿必须分文件发**:位姿就是完整轨迹。Track A 的任务定义是"只给方向
词",一旦 Track A 参赛者拿到位姿,就能绕过 22.5° 的量化地板 —— 那条地板是
Track A 分数的下端参照,绕过去之后两个 Track 的角误差不再可比。所以位姿放在
单独的 `rswm_test_trackB_poses.h5` 里,Track A 参赛者**物理上拿不到**。

---

## 3. 你会拿到什么

`make_official_splits.py --redact-test` 产出三个文件:

```
test_public/
  rswm_test_public.h5          # 输入(Track A/B 共用)
  rswm_test_trackB_poses.h5    # Track B 附加:逐样本 poses (T,4,4)
  rswm_test_manifest.json      # 提交合同 + 逐样本 T/K/scene/prompt + 各文件 SHA256
```

`rswm_test_public.h5` 的结构:

```
metadata/  .attrs: schema, split=test, total_instances, target_size,
                   pose_window_size, prompt_schema, angle_quantize,
                   min/max_overlap_ratio, steps_per_segment, builder_version,
                   taxonomy_version, source_h5_sha256, official_splits_sha256
data/<sample_id>/
    image        (480, 832, 3) float32,取值域 [0,255]  ← 首帧,唯一给定的画面
    intrinsics   (T, 4) float32                        ← 全库同一常量(dummy)
  .attrs: prompt, num_frames, num_segments, source_scene_category
```

**被移除的东西**(逐条都有理由,不是保守起见随手删的):

| 移除 | 为什么 |
|---|---|
| `video_frames` | 外观真值。留着 = 直接复制 GT |
| `full_scene` | 完整 512×512 场景。有它 + 窗口尺寸就能裁出整段 GT 视频(内部 `deterministic_translation` 基线就是这么合成的) |
| `waypoints` / `segment_angles` / `trajectory` | 折线命令真值(scene 像素坐标)。Track A 拿到就绕过量化地板 |
| `poses` | 移入 Track B 附件(见 §2) |
| `source_dataset` / `source_split` / `source_index` / `source_path` / `source_class` / `source_tile_indices` / `source_scene_id` | 源图身份。AID/RESISC45 是公开语料,拿到它就能下载原图、按公开的窗口尺寸暴力搜出首帧位置,直接裁出 GT |

去标签之后,写出的文件会被**重新打开逐组逐 attr 扫一遍禁忌名单**;扫出任何一项
→ `exit 5`。发布物的机密性不靠"写的时候记得"。

---

## 4. 提交包

### 4.1 目录结构

```
my_submission/
  submission.json
  generated/
    sample_00007.npy
    sample_00013.npy
    ...            # test 的每一个 sample_id 各一个,不多不少
```

打包成目录、`.tar`、`.tar.gz`、`.tgz` 都行。多包一层
(`my_submission/my_submission/generated/`)也认 —— 评分器在**定位** `generated/`
上宽容一层,但**内容**一步不让。

⚠️ tar 里出现 `../`、绝对路径、软链/设备文件 → 直接 `exit 2` 拒绝解包。

### 4.2 `submission.json`

```json
{
  "team": "your-team-name",
  "method": "简短方法名,例:WAN-I2V + LoRA r16 + pose adapter",
  "track": "A",
  "contact": "you@example.org",
  "used_external_data": false,
  "external_source_ids": [],
  "notes": "任何你希望组织者知道的口径说明"
}
```

`team` / `method` / `track` 必填,`track` 只能是 `"A"` 或 `"B"`。
`external_source_ids` 见 §7。

### 4.3 数组合同(硬性)

| 项 | 要求 |
|---|---|
| 路径 | `generated/<sample_id>.npy`,`sample_id` 与 manifest 的键**完全一致** |
| 形状 | `(T, 480, 832, 3)` —— `T` 取 `manifest.samples[<id>].num_frames`,逐样本不同 |
| dtype | `uint8`(浮点也收,但取值域必须是 `[0,255]`;`[0,1]` 归一化会被当错误拒收) |
| 通道序 | RGB |
| 帧数契约 | `T = 9 + (K-1)*8`,`T ≡ 1 (mod 4)`;K=1→9, K=2→17, K=3→25, K=4→33 |
| 数值 | 不得含 NaN/Inf |
| 集合 | 与 test 集合**恰好相等**,不接受部分提交 |

> **不接受部分提交**是有意的:允许交一部分,就等于允许挑样本 —— 只交容易的
> 那些也能得一个漂亮均值。benchmark 的分数只有在全集上才有意义。

### 4.4 拒收码表

评分器把每条拒收理由都写成机器可读的 `code`(在 `score_report.json` 的
`errors[]` 里),按 code 自查即可:

| code | 含义 |
|---|---|
| `NO_SUBMISSION_JSON` / `BAD_SUBMISSION_JSON` | 缺 `submission.json` 或解析失败 |
| `SUBMISSION_META_INCOMPLETE` / `BAD_TRACK` | 必填字段缺失 / track 不是 A 或 B |
| `NO_GENERATED_DIR` | 找不到 `generated/` |
| `MISSING_SAMPLES` | 少交了 test 样本 |
| `EXTRA_SAMPLES` | 交了 test 之外的 id(train/val 的 id?改名?) |
| `UNREADABLE_NPY` | npy 读不出来 |
| `RESOLUTION_MISMATCH` | 形状不是 `(T,480,832,3)` |
| `FRAME_COUNT_MISMATCH` | `T` 与该样本的 `num_frames` 不等 |
| `BAD_DTYPE` | 不是数值类型 |
| `SUSPECT_VALUE_RANGE` | 取值域不像 `[0,255]`(最常见:存成了 `[0,1]` 浮点) |
| `NON_FINITE_VALUES` | 含 NaN/Inf |
| `EXTERNAL_SOURCE_OVERLAP` | 声明的外部语料与 test 源图重合(见 §7) |

告警(不拒收,但会写进报告):`UNEXPECTED_FILES`、`DUPLICATE_VIDEOS`
(多个样本交了同一段视频)、`EXTERNAL_DATA_UNDECLARED`、`FID_UNAVAILABLE` /
`FVD_UNAVAILABLE`(评分机离线取不到预训练权重)、`NON_FINITE_METRIC`
(如逐帧完全一致导致 PSNR=inf,报告里写成 `null` 以保证 JSON 合法)。

### 4.5 交之前自查

不需要真值,只用公开 manifest 就能把 §4.3 的合同查一遍:

```python
import json, numpy as np
from pathlib import Path

m = json.load(open("test_public/rswm_test_manifest.json"))
H, W = m["target_size"]
gen = Path("my_submission/generated")
have, want, ok = {p.stem for p in gen.glob("*.npy")}, set(m["samples"]), True
for k in sorted(want - have):
    print("缺样本:", k); ok = False
for k in sorted(have - want):
    print("多余文件:", k); ok = False
for k in sorted(want & have):
    a = np.load(gen / f"{k}.npy", mmap_mode="r")
    want_shape = (m["samples"][k]["num_frames"], H, W, 3)
    if tuple(a.shape) != want_shape:
        print(f"{k}: shape={tuple(a.shape)} 应为 {want_shape}"); ok = False
    if a.dtype != np.uint8:
        print(f"{k}: dtype={a.dtype} 建议 uint8"); ok = False
print("SELF-CHECK:", "PASS" if ok else "FAIL")
```

---

## 5. 评分口径

`server_run/score_submission.py`。两轨指标**不在评分器里重写**,分别复用
`lora_remotesense/new/metrics.py` 与 `server_run/run_trajectory_eval.py` 的函数
——两套口径各写一遍迟早漂移,分数就不可比了。

### 5.1 Track A(保真 + 净方向)

| 指标 | 方向 | 口径 |
|---|---|---|
| `SSIM` / `PSNR` | ↑ | 逐帧后取平均(`metrics.compute_video_ssim/psnr`) |
| `LPIPS` | ↓ | 需要 `lpips` 包;取不到记 `null` 并计数,不臆造 |
| `FID` | ↓ | 每条视频取中间帧,InceptionV3 特征 |
| `FVD` | ↓ | **r3d_18 代理口径**,只在本 benchmark 内部可比,**不可与文献 I3D-FVD 并列**;文献可比口径见 `server_run/recompute_fvd_i3d.py` |
| `Angular_Error_Mean/Median/Std` | ↓ | 光流主方向 vs 目标角 |
| `Angular_Error_Mean_wellposed` | ↓ | **正式读数以这一栏为准** |

**角误差口径(必须随数字一起报,否则不可比)**:
`flow_size = metadata['pose_window_size']`(canonical `(128,222)`)、`step=1`。
在 480×832 生成分辨率上直接跑 Farnebäck 是已知的量程事故 —— GT 会读出 82°
的垃圾读数(`REPRODUCIBILITY.md` §5)。

**目标角**:`waypoints` 端点净位移 → 四基向量化 → `+180°`(内容角)。
prompt 里的方向词说的是**相机/窗口**的移动方向,而光流量的是**画面内容**的
运动方向,滑窗向东则内容向西 —— 两者恰好相反。prompt 只在没有 waypoints 时
回退;prompt 与 waypoints 矛盾 = 真值不自洽,评分器**硬失败**。

**well-posed 双栏**:GT 命令轨迹近闭合时,"净方向"这个问题本身无定义。门限
= `evaluate.WELL_POSED_MIN_NET_RATIO`(0.2),与 Track B 的净方向门控**同一真源
同一阈值**。两栏并报、各印 n。

### 5.2 Track B(轨迹控制)

复用 `run_trajectory_eval._score_v2`:窗口分辨率光流 + 真实结点
(`trajectory` → `poses` → 均匀 linspace 的回退链,`knots_source` 如实标注)
+ 时间轴自适配。聚合键 = `run_trajectory_eval.TRACKB_KEYS`:

`net_direction_error`(↓)、`segment_angular_mean`(↓)、
`segment_success_rate_30/45`(↑)、`turn_recall/precision/f1`(↑)、
`turn_timing_error`(↓)、`waypoint_error` / `waypoint_endpoint_error`(↓)、
`frechet` / `dtw`(↓)、`flow_magnitude`(匹配量)、`valid_pct`(↑)。

**Track A 提交同样会跑 Track B 指标,但整块标 `commanded=False`** —— 它没拿到
折线,逐段/转弯/waypoint/形状只作诊断,**不能算它的失败**。只有净方向与段数
对 Track A 是达标量。

近共线样本(相邻两段折叠夹角 < 20° = knot 检测容差)的 knot 依赖指标统一置
`null` 不进均值,`knots_source` 标 `degenerate_geometry` 并公开计数 ——
"该样本对这个问题无定义",不是回退。

### 5.3 仪器地板(报告里必须一起读)

`--floor measured`(默认)会把**真值视频**喂进同一条管线当场实测:

| 参照线 | 值 | 来源 |
|---|---|---|
| 四方向 prompt 的量化地板 | **22.5°** | 解析值;**仅当 `angle_quantize=0`(连续角)时存在**。判门库 `angle_quantize=4` 的轨迹角本身就是四基向,该项 = 0,两库地板**不可混用** |
| 随机方向的期望角误差 | **90°** | 可用量程的上端 |
| 光流估计器在 GT 上的实测地板 | 当场实测(既往 `step=1` 口径 ≈25°:median 24.45° / mean 35.93°) | `instrument_floor_report.measure_gt_floor`,与论文地板同一函数 |
| Track B 标定门 | gt_oracle 必须近满分 | `run_trajectory_eval._calibration_gate` |

报告里同时给 `vs_quantized_target` 与 `vs_continuous_target` 两个地板,两者之差
就是量化项的实际贡献 —— 比口头比较"22.5° 和 25° 谁大"可靠得多。

**Track B 标定门 FAIL = 真值视频自己都拿不到分 = 指标失效而非模型失效。**
这种情况下任何提交分数都不该被解读(`--require-floor-gate` 会让评分器 `exit 3`)。

---

## 6. 组织者怎么跑

```bash
# ① 生成官方划分 + 公开 test(建库之后、放榜之前,只跑一次)
python3 server_run/make_official_splits.py \
    --gt-h5 $WORK/data/train_geo.h5 \
    -o $WORK/official_splits.json \
    --redact-test --public-dir $WORK/release/test_public

# ② 评一份提交
python3 server_run/score_submission.py \
    --gt-h5 $WORK/data/train_geo.h5 \
    --splits $WORK/official_splits.json \
    --submission /path/to/team_x.tar.gz \
    -o $WORK/score_team_x.json --device cpu
```

退出码:`0` 通过并出分 / `2` 用法或环境错误 / `3` 仪器地板门 FAIL
(`--require-floor-gate`)/ `4` 提交不合规(已拒收)。

评分器会核对 `--gt-h5` 的 SHA256 与 `official_splits.json` 记录的是否一致;
不一致直接 `exit 2` —— 换了数据集,"test" 指的就不是同一批样本。

两个脚本都是**纯 CPU**。PBS(Gadi/NCI 扁平格式)作业头:

```bash
#!/bin/bash
#PBS -P rx40
#PBS -q normal
#PBS -l ncpus=12,mem=128GB,jobfs=100GB,walltime=12:00:00
#PBS -l storage=scratch/qk73+scratch/rx40
#PBS -l wd
#PBS -j oe
_ENVF="${PBS_O_WORKDIR:-$PWD}/server_run/pbs/_pbs_env.sh"
[ -f "$_ENVF" ] || { echo "FATAL: 找不到 $_ENVF"; exit 1; }
source "$_ENVF"
export RSWM_PROFILE=official
export RSWM_RUN_ROLE=probe_gate     # 见下方说明
```

**耗时**:评分是逐样本流式的(GT 视频 float32 单条 ≈158 MB,绝不整批读进内存)。
实测约 **8 s/样本**(SSIM 33 帧 480×832 + 两轨光流),`--floor measured` 再按
`--floor-max-samples`(默认 100)加约 14 s/样本 → 800 条 test ≈ 2–3 小时。
walltime 给 12 小时是留余量。

**role 说明**:`submission_scoring` 目前不在 `run_manifest._NEVER_HEADLINE_ROLES`
里,所以暂时用 `probe_gate`(保证永不入 headline)。正解是把
`submission_scoring` 同时加进 `run_manifest.py` 的 `_NEVER_HEADLINE_ROLES` 与
`common.sh` 的 `_RSWM_NEVER_HEADLINE_ROLES`(两处必须同改,
`test_pose_timeline.py` 有用例钉住二者一致)—— 提交分数是外部产物,任何情况下
都不该出现在论文主表里。评分报告里内嵌了 `run_manifest` 块可供追溯。

---

## 7. 允许与禁止

### 允许

- 用官方 `train` + `val` 的**全部**内容(图像、视频、轨迹、位姿、prompt)训练与调参。
- 用任何公开预训练权重(骨干、VAE、文本编码器…),在 `submission.json` 里写清楚。
- 用外部遥感语料预训练/微调 —— **但必须声明**(见下)。
- 对同一模型交 Track A 与 Track B 两份提交(分别评,分别上榜)。

### 禁止

1. **不得在 test 源图上训练。** 注意是**源图**,不是样本 —— 从同一张 AID 影像
   裁出的任何窗口、任何轨迹都算。
2. **不得用任何方式获取 test 的真值帧、轨迹、位姿(Track A)或源图身份。**
3. **不得对不同样本交同一段视频**冒充多样性(会被 `DUPLICATE_VIDEOS` 记录;
   Track B 的 `valid` 与净方向也会如实反映)。
4. **不得只交一部分样本**(见 §4.3)。

### 外部数据怎么声明与核验

```json
{"used_external_data": true,
 "external_source_ids": ["aid/train/1234", "/data/AID/Farmland/farmland_12.jpg",
                         "farmland_12.jpg"]}
```

组织者侧 `check_external_sources()` 会把它与 test 的源身份
(`dataset/split/index`、完整路径、文件名三种形式)求交集,命中即
`EXTERNAL_SOURCE_OVERLAP` 拒收。

**为什么协议是"你声明 + 我核验",而不是直接公布 test 源图黑名单**:
AID/RESISC45 是公开语料。一旦公布 test 用了哪些源图,配上公开的窗口尺寸就能
反推出 GT 窗口(Track B 更是连轨迹都是给定的)—— 黑名单本身就是一份作弊说明书。
组织者手里有真值,这道核验只有组织者做得了。

`used_external_data: true` 却不给 `external_source_ids` → 记
`EXTERNAL_DATA_UNDECLARED` 告警(无法核验),组织者可据此要求补充。

---

## 8. 诚实的局限声明

这些不是待办事项,是这个 benchmark 的**结构性限制**,写在这里免得被误当成保证:

1. **首帧公开 ⇒ 理论上可以图像检索反查源图。** RSWM-Bench 的源语料
   (AID/RESISC45)是公开的,而首帧按任务定义必须给。有心人可以用首帧去公开
   语料里检索出源图,再暴力搜窗口位置反推 GT。我们不假装这在密码学意义上不可能
   —— 治理靠 §7 的声明+核验,以及赛后公开源图供复核,而不是靠"藏得住"。
2. **`--redact-test` 只在 attrs/datasets 层面去标签。** 它不做像素级的水印或
   扰动;`image` 是原样拷贝(保真度优先 —— 改了首帧,提交与内部实验的条件就不
   是同一个了)。
3. **FVD 是 r3d_18 代理口径**,只在本 benchmark 内部可比(§5.1)。
4. **FID/FVD 需要下载预训练权重**;离线评分机上会记 `null` 并告警,不是 0。
5. **`p < 0.01` 的卡方门只查类别一维。** 帧数/段数/overlap 的分布只报表不判门
   —— 它们由源图组的结构决定,强行平衡会破坏 source-disjoint(那个更重要)。

---

## 9. 版本

| 项 | 值 |
|---|---|
| 划分 schema | `rswm_official_splits/v1` |
| 公开 manifest schema | `rswm_public_test_manifest/v1` |
| 评分报告 schema | `rswm_submission_score/v1` |
| 划分脚本 | `server_run/make_official_splits.py` |
| 评分脚本 | `server_run/score_submission.py` |

`official_splits.json` 里的 `splits_sha256` 是划分的身份;引用本 benchmark 的
结果时请连它一起写,这样别人能确认你评的是同一个 test。
