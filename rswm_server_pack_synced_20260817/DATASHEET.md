# Datasheet for RSWM-Bench

按 Gebru et al., *Datasheets for Datasets* (CACM 64(12), 2021) 的问题清单组织。

**本文档的取数规则**：每个数字都必须能在本仓库代码里指到出处，或标注核实方式与
日期。指不到出处的数字一律不写；已知不确定的一律显式写「待确认」。
文中形如 `文件:行号` 的引用指向本仓库当时的代码；仓库在演进，行号可能漂移，
以标识符名为准。

**核实日期**：2026-07-29。Hugging Face 侧的数字通过 HF public API 现场取得
（`/api/datasets/...`、`/api/models/...`、`/api/datasets/.../tree/main/...`）。

---

## 1. Motivation

**为什么创建这个数据集？它要解决什么任务？**

RSWM-Bench 用来测一件很窄的事：**一个视频世界模型在遥感影像上，能不能听懂
"往哪个方向走" 这个指令**，以及这个指令通过**文本通道**给和通过**相机位姿通道**
（Plücker embedding）给，效果差多少。为此需要一批「同一张源图 + 已知的窗口
运动轨迹 + 与之严格对应的文本描述」的样本，且轨迹方向是可被独立测量的
（用光流反解），而不是靠模型自评。

**谁创建的、谁资助的？**
RSWM-Bench 的作者（见 `CITATION.cff`）。资助方：**待确认**（提交前补齐）。

**这不是什么？**
不是遥感语义分割/分类基准，不是真实卫星视频数据集（见 §2 的「已知失真」），
也不是通用视频生成基准。

---

## 2. Composition

### 2.1 一条样本是什么

一条样本 = 一段合成的「滑窗视频」+ 一条文本 prompt + 一条相机轨迹：

HDF5 布局 `f['data']/sample_XXXXX`。下表的形状与 dtype 是本机 CPU 实跑
`-n 8 --no-hf` 后从产物读出来的，不是照抄注释：

| 字段 | 内容 | 出处 |
|---|---|---|
| `video_frames` | `(T, 480, 832, 3)` **float32** | `convert_benchmark_to_lingbot` |
| `image` | `(480, 832, 3)` float32，= 首帧（I2V 条件图） | 同上 |
| `full_scene` | `(512, 512, 3)` float32，滑窗的底图 | `DEFAULT_SCENE_SIZE = 512` |
| `waypoints` | `(K+1, 2)`，折点的窗口中心 `(cy, cx)` 像素坐标 | `server_run/baselines/registry.py` 文件头约定 |
| `trajectory` | `(T, 2)`，逐帧窗口中心 `(cy, cx)` | 同上 |
| `segment_angles` | `(K,)`，每段的**相机**方向角（度） | 同上 |
| `poses` | `(T, 4, 4)` float32，camera-to-world | `lora_remotesense/new/camera_trajectory.py` |
| `intrinsics` | `(T, 4)` float32（占位内参） | `_make_dummy_intrinsics` |
| attrs | `prompt` / `num_frames` / `num_segments` / `overlap_ratio` / `source_scene_category` / `source_scene_id` / `pose_source` | 建库期写入 |

`pose_source ∈ {plucker_from_trajectory, plucker_from_waypoints, dummy_identity}`
如实标注位姿是从逐帧 `trajectory` 还是从 waypoints 均匀插值来的 —— 后者在转弯
时刻会与视频错位，下游据此判读。

### 2.2 帧数契约

`steps = 9`（`DEFAULT_STEPS_PER_SEGMENT`），`T = 9 + (K-1)*8`，因此
`T ≡ 1 (mod 4)`（Wan 系 VAE 的时间下采样要求）：

| K（段数） | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| T（帧数） | 9 | 17 | 25 | 33 |

主库的段数默认区间是 `[min_segments=2, max_segments=4]`
（`build_dataset_from_benchmark.py` 里 `_min_seg = ... or 2` / `_max_seg = ... or 4`），
所以**主库不含 T=9 的样本**，实际 T ∈ {17, 25, 33}。判门探针库显式用
`--min-segments 1 --max-segments 1`，因此 T=9。

实测确认（本机 CPU，`-n 8 --no-hf`）：8 条样本的 `num_frames` 分别为
17/33/17/33/17/25/33/33，与上表逐条一致。

### 2.3 几何配置

* 源图 → 场景：AID 是 600×600；WHU-RS19 当前快照为 1001 张 600×600
  加 4 张异形图。WHU 全部保持大图模式、每样本恰用 1 张唯一源图；短边不足
  512 的 726×456 与 600×500 先做确定性的等比例 resize-to-cover，再走既有
  512×512 随机裁剪。958×612 与 711×718 无需归一化。RESISC45/PatternNet
  是 256×256，走**小图拼接
  模式**：2×2 拼成 512×512，每样本消耗 4 张源图。
* 滑窗：128×222（`DEFAULT_WINDOW_HEIGHT=128` / `DEFAULT_WINDOW_WIDTH=222`），
  宽高比对齐目标 480:832 ≈ 0.577，避免 letterbox。窗口再插值放大到 480×832。
* 相机姿态：**旋转恒为单位阵，只有平移**，光轴垂直向下看地面
  （`camera_trajectory.py` 文件头 "keep rotation = identity (pure translation,
  camera orientation fixed looking straight down at the ground plane)"）。

### 2.4 重叠率（overlap）

`overlap_ratio` 是**区间**约束，不是单侧上界：

| 库 | 区间 | 出处 |
|---|---|---|
| 主 benchmark | `[0.00, 0.10)` | builder argparse 默认（`--max-overlap-ratio` help：默认 0.10） |
| 34a/34b 判门探针 | `[0.50, 0.95)` | `server_run/pbs/01_build_data.pbs`：`PROBE34_OV_LO:-0.50` / `PROBE34_OV_HI:-0.95` |

### 2.5 规模

`N_MAIN` 默认 4000（`server_run/common.sh`：`N_MAIN="${N_MAIN:-4000}"`），
四个场景类各 1000 条。评估分层：headline 臂 400 条、消融/诊断臂 100 条
（`N_EVAL_HEADLINE:-400` / `N_EVAL_ABLATION:-100`，后者是前者的前缀子集）。

`-n` 必须是 4 的倍数（四场景等分）。

### 2.6 源数据池的实际容量（AID，HF 现场核实 2026-07-29）

`blanchon/AID` 的 `train` split 有 **10001** 条记录 = 10000 张 jpg + 1 张
`thumbnail.png`（后者被 `check_aid_pools.py` 的 `img_ok()` 与
`_class_from_data_path` 排除）。30 个类目录的逐类张数相加恰为 10000。

| 场景池 | 类数 | 可用源图 |
|---|---|---|
| water | 4（Pond 420 / Port 380 / River 410 / Beach 400） | **1610** |
| urban | 9（Commercial 350 / DenseResidential 410 / Industrial 390 / MediumResidential 290 / Parking 390 / RailwayStation 260 / School 300 / StorageTanks 360 / Church 240） | **2990** |
| rural | 5（BareLand 310 / Farmland 370 / Meadow 280 / SparseResidential 300 / Forest 250） | **1510** |
| general | 11（Airport 360 / Bridge 360 / Center 260 / Desert 300 / Mountain 340 / Park 350 / Playground 370 / Square 330 / Stadium 290 / Viaduct 420 / Resort 290） | **3670** |
| 合计（进池） | 29 | 9780 |
| **未进任何池** | 1（BaseballField） | **220** |

两条由此推出的硬约束：

1. **AID 上 `N_MAIN` 的理论上限 = 4 × 1510 = 6040**。builder 硬禁 `%` 回绕
   （`SceneSelectionError: ... 禁止回绕`），池抽干就报错而不是复用源图。
   语义门的重采还会额外消耗池（§4.2），所以实际可用上限低于 6040 且没有
   闭式表达 —— 想上更大规模必须先跑一次建库看池够不够，不能按 6040 规划。
2. **AID 的 BaseballField（220 张，占 AID 的 2.2%）永远不会出现在数据里**，
   因为 `scene_taxonomy.json` 的 AID 四池只列了 29 个类。builder 的导入期
   断言 `_assert_scene_groups_disjoint()` 只校验四池**互斥**，不校验对源数据集
   类别的**覆盖完整性** —— 漏一个类不会报错，会静默不采样。

### 2.7 有标签吗？

有三层「标签」，性质完全不同，不要混：

* `scene_category ∈ {general, water, urban, rural}` —— **prompt 唯一声明的语义**，
  由 `scene_taxonomy.json` 从源图的类别标签映射而来。
* `source_class` —— 源图在原数据集里的类别（provenance，对**整张源图**为真）。
  裁剪后不保证仍成立，因此**不作为逐帧保持目标**（`semantic_gate.py` 文件头明确
  写了这条口径）。
* 方向词 ∈ {east, north, west, south} —— 由首末 waypoint 的净位移角落到 4 个
  90° 宽的桶（`_make_prompt`：`<45 或 ≥315 → east`，`[45,135) → north`，
  `[135,225) → west`，其余 → south）。

### 2.8 缺失信息 / 噪声

* 没有像素级标注。「轨迹是否离开类别主体」只能用场景级逐帧评分做代理
  （§4.2），这是本数据集最大的方法学让步。
* `general` 类的 prompt 是 "a mixed remote-sensing scene"，不含语义声明，
  因此**不设语义门**（§4.2 的偏置只作用在 water/urban/rural 三类上）。

---

## 3. Collection Process

### 3.1 数据从哪来

不采集新影像。四个公开遥感场景分类数据集（`--dataset-tier`）：

| tier | HF 仓库 | 原生尺寸 | 用途 |
|---|---|---|---|
| `aid` | `blanchon/AID` | 600² | canonical 主集 |
| `whurs19` | `jonathan-roberts1/WHU-RS19` | 600²×1001 + 4 个异形尺寸 | 泛化**主**集（零拼接；短边不足 512 时等比例 resize-to-cover） |
| `resisc45` | `blanchon/RESISC45` | 256² | 泛化辅助（2×2 拼接） |
| `patternnet` | `blanchon/PatternNet` | 256² | 泛化辅助（2×2 拼接） |

选 WHU-RS19 而非 RESISC45 作泛化主集的理由写在 `offline_prefetch.sh` 里：
256² 拼接会引入接缝伪影这一**额外域偏移**，跨数据集泛化差距就无法单独归因。

### 3.2 ⚠️ 这是静态影像上的滑窗，不是真实卫星视频

这是理解本 benchmark 一切结论的前提，必须写在最前面：

**每条"视频"都是在一张静止的卫星/航拍图上滑动一个矩形窗口截出来的。**

由此，真实卫星视频里存在而本数据集里**结构性缺席**的因素至少有：

* **时间变化**：地物不动，车不开，船不走，水面不流，云不飘。相邻帧之间除了
  平移没有任何内容变化。
* **光照/季节/大气**：单张源图内恒定。真实视频里的日照角变化、云影、雾霾、
  传感器增益漂移全部为零。
* **视差与三维结构**：相机是正射向下的纯平移（§2.3），场景被当作一个平面。
  高楼、山体在真实斜视卫星视频里会产生视差与遮挡变化，这里一律没有。
  因此本 benchmark 里的 "camera pose" 实质是**二维窗口轨迹被提升进
  Plücker 表示**，不是真正的 6-DoF 视点变化。
* **重采样伪影**：128×222 窗口被插值放大到 480×832（放大约 3.75×），
  所有帧都带同一套上采样模糊；256² tier 还额外带 2×2 拼接的接缝。

**对结论的影响**：在本数据集上"方向可控"成立，只能推出模型学会了
**图像平面上的二维平移指令跟随**。推不出它理解三维视点运动，也推不出它在
真实卫星视频上还成立。任何跨到真实视频的断言都需要另做实验。

### 3.3 采样机制

* 每个场景池在建库开始时用固定 RNG 做一次确定性 `rng.permutation`，
  样本 `scene_id` 按序消费该排列 → 同一次建库内**不同样本不共享源图**。
* 大图模式：`scene_id >= len(order)` 直接抛 `SceneSelectionError`，
  **禁止 `%` 回绕**。回绕曾是「四重完整重复样本」事故的根因之一
  （见 `check_benchmark_duplicates.py` 文件头的尸检记录）。
* 小图拼接模式：一条样本消费 `tiles = 4` 张源图，起点 `scene_id * tiles`，
  越界同样硬错。

---

## 4. Preprocessing / Cleaning / Labeling

### 4.1 四道门

| 门 | 脚本 | 判据 |
|---|---|---|
| 池预检 | `server_run/check_aid_pools.py` | 四池互斥 / 池内无重复 / 每个索引是合法训练图 / `len(pool) >= threshold` / 并集 = 各池之和；FAIL 退出码 4 |
| 重复 | `server_run/check_benchmark_duplicates.py --expect-zero` | `traj_fp` / `video_fp` / `scene_fp` 三类指纹分组，任何组 >1 即报 |
| split 泄漏 | 同上 `--split` | 同一 `video_fp` 组横跨 train/val = 保真指标泄漏（最严重） |
| 语义保持 | `server_run/check_semantic_consistency.py` | 见 §4.2 |

`server_run/make_precheck_split.py` 在建库后立刻预生成 split，使泄漏门不必等到
训练作业产出 split。它与 `lora_remotesense/new/dataset_loader.py` 的
`split_dataset`（`train_ratio=0.8`，`split_seed` 默认 42）必须产生**逐位相同**
的划分，脚本头把这条同源约定写死了。

### 4.2 ⚠️ 语义门的系统性偏置（本数据集最重要的已知缺陷）

**背景**：AID 的类别标签是**整图**属性。滑窗裁剪后窗口可能离开类别主体，
此时 prompt 的场景语义（例如 "a water-dominated remote-sensing scene"）
对这些帧就是**假监督** —— 而文本条件通道正是本 benchmark 要测的东西。
2026-07-27 的人检坐实了这一点：`sample_00009`（water/River）12 帧里 10 帧
是纯草地，`sample_00011`（urban/StorageTanks）有 4 帧是 100% 河水
（`semantic_gate.py` 文件头原话）。

> ⚠️ 源文档内部不一致，如实记下：这里写的是「12 帧」，而冻结规则的
> `FROZEN_P0_FRAMES = 9`、阈值也是按 9 帧的 2/9–3/9 中点定的。两处说的
> 大概率不是同一批帧（人检发生在早期 smoke H5 上，规则冻结在后来的 9 帧
> 口径上），但仓库里没有留下能判定的记录。**待确认**：以服务器上的
> `rule_decision_v2712.txt`（只读 + SHA256 留档）为准。

**门的实现**：`lora_remotesense/semantic_gate.py`，CLIP 零样本逐帧评分
（默认 `openai/clip-vit-base-patch32`）。正类 prompt = 本池全部类名成句，
负类 = 其余三池类名成句；帧对正类集的组均值相似度 > 对负类集的组均值
⇒ 该帧"同意"声明。样本通过 ⇔ 同意帧占比 ≥ 阈值。

**冻结规则**（`FROZEN_P0_RULE = "frac"`）：

```
frac >= 0.2777777777777778        # = 9 帧下 2/9 与 3/9 的分界中点，必须全精度
FROZEN_P0_FRAMES = 9
SEMGATE_VERSION = "semgate_v3_1_20260729"
```

**校准结果 —— 类别层面的欠采样偏置**：

在 20 样本人检真值集（`server_run/calibration_labels_smoke_v261.json`，
`_human_check_date = 2026-07-28`）上，这条规则的表现是：

* **0 个坏样本被放行**（两个真坏样本 `sample_00009`、`sample_00011` 都被拒）；
* **3 个有效样本被误拒**：`sample_00006`（water / **Port**）、
  `sample_00016`（rural / **BareLand**）、`sample_00019`（rural / **Farmland**）。

分母要说清楚：20 条里有 5 条是 `general`（不设门），受门管辖的只有 15 条；
其中 13 条有效、2 条无效。所以**在这个校准集上，有效样本的误拒率是
3/13 ≈ 23%**，不是 3/20。

**为什么这构成类别层面的偏置**：误拒的机制不是随机的，是可复现的零样本混淆，
代码注释里逐条记着：

* Port 的帧里"码头船排"在 CLIP 特征上更像 **parking lot**（urban 池）；
* BareLand 的帧像 **Desert**（general 池）。

两个混淆都是**跨池**的，于是负类池的最大相似度压过正类池，帧被判"不同意"。
被拒的样本会由建库器换一张源图重采（确定性游标推进，池耗尽硬错，
`RSWM_SEMGATE_MAX_TRIES` 默认 6），于是：

> **Port / BareLand / Farmland 三类在最终数据集里的占比，低于它们在源池中的
> 占比。**数据没有变脏（被拒的是有效样本，只是被丢掉了），但**类别分布被系统性
> 扭曲了**。按 §2.6 的池容量算，Port 占 AID water 池的 380/1610 ≈ 23.6%，
> BareLand+Farmland 合计占 rural 池的 680/1510 ≈ 45.0% —— 受影响的份额不小。

**这条证据的强度必须如实说**：上面是 **3 个样本**上的观察，加上代码注释里
记录的混淆机制。**类级拒收率没有被测量过** —— 没有人跑过"把 Port 类全部样本
过一遍门，统计拒收比例"。`server_run/semgate_build_report.py` 会把这三类
（代码里称"已知误拒类"）的重采次数单列，正式建库跑完后应当用它把类级拒收率
补测出来，届时本节的数字要更新。

**在哪些结论上要小心**：
* 任何按 `source_class` 分层的分析（例如"哪类场景更难控"）都会被这个偏置污染。
* 跨数据集泛化实验里，RESISC45/PatternNet/WHU-RS19 的类构成不同，门的偏置
  也不同 —— 泛化差距里有一部分来自**门的行为差异**，不能全部归因于域偏移。

### 4.3 ⚠️ "语义保持"这个判据本身的证据基础很薄

必须与 §4.2 分开说：§4.2 是"门判错了谁"，本节是"门这个判据本身可不可信"。

* **判据来源**：CLIP 零样本文本-图像相似度比较。这是一个**代理**，不是"语义
  是否保持"的定义。`semantic_gate.py` 文件头自己写了：没有像素级标注时，
  "轨迹不离开类别主体"只能以场景级逐帧评分为代理。
* **验证规模**：**只有 20 个人检样本**，其中只有 15 个进入门的管辖范围。
* **无标注者一致性数据**：`calibration_labels_smoke_v261.json` 只记录了一个
  `_human_check_date` 和一段 `_note`，**没有标注者身份、没有第二遍独立标注、
  没有 Cohen's κ 或任何一致性统计**。也就是说，"哪些样本算无效"这个真值本身
  是**单人单遍**判断，不可复核。
* **规则冻结范围有限**：代码注释明确写着 "范围 = 受控 P0；生产/全量规则未冻结"，
  以及"第五轮特征迭代禁止"（防止看到结果后再改规则的预注册纪律）。
  换言之，这条阈值是在 20 样本上选的，**没有在 4000 样本的正式库上验证过**。
* **阈值的精度陷阱**：0.2777777777777778 是 2/9 与 3/9 的中点，
  写成 0.28 会改变判定。任何复现必须用全精度值。

**给复现者的建议**：不要把语义门当作"数据集干净"的证明。它是一道**召回优先**
的粗筛（0 漏放 / 23% 误拒），其代价已经在 §4.2 里量化。如果你的研究依赖
per-class 的干净度，请自己重做标注。

### 4.4 prompt 的构造

```
A satellite video of {场景模板}. The camera follows a {K}-segment
trajectory with net displacement toward the {DIR}.
```

* 场景模板来自 `scene_taxonomy.json` 的 `prompt_templates`（4 条固定句）。
  旧版把普通水景写成 "a flooded area" 的灾害语义已废除 —— 那是给文本通道
  灌错误语义，对专门研究文本方向信号的实验是混淆变量。
* `{K}` 是**段数**，K 段折线有 K-1 个转弯，别读成转弯数。
* `{DIR}` 是**相机**方向词。注意：**内容光流 = 相机运动取反**（窗口往东 →
  画面内容往西），评估侧不要再自己翻 180°。

### 4.5 原始数据保留了吗

源数据集不修改、不重分发（见 `LICENSE` APPENDIX B）。每条样本的 provenance
写进 `_HD_SOURCE_REGISTRY`：`source_dataset / source_split / source_index /
source_path / source_class`。WHU 另外记录 `source_original_hw /
source_normalized_hw / source_preprocess / source_preprocess_scale`。

**已知局限**：小图拼接模式（RESISC45 / PatternNet）下一条样本用了 4 张源图，
但登记表只记**首块 tile** 的索引作代表（代码里写着
`"source_index": int(tile_indices[0]),   # 代表索引 = 首 tile`）。
所以这两个 tier 的源图 provenance 是**不完整的**：无法从产物反查出另外 3 张
源图是谁。AID / WHU-RS19（大图模式，1 样本 1 源图）不受此限。

---

## 5. Uses

**已经/打算用来做什么**：方向可控性评测（文本通道 vs Plücker 位姿通道）、
LoRA rank 消融、overlap 难度消融、跨数据集泛化。

### 5.1 ⚠️ 度量地板：低于这些数字的差异不构成证据

角误差指标有三个必须同时摆在表里的参照线：

| 参照线 | 数值 | 含义 |
|---|---|---|
| 四方向 prompt 的量化地板 | **≈ 22.5°** | prompt 只给 4 个 90° 宽的方向桶（§2.7）。即使模型完美听话，落在桶中心，对均匀分布的真值角，平均绝对误差也是 22.5°。 |
| 随机方向的期望 | **90°** | 均匀随机朝向的期望绝对角误差。 |
| 光流估计器在 **GT** 上的实测地板 | **≈ 25°** | 这是**仪器噪声**，不是模型的错。 |

第三条的精确出处（`server_run/validate_direction_metric.py`）：在 ground-truth
滑窗视频上跑同一条 Farnebäck 管线，**step=1 口径**下实测 mean 35.93° /
**median 24.45°**（≈ 量化地板）。默认 `step=3`（每对约 27 px）下读数是
**82.67°**，那是**超出 Farnebäck 量程的垃圾读数**，不是指标语义；脚本在
全分辨率帧上跑时会大声警告（旧版曾在 480×832 原始帧上直接跑，地板读数
82.2°）。**任何角误差表都必须声明 step 口径**，否则数字不可比。

**推论**：一个"平均角误差 30°"的结果，距离仪器地板只有几度。要主张方向可控，
必须同时报出 GT 地板与随机基线，并给出样本量下的最小可检测效应
（`server_run/metric_sample_size_study.py` 按 `1.96·√2·s/√n` 的闭式算 MDE）。

### 5.2 不适合的用途

* **不要**用来评测三维视点控制 / 新视角合成能力（§3.2：无视差、无旋转）。
* **不要**用来做遥感变化检测、时序分析（无时间变化）。
* **不要**在 per-class 层面下强结论（§4.2 的类别偏置）。
* **不要**把 `deterministic_translation` / `gt_oracle` 这类诊断基线与真实方法
  同表排名 —— 它们能看见真实静态场景，是构造上的上界不是可部署方法
  （`server_run/baselines/baseline_status.json` 里逐条写了这条警告）。

---

## 6. Distribution

* **像素不分发**。分发的是建库配方、评估协议与审计元数据；使用者按
  `REPRODUCIBILITY.md` 从上游取源数据后本地重建。
* 许可：代码 Apache-2.0，原创非代码数据 CC BY 4.0，源影像条款见
  `LICENSE` APPENDIX B（含四个数据集逐条的"待确认"标注）。
* 出口管制 / 第三方 IP 限制：源影像大量来自 Google Earth 渲染图，
  上游权利链有断点，这是"不分发像素"的直接原因。

---

## 7. Maintenance

* 维护者与联系方式：见 `CITATION.cff`（**待确认**：投稿匿名期用 OpenReview
  匿名渠道）。
* **产物身份的单一来源**是 `lora_remotesense/new/run_manifest.py`。任何一份
  结果都带 `run_manifest.json`，字段缺失一律 fail-closed。
* 数据集版本：`scene_taxonomy.json` 的 `taxonomy_version`
  （当前 `water_v1_20260726`）、`prompt_template_version`
  （`pt_v3_water_20260726`）、`semantic_gate.py` 的 `SEMGATE_VERSION`
  （`semgate_v3_1_20260729`）。改 taxonomy 必须同步改版本号。
* 勘误发布方式：更新本文件 + 版本号；**不得**在不改版本号的情况下改判定规则。

---

## 附：本文档中所有数字的核实方式

| 数字 | 核实方式 |
|---|---|
| T = 9+(K-1)·8，主库 T ∈ {17,25,33} | 本机 CPU 实跑 `-n 8 --no-hf` 后读 H5 属性 |
| AID 10001 条 / 30 类 / 逐类张数 / 池容量 | HF public API 现场取，2026-07-29；30 类张数之和 = 10000 自洽 |
| BaseballField 未进任何池 | 脚本比对 `scene_taxonomy.json` 的 AID 四池与 HF 上 30 个类目录 |
| `frac >= 0.2777777777777778` / 0 放行 3 误拒 / Port·BareLand·Farmland | `lora_remotesense/semantic_gate.py` 的 `FROZEN_P0_*` 与其上方的冻结注释 |
| 20 样本、其中 5 条 general、2 条无效 | `server_run/calibration_labels_smoke_v261.json` 逐条计数 |
| 无标注者一致性数据 | 同一 JSON 中不存在 annotator / 第二遍 / κ 字段 |
| 22.5° / 90° / 25° 地板 | `_make_prompt` 的四桶逻辑；`server_run/validate_direction_metric.py` 的 35.93° / 24.45° / 82.67° 实测记录 |
| overlap 区间 | builder argparse 默认 + `server_run/pbs/01_build_data.pbs` |
| N_MAIN=4000 / N_EVAL 400·100 | `server_run/common.sh` |
| 拼接 tier 只记首块 tile | `build_dataset_from_benchmark.py` 的 `_HD_SOURCE_REGISTRY` 小图分支 |
