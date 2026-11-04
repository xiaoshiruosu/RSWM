# RSWM-Inria-120：从 Hugging Face 原始数据到空间生成 Benchmark 的构建方法

**文档版本：hf-construction-v1；日期：2026-09-17。代码基线：`yedda-tv/rswm-1@e8ab7137ecf9153e06190e7885159854405c36c2`。**

本文以现有 RSWM-Inria-120 定义为核心，不更换 120 世界、四起点、F2/F3/F5/F7 和首次公开内容评分。新增两项明确标识的工程配置：仅用于开发的 `smoke12` 子集，以及独立的 `rswm-levir100-lr64-v0.1` 小地图再入诊断。前者不是新的正式基准，后者不加入 Inria 主榜。原 Benchmark-first 论文计划保留在 `docs/PLAN-BENCHMARK-FIRST-ARCHIVE.md`，论文写作蓝图继续有效。

这里的“生成 benchmark”是生成**世界文件、初始观测、空间查询程序、私有几何标签、数据划分与评分协议**，不是先用一个生成模型合成许多视频。生成模型的预测是以后运行 benchmark 时才产生的结果。

本轮核验了 HF 网页、源提交及大文件指针；实际下载原始 TIFF/Parquet 为 0。附带脚本的 21 项本地测试与六组几何示例已经运行；它们验证选行、解码契约和几何等局部行为，不等于已经下载/构建正式集合，也不是全部仓库回归。联网下载、完整 Inria 新导出器、真实 LEVIR Parquet 的端到端运行仍须在联网机器验收。以下预算明确区分网页约数、源提交精确字节和本方案计算量。

## 1. 数据选择：只用两个现有 HF 仓库，不先收集十几个来源

| 用途 | 现有 HF 仓库 | 实际取什么 | 下载预算与得到的世界 |
|---|---|---|---|
| 核心正式基准 | `blanchon/INRIA-Aerial-Image-Labeling` | 原始 `data/train/images/*.tif` 与 `data/test/images/*.tif` 中确定性选出的 120 幅 | 单 TIFF 页面约 75 MB，选中部分约 9 GB；得到 120 幅 2048² 世界；不下载约 27.2 GB 的整个镜像 |
| 开发起步，不进测试主表 | 同一个 Inria 仓库的 `smoke12` | Austin、Chicago、Vienna、Kitsap 各取正式排序前 3 幅，共 12 幅 | 约 0.90 GB 源文件；9 train、3 dev；不下载 calibration/validation/test 影像 |
| 小地图补充 | `blanchon/LEVIR_CDPlus` | 仅三个 test Parquet；从 348 行中固定选 100 行，每行只使用 `image1` | **1,316,127,801 bytes** 源 Parquet；得到 100 幅 1024² 世界；不下载 train 五片 |

Inria 源说明支持 5000×5000、RGB、标称 0.3 m/px、360 张原始 RGB 影像；目录中 train/images 与 test/images 各约 13.5 GB。[H1–H4] LEVIR-CD+ 镜像包含 985 个图像对，字段为 `image1/image2/mask`，test 为 348 行，图像标称 1024²、0.5 m/px。[H5–H7]

**选择理由是连续幅面，不是下载次数。** 核心需要在同一个世界内移动、离开和返回。已经切成 256²/512²且没有可验证邻接信息的小图不能直接提供本配置的连续世界；不能把无关小图拼成“大地图”。LEVIR 世界较小，因此为它定义独立查询，不放大影像，也不冒充 Inria 等难度实验。

本轮不把没有核实到具体 HF 原始大图文件的 LandCover.ai、NAIP 等目录写成“可直接按本脚本下载”的来源。两个所选镜像许可栏为 unknown；这不是自由再分发授权。代码可单独发布；原图、衍生世界及示例影像再分发前需要核查原始权利。[H1,H5]

## 2. 最终 Benchmark 是什么

定义一幅处理后的静态 RGB 世界 `W∈uint8[2048,2048,3]`。对初始中心 `o` 和相对查询 `q_t`，真实裁剪算子为：

```python
cx, cy = o[0] + q_t[0], o[1] + q_t[1]
I_t = W[cy-120:cy+120, cx-208:cx+208, :]
```

一个逻辑任务包含：`(I0, public_spec, query_program, protocol, private_geometry_labels)`。模型只得到 `I0`、公共规范和该次协议允许的查询/历史；完整 W、源城市、绝对 o、目标区域、未来真实裁剪与私有标签留在评测器侧。

模型返回 `Y_t∈uint8[240,416,3]`。评测器保存其不可补写的首次提交。序列下标 t 是空间查询事件，不是现实时间。`I_t` 可用于 train 监督、Oracle 正对照和质量参考集合；未知区域不要求 `Y_t=I_t`。

因此这不是建筑分割、变化检测或视频预测数据集的简单改名。它把原来的大图或双时相图像中的**单幅静态画布**转换成可查询、可返回、可分支的空间生成测试。

## 3. Inria 的下载单位与版本

### 3.1 明确目录，而不是调用 imagefolder 预览采样

使用：

```text
data/train/images/austin*.tif
data/train/images/chicago*.tif
data/train/images/vienna*.tif
data/train/images/kitsap*.tif
data/test/images/bellingham*.tif
data/test/images/bloomington*.tif
data/test/images/sfo*.tif
data/test/images/innsbruck*.tif
```

排除 `data/train/gt/*`、缩略图、README 中图片及自动预览行。镜像预览实际将 gt 作为 imagefolder 类别展示；它的 540 预览行不能解释成 540 幅 RGB 世界。源数据的建筑掩膜不是空间动作标签，也不是未知区域唯一答案。[H1]

### 3.2 本方案给出可明确引用的历史源版本

```text
Inria: ad8315c82c8a3021b93a7d52922d19ea76201f60
LEVIR-CD+: 8062826f12e4f3cd5690b84278ad67c1f42acc46
```

这是本轮从 HF 提交页取得的完整版本标识，不声称它们均是当前最新版本。[H4,H7] Inria 使用已核验的原始文件提交作为本方案的固定起点；若已有正式 manifest 使用其他 revision，应沿用原 manifest 或显式新建数据发布版本，不能覆盖它。相同源路径、不同 revision 会改变现有代码中的 world_id，进而改变任务种子。

源提交中能核验的一个实际例子是：

```json
{
  "repo_id": "blanchon/INRIA-Aerial-Image-Labeling",
  "revision": "ad8315c82c8a3021b93a7d52922d19ea76201f60",
  "path": "data/train/images/austin12.tif",
  "size_bytes": 75040370,
  "lfs_sha256": "0234ee92af8b804122cd331c080df04cf4c16be6bc5898a1adc84121ece40b35"
}
```

这是文件指针信息，不是已经下载的 TIFF 的本地验收。**Xet hash 与原文件 SHA256 不同，下载校验使用原文件 SHA256，不使用 Xet 内容寻址字段。** [H4]

### 3.3 先分地域，再在地域内选择文件

| benchmark 角色 | 原始区域前缀 | 源 split | 每区 k | 世界数 |
|---|---|---|---:|---:|
| train | austin/chicago/vienna | train | 20 | 60 |
| dev | kitsap | train | 12 | 12 |
| metric_calibration | bellingham | test | 12 | 12 |
| metric_validation | bloomington | test | 12 | 12 |
| test_unseen_city | sfo/innsbruck | test | 12 | 24 |

源 split 与 benchmark 角色是两套概念。例如本方案可将原数据 test/bellingham 用作仪器校准，因为这里不参加原建筑分割竞赛。校准不得读取本 benchmark 的两个测试地域后调参。

固定选择算法：HfApi 获取整个固定 revision 的文件元数据；过滤正则 `data/(train|test)/images/([A-Za-z-]+)(\d+)\.tif`；只把 `sanfrancisco` 明确映射为 `sfo`；每城市按 `SHA256('rswm-inria-120-v0.1/' + path)` 升序，哈希并列时按 path 排序；取前 k。它不读取图像像素，不按画面质量或生成难度筛选。

现有代码 world_id 为 `inria-` 加 `SHA256(canonical_json([repo_id, revision, path]))` 的前16个十六进制字符。canonical JSON 使用 UTF-8、键排序、无额外分隔空格、禁止 NaN。上面的 austin12 在指定 revision 下对应 `inria-5a211398adc1d270`；其选择哈希为 `0148fb2ce5cb046cfe05a6ef775957347a3d0f21abb28365d9cdf4542887e9c4`。这些值由指定公式复算，不表示它的随机任务必然等于本文的固定几何示例。

必须保存完整 `source-inventory.json` 和选中 `manifest.json`。manifest 保存 revision、path、source_split、city、role、selection_rank、size_bytes、lfs_sha256、world_id 和总字节。只有选中120行时，不能证明全部候选的 top-k；运行 `verify_manifest(manifest, inventory)` 对完整目录核对。

### 3.4 低预算起步具体选哪12幅

`smoke12` 不重新抽样：从正式 manifest 中保留四个 train/dev 地域各 selection_rank≤3。按原始 1–36 文件命名与同一哈希规则，开发文件名如下；联网运行时仍必须与固定 revision 的完整目录核对。

| 区域 | 文件名（选择顺序） |
|---|---|
| Austin | austin12.tif、austin11.tif、austin5.tif |
| Chicago | chicago31.tif、chicago4.tif、chicago12.tif |
| Vienna | vienna29.tif、vienna35.tif、vienna10.tif |
| Kitsap | kitsap31.tif、kitsap19.tif、kitsap14.tif |

这个阶段只有约0.90GB源影像、12世界、48张不同I0、192任务包；train9世界、dev3世界。没有正式 test，不产生论文主测试结果。后续扩为core120时复用 raw 下载目录即可；源文件不变，但不同构建输出目录保留独立清单。

## 4. 原始 TIFF 怎样处理为世界

### 4.1 校验与解码

对每个选中源文件依次检查：磁盘精确字节数与 manifest 相同；可用 SHA256 相同；Pillow 解码 `size=(5000,5000)`、`mode='RGB'`；NumPy 类型为 uint8、HWC三通道。不调用 `.convert('RGB')` 把单通道建筑掩膜或其他波段图自动伪装成合格源。

坏文件、尺寸不符、hash不符立即停止构建，写入失败记录或保留失败目录。不能跳过后取下一个“更好”文件补足配额。

记录可用 GeoTIFF 标签；没有 CRS/仿射时标记 unavailable。源自大型正射产品不意味着每个镜像文件一定携带经纬度。只可推导标称分辨率，不自行赋地理坐标。

### 4.2 中央裁剪与确定性二倍平均

```python
src = np.asarray(image)                 # [5000,5000,3], uint8
center = src[452:4548, 452:4548, :]      # [4096,4096,3]
sums = center.reshape(2048,2,2048,2,3).sum(axis=(1,3), dtype=np.uint16)
world = ((sums + 2) // 4).astype(np.uint8)
```

即每个世界像素：

`W[y,x,c] = floor((Σ_{i,j∈{0,1}} src[452+2y+i,452+2x+j,c] + 2)/4)`。

2×2块最大和1020，uint16可以容纳；加2再整除4定义半向上舍入，不能替换成 `np.rint(mean)` 的半偶数舍入。例：四个值0、0、1、1，其和2，结果1，不是0。

源标称0.3m/px，因此世界标称0.6m/px；2048像素覆盖1228.8m。不能写成2048m或1m/px。平均在世界层做一次，之后全部视图直接裁剪；不逐帧调整直方图、自动曝光、旋转或有损压缩。

### 4.3 输出文件与内存使用

保存一个 `private/worlds/<world_id>.png` 和一个同名JSON侧车。侧车至少包括 `source_path/source_sha256/world_pixel_sha256/world_file_sha256/source_crop_xyxy/resampling/shape/nominal_gsd_m/source_geotiff_tags`。PNG字节可能受编码版本影响，因此像素hash与文件hash分列。

逐幅解码、转换、写盘并释放，不把120幅5000²数组同时载入内存。单个源RGB数组75,000,000字节，单个世界RGB数组12,582,912字节；这不是完整进程内存。原始源约9GB；120个世界未压缩像素1,509,949,440字节。实际PNG占用必须转换后统计，不预先承诺压缩率。

## 5. 如何从世界得到初始图像与位置查询

四个初始中心保持 `(960,960),(1088,960),(960,1088),(1088,1088)`。每世界保存四张I0，每张供该起点的F2/F3/F5/F7共享；不复制成1920张不同初始图。

以o=(960,960)为例：

```python
I0 = W[840:1080, 752:1168]  # 高240、宽416
```

在原始5000²图中，这块I0对应源范围 `y=[2132,2612)`、`x=[1956,2788)`，再经过相同2×2平均。因而I0左上角像素由源 `[2132:2134,1956:1958]` 的四个RGB像素得到；不存在逐视图再次resize。

对q=(24,0)：

```python
I1 = W[840:1080, 776:1192]
# 命令向右；共同地物在输出图中向左。
I1[:, :392] 对应 I0[:, 24:416]
```

这一事件的真实已知面积是392×240=94,080像素，新出现面积为24×240=5,760像素。该对应只来自坐标，不使用图像匹配估计。

合法相对坐标两轴[-704,704]；所有四起点的整个视口均不越界。模型条件使用 `q/d` 和 `delta_q/d`，`d=sqrt(416²+240²)=480.2665926337163`。右正、下正；跨调用不能重置原点；内部480×832只属于模型适配。

## 6. F2/F3/F5/F7怎样具体生成

正式生成调用仓库已有 `rswm_inria.tasks.packages(world_id, role, origin_index)`，不是把本节示例复制到120幅图。随机数采用当前代码的PCG64；种子来自 `SHA256(canonical_json([VERSION,*parts]))` 的前16字节大端整数。F2、F3和F5-F7使用不同parts。world_id/role/origin固定后，任务可重复；记录NumPy版本和最终坐标JSON，未来不要只依赖随机种子跨版本复现。

### 6.1 F2：多段数值控制

对每起点采样一条路径，q0为0，新查询32。

一次候选：k从{2,3,4}均匀抽取；各段先分6步；剩余32-6k步按k类等概率multinomial分配。方向索引0–7对应东、东南、南、西南、西、西北、北、东北；首方向随机；之后加{-3,-2,-1,1,2,3}之一，即±45/90/135度。正角在y向下图像坐标中顺时针。每段方向向量分量乘{16,24,32}之一，整段增量固定。

累加32个增量得到q1…q32。逐点检查范围，逐边交叠 `max(0,1-|dx|/416)*max(0,1-|dy|/240)≥0.65`，总欧氏路长处于[1d,2.5d]。对角方向更长，不称等速。最多10000候选，失败停止，不换固定fallback。

确定合法示例：16步(24,0)，再16步(0,24)。q1=(24,0)、q16=(384,0)、q17=(384,24)、q32=(384,384)；路长768像素；转弯事件16；最大半径约543.06像素。模型一次收到全部32新查询，返回32张图。

### 6.2 F3：相同形状、两倍幅度

沿用上面的段数与方向算法，基础分量改为{8,12,16}；基础路长[0.60d,1.25d]。构造 `small=r`、`large=2r`，两条路径一起检查边界、交叠和长度，任何一条失败则整对重采样。

示例small：16步(12,0)+16步(0,12)，终点(192,192)；large：对应终点(384,384)。二者分别reset模型，但共享I0与推理seed设置；不要求不同空间位置的未知RGB相同。一个逻辑包包含两个离线流，共64新输出。

本训练配方已包含这些幅度，所以叫尺度响应，不叫未见尺度外推。要研究后者必须另行限制训练支持范围，并使用新的study标签。

### 6.3 F5：首次生成目标的离开与重入

未变换端点：

```text
A=(0,0)
B=(bx,0),             bx∈{448,480,512,544}
C=(bx+cx,cy),         cx∈{-64,-32,0,32,64}; cy∈{416,448,480,512,544}
D=(dx,dy),            dx∈{-544,-512,-480,-448}; dy∈{416,448,480,512,544}
E=(ex,ey),            ex,ey∈{-544,-512,-480,-448}
end=B 或 B+(64,48)
```

段步数依次16、16、32、16、16，总96。每段`a→b`分n步，以 `np.rint(a+(b-a)*j/n)` 得第j步整数坐标。然后对整个命令施加0/90/180/270度旋转以及可选x反射，共8朝向；不旋转图像与矩形视口。核心起点0/2为完全返回，1/3为偏移返回。

参考完全返回路径：

| 事件 | 端点 | q |
|---:|---|---|
| 0 | A | (0,0) |
| 16 | B | (480,0) |
| 32 | C | (480,480) |
| 64 | D | (-480,480) |
| 80 | E | (-480,-480) |
| 96 | end | (480,0) |

偏移参考将最终位置改为(544,48)。该示例完全返回相邻最小交叠约0.748798；偏移约0.729808；不是把B到偏移B的差直接当成相邻运动。矩形视口导致旋转后难度不严格不变，实际接受任务必须记录朝向与交叠。

### 6.4 F7：共享32查询前缀，再走两条路线

前缀A→B→C同F5。后缀a：C→D→E→F→end，其中 `F=(B.x+{-32,0,32},E.y+{-32,0,32})`。后缀b：C→P→Q→R→end，其中：

```text
P.x∈{-64,0,64};          P.y∈{576,608,640,672}
Q.x∈{-672,-640,-608,-576}; Q.y∈{-64,0,64}
R.x∈{-64,0,64};          R.y∈{-672,-640,-608,-576}
```

每后缀4段，每段16查询。F5、F7-a、F7-b一起接受或拒绝，最多1000个候选；三条路径都必须满足边界、相邻交叠和目标再入约束。

参考a：C→(-480,480)→(-480,-480)→(480,-480)→B；参考b：C→(0,640)→(-640,0)→(0,-640)→B。

私有JSON为了便于标签计算，可以在a/b中各保存97个坐标，包含重复前缀；真正执行只运行一次32查询前缀，之后各64，因此**唯一新输出160，不是192**。在C保存状态，两支独立恢复，不共用可写地图。它们分别与前缀首次内容比较，不只比较a与b相似。

## 7. 私有几何标签怎样生成：给出逐像素例子

以未旋转B=(480,0)为例，128²目标在相对坐标中为 `x=[416,544), y=[-64,64)`。起点(960,960)时，它的绝对世界范围是 `x=[1376,1504), y=[896,1024)`。它不在I0中，第一次出现只能由模型生成。

对每个目标像素u=(ux,uy)，逐事件计算：

```text
visible[t,u] = (q_t.x-208 <= ux < q_t.x+208)
               and (q_t.y-120 <= uy < q_t.y+120)
first_event(u) = min{t∈[1,32] : visible[t,u]}
last_prefix_visible_event(u) = max{t∈[1,32] : visible[t,u]}
reentry_event(u) = min{t>32 : visible[t,u]}
invisible_query_gap(u) = reentry_event(u)-last_prefix_visible_event(u)-1
fully_absent_calls(u) = floor((reentry_event-1)/16)-floor((last_prefix_visible_event-1)/16)-1
```

先保证集合非空、t0/t32不可见，再取min/max；不能用空集合argmax伪造事件0。所有目标像素gap≥32，首次和再入跨调用。NPZ保存x/y、first、last、reentry、gap和完整缺席调用数；每张数组128×128。其他坐标与统计放JSON。K/P/N可即时计算，不永久存每帧三张大mask。

**一个可以逐步验算的像素：u=(480,0)。** 在上述F5参考中：first=10，last=20，reentry=93，gap=72。q10=(300,0)，所以它在首次图Y10的行120、列388；q93=(300,-90)，所以它在再入图Y93的行210、列388。评分比较：

`Y10[120,388,:]` 与 `Y93[210,388,:]`。

不是比较两张图同一个像素下标，也不是拿Y93与隐藏真实W比较。Y10若缺失，参考永久无效；Y11、Y16或Y96正常也不能补写这个first槽位。

本轮脚本实际复算的固定示例（几何，不是生成模型实验）：

| 示例 | first范围 | last范围 | 首次再入范围 | 不可见gap | 完整缺席调用 |
|---|---|---|---|---|---|
| F5完全返回 | 7–12 | 17–22 | 92–95 | 72–76 | 3 |
| F5偏移返回 | 7–12 | 17–22 | 91–93 | 70–75 | 3 |
| F7-a参考 | 7–12 | 17–22 | 90–95 | 72–72 | 3 |
| F7-b参考 | 7–12 | 17–22 | 92–95 | 72–74 | 3 |

因此“第16帧到B”不是每个像素首次出现，“第96帧返回B”也不是首次再入。正式参数化任务应报告各自范围，不能复制本表数值。

## 8. Benchmark落盘后长什么样

新增 `scripts/hf_materialize.py inria` 按如下布局导出；核心源处理/任务生成复用原包，不另写一套随机采样。

```text
build/core120/
├── build-selection.json              # fixed revision、完整核心manifest hash、选择列表
├── BUILD_COMPLETE.json               # 只有全部完成才存在
├── public/
│   ├── initial/<opaque_initial_id>.png  # 共480张；一世界四起点
│   └── sessions/<opaque_session_id>.json
└── private/
    ├── worlds/<world_id>.png
    ├── worlds/<world_id>.json
    ├── tasks/<task_id>.json          # 完整程序、起点、角色、元数据
    ├── labels/<task_id>-<stream>.npz
    ├── training-clips.json
    └── index.json                   # task→world→city→role及源manifest关联
```

离线公开session包含I0位置、视口、归一化、合法范围和完整q。F3明确两个流之间reset。online16公开session不包含完整未来程序，只说明由评测器逐块发送当前查询。公开ID不直接包含城市或源文件名；但这是文件组织，不是匿名保证或安全沙箱。

实际调用序列：

| 任务 | 初始化 | generate调用 | 输出事件 |
|---|---:|---|---:|
| F2 | 1 | 一次给q1…q32 | 32 |
| F3 | 2 | small一次、reset、large一次 | 64 |
| F5 | 1 | q1…16、17…32、…、81…96，共6次 | 96 |
| F7 | 1＋两次父状态恢复 | prefix两次、a四次、恢复、b四次，共10次 | 160 |

每起点5次初始化、19次generate调用，352新输出。这里按基准接口计数，视频模型内部可能生成更多上下文/额外帧；额外帧不作为新的受控查询，但计成本。

原仓库`run`入口读私有任务和世界，由评测器裁I0；新公开导出文件用于明确接口及后续发布，不表示原runner已经实现进程隔离。完整`private`目录不能挂载给被测模型。任务文件评分和哈希链只能帮助复算，不能事后证明没有偷看真实地图。

## 9. 如何评价模型返回的图像

令O为I0覆盖，U为过去**计划**覆盖并集，每次查询区域A：`K=A∩O`、`P=(A∩U)\O`、`N=A\(U∪O)`。

**命令坐标交叠**：相邻合法输出按命令建立世界像素对应，计算归一化RGB MAE，不先最佳平移配准。

**真实已知保持**：K对I0。若K为空，是不适用；若有K但输出缺失，是缺失，不是0分。

**生成历史保持**：P对逐像素first槽位公开值。即使first槽位失败，P仍由计划覆盖决定；不能改算为N重新评分。

**首次再入保持**：每个目标像素只在其reentry_event比较一次，与它的first参考对应。F7两支独立锚定前缀。输出完整但first参考无效，也不算完整保持。

每项都保存expected_pixels、valid_pixels、absolute_error_rgb、MAE与coverage：`MAE=error/(255*3*valid_pixels)`，valid=0时MAE=null；不丢掉失败分母。固定噪声地图可能在这些一致性量上全部为0，这恰好说明一致性不等于未知图像合理。

**运动诊断**另从图像估计位移，不以命令作为估计真值；bellingham选参、bloomington冻结验证，报告可信覆盖/歧义/证据不足/失败。现有motion函数尚不等于实景认证。

**未知分布诊断**：对每条计划流，用128像素网格寻找完全在计划未知并集中的块，排除I0；按`SHA256(namespace/x/y/128)`排序取最多16块。先选块后看输出，从跨事件首次台账组装；不完整则标缺失，不从后续输出补齐。F7前缀共享内容若进入分支质量统计，必须单独登记一次或保持branch内配对，不当成独立重复样本。冻结特征骨干、预处理、参考真实块及每世界权重，做真实—真实参照和缺失覆盖报告。不要把任意特征MMD都叫标准KID，更不要把隐藏未来PSNR称为未知生成正确率。

模型结果文件应包括原始PNG/事件日志、RGB评分、运动状态、未知块完整率/特征结果和资源记录；后面三项完整实验集成不由本次数据导出脚本完成。论文必须区分“任务构建完成”“评分代码存在”“指标经独立验证”“真实模型结果取得”。

## 10. 规模：存什么、不存什么

| 角色 | 世界 | I0 | 核心任务包 | 全部任务唯一新查询事件 |
|---|---:|---:|---:|---:|
| train | 60 | 240 | 960 | 84,480 |
| dev | 12 | 48 | 192 | 16,896 |
| metric_calibration | 12 | 48 | 192 | 16,896 |
| metric_validation | 12 | 48 | 192 | 16,896 |
| test_unseen_city | 24 | 96 | 384 | 33,792 |
| 合计 | 120 | 480 | 1,920 | 168,960 |

一个F3包包含两个流，一个F7包包含一前缀和两分支；不能把流数、PNG数、包数混用。120瓦片来自8个预先选定地域，不是120城市；测试只有两个城市。每城市分表，然后等城市权重均值；起点和任务先在世界内聚合。

训练另导出每train世界32个F2 clip、16对F3 clip，共64个33视图clip；60世界为3840 clips、126720真实视图事件。只存q、origin、world_id与pair_id，训练batch时按world裁图。若全部预存这些RGB，会占37,955,174,400未压缩字节，完全没有必要。

120世界原始RGB约1.51GB；480张I0原始RGB约143.77MB。正式测试每方法每种子输出RGB为10,121,379,840字节；3种子约30.36GB。PNG实际压缩和视频原生额外帧另算。数据下载小不意味着模型推理小。

## 11. LEVIR-CD+：1.316GB的小地图补充，具体到文件和标签

### 11.1 固定三个文件，绝不下载全库后再删

固定revision为 `8062826f12e4f3cd5690b84278ad67c1f42acc46`。源提交的大文件指针给出：[H7]

| HF路径 | 精确下载文件字节 | 原文件SHA256 |
|---|---:|---|
| data/test-00000-of-00003.parquet | 439171835 | 75f64aef4b9969de998d3a3454802bf21da01cd1fbe3980a191d15181f3f1353 |
| data/test-00001-of-00003.parquet | 439361709 | 6e34db27af2b3b27825134ceba85afd504745359e70780d00ec4fd7949eab6f6 |
| data/test-00002-of-00003.parquet | 437594257 | 763eb1135eed51b24d50f24a05c67bb11619a08a8b87c5404227548c99af90fb |
| 合计 | **1316127801** | 每片独立校验 |

数据卡test `num_bytes=1316259239` 不是这三份Parquet下载文件大小的和；预算不能把二者混同。读取 `image1` 一列减少解码/内存，并不把需要下载的完整Parquet文件自动缩成三分之一。

### 11.2 在看像素前，选择100行

用PyArrow读取各片footer取得行数N0/N1/N2，不假设三片等长；总数应为348。为所有合法行建立候选key：

`SHA256('rswm-levir100-lr64-v0.1/' + shard_path + '/' + decimal_row_index)`。

按key、shard_path、row_index升序取100行。row_index是片内0起算索引。选行发生在解码图像前；不因图像纹理或模型失败替换候选。保存全部候选片行数、选中排序、revision、片hash。源没有真实pair_id或地理坐标字段时，用`shard:row_index`作来源键，不能虚构地理组。

逐片 `ParquetFile.iter_batches(batch_size=8, columns=['image1'])`，不载入整个985对数据；不读取`image2`与`mask`供轨迹构建。只对选中行的image1解码；字段须是内嵌bytes，外部路径或意外schema停止而不是自动猜测。[H10]

### 11.3 世界处理不resize、不拼接

校验image1为1024²、RGB、uint8；原样保存无损PNG，记录原始编码bytes的SHA256、像素SHA256、可用源image1.path、片行键。100世界原始RGB共314,572,800字节。第二时相不是下一空间帧；变化mask不作为建筑真值。

固定o=(512,512)，I0=`world[392:632,304:720]`。viewport仍240×416；公共q范围[-288,288]²；世界标称512m宽、视口208m×120m。不能和Inria的物理视野或历史难度当成相同条件混榜。

### 11.4 本轮给出并复算一个完整LR64任务，不再只说“重新采样”

新study为 `rswm-levir100-lr64-v0.1`，只测小地图首次再入，不冒充已经实现小地图F2/F3/F7全集。

| 事件 | 相对位置 |
|---:|---|
| 0 | A=(0,0) |
| 16 | B=(256,256) |
| 32 | C=(-256,256) |
| 48 | D=(-256,-256) |
| 64 | 完全返回B，或偏移返回(288,280) |

每段16步，用同一np.rint线性插值。两变体相同I0、相同前48个查询，各自reset；不旋转命令、不声称拓扑泛化。目标R=B中心128²，相对范围x/y均[192,320)，绝对世界范围[704,832)²，不在I0中；q32时完全不可见。

实际几何复算：完全返回first=5–13、last=20–24、reentry=59–63、gap=34–42、相邻最小交叠0.8；偏移返回first/last相同、reentry=58–62、gap=33–41、最小交叠约0.788181。均有1个完整不可见调用块。128²目标不是随意假定能满足约束，而是逐像素验证通过。

每世界导出完全/偏移两个LR64包，共100I0、200包；每方法每种子12800新输出，原始RGB3,833,856,000字节。只是固定拓扑、短间隔的补充诊断，不能代替核心跨路线分支研究。HF说明包含Texas/Austin等来源，缺坐标时不能排除与Inria Austin地理接近或交叉，不称严格地理无交叉OOD。[H5]

当前脚本可构建此扩展的世界、公开session、私有q和NPZ标签。**现有Inria `run_task/Ledger`硬编码2048世界，不能直接运行1024世界；本轮没有把它悄悄填充/放大来兼容。** LR64的模型执行及通用尺寸评分需单独适配后报告结果；本轮只完成构建定义、代码和几何验证，不称完成第二源模型实验。

## 12. 可直接执行的构建命令

### 12.1 安装并获取现有仓库

```bash
git clone https://github.com/yedda-tv/rswm-1.git
cd rswm-1
python -m pip install -e '.[dev,data,motion]'
# 仅LEVIR Parquet读取需要：
python -m pip install pyarrow
```

已有工作区先保存本地修改，再`git pull --ff-only`，不要重新git init或强制覆盖。

### 12.2 固定源清单（不下载全部影像）

```bash
python -m rswm_inria.cli freeze \
  --revision ad8315c82c8a3021b93a7d52922d19ea76201f60 \
  --out data/inria-lock/manifest.json
```

同时生成`data/inria-lock/source-inventory.json`。这一步由HfApi执行，需要联网；不是已经在本次环境中跑通的真实下载。输出缺完整SHA或精确字节时停止。[H8,H9]

### 12.3 先跑12世界开发构建（约0.90GB）

```bash
python scripts/hf_materialize.py inria \
  --manifest data/inria-lock/manifest.json \
  --inventory data/inria-lock/source-inventory.json \
  --profile smoke12 --source-dir data/inria-raw \
  --out build/inria-smoke12 --download --max-source-gb 1.0
```

它检查完整清单，保留四地域各前3幅，逐幅下载/校验/转换，调用现有随机任务生成器，保存I0、公开session、私有任务/标签、训练索引与完成标志。不是仅打印待办事项。`--max-source-gb`约束选中源payload，不代表包括世界/模型/临时缓存的峰值磁盘。已有源文件直接验证并复用。

### 12.4 扩展正式120世界（约9GB）

```bash
python scripts/hf_materialize.py inria \
  --manifest data/inria-lock/manifest.json \
  --inventory data/inria-lock/source-inventory.json \
  --profile core120 --source-dir data/inria-raw \
  --out build/inria-core120 --download --max-source-gb 9.5
```

当前默认预算1GB会拒绝core120，因此必须显式给9.5GB上限。原始下载复用，构建目录分开。输出非空时拒绝覆盖；失败目录留存，修正后用新输出目录。只有所有世界/任务完成才生成`BUILD_COMPLETE.json`。完成后核对120、480、1920和168960这四个计数。

### 12.5 从新索引挑一个dev任务，实际运行已有评分器

```python
import json, subprocess, sys
from pathlib import Path
root = Path('build/inria-core120/private')
index = json.loads((root/'index.json').read_text(encoding='utf-8'))
row = next(r for r in index['tasks'] if r['role']=='dev' and r['family']=='F5')
subprocess.run([
    sys.executable, '-m', 'rswm_inria.cli', 'run',
    '--task', str(root/row['task_path']), '--world', str(root/row['world_path']),
    '--model', 'oracle', '--seed', '0', '--out', 'runs/dev-oracle'
], check=True)
subprocess.run([
    sys.executable, '-m', 'rswm_inria.replay',
    '--run-dir', str(Path('runs/dev-oracle')/row['task_id']),
    '--out', 'runs/dev-oracle-recomputed.json'
], check=True)
```

Oracle只验证仪器。真实增量补全把`--model`改为`lazy-map`并提供真实本地`--model-dir`，安装生成依赖后运行；不提供模型目录时是诊断噪声，不能当论文真实生成基线。文件名、尺寸或输入图不正确时停止，不根据结果删除这个任务。

### 12.6 构建LEVIR100小地图补充

```bash
python scripts/hf_materialize.py levir \
  --source-dir data/levir-raw --out build/levir100-lr64 \
  --download --max-source-gb 1.4
```

该命令固定revision与三片hash/大小，核对schema与总348行，在看像素前确定100行，读取image1并导出两个LR64变体。它不调用Inria scorer，也不声称推理完成。

### 12.7 不下载任何数据即可检查本文几何例子

```bash
python scripts/hf_materialize.py examples --out build/reference-geometry
python -m pytest -q tests/test_hf_materialize.py
```

产生六组完整坐标JSON、128²标签NPZ和geometry-summary。这里的F5/F7是说明用确定路径，不是正式随机核心任务；其中F7-a/b各含前缀，执行计数不能把两条97点路径简单当192输出。

## 13. 最终如何用于一篇Benchmark论文

数据章节应写“复用现有影像并确定性构建空间查询任务”，不是“重新采集120幅遥感影像”。完整论证顺序为：源与文件筛选→地域角色→世界转换→I0与查询→自动几何标签→受限调用→不可补写首次提交→分项评分。

| 科学问题 | 对应任务/对照 | 能回答什么 |
|---|---|---|
| 方向、距离、转弯是否按数值指令执行 | F2＋图像独立运动诊断 | 有移动外观是否等于执行正确命令 |
| 两倍指令是否产生两倍幅度 | F3严格配对 | 固定单位下的尺度响应，不是未知RGB相同 |
| 第一次重新遇见旧区域是否保持 | F5逐像素首次再入 | 避免仅最后B图像掩盖早期重绘 |
| 分叉后的两条路线是否保留共同历史 | F7共同前缀锚 | 分别对共同内容保持，而非仅a/b相似 |
| 保持是网络产生还是地图保存保证 | 同骨干无历史/最近帧/帧缓存/坐标地图 | 分离确定性系统机制与生成行为 |
| 一致的画布是否也像遥感影像 | 固定噪声/平铺＋未知分布诊断 | 一致性与内容分布不能由同一量替代 |
| 小地图、不同来源下是否仍有相同行为 | 独立LR64，不混主榜 | 有限的来源/范围补充，不证明全球泛化 |

最低实际实验要有无数值视频、实际消费数值的生成、真实整图补全、真实增量地图；Oracle和随机图不顶替。地图基线应使用公平的上下文和生成顺序；当前raster整图首窗口看不到中央I0的问题仍须在真实基线前解决。先验证测量，再跑模型；像素错误、缺失、不可测、资源分别报告，不合成世界模型总分。

主文建议设置三节明确的方法：**Data and World Construction**（本文件3–5节）、**Query Programs and Geometric Annotation**（6–8节）、**Evaluation Protocol**（9节）。实验章节不是再描述构建流程，而是报告实际强模型区分度、受控错误验证、同骨干机制对照及来源限制。现阶段可以写方法定义和预注册实验，不能填造模型数、性能改善或实景验证结果。

## 14. 本轮交付与未完成项

本轮新增：具体HF来源与固定提交证据；三片LEVIR精确下载bytes/hash；12世界低预算开发导出；不改变核心的Inria世界→I0→公开/私有任务物化脚本；独立LEVIR100/LR64构建脚本；六组完整几何示例；21项局部单测实际通过。核心`src/rswm_inria`未修改。

本轮没有完成：真实TIFF或Parquet下载、正式120世界与100小世界构建、全部仓库本地回归、真实权重推理、LR64通用尺寸runner/scorer、实景运动校准、冻结特征质量实验、公开安全沙箱。源码/网页可核验不等于真实数据已下载。脚本所在环境无HF外部网络，PyArrow也未安装，因此不能以模拟或几何测试代替真实Parquet验收。

## 15. 可核查来源

[H1] HF Inria数据卡、目录说明、预览与许可：https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling
[H2] HF Inria原始train/images：https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling/tree/main/data/train/images
[H3] HF Inria原始test/images：https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling/tree/main/data/test/images
[H4] 固定Inria原始文件提交及LFS指针：https://huggingface.co/datasets/blanchon/INRIA-Aerial-Image-Labeling/commit/ad8315c82c8a3021b93a7d52922d19ea76201f60.diff
[H5] HF LEVIR-CD+数据卡与来源限制：https://huggingface.co/datasets/blanchon/LEVIR_CDPlus
[H6] HF LEVIR-CD+实际Parquet目录：https://huggingface.co/datasets/blanchon/LEVIR_CDPlus/tree/main/data
[H7] 固定LEVIR提交中的schema、行数和精确文件指针：https://huggingface.co/datasets/blanchon/LEVIR_CDPlus/commit/8062826f12e4f3cd5690b84278ad67c1f42acc46
[H8] HfApi元数据接口：https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
[H9] HF按文件及固定revision下载：https://huggingface.co/docs/huggingface_hub/guides/download
[H10] PyArrow ParquetFile与逐列batch读取：https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html

源事实来自以上网页或文件指针；文件名排序、world_id、几何范围、目标事件与数据规模来自本方案公式/本轮脚本复算。后者不是新增地理测绘或真实生成性能。
