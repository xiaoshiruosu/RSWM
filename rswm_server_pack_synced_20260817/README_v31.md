# RSWM-Bench v3.1 — 变更与验证状态

> **当前部署说明**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FA2→SDPA）；本文件中的 H200 内容属于 legacy/reference，
> sm90/FA3 支持仍保留，但本轮没有 H200 实机回归。

相对 v3.0(`rswm_server_pack_v30.tar.gz`)。

---

## 一句话

v3.0 之后修的都是**会让正式跑挂掉或让数字失去意义**的东西:一个 4000 样本库上
必然 OOM 的哈希、一个自己拦死自己的自检、一批从未在 CI 里跑过的自检模块,
以及新增的跨库/跨划分查重与拼接接缝检测。

---

## 1. P0:`make_variant_h5.py` 在正式规模上必然 OOM

```python
# 旧
src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
```

`read_bytes()` 把**整个源 H5** 读进内存。canonical 库带 `full_scene`,
12 个样本就 1.3 GB;`N_MAIN=4000` 时是几百 GB。

这个脚本在 **32 / 33 / 35 变体臂的生产路径上** —— 正式跑会 OOM 被 kill
(exit 137),而且是在 `afterok` 链中间挂,后面全部作业跟着废。

写全自动语义门审计时我自己先踩了这个坑(实测 exit 137),然后全仓扫同款模式,
发现四处,其中两处对 H5 用整文件读:

| 位置 | 对象 | 风险 |
|---|---|---|
| `server_run/make_variant_h5.py:64` | **源 H5** | **P0 必 OOM** |
| `server_run/recompute_fvd_i3d.py:264` | i3d torchscript 权重 | P2 浪费内存 |
| `server_run/baselines/registry.py:306` | 一个 .py | 无风险 |
| `lora_remotesense/build_dataset_from_benchmark.py:77` | taxonomy JSON | 无风险 |

修法:`run_manifest.py` 里已有的分块实现改成公开的 `sha256_file` 作单一来源,
两处 H5/权重路径改用它。已验证分块与整文件哈希在非 chunk 整数倍的文件上
**逐位一致**。

---

## 2. 新增:`server_run/check_near_duplicates.py`(44 条自检)

### 为什么

`make_official_splits.py` 保证的是 **source-disjoint** —— 同一个 source id 不会
同时进 train 和 test。这**不等于** content-disjoint:它只看 id,不看像素。

更要紧的是四个 tier 全是 Google Earth 派生:

| tier | 源 | 512 目标下的网格 | 接缝 |
|---|---|---|---|
| canonical(train) | AID 600² | `ceil(512/600)=1` → 1×1 | 无 |
| 泛化 #1 resisc45 | RESISC45 256² | `ceil(512/256)=2` → **2×2** | **第 256 行/列** |
| 泛化 #2 patternnet | PatternNet 256² | 2×2 | **第 256 行/列** |
| 泛化 #3 whurs19 | WHU-RS19（600²×1001 + 4 个异形尺寸） | 1×1；短边不足 512 的 2 张先等比例 resize-to-cover | 无 |

网格数字来自建库器 `build_dataset_from_benchmark.py:480-483`,不是推测。

### 三个检查

```bash
# 0) 阈值必须先标定 —— 不标定就拒绝判定
python3 server_run/check_near_duplicates.py --calibrate --h5 canonical.h5

# 1) 划分泄漏(内容级,不只是 id 级)
python3 server_run/check_near_duplicates.py --check leakage \
    --h5 canonical.h5 --splits official_splits.json

# 2) 跨库重复(canonical vs 各泛化 tier)
python3 server_run/check_near_duplicates.py --check cross \
    --h5 canonical.h5 --other resisc45.h5 --other whurs19.h5

# 3) 拼接接缝
python3 server_run/check_near_duplicates.py --check seam --h5 resisc45.h5
```

退出码:`0` 通过 / `3` 无法标定→拒绝出数 / `4` 划分泄漏 / `5` 跨库重复超上限 /
`6` 输入不可用。

### 与 `check_benchmark_duplicates.py` 的分工(**先跑那个**)

| | 口径 | 范围 | 特点 |
|---|---|---|---|
| `check_benchmark_duplicates.py`(已有) | md5 逐位精确 | 单库内 | 便宜,证据是字节级 |
| `check_near_duplicates.py`(新) | 感知哈希 | 跨库 | 抗重编码/缩放/翻转 |

精确 ⊂ 近似,所以新工具覆盖它的功能,但**它更该先跑**:md5 相同是不可争辩的,
感知哈希判定永远带 TPR/FPR。

### 已验证(真数据故障注入,不是只跑了自检)

| 注入 | 期望 | 实测 |
|---|---|---|
| tier = canonical 的重发布副本 | 高重复率 → FAIL | `dup_rate=1.0000` rc=5 |
| tier = 独立 1/f 纹理 | 0 → PASS | `dup_rate=0.0000` rc=0 |
| 同库内跨 split 内容重复 | 抓到并 rc=4 | 抓到 `train/sample_00000 ↔ test/sample_00008` d=2 |
| 无泄漏对照 | PASS | rc=0(证明不是误报机器) |
| 真 2×2 拼接图 | 接缝可检出 | `seam_z=21.37`,100% 场景 |
| 直裁库同口径 | 不可检出 | `seam_z=0.59` |
| 600² 源 → 1×1 网格 | N/A 而非假阴 | N/A |

### 工具本身在自检中暴露的真缺陷(已修)

**pHash 在低纹理场景上是噪声。** 线性渐变图经 JPEG-70 后 pHash 距离 22/63,
而两张**完全无关**的图距离也是 22 —— 同源与异源在这种图上没有区别。

后果不是误报而是**漏报**:真重复被噪声抬到阈值之上,检测器安静地报
"未检出重复"。大片水面/裸地正是这种图,而 taxonomy 里 BareLand+Farmland 占 45%。

已加 `hash_stability()`:±1 LSB 抖动下翻位数 ≥ 判定阈值的场景,**如实剔除并计数**,
不算作"已检查过"。实测渐变图翻 20 位、1/f 纹理图翻 0 位。
标定输出里报 `n_dropped_low_texture` 与 `n_blind_at_threshold`。

**标定失败原来在浪费信息。** 泄漏检查最初返回 rc=3,因为注入的泄漏对
污染了负例 —— 标定假设"source_key 不同 ⇒ 内容不同",而这假设**恰好在有泄漏时
不成立**。原来一律 rc=3,把一个发现当成了工具故障。现在分开诊断,直接指出
`(0,8)@d=2`,并给两条出路:`--calib-h5 <干净库>` 或 `--calib-allow-exclude`
(被排除的对**登记为重复发现**,不隐藏)。两条都验证过能抓到泄漏。

### 必须照写的盲区

**能**检出:同一张源影像经不同管线(重编码/缩放/轻微裁切/翻转旋转)。

**不能**检出:同一地理区域被**不同取景**采集(裁框不同、季节不同、分辨率差一档)
—— 需要地理坐标,而这四个 HF 数据集都不带 geo-metadata。

所以 dup_rate 是**下界**。论文/datasheet 里不得写"两库无交集",
只能写"未检出可由感知哈希识别的重复"。这条盲区恰恰覆盖了遥感库之间
最常见的交集形态。

---

## 3. CI:11 个自检模块从未跑过

CI 原来只跑 `server_run/test_*.py`(8 个文件,711 条断言)。实测仓库里另有
**11 个模块自带 `--self-test`,CI 一个都没跑**:

```
lora_remotesense/new/backbone.py          lora_remotesense/new/baseline_outpaint.py   19
lora_remotesense/new/camera_trajectory.py lora_training/camera_trajectory.py
server_run/check_near_duplicates.py    44 server_run/check_resisc45.py                 9
server_run/instrument_floor_report.py     server_run/recompute_fvd_i3d.py
server_run/semgate_arbiters.py         26 server_run/semgate_auto_auditset.py         20
server_run/semgate_frozen_inputs.py
```

新增 CI 步骤 **C2**,用 `find`+`grep` 自动发现(不写死清单,新增模块自动纳入),
并断言发现数 ≥10 —— 发现逻辑坏掉时 fail 而不是静默少跑。

写这个步骤时踩了两个坑,都已在注释里记下:
1. `ci.yml` 自身含 `"--self-test"` 字面量 → 会被匹配进来,然后
   `python ci.yml --self-test` 必然失败;
2. 不同 grep 实现对 `-r` 与 `--include` 的组合行为不一致(ugrep 把
   `--include` 的值当路径找)。

所以改成先 `find` 限定 `.py` 再 grep。

---

## 4. 两个自检模块此前是红的

### `recompute_fvd_i3d.py` —— 自己的门拦死自己的自检

v2.9.4 给它加了 headline 资格门(`collect_pairs` → `_assert_headline_eligible`),
但自检夹具没跟着更新,于是整个 `--self-test` 卡死在 `[3/4]`:

```
[headline] 被重算 FVD 的 eval 目录 .../eval_x 没有 run_manifest.json
```

没有只"补个文件让它过",而是升级成**真验门**——三条故障注入:
缺 `run_manifest.json` / `headline_eligible=false` / 带 `manifest_override` 痕迹,
逐条确认拒跑,再确认合规 manifest 下放行。

### `semgate_frozen_inputs.py` —— 一条**依环境而变**的断言

它断言 `unverifiable` 理由里含"证明不了",但代码有两个分支:
"当前可算出值"那支写的是"不能证明",而本机没有 CLIP 权重时走的是
"两边都取不到"那支、根本没写这句话 → 有权重的机器绿、没权重的机器红。

语义上两种情况的理由是**同一个**(冻结时没记上,现在能不能算出来都不改变
"证明不了当时用的是什么"),所以统一了两支措辞,并让单测显式断言
"无论当前环境可否取得,结论都不是 match"。

---

## 5. 验证状态(本包)

| 项 | 结果 |
|---|---|
| `py_compile` 全仓 | 219 个 `.py` 全通过 |
| `bash -n` 全仓 | 75 个 `.sh`/`.pbs`,失败 0 |
| `server_run/test_*.py` | 8 个文件、**711 条断言**,失败 0 |
| `--self-test` 模块 | 11 个,失败 0,**~118 条断言** |
| `--no-hf` 建库冒烟 | 8 样本,帧数契约 `T=9+(K-1)·8` 且 `T≡1 (mod 4)` 全满足 |
| 查重器真数据故障注入 | 7 项全按预期开火(见 §2) |

---

## 6. 本包**不含**、以及仍然没有的

### 交付中(下一版并入)

`server_run/make_counterfactual_set.py` 与 `server_run/counterfactual_report.py`
由并行 agent 正在写,已落盘但**未经我复核**,且后者的 `--self-test` 还需要
`--gt-h5/--pairs`(不符合"自检必须自带夹具"的约定,会让 CI 步骤 C2 变红)。
故本包排除,待复核后进 v3.2。

同批还在进行的:K↔T 解混淆(`deconfound_spec.json` / `61` / `62` / 报告)、
`score_submission.py` 九类拒绝规则故障注入、torch 惰性导入的独立复核、
12 个既有交付文件的第二双眼睛。

### 需要真数据/GPU 才能有结论

**跨库重复率没有真数字。** 本机无 HF 缓存,四个 tier 的 H5 得在集群上建。
工具已就绪,但结论只能等真数据。这条直接决定泛化 tier 还剩几个可用:

* `resisc45` / `patternnet` 的 gap **混着一个纯人造伪影项**(2×2 接缝,
  实测同口径 z=21.4 可检出 vs 直裁 0.59)。模型没见过接缝,在带缝场景上
  必然变差 —— 与地理/传感器域移无关。这两个 tier 的 gap 不能单独当泛化证据。
* `whurs19` 是**唯一零拼接**的泛化 tier,但它与 AID 同课题组且 1001/1005
  为 600² Google Earth 影像,**恰恰是最可能有源影像交集的那个**。

也就是说:两个 tier 被伪影污染,剩下那个可能是同分布重测。跑
`--check cross --h5 canonical.h5 --other whurs19.h5` 才知道。

### 仍未解决(需要论文层面决定)

* 文本/drop 任务欠定 —— 单 GT PSNR 评判多模态未来
* world-model 命名
* 六个外部 baseline 全部 `tested=false`
* 派生 H5/帧数据的再分发许可(未获明确许可前只发生成器 + 源 ID + 校验清单)
