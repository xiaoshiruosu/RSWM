# RESISC45 第二数据集泛化(advisor point 5)

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FA2→SDPA）；H200 141GB 为 legacy/reference，路径保留但本轮未实机回归。

**定位**:训练集固定为 AID,测试集换成 RESISC45 → 掉点 = 泛化间隙。
**不在 RESISC45 上训练**(test-only):便宜、不需要新的 GPU 训练预算,也最站得住。

> 本文档区分三种说法,请勿混读:
> **[已接通]** = 代码里真的通了;**[破损/缺失]** = 不工作或不存在;**[风险]** = 能跑但会静默出错。
> 所有结论都标了 `文件:行号`,可自行核对。**本沙箱没有 numpy/h5py/torch,
> 下面的行为断言来自读码,不是运行结果**;实际可执行的验证见文末「我跑了什么/没跑什么」。

---

## 1. 审计:`--dataset-tier resisc45` 到底通没通

### [已接通]

| 项 | 出处 |
|---|---|
| `--dataset-tier {aid,resisc45,mixed}` argparse 存在 | `lora_remotesense/build_dataset_from_benchmark.py:853-854` |
| tier → 数据集配置映射 | `build_dataset_from_benchmark.py:139-143` → `HD_DATASET_CONFIG_RESISC45:107-118` |
| `DATASET_TIER` 环境变量(默认 aid)可覆盖 | `server_run/common.sh:47` |
| `01_build_data.pbs` 把 `$DATASET_TIER` 透传给构建器 | `server_run/pbs/01_build_data.pbs:19` |
| RESISC45 45 类 → 4 个 scenario group 的映射表存在 | `build_dataset_from_benchmark.py:81-98` (`RESISC45_SCENE_GROUPS`) |
| 256→512 小图拼接分支存在 | `build_dataset_from_benchmark.py:310-340` |
| canonical 参数(window 128×222 / steps 9 / target 480×832 / seed 42)与 tier **无关**,换 tier 不改 | `build_dataset_from_benchmark.py:204-207, 663-674` |
| 训练/评估路径 **tier 无关**(只认 H5,不认来源)→ 无需额外接线 | `common.sh:102-153`, `new/evaluate.py:105-131` |
| `offline_prefetch.sh TIER2=1` 会预取 RESISC45 | `server_run/offline_prefetch.sh:44-48` |

**结论**:「建 RESISC45 数据集」这条链路本身是通的。**但**下面这些问题让它在
**失败时不报错**,以及让「泛化」这个 claim 需要额外披露。

### [破损/缺失]

1. **没有任何 PBS / 编排脚本跑第二数据集** —— 此前只有 `pbs/README_PBS.md:29-39`
   的一段散文说「设 `DATASET_TIER=resisc45` + `WORK=…/work_resisc45`」。
   → 本次补齐:`pbs/40_build_resisc45.pbs` + `pbs/qsub_resisc45.sh`。

2. **裸 qsub 会覆盖旧 AID 结果(严重)**:`_pbs_env.sh:20` 把 `WORK` 默认成
   `$CODE_ROOT/server_run/work`(= 旧 AID work)。若有人
   `qsub -v DATASET_TIER=resisc45 01_build_data.pbs` 而**忘了传 WORK**,
   `01_build_data.pbs:19` 就会把 RESISC45 数据写进 `work/data/train_geo.h5`,
   **直接覆盖旧 AID 数据**。
   → `40_build_resisc45.pbs` 里加了硬守卫:`WORK == 旧 work` 或 `DATASET_TIER != resisc45` 一律拒跑。

3. **H5 不记录真实数据来源**:`build_dataset_from_benchmark.py:749` 把
   `metadata['source']` 硬编码成 `'remotesensebenchmark_hd_v3_plucker'`,
   与 tier 无关。→ 建完之后**无法从 H5 判断它到底是 AID / RESISC45 / 合成**。
   这使下面第 4 条的静默失败不可事后审计。**未修**(构建器是只读文件),
   缓解办法是建库前先 `--check-source`、并保留建库日志。

4. **45 类的细粒度标签没有落盘**:`build_dataset_from_benchmark.py:801` 只存
   `source_scene_category` = benchmark 的 4 个 group(general/flood/urban/rural),
   源图的 RESISC45 类名**丢弃**。
   → 无法做「按 RESISC45 类别的 per-class 泛化分析」;`check_resisc45.py --check-h5`
   也只能报 4-way 分布(45 类分布只能在 `--check-source` 阶段从源数据集看)。

### [风险] 静默失败(能跑完、结果是错的,这是最危险的一类)

5. **加载失败 → 静默回退到合成噪声场景**。
   `build_multi_direction_benchmark_v5.py:227-228` 里 `load_dataset` 失败只
   `print` 不抛;`hf_datasets[cat]` 不被设置 → `:270-284` `generate_scene` 直接改用
   **程序合成的噪声场景**。上层 `build_dataset_from_benchmark.py:347-351` 也吞异常
   → 回退 `:406-410` 原始 loader → 同样合成。
   **后果**:RESISC45 缓存没预取好时,建库会「成功」产出一个 100% 合成的
   「RESISC45 数据集」,配合第 3 条(H5 不记来源)**事后完全查不出来**,
   泛化实验的数字全废。
   → 缓解:`40_build_resisc45.pbs` 在**建库前**强制跑 `--check-source`(离线加载真数据),
   不绿就不建库。

6. **类名对不上 → 场景过滤静默失效**。`_get_scene_filter_indices` 在三处
   (`:227-228` 无 label 特征 / `:237-238` 目标类空 / `:248-249` 匹配到 0 个类)
   都 `return list(range(n))` —— **静默返回全量索引**。
   **后果**:4 个 scene group 抽同一个 45 类池子,「场景」区分失效;更糟的是
   prompt 由 `_make_prompt` 的 `desc_map`(`:569-575`)按 group 硬编码文本,
   于是会出现「urban 的 prompt 配 desert 的图」—— 训练/评估的文本条件与像素不符。
   → 缓解:`--check-source` 对每个 group 调用**构建器真身**
   `_get_scene_filter_indices`,若返回长度 == 全量 → FAIL LOUD。

7. **RESISC45 只用了 45 类中的一个子集**(设计如此,非 bug,但必须写进论文)。
   `RESISC45_SCENE_GROUPS`(`:81-98`)列了 29 个类名;经双向子串匹配
   (`:243-245`)实际命中 **30/45** 类(`railway` 是被 `railway_station` 以
   `name_lower in tc` 反向匹配捎带进 urban 的)。
   未被任何 group 采样的 15 类:`baseball_diamond, basketball_court, cloud,
   golf_course, ground_track_field, harbor, intersection, mobile_home_park,
   roundabout, runway, ship, snowberg, storage_tank, tennis_court,
   thermal_power_station`。
   → 论文应写「RESISC45 的 30 类子集」而非「RESISC45(45 类)」。
   ⚠ 这个 30/45 是用 `check_resisc45.py --self-test` 在**上游 NWPU-RESISC45 标准类名夹具**上
   算出来的(可复现,见文末),**不是**在 `blanchon/RESISC45` 真实类名上算的 ——
   权威数字以集群上 `--check-source` 的输出为准。

8. **拼接接缝 = 泛化实验的混杂因子(必须披露)**。
   AID(600×600 ≥ 512)走 `:303-309` **单图随机裁剪** → 场景是一张连续的真实影像。
   RESISC45(256×256 < 512)走 `:310-340` **2×2 拼接**:`num_tiles=2×2`、tile 256×256,
   四格分别取 `valid_indices[linear_idx % len]` 的**四张互不相关的图**,还各自随机
   翻转/`rot90`。滑窗是 128×222(< 256),所以窗口**跨越 tile 边界时会看到硬内容跳变**,
   而 AID 上不存在这种跳变。
   → **AID→RESISC45 的掉点混杂了三样东西**:(a) 域偏移、(b) 分辨率/GSD 偏移
   (600×600 单图 vs 256×256 上采样)、(c) 拼接接缝伪影。
   这**不是**一个干净的「换数据集」实验,论文里必须说清,否则会被审稿人打。

9. **样本按场景成块排列 → 「评前 N 条」= 只评一个场景**。
   `build_multi_direction_benchmark_v5.py:717-727` 一个 scene 一整块顺序写,
   所以 `sample_00000..N/4` 全是 `general`。而 `evaluate.py:109-115` 是
   `keys=sorted(...)`、按 `val_indices` **顺序**取前 `--max_samples` 条。
   AID 侧没踩到是因为 `train.py` 的 split 本身是随机化的;但 test-only 场景下
   若直接拿 `range(N)` 当 val_indices,`--max_samples 50` 就只会评到 general。
   → 缓解:`check_resisc45.py --make-split` 生成**按场景分层交错**的 split
   (`[general_0, flood_0, urban_0, rural_0, general_1, …]`),取前 N 条自动 4 场景均衡。

10. **误用 AID 的 split 会静默少评样本**。AID 的 `train_val_split.json` 里的索引
    指向 AID H5;拿去评 RESISC45 H5 时,`evaluate.py:113-114` 对越界索引
    **静默 `continue`** → 评估样本数悄悄变少且无告警。
    → 缓解:`--check-split` 校验索引全部落在该 H5 范围内;`40` 的 eval 阶段强制跑。

11. `blanchon/RESISC45` 的 split 名(`"train"`,`HD_DATASET_CONFIG_RESISC45:109`)与
    `trust_remote_code=True`(`v5:217`)在断网节点是否可用,**本沙箱无法验证**
    (无 `datasets`、无网络)。若该 repo 是 script-based,`HF_HUB_OFFLINE=1` 可能加载失败
    → 会触发第 5 条的静默合成回退。**上集群第一件事就是跑 `--check-source`。**

### [风险] 旧 AID 侧的既有 bug —— 只报告,不修

12. `AID_SCENE_GROUPS['flood']` 含 `"port"`(`:70-72`),而匹配是双向子串
    (`:243`:`tc in name_lower`)→ `"port" in "airport"` 为真 → **AID 的 flood 池子
    混进了 airport**。这影响的是**已产出论文数字的旧 AID 结果**,属「不可覆盖」范围,
    本次**不动**。RESISC45 侧不受此影响(已用 `--self-test` 逐类核对,见文末输出)。

---

## 2. RESISC45-only vs RESISC45 + PatternNet(**默认:RESISC45-only**)

**默认只用 RESISC45,不下载 PatternNet。**

`HD_DATASET_CONFIG_RESISC45`(`build_dataset_from_benchmark.py:107-118`)在
`general` 和 `urban` 两个 group 的 `datasets` 列表里写了 `[RESISC45, PatternNet]`。
但底层 `_load_hf_datasets`(`build_multi_direction_benchmark_v5.py:200-228`)
**遍历列表、第一个加载成功就 `break`**(`:213` / `:226`),而 `datasets[0]` 恒为 RESISC45。

> **所以 PatternNet 只是「RESISC45 加载失败时的 fallback」,正常路径下永远不会被用到。**
> 反过来说:如果你**看到** PatternNet 被加载,说明 RESISC45 加载失败了 —— 那是个事故信号。

### 怎么避免下载 PatternNet

`offline_prefetch.sh:44-48` 的 `TIER2=1` 目前**同时**拉 RESISC45 **和** PatternNet:

```bash
if [ "${TIER2:-0}" = "1" ]; then
    huggingface-cli download blanchon/RESISC45   --repo-type dataset
    huggingface-cli download blanchon/PatternNet --repo-type dataset   # ← 默认不需要
fi
```

`offline_prefetch.sh` 是只读文件(未改)。**在联网机上只取 RESISC45**,请直接跑:

```bash
# 推荐:只预取 RESISC45,跳过 PatternNet(省带宽/空间/inode)
export HF_HOME=<你的 STAGE>/hf_home
huggingface-cli download blanchon/RESISC45 --repo-type dataset
# 国内: export HF_ENDPOINT=https://hf-mirror.com
```

即**不要**用 `TIER2=1`;其余(基模 / AID / 评估权重)照常 `bash server_run/offline_prefetch.sh`。

**什么时候才需要 PatternNet**:只有当你**故意**要 `general`/`urban` 混入 PatternNet 时。
但注意:按 `:200-228` 的 `break` 语义,**只要 RESISC45 能加载,PatternNet 就进不来** ——
想真正混入必须改构建器(只读文件,不在本次范围)。所以「RESISC45+PatternNet」目前
**并不是一个可达的配置**,别在论文里这么写。

---

## 3. 独立 WORK 规则(旧 AID 结果神圣不可覆盖)

- 旧 AID 产物 `server_run/work/`(H5、checkpoint、eval 输出、summary.md)**只读**。
- 本实验一律写入独立根:`WORK_RESISC45`(默认 `server_run/work_resisc45`)。
- 三重守卫:
  1. `qsub_resisc45.sh` 拒绝 `WORK_RESISC45 == server_run/work`;
  2. `40_build_resisc45.pbs` **自己**也拒绝(防止有人绕过编排器裸 qsub;
     `_pbs_env.sh:20` 的 WORK 默认值正是旧 work,不挡会覆盖旧数据);
  3. `qsub_resisc45.sh` 还拒绝 `--lora-ckpt` 落在 `WORK_RESISC45` 内部(待评 checkpoint 必须是外部只读引用)。
- `40` 只在自己的 `$WORK/tmp_bench_resisc45` 里做清理,绝不 `rm` 任何 `$WORK` 之外的东西。

---

## 4. 存储 / inode(qk73 容量紧、rx40 inode 紧)

- **聚合优先**:数据只落 2 个 H5(`train_geo.h5` / `train_semantic.h5`)+ 1 个 split JSON
  + 2 个 check 报告 JSON。`check_resisc45.py` 的所有结果写**一个** `--report` JSON,不产生零散文件。
- **中间产物会造 inode**:构建器每个样本写一个 `.mp4`
  (`build_multi_direction_benchmark_v5.py:780-782`)到 `--temp-dir`,外加一份 `benchmark.h5`。
  N=200 → ~200 个小文件。→ `40_build_resisc45.pbs` 在建库成功后 `rm -rf "$TMPB"`
  (只删自己的临时目录)。**别把 `--temp-dir` 指到需要长期保留的地方。**
- **eval 输出是每样本 3 个小文件**(`generated/<key>.npy` + `gt/<key>.npy` + `meta/<key>.json`)。
  `EVAL_N_RESISC45=50` → ~150 个文件。这是既有 evaluate.py 的行为(只读,未改)。
  **不要**把 `--eval-n` 开到几千,否则 rx40 的 inode 会被吃掉。
- 预检阈值(比主 rerun 低,因为这里不训练):可用空间 ≥ 60 GiB、可用 inode ≥ 20k,
  不满足直接拒跑(`qsub_resisc45.sh` 与 `40_build_resisc45.pbs` 各查一次)。
- 规模估算:N=200、480×832、平均 27 帧、gzip → GEO+SEM 合计约 **4-10 GiB**(未实测)。

---

## 5. 精确命令

### 第 0 步(联网机):只预取 RESISC45

```bash
export HF_HOME=$PWD/rswm_offline_stage/hf_home
huggingface-cli download blanchon/RESISC45 --repo-type dataset   # ← 不要 TIER2=1(那会连 PatternNet 一起拉)
# 拷到节点: STAGE/hf_home/ → 节点 $HF_HOME(与已有缓存目录合并)
```

### 第 1 步(集群登录节点):建库**之前**必须全绿

```bash
cd <包根>
export HF_HOME=/scratch/qk73/$USER/RSWM/hf_cache        # 按 _pbs_env.sh:28 的实际值
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python server_run/check_resisc45.py --self-test                    # 纯 stdlib,无需 numpy
python server_run/check_resisc45.py --check-offline --check-source \
    --report /scratch/rx40/$USER/rswm_resisc45/source_check.json
```

`--check-source` 若 FAIL(离线加载不了 / 某个 group 过滤失效)**就不要建库** ——
建了也是第 5、6 条的静默错误数据。

### 第 2 步:提交(DRYRUN 先看)

```bash
export CHECKPOINT_DIR=/scratch/qk73/$USER/RSWM/lingbot-world-base-cam
export WORK_RESISC45=/scratch/rx40/$USER/rswm_resisc45          # 独立 WORK,绝不等于旧 work

# 先干跑:只打印 qsub 命令与依赖,不提交
DRYRUN=1 bash server_run/pbs/qsub_resisc45.sh \
    --lora-ckpt /scratch/rx40/$USER/rswm_corrected/train_r16_sem_s42/best_checkpoint

# 真提交(两个作业:build → eval,串 afterok)
bash server_run/pbs/qsub_resisc45.sh \
    --lora-ckpt /scratch/rx40/$USER/rswm_corrected/train_r16_sem_s42/best_checkpoint
```

可选参数:`--channel sem|geo`(默认 sem)、`--n 200`(建库样本数)、`--eval-n 50`(评估条数);
`FUSED=1` 把 build+eval 合成一个作业(省一次排队,代价是建库那 1-2h GPU 闲置)。

### 第 3 步:几何通道(可选,复用同一份 H5,不会重建)

```bash
bash server_run/pbs/qsub_resisc45.sh --channel geo \
    --lora-ckpt /scratch/rx40/$USER/rswm_corrected/train_r16_geo_s42/best_checkpoint
```

### 第 4 步:读结果 / 算泛化间隙

```
$WORK_RESISC45/eval_resisc45_sem_*/eval_results.json   ← RESISC45(本次)
<WORK_CORRECTED>/eval_r16_sem_s42/eval_results.json     ← AID(旧结果,只读)
```
同一个 checkpoint 的 SSIM / PSNR / LPIPS / FID / FVD / Angular_Error_Mean 逐项对比即泛化间隙。
**报告时必须同时披露第 8 条**(拼接接缝 + 分辨率偏移是混杂因子)与第 7 条(30/45 类子集)。

### 单独用检查器

```bash
python server_run/check_resisc45.py --check-h5 $WORK_RESISC45/data/train_geo.h5 \
    --make-split $WORK_RESISC45/data/resisc45_val_split.json \
    --check-split $WORK_RESISC45/data/train_geo.h5 $WORK_RESISC45/data/resisc45_val_split.json \
    --report $WORK_RESISC45/resisc45_h5_check.json
DRYRUN=1 python server_run/check_resisc45.py --check-source     # 只打印计划
```
任何一项失败 → 非零退出(0=全绿,1=有 FAIL,2=用法错)。

---

## 6. 我跑了什么 / 没跑什么(authoring 沙箱无 numpy/cv2/h5py/torch/网络/pip)

**真的执行过(输出已核对)**:
- `python3 -m py_compile server_run/check_resisc45.py` ✓
- `bash -n server_run/pbs/40_build_resisc45.pbs` / `bash -n server_run/pbs/qsub_resisc45.sh` ✓
- `python3 server_run/check_resisc45.py --self-test` → 9 PASS / 0 FAIL,退出 0。
  真跑了 ast 提取 `RESISC45_SCENE_GROUPS` + 分组匹配镜像(得出上面第 7 条的 30/45)
  + 分层交错的双射性/前缀均衡/不等长桶三个断言。
- `--check-offline`:环境变量未设时 3 FAIL 退出 1;设好后 4 PASS 退出 0、报告 JSON 写出 ✓
- 用法错误:无参数 → 退出 2;`--make-split` 缺 `--check-h5` → 退出 2 ✓
- `DRYRUN=1` 下 `--check-source/--check-h5/--make-split/--check-split` 均只打印计划、
  不 import numpy/h5py ✓
- `qsub_resisc45.sh` 全部守卫:缺 `--lora-ckpt`、ckpt 不存在、`--channel` 非法、
  `WORK_RESISC45 == 旧 work`、ckpt 落在 WORK 内部 → 全部非零退出 ✓
- `qsub_resisc45.sh` DRYRUN happy path:两作业链 + `afterok` 依赖 + `-v WORK=…` 透传 ✓(FUSED=1 同)

**没能执行(沙箱缺依赖/断网,严禁当成已验证)**:
- `--check-source` 的真实路径:需要 `datasets` + RESISC45 HF 缓存。**从未运行。**
  `blanchon/RESISC45` 的真实类名 / split 名 / 图像尺寸 / 离线可加载性 **全部未验证**(见第 11 条)。
- `--check-h5` / `--make-split` / `--check-split` 的真实路径:需要 `h5py` + 一个真 H5。
  **从未运行**(只有其中的纯逻辑 `_interleave_by_category` 被 `--self-test` 覆盖)。
  H5 里 attrs 的 bytes/str 类型、`num_frames` 与 `waypoints` 的实际 shape 等断言 **未经真数据检验**。
- `40_build_resisc45.pbs` **从未 qsub**(按要求不提交作业、不启动训练/GPU 评估)。
  PBS 头部 `-P rx40 / -q gpuhopper / -l storage` 是照抄 `01_build_data.pbs:2-6`,未在 Gadi 上验证过。
- 上面所有关于「静默回退」「break 语义」「成块排列」的断言 **来自读码**,未在运行时观测。
- 容量估算(4-10 GiB)是算术推演,**未实测**。

**已知未修(构建器等是只读文件,不在本次可写范围)**:第 3、4 条(H5 不记来源 / 丢 45 类标签)、
第 5、6 条的根因(静默回退本身 —— 只能靠建库前 `--check-source` 拦截,不能根治)、
第 8 条(拼接接缝 —— 属方法学,须论文披露)、第 12 条(AID flood/airport,旧结果不可动)。
