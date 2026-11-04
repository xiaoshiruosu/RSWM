# RSWM-Bench 复现说明

面向 ICLR reproducibility checklist。**本文件只写已核实的事实**；没核实的一律
标「待确认」，不做乐观假设。核实日期 **2026-07-29**。

---

## 0. 三句话版本

1. **基模是公开的。** `robbyant/lingbot-world-base-cam` 在 Hugging Face 上
   非 gated、非 private、Apache-2.0（下面给了核实方法与可钉的 revision）。
   所以「基模不公开导致无法复现」这个担心，就本项目而言**不成立**。
2. **真正的门槛不是可得性，是算力**：~56GB 权重 + 单卡 80GB 级显存 +
   十几个各 24–40 小时的作业。
3. **即使完全拿不到基模，本 benchmark 仍可用**：数据构建与评估协议对骨干
   零耦合，换任何 I2V 模型都能跑同一张表；诊断基线与全部度量地板甚至
   **纯 CPU 就能复现**（§4 的 Tier 2）。

---

## 1. 基模 `lingbot-world-base-cam` 到底公不公开？

### 1.1 结论与证据

**公开。** 2026-07-29 用 Hugging Face public API 现场核实
（`GET https://huggingface.co/api/models/robbyant/lingbot-world-base-cam`）：

| 字段 | 值 |
|---|---|
| `gated` | `false` —— 不需要接受协议、不需要申请 |
| `private` | `false` |
| `cardData.license` | `apache-2.0` |
| `lastModified` | `2026-02-02T11:44:27Z` |
| `sha`（当前 main 的 commit） | `6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd` |
| 模型卡标签 | 含 `arxiv:2601.20540`、`image-to-video`、`World Model` |
| 文件数 | 31 |

仓库结构（同一次 API 调用取得的文件清单）：

```
low_noise_model/  diffusion_pytorch_model-0000{1..8}-of-00008.safetensors + index.json
high_noise_model/ diffusion_pytorch_model-0000{1..8}-of-00008.safetensors + index.json
Wan2.1_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
google/umt5-xxl/{tokenizer.json, spiece.model, tokenizer_config.json, special_tokens_map.json}
configuration.json  README.md  assets/  examples/
```

即：**两份 DiT（low/high noise 专家）+ Wan2.1 VAE + umT5-XXL 文本编码器**。
这与 `server_run/common.sh` 里 80GB 卡策略的描述自洽
（`--inactive_model_cpu` 把其中一份 14B bf16（~28GB）放 CPU）。

### 1.2 怎么自己复核一遍（不要信本文件，自己跑）

```bash
curl -s https://huggingface.co/api/models/robbyant/lingbot-world-base-cam \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:d.get(k) for k in ['gated','private','sha','lastModified']})"
```

### 1.3 ⚠️ 「现在公开」不等于「永远可得」——必须钉住 revision

上游仓库随时可能被删除、改名或转为 gated；`lastModified` 也说明它被改过。
因此**复现时必须钉 commit**，不要拉 `main`：

```bash
huggingface-cli download robbyant/lingbot-world-base-cam \
    --revision 6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd \
    --local-dir /nix/rsroll_25828/models/lingbot-world-base-cam
```

（本仓库自带的 `server_run/offline_prefetch.sh` **没有**带 `--revision`，
它拉的是 `main`。要严格复现请自行加上上面的参数，或把该 sha 写进你自己的
预取脚本。这是现状陈述，不是本文件在建议改动既有脚本。）

### 1.4 ⚠️ 已知缺口：基模权重没有校验和留档

`lora_remotesense/new/run_manifest.py` 的 `checkpoint_id` 记的是**路径字符串**
（`train.py` 传 `checkpoint_dir`，`evaluate.py` 传 lora 路径或 `"base"`），
**不是权重内容的哈希**。`code_sha256()` 只覆盖代码文件，不覆盖 checkpoint。

后果：如果上游悄悄换了权重，本仓库的产物身份机制**察觉不到**。
在真正下载之前，把 §1.1 的 revision sha 当作唯一的内容锚点。
建议（未实现，留给主协调者决定是否接线）：预取后对
`low_noise_model/` 与 `high_noise_model/` 逐分片算 SHA256 存一份清单，
并把它并入 manifest 的身份字段。

---

## 2. 完整复现（Tier 0）：跑出论文主表

### 2.1 硬件与环境（出处：`server_run/README_RUN.md` §0）

| 项 | 要求 |
|---|---|
| GPU | **Primary validated：A800-SXM4-80GB ×1**（auto offload）；Legacy/reference：H200 141GB ×1 |
| 系统内存 | H200 ≥64GB；80GB 卡 ≥100GB |
| 磁盘 | ≥350GB（基模 56G + 数据 2×~50G + ckpt 5×~10G + 评估 npy ~20G） |
| CUDA | 12.1（驱动 ≥530） |
| Python | 3.10 |
| 依赖 | `requirements-lock.txt`（flash-attn 单独装，见该文件 §9） |

调度器口径是 Gadi/NCI 的 PBS（扁平 `-l ncpus=...,ngpus=1,mem=...`，
**不是** `-l select=`）。换集群要改 `#PBS` 头与
`server_run/pbs/_pbs_env.sh` 里的 conda / 路径。

### 2.2 步骤

```bash
# 1) 环境
cd lora_remotesense/new && bash setup_env.sh     # 装完会硬校验 import flash_attn

# 2) 基模 + 数据 + 评估权重（联网机上做，产出一个可整体拷贝的 STAGE）
bash server_run/offline_prefetch.sh              # ~60GB；见 §1.3 关于 --revision

# 3) 提交作业（包根）
DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh   # 先干跑看清依赖链
bash server_run/pbs/qsub_corrected_rerun.sh
```

作业链（从 `qsub_corrected_rerun.sh` 逐行读出，共 **19** 个 PBS 作业）：

```
01_build_data
   └─(人工批准闸 APPROVED_01_DATA)→ 34a_learnability_drop / 34b_learnability_native   [判门]
        └─→ 20~25（主线 3 种子 × 2 臂）、26/27（rank128）、28/29（attncam）、
            30（rank8）、32（shufpose）、33（geostrip）、35（richtext）、
            36（nopose 静止相机诊断臂）   共 15 个训练+评估作业
        └─→ 31_eval_base（基线评估，取 20 的 split）
```

### 2.3 两道闸，别绕过

* **人工批准闸**：01 建库完成后必须人工核对
  （`semgate_attempts.json` / `overlap_audit.json` / 光流量程标定产物 / 抽检帧），
  然后把 `approval_token` 写进 `$WORK/APPROVED_01_DATA`。批准**绑定这一批数据**
  ——数据重建后 token 变化，旧批准自动失效。
* **可学习性判门（34a/34b）**：用最简单的方向任务（K=1 单段 / 四基本方向 /
  高重叠 `[0.50,0.95)`）分别给 `drop`（文本）与 `native`（Plücker）两条通路
  做正对照。论文的 headline 是一个 null 结论，所以「范式不迁移」必须与
  「这套配方教不会任何东西」可区分。判门 FAIL 就不要往下烧 GPU。

### 2.4 复现时必须对齐的身份字段

任何一次运行都由 `lora_remotesense/new/run_manifest.py` 写出
`run_manifest.json`，**字段缺失一律 fail-closed**。对比结果前先核对：

| 字段 | 说明 |
|---|---|
| `rswm_profile` | `official`（headline，`SEM_ARM_MODE=drop`）/ `legacy`（复现旧数字，`identity`）。由 `server_run/common.sh` 顶部统一裁决 |
| `pose_mode` | `native`（真实轨迹 Plücker）/ `identity`（中和位姿但**仍走 Plücker 通路** = 静止相机诊断臂，**不是** text-only）/ `drop`（位姿通路不接 = 真 text-only） |
| `role` | 每个作业显式声明；`diagnostic_stationary` / `probe_gate` / `variant_attribution` / `legacy` 四类**永不入 headline** |
| `code_sha256` | 覆盖 train/evaluate/metrics/dataset_loader/builder 等核心文件 |
| `data_h5_sha256` / `split_sha256` / `split_seed` / `lora_rank` / `lora_alpha` / `gen_seed` / `frame_num` | 数据与超参身份 |

**最容易踩的坑**：`pose_mode=identity` 不是 text-only。它把位姿中和成单位阵
但仍然把 Plücker 嵌入喂进去，等价于「命令相机保持静止」。真 text-only 是
`drop`。把 identity 当 text-only 报，结论会反过来。

### 2.5 ⚠️ 数值可复现性的边界

`train.py` 只调了 `torch.manual_seed(args.seed)`（seed 默认 42，
`split_seed` 默认 42，`gen_seed` 默认 42）。**没有**设
`torch.use_deterministic_algorithms(True)`，**没有**关 cuDNN benchmark。
因此：

* 同机同环境重跑：结果应当接近但**不保证逐位相同**；
* 换 GPU 型号 / CUDA 版本 / attention 实现（flash-attn 版本或回退路径）：
  数值会漂移，幅度未测。

对齐的正确姿势是比对 §2.4 的 manifest 字段 + 报告多种子（20–25 是 s42/s43/s44
三种子的用途），而不是期待逐位一致。

---

## 3. 拿不到基模怎么办？（这是本节的正题）

即使 §1 的结论有一天失效（仓库被撤、你所在的机构不能用它、或你就是想换骨干），
**RSWM-Bench 的主体依然可用**，因为它的三个层次是解耦的：

```
  [层 1] 数据构建      纯 CPU，只依赖公开遥感数据集，与任何生成模型无关
  [层 2] 评估协议/指标  纯 CPU，输入是"生成帧序列 + GT"，与产生它的模型无关
  [层 3] 被测模型      ← 只有这一层依赖 lingbot-world-base-cam
```

层 3 是可替换件。下面按「你能拿到什么」给三条降级路径。

### Tier 1 —— 换骨干，跑同一张表（推荐的降级路径）

**能复现什么**：全部相对结论（文本通道 vs 位姿通道的差距、rank 消融趋势、
overlap 难度曲线、判门是否通过）。
**不能复现什么**：论文表里的绝对数值（那是 lingbot 骨干的数值）。

怎么做：

1. 用 `lora_remotesense/build_dataset_from_benchmark.py` 建库（层 1 完全不变），
   记下 `data_h5_sha256`，这样你和别人用的是同一份数据。
2. 把你的 I2V 模型接进来。接口契约已经被 `server_run/baselines/registry.py`
   固化成「统一 baseline 适配器」：诊断基线与外部生成式基线**吐出完全相同的
   记录格式**，同一套评分脚本（`trajectory_metrics.score_trajectory` /
   `metrics.py`）无差别打分。照它写一个 adapter 即可，不要另写评估。
3. 关键约定别踩（registry.py 文件头逐条写了，照抄）：
   * `waypoints [K+1,2] = (cy,cx)` 是**相机/窗口中心**路径；
   * 相机方向角 `= degrees(atan2(-dy, dx)) % 360`，east=0/north=90/west=180/south=270；
   * **内容光流 = 相机运动取反**（窗口往东 → 画面内容往西）。
     `direction_word` 记的是**相机**方向词，不要自己再翻 180°；
   * `{K}-segment` 是**段数**，K 段折线有 K−1 个转弯。
4. 位姿通路怎么接决定了你在测哪条臂：`native` / `identity` / `drop` 三态
   见 §2.4，不要混。
5. 报告时明确声明骨干已更换，并把 §5 的三条度量地板一起报出来 —— 换骨干后
   「30° 平均角误差」是好是坏，只有对着地板才读得出来。

### Tier 2 —— 完全不用任何生成模型（纯 CPU 可跑）

`lora_remotesense/new/diagnostic_baselines.py` 提供四条诊断基线，
经 `server_run/baselines/registry.py` 包装（**不重新实现**）：

| 基线 | 性质 | 探测什么 |
|---|---|---|
| `gt_oracle` | **上界 / 指标天花板** | GT 帧原样透传。**永远不要**和真实方法同表排名 |
| `deterministic_translation` | **构造感知上界** | 沿 GT waypoints 直接裁真实窗口。需要完整静态场景 —— 真实世界模型看不到它。这里分数高 = 场景查表，不是生成 |
| `persistence` | 退化对照 | 全 T 帧都用源图。故障模式：几乎零光流，方向不可估 |
| `random_direction_translation` | 错向对照 | 帧很清晰、光流幅度也对，但朝向确定性地与 GT 象限相反。故障模式：「看起来很好，方向全错」 |

**这条路能复现本 benchmark 最关键的可信度部件**：三条度量地板（§5）、
指标实现、判门逻辑、以及「一个方法必须比什么强才算强」的全部参照线。
不需要 GPU，不需要基模，不需要 HF 网络（用 `--no-hf` 建合成库即可先打通管线）。

⚠️ 现状如实说：`server_run/baselines/baseline_status.json` 里
**每一条**的 `tested` 与 `result_available` 都是 `false`，文件顶部的
`warning` 字段自己写着「没有任何一条经过运行时测试，也没有任何一条产出过
数字，不要据此宣称基线对比已完成」。所以 Tier 2 是**可跑的路径**，
不是**已跑完的结果**。

### Tier 3 —— 只验证管线是活的（几分钟，笔记本即可）

```bash
python lora_remotesense/build_dataset_from_benchmark.py \
    -o "$TMPDIR/smoke.h5" -n 8 --no-hf --temp-dir "$TMPDIR/tmp_bench"
for t in server_run/test_*.py; do python "$t"; done
```

`--no-hf` 走程序化合成场景，**不联网、不加载 CLIP**（builder 里语义门的挂载
条件是 `use_hf and gate_enabled()`）。这正是 `.github/workflows/ci.yml` 做的事，
已在纯 CPU、无网、无 torch 的环境下实跑通过。

---

## 4. 数据怎么拿

本仓库**不分发影像像素**（权利链断点，见 `LICENSE` APPENDIX B）。
四个源数据集都能从 Hugging Face 公开取得（2026-07-29 核实均非 gated）：

| tier | 仓库 | HF 声明的许可 |
|---|---|---|
| `aid`（canonical 主集） | `blanchon/AID` | `unknown` |
| `whurs19`（泛化主集） | `jonathan-roberts1/WHU-RS19` | `cc-by-4.0` |
| `resisc45` | `blanchon/RESISC45` | `unknown` |
| `patternnet` | `blanchon/PatternNet` | `unknown` |

许可条款的逐条分析与「待确认」项见 `LICENSE` APPENDIX B。
用 `--local-aid-dir` 也可以走本地目录（`root/<CLASS>/*.jpg`），
`check_aid_pools.py` 对本地与 HF 两种布局分路判别合法性。

复现数据身份的方式：**分发 `run_manifest.json` + 建库参数，而不是分发 H5**。
建库对 `(源数据集, -n, seed, 各旋钮)` 确定性，别人照参数重建后
`data_h5_sha256` 应当对得上；对不上就说明环境或上游数据变了，这正是要发现的。

---

## 5. 报结果时必须一起报的三条地板

（详细出处见 `DATASHEET.md` §5.1，这里只列结论）

| 参照线 | 数值 |
|---|---|
| 四方向 prompt 的量化地板 | **≈ 22.5°** |
| 随机方向的期望角误差 | **90°** |
| 光流估计器在 **GT** 上的实测地板（**仪器噪声**） | **≈ 25°**（`step=1` 口径实测 median 24.45° / mean 35.93°） |

⚠️ 角误差必须声明 **step 口径**。默认 `step=3` 在 GT 上读出 82.67°，那是超出
Farnebäck 量程的垃圾读数，不是指标语义（`server_run/validate_direction_metric.py`
里有实测记录和大声警告）。不声明口径的角误差数字之间不可比。

---

## 6. 已知的复现障碍（诚实清单）

按「会不会挡住别人」排序：

1. **算力**：19 个 PBS 作业，训练作业 walltime 上到 40 小时，单卡 80GB 级。
   这是最实在的门槛，没有软化办法。降级见 §3。
2. **flash-attn 与 torch 的版本不一致**：`requirements.txt` 注释里给的本地
   wheel 是对 **torch 2.6** ABI 编的，而本栈钉的是 **torch 2.4.0**；而且那个
   `.whl` 并不在仓库里（`find . -name "*.whl"` 为空）。必须走 `setup_env.sh`
   的自动匹配逻辑（按已装 torch 的版本与 CXX11 ABI 拼 wheel 名去官方 release 拉，
   以 `import flash_attn` 成功与否为唯一判据）。详见 `requirements-lock.txt` §9。
3. **依赖未 freeze**：除 torch/torchvision/flash-attn 外，其余依赖仓库里只有
   下界。`requirements-lock.txt` 里逐行标了「待实测确认」，服务器跑通后应当
   `pip freeze` 回填。
4. **基模无内容校验和**（§1.4）。
5. **数值非逐位确定**（§2.5）。
6. **基线尚未产出任何数字**（§3 Tier 2 末尾）。
7. **调度器绑定 Gadi/NCI**：PBS 头是扁平格式，换集群需要改
   `#PBS` 行与 `server_run/pbs/_pbs_env.sh`（conda 路径、`ASSET_ROOT`、
   离线开关）。作业逻辑本身与调度器无关。
8. **语义门的偏置与其证据基础**：见 `DATASHEET.md` §4.2 / §4.3。
   这不挡住复现，但会影响你怎么解读 per-class 的结果。

---

## 7. checklist 速查

| ICLR 常见问项 | 本仓库的位置 |
|---|---|
| 代码是否公开、许可 | `LICENSE`（Apache-2.0）+ `NOTICE` |
| 数据是否公开、许可 | `LICENSE` APPENDIX B + 本文件 §4 |
| 数据集文档 | `DATASHEET.md`（Gebru et al. 体例） |
| 依赖与环境 | `requirements-lock.txt`、`lora_remotesense/new/setup_env.sh` |
| 自动化测试 | `.github/workflows/ci.yml`（纯 CPU、无网） |
| 随机种子与确定性 | 本文件 §2.5 |
| 算力开销 | 本文件 §2.1 / §6.1 |
| 结果的身份可追溯 | `lora_remotesense/new/run_manifest.py`（fail-closed） |
| 已知局限 | `DATASHEET.md` §3.2 / §4.2 / §4.3 / §5.2，本文件 §6 |
| 如何引用 | `CITATION.cff` / `CITATION.bib` |
