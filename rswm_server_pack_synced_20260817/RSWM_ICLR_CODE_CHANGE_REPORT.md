# RSWM ICLR Full Pipeline 最新代码修改说明

**文档日期：** 2026-08-17  
**项目：** Remote Sensing World Model（RSWM）/ RS-Roll  
**用途：** 师兄内部技术审查  
**审查结论：** 当前代码已形成较完整的、失败即停的 ICLR 主实验工程链，但服务器最新源码与运行产物尚未在本次审查环境中出现，因而本文可以作为代码走查材料，不能替代服务器验收记录或实验完成证明。

## 0. 审查范围与证据边界

用户指定的服务器源码目录为：

```text
/nix/rsroll_25828/src/rsroll-bench-iclr-20260817-full-pipeline/rswm_server_pack
```

服务器运行目录为：

```text
/nix/rsroll_25828/work/rsroll_official/
```

本次执行环境无法访问上述两个 `/nix` 路径，当前任务终端也没有服务器 `ls`、日志、`tmux` 或 `nvidia-smi` 输出。因此，本文的代码结论来自工作区内可访问的 2026-08-17 交付包解压副本；该交付 ZIP 的 SHA-256 为：

```text
23F499553446DB4D2A42DDCD3B2EE53083814E6D0FFDEDBB44A6D7069908532A
```

为避免把本地副本冒充服务器现状，本文使用三类状态：

- **代码核验：** 可由本次实际读取的源码、配置或测试直接证明。
- **用户报告：** 来自服务器现象描述，但本次没有相应服务器文件或日志可复核。
- **待服务器核验：** 必须以指定 `/nix` 目录的实际文件、报告或终端输出为准。

特别说明：可访问副本中不存在任何 `server_run/start_full_auto_pipeline_v*.sh`。因此，这组脚本及其“GPU 等待—自动重建—后续实验”行为不能在本文中写成已核验事实；如服务器上存在，它们应在最终签字前补充源码哈希和逐段审查。

## 1. 项目背景

RS-Roll 将遥感空间外推表述为受连续相机轨迹控制的视频生成问题：输入单张遥感图像、场景提示和连续轨迹，输出按轨迹排列的 17、25 或 33 帧 rollout。当前主干是 camera-enabled LingBot-World/Wan I2V video diffusion backbone，采用 LoRA 做单卡 A800 适配，并把 camera-to-world 位姿转换为 Plücker ray condition 注入 Video DiT。

该任务模拟的是静态遥感场景上的连续空间视点移动，不是地表随时间变化的预测。正式表述宜使用“camera-coordinate-conditioned spatial rollout”或“viewpoint-conditioned spatial consistency”，不应把当前二维平面位姿代理写成显式三维重建。

## 2. 数据集构建部分

### 2.1 原构建流程

旧版 `server_run/build_correct.sh` 已经调用 canonical builder，而不是错误地调用 `lora_remotesense/new/build_from_remotesense.py`。其主要不足不是 builder 选错，而是工程保护较弱：

1. 直接把构建结果写到最终 `OUT`，进程被终止时可能留下看似正式、实则未闭合的 HDF5。
2. 已存在的 `OUT` 会直接跳过构建，随后只做内嵌的首样本/有限字段检查。
3. 未强制 AID-only、离线 Hugging Face、绝对路径本地 CLIP snapshot 和 `RSWM_SEMGATE=1`。
4. 未显式钉死 overlap 区间，也没有对全部样本做 source identity、SemanticGate 证据和 duplicate gate 审计。

仓库仍保留 `lora_remotesense/new/build_from_remotesense.py` 作为历史/其他路径，但它不是论文协议入口。当前正式流程不得直接调用它。

### 2.2 当前 canonical builder

当前唯一正式数据入口为：

```text
server_run/build_correct.sh
  -> lora_remotesense/build_dataset_from_benchmark.py
  -> wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py
```

正式参数由入口显式固定：

| 项目 | 当前协议 |
|---|---|
| 数据源 | AID（`blanchon/AID` 或显式 `LOCAL_AID_DIR`） |
| 样本数 | 2000；四个 scene pool 均分，因此必须为 4 的倍数 |
| 原始滑窗 | 128 × 222 |
| 训练/评估分辨率 | 480 × 832 |
| 每段采样点 | 9 |
| 轨迹段数 | 2–4 |
| rollout 帧数 | 17 / 25 / 33；段边界去重 |
| 轨迹角度 | 连续角，`angle_quantize=0` |
| 起点—终点 overlap | `[0.0, 0.1)` |
| 数据随机种子 | builder 默认 42 |

128×222 与 480×832 具有一致宽高比，避免旧 96×96 正方形窗口拉伸到宽屏目标所造成的额外横向模糊。帧数遵循

```text
T = 9 + (K - 1) × 8,  K ∈ {2, 3, 4}
```

故得到 17、25、33，并满足视频 VAE 的 `4n+1` 时间长度约束。

### 2.3 离线 AID 与本地 CLIP

**代码核验：** `build_correct.sh` 强制：

- `HF_HUB_OFFLINE=1`；
- `HF_DATASETS_OFFLINE=1`；
- `TRANSFORMERS_OFFLINE=1`；
- `RSWM_SEMGATE=1`；
- `RSWM_SEMGATE_MODEL` 必须是绝对路径，且包含 CLIP config、processor config 和模型权重；
- AID 必须已存在于本地 Hugging Face cache，或由 `LOCAL_AID_DIR` 显式提供。

`run_iclr_full_pipeline.sh` 在正式构建前另设 AID pool 检查和 CLIP 本地加载检查。CLIP 预检使用 `local_files_only=True`，不会把缺失权重静默转化为联网下载。

### 2.4 SemanticGate

SemanticGate 用于数据质量控制，不是训练 loss，也不参与模型推理。它对 `water`、`urban`、`rural` 的抽样帧做 CLIP prompt-bank 判定，`general` 因无具体场景声明而跳过。未通过的候选会换源图重采；连续失败达到 `RSWM_SEMGATE_MAX_TRIES`（默认 6）时硬失败，不会静默放行。

当前代码支持：

```text
RSWM_SEMGATE_MIN_FRAC
RSWM_SEMGATE_MIN_FRAC_WATER
RSWM_SEMGATE_MIN_FRAC_URBAN
RSWM_SEMGATE_MIN_FRAC_RURAL
```

但 **可访问正式入口没有设置具体的 class-specific 值**。没有环境覆盖时，三个语义类均使用源码中的冻结基础阈值 `0.2777777777777778`。因此，“已经调整 water 阈值”不能仅凭代码成立，必须同时提供服务器启动环境、构建日志和最终 HDF5 中逐样本 `semgate_threshold` 证据。

### 2.5 overlap、重复与来源审计

当前验证分为三层：

1. `server_run/check_dataset.py`：检查 2000 样本、HDF5 结构、480×832、128×222、steps=9、17/25/33、AID 来源、连续角、逐样本 overlap、Plücker trajectory pose、像素方差及 SemanticGate 落盘证据。
2. `server_run/check_semantic_consistency.py`：使用本地 CLIP 对最终 HDF5 独立复核；门模式禁止只抽部分样本后签发 PASS。
3. `server_run/check_benchmark_duplicates.py`：检查 trajectory template、video content、full-scene image 和 source identity 重复；正式参数要求零重复、零缺失来源身份。

新版 `build_correct.sh` 先写 `train_geo.h5.partial`，只有完整 validator 通过后才在同一目录原子重命名为 `train_geo.h5`。已有 `.partial` 会被保留并硬失败，避免自动覆盖诊断证据。

## 3. 自动化实验流水线

### 3.1 已核验的正式总控

`server_run/run_iclr_full_pipeline.sh` 是可访问副本中的端到端总控。其实际流程为：

```text
[默认只读 plan / status]
          |
          v
[四项显式 execute 确认 + 全局 flock]
          |
          v
[01 环境、基础模型、I3D 与 GPU inventory]
          |
          v
[02 离线 AID snapshot 与四池检查]
          |
          v
[03 本地 CLIP snapshot 加载检查]
          |
          v
[04 复用有效 train_geo.h5 / 等待外部构建 / canonical 新建]
          |
          v
[05a HDF5 结构与冻结协议验证]
          |
          v
[05b 独立 SemanticGate 复核]
          |
          v
[05c duplicate 与 source-reuse gate]
          |
          v
[06 source-disjoint official train/val/test split]
          |
          v
[06b 冻结 400 样本 AID test manifest]
          |
          v
[07 Primary LoRA：seed 42 -> 43 -> 44]
          |
          v
[08a 每个 seed 单独生成 official test rollout]
          |
          v
[08b 标准化三个 Primary 结果目录]
          |
          v
[08c 计算 Primary paper metrics]
          |
          v
[09 导出 paper CSV 与 export_manifest.json]
```

每一步有独立追加日志和原子 `.done` checkpoint。恢复方式是用完全相同的参数重新执行；只有 marker schema、signature 和产物验证同时通过时才跳过。checkpoint 与产物不一致时 fail closed，不会自动重跑覆盖旧结果。

### 3.2 GPU 等待的实际职责边界

**代码核验：** `run_iclr_full_pipeline.sh` 不实现持续 GPU 等待。它在环境步骤读取 `nvidia-smi`，训练阶段委托给 `server_run/iclr_a800/run_iclr_main_a800.sh`，后者再委托正式 `server_run/rswm_train_runner.sh`。正式 runner 要求：

- `/nix/rsroll_25828/work/rswm_scheduler_status.env` 存在；
- scheduler 状态为 `READY_FOR_TRAINING`；
- data、CPU tests、GPU preflight、dataset smoke、GPU smoke 五个 gate 全为 1；
- 训练前再次检查 GPU 使用率、显存和独占锁。

`server_run/rswm_auto_scheduler.sh` 是独立 readiness scheduler：它等待 RESISC45 下载完成，检查 AID/WHU/RESISC/模型/评估权重，执行 CPU dataset smoke，连续观察空闲 GPU，再做 A800 单步 smoke，最后只写 `READY_FOR_TRAINING`；它明确不启动正式训练。该脚本属于较广的多数据集 readiness 基础设施，不等同于 AID full pipeline。

### 3.3 `start_full_auto_pipeline_v*.sh` 状态

**待服务器核验：** 可访问副本中没有这些文件，故无法确认其版本号、GPU 等待算法、损坏 HDF5 处理方式、环境变量、重建策略、日志和 resume 语义。用户描述的目标流程可写为：

```text
GPU 等待
  -> 数据集状态检查
  -> 必要时重新构建
  -> dataset / SemanticGate / duplicate 验证
  -> official split 与 manifest
  -> 三个 Primary LoRA 训练
  -> official-test evaluation
  -> baseline comparison
  -> paper table export
```

但在取得服务器脚本前，上图只能视为预期设计，不能视为已实现审查结论。

### 3.4 baseline comparison 的实际范围

当前 full pipeline 只运行 `baseline_groups.primary`，即三个 Primary seed 的标准化、指标计算与表格导出。仓库虽然配置了 `persistence`、frozen I2V、image outpainting、RSWISE bridge 和 no-camera 等 `main` baseline，但外部 checkpoint/producer output 不齐时会硬失败，full pipeline 也没有自动运行整个 `main` 组。

因此，当前流程更准确的表述是“Primary multi-seed comparison and table export”，而不是“完整外部 baseline comparison 已自动完成”。

## 4. 正式实验配置

### 4.1 Primary 配置

当前主实验固定为：

| Experiment ID | Pose | LoRA targets | Rank/alpha | Train seed | Split seed | Generation seed |
|---|---|---|---|---:|---:|---:|
| `iclr_primary_native_attn_cam_r16_s42` | native | `attn_cam` | 16/16 | 42 | 42 | 42 |
| `iclr_primary_native_attn_cam_r16_s43` | native | `attn_cam` | 16/16 | 43 | 42 | 42 |
| `iclr_primary_native_attn_cam_r16_s44` | native | `attn_cam` | 16/16 | 44 | 42 | 42 |

共同训练配置为 BF16、batch size 1、gradient accumulation 4、5 epochs、最大 33 帧、inactive expert offload 和 A800 70 GiB free-memory gate。训练时只改变 optimization seed；split 和 generation seed 固定，以便把差异主要归因于训练随机性，而不是更换测试样本或扩散噪声。

### 4.2 选择 `native_attn_cam_r16` 作为主实验的理由

1. **与研究问题直接一致。** `native` 保留 camera trajectory/Plücker 条件通路；`attn_cam` 同时允许注意力投影和 camera injector 相关线性层学习 LoRA 更新。
2. **参数效率与单卡可行性折中。** rank 16 是当前 rank sweep（4/8/16/32/64）的中心配置，能够在单 A800 约束下控制训练参数量和存储成本。
3. **适合作为统一参照单元。** matched 2×2 和 rank sweep 都复用 seed-42 的同一 canonical r16 cell，避免重复训练同一个实验。
4. **满足主结果不确定性报告。** 三个训练 seed 支持报告均值、标准差及同一测试轨迹上的配对分析。

需要注意：Primary multi-seed 只能估计该主配置自身的训练方差。若要因果归因于 camera condition，仍需完成相同 LoRA target 下的 native/drop matched controls；仅比较 `native+attn_cam` 与 `drop+attn_only` 会同时改变 pose route 和 target scope。

## 5. 数据集与运行产物状态

### 5.1 当前可下结论的状态

| 产物 | 完整性 | 是否损坏 | 是否已重新生成 | 是否需要 rebuild | 本次依据 |
|---|---|---|---|---|---|
| `work/rsroll_official/data/train_geo.h5` | **不可判定** | **不可判定** | **不可判定** | **不可判定** | 服务器目录不可访问；本地包不包含数据产物 |
| `official_splits.json` | **不可判定** | — | **不可判定** | 依赖有效 HDF5 | 服务器文件未提供 |
| `manifests/aid_test.json` | **不可判定** | — | **不可判定** | 依赖有效 split | 服务器文件未提供 |
| `reports/dataset_validation.json` | **不可判定** | — | **不可判定** | 必须重跑/核验 | 可访问 `reports/` 只有静态 Markdown，无运行期 JSON |
| `reports/semantic_consistency.json` | **不可判定** | — | **不可判定** | 必须重跑/核验 | 同上 |
| `reports/duplicate_report.json` | **不可判定** | — | **不可判定** | 必须重跑/核验 | 同上 |
| `pipeline_state/*.done` | **不可判定** | — | — | — | 服务器 checkpoint 未提供 |

这里不能根据“曾出现 addr overflow”直接推断当前 `train_geo.h5` 仍损坏，也不能根据“启动过重建”推断它已经完整。

### 5.2 正式放行标准

只有以下条件同时满足，才可写“无需重新 build”：

1. `check_dataset.py train_geo.h5 --validate --expect-n 2000` exit 0；
2. `dataset_validation.json.status == PASS`，且文件 size/mtime 与当前 HDF5 一致；
3. SemanticGate 复核 `n_fail == 0`、`bad_attr == []`、`enabled_mismatch == []`；
4. duplicate report 中 trajectory/video/scene/source duplicate 均为空；
5. official split `self_check.pass == true`，总数 2000，train/val/test source-disjoint；
6. AID test manifest 为 headline eligible、`source_split=test`、400 个固定 sample key；
7. 对应 `.done` marker 的 signature 与上述产物仍一致。

若 HDF5 打开即出现 `addr overflow`、缺 object、样本数不足，或上述任一正式 gate 失败，则当前文件不得复用。应先保留文件和日志用于诊断，再由 `server_run/build_correct.sh` 的 canonical `.partial -> validate -> atomic rename` 流程重建；不得直接调用 legacy builder。

## 6. 已知问题

### 6.1 HDF5 中断与 `addr overflow`

**用户报告：** 服务器曾出现：

```text
Unable to synchronously open object
addr overflow
```

该错误通常表示 HDF5 元数据引用的对象地址超出实际文件末尾，符合构建进程在写文件期间被 kill、磁盘写入未完成或文件被截断的情形。当前 `.partial` 和原子发布设计可以降低未完成文件被误认成正式数据的风险，但无法自动修复已经损坏的旧 `train_geo.h5`。

此外，当前 `build_correct.sh` 对“最终文件已存在但 validator 失败”的策略是硬失败，而不是自动删除或覆盖。这是审慎行为，但意味着自动启动脚本若声称“自动重新 build”，必须明确实现安全隔离旧文件、保留诊断信息和重新发布的逻辑；该部分需要检查服务器上的 `start_full_auto_pipeline_v*.sh`。

### 6.2 CLIP API 兼容问题

**用户报告：** 部分服务器 transformers/CLIP 版本在调用

```python
get_text_features(return_dict=True)
```

时抛出 `TypeError`。

**代码核验：** 可访问的 `lora_remotesense/semantic_gate.py` 仍直接调用：

```python
self._model.get_text_features(**tok, return_dict=True)
self._model.get_image_features(**px, return_dict=True)
```

虽然 `_coerce_features()` 已兼容 Tensor、`*_embeds` 和 `pooler_output` 等返回类型，但它不能处理“方法签名根本不接受 `return_dict`”的情况。因此，就本次实际读取的代码而言，此问题仍未闭环；真实 CLIP eager smoke 是当前构建前的 P0 gate。修复后还应增加覆盖两种 helper 签名的单元测试，防止仅在服务器长时构建时暴露。

### 6.3 water 类连续六次不过门

**用户报告：** water 第 11 个样本连续 6 个候选都未通过 SemanticGate。

**代码核验：** builder 确实会在连续 `max_tries=6` 后抛错；这表示 fail-closed 语义约束正在生效，而不是应当关闭 SemanticGate。当前合理方向是：

1. 保持 `RSWM_SEMGATE=1` 和本地离线 CLIP；
2. 用失败候选、逐帧 margin、source class、scene pool mass 和人工标签做 class-specific 校准；
3. 通过 `RSWM_SEMGATE_MIN_FRAC_WATER` 设置经预注册/留档的 water 阈值；
4. 在 100 样本 rehearsal 中同时检查坏样本放行率和有效 water 样本误拒率；
5. 把实际阈值、CLIP snapshot 身份、拒绝/接受分布写入报告。

不应通过 `RSWM_SEMGATE=0`、无限重试或对失败样本静默放行来绕过问题。另一方面，也不能仅为了让第 11 个样本通过而事后任意降低阈值；需要明确的校准集与人检标准。

### 6.4 文档与当前协议存在漂移

当前 `server_run/README_ICLR_FULL_PIPELINE.md` 和总控固定为 AID、N=2000、Primary-only table scope；但 `paper/paper_draft.md` 仍写有“4,000-sample official train artifact”，部分报告仍把 WHU/RESISC cross-domain 和完整 main baseline 作为当前正式范围。`reports/TEST_REPORT.md` 也记录的是旧测试数量和旧 Bash 检查状态。

这不会直接改变代码执行，但会使师兄无法判断“当前已实现范围”和“未来实验计划”的边界。正式审查包应统一 N=2000、AID Primary 主线，并把 cross-domain/外部 baseline 明确标为后续扩展或独立流水线。

### 6.5 可复现性证据仍不完整

HDF5 会保存 SemanticGate 版本、逐样本阈值与判定证据，但当前可见代码没有把 CLIP snapshot 的 revision/权重 SHA-256 固定写入最终 HDF5 或 semantic report。full-pipeline 的 CLIP checkpoint signature 主要绑定路径而非权重内容。对于论文级复现，建议后续归档 CLIP config/processor/weight hashes、transformers 版本和实际 class-specific threshold 环境。

## 7. 当前代码是否适合给师兄看

### 7.1 总体评价

**适合进行内部工程与实验协议走查；暂不适合作为“服务器最新状态已验收、实验已可一键完成”的最终签字包。**

### 7.2 优点

1. **工程完整性较高。** canonical 入口、离线资源检查、`.partial` 原子发布、逐步日志、flock、checkpoint、artifact verification、resume 和失败状态记录均已形成。
2. **实验可复现性框架清晰。** 数据、split、test manifest、训练 seed、split seed、generation seed、config hash 和运行 manifest 分层管理；test 与训练期 val 明确隔离。
3. **ICLR 实验规范意识较强。** 使用 source-disjoint split、future-only scoring、固定 400 个 test 样本、多 seed Primary、matched controls/rank sweep 预注册、分布指标最小样本数和 I3D checkpoint gate。
4. **失败路径较诚实。** 缺外部 baseline、缺 checkpoint、SemanticGate 失败、重复或 stale checkpoint 都不会用黑帧、Persistence 或旧结果静默代替。
5. **主模型边界表述合理。** 代码与 paper 草稿区分 camera-coordinate condition、二维 trajectory proxy 和显式三维几何，避免过度主张。

### 7.3 不足与审查阻塞项

1. 服务器 `20260817-full-pipeline` 源码未提供，特别是 `start_full_auto_pipeline_v*.sh` 无法审查或哈希绑定。
2. `train_geo.h5`、split、manifest、runtime reports 和 pipeline checkpoints 状态均无服务器证据。
3. 可访问 SemanticGate 仍存在 `return_dict=True` API 兼容阻塞风险。
4. water class-specific threshold 尚无可验证的冻结值和校准报告。
5. full pipeline 实际只处理三个 Primary seed，不等于完整 external baseline comparison。
6. `rswm_auto_scheduler.sh` 与 AID full pipeline 的职责衔接较间接，且 readiness scheduler 还依赖 WHU/RESISC 资源；服务器 v* 启动脚本是否正确桥接需要专门检查。
7. paper/reports 中仍有 N=4000、cross-domain 和旧测试状态等历史表述，尚未与 N=2000 Primary 主线完全统一。
8. 当前结果表均为空；没有正式训练 checkpoint、真实 GPU 资源记录或论文指标，不能做性能结论。

### 7.4 本地代码回归证据及其边界

对本文所依据的可访问副本，本地已完成以下非服务器回归：

- `python -m unittest discover -s tests -v`：67/67 通过；
- `python server_run/test_iclr_protocol.py`：26/26 通过；
- 相关 Bash 入口的 `bash -n`：通过；
- full pipeline 与 evaluation wrapper 的 `--plan`：通过，输出明确为 `GPU_ACCESSED=0`、`TRAINING_STARTED=0`。

这些结果能够证明配置解析、失败策略、数据 validator fixture、split/test 隔离、Primary 三 seed wiring、checkpoint 结构和 shell 语法，但不能证明服务器 CLIP API、真实 2000 样本 HDF5、A800 model load、训练收敛或论文指标。仓库内旧 `reports/TEST_REPORT.md` 尚未同步这组最新本地回归结果。

## 8. 下一步计划

建议按以下顺序推进，不改变已经冻结的实验设计：

1. **冻结服务器源码证据。** 从指定服务器目录导出文件清单、Git commit（若存在）及关键脚本 SHA-256，补齐所有 `start_full_auto_pipeline_v*.sh` 后重新审查本文自动化章节。
2. **先解决 CLIP P0 兼容。** 在服务器正式 Python/transformers 环境做 local-only text/image feature smoke，确认 helper 签名和输出维度，并补回归测试。
3. **完成 water 阈值校准。** 保留 SemanticGate，基于人检标注冻结 class-specific threshold；重新跑 100 样本 rehearsal，归档通过证据。
4. **判定而非猜测 HDF5。** 对当前 `train_geo.h5` 运行正式 validator。损坏则隔离旧文件并从 canonical `.partial` 流程重建；通过则禁止无意义重复构建。
5. **生成并冻结数据证据链。** 依次产生 dataset validation、semantic consistency、duplicate report、official split 和 400-sample AID test manifest。
6. **确认 scheduler 与 GPU。** 以服务器 `nvidia-smi`、scheduler status、tmux 和 preflight 输出为准，达到 `READY_FOR_TRAINING` 后再启动三 seed Primary。
7. **完成正式结果链。** 训练 seed 42/43/44，单独生成 official-test rollout，标准化结果，计算 SSIM/PSNR/LPIPS/FID/FVD proxy/I3D 及轨迹指标，再导出 paper tables。
8. **分开处理外部 baseline。** Primary 流水线通过后，再按 `main` baseline 依赖逐项补 producer/checkpoint，不得把未运行方法混入主表。
9. **统一文档。** 将 paper 和 reports 中的 4000 改为当前冻结的 2000，并同步标注 AID-only Primary 与尚未运行的扩展实验。

## 9. 服务器最终签字所需最小证据

在把本文发送给师兄作为“服务器最新代码与状态报告”之前，至少需要补充以下只读结果：

```bash
cd /nix/rsroll_25828/src/rsroll-bench-iclr-20260817-full-pipeline/rswm_server_pack

find server_run -maxdepth 1 -type f -name 'start_full_auto_pipeline_v*.sh' -print
sha256sum server_run/run_iclr_full_pipeline.sh \
  server_run/start_full_auto_pipeline_v*.sh \
  server_run/build_correct.sh \
  server_run/rswm_auto_scheduler.sh \
  lora_remotesense/build_dataset_from_benchmark.py \
  lora_remotesense/semantic_gate.py

ls -lh /nix/rsroll_25828/work/rsroll_official/data/train_geo.h5*
python server_run/check_dataset.py \
  /nix/rsroll_25828/work/rsroll_official/data/train_geo.h5 \
  --validate --expect-n 2000

ls -lh /nix/rsroll_25828/work/rsroll_official/official_splits.json \
  /nix/rsroll_25828/work/rsroll_official/manifests/aid_test.json \
  /nix/rsroll_25828/work/rsroll_official/reports/*.json \
  /nix/rsroll_25828/work/rsroll_official/pipeline_state/*.done

tmux ls
nvidia-smi
```

这些证据补齐后，本文第 3.3 节和第 5 节才能从“待服务器核验”更新为确定结论。
