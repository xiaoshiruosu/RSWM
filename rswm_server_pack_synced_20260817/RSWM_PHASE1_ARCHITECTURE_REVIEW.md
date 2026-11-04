# RSWM-Bench Phase 1 当前架构审查

> 文档状态：Phase 1 历史设计审查记录；后续设计已于 2026-08-23 获批  
> 审查日期：2026-08-22  
> 审查时规范基线：`RSWM_BENCH_PROTOCOL_V1.md`，SHA-256 `0EFCA1BECBC2A21148E5C7022D2DAFB0D7FD8917AADED6843019F76C3B66F5AE`  
> 冻结批准记录写入后的规范副本：SHA-256 `F9C3332C917C1936665A920364BDFAA8886FB822AA040213519F36F49E01D301`；科学定义未改，仅更新状态、批准记录与决策表状态  
> 变更边界：本阶段未修改模型、训练、数据、配置、评测或 baseline 代码；未启动 GPU、训练、数据构建或 evaluation。

## 1. 审查结论

当前仓库已经具备一个较完整的 **RSWM/RS-Roll 实验执行系统**：它能够构建受控遥感 rollout 数据，进行 LoRA 训练，生成 17/25/33 帧视频，执行 future-only 指标与轨迹指标，标准化 baseline 结果，并导出论文表格。现有正式 ICLR 主链路具有数据门禁、manifest、断点恢复、结果身份和失败退出机制，不能被 Phase 1 重写或重新解释。

当前仓库尚未形成清晰的 **benchmark 公共抽象层**。任务定义、模型调用、HDF5 读取、轨迹语义、评测聚合和 baseline 能力声明分散在 `lora_remotesense/new/`、`server_run/`、`wan/remotesensebenchmark/` 与文档中。尤其是：

1. `lora_remotesense/new/backbone.py` 已经提供推理级 `VideoBackbone`、`WanBackbone`、能力声明和输出合同，不能再平行复制一套等价接口；
2. `lora_remotesense/new/evaluate.py` 的默认路径仍直接调用 Wan 的 `load_lora_model()` / `generate_video()`，只有显式选择非 `wan-inline` 时才经过 backbone 注册表；
3. 训练数据加载器是 PyTorch 训练专用接口，不适合作为 benchmark 数据 API；
4. manifest v2 已冻结样本、checksum、轨迹和 overlap 证据，但尚未表达 V1 的 task、track、data domain 与 capability partition；
5. 现有 metrics 和 baseline 框架可复用，但公共接口、未来帧裁剪、能力分榜和 hidden-GT 边界尚未集中管理。

因此，Phase 1 的正确方向是增加一层 **旁路、只包装、可回退** 的 benchmark abstraction：以 `WorldModel` 作为面向 task 的高层门面，以现有 `VideoBackbone` 作为 Wan 推理桥，以 `LegacyH5Adapter` 只读映射旧 HDF5/manifest v2，以 task 与 metric registry 组织冻结协议。初始迁移不改变旧入口默认行为，也不改变任何已有 checkpoint、结果或论文实验身份。

## 2. 证据边界

本报告按以下三种状态陈述结论：

| 状态 | 含义 | 本次使用方式 |
|---|---|---|
| 已观察 | 已从当前规范副本的源码、配置或项目文档直接确认 | 用于描述现状和接口约束 |
| 合理推断 | 由多处静态证据推导，但未运行真实模型或正式数据验证 | 明确标注为迁移设计判断 |
| 尚未验证 | 需要服务器文件、GPU 权重、正式 HDF5 或真实运行才能确认 | 不写成“已通过” |

本次完成了顶层目录清单、关键入口与调用链审读，并重点检查了模型、训练、数据构建、HDF5/manifest、轨迹、评测、baseline、流水线、测试和论文导出模块。未读取或修改 `work/` 中历史产物内容；它们按 `AGENTS.md` 作为只读历史证据保留。未将嵌套的同名 `rswm_server_pack_synced_20260817/` 目录当作当前规范副本。

### 2.1 版本身份

- 已观察：`AGENTS.md` 指定顶层目录为规范副本，嵌套同名目录是冗余副本，不应删除或混用。
- 已观察：`SYNC_REPORT.md` 表明当前包是“本地基线 + 用户提供的服务器运行记录”同步结果，不是本次直接从服务器下载得到。
- 已观察：工作区 Git 仅把整个顶层项目显示为未跟踪目录，没有可用于定位版本的提交历史或远程地址。
- 结论：本次以 `AGENTS.md`、`SYNC_REPORT.md`、文件清单和关键文件 SHA-256 作为版本证据，而不以目录修改时间判断。

### 2.2 审查范围规模

排除 `work/`、嵌套冗余副本与缓存后，当前活动包约 474 个文件，主要分布为：`server_run/` 162、`lora_remotesense/` 132、`wan/` 85、`configs/` 25、`reports/` 14、`tests/` 9，其余为文档、脚本和论文材料。仓库根目录当前不存在 `models/`、`benchmark/`、`evaluation/`、`baselines/` 四个公共包。

## 3. 当前模块结构

```text
配置 / 自动化入口
  configs/iclr_main/*
  server_run/run_iclr_full_pipeline.sh
  server_run/iclr_a800/*
            |
            +--> 数据构建与验证
            |      server_run/build_correct.sh
            |      lora_remotesense/build_dataset_from_benchmark.py
            |      wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py
            |      server_run/check_dataset.py
            |      server_run/make_official_splits.py
            |      server_run/baselines/build_benchmark_manifest.py
            |
            +--> 训练
            |      server_run/rswm_train_runner.sh
            |      lora_remotesense/new/train.py
            |      lora_remotesense/new/dataset_loader.py
            |      wan/* + LoRA/PEFT
            |
            +--> 推理与评测
            |      lora_remotesense/new/evaluate.py
            |      lora_remotesense/new/inference.py
            |      lora_remotesense/new/backbone.py（可选路径）
            |      lora_remotesense/new/metrics.py
            |      lora_remotesense/new/trajectory_metrics.py
            |      server_run/run_trajectory_eval.py
            |
            +--> Baseline 与标准化
            |      server_run/baselines/*
            |      server_run/evaluate_baseline_results.py
            |
            +--> 论文产物
                   tools/export_paper_tables.py
                   paper/* / reports/*
```

这一结构以实验执行为中心：上层 shell/PBS runner 直接组织数据、模型和评测脚本；公共 benchmark 概念没有独立命名空间。

## 4. 分层归类

### 4.1 Model layer

| 位置 | 当前职责 | 审查判断 |
|---|---|---|
| `wan/` | Wan I2V/Video DiT、VAE、T5、camera utils、采样器 | 核心模型实现；Phase 1 必须保持不变 |
| `lora_remotesense/new/wan/` | 与现有运行包兼容的 Wan 副本 | 兼容性实现；不得在抽象迁移时顺手合并 |
| `lora_remotesense/new/inference.py` | 权重加载、LoRA 合并、条件构造、采样、VAE decode、输出 resize | 现有 Wan 推理真源；应被 adapter 调用而非重写 |
| `lora_remotesense/new/backbone.py` | `PoseCond`、`BackboneCapabilities`、`VideoBackbone`、`WanBackbone`、注册表和 CPU 诊断 backbone | 已有推理级模型协议；新 `WorldModel` 必须组合而不是复制它 |
| `docs/ADDING_A_BACKBONE.md` | 现有 backbone 接入合同与能力说明 | 新设计的兼容依据 |

已观察：`VideoBackbone.generate()` 已约束输入首帧、prompt、帧数、尺寸、可选 pose 与 seed，输出为 `(T,H,W,3) uint8`；`WanBackbone` 是薄适配器。已观察：相关 CPU/mock 协议测试存在，但文件明确说明真实 Wan 权重的 GPU 端到端尚未由该测试验证。

### 4.2 Training layer

| 位置 | 当前职责 | 审查判断 |
|---|---|---|
| `lora_remotesense/new/train.py` | 双专家 Wan LoRA 训练、offload、DDP 防护、参数审计、checkpoint/resume | 正式训练核心；保持不变 |
| `lora_remotesense/new/dataset_loader.py` | HDF5 到 PyTorch tensor 的训练加载、归一化、split 与截断保护 | 训练专用，不作为 benchmark dataset API |
| `lora_training/` | LoRA/轨迹相关旧兼容实现 | 保持兼容，不在 Phase 1 迁移 |
| `configs/iclr_main/` | 主实验、matched、rank sweep、target sweep 与实验计划 | 冻结 ICLR 参数和身份 |
| `server_run/rswm_train_runner.sh`、`server_run/iclr_a800/*` | 配置解析、训练/评测调度与安全门 | 既有实验入口；初始迁移不改默认行为 |

已观察：正式 primary 为 `native_attn_cam_r16`，训练种子 42/43/44；batch size 1、gradient accumulation 4、33 最大帧、固定 split seed 42。Phase 1 不对这些参数提出优化或变更。

### 4.3 Benchmark/data layer

| 位置 | 当前职责 | 审查判断 |
|---|---|---|
| `server_run/build_correct.sh` | 唯一 canonical、offline、SemanticGate 强制的数据构建入口；partial 验证后原子发布 | 正式数据入口，保持不变 |
| `lora_remotesense/build_dataset_from_benchmark.py` | AID 等 tier 注入、v5 benchmark 转换、HDF5 写入、prompt/pose/provenance | 数据核心，保持不变 |
| `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py` | 场景、连续轨迹、方向 embedding、overlap 约束与视频窗口生成 | trajectory/data generation core，保持不变 |
| `server_run/check_dataset.py` | HDF5 协议与结构门 | 复用验证，不重写 |
| `server_run/make_official_splits.py` | source-disjoint official split | 冻结划分逻辑 |
| `server_run/baselines/build_benchmark_manifest.py` | 从 HDF5 + split 生成 manifest v2、checksum、overlap profile、difficulty | 旧数据与新抽象之间的重要桥梁 |
| `server_run/baselines/protocol.py` | manifest/result/card schema 与验证器 | Phase 1 兼容真源，不能静默改 schema |

HDF5 的已观察核心结构为：

```text
/metadata
  attrs: total_instances, target_size, pose_window_size,
         pose_injection, prompt_schema, overlap bounds, provenance...
/data/<sample_key>
  image                 condition image
  video_frames          rollout，含 t=0 condition frame
  intrinsics
  poses
  trajectory            逐帧窗口中心（存在于 canonical 新数据）
  waypoints
  segment_angles
  full_scene            评测/诊断需要时存在
  attrs: prompt, num_frames, overlap_ratio, source_*, pose_source...
```

manifest v2 已冻结每个样本的 `sample_id`、source、trajectory ID/checksum、frame count、condition/GT checksum、condition-to-frame overlap profile、adjacent overlap profile、endpoint/mean/min/max overlap、difficulty 与总 manifest checksum。它目前把 `task` 写为自然语言，且文档级 `source_dataset/source_split` 会约束所有样本一致；尚未编码 V1 的 task/track/domain/capability partition。

### 4.4 Evaluation layer

| 位置 | 当前职责 | 审查判断 |
|---|---|---|
| `lora_remotesense/new/evaluate.py` | HDF5 取样、推理、保存 generated/GT/meta、结果身份、指标聚合 | 当前 experiment-specific 总入口，耦合较重 |
| `lora_remotesense/new/metrics.py` | SSIM、PSNR、LPIPS、FID、R3D proxy、光流方向等 | 数值实现复用，不重写 |
| `lora_remotesense/new/trajectory_metrics.py` | 轨迹恢复与方向、分段、turn、waypoint/path 指标 | 数值实现复用，不重写 |
| `server_run/run_trajectory_eval.py` | Track B 聚合、timeline 对齐、commanded 语义与标定门 | 现有结果解释的重要真源 |
| `server_run/evaluate_baseline_results.py` | 标准化 baseline 结果、共享指标与聚合 | 新 runner 可通过 wrapper 调用 |
| `lora_remotesense/new/submission_contract.py` | ID、shape、失败率、future-only 等提交约束 | 新输出验证应委托或镜像其合同，不分叉 |

已观察：`compute_all_metrics()` 已执行 future-only 评分并严格区分标准 I3D-FVD 与 `FVD_R3D_PROXY`；`tools/export_paper_tables.py` 对缺失值保留空值，不以 0 填充，也不构造 composite score。合理推断：若新 task、metric wrapper 和旧聚合同时裁去 t=0，会造成“双重 future-only 裁剪”，因此必须把裁剪责任集中到一个 runner boundary。

### 4.5 Baseline layer

| 位置 | 当前职责 | 审查判断 |
|---|---|---|
| `server_run/baselines/registry.py` | `BaselineAdapter`、注册表、输出 record 验证、诊断/外部适配器 | 可由新公共 baseline 层包装 |
| `server_run/baselines/cards.py` + `cards/*.json` | card v1 加载与方法披露 | 字段不足以表达 V1 capability partition 与 license 合同 |
| `server_run/baselines/result_adapters.py` | 将已有 rollout 目录标准化为 baseline 输出 | 重要兼容桥 |
| `server_run/baselines/external_stubs.py` | 外部方法 stub/桥接状态 | 不等于外部 baseline 已实现或已复现 |
| `server_run/baselines/baseline_status.json` | 实现/结果可用状态 | 当前所有正式结果可用性必须按文件实际状态解释 |

已观察：现有 registry 的 condition 类型主要是 `text_only`、`text_plucker`、`none`，方向解析主要覆盖四方向词；这不能直接代表 V1 `net_continuous` 或多视图 capability。已观察：card v1 有 checkpoint/adaptation/trajectory support 等描述，但没有可验证的 license、task/track 列表、输出 schema 和完整能力分区字段。

### 4.6 Orchestration 与 artifact layer

`server_run/run_iclr_full_pipeline.sh` 将环境/AID/CLIP 检查、canonical 数据获取、HDF5 结构检查、SemanticGate 重验、duplicate gate、official split、manifest、三个 primary seed、rollout 标准化、正式 evaluation 和论文表格导出串成带签名 checkpoint 的流水线。`tools/export_paper_tables.py` 消费标准化结果并输出 main、overlap、trajectory、rank、target、cross-dataset 与 efficiency 表。

这条链路是现有 ICLR 实验的可复现资产。新 benchmark runner 应作为独立入口逐步验证，不能直接替换该脚本或重新认证已有产物。

## 5. 当前端到端数据流

```text
AID local snapshot + local CLIP
        |
        v
build_correct.sh
        |
        v
canonical builder -> train_geo.h5.partial -> validate -> atomic rename
        |
        +--> check_dataset / SemanticGate recheck / duplicate audit
        |
        +--> official_splits.json
        |
        +--> benchmark manifest v2 (sample/checksum/overlap freeze)
        |
        v
configs/iclr_main + rswm_train_runner
        |
        v
train.py -> LoRA checkpoint + run identity
        |
        v
evaluate.py -> inference.py/Wan -> generated + gt + meta
        |
        +--> metrics.py (future-only visual/distribution/flow)
        +--> trajectory_metrics.py / run_trajectory_eval.py
        |
        v
server_run/baselines 标准化 -> aggregate_metrics.json + sample_metrics.jsonl
        |
        v
export_paper_tables.py -> CSV + export_manifest.json
```

该流中 model、task 和 evaluator 之间没有稳定的公共对象：数据以 dict/HDF5 group 传递，模型输出在 Wan tensor 和 canonical uint8 video 之间转换，task 语义依赖 prompt/environment/目录元数据，导致接入外部 world model 时容易复制逻辑或改变口径。

## 6. 必须保持不变的代码与事实

### 6.1 代码冻结区

Phase 1 及后续初始实现不得修改：

- `wan/**`；
- `lora_remotesense/new/wan/**`；
- `lora_remotesense/new/train.py`；
- `lora_remotesense/new/inference.py` 的模型结构、采样和 checkpoint 解释；
- `lora_remotesense/new/dataset_loader.py` 的正式训练行为；
- `lora_training/**`；
- `lora_remotesense/build_dataset_from_benchmark.py`；
- `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py`；
- `server_run/build_correct.sh` 与正式数据门；
- `server_run/make_official_splits.py`；
- `configs/iclr_main/**`；
- 已有 checkpoint、manifest、HDF5、实验输出和 paper tables；
- 现有数值 metric 与 trajectory metric 的定义。

### 6.2 科学协议冻结区

- 现有 ICLR AID 主实验仍为 N=2000、window 128×222、target 480×832、steps 9、T=17/25/33、endpoint overlap `[0,0.1)`、SemanticGate enabled；
- primary 仍为 `native_attn_cam_r16`，训练种子 42/43/44；
- 不把受控静态场景 rollout 表述为真实时间预测、6DoF navigation、3D 重建或真实地理定位；
- V1 不把真实世界数据设为首发强制条件；
- V1 不定义 RSWM Score 为主指标或单一排行榜；
- t=0 condition frame 不进入生成质量评分；
- camera direction 与 content direction 保持 180° 差，不能混写；
- endpoint、mean future 与 adjacent overlap 保持不同语义。

## 7. 需要新增的 abstraction

| 新抽象 | 解决的问题 | 与现有代码关系 |
|---|---|---|
| `WorldModel` | benchmark 不直接依赖 Wan；统一请求、能力与 rollout 合同 | 高层门面，内部组合现有 `VideoBackbone` |
| `WanWorldModelAdapter` | 将 V1 track/task 输入转成现有 Wan prompt/pose 调用 | 薄包装 `WanBackbone`，不复制采样 |
| `BenchmarkSample` / `GroundTruthBundle` | 将公开模型输入与 organizer-only GT 分离 | 只读映射 HDF5/manifest v2 |
| `BaseTask` + T1/T2/T3 | 将任务选择、输入裁剪、输出验证和评测调度从模型移出 | task 不导入 Wan 或训练代码 |
| `LegacyH5Adapter` | 旧 HDF5 无重建接入统一 sample | 复用 manifest v2 验证/checksum，官方模式禁止插值 |
| trajectory domain types + legacy adapter | 集中方向、waypoint、overlap level 语义 | 调用现有轨迹实现，不改生成核心 |
| `Metric` / `EvaluationRunner` | 统一 future-only、invalid、scope、版本与聚合 | wrapper 调用现有 metric 真源 |
| `BaselineCapabilities` / card v2 overlay | 按 native/adapted/multiview 和实际输入能力分榜 | 包装现有 registry/card v1，不伪造外部结果 |
| benchmark spec sidecar | 在不改 manifest v2 的前提下声明 task/track/domain/partition | 引用 manifest checksum，不重写旧 manifest |

## 8. 主要迁移风险

### P0：会改变结论或破坏公平性的风险

1. **重复模型协议。** 若新 `WorldModel` 平行实现 `VideoBackbone` 的输出与 pose 合同，两套接口会漂移。缓解：`WorldModel` 只承担 benchmark 语义，Wan adapter 必须组合现有 `WanBackbone`。
2. **Track A 信息泄漏。** HDF5 同时含 trajectory、poses、intrinsics 和 GT；若 sample 对象原样交给模型，`A_text` 不再是 text-commanded。缓解：公开 `ModelInput` 与 organizer-only `GroundTruthBundle` 分离，`BaseTask.prepare_input()` 白名单投影字段。
3. **future-only 双重裁剪。** 旧 metrics 已裁掉 t=0；新 runner 再裁一次会从 t=2 开始评分。缓解：定义唯一 `EvaluationView`，wrapper 声明旧函数是否自行裁剪，并做 poison-frame 已知答案测试。
4. **方向 180° 翻转。** 轨迹的 camera bearing 与像素内容光流相反。缓解：类型中同时记录 `camera_bearing_deg` 与 `content_bearing_deg`，禁止无命名 `angle`。
5. **overlap 口径混用。** T3 只能以 condition-to-endpoint overlap 分档，不能以 mean/adjacent 替代。缓解：使用结构化 `OverlapProfile`，level 由唯一函数根据 endpoint 值计算。
6. **旧结果被新规范追溯重标。** 当前 `<10%` ICLR 数据只能作为 legacy-compatible T3/L4 artifact，不能据此声称已完成完整 T2/T3 ladder。缓解：sidecar 明确 `provenance_status` 与 `coverage_status`，默认不提升既有 result status。
7. **分布指标接口误用。** FID/FVD 不是单样本可解释标量，简单的 `Metric.compute(sample)` 会产生错误聚合。缓解：Metric 声明 `scope=sample|dataset`，dataset metric 使用 accumulator/finalize。
8. **manifest v2 被原地扩写。** 这会使 checksum 和旧结果身份失效。缓解：V1 task/track 先放入带自身 checksum 的 sidecar；新 schema 需单独版本化。

### P1：高概率工程风险

1. **默认 Wan 路径与 registry 路径不等价。** 现有 `evaluate.py` 默认仍是 `wan-inline`。迁移前必须比较 prompt、pose、seed、帧数、resize、dtype、首帧与元数据。
2. **HDF5 样本顺序。** split 使用索引，而 manifest 冻结 sample key 顺序。adapter 必须以 manifest `sample_keys` 为权威，并验证与 split/HDF5 对应，不能自行重新排序。
3. **旧数据缺字段。** 非 canonical HDF5 可能缺 trajectory/full_scene/pose window。official 模式应 fail closed；diagnostic 模式可以显式标记 `NOT_AVAILABLE`，不得静默插值后进入 headline。
4. **baseline capability 过度概括。** 现有四方向词与 `text_only/text_plucker` 不足以代表 continuous bearing、full camera 或 multiview。必须通过 capability validator 分榜。
5. **GT 泄漏到外部 adapter。** `full_scene`、GT frames、source path 等不能进入模型调用或提交包。
6. **重型依赖与 import side effect。** 公共 core types 应不依赖 torch/h5py/Wan；这些依赖留在具体 adapter 中，确保协议测试可在 CPU 最小环境运行。
7. **嵌套副本导入错误。** runner 必须解析并记录规范项目根，禁止因 `sys.path` 顺序加载嵌套冗余副本。

### P2：维护与发布风险

- card v1 缺少可验证 license、checkpoint hash、task/track 与输出 schema；
- manifest v2 的全局 source 字段不适合未来混合数据域；
- output-level semantic metric 当前没有冻结、校准的实现，不能提前成为 headline；
- 文档中存在历史尺寸、样本量或 profile 叙述，不能用它们覆盖当前代码与 frozen protocol；
- 当前缺少可用 Git commit/remote 身份，正式发布前应建立不可变 release tag 或 source archive hash。

## 9. 当前测试与验证证据

### 9.1 已观察到的测试资产

- `server_run/test_backbone_protocol.py`：CPU/Mock 检查 `VideoBackbone`、输出合同、pose mode、方向与 Wan 参数转发；
- `tests/test_baseline_platform.py`：manifest/baseline/export 的 fixture 测试；
- `tests/test_full_pipeline.py`：canonical HDF5 validator 与流水线 wiring；
- `server_run/test_metrics_known_answer.py`、`server_run/validate_direction_metric.py`：指标已知答案与方向校准；
- data tier、scheduler、experiment runner 和 A800 入口相关测试。

### 9.2 本次未执行

按用户限制，本次没有运行上述测试、compile、evaluation、模型加载、HDF5 构建或任何 GPU 作业。因此本报告只证明“设计基于当前代码静态证据”，不证明新 abstraction 已实现或端到端通过。

## 10. Phase 1 审查结论与进入实现的门

架构层面可以进入抽象实现，但必须满足以下前置确认：

1. 接受 `WorldModel` 作为高层 benchmark 门面、现有 `VideoBackbone` 作为推理桥，而不是建立第二套 backbone；
2. 接受 manifest v2 保持不变，task/track/domain 先以 checksum sidecar 表达；
3. 接受 Track A 模型输入与 GT/hidden trajectory 物理分离；
4. 接受新 runner 初期为 opt-in，不替换 `run_iclr_full_pipeline.sh` 或 `wan-inline` 默认路径；
5. 接受所有 metric 只包装，且 future-only 由一个明确边界负责；
6. 接受 output-level semantic metric 暂为扩展位，不进入 V1 headline；
7. 在实现前将 `RSWM_BENCH_PROTOCOL_V1.md` 内“待确认”决策表与本次“协议已冻结”的执行指令形成一致、可审计的批准记录。

在这些门未确认前，下一阶段只应停留在本文与 `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md` 的设计审阅，不应修改代码或启动实验。
