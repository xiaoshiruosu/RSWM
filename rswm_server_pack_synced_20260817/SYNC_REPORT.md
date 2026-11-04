# RSWM ICLR Pipeline 本地代码同步报告

**同步时间：** 2026-08-17 21:19:24（Asia/Shanghai）  
**本地基线：** `rsroll-bench-iclr-20260816-full-pipeline/rswm_server_pack`  
**交付目录：** `rswm_server_pack_synced_20260817`  
**同步依据：** 用户提供的服务器运行记录、报错信息与已确认修改结果；本次未从服务器下载或覆盖本地项目。

## 1. 同步范围与原则

本次仅同步服务器运行记录中已经确认的工程修改，不调整模型结构、LoRA 配置或论文实验协议。现有 `run_iclr_full_pipeline.sh`、`configs/iclr_main/`、训练入口、评估入口、baseline 与 paper-table 导出结构均予以保留。

仓库中的 `RSWM_ICLR_CODE_CHANGE_REPORT.md` 是本次同步前的审查基线；其中关于“启动器缺失”和“仍含 `return_dict=True`”的描述仅代表修改前状态。同步后的最终状态以本报告为准。

冻结协议保持如下：

- 数据源：AID；正式样本数：2000；
- 轨迹窗口：128×222；输出目标：480×832；每段 steps：9；
- rollout：17/25/33 frames；overlap：`[0.0, 0.1)`；
- 数据构建唯一入口：`server_run/build_correct.sh`；
- canonical builder：`lora_remotesense/build_dataset_from_benchmark.py`；
- SemanticGate 强制启用并使用本地 CLIP snapshot；
- Primary：`native_attn_cam_r16`，seeds 42/43/44；
- 未修改模型、LoRA rank/alpha、target modules、seed 或 split 协议。

## 2. 修改文件列表

| 文件 | 类型 | 同步内容 | 修改原因 |
|---|---|---|---|
| `server_run/start_full_auto_pipeline_v20260817.sh` | 新增 | 30 秒 GPU polling、持续空闲确认、显存/利用率门、启动前即时复查、纯数字物理 GPU 校验、`CUDA_VISIBLE_DEVICES` 设置、tmux 友好日志与 full pipeline 自动启动 | 补齐服务器记录中的版本化自动入口，并修复 `GPU=3,` 一类解析污染 |
| `server_run/build_correct.sh` | 修改 | 保留 `.partial` 构建；最终 H5 或 partial 验证失败时移至 `.invalid.<UTC>.<PID>` 隔离文件；随后 canonical rebuild；完整验证后原子发布 | 防止被 kill 的半成品或损坏 HDF5 被复用 |
| `lora_remotesense/semantic_gate.py` | 修改 | 删除 `get_text_features()` 与 `get_image_features()` 的 `return_dict=True` 参数；保留 `_coerce_features()` | 兼容当前服务器 transformers/CLIP API |
| `tests/test_full_pipeline.py` | 修改 | 增加 HDF5 隔离/重建、CLIP helper 调用、四类阈值环境变量、GPU 启动器回归断言 | 将服务器问题固化为本地回归约束 |
| `SYNC_REPORT.md` | 新增 | 记录同步范围、服务器记录对应关系与验证结果 | 供重新上传与复核 |

`server_run/run_iclr_full_pipeline.sh` 与 `configs/iclr_main/` 未修改；现有 full pipeline 已包含环境、AID snapshot、CLIP snapshot、dataset、SemanticGate、duplicate、official split、Primary 三 seed、evaluation、baseline 与 paper table 的逐步日志、checkpoint 和 resume 机制。

## 3. 与服务器运行记录的对应关系

### 3.1 CLIP API TypeError

服务器记录：`CLIPModel.get_text_features()` 不接受 `return_dict`。  
本地同步：调用统一为 `get_text_features(**tok)` 与 `get_image_features(**px)`；不同 transformers 返回格式继续由已有 `_coerce_features()` 归一。

### 3.2 water 类连续拒绝

SemanticGate 未关闭。现有阈值机制继续支持：

- `RSWM_SEMGATE_MIN_FRAC`
- `RSWM_SEMGATE_MIN_FRAC_WATER`
- `RSWM_SEMGATE_MIN_FRAC_URBAN`
- `RSWM_SEMGATE_MIN_FRAC_RURAL`

构建入口和自动启动器均记录实际阈值环境。未在代码中臆造或冻结新的 water 数值；服务器应使用已校准并留档的环境值。

### 3.3 HDF5 `addr overflow` / object open failure

`build_correct.sh` 现在执行：

```text
已有 train_geo.h5
  → canonical validate
  → PASS：复用
  → FAIL：隔离为 .invalid.*
       → 检查 train_geo.h5.partial
       → 完整则原子发布；损坏则隔离并 rebuild

新建数据
  → train_geo.h5.partial
  → canonical validate
  → atomic rename 为 train_geo.h5
```

损坏文件不会被直接复用或静默覆盖，隔离文件保留诊断与恢复可能性。

### 3.4 GPU 变量尾逗号

新启动器不解析带单位或逗号的 `nvidia-smi` 展示文本，而是复用 `rswm_gpu_selector.py` 的原子结果文件。启动前要求 `PHYSICAL_GPU_ID` 严格匹配 `^[0-9]+$`，日志输出固定为 `GPU=<数字>`，再设置单卡 `CUDA_VISIBLE_DEVICES`。默认 GPU polling interval 为 30 秒；默认门限沿用现有共享 A800 选择器：utilization `<10%`、used memory `<5000 MiB`，持续确认 600 秒，均可通过显式 `RSWM_GPU_*` 环境变量调整。

## 4. 验证结果

| 检查 | 结果 |
|---|---|
| `bash -n server_run/*.sh` | PASS |
| `python -m py_compile lora_remotesense/*.py` | PASS，9 files |
| `bash server_run/run_iclr_full_pipeline.sh --plan` | PASS；输出 `GPU_ACCESSED=0`、`TRAINING_STARTED=0`、`FINAL_RESULT=PLAN_ONLY` |
| `python -m unittest tests.test_full_pipeline tests.test_gpu_scheduler` | PASS，16 tests |
| `python server_run/test_semantic_gate.py` | PASS，88 checks / 0 failures |

说明：`run_iclr_full_pipeline.sh` 是 Bash 入口，因此 plan 验证使用 Bash 解释器；不会使用 Python 直接解释 `.sh` 文件。

## 5. 关键文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `server_run/start_full_auto_pipeline_v20260817.sh` | `286FE6BF260CB40A9F418DE38FC8AD14534FEB5071AED0CBCE481EEBC204138B` |
| `server_run/run_iclr_full_pipeline.sh` | `D8073D89FB210DE240737E18ABB2CDC9E218787C1EC200B158C850A43E753342` |
| `server_run/build_correct.sh` | `9512D1C26B575190DBD3D58E04FEBCF903B5BE94150D9E5CFB14F337990D77A5` |
| `lora_remotesense/semantic_gate.py` | `1164AF63545D9E769BF7233DD76A95B3DDEAF62735F9CD6875DF1EEC3EDFB93F` |
| `lora_remotesense/build_dataset_from_benchmark.py` | `12DA38C7443F2204D7CB9B9AB2BC8038BF8CA8CC3D7F2A2C0330EE6318E2EC6F` |
| `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py` | `EE6CC6DDB9163F33E17DE8CBA0EBF5C333DB45ABA05E0CB287B394F77E8A5FF5` |

## 6. 交付边界

交付目录排除 `work/`、`*.h5`、`*.pt`、`*.pth`、`*.ckpt`、`*.bin`、checkpoint/cache 目录、`__pycache__` 与 `*.pyc`。本次只完成本地代码同步与离线回归，不声称服务器 GPU、AID/CLIP snapshot、2000 样本数据构建、LoRA 训练、evaluation 或论文结果已经完成。

上传服务器后仍应在实际服务器环境中执行资源检查、100 样本 rehearsal、正式数据验证与 GPU/tmux 状态确认，再继续 full pipeline。
