# RSWM 从头复跑 — 单卡服务器运行包(H200 / A100-80G / A800)

> **共享 A800 无人值守就绪检查**：请先读
> [`README_AUTO_SCHEDULER.md`](README_AUTO_SCHEDULER.md)。该入口只做数据完整性、
> 资源检测和一步 smoke，输出 `READY_FOR_TRAINING` 后停止，不会启动正式训练。

> **Primary validated hardware：NVIDIA A800-SXM4-80GB**（sm80、BF16、FA2→SDPA）。
> H200 141GB 是保留的 legacy/reference 路径（sm90、可选 FA3），本轮未做 H200 实机回归。

> **2026-08-14 A800 说明**：共享 4×A800-SXM4-80GB 的环境、存储、preflight
> 与三级 smoke 以 `server_run/A800_80G.md` 为准。`setup_env.sh` 现为 verify-first，
> 不再自动升级、下载或源码编译；下文旧流程中的自动安装描述仅为历史记录。

> ## ⚠️ 先读:这是**旧流程**的手册(2026-07-04)
>
> 本文件描述的是 **01–19 旧 AID 流水线**(已全部跑完,产物在 `server_run/work/`,**只读勿动**)。
> 它**不包含** 2026-07-11 之后的:修正 prompt、waypoints 持久化、方向角 180° 修正、
> Track B 轨迹指标、多种子、`--split_seed`/`--gen_seed`/`--lora_targets`、独立 WORK 安全机制。
>
> ### 👉 现在要跑代码,请看仓库根目录的 **`README_RUN.md`**(总入口)
>
> 本文件保留作为旧流程的历史记录与单机(非 PBS)运行参考。

---

> 目标:在一台租用的单卡上从零复跑全部论文实验,产出一套
> **单版本、自洽、可审计**的数字(消除历史上 r64/r128 口径、代码版本混淆等问题)。
> 本包**自包含**:上传 `rswm_server_pack.tar.gz` 解压,解压目录即工程根,照下面顺序执行。
>
> **H200(141GB)**:零改动直接跑。`common.sh` 自动探测显存并关掉 CPU offload
> (两份 DiT 常驻 GPU,比 80GB 卡快 30-50%);论文原始实验就是 H200 跑的,
> 下面时间表 ×0.65-0.75。
>
> ⚠️ **建数据只有一条正路:`bash server_run/build_correct.sh`**(canonical,产 17/25/33 帧)。
> `new/run_build_dataset.sh` 和 `new/build_from_remotesense.py` 是**错构建器**
> (96×96/steps8 → 16/24/32 帧糊图,与论文不可比),本包已把它们**默认锁死拒跑**
> (报 `[REFUSED]` 属正常,不是 bug);`verify_remotesense_h5.py` 也会对
> 16/24/32 帧数据直接判 FAIL。

## 包内结构

```
rswm_server_pack/
├── server_run/            # 本手册 + 全部运行脚本(run_all.sh 一键入口)
├── lora_remotesense/
│   ├── build_dataset_from_benchmark.py   # canonical 数据构建器(论文管道)
│   ├── build_overlap_benchmark.py        # overlap 分层构建器(拒绝采样)
│   ├── rebin_overlap_metrics.py          # overlap 重分箱
│   ├── patch_lpips.py                    # LPIPS 兜底补算
│   └── new/               # 训练/评估/推理 + wan 副本 + verify-first setup_env.sh
├── lora_training/         # camera_trajectory.py 等(构建器 import 依赖)
└── wan/                   # v5 benchmark 引擎(构建器路径锚点依赖)
```

---

## 0. 机器要求(下单前自查)

| 项 | 要求 | 原因 |
|---|---|---|
| GPU | **Primary：A800-SXM4-80GB ×1**；Legacy/reference：H200 141GB ×1 | A800 `OFFLOAD=auto→1`；H200 空闲显存足够时 auto→0 |
| 系统内存 | 当前 A800 服务器 1TiB；建议至少 128GB | offload 时专家、T5/VAE 与 optimizer state 会占用主存 |
| 磁盘 | **≥350GB** | 基模 56G + 数据 2×~50G + ckpt 5×~10G + 评估 npy ~20G |
| CUDA | 12.1(驱动 ≥530) | flash-attn 预编译 wheel 匹配;H200(sm90)同样适用 |
| Python | 3.10 | 同上 |
| 网络 | 可访问 HuggingFace(或配镜像) | 基模 + AID 数据下载 |

国内机器先设镜像:`export HF_ENDPOINT=https://hf-mirror.com`

## 1. 装环境(~10-25 分钟)

```bash
cd lora_remotesense/new
bash setup_env.sh          # 默认只核实固定版本，不修改环境
# 明确重建才允许：RSWM_ALLOW_INSTALL=1 CREATE_CONDA=1 ENV_PREFIX=/nix/rsroll_25828/envs/rswm-a800 bash setup_env.sh
```

flash-attn 固定为已验证的 2.7.3（torch2.4/cu121/Python3.10 ABI）。脚本只做导入和版本
核实；缺失或 ABI 不匹配时直接非零退出，不会临时升级或回退源码编译。

> 前置:**Python 3.10**(flash-attn 预编译 wheel 只覆盖到 cp310/311/312;3.10 最稳)、
> 能访问 pypi + github(拉 wheel)。若机器只有 CUDA runtime 没有 toolkit,预编译 wheel 拉到就不需要编译;
> 万一要编译又没 nvcc,自检会报错并给出手动装 wheel 的指引。

## 2. 下基模(~56GB)

```bash
huggingface-cli login       # 首次
huggingface-cli download robbyant/lingbot-world-base-cam \
    --local-dir ~/rswm/lingbot-world-base-cam
export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam
```

## 2b. 计算节点断网?一键离线预取

计算节点不能联网时,在**任何一台能联网的 Linux 机器**(登录节点/本地)上:

```bash
export HF_ENDPOINT=https://hf-mirror.com          # 国内
bash server_run/offline_prefetch.sh               # 基模56G + AID~3G + 评估权重0.5G
WHEELS=1 bash server_run/offline_prefetch.sh      # 节点连 pypi 都不通时,追加 wheelhouse ~4G
```

按脚本末尾打印的指引把 STAGE 目录拷进节点（基模、HF cache、torch cache 全部放在
`/nix/rsroll_25828`），节点上再：

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 写进 ~/.bashrc
# 完全无 pypi 时装环境:
export PIP_NO_INDEX=1 PIP_FIND_LINKS=/nix/rsroll_25828/wheelhouse
cp /nix/rsroll_25828/wheelhouse/flash_attn-*.whl lora_remotesense/new/    # setup_env 优先用本地 wheel
```

之后全流程(建库/训练/评估/P0-P5)零联网。断网节点**别跑 P6**(要在线拉 gated SVD ~9.5GB
和独立环境;真要跑,先在联网机把 `setup_mvsplat360_autodl.sh` 的产物整目录拷过来)。

## 2c. PBS 集群?用 qsub,一实验一作业

集群是 PBS(登录节点提交 / 计算节点执行 / 计算节点常断网)时,**不要**在登录节点直接
`bash run_all.sh`(会在无 GPU 的登录节点跑训练)。用 `server_run/pbs/` —— 全流程拆成
**19 个独立作业**,每个 `.pbs` = 一个实验:

```bash
# 本节仅复现 01-19 旧流程;qsub_all 默认拒绝,必须显式确认 legacy。
# 登录节点(有网):先按 2b 预取 + 装环境;改各 .pbs 的 #PBS 头 + _pbs_env.sh 的 conda 路径
export CHECKPOINT_DIR=$HOME/RSWM/lingbot-world-base-cam
export CONDA_ENV=rswm                              # 或 /home/user/RSWM/envs/rswm
DRYRUN=1 FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
```

也可 `qsub server_run/pbs/01_build_data.pbs` 逐个手动提交。训练作业超时被杀后重投会
**按 epoch 自动续训**(`train_lora` 检测 `checkpoint-epoch-*` → `--resume_from`)。
作业清单、依赖图、Torque/PBS Pro 指令差异见 `server_run/pbs/README_PBS.md`。
下面第 3 节的 tmux 方式是给**单机(非集群)**用的。

## 3. 跑(单机非集群;tmux 里)

```bash
tmux new -s rswm
export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam
export HF_ENDPOINT=https://hf-mirror.com        # 国内机器
cd <解压后的 rswm_server_pack/>                  # 工程根
bash server_run/run_all.sh                       # P0 smoke → P1..P5 全自动
```

**断点续跑**:每个子步骤完成会在 `server_run/work/stamps/` 打点;机器重启/中断后
重跑 `run_all.sh` 自动跳过已完成段。想强制重跑某段,删对应 `stamps/xxx.done`。

也可单独跑某阶段:`bash server_run/p1_main.sh` 等。

## 4. 各阶段 & 预算(下表为 A800-80G 口径;**H200 ×0.65-0.75**,即合计 ~85-110h ≈ 4-5 天)

| 阶段 | 内容 | GPU 时 | 产出 |
|---|---|---:|---|
| P0 | smoke:20样本/1ep/评4 | 1-2h | 端到端验证,**不通过别往下跑** |
| P1 | build 500 → 训 r128(语义) → 评 base+LoRA(全val≈100) → 对齐 | 32-40h | **主表 headline** |
| P2 | 训 r16/32/64 + 各评@50(baseline 复用 P1) | 50-60h | rank 饱和曲线 |
| P3 | 训 r128(几何 Plücker) + 评全val | 21-26h | **方向路由 decisive analysis** |
| P4 | overlap 4 档 build+convert+评(LoRA+base 各@50)+re-bin | 15-20h | overlap 表(baseline 4 档全補齐) |
| P5 | 定性 mp4(4 场景×4,带 GT 对照) | ~2h | 定性图素材 |
| P6 | (可选)MVSplat360 外部 baseline @20 样本 | 4-8h | **domain-gap 对比行**(审稿人必问的外部 baseline) |
| | **合计(P0-P5)** | **~125-150h ≈ 6 天纯 GPU** | |

> **P6 必须独立 conda env**(它装 torch 2.2.2,会毁掉主环境)——`p6_mvsplat.sh` 内置守卫,
> 检测到主环境直接拒跑;结束时强制核查 `generation_success >= 80%`,防黑帧假指标。
> SVD 权重是 gated:先在 HF 网页 accept license 再 `huggingface-cli login`。

结束后:`python server_run/collect_results.py` → `work/summary.md` 一张总表。

## 5. 本包锁定的配置(全部按仓库真实代码,勿改)

| 决定 | 取值 | 代码依据 |
|---|---|---|
| 构建器 | `lora_remotesense/build_dataset_from_benchmark.py --dataset-tier aid` | 产出论文数据 train_hd_500 的 canonical 管道(AID/128×222/steps9/480×832) |
| 训练器 | `lora_remotesense/new/train.py` | **已核实**含逐样本 low/high 路由修复(:587-595)+ 80G offload 三件套默认 ON |
| 评估器 | `new/evaluate.py` / `new/evaluate_baseline.py` | `metrics.py` 与产出论文数字的版本 **md5 相同**;new 版仅多 CPU-offload 开关,不动数值 |
| MAX_FRAMES | **33** | 正式训练、推理与 smoke 的统一默认值 |
| epochs / lr / seed | 5 / 1e-4 / 42 | 各 ablation pbs 一致 |
| grad accum | **4** | `ablation_rank128.pbs`(产出 headline 的作业) |
| LoRA α | **α=rank**(16/16…128/128) | `ablation_rank{16,32,128}.pbs` 实际取值;⚠️ 论文文本写 α=64 需改稿 |
| 语义通道 | 位姿中和 identity(`neutralize_poses.py`) | 当前构建器一律注入方向位姿(BUG-4 修复),中和后 = 原主实验 regime |
| 几何通道 | 构建器原生 waypoint Plücker 位姿 | 与语义版**同样本同 prompt**,位姿是唯一变量 → 干净单变量对比 |
| overlap eval 位姿 | 保持转换器原样(不中和) | 与产出旧 overlap 数字的 5/3-5/4 作业一致 |
| ❌ 禁用 | `new/build_from_remotesense.py` | 96×96/steps8/eurosat,**不是**论文管道,用了数据不可比 |

## 6. 每阶段结束的人工核对点

- **P0**:`work/smoke/eval_aligned/comparison.json` 存在;训练无 OOM(峰值 <75GB);npy 非全黑。
- **P1**:`work/aligned_main/comparison.json` —— LoRA FVD 应显著低于 baseline(历史参考:79→31-37 区间);
  Angular_Error 两者都 ~85-95°(≈随机,正常)。
  **另跑一次 `python server_run/check_aid_pools.py`**(泄漏验证):PASS → 论文 disjoint-source claim 成立,
  删除 tex 里 splits 段的 `% TODO`;FAIL → 按脚本提示处理。
- **P2**:r16 的 FVD 应与 r128 接近(饱和结论)。
- **P3**:几何通道 Angular_Error 若仍 ~90° → 论文走 NEGATIVE 分支(结构性局限);若 <60° → 方向 claim 回正文(改稿框架)。
- **P4**:`overlap/rebin_*` 4 个目录齐;FVD 随 overlap 上升的趋势是否复现;**这次有完整 baseline 4 档**。
- 全程:`nvidia-smi` 训练期 55-65GB 正常;>75GB 检查是否误改 batch。

## 7. 故障速查

| 现象 | 处理 |
|---|---|
| HF 下载 401/超时 | `huggingface-cli login`;国内 `export HF_ENDPOINT=https://hf-mirror.com` |
| AID 数据集拉不动 | 手动下好后 `export LOCAL_AID_DIR=/path/aid/images`(P4 自动透传) |
| 训练 OOM | 确认 batch=1、offload 三件套没被关;`PYTORCH_CUDA_ALLOC_CONF` 已由脚本设置 |
| CPU 内存爆 | 机器内存 <100G;换机或减 num_workers(已默认 0) |
| bitsandbytes 缺失 | 自动 fallback torch.AdamW,不影响跑通;`pip install bitsandbytes>=0.43` 消警告 |
| LPIPS/FID 首跑要下权重 | 联网即可;离线机先 `python -c "import lpips; lpips.LPIPS(net='alex')"` 预缓存 |
| 断电/中断 | 重跑 `run_all.sh`,stamps 自动续 |

## 8. 跑完带走什么(rsync 回本地)

```
server_run/work/summary.md + summary.json     # 总表
server_run/work/aligned_main/                 # 主表对齐数字
server_run/work/eval_*/eval_results.json      # 各配置原始指标
server_run/work/overlap/rebin_*/              # overlap 重分箱
server_run/work/*/train_val_split.json        # split 存档(审计用)
server_run/work/quals/                        # 定性 mp4
server_run/work/train_*/best_checkpoint/      # (可选)LoRA 权重,各 ~10G
server_run/work/logs/                         # 全部日志
```
评估 npy(generated/gt)体积大,至少带走 `eval_r128_sem` 和 `eval_base` 两套(拼定性图用)。

## 9. 跑完后还差的两件小事(不在本包,出数后补)

- **多噪声种子方差**(审稿人要 error bar):同一 ckpt 换 3 个生成种子跑 ~20 样本报 std,
  需要一个小 glue 脚本(evaluate.py 无 seed 参数)——出数后向 Claude 要 `p7_variance.sh`。
- **标准 I3D FVD 补算**(堵"你的 FVD 是 r3d_18 代理"的口):在已有 generated/gt npy 上
  离线重算,零推理成本——同上,出数后要 `compute_i3d_fvd.py`。
