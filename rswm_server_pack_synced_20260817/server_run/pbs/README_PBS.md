# 在 PBS 集群上跑 RSWM(一实验一作业)

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**；
> 本文 PBS/H200 队列属于 legacy/reference，sm90/FA3 路径保留但本轮未实机回归。

> **当前唯一推荐入口**:`qsub_corrected_rerun.sh`(20–33 corrected 链)。
> 下方 01–19 作业表属于历史 legacy 链;`qsub_all.sh` 默认 `[REFUSED]`,
> 仅允许用 `FORCE_LEGACY=1` 调试或复现旧结果。

以下先保留 **19 个 legacy 独立作业**的依赖资料。每个 `.pbs` = 一个实验,
适用于**登录节点提交 / 计算节点执行 / 计算节点常断网**的 PBS 集群。

## 作业清单与依赖

| # | 文件 | 实验 | 依赖 | 墙钟 |
|--:|---|---|---|---|
| 01 | `01_build_data.pbs` | 建 canonical 数据(GEO_H5+SEM_H5) | — | 6h |
| 02 | `02_train_r128_sem.pbs` | 主训练 r128 语义 | 01 | 48h |
| 03 | `03_eval_base.pbs` | baseline 全 val 评估 | 02 | 12h |
| 04 | `04_eval_r128_sem.pbs` | LoRA r128 全 val 评估 | 02 | 12h |
| 05/06/07 | `0{5,6,7}_train_r{16,32,64}.pbs` | rank 消融训练 | 01 | 48h |
| 08/09/10 | `0{8,9}_/10_eval_r{16,32,64}.pbs` | rank 评估@50 | 05/06/07 | 6h |
| 11 | `11_train_geo.pbs` | 几何通道训练 | 01 | 48h |
| 12 | `12_eval_geo.pbs` | 几何通道全 val 评估 | 11 | 12h |
| 13-16 | `1{3,4,5,6}_overlap_{trivial,easy,medium,hard}.pbs` | overlap 每档 build+eval | 02 | 8h |
| 17 | `17_overlap_rebin.pbs` | overlap 重分箱 | 13-16 | 3h |
| 18 | `18_quals.pbs` | 定性 mp4 | 02 | 4h |
| 19 | `19_collect.pbs` | 对齐对比 + 总表 | 03,04,08,09,10,12,17 | 2h |

> baseline 只在 03 评一次,rank/geo/overlap 全复用它。`03` 需要 `02` 产出的
> `train_val_split.json`,故 `03/04` 依赖 `02`(不是 `01`)。

## 第二数据集(泛化到 RESISC45,堵"单数据集特例"质疑)

**不用重写作业** —— 传两个环境变量复用现有的 6 个作业(01/02/03/04/11/12),
用 `DATASET_TIER=resisc45` 换数据源、用 `WORK=…/work_resisc45` 把产物隔离到独立目录
(否则会覆盖 AID 的结果):

```bash
# 登录节点先预取第二数据集(计算节点断网):
TIER2=1 bash server_run/offline_prefetch.sh          # 拉 RESISC45 + PatternNet

# 提交 RESISC45 的最小集(1 build + 2 train + 3 eval,复现 主结果 + 方向 两条 headline):
export DATASET_TIER=resisc45
export WORK=/scratch/qk73/zz1230/RSWM/rswm_server_pack/server_run/work_resisc45
V="DATASET_TIER=$DATASET_TIER,WORK=$WORK,CHECKPOINT_DIR=$CHECKPOINT_DIR"
j01=$(qsub -v "$V" server_run/pbs/01_build_data.pbs)
j02=$(qsub -v "$V" -W depend=afterok:$j01 server_run/pbs/02_train_r128_sem.pbs)
j11=$(qsub -v "$V" -W depend=afterok:$j01 server_run/pbs/11_train_geo.pbs)
qsub -v "$V" -W depend=afterok:$j02 server_run/pbs/03_eval_base.pbs
qsub -v "$V" -W depend=afterok:$j02 server_run/pbs/04_eval_r128_sem.pbs
qsub -v "$V" -W depend=afterok:$j11 server_run/pbs/12_eval_geo.pbs
```

> 约 38 GPU 时(2 训练并行 + 3 评估)。rank(05-10)、overlap(13-17)、quals(18)
> **不必在 RESISC45 重跑**——那几条结论 AID 上做即可,论文里注明。
> RESISC45 是 256×256,构建器会同类别拼接到 512 再裁,canonical 参数(128×222/steps9/480×832)原样可用。

## 一次性准备(登录节点,有网)

```bash
tar xzf rswm_server_pack.tar.gz && cd rswm_server_pack
export HF_ENDPOINT=https://hf-mirror.com
bash server_run/offline_prefetch.sh          # 基模56G+AID3G+评估权重0.5G(计算节点断网必做)
WHEELS=1 bash server_run/offline_prefetch.sh # 计算节点也没 pypi 时
cd lora_remotesense/new && bash setup_env.sh && cd ../..
# 把 hf_home / torch_cache 放到计算节点能读到的共享盘 $HOME/.cache/
```

## 集群头(默认已是 Gadi/NCI 实测值)

每个 `.pbs` 顶部已用 **Gadi 原生格式**(与你跑通的 p0 一致):

```
#PBS -P rx40                                   # 项目/记账
#PBS -q gpuhopper                              # 旧 H200 队列；A800 必须换成集群实际队列
#PBS -l ncpus=12,ngpus=1,mem=128GB,jobfs=100GB,walltime=<各作业不同>
#PBS -l storage=scratch/qk73+scratch/rx40      # ★ 必须,否则计算节点看不到 /scratch
#PBS -l wd                                     # cwd=提交目录
#PBS -j oe
```

**A800 80GB**:先读 `server_run/A800_80G.md`。共享 4×A800 服务器使用
`source server_run/a800_env.sh`；`_pbs_env.sh` 保持硬件中立的 `auto`，以保留 H200。
如果管理员明确要求 PBS，必须把 `gpuhopper` 换成管理员提供的 A800 队列，并显式传
`HARDWARE_PROFILE=a800_80g,OFFLOAD=1`（保留 `mem=128GB`）。

**换项目/队列/存储**:直接改这四行(`-P` / `-q` / `-l storage` / `ncpus,mem,jobfs`),
或重生成时 `PBS_PROJECT=.. PBS_QUEUE=.. PBS_STORAGE=..` 传给生成脚本。
> ⚠️ Gadi 用**扁平** `-l ncpus=,ngpus=,mem=` 且**必须**有 `-l storage=`;`-l select=` 那种写法 Gadi 会拒。

`_pbs_env.sh` 里 conda 路径 / `ASSET_ROOT` / 缓存目录也已填成你的实测值(换机器改那里或 `-v` 覆盖)。

## 提交

**当前推荐:corrected 20–33 链(自动串依赖):**
```bash
export CHECKPOINT_DIR=$HOME/RSWM/lingbot-world-base-cam
export CONDA_ENV=rswm                         # 或 /home/user/RSWM/envs/rswm
DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh
bash server_run/pbs/qsub_corrected_rerun.sh
```

**legacy 01–19(仅调试/复现旧结果):**
```bash
DRYRUN=1 FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
```

**手动逐个 legacy 作业**(注意上表依赖顺序,从包根提交使 `PBS_O_WORKDIR`=包根):
```bash
cd rswm_server_pack
qsub -v CHECKPOINT_DIR=$HOME/RSWM/lingbot-world-base-cam,CONDA_ENV=rswm server_run/pbs/01_build_data.pbs
# 等 01 完成后再 02,依此类推
```

## 超时 / 断点

- **训练自动续**:`train_lora` 检测 `checkpoint-epoch-*`(train.py 每 epoch 存)→ 超时重投自动 `--resume_from`,不从头跑。
- **子步骤跳过**:build/eval 用 `work/stamps/*.done` 打点,已完成的重投直接跳。
- **某作业失败**:依赖它的会被 PBS 置 held;修好/等到资源后,重投该作业及其下游即可(前面不重跑)。

## 注意

- 一切数据/权重/输出放**共享文件系统**(计算节点能读写),别放本地 `/tmp`。
- `01_build` 是 CPU/IO 密集不用 GPU,想省卡可把它的 `ngpus=1` 改 `ngpus=0`(或投 CPU 队列)。
- P6(MVSplat360)未做成 `.pbs`:它要独立 torch2.2.2 环境 + gated SVD,断网集群需登录节点先装好整目录再拷入,参照 `server_run/p6_mvsplat.sh` 自封装。

## 档2 修正-prompt 全量重跑(2026-07-11 拍板 rank=16 + 档2 + 师兄5点)

**第 0 步(必做,登录节点纯 CPU,不烧 GPU;全绿才提交训练)**:
```bash
# a. 两个新模块自测(确认 Track B 数值/基线合成正确 —— 沙箱没 numpy 没跑过)
python lora_remotesense/new/trajectory_metrics.py     # 合成直线/L形应给小误差
python lora_remotesense/new/diagnostic_baselines.py   # 打印各基线 shape + 流幅度
# b. 方向指标校准(需已有一份 eval 输出;首轮可只看 GT 地板/人工平移那几项)
python server_run/validate_direction_metric.py --gt-h5 $WORK/data/train_semantic.h5 \
    --split $WORK/train_r16_sem_s42/train_val_split.json --eval-dirs base=$WORK/eval_base \
    -o $WORK/direction_validation.json
```
判停:GT 地板≈22.5°、人工平移≈0°、各场景流幅度>0。任一不过 → 停,先修指标。

**第 1 步:一键提交全量重跑(01 重建→13 训练+评估→baseline)**:
```bash
DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh   # 先看命令
bash server_run/pbs/qsub_corrected_rerun.sh            # 真提交(会强制重建数据集)
```
矩阵(全修正 prompt,train+eval 融合,全并行,SPLIT_SEED=42):
- r16 核心 6:sem/geo × seed 42/43/44(方向+保真,3 种子)
- r128 容量 2:sem/geo s42(误接地/容量对照,单种子 secondary)
- 探针 3:r16 sem/geo attn_cam(去 FFN LoRA,点5参数效率)+ r8 sem(更小 rank)
- 归因 2:shuffled_pose / direction_stripped(单种子)
成本≈13×(10训+10评)全并行 ≈ 墙钟 20h;baseline 评估 1 作业。

**第 2 步:全跑完后登录节点跑 Track B + 校准**(命令见 qsub_corrected_rerun.sh 末尾打印),
回传 direction_validation.json + trajectory_eval.json + 各 eval_results.json → 填论文。

报告口径:r16 sem/geo 各 3 训练种子 mean±std + 逐轨迹 bootstrap CI + 配对检验;
参数效率表对比 all/attn_cam/r8 的可训参数量 vs 指标。
