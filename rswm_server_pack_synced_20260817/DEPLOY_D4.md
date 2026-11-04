# D4 部署手册(v3.0)

> 生成:2026-07-28 | 目标:Gadi/NCI(PBS + H200)| 距 ICLR 冻结日(08-15)18 天
>
> 这份是**本次部署的可执行清单**。原理/依赖图/单作业细节见 `README_RUN.md`
> 与 `server_run/pbs/README_PBS.md`;本文件只回答"现在按什么顺序敲什么命令、
> 每步看到什么才算过"。
>
> **一句话**:旧数据全部作废(125 模板缺陷 + 泄漏 + 口径),这次是从零重建的
> D4。数据质量四门(池 / 重复 / split 泄漏 / 裁剪后语义)+资源配额门会拦住
> 坏数据,**门不绿就别往下走**。

---


## 正式矩阵(v2.9 拍板,勿改)

| 作业 | H5 | pose_mode | 角色 |
|---|---|---|---|
| 20-22 | GEO_H5 | drop | **headline 真·text-only**(三种子) |
| 23-25 | GEO_H5 | native | **headline Plücker**(三种子) |
| 26/28/30 | GEO_H5 | drop | Rank/target 消融(跟随主线口径) |
| 27/29 | GEO_H5 | native | 对应 native 消融 |
| 31 | 三份 | drop/native/identity | 每种口径各一份 baseline |
| 32/33 | 变体 | native | 归因(shufpose / geostrip) |
| **34a** | probe | drop | 判门:文本方向可学习性 |
| **34b** | probe | native | 判门:Plücker 方向可学习性 |
| 35 | GEO_H5 | drop | 逐段富文本臂 |
| 36 | SEM_H5 | **identity** | 静止相机命令**诊断**臂 |

* `identity` ≠ text-only。它是"显式命令相机不动",`headline_eligible` 恒为 false。
* **34a 与 34b 都 PASS 才放行 20-30/35/36**。只让 drop PASS 就解锁 native 实验,
  证据链不完整 —— native 主线失败时仍可被归因为"pose 表示学不会"。
* Δ 只许减**同 pose_mode** 的 baseline;`collect_results` 会自动拒收不合格产物。

## 0. 部署(5 分钟)

```bash
# 登录节点
cd /scratch/qk73/$USER/RSWM          # = ASSET_ROOT,基模与缓存都在这
mv rswm_server_pack rswm_server_pack_OLD_$(date +%Y%m%d)   # 旧包留档,不删
tar -xzf ~/rswm_server_pack_v30.tar.gz                    # 解出 rswm_server_pack/
cd rswm_server_pack

# 环境(与旧版一致,不用重装)
source /scratch/qk73/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/rx40/$USER/envs/rswm

# 认包:必须看到 v3.0、run_manifest、34a/34b 双判门、语义门三件套
head -3 CHANGELOG_vs_0710.md   # 应显示 v3.0
ls lora_remotesense/semantic_gate.py server_run/check_semantic_consistency.py \
   server_run/test_semantic_gate.py   # 全部都在才是 v3.0 包
ls server_run/test_pose_timeline.py \
   server_run/pbs/34a_learnability_drop.pbs \
   server_run/pbs/34b_learnability_native.pbs \
   lora_remotesense/new/run_manifest.py
```

**离线预取**(只在联网机跑;已做过可跳过)。断网计算节点的一切权重都靠它:

```bash
bash server_run/offline_prefetch.sh                 # 基模+AID+FID/FVD/LPIPS 权重(~60GB)
I3D=1        bash server_run/offline_prefetch.sh    # 标准 FVD 的 i3d_torchscript.pt(~50MB,必做)
BASELINE_A=1 bash server_run/offline_prefetch.sh    # outpaint 基线的 SD-2-inpainting(~5GB)
TIER2=1      bash server_run/offline_prefetch.sh    # 泛化集(WHU-RS19 主 + RESISC45/PatternNet 辅)
# MVSPLAT=1 …… 只在决定做档 B 时再拉(含 gated SVD,须先 HF login)
```

---

## 1. 纯 CPU 自检(登录节点,30 秒,不烧 GPU)

```bash
python server_run/test_taxonomy_prompt.py        # 期望:全绿 ✓(49 项)
python server_run/test_route_resolution.py       # 期望:全绿 ✓(15 项)
python server_run/test_wellposed_dualcolumn.py   # 期望:全绿 ✓(17 项)
python server_run/check_resisc45.py --self-test  # 期望:全绿 ✓(9/9)
python server_run/recompute_fvd_i3d.py --self-test
python server_run/baselines/registry.py          # 期望:接线自检全部通过
```

**任一不绿 → 停,把输出发回**。这几项全是纯逻辑,在服务器上应与本地一致。

---

## 2. 配额实测(**一切提交的前置**)

07-25 删过旧 checkpoint,余量必须重测:

```bash
lquota
```

把 **rx40 与 qk73 两行原样发回**。判据(D4 全量足迹 ≈ 107–127 GiB / 3.1–3.7k inode):

| 情况 | 动作 |
|---|---|
| rx40 空间 ≥ 300 GiB 且 inode ≥ 20k | 默认阈值直接过,不用带 env |
| 空间 ≥ 150 GiB 且 inode ≥ 8k | 提交时显式下调:`MIN_QUOTA_GIB=150 MIN_QUOTA_INODES=8000` |
| 低于上面 | 先清理(旧 WORK 的 `eval_*/generated/*.npy` 是 inode 大户),别硬跑 |

> **存储决策(07-28)**:新 WORK 放 qk73(inode 富余),计算仍 `-P rx40`;
> rx40 的 5,276 inode 不再直接限制新产出。
> lquota 的列位解析是按你给的样式写的;若排版不同,预检会**硬错而不是误判**——
> 报错就把 lquota 原始输出发回,一行适配即可。

---

## 3. 池预检(P0 重提五项里唯一没齐的一项)

```bash
python server_run/check_aid_pools.py --threshold 125
echo "RC=$?"
```

**期望**:`RC=0`,四池互斥、各 ≥125、无 thumbnail。注意场景名现在是
`general/water/urban/rural`(flood 已更名 water)。

FAIL 就停——它是三道门的第一道,后面全白搭。

---

## 3b. 【已完成·规则冻结】语义门校准终局(2026-07-29 拍板)

> 四轮真值校准结束(预注册决策规则执行完毕):
> * **冻结规则 = `frac >= 0.2777777777777778`**(9 帧;必须保存全精度,
>   已写死为代码默认 `FROZEN_P0_THRESHOLD`,不再依赖 env);
> * 校准结果:0 坏样本放走 / 3 有效样本误拒(00006 Port、00016 BareLand、
>   00019 Farmland)→ datasheet 记录这三类的系统性欠采样偏置;
> * clspres 族(含事后 OR 组合)不采纳;clspres/raw/scene_cls_mass 仅作
>   诊断证据;**第五轮特征迭代禁止**;
> * 决策留档:服务器 `rule_decision_v2712.txt`(只读 + SHA256)。
> 本节不再需要执行任何校准命令;v261 smoke H5 继续只读保留(永久真值集)。

## 4. Canary:20 样本冒烟(~1–2h,一步都不能跳)

> ⚠️ v2.7 起建库入口变了行为:建库期语义门会拒绝采样换源图(日志里
> `[semgate] ... 被拒 ... 换图重采` 属正常),p0_smoke 里多一道
> `SEMANTIC GATE`;**必须删掉旧 smoke 目录全新跑**(builder 变了,
> sha-bound stamp 也会强制重建)。

```bash
export WORK=/scratch/rx40/$USER/RSWM_D4_smoke_$(date +%Y%m%d)
bash server_run/p0_smoke.sh
```

内含:20 样本建库 → **重复/源身份门**(`--expect-zero --expect-n 20`)→
**split 泄漏门** → neutralize → 1 epoch 训练 → 4 样本评估。

**人眼验收(必做,自动门管不了"内容对不对"):**

```bash
python server_run/visualize_benchmark.py --gt-h5 $WORK/smoke/smoke_geo.h5 \
       -o $WORK/smoke/viz --max 20        # [拍板] 全部样本,不只前 12
```

看 `sheet_*.png`:场景标签与画面是否相符(water 是不是真水面)、轨迹折线是否
合理、起始窗口框尺度对不对;看 `summary.json`:四场景是否均衡、方向分布、
净位移比 <0.2 的占比。

**[拍板 2026-07-29] 受控 P0 附加验收证据**(自动落盘,人工核对):
- `$WORK/smoke/semgate_build_report.json`:各 source_class 尝试/拒绝/接受数、
  已知误拒类(Port/BareLand/Farmland)重采单列、最大 tries 对照池耗尽路径、
  全部语义样本的 semgate_frac 分布;
- 人工视觉五类重点:River 河体是否真在窗内 / StorageTanks 储罐是否在窗内 /
  Port 码头船在时是否被不必要重采 / BareLand、Farmland 纹理清楚却被拒的比例。

**判停**:任一门 FAIL,或样张明显不对 → 停,把 `duplicate_report_smoke.json`、
`semgate_build_report.json` 与 `summary.json` 发回。

---

## 5. 全量建库 + 训练矩阵(主战场)

```bash
export WORK_CORRECTED=/scratch/rx40/$USER/RSWM_D4_work_$(date +%Y%m%d)

DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh    # 先干跑看依赖与作业名
bash server_run/pbs/qsub_corrected_rerun.sh             # 真提交
# 配额紧时:MIN_QUOTA_GIB=150 MIN_QUOTA_INODES=8000 bash ...
```

提交内容(afterok 依赖链自动串好):

| 作业 | 内容 |
|---|---|
| 01 | 建库(**重复门 + split 泄漏门 + 语义门 + overlap enforce<0.10**,FAIL 则下游全挂) |
| **34** | **可学习性判门**(单段/四方向/高重叠独立 profile;角误差 >45° → exit 3,**后续全部不执行**) |
| 20–25 | 主线六训练:r16 sem/geo × seed 42/43/44(afterok:34) |
| 26–30 | 单种子消融:r128 sem/geo、attn_cam ×2、r8_sem |
| 31 | zero-shot baseline(afterok:20,复用其 split) |
| 32/33 | 归因实验:shuffled_pose / direction_stripped(单种子) |
| **35/36** | **逐段富文本臂 / 真 no-pose 臂**(pose_mode=drop;拆解"通道 vs 信息量"与"静止命令污染") |

**建库后立刻核对**(01 跑完、训练还在排队时):

```bash
python server_run/visualize_benchmark.py --gt-h5 $WORK_CORRECTED/data/train_geo.h5 \
       -o $WORK_CORRECTED/viz --max 24
cat $WORK_CORRECTED/duplicate_report_build.json | head -30
```

---

## 6. 基线(与训练并行,不抢 GPU)

基线走**共享 manifest 协议**(同一 H5/split/帧数/分辨率/gen_seed 钉死,
逐条 record 核对,不符即非零退出)。先写 manifest,再按名跑:

```bash
W=$WORK_CORRECTED
mkdir -p $W/baselines
cat > $W/baselines/manifest.json <<EOF
{
  "h5":         "$W/data/train_geo.h5",
  "split":      "$W/train_r16_sem_s42/train_val_split.json",
  "num_frames": null,
  "resolution": [480, 832],
  "gen_seed":   42,
  "sample_keys": null,
  "max_samples": null
}
EOF
# num_frames 用 null:canonical 帧数按样本变(v2.8 起 17/25/33),单值会硬比失败

python server_run/baselines/run_baseline.py --list          # 先看注册表与状态

# 四个诊断基线:纯 CPU,至今一个数都没出过,必跑
for BL in persistence deterministic_translation random_direction_translation gt_oracle; do
  python server_run/baselines/run_baseline.py --baseline $BL \
      --manifest $W/baselines/manifest.json --out $W/baselines/$BL
done

# 档 A 外部基线:平移+outpaint(需先 BASELINE_A=1 预取,~3-5 GPU 时,要 GPU)
python server_run/baselines/run_baseline.py --baseline image_outpainting \
    --manifest $W/baselines/manifest.json --out $W/baselines/image_outpainting
```

> 先加 `--dryrun` 跑一遍(不生成、不写盘,只查依赖/文件/样本解析)最省事。
> `gt_oracle` 与 `deterministic_translation` 是**指标上界**,论文里必须标注,
> 不得与真实方法并列排名(脚本自己也会打这条警告)。

档 B(MVSplat360)先跑冒烟门,过了再投全量:

```bash
conda activate mvsplat        # 独立 torch2.2.2 env
MVSPLAT_DIR=/path/mvsplat360 MVSPLAT_SMOKE_N=3 bash server_run/mvsplat_smoke.sh
# 期望:MVSPLAT360 SMOKE: PASS;不过就转 scope 声明,别追加 GPU
```

---

## 7. 分析出表(登录节点,纯 CPU)

```bash
W=$WORK_CORRECTED
S=$W/train_r16_sem_s42/train_val_split.json

# 7.1 方向指标校准(step=1 正式口径)
python server_run/validate_direction_metric.py --gt-h5 $W/data/train_semantic.h5 \
    --split $S --eval-dirs textonly=$W/eval_r16_sem_s42 \
    textpluecker=$W/eval_r16_geo_s42 base=$W/eval_base -o $W/direction_validation.json

# 7.2 Track B:先 5 样本冒烟,PASS 才全量
python server_run/run_trajectory_eval.py --gt-h5 $W/data/train_geo.h5 --split $S \
    --eval-dirs textonly=$W/eval_r16_sem_s42:text \
    --max-samples 5 --with-baselines -o $W/trajectory_eval_smoke.json
#   ↑ 必须出现 TRACK B V2 CALIBRATION: PASS(FAIL 会 exit 3)
python server_run/run_trajectory_eval.py --gt-h5 $W/data/train_geo.h5 --split $S \
    --eval-dirs textonly=$W/eval_r16_sem_s42:text \
                textpluecker=$W/eval_r16_geo_s42:geo \
                geostrip=$W/eval_r16_geostrip_s42:geo \
    --with-baselines -o $W/trajectory_eval.json

# 7.3 标准 I3D-FVD 重算(论文主 FVD;R3D 版是 FRD,不可与文献比)
python server_run/recompute_fvd_i3d.py --eval-dirs base=$W/eval_base \
    textonly=$W/eval_r16_sem_s42 textpluecker=$W/eval_r16_geo_s42 -o $W/fvd_i3d.json

# 7.4 多种子聚合 + 参数效率
python server_run/aggregate_seeds.py --seeds 42 43 44 \
    --eval-dirs r16_sem=$W/eval_r16_sem_s{42,43,44} r16_geo=$W/eval_r16_geo_s{42,43,44} \
    --pair r16_sem r16_geo --out-prefix $W/seed_summary
python server_run/parameter_efficiency_summary.py --exp-dirs $W/train_r16_* $W/train_r8_* $W/train_r128_* \
    --out-prefix $W/param_efficiency

# 7.5 统一结果总表(corrected 目录优先,run_all 目录回退)
python server_run/collect_results.py $W
```

> **`:text` / `:geo` 后缀 v2.6 起强烈建议总是写**——它是路由的权威声明。
> 不写时:有 meta 的目录按 meta 判;旧目录靠名字启发式,`base`/`shufpose`
> 这类歧义名会**直接 FATAL 要求你显式声明**(这是特性,不是 bug)。

---

## 8. 泛化 tier(主线出数后错峰)

```bash
CK=$WORK_CORRECTED/train_r16_sem_s42/best_checkpoint

# 主集:WHU-RS19(1001 张 600² + 4 个异形尺寸,仍为零拼接)。先量化池容量
python server_run/check_resisc45.py --tier whurs19 --check-source
bash server_run/pbs/qsub_resisc45.sh --tier whurs19 --lora-ckpt $CK

# 辅助(GPU 有余再跑;2×2 拼接有接缝伪影,论文里须标 caveat)
bash server_run/pbs/qsub_resisc45.sh --tier resisc45   --lora-ckpt $CK
bash server_run/pbs/qsub_resisc45.sh --tier patternnet --lora-ckpt $CK
```

**只要 WHU-RS19 一个就够撑泛化 claim**(已拍板),三个都赶不上冻结日可只报它。

---

## 判停红线(任何一条命中就停,把输出发回)

1. 第 1 步自检不全绿;
2. `check_aid_pools` RC≠0;
2b. §3b 真实 CLIP 校准 RC≠0、JSON 缺失/为空,或分不开人检已知的坏样本
    (00009/00011);
3. 冒烟的重复门 / split 泄漏门 FAIL;
4. 样张与标签明显不符;
5. 全量 01 建库的门 FAIL(此时 20–33 会自动挂住,**别手工绕过**);
6. Track B 5 样本冒烟没打出 `TRACK B V2 CALIBRATION: PASS`;
7. MVSplat 冒烟 <80% 成功率。

---

## 已知会遇到的几件事(不是 bug)

- **`base` / `shufpose` 目录在 Track B 里会 FATAL 要求 `:text`/`:geo`** ——
  路由无法从名字唯一判定时故意不猜。按实际配置显式写即可。
- **Track B oracle 净方向门可能仍 FAIL**(上轮 25.11°)。去重后坏模板不再多副本
  计权,大概率下降;若仍 FAIL,跑 `server_run/autopsy_trackb.py` 尸检后再定门,
  **不要为了过门去调 min_net_ratio**。
- **旧 eval 目录不产 well-posed 双栏**(meta 无 `gt_net_displacement_ratio`),
  会打 WARNING、其余指标照常;混版本目录则**硬失败**(这是防静默混算)。
- **`p1_main.sh` / `p0_smoke.sh` 也带门了**,与 PBS 路径同构,不会绕过。

---

## 出问题时最有用的三个文件

```
$WORK/duplicate_report_build.json     # 重复/源身份门的完整证据
$WORK/logs/p1_*.log                   # 建库、门、split 的全过程
$WORK/viz/summary.json                # 数据集统计(场景/方向/段数/净位移比分布)
```

发回时带上这三个 + 报错那段日志,我这边能直接定位。
