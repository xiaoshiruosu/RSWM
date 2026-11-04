# ICLR benchmark 化改造（2026-08-13）

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FlashAttention-2→SDPA）；Legacy/reference = **H200 141GB**
>（sm90、可选 FA3 路径保留，本轮未做 H200 实机回归）。

## 结论与任务边界

本版本保留现有 Wan/LingBot I2V Video DiT 主干，不把任务改写成真实时间预测。
正式任务定义是：

> Continuous Trajectory-conditioned Remote Sensing Video Rollout：给定单帧遥感
> 观测与连续相机轨迹，生成未观测空间区域的连续视频 rollout。

因此本文可以讨论连续轨迹控制、低条件重叠与长程误差累积；不能把 AID 合成滑窗
视频称为 future Earth dynamics，也不能仅凭 video diffusion 主干声称模型具有显式
3D geometry awareness。

## 已完成的 P0 修复

1. `MAX_TRAIN_SAMPLES` 已从 `common.sh` 透传到 `train.py`，按预注册划分后的
   固定前缀确定性截断，并写入 split、checkpoint config 与 run manifest。
2. 训练支持 `--split_file` / `TRAIN_SPLIT_FILE`。official 链由
   `make_official_splits.py` 生成 source-disjoint 70/10/20 三路划分，训练器只读取
   train/val，hidden test 不进入 DataLoader。
3. official 主通路默认 `prompt_schema=net_continuous` 与
   `path_estimator=registration`；缺失或被改为 legacy 口径时 fail-closed，不能进入
   headline。
4. manifest v3.2 记录 dataset tier、train/split seed、rank/alpha/targets、训练样本
   上限、epoch、batch、梯度累积、学习率以及 prompt/path 口径；续训或复用时任一身份
   不同均拒绝。
5. 官方评分器和标准 I3D-FVD 重算都从 `t=1` 开始，只评预测帧。FID 使用全部预测帧，
   不再只抽中间帧；FID/FVD 的样本级计算统一执行 `N>=32` 门。

## ICLR 实验协议

机器可读真源是 `server_run/iclr_experiment_matrix.json`，用下面的 CPU 命令校验并
物化计划（不会启动训练）：

```bash
python server_run/iclr_matrix.py --check-only
python server_run/iclr_matrix.py -o /path/to/work/iclr_execution_plan.json
```

计划包含：

- rank sweep：`r=4/8/16/32/64`，`alpha=rank`；
- target ablation：`attn_only / attn_cam / all`；
- primary 配方三个训练种子 `42/43/44`；
- Frozen backbone baseline；
- Full FT 占位明确标成 `not_implemented/runnable=false`。

Full FT 没有被伪装成已实现：当前是两份 14B 专家，现有 8-bit LoRA optimizer 与
inactive-model CPU swap 不是公平、可验证的全参训练方案。若要把 Full FT 放主表，需先
实现并验证独立的 FSDP/ZeRO 配方、优化器状态、峰值显存与收敛性。

## Overlap 与 horizon 的正确口径

论文中的 80/50/30/10/5% 定义为：

```text
overlap(condition frame t=0, predicted frame t)
```

而不是相邻两帧的 overlap。相邻帧保持高 overlap 是连续视频的必要条件；要求每个
相邻 step 只有 5% overlap 会把视频变成不连续图像跳跃。工具会把相邻 step overlap
作为独立 continuity audit 报告，绝不与主自变量混用。

已有评估输出上运行：

```bash
python server_run/evaluate_rollout_protocol.py \
  --gt-h5 "$WORK/data/train_geo.h5" \
  --eval-dir "$WORK/eval_r16_geo_s42" \
  --split-file "$WORK/train_r16_geo_s42/train_val_split.json" \
  -o "$WORK/eval_r16_geo_s42/rollout_protocol_results.json"
```

输出包含：

- 80/50/30/10/5% 条件重叠分箱的 future-frame-only SSIM/PSNR；
- `t=1..32` 的逐帧曲线；
- 17/25/33 帧 prefix 与 exact-length cohort；
- 相邻 step overlap 审计；
- 生成与 GT 的 2D 光流方向/幅度一致性 proxy。

该工具故意不在 frame bin 内算 FID/FVD：同一视频内的相关帧不能冒充独立样本来放大
N。样本级 FID/FVD 用官方评分器与 `recompute_fvd_i3d.py`。

## 正式集群入口

仍使用 `server_run/pbs/qsub_corrected_rerun.sh`。`01_build_data.pbs` 现在生成
`$WORK/official_splits.json`，后续 official 训练通过 `TRAIN_SPLIT_FILE` 自动复用。
rank/target 矩阵的未覆盖作业可依据 `iclr_execution_plan.json` 新建独立 PBS；在未真跑
前，不得把配置文件当成实验结果。

跨域表限定为 AID 训练权重在 WHU-RS19 / RESISC45 上 test-only 评估。先运行已有
近重复/跨库重复门，再用 `generalization_gap_report.py` 汇总；目标域重新训练的结果不应
混入“cross-domain generalization”主表。

## 验证与限制

已运行的 CPU 回归见 `reports/test_report.md`。本地没有 56GB 基模与匹配的 GPU 运行
条件，因此以下项目明确未执行：真实模型初始化、LoRA 真注入参数计数、1-batch forward、
长时训练、真实 I3D 权重前向与任何论文数值。已有 mock/假提取器测试只能验证参数转发、
形状、契约与统计管线，不能替代服务器 smoke run。
