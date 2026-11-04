# RSWM P0 + ICLR benchmark 改造交付说明

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FlashAttention-2→SDPA）；Legacy/reference = **H200 141GB**
>（sm90、可选 FA3 路径保留，本轮未做 H200 实机回归）。

本目录是只读基线 `rsroll-bench-server-20260731c.tar.gz` 的独立修改版。原始归档
SHA-256 为：

```text
FCA092BA5F294EE7FD3DA3A485484CA4716698237BCEE8ABBA9CB20B6591E9D6
```

本次保留 Wan/LingBot I2V Video DiT 主干，将任务严格定义为 **Continuous
Trajectory-conditioned Remote Sensing Video Rollout**（静态场景的空间相机 rollout），
不把 AID 合成序列描述成真实未来地球动力学。

## 从这里开始

1. 阅读 `reports/code_audit.md` 和 `reports/version_verification.md` 了解基线与架构。
2. 阅读 `README_ICLR_MODIFICATIONS.md` 和 `docs/ICLR_EXPERIMENT_PROTOCOL.md` 了解正式协议。
3. 用 CPU 校验机器可读实验矩阵：

   ```bash
   python server_run/iclr_matrix.py --check-only
   ```

4. Linux/PBS 集群正式入口仍是：

   ```bash
   bash server_run/pbs/qsub_corrected_rerun.sh
   ```

5. 已有推理结果可追加 overlap/horizon/future-only 分析：

   ```bash
   python server_run/evaluate_rollout_protocol.py \
     --gt-h5 "$WORK/data/train_geo.h5" \
     --eval-dir "$WORK/eval_r16_geo_s42" \
     --split-file "$WORK/train_r16_geo_s42/train_val_split.json" \
     -o "$WORK/eval_r16_geo_s42/rollout_protocol_results.json"
   ```

## 关键修复

- official 训练复用 source-disjoint 70/10/20 固定划分，hidden test 不进入训练器；
- `MAX_TRAIN_SAMPLES`、rank/alpha/targets、训练配方、prompt/path 口径均进入 manifest identity；
- official 默认且强制 `net_continuous + registration`，不合格运行 fail-closed；
- SSIM/PSNR/LPIPS/FID/FVD 全部排除条件首帧，FID 使用全部预测帧；
- 分布指标执行 `N>=32` 样本门；
- 增加 80/50/30/10/5% condition-overlap、17/25/33 帧 horizon 与 2D flow proxy 协议；
- rank `4/8/16/32/64`、target `attn_only/attn_cam/all`、Frozen 和三种子主配置已机器可读预注册。

## 尚未完成的论文证据

未下载模型或数据，未运行 GPU forward、LoRA 真注入计数、训练、跨域评估或真实 I3D
特征抽取。Full FT 也没有被伪造成可运行项：双 14B 专家需要先实现并验证独立的
FSDP/ZeRO 配方。任何表格行在对应 checkpoint、manifest、gate 和评估文件真实存在前，
都只是实验计划，不是实验结果。

完整变更见 `reports/file_manifest.md`、`reports/modification_summary.md` 和
`patches/rswm_iclr_20260813.patch`；测试边界见 `reports/test_report.md`。
