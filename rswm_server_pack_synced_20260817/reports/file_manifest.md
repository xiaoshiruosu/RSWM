# 文件修改清单

基线源归档：
`C:\Users\25828\Desktop\游戏\ctw\rsroll-bench-server-20260731c.tar.gz`。

下面所有路径均相对于归档内 `rswm_server_pack/`；修改版保持相同相对结构。原始
`ctw` 目录没有被覆盖、移动或删除。

## 新增文件（12）

1. `README_MODIFICATIONS.md`
2. `README_ICLR_MODIFICATIONS.md`
3. `docs/ICLR_EXPERIMENT_PROTOCOL.md`
4. `reports/code_audit.md`
5. `reports/version_verification.md`
6. `reports/modification_summary.md`
7. `reports/test_report.md`
8. `reports/file_manifest.md`
9. `server_run/iclr_experiment_matrix.json`
10. `server_run/iclr_matrix.py`
11. `server_run/evaluate_rollout_protocol.py`
12. `server_run/test_iclr_protocol.py`

交付时另生成 `patches/rswm_iclr_20260813.patch` 和 `patches/SHA256SUMS.txt`；它们是
交付元数据，不计入源代码 diff 统计。

## 修改文件（19）

1. `lora_remotesense/new/dataset_loader.py`
2. `lora_remotesense/new/evaluate.py`
3. `lora_remotesense/new/evaluate_baseline.py`
4. `lora_remotesense/new/metrics.py`
5. `lora_remotesense/new/run_manifest.py`
6. `lora_remotesense/new/train.py`
7. `server_run/check_benchmark_duplicates.py`
8. `server_run/check_near_duplicates.py`
9. `server_run/common.sh`
10. `server_run/pbs/01_build_data.pbs`
11. `server_run/pbs/34a_learnability_drop.pbs`
12. `server_run/pbs/34b_learnability_native.pbs`
13. `server_run/pbs/51_ladder_train_eval.pbs`
14. `server_run/pbs/_pbs_env.sh`
15. `server_run/pbs/qsub_corrected_rerun.sh`
16. `server_run/recompute_fvd_i3d.py`
17. `server_run/score_submission.py`
18. `server_run/test_pose_timeline.py`
19. `server_run/test_submission_contract.py`

## 未修改但重要的文件

- `lora_remotesense/new/backbone.py`：两专家 Video DiT 与真实 LoRA target 分类/审计入口。
- `wan/modules/model.py`：Wan Video DiT 主干；本次有意保持 checkpoint 兼容。
- `lora_remotesense/build_dataset_from_benchmark.py`：AID/cross-domain 轨迹视频 HDF5 构建入口。
- `server_run/make_official_splits.py`：已有的 source-disjoint 三路划分生成器，本次接入正式链。
- `requirements-lock.txt`：正式 Linux/CUDA 环境锁定依据。
- `server_run/baselines/baseline_status.json`：外部 baseline 仍为未运行状态，未伪造结果。

## 删除文件

无。

## 路径映射

```text
原始：源归档 / rswm_server_pack / <relative_path>
修改：交付目录 / rswm_server_pack / <relative_path>
```

逐行 unified diff 位于 `patches/rswm_iclr_20260813.patch`。
