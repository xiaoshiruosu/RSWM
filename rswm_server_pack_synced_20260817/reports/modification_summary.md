# 修改摘要

## 修改文件

| 文件 | 原行为 | 修改后行为 | checkpoint 影响 |
|---|---|---|---|
| `server_run/common.sh` | `MAX_TRAIN_SAMPLES` 不透传；official 默认 net/flow | 透传 cap/split；official 默认 continuous/registration；错误口径不进 headline | 旧 adapter 可加载；续训 identity 更严格 |
| `lora_remotesense/new/train.py` | 随机两路 split；manifest 缺训练配方 | 支持预注册三路 split；test 排除；记录完整配方/prompt/cap | 旧无完整 manifest 的断点默认拒绝复用，防污染 |
| `dataset_loader.py` | 只支持随机 80/20 | 支持两路和 official 三路 JSON，范围/重复/交集硬门 | 不改变样本张量格式 |
| `run_manifest.py` | identity 字段不足，official prompt/path 可未知 | manifest 3.2，配方完整，official fail-closed | 不改权重，只改复用资格 |
| `evaluate.py` / `evaluate_baseline.py` | eval manifest 无 rank/target；baseline 缺 prompt/backbone | 从 checkpoint 继承训练身份；Frozen 明确标注 | 推理 API 不变 |
| `metrics.py` | FID 特征入口不可流式复用 | 暴露统一 frame feature loader/extractor | 数值口径不变 |
| `score_submission.py` | 条件帧进入 SSIM/PSNR/LPIPS/FID/FVD；FID 仅中帧 | 全部保真只看 future；FID 全预测帧；N>=32 | 不涉及权重 |
| `recompute_fvd_i3d.py` | 标准 FVD 包含条件帧，极小 N 可出数 | 条件帧后取 16 帧；N>=32；报告 selection | 数字与旧版不可直接混用（语义修正） |
| split/duplicate 检查器 | official 三路 schema 可能绕过 | 两路/三路兼容，val+test 作为 holdout | 无 |
| PBS official 链 | 训练器仍自己随机 split | 01 生成 official split，后续训练自动读取 | 新跑结果与旧随机 split 不可混用 |
| tests | 未覆盖上述协议 | 新增行为级矩阵/split/rollout 测试；评分入口未来帧测试 | 无 |

## 新增文件

- `server_run/iclr_experiment_matrix.json`
- `server_run/iclr_matrix.py`
- `server_run/evaluate_rollout_protocol.py`
- `server_run/test_iclr_protocol.py`
- `docs/ICLR_EXPERIMENT_PROTOCOL.md`
- `README_ICLR_MODIFICATIONS.md`
- `reports/*`

## 明确未做

- 未修改 Video DiT 主干架构。
- 未启动长时训练或下载数据/权重。
- 未产生任何 LoRA、Frozen、Full FT 或外部 baseline 的论文结果。
- 未实现 Full FT；矩阵中为 `runnable=false`，防止空行被误认为结果。
- 未声称 2D optical-flow proxy 等同于 3D geometry awareness。
- 未删除任何源文件或历史结果。
