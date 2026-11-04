# 本地 DINOv2 特征导出

`python -m rswm_inria.feature_export dinov2` 填补“本地原始预训练 state_dict → 既有 TorchScript 特征配置”的部署步骤。它只生成候选，不选择正式仪器，不生成 reviewer 决定或 held-out 验收。后续继续走 `rswm-release-features calibrate/validate`、`rswm-quality` 和既有指标消费者。

本实现按官方 [hub 入口](https://github.com/facebookresearch/dinov2/blob/main/hubconf.py)、[backbone 工厂](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py) 及 [forward_features 返回值](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py)核对接口（核查日期：2026-10-02）。实际执行锁定 owner 给出的完整 Git revision，不沿用网页的 main 作为运行身份。

## 输入与执行

服务器需已有官方 `facebookresearch/dinov2` 的干净 Git checkout、明确完整 40 位 commit、本地原始 backbone state_dict 和实际使用决定。程序没有下载源码/权重路径，没有从网络初始化模型的分支。

```bash
python -m rswm_inria.feature_export dinov2 \
  --source-dir /srv/rswm/vendor/dinov2 \
  --weights /srv/rswm/weights/dinov2/OWNER_CHOSEN_BACKBONE.pth \
  --architecture OWNER_CHOSEN_SUPPORTED_ARCHITECTURE \
  --revision OWNER_VERIFIED_FULL_40_CHARACTER_COMMIT \
  --decisions /srv/rswm/inputs/dinov2-use-decision.json \
  --preprocessing /srv/rswm/inputs/dinov2-preprocessing.json \
  --device cpu --batch-size 2 \
  --out /srv/rswm/deliveries/feature-candidate-001
```

支持 `dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`、`dinov2_vitg14` 及各自 `_reg` 版本；不支持 classifier、depth head、DINOv3 或任意 HF 权重。所选架构必须与 owner 的完整预训练 backbone 权重匹配。没有 missing/unexpected keys、前缀剥离、随机权重或另一个模型的回退。

决策 JSON 恰有 `approved`、`license_record`、`decision_reference`。运行者据实际来源/权重使用决定填写；`approved=false`、空记录或缺参考被拒绝。源码 origin、HEAD、工作树和逐文件 SHA 用于本地一致性核对，不能证明外部签名或权重真实性；保留官方原权重来源与使用记录仍是 owner 的输入责任。

预处理 JSON 恰有 `resize`、`center_crop`、`resample`、`scale`、`mean`、`std`。以下仅示范官方分类评估预处理形状和归一化参数的表达方式；是否用于 RSWM 正式未知分布仪器仍需事前决定与验证：[官方 transforms](https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py)。

```json
{"resize":[256,256],"center_crop":[224,224],"resample":"bicubic","scale":"uint8/255","mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}
```

维度顺序均为 `[height,width]`。最终模型输入大小必须是 patch size 14 的整数倍。既有 `features.py` 仍接收固定 128² uint8 RGB 图块，然后执行此明确 resize/crop/normalization；导出的模型接收已预处理 float32 BCHW 张量，输出经过 normalization 的 class token，不调用分类 head。

## 实际导出与读回检查

导出在 CPU float32 执行，因此无需 GPU；大架构仍需相应 RAM、墙钟与磁盘额度，不能把显存设备数量当作 CPU 内存保证。原 hub 工厂调用固定为 `torch.hub.load(..., source="local", pretrained=False)`，随后使用 `torch.load(..., weights_only=True)` 严格读入完整原始 state_dict。程序逐键检查 tensor、shape、有限数和实际加载值；任何缺失、额外、嵌套 checkpoint 或无法识别权重均拒绝。

为序列化使用官方源码的普通 PyTorch attention 分支，原源码导入前明确设置 `XFORMERS_DISABLED=1`，记录该导出设置；原代码提供这一开关，见[官方 attention](https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/attention.py)。这是候选导出的工程设置，不是原生成模型的相机或采样配置变更。应在新进程导出，不混用先前已经导入的其他 DINOv2 源树。

输出目录必须全新，且与原源码、权重及输入树独立。真实加载前先保存 `export-contract.json`，固定 CPU float32、batch 1/2、冻结空间尺寸、`atol=1e-5` 与 `rtol=1e-5`；检查后不调整容差。实际 wrapper 调用 `forward_features(x)["x_norm_clstoken"]`，比较 eager、同进程 JIT 读回和新进程 JIT 的有限 `(B,D)` 输出。新进程 PID、命令、退出码、时长、输入/输出数组和 stdout/stderr 存在 `roundtrip/`。子进程超时上限默认 300 秒，可显式传 `--timeout-seconds`；超时保留失败与已发生成本。

成功保留 `encoder.torchscript.pt`、可直接给 `rswm-features` 的 `feature-config.json` 和 `export-report.json`。配置锁定实际 JIT SHA、repo/revision、输出维数/层、完整预处理、device/batch 与 Torch/NumPy/Pillow 版本。`--device` 指后续特征执行设备，导出与软件 round-trip 仍在 CPU；CUDA 尚未因此验证。应在目标特征运行环境导出或明确重新形成对应版本身份，不能随意改 runtime version 来绕过检查。

失败退出 2 并保留 `export-failure.json` 与部分文件，重新执行使用新目录；不覆盖成功包或失败现场。成功退出 0 的状态为 `EXPORTED_FEATURE_CANDIDATE_NOT_INSTRUMENT_VALIDATED`，`release_validated=false`、`heldout_validation=NOT_RUN`、`research_acceptance=NOT_EVALUATED` 始终明确。候选序列化通过不能代替真实校准、曝光账本、独立留出验证或正式模型评价。

## 软件验证范围

单元测试使用实际小型 Torch 模块，检查严格 checkpoint 载入、缺键/额外键/shape/非有限拒绝、batch 1/2 JIT 新进程读回，以及既有 `FrozenTorchScriptFeatures.encode`。这些输出明确 `diagnostic_features=true`，不是官方预训练 DINOv2，也不能升级为正式仪器。正式 DINOv2 源码与预训练权重未具备时，实模导出保持 `NOT_RUN`；交付的是可在服务器执行的代码。
