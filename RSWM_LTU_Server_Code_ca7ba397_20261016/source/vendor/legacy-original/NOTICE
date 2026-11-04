RSWM-Bench
Copyright (c) 2026 The RSWM-Bench Authors

本产品包含由第三方开发的软件与模型。按 Apache License 2.0 §4(d)，
下游再分发必须保留本文件。

--------------------------------------------------------------------
1. Wan 视频扩散代码（Alibaba / Wan-AI），Apache License 2.0
--------------------------------------------------------------------
路径：
    wan/
    lora_remotesense/new/wan/
这两棵子树是 Wan 系列视频扩散实现的改写版（VAE 见
`wan/modules/vae2_1.py` 的 `Wan2_1_VAE` 与 `wan/modules/vae2_2.py`
的 `Wan2_2_VAE`；目标骨干见 `lora_remotesense/new/camera_trajectory.py`
文件头写明的 "Wan2.2 i2v-A14B"）。

修改说明（Apache §4(b) 要求的 "prominent notices"）：
  * 加入了 Plücker 位姿注入的三态开关（native / identity / drop），
    见 `lora_remotesense/new/camera_trajectory.py` 与 `wan/utils/cam_utils.py`
    的调用侧；
  * 加入了 LoRA 注入与逐样本 low/high 噪声路由（`lora_remotesense/new/train.py`）；
  * 评估侧改写了 SciPy `sqrtm` 调用签名以兼容新版 SciPy，并加入
    well-posed 双栏口径（`lora_remotesense/new/metrics.py`、`evaluate.py`）。

⚠️ 待确认：本仓库内的 `wan/` 子树在打包时已不含上游的版权头，因此
无法从文件本身逐行核对具体上游 commit。上游归属按 Wan 官方发布
（Apache-2.0）记载；若你要做严格的许可证审计，请以 Wan 官方仓库的
原始文件为准，并把上游 LICENSE/NOTICE 一并取来。

--------------------------------------------------------------------
2. 基模权重 robbyant/lingbot-world-base-cam，Apache License 2.0
--------------------------------------------------------------------
不随本仓库分发。2026-07-29 经 Hugging Face API 核实：
`gated=false`、`private=false`、`cardData.license=apache-2.0`，
模型卡标签含 `arxiv:2601.20540`。
下载方式见 `server_run/offline_prefetch.sh`。

--------------------------------------------------------------------
3. 评估用第三方预训练权重
--------------------------------------------------------------------
不随本仓库分发，各自适用上游许可：
    torchvision Inception-v3 （FID）
    torchvision R3D-18       （FVD 代理）
    LPIPS + AlexNet backbone （感知距离）
预取脚本见 `server_run/offline_prefetch.sh`。

--------------------------------------------------------------------
4. 源遥感影像数据集
--------------------------------------------------------------------
不随本仓库分发。上游条款逐条见 LICENSE 的 APPENDIX B（含明确标注的
"待确认"项）。
