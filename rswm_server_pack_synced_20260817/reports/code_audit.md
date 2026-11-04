# 代码审计结论（更新版）

## 项目识别

- 项目：RSWM / RSRoll-Bench server pack，遥感连续轨迹视频 rollout 与 LoRA 评估工程。
- 只读来源：`C:\Users\25828\Desktop\游戏\ctw\rsroll-bench-server-20260731c.tar.gz`。
- 源包 SHA-256：`FCA092BA5F294EE7FD3DA3A485484CA4716698237BCEE8ABBA9CB20B6591E9D6`。
- 选择理由：该包包含 v5 trajectory builder、双 14B I2V 专家、LoRA target 审计、
  corrected 20–36 PBS 链、Track B v2、submission contract 与标准 I3D-FVD 重算，
  比目录内 legacy 01–19 链完整。原始包没有可用于确定 commit 的 `.git` 元数据，
  因而 Git commit 为“不可核实”，不能根据文件日期伪造。
- 工程声明的正式环境：Linux x86_64、Python 3.10、PyTorch 2.4.0+cu121、
  torchvision 0.19.0、CUDA 12.1；详见 `requirements-lock.txt`。
- 本机测试环境：Python 3.13.5、PyTorch 2.11.0 CPU、无 CUDA。它只用于 CPU
  契约回归，不是论文数值复现环境。

## 实际模型与数据流

```text
AID / cross-domain RGB source image
  -> v5 continuous polyline trajectory + sliding-window pseudo video
  -> HDF5 (image, T-frame video, prompt, intrinsics, c2w/Pluecker, trajectory)
  -> frozen T5 text encoder + Wan 2.1 VAE
  -> low-noise / high-noise Wan i2v-A14B Video DiT experts
  -> LoRA adapters (all | attn_cam | attn_only)
  -> diffusion denoising loss in latent video space
  -> video rollout inference
  -> future-frame fidelity + trajectory/control metrics
```

这是“静态场景上的空间相机 rollout”，不是 Earth future dynamics prediction。

## LoRA 核实

- 已支持 PEFT LoRA；两个专家分别调用 `get_loraconfig(..., targets=...)` 后
  `get_peft_model`。
- target 不是猜测的 `q_proj/v_proj` 字符串，而由真实 Wan block 分类器选择：
  `attn_only`、`attn_cam`、`all`；训练初始化时会输出 `param_audit.json`，并在发现
  非 LoRA 参数意外可训练或零可训练参数时失败。
- 本机没有 56GB 基模和 GPU，因此真实 module 数、可训练参数总量与比例未重新实例化；
  需要在服务器 smoke run 的 `param_audit.json` 中取权威数字。

## 审计后判断

- Video diffusion backbone、视频输入输出、连续 trajectory/Pluecker 通路与 benchmark
  builder 已存在，不需要改造成另一套主干。
- “低 overlap”旧证据只约束首尾窗口；相邻帧实际仍是高 overlap。论文应把自变量定义成
  条件帧到未来帧 overlap，相邻 overlap 作为连续性审计。
- 旧 official 运行仍有三个 P0：样本 cap 未从 shell 到训练器、manifest 缺关键配方、
  server scorer 将条件帧算进保真指标。此版本已修复。
- 原工程已有外部 baseline registry/stub，但没有运行结果。MVSplat360 的多视图+pose 输入
  与本任务单图输入不公平，不进入主表。
- Full FT 不存在可验证实现；双 14B 专家不能用现有 LoRA optimizer/offload 路径简单
  `requires_grad_(True)` 后冒充公平 baseline。

## 风险分级

- P0 已处理：official split 没接训练、样本上限断链、manifest 不完整、条件帧评分虚高、
  official prompt/path 默认仍是 legacy、分布指标极小 N。
- P1 已加入基础设施：overlap/horizon 曲线、2D flow consistency proxy、rank/target 预注册。
- P1 待真跑：低 overlap 优势、任意方向、长程累积、跨域泛化都还没有数值证据。
- P2 待工程：公平 Full FT（FSDP/ZeRO）和 image extrapolation 外部 baseline adapter。
- P3：历史文档和 legacy PBS 仍较多；均保留以避免破坏旧结果，但正式入口已明确。
