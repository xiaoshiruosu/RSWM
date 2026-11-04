# User Requirements — RS-Roll Baseline Extension
> Created: 2026-08-16 | Project: RS-Roll / Remote Sensing Video World Model
> 由对话中的明确要求整理；本文件只记录约束，不记录实验结论。

## 阶段 A：方向与 RQ 约束

### 研究方向约束

**希望侧重的方向**

以 Benchmark / Task Definition 为核心贡献，评估单张遥感图像与连续相机轨迹条件下的连续视频 rollout，重点覆盖 long-horizon 与 reduced-overlap 条件。

**不希望涉及的方向**

不重写 Wan/LingBot Video Diffusion 主干，不增加未经必要性证明的大型显式 3D geometry 模块，不提前声称视频模型优于图像外推模型。

**已确认研究方向**

将现有代码扩展为可对多类 baseline 进行公平、可审计比较的 RS-Roll Benchmark 平台。

### 研究问题（RQ）约束

RQ 必须能够区分视频先验、相机轨迹条件、LoRA target/rank、低重叠鲁棒性以及 AID 到 WHU-RS19/RESISC45 的跨数据集泛化。

### 明确参考论文

Remote Sensing-Oriented World Model / RSWISE；只使用可核实的公开资料，无法获得的代码或 checkpoint 必须标注为不可复现。

### 文档偏好

中文说明为主，代码字段与论文术语保留英文；所有未运行实验标记 `NOT RUN` 或 `NOT VERIFIED`。

## 阶段 C：硬件约束

### GPU 约束

正式环境为 A800-SXM4-80GB；本地阶段禁止访问 GPU 或启动训练。

### 训练时长上限

本次任务不执行训练。

## 阶段 D：实现约束

### 基于已有项目

基于 `rsroll-bench-iclr-20260815-a800-whurs19-experiment-runner-safety-patched` 的独立副本增量开发。

### 框架偏好

保持现有 Python/PyTorch 项目与接口。

### 其他硬性要求

- 核心 Video DiT、VAE、attention、3D RoPE、camera/Plücker 逻辑保持不变。
- 所有方法使用同一冻结 test manifest 与 future-only evaluator。
- adapter 存在不等于 checkpoint、运行验证或论文结果存在。
- `FVD_R3D_PROXY` 与标准 `FVD_I3D` 严格分开。
- 所有修改在独立版本中交付到 `C:\Users\25828\Desktop\lora`。

## 阶段 E：ICLR 主实验与 A800 运行层

- 基于已完成的 `baseline-extension` 版本继续增量开发，不覆盖原版本。
- 生成可被正式 experiment runner 直接解析的 ICLR 主实验配置。
- 主配置包含 native + attn_cam + rank 16 的 42/43/44 三训练种子，并固定
  split seed、generation seed、数据、预算和评估样本数。
- 提供 matched pose×target、all-target 补充臂和 rank 4/8/16/32/64 配置；
  完全相同的 canonical r16 点只运行一次。
- A800 脚本默认必须是 plan/dry-run，正式训练、baseline 和 evaluation 分别
  需要显式确认；不得在本地启动 GPU、训练或下载。
