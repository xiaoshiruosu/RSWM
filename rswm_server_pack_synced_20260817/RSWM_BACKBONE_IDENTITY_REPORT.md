# RSWM Backbone Identity Audit Report

> 审计日期：2026-08-23（Asia/Shanghai）  
> 审计阶段：Phase 2 M4 evaluator 之前  
> 审计对象：当前顶层同步运行包 `rswm_server_pack_synced_20260817`  
> 审计性质：只读代码、配置与文档审计；除本报告外未修改实现  
> 阶段边界：未进入 M4，未加载真实权重，未运行 GPU 推理、训练或正式 evaluation

## 1. 结论先行

### 1.1 当前 backbone 的最准确身份表述

当前仓库中，正式 ICLR 配置指向的 checkpoint 发布身份是 **LingBot-World base-cam**，代码执行栈是项目内置 `wan` 包中的 **Wan I2V A14B / `WanModel`**，并带有显式的 Plücker 相机控制注入层。两者不是互斥命名，而是两个不同层级：

| 身份层级 | 审计结论 | 证据状态 |
|---|---|---|
| 权重/发布身份 | 配置目录名为 `lingbot-world-base-cam`；预取脚本声明 Hugging Face 仓库为 `robbyant/lingbot-world-base-cam` | **已观察（仓库声明）** |
| 架构家族 | `WAN_CONFIGS["i2v-A14B"]`、`WanModel`、Wan 2.1 VAE、low/high-noise 双 transformer | **已观察** |
| 相机条件变体 | 代码包含 `patch_embedding_wancamctrl`、全局 pose MLP 和逐 block camera injector/scale/shift | **已观察** |
| 推荐科研表述 | “LingBot-World base-cam checkpoint，基于 Wan I2V A14B 的相机条件 video diffusion 实现” | **由上述事实得到的合理推断** |
| “LingBot-Wan” | 当前代码、配置和 checkpoint 路径中未发现它作为正式类名或发布名 | **尚未验证为正式名称** |
| 精确 Wan 版本/上游 fork 谱系 | `WanBackbone.name` 写有 `wan2.2-i2v-A14B`，但当前本地没有 checkpoint 配置、权重清单或上游 Git 谱系可独立证明该版本标签 | **尚未验证** |

因此，不宜只写“Wan”，因为这会丢失实际配置绑定的 LingBot checkpoint 身份；也不应把“LingBot-Wan”写成已证实的正式模型名。本文建议对外使用：

> **LingBot-World base-cam (Wan I2V A14B camera-conditioned backbone)**

### 1.2 Adapter 命名结论

当前 `models/` 中**尚不存在** `WanWorldModelAdapter`、`LingBotWorldModelAdapter` 或 `VideoDiffusionWorldModelAdapter` 的 Python 实现；`WanWorldModelAdapter` 仅存在于已确认的 Phase 1 设计文档中。`models/__init__.py:3` 也明确说明 M1-M3 不导入具体 Wan 或外部模型。

对未来的具体实现，建议将设计名 `WanWorldModelAdapter` 改为 **`LingBotWorldModelAdapter`**，理由是当前 official 配置固定绑定 LingBot-World base-cam checkpoint。`VideoDiffusionWorldModelAdapter` 过于宽泛，更适合作为抽象基类或模型族接口，而不是这个具体 checkpoint adapter 的名字。

该建议只涉及未来抽象层身份命名，不建议在本次审计中改动现有低层 `WanBackbone`；后者封装的是 Wan 格式的加载与采样实现。若未来同一个 adapter 确实要支持多个非 LingBot 的 Wan I2V checkpoint，则应再采用能明确架构范围的名称，例如 `WanI2VA14BWorldModelAdapter`，而不是把泛化能力预先写进当前实现身份。

## 2. 审计证据边界

### 2.1 当前被审计版本

- `AGENTS.md:5` 指定当前顶层同步目录为主运行包，内部同名目录是冗余副本；本报告只审计顶层包。
- `SYNC_REPORT.md:6` 说明该包依据服务器运行记录同步，**并非从服务器直接下载**。因此本报告确认的是当前本地同步包的代码身份，不能替代服务器现状核验。
- 工作区 Git 分支没有 commit，且没有可用 remote/commit 历史；本次不能用 Git commit 作为版本锚点，只能使用文件路径、同步记录与 SHA-256。
- 本地不存在 `models/lingbot-world-base-cam/`，因此没有读取真实 checkpoint 的 `configuration.json`、subfolder config 或 safetensor 内容。

### 2.2 核心文件 SHA-256

| 文件 | SHA-256 |
|---|---|
| `lora_remotesense/new/evaluate.py` | `BD08CAC22F7C2D2EC835E2103B79982A0BA2B327C40BB63F051654770B2565E0` |
| `lora_remotesense/new/inference.py` | `9703FA88143DF80D146815555DF7D48D4B815BB942E12994D6BF8BFF185D1008` |
| `lora_remotesense/new/backbone.py` | `590A759CBCAB24A21226D052C3B6F8FB18B0F294AF9761E3B4814C97C6570F13` |
| `lora_remotesense/new/train.py` | `DF585FE0EFD083B16484CA1F5DCA1263AB29ADFE04E2228BA9FC5A0118A0DFA4` |
| `lora_remotesense/new/wan/modules/model.py` | `5C43E4496CF2E966BBB0FF82ABEE2D89F29D7717DE166D4D22BEDAA113C71D35` |
| `lora_remotesense/new/wan/utils/cam_utils.py` | `27D4801792316322E09711B31CDDE595940AB2504E5580A0CF290B72C9789951` |
| `lora_remotesense/new/wan/configs/wan_i2v_A14B.py` | `34E360422D0F445F5351BC1D0DABA28C9577228D026778C61756D882358EA933` |
| `lora_remotesense/new/wan/modules/animate/animate_utils.py` | `56860788D020B5B4F52FFB2C51A95119B5E6EC7AC096357A004021E138D38865` |
| `server_run/generate_manifest.py` | `CC1CB23B3B1BD274A5FF020AFC62F1D11FC29D941B02D631AF383FB21EFCC4DE` |
| `configs/iclr_main/primary_native_attn_cam_r16_s42.yaml` | `07B82612F86953BA5EE9F8C38D2ECA2143573B79F69D007EACEB4EAED6514BC7` |
| `models/world_model.py` | `B8DFAB9B189BAA4953E49CA4EE372EDE1D7B52C85D32DAD8862D15968D9ECD92` |

### 2.3 `wan` 模块实际解析位置

`evaluate.py` 把 `lora_remotesense/new/` 插入 `sys.path` 首位。实际 import 探测结果为：

```text
lora_remotesense/new/wan/__init__.py
```

因此正式 `evaluate.py -> inference.py` 路径使用的是 `lora_remotesense/new/wan/`，不是依赖环境中另一个同名 pip 包。去除 `__pycache__`/`.pyc` 后，该目录与仓库根部 `wan/` 均有 86 个源文件，逐文件相对路径和 SHA-256 比较差异为 0。

## 3. 真实模型名称与组成

### 3.1 已观察

1. 三个 Primary 配置均把 `model.checkpoint` 指向 `${RSWM_BASE}/models/lingbot-world-base-cam`，并使用 `native` pose、LoRA rank/alpha 16、`attn_cam` targets（例如 `primary_native_attn_cam_r16_s42.yaml:12-20`）。
2. `server_run/offline_prefetch.sh:14,56-58` 把 `robbyant/lingbot-world-base-cam` 声明为训练与评估的唯一基模源。
3. `server_run/generate_manifest.py:921-926` 对 checkpoint 目录要求：
   - `configuration.json`；
   - `Wan2.1_VAE.pth`；
   - umT5 XXL encoder；
   - `low_noise_model/`；
   - `high_noise_model/`。
4. `inference.py:99-107` 用 `WanModel.from_pretrained()` 分别加载 low/high-noise transformer；`wan_i2v_A14B.py:6-30` 将静态配置命名为 Wan I2V A14B，并声明 40 层、40 heads、dim 5120、FFN dim 13824。
5. 该配置的采样参数为 70 steps、boundary 0.947、双 expert guidance `(5.0, 5.0)`（`wan_i2v_A14B.py:32-36`）。

### 3.2 合理推断

- “LingBot”描述权重发布/模型产品身份；“Wan I2V A14B”描述代码与 transformer 架构家族。
- 当前模型不是未修改的通用 Wan I2V：本地 `WanModel` 明确包含 camera-control 专用层和 `dit_cond_dict` 路径。
- 现有 `WanBackbone.name = "wan2.2-i2v-A14B"` 是代码标签，不能单独替代 checkpoint 发布身份。

### 3.3 尚未验证

- 本地没有真实权重，未验证实际服务器加载的目录是否确为上述 Hugging Face revision 或是否被替换。
- 未读取真实 subfolder model config，因此静态 `wan_i2v_A14B.py` 的层数/维度与真实权重是否逐项一致尚未运行期确认。
- 未确认 camera-control 修改最初来自 LingBot 上游、项目本地 fork，还是另一个中间 fork。
- `REPRODUCIBILITY.md:66-73` 记录过 revision `6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd`，但当前 `offline_prefetch.sh` 没有 `--revision`；本次没有联网重新核验该 revision，也没有 checkpoint 分片哈希。

## 4. 推理调用链

### 4.1 Official 执行路线

```text
configs/iclr_main/primary_*.yaml
  └─ model.checkpoint = .../lingbot-world-base-cam
  └─ lora.target_modules = attn_cam
          ↓
server_run/generate_manifest.py
  └─ 生成 evaluate.py --checkpoint_dir ... --lora_path .../best_checkpoint
  └─ PLAN_BACKBONE=wan-inline, PLAN_POSE_MODE=native
          ↓
server_run/run_evaluation.sh
  └─ export RSWM_BACKBONE=wan-inline
  └─ export RSWM_POSE_MODE=native
          ↓
lora_remotesense/new/evaluate.py
  └─ load_val_samples(): 从 HDF5 读 image/video_frames/poses/intrinsics/prompt
  └─ run_inference_batch()
          ↓
lora_remotesense/new/inference.py::load_lora_model()
  ├─ T5EncoderModel
  ├─ Wan2_1_VAE
  ├─ WanModel.from_pretrained(.../low_noise_model)
  ├─ WanModel.from_pretrained(.../high_noise_model)
  └─ PeftModel.from_pretrained() ×2 → merge_and_unload()
          ↓
evaluate.py 每个 sample 写临时 poses.npy / intrinsics.npy
          ↓
inference.py::generate_video()
  ├─ 文本条件
  ├─ 首帧 image latent + first-frame mask
  ├─ camera pose → relative pose → Plücker ray tensor
  ├─ FlowUniPC 70-step 手工 denoising loop
  │    └─ high-noise / low-noise WanModel 按 boundary 切换
  ├─ Wan2.1 VAE decode
  └─ resize 到 480×832（如原生 decode 尺寸不同）
          ↓
evaluate.py: C,T,H,W float[-1,1] → T,H,W,C uint8
```

### 4.2 Pipeline 形态

这里没有实例化 Diffusers 的通用 `DiffusionPipeline` 对象。实际 pipeline 是 `evaluate.py` 与 `inference.py` 组成的程序化调用链，去噪循环在 `generate_video()` 内手工执行。

### 4.3 `wan-inline` 与 registry 路径

- `generate_manifest.py:981` 固定 official 路径为 `wan-inline`。
- `evaluate.py:472-490` 对 `wan-inline` 直接调用 `load_lora_model()`/`generate_video()`，因此 official 当前不经过 `backbone.py` 的 `WanBackbone`。
- 只有 `RSWM_BACKBONE != "wan-inline"` 时才进入注册表路径。

## 5. Image condition 如何进入模型

### 5.1 已观察

1. HDF5 的 `image` 被转为 PIL，再由 `TF.to_tensor()` 转成 `[-1,1]`（`inference.py:217-218`）。
2. 代码构造 `[首帧 RGB, 后续全零帧]` 的条件视频并用 VAE 编码（`inference.py:324-332`）。
3. 同时构造首帧为 1、未来帧为 0 的 stride-aligned mask；最终 `y = cat(mask, encoded_condition)`（`inference.py:245-252,333`）。
4. conditional 与 classifier-free unconditional 两个分支都携带同一个 `y`（`inference.py:346-357`）；负条件只替换文本 context，不删除首帧条件。
5. `WanModel.forward()` 在 3D patch embedding 之前按通道把噪声 latent `x` 与 `y` 拼接（`model.py:465-476`）。

### 5.2 结论

Image condition 是 **VAE latent 级首帧条件 + 首帧 mask**，直接作为 transformer patch-embedding 输入的一部分；它不是只通过文本 cross-attention 进入模型。

## 6. Trajectory、camera pose 与 Plücker 注入

### 6.1 数据侧 pose 来源

已观察：canonical builder 优先从逐帧滑窗中心 `trajectory (T,2)=(cy,cx)` 生成 `(T,4,4)` c2w；旋转保持单位阵，平移编码滑窗运动（`build_dataset_from_benchmark.py:947-964,1411-1433`）。仅旧源缺 trajectory 时才回退 waypoint 插值。内参保存为 `(T,4)` 的 `[fx,fy,cx,cy]`（`build_dataset_from_benchmark.py:988-991,1418,1442-1443`）。

### 6.2 推理侧 pose 处理

已观察：

1. `evaluate.py:521-522` 把样本 `poses`/`intrinsics` 写入临时 action 目录。
2. `generate_video()` 默认 `framewise=True`（`inference.py:173-191`）。
3. c2w 先插值到 latent frame 数，再经 `compute_relative_poses()`：先相对首帧，启用 framewise 时再转为相邻帧变换；平移按最大范数归一化（`cam_utils.py:54-70`）。
4. 内参会从 480×832 合同坐标变换到内部生成尺寸。
5. `pose_mode=drop` 时整条 pose 通路置空；`identity` 和 `native` 都走相同 Plücker 路径，差异来自 pose 内容。

### 6.3 Plücker ray 是否使用

**是，已观察。** `get_plucker_embeddings()` 对每帧、每像素计算：

```text
direction = d
origin = o
moment = o × d
plucker = concat(d, o × d)    # 6 channels
```

证据位于 `cam_utils.py:86-110`。所得 6D tensor 被重排到 latent 网格，并作为 `dit_cond_dict["c2ws_plucker_emb"]` 传给两份 Wan transformer（`inference.py:295-322,346-357`）。

### 6.4 Pose embedding 在哪里注入

存在两级注入，均为已观察：

1. **全局 pose embedding**（`model.py:396-399,508-528`）  
   Plücker patch 先经过 `patch_embedding_wancamctrl`，再经过两层 SiLU MLP 残差：`c2ws_hidden_states_layer1/2`。
2. **每个 Wan attention block 内注入**（`model.py:220-223,249-265`）  
   每个 block 使用 `cam_injector_layer1/2` 生成 camera hidden state，再由 `cam_scale_layer` 与 `cam_shift_layer` 对 self-attention residual 后的 token 做：

```text
x = (1 + cam_scale) * x + cam_shift
```

之后才进入 cross-attention 与 FFN。因此 pose 不是只在输入端拼接一次，而是在全局投影后传入每一个 transformer block 进行调制。

## 7. LoRA target modules

### 7.1 Primary 配置

三个正式 seed 配置均为：

```text
rank = 16
alpha = 16
target_modules = attn_cam
```

### 7.2 `attn_cam` 的实际筛选规则

`get_loraconfig()` 先枚举名字含 `blocks` 的全部 `nn.Linear`，排除 `face` 与 `modulation`，然后为 `attn_cam` 保留以下名字族（`animate_utils.py:20-57`）：

| 目标族 | 具体静态模块名 |
|---|---|
| Self-attention | `blocks.N.self_attn.{q,k,v,o}` |
| Cross-attention | `blocks.N.cross_attn.{q,k,v,o}` |
| Camera injector | `blocks.N.cam_injector_layer1`, `cam_injector_layer2` |
| Camera affine | `blocks.N.cam_scale_layer`, `cam_shift_layer` |

明确不包含：

- FFN/MLP 的 `blocks.N.ffn.*`；
- 全局 `patch_embedding_wancamctrl`；
- 全局 `c2ws_hidden_states_layer1/2`。

训练时 low/high-noise 两个 expert 分别执行 `get_peft_model()`；推理时分别加载 `low_noise/` 与 `high_noise/` adapter，并各自 `merge_and_unload()`（`train.py:350-380`；`inference.py:110-138`）。

由于本地没有真实 checkpoint，本报告没有把静态配置推算出的模块数写成“实际参数数”；最终展开的 target 列表和 `param_audit.json` 数量仍属**尚未验证**。

## 8. Benchmark abstraction 审查

### 8.1 已观察

- Phase 1 设计在 `RSWM_PHASE1_BENCHMARK_ARCHITECTURE.md:1001-1003` 冻结了两层抽象：benchmark `WorldModel` 门面组合现有 `WanBackbone`，设计名为 `WanWorldModelAdapter`。
- 当前 `models/world_model.py` 只有通用 `WorldModel` Protocol；`models/registry.py` 只有空的 factory registry；`models/` 中只有 deterministic mock。
- 在 `models/`、`lora_remotesense/`、`server_run/`、`benchmark/`、`tasks/` 和 `tests/` 的 Python 源码中搜索三个候选 adapter 名称，匹配数为 0。
- 现有低层 `WanBackbone` 仍是参考薄封装，注册名为 `wan` 与 `wan2.2-i2v-A14B`（`backbone.py:475-528,945-968`）。

### 8.2 命名建议（推断，不是已实施事实）

| 名称 | 适用性判断 |
|---|---|
| `LingBotWorldModelAdapter` | **建议用于当前 official 具体 adapter**；保留 checkpoint 发布身份 |
| `WanWorldModelAdapter` | 只表达架构家族，无法区分 LingBot checkpoint 与其他 Wan-compatible checkpoint |
| `VideoDiffusionWorldModelAdapter` | 过宽；适合作为抽象层，不适合作为当前具体实现名 |

为满足 P1-16/P1-17，未来 adapter identity 至少应分别记录：checkpoint repo/local identity、固定 revision 或分片哈希、transformer 类与配置哈希、camera-control 代码哈希、low/high LoRA adapter 哈希、采样配置以及执行路线（`wan-inline` 或 adapter）。这是审计建议，不是本次实现内容。

## 9. 优先级发现

### F-01 — checkpoint 身份尚未形成可验证闭环（P1，高影响）

- **已观察：** official 配置只给目录路径；预取脚本下载 HF `main`，没有 `--revision`；当前 manifest 的代码哈希不覆盖 checkpoint 内容。`REPRODUCIBILITY.md:60-85` 也明确记录了这一缺口。
- **影响：** P1-16 release identity 与 P1-17 artifact lineage 无法证明两次结果使用了相同 base weights。
- **可能性：** 中；上游更新或本地目录被替换时，现有身份字段不能检测。
- **最小后续动作：** 在进入正式 release/evaluation 前绑定固定 revision，并生成 checkpoint 分片 SHA-256 manifest；本次未实施。

### F-02 — 非 inline 的 Wan registry 路径不能按当前调用直接构造（P1，集成阻断）

- **已观察：** `evaluate.py:474-478` 调用 `build_backbone(_BACKBONE)` 时没有传 kwargs；`WanBackbone.__init__()` 在 `backbone.py:500-503` 要求必填 `checkpoint_dir`。
- **影响：** `RSWM_BACKBONE=wan` 或 `wan2.2-i2v-A14B` 会在构造阶段缺少参数。当前 official `wan-inline` 不受影响，但未来真实 `WorldModelAdapter` 不能把这条路视为已接通。
- **最小后续动作：** 在获批实现 concrete adapter 时显式传递 checkpoint/LoRA/runtime 参数并做 CPU/mock 参数转发测试；本次未修改。

### F-03 — 一个字符串同时承担发布身份与架构身份（P2，可审计性）

- **已观察：** checkpoint 配置使用 `lingbot-world-base-cam`，manifest 执行标签为 `wan-inline`，`WanBackbone.name` 又是 `wan2.2-i2v-A14B`。
- **影响：** 论文、leaderboard、manifest 若只记录一个 `backbone` 字段，可能把执行路线、架构和权重发布版本混为一谈。
- **最小后续动作：** 在 P1-16 identity 中分字段记录，而不是用一次改名代替身份链；本次未修改。

### F-04 — camera fork 的上游来源未证实（P2，来源说明）

- **已观察：** camera-control 层真实存在且进入前向计算。
- **尚未验证：** 这些层最初属于 LingBot 官方实现还是项目/第三方 fork。
- **影响：** 不影响静态调用链判断，但影响方法归属、引用和 release note 的准确性。
- **最小后续动作：** 取得 checkpoint README、固定 revision 源码或上游 commit 后再归属；在此之前仅写“package-local Wan-derived camera-conditioned implementation”。

## 10. 实际检查与结果

| 检查 | 真实结果 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| Python import resolution | `wan` 解析到 `lora_remotesense/new/wan/__init__.py` | 当前 evaluate/inference 使用的源码包位置 | 真实权重来源 |
| 两份 `wan/` 源码哈希对比 | 86 vs 86，差异 0 | 顶层与 package-local 源文件当前一致 | 上游 fork 谱系 |
| checkpoint 存在性 | `models/lingbot-world-base-cam` 不存在 | 本地不能做真实权重核验 | 服务器目录状态 |
| `python server_run/test_backbone_protocol.py` | 最终结果 `86 PASS, 0 FAIL` | mock/CPU 协议、参数转发与输出合同 | 56GB 真权重加载、GPU 数值或正式 evaluation |

说明：该协议测试最初在受限临时目录因 Windows ACL 无法写 `poses.npy`；改为获准的非受限测试执行后得到上述 86/0。测试自身也明确标注真实 Wan 权重端到端仍未验证。

## 11. 审计判定与停止点

### 已确认

- 当前 official checkpoint 的**仓库声明身份**是 LingBot-World base-cam。
- 当前实际代码架构是 package-local Wan I2V A14B `WanModel`，包含 low/high-noise 双 expert。
- image condition 以 VAE 首帧 latent + mask 在 patch embedding 前进入模型。
- native/identity pose 会生成 6D Plücker ray，先全局编码，再注入每个 transformer block；drop 不传 pose。
- Primary LoRA targets 是 attention projections + block 内 camera injector/scale/shift，不含 FFN 和全局 camera projection。
- official 评测当前仍走 `wan-inline`，未使用计划中的 benchmark `WanWorldModelAdapter`。

### 尚未确认

- 服务器实际 checkpoint 内容、固定 revision 和分片哈希；
- 真实 checkpoint 的完整 transformer config 与静态配置是否逐项相同；
- camera-control fork 的上游归属；
- 真权重 `WanBackbone`/未来 `WorldModelAdapter` 的 GPU 端到端行为；
- 正式 evaluation 数值。

### M4 前结论

Backbone 的**静态身份与调用链审计已完成**，但 benchmark release identity 仍不是 verified/release-ready。建议在未来实现具体 adapter 时使用 `LingBotWorldModelAdapter`，并将架构身份另记为 Wan I2V A14B camera-conditioned；不要用 `VideoDiffusionWorldModelAdapter` 作为当前具体模型名。

本报告到此停止，**未进入 Phase 2 M4 evaluator**。
