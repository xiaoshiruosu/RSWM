# A800 compatibility audit

日期：2026-08-14  
审查对象：`rsroll-bench-iclr-20260813-a800`  
定位：A800 服务器部署前小范围审查；不重构模型，不修改论文任务和 benchmark 协议。

## 1. 结论

代码层部署审查 **PASS**。Primary validated hardware 明确为
**NVIDIA A800-SXM4-80GB**（sm80、BF16、FlashAttention-2→SDPA）；
H200 141GB 的 sm90/FA3 路径保留为 legacy/reference。本轮没有 A800/H200 实机训练，
因此不能把代码审查 PASS 写成实验 PASS。

本轮唯一实质运行策略变化是把 A800 的硬锁 `OFFLOAD=1` 改成
`OFFLOAD=auto|0|1`：

- `auto` 只有在当前可见逻辑 GPU 同时满足总显存 ≥97000MiB、空闲显存
  ≥90000MiB 时才解析为 0。
- A800-80GB 的 `auto` 必然解析为 1，仍是正式默认安全路径。
- 显式 `OFFLOAD=0` 被保留用于用户批准的吞吐/OOM 边界测试；会打印警告，
  未通过真实 33 帧 smoke 前不得进入正式矩阵。
- A800 smoke 固定 `OFFLOAD=1`，不会因自动探测改变测试口径。

## 2. 兼容性审计

| 文件/模块 | 问题或证据 | 风险 | 是否修改 |
| --- | --- | --- | --- |
| `server_run/a800_env.sh` | A800 原先硬锁 `OFFLOAD=1` | 不满足 auto/0/1 和吞吐边界测试要求 | 是：默认 auto，校验三态；不自动选 GPU |
| `server_run/common.sh` | 只接受 0/1，且 A800 profile 拒绝 0 | 无法按当前可见卡总/空闲显存决策 | 是：按逻辑 GPU total/free 解析，记录 requested/resolved |
| `lora_remotesense/new/train.py` | profile 无条件覆写 `inactive_model_cpu=True` | launcher 的显式 0 无效；运行身份不完整 | 是：支持 `--offload auto/0/1`，保留旧 flags，manifest/checkpoint 记录解析结果 |
| `run_train.sh`、`run_p0.sh` | 旧变量默认 1，A800 0 被拒绝 | 与最终部署策略不一致 | 是：OFFLOAD 为主，旧 `INACTIVE_MODEL_CPU` 仅作兼容别名 |
| `run_a800_smoke.sh` | 部署默认改 auto 后，smoke 口径可能漂移 | smoke 不能稳定证明安全路径 | 是：显式固定 `OFFLOAD=1` |
| `a800_80g.json`、`h200_141g.json` | 没有机器可读的 offload policy | profile 审计不完整 | 是：两者增加 `offload: auto`，H200 原优化保留 |
| `wan/modules/attention.py` 及镜像 | FA3 导入存在 | A800 若误进 sm90 路径会失败 | 否：已有 capability guard；sm80=FA2，sm90 可 FA3，异常回退 SDPA |
| `animate/face_blocks.py`、`s2v/auxi_blocks.py` 及镜像 | 辅助 attention 绕过主入口的历史风险 | ABI/kernel 失败可能中止 | 否：现有版本已捕获导入/ABI/kernel 异常并回退 SDPA |
| `server_run/preflight_a800.py` | 未完整报告 XDG/Triton cache、offload 和 sm90 env | 部署前无法发现缓存或 Hopper-only 环境误配 | 是：补齐路径、offload、attention policy、FP8/sm90 检查 |
| `scripts/`、`tests/` | 用户指定入口缺失 | 部署说明与实际目录不一致 | 是：增加无重复实现的稳定 wrapper |
| `server_run/semgate_arbiters.py` | 读取 `~/.cache/huggingface` 候选 | 共享服务器可能误用 home cache | 是：只读取 HF 环境变量指定目录 |
| README 共 12 份 | H200 历史表述仍可能像当前主硬件；存在 home/tmp 示例 | 读者误用旧路径或误判验证状态 | 是：全部加 Primary A800 / Legacy H200 标记并修正活跃示例 |
| `requirements-a800-verified.txt`、`environment-a800.lock.json` | 关键栈已锁；硬件资源信息不完整 | 环境身份缺 CPU/RAM/offload policy | lock 不改版本；JSON 补 CPU、128 threads、1TiB 与 runtime policy |
| `run_distributed.sh` 及镜像 | 多卡入口可见 | 共享节点可能误占 4 卡 | 否：已有显式 `NUM_GPUS`/CVD 守卫，不自行设置 CVD |
| 41 个 `.pbs` + legacy 文档 | `gpuhopper` 队列属于旧 H200 集群 | 不能猜测 A800 队列名 | 代码不改；PBS README 明确 legacy，等待管理员队列信息 |
| FP8/TE/sm90 编译/`torch.compile` | 未发现实际 FP8、TransformerEngine 或 sm90-only 自定义编译 | 无需为 A800 删除不存在的路径 | 否；profile 继续 `fp8=false`、`torch_compile=false` |
| `cuda:0`、`cuda:1`、全四卡硬编码 | 运行源码复查 | 可能越过 CVD 或占用他人 GPU | 否：运行源码搜索为 0；历史审计文档中的旧代码引用保留 |

## 3. 修改文件

相对本轮审查前的 A800 交付版：**36 个文件修改、3 个文件新增、0 删除**。

新增：

```text
A scripts/preflight_a800.py
A tests/test_a800_smoke.py
A reports/A800 compatibility audit.md
```

修改：

```text
M environment-a800.lock.json
M lora_remotesense/build_overlap_benchmark.py
M lora_remotesense/new/configs/hardware/a800_80g.json
M lora_remotesense/new/configs/hardware/h200_141g.json
M lora_remotesense/new/README.md
M lora_remotesense/new/run_p0.sh
M lora_remotesense/new/run_train.sh
M lora_remotesense/new/train.py
M lora_remotesense/new/wan/remotesensebenchmark/download_rsvideowise_data.py
M lora_remotesense/new/wan/remotesensebenchmark/README.md
M README_ICLR_MODIFICATIONS.md
M README_MODIFICATIONS.md
M README_RUN.md
M README_v30.md
M README_v31.md
M reports/A800_H200_AUDIT_20260814.md
M REPRODUCIBILITY.md
M server_run/A800_80G.md
M server_run/a800_env.sh
M server_run/baselines/README_BASELINES.md
M server_run/build_correct.sh
M server_run/calibrate_semantic_gate.py
M server_run/check_resisc45.py
M server_run/common.sh
M server_run/offline_prefetch.sh
M server_run/pbs/README_PBS.md
M server_run/preflight_a800.py
M server_run/README_RESISC45.md
M server_run/README_RUN.md
M server_run/run_a800_smoke.sh
M server_run/run_all.sh
M server_run/semgate_arbiters.py
M server_run/test_a800_profile.py
M SUBMISSION.md
M wan/remotesensebenchmark/download_rsvideowise_data.py
M wan/remotesensebenchmark/README.md
```

这些修改只涉及硬件策略、路径、检查、兼容入口和文档；没有修改
continuous trajectory-conditioned rollout、`net_continuous`、registration、
low-overlap、camera/Plücker conditioning、LoRA rank/target 消融、外部 baseline
接口或数据集/评分协议。

## 4. 最终环境版本

| 项目 | 固定值 |
| --- | --- |
| Python | 3.10.20 |
| torch | 2.4.0+cu121 |
| torchvision | 0.19.0+cu121 |
| CUDA runtime / driver CUDA | 12.1 / 12.2 |
| Driver | 535.261.03 |
| flash-attn | 2.7.3 |
| transformers | 4.51.3 |
| tokenizers | 0.21.4 |
| diffusers | 0.39.0 |
| accelerate | 1.14.0 |
| peft | 0.19.1 |
| numpy | 1.26.4 |
| huggingface_hub（训练） | 0.36.2 |
| 下载环境 | Python 3.11.15 + huggingface_hub 0.35.3 |

torch、torchvision、flash-attn 未修改，也没有执行 pip 安装或升级。

## 5. A800 默认配置

| 参数 | 默认值 |
| --- | --- |
| precision | BF16 |
| micro batch | 1 |
| gradient accumulation | 4 |
| gradient checkpointing | true |
| frames | 33 |
| attention | auto：FA2→SDPA |
| offload | auto；A800-80GB 解析为 1 |
| smoke offload | 1（固定） |
| T5 / VAE | CPU，串行 swap |
| optimizer | torch.optim.AdamW（A800） |
| workers / prefetch / persistent | 0 / disabled / false |
| pin memory | false |
| torch.compile / FP8 | false / false |
| cache/tmp 根 | `/nix/rsroll_25828` |

## 6. 测试状态

### PASS

- 230 个 Python 文件 source compile。
- 14 个 JSON 文件解析。
- 10 个 CPU 回归脚本全部退出 0；A800 profile 专项 35 项断言通过（含
  A800 auto→1、空闲 H200 auto→0、空闲不足 H200 auto→1 三种合成显存分支）。
- backbone、ICLR protocol、trajectory、known-answer metrics、semantic gate、
  submission contract、taxonomy、well-posed 双栏均为 0 FAIL。
- LPIPS 已知答案有 1 项因本机无 `lpips` 包按设计 SKIP，不计 PASS，也不计失败。
- 公共 attention 入口的 variable-length CPU SDPA fallback：shape、finite、padding 通过。
- root `wan/` 与 `new/wan/` 共 86 个对应文件 SHA-256 一致，0 drift。
- 运行源码中的 `cuda:0`、`cuda:1`、固定四卡、`/root` 路径扫描为 0。
- 12 份 README 均明确 Primary A800 / Legacy H200。
- 标准 wrapper 可执行 `--help`；本机非 A800 preflight 按设计报告 FAIL，
  没有把 RTX 5060/Python3.13/CPU torch 环境误报为可部署。

### FAIL

- 无代码断言失败。

### NOT TESTED

- Linux `bash -n`/shellcheck：本机 WSL 启动返回 `E_ACCESSDENIED`；需在服务器执行。
- A800 smoke Level 1/2/3：当前机器不是 A800 且缺固定训练依赖。
- 真实 33 帧、1 train + 1 val 权重 smoke。
- `OFFLOAD=0` 的 A800 resident throughput/OOM 边界。
- 正式训练、长时间稳定性、峰值显存、吞吐和收敛。
- H200/FA3 实机回归、多 A800、FSDP/model parallel。
- 真实外部 baseline 和完整第二数据集结果。

## 7. A800 服务器下一步

管理员分配一张空闲 A800 后：

```bash
cd /nix/rsroll_25828/<代码目录>/rswm_server_pack
export CUDA_VISIBLE_DEVICES=<管理员分配的单个物理卡号>
source server_run/a800_env.sh
python scripts/preflight_a800.py --strict --require-single-visible --min-free-gib 70
MIN_FREE_GIB=70 SMOKE_LEVEL=3 bash server_run/run_a800_smoke.sh
bash server_run/capture_a800_environment.sh
```

到此停止。只有上述步骤全部 PASS 且用户再次批准，才运行
`REAL_MODEL_SMOKE=1`；仍不直接启动正式训练。
