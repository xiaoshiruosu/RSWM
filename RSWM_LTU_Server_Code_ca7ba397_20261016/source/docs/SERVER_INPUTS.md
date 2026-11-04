# 服务器输入准备

`python -m rswm_inria.server_inputs` 从服务器实际文件生成既有 registry、suite 和 training 输入。准备期间读取并计算源文件/权重完整 SHA，不导入原模型入口、不加载 Torch tensor、不下载、不给模型调用或科研结论签收。成功退出 0，缺输入、配置/身份错误和已存在输出退出 2。所有输出排他发表；同名文件即使内容相同也保留原件并拒绝重写。

## 数值模型配置

先按 [原 Wan 指南](LEGACY_WAN_BRIDGE.md) 把明确的 `backend_config.json` 锁入新部署树。`PREPARED_SOURCE_ONLY` 目录不能使用本入口。

```bash
python -m rswm_inria.server_inputs numeric-model \
  --source-dir /srv/rswm/build/wan-configured \
  --weights-dir /srv/rswm/weights/wan \
  --camera /srv/rswm/inputs/camera.json \
  --decisions /srv/rswm/inputs/wan-use-decision.json \
  --model-id wan-fixed-units --device cuda:0 \
  --out /srv/rswm/inputs/generated/wan-fixed-units.json
```

`camera.json` 恰有现有四字段。例如下面是一个固定单位表示配置；改变它形成新的相机身份，并仍须在对应研究范围内选择：

```json
{"translation_scaling":"fixed_units","framewise":false,"camera_channel":"native","temporal_stride":4}
```

原选定 Wan VAE 为 stride 4，因此该准备入口拒绝其他 temporal stride。`camera_channel` 可为 `native`、`identity` 或 `drop`；`translation_scaling` 可为 `fixed_units` 或 `legacy_maxnorm`，语义沿用原合同。

`wan-use-decision.json` 恰有 `approved` 布尔值与非空 `license_record` 字符串。运行者据实际授权文件填写；空白模板或 `approved=false` 被拒绝，程序不会替换为 true。使用决定应指向适用的实际基础模型/恢复适配器，不把示例记录当许可。

输出为一个现有 `numeric-camera` 模型配置，可加入 suite 的模型 JSON 数组。基础模型 manifest 的 source/weights hash 来自本次实际文件，训练相机 hash 来自同一 `CameraConfig.manifest()`。缺 T5、VAE、tokenizer、low/high 任一组件即拒绝。实际受训 v2 release 沿用其保留 binding、基础权重与训练 seed，核对相机关联，不能降级到基础模型。成功只表示输入准备完成，随后仍需新进程真权重加载。

## 固定 8 包 pilot 索引

完整 `pilot-dev2` 数据构建包含两个固定 Kitsap 世界、四 origins、32 个任务包。运行 pilot 选择 origin 0 的八包，完整审计树保持原样：

```bash
python -m rswm_inria.server_inputs pilot-index \
  --build /srv/rswm/build/pilot-dev2 \
  --out /srv/rswm/inputs/generated/pilot-index.json
rswm-suite plan \
  --task-index /srv/rswm/inputs/generated/pilot-index.json \
  --world-dir /srv/rswm/build/pilot-dev2/private/worlds \
  --models /srv/rswm/inputs/models-pilot.json --seeds 0 \
  --code-commit ACTUAL_SOURCE_COMMIT \
  --out /srv/rswm/plans/pilot-plan.json
```

选择器核对固定 revision、原 `kitsap31.tif`/`kitsap19.tif` 来源身份、角色和完整 32 包任务序列，校验实际 world 文件/像素及任务标签 hash。派生索引中 `task_path` 为绝对路径，`worlds` 保留完整父索引原记录；必须提供上例 `--world-dir`。额外 `world_root`、`parent_build` 和父身份字段供服务器 driver 使用。它不是源 TIFF→world 的独立审计，实际 experiment 仍需当前代码生成的完整来源审计。

派生索引必须存到 build 之外，否则完整审计的产物清单会被污染。程序拒绝写进原 build。它不更换固定样本、不改首次提交、不向模型提供私有世界路径。

## 训练配置派生

`training-config` 只接受完整 core120 和当前源码指纹对应的完整来源审计。它读回 retained artifact 的实际字节，并复用 `ClipDataset` 与 `validate_dev_index`；dev2 或旧源码审计无法替代。

```bash
python -m rswm_inria.server_inputs training-config \
  --spec /srv/rswm/inputs/wan-training-spec.json \
  --build /srv/rswm/build/core120 \
  --source-audit /srv/rswm/evidence/core120-current-audit.json \
  --out /srv/rswm/inputs/generated/wan-train.json
rswm-train --config /srv/rswm/inputs/generated/wan-train.json \
  --index /srv/rswm/build/core120/private/index.json \
  --clips /srv/rswm/build/core120/private/training-clips.json \
  --root /srv/rswm/build/core120/private/worlds \
  --source-audit /srv/rswm/evidence/core120-current-audit.json \
  --out /srv/rswm/runs/wan-train
```

spec 恰有以下十一字段，全部由运行者明确提供。输出是 `rswm-train` 原有十二字段配置，第十二项 `validation_index` 自动派生，不增加另一套训练协议：

| 字段 | 输入含义 |
| --- | --- |
| `updates`, `batch_size`, `accumulation`, `checkpoint_every` | 已决定的正整数更新/批量/累积/checkpoint 范围 |
| `lr`, `seed` | 正有限学习率与非负整数训练 seed |
| `precision`, `device` | 原选定 Wan 的 `bf16` 与 `cuda` 或 `cuda:N` |
| `camera` | 四键相机 JSON 文件路径 |
| `backend_manifest` | 已批准八键 backend manifest JSON，或包含该 manifest 的 numeric 模型配置文件路径 |
| `trainable_policy` | 原已冻结 LoRA 对象：`mode`, `rank`, `alpha`, `targets`, `gradient_checkpointing` |

spec 的相机与 manifest 路径相对于 spec 文件目录解析；manifest 内 source/weights 相对路径相对于 manifest 文件目录解析。输出中的实际目录变成明确绝对路径，SHA 不手抄或重标签。

部署源 `backend_config.json` 必须已有非空 `training`，spec 中 LoRA policy、precision 与原配置一致；new LoRA 使用 base 初始化，恢复 LoRA 使用对应恢复模式。dev 世界列表、验证 family/origin/seed 和累计 forward 上限沿用该 source 配置的实际冻结 `training.validation`。程序从当前 core120 派生其完整 dev 哈希记录，并核对验证 forward 上限为 `updates × dev 世界数`，不会替用户追加验证预算。

训练输出目录、恢复 checkpoint 与尝试成本仍由原 `rswm-train` 管理。恢复需保留同一目录与全部账本；本准备入口不移动或修改它们。后续导出继续使用 [原训练到推理入口](LEGACY_WAN_BRIDGE.md#训练到推理交付)。

## 源码身份与验收边界

新的 Python 实现会改变当前 `code_fingerprint()`。应对保留的同一物化树重新执行独立审计并存到新 evidence 文件；不要编辑旧 audit 的 source hash，也不需要重做已验证的 TIFF 物化。代码、输入、真正模型执行与仪器验收仍分别留证。准备工具输出的 `INPUT_PREPARED_NOT_MODEL_VALIDATED` 不意味着 GPU、训练、704 输出或正式科研通过。
