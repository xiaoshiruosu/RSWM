# ReproductionGuide

## 软件重现

```bash
python -m pip install -e '.[dev,data,motion]' build 'setuptools>=68' wheel
python -m pytest -q --junitxml=local-tests.xml
python -m build --no-isolation
python -m venv /tmp/rswm-release-env
/tmp/rswm-release-env/bin/python -m pip install dist/*.whl
cd /tmp
/tmp/rswm-release-env/bin/rswm-acceptance --out rswm-cpu-new
```

最后命令脱离checkout执行已安装包，默认实际F2/F7 Oracle 192输出并replay、资源检查、出表。包内`_build_info.json`记录真实source commit、source SHA与dirty状态；构建验收必须对应具体commit。GPU/模型/实景不在默认CPU软件通过范围。

## 独立 CPU Torch 工程验证

基础安装不要求 Torch。需要执行可选适配器接线与本地 TorchScript 控制时，使用独立环境；CI 的 `torch-cpu-controls` 作业使用本轮实际测试的官方 CPU 版本：

```bash
python -m venv .venv-torch-cpu
.venv-torch-cpu/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
.venv-torch-cpu/bin/python -m pip install -e '.[dev,data,motion]' build 'setuptools>=68' wheel
.venv-torch-cpu/bin/python -m pytest -q --junitxml=torch-cpu-tests.xml
```

CPU Torch 可用不代表批准原 backend、真实权重或 GPU 可用。完成下方 quality synthetic reproduction 后，可从当前 checkout 执行实际特征 CLI 控制：

```bash
.venv-torch-cpu/bin/python scripts/reproduce_torch_feature_cli.py \
  --mosaics runs/quality-new/generated/mosaics.json \
  --reference-mosaics runs/quality-new/reference/mosaics.json \
  --work-dir runs/torch-feature-raw-new \
  --evidence-dir runs/torch-feature-receipt-new
```

该脚本只接受 `diagnostic_only` 输入，生成明确标记的本地 channel-mean TorchScript 控制模块，实际执行提取、缓存复用、14 个 controls、质量统计，以及缺失权重/损坏权重/损坏缓存的非零退出。原始控制模块和缓存留在 work 目录。全部结果 `diagnostic_features=true`、`release_validated=false`，不能作为真实特征仪器 release。命令与 stdout/stderr 独立留证；重复执行使用新目录。

## 原数据和smoke12

依[README](../README.md)和[G2指南](G2_DATA_ACCEPTANCE.md)顺序freeze、first-source、materialize、audit和smoke12 `--execute`。源必须真实存在且通过完整inventory；源/输出预算先核查。不要将`--task-limit`子集或synthetic fixture作为正式门。所有新输出用新目录，避免覆盖原文件。

## 最终工程CLI证据

以下命令均产生明确标记的synthetic工程对照，不读取/下载真实源或模型。先完成全部源码编辑；计划的完整source fingerprint在执行和replay之间必须一致。保留原失败目录，重跑选新位置。

```bash
python evidence/build-closure/reproduce_export_cli.py --out runs/g1-fault-new
python evidence/build-closure/reproduce_g2_synthetic.py --out runs/g2-subset-new
python scripts/g3_research_acceptance.py --work-dir runs/g3-raw-new --evidence-dir runs/g3-receipt-new
python scripts/reproduce_motion_protocol.py --work runs/motion-raw-new --out runs/motion-receipt-new
python evidence/build-closure/G5-quality/reproduce_quality_pipeline.py --out runs/quality-new
python scripts/g7_diagnostics.py --out runs/g7-new
python scripts/reproduce_worker_acceptance.py --out runs/worker-new
```

CLI健康验收返回0；坏run资源/outcome在保留分母的可导出结果上返回1；坏root plan返回2且不发布分析。受限OS probe失败保持BLOCKED；实际截图/输出不足不能补写参考。原审核包的固定旧文件导入脚本仅用于证明历史bug，修复回归由当前工作树的pytest执行。

## 实际模型、训练与正式运行

features依赖本地批准TorchScript及冻结预处理，generate依赖本地批准原backend和真实权重。manifest与rights必须实际填写，不能填测试文件、假SHA或以噪声fallback替代。数值camera虚拟几何、字段及实际原模型接口见[AdapterGuide](AdapterGuide.md)。

```bash
rswm-train --config training.json --index build/core120/private/index.json --clips build/core120/private/training-clips.json --root build/core120/private/worlds --source-audit core120-audit.json --out runs/train-new
rswm-experiment freeze --plan dev-plan.json --kind pilot --purpose real --inputs pilot-inputs.json --out pilot-experiment.json
rswm-experiment run --plan dev-plan.json --experiment pilot-experiment.json --out runs/real-pilot-new
```

训练config必须完整冻结updates/batch_size/accumulation/lr/seed/precision/device/camera/trainable_policy/checkpoint_every/backend_manifest/validation_index。loader锁实际60训练世界及同audit的dev来源；checkpoint为优化器边界，包含模型/优化器/scheduler/scaler/Python/NumPy/Torch/CUDA RNG、cursor和独立SHA sidecar。恢复用`--resume CHECKPOINT.pt`，预算与cursor不变。当前已有独立 CPU Torch 工程验证环境，但批准原objective/weights与真实数据缺失，原模型实际梯度、训练恢复和GPU均NOT_RUN。

正式experiment freeze inputs包括budget、core120 source_audit、rights、每当前model实际pilot_receipts、两仪器实际metric_receipts、显式instrument_config_sha256及真实smoke12_evidence_root。它重验真实576run/50,688输出的三诊断门和保留pilot PNG/资源/analysis链；薄PASS字典或自己重新算digest不能替代实际证据。resume采用持久累计预算与新invocation分析/receipt，不覆盖首轮。正式RGB执行完成状态仍是`EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS`，不自动发布论文结论。

motion用Bellingham calibration、Bloomington validate及曝光ledger。quality先实际feature calibration，再冻结reviewed acceptance-config与held-out patch计划，使用`rswm-release-features validate`；无approved权重或reviewed范围时BLOCKED。具体合同见[计量协议](MeasurementProtocol.md)。

`rswm-acceptance --profile metric-validation --inputs instruments.json --out runs/instruments-new` 的motion对象提供manifest/calibration/root/exposure_ledger；quality对象提供config/acceptance_config/reference_mosaics/patch_plan/suite_plan/source_audit/cache/calibration_receipt/exposure_ledger。该profile实际运行两held-out验证及严格release validators。其他profile需要各自实际输入；full-build会独立记录所有门，不把坏一个输入隐藏成其余门通过。

真实数据、权重、设备或系统能力缺失时保存NOT_RUN/BLOCKED原因、命令、返回码、环境与hash，继续纯工程路径。最终报告以[阶段证据](../evidence/build-closure)为准，不能用旧CI或不同source snapshot给当前commit背书。

当前构建 provenance 将 README、docs、tests、MANIFEST 的未提交改动也计为 dirty。物化 PNG 完整编码后独占发布，已有产物不被覆盖；保留 hash 和独立像素/文件清单审计仍是必需条件。历史两次 smoke12 物化审计拒绝记录见 `evidence/build-closure/G2-real-source-continuation-report.md`，失败目录保持原样。其后 f1 源码已在新目录完成真实 smoke12 三诊断 576 runs / 50,688 outputs；该验收不代表真实生成模型，也不签收当前后继源码的 core120。

## 2026-10-01 指标消费与实际分发

运动和质量评价使用冻结 suite 绑定的结果生产者，再由 `rswm-export --motion-result ... --quality-result ...` 独立复核保留输入。完整命令、统计单位及诊断边界见 [C05 指标消费](C05_METRIC_CONSUMPTION.md)。仪器收据本身不生成模型成绩；没有有效评价时 RGB 继续导出，正式指标保持 `NOT_AVAILABLE`。

最终实现提交固定后，由 [C07 分发留存](C07_DISTRIBUTION_RETENTION.md) 构建实际 wheel/sdist、保存归档成员映射和 SHA256SUMS，并下载读回验收。CI 对外层 artifact ZIP 和内部两个包分别核对 hash，在源码目录外分别安装 wheel 与实际重建的 sdist。提交前 prototype 包、旧提交 CI 或合成贯通均不能替最终 HEAD 签收。
