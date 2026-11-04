# C05 保留结果消费与分表

`rswm-export` 保持已有 RGB 导出接口。可选的运动和质量输入必须是实际模型评价产物路径；仪器发布收据本身不能产生模型分数。未提供输入时照常导出 RGB，并记录 `NOT_AVAILABLE` 及原因。

在完整源码固定后，由同一冻结 suite plan 和保留 suite 产生结果。运动继续使用原发布配置和受限范围；质量使用预先冻结的图块集合和实际批准的特征配置。正常运行与故障控制使用不同目录。

```bash
rswm-motion-validate evaluate-suite --plan runs/plan.json --suite runs/suite \
  --config instruments/motion-config.json --validation instruments/motion/validation.json \
  --selection first_attempt_only --out runs/motion-model
rswm-quality --generated runs/features-generated/features.json \
  --reference runs/features-reference/features.json --suite-plan runs/plan.json \
  --suite runs/suite --generated-mosaics runs/mosaics-generated/mosaics.json \
  --reference-mosaics runs/mosaics-reference/mosaics.json --patch-plan runs/patch-plan.json \
  --validation-receipt instruments/features/validation.json \
  --feature-config-sha256 "$FEATURE_CONFIG_SHA256" --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --cap "$FROZEN_SAMPLE_CAP" --bootstrap "$FROZEN_BOOTSTRAP" --seed "$FROZEN_STATISTICAL_SEED" \
  --selection first_attempt_only --out runs/quality-model.json
rswm-export --plan runs/plan.json --suite runs/suite --out runs/tables \
  --motion-result runs/motion-model/motion.json --quality-result runs/quality-model.json
```

重复 `--motion-result` 或 `--quality-result` 可输入独立 attempt 选择的结果；重复结果、重复组和配置混用会失败。恢复分析必须由计划允许的 `latest_attempt_within_budget` 重新生成，保留首轮与恢复的独立身份。分布统计按实际冻结集合及 training/inference seed 分组，不广播成逐任务分数。F7 共同前缀只计一次。

消费者重新核对完整 expected run 集合、model/weights manifest、任务和源集合、study、seed、attempt、指标配置、当前代码和预处理身份，并检查实际保留文件。运动重新从提交 PNG 测量；质量重新组装首次提交图块和真实参考，重算保留特征的 MMD-U。正式质量还重新加载冻结权重、执行特征推理及校准/留出控制。正式运动重验保留校准、留出像素、发布收据和曝光账本；读表不会增加曝光。历史缺少保留输入的收据仍保留，但不能单独为当前模型出表。

输出新增 `motion.csv`、`quality.csv`、`metric-results.json`、`tables/motion.tex`、`tables/quality.tex` 和两个 `figure-data` 文件；`provenance.json` 记录消费身份及产物 hash。RGB 主表只引用适用的正式分表，不填入分布统计或发明总分。运动报告计划、可测、可信覆盖、歧义、证据不足、生成失败和条件误差；质量报告计划/完整图块、缺失像素、参考覆盖、真实—真实对照及特征配置。

零可测覆盖的误差为 null；缺失世界时固定世界/城市权重的主值为 null，条件子集值单独保留。合法负 MMD-U 保留，非有限数拒绝。此指标称为 generic feature polynomial MMD-U，不自动称为标准 KID。

合成模型或诊断特征贯通时，结果标记 `DIAGNOSTIC_ONLY`，正式 availability 仍为 `NOT_AVAILABLE`；不能用于批准仪器、真实生成 pilot、正式研究或论文门。正式特征权重、完整数据和实际模型未具备时，只签收软件消费能力。最终测试和 CI 需对应实际后继 HEAD，历史 f1 收据不追改。
