# Runtime experiment directory

The formal runner creates one immutable experiment namespace here:

```text
experiments/<experiment_id>/
├── config.yaml
├── manifest.yaml
├── logs/
├── checkpoints/
└── results/
```

Runtime experiment outputs are intentionally ignored by Git. `config.yaml` is
the exact launch snapshot. `manifest.yaml` is the orchestration record; the
existing training code still writes its native `run_manifest.json` and native
LoRA/optimizer/scheduler state inside each checkpoint.
