# WHU-RS19 数据兼容性专项审查（2026-08-14）

## 结论

本轮仅修复 `tier=whurs19` 的真实 Parquet 读取、异构尺寸鲁棒性、来源
fail-close 和完整性验证。模型、LoRA、A800 配置、continuous trajectory、
net_continuous、registration、low-overlap、Plücker/camera conditioning、rank
ablation、外部 baseline 接口均未修改，也没有启动训练或完整 benchmark。

根因是当前 WHU-RS19 Hugging Face snapshot 为 Parquet，`image` 可能以
`{"bytes": ..., "path": ...}` 返回，而旧 builder 把该值直接交给
`np.asarray`/PIL 假定路径；同时旧代码只探测每个场景池的第一张图并缓存尺寸，
无法代表真实的异构尺寸。底层 Hugging Face loader 还会吞掉加载异常，随后让
场景生成器转入合成数据路径。

## 修改前审计

| 文件/链路 | 修改前处理方式 | PIL 假定 | 600×600 假定 | silent fallback | 处置 |
|---|---|---:|---:|---:|---|
| `lora_remotesense/build_dataset_from_benchmark.py` | `dataset[i][image]` 直接交给 `_to_numpy_image`；首图尺寸决定整个池的大/小图分支 | 是 | 是（WHU 注释和池级探针） | 间接存在 | 统一解码；WHU 始终单源大图路径；短边不足 512 时等比例 resize-to-cover；逐样本记录尺寸 |
| `wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py` | `load_dataset` 异常只打印，未加载场景可转合成 | 否 | 否 | 是 | 不改第三方任务定义；在注入层对 official WHU 做加载后硬校验，并优先加载可见本地 Parquet |
| `server_run/check_resisc45.py --tier whurs19` | 只看首图 `.size`，未全量解码、未核对 1005/19/RGB/尺寸分布 | 是 | 是 | 检查器不能排除 | 扩展为全量 CPU 严格检查；记录真实 Parquet 路径和来源 |
| `lora_remotesense/semantic_gate.py` | 接收正式裁剪后的数组/帧 | 否 | 否 | 否 | 无需修改 |
| v5 trajectory/window builder | 接收 512×512 `full_scene`，提取 128×222 窗口 | 否 | 只要求正式场景为 512 | 否 | 无需修改；测试真实调用该链路 |
| evaluation/data loader | 读取最终 H5，不读取 WHU 原始字段 | 否 | 否 | 否 | 无需修改 |

## Loader 新行为

统一 helper：`lora_remotesense/hf_image_utils.py::decode_hf_image`。

支持：

1. `PIL.Image.Image`；
2. `{"bytes": bytes-like, "path": ...}`，bytes 非空时优先；
3. bytes 为 `None` 且 path 非空时从 path 读取。

输出总是一个已完整加载、与源 stream/file 解耦的 RGB PIL image。BytesIO、
打开的文件和临时 PIL 对象均在 helper 内关闭。`{}`、`None`、整数、非法 bytes、
非法 path 类型均显式抛错；不返回黑图、不跳过样本、不回退其他图像。

当 `$HF_HOME/hub/datasets--jonathan-roberts1--WHU-RS19/snapshots/*/data/`
存在 `train-*.parquet` 时，official WHU builder 直接以本地 Parquet 加载；可用
`WHURS19_SNAPSHOT_ROOT` 显式指定 repo cache、snapshot、data 目录或 Parquet。
找不到显式指定的路径会硬错误。没有显式路径且本地 Parquet 不可见时，才调用
原有的同名 HF repo loader，之后仍执行严格来源校验。

Parquet 若把 `label` 暴露为 plain int64 而不是 `ClassLabel`，仅对精确 repo
`jonathan-roberts1/WHU-RS19` 使用其官方 dataset card 发布的固定 0..18
类名映射，并先验证所有 observed id 都在范围内。其他数据集行为不变。

## 非 600×600 图像

所有 WHU 样本仍使用原项目的单源大图 canonical pipeline，不删除、不拼接、
不补黑边，也不改变轨迹或输出协议：

| 原始 W×H | 进入 512 场景前 | 后续路径 |
|---:|---|---|
| 600×600（1001 张） | identity | 既有 seeded 512 crop → trajectory → 128×222 windows → 480×832 |
| 958×612 | identity | 同上 |
| 726×456 | 确定性等比例 resize-to-cover 为 816×512 | 同上 |
| 600×500 | 确定性等比例 resize-to-cover 为 615×512 | 同上 |
| 711×718 | identity | 同上 |

归一化复用正式 builder 的 `_resize_image`，scale 为
`max(512/H, 512/W)`，因此不拉伸宽高比。每条 WHU 样本将以下 attrs 透传到最终
H5：`source_original_hw`、`source_normalized_hw`、`source_preprocess`、
`source_preprocess_scale`。benchmark 的 scene size、trajectory、overlap、帧数、
窗口和 480×832 输出均未改变。

## Fallback 判定

official `tier=whurs19` 现在必须让每个请求场景都实际加载自精确来源
`jonathan-roberts1/WHU-RS19`。以下情况均抛 `SceneSelectionError`：

- WHU 加载失败后场景缺失（阻止 synthetic fallback）；
- 实际来源名为 RESISC45；
- 实际来源名为 PatternNet；
- 显式 `WHURS19_SNAPSHOT_ROOT` 无合法 `train-*.parquet`。

不存在默认 debug fallback；H5 的 `source_dataset` 仍记录精确真实来源。

## 严格完整性命令

服务器执行：

```bash
export RSWM_BASE=/nix/rsroll_25828
export HF_HOME=/nix/rsroll_25828/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python server_run/check_resisc45.py \
  --tier whurs19 \
  --check-offline \
  --check-source \
  --snapshot-root /nix/rsroll_25828/cache/huggingface/hub/datasets--jonathan-roberts1--WHU-RS19 \
  --report /nix/rsroll_25828/whurs19_source_check.json
```

硬检查：source、1005 samples、1005 decode success、0 errors、19 unique labels、
ids 0..18、RGB 1005、四个 taxonomy pool 可用。尺寸分布全量输出；当前参考指纹
为 `600×600:1001` 和四个 singleton。尺寸指纹变化只发 WARN，不把其他数据版本
武断判为非法，也不会据此删除样本。

## 测试结果

| 项目 | 状态 | 证据 |
|---|---|---|
| WHU unit tests（PIL/bytes/path/非法输入/plain int64/本地 snapshot/fail-close） | PASS | `python tests/test_whurs19_data.py`：11 tests, OK |
| WHU synthetic integrity 1005/19/RGB/尺寸指纹逻辑 | PASS | 上述 unit suite，1005 行检查器夹具全绿 |
| 600×600 canonical preprocessing | PASS | 实际调用 512 scene + v5 trajectory extractor，得到 17×128×222×3 |
| 四个非 600×600 canonical preprocessing | PASS | 958×612、726×456、600×500、711×718 同一正式链路，无越界/负 crop/shape error |
| official WHU 缺失/RESISC45/PatternNet fallback | PASS | 三种输入均显式 `SceneSelectionError` |
| WHU taxonomy/checker self-test | PASS | `--tier whurs19 --self-test`：9 PASS / 0 FAIL |
| RESISC checker shared-code regression | PASS | `--tier resisc45 --self-test`：9 PASS / 0 FAIL |
| AID regression | PASS | 600×600 PIL 像素级 crop 结果逐位相等；taxonomy/prompt 49 PASS / 0 FAIL |
| ICLR protocol regression | PASS | 25 PASS / 0 FAIL |
| A800 profile regression | PASS | 35 checks 全通过 |
| Python source compile | PASS | 232/232 |
| 当前电脑真实 WHU 1005 decode smoke | NOT RUN | `/nix/rsroll_25828/...` 与本地 HF cache 均不可见；未下载替代数据、未伪报 |
| 训练 / GPU / 完整 benchmark | NOT RUN | 按要求未启动 |

真实服务器 smoke 必须运行上面的 `--check-source` 命令；只有其 JSON 报告
`n_fail=0` 才能把 “real 1005 decode test” 从 NOT RUN 改为 PASS。

## 修改文件

```text
A lora_remotesense/hf_image_utils.py
M lora_remotesense/build_dataset_from_benchmark.py
M server_run/check_resisc45.py
M server_run/check_near_duplicates.py
M server_run/offline_prefetch.sh
A tests/test_whurs19_data.py
A tests/fixtures/whurs19_path_fixture.ppm
M DATASHEET.md
M README_v31.md
M DEPLOY_D4.md
M CHANGELOG_vs_0710.md
A reports/WHU_RS19_DATA_AUDIT_20260814.md
```

## 人工 diff 复核重点

1. decoder 对 bytes/path 的优先级和资源关闭；
2. `_normalize_whurs19_source` 只在精确 WHU repo 生效；
3. per-sample metadata 是否完整透传；
4. `_assert_official_whurs19_loaded` 是否覆盖缺失和替代来源；
5. `check_resisc45.py` 的 1005/19/RGB/尺寸统计是否与服务器 JSON 一致。
