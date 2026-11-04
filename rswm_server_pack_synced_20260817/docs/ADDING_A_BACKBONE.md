# 把你自己的模型接进 RSWM-Bench

> 参考实现与协议本体:`lora_remotesense/new/backbone.py`
> 单测(纯 CPU、无 torch 可跑):`server_run/test_backbone_protocol.py`

RSWM-Bench 的评测入口原本把生成器钉死成一个具体实现 —— `WAN_CONFIGS["i2v-A14B"]`
在 `train.py:236` / `evaluate.py:384` / `evaluate_baseline.py:171` /
`generate_samples.py:183` / `inference.py:406` 各写了一遍,适配方法只有 LoRA。
`backbone.py` 把"生成一段视频"这个动作抽成一个最小协议,让第三方无需改动
评测代码就能换成自己的模型。

**这个协议只覆盖推理侧。** 训练/适配(LoRA、full-ft、prefix……)高度依赖具体
骨干,不在协议范围内:你用自己的代码训练,只需要把训练好的模型包成一个
`VideoBackbone` 交给评测入口。

---

## 1. 三十秒版本

实现三个成员即可:

```python
import numpy as np
from backbone import BackboneCapabilities, check_output_contract, parse_size

class MyBackbone:
    name = "my-video-model-v1"

    capabilities = BackboneCapabilities(
        supports_pose_cond=False,     # 我的模型只吃图 + 文本
        adaptation="lora",            # none | lora | full-ft | prefix | adapter | ...
        frame_count_rule="4n+1",
        stochastic=True,
    )

    def generate(self, *, first_frame, prompt, num_frames, size,
                 pose_cond=None, seed=None) -> np.ndarray:
        h, w = parse_size(size)               # "480*832" → (480, 832),高在前
        video = my_model.sample(first_frame, prompt, num_frames, (h, w), seed)
        return check_output_contract(video, num_frames, size, who="MyBackbone")
```

然后在你的入口脚本里注册:

```python
from backbone import register_backbone, build_backbone, validate_backbone
register_backbone("my-video-model-v1", MyBackbone)
bb = build_backbone("my-video-model-v1")     # 内部会跑 validate_backbone
```

`validate_backbone()` 会检查签名而不只是"有没有这个方法" —— 最常见的接错是把
参数写成位置参数,或者给 `pose_cond` 漏掉默认值 `None`(见 §3)。

---

## 2. 输入/输出契约(逐条都会被硬校验)

| 参数 | 类型 | 说明 |
|---|---|---|
| `first_frame` | `(H0, W0, 3)` `uint8` RGB | 就是 H5 里的 `image`(`evaluate.py:211`)。`H0/W0` 不保证等于 `size`,你负责缩放。 |
| `prompt` | `str` | 方向词措辞见 `evaluate.parse_direction_word_from_prompt`(兼容 `"toward the east"` 与旧的 `"eastward"`)。 |
| `num_frames` | `int` | **必须 ≡1 (mod 4)**。见 §5。 |
| `size` | `str` | `"H*W"`,**高在前**(与 `wan.configs.SIZE_CONFIGS` 同格式)。 |
| `pose_cond` | `PoseCond` 或 `None` | `None` 是**合法输入**,含义见 §3。 |
| `seed` | `int` 或 `None` | 评测入口总会显式传;`None` 表示用骨干自己的默认种子。 |

返回值必须是 `(num_frames, H, W, 3)` 的 `uint8` ndarray、RGB、值域 `[0,255]`,
其中 `(H, W) = parse_size(size)`。

**为什么这几条要硬校验而不是"尽量满足":**
`metrics.py:103-104` 在算指标前会把 gen 与 gt 裁到两者的最小形状。也就是说
返回错帧数或错分辨率**不会报任何错**,只会静默地改变"被评的到底是什么" ——
FID/FVD/SSIM 照样出一组看起来合理的数字。转置(把 `size` 按"宽\*高"理解)同理。
所以 `check_output_contract()` 存在的意义就是把这类静默改成异常;请在你的
`generate()` 末尾调用它。

`PoseCond` 的两个字段:

```python
PoseCond(
    c2w,          # (T, 4, 4) camera-to-world
    intrinsics,   # (T, 4) = [fx, fy, cx, cy],可选
)
```

`c2w` 的约定来自建库器(`lora_training/camera_trajectory.py:112-127`):旋转恒为
单位阵(天底视角固定朝向),平移列 `tx` = +图像右(东)、`ty` = +图像下(南)、
`tz` = 0,**单位 = 1 个窗口边长**。`intrinsics` 的像素口径是 H5 的帧尺寸
(canonical 480×832;`inference.py:261-262` 把 `height_org/width_org` 硬编码成 480/832)。

> **方向约定(接错整表反号)**:benchmark 把相机平移实现为静态大图上的滑窗。
> **窗口向东移 ⟹ 画面内容向西流**。所以 prompt 说 "toward the east" 时,
> 合规的画面运动方向是 180°,不是 0°(`evaluate.py:156-170`)。

---

## 3. `pose_mode` 三态与 `pose_cond` 的一一对应

这是本 benchmark 最容易接错、且接错后完全静默的地方。三个臂:

| `RSWM_POSE_MODE` | 含义 | 你会收到的 `pose_cond` |
|---|---|---|
| `native` | 真实轨迹位姿进模型 | 非单位阵的 `PoseCond` |
| `identity` | **中和位姿仍走位姿通路**的静止相机命令诊断臂 | 全单位阵的 `PoseCond` |
| `drop` | 位姿通路整条不接 = 真 text-only | `None` |

`identity` **不是** text-only —— 位姿张量照样被编码进模型,只是内容是单位阵。
按 v2.9 拍板它永远不进 headline(`run_manifest.py:121`)。正因为如此,协议必须
能把 `drop` 和 `identity` 分开:如果"没有位姿"只能用"传单位阵"表达,这两个臂在
API 层就分不开了。所以 **`pose_cond=None` 必须是合法输入**,而不是让调用方传
一堆单位阵。

`backbone.assert_pose_cond_matches_mode()` 对四种矛盾组合全部硬失败:

* `drop` + 传了 `pose_cond` → 你以为传进去了,其实 `inference.py:254` 会悄悄丢掉;
* `native`/`identity` + `pose_cond=None` → 跑出的是 text-only 结果,却会被
  `evaluate.py:494` 写上 `pose_mode="native"`,再被 `run_manifest` 认证成合格 headline;
* `native` + 全单位阵位姿 → 几何通道实际全零却按几何臂统计;
* `identity` + 真实轨迹位姿 → 静止相机诊断臂的语义直接不成立。

（后两条与 `evaluate.py:250-258` 的两道门同义,协议在骨干边界上再守一次。）

---

## 4. 能力声明:哪些臂对你的骨干**有意义**

```python
BackboneCapabilities(
    supports_pose_cond = False,   # 决定 applicable_pose_modes
    pose_cond_format   = None,    # 支持时必填:位姿到底怎么被消费
    frame_count_rule   = "4n+1",  # 或 "any"
    stochastic         = True,    # 换 seed 是否真的换输出
    adaptation         = "lora",
    trainable_params   = 12_345_678,
    supported_sizes    = ("480*832",),
    notes              = "……",
)
```

### 不支持位姿条件怎么办

`supports_pose_cond=False` 是一等公民,不是缺陷。它的后果是
`capabilities.applicable_pose_modes == ("drop",)`:

* ✅ **可以跑**:`drop` 臂(真 text-only)。这是 official profile 下 headline 的
  文本臂口径(`common.sh` 的 profile 段:20-22 号 `drop` + GEO_H5)。
* ❌ **不能跑**:`native`、`identity`。骨干请求这两个臂时协议直接抛异常。

**为什么必须拒绝而不是"跑了再说"**:一个没有位姿输入通道的模型在 `native` 臂上
的分数,与一个有通道但学不会的模型,在 Δ 表里长得一模一样。前者是"没有这个输入",
后者才是"不听话"。把前者填进 native↔drop 的 Δ 表 = 把缺失的通道读成失败的控制。

如果你的模型有位姿通道但表示形式不同(6-DoF ray map、外参 JSON、相机轨迹 token
……),照样声明 `supports_pose_cond=True`,在 `pose_cond_format` 里写清楚你把
`PoseCond` 转成了什么 —— 读者据此判断你和 Wan 的 Plücker 条件是不是同一类方法。

### `stochastic=False` 的后果

多种子聚合在确定性骨干上恒得 0 方差。那**不是**稳定性证据。`aggregate_seeds.py`
汇总时应当据此把它排除在种子方差之外,而不是报一个 `std=0`。

---

## 5. 帧数契约:`T ≡ 1 (mod 4)`

benchmark 的产物契约是 `steps=9` 时 `T = 9 + (K-1)*8`,即 K=1..4 → 9 / 17 / 25 / 33,
全部 ≡1 (mod 4)(CI 的 D 步骤逐样本断言这一点)。Wan 系 VAE 的时间下采样也要求它。

现有 `inference.py:188` 有一段隐式钳制:

```python
len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
frame_num = min(frame_num, len_c2ws)
```

它**只在传了 `action_path` 时执行**。`v2.8.2 N6` 的注释记着:`drop` 臂必须先让
这段钳制跑完、再把 `action_path` 置 `None`,否则 `drop` 臂的 T 会和 `native` 臂不同
—— 而 FVD/FID 照样出数,只是不可比。这是一个"靠代码行顺序维持的不变量",
对第三方骨干完全不可移植。

协议把它换成三条显式契约,由 `check_frame_count` / `WanBackbone` / `check_output_contract`
分别强制:

1. `num_frames ≡ 1 (mod 4)`;
2. `len(pose_cond) == num_frames`(于是那段钳制退化成恒等操作);
3. 返回的 `T == num_frames`。

这三条在既有 canonical 数据上恒成立(T∈{9,17,25,33},`poses` 与 `video_frames` 等长),
所以**不改变任何已有数字**,只是把不变量从"行序"搬到了"断言"。

---

## 6. 两个参考实现

### `WanBackbone` —— 现有 `inference.py` 的薄封装

* 不改 `inference.py` 一行:只 `import` 它的 `load_lora_model` / `generate_video`;
  基模型走 `evaluate_baseline.load_base_model`(因为 `inference.load_lora_model`
  在 LoRA 目录缺失时会硬失败 —— 那是 v2.7.1 F2 刻意的"拒绝静默退化为基模型")。
* 位姿以 `poses.npy` / `intrinsics.npy` 落进临时目录,即 `evaluate.py:426-428` 现在的做法;
  调用结束后清理(`evaluate.py` v2.7.1 F9 修过同一个泄漏)。
* 默认超参逐个对齐 `evaluate.py:432-448` 现在实际吃到的值:
  `sampling_steps=70` / `shift=10.0` / `guide_scale=(5.0,5.0)` /
  `t5_cpu=True` / `inactive_model_cpu=True` / `framewise=True`。
  也就是说"换成 WanBackbone"是数值恒等的改造,不会顺手改采样配置。

> ⚠️ **`WanBackbone` 尚未在真权重上端到端验证** —— 需要 56GB 基模 + GPU。
> `server_run/test_backbone_protocol.py` 的 T8 用 mock 固定了它的参数转发、
> `pose_cond=None → action_path=None`、位姿落盘内容、临时目录清理、
> 基模 vs LoRA 走不同加载函数、以及矛盾组合在加载权重**之前**就失败。
> 首次真跑请与旧路径做一次逐样本 npy 对拍(同 seed、同数据、同 profile,
> 生成结果应逐位相同)。

### `TrivialShiftBackbone` —— 纯图像平移,无 GPU / 无权重

把首帧按命令方向逐帧平移(空出的区域用边缘像素复制填充),不生成任何新内容。
它有三个用途:

1. **证明协议真的可插拔** —— 一个连权重都没有的东西也能走完整条评测路径;
2. **给 CI 一条无 GPU 的端到端通路**(`.github/workflows/ci.yml` 是纯 CPU、连
   torch 都不装的);
3. **一个诚实的参照点**:它对方向命令的服从度是"完美"的(平移量直接来自命令),
   实测四方位 prompt 下光流读数与目标角的误差 < 0.05°(本机 2026-07-29,
   `flow_size=(128,222)`、`step=1`)。

第 3 点值得对审稿人讲清楚:**"角误差很低"本身不是世界建模能力的证据。**
一个零生成能力的平移基线就能把角误差压到接近 0° —— 比 GT 视频的 22.5° 量化地板
还低,恰恰因为它无视了真实轨迹的连续方向、只照着量化后的方向词直线平移。
方向指标必须与画质指标(SSIM/PSNR/FID/FVD)一起读。

> `TrivialShiftBackbone` **不是论文里的 baseline**。它没有生成能力,SSIM/PSNR
> 只反映"首帧有多像后续帧",与 `validate_direction_metric.py` 的 persistence
> 基线是近亲。任何把它的分数与真实模型并列的表都必须标注这一点,建议给它一个
> 专门的 role(见 §8)使其永不进 headline。

它同时是"协议能表达『不支持位姿』"的可执行样例:

```python
TrivialShiftBackbone(use_pose_cond=False)   # → applicable_pose_modes == ("drop",)
```

---

## 7. 结果如何声明

在论文/README 里报告用非 canonical 骨干跑出的数字时,**必须**同时给出:

1. **骨干标识与适配方法**:`capabilities.adaptation`(`none` = 零样本)、
   `trainable_params`。"零样本失败"和"适配后仍失败"是两个不同的结论。
2. **跑了哪些臂,以及为什么**。如果 `supports_pose_cond=False`,明写
   "只跑 drop 臂,native/identity 不适用",不要留空格让读者以为是漏跑。
3. **不要跨骨干合并平均**。不同骨干的分数分开成行;跨骨干只报"结论方向是否一致"
   (例如"三个骨干上 native−drop 的 Δ 都不显著"),不报一个混合均值。
4. **确定性骨干不报种子方差**(`stochastic=False` 时 std 恒 0,那不是稳定性)。
5. **`headline_eligible`**:非 canonical 骨干的产物是否算 headline 由 profile /
   role 规则决定(`run_manifest.headline_eligible`)。参照点类骨干(如
   `TrivialShiftBackbone`)不应进 headline。

`backbone.py` 提供了两个帮手,让上面这些进得了产物身份:

```python
bb.checkpoint_id()      # → "wan2.2-i2v-A14B+lora:/path/to/lora" / "trivial-shift:frac=0.04,pose=on"
bb.manifest_fields()    # → {"backbone": ..., "backbone_adaptation": ..., ...}
```

`checkpoint_id()` 特意把骨干名拼进去:`run_manifest.IDENTITY_FIELDS` 目前**没有**
`backbone` 字段,两个不同骨干的产物落在同一目录下会被判为"同源"从而被复用。在
主协调者把 `backbone` 加进 `IDENTITY_FIELDS` 之前,`checkpoint_id` 是唯一既已参与
身份比对、又能承载骨干信息的字段。

---

## 8. 主协调者要做的接线(本文件与 `backbone.py` 都没有改任何既有文件)

按依赖顺序,具体到函数:

### 必做(不做的话协议接不上评测)

1. **`lora_remotesense/new/evaluate.py::run_inference_batch`**(现 334-525 行)
   * 348 行 `from inference import load_lora_model, generate_video`、
     384 行 `config = WAN_CONFIGS["i2v-A14B"]`、388-395 行的模型加载、
     432-448 行的 `generate_video(...)`、451-452 行的张量→uint8:
     整段替换为 `backbone.generate(first_frame=..., prompt=..., num_frames=...,
     size=..., pose_cond=..., seed=gen_seed)`。
   * 新增形参 `backbone`(默认 `None` = 按现有 `checkpoint_dir/lora_path` 造
     `WanBackbone`,保持旧行为)。
   * 368-375 行 `build_manifest(...)`:`checkpoint_id=str(lora_path or "base")`
     → `backbone.checkpoint_id()`;加 `extra=backbone.manifest_fields()`。
   * 470-498 行的 meta dict:加 `"backbone": backbone.name`。
   * 进循环之前调一次 `backbone.capabilities.assert_pose_mode_supported(
     os.environ["RSWM_POSE_MODE"])` —— 不适用的臂要在烧 GPU 之前拒。

2. **`lora_remotesense/new/evaluate.py::main`**(899 行起 argparse)
   * 加 `--backbone`(默认 `"wan"`)与 `--backbone_kwargs`(JSON 串)。
   * 939-940 行的 `assert args.checkpoint_dir/--lora_path 必须指定` 要放宽:
     无权重骨干(如 `trivial-shift`)不需要它们。建议改成"由 backbone 自己声明
     需要哪些参数",而不是在 main 里写死。
   * 954-967 行的 `run_inference_batch(...)` 调用加 `backbone=`。

3. **`lora_remotesense/new/evaluate_baseline.py::run_baseline_inference`**
   (现 128-276 行)同 1,调用点在 140 / 171 / 156-167(manifest,`checkpoint_id="base"`
   在 162 行)/ 199-216 行。**两个入口必须一起改** —— v2.9.2 只给 `evaluate.py` 加了
   `--metrics_only` 身份门、`evaluate_baseline.py` 漏了,`run_manifest.py:241-243`
   把那次教训写在注释里了。

4. **`lora_remotesense/new/run_manifest.py`**
   * `IDENTITY_FIELDS`(34-51 行)加 `"backbone"`(建议再加 `"backbone_adaptation"`)。
     不加的话,同一目录下换骨干重跑会被判"同源"而复用旧产物。
   * `code_sha256()` 的默认文件清单(84-94 行)加 `here / "backbone.py"`。
     否则改了 backbone.py 不会改变代码指纹,产物看起来仍"合格"。

5. **`server_run/common.sh`**:加 `BACKBONE`/`BACKBONE_KWARGS` 环境变量并透传
   `--backbone`;缺省 `wan` 以保持既有 20-36 号作业逐字节不变。

### 建议做(不做也能跑,但结论会被稀释)

6. **`server_run/collect_results.py` / `aggregate_seeds.py`**:按 `backbone`
   分组,禁止跨骨干求均值(§7 第 3 条)。`stochastic=False` 的骨干不报种子 std。

7. **`run_manifest._NEVER_HEADLINE_ROLES`**(53-56 行):为参照点类骨干新增一个
   role(例如 `reference_degenerate`)并加入该集合,使 `TrivialShiftBackbone`
   之类的产物永不进 headline。**这是一个需要拍板的决定,本文件没有替你做。**

8. **`lora_remotesense/new/generate_samples.py`**(58-59 / 183 / 185 / 217 行):
   出样脚本同样绑死 Wan,改法同 1。

9. **`lora_remotesense/new/train.py:236`** 的 `WAN_CONFIGS["i2v-A14B"]`:
   训练侧**不在本协议范围内**(见开头)。但至少应把 `"i2v-A14B"` 提成 CLI 参数,
   否则连"换一个 Wan 配置"都做不到。给第三方骨干做适配训练应当由他们自己的
   代码完成,评测只认 `VideoBackbone`。

10. **`.github/workflows/ci.yml`**:无需改动 —— C 步骤用 `server_run/test_*.py`
    的 glob 自动发现,`test_backbone_protocol.py` 会被自动纳入。

---

## 9. 验收清单

接好自己的骨干后,按顺序跑:

```bash
# 1) 协议自检(纯 CPU)
python lora_remotesense/new/backbone.py --self-test

# 2) 协议单测(纯 CPU、无 torch 亦可)
python server_run/test_backbone_protocol.py

# 3) 你自己的骨干:结构 + 契约
python - <<'PY'
import numpy as np, sys
sys.path.insert(0, "lora_remotesense/new")
from backbone import validate_backbone, check_output_contract
import my_backbone_module                     # 你的实现
bb = my_backbone_module.MyBackbone()
validate_backbone(bb)                          # 签名与能力声明
img = np.zeros((480, 832, 3), np.uint8)
v = bb.generate(first_frame=img, prompt="Net displacement toward the east.",
                num_frames=33, size="480*832", seed=42)
check_output_contract(v, 33, "480*832", who="MyBackbone")
print("OK", v.shape, v.dtype, v.min(), v.max())
PY
```

三步全绿之后再烧 GPU 跑正式矩阵。三步都不需要 GPU,失败在这里比失败在
24 小时的 PBS 作业里便宜四个数量级。
