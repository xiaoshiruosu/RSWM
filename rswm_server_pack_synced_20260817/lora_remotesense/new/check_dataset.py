#!/usr/bin/env python3
"""
检查 LingBot 训练数据集的转换效果

功能:
  1. 打印数据集统计摘要（样本数、帧数、分辨率、prompt 等）
  2. 生成样本预览图（关键帧 + 轨迹 + 嵌入信息）
  3. 生成 GIF 动画（滑动窗口效果）
  4. 数据完整性校验

用法:
  # 基本检查（打印统计信息）
  python check_dataset.py train.h5

  # 生成预览图（保存为 PNG）
  python check_dataset.py train.h5 --preview --num-samples 5

  # 生成 GIF 动画
  python check_dataset.py train.h5 --gif --sample 0

  # 完整检查（统计 + 预览 + GIF）
  python check_dataset.py train.h5 --all
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path

# ============================================================================
# 统计信息
# ============================================================================

def print_stats(h5_path):
    """打印数据集完整统计信息"""
    print(f"\n{'='*60}")
    print(f"数据集: {h5_path}")
    print(f"{'='*60}")

    with h5py.File(h5_path, 'r') as f:
        # 元数据
        if 'metadata' in f:
            meta = f['metadata']
            print(f"\n--- 元数据 ---")
            for key in meta.attrs:
                val = meta.attrs[key]
                if isinstance(val, (np.ndarray, list, tuple)):
                    print(f"  {key}: {list(val)}")
                else:
                    print(f"  {key}: {val}")

        # 样本统计
        if 'data' not in f:
            print("  错误: 无 data 组!")
            return

        keys = sorted(f['data'].keys())
        total = len(keys)
        print(f"\n--- 样本统计 ---")
        print(f"  总样本数: {total}")

        if total == 0:
            return

        # 收集所有样本的信息
        frame_counts = []
        prompts = []
        scene_cats = {}
        shapes = {}

        for i, key in enumerate(keys):
            s = f['data'][key]
            nf = int(s.attrs.get('num_frames', 0))
            frame_counts.append(nf)

            prompt = s.attrs.get('prompt', '')
            prompts.append(prompt)

            # 统计 prompt 中的场景类别
            # [v2.5] 场景统计改读 source_scene_category attr:pt_v3 模板里
            # water 句无 'flood'、general 句无 'general',按 prompt 关键词数会恒为 0。
            _sc = str(s.attrs.get('source_scene_category', '')) if hasattr(s, 'attrs') else ''
            _sc = {'flood': 'water'}.get(_sc, _sc)
            if _sc:
                scene_counts[_sc] = scene_counts.get(_sc, 0) + 1
            # 记录第一个样本的各字段形状
            if i == 0:
                for ds_name in s:
                    shapes[ds_name] = s[ds_name].shape

        frame_counts = np.array(frame_counts)
        print(f"  帧数范围: {frame_counts.min()} ~ {frame_counts.max()}")
        print(f"  平均帧数: {frame_counts.mean():.1f}")

        print(f"\n--- 场景分布 ---")
        for cat, cnt in sorted(scene_cats.items()):
            print(f"  {cat}: {cnt} 样本")

        print(f"\n--- 数据字段（sample_00000）---")
        for name, shape in shapes.items():
            s0 = f['data'][keys[0]]
            dtype = s0[name].dtype
            data = s0[name][:]
            vmin, vmax = data.min(), data.max()
            print(f"  {name:20s}  shape={str(shape):20s}  dtype={dtype}  range=[{vmin:.2f}, {vmax:.2f}]")

        # 属性
        s0 = f['data'][keys[0]]
        print(f"\n--- 属性 ---")
        for attr_name in s0.attrs:
            val = s0.attrs[attr_name]
            print(f"  {attr_name}: {val}")

        print(f"\n--- Prompt 示例（前 5 条）---")
        for i, p in enumerate(prompts[:5]):
            print(f"  [{i}] {p}")

        # 文件大小
        file_size = Path(h5_path).stat().st_size
        print(f"\n--- 文件信息 ---")
        print(f"  文件大小: {file_size / 1024 / 1024:.1f} MB")
        print(f"  每样本平均: {file_size / total / 1024:.1f} KB")


# ============================================================================
# 数据完整性校验
# ============================================================================

def validate_dataset(h5_path):
    """校验数据集完整性"""
    print(f"\n{'='*60}")
    print("数据完整性校验")
    print(f"{'='*60}")

    errors = []
    warnings = []

    with h5py.File(h5_path, 'r') as f:
        if 'metadata' not in f:
            errors.append("缺少 metadata 组")
        if 'data' not in f:
            errors.append("缺少 data 组")
            print(f"  ✗ 致命错误: {errors[-1]}")
            return False

        keys = sorted(f['data'].keys())
        required_fields = ['image', 'video_frames', 'intrinsics', 'poses']
        required_attrs = ['prompt', 'num_frames']

        for i, key in enumerate(keys):
            s = f['data'][key]

            # 检查必需字段
            for field in required_fields:
                if field not in s:
                    errors.append(f"{key}: 缺少字段 '{field}'")

            for attr in required_attrs:
                if attr not in s.attrs:
                    errors.append(f"{key}: 缺少属性 '{attr}'")

            # 检查数据一致性
            if 'video_frames' in s and 'num_frames' in s.attrs:
                vf_shape = s['video_frames'].shape
                nf = s.attrs['num_frames']
                if vf_shape[0] != nf:
                    errors.append(f"{key}: video_frames 帧数 {vf_shape[0]} != num_frames {nf}")

            if 'poses' in s and 'num_frames' in s.attrs:
                p_shape = s['poses'].shape
                nf = s.attrs['num_frames']
                if p_shape[0] != nf:
                    errors.append(f"{key}: poses 帧数 {p_shape[0]} != num_frames {nf}")

            if 'intrinsics' in s and 'num_frames' in s.attrs:
                intr_shape = s['intrinsics'].shape
                nf = s.attrs['num_frames']
                if intr_shape[0] != nf:
                    errors.append(f"{key}: intrinsics 帧数 {intr_shape[0]} != num_frames {nf}")

            # 检查值范围
            if 'video_frames' in s:
                vf = s['video_frames']
                # 只检查第一帧避免内存问题
                frame0 = vf[0]
                if frame0.max() > 256:
                    warnings.append(f"{key}: video_frames 值超出 [0,255] 范围 (max={frame0.max():.1f})")
                if frame0.min() < 0:
                    warnings.append(f"{key}: video_frames 存在负值 (min={frame0.min():.1f})")

            if 'poses' in s:
                pose = s['poses'][:]
                if np.any(np.isnan(pose)):
                    errors.append(f"{key}: poses 包含 NaN")
                if np.any(np.isinf(pose)):
                    errors.append(f"{key}: poses 包含 Inf")

    # 输出结果
    if errors:
        print(f"\n  ✗ 发现 {len(errors)} 个错误:")
        for e in errors[:20]:
            print(f"    - {e}")
        if len(errors) > 20:
            print(f"    ... 还有 {len(errors)-20} 个错误")
    else:
        print(f"\n  ✓ 无错误")

    if warnings:
        print(f"\n  ⚠ 发现 {len(warnings)} 个警告:")
        for w in warnings[:10]:
            print(f"    - {w}")
    else:
        print(f"  ✓ 无警告")

    ok = len(errors) == 0
    print(f"\n  结论: {'✓ 数据集有效' if ok else '✗ 数据集存在问题'}")
    return ok


# ============================================================================
# 预览图生成
# ============================================================================

def generate_preview(h5_path, output_dir="preview", num_samples=5):
    """生成样本预览图（PNG）"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("错误: 需要 matplotlib。运行: pip install matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"生成预览图 → {output_dir}/")
    print(f"{'='*60}")

    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['data'].keys())
        num_samples = min(num_samples, len(keys))

        for si in range(num_samples):
            key = keys[si]
            s = f['data'][key]

            vf = s['video_frames'][:]       # (T, H, W, 3)
            image = s['image'][:]           # (H, W, 3)
            poses = s['poses'][:]           # (T, 4, 4)
            intrinsics = s['intrinsics'][:] # (T, 4)
            prompt = s.attrs.get('prompt', '')
            num_frames = int(s.attrs.get('num_frames', vf.shape[0]))

            T, H, W, C = vf.shape

            # 归一化到 [0, 1] 用于 imshow
            def norm(x):
                x = x.astype(np.float32)
                if x.max() > 1.0:
                    x = x / 255.0
                return np.clip(x, 0, 1)

            # ---- 图1: 关键帧展示 ----
            n_show = min(8, T)
            indices = np.linspace(0, T - 1, n_show).astype(int)

            fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 7))
            fig.suptitle(f"{key} | {num_frames} frames | {H}×{W}\n{prompt}", fontsize=10, y=0.98)

            # 第一行：关键帧
            for i, idx in enumerate(indices):
                axes[0, i].imshow(norm(vf[idx]))
                axes[0, i].set_title(f"F{idx}", fontsize=8)
                axes[0, i].axis('off')

            # 第二行：image + 轨迹 + 位姿分析
            axes[1, 0].imshow(norm(image))
            axes[1, 0].set_title("image (首帧)", fontsize=8)
            axes[1, 0].axis('off')

            # 位姿轨迹 (XY)
            tx = poses[:, 0, 3]
            ty = poses[:, 1, 3]
            axes[1, 1].plot(tx, ty, 'b.-', markersize=3, linewidth=1)
            axes[1, 1].plot(tx[0], ty[0], 'go', markersize=8, label='start')
            axes[1, 1].plot(tx[-1], ty[-1], 'r*', markersize=10, label='end')
            axes[1, 1].set_title("位姿轨迹 (tx, ty)", fontsize=8)
            axes[1, 1].legend(fontsize=6)
            axes[1, 1].set_aspect('equal')
            axes[1, 1].grid(True, alpha=0.3)

            # 内参
            axes[1, 2].plot(intrinsics[:, 0], label='fx')
            axes[1, 2].plot(intrinsics[:, 1], label='fy')
            axes[1, 2].plot(intrinsics[:, 2], label='cx')
            axes[1, 2].plot(intrinsics[:, 3], label='cy')
            axes[1, 2].set_title("内参", fontsize=8)
            axes[1, 2].legend(fontsize=6)

            # 帧间差异（检查帧是否在变化）
            diffs = []
            for i in range(1, min(T, 20)):
                diff = np.abs(vf[i].astype(float) - vf[i-1].astype(float)).mean()
                diffs.append(diff)
            axes[1, 3].bar(range(len(diffs)), diffs, color='steelblue')
            axes[1, 3].set_title("帧间差异", fontsize=8)
            axes[1, 3].set_xlabel("frame", fontsize=7)

            # 首尾帧对比
            if n_show > 4:
                axes[1, 4].imshow(norm(vf[0]))
                axes[1, 4].set_title("第一帧", fontsize=8)
                axes[1, 4].axis('off')

            if n_show > 5:
                axes[1, 5].imshow(norm(vf[-1]))
                axes[1, 5].set_title("最后帧", fontsize=8)
                axes[1, 5].axis('off')

            if n_show > 6:
                # 首尾差异热图
                diff_map = np.abs(vf[-1].astype(float) - vf[0].astype(float)).mean(axis=-1)
                axes[1, 6].imshow(diff_map, cmap='hot')
                axes[1, 6].set_title("首尾差异热图", fontsize=8)
                axes[1, 6].axis('off')

            # 隐藏多余的 axes
            for j in range(max(n_show, 7) if n_show > 6 else max(n_show, 4), n_show):
                if j < len(axes[1]):
                    axes[1, j].axis('off')
            # 隐藏第二行超出部分
            for j in range(n_show):
                if j > 6:
                    axes[1, j].axis('off')

            plt.tight_layout()
            out_path = output_dir / f"preview_{key}.png"
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {out_path}")

    print(f"\n预览图已保存到 {output_dir}/")
    print(f"提示: 用 scp 拷贝到本地查看")


# ============================================================================
# GIF 动画生成
# ============================================================================

def generate_gif(h5_path, sample_idx=0, output_path=None, fps=5):
    """生成单个样本的 GIF 动画"""
    try:
        import imageio
    except ImportError:
        print("错误: 需要 imageio。运行: pip install imageio")
        return

    if output_path is None:
        output_path = f"lingbot_sample{sample_idx}.gif"

    print(f"\n生成 GIF: sample {sample_idx} → {output_path}")

    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['data'].keys())
        if sample_idx >= len(keys):
            print(f"错误: sample_idx {sample_idx} 超出范围 (共 {len(keys)} 样本)")
            return

        key = keys[sample_idx]
        s = f['data'][key]

        vf = s['video_frames'][:]  # (T, H, W, 3)
        prompt = s.attrs.get('prompt', '')
        T = vf.shape[0]

        print(f"  样本: {key}")
        print(f"  帧数: {T}")
        print(f"  分辨率: {vf.shape[1]}×{vf.shape[2]}")
        print(f"  Prompt: {prompt}")

        # 归一化到 uint8
        frames = []
        for i in range(T):
            frame = vf[i].astype(np.float32)
            if frame.max() > 1.0:
                frame = frame / 255.0
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            frames.append(frame)

        duration = 1.0 / fps
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
        print(f"  ✓ 已保存: {output_path} ({T} 帧, {fps} FPS)")


# ============================================================================
# 快速对比：合成 vs 高清
# ============================================================================

def compare_datasets(path_a, path_b, label_a="A", label_b="B", output="compare.png"):
    """对比两个数据集的样本质量"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("错误: 需要 matplotlib")
        return

    print(f"\n对比: {label_a} vs {label_b}")

    def norm(x):
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        return np.clip(x, 0, 1)

    fig, axes = plt.subplots(2, 6, figsize=(18, 7))
    fig.suptitle(f"数据集对比: {label_a} (上) vs {label_b} (下)", fontsize=12)

    for row, (path, label) in enumerate([(path_a, label_a), (path_b, label_b)]):
        with h5py.File(path, 'r') as f:
            keys = sorted(f['data'].keys())
            if not keys:
                continue
            s = f['data'][keys[0]]
            vf = s['video_frames'][:]
            T = vf.shape[0]

            # 展示 6 个均匀采样帧
            indices = np.linspace(0, T - 1, 6).astype(int)
            for col, idx in enumerate(indices):
                axes[row, col].imshow(norm(vf[idx]))
                axes[row, col].set_title(f"{label} F{idx}", fontsize=8)
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 对比图保存: {output}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="检查 LingBot 训练数据集的转换效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本检查
  python check_dataset.py train.h5

  # 生成预览图
  python check_dataset.py train.h5 --preview

  # 生成 GIF
  python check_dataset.py train.h5 --gif --sample 0

  # 全面检查
  python check_dataset.py train.h5 --all

  # 对比两个数据集
  python check_dataset.py train_hd.h5 --compare demo_synth.h5
        """
    )
    parser.add_argument("input", help="输入 HDF5 文件路径")
    parser.add_argument("--preview", action="store_true", help="生成预览图")
    parser.add_argument("--gif", action="store_true", help="生成 GIF 动画")
    parser.add_argument("--validate", action="store_true", help="数据完整性校验")
    parser.add_argument("--all", action="store_true", help="执行全部检查")
    parser.add_argument("--num-samples", type=int, default=5, help="预览样本数")
    parser.add_argument("--sample", type=int, default=0, help="GIF 的样本编号")
    parser.add_argument("--output-dir", default="preview", help="预览图输出目录")
    parser.add_argument("--compare", default=None, help="对比的第二个数据集路径")
    args = parser.parse_args()

    h5_path = args.input
    if not Path(h5_path).exists():
        print(f"错误: 文件不存在 {h5_path}")
        return 1

    # 始终打印统计
    print_stats(h5_path)

    if args.all:
        args.validate = True
        args.preview = True
        args.gif = True

    if args.validate or args.all:
        validate_dataset(h5_path)

    if args.preview:
        generate_preview(h5_path, output_dir=args.output_dir, num_samples=args.num_samples)

    if args.gif:
        generate_gif(h5_path, sample_idx=args.sample)

    if args.compare:
        if not Path(args.compare).exists():
            print(f"错误: 对比文件不存在 {args.compare}")
        else:
            compare_datasets(h5_path, args.compare,
                           label_a="HD", label_b="对比",
                           output="compare.png")

    # 提示下载命令
    print(f"\n{'─'*60}")
    print("下载到本地查看:")
    print(f"  scp zz1230@gadi-dm.nci.org.au:{Path(h5_path).resolve().parent}/'*.png' .")
    print(f"  scp zz1230@gadi-dm.nci.org.au:{Path(h5_path).resolve().parent}/'*.gif' .")
    return 0


if __name__ == "__main__":
    sys.exit(main())

