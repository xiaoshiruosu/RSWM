#!/usr/bin/env python
"""augment_h5_trajectory.py — 向既有 final H5 原地回填 trajectory / full_scene(免重建)。

背景(服务器 Track B 校准复盘, gt_oracle net_direction_error=80.67°):
  final H5(build_dataset_from_benchmark.py 产物, 布局 f['data']/sample_xxxxx)
  此前没有从 v5 源 benchmark 复制这两个字段:
    - trajectory  (T,2)=(cy,cx) 真实逐帧窗口中心 —— Track B 真实拐点时序的唯一
      来源(真实 knot 帧如 [0,9,17] ≠ 均匀 linspace 的 [0,8,17])。
    - full_scene  (512,512,3) uint8 完整场景 —— deterministic/random 诊断基线
      必须从它按原生窗口尺寸裁剪移动窗口;首帧 'image' 只是首帧窗口的 480×832
      上采样, 服务器上误传 scene_image=g['image'] 曾使两基线 valid_pct=0%。
  本脚本把它们(以及目标缺失时的 waypoints/segment_angles)从 v5 源 benchmark
  (tmp_bench, 布局 data/sample_xxxxx, 见 build_multi_direction_benchmark_v5.py
  _write_sample_to_hdf5) 回填进既有 final H5, 无需整库重建。

安全性:
  * 匹配 = 同名 sample key。逐样本校验: 帧数(目标 attrs num_frames == 源
    trajectory 长度 == 源 attrs total_frames)、num_segments(双方都有时)、
    waypoints allclose(双方都有时)。任一不匹配 -> 整体失败并列出全部违例样本,
    绝不静默写入错误轨迹;全部校验通过后才开始写。
  * 目标以 'r+' 原地打开, 单文件编辑(inode 不变, 绝不走「复制-改名」)。
    注意: --overwrite 删除旧 dataset 后 HDF5 不回收空间, 文件会略增大(有意
    不 repack —— repack 是新文件+改名, 会换 inode)。
  * 默认只新增缺失 dataset;--overwrite 才替换已存在的 trajectory/full_scene。
    waypoints/segment_angles 永远只补缺(它们同时是本脚本的校验锚点)。
  * 目标位于旧 AID 工作目录(server_run/work, 旧结果神圣)下时拒绝写入,
    除非显式 --i-know。
  * DRYRUN=1 环境变量: 只打印计划, 不写入任何数据。

用法:
  python server_run/augment_h5_trajectory.py --src tmp_bench.h5 --dst final.h5
  DRYRUN=1 python server_run/augment_h5_trajectory.py --src tmp_bench.h5 --dst final.h5
  python server_run/augment_h5_trajectory.py --src tmp_bench.h5 --dst final.h5 --overwrite

退出码: 0 成功(含 DRYRUN / 无事可做);非 0 任何校验失败或异常。
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

# 旧 AID 工作目录(旧结果神圣, 默认禁止写)。common.sh:26 WORK 默认 $ROOT/server_run/work。
OLD_AID_WORK_DIR = Path(__file__).resolve().parent / "work"

COPY_ALWAYS = ("trajectory", "full_scene")           # 主回填目标(受 --overwrite 控制)
COPY_IF_MISSING = ("waypoints", "segment_angles")    # 仅目标缺失时顺带补齐


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def _convert(name: str, data):
    """写入前的 dtype 规范化 —— 与 build_dataset_from_benchmark.py 转换器保持一致。"""
    if name in ("trajectory", "waypoints", "segment_angles"):
        return np.asarray(data, dtype=np.float32)
    return data  # full_scene: 原样(实际链路为 float32 0-255;v2.7.11 更正)


def build_plan(src_f, dst_f, overwrite: bool):
    """扫描 + 逐样本身份校验;只读, 不做任何写入。

    返回 (plan, offenders, weak):
      plan      [(key, dataset_name, action)]  action ∈ create/overwrite/skip_exists
      offenders [(key, reason)]  任何一条都必须让整个运行失败
      weak      [key]  只有帧数校验可用的样本(双方都缺 num_segments/waypoints)
    """
    if 'data' not in src_f:
        fail("源文件缺 'data' 组(应为 v5 benchmark 布局 data/sample_xxxxx)")
    if 'data' not in dst_f:
        fail("目标文件缺 'data' 组(应为 final LingBot 布局 f['data']/sample_xxxxx;"
             "注意 evaluate.py 读的就是 f['data'], 不是 'samples')")

    src_data, dst_data = src_f['data'], dst_f['data']
    dst_keys = sorted(dst_data.keys())
    if not dst_keys:
        fail("目标 'data' 组为空(0 个样本) —— 不是有效的 final H5")
    plan, offenders, weak = [], [], []

    for key in dst_keys:
        g = dst_data[key]
        if key not in src_data:
            offenders.append((key, "目标样本在源 benchmark 中不存在"))
            continue
        s = src_data[key]

        # ---- 源必须有主回填字段(否则传错了源文件) ----
        missing_src = [n for n in COPY_ALWAYS if n not in s]
        if missing_src:
            offenders.append((key, f"源样本缺 {missing_src}(不是 v5 源 benchmark?)"))
            continue

        # ---- 身份校验: 绝不静默写入错误轨迹 ----
        traj_len = int(s['trajectory'].shape[0])
        nf = g.attrs.get('num_frames')
        if nf is None:
            offenders.append((key, "目标缺 attrs['num_frames'](final H5 结构异常)"))
            continue
        if int(nf) != traj_len:
            offenders.append((key, f"帧数不匹配: 目标 num_frames={int(nf)} "
                                   f"vs 源 trajectory 长度={traj_len}"))
            continue
        stf = s.attrs.get('total_frames')
        if stf is not None and int(stf) != int(nf):
            offenders.append((key, f"帧数不匹配: 目标 num_frames={int(nf)} "
                                   f"vs 源 total_frames={int(stf)}"))
            continue

        strong = False
        s_nseg, d_nseg = s.attrs.get('num_segments'), g.attrs.get('num_segments')
        if s_nseg is not None and d_nseg is not None:
            if int(s_nseg) != int(d_nseg):
                offenders.append((key, f"num_segments 不匹配: 目标 {int(d_nseg)} "
                                       f"vs 源 {int(s_nseg)}"))
                continue
            strong = True
        if 'waypoints' in s and 'waypoints' in g:
            swp = np.asarray(s['waypoints'][:], dtype=np.float64)
            dwp = np.asarray(g['waypoints'][:], dtype=np.float64)
            if swp.shape != dwp.shape:
                offenders.append((key, f"waypoints 形状不匹配: 目标 {dwp.shape} "
                                       f"vs 源 {swp.shape}"))
                continue
            # 转换器写 float32, 源可能是 int/float64 -> 容忍量化误差, 不容忍错配
            if not np.allclose(swp, dwp, atol=1e-2):
                offenders.append((key, "waypoints 数值不匹配(allclose 失败)"))
                continue
            strong = True
        if not strong:
            weak.append(key)

        # ---- 该样本通过校验, 生成动作 ----
        for name in COPY_ALWAYS:
            if name in g:
                plan.append((key, name, 'overwrite' if overwrite else 'skip_exists'))
            else:
                plan.append((key, name, 'create'))
        for name in COPY_IF_MISSING:
            if name in s and name not in g:
                plan.append((key, name, 'create'))

    return plan, offenders, weak


def apply_plan(src_f, dst_f, plan):
    """执行计划(dst 已以 r+ 打开)。返回动作计数。"""
    counts = Counter()
    src_data, dst_data = src_f['data'], dst_f['data']
    for key, name, action in plan:
        if action == 'skip_exists':
            counts[f'{name}:skip_exists'] += 1
            continue
        g, s = dst_data[key], src_data[key]
        if action == 'overwrite':
            del g[name]  # 注: HDF5 删除不回收空间(有意不 repack, 保 inode)
        data = _convert(name, s[name][:])
        if name == 'full_scene':
            g.create_dataset('full_scene', data=data, compression='gzip')
        else:
            g.create_dataset(name, data=data)
        if name == 'waypoints' and 'num_segments' not in g.attrs \
                and 'num_segments' in s.attrs:
            # 保持 final H5 不变式: 有 waypoints 必有 attrs['num_segments']
            # (build_dataset_from_benchmark.py 转换器同时写这三者)
            g.attrs['num_segments'] = int(s.attrs['num_segments'])
        counts[f'{name}:{action}'] += 1
    return counts


def post_verify(src_path: Path, dst_path: Path, plan) -> None:
    """写后抽验: 重开文件, 逐字节比对第一个实际写入的 dataset。失败即非零退出。"""
    written = [(k, n, a) for (k, n, a) in plan if a != 'skip_exists']
    if not written:
        return
    key, name, _ = written[0]
    with h5py.File(src_path, 'r') as sf, h5py.File(dst_path, 'r') as df:
        expect = _convert(name, sf['data'][key][name][:])
        got = df['data'][key][name][:]
        if got.shape != expect.shape or not np.array_equal(got, expect):
            fail(f"写后抽验失败: {key}/{name} 与源不一致 "
                 f"(shape {got.shape} vs {expect.shape})")
        if name == 'trajectory' and (got.ndim != 2 or got.shape[1] != 2):
            fail(f"写后抽验失败: {key}/trajectory 形状 {got.shape} 不是 (T,2)")
    print(f"[verify] 写后抽验通过: {key}/{name}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 v5 源 benchmark H5 向既有 final H5 原地回填 "
                    "trajectory/full_scene(以及缺失的 waypoints/segment_angles)。")
    ap.add_argument('--src', required=True,
                    help='v5 源 benchmark H5(tmp_bench, 布局 data/sample_xxxxx)')
    ap.add_argument('--dst', required=True,
                    help='既有 final H5(布局 f[\'data\']/sample_xxxxx), 原地追加')
    ap.add_argument('--overwrite', action='store_true',
                    help='替换目标中已存在的 trajectory/full_scene(默认只补缺)')
    ap.add_argument('--i-know', dest='i_know', action='store_true',
                    help=f'允许写旧 AID 工作目录({OLD_AID_WORK_DIR})下的目标文件')
    args = ap.parse_args()

    dryrun = os.environ.get('DRYRUN', '') == '1'
    src_p, dst_p = Path(args.src), Path(args.dst)
    if not src_p.is_file():
        fail(f"--src 不存在或不是文件: {src_p}")
    if not dst_p.is_file():
        fail(f"--dst 不存在或不是文件: {dst_p}(本脚本只原地增补, 不新建)")
    if src_p.resolve() == dst_p.resolve():
        fail("--src 与 --dst 是同一个文件")

    dst_resolved = dst_p.resolve()
    try:
        dst_resolved.relative_to(OLD_AID_WORK_DIR)
        under_old_aid = True
    except ValueError:
        under_old_aid = False
    if under_old_aid and not args.i_know:
        fail(f"目标位于旧 AID 工作目录(旧结果神圣): {dst_resolved}\n"
             f"确认要写请加 --i-know")

    # ---- 阶段 1: 只读扫描 + 校验(dst 以 'r' 打开, 校验期零写入) ----
    with h5py.File(src_p, 'r') as sf, h5py.File(dst_p, 'r') as df:
        plan, offenders, weak = build_plan(sf, df, args.overwrite)
        extra_src = len(set(sf['data'].keys()) - set(df['data'].keys()))

    if offenders:
        print(f"[FAIL] 校验失败, {len(offenders)} 个违例样本(未写入任何数据):",
              file=sys.stderr)
        for key, reason in offenders:
            print(f"  {key}: {reason}", file=sys.stderr)
        sys.exit(1)

    counts = Counter(f'{n}:{a}' for _, n, a in plan)
    print(f"[plan] 目标样本 {len(set(k for k, _, _ in plan))} 个全部通过校验"
          f"(源多出 {extra_src} 个样本, 忽略)")
    for tag in sorted(counts):
        print(f"  {tag}: {counts[tag]}")
    if weak:
        print(f"  ⚠ {len(weak)} 个样本仅帧数校验可用(双方缺 num_segments/waypoints), "
              f"例如: {weak[:5]}")
    for key, name, action in plan[:10]:
        print(f"    {key}/{name} -> {action}")
    if len(plan) > 10:
        print(f"    ... 共 {len(plan)} 条动作")

    if dryrun:
        print("[DRYRUN] 只打印计划, 未写入任何数据")
        return 0

    todo = [p for p in plan if p[2] != 'skip_exists']
    if not todo:
        print("[done] 无事可做: 目标已含全部字段(替换请加 --overwrite)")
        return 0

    # ---- 阶段 2: 原地写入(dst 以 'r+' 打开, 单文件编辑, inode 不变) ----
    t0 = time.time()
    with h5py.File(src_p, 'r') as sf, h5py.File(dst_p, 'r+') as df:
        done = apply_plan(sf, df, plan)
        if 'metadata' in df:  # 溯源(与转换器 metadata 约定同风格)
            df['metadata'].attrs['trajectory_backfill_src'] = str(src_p.resolve())
            df['metadata'].attrs['trajectory_backfill_time'] = \
                time.strftime('%Y-%m-%d %H:%M:%S')

    post_verify(src_p, dst_p, plan)

    print(f"[done] 写入完成, 耗时 {time.time() - t0:.1f}s: {dst_p}")
    for tag in sorted(done):
        print(f"  {tag}: {done[tag]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
