#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pack_path.py — 定位既有工程根，并把"找不到"变成**硬失败**而不是静默跳过。

**为什么单独有这个文件**（2026-07-30 审查发现的致命问题）：
各模块最强的断言——与真实 `trajectory_metrics.py` 的互操作、与真实 `_make_prompt`
的 310 角逐位一致、与真实 `_calculate_overlap` 的 400 窗逐位一致——都依赖找得到既有工程。
最初把路径**写死在本次会话的 /tmp 目录**里，于是换任何一台机器（尤其是最终要跑的
H200）那些断言全部变成 SKIP，而自检**照样打印"全部通过"并退出 0**。
实测：把路径指向 /nonexistent，`geom_registration.py --self-test` 仍然退出 0。
一个"绿"的自检套件在目标机器上其实什么跨模块的东西都没验证——这比没有自检更危险。

规则（本文件强制）：
  * 工程根优先取环境变量 `RSWM_PACK_ROOT`；否则按候选列表探测。
  * 找不到时**默认退出码非 0**；要在无工程的机器上只跑纯本地断言，
    必须显式传 `--allow-skip`，且会打印醒目横幅说明"哪些断言没跑"。
"""

from __future__ import annotations

import os
from typing import List, Optional

ENV_VAR = "RSWM_PACK_ROOT"

# 候选路径按可信度排序，第一个通过哨兵校验的胜出。
# 刻意**不**放任何开发机专属的绝对路径：那种路径发到服务器上就是死的，
# 而"候选里有一条永远不成立的路径"会让 find_pack_root 的行为随机器而变。
_THIS_PACK = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_SAFE_BASE = os.environ.get("RSWM_BASE", "/nix/rsroll_25828")
_CANDIDATES: List[str] = [
    _THIS_PACK,                                      # 当前交付包根
    os.getcwd(),                                     # 已经 cd 到包根
    os.path.join(os.getcwd(), "rswm_server_pack"),  # 当前目录下的解压包
    os.path.join(_SAFE_BASE, "rswm_server_pack"),   # 共享 A800 标准根
]

# 用来确认"这确实是工程根"而不是同名空目录
_SENTINELS = ("lora_remotesense/new/trajectory_metrics.py",
              "wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py")


def _looks_like_pack(path: str) -> bool:
    return bool(path) and all(os.path.isfile(os.path.join(path, s)) for s in _SENTINELS)


def find_pack_root() -> Optional[str]:
    """返回既有工程根，找不到返回 None。会校验哨兵文件，避免指到空壳目录。

    **显式设了 `RSWM_PACK_ROOT` 却指错时不回退候选**：静默回退等于把
    "你的配置写错了"变成"看起来一切正常"，与本项目的 fail-loud 规范相悖。
    """
    env = os.environ.get(ENV_VAR)
    if env:
        if _looks_like_pack(env):
            return env
        print("[pack_path] %s=%r 不是工程根（缺 %s）；**不回退**候选路径。"
              % (ENV_VAR, env, _SENTINELS[0]))
        return None
    for c in _CANDIDATES:
        if _looks_like_pack(c):
            return c
    return None


def skip_banner(missing_what: str) -> None:
    """打印醒目横幅，明确说明哪些断言**没有**被验证。"""
    bar = "!" * 74
    print("\n" + bar)
    print("!! 跳过：找不到既有工程根，以下断言【未验证】：%s" % missing_what)
    print("!! 设 %s=<工程根> 后重跑；该目录须含 %s" % (ENV_VAR, _SENTINELS[0]))
    print("!! 本次退出码将为非 0（除非显式 --allow-skip）——跳过不算通过。")
    print(bar + "\n")


def resolve_or_exit_code(allow_skip: bool, missing_what: str):
    """返回 (pack_root, exit_code_if_missing)。

    pack_root 为 None 时：allow_skip=True → 建议退出码 0（但已打横幅）；
    否则 → 建议退出码 2（跳过不算通过）。
    """
    root = find_pack_root()
    if root is not None:
        return root, 0
    skip_banner(missing_what)
    return None, (0 if allow_skip else 2)
