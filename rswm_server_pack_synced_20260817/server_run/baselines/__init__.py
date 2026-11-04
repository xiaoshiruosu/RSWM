#!/usr/bin/env python3
# =============================================================================
# server_run/baselines — RSWM 统一 baseline 适配器层
#
# 契约、两类基线的定义、共享评估协议:见仓库根 README_BASELINES.md。
#
# import 本包 **不需要** numpy(registry/external_stubs 只用 stdlib;numpy 在
# run()/validate_record() 里惰性 import),所以在没装 numpy 的机器上也能列基线、
# 查状态、跑接线自检:
#     python3 server_run/baselines/registry.py
#
# 两类基线:
#   diagnostic          —— 从静态场景 + GT 轨迹合成的非生成式对照,用来证明指标
#                          能区分四种失败模式。其中 gt_oracle 是 **指标上界**,
#                          不是可部署方法。
#   external_generative —— 外部生成式模型。**截至目前一个都没跑过**,全是 stub。
# =============================================================================
from __future__ import annotations

from .registry import (  # noqa: F401
    ADAPTER_VERSION,
    CONDITION_TYPES,
    KINDS,
    RECORD_KEYS,
    ROUTE_KEYS,
    RUNTIME_KEYS,
    STATUSES,
    BaselineAdapter,
    DiagnosticAdapter,
    direction_word,
    get,
    list_baselines,
    register,
    validate_record,
)

__all__ = [
    "ADAPTER_VERSION",
    "CONDITION_TYPES",
    "KINDS",
    "RECORD_KEYS",
    "ROUTE_KEYS",
    "RUNTIME_KEYS",
    "STATUSES",
    "BaselineAdapter",
    "DiagnosticAdapter",
    "direction_word",
    "get",
    "list_baselines",
    "register",
    "validate_record",
]
