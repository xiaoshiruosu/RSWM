"""Small explicit registry for WorldModel factories."""

from __future__ import annotations

from typing import Callable, Dict

from .world_model import validate_world_model

_BUILDERS: Dict[str, Callable[..., object]] = {}


def register_world_model(name: str, builder: Callable[..., object]) -> None:
    key = str(name).strip()
    if not key or not callable(builder):
        raise TypeError("world model registry requires a non-empty name and callable builder")
    if key in _BUILDERS:
        raise KeyError(f"world model {key!r} is already registered")
    _BUILDERS[key] = builder


def build_world_model(name: str, **kwargs):
    try:
        builder = _BUILDERS[str(name)]
    except KeyError as exc:
        raise KeyError(f"unknown world model {name!r}; available={sorted(_BUILDERS)}") from exc
    value = builder(**kwargs)
    validate_world_model(value, label=f"world_model[{name}]")
    return value


def list_world_models() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))
