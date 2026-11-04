#!/usr/bin/env python3
"""Load and validate immutable baseline-card templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from protocol import CARD_SCHEMA, ProtocolError

HERE = Path(__file__).resolve().parent
CARD_DIR = HERE / "cards"
REQUIRED = (
    "name", "source", "checkpoint", "native_input", "native_output",
    "number_of_condition_views", "uses_camera_pose", "uses_future_gt",
    "training_data", "adaptation", "notes",
)


def load_card(name: str) -> dict[str, Any]:
    path = CARD_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"baseline card missing for {name!r}: {path}; paper runs require a card")
    card = json.loads(path.read_text(encoding="utf-8"))
    if card.get("schema") != CARD_SCHEMA:
        raise ProtocolError(f"{path}: schema must be {CARD_SCHEMA!r}")
    missing = [key for key in REQUIRED if key not in card]
    if missing:
        raise ProtocolError(f"{path}: missing fields {missing}")
    if card["name"] != name:
        raise ProtocolError(f"{path}: card.name={card['name']!r} != {name!r}")
    if not isinstance(card["uses_future_gt"], bool):
        raise ProtocolError(f"{path}: uses_future_gt must be bool")
    if not isinstance(card["uses_camera_pose"], bool):
        raise ProtocolError(f"{path}: uses_camera_pose must be bool")
    views = card["number_of_condition_views"]
    if isinstance(views, bool) or not isinstance(views, int) or views < 1:
        raise ProtocolError(
            f"{path}: number_of_condition_views must be a positive integer")
    for key in ("source", "native_input", "native_output", "training_data",
                "adaptation", "notes"):
        if not isinstance(card[key], str) or not card[key].strip():
            raise ProtocolError(f"{path}: {key} must be a non-empty string")
    if "main_table_eligible" in card and not isinstance(
            card["main_table_eligible"], bool):
        raise ProtocolError(f"{path}: main_table_eligible must be bool")
    if card["uses_future_gt"] and card.get("main_table_eligible", False):
        raise ProtocolError(
            f"{path}: a future-GT diagnostic cannot be main-table eligible")
    return card


def list_cards() -> list[str]:
    return sorted(path.stem for path in CARD_DIR.glob("*.json"))
