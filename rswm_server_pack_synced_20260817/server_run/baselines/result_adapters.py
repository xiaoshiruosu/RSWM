#!/usr/bin/env python3
"""Adapters for methods whose real inference is produced by existing runners.

These adapters do not load or modify the Video DiT.  They ingest per-sample `.npy`
rollouts from an explicit producer directory and convert them to the existing
`registry.py` record contract.  Missing products fail loudly; no black/static fallback
is synthesized.  RSWISE image outputs are accepted only after an explicitly labelled
recursive/per-waypoint producer has created a full T-frame adapted rollout.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from protocol import normalize_video
from registry import (ADAPTER_VERSION, BaselineAdapter, SampleUnavailable,
                      direction_word, register, validate_record)


class RolloutDirectoryAdapter(BaselineAdapter):
    kind = "external_generative"
    requires_gpu = True
    status = "implemented"
    upper_bound = False
    output_env = ""
    condition_type = "text_plucker"
    method_scope = "result_ingest_bridge"

    def missing_dependencies(self) -> list[str]:
        raw = os.environ.get(self.output_env, "").strip()
        if not raw:
            return [f"set {self.output_env}=<producer eval directory>"]
        root = Path(raw)
        generated = root / "generated"
        if not generated.is_dir():
            return [f"missing {generated}"]
        return []

    def _load_meta(self, root: Path, sample_id: str) -> dict[str, Any]:
        for path in (root / "meta" / f"{sample_id}.json",
                     root / f"{sample_id}.json"):
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: metadata must be an object")
                return value
        return {}

    def run(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key in ("sample_id", "prompt", "source_frame", "gt_frames",
                    "waypoints", "segment_angles", "gen_seed"):
            if key not in sample:
                raise KeyError(f"{self.name}: sample missing {key!r}")
        raw = os.environ.get(self.output_env, "").strip()
        if not raw:
            raise SampleUnavailable(
                f"{self.name}: {self.output_env} is unset; bridge exists but no real "
                "producer output was supplied")
        root = Path(raw)
        product = root / "generated" / f"{sample['sample_id']}.npy"
        if not product.is_file():
            raise SampleUnavailable(f"{self.name}: missing {product}")
        import numpy as np

        started = time.perf_counter()
        target = normalize_video(sample["gt_frames"], name="target",
                                 expected_t=int(sample.get("T", len(sample["gt_frames"]))))
        generated = normalize_video(np.load(product, allow_pickle=False),
                                    name=str(product), expected_t=len(target))
        if generated.shape != target.shape:
            raise SampleUnavailable(
                f"{self.name}: generated {generated.shape} != target {target.shape}; "
                "the producer must export the shared benchmark resolution")
        source = np.asarray(sample["source_frame"])
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError(f"{self.name}: source_frame shape={source.shape}")
        if source.dtype != np.uint8:
            source = np.clip(source, 0, 255).astype(np.uint8)
        if source.shape != target.shape[1:]:
            # Existing evaluators already permit a condition crop at another native
            # resolution.  Resize only the source record, never the generated/GT video.
            import cv2
            source = cv2.resize(source, (target.shape[2], target.shape[1]),
                                interpolation=cv2.INTER_AREA)
        meta = self._load_meta(root, str(sample["sample_id"]))
        producer_method = str(meta.get("method") or meta.get("experiment_id") or self.name)
        if producer_method != self.name and meta.get("method") is not None:
            raise SampleUnavailable(
                f"{product}: producer metadata method={producer_method!r} != {self.name!r}")
        if meta.get("condition_type") is not None and \
                str(meta["condition_type"]) != self.condition_type:
            raise SampleUnavailable(
                f"{product}: producer condition_type={meta['condition_type']!r} != "
                f"adapter contract {self.condition_type!r}")
        if meta.get("gen_seed") is not None and \
                int(meta["gen_seed"]) != int(sample["gen_seed"]):
            raise SampleUnavailable(
                f"{product}: producer gen_seed={meta['gen_seed']} != frozen "
                f"manifest seed={sample['gen_seed']}")
        producer_key = meta.get("sample_id", meta.get("key"))
        if producer_key is not None and str(producer_key) != str(sample["sample_id"]):
            raise SampleUnavailable(
                f"{product}: producer sample identity {producer_key!r} disagrees with "
                f"{sample['sample_id']!r}")
        producer_t = meta.get("num_frames_gen", meta.get("num_frames"))
        if producer_t is not None and int(producer_t) != len(generated):
            raise SampleUnavailable(
                f"{product}: producer metadata T={producer_t} != saved rollout "
                f"T={len(generated)}")
        condition_type = self.condition_type
        record = {
            "sample_id": str(sample["sample_id"]),
            "generated_frames": generated,
            "source_frames": source[None],
            "target_frames": target,
            "route_metadata": {
                "prompt": str(sample["prompt"]),
                "waypoints": np.asarray(sample["waypoints"]).tolist(),
                "segment_angles": np.asarray(sample["segment_angles"]).reshape(-1).tolist(),
                "num_segments": int(len(np.asarray(sample["waypoints"])) - 1),
                "direction_word": direction_word(sample["prompt"], sample["waypoints"]),
                "condition_type": condition_type,
            },
            "runtime_metadata": {
                "wall_seconds": float(meta.get("infer_seconds", time.perf_counter() - started)),
                "gen_seed": int(meta.get("gen_seed", sample["gen_seed"])),
                "resolution": [int(generated.shape[1]), int(generated.shape[2])],
                "num_frames": int(len(generated)),
                "device": str(meta.get("device", "external-producer")),
                "method_version": str(meta.get(
                    "method_version", f"{ADAPTER_VERSION}/{self.method_scope}")),
            },
        }
        return validate_record(record)


def _adapter(name: str, env: str, condition: str, description: str,
             *, requires_gpu: bool = True) -> RolloutDirectoryAdapter:
    cls = type(
        "ConfiguredRolloutAdapter_" + name,
        (RolloutDirectoryAdapter,),
        {
            "name": name,
            "output_env": env,
            "condition_type": condition,
            "description": description,
            "requires_gpu": requires_gpu,
        },
    )
    return cls()


register(_adapter(
    "rswise_recursive_bridge", "RSWM_RSWISE_EVAL_DIR", "text_only",
    "Ingests an explicitly adapted recursive/per-waypoint RSWISE-style rollout. "
    "The original method remains a single-image method and is not available here."))
register(_adapter(
    "frozen_i2v", "RSWM_FROZEN_I2V_EVAL_DIR", "text_plucker",
    "Frozen pretrained Wan/LingBot I2V output; zero adaptation parameters."))
register(_adapter(
    "video_no_camera", "RSWM_NO_CAMERA_EVAL_DIR", "text_only",
    "Same video backbone with camera/Plucker conditioning disabled."))
register(_adapter(
    "rsroll_native_attn_only_r16", "RSWM_NATIVE_ATTN_ONLY_R16_EVAL_DIR", "text_plucker",
    "Matched ablation: native pose route, attention-only LoRA, rank 16."))
register(_adapter(
    "rsroll_native_attn_cam_r16", "RSWM_NATIVE_ATTN_CAM_R16_EVAL_DIR", "text_plucker",
    "Main camera-aware arm: native pose route, attention+camera LoRA, rank 16."))
register(_adapter(
    "rsroll_drop_attn_only_r16", "RSWM_DROP_ATTN_ONLY_R16_EVAL_DIR", "text_only",
    "Matched ablation: dropped pose route, attention-only LoRA, rank 16."))
register(_adapter(
    "rsroll_drop_attn_cam_r16", "RSWM_DROP_ATTN_CAM_R16_EVAL_DIR", "text_only",
    "Matched ablation: dropped pose route, attention+camera LoRA, rank 16."))
register(_adapter(
    "rsroll_native_all_r16", "RSWM_NATIVE_ALL_R16_EVAL_DIR", "text_plucker",
    "Optional matched ablation: native pose route, all LoRA targets, rank 16."))
register(_adapter(
    "rsroll_drop_all_r16", "RSWM_DROP_ALL_R16_EVAL_DIR", "text_only",
    "Optional matched ablation: dropped pose route, all LoRA targets, rank 16."))
