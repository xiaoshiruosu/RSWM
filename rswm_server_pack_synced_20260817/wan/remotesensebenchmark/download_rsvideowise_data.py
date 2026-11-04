"""Utility script to pre-download HuggingFace datasets used by build_rsvideowise.

Run this once on a machine with internet access. It will download and cache
all configured HuggingFace datasets locally (under ``HF_HOME`` /
``HF_DATASETS_CACHE``; the A800 deployment maps these to
``/nix/rsroll_25828/cache/huggingface``). The main benchmark script
`build_rsvideowise.py` will then load from this cache automatically.
"""

import sys

from build_rsvideowise import HF_SCENE_CONFIG, hf_load_dataset


def main() -> None:
    if hf_load_dataset is None:
        raise RuntimeError(
            "HuggingFace 'datasets' library is not available. Install it via 'pip install datasets'."
        )

    for scene, cfg in HF_SCENE_CONFIG.items():
        dataset_id = cfg.get("dataset") or ""
        split = cfg.get("split") or "train"
        if not dataset_id:
            continue
        print(f"Downloading scene '{scene}' from HuggingFace dataset '{dataset_id}' (split='{split}')...")
        ds = hf_load_dataset(dataset_id, split=split)
        print(f"  -> downloaded/cached {len(ds)} samples for scene '{scene}'.")

    print("All configured HuggingFace datasets are downloaded and cached.")


if __name__ == "__main__":
    main()
