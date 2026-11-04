#!/usr/bin/env python3
"""Small, dependency-light helpers for Hugging Face image records.

Hugging Face image columns are not guaranteed to be decoded eagerly.  In
particular, a Parquet-backed ``datasets.Image`` value may arrive as a mapping
with ``bytes`` and ``path`` members instead of as ``PIL.Image.Image``.  Keep
the conversion in one place so benchmark builders and integrity checks use
identical, fail-closed semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Optional, Union


WHURS19_REPO = "jonathan-roberts1/WHU-RS19"

# ClassLabel order published in the WHU-RS19 Hugging Face dataset card.  The
# local Parquet path can expose ``label`` as plain int64 on some datasets
# versions, so the official mapping is retained for deterministic filtering.
WHURS19_LABEL_NAMES = (
    "airport", "beach", "bridge", "commercial", "desert", "farmland",
    "football field", "forest", "industrial", "meadow", "mountain",
    "park", "parking", "pond", "port", "railway station", "residential",
    "river", "viaduct",
)


class ImageFieldDecodeError(ValueError):
    """The image field had a supported shape but its content was not decodable."""


def decode_hf_image(field, *, context: str = "image"):
    """Decode a PIL image or a Hugging Face ``{bytes, path}`` image field.

    Bytes are authoritative when present.  ``path`` is consulted only when
    ``bytes`` is ``None``.  The returned RGB image is fully loaded and detached
    from its source stream/file, so callers do not keep file descriptors or
    ``BytesIO`` objects alive.
    """

    try:
        from PIL import Image, UnidentifiedImageError
    except Exception as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("Pillow is required to decode image fields") from exc

    def _detach_rgb(opened):
        opened.load()
        return opened.convert("RGB").copy()

    if isinstance(field, Image.Image):
        try:
            return _detach_rgb(field)
        except (OSError, ValueError) as exc:
            raise ImageFieldDecodeError(
                f"{context}: failed to decode PIL image: {type(exc).__name__}: {exc}"
            ) from exc

    if not isinstance(field, Mapping):
        raise TypeError(
            f"{context}: Unsupported image field type: {type(field).__name__}; "
            "expected PIL.Image.Image or a mapping with 'bytes'/'path'"
        )

    payload = field.get("bytes")
    path = field.get("path")

    if payload is not None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"{context}: image['bytes'] must be bytes-like or None, got "
                f"{type(payload).__name__}"
            )
        try:
            with BytesIO(bytes(payload)) as stream:
                with Image.open(stream) as opened:
                    return _detach_rgb(opened)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageFieldDecodeError(
                f"{context}: failed to decode image['bytes']: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if path is not None:
        if not isinstance(path, (str, PathLike)):
            raise TypeError(
                f"{context}: image['path'] must be str/path-like or None, got "
                f"{type(path).__name__}"
            )
        if not str(path):
            raise ValueError(f"{context}: image['path'] is empty")
        try:
            with Image.open(path) as opened:
                return _detach_rgb(opened)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageFieldDecodeError(
                f"{context}: failed to decode image['path']={path!s}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise ValueError(
        f"{context}: image mapping contains neither non-null 'bytes' nor 'path'"
    )


def find_hf_snapshot_parquet(
        repo_id: str,
        *,
        hf_home: Optional[Union[str, PathLike]] = None,
        snapshot_root: Optional[Union[str, PathLike]] = None,
        pattern: str = "train-*.parquet") -> Optional[Path]:
    """Locate a cached dataset Parquet without scanning unrelated cache trees.

    ``snapshot_root`` may point at the repository cache directory, one concrete
    snapshot, its ``data`` directory, or the Parquet file itself.  Otherwise
    the conventional ``$HF_HOME/hub/datasets--ORG--NAME`` location is used.
    The newest matching snapshot is preferred; ambiguity is resolved
    deterministically by path name.
    """

    roots = []
    if snapshot_root is not None:
        roots.append(Path(snapshot_root))
    elif hf_home is not None:
        cache_name = "datasets--" + repo_id.replace("/", "--")
        roots.append(Path(hf_home) / "hub" / cache_name)

    matches = []
    for root in roots:
        if root.is_file():
            if root.match(pattern):
                matches.append(root)
            continue
        if not root.is_dir():
            continue
        for candidate_pattern in (
                f"data/{pattern}",
                f"snapshots/*/data/{pattern}",
                pattern):
            matches.extend(p for p in root.glob(candidate_pattern) if p.is_file())

    if not matches:
        return None
    unique = {str(p.resolve()): p for p in matches}
    return sorted(unique.values(), key=lambda p: (p.stat().st_mtime_ns, str(p)))[-1]
