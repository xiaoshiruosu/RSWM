"""A materialized image must appear complete and cannot replace an old artifact."""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rswm_inria import data


@pytest.fixture
def world_input(monkeypatch):
    source = np.full((5000, 5000, 3), 37, np.uint8)
    monkeypatch.setattr(data, "source_pixels", lambda path, row: (source, "a" * 64, {}))
    return {"world_id": "publication-fixture", "path": "fixture.tif"}


def test_world_png_is_not_visible_while_encoder_writes(tmp_path, world_input, monkeypatch):
    target = tmp_path / "publication-fixture.png"
    original = Image.Image.save

    def during_write(image, stream, *args, **kwargs):
        if isinstance(stream, (str, Path)):
            Path(stream).write_bytes(b"partial encoder output")
        else:
            stream.write(b"partial encoder output")
            stream.flush()
        assert not target.exists(), "A partial PNG was publicly visible before completion"
        if not isinstance(stream, (str, Path)):
            stream.seek(0)
            stream.truncate()
        return original(image, stream, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", during_write)
    record = data.build_world("fixture.tif", world_input, tmp_path)
    with Image.open(target) as image:
        assert image.size == (2048, 2048) and np.asarray(image).min() == 37
    assert record["world_file_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_encoder_failure_does_not_publish_a_partial_world(tmp_path, world_input, monkeypatch):
    def fail(image, stream, *args, **kwargs):
        if isinstance(stream, (str, Path)):
            Path(stream).write_bytes(b"partial encoder output")
        else:
            stream.write(b"partial encoder output")
        raise OSError("injected encoder failure")

    monkeypatch.setattr(Image.Image, "save", fail)
    with pytest.raises(OSError, match="encoder failure"):
        data.build_world("fixture.tif", world_input, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_publication_preserves_the_original_lossless_encoding(tmp_path, world_input):
    expected = io.BytesIO()
    Image.fromarray(np.full((2048, 2048, 3), 37, np.uint8)).save(expected, format="PNG", compress_level=6)
    record = data.build_world("fixture.tif", world_input, tmp_path)
    assert (tmp_path / record["path"]).read_bytes() == expected.getvalue()


def test_encoder_has_no_named_partial_staging_file(tmp_path, monkeypatch):
    from rswm_inria.artifact_io import save_png_exclusive
    original = Image.Image.save

    def during_encode(image, stream, *args, **kwargs):
        assert list(tmp_path.iterdir()) == []
        result = original(image, stream, *args, **kwargs)
        assert list(tmp_path.iterdir()) == []
        return result

    monkeypatch.setattr(Image.Image, "save", during_encode)
    save_png_exclusive(tmp_path / "output.png", np.zeros((4, 4, 3), np.uint8))
    assert list(tmp_path.iterdir()) == [tmp_path / "output.png"]


def test_existing_image_cannot_be_overwritten(tmp_path):
    from rswm_inria.artifact_io import save_png_exclusive
    target = tmp_path / "existing.png"
    target.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError):
        save_png_exclusive(target, np.zeros((4, 4, 3), np.uint8))
    assert target.read_bytes() == b"existing user artifact"
    assert list(tmp_path.iterdir()) == [target]


def test_file_sync_failure_prevents_publication(tmp_path, monkeypatch):
    from rswm_inria import artifact_io
    def fail(descriptor):
        raise OSError("injected file synchronization failure")
    monkeypatch.setattr(artifact_io.os, "fsync", fail)
    with pytest.raises(OSError, match="synchronization failure"):
        artifact_io.save_png_exclusive(tmp_path / "output.png", np.zeros((4, 4, 3), np.uint8))
    assert list(tmp_path.iterdir()) == []


def test_preexisting_staging_name_is_not_deleted(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rswm_inria import artifact_io
    monkeypatch.setattr(artifact_io.uuid, "uuid4", lambda: SimpleNamespace(hex="collision"))
    existing = tmp_path / ".output.png.collision.tmp"
    existing.write_bytes(b"existing user staging artifact")
    with pytest.raises(FileExistsError):
        artifact_io.save_png_exclusive(tmp_path / "output.png", np.zeros((4, 4, 3), np.uint8))
    assert existing.read_bytes() == b"existing user staging artifact"
    assert list(tmp_path.iterdir()) == [existing]
