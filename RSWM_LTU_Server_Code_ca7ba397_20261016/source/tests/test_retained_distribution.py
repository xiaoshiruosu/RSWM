"""Retained archives must identify actual bytes, independent of their receipt."""
from argparse import Namespace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).parents[1]


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def distribution(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return load_script("retain_distribution")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def save_tar(path, contents):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def save_wheel(path, contents):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)


@pytest.fixture
def archives(tmp_path, distribution):
    commit = "a" * 40
    contents = {"src/rswm_inria/__init__.py": b"__version__ = '0.3.0'\n",
                "README.md": b"fixture retained archive\n", "setup.py": b"# source fixture\n"}
    source = sha(json.dumps({"__init__.py": sha(contents["src/rswm_inria/__init__.py"])},
                            sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    inputs = {name: sha(content) for name, content in contents.items()}
    provenance = {"schema": "rswm-build-info-v1", "source_commit": commit,
                  "source_dirty": False, "source_code_sha256": source,
                  "source_archive_inputs": inputs}
    metadata = b"Name: rswm-inria-120\nVersion: 0.3.0\n"
    wheel_contents = {"rswm_inria/__init__.py": contents["src/rswm_inria/__init__.py"],
                      "rswm_inria/_build_info.json": json.dumps(provenance).encode(),
                      "rswm_inria_120-0.3.0.dist-info/METADATA": metadata}
    sdist_contents = {"rswm_inria_120-0.3.0/" + name: content for name, content in contents.items()}
    sdist_contents["rswm_inria_120-0.3.0/src/rswm_inria/_build_info.json"] = json.dumps(provenance).encode()
    sdist_contents["rswm_inria_120-0.3.0/PKG-INFO"] = metadata
    dist = tmp_path / "retained" / "dist"
    dist.mkdir(parents=True)
    wheel, sdist = dist / "rswm_inria_120-0.3.0-py3-none-any.whl", dist / "rswm_inria_120-0.3.0.tar.gz"
    save_wheel(wheel, wheel_contents)
    save_tar(sdist, sdist_contents)
    return Namespace(root=dist.parent, dist=dist, wheel=wheel, sdist=sdist, commit=commit,
                     source=source, inputs=inputs, wheel_contents=wheel_contents, sdist_contents=sdist_contents)


def write_retained(distribution, archives):
    (archives.root / "dependencies").mkdir(exist_ok=True)
    (archives.root / "build.stdout.log").write_text("fixture build command output\n")
    (archives.root / "build.stderr.log").write_text("")
    inventory = distribution.inspect_distributions(archives.dist, archives.commit, archives.source)
    manifest = {"schema": distribution.SCHEMA, "source_commit": archives.commit, "source_dirty": False,
                "worktree_dirty": False, "source_code_sha256": archives.source,
                "source_archive_inputs": archives.inputs, "archives": inventory,
                "retained_files": distribution.retained_files(archives.root)}
    distribution.save_json(archives.root / "build-manifest.json", manifest)
    refresh_sums(distribution, archives.root, manifest)
    return manifest


def refresh_sums(distribution, root, manifest):
    hashes = {name: row["sha256"] for name, row in manifest["retained_files"].items()}
    hashes["build-manifest.json"] = distribution.file_hash(root / "build-manifest.json")
    (root / "SHA256SUMS").write_text("".join(value + "  " + name + "\n" for name, value in sorted(hashes.items())))


def test_retained_inventory_maps_actual_authored_inputs(distribution, archives):
    write_retained(distribution, archives)
    result = distribution.verify_retained(archives.root, archives.commit, archives.source)
    mappings = {row["kind"]: row["packaged_inputs"] for row in result["archives"]}
    assert {row["source_input"] for row in mappings["sdist"] if row["source_input"]} == set(archives.inputs)
    assert {row["source_input"] for row in mappings["wheel"] if row["source_input"]} == {"src/rswm_inria/__init__.py"}
    assert all(row["bytes"] > 0 and row["sha256"] for row in result["archives"])


def test_wheel_legal_notice_is_mapped_to_actual_authored_input(distribution, archives):
    notice = b"fixture legal notice\n"
    archives.inputs["NOTICE.md"] = sha(notice)
    for kind in ("wheel", "sdist"):
        contents = getattr(archives, kind + "_contents")
        info_name = "rswm_inria/_build_info.json" if kind == "wheel" else "rswm_inria_120-0.3.0/src/rswm_inria/_build_info.json"
        info = json.loads(contents[info_name])
        info["source_archive_inputs"] = archives.inputs
        contents[info_name] = json.dumps(info).encode()
        name = "rswm_inria_120-0.3.0.dist-info/licenses/NOTICE.md" if kind == "wheel" else "rswm_inria_120-0.3.0/NOTICE.md"
        contents[name] = notice
        (save_wheel if kind == "wheel" else save_tar)(getattr(archives, kind), contents)
    result = distribution.inspect_distributions(archives.dist, archives.commit, archives.source)
    assert [row for row in result[0]["packaged_inputs"] if row["source_input"] == "NOTICE.md"][0]["sha256"] == sha(notice)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_missing_required_archive_fails(distribution, archives, kind):
    getattr(archives, kind).unlink()
    with pytest.raises(ValueError, match="Exactly one wheel"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_unexpected_archive_is_not_silently_ignored(distribution, archives):
    (archives.dist / "old-wheel.whl").write_bytes(b"old archive")
    with pytest.raises(ValueError, match="Exactly one wheel"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


@pytest.mark.parametrize("name", ["data/raw.tif", "weights/backend.safetensors", ".env", "__pycache__/x.pyc",
                                  ".cache/run.json", ".aws/credentials", "model.ckpt", "credentials.pem",
                                  "pytorch_model.bin", "model.onnx", "model.gguf", "source.parquet"])
def test_forbidden_assets_rejected_from_actual_archive(distribution, archives, name):
    contents = {**archives.wheel_contents, name: b"forbidden fixture bytes"}
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="Forbidden packaged file"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_private_key_material_rejected(distribution, archives):
    contents = {**archives.wheel_contents, "unexpected.txt": b"-----BEGIN " + b"PRIVATE KEY-----\nfixture"}
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="Credential material"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_sdist_claimed_input_hash_cannot_replace_actual_bytes(distribution, archives):
    contents = {**archives.sdist_contents, "rswm_inria_120-0.3.0/README.md": b"tampered documentation\n"}
    save_tar(archives.sdist, contents)
    with pytest.raises(ValueError, match="Actual packaged source input"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_missing_sdist_authored_input_rejected(distribution, archives):
    contents = dict(archives.sdist_contents)
    del contents["rswm_inria_120-0.3.0/README.md"]
    save_tar(archives.sdist, contents)
    with pytest.raises(ValueError, match="Required authored inputs missing"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


@pytest.mark.parametrize("kind,name", [("wheel", "model.unknown"),
                                     ("wheel", "rswm_inria_120-0.3.0.dist-info/model.unknown"),
                                     ("sdist", "rswm_inria_120-0.3.0/unrecorded.txt"),
                                     ("sdist", "rswm_inria_120-0.3.0/src/rswm_inria_120.egg-info/model.unknown")])
def test_unrecorded_packaged_input_is_not_called_generated_metadata(distribution, archives, kind, name):
    contents = {**getattr(archives, kind + "_contents"), name: b"unexpected packaged input"}
    (save_wheel if kind == "wheel" else save_tar)(getattr(archives, kind), contents)
    with pytest.raises(ValueError, match="Unexpected unrecorded packaged input"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


@pytest.mark.parametrize("field,value", [("source_commit", "b" * 40), ("source_code_sha256", "b" * 64),
                                        ("source_dirty", True)])
def test_archive_provenance_must_name_expected_clean_identity(distribution, archives, field, value):
    contents = dict(archives.wheel_contents)
    info = json.loads(contents["rswm_inria/_build_info.json"])
    info[field] = value
    contents["rswm_inria/_build_info.json"] = json.dumps(info).encode()
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="supplied clean commit"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_duplicate_wheel_member_rejected(distribution, archives):
    with zipfile.ZipFile(archives.wheel, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("rswm_inria/__init__.py", b"duplicate")
    with pytest.raises(ValueError, match="Duplicate archive member"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


@pytest.mark.parametrize("change", ["missing", "tampered", "unexpected"])
def test_retained_files_must_match_manifest(distribution, archives, change):
    write_retained(distribution, archives)
    if change == "missing":
        archives.wheel.unlink()
    elif change == "tampered":
        archives.wheel.write_bytes(b"tampered wheel")
    else:
        (archives.root / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="Retained bytes"):
        distribution.verify_retained(archives.root, archives.commit)


def test_expected_commit_and_fingerprint_are_external(distribution, archives):
    write_retained(distribution, archives)
    with pytest.raises(ValueError, match="supplied clean commit"):
        distribution.verify_retained(archives.root, "b" * 40)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        distribution.verify_retained(archives.root, archives.commit, "b" * 64)


def test_manifest_and_inner_archive_hashes_are_retained_separately(distribution, archives, tmp_path):
    write_retained(distribution, archives)
    result = distribution.pack_retained(Namespace(retained=archives.root, commit=archives.commit,
                                                  source_fingerprint=archives.source, bundle=tmp_path / "artifact.zip"))
    readback = distribution.readback_bundle(Path(result["artifact_zip"]), result["artifact_zip_sha256"], tmp_path / "readback")
    read = distribution.verify_retained(readback, archives.commit, archives.source)
    assert read["archives"][0]["sha256"] == distribution.file_hash(archives.wheel)
    assert result["artifact_zip_sha256"] != read["archives"][0]["sha256"]


def test_wrong_outer_zip_hash_fails_before_extraction(distribution, archives, tmp_path):
    write_retained(distribution, archives)
    bundle = tmp_path / "artifact.zip"
    distribution.pack_retained(Namespace(retained=archives.root, commit=archives.commit,
                                        source_fingerprint=None, bundle=bundle))
    target = tmp_path / "readback"
    with pytest.raises(ValueError, match="artifact ZIP SHA256"):
        distribution.readback_bundle(bundle, "b" * 64, target)
    assert not target.exists()


def test_correct_zip_digest_cannot_hide_tampered_inner_wheel(distribution, archives, tmp_path):
    write_retained(distribution, archives)
    bundle = tmp_path / "artifact.zip"
    contents = {path.relative_to(archives.root).as_posix(): path.read_bytes()
                for path in archives.root.rglob("*") if path.is_file()}
    contents["dist/" + archives.wheel.name] = b"tampered retained wheel"
    save_wheel(bundle, contents)
    readback = distribution.readback_bundle(bundle, distribution.file_hash(bundle), tmp_path / "readback")
    with pytest.raises(ValueError, match="Retained bytes"):
        distribution.verify_retained(readback, archives.commit)


def test_build_refuses_dirty_checkout_before_output(distribution, monkeypatch, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    commit = "a" * 40
    monkeypatch.setattr(distribution.subprocess, "check_output", lambda command, **kwargs:
                        str(root) if "--show-toplevel" in command else commit)
    monkeypatch.setattr(distribution, "source_identity", lambda root: {"source_dirty": True})
    monkeypatch.setattr(distribution, "worktree_status", lambda root: [" M setup.py"])
    output = tmp_path / "retained"
    with pytest.raises(ValueError, match="clean committed worktree"):
        distribution.build_retained(Namespace(root=root, commit=commit, out=output, dependencies=None))
    assert not output.exists()


def test_build_refuses_stale_commit_before_output(distribution, monkeypatch, tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(distribution.subprocess, "check_output", lambda command, **kwargs:
                        str(root) if "--show-toplevel" in command else "b" * 40)
    with pytest.raises(ValueError, match="Build HEAD differs"):
        distribution.build_retained(Namespace(root=root, commit="a" * 40, out=tmp_path / "retained", dependencies=None))


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "dir/../outside.txt", "back\\slash.txt"])
def test_unsafe_downloaded_zip_member_fails(distribution, tmp_path, name):
    bundle = tmp_path / "artifact.zip"
    save_wheel(bundle, {name: b"unsafe member"})
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        distribution.readback_bundle(bundle, distribution.file_hash(bundle), tmp_path / "readback")


def test_sums_missing_duplicate_or_tampered_entries_fail(distribution, archives):
    manifest = write_retained(distribution, archives)
    sums = archives.root / "SHA256SUMS"
    correct = sums.read_text()
    for bad in ("", correct + correct.splitlines()[0] + "\n", correct.replace("  dist/", "  ../dist/")):
        sums.write_text(bad)
        with pytest.raises(ValueError):
            distribution.verify_retained(archives.root, archives.commit)
    refresh_sums(distribution, archives.root, manifest)
    distribution.verify_retained(archives.root, archives.commit)


def test_retained_symlink_rejected(distribution, archives):
    write_retained(distribution, archives)
    (archives.root / "link").symlink_to(archives.wheel)
    with pytest.raises(ValueError, match="symbolic links"):
        distribution.verify_retained(archives.root, archives.commit)


def test_download_does_not_print_or_accept_absent_token(distribution, monkeypatch, tmp_path):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="GH_TOKEN is required"):
        distribution.download_artifact(Namespace(repository="owner/repository", artifact_id="123",
                                               bundle=tmp_path / "download.zip", bundle_sha256="a" * 64))


def add_resource(archives, relative, content):
    source_name = "src/rswm_inria/" + relative
    archives.inputs[source_name] = sha(content)
    for kind in ("wheel", "sdist"):
        contents = getattr(archives, kind + "_contents")
        info_name = "rswm_inria/_build_info.json" if kind == "wheel" else "rswm_inria_120-0.3.0/src/rswm_inria/_build_info.json"
        info = json.loads(contents[info_name])
        info["source_archive_inputs"] = archives.inputs
        contents[info_name] = json.dumps(info).encode()
        name = "rswm_inria/" + relative if kind == "wheel" else "rswm_inria_120-0.3.0/" + source_name
        contents[name] = content
        (save_wheel if kind == "wheel" else save_tar)(getattr(archives, kind), contents)


def test_packaged_frozen_json_is_mapped_and_required(distribution, archives):
    relative = "configs/frozen/manifest.json"
    add_resource(archives, relative, b'{"frozen_fixture":true}\n')
    mapped = distribution.inspect_distributions(archives.dist, archives.commit, archives.source)
    assert all(any(row["source_input"] == "src/rswm_inria/" + relative
                   for row in archive["packaged_inputs"]) for archive in mapped)
    contents = dict(archives.wheel_contents)
    del contents["rswm_inria/" + relative]
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="Required authored inputs missing"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_packaged_frozen_json_tamper_is_rejected(distribution, archives):
    relative = "configs/frozen/manifest.json"
    add_resource(archives, relative, b'{"frozen_fixture":true}\n')
    contents = {**archives.wheel_contents, "rswm_inria/" + relative: b'{"changed":true}\n'}
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="Actual packaged source input differs"):
        distribution.inspect_distributions(archives.dist, archives.commit, archives.source)


def test_dirty_candidate_requires_explicit_mode_and_never_claims_clean(distribution, archives):
    contents = dict(archives.wheel_contents)
    info = json.loads(contents["rswm_inria/_build_info.json"])
    info["source_dirty"] = True
    contents["rswm_inria/_build_info.json"] = json.dumps(info).encode()
    save_wheel(archives.wheel, contents)
    with pytest.raises(ValueError, match="clean commit"):
        distribution.scan_archive(archives.wheel, archives.commit, archives.source)
    candidate = distribution.scan_archive(archives.wheel, archives.commit, archives.source, allow_dirty_candidate=True)
    assert candidate["status"] == "VERIFIED_DIRTY_ENGINEERING_CANDIDATE"
    assert candidate["build_info"]["source_dirty"] is True
    with pytest.raises(ValueError, match="fingerprint"):
        distribution.scan_archive(archives.wheel, archives.commit, "b" * 64, allow_dirty_candidate=True)


def test_cli_missing_distributions_has_explicit_nonzero_exit(distribution, tmp_path, capsys):
    code = distribution.main(["verify", "--retained", str(tmp_path), "--commit", "a" * 40])
    assert code == 1 and json.loads(capsys.readouterr().out)["status"] == "FAIL"
