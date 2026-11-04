"""Build identity must include every source archive input, not just package code."""
from pathlib import Path
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
import setuptools


@pytest.fixture
def provenance(monkeypatch):
    # Load the current build implementation without invoking setup's CLI.
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    return runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))["provenance"]


@pytest.fixture
def committed_tree(tmp_path):
    files = {
        "src/rswm_inria/__init__.py": "__version__ = 'fixture'\n",
        "README.md": "build fixture\n",
        "NOTICE.md": "build legal notice fixture\n",
        "docs/Card.md": "build fixture\n",
        "tests/test_contract.py": "# build fixture\n",
        "scripts/check_fixture.py": "print('archive fixture helper')\n",
        "MANIFEST.in": "recursive-include docs *.md\n",
        "pyproject.toml": "# build fixture\n",
        "setup.py": "# build fixture\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    git("add", ".")
    git("-c", "user.name=Build fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "Commit fixture source inputs")
    return tmp_path, git("rev-parse", "HEAD")


def test_clean_committed_archive_has_actual_identity(provenance, committed_tree):
    root, head = committed_tree
    result = provenance(root)
    assert result["source_commit"] == head and result["source_dirty"] is False


def test_frozen_json_resource_is_an_authored_build_input(provenance, committed_tree):
    root, _ = committed_tree
    resource = root / "src/rswm_inria/configs/frozen/manifest.json"
    resource.parent.mkdir(parents=True)
    resource.write_text('{"version":"fixture"}\n')
    result = provenance(root)
    assert "src/rswm_inria/configs/frozen/manifest.json" in result["source_archive_inputs"]
    assert result["source_dirty"] is True


@pytest.mark.parametrize("name", ["README.md", "NOTICE.md", "docs/Card.md", "tests/test_contract.py", "scripts/check_fixture.py", "MANIFEST.in"])
def test_changed_archive_input_is_not_a_clean_build(provenance, committed_tree, name):
    root, head = committed_tree
    before = provenance(root)
    (root / name).write_text("changed archive input\n")
    result = provenance(root)
    assert result["source_commit"] == head
    assert result["source_code_sha256"] == before["source_code_sha256"]
    assert result["source_dirty"] is True


@pytest.mark.parametrize("name", ["docs/NewCard.md", "tests/test_new_contract.py", "scripts/new_helper.py"])
def test_uncommitted_archive_input_is_not_a_clean_build(provenance, committed_tree, name):
    root, _ = committed_tree
    (root / name).write_text("new archive input\n")
    assert provenance(root)["source_dirty"] is True


def test_untracked_run_assets_do_not_dirty_the_archive(provenance, committed_tree):
    root, _ = committed_tree
    for name in ("evidence/receipt.json", "runs/output.png", "data/source.tif"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"work asset fixture only")
    assert provenance(root)["source_dirty"] is False


@pytest.fixture
def extracted_tree(provenance, committed_tree):
    root, head = committed_tree
    extracted = root / "extracted-source"
    # Keep the source archive in an unrelated parent checkout to exercise the
    # no-Git path even when git can discover a repository above the extraction.
    shutil.copytree(root, extracted, ignore=shutil.ignore_patterns(".git", "extracted-source"))
    original = provenance(root)
    (extracted / "src/rswm_inria/_build_info.json").write_text(json.dumps(original))
    return extracted, head, original


def test_unchanged_extracted_archive_keeps_verified_clean_identity(provenance, extracted_tree):
    root, head, original = extracted_tree
    result = provenance(root)
    assert result["source_commit"] == head
    assert result["source_dirty"] is False
    assert result["source_archive_inputs"] == original["source_archive_inputs"]


@pytest.mark.parametrize("name", ["src/rswm_inria/__init__.py", "README.md", "NOTICE.md", "docs/Card.md",
                                  "tests/test_contract.py", "scripts/check_fixture.py", "MANIFEST.in", "pyproject.toml", "setup.py"])
def test_changed_extracted_archive_input_is_dirty(provenance, extracted_tree, name):
    root, head, _ = extracted_tree
    (root / name).write_text("changed after source extraction\n")
    result = provenance(root)
    assert result["source_commit"] == head and result["source_dirty"] is True


@pytest.mark.parametrize("name", ["src/rswm_inria/new.py", "tests/test_added.py", "scripts/added_helper.py", "docs/NewCard.md"])
def test_added_extracted_archive_input_is_dirty(provenance, extracted_tree, name):
    root, _, _ = extracted_tree
    (root / name).write_text("new archive input\n")
    assert provenance(root)["source_dirty"] is True


def test_deleted_extracted_archive_input_is_dirty(provenance, extracted_tree):
    root, _, _ = extracted_tree
    (root / "docs/Card.md").unlink()
    assert provenance(root)["source_dirty"] is True


@pytest.mark.parametrize("prior_dirty", [False, None, True])
def test_legacy_archive_without_input_manifest_never_gains_clean_identity(provenance, extracted_tree, prior_dirty):
    root, _, original = extracted_tree
    original.pop("source_archive_inputs")
    original["source_dirty"] = prior_dirty
    (root / "src/rswm_inria/_build_info.json").write_text(json.dumps(original))
    rebuilt = provenance(root)
    expected = True if prior_dirty is True else None
    assert rebuilt["source_dirty"] is expected
    (root / "src/rswm_inria/_build_info.json").write_text(json.dumps(rebuilt))
    assert provenance(root)["source_dirty"] is expected
    (root / "src/rswm_inria/__init__.py").write_text("changed legacy package\n")
    assert provenance(root)["source_dirty"] is True


def test_rebuilt_dirty_archive_cannot_launder_changes(provenance, extracted_tree):
    root, _, _ = extracted_tree
    (root / "docs/Card.md").write_text("changed archive input\n")
    dirty = provenance(root)
    (root / "src/rswm_inria/_build_info.json").write_text(json.dumps(dirty))
    assert provenance(root)["source_dirty"] is True


def test_actual_sdist_to_wheel_preserves_or_marks_source_identity(tmp_path):
    """Exercise both setuptools commands and receipts inside their real archives."""
    project = tmp_path / "source"
    package = project / "src/rswm_inria"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'fixture'\n")
    for name, content in {"README.md": "fixture\n", "docs/Card.md": "fixture\n",
                          "tests/test_contract.py": "# fixture\n",
                          "scripts/check_fixture.py": "print('archive fixture helper')\n"}.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    repository = Path(__file__).parents[1]
    for name in ("setup.py", "MANIFEST.in"):
        shutil.copyfile(repository / name, project / name)
    (project / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[project]
name = "rswm-provenance-fixture"
version = "0.0.1"
[tool.setuptools.packages.find]
where = ["src"]
''')
    (project / ".gitignore").write_text("*.egg-info/\nbuild/\n")

    def command(root, *args):
        result = subprocess.run(args, cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout.strip()

    command(project, "git", "init")
    command(project, "git", "add", ".")
    command(project, "git", "-c", "user.name=Build fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-m", "Commit fixture source inputs")
    head = command(project, "git", "rev-parse", "HEAD")
    archives = tmp_path / "archives"
    command(project, sys.executable, "setup.py", "sdist", "--dist-dir", str(archives))
    extracted = tmp_path / "extracted"
    with tarfile.open(next(archives.glob("*.tar.gz"))) as archive:
        # These are our own fixture members; use the safe filter where the
        # supported Python patch release supplies it.
        extraction = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(extracted, **extraction)
    root = next(extracted.iterdir())

    def wheel_identity(label):
        output = tmp_path / label
        command(root, sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output))
        with zipfile.ZipFile(next(output.glob("*.whl"))) as wheel:
            identity = json.loads(wheel.read("rswm_inria/_build_info.json"))
            files = {name.removeprefix("rswm_inria/"): hashlib.sha256(wheel.read(name)).hexdigest()
                     for name in sorted(wheel.namelist()) if name.startswith("rswm_inria/") and name.endswith(".py")}
            fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"),
                                                     ensure_ascii=False).encode()).hexdigest()
            assert identity["source_code_sha256"] == fingerprint
            return identity

    clean = wheel_identity("clean-wheel")
    assert clean["source_commit"] == head and clean["source_dirty"] is False
    assert "docs/Card.md" in clean["source_archive_inputs"]
    helper = root / "scripts/check_fixture.py"
    helper_source = "print('archive fixture helper')\n"
    assert helper.read_text() == helper_source
    assert clean["source_archive_inputs"]["scripts/check_fixture.py"] == hashlib.sha256(helper_source.encode()).hexdigest()
    helper.write_text("print('modified helper after extraction')\n")
    changed_script = wheel_identity("changed-script-wheel")
    assert changed_script["source_commit"] == head and changed_script["source_dirty"] is True
    assert changed_script["source_code_sha256"] == clean["source_code_sha256"]
    helper.write_text(helper_source)
    (root / "docs/Card.md").write_text("modified after sdist extraction\n")
    changed_document = wheel_identity("changed-document-wheel")
    assert changed_document["source_commit"] == head and changed_document["source_dirty"] is True
    assert changed_document["source_code_sha256"] == clean["source_code_sha256"]
    (root / "src/rswm_inria/__init__.py").write_text("__version__ = 'changed'\n")
    changed_package = wheel_identity("changed-package-wheel")
    assert changed_package["source_commit"] == head and changed_package["source_dirty"] is True
    assert changed_package["source_code_sha256"] != clean["source_code_sha256"]
