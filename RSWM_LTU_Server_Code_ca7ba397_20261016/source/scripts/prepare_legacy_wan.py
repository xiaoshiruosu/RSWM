#!/usr/bin/env python3
"""Deploy one verified legacy Wan tree and minimal, reviewable integration patches.

No weights or old evaluation data protocol are copied. The original inputs remain
read-only; an explicit fresh output directory is required. Backend approval and
weight use remain the existing registry/owner decisions.
"""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile
from rswm_inria.geometry import canonical, digest, file_hash
from rswm_inria.replay import load_json

SOURCE_ROOT = "rswm_server_pack_synced_20260817"
SOURCE_ZIP_SHA256 = "2d079e768bdf66a06cc7cb8891f408c7adbcb1985424b801fa962da97400ce73"
SELECTED = "lora_remotesense/new/"
EXCLUDED_SOURCE_SUBTREE = SELECTED + "wan/remotesensebenchmark/"
SHIM = ('from pathlib import Path\nfrom rswm_inria.legacy_wan import load_backend as _load\n\n'
        'def load_backend(*, weights_dir, device):\n'
        '    return _load(source_dir=Path(__file__).resolve().parent, weights_dir=weights_dir, device=device)\n')


def modified_notice(scope):
    return ('# RSWM integration modifications: ' + scope + '.\n'
            '# Original source ZIP SHA256: ' + SOURCE_ZIP_SHA256 + '\n'
            '# Changes are recorded in source-provenance.json and prepared-source.patch.\n')


def _safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name.rstrip("/"):
        raise ValueError("Unsafe legacy source path")
    return path.as_posix()


def _not_symlink(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Legacy source/output symlinks are not accepted")
    return path


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _source_files(source_dir=None, source_zip=None, source_inventory=None, expected_source_sha256=SOURCE_ZIP_SHA256):
    if (source_dir is None) == (source_zip is None):
        raise ValueError("Choose exactly one source directory or source ZIP")
    if source_zip is not None:
        path = _not_symlink(source_zip)
        if not _sha(expected_source_sha256) or file_hash(path) != expected_source_sha256:
            raise ValueError("Original legacy source ZIP SHA256 mismatch")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or archive.testzip() is not None:
                raise ValueError("Legacy ZIP duplicate inventory or failed CRC")
            files = {}
            for item in archive.infolist():
                name = _safe_relative(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Legacy ZIP symlink")
                if item.is_dir() and name == SOURCE_ROOT:
                    continue
                if not name.startswith(SOURCE_ROOT + "/"):
                    raise ValueError("Legacy ZIP root mismatch")
                if not item.is_dir():
                    files[name.removeprefix(SOURCE_ROOT + "/")] = archive.read(item)
        if file_hash(path) != expected_source_sha256:
            raise ValueError("Legacy ZIP changed during source readback")
        source = {"kind": "verified_zip", "path": str(path.resolve()), "sha256": file_hash(path),
                  "bytes": path.stat().st_size, "crc": "PASS", "files": len(files)}
    else:
        root = _not_symlink(source_dir).resolve()
        if source_inventory is None:
            raise ValueError("Directory input requires the retained source inventory")
        index_path = _not_symlink(source_inventory)
        index = load_json(index_path.read_text())
        if not isinstance(index, list) or not index:
            raise ValueError("Legacy inventory must be a nonempty file list")
        files = {}
        for item in index:
            if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"} or not _sha(item["sha256"]) or type(item["bytes"]) is not int or item["bytes"] < 0:
                raise ValueError("Legacy inventory row mismatch")
            name = _safe_relative(item["path"])
            if not name.startswith(SOURCE_ROOT + "/"):
                raise ValueError("Legacy directory inventory root mismatch")
            relative = name.removeprefix(SOURCE_ROOT + "/")
            if relative in files:
                raise ValueError("Duplicate legacy inventory path")
            path = _not_symlink(root / relative)
            data = path.read_bytes()
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Legacy directory file SHA256/size mismatch: " + relative)
            files[relative] = data
        source = {"kind": "inventory_verified_directory", "path": str(root),
                  "inventory_path": str(index_path.resolve()), "inventory_sha256": file_hash(index_path),
                  "files": len(files), "scope": "Actual directory bytes match caller-supplied retained inventory"}
    return files, source


def _function(source, name):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise ValueError("Expected exactly one legacy function " + name)
    fn = functions[0]
    text = ast.get_source_segment(source, fn) + "\n"
    return fn, text


def _once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unsupported legacy source patch anchor")
    return text.replace(old, new, 1)


def patch_sampler(source):
    """Preserve the original sampler body and old path; add one mutually exclusive input."""
    fn, old = _function(source, "generate_video")
    sampler = _once(old, "    inactive_model_cpu: bool = True,\n):",
                    "    inactive_model_cpu: bool = True,\n    prepared_camera=None,\n    block_cpu_offload: bool = False,\n):")
    sampler = _once(sampler, "            if inactive_model_cpu and tag != current_model_tag:",
                    "            if inactive_model_cpu and not block_cpu_offload and tag != current_model_tag:")
    sampler = _once(sampler,
                    "            pred_cond = model(latent_input, t=t_batch, **arg_c)[0]",
                    "            pred_cond = rswm_expert_forward(model, latent_input, t=t_batch, **arg_c)[0] if block_cpu_offload else model(latent_input, t=t_batch, **arg_c)[0]")
    sampler = _once(sampler,
                    "            pred_uncond = model(latent_input, t=t_batch, **arg_null)[0]",
                    "            pred_uncond = rswm_expert_forward(model, latent_input, t=t_batch, **arg_null)[0] if block_cpu_offload else model(latent_input, t=t_batch, **arg_null)[0]")
    sampler = _once(sampler, "    boundary = config.boundary * config.num_train_timesteps",
                    '    if prepared_camera is not None and action_path is not None:\n'
                    '        raise ValueError("prepared_camera and action_path are mutually exclusive")\n\n'
                    '    boundary = config.boundary * config.num_train_timesteps')
    # The original square-root/floor expression yields 479.99999999999994
    # for the exact native grid on current NumPy, silently shrinking height to
    # 464. Prepared camera is a fixed-grid contract, not a resize instruction.
    start_grid = sampler.index("    lat_h = round(\n")
    end_grid = sampler.index("    lat_f = (F_total - 1) // vae_stride[0] + 1\n", start_grid)
    old_grid = sampler[start_grid:end_grid]
    new_grid = ('    if prepared_camera is not None:\n'
                '        if size != "480*832" or (h, w) != (480, 832):\n'
                '            raise ValueError("Prepared sampler requires exact native480x832 input and size")\n'
                '        lat_h, lat_w = 60, 104\n'
                '        h, w = 480, 832\n'
                '    else:\n' + ''.join('    '+line if line.strip() else line for line in old_grid.splitlines(keepends=True)))
    sampler = sampler[:start_grid] + new_grid + sampler[end_grid:]
    start = sampler.index("    # 相机条件\n")
    end = sampler.index("    # 图像条件 y\n", start)
    legacy = sampler[start:end]
    block = ('    if prepared_camera is not None:\n'
             '        from rswm_inria.legacy_camera import camera_condition\n'
             '        dit_cond_dict = camera_condition(prepared_camera, F_total, config, h, w, lat_h, lat_w, device)\n'
             '    else:\n' + ''.join('    '+line if line.strip() else line for line in legacy.splitlines(keepends=True)))
    sampler = sampler[:start] + block + sampler[end:]
    offset = source.index(old)
    source = source[:offset] + sampler + source[offset+len(old):]
    ast.parse(source)
    return source


def patch_inference(source):
    source = patch_sampler(source)
    # Exact legacy loading implementation, now fail closed on incomplete model keys.
    source = source.replace("WanModel.from_pretrained(", "strict_from_pretrained(WanModel,")
    if source.count("strict_from_pretrained(WanModel,") != 2:
        raise ValueError("Expected both original low/high model loader calls")
    source = _once(source, "from wan.modules.model import WanModel\n",
                   "from wan.modules.model import WanModel\nfrom rswm_inria.legacy_wan import strict_from_pretrained, strict_lora_from_pretrained\nfrom rswm_inria.legacy_offload import rswm_expert_forward\n")
    # Preserve inference's merged default. Training explicitly retains the actual
    # restored adapters rather than constructing replacements after merging.
    _, old_loader = _function(source, "load_lora_model")
    loader = _once(old_loader, "    inactive_model_cpu: bool = True,\n):",
                   "    inactive_model_cpu: bool = True,\n    merge_lora: bool = True,\n):")
    for expert in ("low", "high"):
        name = expert + "_noise_model"
        loader = _once(loader,
            f"    {name} = PeftModel.from_pretrained({name}, str({expert}_lora))\n"
            f"    {name} = {name}.merge_and_unload()",
            f"    {name} = strict_lora_from_pretrained(PeftModel, {name}, str({expert}_lora), is_trainable=not merge_lora)\n"
            f"    if merge_lora:\n        {name} = {name}.merge_and_unload()")
    source = _once(source, old_loader, loader)
    source = modified_notice("prepared-only camera/grid and strict local base/LoRA loading") + source
    ast.parse(source)
    return source


def base_loader(source):
    _, text = _function(source, "load_base_model")
    if text.count("WanModel.from_pretrained(") != 2:
        raise ValueError("Expected both original base expert loads")
    text = text.replace("WanModel.from_pretrained(", "strict_from_pretrained(WanModel,")
    result = (modified_notice("original base function extracted; strict local expert loading") +
              'from __future__ import annotations\nimport os\nimport logging\nimport torch\n'
              'from rswm_inria.legacy_wan import strict_from_pretrained\n'
              'logger = logging.getLogger(__name__)\n\n' + text)
    ast.parse(result)
    return result


def patch_tokenizer(source):
    return modified_notice("local-only tokenizer loading") + _once(source, "        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)",
                 '        if kwargs.get("local_files_only", True) is not True:\n'
                 '            raise ValueError("Selected backend forbids tokenizer network fallback")\n'
                 '        kwargs["local_files_only"] = True\n'
                 '        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)')


def prepare(*, out, source_dir=None, source_zip=None, source_inventory=None,
            expected_source_sha256=SOURCE_ZIP_SHA256, backend_config=None):
    root = _not_symlink(out).absolute()
    if root.exists():
        raise FileExistsError("Explicit deployment output must be a fresh directory")
    files, source = _source_files(source_dir, source_zip, source_inventory, expected_source_sha256)
    needed = (SELECTED+"inference.py", SELECTED+"evaluate_baseline.py", SELECTED+"backbone.py", "LICENSE", "NOTICE")
    if any(name not in files for name in needed) or not any(n.startswith(SELECTED+"wan/") for n in files):
        raise ValueError("Selected actual Wan source or original licensing files are missing")
    deployment = {name.removeprefix(SELECTED): data for name, data in files.items()
                  if name.startswith(SELECTED+"wan/") and not name.startswith(EXCLUDED_SOURCE_SUBTREE)}
    deployment.update({"inference.py": files[SELECTED+"inference.py"], "backbone.py": files[SELECTED+"backbone.py"],
                       "LICENSE": files["LICENSE"], "NOTICE": files["NOTICE"]})
    deployment["inference.py"] = patch_inference(deployment["inference.py"].decode()).encode()
    deployment["base_loader.py"] = base_loader(files[SELECTED+"evaluate_baseline.py"].decode()).encode()
    tokenizer = "wan/modules/tokenizers.py"
    deployment[tokenizer] = patch_tokenizer(deployment[tokenizer].decode()).encode()
    raw_config = None
    if backend_config is not None:
        from rswm_inria.legacy_wan import validate_backend_config
        config_path = _not_symlink(backend_config)
        raw_config = config_path.read_bytes()
        validate_backend_config(load_json(raw_config.decode()))
    mapping = {}
    for target, data in sorted(deployment.items()):
        original = (SELECTED+"evaluate_baseline.py" if target=="base_loader.py" else
                    target if target in ("LICENSE", "NOTICE") else SELECTED+target)
        mapping[target] = {"original_path": SOURCE_ROOT+"/"+original,
                           "original_file_sha256": hashlib.sha256(files[original]).hexdigest(),
                           "deployed_sha256": hashlib.sha256(data).hexdigest(),
                           "deployed_bytes": len(data), "modified": data != files[original]}
    patches = []
    for target in ("inference.py", tokenizer):
        patches.extend(difflib.unified_diff(files[SELECTED+target].decode().splitlines(keepends=True),
            deployment[target].decode().splitlines(keepends=True), fromfile="original/"+target, tofile="prepared/"+target))
    _, original_base = _function(files[SELECTED+"evaluate_baseline.py"].decode(), "load_base_model")
    patches.extend(difflib.unified_diff(original_base.splitlines(keepends=True),
        deployment["base_loader.py"].decode().splitlines(keepends=True),
        fromfile="original/evaluate_baseline.py#load_base_model", tofile="prepared/base_loader.py"))
    selected_functions = {}
    for name in ("generate_video", "load_lora_model"):
        _, original_text = _function(files[SELECTED+"inference.py"].decode(), name)
        _, deployed_text = _function(deployment["inference.py"].decode(), name)
        selected_functions[name] = {"original_text_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
                                    "deployed_text_sha256": hashlib.sha256(deployed_text.encode()).hexdigest()}
    provenance = {"source": source, "selected_vendor_tree": SOURCE_ROOT+"/"+SELECTED+"wan",
                  "selection_reason": "The actual adjacent inference.py production tree; top-level Wan is excluded",
                  "excluded_source_subtrees": [{"path": SOURCE_ROOT+"/"+EXCLUDED_SOURCE_SUBTREE,
                     "file_count": sum(name.startswith(EXCLUDED_SOURCE_SUBTREE) for name in files),
                     "reason": "Unreferenced legacy benchmark/tooling with provider credential configuration; no selected sampler/model dependency. Original inputs remain unchanged; excluded configuration bytes are not deployed or recorded."}],
                  "files": mapping, "base_function_original_text_sha256": hashlib.sha256(original_base.encode()).hexdigest(),
                  "selected_functions": selected_functions,
                  "patch_scope": ["mutually exclusive prepared camera; old action/env path retained",
                                  "prepared-only exact480x832 grid avoids original sqrt/floor464 height; old resize formula retained",
                                  "original base function isolated from old evaluation/HDF5 protocol",
                                  "strict actual low/high base and retained LoRA tensor loads; local-only tokenizer",
                                  "explicit unmerged restored LoRA training; original inference merge default retained",
                                  "explicit CPU block offload bypasses whole-expert GPU swap; original default retained"],
                  "real_model_validation": "NOT_RUN", "license_approval_inferred": False, "weights_copied": False}
    deployment["prepared-source.patch"] = ''.join(patches).encode()
    deployment["source-provenance.json"] = canonical(provenance)+b"\n"
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(deployment.items())}
    deployment["source_inventory.json"] = canonical(inventory)+b"\n"
    deployment["backend.py"] = SHIM.encode()
    if raw_config is not None:
        deployment["backend_config.json"] = raw_config
    root.mkdir(parents=True)
    for name, data in sorted(deployment.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    if any(file_hash(root/name) != hashlib.sha256(data).hexdigest() for name,data in deployment.items()):
        raise ValueError("Deployed source readback mismatch")
    return {"status": "PREPARED_SOURCE_ONLY", "out": str(root), "source": source,
            "files": len(deployment), "source_inventory_sha256": digest(inventory),
            "backend_config_supplied": raw_config is not None, "real_model_validation": "NOT_RUN"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir")
    source.add_argument("--source-zip")
    parser.add_argument("--source-inventory", help="Required full retained inventory for directory input")
    parser.add_argument("--expected-source-sha256", default=SOURCE_ZIP_SHA256)
    parser.add_argument("--backend-config", help="Explicit caller sampling/mode JSON, copied exactly")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(out=args.out, source_dir=args.source_dir, source_zip=args.source_zip,
                         source_inventory=args.source_inventory, expected_source_sha256=args.expected_source_sha256,
                         backend_config=args.backend_config)
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "REJECTED", "error": type(exc).__name__, "detail": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
