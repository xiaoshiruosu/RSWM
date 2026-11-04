"""Read-only integrity and observed LTU source identity verification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = "ca7ba3977bba0f71d1165526e1caf02ada6d67c555086e3636679ece647161ca"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def fingerprint(folder):
    files = {p.relative_to(folder).as_posix(): sha(p.read_bytes())
             for p in sorted(folder.rglob("*.py"))}
    value = sha(json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    return value, len(files)

errors = []
listed = set()
for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    digest, relative = line.split("  ", 1)
    candidate = (ROOT / relative).resolve()
    if not candidate.is_relative_to(ROOT) or candidate.is_symlink():
        errors.append("Unsafe manifest path: " + relative)
        continue
    listed.add(relative)
    if not candidate.is_file() or sha(candidate.read_bytes()) != digest:
        errors.append("Hash mismatch: " + relative)
actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
          if p.is_file() and p.name != "SHA256SUMS"}
if actual != listed:
    errors.append("Archive file set differs from SHA256SUMS")
packages = {}
for relative in ("runtime/rswm-code-v0.4.2-dynstate-na/rswm_inria", "source/src/rswm_inria"):
    value, count = fingerprint(ROOT / relative)
    packages[relative] = {"fingerprint": value, "python_files": count}
    if value != EXPECTED or count != 54:
        errors.append("Observed server identity mismatch: " + relative)
print(json.dumps({"status": "PASS" if not errors else "FAIL", "files": len(listed),
                  "packages": packages, "errors": errors}, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
