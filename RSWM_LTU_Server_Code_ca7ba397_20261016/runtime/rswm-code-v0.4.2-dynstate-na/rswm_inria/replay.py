"""Recompute RGB scores from the frozen task, initial image, log and PNGs.

A correct hash chain verifies internal integrity, not trusted timestamping,
network isolation, absence of hidden input, or a model's internal branch state.
Pass an independently retained expected_head to detect whole-log replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import canonical, rgb, digest
from .tasks import validate_task
from .score_rgb import Ledger, branch_report
from .runner import verify_journal
from .data import write_json

FAILURES = {"missing", "invalid_image", "invalid_batch", "model_exception"}
AUXILIARY = {"session", "initialize", "snapshot", "restore", "adapter_failure"}


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def load_json(text):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON number: "+value)
    return json.loads(text, object_pairs_hook=_strict_object, parse_constant=invalid_constant)


def load_rgb_png(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Required PNG is missing or is a symlink.")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("Replay requires lossless RGB PNGs; no automatic conversions.")
        return rgb(np.asarray(image)).copy()


def schedule(task):
    """Yield the exact permitted call/output trace, including prefix only once."""
    family, programs = task["family"], task["programs"]
    if family == "F7":
        streams = [("prefix", programs["a"], 1,33,16),
                   ("a",programs["a"],33,97,16),("b",programs["b"],33,97,16)]
    else:
        names = ("small","large") if family == "F3" else ("main",)
        streams = [(name,programs[name],1,len(programs[name]),16 if family=="F5" else 32) for name in names]
    for stream, queries, start, stop, size in streams:
        for lo in range(start,stop,size):
            hi = min(stop,lo+size)
            yield {"kind":"request","stream":stream,"start":lo,"stop":hi,"queries":queries[lo:hi]}
            yield {"kind":"response","stream":stream,"start":lo,"stop":hi}
            for event in range(lo,hi):
                yield {"kind":"output","stream":stream,"event":event,"q":queries[event]}


def replay_run(task, initial, run_dir, expected_head=None, check_summary=True):
    labels = validate_task(task)
    initial = rgb(initial).copy()
    root = Path(run_dir).resolve()
    log = root/"events.jsonl"
    if not log.is_file() or log.is_symlink():
        raise ValueError("Missing or symlinked journal.")
    rows = []
    with log.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank journal row.")
            rows.append(load_json(line))
            if len(rows) > 20000:
                raise ValueError("Journal exceeds the bounded task format.")
    if not rows or not all(isinstance(r,dict) for r in rows):
        raise ValueError("Empty or malformed journal.")
    try:
        head = verify_journal(rows)
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed journal chain.") from exc
    if expected_head is not None:
        if not isinstance(expected_head,str) or not re.fullmatch(r"[0-9a-f]{64}",expected_head) or head != expected_head:
            raise ValueError("Journal does not match the independently retained head.")
    header = rows[0]
    if header.get("kind") != "session" or sum(r.get("kind")=="session" for r in rows) != 1:
        raise ValueError("Exactly one initial session row is required.")
    required = {"journal_schema":"rswm-journal-v1", "task_id":task["task_id"],
                "task_sha256":digest(task), "protocol":task["protocol"],
                "initial_sha256":hashlib.sha256(initial.tobytes()).hexdigest()}
    if any(header.get(key)!=value for key,value in required.items()) or type(header.get("seed")) is not int:
        raise ValueError("Session does not match the frozen task, image or journal schema.")
    expected = list(schedule(task))
    actual = []
    for row in rows:
        if row.get("kind") in {"request","response","output"}:
            actual.append(row)
        elif row.get("kind") not in AUXILIARY:
            raise ValueError("Unknown journal event kind.")
    if len(actual) != len(expected):
        raise ValueError("Missing or extra planned calls/outputs.")
    for row, want in zip(actual, expected):
        if any(canonical(row.get(key)) != canonical(value) for key,value in want.items()):
            raise ValueError("Journal call/coordinate/event order differs from the planned protocol.")

    ledgers = {}
    prefix = None
    logged_files = set()
    valid_images = 0
    for row in actual:
        if row["kind"] != "output":
            continue
        stream, event, status = row["stream"], row["event"], row.get("status")
        path = root/stream/f"{event:04d}.png"
        if path.parent.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Output path escapes the run directory or follows a symlink.")
        if status == "valid":
            image = load_rgb_png(path)
            if hashlib.sha256(image.tobytes()).hexdigest() != row.get("pixel_sha256"):
                raise ValueError("Submitted PNG pixels disagree with the journal.")
            logged_files.add(path.relative_to(root).as_posix()); valid_images += 1
        elif status in FAILURES:
            image = None
            if row.get("pixel_sha256") is not None or path.exists():
                raise ValueError("Failed output has a pixel hash or a later replacement file.")
        else:
            raise ValueError("Unrecognized submission status.")
        if stream not in ledgers:
            if stream in ("a","b"):
                if prefix is None or prefix.event != 32:
                    raise ValueError("F7 branch lacks its complete planned prefix.")
                ledgers[stream] = prefix.fork(labels[stream])
            else:
                ledgers[stream] = Ledger(initial,task["origin"],labels.get(stream))
                if stream == "prefix":
                    prefix = ledgers[stream]
        ledgers[stream].submit(event,row["q"],image)
    # Reject unlogged PNGs, including hidden post-hoc replacement submissions.
    present = {p.relative_to(root).as_posix() for p in root.rglob("*.png")}
    if present - logged_files - {"initial.png"}:
        raise ValueError("Run directory contains unlogged PNG outputs.")
    reports = (branch_report(prefix,ledgers["a"],ledgers["b"]) if task["family"]=="F7"
               else {name:ledger.report() for name,ledger in ledgers.items()})
    summary_matches = None
    summary = root/"summary.json"
    if check_summary and summary.exists():
        original = load_json(summary.read_text(encoding="utf-8"))
        if original.get("journal_head") != head or canonical(original.get("reports")) != canonical(reports):
            raise ValueError("Saved summary disagrees with independent replay.")
        summary_matches = True
    return {"task_id":task["task_id"],"family":task["family"],"seed":header["seed"],
            "task_sha256":digest(task),"journal_head":head,"reports":reports,
            "verified_png_outputs":valid_images,"planned_output_events":sum(r["kind"]=="output" for r in actual),
            "artifact_checks":{"trace_matches_plan":True,"pixel_hashes_verified":True,
                               "summary_matches":summary_matches,"externally_anchored_head":expected_head is not None},
            "protocol_security_verified":False,"real_model_validation":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--task",type=Path)
    parser.add_argument("--initial",type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    task_path = args.task or args.run_dir/"task.json"
    initial_path = args.initial or args.run_dir/"initial.png"
    task = load_json(task_path.read_text(encoding="utf-8"))
    result = replay_run(task,load_rgb_png(initial_path),args.run_dir,args.expected_head)
    write_json(args.out,result)
    print(json.dumps({"task_id":result["task_id"],"journal_head":result["journal_head"],
                      "verified_png_outputs":result["verified_png_outputs"]},sort_keys=True))


if __name__ == "__main__":
    main()
