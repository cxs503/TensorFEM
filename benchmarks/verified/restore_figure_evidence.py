#!/usr/bin/env python3
"""Restore the figure artifacts that ``gates_verified.json`` promises but the repo lacks.

Why this exists
---------------
``tools/verified_anchor_audit.py`` recomputes every gate from the numbers inside
the artifact, but it also checks something cheaper and more brutal: **does the
file the certificate names exist?**  A certificate that says

    "artifacts": {"figures": ["benchmarks/verified/<anchor>/foo.png"]}

is a *claim of availability*.  When ``foo.png`` is not in the tree, the gate
numbers may still be right while the evidence trail is broken: a reader who
follows the certificate gets nothing, and ``E_ARTIFACT_MISSING`` is raised.
Fourteen anchors were in exactly that state (36 missing figures), because a
figure is a **build product** -- drivers write it next to themselves at run time
-- and a build product is invisible to ``git status`` unless someone remembers
to ``git add`` it.  Nothing in the pipeline remembered.

What this driver does
---------------------
For every anchor with a declared-but-missing artifact *and* a runner in the
tree, it executes the anchor's own driver (never a private re-implementation,
which would be a second source of truth) and then verifies that the named files
actually appeared and are real PNGs.  It cannot fabricate evidence: if the
driver fails, times out, or does not emit the declared path, the anchor is
recorded as ``runner_failed`` / ``timeout`` / ``figure_missing`` and stays in
the certificate-debt registry.  Honest failure is the whole point.

Reproducibility side channel
----------------------------
Re-running a driver also rewrites ``result.json`` / ``gates_verified.json``.
Those rewrites are compared against ``git show HEAD:<path>`` leaf by leaf: any
changed leaf outside ``generated_at`` / ``wall_seconds`` is reported as
``physics_drift`` (a nondeterministic anchor is a finding, not a nuisance
detail).  With the default ``--keep-json`` the committed JSON is left untouched,
so the merged evidence chain keeps its original numbers while the figures are
restored.

Usage
-----
    python benchmarks/verified/restore_figure_evidence.py --dry-run
    python benchmarks/verified/restore_figure_evidence.py --jobs 3 --timeout 600
    python benchmarks/verified/restore_figure_evidence.py --report-only --write

Exit code is 0 when every *attempted* anchor ended ``restored`` or
``already_complete``; anchors with no runner are reported but do not fail the
run (they need a driver written, which is a different job).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
REPORT_PATH = VERIFIED_DIR / "figure_evidence_restore.json"
REPORT_SCHEMA = "tensorfem.figure-evidence-restore/1"

sys.path.insert(0, str(REPO_ROOT / "tools"))
import verified_anchor_audit as audit  # noqa: E402  (single source of truth for gate semantics)

#: leaves that legitimately change between two runs of a deterministic driver
VOLATILE_LEAVES = (
    "/generated_at",
    "/wall_seconds",
    "/solve_seconds",
    "/seconds",
    "/elapsed_s",
    "/elapsed_total_s",
)

#: two floating-point residuals below this are roundoff, not a changed answer.
#: 1e-9 is the tightest threshold any certificate in the registry declares, so a
#: leaf that moves by less than it cannot by itself flip a verdict.
ROUNDOFF_TOL = 1e-9

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: a figure smaller than this carries no readable content
MIN_PNG_BYTES = 2048


def _flatten(obj, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _flatten(value, f"{prefix}/{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _flatten(value, f"{prefix}[{index}]")
    else:
        yield prefix, obj


def _volatile(leaf: str) -> bool:
    return any(leaf.endswith(tag) for tag in VOLATILE_LEAVES)


def _roundoff(before, after) -> bool:
    """Same answer to within double-precision noise (thread/BLAS ordering)."""
    if isinstance(before, bool) or isinstance(after, bool):
        return False
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return False
    return abs(float(after) - float(before)) <= ROUNDOFF_TOL * max(1.0, abs(float(before)))


def declared_artifacts(payload: dict) -> list[str]:
    """Every path the certificate claims, in declaration order."""
    out: list[str] = []
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        for item in artifacts.values():
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list):
                out.extend(sub for sub in item if isinstance(sub, str))
    return out


def resolve_declared(text: str, anchor_dir: Path) -> Path | None:
    """Anchor-relative or repo-relative path -> absolute path (None if not a path)."""
    if not audit._looks_like_path(text):
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    anchor_relative = anchor_dir / candidate
    if anchor_relative.exists():
        return anchor_relative
    repo_relative = REPO_ROOT / candidate
    if repo_relative.exists():
        return repo_relative
    # neither exists: decide by shape -- a bare filename belongs to the anchor dir
    if len(candidate.parts) == 1:
        return anchor_relative
    return repo_relative


def png_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        blob = path.read_bytes()
    except OSError as exc:  # pragma: no cover - unreadable file
        return False, f"unreadable: {exc}"
    if not blob.startswith(PNG_MAGIC):
        return False, "not a png (bad magic)"
    if len(blob) < MIN_PNG_BYTES:
        return False, f"png too small ({len(blob)} B)"
    return True, f"{len(blob)} B"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_for(anchor_dir: Path) -> Path | None:
    for name in audit.RUNNER_NAMES:
        candidate = anchor_dir / name
        if candidate.exists():
            return candidate
    return None


def git_head_json(rel: str) -> dict | None:
    try:
        raw = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def json_drift(anchor_rel: str) -> dict:
    """Leaf-level comparison of rewritten artifacts against HEAD."""
    report = {"files": [], "physics_drift": [], "roundoff": [], "volatile_changed": 0}
    for name in ("result.json", "gates_verified.json"):
        rel = f"{anchor_rel}/{name}"
        old = git_head_json(rel)
        path = REPO_ROOT / rel
        if old is None or not path.exists():
            continue
        new = json.loads(path.read_text())
        before, after = dict(_flatten(old)), dict(_flatten(new))
        changed = [key for key in set(before) | set(after) if before.get(key) != after.get(key)]
        volatile = [key for key in changed if _volatile(key)]
        drift, roundoff = [], []
        for key in changed:
            if _volatile(key):
                continue
            (roundoff if _roundoff(before.get(key), after.get(key)) else drift).append(key)
        report["files"].append(
            {"path": rel, "changed_leaves": len(changed), "volatile_leaves": len(volatile),
             "roundoff_leaves": len(roundoff)}
        )
        report["volatile_changed"] += len(volatile)
        report["roundoff"].extend({"path": rel, "leaf": key, "head": before.get(key),
                                   "run": after.get(key)} for key in sorted(roundoff))
        report["physics_drift"].extend({"path": rel, "leaf": key, "head": before.get(key),
                                        "run": after.get(key)} for key in sorted(drift))
    return report


def restore_json(anchor_rel: str) -> list[str]:
    """Revert the rewritten certificates so the merged numbers stay the reference."""
    paths = [f"{anchor_rel}/result.json", f"{anchor_rel}/gates_verified.json"]
    present = [p for p in paths if (REPO_ROOT / p).exists()]
    if not present:
        return []
    subprocess.check_call(["git", "checkout", "--", *present], cwd=REPO_ROOT)
    return present


def inspect(anchor_dir: Path, tracked: set[str] | None) -> dict:
    gates_path = anchor_dir / "gates_verified.json"
    anchor_rel = str(anchor_dir.relative_to(REPO_ROOT))
    record = {
        "anchor": anchor_dir.name,
        "anchor_dir": anchor_rel,
        "has_gates": gates_path.exists(),
        "runner": None,
        "declared": [],
        "missing": [],
        "status": "unknown",
    }
    if not gates_path.exists():
        record["status"] = "no_gates"
        return record
    payload = json.loads(gates_path.read_text())
    runner = runner_for(anchor_dir)
    record["runner"] = str(runner.relative_to(REPO_ROOT)) if runner else None
    declared = declared_artifacts(payload)
    paths: list[dict] = []
    for text in declared:
        path = resolve_declared(text, anchor_dir)
        if path is None:
            continue
        ok, note = png_ok(path) if path.suffix.lower() == ".png" else (path.exists(), "")
        entry = {
            "declared": text,
            "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            "present": ok,
            "note": note,
        }
        paths.append(entry)
        if not ok:
            record["missing"].append(entry["path"])
    record["declared"] = paths
    if not record["missing"]:
        record["status"] = "already_complete"
    elif runner is None:
        record["status"] = "no_runner"
    return record


def _run_one(anchor_dir: Path, runner: Path, timeout: int, keep_json: bool) -> dict:
    env = dict(os.environ)
    # drivers import the shared harness from benchmarks/verified and the library from src/
    base = f"src:{VERIFIED_DIR.relative_to(REPO_ROOT)}"
    env["PYTHONPATH"] = f"{base}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else base
    env.setdefault("OMP_NUM_THREADS", "4")
    env["MPLBACKEND"] = "Agg"
    started = time.time()
    outcome = {"exit_code": None, "timed_out": False, "stderr_tail": ""}
    try:
        proc = subprocess.run(
            [sys.executable, str(runner.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        outcome["exit_code"] = proc.returncode
        outcome["stderr_tail"] = proc.stderr.strip().splitlines()[-3:]
    except subprocess.TimeoutExpired:
        outcome["timed_out"] = True
    outcome["wall_seconds"] = round(time.time() - started, 3)
    anchor_rel = str(anchor_dir.relative_to(REPO_ROOT))
    outcome["json"] = json_drift(anchor_rel)
    if keep_json:
        outcome["reverted_json"] = restore_json(anchor_rel)
    return outcome


def restore_one(anchor_dir: Path, timeout: int, keep_json: bool, force: bool = False) -> dict:
    record = inspect(anchor_dir, audit.tracked_files())
    if record["status"] != "unknown" and not (force and record["runner"]):
        return record
    record["forced"] = bool(force)
    runner = REPO_ROOT / record["runner"]
    record["attempted"] = True
    run = _run_one(anchor_dir, runner, timeout, keep_json)
    record["run"] = run
    # re-inspect the files the certificate asks for
    paths = []
    missing: list[str] = []
    for text in declared_artifacts(json.loads((anchor_dir / "gates_verified.json").read_text())):
        path = resolve_declared(text, anchor_dir)
        if path is None:
            continue
        ok, note = png_ok(path) if path.suffix.lower() == ".png" else (path.exists(), "")
        rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        entry = {"path": rel, "present": ok, "note": note}
        if ok and not path.is_dir():
            entry["sha256"] = sha256(path)
        else:
            missing.append(rel)
        paths.append(entry)
    record["declared_after"] = paths
    record["missing_after"] = missing
    record["missing_before"] = record["missing"]
    if run["timed_out"]:
        record["status"] = "timeout"
    elif run["exit_code"] != 0:
        record["status"] = "runner_failed"
    elif missing:
        record["status"] = "figure_missing"
    elif run["json"]["physics_drift"]:
        record["status"] = "restored_with_drift"
    elif record["forced"] and not record["missing_before"]:
        record["status"] = "reproduced"
    else:
        record["status"] = "restored"
    return record


def _previous_entries() -> dict[str, dict]:
    """Carry the run history (what was executed, how long, what drifted) forward."""
    if not REPORT_PATH.exists():
        return {}
    try:
        payload = json.loads(REPORT_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if payload.get("schema") != REPORT_SCHEMA:
        return {}
    return {entry["anchor"]: entry for entry in payload.get("entries", [])}


def report_only() -> dict:
    tracked = audit.tracked_files()
    history = _previous_entries()
    entries = []
    for anchor_dir in sorted(audit.discover()):
        record = inspect(anchor_dir, tracked)
        if record["status"] == "unknown":
            record["status"] = "not_attempted"
        prior = history.get(record["anchor"], {})
        for key in ("runner", "run", "forced", "missing_before"):
            if key in prior and key not in record:
                record[key] = prior[key]
        if prior.get("missing_before") and not record.get("missing_after"):
            record["missing_after"] = []
            record["status"] = ("restored_with_drift"
                                if prior.get("run", {}).get("json", {}).get("physics_drift")
                                else "restored")
        missing_now: list[str] = []
        for entry in record.get("declared", []):
            ok, note = (png_ok(REPO_ROOT / entry["path"]) if entry["path"].endswith(".png")
                        else ((REPO_ROOT / entry["path"]).exists(), ""))
            entry["present"] = ok
            entry["note"] = note
            if ok and not (REPO_ROOT / entry["path"]).is_dir():
                entry["sha256"] = sha256(REPO_ROOT / entry["path"])
            if not ok:
                missing_now.append(entry["path"])
            entry["git_tracked"] = tracked is not None and entry["path"] in tracked
        record["missing_after"] = missing_now
        entries.append(record)
    return entries


def summarise(entries: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    missing_after = sum(
        len(e.get("missing_after") if e.get("missing_after") is not None else e.get("missing") or [])
        for e in entries
    )
    drift = [
        f"{e['anchor']}:{d['leaf']}"
        for e in entries
        for d in e.get("run", {}).get("json", {}).get("physics_drift", [])
    ]
    restored = [e for e in entries if e["status"] in ("restored", "restored_with_drift")]
    return {
        "anchors": len(entries),
        "status_counts": counts,
        "figures_restored": sum(len(e.get("missing_before", [])) for e in restored),
        "missing_after": missing_after,
        "physics_drift": drift,
    }


def build_payload(entries: list[dict]) -> dict:
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": audit.__dict__.get("utc_now", lambda: "")()
        if hasattr(audit, "utc_now")
        else time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": (
            "Every entry is the anchor's own driver re-executed, never a private "
            "re-implementation; status=figure_missing/timeout/runner_failed means the "
            "evidence is still absent and the anchor stays in "
            "anchor_certificate_debt.json."
        ),
        "summary": summarise(entries),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", nargs="*", help="restrict to these anchor directory names")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent drivers (default 1)")
    parser.add_argument("--timeout", type=int, default=900, help="per-driver timeout [s]")
    parser.add_argument("--keep-json", action="store_true", default=True,
                        help="revert rewritten result.json/gates (default: on)")
    parser.add_argument("--no-keep-json", dest="keep_json", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="re-run drivers even when every declared figure is present "
                             "(reproducibility re-check; pairs with --no-keep-json)")
    parser.add_argument("--report-only", action="store_true",
                        help="no execution: re-derive presence/tracking of declared artifacts")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--write", action="store_true", help=f"write {REPORT_PATH.name}")
    args = parser.parse_args(argv)

    if args.report_only:
        entries = report_only()
    else:
        dirs = sorted(audit.discover())
        if args.anchors:
            wanted = set(args.anchors)
            dirs = [d for d in dirs if d.name in wanted]
        plan = [
            d for d in dirs
            if args.force or inspect(d, audit.tracked_files())["status"] == "unknown"
        ]
        if args.dry_run:
            for anchor_dir in plan:
                runner = runner_for(anchor_dir)
                missing = inspect(anchor_dir, audit.tracked_files())["missing"]
                print(f"{anchor_dir.name}: run {runner.relative_to(REPO_ROOT)} "
                      f"to emit {len(missing)} figure(s)")
            print(f"plan: {len(plan)} driver(s)")
            return 0
        if args.jobs > 1:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                entries = list(pool.map(
                    lambda d: restore_one(d, args.timeout, args.keep_json, args.force), plan))
        else:
            entries = [restore_one(d, args.timeout, args.keep_json, args.force) for d in plan]
        # anchors that needed nothing are still worth reporting
        done = {e["anchor"] for e in entries}
        for anchor_dir in dirs:
            if anchor_dir.name not in done:
                record = inspect(anchor_dir, audit.tracked_files())
                if record["status"] == "unknown":
                    record["status"] = "not_attempted"
                entries.append(record)

    payload = build_payload(entries)
    summary = payload["summary"]
    for entry in entries:
        extra = ""
        if entry.get("run"):
            extra = f" ({entry['run']['wall_seconds']}s, exit={entry['run']['exit_code']})"
        print(f"[{entry['status']:>20}] {entry['anchor']}{extra}")
        for path in entry.get("missing_after", []):
            print(f"                       still missing: {path}")
    print(json.dumps(summary, indent=2))
    if args.write:
        REPORT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    blocking = [e for e in entries if e["status"] in
                ("timeout", "runner_failed", "figure_missing", "restored_with_drift")]
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())