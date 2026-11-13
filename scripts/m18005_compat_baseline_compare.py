#!/usr/bin/env python3
"""Record the evidence for the public-API compatibility gate (mission m18005).

Two independent measurements, both written to one JSON artifact so the numbers in
the report are machine-generated instead of hand-copied:

1. **Real history.**  Walk the commits that touched ``api/public-api-v1.json``,
   compare every consecutive pair with the new gate in both modes, and record the
   verdict.  This shows what the gate says about the API this repository actually
   shipped (395 names, six transitions) instead of about toy fixtures.
2. **Detection contrast.**  Build a scratch copy of the tree, inject *code
   consistent* API breaks into it -- the manifest and the module agree, so the
   point-in-time snapshot gate cannot see anything wrong -- and run both gates.
   The point-in-time audit is the baseline (`main` before this change had nothing
   else); the compatibility gate is the treatment.

Usage
-----
    python scripts/m18005_compat_baseline_compare.py [--out artifacts/...json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "api/public-api-v1.json"
PACKAGED_REL = "src/tensorfem/public_api_manifest.json"
DEFAULT_OUT = ROOT / "artifacts/m18005_public_api_compat_baseline.json"


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict[str, object]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-1500:],
        "seconds": round(time.time() - started, 2),
    }


def manifest_commits() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--format=%H", "--", MANIFEST_REL],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out


def history_transitions() -> list[dict[str, object]]:
    from public_api_compat import classify, load_manifest  # local import: sibling script

    commits = manifest_commits()
    rows = []
    for older, newer in zip(commits[1:], commits[:-1]):  # oldest -> newest
        base = load_manifest(f"git:{older}:{MANIFEST_REL}")
        head = load_manifest(f"git:{newer}:{MANIFEST_REL}")
        pr = classify(base, head, mode="pr")
        release = classify(base, head, mode="release")
        rows.append(
            {
                "base_commit": older[:8],
                "head_commit": newer[:8],
                "base_version": pr["base_version"],
                "head_version": pr["head_version"],
                "entries": pr["entries"],
                "counts": pr["counts"],
                "breaking": [change["kind"] + ":" + change["name"] for change in pr["breaking"]],
                "pr_mode": {"required_bump": pr["required_bump"], "compliant": pr["compliant"]},
                "release_mode": {
                    "required_bump": release["required_bump"],
                    "compliant": release["compliant"],
                },
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# detection contrast on a scratch copy of the tree
# --------------------------------------------------------------------------- #
def _scratch_tree(tmp: Path) -> Path:
    """Copy only what the two gates need (package sources, manifests, scripts)."""
    for rel in ("src", "api", "scripts", "pyproject.toml"):
        source = ROOT / rel
        target = tmp / rel
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    return tmp


def _rewrite_manifests(tree: Path, mutate) -> None:
    for rel in (MANIFEST_REL, PACKAGED_REL):
        path = tree / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["entries"] = mutate(payload["entries"])
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _drop_from_toplevel(tree: Path, name: str) -> None:
    """Remove ``name`` from ``__init__.py`` (import line + ``__all__`` token)."""
    path = tree / "src/tensorfem/__init__.py"
    text = path.read_text(encoding="utf-8")
    lines = [
        line
        for line in text.splitlines(keepends=True)
        if not (line.startswith("from .") and line.rstrip().endswith(name))
    ]
    text = "".join(lines).replace(f'"{name}", ', "", 1)
    path.write_text(text, encoding="utf-8")


def _demote(entries: list[dict], name: str, stability: str) -> list[dict]:
    for entry in entries:
        if entry["name"] == name:
            entry["stability"] = stability
    return entries


def detection_contrast(tmp_root: Path) -> dict[str, object]:
    """Same break, two gates: the snapshot gate is blind, the release gate is not."""
    faults = [
        {
            "id": "F1-remove-stable-name",
            "what": "TrussModel (stable) deleted from __init__ imports + __all__ + both manifests",
            "apply_manifest": lambda entries: [e for e in entries if e["name"] != "TrussModel"],
            "apply_code": lambda tree: _drop_from_toplevel(tree, "TrussModel"),
        },
        {
            "id": "F2-demote-stable-name",
            "what": "solve_linear_static demoted stable -> experimental in both manifests",
            "apply_manifest": lambda entries: _demote(
                entries, "solve_linear_static", "experimental"
            ),
            "apply_code": lambda tree: None,
        },
    ]
    rows = []
    pristine = tmp_root / "_pristine"
    shutil.copytree(tmp_root / "src", pristine / "src")
    shutil.copytree(tmp_root / "api", pristine / "api")
    for fault in faults:
        tree = tmp_root / fault["id"]
        shutil.copytree(tmp_root / "src", tree / "src")
        shutil.copytree(tmp_root / "api", tree / "api")
        shutil.copytree(tmp_root / "scripts", tree / "scripts")
        _rewrite_manifests(tree, fault["apply_manifest"])
        fault["apply_code"](tree)
        # Point the import at *this* fault tree: a PYTHONPATH left at the pristine
        # copy would audit the unmodified package and fake a detection.
        env = {**os.environ, "PYTHONPATH": str(tree / "src")}
        import_check = _run(
            [
                sys.executable,
                "-c",
                "import tensorfem;print('import-ok', tensorfem.__file__, len(tensorfem.__all__))",
            ],
            tree,
            env=env,
        )
        snapshot = _run([sys.executable, str(tree / "scripts/audit_public_api.py")], tree, env=env)
        release = _run(
            [
                sys.executable,
                str(ROOT / "scripts/public_api_compat.py"),
                "--base",
                str(ROOT / MANIFEST_REL),
                "--head",
                str(tree / MANIFEST_REL),
            ],
            ROOT,
        )
        rows.append(
            {
                "id": fault["id"],
                "what": fault["what"],
                "import_ok": import_check["returncode"] == 0
                and "import-ok" in str(import_check["stdout"]),
                "snapshot_gate_detected": snapshot["returncode"] != 0,
                "snapshot_gate_stdout": snapshot["stdout"],
                "compat_gate_detected": release["returncode"] != 0,
                "compat_gate_stdout": release["stdout"],
                "compat_gate_stderr": release["stderr"],
            }
        )
    injected = len(rows)
    return {
        "injected_faults": injected,
        "snapshot_gate_detections": sum(1 for row in rows if row["snapshot_gate_detected"]),
        "compat_gate_detections": sum(1 for row in rows if row["compat_gate_detected"]),
        "rows": rows,
        "note": (
            "Faults are code-consistent: module and manifest were edited together, which is exactly how a "
            "real release edits them. A point-in-time snapshot check has no second opinion to compare against "
            "and therefore passes; the release gate compares against the previous reviewed snapshot and fails."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))

    started = time.time()
    history = history_transitions()
    with tempfile.TemporaryDirectory(prefix="m18005-compat-") as tmp:
        tree = _scratch_tree(Path(tmp))
        contrast = detection_contrast(tree)

    artifact = {
        "mission": "m18005",
        "what": "release-to-release compatibility gate for api/public-api-v1.json",
        "head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "scripts": {
            "scripts/public_api_compat.py": hashlib.sha256(
                (ROOT / "scripts/public_api_compat.py").read_bytes()
            ).hexdigest(),
            "reproduce": "PYTHONPATH=src python scripts/m18005_compat_baseline_compare.py",
        },
        "history": {
            "manifest_commits": len(manifest_commits()),
            "transitions": history,
            "pr_mode_failures": [
                row["base_commit"] + "->" + row["head_commit"]
                for row in history
                if not row["pr_mode"]["compliant"]
            ],
        },
        "detection_contrast": contrast,
        "seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {artifact['seconds']}s")
    print(
        f"history transitions: {len(history)}; pr-mode failures: {len(artifact['history']['pr_mode_failures'])}"
    )
    print(
        f"detection contrast: snapshot gate {contrast['snapshot_gate_detections']}/{contrast['injected_faults']}, "
        f"compat gate {contrast['compat_gate_detections']}/{contrast['injected_faults']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
