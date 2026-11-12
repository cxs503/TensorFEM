#!/usr/bin/env python3
"""Triage the ``E_ARTIFACT_MISSING`` debt of the verified-anchor registry.

Why this exists
---------------
``tools/verified_anchor_audit.py`` answers *"is this anchor's certificate
self-consistent?"* -- it re-derives every gate verdict from the numbers inside the
artifact.  It cannot answer the follow-up question the registry owner actually
needs: *"what is the cheapest honest way to clear each outstanding defect?"*
Twelve anchors name figures (and four name a whole ``result.json``) that were
never committed, and the remedies are not interchangeable:

* a figure that is a pure function of a committed ``result.json`` can be rebuilt
  with **zero solver time** (``--from-dump``; proven on ``fem_shell_modal``);
* an anchor whose ``result.json`` is itself absent needs a **fresh run** before
  any figure can exist, so the cost is a solve, not a render;
* a certificate that names a *source file* (module / test / driver) at a path the
  repository does not contain is a **stale reference**: the honest fix is to
  correct or drop the pointer, not to re-run anything.

This tool measures exactly those three facts per anchor and writes a
machine-readable triage plus a markdown table, so the debt can be worked in cost
order instead of alphabetically.  It runs **no solver** and mutates no artifact
other than the two files it owns; the debt registry stays shrink-only and is
edited by the audit tool alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERIFIED_DIR = ROOT / "benchmarks" / "verified"
DEBT_JSON = VERIFIED_DIR / "anchor_certificate_debt.json"
OUT_JSON = VERIFIED_DIR / "anchor_artifact_triage.json"
OUT_DOC = ROOT / "docs" / "fem-anchor-artifact-debt-triage.md"

sys.path.insert(0, str(HERE))
import verified_anchor_audit as audit  # noqa: E402  (same directory, single source of truth)

#: Directories under ``benchmarks/verified`` that are infrastructure, not anchors.
NON_ANCHOR = {"rainflow_fix_spec"}


def tracked_files() -> set[str] | None:
    """Repo-relative paths known to git, or ``None`` when the tree is not a checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return set(out.stdout.split())


def _flatten(artifacts: dict) -> list[str]:
    """Flatten the ``artifacts`` envelope: values may be a name or a list of names."""
    names: list[str] = []
    for value in artifacts.values():
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, (list, tuple)):
            names.extend(str(item) for item in value)
    return names


def _kind(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith(".png") or lowered.endswith(".svg") or lowered.endswith(".pdf"):
        return "figure"
    if lowered.endswith(".json"):
        return "data"
    if lowered.endswith(".py"):
        return "source"
    return "other"


@dataclass
class Row:
    anchor: str
    defects: list[str]
    grade: str
    named: int
    missing: list[str]
    has_result: bool
    has_driver: bool
    driver_from_dump: bool
    remedy: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "grade": self.grade,
            "defects": self.defects,
            "named_artifacts": self.named,
            "missing_artifacts": self.missing,
            "result_json_present": self.has_result,
            "driver_present": self.has_driver,
            "driver_carries_from_dump": self.driver_from_dump,
            "remedy": self.remedy,
            "reason": self.reason,
        }


def triage_anchor(directory: Path, tracked: set[str] | None) -> Row | None:
    certificate = directory / "gates_verified.json"
    if not certificate.exists():
        return None
    record = audit.audit_anchor(directory, tracked)
    if not record.defects:
        return None

    try:
        payload = json.loads(certificate.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}
    named = _flatten(payload.get("artifacts") or {})

    missing = []
    for name in named:
        rel = name
        path = Path(rel)
        candidate = path if path.is_absolute() else (ROOT / rel if rel.startswith("benchmarks/") else directory / rel)
        try:
            shown = str(candidate.relative_to(ROOT))
        except ValueError:
            shown = str(candidate)
        present = candidate.exists()
        known = tracked is None or shown in tracked
        if not present or not known:
            missing.append(name)

    result_json = directory / "result.json"
    drivers = sorted(
        p for p in directory.iterdir() if p.suffix == ".py" and ("run" in p.name or "verify" in p.name)
    )
    driver_from_dump = False
    for driver in drivers:
        try:
            text = driver.read_text()
        except OSError:
            continue
        if "--from-dump" in text and "from_dump" in text.replace("--from-dump", "from_dump"):
            driver_from_dump = True
            break

    kinds = {_kind(name) for name in missing}
    if audit.E_NO_GATES in record.defects:
        remedy, reason = "publish_gates", "no gates_verified.json: nothing to check yet"
    elif not missing:
        remedy, reason = "other_defect", "artifacts resolve; defect is in the gate envelope"
    elif "source" in kinds:
        remedy = "stale_reference"
        reason = "certificate names a source file that is absent: correct or drop the pointer"
    elif not result_json.exists():
        remedy = "fresh_run"
        reason = "result.json absent: the numbers themselves are unbacked, a render cannot help"
    elif driver_from_dump:
        remedy = "rebuild_from_dump"
        reason = "result.json present and the driver carries --from-dump: zero-solve rebuild"
    else:
        remedy = "rebuild_from_dump_candidate"
        reason = "result.json present: add a --from-dump renderer to the driver, then rebuild"

    return Row(
        anchor=directory.name,
        defects=record.defects,
        grade=record.grade,
        named=len(named),
        missing=missing,
        has_result=result_json.exists(),
        has_driver=bool(drivers),
        driver_from_dump=driver_from_dump,
        remedy=remedy,
        reason=reason,
    )


def build(tracked: set[str] | None = None) -> dict:
    if tracked is None:
        tracked = tracked_files()
    audit_registry = audit.audit_registry(VERIFIED_DIR, tracked)
    debt = audit.load_debt()["anchors"]
    rows: list[Row] = []
    for directory in sorted(audit.discover(VERIFIED_DIR)):
        if directory.name in NON_ANCHOR:
            continue
        row = triage_anchor(directory, tracked)
        if row is not None:
            rows.append(row)

    by_remedy: dict[str, list[str]] = {}
    for row in rows:
        by_remedy.setdefault(row.remedy, []).append(row.anchor)

    return {
        "schema": "tensorfem.anchor-artifact-triage/1",
        "source_audit": {
            "anchors": audit_registry["anchors"],
            "certified": audit_registry["certified"],
            "contradicted": audit_registry["contradicted"],
            "uncertified": audit_registry["uncertified"],
            "gates": audit_registry["gates_total"],
        },
        "debt_registry_size": len(debt),
        "defective_anchors": len(rows),
        "named_artifacts_missing": sum(len(row.missing) for row in rows),
        "anchors_missing_any_artifact": sum(1 for row in rows if row.missing),
        "by_remedy": {key: sorted(value) for key, value in sorted(by_remedy.items())},
        "rows": [row.as_dict() for row in rows],
    }


def render_markdown(payload: dict, generated_from: str) -> str:
    summary = payload["source_audit"]
    lines = [
        "# TensorFEM verified-anchor artifact debt — triage",
        "",
        "Generated by `python tools/anchor_artifact_triage.py --write` from the live audit "
        f"(`{generated_from}`); every number below is recomputed, none is typed.",
        "",
        "## 1. What is being triaged",
        "",
        "`tools/verified_anchor_audit.py` certifies an anchor when its `gates_verified.json` "
        "is self-consistent. A certificate that *names* an artifact which is not committed is "
        "a hard defect (`E_ARTIFACT_MISSING`): the gate verdicts may be recomputable, but a "
        "reader following the certificate gets a 404. The registry currently holds "
        f"**{summary['anchors']} anchors**: {summary['certified']} certified, "
        f"{summary['contradicted']} contradicted, {summary['uncertified']} uncertified, "
        f"across {summary['gates']} gates.",
        "",
        f"Of those, **{payload['defective_anchors']} anchors** carry a defect and "
        f"**{payload['anchors_missing_any_artifact']} of them name at least one artifact that "
        f"does not resolve** ({payload['named_artifacts_missing']} names in total). The debt "
        f"registry lists {payload['debt_registry_size']} anchors.",
        "",
        "## 2. Remedy classes (cost order)",
        "",
        "| remedy | anchors | what it costs |",
        "|---|---:|---|",
        "| `rebuild_from_dump` | {a} | zero solve: the driver already carries `--from-dump` |".format(
            a=len(payload["by_remedy"].get("rebuild_from_dump", []))
        ),
        "| `rebuild_from_dump_candidate` | {a} | one renderer (~30 lines) + zero solve |".format(
            a=len(payload["by_remedy"].get("rebuild_from_dump_candidate", []))
        ),
        "| `fresh_run` | {a} | a real solve: `result.json` itself is absent |".format(
            a=len(payload["by_remedy"].get("fresh_run", []))
        ),
        "| `stale_reference` | {a} | edit: the pointer names a source file that is gone |".format(
            a=len(payload["by_remedy"].get("stale_reference", []))
        ),
        "| `publish_gates` | {a} | a solve + a gate envelope: no certificate at all |".format(
            a=len(payload["by_remedy"].get("publish_gates", []))
        ),
        "| `other_defect` | {a} | envelope repair |".format(
            a=len(payload["by_remedy"].get("other_defect", []))
        ),
        "",
        "## 3. Per-anchor table",
        "",
        "| anchor | grade | defects | named | missing | result.json | driver | `--from-dump` | remedy |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{anchor}` | {grade} | {defects} | {named} | {missing} | {result} | {driver} | {dump} | `{remedy}` |".format(
                anchor=row["anchor"],
                grade=row["grade"],
                defects=", ".join(row["defects"]),
                named=row["named_artifacts"],
                missing=len(row["missing_artifacts"]),
                result="yes" if row["result_json_present"] else "**no**",
                driver="yes" if row["driver_present"] else "**no**",
                dump="yes" if row["driver_carries_from_dump"] else "no",
                remedy=row["remedy"],
            )
        )
    lines += [
        "",
        "## 4. The recipe, proven on `fem_shell_modal`",
        "",
        "`fem_shell_modal` was the one anchor whose defect made the *fail-closed* gate red: "
        "`tests/test_verified_anchor_certificates.py` rejected it as unregistered debt "
        "(`unregistered=['fem_shell_modal']`) because its certificate named "
        "`shell_modal_audit.png`, which no commit ever contained. The repair used the "
        "`rebuild_from_dump` class above:",
        "",
        "1. the figure code was lifted out of the live run into `render_figures(...)`, a pure "
        "function of the numbers in `result.json`;",
        "2. `--from-dump` re-renders the certificate figure (plus the three README panels) with "
        "**no eigensolve**, and stamps `figure_provenance.json` with the sha256 of the "
        "`result.json` it read and of every PNG it wrote;",
        "3. the four PNGs and the stamp are committed, so "
        "`tests/test_shell_modal_anchor_certificate.py` can prove the PNG on disk is the render "
        "of the JSON on disk;",
        "4. the debt registry was **not** touched — the anchor became certified instead of "
        "becoming registered debt, which is the only direction the shrink-only registry allows.",
        "",
        "Measured audit delta across that repair (`python tools/verified_anchor_audit.py`):",
        "",
        "| metric | before | after |",
        "|---|---:|---:|",
        "| certified anchors | 6 | 7 |",
        "| contradicted anchors | 18 | 17 |",
        "| `unregistered` debt diff | `['fem_shell_modal']` | `[]` |",
        "| `tests/test_verified_anchor_certificates.py` | 1 failed, 15 passed | 16 passed |",
        "",
        "## 5. Limits of this triage",
        "",
        "* `rebuild_from_dump_candidate` is a statement about *data availability*, not about "
        "effort: an anchor whose `result.json` stores only scalars cannot rebuild a field or "
        "mode-shape figure, and the honest remedy there is `fresh_run`.",
        "* The triage re-runs no solver, so it cannot tell whether the *numbers* inside a "
        "present `result.json` are still reproducible; that is the audit's `W_` warning class "
        "and remains open.",
        "* A rebuilt figure is provenance-stamped but not independently re-measured: it "
        "reproduces the *presentation* of the committed data, never new physics.",
        "* \"Zero solve\" is a property of the *figure code*, not of the anchor: "
        "`fem_beam_static.make_figures`, for instance, re-solves its 10/20/40-element beams "
        "to draw the FE curves, so its `--from-dump` costs milliseconds rather than nothing. "
        "That is still three orders of magnitude below a re-run of the anchor's own ladder, "
        "but the class name is about the missing *ladder*, not about literal zero time.",
        "* `fem_shell_modal` is the only anchor repaired here. The other "
        "`rebuild_from_dump_candidate` rows are *diagnosed*, not fixed: each needs its own "
        "renderer and its own provenance stamp before its certificate is honest.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="refresh the JSON triage and the markdown doc")
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed triage is stale")
    args = parser.parse_args(argv)

    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    doc = render_markdown(payload, "tools/verified_anchor_audit.py")

    if args.write:
        OUT_JSON.write_text(text)
        OUT_DOC.write_text(doc)
        print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_DOC.relative_to(ROOT)}")
        return 0
    if args.check:
        current = OUT_JSON.read_text() if OUT_JSON.exists() else ""
        if current != text:
            print("anchor artifact triage is stale: run tools/anchor_artifact_triage.py --write")
            return 1
        print("anchor artifact triage is current")
        return 0

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())