"""Certificate audit of the ``benchmarks/verified/*`` anchor registry.

Why this exists
---------------
Every anchor under ``benchmarks/verified/<anchor>/`` publishes a
``gates_verified.json`` whose gates are claims of the form

    quantity <operator> threshold   declared passed: true / false

``value`` and ``threshold`` are numbers *inside the artifact*, so the verdict is
recomputable without re-running the solver.  That turns the anchor registry into
a machine-checkable certificate store rather than a set of prose claims:

* a gate whose declared ``passed`` contradicts ``value <operator> threshold`` is
  an **integrity defect** -- no physics is needed to see it;
* a top-level ``passed: true`` with a failing gate inside is the same defect one
  level up;
* an anchor with no ``gates_verified.json`` has no certificate at all, and the
  historical ladder (``benchmarks/verified/test_verification_ladder.py``)
  silently **skips** such a case, so its absence is invisible in a green suite;
* a gate that declares PASS while carrying no numeric threshold is
  **unfalsifiable**: nothing can ever turn it red.

``tests/test_verified_anchor_certificates.py`` runs this audit and fails closed:
the set of uncertified anchors must equal a shrink-only debt registry, so a new
anchor cannot be added without a self-consistent certificate, and a repaired
anchor must be deleted from the debt list rather than silently tolerated.

Severity model
--------------
``E_`` codes are hard defects (they block certification).  ``W_`` codes are
reported but tolerated, because they describe *ambiguity*, not falsity: an
operator that only exists in the prose, an artifact named by bare filename, a
gate whose verdict cannot be recomputed but is conservatively declared FAIL.

What this deliberately does **not** do
--------------------------------------
It does not judge whether the oracle named in a gate is the *right* one -- that
needs a domain reviewer reading ``quantity`` -- and it re-runs no solver.  It
catches "the certificate contradicts its own numbers", "the evidence is
missing", "the evidence is untracked", "the pass is unfalsifiable"; not "the
physics is wrong".

Usage
-----
    python tools/verified_anchor_audit.py            # audit + summary
    python tools/verified_anchor_audit.py --check    # exit 1 if the debt registry drifted
    python tools/verified_anchor_audit.py --write    # refresh artifact, debt registry, doc table
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_DIR = REPO_ROOT / "benchmarks" / "verified"
GATE_SCHEMA = "tensorfem.verified-gates.v1"
ARTIFACT_JSON = VERIFIED_DIR / "anchor_certificates.json"
DEBT_JSON = VERIFIED_DIR / "anchor_certificate_debt.json"
DOC_PATH = REPO_ROOT / "docs" / "verified-anchor-certificate-table.md"

RUNNER_NAMES = ("run.py", "run_verification.py")

#: hard defect codes (block certification)
E_NO_GATES = "E_NO_GATES"
E_GATES_UNREADABLE = "E_GATES_UNREADABLE"
E_SCHEMA = "E_SCHEMA"
E_NO_GATES_LIST = "E_NO_GATES_LIST"
E_GATE_MALFORMED = "E_GATE_MALFORMED"
E_GATE_CONTRADICTION = "E_GATE_CONTRADICTION"
E_OVERALL_CONTRADICTION = "E_OVERALL_CONTRADICTION"
E_OVERALL_ABSENT = "E_OVERALL_ABSENT"
E_FAILED_GATES_LIST = "E_FAILED_GATES_LIST"
E_ANCHOR_NAME_MISMATCH = "E_ANCHOR_NAME_MISMATCH"
E_ARTIFACT_MISSING = "E_ARTIFACT_MISSING"
E_ARTIFACT_UNTRACKED = "E_ARTIFACT_UNTRACKED"
E_PASS_WITHOUT_CRITERION = "E_PASS_WITHOUT_CRITERION"
E_GATE_VALUE_NOT_FINITE = "E_GATE_VALUE_NOT_FINITE"

#: soft codes (reported, tolerated)
W_NO_RUNNER = "W_NO_RUNNER"
W_NO_RESULT = "W_NO_RESULT"
W_UNDECLARED_ARTIFACTS = "W_UNDECLARED_ARTIFACTS"
W_ARTIFACT_AMBIGUOUS = "W_ARTIFACT_AMBIGUOUS"
W_OPERATOR_INFERRED = "W_OPERATOR_INFERRED"
W_OPERATOR_ABSENT = "W_OPERATOR_ABSENT"
W_CONTRADICTION_INFERRED = "W_CONTRADICTION_INFERRED"
W_UNRECOMPUTABLE_FAIL = "W_UNRECOMPUTABLE_FAIL"
W_ARTIFACT_NOT_A_PATH = "W_ARTIFACT_NOT_A_PATH"
W_THRESHOLD_FROM_PROSE = "W_THRESHOLD_FROM_PROSE"
W_FAILED_GATES_LIST_UNCHECKED = "W_FAILED_GATES_LIST_UNCHECKED"

HARD_PREFIX = "E_"

#: operator -> predicate on (value, threshold)
OPERATORS = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: math.isclose(v, t, rel_tol=1e-12, abs_tol=0.0),
    "abs<=": lambda v, t: abs(v) <= t,
}

#: prose -> operator, only for explicit symbols.  Deliberately does not read
#: plain English ("below", "at least") -- that is a judgement, not a parse.
_PROSE_OPERATORS = (
    ("<=", re.compile(r"(<=|\u2264)")),
    (">=", re.compile(r"(>=|\u2265)")),
    ("<", re.compile(r"(?<![<>])<(?![=])")),
    (">", re.compile(r"(?<![<>])>(?![=])")),
)

#: directories under benchmarks/verified/ that are not anchors
NON_ANCHOR_DIRS = {"_cache", "anchors", "artifacts", "__pycache__"}

_BOOL_TRUE = {"PASS", "PASSED", "TRUE"}
_BOOL_FALSE = {"FAIL", "FAILED", "FALSE"}


@dataclass
class GateAudit:
    id: str
    quantity: str
    operator: str | None
    operator_source: str
    value: float | None
    threshold: float | None
    declared_passed: bool | None
    recomputed_passed: bool | None
    defects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.recomputed_passed is None or self.declared_passed is None:
            return "unrecomputable"
        if not self.agrees:
            return "contradicted"
        return "agrees"

    @property
    def agrees(self) -> bool:
        return self.declared_passed is not None and self.declared_passed == self.recomputed_passed

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "quantity": self.quantity,
            "operator": self.operator,
            "operator_source": self.operator_source,
            "value": self.value,
            "threshold": self.threshold,
            "declared_passed": self.declared_passed,
            "recomputed_passed": self.recomputed_passed,
            "status": self.status,
            "defects": self.defects,
            "warnings": self.warnings,
        }


@dataclass
class AnchorAudit:
    anchor: str
    directory: str
    has_runner: bool
    has_result: bool
    has_gates: bool
    declared_overall: bool | None
    recomputed_overall: bool | None
    passed_gates: int
    failed_gates: int
    defects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gates: list[GateAudit] = field(default_factory=list)

    @property
    def hard_defects(self) -> list[str]:
        return [code for code in self.defects if code.startswith(HARD_PREFIX)]

    @property
    def certified(self) -> bool:
        return self.has_gates and not self.hard_defects

    @property
    def grade(self) -> str:
        """``certified`` | ``contradicted`` | ``uncertified``."""
        if self.certified:
            return "certified"
        if self.has_gates and self.hard_defects:
            return "contradicted"
        return "uncertified"

    def as_dict(self) -> dict:
        return {
            "anchor": self.anchor,
            "directory": self.directory,
            "grade": self.grade,
            "certified": self.certified,
            "has_runner": self.has_runner,
            "has_result": self.has_result,
            "has_gates": self.has_gates,
            "declared_overall": self.declared_overall,
            "recomputed_overall": self.recomputed_overall,
            "gates": len(self.gates),
            "gates_passed": self.passed_gates,
            "gates_failed": self.failed_gates,
            "gates_contradicted": sum(
                1 for g in self.gates if g.status == "contradicted" and g.operator_source == "declared"
            ),
            "gates_contradicted_inferred": sum(
                1 for g in self.gates if g.status == "contradicted" and g.operator_source == "inferred"
            ),
            "gates_unrecomputable": sum(1 for g in self.gates if g.status == "unrecomputable"),
            "defects": self.defects,
            "warnings": self.warnings,
            "gate_detail": [g.as_dict() for g in self.gates],
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def tracked_files(repo_root: Path = REPO_ROOT) -> set[str] | None:
    """Return repo-relative tracked paths, or ``None`` when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def _as_float(raw) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
    else:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _as_bool(raw) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        token = raw.strip().upper()
        if token in _BOOL_TRUE:
            return True
        if token in _BOOL_FALSE:
            return False
    return None


#: a bound is only read when a comparison symbol is *adjacent* to the number, so
#: that trend statements such as ``e(4) > e(8) > e(32)`` are not mistaken for one
_PROSE_BOUND = re.compile(r"(<=|>=|\u2264|\u2265|<|>)\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)")


def _threshold_from_prose(*texts: str) -> float | None:
    """Extract a numeric bound from prose such as ``rel err < 3% at facets=8``.

    Only used when the gate carries no numeric ``threshold``.  The first
    number in the criterion text is taken -- anchor authors write the bound
    first, the recorded value second -- and a trailing ``%`` divides by 100.
    Callers must record ``W_THRESHOLD_FROM_PROSE`` so the reading is auditable.
    """
    for text in texts:
        if not text:
            continue
        match = _PROSE_BOUND.search(text)
        if match:
            value = float(match.group(2))
            return value / 100.0 if match.group(3) == "%" else value
    return None


def _infer_operator(*texts: str) -> str | None:
    joined = " ".join(t for t in texts if t)
    if not joined:
        return None
    for operator, pattern in _PROSE_OPERATORS:
        if pattern.search(joined):
            return operator
    return None


def _add(target: list[str], code: str) -> None:
    if code not in target:
        target.append(code)


def _resolve(text: str, anchor_dir: Path, tracked: set[str] | None) -> tuple[str, str]:
    """Resolve a declared artifact path.

    Returns ``(state, resolved_rel)`` with state in
    ``{"ok", "ok_dir", "ambiguous", "missing"}``.  ``resolved_rel`` is the
    repo-relative path when resolved.
    """
    raw = Path(text)
    candidates = [raw] if raw.is_absolute() else [REPO_ROOT / raw, anchor_dir / raw]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            rel = candidate.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return ("ok", candidate.as_posix())
        staged = "ok_dir" if candidate.is_dir() else "ok"
        return (staged, rel)
    # Not resolvable where declared: is the basename unique in the repo?
    if tracked:
        name = raw.name
        matches = [path for path in tracked if path.rsplit("/", 1)[-1] == name]
        if len(matches) == 1:
            return ("ambiguous", matches[0])
    return ("missing", "")


def _looks_like_path(text: str) -> bool:
    """Reject prose that anchors put in the ``artifacts`` map.

    A declared artifact is a path: it has no spaces and either a '/' or a file
    suffix.  ``"every number of the ladder, cross-checks and sensitivity
    studies"`` is a description, not evidence, and must not be reported as a
    missing file -- it is reported as ``W_ARTIFACT_NOT_A_PATH`` instead.
    """
    stripped = text.strip()
    if not stripped or " " in stripped:
        return False
    suffix = Path(stripped).suffix
    return "/" in stripped or bool(suffix)


def _tracked_state(rel: str, is_dir: str, tracked: set[str] | None) -> bool:
    if tracked is None:
        return True
    if is_dir == "ok_dir":
        return any(path == rel or path.startswith(rel + "/") for path in tracked)
    return rel in tracked


# --------------------------------------------------------------------------- #
# gate normalisation
# --------------------------------------------------------------------------- #
def audit_gate(raw: dict, index: int, inherited_id: str | None = None) -> GateAudit:
    gate_id = inherited_id or str(raw.get("id") or raw.get("gate") or f"gate[{index}]")
    quantity = str(
        raw.get("quantity") or raw.get("description") or raw.get("criterion") or raw.get("detail") or ""
    )
    operator_raw = raw.get("operator")
    operator = operator_raw.strip() if isinstance(operator_raw, str) else None
    source = "declared"
    if operator not in OPERATORS:
        inferred = _infer_operator(quantity, str(raw.get("note") or ""))
        operator, source = inferred, ("inferred" if inferred else "absent")
    raw_value = raw.get("value", raw.get("actual"))
    raw_threshold = raw.get("threshold", raw.get("limit"))
    value = _as_float(raw_value)
    threshold = _as_float(raw_threshold)
    declared = _as_bool(raw.get("passed", raw.get("pass", raw.get("status", raw.get("verdict")))))
    value_not_finite = False
    if value is None and isinstance(raw_value, float) and not math.isfinite(raw_value):
        value_not_finite = True
    threshold_from_prose = False
    if threshold is None and declared is not None:
        parsed = _threshold_from_prose(quantity, str(raw.get("note") or ""))
        if parsed is not None:
            threshold, threshold_from_prose = parsed, True

    gate = GateAudit(
        id=gate_id,
        quantity=quantity,
        operator=operator,
        operator_source=source,
        value=value,
        threshold=threshold,
        declared_passed=declared,
        recomputed_passed=None,
    )

    if threshold_from_prose and threshold is not None:
        _add(gate.warnings, W_THRESHOLD_FROM_PROSE)
    if value_not_finite:
        # a non-finite "value" can never be compared: it is not a measurement
        _add(gate.defects, E_GATE_VALUE_NOT_FINITE)
        return gate
    if operator is not None and value is not None and threshold is not None:
        gate.recomputed_passed = bool(OPERATORS[operator](value, threshold))

    if declared is None:
        _add(gate.defects, E_GATE_MALFORMED)
        return gate

    if gate.recomputed_passed is None:
        # cannot machine-check this gate's verdict
        if threshold is None and declared:
            _add(gate.defects, E_PASS_WITHOUT_CRITERION)
        elif operator is None and threshold is not None:
            _add(gate.warnings, W_OPERATOR_ABSENT)
        else:
            _add(gate.warnings, W_UNRECOMPUTABLE_FAIL if not declared else E_GATE_MALFORMED)
        return gate

    if source == "inferred":
        _add(gate.warnings, W_OPERATOR_INFERRED)
        if declared != gate.recomputed_passed:
            _add(gate.warnings, W_CONTRADICTION_INFERRED)
        return gate

    if declared != gate.recomputed_passed:
        _add(gate.defects, E_GATE_CONTRADICTION)
    return gate


def _iter_gates(raw_gates) -> list[tuple[dict, str | None]]:
    """Yield ``(gate_dict, inherited_id)`` for list- and dict-shaped registries."""
    if isinstance(raw_gates, list):
        return [(item, None) for item in raw_gates if isinstance(item, dict)]
    if isinstance(raw_gates, dict):
        out: list[tuple[dict, str | None]] = []
        for key, item in raw_gates.items():
            if isinstance(item, dict):
                out.append((item, str(key)))
        return out
    return []


def audit_anchor(directory: Path, tracked: set[str] | None) -> AnchorAudit:
    name = directory.name
    record = AnchorAudit(
        anchor=name,
        directory=directory.relative_to(REPO_ROOT).as_posix(),
        has_runner=any((directory / runner).is_file() for runner in RUNNER_NAMES),
        has_result=(directory / "result.json").is_file(),
        has_gates=(directory / "gates_verified.json").is_file(),
        declared_overall=None,
        recomputed_overall=None,
        passed_gates=0,
        failed_gates=0,
    )

    if not record.has_runner:
        _add(record.warnings, W_NO_RUNNER)
    if not record.has_gates:
        _add(record.defects, E_NO_GATES)
        return record
    if not record.has_result:
        _add(record.warnings, W_NO_RESULT)

    gates_path = directory / "gates_verified.json"
    try:
        payload = json.loads(gates_path.read_text())
    except (OSError, json.JSONDecodeError):
        _add(record.defects, E_GATES_UNREADABLE)
        return record

    if isinstance(payload, list):
        # ``fem_shell_modal`` ships a bare gate list: no envelope, no schema tag,
        # and ``limit`` where the schema says ``threshold``.
        _add(record.defects, E_SCHEMA)
        payload = {"gates": payload}
    if not isinstance(payload, dict):
        _add(record.defects, E_GATES_UNREADABLE)
        return record

    if payload.get("schema") != GATE_SCHEMA:
        _add(record.defects, E_SCHEMA)

    anchor_field = payload.get("anchor")
    if isinstance(anchor_field, str) and anchor_field.strip() and anchor_field.strip() != name:
        _add(record.defects, E_ANCHOR_NAME_MISMATCH)

    entries = _iter_gates(payload.get("gates"))
    if not entries:
        # Legacy shape (``fem_wave_dispersion``): gates are top-level keys, each a
        # dict carrying its own value/threshold/verdict and no envelope at all.
        flat = [
            (value, str(key))
            for key, value in payload.items()
            if isinstance(value, dict)
            and ("value" in value or "actual" in value)
            and any(tag in value for tag in ("threshold", "limit", "verdict", "passed", "status"))
        ]
        if flat:
            entries = flat
    if not entries:
        _add(record.defects, E_NO_GATES_LIST)
        return record

    for index, (raw, inherited) in enumerate(entries):
        gate = audit_gate(raw, index, inherited)
        record.gates.append(gate)
        if gate.recomputed_passed:
            record.passed_gates += 1
        elif gate.recomputed_passed is False:
            record.failed_gates += 1
        for code in gate.defects:
            _add(record.defects, code)
        for code in gate.warnings:
            _add(record.warnings, code)

    recomputable = [g for g in record.gates if g.recomputed_passed is not None]
    if len(recomputable) == len(record.gates):
        record.recomputed_overall = all(g.recomputed_passed for g in record.gates)

    record.declared_overall = _as_bool(
        payload.get("passed", payload.get("all_gating_pass", payload.get("overall_pass")))
    )
    if record.declared_overall is None:
        _add(record.defects, E_OVERALL_ABSENT)
    elif (
        record.recomputed_overall is not None
        and record.declared_overall != record.recomputed_overall
    ):
        _add(record.defects, E_OVERALL_CONTRADICTION)

    declared_failed = payload.get("failed_gates")
    if isinstance(declared_failed, list) and record.recomputed_overall is None:
        _add(record.warnings, W_FAILED_GATES_LIST_UNCHECKED)
    elif isinstance(declared_failed, list):
        expected = {g.id for g in record.gates if g.recomputed_passed is False}
        listed = {str(item) for item in declared_failed}
        if expected != listed:
            _add(record.defects, E_FAILED_GATES_LIST)

    # Declared artifacts: existence on disk and (optionally) git tracking.
    artifacts = payload.get("artifacts")
    declared_paths: list[str] = []
    if isinstance(artifacts, dict):
        for item in artifacts.values():
            if isinstance(item, str):
                declared_paths.append(item)
            elif isinstance(item, list):
                declared_paths.extend(sub for sub in item if isinstance(sub, str))
    for text in declared_paths:
        if not _looks_like_path(text):
            _add(record.warnings, W_ARTIFACT_NOT_A_PATH)
            continue
        state, rel = _resolve(text, directory, tracked)
        if state == "missing":
            _add(record.defects, E_ARTIFACT_MISSING)
            continue
        if state == "ambiguous":
            _add(record.warnings, W_ARTIFACT_AMBIGUOUS)
            continue
        if not _tracked_state(rel, state, tracked):
            _add(record.defects, E_ARTIFACT_UNTRACKED)
    if not declared_paths:
        _add(record.warnings, W_UNDECLARED_ARTIFACTS)

    return record


def discover(directory: Path = VERIFIED_DIR) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and path.name not in NON_ANCHOR_DIRS and not path.name.startswith(".")
    )


def audit_registry(directory: Path = VERIFIED_DIR, tracked: set[str] | None = None) -> dict:
    if tracked is None:
        tracked = tracked_files()
    records = [audit_anchor(path, tracked) for path in discover(directory)]
    return {
        "schema": "tensorfem.anchor-certificates/1",
        "gate_schema": GATE_SCHEMA,
        "git_tracking_checked": tracked is not None,
        "anchors": len(records),
        "certified": sum(1 for r in records if r.certified),
        "contradicted": sum(1 for r in records if r.grade == "contradicted"),
        "uncertified": sum(1 for r in records if r.grade == "uncertified"),
        "gates_total": sum(len(r.gates) for r in records),
        "gates_recomputed": sum(
            len([g for g in r.gates if g.recomputed_passed is not None]) for r in records
        ),
        "gates_contradicted": sum(
            len([g for g in r.gates if g.status == "contradicted" and g.operator_source == "declared"])
            for r in records
        ),
        "gates_contradicted_inferred": sum(
            len([g for g in r.gates if g.status == "contradicted" and g.operator_source == "inferred"])
            for r in records
        ),
        "gates_without_numeric_criterion": sum(
            len([g for g in r.gates if g.threshold is None]) for r in records
        ),
        "gates_unrecomputable": sum(
            len([g for g in r.gates if g.status == "unrecomputable"]) for r in records
        ),
        "defect_counts": _defect_counts(records),
        "records": [record.as_dict() for record in records],
    }


def _defect_counts(records: list[AnchorAudit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for code in record.defects:
            counts[code] = counts.get(code, 0) + 1
        for gate in record.gates:
            for code in gate.defects:
                if code not in record.defects:
                    counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# debt registry
# --------------------------------------------------------------------------- #
def debt_from_audit(audit: dict) -> dict:
    return {
        "schema": "tensorfem.anchor-certificates.debt/1",
        "note": (
            "Shrink-only registry of anchors that do not yet carry a self-consistent certificate. "
            "tests/test_verified_anchor_certificates.py fails when the live audit differs from this "
            "list, so debt cannot grow silently and a repaired anchor must be deleted here. "
            "Generated by tools/verified_anchor_audit.py --write."
        ),
        "anchors": {
            record["anchor"]: {
                "grade": record["grade"],
                "defects": record["defects"],
                "remedy": _remedy(record["defects"]),
            }
            for record in audit["records"]
            if not record["certified"]
        },
    }


_REMEDIES = {
    E_NO_GATES: "publish gates_verified.json (schema tensorfem.verified-gates.v1) from a fresh run",
    E_GATE_CONTRADICTION: "recompute the gate from its own value/threshold and fix the number or the verdict",
    E_OVERALL_CONTRADICTION: "set the top-level passed flag from the gates it aggregates",
    E_OVERALL_ABSENT: "add an aggregate passed flag so the anchor's verdict needs no prose",
    E_SCHEMA: "re-emit the file in the canonical envelope (schema/anchor/passed/gates)",
    E_FAILED_GATES_LIST: "regenerate failed_gates from the gate verdicts",
    E_ARTIFACT_MISSING: "commit the artifact the certificate names, or drop the stale reference",
    E_ARTIFACT_UNTRACKED: "git add the artifact the certificate names",
    E_PASS_WITHOUT_CRITERION: "give the gate a numeric threshold, or mark it informational (not a pass)",
    E_GATE_MALFORMED: "give the gate a parseable verdict",
    E_NO_GATES_LIST: "emit a non-empty gates list",
    E_ANCHOR_NAME_MISMATCH: "align the anchor field with the directory name",
}


def _remedy(defects: list[str]) -> str:
    hints = [_REMEDIES[code] for code in defects if code in _REMEDIES]
    return "; ".join(dict.fromkeys(hints)) or "add gates_verified.json"


def load_debt(path: Path = DEBT_JSON) -> dict:
    return json.loads(path.read_text())


def debt_diff(audit: dict, debt: dict) -> dict:
    live = {r["anchor"]: r for r in audit["records"] if not r["certified"]}
    registered = debt.get("anchors", {})
    return {
        "unregistered": sorted(set(live) - set(registered)),
        "stale": sorted(set(registered) - set(live)),
        "defect_drift": sorted(
            anchor
            for anchor in set(live) & set(registered)
            if sorted(live[anchor]["defects"]) != sorted(registered[anchor].get("defects", []))
        ),
    }


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def render_markdown(audit: dict, diff: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Verified-anchor certificate audit")
    add("")
    add(
        "Generated by `tools/verified_anchor_audit.py --write` from the shipped artifacts in "
        "`benchmarks/verified/`. Every number below is read from a committed JSON file; nothing is "
        "typed by hand."
    )
    add("")
    add("## Registry totals")
    add("")
    add("| metric | value |")
    add("|---|---:|")
    add(f"| anchors | {audit['anchors']} |")
    add(f"| certified (self-consistent certificate) | {audit['certified']} |")
    add(f"| contradicted (certificate disagrees with its own numbers) | {audit['contradicted']} |")
    add(f"| uncertified (no gate file) | {audit['uncertified']} |")
    add(f"| gates total | {audit['gates_total']} |")
    add(f"| gates whose verdict was recomputed from `value`/`threshold` | {audit['gates_recomputed']} |")
    add(f"| gates contradicted under a **declared** operator | {audit['gates_contradicted']} |")
    add(f"| gates contradicted under an operator read from prose | {audit['gates_contradicted_inferred']} |")
    add(f"| gates with no numeric criterion at all | {audit['gates_without_numeric_criterion']} |")
    add(f"| gates unrecomputable (operator only in prose) | {audit['gates_unrecomputable']} |")
    add(f"| git tracking check | {'enabled' if audit['git_tracking_checked'] else 'disabled (no git)'} |")
    add("")
    if audit.get("defect_counts"):
        add("### Hard-defect census")
        add("")
        add("| code | anchors (+ gates) |")
        add("|---|---:|")
        for code, count in audit["defect_counts"].items():
            if code.startswith(HARD_PREFIX):
                add(f"| `{code}` | {count} |")
        add("")
    if diff is not None:
        add("## Debt-registry consistency")
        add("")
        add(f"* unregistered defects (new debt the CI gate rejects): `{diff['unregistered'] or 'none'}`")
        add(f"* stale debt entries (repaired anchors not yet removed): `{diff['stale'] or 'none'}`")
        add(f"* defect drift (same anchor, different defect list): `{diff['defect_drift'] or 'none'}`")
        add("")
    add("## Per-anchor certificate")
    add("")
    add("| anchor | grade | gates | failed | contradicted | unrecomputable | declared | recomputed | defects |")
    add("|---|---|---:|---:|---:|---:|---|---|---|")
    for record in audit["records"]:
        declared = "-" if record["declared_overall"] is None else str(record["declared_overall"]).lower()
        recomputed = "-" if record["recomputed_overall"] is None else str(record["recomputed_overall"]).lower()
        add(
            f"| `{record['anchor']}` | {record['grade']} | {record['gates']} | {record['gates_failed']} | "
            f"{record['gates_contradicted']} | {record['gates_unrecomputable']} | {declared} | "
            f"{recomputed} | {', '.join(record['defects'])} |"
        )
    add("")
    contradicted = [r for r in audit["records"] if r["grade"] == "contradicted"]
    if contradicted:
        add("## Contradiction detail")
        add("")
        for record in contradicted:
            add(f"### `{record['anchor']}` ({record['directory']})")
            add("")
            add("| gate | operator | value | threshold | declared | recomputed | defect |")
            add("|---|---|---:|---:|---|---|---|")
            for gate in record["gate_detail"]:
                if gate["defects"] or gate["status"] == "contradicted":
                    add(
                        f"| `{gate['id']}` | {gate['operator']} ({gate['operator_source']}) | "
                        f"{gate['value']} | {gate['threshold']} | {gate['declared_passed']} | "
                        f"{gate['recomputed_passed']} | {', '.join(gate['defects']) or gate['status']} |"
                    )
            add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the audit disagrees with the debt registry")
    parser.add_argument("--write", action="store_true", help="rewrite anchor_certificates.json, the debt registry and the doc")
    parser.add_argument("--json", action="store_true", help="print the full audit record")
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit = audit_registry()
    diff = None
    if DEBT_JSON.is_file():
        diff = debt_diff(audit, load_debt())

    if args.write:
        ARTIFACT_JSON.write_text(json.dumps(audit, indent=2) + "\n")
        DEBT_JSON.write_text(json.dumps(debt_from_audit(audit), indent=2) + "\n")
        diff = debt_diff(audit, load_debt())
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(render_markdown(audit, diff))

    if args.json:
        print(json.dumps(audit, indent=2))
    else:
        print(
            f"anchors={audit['anchors']} certified={audit['certified']} "
            f"contradicted={audit['contradicted']} uncertified={audit['uncertified']} "
            f"gates={audit['gates_total']} recomputed={audit['gates_recomputed']} "
            f"contradicted_gates={audit['gates_contradicted']} unrecomputable={audit['gates_unrecomputable']}"
        )
        for record in audit["records"]:
            if not record["certified"]:
                print(f"  [{record['grade']:13s}] {record['anchor']}: {', '.join(record['defects'])}")
        if diff is not None:
            print(
                "debt-diff unregistered={} stale={} drift={}".format(
                    diff["unregistered"], diff["stale"], diff["defect_drift"]
                )
            )

    if args.check:
        if diff is None:
            print("no debt registry; run --write first", file=sys.stderr)
            return 1
        if diff["unregistered"] or diff["defect_drift"]:
            print("FAIL: unregistered certificate defects", file=sys.stderr)
            return 1
        if diff["stale"]:
            print("FAIL: debt registry lists anchors that are now certified", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())