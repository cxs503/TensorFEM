"""Tensor knowledge-object (KO) ontology v1: compiler + validator (FEM-axis pilot).

The ontology answers three questions a downstream agent actually asks:

* **what kind of thing is this?** -> ``form`` (six closed forms: CONTRACT, METHOD,
  CASE, CRITERION, EVIDENCE, DEFECT)
* **what is it about?**        -> ``method_axis`` x ``domain_axis`` (dual axis)
* **what may I derive from it?** -> typed ``relations`` over a closed vocabulary

This module derives a KO index mechanically from evidence that already exists in
the repository (``docs/CASE-*.md``, ``benchmarks/**/*evidence*.json``, audit
documents) plus a small curated seed for methods/contracts, validates it against
``schema_v1.json`` / ``relations_v1.json``, and reports coverage.

Design rule that makes the index trustworthy: nothing is inferred silently. Every
field carries a ``field_confidence`` entry, unestablished values are explicit
``null``, and the *mechanical* evidence grade (derived from the artefact on disk)
is stored next to the *declared* grade (quoted from the document) so an overstated
claim shows up as a disagreement instead of disappearing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

SCHEMA_VERSION = "1"
FORMS = ("CASE", "METHOD", "CRITERION", "EVIDENCE", "DEFECT", "CONTRACT")
GRADES = ("REAL", "MOCK", "STUB", "WITHHELD", "NOT_SUPPORTED")
HERE = Path(__file__).resolve().parent

# --- domain-axis keyword map (dual axis, domain side) -----------------------
# Ship-performance disciplines first (ECOSYSTEM-MAP v2), then the general
# mechanics buckets TensorFEM's remaining corpus falls into. First hit wins by
# design: the map is ordered from most specific to most generic.
DOMAIN_KEYWORDS = (
    ("ice", ("冰", "ice")),
    ("vibration_noise", ("振动", "vibration", "acoustic", "声学", "modal", "模态", "frequency", "Hz")),
    ("contact", ("接触", "contact", "hertz", "压入", "indentation")),
    ("heat_transfer", ("热", "heat", "thermal", "conduction", "temperature", "温度")),
    ("stability", ("屈曲", "buckling", "buckl", "collapse", "极限强度", "ultimate strength")),
    ("structure", ("plate", "shell", "梁", "beam", "板", "壳", "stress", "应力", "plastic", "塑性", "deflection", "挠度")),
)
METHOD_TOKENS = ("FEM", "LBM", "FVM", "BEM", "SPH", "DEM", "DG", "CAD")
# keyword -> METHOD seed id, used to derive CASE --uses_method--> METHOD edges
METHOD_KEYWORDS = (
    ("METHOD-FEM-IBM-AITKEN", ("aitken", "ibm", "immersed boundary", "added mass", "附加质量")),
    ("METHOD-FEM-CZM-COHESIVE", ("cohesive", "内聚", "delamination", "分层")),
    ("METHOD-FEM-ALM-CONTACT", ("hertz", "contact", "接触", "压入", "indentation", "ice panel")),
    ("METHOD-FEM-J2-PLASTICITY", ("plastic", "塑性", "collapse", "j2", "von mises")),
    ("METHOD-FEM-HEAT-L2", ("heat", "thermal", "热", "conduction", "温度")),
    ("METHOD-FEM-TIMOSHENKO-BEAM", ("timoshenko", "beam modal", "euler-bernoulli", "悬臂梁", "cantilever")),
    ("METHOD-FEM-COROTATIONAL-BEAM", ("large deflection", "elastica", "corotational", "大变形")),
    ("METHOD-FEM-MITC4-SHELL", ("mitc", "shell4", "shell", "圆柱壳", "cylindrical shell")),
    ("METHOD-FEM-MINDLIN-PLATE", ("mindlin", "plate", "板", "kirchhoff")),
)

_TOL_PATTERNS = (
    re.compile(r"(?:gate|判据|验收|tolerance|tol|阈值)[^\n]{0,60}?(<=|≤|<)\s*(\d+(?:\.\d+)?)\s*%"),
    re.compile(r"(<=|≤|<)\s*(\d+(?:\.\d+)?)\s*%"),
)
_VERDICT_PATTERNS = (
    re.compile(r"Verdict\s*[:：]\s*\*{0,2}\s*(PASS|FAIL)", re.IGNORECASE),
    re.compile(r"结论\s*[:：]\s*\*{0,2}\s*(PASS|FAIL)", re.IGNORECASE),
)
_GRADE_PATTERN = re.compile(r"证据分级\s*[:：]\s*`{0,2}(REAL|MOCK|STUB|WITHHELD|NOT_SUPPORTED)")
_ANCHOR_PATTERN = re.compile(r"【锚点=([A-Za-z0-9._-]+)】")
_ARTIFACT_PATTERN = re.compile(r"`([A-Za-z0-9_./-]+\.(?:json|md|csv|npz))`")
_RUN_PATTERN = re.compile(r"(?:Reproduce|复现|运行)\s*[:：]?\s*`([^`\n]{4,200})`")
_SEVERITY_PATTERN = re.compile(r"\bP([0-3])\b")
_H1_PATTERN = re.compile(r"^#\s+(.{3,200})$", re.MULTILINE)

DEFECT_DOC_PATTERN = re.compile(r"(?i)(audit|verdict|defect|honest|round[23]|CASE-.*-(repair|closure|review))")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def schema(root: Path) -> dict:
    return load_json(Path(root) / "tools/tk_ontology/schema_v1.json")


def relation_vocabulary(root: Path) -> dict:
    return load_json(Path(root) / "tools/tk_ontology/relations_v1.json")


def count_numeric_leaf(obj) -> int:
    if isinstance(obj, bool):
        return 0
    if isinstance(obj, (int, float)):
        return 1
    if isinstance(obj, dict):
        return sum(count_numeric_leaf(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_numeric_leaf(v) for v in obj)
    return 0


def _domain_axis(text: str) -> list[str]:
    low = text.lower()
    for name, keys in DOMAIN_KEYWORDS:
        if any(k.lower() in low for k in keys):
            return [name]
    return ["general_mechanics"]


def _method_axis(stem: str, text: str) -> list[str]:
    axes = set()
    m = re.match(r"CASE-([A-Z]+)-", stem)
    if m and m.group(1) in METHOD_TOKENS:
        axes.add(m.group(1))
    for token in METHOD_TOKENS:
        if re.search(rf"\b{token}\b", text):
            axes.add(token)
    # this repository is the FEM site of the method axis
    axes.add("FEM")
    return sorted(axes)


def _first_match(patterns, text):
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m
    return None


def _ko(kid, form, title, method_axis, domain_axis, provenance, fields, confidence, relations=None, notes=None):
    return {
        "id": kid,
        "form": form,
        "title": title,
        "method_axis": method_axis,
        "domain_axis": domain_axis,
        "provenance": provenance,
        "fields": fields,
        "field_confidence": confidence,
        "relations": relations or [],
        "notes": notes,
    }


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # pragma: no cover - only when git is unavailable
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


PLACEHOLDER_VALUES = {"todo", "tbd", "placeholder", "stub", "n/a", "not implemented", "xxx"}
_SCAN_SUFFIXES = (".json", ".csv", ".npz")


def _name_index(root: Path) -> dict[str, list[str]]:
    """basename -> repo-relative paths, for resolving citations given relative to a script directory."""
    cache = getattr(_name_index, "_cache", None)
    if cache is not None and cache[0] == str(root):
        return cache[1]
    index: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        rel = str(path.relative_to(root))
        if rel.startswith((".git/", "node_modules/")):
            continue
        index.setdefault(path.name, []).append(rel)
    result = {k: sorted(v) for k, v in index.items()}
    _name_index._cache = (str(root), result)
    return result


def _placeholder_left(payload) -> bool:
    if isinstance(payload, str):
        return payload.strip().lower() in PLACEHOLDER_VALUES
    if isinstance(payload, dict):
        return any(_placeholder_left(v) for v in payload.values())
    if isinstance(payload, list):
        return any(_placeholder_left(v) for v in payload)
    return False


def _grade_file(path: Path):
    try:
        digest = _sha256(path)
    except OSError:
        return "WITHHELD", "unreadable_artifact", None, None
    if path.suffix.lower() != ".json":
        return "REAL", "non_json_artifact_present", digest, None
    try:
        payload = load_json(path)
    except Exception:
        return "WITHHELD", "unparsable_json", digest, None
    n_num = count_numeric_leaf(payload)
    if _placeholder_left(payload):
        return "STUB", "placeholder_values_in_artifact", digest, n_num
    if n_num == 0:
        return "STUB", "no_numeric_payload", digest, 0
    return "REAL", "numeric_payload_present", digest, n_num


def mechanical_grade(root: Path, rel: str | None):
    """Grade an evidence artefact from what is actually on disk.

    This is an *upper bound* derived mechanically: a numeric payload cannot prove a
    number was computed rather than hard-coded, so REAL here means 'a recomputable
    numeric artefact exists at (or resolvable from) the cited path', which is the
    precondition for a reviewer to confirm REAL. The vocabulary is the platform's
    (REAL/MOCK/STUB/WITHHELD/NOT_SUPPORTED); MOCK is never assigned mechanically
    because a hard-coded value can only be proven by reading the producer.

    Citation resolution: documents in this repository routinely cite artefacts
    relative to the script directory (``evidence.json``), so a bare basename is
    looked up by name before being declared missing. Ambiguous names stay
    WITHHELD -- guessing which of several same-named files the author meant would
    be exactly the kind of silent inference this ontology forbids.
    """
    if not rel:
        return "WITHHELD", "no_artifact_path", None, None
    path = root / rel
    if path.is_file():
        return _grade_file(path)
    candidates = _name_index(root).get(Path(rel).name, [])
    if len(candidates) == 1:
        grade, basis, digest, n_num = _grade_file(root / candidates[0])
        return grade, f"resolved_by_name:{candidates[0]}", digest, n_num
    if len(candidates) > 1:
        suffix_hits = [c for c in candidates if c.endswith(rel.lstrip("./"))]
        if len(suffix_hits) == 1:
            grade, basis, digest, n_num = _grade_file(root / suffix_hits[0])
            return grade, f"resolved_by_suffix:{suffix_hits[0]}", digest, n_num
        return "WITHHELD", f"ambiguous_relative_reference:{len(candidates)}", None, None
    return "WITHHELD", "missing_artifact", None, None


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def parse_case_doc(path: Path, root: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    stem = path.stem
    m_anchor = _ANCHOR_PATTERN.search(text)
    code = m_anchor.group(1) if m_anchor else (stem if stem.startswith("CASE-") else f"CASE-{stem}")
    if not code.startswith("CASE-"):
        code = f"CASE-{code}"

    verdict = "UNKNOWN"
    m_v = _first_match(_VERDICT_PATTERNS, text)
    if m_v:
        verdict = m_v.group(1).upper()
    elif re.search(r"(?i)honest[\s-]*fail", text):
        verdict = "FAIL"

    tol_text, comparator, threshold = None, "unknown", None
    m_t = _first_match(_TOL_PATTERNS, text)
    if m_t:
        comparator = "<=" if m_t.group(1) in ("<=", "≤") else "<"
        threshold = float(m_t.group(2))
        start = max(0, m_t.start() - 80)
        tol_text = " ".join(text[start:m_t.end() + 20].split())

    m_g = _GRADE_PATTERN.search(text)
    declared_grade = m_g.group(1) if m_g else "UNKNOWN"

    artifacts, seen = [], set()
    for am in _ARTIFACT_PATTERN.finditer(text):
        rel = am.group(1).strip().lstrip("./")
        if rel in seen:
            continue
        seen.add(rel)
        if not ("/" in rel or rel.endswith(".json")):
            continue
        artifacts.append(rel)

    m_run = _RUN_PATTERN.search(text)
    run_command = m_run.group(1).strip() if m_run else None

    bench = None
    for line in text.splitlines():
        low = line.lower()
        if ("基准" in line or "reference (" in low or low.strip().startswith("**reference")) and len(line.strip()) > 8:
            bench = " ".join(line.strip().lstrip("-* ").split())[:240]
            break

    tier = None
    m_tier = re.search(r"tier\s*`?([a-z]+)`?", text)
    if m_tier:
        tier = m_tier.group(1)

    rel_path = str(path.relative_to(root))
    return {
        "code": code,
        "file": rel_path,
        "verdict": verdict,
        "tolerance_text": tol_text,
        "comparator": comparator,
        "threshold_pct": threshold,
        "declared_grade": declared_grade,
        "artifact_paths": artifacts[:6],
        "run_command": run_command,
        "benchmark_reference": bench,
        "tier": tier,
        "domain_axis": _domain_axis(stem + "\n" + text[:4000]),
        "method_axis": _method_axis(stem, text),
    }


def _methods_for(text: str, available: set[str]) -> list[str]:
    low = text.lower()
    hits = []
    for mid, keys in METHOD_KEYWORDS:
        if mid not in available:
            continue
        if any(k.lower() in low for k in keys):
            hits.append(mid)
    return hits[:3]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_index(root, commit=None) -> dict:
    root = Path(root)
    commit = commit if commit is not None else _git_commit(root)
    kos: list[dict] = []
    edges: list[dict] = []

    seed_m = load_json(root / "tools/tk_ontology/seeds/fem_methods_v1.json")
    seed_c = load_json(root / "tools/tk_ontology/seeds/contracts_v1.json")

    for m in seed_m["methods"]:
        for impl in m["implementation_paths"]:
            if not (root / impl).is_file():
                raise FileNotFoundError(f"METHOD {m['id']} points at missing implementation {impl}")
        kos.append(_ko(
            m["id"], "METHOD", m["title"], ["FEM"], m["domain_axis"],
            {"repo": "TensorFEM", "path": m["implementation_paths"][0], "line": None, "commit": commit},
            {"name": m["title"], "family": m["family"], "applicability": m["applicability"],
             "validity_limits": m["validity_limits"], "implementation_paths": m["implementation_paths"],
             "references": m["references"]},
            {k: "declared" for k in ("name", "family", "applicability", "validity_limits",
                                     "implementation_paths", "references")},
        ))
    method_ids = {k["id"] for k in kos}

    for c in seed_c["criteria"]:
        kos.append(_ko(
            c["id"], "CRITERION", c["title"], ["ANALYTIC"], c["domain_axis"],
            {"repo": "TensorFEM", "path": "tools/tk_ontology/seeds/contracts_v1.json", "line": None, "commit": commit},
            {"metric": c["metric"], "comparator": c["comparator"], "threshold_pct": c["threshold_pct"],
             "unit": c["unit"], "source": c["source"]},
            {"metric": "declared", "comparator": "declared", "threshold_pct": "declared",
             "unit": "declared", "source": "declared"},
        ))
    for c in seed_c["contracts"]:
        kos.append(_ko(
            c["id"], "CONTRACT", c["title"], ["FEM"], c["domain_axis"],
            {"repo": "TensorFEM", "path": "tools/tk_ontology/seeds/contracts_v1.json", "line": None, "commit": commit},
            {"scope": c["scope"], "obligations": c["obligations"],
             "tolerance_policy": c["tolerance_policy"], "consumer": c["consumer"]},
            {"scope": "declared", "obligations": "declared",
             "tolerance_policy": "declared", "consumer": "declared"},
            relations=[{"verb": "requires", "target": "CRITERION-PLATFORM-15PCT"},
                       {"verb": "requires", "target": "CRITERION-PLATFORM-3PCT"}],
        ))

    # ---- CASE / CRITERION / EVIDENCE from case documents -------------------
    # One case is frequently described by several documents (the case, then a
    # repair round, then an audit). They are merged under the anchor id instead of
    # duplicated: a consumer asking "what do we know about CASE-X" must receive one
    # object, and the merge keeps every source path so nothing is silently dropped.
    records = [parse_case_doc(doc, root) for doc in sorted((root / "docs").glob("CASE-*.md"))]
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec["code"], []).append(rec)

    evidence_by_path: dict[str, dict] = {}
    for code in sorted(groups):
        recs = sorted(groups[code], key=lambda r: r["file"])
        base = recs[0]

        def pick(key):
            for r in recs:
                val = r[key]
                if val not in (None, "", [], "UNKNOWN"):
                    return val
            return None

        verdicts = [r["verdict"] for r in recs if r["verdict"] != "UNKNOWN"]
        verdict = "FAIL" if verdicts and verdicts[-1] == "FAIL" else (verdicts[-1] if verdicts else "UNKNOWN")
        grades = [r["declared_grade"] for r in recs if r["declared_grade"] != "UNKNOWN"]
        declared_grade = max(sorted(set(grades)), key=lambda g: grades.count(g)) if grades else "UNKNOWN"
        artifacts, seen_art = [], set()
        for r in recs:
            for art in r["artifact_paths"]:
                if art not in seen_art:
                    seen_art.add(art)
                    artifacts.append(art)
        artifacts = artifacts[:8]
        cid = code
        full_text = "\n".join((root / r["file"]).read_text(encoding="utf-8", errors="replace") for r in recs)
        method_axis = sorted({a for r in recs for a in r["method_axis"]})
        domain_axis = []
        for r in recs:
            for a in r["domain_axis"]:
                if a not in domain_axis:
                    domain_axis.append(a)
        rel = [{"verb": "uses_method", "target": mid} for mid in _methods_for(full_text, method_ids)]
        case_ko = _ko(
            cid, "CASE", f"case {cid}", method_axis, domain_axis,
            {"repo": "TensorFEM", "path": base["file"], "line": None, "commit": commit,
             "supporting_paths": [r["file"] for r in recs[1:]]},
            {"code": cid, "tier": pick("tier"),
             "benchmark_reference": pick("benchmark_reference"),
             "tolerance_text": pick("tolerance_text"), "run_command": pick("run_command"),
             "artifact_paths": artifacts, "verdict": verdict},
            {"code": "measured", "tier": "parsed" if pick("tier") else "unknown",
             "benchmark_reference": "parsed" if pick("benchmark_reference") else "unknown",
             "tolerance_text": "parsed" if pick("tolerance_text") else "unknown",
             "run_command": "parsed" if pick("run_command") else "unknown",
             "artifact_paths": "parsed" if artifacts else "unknown",
             "verdict": "parsed" if verdict != "UNKNOWN" else "unknown"},
            relations=rel,
            notes=(f"merged from {len(recs)} documents" if len(recs) > 1 else None),
        )
        kos.append(case_ko)

        threshold_pct = pick("threshold_pct")
        if threshold_pct is not None:
            crit_id = f"CRITERION-{cid[5:]}"
            crit_rec = next(r for r in recs if r["threshold_pct"] is not None)
            kos.append(_ko(
                crit_id, "CRITERION", f"declared acceptance for {cid}",
                method_axis, domain_axis,
                {"repo": "TensorFEM", "path": crit_rec["file"], "line": None, "commit": commit},
                {"metric": f"reported key quantity of {cid} vs declared benchmark",
                 "comparator": pick("comparator") or "unknown", "threshold_pct": float(threshold_pct),
                 "unit": "percent", "source": f"{crit_rec['file']} (parsed tolerance sentence)"},
                {"metric": "declared", "comparator": "parsed", "threshold_pct": "parsed",
                 "unit": "parsed", "source": "parsed"},
            ))
            case_ko["relations"].append({"verb": "has_criterion", "target": crit_id})

        for art in artifacts:
            if not art.lower().endswith(".json"):
                continue
            grade, basis, digest, n_num = mechanical_grade(root, art)
            ev_id = f"EVIDENCE-{re.sub(r'[^A-Za-z0-9]+', '-', art).strip('-').upper()}"
            if ev_id in evidence_by_path:
                evidence_by_path[ev_id]["_cases"].add(cid)
                case_ko["relations"].append({"verb": "has_evidence", "target": ev_id})
                continue
            # provenance of the *knowledge object* is the document that cites the
            # artefact (that file exists); the artefact path itself lives in fields so
            # a citation pointing outside the repo root is reported as a grading
            # finding instead of crashing the compiler.
            ev = _ko(
                ev_id, "EVIDENCE", f"artefact {art}", method_axis, domain_axis,
                {"repo": "TensorFEM", "path": base["file"], "line": None, "commit": commit},
                {"case_id": cid, "artifact_path": art,
                 "declared_grade": declared_grade, "mechanical_grade": grade,
                 "grade_basis": basis, "sha256": digest, "n_numeric_leaf": n_num},
                {"case_id": "parsed", "artifact_path": "parsed", "declared_grade": "declared",
                 "mechanical_grade": "measured", "grade_basis": "measured",
                 "sha256": "measured" if digest else "unknown",
                 "n_numeric_leaf": "measured" if n_num is not None else "unknown"},
            )
            ev["_cases"] = {cid}
            evidence_by_path[ev_id] = ev
            kos.append(ev)
            case_ko["relations"].append({"verb": "has_evidence", "target": ev_id})

    # evidence artefacts that exist but no case document cites: link them to a case
    # when the artefact's own "case" field names one, otherwise leave them orphaned
    # and let the gap report surface them (an uncited artefact is a real finding).
    uncited: list[dict] = []
    for art in sorted((root / "benchmarks").rglob("*evidence*.json")):
        rel = str(art.relative_to(root))
        if any(e["fields"]["artifact_path"] == rel for e in evidence_by_path.values()):
            continue
        try:
            payload = load_json(art)
        except Exception:
            payload = {}
        case_id = payload.get("case") if isinstance(payload, dict) else None
        grade, basis, digest, n_num = mechanical_grade(root, rel)
        ev_id = f"EVIDENCE-{re.sub(r'[^A-Za-z0-9]+', '-', rel).strip('-').upper()}"
        ev = _ko(
            ev_id, "EVIDENCE", f"uncited artefact {rel}", ["FEM"],
            _domain_axis(rel + " " + str(case_id or "")),
            {"repo": "TensorFEM", "path": rel, "line": None, "commit": commit},
            {"case_id": case_id if isinstance(case_id, str) else None, "artifact_path": rel,
             "declared_grade": "UNKNOWN", "mechanical_grade": grade, "grade_basis": basis,
             "sha256": digest, "n_numeric_leaf": n_num},
            {"case_id": "parsed" if isinstance(case_id, str) else "unknown",
             "artifact_path": "parsed", "declared_grade": "unknown",
             "mechanical_grade": "measured", "grade_basis": "measured",
             "sha256": "measured", "n_numeric_leaf": "measured"},
            notes="artefact exists on disk but no CASE document cites it",
        )
        ev["_cases"] = set()
        evidence_by_path[ev_id] = ev
        kos.append(ev)
        uncited.append(ev)

    case_by_code = {k["fields"]["code"]: k for k in kos if k["form"] == "CASE"}
    for ev in uncited:
        cited_code = ev["fields"]["case_id"]
        if not cited_code:
            continue
        target = case_by_code.get(cited_code)
        if target is None:
            hits = [k for k in case_by_code.values() if cited_code.upper() in k["id"].upper()]
            target = hits[0] if len(hits) == 1 else None
        if target is not None:
            target["relations"].append({"verb": "has_evidence", "target": ev["id"]})
            ev["fields"]["case_id"] = target["id"]

    # ---- DEFECT from audit / verdict / repair documents --------------------
    case_ids = [k["id"] for k in kos if k["form"] == "CASE"]
    for doc in sorted((root / "docs").glob("*.md")):
        if not DEFECT_DOC_PATTERN.search(doc.name):
            continue
        rel = str(doc.relative_to(root))
        text = doc.read_text(encoding="utf-8", errors="replace")
        m_h1 = _H1_PATTERN.search(text)
        symptom = " ".join((m_h1.group(1) if m_h1 else doc.stem).split())[:200]
        m_sev = _SEVERITY_PATTERN.search(text)
        severity = f"P{m_sev.group(1)}" if m_sev else "UNKNOWN"
        refuted = None
        for line in text.splitlines():
            low = line.lower()
            if any(k in low for k in ("不可复现", "not reproducible", "overstated", "虚报", "误差超标",
                                      "defect", "失效", "错误", "不合规")):
                cand = " ".join(line.strip().lstrip("-*#> ").split())
                if len(cand) > 12:
                    refuted = cand[:240]
                    break
        # Direct id citation first, then a conservative token-level match: an audit
        # document titled CASE-FEM-X-round2 is about CASE-FEM-X even when it never
        # spells the id out. A suffix shorter than 8 characters is too generic to
        # match on, and the link set is capped so one audit cannot claim the corpus.
        links = [cid for cid in case_ids if cid in text]
        text_up = text.upper()
        for cid in case_ids:
            suffix = cid[5:].upper()
            if cid in links or len(suffix) < 8:
                continue
            if suffix in text_up:
                links.append(cid)
        links = links[:3]
        m_ev = re.search(r"`([A-Za-z0-9_./-]+\.(?:json|md))`", text)
        evidence_path = m_ev.group(1) if m_ev and (root / m_ev.group(1)).is_file() else None
        did = f"DEFECT-{re.sub(r'[^A-Za-z0-9]+', '-', doc.stem).strip('-').upper()}"
        kos.append(_ko(
            did, "DEFECT", symptom, _method_axis(doc.stem, text), _domain_axis(text[:4000]),
            {"repo": "TensorFEM", "path": rel, "line": None, "commit": commit},
            {"symptom": symptom, "severity": severity, "scope": "verification of the linked case(s) or method",
             "refuted_claim": refuted, "evidence_path": evidence_path},
            {"symptom": "parsed", "severity": "parsed" if severity != "UNKNOWN" else "unknown",
             "scope": "declared", "refuted_claim": "parsed" if refuted else "unknown",
             "evidence_path": "parsed" if evidence_path else "unknown"},
            relations=([{"verb": "found_in", "target": cid} for cid in links]),
        ))

    # ---- assemble edges (declared only in KO.relations) --------------------
    by_id = {k["id"]: k for k in kos}
    vocab = {row["verb"]: row for row in relation_vocabulary(root)["verbs"]}
    for ko in kos:
        for r in ko["relations"]:
            row = vocab.get(r["verb"])
            if row is None:
                continue
            edges.append({
                "subject": ko["id"], "verb": r["verb"], "object": r["target"],
                "subject_form": ko["form"], "object_form": by_id[r["target"]]["form"] if r["target"] in by_id else None,
                "derived_inverse": False,
            })
            if row["inverse"] and r["target"] in by_id:
                edges.append({
                    "subject": r["target"], "verb": row["inverse"], "object": ko["id"],
                    "subject_form": by_id[r["target"]]["form"], "object_form": ko["form"],
                    "derived_inverse": True,
                })
    kos = sorted(kos, key=lambda k: (FORMS.index(k["form"]), k["id"]))
    edges = sorted(edges, key=lambda e: (e["subject"], e["verb"], e["object"]))
    for ko in kos:  # internal book-keeping keys never reach the artefact
        ko.pop("_cases", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_id": "https://tensor-ecosystem/knowledge/ontology/schema_v1.json",
        "repo": "TensorFEM",
        "method_axis_pilot": "FEM",
        "commit": commit,
        "kos": kos,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
def validate_index(index: dict, root, schema_doc=None, vocab=None) -> list[str]:
    """Return a list of human-readable problems. Empty list == valid."""
    import jsonschema

    root = Path(root)
    schema_doc = schema_doc or schema(root)
    vocab = vocab or relation_vocabulary(root)
    problems: list[str] = []

    validator = jsonschema.Draft202012Validator(schema_doc)
    ids = set()
    for ko in index["kos"]:
        for err in validator.iter_errors(ko):
            problems.append(f"{ko.get('id', '<no id>')}: schema violation at {list(err.absolute_path)}: {err.message}")
        if ko["id"] in ids:
            problems.append(f"{ko['id']}: duplicate id")
        ids.add(ko["id"])
        if not ko["id"].startswith(ko["form"] + "-"):
            problems.append(f"{ko['id']}: id prefix does not match form {ko['form']}")
        if not (root / ko["provenance"]["path"]).exists():
            problems.append(f"{ko['id']}: provenance path missing: {ko['provenance']['path']}")

    prim = {row["verb"]: row for row in vocab["verbs"]}
    inv = {row["inverse"]: row for row in vocab["verbs"] if row["inverse"]}
    rows = prim
    declared = {(e["subject"], e["verb"], e["object"]) for e in index["edges"] if not e["derived_inverse"]}
    by_id = {ko["id"]: ko for ko in index["kos"]}
    for ko in index["kos"]:
        for r in ko["relations"]:
            row = rows.get(r["verb"])
            if row is None:
                problems.append(f"{ko['id']}: undeclared relation verb {r['verb']}")
                continue
            if row["from"] != ko["form"]:
                problems.append(f"{ko['id']}: verb {r['verb']} not allowed on subject form {ko['form']}")
            target = by_id.get(r["target"])
            if target is None:
                problems.append(f"{ko['id']}: dangling relation target {r['target']}")
                continue
            if row["to"] != target["form"]:
                problems.append(
                    f"{ko['id']}: verb {r['verb']} expects object form {row['to']}, got {target['form']}")
            if (ko["id"], r["verb"], r["target"]) not in declared:
                problems.append(f"{ko['id']}: relation {r['verb']}->{r['target']} missing from edge list")

    # inverse edges must be present and typed correctly
    for e in index["edges"]:
        if e["derived_inverse"]:
            row = inv.get(e["verb"])
            if row is None:
                problems.append(f"edge {e['subject']}-{e['verb']}->{e['object']}: inverse verb not in vocabulary")
                continue
            fwd = (e["object"], row["verb"], e["subject"])
            if fwd not in declared:
                problems.append(f"inverse edge {e['subject']}-{e['verb']}->{e['object']} has no forward twin")
            if e["subject_form"] != row["to"] or e["object_form"] != row["from"]:
                problems.append(f"inverse edge {e['subject']}-{e['verb']}->{e['object']}: form orientation wrong")
            continue
        row = prim.get(e["verb"])
        if row is None:
            problems.append(f"edge {e['subject']}-{e['verb']}->{e['object']}: undeclared verb")
            continue
        if e["subject_form"] != row["from"] or e["object_form"] != row["to"]:
            problems.append(f"edge {e['subject']}-{e['verb']}->{e['object']}: form pair not allowed by vocabulary")

    # cardinality: one_to_one verbs may not fan out
    fan = {}
    for e in index["edges"]:
        if e["derived_inverse"]:
            continue
        row = rows.get(e["verb"])
        if row and row["cardinality"] == "one_to_one":
            fan.setdefault((e["subject"], e["verb"]), set()).add(e["object"])
            if len(fan[(e["subject"], e["verb"])]) > 1:
                problems.append(f"{e['subject']}: verb {e['verb']} is one_to_one but points at {len(fan[(e['subject'], e['verb'])])} objects")

    # evidence grade bookkeeping
    for ko in index["kos"]:
        if ko["form"] != "EVIDENCE":
            continue
        f = ko["fields"]
        if f["mechanical_grade"] not in GRADES:
            problems.append(f"{ko['id']}: mechanical_grade {f['mechanical_grade']} not in vocabulary")
        if f["mechanical_grade"] == "REAL" and not f["sha256"]:
            problems.append(f"{ko['id']}: REAL evidence without content digest")
        if f["mechanical_grade"] == "WITHHELD" and not (
                f["grade_basis"] in ("missing_artifact", "unparsable_json", "unreadable_artifact", "no_artifact_path")
                or f["grade_basis"].startswith(("ambiguous_relative_reference", "no_"))):
            problems.append(f"{ko['id']}: WITHHELD with implausible basis {f['grade_basis']}")
    return problems


# ---------------------------------------------------------------------------
# metrics / gap report
# ---------------------------------------------------------------------------
def metrics(index: dict) -> dict:
    kos = index["kos"]
    by_id = {k["id"]: k for k in kos}
    form_counts = {f: sum(1 for k in kos if k["form"] == f) for f in FORMS}
    ev = [k for k in kos if k["form"] == "EVIDENCE"]
    mech = {}
    for k in ev:
        mech[k["fields"]["mechanical_grade"]] = mech.get(k["fields"]["mechanical_grade"], 0) + 1
    decl = {}
    for k in ev:
        g = k["fields"]["declared_grade"]
        decl[g] = decl.get(g, 0) + 1

    comparable = [k for k in ev if k["fields"]["declared_grade"] in GRADES]
    agree = [k for k in comparable if k["fields"]["declared_grade"] == k["fields"]["mechanical_grade"]]

    cases = [k for k in kos if k["form"] == "CASE"]
    rel_by_case = {}
    for e in index["edges"]:
        if e["derived_inverse"]:
            continue
        rel_by_case.setdefault(e["subject"], set()).add(e["verb"])

    def frac(n, d):
        return round(n / d, 4) if d else 0.0

    closure = sum(1 for c in cases
                  if any(r["verb"] == "has_evidence" and by_id[r["target"]]["fields"]["mechanical_grade"] == "REAL"
                         for r in c["relations"]))
    with_crit = sum(1 for c in cases if "has_criterion" in rel_by_case.get(c["id"], set()))
    with_method = sum(1 for c in cases if "uses_method" in rel_by_case.get(c["id"], set()))
    with_verdict = sum(1 for c in cases if c["fields"]["verdict"] in ("PASS", "FAIL"))
    orphan = [k["id"] for k in kos if not any(e["subject"] == k["id"] or e["object"] == k["id"]
                                              for e in index["edges"])]
    dangling = [e for e in index["edges"] if e["object"] not in by_id]

    return {
        "ko_total": len(kos),
        "form_counts": form_counts,
        "edge_total": len(index["edges"]),
        "edge_declared": sum(1 for e in index["edges"] if not e["derived_inverse"]),
        "relation_verb_counts": _verb_counts(index),
        "evidence_mechanical_grades": mech,
        "evidence_declared_grades": decl,
        "grade_agreement": {
            "comparable": len(comparable),
            "agree": len(agree),
            "rate": frac(len(agree), len(comparable)),
        },
        "case_coverage": {
            "cases": len(cases),
            "with_real_evidence": closure,
            "rate_real_evidence": frac(closure, len(cases)),
            "with_criterion": with_crit,
            "rate_criterion": frac(with_crit, len(cases)),
            "with_method": with_method,
            "rate_method": frac(with_method, len(cases)),
            "with_verdict": with_verdict,
            "rate_verdict": frac(with_verdict, len(cases)),
        },
        "domain_axis_counts": _axis_counts(kos, "domain_axis"),
        "method_axis_counts": _axis_counts(kos, "method_axis"),
        "orphan_ko": orphan,
        "dangling_edges": len(dangling),
        "declared_vs_mechanical_disagreements": [
            {
                "id": k["id"],
                "artifact_path": k["fields"]["artifact_path"],
                "declared": k["fields"]["declared_grade"],
                "mechanical": k["fields"]["mechanical_grade"],
                "basis": k["fields"]["grade_basis"],
            }
            for k in comparable if k["fields"]["declared_grade"] != k["fields"]["mechanical_grade"]
        ],
    }


def _verb_counts(index: dict) -> dict:
    out = {}
    for e in index["edges"]:
        if e["derived_inverse"]:
            continue
        out[e["verb"]] = out.get(e["verb"], 0) + 1
    return dict(sorted(out.items()))


def _axis_counts(kos, axis) -> dict:
    out = {}
    for k in kos:
        for a in k[axis]:
            out[a] = out.get(a, 0) + 1
    return dict(sorted(out.items()))


def gap_report(index: dict, m: dict) -> str:
    by_id = {k["id"]: k for k in index["kos"]}
    lines = ["# KO index gap report (FEM method axis, schema v1)", ""]
    lines.append(f"- KO total: **{m['ko_total']}** — " + ", ".join(f"{k}={v}" for k, v in m["form_counts"].items()))
    lines.append(f"- declared edges: {m['edge_declared']} (with inverse twins: {m['edge_total']})")
    lines.append(f"- CASE with re-runnable REAL evidence: **{m['case_coverage']['with_real_evidence']}"
                 f"/{m['case_coverage']['cases']}** = {m['case_coverage']['rate_real_evidence']:.1%}")
    lines.append("")
    lines.append("## 1. Cases whose cited artefacts cannot be found / parsed (WITHHELD)")
    lines.append("")
    lines.append("| case | artefact | basis |")
    lines.append("|---|---|---|")
    rows = 0
    for c in index["kos"]:
        if c["form"] != "CASE":
            continue
        for r in c["relations"]:
            if r["verb"] != "has_evidence":
                continue
            e = by_id[r["target"]]["fields"]
            if e["mechanical_grade"] in ("WITHHELD", "STUB"):
                lines.append(f"| {c['id']} | `{e['artifact_path']}` | {e['grade_basis']} |")
                rows += 1
    if rows == 0:
        lines.append("| — | none | — |")
    lines.append("")
    lines.append("## 2. Declared grade vs mechanical grade disagreements")
    lines.append("")
    dis = m["declared_vs_mechanical_disagreements"]
    if dis:
        lines.append("| evidence | artefact | declared | mechanical | basis |")
        lines.append("|---|---|---|---|---|")
        for d in dis:
            lines.append(f"| {d['id']} | `{d['artifact_path']}` | {d['declared']} | {d['mechanical']} | {d['basis']} |")
    else:
        lines.append("none")
    lines.append("")
    lines.append("## 3. Cases with no parsed acceptance criterion")
    lines.append("")
    no_crit = [c["id"] for c in index["kos"]
               if c["form"] == "CASE" and not any(r["verb"] == "has_criterion" for r in c["relations"])]
    lines.append(", ".join(no_crit) if no_crit else "none")
    lines.append("")
    lines.append("## 4. Orphan objects (no relation in either direction)")
    lines.append("")
    lines.append(", ".join(m["orphan_ko"]) if m["orphan_ko"] else "none")
    lines.append("")
    return "\n".join(lines)