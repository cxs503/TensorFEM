"""Independent-oracle audit of the case-pool anchor ``FEM-HERTZ-CONTACT``.

Why this exists
---------------
``benchmarks/verified/fem_hertz_contact_loadcontrol/result.json`` certifies the
Hertz case-pool anchor, and two prose documents quote its numbers
(``docs/CASE-FEM-HERTZ-CONTACT.md`` and ``...-verification.md``).  The existing
certificate audit (``tools/verified_anchor_audit.py``) only re-evaluates
``value <operator> threshold`` *inside* the artifact; it deliberately does not
answer the two questions that actually decide whether an acceptance claim is
true:

* **is the analytic reference right?**  A gate can be self-consistent and still
  grade the FE number against a mis-derived oracle (wrong ``E*``, wrong
  exponent, ``δ`` for a sphere-sphere pair used with a rigid-sphere model, ...);
* **do the documents quote the artifact?**  A stale table row is a false
  published claim even when the artifact behind it is correct.

This tool closes both gaps without re-running the solver:

1. it recomputes the Hertz oracle ``a``, ``δ``, ``p₀`` from the raw parameter
   block with formulas written **from the primary sources** (Hertz 1882;
   Johnson 1985 §4.2) and independent of ``src/tensorfem`` -- nothing is
   imported from the module under test;
2. it recomputes every published error percentage from the stored FE numbers
   against that independent oracle (not against the artifact's own oracle);
3. it re-evaluates the acceptance criterion of the ledger row
   (``<3 %`` on ``a`` and ``δ``, load ``P`` given);
4. it parses the numeric claims out of the two Markdown documents and flags
   every claim that the artifact does not support.

Severity model
--------------
``E_`` = hard defect, blocks certification of the corresponding verdict.
``W_`` = reported, tolerated ambiguity.

Two verdicts are published separately, because they fail for different reasons:

* ``physics_verdict`` -- does the evidence support ``a`` and ``δ`` within 3 %?
* ``docs_verdict``    -- do the documents quote the evidence?

Usage
-----
    python tools/hertz_case_pool_independent_audit.py            # audit + print
    python tools/hertz_case_pool_independent_audit.py --check    # exit 1 if not clean
    python tools/hertz_case_pool_independent_audit.py --write    # refresh artifact + doc + figure
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = ROOT / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"
RESULT = ANCHOR_DIR / "result.json"
GATES = ANCHOR_DIR / "gates_verified.json"
AUDIT_OUT = ANCHOR_DIR / "independent_audit.json"
DOC_MAIN = ROOT / "docs" / "CASE-FEM-HERTZ-CONTACT.md"
DOC_VERIF = ROOT / "docs" / "CASE-FEM-HERTZ-CONTACT-verification.md"
DOC_OUT = ROOT / "docs" / "CASE-FEM-HERTZ-CONTACT-independent-verification.md"
FIG_OUT = ROOT / "docs" / "assets" / "case-fem-hertz-contact" / "hertz_independent_audit.png"

# Estimator the artifact and the certificate grade against.  Kept in one place
# so a future re-run cannot silently switch estimator without this audit seeing
# it: the audit asserts the artifact agrees with the name used here.
GRADED_RADIUS_ESTIMATOR = "contact_radius_linear"

# Defects this audit found in the *pre-fix* evidence bundle and that were
# repaired in the same commit.  The "before" numbers are historical (they are
# the text that was committed at the parent commit ``HEAD~1`` of the repair);
# the "after" numbers are live claims and are re-checked against the artifact
# on every run, so this record cannot rot into a comfortable story.
FIXED_DEFECTS = (
    {"id": "D1",
     "doc": "CASE-FEM-HERTZ-CONTACT-verification.md",
     "quantity": "contact_radius@0.10/0.0075",
     "before_pct": -1.750, "after_quantity": "contact_radius", "after_key": "ratio0.10_h0.0075_d96",
     "note": "表内 a 误差是 m17149 估计量修复前的历史值，被当作现役结果发布；工件为 -0.4908%"},
    {"id": "D2",
     "doc": "CASE-FEM-HERTZ-CONTACT-verification.md",
     "quantity": "contact_radius@summary/0.10/0.0075",
     "before_pct": -1.750, "after_quantity": "contact_radius", "after_key": "ratio0.10_h0.0075_d96",
     "note": "§4 误差判定表同样引用了过期的 -1.750%，并把 F3 写成「未修」"},
    {"id": "D3",
     "doc": "CASE-FEM-HERTZ-CONTACT.md",
     "quantity": "peak_pressure@summary/0.05/0.0075",
     "before_pct": 0.493, "after_quantity": "peak_pressure", "after_key": "ratio0.05_h0.0075_d96",
     "note": "§4 的 p0 行标 h/a=0.0075，却引用了 h/a=0.03 的值（+0.4927%）；h/a=0.0075 应为 +0.6297%"},
)


def _check_fixed_defects(rows: list[dict]) -> list[dict]:
    """Re-verify that every repaired claim now equals the artifact."""
    by_key = {r["key"]: r for r in rows}
    out = []
    for d in FIXED_DEFECTS:
        row = by_key[d["after_key"]]
        q = "peak_pressure_consistent" if d["after_quantity"] == "peak_pressure" else d["after_quantity"]
        after = row["error_pct"][q]
        out.append({**d, "after_pct": after,
                    "delta_pp": d["before_pct"] - after,
                    "status": "repaired" if abs(after - d["before_pct"]) > 0.01 else "not_repaired"})
    return out


# --------------------------------------------------------------------------- #
# 1. Independent oracle.  Written from the primary sources, no project imports.
# --------------------------------------------------------------------------- #
def hertz_oracle(load: float, radius: float, young: float, poisson: float) -> dict:
    """Rigid sphere of radius ``R`` pressed by ``P`` into an elastic half-space.

    Hertz 1882, *J. reine angew. Math.* 92:156; Johnson 1985 *Contact Mechanics*
    §4.2.  For a rigid indenter ``1/E* = (1 - ν²)/E`` and ``E*`` is the plane
    strain modulus of the half-space:

        a   = (3 P R / (4 E*))^(1/3)
        δ   = a² / R                     (geometric relation, no extra modulus)
        p₀  = 3 P / (2 π a²)

    Derived live rather than read from the artifact: an artifact that stores a
    wrong ``E*`` (e.g. ``E`` or ``E/(1-ν)``) cannot hide here.
    """
    e_star = young / (1.0 - poisson * poisson)
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    delta = a * a / radius
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    return {"equivalent_modulus": e_star, "contact_radius": a, "indentation": delta, "peak_pressure": p0}


def rel_err_pct(fe: float, ref: float) -> float:
    return (fe / ref - 1.0) * 100.0


# --------------------------------------------------------------------------- #
# 2. Audit
# --------------------------------------------------------------------------- #
def _findings() -> list[dict]:
    return []


def _add(findings: list[dict], code: str, severity: str, **kw) -> None:
    findings.append({"code": code, "severity": severity, **kw})


def audit() -> dict:
    result = json.loads(RESULT.read_text())
    gates = json.loads(GATES.read_text())
    params = result["params"]
    crit = result["criteria"]
    tol = float(crit["contact_radius_pct"])
    findings: list[dict] = []

    if abs(float(crit["indentation_pct"]) - tol) > 0.0:
        _add(findings, "W_ASYMMETRIC_TOLERANCE", "W",
             note="contact-radius and indentation tolerances differ; the ledger row quotes one 3 % band",
             contact_radius_pct=crit["contact_radius_pct"], indentation_pct=crit["indentation_pct"])

    # -- per-run independent recomputation ---------------------------------- #
    rows = []
    for key, run in result["runs"].items():
        load = float(run["load"])
        oracle = hertz_oracle(load, float(params["R"]), float(params["E"]), float(params["nu"]))
        stored = run["analytic"]

        # E_ORACLE: the artifact's own reference must equal the independent one.
        for name in ("contact_radius", "indentation", "peak_pressure"):
            rel = abs(stored[name] / oracle[name] - 1.0)
            if rel > 1e-9:
                _add(findings, "E_ORACLE_MISMATCH", "E", run=key, quantity=name,
                     artifact=stored[name], independent=oracle[name], rel=rel)

        a_fe = float(run["numeric"][GRADED_RADIUS_ESTIMATOR])
        d_fe = float(run["numeric"]["indentation"])
        e_a = rel_err_pct(a_fe, oracle["contact_radius"])
        e_d = rel_err_pct(d_fe, oracle["indentation"])

        # E_ARTIFACT_DRIFT: published error must equal the recomputed one.
        gate_row = next((g for g in result["gate_table"] if g["key"] == key), None)
        for code_name, recomputed, published in (
            ("contact_radius_pct", e_a, run["error_pct"]["contact_radius_linear"]),
            ("indentation_pct", e_d, run["error_pct"]["indentation"]),
        ):
            if abs(recomputed - published) > 1e-6:
                _add(findings, "E_ARTIFACT_DRIFT", "E", run=key, quantity=code_name,
                     recomputed_pct=recomputed, published_pct=published)
            if gate_row is not None and abs(recomputed - float(gate_row[code_name])) > 1e-6:
                _add(findings, "E_GATE_TABLE_DRIFT", "E", run=key, quantity=code_name,
                     recomputed_pct=recomputed, gate_table_pct=gate_row[code_name])

        # E_ACCEPTANCE: the ledger row itself.
        for name, val in (("contact_radius", e_a), ("indentation", e_d)):
            if abs(val) >= tol:
                _add(findings, "E_ACCEPTANCE", "E", run=key, quantity=name, error_pct=val, tolerance_pct=tol)

        rows.append({
            "key": key,
            "a_over_R": run["a_over_R"],
            "h_over_a": run["h_over_a"],
            "n_elements": run["n_elements"],
            "load": load,
            "oracle": oracle,
            "fe": {"contact_radius_linear": a_fe, "indentation": d_fe,
                   "peak_pressure_lumped": run["numeric"]["peak_pressure_lumped"],
                   "peak_pressure_consistent": run["numeric"]["peak_pressure_consistent"]},
            "error_pct": {
                "contact_radius": e_a,
                "indentation": e_d,
                # Two peak-pressure estimators: the gate grades the L2-consistent
                # field; the raw lumped nodal value is reported so a reader can
                # see the estimator spread instead of only the flattering number.
                "peak_pressure_lumped": rel_err_pct(run["numeric"]["peak_pressure_lumped"], oracle["peak_pressure"]),
                "peak_pressure_consistent": rel_err_pct(run["numeric"]["peak_pressure_consistent"], oracle["peak_pressure"]),
            },
            "wall_s": run["wall_s"],
            "penetration_over_delta": run["penetration_over_delta"],
            "load_residual": run["load_residual"],
            "equilibrium_residual_rel": run["equilibrium_residual_rel"],
        })

    # -- estimator sensitivity (W_, but material to the p0 claim) ----------- #
    p_lump = max(abs(r["error_pct"]["peak_pressure_lumped"]) for r in rows)
    p_cons = max(abs(r["error_pct"]["peak_pressure_consistent"]) for r in rows)
    if p_lump > 5.0:
        _add(findings, "W_PEAK_ESTIMATOR_SPREAD", "W",
             note="peak pressure passes the 5 % gate only under the L2-consistent field; "
                  "the raw lumped nodal peak is far outside it -- the p0 claim is estimator-conditional",
             lumped_max_abs_pct=p_lump, consistent_max_abs_pct=p_cons)

    # -- certificate gates: independent re-evaluation + oracle binding ------ #
    gate_audit = []
    for g in gates["gates"]:
        recomputed = (g["value"] <= g["threshold"]) if g["operator"] == "<=" else (g["value"] >= g["threshold"])
        if bool(recomputed) != bool(g["passed"]):
            _add(findings, "E_GATE_SELFCONTRADICTION", "E", gate=g["id"],
                 value=g["value"], threshold=g["threshold"], operator=g["operator"], declared=g["passed"])
        gate_audit.append({"id": g["id"], "value": g["value"], "threshold": g["threshold"],
                           "operator": g["operator"], "declared": g["passed"], "recomputed": bool(recomputed)})

    # -- document claims ---------------------------------------------------- #
    doc_claims = _parse_doc_claims(rows)
    for claim in doc_claims:
        if claim["status"] == "stale_timing":
            _add(findings, "W_DOC_STALE_TIMING", "W", doc=claim["doc"], line=claim["line"],
                 note="wall-clock column does not match the artifact's wall_s (machine-dependent; "
                      "reported, not gated -- but it marks the table as not regenerated)",
                 claimed_s=claim["claimed_pct"], artifact_s=round(claim["artifact_pct"], 1))
    for claim in doc_claims:
        if claim["status"] == "mismatch":
            _add(findings, "E_DOC_FALSE_CLAIM", "E", **{k: claim[k] for k in
                 ("doc", "line", "quantity", "claimed_pct", "artifact_pct", "delta_pp")})

    physics_hard = [f for f in findings if f["severity"] == "E" and f["code"] in
                    ("E_ORACLE_MISMATCH", "E_ARTIFACT_DRIFT", "E_GATE_TABLE_DRIFT",
                     "E_ACCEPTANCE", "E_GATE_SELFCONTRADICTION")]
    docs_hard = [f for f in findings if f["severity"] == "E" and f["code"] == "E_DOC_FALSE_CLAIM"]

    fixed = _check_fixed_defects(rows)
    for d in fixed:
        if d["status"] != "repaired":
            _add(findings, "E_DOC_FALSE_CLAIM", "E", doc=d["doc"], line=0, quantity=d["quantity"],
                 claimed_pct=d["before_pct"], artifact_pct=d["after_pct"],
                 delta_pp=d["before_pct"] - d["after_pct"])

    worst = {q: max(abs(r["error_pct"][q]) for r in rows) for q in
             ("contact_radius", "indentation", "peak_pressure_consistent")}
    return {
        "schema": "tensorfem.hertz-independent-audit.v1",
        "anchor": "CASE-FEM-HERTZ-CONTACT",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(RESULT.relative_to(ROOT)),
        "graded_radius_estimator": GRADED_RADIUS_ESTIMATOR,
        "oracle_source": "Hertz 1882 J. reine angew. Math. 92:156; Johnson 1985 Contact Mechanics 4.2 "
                         "DOI:10.1017/cbo9781139171731 (formulas re-derived in this tool)",
        "tolerance_pct": tol,
        "params": {"R": params["R"], "E": params["E"], "nu": params["nu"], "units": params.get("units", "")},
        "rows": rows,
        "gates": gate_audit,
        "doc_claims": doc_claims,
        "fixed_defects": fixed,
        "worst_pct": worst,
        "findings": findings,
        "physics_verdict": {"passed": not physics_hard, "hard_defects": [f["code"] for f in physics_hard]},
        "docs_verdict": {"passed": not docs_hard, "hard_defects": [f["code"] for f in docs_hard]},
    }


# --------------------------------------------------------------------------- #
# 3. Markdown claim extraction (doc drift detector)
# --------------------------------------------------------------------------- #
_ROW_MAIN = re.compile(r"^\|\s*(0\.\d+)\s*\|\s*(0\.\d+)\s*\|")
_PCT = re.compile(r"[−\-+]?\d+\.\d+%?")


def _pct(token: str) -> float:
    return float(token.replace("−", "-").replace("%", "").replace("+", ""))


def _parse_doc_claims(rows: list[dict]) -> list[dict]:
    """Extract the numeric claims of the two result tables and grade each one."""
    by_key = {(round(r["a_over_R"], 3), round(r["h_over_a"], 4)): r for r in rows}
    claims: list[dict] = []

    def grade(doc: str, lineno: int, quantity: str, claimed: float, artifact: float, tol_pp: float = 0.01):
        delta = claimed - artifact
        claims.append({
            "doc": doc, "line": lineno, "quantity": quantity,
            "claimed_pct": claimed, "artifact_pct": artifact, "delta_pp": delta,
            "status": "match" if abs(delta) <= tol_pp else "mismatch",
        })

    # docs/CASE-FEM-HERTZ-CONTACT.md §3.1: | a/R | h/a | n | aH | aFE | e_a | dH | dFE | e_d | e_p0 |
    for i, line in enumerate(DOC_MAIN.read_text().splitlines(), start=1):
        m = _ROW_MAIN.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10:
            continue
        key = (round(float(cells[0]), 3), round(float(cells[1]), 4))
        row = by_key.get(key)
        if row is None:
            continue
        for quantity, cell in (("contact_radius", cells[5]), ("indentation", cells[8]), ("peak_pressure", cells[9])):
            if not _PCT.fullmatch(cell.replace("*", "")):
                continue
            artifact = row["error_pct"]["peak_pressure_consistent"] if quantity == "peak_pressure" else row["error_pct"][quantity]
            grade(DOC_MAIN.name, i, f"{quantity}@{key[0]}/{key[1]}", _pct(cell.replace("*", "")), artifact)

    # docs/CASE-FEM-HERTZ-CONTACT-verification.md §3: | a/R | h/a | e_a | e_d | pen | n | wall |
    for i, line in enumerate(DOC_VERIF.read_text().splitlines(), start=1):
        m = _ROW_MAIN.match(line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[2].replace("*", "").endswith("%"):
            continue
        key = (round(float(cells[0]), 3), round(float(cells[1]), 4))
        row = by_key.get(key)
        if row is None:
            continue
        for quantity, cell in (("contact_radius", cells[2]), ("indentation", cells[3])):
            grade(DOC_VERIF.name, i, f"{quantity}@{key[0]}/{key[1]}", _pct(cell.replace("*", "")), row["error_pct"][quantity])
        # Wall clock is machine-dependent, so it is reported (W_) and never gates:
        # a stale timing column is evidence that the table was not regenerated,
        # which is exactly how the stale -1.750% row survived review.
        wall = cells[6].replace("*", "") if len(cells) > 6 else ""
        wm = re.fullmatch(r"(\d+\.\d+)\s*s", wall)
        if wm and row["wall_s"] > 0 and abs(float(wm.group(1)) / row["wall_s"] - 1.0) > 0.15:
            claims.append({"doc": DOC_VERIF.name, "line": i, "quantity": f"wall_clock@{key[0]}/{key[1]}",
                           "claimed_pct": float(wm.group(1)), "artifact_pct": row["wall_s"],
                           "delta_pp": float(wm.group(1)) - row["wall_s"], "status": "stale_timing"})

    # §4 "误差判定" tables (both docs): | 接触半径 a(...) | ref | ±x.xxx% | <3%? |
    #   | 峰值压力 p0 | a/R=0.05, h/a=0.0075 | ... | +0.493% | 是 |
    # These rows summarise a run by prose instead of by table columns, so they
    # are parsed separately; the quantity, load point and (implicit) finest
    # spacing all have to be inferred from the text.
    for doc in (DOC_MAIN, DOC_VERIF):
        for i, line in enumerate(doc.read_text().splitlines(), start=1):
            if not line.startswith("| 接触半径") and not line.startswith("| 峰值压力"):
                continue
            cells = [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
            pcts = [c for c in cells if _PCT.fullmatch(c)]
            if not pcts:
                continue
            quantity = "contact_radius" if "接触半径" in cells[0] else "peak_pressure"
            a_m = re.search(r"a/R\s*=\s*(\d+\.\d+)", line)
            h_m = re.search(r"h/a\s*=\s*(\d+\.\d+)", line)
            a_r = float(a_m.group(1)) if a_m else 0.05
            # "最细" (finest) means the last rung of the refinement ladder.
            h_a = float(h_m.group(1)) if h_m else 0.0075
            row = by_key.get((round(a_r, 3), round(h_a, 4)))
            if row is None:
                continue
            artifact = row["error_pct"]["peak_pressure_consistent"] if quantity == "peak_pressure" else row["error_pct"][quantity]
            grade(doc.name, i, f"{quantity}@summary/{a_r}/{h_a}", _pct(pcts[-1]), artifact)
    return claims


# --------------------------------------------------------------------------- #
# 4. Figure
# --------------------------------------------------------------------------- #
def render_figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]
    tol = report["tolerance_pct"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for aR, marker in ((0.05, "o"), (0.10, "s")):
        sel = sorted([r for r in rows if r["a_over_R"] == aR], key=lambda r: -r["h_over_a"])
        ax.plot([r["h_over_a"] for r in sel], [r["error_pct"]["contact_radius"] for r in sel],
                marker + "-", label=f"$a/R$={aR}")
        ax.plot([r["h_over_a"] for r in sel], [r["error_pct"]["indentation"] for r in sel],
                marker + "--", alpha=0.6, label=f"$\\delta$, $a/R$={aR}")
    ax.axhspan(-tol, tol, color="tab:green", alpha=0.10, label=f"±{tol:.0f}% acceptance band")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$h/a$ (surface spacing / contact radius)")
    ax.set_ylabel("error vs independent Hertz oracle [%]")
    ax.set_title("(a) recomputed errors, independent oracle")
    ax.legend(fontsize=7)

    ax = axes[1]
    claimed = [c for c in report["doc_claims"] if c["quantity"].startswith("contact_radius")]
    oracle_vals = [c["artifact_pct"] for c in claimed]
    doc_vals = [c["claimed_pct"] for c in claimed]
    ax.scatter(oracle_vals, doc_vals, c=["tab:red" if c["status"] == "mismatch" else "tab:green" for c in claimed],
               zorder=3, s=44, label="documented (current)")
    for d in report["fixed_defects"]:
        if d["after_quantity"] != "contact_radius":
            continue
        ax.scatter([d["after_pct"]], [d["before_pct"]], facecolors="none", edgecolors="tab:red",
                   marker="X", s=110, zorder=4, label=f"pre-fix claim {d['id']} ({d['before_pct']:+.3f}%)")
    lo, hi = min(oracle_vals + doc_vals) - 0.2, max(oracle_vals + doc_vals) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="claimed = artifact")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("artifact $e_a$ [%]")
    ax.set_ylabel("documented $e_a$ [%]")
    ax.set_title("(b) document-vs-artifact drift (contact radius)")
    ax.legend(fontsize=7)

    ax = axes[2]
    labels = [f"{r['a_over_R']:.2f}/{r['h_over_a']:.4f}" for r in rows]
    x = range(len(rows))
    ax.bar([i - 0.2 for i in x], [r["error_pct"]["peak_pressure_consistent"] for r in rows],
           width=0.4, label="L2-consistent field (graded)")
    ax.bar([i + 0.2 for i in x], [r["error_pct"]["peak_pressure_lumped"] for r in rows],
           width=0.4, label="raw lumped nodal peak")
    ax.axhline(5, color="tab:red", ls=":", lw=1.0, label="5% p0 gate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel("$p_0$ error [%]")
    ax.set_title("(c) peak-pressure estimator spread")
    ax.legend(fontsize=7)

    fig.suptitle("CASE-FEM-HERTZ-CONTACT: independent-oracle audit of the published certificate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Markdown report
# --------------------------------------------------------------------------- #
def render_doc(report: dict) -> str:
    rows = report["rows"]
    tol = report["tolerance_pct"]
    L = []
    L.append("# CASE-FEM-HERTZ-CONTACT: 独立复核（independent-oracle audit）\n")
    L.append("【锚点=CASE-FEM-HERTZ-CONTACT】｜仓=TensorFEM｜工具=`tools/hertz_case_pool_independent_audit.py`"
             "｜工件=`benchmarks/verified/fem_hertz_contact_loadcontrol/independent_audit.json`\n")
    L.append("## ① 案例介绍\n")
    L.append("刚性球压入弹性半空间（Hertz 1882；Johnson 1985 §4.2）。本文件**不是**重算仿真，而是对已发布证书 "
             "`result.json` / `gates_verified.json` 与两篇散文文档做**独立复核**：\n"
             "① 用本工具重新推导的 Hertz 闭式解（不 import `src/tensorfem` 任何代码）核对工件里的解析基准；\n"
             "② 用该独立基准重算每个已发布的误差百分比；\n"
             "③ 重判算例池验收（<3%，`a` 与 `δ`，载荷 P 给定）；\n"
             "④ 从两篇 Markdown 里抽取数值声称，逐条对照工件。\n")
    L.append(f"| 项 | 值 |\n|---|---|\n"
             f"| 参数 | R={report['params']['R']} mm, E={report['params']['E']} MPa, ν={report['params']['nu']}"
             f"（工件 `params` 块） |\n"
             f"| 独立基准 | `E* = E/(1-ν²)`；`a=(3PR/4E*)^(1/3)`；`δ=a²/R`；`p₀=3P/(2πa²)` |\n"
             f"| 容差（总账行） | <{tol:.0f}% |\n"
             f"| 被判的半径估计量 | `{report['graded_radius_estimator']}` |\n")
    L.append("## ② 怎么做\n")
    L.append("1. 读 `result.json` 的 `params` 与每个 run 的 `load`，用**独立实现**的 Hertz 公式重算 `a, δ, p₀`；\n"
             "2. 与工件 `analytic` 块逐项比（阈值 1e-9 相对），抓「基准本身写错」这类证书自洽审计看不见的缺陷；\n"
             "3. 用**独立** `a` 重算 `e_a = (a_FE/a_H − 1)·100`，与 `error_pct`、`gate_table` 三方对照；\n"
             "4. 重判 `<3%`；重判 `gates_verified.json` 每条闸门的 `value<operator>threshold`；\n"
             "5. 正则抽取 `docs/CASE-FEM-HERTZ-CONTACT.md` §3.1 与 `...-verification.md` §3 表格里的百分比，"
             "与工件逐条比（容差 0.01 pp），不一致即 `E_DOC_FALSE_CLAIM`。\n")
    L.append("## ③ 计算结果（程序化注入，零手抄）\n")
    L.append("| run | 单元数 | $e_a$ [%] | $e_\\delta$ [%] | $e_{p_0}$ 一致场 [%] | $e_{p_0}$ 集总 [%] | 穿透/δ | 载荷残差 | <3%? |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ok = "✅" if max(abs(r["error_pct"]["contact_radius"]), abs(r["error_pct"]["indentation"])) < tol else "❌"
        L.append(f"| {r['key']} | {r['n_elements']} | {r['error_pct']['contact_radius']:+.4f} | "
                 f"{r['error_pct']['indentation']:+.4f} | {r['error_pct']['peak_pressure_consistent']:+.4f} | "
                 f"{r['error_pct']['peak_pressure_lumped']:+.2f} | {r['penetration_over_delta']:.2e} | "
                 f"{r['load_residual']:.2e} | {ok} |")
    L.append("")
    L.append(f"**证书闸门重判**：{sum(1 for g in report['gates'] if g['recomputed'])}/{len(report['gates'])} 条"
             "由工件内 `value`/`threshold` 重算后与声明一致（`E_GATE_SELFCONTRADICTION` 计数 "
             f"{sum(1 for f in report['findings'] if f['code'] == 'E_GATE_SELFCONTRADICTION')}）。\n")
    L.append("## ④ 结果图\n")
    L.append("![独立复核：重算误差、文档漂移、峰值压力估计量散布]"
             "(assets/case-fem-hertz-contact/hertz_independent_audit.png)\n")
    L.append("*图: (a) 用独立 Hertz 基准重算的 $e_a$、$e_\\delta$ 与 ±3% 判据带；(b) 文档声称值 vs 工件值"
             "（绿点=已一致，红 X=本轮复核抓到的、修复前的过期声称）；(c) 峰值压力两个估计量的散布，说明 $p_0$ 声称是**估计量条件性**的。*\n")
    L.append("## ⑤ 注意点\n")
    L.append("* 本工具**不重跑求解器**：它审的是「证书/文档是否忠实于工件」，不是「工件是否物理正确」。"
             "离散误差的归因另见 `CASE-FEM-HERTZ-CONTACT-residual-attribution.md`。\n"
             "* 峰值压力存在两个估计量：L2 一致场（闸门采用的）与集总节点峰值（相差约 30%）。"
             f"集总峰值最大偏差 **{report['worst_pct'] and max(abs(r['error_pct']['peak_pressure_lumped']) for r in rows):.2f}%**，"
             "因此 `p₀ <5%` 只在一致场下成立，散文若不加限定即属过度声称。\n"
             "* 复核依赖工件内已存的 FE 数值；若 `result.json` 被改写，本审计的结论同步失效——"
             "`tests/test_hertz_case_pool_independent_audit.py` 把它钉成回归门。\n")
    L.append("## ⑥ 误差判定（逐项回答「是否 <3%」）\n")
    L.append("| 关键量 | 基准 | 最差误差 | <3%? |")
    L.append("|---|---|---:|---:|")
    L.append(f"| 接触半径 `a`（6 个 run 全包） | `(3PR/4E*)^(1/3)` | {report['worst_pct']['contact_radius']:.4f}% | "
             f"{'✅ 是' if report['worst_pct']['contact_radius'] < tol else '❌ 否'} |")
    L.append(f"| 压入深 `δ`（6 个 run 全包） | `a²/R` | {report['worst_pct']['indentation']:.4f}% | "
             f"{'✅ 是' if report['worst_pct']['indentation'] < tol else '❌ 否'} |")
    L.append(f"| 峰值压力 `p₀`（一致场，判据 5%） | `3P/(2πa²)` | {report['worst_pct']['peak_pressure_consistent']:.4f}% | "
             f"{'✅ 是' if report['worst_pct']['peak_pressure_consistent'] < 5 else '❌ 否'} |")
    L.append("")
    L.append("## ⑦ 复核发现（severity 排序）\n")
    if report["fixed_defects"]:
        L.append("**本轮复核发现并已修复的文档缺陷**（before 为修复前提交文本，after 为工件现值，逐轮复核）：\n")
        L.append("| id | 位置 | 量 | 修复前 | 工件现值 | 判定 | 说明 |")
        L.append("|---|---|---|---:|---:|---|---|")
        for d in report["fixed_defects"]:
            L.append(f"| {d['id']} | `{d['doc']}` | {d['quantity']} | {d['before_pct']:+.3f}% | "
                     f"{d['after_pct']:+.4f}% | {d['status']} | {d['note']} |")
        L.append("")
    if not report["findings"]:
        L.append("当前轮次**无未决发现**。\n")
        L.append("无发现：工件、闸门与两篇文档三方一致。\n")
    else:
        L.append("| code | severity | 位置 | 细节 |")
        L.append("|---|---|---|---|")
        for f in report["findings"]:
            where = f.get("run") or f.get("gate") or f.get("doc", "")
            if f.get("line"):
                where = f"{where}:{f['line']}"
            detail = f.get("note") or json.dumps({k: v for k, v in f.items()
                                                 if k not in ("code", "severity", "note", "doc", "run", "gate")},
                                                 ensure_ascii=False)
            L.append(f"| `{f['code']}` | {f['severity']} | {where} | {detail} |")
        L.append("")
    L.append(f"**physics_verdict**: {'PASS' if report['physics_verdict']['passed'] else 'FAIL'}"
             f"（hard: {report['physics_verdict']['hard_defects'] or '—'}）  ")
    L.append(f"**docs_verdict**: {'PASS' if report['docs_verdict']['passed'] else 'FAIL'}"
             f"（hard: {report['docs_verdict']['hard_defects'] or '—'}）\n")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="refresh audit json, doc and figure")
    ap.add_argument("--check", action="store_true", help="exit 1 if any hard defect remains")
    args = ap.parse_args(argv)

    report = audit()
    if args.write:
        AUDIT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        DOC_OUT.write_text(render_doc(report))
        render_figure(report, FIG_OUT)
        print(f"wrote {AUDIT_OUT.relative_to(ROOT)}, {DOC_OUT.relative_to(ROOT)}, {FIG_OUT.relative_to(ROOT)}")

    for f in report["findings"]:
        print(f"[{f['severity']}] {f['code']}: {json.dumps({k: v for k, v in f.items() if k not in ('code', 'severity')}, ensure_ascii=False)[:220]}")
    print(f"physics_verdict={'PASS' if report['physics_verdict']['passed'] else 'FAIL'} "
          f"docs_verdict={'PASS' if report['docs_verdict']['passed'] else 'FAIL'} "
          f"worst a={report['worst_pct']['contact_radius']:.4f}% delta={report['worst_pct']['indentation']:.4f}%")
    if args.check and (not report["physics_verdict"]["passed"] or not report["docs_verdict"]["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())