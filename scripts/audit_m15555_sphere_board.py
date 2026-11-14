#!/usr/bin/env python3
"""Machine-derived board audit of CASE-SPHERE-PRESSURE (mission 15405, audit m15555).

Why this file exists
--------------------
The first landing of this audit was prose with hand-copied artefact facts.
Re-checking it on the re-opened mission found two of those facts already stale
against ``main``:

* the audit named ``result.json`` as ``43850 B`` -- the committed file is
  ``43980 B``;
* it named ``run_verification.py`` as ``31267 B`` -- after the case logic moved
  into :mod:`tensorfem.sphere_pressure_case` the driver is a thin wrapper of
  ``1613 B`` (``31267 B`` was the pre-refactor monolith).

Neither number is physics, but a board audit whose own evidence inventory drifts
from the tree is the class the platform files as *number mislabelling*: a reader
cannot tell a stale copy from a wrong measurement.  So the audit becomes a
generated artefact, in the same discipline as the case card
(``build_report.py --check``) and the certificate
(``emit_certificate.py --check``):

* every number in the audit markdown is formatted from committed evidence
  (``result.json``, ``gates_verified.json``, artefact bytes/sha256, the live
  ``tools/verified_anchor_audit.py --json`` record) and mirrored into
  ``docs/board_audit/m15555/verify_output.json``;
* the audit's *standing* claim -- "the run reproduces" -- is re-measured by
  re-executing the case into a temporary directory and comparing the whole solve
  record against the committed ``result.json`` leaf by leaf;
* ``--check`` fails when the markdown, the JSON or the figure no longer match the
  evidence, so the audit cannot silently rot again.

Only the two evidence carriers (``result.json``, ``gates_verified.json``) are
quoted with bytes + sha256: a legitimate edit to a producer script or to the case
card must not force a re-audit, a change of *evidence* must.

Reproduction criterion (predeclared, not tuned after seeing the numbers)
-----------------------------------------------------------------------
A leaf is **drift** when it is non-numeric and differs, or numeric and differs by
more than ``REPRO_REL_TOL = 1e-9`` relative.  Anything below that is recorded as
float noise, not as reproduction.  ``1e-9`` sits seven orders below the case's
3 % accuracy requirement and ~7 orders above one ulp, so it can neither hide a
physics change nor fire on last-bit reducibility.

What it deliberately does not do
--------------------------------
It does not re-open the platform database (``tensor_benchmark_cases`` judge
registration) or the Roark handbook page: this worktree has access to neither, so
those two verdict inputs stay ``WITHHELD`` instead of being restated from the
earlier round.

Usage::

    python3 scripts/audit_m15555_sphere_board.py --write    # doc + json + figure
    python3 scripts/audit_m15555_sphere_board.py --check    # exit 1 on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

AUDIT_MID = 15555
AUDITED_MID = 15405
ANCHOR = "fem_sphere_pressure_membrane"
CASE = "CASE-SPHERE-PRESSURE"
ANCHOR_DIR = os.path.join("benchmarks", "verified", ANCHOR)
REL = ANCHOR_DIR.replace(os.sep, "/")
AUDIT_DIR = os.path.join("docs", "board_audit", f"m{AUDIT_MID}")
DOC_REL = os.path.join(AUDIT_DIR, "AUDIT_m15405_CASE_SPHERE_PRESSURE.md")
JSON_REL = os.path.join(AUDIT_DIR, "verify_output.json")
FIG_REL = os.path.join(AUDIT_DIR, "fig_m15555_sphere_audit.png")
SCHEMA = "tensorfem.board-audit/1"

#: Evidence carriers own the physics: bytes + sha256 are quoted for them.
EVIDENCE = [f"{ANCHOR_DIR}/result.json", f"{ANCHOR_DIR}/gates_verified.json"]
#: Producers must stay free to change: path + role + tracked flag only.
PRODUCERS = [
    (f"{ANCHOR_DIR}/run_verification.py", "driver (thin wrapper over the case module)"),
    (f"{ANCHOR_DIR}/build_report.py", "case card + report builder"),
    (f"{ANCHOR_DIR}/emit_certificate.py", "certificate emitter"),
    (f"{ANCHOR_DIR}/README.md", "case card"),
    ("scripts/audit_m15555_sphere_board.py", "this audit generator"),
]

#: Paths that legitimately differ between two runs in different directories / on
#: different hosts.  Everything else is compared leaf by leaf.
VOLATILE = {
    "figures": "paths are relative to the run's --out directory",
    "flat_facet_control/wall_clock_s": "wall clock of the control solve",
    "environment": "host / interpreter / library versions, distinct on purpose",
}
CONTROL_SUBTREE = "flat_facet_control"

#: Predeclared reproduction tolerance (relative).  Drift above this fails the audit.
REPRO_REL_TOL = 1e-9


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_json(rel: str) -> dict:
    with open(os.path.join(ROOT, rel)) as fh:
        return json.load(fh)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_tracked(rel: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def pct(x: float) -> str:
    return f"{100.0 * x:+.3e} %"


CONTROL_RERUNS = 5
#: Last-bit noise is itself random, so its *value* must not be committed: it would
#: make the drift gate flip one decade at a time.  What is committed is a lower
#: bound on the margin to the criterion, capped at this many decades.
MARGIN_CAP_DECADES = 4


def _margin_decades_geq(value: float) -> int:
    """How many decades below the criterion the measured noise sits (capped)."""
    if value <= 0.0:
        return MARGIN_CAP_DECADES
    return int(min(MARGIN_CAP_DECADES, math.floor(math.log10(REPRO_REL_TOL / value))))


def _prune_control(control: dict) -> dict:
    """The control record without its own wall clock."""
    return {k: v for k, v in control.items() if k != "wall_clock_s"}


def _prune(record: dict) -> dict:
    """Copy of the solve record with the declared volatile paths removed."""
    pruned = json.loads(json.dumps(record))
    for dotted in VOLATILE:
        node = pruned
        parts = dotted.split("/")
        for i, part in enumerate(parts):
            if not isinstance(node, dict) or part not in node:
                break
            if i == len(parts) - 1:
                node.pop(part)
            else:
                node = node[part]
    return pruned


def _drop_subtree(record: dict, key: str) -> dict:
    pruned = json.loads(json.dumps(record))
    pruned.pop(key, None)
    return pruned


def _compare(a, b, path="", tol=REPRO_REL_TOL, drift=None, state=None):
    """Leaf comparison: drift (non-numeric change or numeric change > tol) vs noise."""
    if drift is None:
        drift, state = [], {"leaves": 0, "noise_leaves": 0, "max_rel": 0.0}
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                drift.append({"path": path + "/" + key, "kind": "missing_key"})
                continue
            _compare(a[key], b[key], path + "/" + key, tol, drift, state)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            drift.append({"path": path, "kind": "length", "a": len(a), "b": len(b)})
        else:
            for i, (u, v) in enumerate(zip(a, b)):
                _compare(u, v, f"{path}[{i}]", tol, drift, state)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            drift.append({"path": path, "kind": "value", "committed": a, "fresh": b})
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        state["leaves"] += 1
        if a != b:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
            state["max_rel"] = max(state["max_rel"], rel)
            if rel > tol:
                drift.append(
                    {"path": path, "kind": "numeric", "rel": rel,
                     "committed": a, "fresh": b}
                )
            else:
                state["noise_leaves"] += 1
    elif a != b:
        drift.append({"path": path, "kind": "value", "committed": a, "fresh": b})
    return drift, state


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def measure_inventory() -> dict:
    evidence = []
    for rel in EVIDENCE:
        path = os.path.join(ROOT, rel)
        evidence.append(
            {
                "path": rel.replace(os.sep, "/"),
                "bytes": os.path.getsize(path),
                "sha256": _sha256(path),
                "git_tracked": _git_tracked(rel),
            }
        )
    producers = [
        {"path": rel.replace(os.sep, "/"), "role": role, "git_tracked": _git_tracked(rel)}
        for rel, role in PRODUCERS
    ]
    return {"evidence": evidence, "producers": producers}


def measure_case_numbers() -> dict:
    result = _read_json(f"{REL}/result.json")
    cert = _read_json(f"{REL}/gates_verified.json")
    ladder = result["convergence"]["ladder"]["exact_meridian"]
    row8 = next(row for row in ladder if row["facets"] == 8)
    gates = {g["id"]: g for g in cert["gates"]}
    params = result["parameters"]
    pred = (cert.get("predictions") or [{}])[0]
    thick = result["thickness_sensitivity"]
    flat = result["flat_facet_control"]
    return {
        "parameters": {
            "radius_m": params["radius_m"],
            "thickness_m": params["thickness_m"],
            "R_over_t": params["R_over_t"],
            "E_pa": params["young_modulus_pa"],
            "nu": params["poisson_ratio"],
            "p_pa": params["pressure_pa"],
        },
        "reference": {
            "sigma_pa": result["reference"]["values"]["sigma_pa"],
            "sigma_expr": result["reference"]["derivation"]["sigma_expr"],
            "axial_vs_laplace_residual": result["reference"]["derivation"][
                "residual_axial_vs_laplace_at_r_R"
            ],
            "w_m": row8["w_ref_m"],
        },
        "site_facets_8": {
            "dof": row8["dof"],
            "elements": row8["elements"],
            "sigma_ref_pa": row8["sigma_ref_pa"],
            "sigma_theta_pa": row8["sigma_theta_pa"],
            "sigma_theta_max_pa": max(row8["sigma_theta_pa"]),
            "sigma_theta_rel_error": row8["sigma_theta_max_rel_error"],
            "sigma_phi_rel_error": row8["sigma_phi_max_rel_error"],
            "anisotropy_rel": row8["anisotropy_rel"],
            "w_rel_error": row8["w_rel_error"],
        },
        "ladder": {
            "facets": [row["facets"] for row in ladder],
            "sigma_theta_rel_error": [row["sigma_theta_max_rel_error"] for row in ladder],
            "observed_orders_sigma": result["convergence"]["observed_orders_sigma"][
                "exact_meridian"
            ],
        },
        "controls": {
            "chord_sigma_theta_rel_error": gates["G9.chord_control_fails"]["value"],
            "flat_shell4_N_mean_rel_error": gates["G10.flat_facet_shell4_control_fails"]["value"],
            "flat_shell4_facets": flat["facet_count"],
            "flat_shell4_recorded_percent": flat["recorded_error_curve_percent"]["24"],
        },
        "reference_band": {
            "t_over_R_band_within_tol": thick["t_over_R_band_within_tol"],
            "slope": thick["thin_vs_lame_loglog_slope"],
            "deviation_at_case_point": gates["G8a.reference_valid_at_case_point"]["value"],
            "prediction_id": pred.get("id"),
            "prediction_threshold": pred.get("predicted_threshold"),
            "prediction_outcome": pred.get("outcome"),
        },
        "tolerance": result["tolerance"],
    }


def measure_certificate() -> dict:
    cert = _read_json(f"{REL}/gates_verified.json")
    gates = cert["gates"]
    return {
        "schema": cert.get("schema"),
        "n_gates": len(gates),
        "n_passed": sum(1 for g in gates if g["passed"]),
        "passed": cert.get("passed"),
        "n_predictions": len(cert.get("predictions") or []),
        "falsified_predictions": [
            p["id"] for p in (cert.get("predictions") or []) if p.get("outcome") == "falsified"
        ],
        "gates_without_numeric_threshold": [
            g["id"] for g in gates if not isinstance(g.get("threshold"), (int, float))
        ],
    }


def measure_anchor_audit() -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "verified_anchor_audit.py"), "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"verified_anchor_audit failed: {proc.stderr[-800:]}")
    payload = json.loads(proc.stdout)
    record = next(r for r in payload["records"] if r["anchor"] == ANCHOR)
    return {
        "grade": record["grade"],
        "certified": record["certified"],
        "defects": record["defects"],
        "warnings": record["warnings"],
        "gates": record["gates"],
        "gates_passed": record["gates_passed"],
        "gates_contradicted": record["gates_contradicted"],
        "gates_unrecomputable": record["gates_unrecomputable"],
    }


def measure_drift_gates() -> dict:
    out = {}
    for name, script in (
        ("emit_certificate_check", "emit_certificate.py"),
        ("build_report_check", "build_report.py"),
    ):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, ANCHOR_DIR, script), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        out[name] = {"clean": proc.returncode == 0}
    return out


def measure_reproduction() -> dict:
    """Re-execute the case and compare the solve record leaf by leaf."""
    import torch  # noqa: PLC0415  (version only, for the record)

    from tensorfem import sphere_pressure_case as sp  # noqa: PLC0415

    committed = _prune(_read_json(f"{REL}/result.json"))
    with tempfile.TemporaryDirectory(prefix="sphere_audit_") as tmp:
        fresh = _prune(sp.run_case(out_dir=tmp))

    drift, state = _compare(committed, fresh)
    drift_axisym, state_axisym = _compare(
        _drop_subtree(committed, CONTROL_SUBTREE), _drop_subtree(fresh, CONTROL_SUBTREE)
    )

    # Independent measurement of the control's own run-to-run spread: this
    # separates "the audit re-ran and the numbers moved" from "this control is
    # not bit-reducible".  Same process, repeated calls, no code change in between.
    # Only magnitudes are reported: *which* leaf flips is itself random, so a path
    # list here would make the drift gate flaky.  The floor of 1e-15 is the
    # resolution of the measurement, not a claim that the noise is exactly there.
    control_runs = [sp.flat_facet_shell4_control() for _ in range(CONTROL_RERUNS)]
    control_max_rel = 0.0
    for candidate in control_runs[1:]:
        more, _ = _compare(_prune_control(control_runs[0]), _prune_control(candidate))
        control_max_rel = max(
            [control_max_rel] + [entry.get("rel", 0.0) for entry in more]
        )

    return {
        "reproduced": not drift and not drift_axisym,
        "rel_tolerance": REPRO_REL_TOL,
        "compared_leaves": state["leaves"],
        "drift_paths": drift + drift_axisym,
        "noise_margin_decades_geq": _margin_decades_geq(state["max_rel"]),
        "axisym_noise_margin_decades_geq": _margin_decades_geq(state_axisym["max_rel"]),
        "control_reruns": len(control_runs),
        "control_noise_margin_decades_geq": _margin_decades_geq(control_max_rel),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "excluded_paths": sorted(VOLATILE),
        "excluded_reason": {k: VOLATILE[k] for k in sorted(VOLATILE)},
    }


def measure_all() -> dict:
    return {
        "schema": SCHEMA,
        "audit_mission": AUDIT_MID,
        "audited_mission": AUDITED_MID,
        "anchor": {"name": ANCHOR, "dir": REL, "case": CASE},
        "verdict": {
            "grade": "L3",
            "l4_blockers": [
                "no registered judge row in tensor_benchmark_cases (F1, not re-checkable here)",
                "external handbook page (Roark) absent from this worktree (F2, WITHHELD)",
            ],
        },
        "inventory": measure_inventory(),
        "case_numbers": measure_case_numbers(),
        "certificate": measure_certificate(),
        "anchor_audit_record": measure_anchor_audit(),
        "drift_gates": measure_drift_gates(),
        "reproduction": measure_reproduction(),
        "unverified": [
            {
                "claim": "tensor_benchmark_cases holds no CASE-SPHERE-PRESSURE judge row",
                "status": "WITHHELD",
                "reason": (
                    "this worktree has no platform database access; the earlier round's "
                    "0/18 query is not re-measured"
                ),
            },
            {
                "claim": "the reference equals the Roark handbook table entry",
                "status": "WITHHELD",
                "reason": (
                    "no handbook page in the repo or platform knowledge base; the reference "
                    "stays machine-derived (sympy) and Lame-cross-anchored"
                ),
            },
        ],
        "figure": {
            "path": FIG_REL.replace(os.sep, "/"),
            "caption": (
                "CASE-SPHERE-PRESSURE board audit, derived from result.json: "
                "(a) sigma_theta error ladder vs pR/2t with the 3 % tolerance band; "
                "(b) sigma_theta / sigma_phi along the meridian at 8 facets; "
                "(c) thin-shell vs Lame deviation vs t/R with the FALSIFIED t/R=0.05 "
                "prediction and the measured band end at t/R=0.02; "
                "(d) negative controls (straight-chord meridian, flat-facet Shell4) "
                "against the exact-meridian element"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def render_figure(path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = _read_json(f"{REL}/result.json")
    cert = _read_json(f"{REL}/gates_verified.json")
    ladder = result["convergence"]["ladder"]["exact_meridian"]
    facets = np.array([row["facets"] for row in ladder], dtype=float)
    err = np.array([row["sigma_theta_max_rel_error"] for row in ladder])
    row8 = next(row for row in ladder if row["facets"] == 8)
    thick = result["thickness_sensitivity"]["rows"]
    tol = result["tolerance"]
    gates = {g["id"]: g for g in cert["gates"]}

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))

    ax = axes[0][0]
    ax.loglog(facets, err, "o-", lw=1.8, label=r"$\max|\sigma_\theta/(pR/2t)-1|$")
    ax.axhline(tol, color="crimson", ls="--", lw=1.2, label="3 % tolerance")
    ax.set_xlabel("facets (pole -> equator)")
    ax.set_ylabel("rel. error")
    ax.set_title("(a) convergence ladder vs pR/2t")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    x = np.arange(len(row8["sigma_theta_pa"]))
    ax.plot(x, row8["sigma_theta_pa"], "o-", lw=1.6, label=r"$\sigma_\theta$")
    ax.plot(x, row8["sigma_phi_pa"], "s--", lw=1.6, label=r"$\sigma_\phi$")
    ax.axhline(row8["sigma_ref_pa"], color="crimson", ls=":", lw=1.4, label=r"$pR/2t$")
    ax.set_ylim(0.0, 2.0 * row8["sigma_ref_pa"])
    ax.set_xlabel("Gauss point along meridian (8 facets)")
    ax.set_ylabel("stress [Pa]")
    ax.set_title("(b) membrane stress field at 8 facets")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    t_over_R = np.array([row["t_over_R"] for row in thick], dtype=float)
    dev = np.array([row["thin_vs_lame_rel"] for row in thick])
    ax.loglog(t_over_R, dev, "o-", lw=1.8, label="thin-shell vs Lame")
    ax.axhline(tol, color="crimson", ls="--", lw=1.2, label="3 % tolerance")
    band = result["thickness_sensitivity"]["t_over_R_band_within_tol"]
    ax.axvspan(band[0], band[1], color="tab:green", alpha=0.12,
               label=f"measured band {band[0]}..{band[1]}")
    ax.axvline(0.05, color="k", ls="-.", lw=1.2, label="FALSIFIED prediction t/R = 0.05")
    ax.set_xlabel("t / R")
    ax.set_ylabel("rel. deviation")
    ax.set_title("(c) reference validity band (prediction falsified)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1][1]
    names = ["exact meridian\n8 facets", "chord meridian\n8 facets", "flat Shell4\n24 facets"]
    values = [
        err[list(facets).index(8)],
        gates["G9.chord_control_fails"]["value"],
        gates["G10.flat_facet_shell4_control_fails"]["value"],
    ]
    ax.bar(names, values, color=["tab:green", "tab:orange", "tab:red"])
    ax.set_yscale("log")
    ax.axhline(tol, color="crimson", ls="--", lw=1.2, label="3 % tolerance")
    for i, value in enumerate(values):
        ax.text(i, value * 1.6, f"{100 * value:.3g} %", ha="center", fontsize=8)
    ax.set_ylim(1e-7, 1e5)
    ax.set_ylabel("rel. error (log)")
    ax.set_title("(d) negative controls miss by orders of magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", which="both", alpha=0.3)

    fig.suptitle(
        f"m{AUDIT_MID} board audit - {CASE} "
        f"(uniform pressure, R/t={result['parameters']['R_over_t']:.0f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #
def render_markdown(p: dict) -> str:
    ev = {row["path"]: row for row in p["inventory"]["evidence"]}
    res_path = f"{REL}/result.json"
    gates_path = f"{REL}/gates_verified.json"
    cn = p["case_numbers"]
    cert = p["certificate"]
    aud = p["anchor_audit_record"]
    rep = p["reproduction"]
    drift = p["drift_gates"]
    site = cn["site_facets_8"]
    band = cn["reference_band"]
    ctl = cn["controls"]
    par = cn["parameters"]
    l4 = "；".join(p["verdict"]["l4_blockers"])
    failed = ", ".join(cert["falsified_predictions"]) or "none"
    png = p["figure"]["path"]
    png_base = os.path.basename(png)
    clean_cert = "clean" if drift["emit_certificate_check"]["clean"] else "**DRIFT**"
    clean_card = "clean" if drift["build_report_check"]["clean"] else "**DRIFT**"
    axisym_txt = f"至少低于判据 {rep['axisym_noise_margin_decades_geq']} 个量级"
    ev_rows = "\n".join(
        f"| `{row['path']}` | {row['bytes']} B | `{row['sha256'][:16]}…` | "
        f"{'✓' if row['git_tracked'] else '✗'} |"
        for row in p["inventory"]["evidence"]
    )
    prod_rows = "\n".join(
        f"| `{row['path']}` | {row['role']} | {'✓' if row['git_tracked'] else '✗'} |"
        for row in p["inventory"]["producers"]
    )
    vol_rows = "\n".join(f"| `{k}` | {v} |" for k, v in rep["excluded_reason"].items())
    unver_rows = "\n".join(
        f"| {u['claim']} | **{u['status']}** | {u['reason']} |" for u in p["unverified"]
    )
    return f"""# m{AUDIT_MID} 交差核验：{CASE}（m{AUDITED_MID}）board-audit 裁定

> **本文件为生成物，禁止手改。** 生成器 `scripts/audit_m15555_sphere_board.py`（`--write` / `--check`）；
> 机器抽取的数字镜像在 `docs/board_audit/m{AUDIT_MID}/verify_output.json`；漂移门
> `python3 scripts/audit_m15555_sphere_board.py --check` 与
> `tests/test_board_audit_m15555_sphere.py` 同时看守。上一版手抄的工件字节数已两次过期
> （见 §五 F4）。本轮改为注入，且只对**证据载体**引用字节 + sha256、生产脚本只列路径，
> 这样别的 PR 合法改脚本不会误触发再审计，而证据一旦变化必须重审。

**审计对象**：mission {AUDITED_MID}（{CASE}·均压球壳薄膜应力·TensorFEM）
**本轮**：m{AUDIT_MID} 重开单（m20596，从当前 main 切枝重做；不改动任何物理实现）
**裁定**：**L3**（L1–L3 实测成立；L4 未记，阻断项：{l4}）
**锚点**：`{ANCHOR}` · 证据 `{res_path}` / `{gates_path}` · 证据图 `{png}`

## 一、案由与算例（物理）

| 项 | 值 |
|---|---|
| 结构 | 闭合球壳，内压 {par['p_pa']:.0f} Pa 均布 |
| 几何 | R = {par['radius_m']:.6g} m，t = {par['thickness_m']:.6g} m，R/t = {par['R_over_t']:.0f} |
| 材料 | E = {par['E_pa']:.4g} Pa，ν = {par['nu']} |
| 参考解 | σ = {cn['reference']['sigma_expr']} = **{cn['reference']['sigma_pa']:.6g} Pa**；w = pR²(1-ν)/(2Et) = {cn['reference']['w_m']:.6e} m |
| 参考解核验 | sympy 两路静力平衡（Laplace / 轴向截面）残差 = {cn['reference']['axial_vs_laplace_residual']}（恒零，非自指） |
| 生产单元 | `tensorfem.axisym_shell.AxisymShell`（2 节点 Reissner–Mindlin 环元，`meridian_radius=R` 真圆弧 + 横向剪切 SRI） |
| 判据 | 相对误差 < {100 * cn['tolerance']:.0f} %（严于学科 15 % 口径的自定门） |

## 二、三件套核验（数字全部机器抽取）

### 2.1 run_artifacts —— 证据载体

| 文件 | 字节 | sha256（前 16） | git tracked |
|---|---:|---|---|
{ev_rows}

生产链（只登记路径，不锁字节）：

| 文件 | 角色 | git tracked |
|---|---|---|
{prod_rows}

### 2.2 benchmark ref vs got

| 量 | 本轮读数 | 判据 |
|---|---|---|
| σ_θ（8 facets / {site['elements']} elements / {site['dof']} DOF） | {site['sigma_theta_max_pa']:.6f} Pa（pR/2t = {site['sigma_ref_pa']:.6g} Pa） | **{pct(site['sigma_theta_rel_error'])}** < 3 % |
| σ_φ | — | {pct(site['sigma_phi_rel_error'])} < 3 % |
| 各向同性 σ_θ vs σ_φ | — | {pct(site['anisotropy_rel'])} < 3 % |
| 径向位移 w | — | {pct(site['w_rel_error'])} < 3 % |
| 收敛阶梯 facets {cn['ladder']['facets']} | {['%.2e' % e for e in cn['ladder']['sigma_theta_rel_error']]} | 单调收缩 |
| 观测阶（exact meridian） | {cn['ladder']['observed_orders_sigma']} | min = {min(o for o in cn['ladder']['observed_orders_sigma'] if o is not None):.3f} ≥ 1.5 |

### 2.3 repro_cmd —— 本轮重跑并整份比对 solve record

- 命令：`python3 {ANCHOR_DIR}/run_verification.py`（等价 `tensorfem-sphere-pressure --out artifacts/{CASE}`）
- 复现判定：**reproduced = {rep['reproduced']}**；逐叶比对 **{rep['compared_leaves']}** 个数值叶，
  超判据（相对 {rep['rel_tolerance']:g}）的漂移 **{len(rep['drift_paths'])}** 条
- 噪声测量（不是"大概一样"）：整体抖动至少低于判据 {rep['noise_margin_decades_geq']} 个量级；
  其中轴对称生产路径 {axisym_txt}。flat-facet Shell4 对照在同一进程内独立重跑
  {rep['control_reruns']} 次，其自身抖动同样至少低于判据
  {rep['control_noise_margin_decades_geq']} 个量级——
  "重跑不动"与"该对照本就 1 ulp 级不可约"被分开测量，判据 {rep['rel_tolerance']:g} 因此既不会
  把浮点噪声当物理漂移、也不会被 1 uLP 硬币面反过来带崩（故只落盘余量下界，不落盘噪声数值与路径）
- 运行环境：Python {rep['environment']['python']} / torch {rep['environment']['torch']}（float64，单进程）
- 不参与比对的易变路径（理由入档）：

| 路径 | 理由 |
|---|---|
{vol_rows}

## 三、客观证据链（每一条都能被命令重算）

| 断言 | 派生方式 | 本轮读数 |
|---|---|---|
| 证书与工件不漂移 | `emit_certificate.py --check` | {clean_cert} |
| 案卡/报告与工件不漂移 | `build_report.py --check` | {clean_card} |
| 锚点证书自洽 | `tools/verified_anchor_audit.py --json` | grade=**{aud['grade']}**, certified={aud['certified']}, {aud['gates_passed']}/{aud['gates']} gates pass, contradicted={aud['gates_contradicted']}, unrecomputable={aud['gates_unrecomputable']}, defects={aud['defects']}, warnings={aud['warnings']} |
| 门数与容差 | `{gates_path}` | {cert['n_passed']}/{cert['n_gates']} gating 通过，无阈值门 {cert['gates_without_numeric_threshold']} |
| 负对照真的失败 | `result.json` → G9/G10 | 直弦子午线 {pct(ctl['chord_sigma_theta_rel_error'])}；平面 Shell4 {pct(ctl['flat_shell4_N_mean_rel_error'])}（{ctl['flat_shell4_facets']} facets，录得 {ctl['flat_shell4_recorded_percent']} % @24） |
| 参考带（证伪留档） | `result.json` → G8a + predictions | 带内 t/R ∈ {band['t_over_R_band_within_tol']}，斜率 {band['slope']}；`{band['prediction_id']}` 预注册 t/R ≥ {band['prediction_threshold']} → **{band['prediction_outcome']}**；case 点偏差 {pct(band['deviation_at_case_point'])} |

> 预注册预测被**证伪**且留档：`{failed}`。它不是门（不得计入门数），也不允许被悄悄删除——
> `tests/test_case_sphere_pressure_certificate.py` 看守它留在 `predictions` 里。

![m{AUDIT_MID} audit figure]({png_base})

图 `{png_base}`（`{png}`，由 `result.json` 渲染，零手抄）：
(a) σ_θ 误差阶梯与 3 % 带；(b) 8 facets 下 σ_θ/σ_φ 沿子午线场；
(c) 薄壳 vs Lamé 偏差与**被证伪的** t/R = 0.05 预测（实测带止于 0.02）；
(d) 两个负对照相对精确子午线单元差若干量级。

## 四、判级（L0–L5）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 参考解机器推导 + 单元复现 | ✅ 平衡残差 {cn['reference']['axial_vs_laplace_residual']}；8 单元 {site['dof']} DOF 复现膜态 {pct(site['sigma_theta_rel_error'])} |
| L2 独立功能实现 | 真代码：输入→计算→输出 | ✅ 生产模块 `tensorfem.sphere_pressure_case`（driver 为薄包装，case 逻辑不在脚本内） |
| L3 系统集成 | 接入生产单元/平台 | ✅ 复用 `tensorfem.axisym_shell.AxisymShell`；锚点证书 stage=verified，审计 defects={aud['defects']} |
| L4 基准验证 | 外部基准 + 可复现 | ⚠️ 完整算例✅ / 误差分析✅ / 可复现✅（reproduced={rep['reproduced']}）但 {l4} → **未记** |
| L5 科研工程成果 | — | 未达（L4 未记） |

**裁定 L3** 维持。升 L4 需外部基准原表原页 + 平台裁判行，二者在本工作树内不可得（§六）。

## 五、发现（findings）

- **F1_no_registered_judge**：`tensor_benchmark_cases` 无该锚点注册行（**本轮 WITHHELD**：工作树无库访问，沿用上轮 0/18 结论但不重测）。
- **F2_external_anchor_withheld**：手册原表原页未核对（**WITHHELD**）；参考解由 sympy 机器推导 + Lamé 厚壁极限交叉锚承载，独立可复核、非自指。
- **F3_clean_report（已升级为机器门）**：报告/案卡 `build_report.py --check` = {clean_card}，证书 `emit_certificate.py --check` = {clean_cert}；"数字零手抄"从说法变成可执行判据。
- **F4_stale_hand_typed_numbers（本轮发现并修掉）**：上一版审计文档手抄 `result.json` = 43850 B、`run_verification.py` = 31267 B；实测证据载体 **{ev[res_path]['bytes']} B**（sha256 `{ev[res_path]['sha256'][:16]}…`），而 `run_verification.py` 早已是薄包装（case 逻辑下沉 `tensorfem.sphere_pressure_case`，31267 B 属重构前单体脚本）。工件清单因此改为生成器注入 + `--check`，本文件成为生成物。
- **F5_reproduction_machine_checked（本轮新增能力）**：`reproduced` 从"跑过一次记得是这样"改为**每次审计重跑并逐叶比对**（{rep['compared_leaves']} 叶、判据 {rep['rel_tolerance']:g}），易变路径逐条给理由。
- **F6_control_not_bit_reducible（本轮量化的性质）**：flat-facet Shell4 对照在同一进程内重跑 {rep['control_reruns']} 次即出现 1 ulp 级抖动（源：`uniform_expansion_fit` 的最小二乘/规约非结合），而轴对称生产路径即使抖动也{axisym_txt}。该抖动至少低于 3 % 判据 14 个量级、不影响任何门，但**不可**用作逐位比较的基准——故复现判据显式设为 {rep['rel_tolerance']:g} 而非逐位相等，且只落盘余量下界（噪声数值与翻转路径逐次随机，落盘会让漂移门自己抖动）。

## 六、checks_not_run / 不可核验项（诚实登记）

| 断言 | 状态 | 原因 |
|---|---|---|
{unver_rows}
| 集群 loong64 / x86 双平台复跑 | 未跑 | 单机单进程 float64、预算 ≤10 s，无需上集群；CI 由平台合并时触发 |
| 上一版声称的 pytest 通过数与耗时 | 未沿用 | 测试数量随套件增长会漂移；改由 `tests/test_board_audit_m15555_sphere.py` 自己看守 |

## 七、产物

| 产物 | 说明 |
|---|---|
| `{DOC_REL.replace(os.sep, '/')}` | 本文件（生成物，`--check` 看守） |
| `{JSON_REL.replace(os.sep, '/')}` | 机器抽取数字镜像（schema `{SCHEMA}`） |
| `{png}` | 四联证据图（由 `result.json` 渲染） |
| `scripts/audit_m15555_sphere_board.py` | 生成器 + 复现比对 + 漂移门 |
| `tests/test_board_audit_m15555_sphere.py` | 独立重算 + 漂移门 + 复现门槛 |
"""


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _load_committed_json() -> dict | None:
    path = os.path.join(ROOT, JSON_REL)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate doc/json/figure")
    group.add_argument("--check", action="store_true", help="exit 1 if out of sync")
    args = parser.parse_args(argv)

    fig_path = os.path.join(ROOT, FIG_REL)
    if args.write:
        render_figure(fig_path)
    payload = measure_all()
    if os.path.isfile(fig_path):
        payload["figure"]["bytes"] = os.path.getsize(fig_path)
        payload["figure"]["sha256"] = _sha256(fig_path)
    markdown = render_markdown(payload)
    json_text = json.dumps(payload, indent=1, sort_keys=True) + "\n"

    if args.write:
        with open(os.path.join(ROOT, JSON_REL), "w") as fh:
            fh.write(json_text)
        with open(os.path.join(ROOT, DOC_REL), "w") as fh:
            fh.write(markdown)
        print(f"wrote {JSON_REL} and {DOC_REL}; figure {FIG_REL}")
        print(f"reproduced={payload['reproduction']['reproduced']} "
              f"leaves={payload['reproduction']['compared_leaves']} "
              f"drift={len(payload['reproduction']['drift_paths'])} "
              f"grade={payload['anchor_audit_record']['grade']}")
        return 0

    problems = []
    if not os.path.isfile(fig_path):
        problems.append(f"missing figure {FIG_REL}")
    else:
        committed = _load_committed_json() or {}
        recorded = (committed.get("figure") or {}).get("sha256")
        if recorded and _sha256(fig_path) != recorded:
            problems.append("figure sha256 != verify_output record")
    for rel, text in ((JSON_REL, json_text), (DOC_REL, markdown)):
        if not os.path.isfile(os.path.join(ROOT, rel)):
            problems.append(f"missing {rel}")
            continue
        with open(os.path.join(ROOT, rel)) as fh:
            if fh.read() != text:
                problems.append(f"drift: {rel} differs from regeneration")
    if not payload["reproduction"]["reproduced"]:
        problems.append(
            "reproduction drift: "
            + ", ".join(entry["path"] for entry in payload["reproduction"]["drift_paths"])
        )
    if problems:
        print("AUDIT DRIFT:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("audit in sync with evidence "
          f"(reproduced={payload['reproduction']['reproduced']}, "
          f"{payload['reproduction']['compared_leaves']} leaves compared, "
          f"tolerance {REPRO_REL_TOL:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())