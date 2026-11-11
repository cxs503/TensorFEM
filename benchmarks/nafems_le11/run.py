"""CASE-NAFEMS-LE11 — spheroidal shell under internal pressure: canonical entry.

One-command reproduction of the case-pool example (anchor ``CASE-NAFEMS-LE11``).
This module is the **entry point** of the case: it

1. runs the TensorFEM axisymmetric shell-of-revolution route on the closed
   spheroid (reusing :mod:`tensorfem.ellipsoid_shell` and the driver helpers of
   ``benchmarks/nafems_le11_ellipsoid.py`` — no physics is duplicated here),
2. writes a *compact, committed* run artifact
   ``benchmarks/nafems_le11/results.json`` (ref vs got, per-item errors, 5 %
   verdict, convergence, three sensitivity studies, provenance), and
3. renders ``benchmarks/nafems_le11/report.md`` from that JSON (no number is
   typed by hand).

Usage::

    PYTHONPATH=src python3 benchmarks/nafems_le11/run.py            # full case
    PYTHONPATH=src python3 benchmarks/nafems_le11/run.py --quick    # coarse (smoke)
    PYTHONPATH=src python3 benchmarks/nafems_le11/run.py --out /tmp/x.json

Exit code 0 iff every gated quantity at the report mesh is within tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DRIVER = ROOT / "benchmarks" / "nafems_le11_ellipsoid.py"
IMPL_FILES = [
    "src/tensorfem/ellipsoid_shell.py",
    "src/tensorfem/axisym_shell.py",
    "src/tensorfem/integration_callers.py",
    "benchmarks/nafems_le11_ellipsoid.py",
    "benchmarks/nafems_le11/run.py",
]
DEFAULT_OUT = HERE / "results.json"
DEFAULT_MD = HERE / "report.md"
TOL = 0.05  # case tolerance (<5 %), pre-declared by the task card


def _load_driver():
    """Import the case driver by path (benchmarks/ is not a package)."""
    spec = importlib.util.spec_from_file_location("nafems_le11_ellipsoid", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance() -> dict:
    import torch

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                    # pragma: no cover
        git = ""
    return {
        "repo": "TensorFEM",
        "case": "CASE-NAFEMS-LE11",
        "git_rev": git,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "impl_sha256": {rel: _sha256(ROOT / rel) for rel in IMPL_FILES
                        if (ROOT / rel).is_file()},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


GATED = (
    ("N_phi_pole", "N_pole_fe", "N_pole_ref", "N_pole_err", "meridional force at pole [p a]"),
    ("N_theta_pole", "N_theta_pole_fe", "N_theta_pole_ref", "N_theta_pole_err",
     "hoop force at pole [p a]"),
    ("N_phi_equator", "N_phi_equator_fe", "N_phi_equator_ref", "N_phi_equator_err",
     "meridional force at equator [p a]"),
    ("N_theta_equator", "N_theta_equator_fe", "N_theta_equator_ref",
     "N_theta_equator_err", "hoop force at equator [p a]"),
    ("w_equator", "w_equator_fe", "w_equator_ref", "w_equator_err",
     "equatorial normal displacement [p a^2 / (E t)]"),
)


def build_case(*, report_facets: int = 256, sweep=(4, 8, 16, 32, 64, 128, 256, 512),
               with_sensitivity: bool = True) -> dict:
    """Run the case and return the compact results payload."""
    t0 = time.time()
    d = _load_driver()
    sweep = [f for f in sweep if f <= report_facets] + [report_facets]
    rows = [d._summary(d.run_case(facets=f)) for f in dict.fromkeys(sweep)]
    at_report = rows[-1]

    ref_rows = []
    for name, fe_key, ref_key, err_key, quantity in GATED:
        err = abs(at_report[fe_key] - at_report[ref_key]) / abs(at_report[ref_key])
        ref_rows.append({
            "quantity": name, "description": quantity,
            "ref": at_report[ref_key], "got": at_report[fe_key],
            "rel_error": err, "tolerance": TOL, "passed": err < TOL,
        })

    payload: dict = {
        "case": "CASE-NAFEMS-LE11",
        "title": "Spheroidal (ellipsoid-of-revolution) shell under uniform internal pressure",
        "solver": "TensorFEM tensorfem.axisym_shell (2-node ring, Reissner-Mindlin, SRI)",
        "tolerance": TOL,
        "report_mesh": {"facets": report_facets, "dof": at_report["dof"]},
        "parameters": {
            "a_equatorial": d.A, "b_polar": d.B, "thickness": d.THICKNESS,
            "young": d.YOUNG, "poisson": d.POISSON, "pressure": d.PRESSURE,
            "aspect_ratio_b_over_a": d.B / d.A,
            "slenderness_a_over_t": d.A / d.THICKNESS,
            "p*a^2/(E*t^2)": d.PRESSURE * d.A ** 2 / (d.YOUNG * d.THICKNESS ** 2),
        },
        "reference": {
            "N_phi": "p*r/(2 sin phi) = p*R2/2  [cap axial statics, exact]",
            "N_theta": "p*R2*(1 - R2/(2*R1))  [normal equilibrium]",
            "displacement": "Rayleigh-Ritz membrane potential minimisation "
                            "(independent discretisation; reproduces the closed-form forces)",
            "provenance_note": (
                "The NAFEMS LE11 manual text/table was NOT retrievable from this sandbox "
                "(no egress to nafems.org documents; search-engine caches returned no table). "
                "The ref column is therefore the closed-form thin-shell membrane solution "
                "(exact statics, recomputed in-process) plus an independent Rayleigh-Ritz "
                "displacement solve -- NOT transcribed manual values. See docs §6 item 4."),
        },
        "ref_vs_got": ref_rows,
        "all_passed": all(r["passed"] for r in ref_rows),
        "convergence": rows,
        "reference_selfcheck": {},
        "provenance": _provenance(),
        "wall_seconds": None,
    }
    if with_sensitivity:
        payload["reference_selfcheck"] = {
            "rr_order": d.ellipsoid_pressure(facets=32)["membrane_shape"]["order"],
            "N_phi_max_rel_error": d.ellipsoid_pressure(facets=32)["membrane_shape"][
                "N_selfcheck_phi"],
            "N_theta_max_rel_error": d.ellipsoid_pressure(facets=32)["membrane_shape"][
                "N_selfcheck_theta"],
            "sphere_limit_w": d._sphere_limit_w(),
        }
        payload["sensitivity"] = {
            "aspect_ratio": [d._aspect_point(r) for r in (0.3, 0.5, 0.75, 1.0, 2.0, 3.0)],
            "thickness": [d._thickness_point(t) for t in (0.02, 0.01, 0.005, 0.002)],
            "grading": [dict(grading=g, **_pick(d._summary(d.run_case(facets=f, grading=g))))
                        for g in (0.5, 0.75, 1.0, 1.5) for f in (32, 64)],
            "shear_integration": [dict(integr=si, **_pick(d._summary(
                d.run_case(facets=128, shear_integration=si))))
                for si in ("reduced", "full")],
        }
    payload["wall_seconds"] = round(time.time() - t0, 2)
    return payload


def _pick(summary: dict) -> dict:
    keys = ("facets", "dof", "N_pole_err", "N_theta_equator_err", "w_equator_err",
            "err_phi_max_scaled", "err_theta_max_scaled")
    return {k: summary[k] for k in keys}


def render_markdown(payload: dict) -> str:
    """Render report.md from the JSON payload (programmatic, zero hand-typed numbers)."""
    p = payload["parameters"]
    lines = [
        "# CASE-NAFEMS-LE11 — spheroidal shell under internal pressure (TensorFEM)",
        "",
        "> Generated by `benchmarks/nafems_le11/run.py` from `results.json`; "
        "every number below is injected programmatically.",
        "",
        "## Parameters (dimensionless)",
        "",
        "| quantity | symbol | value |", "|---|---|---|",
    ]
    for k, v in p.items():
        lines.append(f"| {k} | | {v} |")
    lines += ["", f"## ref vs got @ {payload['report_mesh']['facets']} facets "
                  f"({payload['report_mesh']['dof']} dof), tolerance <{100*payload['tolerance']:.0f}%",
              "", "| quantity | ref | got | rel. error | pass |", "|---|---|---|---|---|"]
    for r in payload["ref_vs_got"]:
        lines.append("| %s | %.6f | %.6f | %.3f%% | %s |"
                     % (r["quantity"], r["ref"], r["got"], 100 * r["rel_error"],
                        "✓" if r["passed"] else "✗"))
    lines += ["", "## Convergence", "",
              "| facets | dof | N_phi@pole err | N_theta@equator err | w@equator err "
              "| max|dN_phi|/N_scale | max|dN_theta|/N_scale |",
              "|---|---|---|---|---|---|---|"]
    for s in payload["convergence"]:
        lines.append("| %d | %d | %.3f%% | %.3f%% | %.3f%% | %.3f%% | %.3f%% |"
                     % (s["facets"], s["dof"], 100 * s["N_pole_err"],
                        100 * s["N_theta_equator_err"], 100 * s["w_equator_err"],
                        100 * s["err_phi_max_scaled"], 100 * s["err_theta_max_scaled"]))
    if "sensitivity" in payload:
        s = payload["sensitivity"]
        lines += ["", "## Sensitivity 1 — aspect ratio b/a (128 facets)", "",
                  "| b/a | N_theta@equator ref | hoop sign change | N_theta@equator err | w err |",
                  "|---|---|---|---|---|"]
        for r in s["aspect_ratio"]:
            lines.append("| %.2f | %.4f | %s | %.3f%% | %.3f%% |"
                         % (r["b_over_a"], r["N_theta_equator_ref"],
                            "yes" if r["hoop_sign_change"] else "no",
                            100 * r["N_theta_equator_err"], 100 * r["w_equator_err"]))
        lines += ["", "## Sensitivity 2 — wall thickness a/t (128 facets)", "",
                  "| a/t | N_theta@equator err | w err | bending/membrane |",
                  "|---|---|---|---|"]
        for r in s["thickness"]:
            lines.append("| %.0f | %.3f%% | %.3f%% | %.4f |"
                         % (r["slenderness"], 100 * r["N_theta_equator_err"],
                            100 * r["w_equator_err"], r["bending_to_membrane"]))
        lines += ["", "## Sensitivity 3 — grading & shear integration", "",
                  "| grading | facets | max|dN_phi|/N_scale | N_theta@equator err |",
                  "|---|---|---|---|"]
        for r in s["grading"]:
            lines.append("| %.2f | %d | %.3f%% | %.3f%% |"
                         % (r["grading"], r["facets"], 100 * r["err_phi_max_scaled"],
                            100 * r["N_theta_equator_err"]))
        lines += ["", "| shear integration | facets | max|dN_phi|/N_scale |",
                  "|---|---|---|"]
        for r in s["shear_integration"]:
            lines.append("| %s | %d | %.3f%% |"
                         % (r["integr"], r["facets"], 100 * r["err_phi_max_scaled"]))
    prov = payload["provenance"]
    lines += ["", "## Provenance", "",
              f"- git rev `{prov['git_rev']}`",
              f"- python {prov['python']} / torch {prov['torch']} / {prov['machine']}",
              f"- generated {prov['generated_utc']}, wall {payload['wall_seconds']} s",
              "", "| implementation file | sha256 |", "|---|---|"]
    for k, v in prov["impl_sha256"].items():
        lines.append(f"| `{k}` | `{v[:16]}…` |")
    lines += ["", f"**Verdict**: {'PASS' if payload['all_passed'] else 'FAIL'} "
                  f"(all gated quantities <{100 * payload['tolerance']:.0f} % at the report mesh)", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    ap.add_argument("--quick", action="store_true",
                    help="coarse smoke run (no sensitivity, 128-facet report mesh)")
    args = ap.parse_args(argv)

    if args.quick:
        payload = build_case(report_facets=128, sweep=(4, 8, 16, 32, 64, 128),
                             with_sensitivity=False)
    else:
        payload = build_case()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(payload) + "\n")

    for r in payload["ref_vs_got"]:
        print("%-16s ref=%.6f got=%.6f err=%6.3f%% %s"
              % (r["quantity"], r["ref"], r["got"], 100 * r["rel_error"],
                 "PASS" if r["passed"] else "FAIL"))
    print("verdict=%s  wall=%.1fs  artifact=%s"
          % ("PASS" if payload["all_passed"] else "FAIL", payload["wall_seconds"],
             args.out.relative_to(ROOT)))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())