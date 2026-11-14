#!/usr/bin/env python3
"""Scordelis-Lo cylindrical roof benchmark verification.

This is one of the most classic benchmarks for shell finite elements, testing
the ability to capture membrane-bending coupling in curved shells.

Problem definition:
- Cylindrical roof: length L=50, radius R=25, half-angle=40°
- Thickness t=0.25, E=4.32e8, ν=0
- Vertical surface load q=90 (gravity-like)
- Boundary conditions: rigid diaphragms at curved ends (uy=uz=0), one axial constraint
- Reference: vertical displacement at midpoint of free edge = -0.3024

This verification compares two formulations:
1. Shell4 (flat facet): Q4 membrane + Mindlin plate, flat facets approximate curvature
2. Cylindrical shell4: Specialized element with exact cylindrical metric and curvature

The flat facet formulation with the historical default (one-point transverse
shear) plateaus ~50 % above the reference.  That plateau is NOT a faceting
limitation: holding the facets, nodes, load and diaphragms fixed and switching
only the transverse-shear rule of the same element brings the same faceted mesh
inside 3 % (see docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md and the case
pool entry CASE-FEM-SCORDELIS-LO-ROOF,
benchmarks/case_scordelis_lo_roof.py).  The cylindrical formulation achieves
<3 % error.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pin the import to *this* checkout.  The box carries several editable
# tensorfem installs (one per agent worktree); without an explicit pin the
# anchor verifies whichever checkout happened to win on sys.path -- the very
# provenance hole that made this anchor import ../m14687 instead of the tree
# under test.
# ---------------------------------------------------------------------------
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(_REPO_SRC))

import tensorfem  # noqa: E402  (must follow the path pin)

if _REPO_SRC.resolve() not in Path(tensorfem.__file__).resolve().parents:
    raise SystemExit(
        f"provenance error: tensorfem resolved to {tensorfem.__file__}, outside "
        f"this checkout ({_REPO_SRC})"
    )

from tensorfem.shell_benchmarks import scordelis_lo_roof
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical
from tensorfem.refinement_gate import evaluate_reference_ladder, summarise

_TENSORFEM_FILE = Path(tensorfem.__file__).resolve()
if _REPO_SRC.resolve() not in _TENSORFEM_FILE.parents:
    raise SystemExit(
        f"provenance error: tensorfem resolved to {_TENSORFEM_FILE}, "
        f"which is outside this checkout ({_REPO_SRC.resolve()})"
    )

PUBLISHED_REFERENCE = -0.3024
OUT = Path(__file__).parent


def mesh_convergence_flat_facet():
    """Run Scordelis-Lo with flat facet Shell4."""
    meshes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    results = []
    
    print("Flat facet Shell4:")
    for nx, nt in meshes:
        print(f"  Solving mesh {nx}x{nt}...")
        result = scordelis_lo_roof(nx, nt)
        w = result.probe_displacement
        error = abs(w / PUBLISHED_REFERENCE - 1.0)
        
        results.append({
            "element": "Shell4 (flat facet)",
            "mesh": f"{nx}x{nt}",
            "nx": nx,
            "nt": nt,
            "elements": nx * nt,
            "nodes": (nx + 1) * (nt + 1),
            "probe_displacement": w,
            "reference": PUBLISHED_REFERENCE,
            "relative_error": error,
            "relative_error_pct": error * 100,
        })
        print(f"    w = {w:.6f}, error = {error*100:.2f}%")
    
    return results


def mesh_convergence_cylindrical():
    """Run Scordelis-Lo with cylindrical shell4."""
    meshes = [(4, 4), (8, 8), (16, 16), (32, 32)]
    results = []
    
    print("\nCylindrical shell4:")
    for nx, nt in meshes:
        print(f"  Solving mesh {nx}x{nt}...")
        result = scordelis_lo_cylindrical(nx, nt)
        w = result.probe_displacement
        error = abs(w / PUBLISHED_REFERENCE - 1.0)
        
        results.append({
            "element": "Cylindrical shell4",
            "mesh": f"{nx}x{nt}",
            "nx": nx,
            "nt": nt,
            "elements": nx * nt,
            "nodes": (nx + 1) * (nt + 1),
            "probe_displacement": w,
            "reference": PUBLISHED_REFERENCE,
            "relative_error": error,
            "relative_error_pct": error * 100,
        })
        print(f"    w = {w:.6f}, error = {error*100:.2f}%")
    
    return results


def grade_ladder(results):
    """Grade one refinement ladder against the published reference.

    All convergence arithmetic is delegated to
    :mod:`tensorfem.refinement_gate` (which in turn uses the canonical
    ``convergence_verification`` module).  The former hand-rolled estimator
    ``order = (e_coarse/e_fine - 1)/(r - 1)`` is *not* the observed order of
    accuracy -- it is not a logarithm and it produced physically impossible
    values (p = 14.1 and 81.1 for a linear shell) plus a spurious
    "superconvergence" reading.
    """
    levels = [(1.0 / r["nx"], r["probe_displacement"], r["mesh"]) for r in results]
    return evaluate_reference_ladder(
        levels,
        reference_value=PUBLISHED_REFERENCE,
        tolerance=0.03,
        min_order=1.0,
        coarsest_sanity=0.5,
        qoi="vertical displacement at free-edge midpoint",
        reference_label="Scordelis & Lo (1960)",
    )


def compute_convergence_order(results, ladder=None):
    """Observed order of every mesh triple, labelled honestly.

    ``observed_order`` is ``None`` (with a reason) whenever the triple cannot
    support an estimate -- e.g. the flat-facet ladder oscillates instead of
    converging.
    """
    ladder = ladder if ladder is not None else grade_ladder(results)
    orders = []
    for entry in ladder["observed_orders"]:
        orders.append({
            "from_mesh": entry["from"],
            "to_mesh": entry["to"],
            "refinement_ratio": entry["ratio"],
            "observed_order": entry["order"],
            "reason": entry.get("reason", ""),
        })
    return orders


def plot_convergence(flat_results, cyl_results):
    """Generate convergence comparison plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs mesh size
    ax = axes[0]
    h_flat = [1.0 / r["nx"] for r in flat_results]
    errors_flat = [r["relative_error_pct"] for r in flat_results]
    h_cyl = [1.0 / r["nx"] for r in cyl_results]
    errors_cyl = [r["relative_error_pct"] for r in cyl_results]
    
    ax.loglog(h_flat, errors_flat, 'bo-', linewidth=2, markersize=8, label='Shell4 (flat facet)')
    ax.loglog(h_cyl, errors_cyl, 'rs-', linewidth=2, markersize=8, label='Cylindrical shell4')
    ax.axhline(3.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='3% threshold')
    ax.set_xlabel('Characteristic mesh size h/L', fontsize=12)
    ax.set_ylabel('Relative error (%)', fontsize=12)
    ax.set_title('Scordelis-Lo: Error vs Mesh Size', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Displacement vs mesh
    ax = axes[1]
    elements_flat = [r["elements"] for r in flat_results]
    disp_flat = [r["probe_displacement"] for r in flat_results]
    elements_cyl = [r["elements"] for r in cyl_results]
    disp_cyl = [r["probe_displacement"] for r in cyl_results]
    
    ax.semilogx(elements_flat, disp_flat, 'bo-', linewidth=2, markersize=8, label='Shell4 (flat facet)')
    ax.semilogx(elements_cyl, disp_cyl, 'rs-', linewidth=2, markersize=8, label='Cylindrical shell4')
    ax.axhline(PUBLISHED_REFERENCE, color='k', linestyle='--', linewidth=2, label=f'Reference: {PUBLISHED_REFERENCE}')
    ax.set_xlabel('Number of elements', fontsize=12)
    ax.set_ylabel('Probe displacement w', fontsize=12)
    ax.set_title('Scordelis-Lo: Displacement Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    fig.savefig(OUT / "scordelis_lo_convergence.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUT / 'scordelis_lo_convergence.png'}")
    plt.close()


def _pct(x):
    """Percent string for a relative error (None-safe)."""
    return "n/a" if x is None else f"{100.0 * x:.4f}%"


def _ladder_claim(ladder):
    """One-sentence verdict for one ladder, derived from its own gates.

    The conclusion used to be a hand-written sentence, which stayed in the
    artifact after the underlying numbers changed (the m18206 element repair
    turned the cylindrical claim from a pass into a ~54 % miss while the
    sentence still read "reproduces the Scordelis-Lo free-edge deflection").
    Composing it from the graded ladder keeps the two in step by construction.
    """
    status = ladder.get("status")
    finest = _pct(ladder.get("reported_accuracy_finest"))
    certified = ladder.get("certified_accuracy")
    if status == "PASS":
        return (f"certifies the free-edge deflection within the 3% tolerance "
                f"(finest-mesh {finest}, Richardson limit {_pct(certified)})")
    if status == "WITHHELD":
        return (f"cannot certify the claim: the mesh ladder is not in the asymptotic "
                f"range (finest-mesh {finest}, Richardson limit {_pct(certified)})")
    return (f"does not certify the claim: finest-mesh error {finest}, certified "
            f"accuracy {_pct(certified)}, outside the 3% tolerance")


def generate_report(flat_results, cyl_results, flat_orders, cyl_orders,
                    flat_ladder=None, cyl_ladder=None):
    """Generate verification report."""
    report = {
        "benchmark": "Scordelis-Lo cylindrical roof",
        "reference": {
            "value": PUBLISHED_REFERENCE,
            "source": "Scordelis & Lo (1960), widely cited in FEM literature",
            "quantity": "Vertical displacement at midpoint of free edge",
        },
        "problem": {
            "geometry": "Cylindrical roof: L=50, R=25, half-angle=40°",
            "material": "E=4.32e8, ν=0",
            "thickness": 0.25,
            "load": "Vertical surface load q=90",
            "boundary_conditions": "Rigid diaphragms at curved ends (uy=uz=0)",
        },
        "elements_compared": [
            {
                "name": "Shell4 (flat facet)",
                "formulation": "Q4 membrane + Mindlin plate with selective reduced integration",
                "curvature_treatment": "Flat facets approximate curved geometry",
                # Corrected in m16588: the ~50 % plateau is a rank deficiency of
                # the default one-point transverse-shear block, not a faceting
                # limitation.  Recorded here (and not only in `gates`) so a
                # regeneration cannot re-emit the falsified cause.
                "limitation": "Default one-point transverse shear is rank deficient: 3 spurious "
                              "zero-energy mechanisms per element leave the ladder ~50 % above the "
                              "reference; with assumed natural shear the same facets reach 2.35 % at "
                              "16x16 (CASE-FEM-SCORDELIS-LO-ROOF)",
            },
            {
                "name": "Cylindrical shell4",
                "formulation": "Specialized cylindrical Reissner-Mindlin shell",
                "curvature_treatment": "Exact cylindrical metric and curvature terms retained",
                "advantage": "Properly captures membrane-bending coupling in cylindrical shells",
            },
        ],
        "flat_facet_results": flat_results,
        "cylindrical_results": cyl_results,
        "flat_facet_convergence": flat_orders,
        "cylindrical_convergence": cyl_orders,
        "ladder_grading": {
            "flat_facet": flat_ladder,
            "cylindrical": cyl_ladder,
        },
        "verdict": {
            "flat_facet": {
                "status": flat_ladder["status"],
                "reason": (
                    f"certified accuracy (Richardson limit of the ladder) "
                    f"{_pct(flat_ladder['certified_accuracy'])} outside the 3% tolerance"
                    if flat_ladder["certified_accuracy"] is not None
                    and flat_ladder["certified_accuracy"] > 0.03
                    else "see ladder_grading.flat_facet.gates"
                ),
                "reported_accuracy_finest": flat_ladder["reported_accuracy_finest"],
                "certified_accuracy": flat_ladder["certified_accuracy"],
                "best_error": _pct(flat_ladder["reported_accuracy_finest"]),
            },
            "cylindrical": {
                "status": cyl_ladder["status"],
                "reason": (
                    "accuracy certified at the Richardson limit; ladder in the asymptotic range"
                    if cyl_ladder["status"] == "PASS"
                    else "ladder cannot certify the claim: " + "; ".join(
                        g["detail"] for g in cyl_ladder["gates"] if g["status"] != "PASS"
                    )
                ),
                "reported_accuracy_finest": cyl_ladder["reported_accuracy_finest"],
                "certified_accuracy": cyl_ladder["certified_accuracy"],
                "best_error": _pct(cyl_ladder["reported_accuracy_finest"]),
            },
            "conclusion": (
                f"Cylindrical shell4 {_ladder_claim(cyl_ladder)}; the flat-facet "
                f"formulation {_ladder_claim(flat_ladder)}. Finest-mesh error alone is "
                f"not evidence: see ladder_grading for the certified numbers."
            ),
        },
    }
    
    with open(OUT / "result.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {OUT / 'result.json'}")
    
    return report


def generate_gates(flat_results, cyl_results, flat_ladder=None, cyl_ladder=None):
    """Machine contract of the anchor, graded on the *asymptotic* accuracy.

    A ladder that is pre-asymptotic or non-monotone cannot certify anything:
    such a criterion is recorded as ``WITHHELD`` (evidence insufficient), never
    as a pass.  ``passed`` is true only when every certifying criterion passes.
    """
    flat_ladder = flat_ladder if flat_ladder is not None else grade_ladder(flat_results)
    cyl_ladder = cyl_ladder if cyl_ladder is not None else grade_ladder(cyl_results)

    gates = []
    for tag, label, ladder in (
        ("flat_facet", "Flat facet Shell4", flat_ladder),
        ("cylindrical", "Cylindrical shell4", cyl_ladder),
    ):
        for g in ladder["gates"]:
            gates.append({
                "id": f"{tag}.{g['id']}",
                "description": f"{label}: {g['description']}",
                "status": g["status"],
                "passed": bool(g["passed"]),
                "operator": g["operator"],
                "threshold": g["threshold"],
                "actual": g["actual"],
                "detail": g["detail"],
            })

    cyl_summary = summarise(cyl_ladder)
    flat_summary = summarise(flat_ladder)
    overall = cyl_summary["overall_verdict"]

    document = {
        "schema": "tensorfem.verified-gates.v1",
        "benchmark": "Scordelis-Lo cylindrical roof",
        "reference": {
            "value": PUBLISHED_REFERENCE,
            "source": "Scordelis & Lo (1960); free-edge midpoint vertical deflection",
        },
        "criteria": {
            "tolerance": 0.03,
            "min_order": 1.0,
            "coarsest_sanity": 0.5,
            "certified_quantity": "Richardson limit of the mesh ladder (not the finest mesh)",
        },
        "element_under_test": "cylindrical",
        "gates": gates,
        "verdicts": {
            "flat_facet": flat_summary,
            "cylindrical": cyl_summary,
        },
        "overall_verdict": overall,
        "passed": overall == "PASS",
        "reported_vs_certified": {
            "flat_facet": {
                "reported_finest_error": flat_ladder["reported_accuracy_finest"],
                "certified_error": flat_ladder["certified_accuracy"],
            },
            "cylindrical": {
                "reported_finest_error": cyl_ladder["reported_accuracy_finest"],
                "certified_error": cyl_ladder["certified_accuracy"],
            },
        },
        # Evidence the certificate names, so a reader can check the claim
        # without re-running the solver: the convergence chart and the two
        # diagnostic studies are committed next to the numbers they back.
        "artifacts": {
            "result": "result.json",
            "gates": "gates_verified.json",
            "fig_convergence": "scordelis_lo_convergence.png",
            "shear_treatment_study": "shear_treatment_study.json",
            "reaction_residual_study": "reaction_residual_study.json",
        },
        "limitations": [
            # Corrected in m16449: the ~50% plateau is a rank deficiency of the
            # one-point transverse-shear block (3 spurious zero-energy modes per
            # element), not a limitation of faceting.  The same faceted mesh with
            # assumed natural shear reaches 2.35% at 16x16 and 0.75% at 32x32 --
            # see shear_treatment_study.py and
            # docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md.
            "Flat facet Shell4 with the default one-point transverse-shear rule is rank "
            "deficient (3 spurious zero-energy modes per element) and its ladder plateaus "
            "~50% above the reference; with assumed natural shear the same facets reach "
            "2.35% at 16x16",
            "Cylindrical shell4 is specialized for cylindrical geometry only",
            "General curved shells require either specialized elements or improved flat facet formulations",
            "Accuracy is certified at the Richardson limit of a 4-level ladder; a finer reference "
            "value (more than four significant digits) is required to certify sub-0.1% claims",
        ],
    }

    with open(OUT / "gates_verified.json", "w") as f:
        json.dump(document, f, indent=2)
    print(f"Saved: {OUT / 'gates_verified.json'}")

    return document


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from_dump = "--from-dump" in argv

    print("=" * 70)
    print("Scordelis-Lo Cylindrical Roof Benchmark Verification")
    print("Comparing flat facet vs cylindrical shell formulations")
    print("=" * 70)
    print()

    if from_dump:
        # Re-grade the committed ladder without touching the solver: the
        # certificate must be rebuildable from the artifact it certifies.
        dumped = json.loads((OUT / "result.json").read_text())
        flat_results = dumped["flat_facet_results"]
        cyl_results = dumped["cylindrical_results"]
        print(f"Re-grading the committed ladder from {OUT / 'result.json'} (no solves)")
    else:
        flat_results = mesh_convergence_flat_facet()
        cyl_results = mesh_convergence_cylindrical()
    print()
    
    flat_orders = compute_convergence_order(flat_results)
    cyl_orders = compute_convergence_order(cyl_results)
    
    if flat_orders:
        print("Flat facet convergence orders:")
        for o in flat_orders:
            shown = "n/a" if o["observed_order"] is None else f"{o['observed_order']:.3f}"
            print(f"  {o['from_mesh']} → {o['to_mesh']}: p = {shown} {o['reason']}")
    else:
        print("Flat facet: non-monotone convergence (no valid orders)")
    
    if cyl_orders:
        print("\nCylindrical shell4 convergence orders:")
        for o in cyl_orders:
            shown = "n/a" if o["observed_order"] is None else f"{o['observed_order']:.3f}"
            print(f"  {o['from_mesh']} → {o['to_mesh']}: p = {shown} {o['reason']}")
    else:
        print("\nCylindrical shell4: non-monotone convergence (no valid orders)")
    
    print()
    if not from_dump:
        plot_convergence(flat_results, cyl_results)
    print()
    
    flat_ladder = grade_ladder(flat_results)
    cyl_ladder = grade_ladder(cyl_results)
    report = generate_report(flat_results, cyl_results, flat_orders, cyl_orders,
                             flat_ladder=flat_ladder, cyl_ladder=cyl_ladder)
    gates = generate_gates(flat_results, cyl_results, flat_ladder, cyl_ladder)
    
    print()
    print("=" * 70)
    print("SUMMARY:")
    print(f"  Flat facet Shell4: {report['verdict']['flat_facet']['status']}")
    print(f"    Best error: {report['verdict']['flat_facet']['best_error']}")
    print(f"  Cylindrical shell4: {report['verdict']['cylindrical']['status']}")
    print(f"    accuracy_verdict={gates['verdicts']['cylindrical']['accuracy_verdict']} "
          f"study_verdict={gates['verdicts']['cylindrical']['study_verdict']}"
          f" unproven={gates['verdicts']['cylindrical']['unproven']}")
    print(f"    Best error: {report['verdict']['cylindrical']['best_error']}")
    print(f"  Conclusion: {report['verdict']['conclusion']}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
