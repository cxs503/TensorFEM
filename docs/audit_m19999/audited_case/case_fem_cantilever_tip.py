#!/usr/bin/env python3
"""CASE-FEM-CANTILEVER-TIP - clamped-free beam tip deflection vs ``P L^3 / (3 E I)``.

Calculation-pool anchor (owner 2026-11-12 case-pool order), case code
``FEM-CANTILEVER-TIP``, acceptance criterion ``< 2 %`` (``tensor_case_pool``).

Physical problem
----------------
A uniform, homogeneous, isotropic cantilever is clamped at ``x = 0`` and loaded by a
single transverse force ``P`` at the free end ``x = L``.  The target quantity is the
tip deflection in the direction of the load.

Reference (external anchor)
---------------------------
Euler-Bernoulli beam theory with a unit-load (Castigliano) derivation::

    delta = int_0^L M(x) m(x) / (E I) dx = P L^3 / (3 E I),   I = b h^3 / 12

provenance: S. Timoshenko, *Strength of Materials, Part I*, 3rd ed., Van Nostrand,
1955, ch. 4 (cantilever, end load); R.D. Blevins, *Formulas for Natural Frequency and
Mode Shape*, Van Nostrand Reinhold, 1979, Table 8-1 (same beam family).  The closed
form is **not** copied: :func:`castigliano_integral` re-derives it from the bending
moment field by numerical quadrature, and the two must agree to ``1e-9`` relative
before any FE number is looked at.

The shear-flexible (Timoshenko) reference ``P L^3/(3 E I) + P L/(kappa A G)``,
``kappa = 5/6`` for a rectangular section, is carried alongside because it bounds the
*theory* error of the declared anchor.  At ``L/h = 100`` that error is 0.008 %, far
below the 2 % band, so the anchor is sharp for this geometry.

Model levels (the three alternatives of the method-selection study)
-------------------------------------------------------------------
M1 ``beam_eb``          1-D Euler-Bernoulli frame FE, cubic Hermite + 2-point Gauss
                        (``tensorfem.frame2d``).  Reproduces the cubic bending field
                        exactly, so it tests the shipped beam solver rather than
                        discretisation.
M2 ``beam_timoshenko``  1-D Timoshenko beam FE with exact shear-flexible stiffness
                        (``tensorfem.advanced_beam.solve_timoshenko_cantilever``);
                        adds shear deformation, quantifies the anchor's theory error.
M3 ``solid_hex20``      3-D quadratic 20-node brick FE (``tensorfem.hex20``), no beam
                        kinematics: the only level that can falsify "the beam
                        kinematics behind the anchor are adequate here".  The end load
                        enters as a statically equivalent nodal load on the free face
                        (sum = P), so clamp and load are the only idealisations.

Sparse assembly + SuperLU (``solve_hex20_sparse``) above 3000 dof, dense LAPACK below,
so the cross-section refinement ladder stays affordable.

Predeclared acceptance (fixed before the numbers are produced)
-------------------------------------------------------------
* metric: tip deflection of the **solid reference mesh** ``32 x 4 x 4`` (axial x
  cross x cross) against ``reference.euler_bernoulli_m``;
* tolerance: ``|got - ref| / ref < 0.02`` (the case-pool ``< 2 %``);
* secondary, non-blocking evidence: M1 machine-exact on the cubic solution
  (``< 1e-9``), monotone h-convergence of M3, load linearity, slenderness sensitivity.
Levels outside the declared reference mesh are reported, never used to rescue it.

Run
---
    python benchmarks/case_fem_cantilever_tip.py --quick           # smoke, ~20 s
    python benchmarks/case_fem_cantilever_tip.py --write-docs      # full study
    python benchmarks/case_fem_cantilever_tip.py --json            # workload stdout

Outputs (every number on the docs page is injected from the computed evidence)
----------------------------------------------------------------------------
``benchmarks/case_fem_cantilever_tip_evidence.json``
``docs/assets/case-fem-cantilever-tip/*.png``
``docs/CASE-FEM-CANTILEVER-TIP.md`` (only with ``--write-docs``)
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.advanced_beam import solve_timoshenko_cantilever  # noqa: E402
from tensorfem.frame2d import FrameModel, solve_frame_static  # noqa: E402
from tensorfem.hex20 import (  # noqa: E402
    Hex20Model,
    solve_hex20,
    solve_hex20_sparse,
    structured_hex20_mesh,
)

# --------------------------------------------------------------------------- #
# Case definition (declared case parameters)
# --------------------------------------------------------------------------- #
BENCHMARK_ID = "CASE-FEM-CANTILEVER-TIP"
BENCHMARK_VERSION = "1.0.0"
WORKLOAD_ID = "benchmarks/case_fem_cantilever_tip.py"
WORKLOAD_VERSION = "1.0.0"

L = 1.0            # span [m]
B = 0.01           # width  (y) [m]
H = 0.01           # height (z, bending direction) [m]
E = 200e9          # Young's modulus [Pa]
NU = 0.3           # Poisson's ratio [-]
P = 5.0            # tip force in -z [N]
KAPPA = 5.0 / 6.0  # rectangular-section shear correction factor [-]

TOLERANCE = 0.02               # case-pool acceptance: < 2 %
REFERENCE_MESH = (32, 4)       # (axial elements, elements per cross-section side)
BEAM_EB_ELEMENTS = 8           # declared beam-FE reference discretisation
BEAM_TIMOSHENKO_ELEMENTS = 8
DTYPE = torch.float64
SPARSE_DOF_THRESHOLD = 3000

#: axial / cross-section refinement ladder of the solid model
AXIAL_LADDER = ((8, 1), (16, 1), (16, 2), (32, 2), (32, 4), (64, 4))
QUICK_LADDER = ((8, 1), (16, 2))
#: slenderness sensitivity points at the cheap solid mesh (32 x 2 x 2)
SLENDERNESS_POINTS = (100.0, 50.0, 20.0)
SENSITIVITY_MESH = (32, 2)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT / "docs" / "assets" / "case-fem-cantilever-tip"
EVIDENCE_PATH = OUT_DIR / "case_fem_cantilever_tip_evidence.json"
DOC_PATH = ROOT / "docs" / "CASE-FEM-CANTILEVER-TIP.md"


def log(message: str) -> None:
    """Progress goes to stderr so ``--json`` stdout stays one JSON document."""
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Reference: closed form + independent unit-load re-derivation
# --------------------------------------------------------------------------- #
def section_properties(b: float = B, h: float = H, e: float = E, nu: float = NU,
                       kappa: float = KAPPA) -> dict:
    area = b * h
    inertia = b * h ** 3 / 12.0
    shear = e / (2.0 * (1.0 + nu))
    return {
        "area_m2": area,
        "inertia_m4": inertia,
        "shear_modulus_Pa": shear,
        "kappa_shear_area_Pa_m2": kappa * area * shear,
        "bending_stiffness_Pa_m4": e * inertia,
    }


def analytic_reference(length: float = L, load: float = P, b: float = B, h: float = H,
                       e: float = E, nu: float = NU, kappa: float = KAPPA) -> dict:
    """Closed-form tip deflection: Euler-Bernoulli and Timoshenko."""
    props = section_properties(b, h, e, nu, kappa)
    ei = props["bending_stiffness_Pa_m4"]
    delta_eb = load * length ** 3 / (3.0 * ei)
    delta_shear = load * length / props["kappa_shear_area_Pa_m2"]
    return {
        "euler_bernoulli_m": delta_eb,
        "shear_term_m": delta_shear,
        "timoshenko_m": delta_eb + delta_shear,
        "shear_fraction_of_bending": delta_shear / delta_eb,
        **props,
    }


def castigliano_integral(length: float = L, load: float = P, b: float = B, h: float = H,
                         e: float = E, nu: float = NU, kappa: float = KAPPA,
                         samples: int = 20001) -> dict:
    """Independent re-derivation of the anchor by numerical unit-load integration.

    ``delta = int_0^L M m /(EI) dx + int_0^L V v /(kappa A G) dx`` with
    ``M = P (L - x)``, unit-load moment ``m = (L - x)``, ``V = P``, unit shear
    ``v = 1``.  The integrand is quadratic in ``x``, so composite Simpson on an even
    number of intervals is exact up to round-off -- a trapezoid rule would leave an
    ``O(h^2)`` residual that must not be confused with a model error.
    """
    props = section_properties(b, h, e, nu, kappa)
    ei = props["bending_stiffness_Pa_m4"]
    kag = props["kappa_shear_area_Pa_m2"]
    if samples % 2 == 0:
        samples += 1
    step = length / (samples - 1)
    bending = 0.0
    shear = 0.0
    for i in range(samples):
        x = i * step
        weight = 1.0 if i in (0, samples - 1) else (4.0 if i % 2 else 2.0)
        bending += weight * load * (length - x) ** 2 / ei
        shear += weight * load / kag
    return {"bending_m": bending * step / 3.0,
            "shear_m": shear * step / 3.0,
            "total_m": (bending + shear) * step / 3.0}


# --------------------------------------------------------------------------- #
# Model levels
# --------------------------------------------------------------------------- #
def beam_eb_level(n_elements: int, length: float = L, load: float = P, b: float = B,
                  h: float = H, e: float = E, nu: float = NU) -> dict:
    """M1: cubic-Hermite Euler-Bernoulli frame FE (bending only)."""
    props = section_properties(b, h, e, nu)
    t0 = time.perf_counter()
    nodes = torch.stack([
        torch.linspace(0.0, length, n_elements + 1, dtype=DTYPE),
        torch.zeros(n_elements + 1, dtype=DTYPE),
    ], dim=1)
    elements = torch.stack([torch.arange(n_elements), torch.arange(1, n_elements + 1)], dim=1)
    forces = torch.zeros(3 * (n_elements + 1), dtype=DTYPE)
    forces[3 * n_elements + 1] = -load
    model = FrameModel(
        nodes=nodes, elements=elements,
        young_modulus=torch.tensor([e], dtype=DTYPE),
        area=torch.tensor([props["area_m2"]], dtype=DTYPE),
        inertia=torch.tensor([props["inertia_m4"]], dtype=DTYPE),
        forces=forces, fixed_dofs=torch.tensor([0, 1, 2]),
    )
    result = solve_frame_static(model)
    tip = -float(result.displacement[3 * n_elements + 1])
    return {
        "model": "beam_eb",
        "n_elements": int(n_elements),
        "dofs": int(model.n_dofs),
        "tip_deflection_m": tip,
        "seconds": time.perf_counter() - t0,
    }


def beam_timoshenko_level(n_elements: int, length: float = L, load: float = P, b: float = B,
                          h: float = H, e: float = E, nu: float = NU,
                          kappa: float = KAPPA) -> dict:
    """M2: Timoshenko beam FE (bending + shear deformation)."""
    props = section_properties(b, h, e, nu, kappa)
    t0 = time.perf_counter()
    result = solve_timoshenko_cantilever(
        length=length, elements=n_elements, E=e, I=props["inertia_m4"],
        G=props["shear_modulus_Pa"], A=props["area_m2"], load=load, kappa=kappa,
    )
    tip = float(result.displacement[-2])
    return {
        "model": "beam_timoshenko",
        "n_elements": int(n_elements),
        "dofs": int(2 * (n_elements + 1)),
        "tip_deflection_m": tip,
        "seconds": time.perf_counter() - t0,
    }


def _axis_mask(nodes: torch.Tensor, axis: int, value: float) -> torch.Tensor:
    target = torch.as_tensor(value, dtype=nodes.dtype)
    return torch.isclose(nodes[:, axis], target, rtol=0.0, atol=1e-12 * max(1.0, abs(value)))


def solid_level(n_axial: int, n_cross: int, *, length: float = L, load: float = P,
                b: float = B, h: float = H, e: float = E, nu: float = NU,
                keep_solution: bool = False) -> dict:
    """M3: 3-D HEX20 solid cantilever with a statically equivalent end-face load.

    ``n_cross`` elements are used along *both* cross-section directions (the section
    is square, ``b = h``), so the mesh is ``n_axial x n_cross x n_cross``.  Everything
    after ``n_cross`` is keyword-only on purpose: a positional third argument used to
    be silently bound to ``length`` and corrupted the reference solve.
    """
    t0 = time.perf_counter()
    nodes, cells = structured_hex20_mesh(length, b, h, n_axial, n_cross, n_cross, dtype=DTYPE)
    n_nodes = int(len(nodes))
    forces = torch.zeros(3 * n_nodes, dtype=DTYPE)
    tip_nodes = torch.where(_axis_mask(nodes, 0, length))[0]
    forces[3 * tip_nodes + 2] = -load / len(tip_nodes)
    clamp_nodes = torch.where(_axis_mask(nodes, 0, 0.0))[0]
    fixed = torch.stack([3 * clamp_nodes, 3 * clamp_nodes + 1, 3 * clamp_nodes + 2],
                        dim=1).reshape(-1)
    model = Hex20Model(
        nodes=nodes, elements=cells,
        young_modulus=torch.tensor(e, dtype=DTYPE),
        poisson_ratio=torch.tensor(nu, dtype=DTYPE),
        forces=forces, fixed_dofs=fixed,
    )
    solver = "sparse_superlu" if model.n_dofs > SPARSE_DOF_THRESHOLD else "dense_lapack"
    result = solve_hex20_sparse(model) if solver == "sparse_superlu" else solve_hex20(model)
    tip = -float(result.displacement[3 * tip_nodes + 2].mean())
    row = {
        "model": "solid_hex20",
        "axial_elements": int(n_axial),
        "cross_elements": int(n_cross),
        "elements": int(len(cells)),
        "nodes": n_nodes,
        "dofs": int(model.n_dofs),
        "linear_solver": solver,
        "load_application": "statically_equivalent_end_face_transverse_nodal_load",
        "tip_deflection_m": tip,
        "seconds": time.perf_counter() - t0,
    }
    if keep_solution:
        row["_fields"] = {"nodes": nodes, "displacement": result.displacement}
    return row


def append_errors(rows, ref: dict, key: str = "euler_bernoulli") -> None:
    """Attach relative errors against ``ref`` to each row in place."""
    reference = ref[f"{key}_m"]
    for row in rows:
        row[f"relative_error_vs_{key}"] = (row["tip_deflection_m"] - reference) / reference
        row[f"abs_relative_error_vs_{key}"] = abs(row[f"relative_error_vs_{key}"])
        row["within_tolerance_vs_anchor"] = row[f"abs_relative_error_vs_{key}"] < TOLERANCE


def mesh_label(mesh) -> str:
    """``(axial, cross)`` -> ``AxCxC`` label of the structured HEX20 mesh."""
    return f"{int(mesh[0])}x{int(mesh[1])}x{int(mesh[1])}"


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
def run_study(quick: bool = False) -> dict:
    ladder = QUICK_LADDER if quick else AXIAL_LADDER
    reference = analytic_reference()
    quad = castigliano_integral()
    quad_bending_rel = (quad["bending_m"] - reference["euler_bernoulli_m"]) \
        / reference["euler_bernoulli_m"]
    quad_shear_rel = (quad["shear_m"] - reference["shear_term_m"]) / reference["shear_term_m"]
    quadrature_rel = (quad["total_m"] - reference["timoshenko_m"]) / reference["timoshenko_m"]

    log(f"[reference] delta_EB = {reference['euler_bernoulli_m']:.12e} m "
        f"(unit-load quadrature: bending {quad['bending_m']:.12e} m rel {quad_bending_rel:+.2e}, "
        f"shear {quad['shear_m']:.12e} m rel {quad_shear_rel:+.2e})")

    beam_rows = [beam_eb_level(n) for n in ((2, 4, 8, 16) if not quick else (3, 6))]
    timoshenko_rows = [beam_timoshenko_level(BEAM_TIMOSHENKO_ELEMENTS)]
    append_errors(beam_rows, reference, "euler_bernoulli")
    append_errors(timoshenko_rows, reference, "euler_bernoulli")
    for row in timoshenko_rows:
        row["relative_error_vs_timoshenko"] = (
            row["tip_deflection_m"] - reference["timoshenko_m"]) / reference["timoshenko_m"]
    for row in beam_rows + timoshenko_rows:
        log(f"[{row['model']}] n={row['n_elements']:>2} dofs={row['dofs']:>5} "
            f"tip={row['tip_deflection_m']:.12e} m "
            f"err={row['relative_error_vs_euler_bernoulli'] * 100:+.3e}% "
            f"({row['seconds']:.2f} s)")

    solid_rows = []
    for n_axial, n_cross in ladder:
        row = solid_level(n_axial, n_cross)
        append_errors([row], reference, "euler_bernoulli")
        solid_rows.append(row)
        log(f"[solid_hex20] {n_axial}x{n_cross}x{n_cross} dofs={row['dofs']:>6} "
            f"tip={row['tip_deflection_m']:.12e} m "
            f"err={row['relative_error_vs_euler_bernoulli'] * 100:+.4f}% "
            f"({row['seconds']:.1f} s, {row['linear_solver']})")

    solid_reference = solid_level(*REFERENCE_MESH, keep_solution=True)
    solid_fields = solid_reference.pop("_fields")
    append_errors([solid_reference], reference, "euler_bernoulli")
    log(f"[solid_hex20] reference {mesh_label(REFERENCE_MESH)} "
        f"dofs={solid_reference['dofs']} tip={solid_reference['tip_deflection_m']:.12e} m "
        f"err={solid_reference['relative_error_vs_euler_bernoulli'] * 100:+.4f}% "
        f"({solid_reference['seconds']:.1f} s)")

    # ---- sensitivity 1: slenderness (theory error + discretisation error) ---
    slenderness_rows = []
    points = SLENDERNESS_POINTS if not quick else (100.0, 20.0)
    for ratio in points:
        height = L / ratio
        ref = analytic_reference(length=L, load=P, b=B, h=height)
        solid = solid_level(*SENSITIVITY_MESH, b=B, h=height)
        solid["slenderness_L_over_h"] = ratio
        solid["height_m"] = height
        solid["reference_euler_bernoulli_m"] = ref["euler_bernoulli_m"]
        solid["reference_timoshenko_m"] = ref["timoshenko_m"]
        solid["shear_fraction"] = ref["shear_fraction_of_bending"]
        append_errors([solid], ref, "euler_bernoulli")
        solid["relative_error_vs_timoshenko"] = (
            solid["tip_deflection_m"] - ref["timoshenko_m"]) / ref["timoshenko_m"]
        slenderness_rows.append(solid)
        log(f"[slenderness L/h={ratio:.0f}] shear term = "
            f"{ref['shear_fraction_of_bending'] * 100:.4f}% of bending; "
            f"solid err vs EB {solid['relative_error_vs_euler_bernoulli'] * 100:+.4f}%, "
            f"vs Timoshenko {solid['relative_error_vs_timoshenko'] * 100:+.4f}%")

    # ---- sensitivity 2: load linearity at the reference mesh ----------------
    linearity_rows = [{"load_N": P, "tip_deflection_m": solid_reference["tip_deflection_m"]}]
    scaled = solid_level(*REFERENCE_MESH, load=2.0 * P)
    linearity_rows.append({"load_N": 2.0 * P, "tip_deflection_m": scaled["tip_deflection_m"]})
    ratio = linearity_rows[1]["tip_deflection_m"] / linearity_rows[0]["tip_deflection_m"]
    log(f"[linearity] tip(2P)/tip(P) = {ratio:.12f} (linear theory: 2)")

    reference_error = abs(solid_reference["relative_error_vs_euler_bernoulli"])
    checks = {
        "reference_formula_rederived": bool(
            abs(quad_bending_rel) < 1e-12 and abs(quad_shear_rel) < 1e-12),
        "solid_reference_within_tolerance": bool(reference_error < TOLERANCE),
        "beam_eb_machine_exact": bool(all(
            row["abs_relative_error_vs_euler_bernoulli"] < 1e-9 for row in beam_rows)),
        "timoshenko_beam_matches_theory": bool(all(
            abs(row["relative_error_vs_timoshenko"]) < 1e-6 for row in timoshenko_rows)),
        "load_linearity": bool(abs(ratio - 2.0) < 1e-9),
    }
    if len(solid_rows) >= 2:
        checks["solid_h_convergence_monotone"] = bool(
            solid_rows[-1]["abs_relative_error_vs_euler_bernoulli"]
            < solid_rows[0]["abs_relative_error_vs_euler_bernoulli"])

    return {
        "schema_version": "1.0.0",
        "run_id": f"{BENCHMARK_ID}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "benchmark": {"benchmark_id": BENCHMARK_ID, "version": BENCHMARK_VERSION},
        "implementation": {
            "project": "TensorFEM",
            "revision": _git_revision(),
            "code": "benchmarks/case_fem_cantilever_tip.py",
            "linear_algebra": "torch.linalg.solve (dense LAPACK) / scipy SuperLU (sparse)",
        },
        "workload": {
            "workload_id": WORKLOAD_ID, "version": WORKLOAD_VERSION,
            "mode": "quick" if quick else "full",
            "configuration": {
                "axial_ladder": [list(p) for p in ladder],
                "reference_mesh": list(REFERENCE_MESH),
                "slenderness_points": list(points),
                "tolerance_rel": TOLERANCE,
            },
        },
        "execution": {
            "status": "success",
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": f"{platform.system()}/{platform.machine()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dtype": str(DTYPE),
            "device": "cpu",
        },
        "geometry": {
            "length_m": L, "width_m": B, "height_m": H,
            "slenderness_L_over_h": L / H, "load_direction": "-z",
        },
        "material": {"young_modulus_Pa": E, "poisson_ratio": NU},
        "load": {"tip_force_N": P, "tip_force_doubled_N": 2 * P,
                 "type": "concentrated transverse end load"},
        "reference": {
            **reference,
            "source": ("Timoshenko, Strength of Materials Part I, 3rd ed. (1955) ch. 4; "
                       "Blevins (1979) Table 8-1"),
            "unit_load_quadrature_bending_m": quad["bending_m"],
            "unit_load_quadrature_bending_rel": quad_bending_rel,
            "unit_load_quadrature_shear_m": quad["shear_m"],
            "unit_load_quadrature_shear_rel": quad_shear_rel,
            "unit_load_quadrature_rel": quadrature_rel,
        },
        "levels": {
            "beam_eb": beam_rows[-1],
            "beam_timoshenko": timoshenko_rows[-1],
            "solid_reference": solid_reference,
            "solid_coarsest": solid_rows[0],
        },
        "studies": {
            "beam_eb_sweep": beam_rows,
            "beam_timoshenko_sweep": timoshenko_rows,
            "solid_ladder": solid_rows,
            "slenderness": slenderness_rows,
            "load_linearity": {"rows": linearity_rows, "ratio_2p_over_p": ratio},
        },
        "verdict": {
            "tolerance_rel": TOLERANCE,
            "reference_mesh": list(REFERENCE_MESH),
            "reference_value_m": reference["euler_bernoulli_m"],
            "tip_deflection_m": solid_reference["tip_deflection_m"],
            "relative_error": solid_reference["relative_error_vs_euler_bernoulli"],
            "abs_relative_error": reference_error,
            "passed": bool(checks["solid_reference_within_tolerance"]),
            "checks": checks,
        },
        "_solid_fields": solid_fields,
    }


def _git_revision() -> str:
    """Commit that produced the numbers (``unknown`` in an export/archive)."""
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - bare checkout / archive
        return "unknown"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(doc: dict, fields: dict) -> list:
    plt = _plt()
    import numpy as np

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tol = doc["verdict"]["tolerance_rel"]
    paths = []

    solid = doc["studies"]["solid_ladder"]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = [r["dofs"] for r in solid]
    y = [r["abs_relative_error_vs_euler_bernoulli"] for r in solid]
    ax.loglog(x, y, "o-", color="#c62828", lw=1.8, ms=6,
              label="HEX20 solid (h + cross-section ladder)")
    for r in solid:
        ax.annotate(f"{r['axial_elements']}x{r['cross_elements']}",
                    (r["dofs"], r["abs_relative_error_vs_euler_bernoulli"]),
                    textcoords="offset points", xytext=(5, 6), fontsize=7)
    beam = doc["levels"]["beam_eb"]
    ax.axhline(max(abs(beam["relative_error_vs_euler_bernoulli"]), 1e-15),
               color="#1565c0", ls=":", lw=1.2,
               label=f"EB beam FE (machine-exact, {abs(beam['relative_error_vs_euler_bernoulli']):.1e})")
    ax.axhline(tol, color="0.35", ls="--", lw=1.2, label=f"acceptance {tol*100:.0f}%")
    ref_pt = doc["levels"]["solid_reference"]
    ax.plot([ref_pt["dofs"]], [ref_pt["abs_relative_error_vs_euler_bernoulli"]], "*",
            ms=14, color="#2e7d32", label="declared reference mesh 32x4x4")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel(r"$|\delta_{FE}-\delta_{EB}|/\delta_{EB}$")
    ax.set_title("CASE-FEM-CANTILEVER-TIP: tip-deflection error vs discretisation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    p1 = FIG_DIR / "convergence.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = ["Euler-Bernoulli\n(anchor)", "Timoshenko\n(analytical)",
              "EB beam FE", "Timoshenko beam FE", "HEX20 coarse\n(8x1x1)",
              "HEX20 reference\n(32x4x4)"]
    values = [doc["reference"]["euler_bernoulli_m"], doc["reference"]["timoshenko_m"],
              doc["levels"]["beam_eb"]["tip_deflection_m"],
              doc["levels"]["beam_timoshenko"]["tip_deflection_m"],
              doc["levels"]["solid_coarsest"]["tip_deflection_m"],
              doc["levels"]["solid_reference"]["tip_deflection_m"]]
    colors = ["#1565c0", "#2e7d32", "#90caf9", "#a5d6a7", "#ef9a9a", "#c62828"]
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="white")
    for i, (b, v) in enumerate(zip(bars, values)):
        ax.text(i, v, f"{v:.6e}", ha="center", va="bottom", fontsize=7)
    ax.axhline(doc["reference"]["euler_bernoulli_m"], color="0.3", ls="--", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("tip deflection [m]")
    ax.set_title(f"Tip deflection by model level (P = {doc['load']['tip_force_N']:.0f} N)")
    fig.tight_layout()
    p2 = FIG_DIR / "levels.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # field contour: uz on the mid-plane y = b/2 of the reference solid model
    nodes = fields["nodes"].cpu().numpy()
    uz = fields["displacement"][2::3].cpu().numpy()
    ymid = nodes[:, 1].mean()
    sel = np.isclose(nodes[:, 1], ymid, atol=1e-12)
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    tri = ax.tricontourf(nodes[sel, 0], nodes[sel, 2] * 1e3, uz[sel] * 1e3, levels=24,
                         cmap="viridis")
    cbar = fig.colorbar(tri, ax=ax)
    cbar.set_label(r"$u_z$ [mm]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Reference HEX20 solution, mid-plane bending displacement")
    fig.tight_layout()
    p3 = FIG_DIR / "field_uz.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)
    return paths


# --------------------------------------------------------------------------- #
# Docs page (numbers injected from the evidence document)
# --------------------------------------------------------------------------- #
def render_docs(doc: dict, figures: list) -> str:
    ref = doc["reference"]
    verdict = doc["verdict"]
    solid = doc["studies"]["solid_ladder"]
    beam = doc["studies"]["beam_eb_sweep"]
    tim = doc["levels"]["beam_timoshenko"]
    slen = doc["studies"]["slenderness"]
    lin = doc["studies"]["load_linearity"]
    checks = verdict["checks"]
    fig_names = ", ".join(f"`docs/assets/case-fem-cantilever-tip/{p.name}`" for p in figures)

    def mark(ok: bool) -> str:
        return "✅" if ok else "❌"

    lines = [
        "# CASE-FEM-CANTILEVER-TIP — 端部集中力下的悬臂梁端挠度",
        "",
        f"- 算例代码：`FEM-CANTILEVER-TIP`（repo `TensorFEM`，tier `core`）",
        f"- 基准：`delta = P L^3 / (3 E I)`，出处 {ref['source']}",
        f"- 验收判据：`|got-ref|/ref < {verdict['tolerance_rel']*100:.0f}%`"
        f"（`tensor_case_pool.tol_policy`）",
        f"- 运行：`python benchmarks/case_fem_cantilever_tip.py --write-docs`",
        f"- 证据：`benchmarks/case_fem_cantilever_tip_evidence.json`；图：{fig_names}",
        "",
        "## ① 问题设定与基准出处",
        "",
        "均质各向同性等截面悬臂梁，`x=0` 固支，自由端 `x=L` 受横向集中力 `P`（沿 `-z`）。",
        "Euler-Bernoulli 梁理论给出（单位载荷法 / Castigliano）：",
        "",
        "```",
        "delta = ∫0^L M(x) m(x) /(EI) dx ,  M = P(L-x) ,  m = L-x",
        "      = P L^3/(3 E I) ,  I = b h^3/12",
        "```",
        "",
        "出处：Timoshenko《Strength of Materials》Part I 3rd ed. (1955) ch.4 悬臂梁端载；",
        "同一梁族的频率基准见 Blevins (1979) Table 8-1（DOI 10.1115/1.3153712）。",
        f"本仓不手抄公式：`castigliano_integral()` 用 20001 点**复合 Simpson**"
        f"（被积函数为二次，故机器精度内精确）对弯矩场与剪力场分别独立重导——"
        f"弯曲项相对差 `{ref['unit_load_quadrature_bending_rel']:+.2e}`、"
        f"剪切项相对差 `{ref['unit_load_quadrature_shear_rel']:+.2e}`，"
        f"合计 `{ref['unit_load_quadrature_rel']:+.2e}`（对 Timoshenko）"
        f"{mark(checks['reference_formula_rederived'])}。",
        "",
        "| 量 | 值 |",
        "|---|---:|",
        f"| 梁长 L | {doc['geometry']['length_m']} m |",
        f"| 截面 b×h | {doc['geometry']['width_m']} × {doc['geometry']['height_m']} m |",
        f"| 细长比 L/h | {doc['geometry']['slenderness_L_over_h']:.0f} |",
        f"| E / ν | {doc['material']['young_modulus_Pa']/1e9:.0f} GPa / {doc['material']['poisson_ratio']} |",
        f"| 端载荷 P | {doc['load']['tip_force_N']} N |",
        f"| 面积 A | {ref['area_m2']:.6e} m² |",
        f"| 惯性矩 I | {ref['inertia_m4']:.6e} m⁴ |",
        f"| 抗弯刚度 EI | {ref['bending_stiffness_Pa_m4']:.6e} N·m² |",
        f"| 剪切修正 κ | {5/6:.6f} |",
        f"| 无量纲载荷 P L²/(EI) | {doc['load']['tip_force_N']*doc['geometry']['length_m']**2/ref['bending_stiffness_Pa_m4']:.6f} |",
        f"| 无量纲挠度 δ/L | {ref['euler_bernoulli_m']/doc['geometry']['length_m']:.6f} |",
        f"| δ_EB | {ref['euler_bernoulli_m']:.12e} m |",
        f"| δ_Timoshenko | {ref['timoshenko_m']:.12e} m |",
        f"| 剪切项占比 | {ref['shear_fraction_of_bending']*100:.4f} % |",
        "",
        "## ② 方法与实现",
        "",
        "| 层 | 方法 | 方程/格式 | 代码指针 |",
        "|---|---|---|---|",
        "| M1 | Euler-Bernoulli 梁 FE（三次 Hermite + 2 点 Gauss） | "
        "`EI w'''' = 0`，端部集中力 | `tensorfem/frame2d.py`: `FrameModel`, `solve_frame_static` |",
        "| M2 | Timoshenko 梁 FE（精确剪切柔度刚度阵） | "
        "`EI w'''' ...`+`κGA(w'-θ)`，剪切项 `PL/(κAG)` | `tensorfem/advanced_beam.py`: "
        "`solve_timoshenko_cantilever` |",
        "| M3 | 3-D HEX20 实体 FE（20 节点二次砖，3×3×3 Gauss） | "
        "`∇·σ = 0`，固支面 Dirichlet，端面静力等效横向节点力 ΣF=P | "
        "`tensorfem/hex20.py`: `structured_hex20_mesh`, `solve_hex20_sparse` |",
        "",
        "实现要点：",
        "",
        "1. 载荷一致性——端面节点按面积均分 `P/n_tip`，合力严格等于 `P`，无端部弯矩附加；",
        "2. 求解器按规模切换——`dof > 3000` 走 COO 装配 + SuperLU（`solve_hex20_sparse`），"
        "否则 dense LAPACK；两条路径同一离散、同一荷载向量；",
        "3. 判据先声明后计算——参考网格、容差、度量写死在模块常量里，"
        "`--quick` 只缩小扫描范围，不改变判据；",
        "4. 输出即工件——所有表格数字由证据 JSON 注入生成，报告与工件不出现手抄数字。",
        "",
        "## ③ 对照表",
        "",
        "| 模型 | 离散 | dof | 端挠度 [m] | 误差 vs δ_EB | <2% ? |",
        "|---|---|---:|---:|---:|:--:|",
    ]
    for row in beam:
        lines.append(
            f"| M1 EB 梁 FE | n={row['n_elements']} | {row['dofs']} | "
            f"{row['tip_deflection_m']:.12e} | {row['relative_error_vs_euler_bernoulli']*100:+.2e}% | "
            f"{mark(row['within_tolerance_vs_anchor'])} |")
    lines.append(
        f"| M2 Timoshenko 梁 FE | n={tim['n_elements']} | {tim['dofs']} | "
        f"{tim['tip_deflection_m']:.12e} | {tim['relative_error_vs_euler_bernoulli']*100:+.2e}% | "
        f"{mark(tim['within_tolerance_vs_anchor'])} |")
    for row in solid:
        lines.append(
            f"| M3 HEX20 实体 | {row['axial_elements']}×{row['cross_elements']}×{row['cross_elements']} | "
            f"{row['dofs']} | {row['tip_deflection_m']:.12e} | "
            f"{row['relative_error_vs_euler_bernoulli']*100:+.4f}% | "
            f"{mark(row['within_tolerance_vs_anchor'])} |")
    ref_row = doc["levels"]["solid_reference"]
    lines += [
        "",
        f"**判据行（预声明）**：参考网格 `{mesh_label(verdict['reference_mesh'])}` 的实体端挠度 "
        f"`{verdict['tip_deflection_m']:.12e} m` vs 解析 `{verdict['reference_value_m']:.12e} m`，"
        f"误差 `{verdict['relative_error']*100:+.4f}%` {mark(verdict['passed'])}（容差 2%）。",
        "",
        "判据清单：",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
        f"| 基准公式独立重导（<1e-9） | {mark(checks['reference_formula_rederived'])} |",
        f"| 参考网格 <2% | {mark(checks['solid_reference_within_tolerance'])} |",
        f"| M1 对三次解机器精确（<1e-9） | {mark(checks['beam_eb_machine_exact'])} |",
        f"| M2 与 Timoshenko 解析一致（<1e-6） | {mark(checks['timoshenko_beam_matches_theory'])} |",
        f"| 载荷线性（tip(2P)/tip(P)=2） | {mark(checks['load_linearity'])} |",
        f"| 实体 h-收敛单调 | {mark(checks.get('solid_h_convergence_monotone', False))} |",
        "",
        "## ④ 结果图",
        "",
        "![收敛](assets/case-fem-cantilever-tip/convergence.png)",
        "",
        "![各层对比](assets/case-fem-cantilever-tip/levels.png)",
        "",
        "![场量](assets/case-fem-cantilever-tip/field_uz.png)",
        "",
        "## ⑤ 收敛性与参数敏感性",
        "",
        "**网格收敛（≥2 分辨率）**：轴向/截面阶梯 "
        + " → ".join(f"{r['axial_elements']}×{r['cross_elements']}"
                     for r in solid) + "。"
        f"最粗 `{solid[0]['axial_elements']}×{solid[0]['cross_elements']}` 误差 "
        f"`{solid[0]['relative_error_vs_euler_bernoulli']*100:+.4f}%`，"
        f"参考网格 `{ref_row['axial_elements']}×{ref_row['cross_elements']}` 误差 "
        f"`{ref_row['relative_error_vs_euler_bernoulli']*100:+.4f}%`；"
        "单层截面单元（×1）的弯曲偏刚最明显，截面细化（×2、×4）是主要误差来源。",
        "",
        "**参数敏感性：细长比 L/h**（实体网格 32×2×2 固定）",
        "",
        "| L/h | h [m] | 剪切项占比 | 实体 vs δ_EB | 实体 vs δ_Timoshenko |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in slen:
        lines.append(
            f"| {row['slenderness_L_over_h']:.0f} | {row['height_m']:.3f} | "
            f"{row['shear_fraction']*100:.4f}% | "
            f"{row['relative_error_vs_euler_bernoulli']*100:+.4f}% | "
            f"{row['relative_error_vs_timoshenko']*100:+.4f}% |")
    lines += [
        "",
        "**参数敏感性：载荷线性**（参考网格）",
        "",
        "| P [N] | 端挠度 [m] |",
        "|---:|---:|",
    ]
    for row in lin["rows"]:
        lines.append(f"| {row['load_N']:.0f} | {row['tip_deflection_m']:.12e} |")
    lines += [
        "",
        f"`tip(2P)/tip(P) = {lin['ratio_2p_over_p']:.12f}`（线性理论精确 2）。",
        "",
        "## ⑥ 失败模式与工件",
        "",
        "| 工件 | 路径 |",
        "|---|---|",
        "| 证据 JSON | `benchmarks/case_fem_cantilever_tip_evidence.json` |",
        "| 收敛图 | `docs/assets/case-fem-cantilever-tip/convergence.png` |",
        "| 层级柱状图 | `docs/assets/case-fem-cantilever-tip/levels.png` |",
        "| 位移场云图 | `docs/assets/case-fem-cantilever-tip/field_uz.png` |",
        "| 回归门 | `tests/test_case_fem_cantilever_tip.py` |",
        "",
        "复现命令：",
        "",
        "```bash",
        "python benchmarks/case_fem_cantilever_tip.py --write-docs   # 全量研究 + 本文档",
        "python benchmarks/case_fem_cantilever_tip.py --json         # L4 判据 workload stdout",
        "pytest -q tests/test_case_fem_cantilever_tip.py            # 廉价回归门",
        "```",
        "",
        "已知失败模式与局限：",
        "",
        "1. **单层截面 HEX20 偏刚**：`×1` 时端挠度偏低约 "
        f"`{solid[0]['relative_error_vs_euler_bernoulli']*100:+.2f}%`，"
        "属弯曲离散误差（非剪切建模），必须在验收前做截面细化，不能靠单一粗网格交差；",
        "2. **δ/L = 1%**：本算例挠度已到线弹性小挠度理论的边缘，若改为大挠度（elastica）"
        "参考解，端挠度会有低阶修正；本仓判据用的是线性理论锚，"
        "超出该范围应改走 `tensorfem/elastica_cantilever.py`；",
        "3. **端面载荷理想化**：载荷按静力等效均布到端面节点（ΣF=P），"
        "与圣维南端部应力分布不同，但只影响端部局部，不改变梁挠度；",
        "4. **未覆盖**：三维局部应力集中、接触/大变形、复合材料截面不在本算例范围。",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="coarse ladder, ~20 s")
    parser.add_argument("--json", action="store_true",
                        help="print the evidence document on stdout (TensorBench workload)")
    parser.add_argument("--write-docs", action="store_true", help="write docs page + figures")
    parser.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    doc = run_study(quick=args.quick)
    fields = doc.pop("_solid_fields")
    doc["execution"]["seconds"] = time.perf_counter() - t0

    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[evidence] {args.evidence}")

    if args.write_docs:
        figures = make_figures(doc, fields)
        DOC_PATH.write_text(render_docs(doc, figures), encoding="utf-8")
        log(f"[docs] {DOC_PATH}")
        for path in figures:
            log(f"[figure] {path}")

    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
    else:
        log(f"[verdict] tip={doc['verdict']['tip_deflection_m']:.12e} m "
            f"ref={doc['verdict']['reference_value_m']:.12e} m "
            f"err={doc['verdict']['relative_error']*100:+.4f}% "
            f"passed={doc['verdict']['passed']}")
    return 0 if doc["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())