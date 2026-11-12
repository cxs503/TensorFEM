"""Robinson patch test A/B/C — constant-strain and linear-stress reproduction.

Why this module
---------------
The repository already carries several *ad hoc* patch assertions inside individual
benchmarks (``benchmarks/beam_foundation_unilateral.py`` C3,
``tests/test_continuum.py``, ``benchmarks/thermal_verification.py`` case 4), but there
is no reusable, published patch-test *suite* with a fixed mesh, a fixed load set and a
machine-checkable verdict.  This module supplies one so that any element family
(Q4 / CST, Q8 serendipity, TET10) can be qualified with the same code path.

The criterion
-------------
Robinson's patch test (Robinson, J., "An attempt to evaluate incompatible mode
elements", in *Finite Element Methods for Plate and Shell Structures*, Vol. 1:
Element Technology, Pineridge Press, 1987, pp. 1-15; the same criterion is stated in
Zienkiewicz & Taylor, *The Finite Element Method*, Vol. 1, §3.6 "The patch test") is:

    a patch of elements, with an irregular (non-affine) mesh and at least one
    interior node, is loaded only through the *boundary* by a displacement field
    that is representable by the element; the solved interior field and the
    recovered stresses must then equal the exact field to round-off.

The load states used here are the three standard ones:

* **A** — uniaxial tension, constant stress ``sigma_xx = sigma0`` (constant strain);
* **B** — pure shear, constant stress ``tau_xy = tau0`` (constant strain);
* **C** — pure bending, *linearly varying* ``sigma_xx = -E*kappa*y`` (linear stress).

Case C is the discriminating one: constant-strain tests are passed by every
conforming isoparametric family, so they cannot separate good from mediocre elements.
Case C is exactly reproducible only by a family whose span contains the complete
quadratic field; on a *distorted* (non-affine) mesh the 8-node serendipity element is
known to miss it (its span omits the ``xi^2*eta^2`` term), which this module measures
rather than asserts.

Target values
-------------
Every reference value in this module is the *analytical* field written down in
``exact_*`` below (stress derived from the prescribed displacement by the constitutive
matrix, i.e. ``sigma = D B u``), so no number is copied from a table.  The Robinson
paper supplies the *criterion*, not a numerical target; the derivation of each target
is printed in ``docs/patch-test-robinson.md``.

Conventions
-----------
Plane-stress 2-D meshes live on ``[0, L] x [0, L]`` with thickness ``t``; the 3-D box is
``[0, L]^3``.  Boundary nodes are identified geometrically (coordinates on the
boundary), interior nodes are the only free unknowns; this keeps the Dirichlet data
independent of the node numbering, which is itself verified.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

import torch

from .continuum import ContinuumModel, solve_continuum
from .quadratic_plane import (
    GAUSS_POINTS_3X3,
    q8_mesh,
    q8_shape,
    q8_stress_at,
    solve_q8_plane,
)
from .quadratic_solid import Tet10Model, solve_tet10, tet4_to_tet10
from .solid3d import hex_to_tet_mesh, structured_hex_mesh

#: Material of the patch (steel, MPa / mm).  Kept in one place so the report and the
#: tests cannot drift apart.
YOUNG = 210000.0
POISSON = 0.3
THICKNESS = 1.0
LENGTH = 1.0

#: Load amplitudes of cases A, B and C.
SIGMA0 = 100.0  # uniaxial tension, MPa
TAU0 = 50.0  # pure shear, MPa
KAPPA = 1.0e-3  # bending curvature, 1/mm

#: A patch test is "exact" when the reproduction error is at round-off level.
EXACT_TOL = 1e-10


def shear_modulus() -> float:
    return YOUNG / (2.0 * (1.0 + POISSON))


# --------------------------------------------------------------------------- #
# meshes
# --------------------------------------------------------------------------- #
def _distortion(u: float, v: float, h: float, amp: float) -> tuple[float, float]:
    """Deterministic, smooth interior-node distortion (no RNG: bit-reproducible)."""
    du = (
        amp
        * h
        * (
            0.35 * math.sin(3.0 * math.pi * u) * math.cos(2.0 * math.pi * v)
            + 0.15 * math.sin(5.0 * math.pi * v)
        )
    )
    dv = (
        amp
        * h
        * (
            0.30 * math.cos(2.0 * math.pi * u) * math.sin(3.0 * math.pi * v)
            + 0.12 * math.sin(4.0 * math.pi * u)
        )
    )
    return du, dv


def q4_patch_mesh(
    nx: int = 3,
    ny: int | None = None,
    distortion: float = 1.0,
    length: float = LENGTH,
    *,
    dtype=torch.float64,
):
    """Irregular Q4 patch on ``[0, L]^2``: boundary nodes exact, interior nodes moved.

    ``distortion = 0`` gives the affine (uniform) grid, which is the control case.
    Raises if distortion inverts an element, so a study cannot silently use a
    meaningless mesh.
    """
    ny = nx if ny is None else ny
    hx, hy = length / nx, length / ny
    pts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = i * hx, j * hy
            if 0 < i < nx and 0 < j < ny:
                du, dv = _distortion(x / length, y / length, min(hx, hy), distortion)
                x, y = x + du, y + dv
            pts.append((x, y))
    nodes = torch.tensor(pts, dtype=dtype)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    elements = torch.tensor(cells, dtype=torch.long)
    _assert_positive_jacobian_q4(nodes, elements)
    return nodes, elements


def _assert_positive_jacobian_q4(nodes: torch.Tensor, elements: torch.Tensor) -> None:
    x = nodes[elements]
    # corner triangles must all be positively oriented (counter-clockwise quads)
    for a, b, c in ((0, 1, 2), (0, 2, 3), (0, 1, 3), (1, 2, 3)):
        cr = (x[:, b, 0] - x[:, a, 0]) * (x[:, c, 1] - x[:, a, 1]) - (x[:, b, 1] - x[:, a, 1]) * (
            x[:, c, 0] - x[:, a, 0]
        )
        if torch.any(cr <= 0):
            raise ValueError("distortion inverted a Q4 element (negative Jacobian)")


def q8_patch_mesh(nx: int = 3, distortion: float = 1.0, length: float = LENGTH):
    """Q8 serendipity patch: Q4 patch upgraded with shared mid-edge nodes."""
    nodes, cells = q4_patch_mesh(nx, nx, distortion, length)
    return q8_mesh(nodes, cells)


def tet10_patch_mesh(nx: int = 2, distortion: float = 1.0, length: float = LENGTH):
    """Distorted TET10 patch on ``[0, L]^3`` (Kuhn six-tet split, conforming)."""
    nodes, hexes = structured_hex_mesh(length, length, length, nx, nx, nx)
    pts = nodes.clone()
    h = length / nx
    for k in range(nx + 1):
        for j in range(nx + 1):
            for i in range(nx + 1):
                if 0 < i < nx and 0 < j < nx and 0 < k < nx:
                    idx = k * (nx + 1) ** 2 + j * (nx + 1) + i
                    du, dv = _distortion(i / nx, j / nx, h, distortion)
                    pts[idx] = pts[idx] + torch.tensor([du, dv, 0.6 * dv])
    tet4_nodes, tet4 = hex_to_tet_mesh(pts, hexes)
    return tet4_to_tet10(tet4_nodes, tet4)


def boundary_nodes_2d(nodes: torch.Tensor, length: float = LENGTH, tol: float = 1e-12):
    on = (
        (nodes[:, 0].abs() < tol)
        | ((nodes[:, 0] - length).abs() < tol)
        | (nodes[:, 1].abs() < tol)
        | ((nodes[:, 1] - length).abs() < tol)
    )
    return on.nonzero().reshape(-1)


def boundary_nodes_3d(nodes: torch.Tensor, length: float = LENGTH, tol: float = 1e-12):
    on = torch.zeros(len(nodes), dtype=torch.bool)
    for d in range(3):
        on |= (nodes[:, d].abs() < tol) | ((nodes[:, d] - length).abs() < tol)
    return on.nonzero().reshape(-1)


# --------------------------------------------------------------------------- #
# exact fields
# --------------------------------------------------------------------------- #
def exact_field(case: str, nodes: torch.Tensor):
    """Return ``(u_exact, sigma_exact_fn, sigma_scale, ex, ey, ez)`` for a case.

    ``sigma_exact_fn`` maps a point (2-D or 3-D) to the exact Voigt stress so that
    probes at Gauss points and centroids are compared at the *same* location.
    """
    if case == "A":
        u = torch.stack((SIGMA0 / YOUNG * nodes[:, 0], -POISSON * SIGMA0 / YOUNG * nodes[:, 1]), 1)
        ex, ey, ez = SIGMA0, 0.0, 0.0

        def fn(p):
            return torch.tensor([SIGMA0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=p.dtype)

        return u, fn, SIGMA0, ex, ey, ez
    if case == "B":
        g = TAU0 / shear_modulus()
        u = torch.stack((0.5 * g * nodes[:, 1], 0.5 * g * nodes[:, 0]), 1)

        def fn(p):
            return torch.tensor([0.0, 0.0, TAU0, 0.0, 0.0, 0.0], dtype=p.dtype)

        return u, fn, TAU0, 0.0, TAU0, 0.0
    if case in ("C", "C3"):
        if case == "C":
            u = torch.stack(
                (
                    -KAPPA * nodes[:, 0] * nodes[:, 1],
                    0.5 * KAPPA * (nodes[:, 0] ** 2 + POISSON * nodes[:, 1] ** 2),
                ),
                1,
            )
        else:  # 3-D: sigma_xx = -E*kappa*y, all other components zero
            u = torch.stack(
                (
                    -KAPPA * nodes[:, 0] * nodes[:, 1],
                    0.5
                    * KAPPA
                    * (nodes[:, 0] ** 2 + POISSON * nodes[:, 1] ** 2 - POISSON * nodes[:, 2] ** 2),
                    POISSON * KAPPA * nodes[:, 1] * nodes[:, 2],
                ),
                1,
            )
        ex, ey, ez = YOUNG * KAPPA, 0.0, 0.0

        # exact stress amplitude grows linearly with y; the error is normalised by the
        # amplitude at the top surface so the number is a fraction of the load range.
        def fn(p):
            return torch.cat([-YOUNG * KAPPA * p[1].reshape(1), torch.zeros(5, dtype=p.dtype)])

        return u, fn, YOUNG * KAPPA * LENGTH, ex, ey, ez
    raise ValueError(f"unknown patch case {case!r}")


def exact_energy(case: str, length: float = LENGTH) -> float:
    """Analytical strain energy of the patch (used as an energy-level cross-check)."""
    if case == "A":
        return 0.5 * SIGMA0**2 / YOUNG * length**3
    if case == "B":
        return 0.5 * TAU0**2 / shear_modulus() * length**3
    if case == "C":
        return 0.5 * YOUNG * KAPPA**2 * length**5 / 3.0
    if case == "C3":
        return 0.5 * YOUNG * KAPPA**2 * length**5 / 3.0
    raise ValueError(case)


# --------------------------------------------------------------------------- #
# case runners
# --------------------------------------------------------------------------- #
@dataclass
class PatchResult:
    case: str
    family: str
    mesh: str
    cells: int
    nodes: int
    free_dofs: int
    disp_error: float  # max |u_h - u_ex| / max |u_ex|
    stress_error: float  # max |sigma_h - sigma_ex| / sigma_scale
    energy_error: float
    expected: str  # "pass" | "documented_defect" | "negative_control"
    probes: list = field(default_factory=list)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.disp_error <= EXACT_TOL and self.stress_error <= EXACT_TOL


def _finish(
    case,
    family,
    mesh_name,
    n_nodes,
    free_dofs,
    u,
    u_ex,
    stress_err,
    energy_h,
    expected,
    probes,
    notes="",
):
    disp_err = float((u.reshape(-1, u_ex.shape[1]) - u_ex).abs().max())
    scale = float(u_ex.abs().max())
    energy_err = abs(energy_h - exact_energy(case, LENGTH)) / exact_energy(case, LENGTH)
    return PatchResult(
        case=case,
        family=family,
        mesh=mesh_name,
        nodes=n_nodes,
        cells=probes and len(probes) or 0,
        free_dofs=free_dofs,
        disp_error=disp_err / scale,
        stress_error=stress_err,
        energy_error=float(energy_err),
        expected=expected,
        probes=probes,
        notes=notes,
    )


def run_q4_patch(case: str = "A", nx: int = 3, distortion: float = 1.0) -> PatchResult:
    """Solve one patch-test case with bilinear Q4 elements."""
    nodes, elements = q4_patch_mesh(nx, nx, distortion)
    u_ex, sig_fn, scale, _, _, _ = exact_field(case, nodes)
    bnd = boundary_nodes_2d(nodes)
    fixed = torch.stack([2 * bnd + 0, 2 * bnd + 1], 1).reshape(-1)
    values = torch.stack([u_ex[bnd, 0], u_ex[bnd, 1]], 1).reshape(-1)
    model = ContinuumModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor(YOUNG),
        poisson_ratio=torch.tensor(POISSON),
        thickness=torch.tensor(THICKNESS),
        forces=torch.zeros(2 * len(nodes)),
        fixed_dofs=fixed,
        element_type="q4",
        plane="stress",
        prescribed_values=values,
    )
    res = solve_continuum(model)
    # Q4 stress from solve_continuum is the (single) element-centre value.
    centres = nodes[elements].mean(1)
    exact = torch.stack([sig_fn(c)[:3] for c in centres])
    err = float((res.stress - exact).abs().max()) / scale
    probes = [
        {
            "centre": c.tolist(),
            "sigma_xx_h": float(s[0]),
            "sigma_xx_exact": float(e[0]),
            "tau_xy_h": float(s[2]),
            "tau_xy_exact": float(e[2]),
        }
        for c, s, e in zip(centres, res.stress, exact)
    ]
    expected = "pass" if case in ("A", "B") else "negative_control"
    out = _finish(
        case,
        "Q4",
        f"distortion={distortion:g}",
        len(nodes),
        len(nodes) * 2 - len(fixed),
        res.displacement,
        u_ex,
        err,
        float(res.strain_energy),
        expected,
        probes,
    )
    out.cells = len(elements)
    return out


def run_q8_patch(case: str = "C", nx: int = 3, distortion: float = 1.0) -> PatchResult:
    """Solve one patch-test case with 8-node serendipity elements (3x3 Gauss)."""
    nodes, elements = q8_patch_mesh(nx, distortion)
    u_ex, sig_fn, scale, _, _, _ = exact_field(case, nodes)
    bnd = boundary_nodes_2d(nodes)
    dirichlet = {}
    for n in bnd.tolist():
        dirichlet[2 * n] = float(u_ex[n, 0])
        dirichlet[2 * n + 1] = float(u_ex[n, 1])
    res = solve_q8_plane(
        nodes,
        elements,
        young=YOUNG,
        poisson=POISSON,
        thickness=THICKNESS,
        plane="stress",
        dirichlet=dirichlet,
    )
    probes, worst = [], 0.0
    for e in range(len(elements)):
        coords = nodes[elements[e]]
        for xi, eta in GAUSS_POINTS_3X3:
            s = q8_stress_at(
                nodes, elements, res.displacement, e, xi, eta, young=YOUNG, poisson=POISSON
            )
            point = q8_shape(xi, eta) @ coords
            exact = sig_fn(point)[:3]
            err = float((s - exact).abs().max()) / scale
            if err > worst:
                worst, worst_probe = (
                    err,
                    {
                        "element": e,
                        "xi": xi,
                        "eta": eta,
                        "y": float(point[1]),
                        "sigma_xx_h": float(s[0]),
                        "sigma_xx_exact": float(sig_fn(point)[0]),
                        "sigma_yy_h": float(s[1]),
                        "tau_xy_h": float(s[2]),
                        "rel_error": err,
                    },
                )
    probes.append(worst_probe)
    expected = "pass" if case in ("A", "B") or distortion == 0.0 else "documented_defect"
    # ``dirichlet`` holds one entry per constrained *dof*, so it is subtracted once
    # (the pre-fix expression used ``2 * len(dirichlet)`` and reported -16 free dofs
    # for the 3x3 Q8 patch: a reporting bug, not a solver effect).
    out = _finish(
        case,
        "Q8",
        f"distortion={distortion:g}",
        len(nodes),
        2 * len(nodes) - len(dirichlet),
        res.displacement,
        u_ex,
        worst,
        float(res.strain_energy),
        expected,
        probes,
        notes="serendipity span omits xi^2*eta^2 -> distorted quadratic patch",
    )
    out.cells = len(elements)
    return out


def run_tet10_patch(nx: int = 2, distortion: float = 1.0) -> PatchResult:
    """Solve the 3-D linear-stress patch (case C3) with TET10 elements."""
    nodes, elements = tet10_patch_mesh(nx, distortion)
    u_ex, sig_fn, scale, _, _, _ = exact_field("C3", nodes)
    bnd = boundary_nodes_3d(nodes)
    fixed = torch.stack([3 * bnd + 0, 3 * bnd + 1, 3 * bnd + 2], 1).reshape(-1)
    values = torch.stack([u_ex[bnd, 0], u_ex[bnd, 1], u_ex[bnd, 2]], 1).reshape(-1)
    model = Tet10Model(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor(YOUNG),
        poisson_ratio=torch.tensor(POISSON),
        forces=torch.zeros(3 * len(nodes)),
        fixed_dofs=fixed,
        prescribed_values=values,
    )
    res = solve_tet10(model)
    corners = nodes[elements[:, :4]].mean(1)  # centroid probe location
    exact = torch.stack([sig_fn(c) for c in corners])
    stress_err = float((res.stress[:, :3] - exact[:, :3]).abs().max()) / scale
    probes = [
        {"centroid": c.tolist(), "sigma_xx_h": float(s[0]), "sigma_xx_exact": float(e[0])}
        for c, s, e in zip(corners, res.stress, exact)
    ]
    out = _finish(
        "C3",
        "TET10",
        f"distortion={distortion:g}",
        len(nodes),
        3 * len(nodes) - len(fixed),
        res.displacement,
        u_ex,
        stress_err,
        float(res.strain_energy),
        "pass",
        probes,
    )
    out.cells = len(elements)
    return out


# --------------------------------------------------------------------------- #
# studies
# --------------------------------------------------------------------------- #
def distortion_study(nx: int = 3, amplitudes=(0.0, 0.5, 1.0, 1.5)):
    """Stress error of Q8 vs interior-node distortion, for cases A, B and C."""
    rows = []
    for amp in amplitudes:
        row = {"distortion": amp}
        for case in ("A", "B", "C"):
            r = run_q8_patch(case, nx, amp)
            row[f"{case}_stress_error"] = r.stress_error
            row[f"{case}_disp_error"] = r.disp_error
        rows.append(row)
    return rows


def refinement_study(sizes=(2, 3, 4, 6), distortion: float = 1.0):
    """h-refinement of case C for Q4, affine Q8 and distorted Q8: error and rate."""
    rows = []
    for n in sizes:
        q4 = run_q4_patch("C", n, distortion)
        q8a = run_q8_patch("C", n, 0.0)
        q8d = run_q8_patch("C", n, distortion)
        rows.append(
            {
                "h": LENGTH / n,
                "cells": q4.cells,
                "Q4_stress_error": q4.stress_error,
                "Q8_affine_stress_error": q8a.stress_error,
                "Q8_distorted_stress_error": q8d.stress_error,
            }
        )
    for key in ("Q4_stress_error", "Q8_distorted_stress_error"):
        rates = []
        for a, b in zip(rows, rows[1:]):
            ea, eb = a[key], b[key]
            if ea > 0 and eb > 0:
                rates.append(math.log(ea / eb) / math.log(a["h"] / b["h"]))
            else:
                rates.append(None)
        for a, p in zip(rows[1:], rates):
            a[key.replace("_error", "_rate")] = p
    return rows


def numbering_study(case: str = "A", nx: int = 3, distortion: float = 1.0):
    """Permute element/grid node numbering; the patch answer must not change."""
    import random

    base = run_q4_patch(case, nx, distortion)
    nodes, elements = q4_patch_mesh(nx, nx, distortion)
    rng = random.Random(20261112)
    perm = list(range(len(nodes)))
    rng.shuffle(perm)  # new->old map
    inv = [0] * len(nodes)
    for new, old in enumerate(perm):
        inv[old] = new
    nodes2 = nodes[torch.tensor(perm)]
    elements2 = torch.tensor([[inv[int(n)] for n in cell] for cell in elements])
    u_ex, sig_fn, scale, _, _, _ = exact_field(case, nodes2)
    bnd = boundary_nodes_2d(nodes2)
    fixed = torch.stack([2 * bnd, 2 * bnd + 1], 1).reshape(-1)
    values = torch.stack([u_ex[bnd, 0], u_ex[bnd, 1]], 1).reshape(-1)
    model = ContinuumModel(
        nodes=nodes2,
        elements=elements2,
        young_modulus=torch.tensor(YOUNG),
        poisson_ratio=torch.tensor(POISSON),
        thickness=torch.tensor(THICKNESS),
        forces=torch.zeros(2 * len(nodes2)),
        fixed_dofs=fixed,
        element_type="q4",
        plane="stress",
        prescribed_values=values,
    )
    res = solve_continuum(model)
    err = float((res.displacement.reshape(-1, 2) - u_ex).abs().max()) / float(u_ex.abs().max())
    return {
        "case": case,
        "base_disp_error": base.disp_error,
        "permuted_disp_error": err,
        "identical": abs(err - base.disp_error) <= 1e-12,
    }


def run_suite() -> dict:
    """Full A/B/C suite plus the three sensitivity studies."""
    cases = [
        run_q4_patch("A"),
        run_q4_patch("B"),
        run_q4_patch("C"),
        run_q8_patch("A"),
        run_q8_patch("B"),
        run_q8_patch("C", 3, 0.0),
        run_q8_patch("C", 3, 1.0),
        run_tet10_patch(2, 1.0),
    ]
    return {
        "material": {
            "E": YOUNG,
            "nu": POISSON,
            "thickness": THICKNESS,
            "L": LENGTH,
            "sigma0": SIGMA0,
            "tau0": TAU0,
            "kappa": KAPPA,
            "G": shear_modulus(),
        },
        "exact_tolerance": EXACT_TOL,
        "cases": [dict(c.__dict__, passed=c.passed) for c in cases],
        "distortion_study": distortion_study(),
        "refinement_study": refinement_study(),
        "numbering_study": numbering_study(),
    }


# --------------------------------------------------------------------------- #
# precision guard
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def double_precision():
    """Force ``torch.float64`` for the enclosed block and restore the caller's dtype.

    Exactness is a round-off statement: at the float32 default the round-off alone
    is ~1e-7, so an entry point that did not pin the dtype would measure torch's
    default instead of the element.  Every entry (CLI, registry, driver) goes
    through this guard.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# --------------------------------------------------------------------------- #
# machine-checkable qualification entry
# --------------------------------------------------------------------------- #
#: ``(id, case, family, distortion, expectation)``.  ``expectation`` is what a
#: *correct* element family must do, so the gate is able to fail: an element that
#: passed everything (including case C with bilinear Q4) would mean the test lost
#: its discriminating power, not that the element improved.
QUALIFICATION_CASES = (
    ("patch.A.Q4", "A", "Q4", 1.0, "exact"),
    ("patch.B.Q4", "B", "Q4", 1.0, "exact"),
    ("patch.C.Q4", "C", "Q4", 1.0, "must_fail"),
    ("patch.A.Q8", "A", "Q8", 1.0, "exact"),
    ("patch.B.Q8", "B", "Q8", 1.0, "exact"),
    ("patch.C.Q8_affine", "C", "Q8", 0.0, "exact"),
    ("patch.C.Q8_distorted", "C", "Q8", 1.0, "must_fail"),
    ("patch.C3.TET10", "C3", "TET10", 1.0, "exact"),
)


def _run_one(case: str, family: str, distortion: float):
    if family == "Q4":
        return run_q4_patch(case, 3, distortion)
    if family == "Q8":
        return run_q8_patch(case, 3, distortion)
    if family == "TET10":
        return run_tet10_patch(2, distortion)
    raise ValueError(f"unknown element family {family!r}")


def qualification_report() -> dict:
    """Run the A/B/C patch cases and return the verdict bundle.

    This is the single entry point shared by the CLI (``tensorfem patch-test``)
    and the test suite, so the number a user sees on the command line is the
    number the gate asserts.
    """
    with double_precision():
        return _qualification_report()


def _qualification_report() -> dict:
    cases = []
    for cid, case, family, distortion, expectation in QUALIFICATION_CASES:
        r = _run_one(case, family, distortion)
        exact = r.disp_error <= EXACT_TOL and r.stress_error <= EXACT_TOL
        ok = exact if expectation == "exact" else not exact
        cases.append(
            {
                "id": cid,
                "case": case,
                "family": family,
                "distortion": distortion,
                "mesh": r.mesh,
                "cells": r.cells,
                "nodes": r.nodes,
                "free_dofs": r.free_dofs,
                "expectation": expectation,
                "disp_error": r.disp_error,
                "stress_error": r.stress_error,
                "energy_error": r.energy_error,
                "tolerance": EXACT_TOL,
                "exact": exact,
                "passed": ok,
            }
        )
    return {
        "schema": "tensorfem.patch_test.qualification/1",
        "reference": "Robinson (1987) patch-test criterion: constant-strain and "
        "linear-stress fields reproduced to round-off on an irregular "
        "patch with free interior nodes; targets derived analytically "
        "from sigma = D B u (no tabulated values copied)",
        "tolerance": EXACT_TOL,
        "material": {
            "E": YOUNG,
            "nu": POISSON,
            "thickness": THICKNESS,
            "L": LENGTH,
            "sigma0": SIGMA0,
            "tau0": TAU0,
            "kappa": KAPPA,
            "G": shear_modulus(),
        },
        "cases": cases,
        "summary": {"total": len(cases), "passed": sum(c["passed"] for c in cases)},
        "passed": all(c["passed"] for c in cases),
    }
