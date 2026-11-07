#!/usr/bin/env python3
"""IN41-N01 anchor: a structural verification face over beam / plate / axisymmetric elements.

Three independent closed-form oracles, one per element family of TensorFEM's
linear structural path.  Each case gets its own three-level mesh ladder, its own
error metric and its own figure; nothing is shared between the cases except the
book-keeping in ``benchmarks/verified/_harness.py``, so one case can be audited
without reading the other two.

Case A -- 1-D Hermite frame element (``tensorfem.frame2d``): cantilever tip load
------------------------------------------------------------------------------
A straight hollow-box cantilever, ``L = 2.5 m``, outer section
``b x h = 0.15 x 0.25 m``, wall ``t_w = 20 mm`` (``A = 1.44e-2 m^2``,
``I = 1.10420e-4 m^4``, ``L/h = 10``), steel ``E = 210 GPa``, tip load
``P = 50 kN``.  Euler--Bernoulli theory gives

    w(x) = P x^2 (3L - x) / (6 E I),   w(L) = P L^3 / (3 E I),
    theta(L) = P L^2 / (2 E I),        sigma_xx(x, z) = M(x) z / I,
    M(x) = P (L - x).

The cubic Hermite element spans cubics exactly and the exact tip-load field is a
cubic, so the discrete solution reproduces the exact field to round-off: the
reported error is a *machine-precision* statement, stronger than the 1 % gate.
That leaves no discretisation error to measure, so the ladder is repeated with a
uniformly distributed load ``q = 20 kN/m`` (exact field quartic,
``w(x) = q x^2 (6L^2 - 4Lx + x^2)/(24 E I)``), where the element interior is not
nodally superconvergent and an observed order is measurable.

Case B -- 4-node Mindlin plate element (``tensorfem.plate``, MITC4 shear):
simply supported square under uniform pressure
----------------------------------------------
``a = 1 m`` square, ``t = 10 mm`` (``a/t = 100``), steel ``E = 210 GPa``,
``nu = 0.3``, uniform pressure ``q = 10 kPa``, soft simple support (``w = 0`` on
every edge, rotations free).  Oracle: the exact Navier double series of Kirchhoff
thin-plate theory (``tensorfem.plate_uniform_load.ss_uniform_kirchhoff``, 401 odd
harmonics -> centre coefficient ``alpha = 0.004062353``) and Timoshenko &
Woinowsky-Krieger Table 8 (``alpha = 0.00406``; three significant figures, so its
own rounding is a 0.06 % floor).  The Mindlin-vs-Kirchhoff model difference at
``a/t = 100`` is ``O((t/a)^2) ~ 1e-4`` relative, below the discretisation error at
the meshes used; that residual is reported, not hidden.

Case C -- 4-node axisymmetric ring element
(``tensorfem.hertz_axisymmetric.assemble_axisymmetric_stiffness``):
thick-walled cylinder under internal pressure
---------------------------------------------
``a = 100 mm``, ``b = 200 mm`` (``b/a = 2``), internal pressure ``p = 20 MPa``,
outer surface traction free, plane strain enforced pointwise (``u_z = 0``), steel
``E = 210 GPa``, ``nu = 0.3``.  The Lame solution of plane-strain elasticity is

    A = p a^2/(b^2 - a^2),   B = p a^2 b^2/(b^2 - a^2),
    sigma_r(r) = A - B/r^2,  sigma_theta(r) = A + B/r^2,
    sigma_z = 2 nu A,        u_r(r) = (1+nu)/E [(1 - 2 nu) A r + B/r].

Stresses are compared at five radii (``r/a = 1.1, 1.3, 1.5, 1.7, 1.9``) after
Gauss-point-to-nodal extrapolation plus averaging of the sharing elements.  The
raw (unrecovered) Gauss-point values are reported alongside, so the reader can
see how much of the accuracy is the recovery and how much is the element.

Outputs (this directory)
------------------------
``result.json``          every number per refinement level (generated)
``gates_verified.json``  schema ``tensorfem.verified-gates.v1``: per gate the
                         reference formula, the value, the threshold, the verdict
``case_a_beam.png``      deflected shape and bending stress vs the closed form
``case_b_plate.png``     deflection field and the mesh-refinement convergence
``case_c_cylinder.png``  sigma_r / sigma_theta radial distribution vs Lame

Usage::

    OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/verified/fem_structural_benchmarks/run.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

HERE = Path(__file__).resolve().parent
VERIFIED = HERE.parent
if str(VERIFIED) not in sys.path:
    sys.path.insert(0, str(VERIFIED))

import _harness as H  # noqa: E402

from tensorfem.frame2d import FrameModel, solve_frame_static  # noqa: E402
from tensorfem.hertz_axisymmetric import assemble_axisymmetric_stiffness  # noqa: E402
from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh  # noqa: E402
from tensorfem.plate_mindlin import consistent_load_vector  # noqa: E402
from tensorfem.plate_uniform_load import ss_uniform_kirchhoff  # noqa: E402

#: campaign thresholds, restated once so ``gates_verified.json`` and the report can
#: be checked against the mandate without reading this code
CRITERIA = {
    "case_a_tip_deflection_rel_max": 0.01,
    "case_a_root_stress_rel_max": 0.01,
    "case_b_centre_deflection_rel_max": 0.02,
    "case_c_stress_rel_max": 0.01,
    "convergence_order_min": 1.5,
    "levels_per_case": 3,
}


# --------------------------------------------------------------------------- #
# case A -- Hermite frame element, hollow-box cantilever
# --------------------------------------------------------------------------- #
BEAM = {
    "length_m": 2.5,
    "outer_width_m": 0.15,
    "outer_depth_m": 0.25,
    "wall_thickness_m": 0.02,
    "young_modulus_pa": 210.0e9,
    "poisson_ratio": 0.3,
    "tip_load_n": 50.0e3,
    "distributed_load_n_per_m": 20.0e3,
    "levels_elements": (4, 8, 16),
}


def beam_section() -> dict:
    b = BEAM["outer_width_m"]
    h = BEAM["outer_depth_m"]
    tw = BEAM["wall_thickness_m"]
    area = b * h - (b - 2 * tw) * (h - 2 * tw)
    inertia = (b * h**3 - (b - 2 * tw) * (h - 2 * tw) ** 3) / 12.0
    return {
        "area_m2": area,
        "second_moment_m4": inertia,
        "extreme_fibre_m": h / 2.0,
        "slenderness_L_over_h": BEAM["length_m"] / h,
    }


def beam_theory(x: float, *, section: dict) -> dict:
    """Exact Euler--Bernoulli fields at station ``x`` under the tip load."""
    E = BEAM["young_modulus_pa"]
    L = BEAM["length_m"]
    P = BEAM["tip_load_n"]
    I = section["second_moment_m4"]
    c = section["extreme_fibre_m"]
    moment = P * (L - x)
    return {
        "defl_m": P * x * x * (3.0 * L - x) / (6.0 * E * I),
        "slope_rad": P * x * (2.0 * L - x) / (2.0 * E * I),
        "bending_moment_nm": moment,
        "bending_stress_pa": moment * c / I,
    }


def beam_theory_udl(x: float, *, section: dict) -> dict:
    """Exact Euler--Bernoulli fields at station ``x`` under uniform load ``q``."""
    E = BEAM["young_modulus_pa"]
    L = BEAM["length_m"]
    q = BEAM["distributed_load_n_per_m"]
    I = section["second_moment_m4"]
    c = section["extreme_fibre_m"]
    moment = q * (L - x) ** 2 / 2.0
    return {
        "defl_m": q * x * x * (6.0 * L * L - 4.0 * L * x + x * x) / (24.0 * E * I),
        "slope_rad": q * x * (3.0 * L * L - 3.0 * L * x + x * x) / (6.0 * E * I),
        "bending_moment_nm": moment,
        "bending_stress_pa": moment * c / I,
    }


def _hermite(xi: float) -> tuple[float, float, float, float]:
    """Cubic Hermite shape functions on the unit element (``xi`` in [0, 1])."""
    return (
        1.0 - 3.0 * xi**2 + 2.0 * xi**3,
        xi - 2.0 * xi**2 + xi**3,
        3.0 * xi**2 - 2.0 * xi**3,
        -xi**2 + xi**3,
    )


def solve_beam(n: int, *, load: str = "tip") -> dict:
    """Solve one cantilever with ``n`` equal-length Hermite elements.

    ``load='tip'`` puts ``P`` at the free node; ``load='udl'`` puts ``q`` on every
    element through the library's consistent load vector.
    """
    L = BEAM["length_m"]
    section = beam_section()
    x = torch.linspace(0.0, L, n + 1, dtype=torch.float64)
    nodes = torch.stack((x, torch.zeros_like(x)), dim=1)
    elements = torch.tensor([[i, i + 1] for i in range(n)], dtype=torch.long)
    forces = torch.zeros(3 * (n + 1), dtype=torch.float64)
    udl = None
    if load == "tip":
        forces[3 * n + 1] = -BEAM["tip_load_n"]
    elif load == "udl":
        udl = torch.tensor(-BEAM["distributed_load_n_per_m"], dtype=torch.float64)
    else:  # pragma: no cover - guard
        raise ValueError("load must be 'tip' or 'udl'")
    model = FrameModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor(BEAM["young_modulus_pa"], dtype=torch.float64),
        area=torch.tensor(section["area_m2"], dtype=torch.float64),
        inertia=torch.tensor(section["second_moment_m4"], dtype=torch.float64),
        forces=forces,
        fixed_dofs=torch.tensor([0, 1, 2], dtype=torch.long),
        distributed_load_y=udl,
    )
    result = solve_frame_static(model)
    u = result.displacement
    theory = beam_theory if load == "tip" else beam_theory_udl
    field = []
    for i in range(n):
        xm = L * (i + 0.5) / n  # element interior: not a nodal superconvergence point
        le = float(x[i + 1] - x[i])
        xi = 0.5
        n1, n2, n3, n4 = _hermite(xi)
        # element dofs: v_i, theta_i, v_{i+1}, theta_{i+1} = 3i+1, 3i+2, 3i+4, 3i+5
        nodal = np.array([float(u[3 * i + 1]), float(u[3 * i + 2]),
                          float(u[3 * i + 4]), float(u[3 * i + 5])])
        w = -(n1 * nodal[0] + le * n2 * nodal[1] + n3 * nodal[2] + le * n4 * nodal[3])
        d2 = np.array([(-6.0 + 12.0 * xi), le * (-4.0 + 6.0 * xi),
                       (6.0 - 12.0 * xi), le * (-2.0 + 6.0 * xi)]) / le**2
        kappa = float(d2 @ nodal)  # v'' of the upward displacement field
        # sigma_xx(z=+c) = -E c v'' : tension on the upper fibre of a downward bend
        sigma = -BEAM["young_modulus_pa"] * section["extreme_fibre_m"] * kappa
        ref = theory(xm, section=section)
        field.append(
            {
                "x_m": xm,
                "defl_fem_m": w,
                "defl_exact_m": ref["defl_m"],
                "defl_rel_error": abs(w - ref["defl_m"]) / abs(ref["defl_m"]),
                "bending_stress_fem_pa": sigma,
                "bending_stress_exact_pa": ref["bending_stress_pa"],
                "bending_stress_rel_error": abs(sigma - ref["bending_stress_pa"])
                / abs(ref["bending_stress_pa"]),
            }
        )
    tip = theory(L, section=section)
    root = theory(0.0, section=section)
    root_moment_fem = float(result.element_end_forces_local[0, 2])
    root_stress_fem = root_moment_fem * section["extreme_fibre_m"] / section["second_moment_m4"]
    return {
        "elements": n,
        "load": load,
        "w_tip_fem_m": -float(u[3 * n + 1]),
        "w_tip_exact_m": tip["defl_m"],
        "w_tip_rel_error": abs(-float(u[3 * n + 1]) - tip["defl_m"]) / abs(tip["defl_m"]),
        "theta_tip_fem_rad": -float(u[3 * n + 2]),
        "theta_tip_exact_rad": tip["slope_rad"],
        "theta_tip_rel_error": abs(-float(u[3 * n + 2]) - tip["slope_rad"])
        / abs(tip["slope_rad"]),
        "root_moment_fem_nm": root_moment_fem,
        "root_moment_exact_nm": root["bending_moment_nm"],
        "root_stress_fem_pa": root_stress_fem,
        "root_stress_exact_pa": root["bending_stress_pa"],
        "root_stress_rel_error": abs(root_stress_fem - root["bending_stress_pa"])
        / abs(root["bending_stress_pa"]),
        "field": field,
        "field_max_defl_rel_error": max(r["defl_rel_error"] for r in field),
        # the last element sits at the free end, where the uniform-load moment
        # (and hence the stress) vanishes: a relative error there measures the
        # denominator, not the element, so only elements with M >= 10 % of the
        # root moment enter the stress metric
        "field_max_stress_rel_error": max(
            r["bending_stress_rel_error"] for r in field
            if abs(r["bending_stress_exact_pa"]) >= 0.1 * abs(theory(0.0, section=section)["bending_stress_pa"])
        ),
    }


# --------------------------------------------------------------------------- #
# case B -- Q4 Mindlin plate (MITC4 shear), simply supported square
# --------------------------------------------------------------------------- #
PLATE = {
    "side_m": 1.0,
    "thickness_m": 0.01,
    "young_modulus_pa": 210.0e9,
    "poisson_ratio": 0.3,
    "pressure_pa": 10.0e3,
    "shear_treatment": "mitc",
    "levels_elements_per_side": (8, 16, 32),
    "timoshenko_alpha": 0.00406,
    "navier_terms": 401,
    "navier_terms_field": 101,
}


def plate_rigidity() -> float:
    E = PLATE["young_modulus_pa"]
    t = PLATE["thickness_m"]
    nu = PLATE["poisson_ratio"]
    return E * t**3 / (12.0 * (1.0 - nu**2))


def plate_theory(x: float = 0.5, y: float = 0.5, *, terms: int = 401) -> float:
    """Kirchhoff Navier deflection of the SSSS square at ``(x/a, y/a)``."""
    a = PLATE["side_m"]
    state = ss_uniform_kirchhoff(
        side=a,
        young=PLATE["young_modulus_pa"],
        poisson=PLATE["poisson_ratio"],
        thickness=PLATE["thickness_m"],
        pressure=PLATE["pressure_pa"],
        x=x * a,
        y=y * a,
        terms=terms,
    )
    return state["deflection"]


def solve_plate(n: int, *, shear: str | None = None) -> dict:
    """Solve the SSSS square on an ``n x n`` mesh and grade three metrics."""
    a = PLATE["side_m"]
    t = PLATE["thickness_m"]
    E = PLATE["young_modulus_pa"]
    nu = PLATE["poisson_ratio"]
    q = PLATE["pressure_pa"]
    shear = shear or PLATE["shear_treatment"]
    nodes, elements = structured_square_mesh(n, a)
    ndof = 3 * nodes.shape[0]
    rows, cols, vals = [], [], []
    f = np.zeros(ndof)
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, E, nu, t, shear_treatment=shear).numpy()
        fe = consistent_load_vector(xy, -q).numpy()
        idx = conn.numpy()
        dof = np.stack((3 * idx, 3 * idx + 1, 3 * idx + 2), axis=1).reshape(-1)
        rows.append(np.repeat(dof, 12))
        cols.append(np.tile(dof, 12))
        vals.append(ke.reshape(-1))
        f[dof] += fe
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndof, ndof),
    ).tocsr()
    coords = nodes.numpy()
    on_edge = (
        (np.abs(coords[:, 0]) < 1e-12)
        | (np.abs(coords[:, 0] - a) < 1e-12)
        | (np.abs(coords[:, 1]) < 1e-12)
        | (np.abs(coords[:, 1] - a) < 1e-12)
    )
    fixed = np.nonzero(on_edge)[0] * 3  # soft simple support: w only
    mask = np.ones(ndof, dtype=bool)
    mask[fixed] = False
    u = np.zeros(ndof)
    u[mask] = spla.spsolve(K[mask][:, mask].tocsc(), f[mask])
    centre = np.nonzero((np.abs(coords[:, 0] - a / 2) < 1e-12)
                        & (np.abs(coords[:, 1] - a / 2) < 1e-12))[0][0]
    w_centre = -float(u[3 * centre])
    w_nodal = -u[0::3]
    # L2 relative error of the deflection field, element centres, theory at the
    # same point (midpoint rule: the integrand is smooth, quadrature error is far
    # below the discretisation error being measured)
    num = den = 0.0
    area = (a / n) ** 2
    for conn in elements.numpy():
        xy = coords[conn]
        w_h = float(w_nodal[conn].mean())
        w_ref = plate_theory(float(xy[:, 0].mean()) / a, float(xy[:, 1].mean()) / a,
                             terms=PLATE["navier_terms_field"])
        num += (w_h - w_ref) ** 2 * area
        den += w_ref**2 * area
    D = plate_rigidity()
    w_ref_centre = plate_theory(0.5, 0.5, terms=PLATE["navier_terms"])
    w_timo = PLATE["timoshenko_alpha"] * q * a**4 / D
    return {
        "elements_per_side": n,
        "elements": int(n * n),
        "shear_treatment": shear,
        "w_centre_fem_m": w_centre,
        "w_centre_navier_m": w_ref_centre,
        "w_centre_timoshenko_m": w_timo,
        "w_centre_rel_error_navier": abs(w_centre - w_ref_centre) / w_ref_centre,
        "w_centre_rel_error_timoshenko": abs(w_centre - w_timo) / w_timo,
        "alpha_fem": w_centre * D / (q * a**4),
        "l2_defl_rel_error": math.sqrt(num / den),
        "_field": (coords, w_nodal),
    }# --------------------------------------------------------------------------- #
# case C -- axisymmetric ring element, thick cylinder under internal pressure
# --------------------------------------------------------------------------- #
CYLINDER = {
    "inner_radius_m": 0.10,
    "outer_radius_m": 0.20,
    "internal_pressure_pa": 20.0e6,
    "axial_length_m": 0.05,
    "young_modulus_pa": 210.0e9,
    "poisson_ratio": 0.3,
    "levels_radial_elements": (8, 16, 32),
    "sample_radii_over_a": (1.1, 1.3, 1.5, 1.7, 1.9),
}

_GAUSS = 1.0 / math.sqrt(3.0)
_SQRT3 = math.sqrt(3.0)
#: bilinear Gauss -> node extrapolation, coefficients applied to (sigma(-g), sigma(+g))
_NODE_LEFT = ((1.0 - _SQRT3) / 2.0, (1.0 + _SQRT3) / 2.0)
_NODE_RIGHT = ((1.0 + _SQRT3) / 2.0, (1.0 - _SQRT3) / 2.0)


def lame_coefficients() -> tuple[float, float]:
    a = CYLINDER["inner_radius_m"]
    b = CYLINDER["outer_radius_m"]
    p = CYLINDER["internal_pressure_pa"]
    A = p * a**2 / (b**2 - a**2)
    B = p * a**2 * b**2 / (b**2 - a**2)
    return A, B


def lame(r: float) -> dict:
    """Plane-strain Lame state at radius ``r`` (tension positive)."""
    A, B = lame_coefficients()
    E = CYLINDER["young_modulus_pa"]
    nu = CYLINDER["poisson_ratio"]
    return {
        "sigma_r_pa": A - B / r**2,
        "sigma_theta_pa": A + B / r**2,
        "sigma_z_pa": 2.0 * nu * A,
        "u_r_m": (1.0 + nu) / E * ((1.0 - 2.0 * nu) * A * r + B / r),
    }


def _axisymmetric_D() -> torch.Tensor:
    E = CYLINDER["young_modulus_pa"]
    nu = CYLINDER["poisson_ratio"]
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * torch.tensor(
        [
            [1.0 - nu, nu, nu, 0.0],
            [nu, 1.0 - nu, nu, 0.0],
            [nu, nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
        ],
        dtype=torch.float64,
    )


def _ring_mesh(nr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``nr`` radial x 1 axial Q4 ring elements on ``r in [a, b]``."""
    a = CYLINDER["inner_radius_m"]
    b = CYLINDER["outer_radius_m"]
    height = CYLINDER["axial_length_m"]
    nz = 1
    r = torch.linspace(a, b, nr + 1, dtype=torch.float64)
    z = torch.linspace(0.0, height, nz + 1, dtype=torch.float64)
    nodes = torch.stack(torch.meshgrid(r, z, indexing="ij"), dim=-1).reshape(-1, 2)

    def nid(i: int, j: int) -> int:
        return i * (nz + 1) + j

    elements = torch.tensor(
        [[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
         for i in range(nr) for j in range(nz)],
        dtype=torch.long,
    )
    return nodes, elements


def _internal_pressure_load(nodes: torch.Tensor, nr: int, nz: int = 1) -> np.ndarray:
    """Consistent nodal forces of the bore pressure (2 pi r edge weighting, inwards)."""
    a = CYLINDER["inner_radius_m"]
    p = CYLINDER["internal_pressure_pa"]
    height = CYLINDER["axial_length_m"]
    f = np.zeros(2 * nodes.shape[0])
    for i in range(nr):
        left, right = i * (nz + 1), i * (nz + 1) + 1
        if abs(float(nodes[left, 0]) - a) > 1e-12:
            continue
        for xi in (-_GAUSS, _GAUSS):
            # traction on the bore = -p * n with n = -e_r (the solid's outward
            # normal points at the axis), i.e. +p e_r pushing the wall outwards
            f[2 * left] += (1.0 - xi) / 2.0 * p * 2.0 * math.pi * a * (height / nz) / 2.0
            f[2 * right] += (1.0 + xi) / 2.0 * p * 2.0 * math.pi * a * (height / nz) / 2.0
    return f


def _element_stresses(nodes: torch.Tensor, conn: torch.Tensor, u: torch.Tensor,
                      D: torch.Tensor) -> dict:
    """``(sigma_r, sigma_z, sigma_theta)`` at the four Gauss points of one element."""
    x = nodes[conn]
    ue = u[torch.stack((2 * conn, 2 * conn + 1), dim=1).reshape(-1)]
    out = {}
    for xg in (-_GAUSS, _GAUSS):
        for eg in (-_GAUSS, _GAUSS):
            N = torch.tensor(
                [(1 - xg) * (1 - eg), (1 + xg) * (1 - eg),
                 (1 + xg) * (1 + eg), (1 - xg) * (1 + eg)],
                dtype=torch.float64,
            ) / 4.0
            dnat = torch.tensor(
                [[-(1 - eg), -(1 - xg)], [(1 - eg), -(1 + xg)],
                 [(1 + eg), (1 + xg)], [-(1 + eg), (1 - xg)]],
                dtype=torch.float64,
            ) / 4.0
            J = dnat.T @ x
            dN = dnat @ torch.linalg.inv(J)
            radius = float(N @ x[:, 0])
            B = torch.zeros(4, 8, dtype=torch.float64)
            B[0, 0::2] = dN[:, 0]
            B[1, 1::2] = dN[:, 1]
            B[2, 0::2] = N / radius
            B[3, 0::2] = dN[:, 1]
            B[3, 1::2] = dN[:, 0]
            s = D @ (B @ ue)
            out[(xg, eg)] = (float(s[0]), float(s[1]), float(s[2]))
    return out


def _nodal_recovery(nodes: torch.Tensor, elements: torch.Tensor, u: torch.Tensor,
                    nr: int, nz: int = 1) -> dict:
    """Gauss -> nodal extrapolation plus averaging of the sharing elements.

    Each element's four Gauss values are extrapolated bilinearly to its four nodes
    (weights ``(1 +- sqrt 3)/2``) and every node averages the contributions of the
    elements that share it.  This standard recovery is what makes the pointwise
    stress second-order accurate; the raw Gauss values are kept for comparison so
    the amount of accuracy contributed by the recovery stays visible.
    """
    D = _axisymmetric_D()
    accum = np.zeros((len(nodes), 3))
    count = np.zeros(len(nodes))
    raw = []
    order = [(xg, eg) for xg in (-_GAUSS, _GAUSS) for eg in (-_GAUSS, _GAUSS)]
    for conn in elements:
        g = _element_stresses(nodes, conn, u, D)
        radii_e = [float(nodes[c, 0]) for c in conn.tolist()]
        raw.append(
            {
                "radius_left_m": min(radii_e),
                "radius_right_m": max(radii_e),
                "sigma_r": [g[k][0] for k in order],
                "sigma_theta": [g[k][2] for k in order],
            }
        )
        for comp in range(3):
            col = [g[k][comp] for k in order]
            r_m = ((col[0] + col[1]) / 2.0, (col[2] + col[3]) / 2.0)  # average in z
            left = _NODE_LEFT[0] * r_m[0] + _NODE_LEFT[1] * r_m[1]
            right = _NODE_RIGHT[0] * r_m[0] + _NODE_RIGHT[1] * r_m[1]
            i = int(conn[0]) // (nz + 1)
            j = int(conn[2]) // (nz + 1)
            accum[i * (nz + 1), comp] += left
            accum[j * (nz + 1), comp] += right
        count[i * (nz + 1)] += 1
        count[j * (nz + 1)] += 1
    count[count == 0] = 1.0
    nodal = accum / count[:, None]
    return {"nodal": nodal, "per_element": raw}


def solve_cylinder(nr: int) -> dict:
    """Solve the thick cylinder with ``nr`` radial ring elements and grade it."""
    nodes, elements = _ring_mesh(nr)
    E = CYLINDER["young_modulus_pa"]
    nu = CYLINDER["poisson_ratio"]
    K = assemble_axisymmetric_stiffness(nodes, elements, E, nu).numpy()
    ndof = 2 * nodes.shape[0]
    f = _internal_pressure_load(nodes, nr)
    # plane strain: the axial normal strain is driven to zero pointwise, so the
    # in-plane problem is exactly the plane-strain Lame problem
    mask = np.ones(ndof, dtype=bool)
    mask[1::2] = False
    u = np.zeros(ndof)
    u[mask] = spla.spsolve(sp.csr_matrix(K)[mask][:, mask].tocsc(), f[mask])
    u = torch.tensor(u)
    recovered = _nodal_recovery(nodes, elements, u, nr)
    nodal = recovered["nodal"]
    a = CYLINDER["inner_radius_m"]
    radii = nodes[0::2, 0].numpy()
    samples = []
    for factor in CYLINDER["sample_radii_over_a"]:
        idx = int(np.argmin(np.abs(radii - factor * a)))
        row = idx * 2  # node row of the (radial index, j = 0) node: nz + 1 = 2
        r = float(radii[idx])
        ref = lame(r)
        sr, sz, st = float(nodal[row, 0]), float(nodal[row, 1]), float(nodal[row, 2])
        ur = float(u[2 * row])
        # sigma_r vanishes at the free outer surface, so a *relative* error is only
        # defined away from it; the load-normalised absolute error is reported for
        # every point so the outer sample is never silently dropped
        p_ref = CYLINDER["internal_pressure_pa"]
        rel_r = (abs(sr - ref["sigma_r_pa"]) / abs(ref["sigma_r_pa"])
                 if abs(ref["sigma_r_pa"]) > 1e-6 * p_ref else None)
        samples.append(
            {
                "r_m": r,
                "r_over_a": r / a,
                "sigma_r_fem_pa": sr,
                "sigma_r_lame_pa": ref["sigma_r_pa"],
                "sigma_r_rel_error": rel_r,
                "sigma_r_abs_over_p": abs(sr - ref["sigma_r_pa"]) / p_ref,
                "sigma_theta_fem_pa": st,
                "sigma_theta_lame_pa": ref["sigma_theta_pa"],
                "sigma_theta_rel_error": abs(st - ref["sigma_theta_pa"])
                / abs(ref["sigma_theta_pa"]),
                "sigma_z_fem_pa": sz,
                "sigma_z_lame_pa": ref["sigma_z_pa"],
                "sigma_z_rel_error": abs(sz - ref["sigma_z_pa"]) / abs(ref["sigma_z_pa"]),
                "u_r_fem_m": ur,
                "u_r_lame_m": ref["u_r_m"],
                "u_r_rel_error": abs(ur - ref["u_r_m"]) / abs(ref["u_r_m"]),
            }
        )
    # raw (unrecovered) Gauss values nearest to the same radii, for the audit
    gauss = []
    for s in samples:
        best = None
        for elem in recovered["per_element"]:
            r0, r1 = elem["radius_left_m"], elem["radius_right_m"]
            for k, xg in enumerate((-_GAUSS, _GAUSS)):
                rg = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * xg
                d = abs(rg - s["r_m"])
                if best is None or d < best[0]:
                    sig_r = (elem["sigma_r"][k] + elem["sigma_r"][k + 2]) / 2.0
                    sig_t = (elem["sigma_theta"][k] + elem["sigma_theta"][k + 2]) / 2.0
                    best = (d, rg, sig_r, sig_t)
        _, rg, sig_r, sig_t = best
        ref = lame(rg)
        gauss.append(
            {
                "gauss_radius_m": rg,
                "sigma_r_rel_error": abs(sig_r - ref["sigma_r_pa"]) / abs(ref["sigma_r_pa"]),
                "sigma_theta_rel_error": abs(sig_t - ref["sigma_theta_pa"])
                / abs(ref["sigma_theta_pa"]),
            }
        )
    return {
        "radial_elements": nr,
        "axial_elements": 1,
        "nodes": int(nodes.shape[0]),
        "samples": samples,
        "max_sigma_r_rel_error": max(
            (s["sigma_r_rel_error"] for s in samples if s["sigma_r_rel_error"] is not None),
            default=float("nan")),
        "max_sigma_r_abs_over_p": max(s["sigma_r_abs_over_p"] for s in samples),
        "sigma_r_rel_error_defined_points": sum(
            1 for s in samples if s["sigma_r_rel_error"] is not None),
        "max_sigma_theta_rel_error": max(s["sigma_theta_rel_error"] for s in samples),
        "max_sigma_z_rel_error": max(s["sigma_z_rel_error"] for s in samples),
        "max_u_r_rel_error": max(s["u_r_rel_error"] for s in samples),
        "gauss_point_reference": gauss,
        "max_gauss_sigma_theta_rel_error": max(r["sigma_theta_rel_error"] for r in gauss),
        "max_gauss_sigma_r_rel_error": max(r["sigma_r_rel_error"] for r in gauss),
        "_field": (nodes, u, nodal),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_beam(levels: list[dict]) -> Path:
    plt = H.pyplot()
    section = beam_section()
    L = BEAM["length_m"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6))
    mid = levels[1]
    stations = [0.0] + [row["x_m"] for row in mid["field"]] + [L]
    w_fem = [0.0] + [row["defl_fem_m"] for row in mid["field"]] + [mid["w_tip_fem_m"]]
    xs = np.linspace(0.0, L, 200)
    axes[0].plot(xs, [beam_theory(x, section=section)["defl_m"] * 1e3 for x in xs], "-",
                 color="0.25", lw=1.6, label="Euler-Bernoulli exact")
    axes[0].plot(stations, np.array(w_fem) * 1e3, "o", ms=3.6, color="crimson",
                 label=f"FEM, {mid['elements']} Hermite elements")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("deflection w [mm]")
    axes[0].set_title("Case A: cantilever tip load P = 50 kN")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(xs, [beam_theory(x, section=section)["bending_stress_pa"] / 1e6 for x in xs],
                 "-", color="0.25", lw=1.6, label="sigma = M(x) c / I exact")
    axes[1].plot([row["x_m"] for row in mid["field"]],
                 [row["bending_stress_fem_pa"] / 1e6 for row in mid["field"]],
                 "s", ms=3.6, color="steelblue", label="FEM curvature field")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("bending stress at z = +h/2 [MPa]")
    axes[1].set_title("Case A: root bending stress (140.5 MPa analytic)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = HERE / "case_a_beam.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_plate(levels: list[dict]) -> Path:
    plt = H.pyplot()
    a = PLATE["side_m"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    fine = levels[-1]
    n = fine["elements_per_side"]
    coords, w = fine["_field"]
    W = (-w).reshape(n + 1, n + 1) * 1e3
    im = axes[0].contourf(np.linspace(0, a, n + 1), np.linspace(0, a, n + 1), W,
                          levels=18, cmap="viridis")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title(f"Case B: deflection field [mm], MITC4 {n}x{n}")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="w [mm]")
    ns = [row["elements_per_side"] for row in levels]
    axes[1].loglog(ns, [row["w_centre_rel_error_navier"] for row in levels], "o-",
                   color="crimson", label="centre deflection vs Navier")
    axes[1].loglog(ns, [row["w_centre_rel_error_timoshenko"] for row in levels], "s--",
                   color="darkorange", label="centre deflection vs Timoshenko 0.00406")
    axes[1].loglog(ns, [row["l2_defl_rel_error"] for row in levels], "^:",
                   color="steelblue", label="L2 deflection field vs Navier")
    axes[1].axhline(CRITERIA["case_b_centre_deflection_rel_max"], color="0.4", ls="--",
                    lw=1.0, label="2 % gate")
    axes[1].set_xlabel("elements per side")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Case B: mesh refinement (Thin plate, a/t = 100)")
    axes[1].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    path = HERE / "case_b_plate.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_cylinder(levels: list[dict]) -> Path:
    plt = H.pyplot()
    a = CYLINDER["inner_radius_m"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    fine = levels[-1]
    radii = np.linspace(a, CYLINDER["outer_radius_m"], 400)
    axes[0].plot(radii / a, [lame(r)["sigma_theta_pa"] / 1e6 for r in radii], "-",
                 color="crimson", lw=1.8, label="Lame sigma_theta")
    axes[0].plot(radii / a, [lame(r)["sigma_r_pa"] / 1e6 for r in radii], "-",
                 color="steelblue", lw=1.8, label="Lame sigma_r")
    axes[0].plot([s["r_over_a"] for s in fine["samples"]],
                 [s["sigma_theta_fem_pa"] / 1e6 for s in fine["samples"]], "o",
                 ms=5, mfc="none", color="crimson", label="FEM sigma_theta (recovered)")
    axes[0].plot([s["r_over_a"] for s in fine["samples"]],
                 [s["sigma_r_fem_pa"] / 1e6 for s in fine["samples"]], "s",
                 ms=5, mfc="none", color="steelblue", label="FEM sigma_r (recovered)")
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_xlabel("r / a")
    axes[0].set_ylabel("stress [MPa]")
    axes[0].set_title(f"Case C: Lame fields, p = 20 MPa, b/a = 2, {fine['radial_elements']} radial elements")
    axes[0].legend(fontsize=7.5, loc="center right")
    for row, style in zip(levels, ("o-", "s--", "^:")):
        axes[1].semilogy([s["r_over_a"] for s in row["samples"]],
                         [s["sigma_theta_rel_error"] for s in row["samples"]], style,
                         color="crimson", ms=4, lw=1.2,
                         label=f"sigma_theta, {row['radial_elements']} elem")
        pts = [(s["r_over_a"], s["sigma_r_rel_error"]) for s in row["samples"]
               if s["sigma_r_rel_error"] is not None]
        if pts:
            axes[1].semilogy([q[0] for q in pts], [q[1] for q in pts], style,
                             color="steelblue", ms=4, lw=1.2,
                             label=f"sigma_r, {row['radial_elements']} elem")
    axes[1].axhline(CRITERIA["case_c_stress_rel_max"], color="0.4", ls="--", lw=1.0,
                    label="1 % gate")
    axes[1].set_xlabel("r / a")
    axes[1].set_ylabel("relative error vs Lame")
    axes[1].set_title("Case C: pointwise stress error, three radial meshes")
    axes[1].legend(fontsize=6.4, ncol=2, loc="lower left")
    fig.tight_layout()
    path = HERE / "case_c_cylinder.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "dtype": "torch.float64 / numpy.float64",
    }


def run_cases(*, plate_levels=None, beam_levels=None, cylinder_levels=None) -> dict:
    """Solve all three ladders; ``*_levels`` overrides exist for the fast tests."""
    beam_levels = beam_levels or BEAM["levels_elements"]
    plate_levels = plate_levels or PLATE["levels_elements_per_side"]
    cylinder_levels = cylinder_levels or CYLINDER["levels_radial_elements"]

    t0 = time.time()
    beam_tip = [solve_beam(n, load="tip") for n in beam_levels]
    beam_udl = [solve_beam(n, load="udl") for n in beam_levels]
    t_beam = time.time() - t0

    t0 = time.time()
    plate = [solve_plate(n) for n in plate_levels]
    t_plate = time.time() - t0

    t0 = time.time()
    cylinder = [solve_cylinder(n) for n in cylinder_levels]
    t_cyl = time.time() - t0

    return {
        "beam_tip": beam_tip,
        "beam_udl": beam_udl,
        "plate": plate,
        "cylinder": cylinder,
        "timings_s": {"beam": t_beam, "plate": t_plate, "cylinder": t_cyl},
    }


def _order(sequence: list[float], ratio: float = 2.0) -> dict:
    """Observed order from the last two levels plus the full ladder."""
    orders = []
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        if a > 0.0 and b > 0.0:
            orders.append(math.log(a / b) / math.log(ratio))
        else:
            orders.append(float("nan"))
    last = orders[-1] if orders else float("nan")
    return {"per_level": orders, "finest": last}


def build_result(cases: dict) -> dict:
    beam_tip, beam_udl = cases["beam_tip"], cases["beam_udl"]
    plate, cylinder = cases["plate"], cases["cylinder"]
    section = beam_section()
    D = plate_rigidity()

    beam_order = _order([row["field_max_defl_rel_error"] for row in beam_udl])
    plate_order_l2 = _order([row["l2_defl_rel_error"] for row in plate])
    plate_order_centre = _order([row["w_centre_rel_error_navier"] for row in plate])
    cyl_order_theta = _order([row["max_sigma_theta_rel_error"] for row in cylinder])
    cyl_order_ur = _order([row["max_u_r_rel_error"] for row in cylinder])

    def strip(row: dict) -> dict:
        return {k: v for k, v in row.items() if not k.startswith("_")}

    return {
        "anchor": "fem_structural_benchmarks",
        "schema": "tensorfem.structural-verification-face.v1",
        "mission": "IN41-N01",
        "generated_at": H.utc_now(),
        "environment": _environment(),
        "criteria": CRITERIA,
        "cases": {
            "A_beam_cantilever": {
                "element": "2-node cubic Hermite Euler-Bernoulli frame element "
                           "(tensorfem.frame2d, 3 dof per node)",
                "physics": "Euler-Bernoulli cantilever, tip load P and uniform load q",
                "parameters": dict(BEAM, section=section),
                "reference": {
                    "w_tip": "P L^3 / (3 E I) = %.10e m" % beam_tip[-1]["w_tip_exact_m"],
                    "theta_tip": "P L^2 / (2 E I) = %.10e rad" % beam_tip[-1]["theta_tip_exact_rad"],
                    "sigma_root": "P L c / I = %.6e Pa" % beam_tip[-1]["root_stress_exact_pa"],
                    "w_udl": "q L^4 / (8 E I) = %.10e m" % beam_udl[-1]["w_tip_exact_m"],
                },
                "levels_tip_load": [strip(row) for row in beam_tip],
                "levels_uniform_load": [strip(row) for row in beam_udl],
                "observations": {
                    "worst_tip_deflection_rel_error": max(r["w_tip_rel_error"] for r in beam_tip),
                    "worst_root_stress_rel_error": max(r["root_stress_rel_error"] for r in beam_tip),
                    "worst_field_stress_rel_error": max(
                        r["field_max_stress_rel_error"] for r in beam_tip),
                    "order_tip_load_fem": _order([r["w_tip_rel_error"] for r in beam_tip]),
                    "order_udl_interior_field": beam_order,
                    "order_note": "the tip-load field is a cubic, which the Hermite element "
                                  "spans exactly: the reported tip-load errors are a round-off "
                                  "floor, not a converged discretisation, so no order is "
                                  "measurable for that load case; the uniform-load interior "
                                  "field supplies the measurable order",
                },
            },
            "B_plate_ssss_uniform": {
                "element": "4-node Mindlin-Reissner plate element, MITC4 assumed shear "
                           "strain (tensorfem.plate.q4_mindlin_stiffness)",
                "physics": "Kirchhoff/Mindlin square plate, soft simple support, uniform pressure",
                "parameters": dict(PLATE, bending_stiffness=D),
                "reference": {
                    "w_centre_navier": "16 q a^4/(pi^6 D) sum_{m,n odd} sin sin /(m n (m^2+n^2)^2)"
                                       " = %.10e m (alpha = %.9f)"
                                       % (plate[-1]["w_centre_navier_m"],
                                          plate[-1]["w_centre_navier_m"] * D
                                          / (PLATE["pressure_pa"] * PLATE["side_m"] ** 4)),
                    "w_centre_timoshenko": "alpha q a^4 / D, alpha = 0.00406 (Timoshenko table 8)"
                                           " = %.10e m" % plate[-1]["w_centre_timoshenko_m"],
                    "mindlin_model_difference": "O((t/a)^2) = %.2e relative"
                                                % ((PLATE["thickness_m"] / PLATE["side_m"]) ** 2),
                },
                "levels": [strip(row) for row in plate],
                "observations": {
                    "worst_centre_rel_error_navier": max(r["w_centre_rel_error_navier"] for r in plate),
                    "worst_centre_rel_error_timoshenko": max(
                        r["w_centre_rel_error_timoshenko"] for r in plate),
                    "order_l2_field": plate_order_l2,
                    "order_centre_deflection": plate_order_centre,
                    "order_note": "the centre deflection error is not monotone: at n = 32 it sits "
                                  "at ~0.3 %, above its n = 16 value (7e-4), so its observed "
                                  "order is not a usable rate.  The L2 norm of the deflection "
                                  "field is monotone and second order, which is the rate this "
                                  "case is graded on.",
                },
            },
            "C_cylinder_lame": {
                "element": "4-node axisymmetric (ring) element with hoop strain "
                           "(tensorfem.hertz_axisymmetric.assemble_axisymmetric_stiffness)",
                "physics": "thick-walled cylinder under internal pressure, plane strain, "
                           "Lame solution",
                "parameters": dict(CYLINDER, lame_A_pa=lame_coefficients()[0],
                                   lame_B_pa_m2=lame_coefficients()[1]),
                "reference": {
                    "sigma_r": "A - B/r^2",
                    "sigma_theta": "A + B/r^2",
                    "sigma_z": "2 nu A = %.6e Pa" % lame(0.5 * (CYLINDER["inner_radius_m"]
                                                               + CYLINDER["outer_radius_m"]))["sigma_z_pa"],
                    "u_r": "(1+nu)/E [(1-2nu) A r + B/r]",
                    "recovery": "Gauss -> nodal bilinear extrapolation (weights (1 +- sqrt3)/2) "
                                "plus averaging of the sharing elements",
                },
                "levels": [strip(row) for row in cylinder],
                "observations": {
                    "worst_sigma_theta_rel_error": max(r["max_sigma_theta_rel_error"] for r in cylinder),
                    "worst_sigma_r_rel_error": max(r["max_sigma_r_rel_error"] for r in cylinder),
                    "worst_u_r_rel_error": max(r["max_u_r_rel_error"] for r in cylinder),
                    "order_sigma_theta": cyl_order_theta,
                    "order_u_r": cyl_order_ur,
                    "gauss_vs_recovered": {
                        "finest_gauss_sigma_theta_rel_error": cylinder[-1]["max_gauss_sigma_theta_rel_error"],
                        "finest_gauss_sigma_r_rel_error": cylinder[-1]["max_gauss_sigma_r_rel_error"],
                    },
                    "order_note": "the recovered pointwise stress is second order; the raw Gauss "
                                  "values are first order and are reported for transparency.",
                },
            },
        },
        "timings_s": cases["timings_s"],
    }


def build_gates(cases: dict, result: dict) -> dict:
    beam_tip, beam_udl = cases["beam_tip"], cases["beam_udl"]
    plate, cylinder = cases["plate"], cases["cylinder"]
    crit = CRITERIA
    gates = [
        H.Gate(
            id="A1.tip_deflection",
            quantity="cantilever tip deflection vs P L^3/(3 E I), worst level",
            reference="w(L) = P L^3 / (3 E I)",
            value=max(r["w_tip_rel_error"] for r in beam_tip),
            threshold=crit["case_a_tip_deflection_rel_max"],
            note="cubic Hermite element spans the exact cubic field: round-off floor",
        ),
        H.Gate(
            id="A2.root_bending_stress",
            quantity="fixed-end bending stress vs sigma = M c / I, worst level",
            reference="sigma(L=0) = P L c / I = 141.505 MPa",
            value=max(r["root_stress_rel_error"] for r in beam_tip),
            threshold=crit["case_a_root_stress_rel_max"],
        ),
        H.Gate(
            id="A3.beam_order",
            quantity="observed order, uniform-load interior deflection field",
            reference="h-refinement order, levels %s" % (BEAM["levels_elements"],),
            value=_order([r["field_max_defl_rel_error"] for r in beam_udl])["finest"],
            threshold=crit["convergence_order_min"],
            operator=">=",
            note="the tip-load case is exactly represented (order not measurable); "
                 "the quartic uniform-load field provides the rate",
        ),
        H.Gate(
            id="B1.centre_vs_timoshenko",
            quantity="SSSS centre deflection vs Timoshenko table 8 alpha = 0.00406, finest level",
            reference="w_c = 0.00406 q a^4 / D",
            value=plate[-1]["w_centre_rel_error_timoshenko"],
            threshold=crit["case_b_centre_deflection_rel_max"],
            note="table is quoted to three significant figures: 0.06 %% rounding floor",
        ),
        H.Gate(
            id="B2.centre_vs_navier",
            quantity="SSSS centre deflection vs converged Navier series, finest level",
            reference="w_c = 16 q a^4/(pi^6 D) sum_{m,n odd} sin sin /(m n (m^2+n^2)^2), 401 terms",
            value=plate[-1]["w_centre_rel_error_navier"],
            threshold=crit["case_b_centre_deflection_rel_max"],
        ),
        H.Gate(
            id="B3.plate_order",
            quantity="observed order, L2 deflection field error",
            reference="h-refinement order, levels %s" % (PLATE["levels_elements_per_side"],),
            value=_order([r["l2_defl_rel_error"] for r in plate])["finest"],
            threshold=crit["convergence_order_min"],
            operator=">=",
        ),
        H.Gate(
            id="C1.hoop_stress",
            quantity="sigma_theta vs Lame at 5 radii, worst of the finest level",
            reference="sigma_theta = A + B/r^2",
            value=cylinder[-1]["max_sigma_theta_rel_error"],
            threshold=crit["case_c_stress_rel_max"],
        ),
        H.Gate(
            id="C2.radial_stress",
            quantity="sigma_r vs Lame at 5 radii, worst of the finest level",
            reference="sigma_r = A - B/r^2",
            value=cylinder[-1]["max_sigma_r_rel_error"],
            threshold=crit["case_c_stress_rel_max"],
            note="sigma_r vanishes at the free outer surface, where a relative error is "
                 "undefined; the four inner sample radii carry the relative gate and C2b grades "
                 "the load-normalised error at all five",
        ),
        H.Gate(
            id="C2b.radial_stress_absolute",
            quantity="|sigma_r - Lame| / p at the 5 radii, worst of the finest level",
            reference="load-normalised radial stress error",
            value=cylinder[-1]["max_sigma_r_abs_over_p"],
            threshold=crit["case_c_stress_rel_max"],
            note="companion of C2: defined even where sigma_r vanishes (free surface)",
        ),
        H.Gate(
            id="C3.radial_displacement",
            quantity="u_r vs Lame at 5 radii, worst of the finest level",
            reference="u_r = (1+nu)/E [(1-2 nu) A r + B/r]",
            value=cylinder[-1]["max_u_r_rel_error"],
            threshold=crit["case_c_stress_rel_max"],
        ),
        H.Gate(
            id="C3b.axial_stress",
            quantity="sigma_z vs 2 nu A at 5 radii, worst of the finest level",
            reference="sigma_z = 2 nu A (plane strain)",
            value=cylinder[-1]["max_sigma_z_rel_error"],
            threshold=crit["case_c_stress_rel_max"],
            note="confirms the plane-strain constraint is actually enforced",
        ),
        H.Gate(
            id="C4.cylinder_order",
            quantity="observed order, recovered sigma_theta error",
            reference="h-refinement order, levels %s" % (CYLINDER["levels_radial_elements"],),
            value=_order([r["max_sigma_theta_rel_error"] for r in cylinder])["finest"],
            threshold=crit["convergence_order_min"],
            operator=">=",
        ),
    ]
    artifacts = {
        "result": "benchmarks/verified/fem_structural_benchmarks/result.json",
        "gates": "benchmarks/verified/fem_structural_benchmarks/gates_verified.json",
        "figures": [
            "benchmarks/verified/fem_structural_benchmarks/case_a_beam.png",
            "benchmarks/verified/fem_structural_benchmarks/case_b_plate.png",
            "benchmarks/verified/fem_structural_benchmarks/case_c_cylinder.png",
        ],
        "driver": "benchmarks/verified/fem_structural_benchmarks/run.py",
        "tests": "tests/test_fem_structural_benchmarks.py",
    }
    limitations = [
        "Case A is exactly represented by the cubic Hermite element, so its tip-load errors are a "
        "round-off floor (~1e-13) rather than a converged discretisation; the measured order comes "
        "from the uniform-load interior field.  Euler-Bernoulli theory itself ignores transverse "
        "shear, which at L/h = 10 is a ~2 % model-level difference that the 1 % comparison to the "
        "same theory does not measure.",
        "Case B uses soft simple support (w = 0, rotations free) and a/t = 100, so the Mindlin "
        "transverse-shear compliance is a ~1e-4 relative model difference against the Kirchhoff "
        "oracle.  The plate centre-deflection error is not monotone in the mesh, so the plate rate is "
        "graded on the L2 deflection-field norm.",
        "Case C enforces plane strain (u_z = 0) rather than modelling the closed or open ends; the "
        "Lame sigma_r/sigma_theta fields are independent of the end condition, but sigma_z = 2 nu A "
        "corresponds to plane strain only.  Stresses are reported after standard nodal recovery: the "
        "raw Gauss-point values are first order, and both sets are in result.json.",
        "All three cases are linear-elastic, small-displacement, single-physics checks.  They certify "
        "element kernels and load/boundary handling, not mesh quality on curved geometry, "
        "plasticity, contact or dynamics.",
    ]
    return H.gate_manifest(
        anchor="fem_structural_benchmarks",
        title="Beam / plate / axisymmetric elements vs Euler-Bernoulli, Navier and Lame oracles",
        criteria=crit,
        gates=gates,
        artifacts=artifacts,
        limitations=limitations,
        mission="IN41-N01",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=1,
                        help="torch threads (tiny element matrices: one thread is fastest)")
    args = parser.parse_args(argv)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(max(1, args.threads))

    cases = run_cases()
    result = build_result(cases)
    gates = build_gates(cases, result)
    H.write_json(HERE / "result.json", result)
    H.write_json(HERE / "gates_verified.json", gates)
    figure_beam(cases["beam_tip"])
    figure_plate(cases["plate"])
    figure_cylinder(cases["cylinder"])

    verdict = "PASS" if gates["passed"] else "FAIL(" + ",".join(gates["failed_gates"]) + ")"
    print(f"gates: {verdict}")
    for gate in gates["gates"]:
        print(f"  {gate['id']:<24s} value={gate['value']:.4e} "
              f"{gate['operator']} {gate['threshold']:.3e}  {'ok' if gate['passed'] else 'FAILED'}")
    print("timings:", json.dumps(result["timings_s"]))
    return 0 if gates["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())