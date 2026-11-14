"""KIRSCH-PLATE field verification: the whole stress field, not only the peak.

Evidence gap this study closes
------------------------------
``run.py`` certifies the anchor with fourteen gates whose stress content is the
*peak* value ``K_t = sigma_tt(a, pi/2)/T`` (one node), a hoop-stress profile on
the hole edge, a load balance and a few invariances.  Nothing in the certificate
compares the rest of the computed field with the closed form, so a solution that
happens to hit ``3T`` at the pole while being wrong in the interior, or wrong in
``sigma_xy`` / in the sign of the transverse component, would still pass.

This driver adds that axis on the *pure discretisation* model of the anchor:
the Q8 polar annulus ``a <= r <= 2a`` whose outer rim carries the exact Kirsch
displacement (production path
:func:`tensorfem.structural_canonical_face.solve_kirsch_q8_concentration`).
Because the exact field is the exact solution of that Dirichlet problem, every
field metric below is discretisation error only - no finite-width or
Saint-Venant model error can hide in it.

Measured per mesh level
-----------------------
* ``kt``                      ``sigma_xx(0, a)/T`` from production nodal recovery
* ``edge_traction_residual``  ``max(|sigma_rr|, |sigma_rt|)/T`` on the free rim
* ``stress_l2_rel_gauss``     ``||sigma_fe - sigma_ex||_L2 / ||sigma_ex||_L2``
                              over the 3x3 Gauss points of every element
* ``stress_l2_rel_nodal``     the same for the recovered nodal field with lumped
                              area weights (the field ``K_t`` is read from)
* ``displacement_l2_rel``     ``||u_fe - u_ex|| / ||u_ex||`` on all nodes
* ``energy_rel_error``        ``|U_fe - U_ex| / U_ex``, ``U`` = strain energy
* ``probe_*``                 pointwise stress at five declared interior probes

All thresholds are predeclared in :data:`CRITERIA` before any number is
produced and are *not* relaxed afterwards; a failing gate is written as failed.

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/field_verification.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
for _path in (_ROOT / "src", _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tensorfem.kirsch_hole import KirschHole  # noqa: E402
from tensorfem.quadratic_plane import (  # noqa: E402
    GAUSS_POINTS_3X3,
    GAUSS_WEIGHTS_3X3,
    q8_element_area,
    q8_strain_matrix,
    q8_stress_at,
    solve_q8_plane,
)
from tensorfem.quadratic_plane import polar_q8_mesh  # noqa: E402

OUTER_OVER_INNER = 2.0
INNER_RADIUS = 1.0
TENSION = 1.0
YOUNG = 1.0e5
POISSON = 0.3
#: ``(n_theta, n_radial)`` with both doubled each level (r = 2 sequence).  Four
#: levels give three pairwise rates, i.e. an asymptotic-range check rather than a
#: single ratio of two errors.
LADDER = ((24, 6), (48, 12), (96, 24), (192, 48))
#: Fixed physical probe locations ``(r/a, theta in degrees)``.  Radii are placed so
#: that each probe stays at least 0.3 away (in parent coordinates) from the element
#: boundary at *every* level of :data:`LADDER`, so the reported stress is
#: single-valued; ``probes_interior`` guards that numerically.
PROBES = ((1.10, 5.0), (1.30, 40.0), (1.70, 87.0), (1.10, 130.0), (1.30, -70.0))

#: Predeclared gates.  Rationale: Q8 stress recovery is O(h^2) at Gauss points
#: and superconvergent after extrapolation, so a 2 % domain-wide band is a
#: generous ceiling for the finest level; the probe band is the campaign's 3 %
#: pointwise line; the free-rim residual is the natural BC and has no reason to
#: exceed 1 % of the peak; energy and displacement converge one order faster.
CRITERIA = {
    "kt_rel_error_max": 0.05,
    "stress_l2_rel_gauss_max": 0.02,
    "stress_l2_rel_nodal_max": 0.03,
    "probe_pointwise_rel_max": 0.03,
    "edge_traction_residual_max": 0.01,
    "displacement_l2_rel_max": 0.005,
    "energy_rel_error_max": 0.01,
    "observed_rate_min": 0.9,
    "levels_min": 3,
}


# --------------------------------------------------------------------------- #
# exact field
# --------------------------------------------------------------------------- #
def kirsch_hole() -> KirschHole:
    return KirschHole(
        radius=INNER_RADIUS,
        far_field=TENSION,
        young=YOUNG,
        poisson=POISSON,
        plane="stress",
    )


def exact_stress_cartesian(x: np.ndarray, y: np.ndarray, hole: KirschHole):
    """``(sigma_xx, sigma_yy, sigma_xy)`` of the Kirsch field at Cartesian points."""
    sxx, syy, sxy = hole.stress_cartesian(np.asarray(x, float), np.asarray(y, float))
    return (
        np.asarray(sxx, float),
        np.asarray(syy, float),
        np.asarray(sxy, float),
    )


def exact_energy_density(x, y, hole: KirschHole):
    """``sigma:eps / 2`` of the plane-stress Kirsch field (compliance form)."""
    sxx, syy, sxy = exact_stress_cartesian(x, y, hole)
    e, nu = hole.young, hole.poisson
    return 0.5 * (sxx * sxx - 2.0 * nu * sxx * syy + syy * syy + 2.0 * (1.0 + nu) * sxy * sxy) / e


def oracle_field_selfcheck(hole: KirschHole | None = None) -> dict:
    """Verify the exact field against its own boundary conditions, densely."""
    hole = hole or kirsch_hole()
    theta = np.linspace(0.0, 2.0 * math.pi, 2001)
    radius = np.full_like(theta, hole.radius)
    srr, stt, srt = hole.stress_polar(radius, theta)
    far = np.full_like(theta, 40.0 * hole.radius)
    srr_f, stt_f, srt_f = hole.stress_polar(far, theta)
    sxx_f = srr_f * np.cos(theta) ** 2 + stt_f * np.sin(theta) ** 2 - 2.0 * srt_f * np.sin(theta) * np.cos(theta)
    syy_f = srr_f * np.sin(theta) ** 2 + stt_f * np.cos(theta) ** 2 + 2.0 * srt_f * np.sin(theta) * np.cos(theta)
    # Cartesian oracle vs the polar oracle at interior points (two code paths).
    rr = np.linspace(1.0, 4.0, 61)
    tt = np.linspace(0.0, 2.0 * math.pi, 97)
    R, T = np.meshgrid(rr, tt, indexing="ij")
    srr_i, stt_i, srt_i = hole.stress_polar(R, T)
    sxx_c, syy_c, sxy_c = exact_stress_cartesian(R * np.cos(T), R * np.sin(T), hole)
    c, s = np.cos(T), np.sin(T)
    sxx_p = srr_i * c**2 + stt_i * s**2 - 2.0 * srt_i * s * c
    syy_p = srr_i * s**2 + stt_i * c**2 + 2.0 * srt_i * s * c
    sxy_p = (srr_i - stt_i) * s * c + srt_i * (c**2 - s**2)
    return {
        "hole_edge_traction_free_max_over_T": float(max(np.abs(srr).max(), np.abs(srt).max()) / hole.far_field),
        "hole_edge_hoop_max_over_T": float(stt.max() / hole.far_field),
        "hole_edge_hoop_min_over_T": float(stt.min() / hole.far_field),
        "far_field_sxx_deviation_over_T": float(np.abs(sxx_f - hole.far_field).max() / hole.far_field),
        "far_field_syy_max_over_T": float(np.abs(syy_f).max() / hole.far_field),
        "cartesian_vs_polar_oracle_max_over_T": float(
            max(
                np.abs(sxx_c - sxx_p).max(),
                np.abs(syy_c - syy_p).max(),
                np.abs(sxy_c - sxy_p).max(),
            )
            / hole.far_field
        ),
        "kt_exact": float(stt.max() / hole.far_field),
    }


# --------------------------------------------------------------------------- #
# FE solve + field metrics
# --------------------------------------------------------------------------- #
def _newton_parent_point(nodes, cells, element: int, target: np.ndarray) -> np.ndarray:
    """Invert the Q8 map of ``element`` for the physical point ``target``."""
    from tensorfem.quadratic_plane import q8_shape, q8_shape_derivative

    corner = nodes[cells[element]].numpy()
    xi, eta = 0.0, 0.0
    for _ in range(40):
        shape = q8_shape(xi, eta).numpy()
        point = shape @ corner
        derivative = q8_shape_derivative(xi, eta).numpy()  # (8, 2) d/d(xi,eta)
        jac = corner.T @ derivative  # dX_b/dxi_a  -> (2, 2)
        step = np.linalg.solve(jac, target - point)
        xi += float(step[0])
        eta += float(step[1])
        if float(np.linalg.norm(step)) < 1e-13:
            break
    return np.array([xi, eta])


def _element_of_point(r: float, theta: float, n_theta: int, n_radial: int, outer: float) -> int:
    """Structured polar lookup: cells are stored with angular index slowest."""
    dr = (outer - INNER_RADIUS) / n_radial
    dtheta = 2.0 * math.pi / n_theta
    i = min(max(int(math.floor((r - INNER_RADIUS) / dr)), 0), n_radial - 1)
    j = min(max(int(math.floor((theta % (2.0 * math.pi)) / dtheta)), 0), n_theta - 1)
    return j * n_radial + i


def solve_level(n_theta: int, n_radial: int) -> dict:
    hole = kirsch_hole()
    outer = OUTER_OVER_INNER * INNER_RADIUS
    nodes, cells = polar_q8_mesh(INNER_RADIUS, outer, n_theta, n_radial)
    coordinates = nodes.numpy()
    exact_u = np.stack(
        hole.displacement_cartesian(coordinates[:, 0], coordinates[:, 1]), axis=1
    )
    radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    theta = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    dirichlet: dict[int, float] = {}
    for idx in np.nonzero(radius > outer * (1.0 - 1e-9))[0]:
        dirichlet[2 * int(idx)] = float(exact_u[idx, 0])
        dirichlet[2 * int(idx) + 1] = float(exact_u[idx, 1])
    solved = solve_q8_plane(
        nodes,
        cells,
        young=YOUNG,
        poisson=POISSON,
        thickness=1.0,
        plane="stress",
        dirichlet=dirichlet,
    )

    element_coords = nodes[cells]
    element_areas = q8_element_area(element_coords).numpy()

    # --- Gauss-point stress error (volume weighted) -----------------------
    gauss_stress = solved.gauss_stress.numpy()
    numerator = 0.0
    denominator = 0.0
    exact_energy = 0.0
    for k, (xi, eta) in enumerate(GAUSS_POINTS_3X3):
        wa, wb = GAUSS_WEIGHTS_3X3[k]
        # sum(wa*wb) over the nine points is 4: element_areas already carries the
        # full rule, so the per-point share is area * wa*wb / 4.
        weight = element_areas * (wa * wb) / 4.0
        x = element_coords[:, :, 0].numpy() @ _q8_vals(xi, eta)
        y = element_coords[:, :, 1].numpy() @ _q8_vals(xi, eta)
        exx, eyy, exy = exact_stress_cartesian(x, y, hole)
        diff = gauss_stress[:, k, :] - np.stack((exx, eyy, exy), axis=1)
        numerator += float(np.sum(weight[:, None] * diff**2))
        denominator += float(np.sum(weight[:, None] * np.stack((exx, eyy, exy), axis=1) ** 2))
        exact_energy += float(np.sum(weight * exact_energy_density(x, y, hole)))
    stress_l2_gauss = math.sqrt(numerator / denominator)

    # --- recovered nodal stress error (lumped area weights) --------------
    nodal = solved.nodal_stress.numpy()
    lumped = np.zeros(len(nodes))
    for local in range(8):
        np.add.at(lumped, cells[:, local].numpy(), element_areas / 8.0)
    exx, eyy, exy = exact_stress_cartesian(coordinates[:, 0], coordinates[:, 1], hole)
    diff = nodal - np.stack((exx, eyy, exy), axis=1)
    stress_l2_nodal = math.sqrt(
        float(np.sum(lumped[:, None] * diff**2)) / float(np.sum(lumped[:, None] * np.stack((exx, eyy, exy), axis=1) ** 2))
    )

    # --- K_t at the pole node, the anchor's own definition ----------------
    target = np.array([0.0, INNER_RADIUS])
    index = int(np.argmin(((coordinates - target) ** 2).sum(axis=1)))
    kt = float(nodal[index, 0]) / TENSION

    # --- natural (traction free) hole rim from the recovered field -------
    on_hole = radius < INNER_RADIUS * (1.0 + 1e-9)
    th = theta[on_hole]
    s = nodal[on_hole]
    c, sn = np.cos(th), np.sin(th)
    srr_edge = s[:, 0] * c**2 + s[:, 1] * sn**2 + 2.0 * s[:, 2] * sn * c
    srt_edge = (s[:, 1] - s[:, 0]) * sn * c + s[:, 2] * (c**2 - sn**2)
    edge_residual = float(max(np.abs(srr_edge).max(), np.abs(srt_edge).max())) / TENSION
    edge_hoop_profile_rms = float(
        np.sqrt(np.mean((s[:, 0] * sn**2 + s[:, 1] * c**2 - 2.0 * s[:, 2] * sn * c - TENSION * (1.0 - 2.0 * np.cos(2.0 * th))) ** 2))
    ) / TENSION

    # --- displacement and energy ----------------------------------------
    displacement = solved.displacement.numpy().reshape(-1, 2)
    disp_rel = math.sqrt(float(np.sum((displacement - exact_u) ** 2)) / float(np.sum(exact_u**2)))
    energy_fe = float(solved.strain_energy)
    energy_rel = abs(energy_fe - exact_energy) / exact_energy

    # --- pointwise probes (fixed physical locations) ---------------------
    probes = []
    interior_margin = 1.0
    for r_over_a, deg in PROBES:
        r = r_over_a * INNER_RADIUS
        ang = math.radians(deg)
        point = np.array([r * math.cos(ang), r * math.sin(ang)])
        element = _element_of_point(r, ang, n_theta, n_radial, outer)
        xi, eta = _newton_parent_point(nodes, cells, element, point)
        interior_margin = min(interior_margin, 1.0 - max(abs(xi), abs(eta)))
        stress = q8_stress_at(
            nodes, cells, solved.displacement, element, float(xi), float(eta),
            young=YOUNG, poisson=POISSON, plane="stress",
        ).numpy()
        ex = exact_stress_cartesian(np.array([point[0]]), np.array([point[1]]), hole)
        exact_vec = np.array([ex[0][0], ex[1][0], ex[2][0]])
        probes.append(
            {
                "r_over_a": r_over_a,
                "theta_degrees": deg,
                "parent_xi": float(xi),
                "parent_eta": float(eta),
                "sigma_xx_over_T": float(stress[0]),
                "sigma_yy_over_T": float(stress[1]),
                "sigma_xy_over_T": float(stress[2]),
                "exact_sigma_xx_over_T": float(exact_vec[0]),
                "exact_sigma_yy_over_T": float(exact_vec[1]),
                "exact_sigma_xy_over_T": float(exact_vec[2]),
                "relative_error": float(np.linalg.norm(stress - exact_vec) / np.linalg.norm(exact_vec)),
            }
        )

    return {
        "n_theta": int(n_theta),
        "n_radial": int(n_radial),
        "outer_over_inner": OUTER_OVER_INNER,
        "elements": int(len(cells)),
        "nodes": int(len(nodes)),
        "dofs": int(solved.displacement.numel()),
        "h_theta": 2.0 * math.pi / n_theta,
        "kt": kt,
        "kt_rel_error": abs(kt - 3.0) / 3.0,
        "edge_traction_residual_over_T": edge_residual,
        "edge_hoop_profile_rms_over_T": edge_hoop_profile_rms,
        "stress_l2_rel_gauss": stress_l2_gauss,
        "stress_l2_rel_nodal": stress_l2_nodal,
        "displacement_l2_rel": float(disp_rel),
        "energy_fe": energy_fe,
        "energy_exact": exact_energy,
        "energy_rel_error": float(energy_rel),
        "probe_interior_margin_min": float(interior_margin),
        "probe_max_relative_error": float(max(p["relative_error"] for p in probes)),
        "probes": probes,
        "cg_relative_residual": float(solved.relative_residual),
        # field snapshot for the contour figure (finest level only, filled later)
        "_nodes": coordinates,
        "_nodal": nodal,
    }


def _q8_vals(xi: float, eta: float) -> np.ndarray:
    from tensorfem.quadratic_plane import q8_shape

    return q8_shape(xi, eta).numpy()


def observed_rate(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """``p`` from ``e ~ h^p`` between two levels with ``h`` divided by ``ratio``."""
    if fine <= 0.0 or coarse <= 0.0:
        return 0.0
    return math.log(coarse / fine) / math.log(ratio)


def build_gates(levels: list, oracle: dict) -> list:
    finest = levels[-1]
    rates = [
        observed_rate(levels[i]["stress_l2_rel_gauss"], levels[i + 1]["stress_l2_rel_gauss"])
        for i in range(len(levels) - 1)
    ]
    disp_rates = [
        observed_rate(levels[i]["displacement_l2_rel"], levels[i + 1]["displacement_l2_rel"])
        for i in range(len(levels) - 1)
    ]
    gates = [
        {
            "id": "oracle_edge_traction_free",
            "quantity": "max(|sigma_rr|,|sigma_rt|)/T on r=a of the analytic oracle",
            "reference": "Kirsch (1898); free circular hole",
            "value": oracle["hole_edge_traction_free_max_over_T"],
            "threshold": 1e-12,
            "operator": "<=",
            "passed": oracle["hole_edge_traction_free_max_over_T"] <= 1e-12,
        },
        {
            "id": "oracle_kt_value",
            "quantity": "analytic sigma_tt(a,pi/2)/T",
            "reference": "sigma_tt(a,pi/2)/T = 3",
            "value": oracle["kt_exact"],
            "threshold": 3.0,
            "operator": ">=",
            "passed": oracle["kt_exact"] >= 3.0,
        },
        {
            "id": "oracle_cartesian_vs_polar",
            "quantity": "max component mismatch between the polar and Cartesian oracle paths, /T",
            "reference": "one field, two evaluation routes (self-consistency)",
            "value": oracle["cartesian_vs_polar_oracle_max_over_T"],
            "threshold": 1e-12,
            "operator": "<=",
            "passed": oracle["cartesian_vs_polar_oracle_max_over_T"] <= 1e-12,
        },
        {
            "id": "levels_declared",
            "quantity": "refinement levels with both n_theta and n_radial doubled",
            "reference": ">= 3 levels for an observed rate",
            "value": len(levels),
            "threshold": CRITERIA["levels_min"],
            "operator": ">=",
            "passed": len(levels) >= CRITERIA["levels_min"],
        },
        {
            "id": "kt_rel_error_finest",
            "quantity": f"|K_t(n_theta={finest['n_theta']}) - 3|/3",
            "reference": "Kirsch limit 3 (pool tolerance 5 %)",
            "value": finest["kt_rel_error"],
            "threshold": CRITERIA["kt_rel_error_max"],
            "operator": "<=",
            "passed": finest["kt_rel_error"] <= CRITERIA["kt_rel_error_max"],
        },
        {
            "id": "stress_field_l2_gauss",
            "quantity": "||sigma_fe - sigma_ex||_L2 / ||sigma_ex||_L2 over all Gauss points (finest)",
            "reference": "domain-wide closed form, pure discretisation model",
            "value": finest["stress_l2_rel_gauss"],
            "threshold": CRITERIA["stress_l2_rel_gauss_max"],
            "operator": "<=",
            "passed": finest["stress_l2_rel_gauss"] <= CRITERIA["stress_l2_rel_gauss_max"],
        },
        {
            "id": "stress_field_l2_nodal",
            "quantity": "same norm for the recovered nodal field (the field K_t is read from)",
            "reference": "domain-wide closed form",
            "value": finest["stress_l2_rel_nodal"],
            "threshold": CRITERIA["stress_l2_rel_nodal_max"],
            "operator": "<=",
            "passed": finest["stress_l2_rel_nodal"] <= CRITERIA["stress_l2_rel_nodal_max"],
        },
        {
            "id": "stress_field_rate",
            "quantity": "min observed rate p of the Gauss-point stress L2 error (e ~ h^p)",
            "reference": "Q8 superconvergent rate, p >= 0.9",
            "value": min(rates),
            "threshold": CRITERIA["observed_rate_min"],
            "operator": ">=",
            "passed": min(rates) >= CRITERIA["observed_rate_min"],
            "note": f"pairwise rates {[round(r, 3) for r in rates]}",
        },
        {
            "id": "displacement_field",
            "quantity": "||u_fe - u_ex||/||u_ex|| on all nodes (finest)",
            "reference": "interior Dirichlet problem, exact field known",
            "value": finest["displacement_l2_rel"],
            "threshold": CRITERIA["displacement_l2_rel_max"],
            "operator": "<=",
            "passed": finest["displacement_l2_rel"] <= CRITERIA["displacement_l2_rel_max"],
            "note": f"pairwise rates {[round(r, 3) for r in disp_rates]}",
        },
        {
            "id": "energy_error",
            "quantity": "|U_fe - U_ex|/U_ex (finest)",
            "reference": "closed-form strain energy over the annulus",
            "value": finest["energy_rel_error"],
            "threshold": CRITERIA["energy_rel_error_max"],
            "operator": "<=",
            "passed": finest["energy_rel_error"] <= CRITERIA["energy_rel_error_max"],
        },
        {
            "id": "pointwise_probes",
            "quantity": "max relative stress error at five declared interior probes (finest)",
            "reference": "closed form at the same physical points",
            "value": finest["probe_max_relative_error"],
            "threshold": CRITERIA["probe_pointwise_rel_max"],
            "operator": "<=",
            "passed": finest["probe_max_relative_error"] <= CRITERIA["probe_pointwise_rel_max"],
            "note": f"probes {[list(pair) for pair in PROBES]}",
        },
        {
            "id": "edge_traction_residual",
            "quantity": "natural (free) hole-rim residual max(|sigma_rr|,|sigma_rt|)/T from nodal recovery (finest)",
            "reference": "traction-free hole: exactly zero",
            "value": finest["edge_traction_residual_over_T"],
            "threshold": CRITERIA["edge_traction_residual_max"],
            "operator": "<=",
            "passed": finest["edge_traction_residual_over_T"] <= CRITERIA["edge_traction_residual_max"],
        },
        {
            "id": "probes_interior",
            "quantity": "min over levels of the probe distance to its element boundary in parent coords (1 - max|xi|,|eta|)",
            "reference": "probe must be interior for a single-valued stress",
            "value": min(lv["probe_interior_margin_min"] for lv in levels),
            "threshold": 0.3,
            "operator": ">=",
            "passed": min(lv["probe_interior_margin_min"] for lv in levels) >= 0.3,
        },
    ]
    return gates


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(levels: list, out_dir: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    finest = levels[-1]
    nodes = finest["_nodes"]
    nodal = finest["_nodal"]
    hole = kirsch_hole()
    exx, eyy, exy = exact_stress_cartesian(nodes[:, 0], nodes[:, 1], hole)

    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1])
    xc = nodes[triangulation.triangles, 0].mean(axis=1)
    yc = nodes[triangulation.triangles, 1].mean(axis=1)
    triangulation.set_mask(np.hypot(xc, yc) < INNER_RADIUS * 1.001)

    panels = (
        (nodal[:, 0] / TENSION, r"FE $\sigma_{xx}/T$ (Q8, recovered)"),
        (np.asarray(exx) / TENSION, r"Kirsch exact $\sigma_{xx}/T$"),
        ((nodal[:, 0] - np.asarray(exx)) / TENSION, r"difference (FE $-$ exact)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    for ax, (values, title) in zip(axes, panels):
        limit = max(abs(values).max(), 1e-12)
        contour = ax.tricontourf(
            triangulation, values, levels=np.linspace(values.min(), values.max(), 24), cmap="coolwarm"
        )
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nrange [{values.min():.3f}, {values.max():.3f}]")
        ax.set_xlabel("x / a")
        fig.colorbar(contour, ax=ax, shrink=0.85)
    fig.suptitle(
        "Kirsch plate field check, Q8 annulus r/a in [1, {:g}], n_theta = {}, n_r = {}: "
        "relative stress L2 error = {:.4f} %".format(
            OUTER_OVER_INNER,
            finest["n_theta"],
            finest["n_radial"],
            100.0 * finest["stress_l2_rel_gauss"],
        )
    )
    path = out_dir / "kirsch_field_contours.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)

    h = np.array([lv["h_theta"] for lv in levels])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    series = (
        ("stress_l2_rel_gauss", r"$\|\sigma_{fe}-\sigma_{ex}\|_{L2}/\|\sigma_{ex}\|_{L2}$ (Gauss)"),
        ("stress_l2_rel_nodal", r"same, recovered nodal field"),
        ("displacement_l2_rel", r"$\|u_{fe}-u_{ex}\|/\|u_{ex}\|$"),
        ("energy_rel_error", r"$|U_{fe}-U_{ex}|/U_{ex}$"),
        ("kt_rel_error", r"$|K_t-3|/3$"),
    )
    for key, label in series:
        values = np.array([max(lv[key], 1e-16) for lv in levels])
        ax.loglog(h, values, "o-", label=label)
    guide = h ** 2
    ax.loglog(h, guide * levels[0]["displacement_l2_rel"] / guide[0], "k--", lw=1.0, label=r"slope 2")
    ax.set_xlabel(r"$h_\theta = 2\pi/n_\theta$")
    ax.set_ylabel("relative error")
    ax.invert_xaxis()
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.set_title("Kirsch annulus: error vs mesh size (pure discretisation model)")
    ax.legend(fontsize=8)
    path = out_dir / "kirsch_field_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path.name)
    return written


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main(argv: list | None = None) -> int:
    started = time.time()
    hole = kirsch_hole()
    oracle = oracle_field_selfcheck(hole)
    levels = []
    for n_theta, n_radial in LADDER:
        t0 = time.time()
        row = solve_level(n_theta, n_radial)
        row["wall_seconds"] = time.time() - t0
        print(
            f"n_theta={n_theta:4d} n_r={n_radial:3d}  K_t={row['kt']:.6f} "
            f"({row['kt_rel_error']:.3e})  sigma_L2_gauss={row['stress_l2_rel_gauss']:.3e}  "
            f"nodal={row['stress_l2_rel_nodal']:.3e}  probes={row['probe_max_relative_error']:.3e}  "
            f"edge={row['edge_traction_residual_over_T']:.3e}  {row['wall_seconds']:.1f}s"
        )
        levels.append(row)

    gates = build_gates(levels, oracle)
    payload = {
        "schema": "tensorfem.verified-field.v1",
        "anchor": "fem_kirsch_plate",
        "study": "kirsch_field_verification",
        "title": "KIRSCH-PLATE: domain-wide stress-field verification against the Kirsch closed form",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "model": {
            "geometry": f"Q8 polar annulus, a={INNER_RADIUS}, r_out={OUTER_OVER_INNER * INNER_RADIUS} (r_out/a={OUTER_OVER_INNER})",
            "boundary_conditions": "exact Kirsch displacement on r = r_out; hole rim traction free (natural BC)",
            "element": "Q8 serendipity plane stress, 3x3 Gauss, production path tensorfem.quadratic_plane.solve_q8_plane",
            "young": YOUNG,
            "poisson": POISSON,
            "thickness": 1.0,
            "far_field_tension": TENSION,
            "error_content": "discretisation only: the exact field solves this Dirichlet problem",
            "n_theta_x_n_radial": [list(pair) for pair in LADDER],
            "probes": [{"r_over_a": r, "theta_degrees": t} for r, t in PROBES],
        },
        "oracle": oracle,
        "criteria": CRITERIA,
        "levels": [{k: v for k, v in row.items() if not k.startswith("_")} for row in levels],
        "gates": gates,
        "gates_passed": sum(1 for gate in gates if gate["passed"]),
        "gates_total": len(gates),
        "wall_seconds": time.time() - started,
    }

    written = figures(levels, _ROOT / "docs" / "assets" / "kirsch_plate_field")
    payload["figures"] = written
    out = _HERE / "field_verification.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"gates {payload['gates_passed']}/{payload['gates_total']} -> {out}")
    failed = [gate["id"] for gate in gates if not gate["passed"]]
    if failed:
        print("FAILED GATES:", failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())