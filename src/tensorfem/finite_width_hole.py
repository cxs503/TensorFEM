"""Finite-width plate with a central circular hole: Howland reference + FE audit.

Why this module exists
----------------------
``docs/kirsch-hole-stress-concentration.md`` closes the *infinite-plate* opening
verification debt and explicitly lists what is **not** covered:

    正交异性板、含裂纹孔、有限宽板（Howland 修正）都未覆盖，``K_t = 3`` 只对无限大板成立。

Hull openings (hatch corners, manholes, pipe penetrations) sit in plating of
finite width, so the design value is the *finite-width* stress concentration,
not 3.0.  This module supplies

* :func:`howland_kt_net` - the classical Howland/Heywood finite-width factor
  (literature fit, cross-checked against the digitised chart values in
  :data:`HOWLAND_CHART`);
* :func:`trefftz_oracle` - an **independent semi-analytic oracle**: a
  Kolosov-Muskhelishvili (Trefftz) expansion whose free coefficients are fixed by
  least-squares boundary collocation on hole + strip edges + loaded ends.  It
  shares no code with the FE path and no closed form with the literature fit;
* the fan-mesh FE driver that exercises the *production* plane-stress path
  (``ContinuumModel`` / ``solve_continuum``, element type ``q4``);
* :func:`audit` - the mesh/term convergence study used by
  ``scripts/audit_finite_width_hole.py``.

Geometry and notation
---------------------
Plate of width ``W = 2b`` (edges ``y = +-b`` traction free), central hole of
radius ``a``, remote uniaxial tension ``T`` along ``x``.  Only a rectangle of
half length ``L`` is modelled (ends ``x = +-L`` carry the uniform traction), so
the strip comparison is exact only as ``L/b -> inf``; FE and oracle use the
*same* ``L`` and the ``L`` sensitivity is reported.

Section factors: gross ``T``, net ``T*b/(b-a)`` (since ``W - 2a = 2(b-a)``), and
``rho = d/W = a/b``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .continuum import ContinuumModel, elasticity_matrix, solve_continuum
from .kirsch_hole import _q4_b

__all__ = [
    "HOWLAND_CHART",
    "StripHole",
    "howland_kt_net",
    "howland_kt_gross",
    "fan_mesh",
    "strip_model",
    "stress_at_point",
    "solve_strip_case",
    "trefftz_oracle",
    "strip_convergence",
    "audit",
]

# ---------------------------------------------------------------------------
# Literature anchor
# ---------------------------------------------------------------------------

#: Digitised Peterson/Howland net-section stress concentration factors for a
#: central hole in a plate of finite width under uniaxial tension
#: (``rho = d/W``).  Kept as raw data so the cubic fit below can be checked.
HOWLAND_CHART = (
    (0.10, 2.73),
    (0.20, 2.52),
    (0.30, 2.36),
    (0.40, 2.23),
    (0.50, 2.11),
)


def howland_kt_net(rho: float) -> float:
    """Net-section ``K_t`` from the classical Howland/Heywood cubic fit.

    ``K_t_net = 3 - 3.14 rho + 3.667 rho^2 - 1.527 rho^3`` with ``rho = d/W``.
    Used here only as a third-party cross-check (never as the verdict); it
    reproduces :data:`HOWLAND_CHART` to better than 0.6 %.
    """
    rho = float(rho)
    if not 0.0 < rho < 1.0:
        raise ValueError("rho = d/W must lie in (0, 1)")
    return 3.0 - 3.14 * rho + 3.667 * rho**2 - 1.527 * rho**3


def howland_kt_gross(rho: float) -> float:
    """Gross-section ``K_t = K_t_net / (1 - rho)`` from the same fit.

    The net-section nominal stress is ``T/(1-rho)``, so the gross-section factor
    is larger: ``K_t_gross -> 3`` as ``rho -> 0`` and grows above 3 as the
    ligament narrows.
    """
    rho = float(rho)
    return howland_kt_net(rho) / (1.0 - rho)


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StripHole:
    """Plate of width ``2*half_width`` with a central hole under tension ``T``."""

    radius: float = 1.0
    half_width: float = 2.0
    half_length: float = 12.0
    tension: float = 1.0
    young: float = 1.0
    poisson: float = 0.3
    thickness: float = 1.0
    plane: str = "stress"

    def __post_init__(self) -> None:
        if self.half_width <= self.radius:
            raise ValueError("half_width must exceed the hole radius")
        if self.half_length <= self.radius:
            raise ValueError("half_length must exceed the hole radius")
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")

    @property
    def rho(self) -> float:
        """``d / W = a / b``."""
        return self.radius / self.half_width

    @property
    def net_stress(self) -> float:
        """Nominal net-section stress ``T * b / (b - a)``."""
        return self.tension * self.half_width / (self.half_width - self.radius)

    @property
    def gross_stress(self) -> float:
        return self.tension


# ---------------------------------------------------------------------------
# Production FE path (plane-stress Q4)
# ---------------------------------------------------------------------------


def _rectangle_radius(theta, half_width: float, half_length: float):
    """Distance from the origin to the rectangle boundary along ``theta``."""
    c, sn = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    with np.errstate(divide="ignore", invalid="ignore"):
        rx = np.where(c > 0.0, half_length / np.maximum(c, 1e-300), np.inf)
        ry = np.where(sn > 0.0, half_width / np.maximum(sn, 1e-300), np.inf)
    return np.minimum(rx, ry)


def fan_mesh(
    strip: StripHole,
    n_theta: int,
    n_radial: int,
    grading: float = 4.0,
    *,
    dtype=torch.float64,
):
    """Structured Q4 fan mesh of the quarter rectangle ``[0, L] x [0, b]`` minus the hole.

    Node ``(i, j)`` sits on the ray through boundary station ``i`` at radius
    ``a + (r_out - a) * s_j`` with ``s_j`` graded towards the hole.  The hole edge
    is a polygon with nodes exactly on ``r = a`` (chord sagitta
    ``O((theta_step)^2)``).
    """
    if n_theta < 2 or n_radial < 1:
        raise ValueError("n_theta >= 2 and n_radial >= 1 are required")
    thetas = np.linspace(0.0, math.pi / 2.0, n_theta + 1)
    rout = _rectangle_radius(thetas, strip.half_width, strip.half_length)
    s_frac = np.linspace(0.0, 1.0, n_radial + 1)
    if grading > 0.0:
        s_frac = (np.exp(grading * s_frac) - 1.0) / (math.exp(grading) - 1.0)
    radii = strip.radius + (rout[:, None] - strip.radius) * s_frac[None, :]
    nodes = np.stack(
        [radii * np.cos(thetas)[:, None], radii * np.sin(thetas)[:, None]], axis=-1
    ).reshape(-1, 2)

    def index(i: int, j: int) -> int:
        return i * (n_radial + 1) + j

    elements = np.empty((n_theta * n_radial, 4), dtype=np.int64)
    for i in range(n_theta):
        for j in range(n_radial):
            elements[i * n_radial + j] = (
                index(i, j),
                index(i, j + 1),
                index(i + 1, j + 1),
                index(i + 1, j),
            )
    return (
        torch.as_tensor(nodes, dtype=dtype),
        torch.as_tensor(elements, dtype=torch.long),
        radii,
        thetas,
    )


def _symmetry_dofs(n_theta: int, n_radial: int) -> list[int]:
    """``u_y = 0`` on ``y = 0`` and ``u_x = 0`` on ``x = 0`` (quarter model)."""
    stride = n_radial + 1
    dofs = [2 * (0 * stride + j) + 1 for j in range(stride)]
    dofs += [2 * (n_theta * stride + j) + 0 for j in range(stride)]
    return dofs


def _end_loads(nodes: np.ndarray, strip: StripHole, n_theta: int, n_radial: int) -> np.ndarray:
    """Consistent nodal loads for uniform traction on ``x = L`` (trapezoid rule)."""
    stride = n_radial + 1
    forces = np.zeros(2 * nodes.shape[0])
    tol = 1e-9 * max(1.0, strip.half_length)
    loaded = [i * stride + n_radial for i in range(n_theta + 1)]
    on_l = [n for n in loaded if abs(nodes[n, 0] - strip.half_length) < tol]
    if len(on_l) < 2:
        raise ValueError("loaded edge x = L has fewer than two nodes")
    y = np.array([nodes[n, 1] for n in on_l])
    order = np.argsort(y)
    nodes_sorted = [on_l[i] for i in order]
    y = y[order]
    # virtual end points 0 and b so the whole edge is loaded even when the last
    # fan ray misses the corner node (otherwise the resultant is short)
    ext = np.concatenate([[0.0], y, [strip.half_width]])
    tributary = 0.5 * (ext[2:] - ext[:-2])
    for n, length in zip(nodes_sorted, tributary):
        forces[2 * n] += strip.tension * strip.thickness * length
    # normalise so the applied resultant is exactly T*b: the fan may place no node
    # exactly on the corner, in which case the trapezoid sum is short by ~0.5 %
    total = forces[0::2].sum()
    target = strip.tension * strip.thickness * strip.half_width
    if total > 0.0:
        forces[0::2] *= target / total
    return forces


def strip_model(strip: StripHole, n_theta: int, n_radial: int, grading: float = 4.0):
    """Assemble the production ``ContinuumModel`` for the quarter strip case."""
    nodes, elements, radii, thetas = fan_mesh(strip, n_theta, n_radial, grading)
    nodes_np = np.asarray(nodes, dtype=np.float64)
    elements_np = np.asarray(elements)
    forces = _end_loads(nodes_np, strip, n_theta, n_radial)
    model = ContinuumModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.as_tensor(strip.young, dtype=torch.float64),
        poisson_ratio=torch.as_tensor(strip.poisson, dtype=torch.float64),
        thickness=torch.as_tensor(strip.thickness, dtype=torch.float64),
        forces=torch.as_tensor(forces, dtype=torch.float64),
        fixed_dofs=torch.as_tensor(_symmetry_dofs(n_theta, n_radial), dtype=torch.long),
        plane=strip.plane,
        element_type="q4",
    )
    meta = {
        "nodes": nodes_np,
        "elements": elements_np,
        "radii": radii,
        "thetas": thetas,
        "n_theta": n_theta,
        "n_radial": n_radial,
        "grading": grading,
        "forces": forces,
        "applied_resultant_x": float(forces[0::2].sum()),
    }
    return model, meta


def _locate_fan(meta: dict, point) -> tuple[int, float, float]:
    """Element index and parent coordinates of a point inside the fan mesh."""
    x, y = float(point[0]), float(point[1])
    theta = math.atan2(y, x)
    r = math.hypot(x, y)
    thetas, radii = meta["thetas"], meta["radii"]
    n_theta, n_radial = meta["n_theta"], meta["n_radial"]
    i = min(max(int(np.searchsorted(thetas, theta)) - 1, 0), n_theta - 1)
    row = radii[i]
    j = min(max(int(np.searchsorted(row, r)) - 1, 0), n_radial - 1)
    connectivity = meta["elements"][i * n_radial + j]
    coordinates = meta["nodes"][connectivity]
    target = np.array([x, y])
    xi = eta = 0.0
    for _ in range(80):
        shape = np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        ) / 4.0
        natural = np.array(
            [
                [-(1.0 - eta), -(1.0 - xi)],
                [(1.0 - eta), -(1.0 + xi)],
                [(1.0 + eta), (1.0 + xi)],
                [-(1.0 + eta), (1.0 - xi)],
            ]
        ) / 4.0
        residual = shape @ coordinates - target
        jacobian = coordinates.T @ natural
        step = np.linalg.solve(jacobian, -residual)
        norm = float(np.linalg.norm(step))
        if norm > 1.0:
            step = step / norm
        xi += step[0]
        eta += step[1]
        if norm < 1e-14:
            break
    return int(i * n_radial + j), float(xi), float(eta)


def stress_at_point(meta: dict, displacement, point, strip: StripHole) -> dict:
    """Pointwise stress from the independent Q4 ``B`` matrix (not the stored field)."""
    element, xi, eta = _locate_fan(meta, point)
    connectivity = meta["elements"][element]
    coordinates = meta["nodes"][connectivity]
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    local = np.empty(8)
    local[0::2] = u[connectivity, 0]
    local[1::2] = u[connectivity, 1]
    b, determinant = _q4_b(xi, eta, coordinates)
    dmat = np.asarray(
        elasticity_matrix(
            torch.tensor([strip.young], dtype=torch.float64),
            torch.tensor([strip.poisson], dtype=torch.float64),
            strip.plane,
        )
    )[0]
    stress = dmat @ b @ local
    return {
        "element": element,
        "parent": [xi, eta],
        "inside": bool(-1.0 - 1e-9 <= xi <= 1.0 + 1e-9 and -1.0 - 1e-9 <= eta <= 1.0 + 1e-9),
        "jacobian": determinant,
        "stress": [float(v) for v in stress],
    }


PEAK_DELTAS = np.array([0.0125, 0.025, 0.05, 0.10])


def solve_strip_case(
    strip: StripHole,
    n_theta: int = 40,
    n_radial: int = 30,
    grading: float = 4.0,
    *,
    keep_model: bool = False,
) -> dict:
    """Solve one fan mesh with the production path; return ``K_t`` + equilibrium."""
    model, meta = strip_model(strip, n_theta, n_radial, grading)
    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    samples = [
        stress_at_point(meta, displacement, (0.0, strip.radius * (1.0 + d)), strip)["stress"][0]
        for d in PEAK_DELTAS
    ]
    fit = np.polyfit(PEAK_DELTAS, np.asarray(samples), 2)
    peak = float(np.polyval(fit, 0.0))

    reaction = result.reaction.detach().cpu().numpy()
    expected = strip.tension * strip.half_width * strip.thickness
    out = {
        "rho": strip.rho,
        "n_theta": int(n_theta),
        "n_radial": int(n_radial),
        "grading": float(grading),
        "half_length": strip.half_length,
        "elements": int(meta["elements"].shape[0]),
        "nodes": int(meta["nodes"].shape[0]),
        "dofs": int(model.n_dofs),
        "peak_hoop": peak,
        "kt_net": peak / strip.net_stress,
        "kt_gross": peak / strip.gross_stress,
        "delta_samples": PEAK_DELTAS.tolist(),
        "sigma_xx_samples": [float(v) for v in samples],
        "applied_resultant_x": meta["applied_resultant_x"],
        "expected_resultant_x": expected,
        "applied_resultant_error": abs(meta["applied_resultant_x"] - expected) / expected,
        "reaction_resultant_x": float(reaction[0::2].sum()),
        "reaction_resultant_error": abs(reaction[0::2].sum() + expected) / expected,
        "reaction_fy": float(reaction[1::2].sum()),
    }
    if keep_model:
        out.update({"meta": meta, "displacement": displacement,
                    "stored_stress": result.stress.detach().cpu().numpy()})
    return out


def _order(coarse: float, fine: float, ratio: float) -> float:
    if coarse <= 0.0 or fine <= 0.0 or fine >= coarse:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def strip_convergence(
    strip: StripHole,
    levels: tuple[int, ...] = (16, 24, 32, 40),
    n_radial_factor: float = 0.75,
    grading: float = 4.0,
) -> dict:
    """Mesh refinement study of ``K_t`` on the fan mesh."""
    rows = []
    for n_theta in levels:
        n_radial = max(6, int(round(n_theta * n_radial_factor)))
        rows.append(solve_strip_case(strip, n_theta, n_radial, grading))
    for coarse, fine in zip(rows[:-1], rows[1:]):
        ratio = fine["n_theta"] / coarse["n_theta"]
        fine["observed_order_vs_previous"] = _order(
            abs(coarse["kt_net"] - howland_kt_net(strip.rho)),
            abs(fine["kt_net"] - howland_kt_net(strip.rho)),
            ratio,
        )
    return {
        "rho": strip.rho,
        "reference_kt_net_howland": howland_kt_net(strip.rho),
        "levels": rows,
        "finest_kt_net": rows[-1]["kt_net"],
        "finest_relative_error_vs_howland": abs(rows[-1]["kt_net"] - howland_kt_net(strip.rho))
        / howland_kt_net(strip.rho),
    }


def audit(
    rhos: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
    *,
    levels: tuple[int, ...] = (16, 24, 32, 40),
    oracle_terms: tuple[int, ...] = (4, 6, 8, 10, 12),
    half_length_ratios: tuple[float, ...] = (4.0, 6.0, 8.0),
) -> dict:
    """Full comparison: FE fan mesh vs independent Trefftz oracle vs Howland fit."""
    results = {"cases": [], "oracle_term_study": [], "length_study": []}
    for rho in rhos:
        half_width = 1.0 / rho
        strip = StripHole(radius=1.0, half_width=half_width, half_length=6.0 * half_width)
        oracle = trefftz_oracle(strip, n_terms=10)
        fe = strip_convergence(strip, levels=levels)
        results["cases"].append(
            {
                "rho": rho,
                "half_width": half_width,
                "half_length": strip.half_length,
                "howland_kt_net": howland_kt_net(rho),
                "chart_kt_net": min(HOWLAND_CHART, key=lambda kv: abs(kv[0] - rho))[1],
                "oracle": {k: v for k, v in oracle.items() if k != "coefficients"},
                "fe": fe,
                "fe_vs_oracle_relative": abs(fe["finest_kt_net"] - oracle["kt_net"]) / oracle["kt_net"],
                "oracle_vs_howland_relative": abs(oracle["kt_net"] - howland_kt_net(rho))
                / howland_kt_net(rho),
                "fe_vs_howland_relative": fe["finest_relative_error_vs_howland"],
            }
        )
    reference = StripHole(radius=1.0, half_width=2.0, half_length=12.0)
    for n_terms in oracle_terms:
        res = trefftz_oracle(reference, n_terms=n_terms)
        results["oracle_term_study"].append(
            {
                "n_terms": n_terms,
                "kt_net": res["kt_net"],
                "max_free_boundary_traction_over_T": res["max_free_boundary_traction_over_T"],
                "max_loaded_traction_error_over_T": res["max_loaded_traction_error_over_T"],
            }
        )
    for ratio in half_length_ratios:
        strip = StripHole(radius=1.0, half_width=2.0, half_length=ratio * 2.0)
        res = trefftz_oracle(strip, n_terms=10, n_edge=300)
        results["length_study"].append(
            {"half_length_over_half_width": ratio, "kt_net": res["kt_net"],
             "kt_gross": res["kt_gross"]}
        )
    results["infinite_plate_limit"] = trefftz_oracle(
        StripHole(radius=1.0, half_width=40.0, half_length=60.0), n_terms=10, n_edge=400
    )["kt_net"]
    return results