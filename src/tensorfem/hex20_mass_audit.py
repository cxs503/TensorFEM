"""Independent oracles for the HEX20 inertia operator.

``tensorfem.hex20`` now ships an inertia operator (``hex20_mass`` and friends),
but a mass matrix is the one place where "it looks symmetric and the total mass
matches" is *not* evidence: any wrong Jacobian, wrong node ordering or wrong
quadrature still reproduces ``rho V``.  This module therefore rebuilds every
check from a second, deliberately separate code path:

* :func:`reference_mass` re-derives the serendipity shapes in NumPy (own node
  table, own derivative code, high-order Gauss) so a shared typing error in the
  torch kernels cannot cancel out,
* :func:`rigid_rotation_energy` compares the mass-weighted kinetic energy of a
  rigid rotation with the analytic box inertia tensor -- this pins the *spatial
  distribution* of mass, not just its sum,
* :func:`axial_modes` compares full 3-D HEX20 rod eigenfrequencies with
  ``omega_n = (2n-1) pi / (2L) * sqrt(E / rho)``, which is only recovered when
  the lateral contraction (hence the ``E`` rather than ``lambda + 2 mu``
  modulus) is resolved by the element itself,
* :func:`free_free_modes` checks that the six rigid-body modes of the assembled
  ``(K, M)`` pair are numerically zero -- a global consistency check of the
  stiffness/mass pairing,
* :func:`lumping_report` measures the row-sum lumping trap (negative corner
  masses) instead of asserting it.

Every number quoted in ``docs/hex20-consistent-mass.md`` is produced here, so
prose and code cannot drift apart.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .hex20 import (
    hex20_lumped_mass,
    hex20_mass,
    hex20_mass_totals,
    hex20_stiffness,
    hex_gauss_rule,
    structured_hex20_mesh,
)
from .solid3d import elasticity_matrix_3d

__all__ = [
    "REFERENCE_CORNERS",
    "REFERENCE_EDGES",
    "reference_shapes",
    "reference_mass",
    "reference_volume",
    "box_inertia",
    "rigid_velocity_energy",
    "rigid_rotation_energy",
    "axial_rod_frequencies",
    "axial_modes",
    "constrained_rod_modes",
    "free_free_modes",
    "lumping_report",
    "rule_error_table",
    "curved_element_residual",
    "distortion_scope",
    "build_evidence",
]

# Node table of the reference implementation, written out longhand (it is
# intentionally not imported from tensorfem.hex20: an independent oracle must
# not share the production node table).
REFERENCE_CORNERS = (
    (-1.0, -1.0, -1.0), (+1.0, -1.0, -1.0), (+1.0, +1.0, -1.0), (-1.0, +1.0, -1.0),
    (-1.0, -1.0, +1.0), (+1.0, -1.0, +1.0), (+1.0, +1.0, +1.0), (-1.0, +1.0, +1.0),
)
REFERENCE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def reference_shapes(r: float, s: float, t: float) -> np.ndarray:
    """Serendipity shape values ``N[20]`` at one reference point (NumPy oracle)."""
    n = np.zeros(20)
    for i, (a, b, c) in enumerate(REFERENCE_CORNERS):
        n[i] = (
            0.125 * (1.0 + a * r) * (1.0 + b * s) * (1.0 + c * t)
            * (a * r + b * s + c * t - 2.0)
        )
    for k, (i, j) in enumerate(REFERENCE_EDGES):
        a, b, c = REFERENCE_CORNERS[i]
        aa, bb, cc = REFERENCE_CORNERS[j]
        if a != aa:
            n[8 + k] = 0.25 * (1.0 - r * r) * (1.0 + b * s) * (1.0 + c * t)
        elif b != bb:
            n[8 + k] = 0.25 * (1.0 - s * s) * (1.0 + a * r) * (1.0 + c * t)
        else:
            n[8 + k] = 0.25 * (1.0 - t * t) * (1.0 + a * r) * (1.0 + b * s)
    return n


def reference_jacobian(x: np.ndarray, r: float, s: float, t: float, h: float = 1e-6):
    """Jacobian and its determinant by central differences (no shared derivative code)."""
    def mapped(rr: float, ss: float, tt: float):
        return reference_shapes(rr, ss, tt) @ x

    dr = (mapped(r + h, s, t) - mapped(r - h, s, t)) / (2 * h)
    ds = (mapped(r, s + h, t) - mapped(r, s - h, t)) / (2 * h)
    dt = (mapped(r, s, t + h) - mapped(r, s, t - h)) / (2 * h)
    jac = np.stack((dr, ds, dt), 1)
    return jac, float(np.linalg.det(jac))


def reference_mass(x, density, *, order: int = 8) -> np.ndarray:
    """High-order NumPy oracle for ``rho * int N_i N_j detJ`` (``[e,20,20]``)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None]
    rho = np.atleast_1d(np.asarray(density, dtype=float))
    if rho.size == 1:
        rho = np.full(len(x), float(rho[0]))
    pts, wts = np.polynomial.legendre.leggauss(order)
    out = np.zeros((len(x), 20, 20))
    cache = {}
    for r, wr in zip(pts, wts):
        for s, ws in zip(pts, wts):
            for t, wt in zip(pts, wts):
                key = (float(r), float(s), float(t))
                if key not in cache:
                    cache[key] = reference_shapes(r, s, t)
                n = cache[key]
                for e in range(len(x)):
                    _, det = reference_jacobian(x[e], r, s, t)
                    out[e] += rho[e] * n[:, None] * n[None, :] * det * wr * ws * wt
    return out


def reference_volume(x) -> np.ndarray:
    """Element volume from the independent Jacobian (``[e]``)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None]
    pts, wts = np.polynomial.legendre.leggauss(8)
    vol = np.zeros(len(x))
    for r, wr in zip(pts, wts):
        for s, ws in zip(pts, wts):
            for t, wt in zip(pts, wts):
                for e in range(len(x)):
                    _, det = reference_jacobian(x[e], r, s, t)
                    vol[e] += det * wr * ws * wt
    return vol


def box_inertia(mass: float, lengths) -> np.ndarray:
    """Principal inertia tensor of a solid cuboid (diagonal, about its centre)."""
    a, b, c = (float(v) for v in lengths)
    return np.diag(
        [
            mass * (b * b + c * c) / 12.0,
            mass * (a * a + c * c) / 12.0,
            mass * (a * a + b * b) / 12.0,
        ]
    )


def rigid_velocity_energy(block: torch.Tensor, velocity) -> float:
    """Kinetic energy of a constant velocity field, ``0.5 * rho int |v|^2 dV``."""
    v = torch.as_tensor(velocity, dtype=block.dtype)
    vel = v.expand(20, 3).reshape(-1)
    return float(0.5 * vel @ block @ vel)


def rigid_rotation_energy(block: torch.Tensor, x: torch.Tensor, omega, about=None) -> float:
    """Kinetic energy of rigid rotation ``omega`` about ``about`` (single element)."""
    centre = x.new_zeros(3) if about is None else torch.as_tensor(about, dtype=x.dtype)
    w = torch.as_tensor(omega, dtype=x.dtype)
    r = x.reshape(20, 3) - centre
    vel = torch.cross(w.expand_as(r), r, dim=1).reshape(-1)
    return float(0.5 * vel @ block @ vel)


def axial_rod_frequencies(mode: int, *, young: float, density: float, length: float) -> float:
    """Exact fixed-free axial rod frequency (mode 1 is the fundamental)."""
    return (2 * mode - 1) * math.pi / (2 * length) * math.sqrt(young / density)


def constrained_rod_frequencies(mode: int, *, young: float, poisson: float,
                                density: float, length: float) -> float:
    """Exact fixed-free rod frequency when lateral strain is suppressed.

    With ``eps_yy = eps_zz = 0`` the axial modulus is the constrained modulus
    ``lambda + 2 mu = E (1 - nu) / ((1 + nu)(1 - 2 nu))``, i.e. +16 % in wave
    speed at ``nu = 0.3`` relative to ``sqrt(E/rho)``.  Quoting ``E`` for a
    laterally clamped model is a mesh-independent 16 % bias, not a
    discretisation error.
    """
    from .solid3d_mass import constrained_modulus

    return (2 * mode - 1) * math.pi / (2 * length) * math.sqrt(
        constrained_modulus(young, poisson) / density
    )


def _rod_model(nex: int, *, young: float, density: float, length: float,
               width: float, height: float):
    nodes, cells = structured_hex20_mesh(length, width, height, nex, 1, 1)
    d = elasticity_matrix_3d(
        nodes.new_full((len(cells),), young), nodes.new_full((len(cells),), 0.3)
    )
    ke, _ = hex20_stiffness(nodes[cells], d)
    me = hex20_mass(nodes[cells], density)
    ed = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 60).long()
    n_dofs = 3 * len(nodes)
    rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1)
    cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1)
    k = nodes.new_zeros((n_dofs, n_dofs)).index_put((rows, cols), ke.reshape(-1), accumulate=True)
    m = nodes.new_zeros((n_dofs, n_dofs)).index_put((rows, cols), me.reshape(-1), accumulate=True)
    return nodes, k, m


def _generalized(k: torch.Tensor, m: torch.Tensor):
    """Generalized symmetric eigenproblem ``K phi = lambda M phi`` (M SPD).

    Uses the Cholesky reduction ``L^-1 K L^-T y = lambda y`` with ``M = L L^T``
    and returns mass-normalised eigenvectors ``phi = L^-T y``.  Treating ``M`` as
    diagonal here would silently bias every frequency low (the consistent mass
    has positive off-diagonals), so the reduction is done properly.
    """
    k = 0.5 * (k + k.t())
    m = 0.5 * (m + m.t())
    chol = torch.linalg.cholesky(m)
    reduced = torch.linalg.solve_triangular(chol, k, upper=False)
    reduced = torch.linalg.solve_triangular(chol, reduced.t(), upper=False).t()
    lam, vec = torch.linalg.eigh(0.5 * (reduced + reduced.t()))
    phi = torch.linalg.solve_triangular(chol, vec, upper=False)
    return lam, phi


def axial_modes(nex: int, *, modes: int = 3, young: float = 200e9,
                density: float = 7850.0, length: float = 10.0,
                width: float = 0.2, height: float = 0.2) -> dict:
    """Full 3-D HEX20 cantilever rod: identify axial modes and compare with theory.

    Modes are classified by the axial participation ``phi_x^T M phi_x / phi^T M
    phi`` so that flexural/torsional modes cannot be mistaken for axial ones --
    the axial spectrum only reproduces ``sqrt(E/rho)`` because the element
    resolves lateral contraction by itself.
    """
    nodes, k, m = _rod_model(nex, young=young, density=density, length=length,
                             width=width, height=height)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3 * left, 3 * left + 1, 3 * left + 2), 1).reshape(-1)
    free = torch.ones(k.shape[0], dtype=torch.bool)
    free[fixed] = False
    lam, phi = _generalized(k[free][:, free], m[free][:, free])
    lam = torch.clamp(lam, min=0.0)
    omega = lam.sqrt()
    # The reduced basis keeps the global ordering, so x-DOFs are every third row.
    xmask = (torch.arange(int(free.sum())) % 3) == 0
    phi_x = phi.clone()
    phi_x[~xmask] = 0.0
    mr = m[free][:, free]
    num = torch.einsum("ij,ik,kj->j", phi_x, mr, phi_x)
    den = torch.einsum("ij,ik,kj->j", phi, mr, phi)
    axial_fraction = (num / den).clamp(0.0, 1.0)
    order = torch.argsort(lam)
    picked = [(int(i), float(omega[i]), float(axial_fraction[i])) for i in order
              if float(axial_fraction[i]) > 0.5][:modes]
    rows, previous = [], None
    for idx, w, frac in picked:
        previous = w if previous is None else previous
        rows.append({"eigen_index": idx, "axial_fraction": frac, "omega_fe": w})
    for n, row in enumerate(rows, start=1):
        exact = axial_rod_frequencies(n, young=young, density=density, length=length)
        row["mode"] = n
        row["omega_exact"] = exact
        row["rel_error"] = (row["omega_fe"] - exact) / exact
    return {"nex": nex, "n_free_dofs": int(free.sum()), "rows": rows}


def constrained_rod_modes(nex: int, *, modes: int = 2, young: float = 200e9,
                          poisson: float = 0.3, density: float = 7850.0,
                          length: float = 10.0, width: float = 0.2,
                          height: float = 0.2) -> dict:
    """1-D reduction (transverse DOFs held): the exact reference is the *constrained* rod.

    This isolates the axial operator from flexural/torsional motion, so the
    convergence rate of the quadratic element can be measured without mode
    classification ambiguity.
    """
    nodes, k, m = _rod_model(nex, young=young, density=density, length=length,
                             width=width, height=height)
    x_dofs = torch.arange(0, k.shape[0], 3)
    free = x_dofs[nodes[x_dofs // 3, 0] > 0]
    lam, _phi = _generalized(k[free][:, free], m[free][:, free])
    omega = torch.clamp(lam, min=0.0).sqrt()
    rows = []
    for n in range(1, modes + 1):
        w = float(omega[n - 1])
        exact = constrained_rod_frequencies(n, young=young, poisson=poisson,
                                            density=density, length=length)
        rows.append({"mode": n, "omega_fe": w, "omega_exact": exact,
                     "rel_error": (w - exact) / exact})
    return {"nex": nex, "rows": rows}


def free_free_modes(nex: int = 2, *, young: float = 200e9, density: float = 7850.0,
                    length: float = 10.0, width: float = 1.0,
                    height: float = 1.0) -> dict:
    """Six rigid-body modes of the assembled ``(K, M)`` pair must be zero."""
    nodes, k, m = _rod_model(nex, young=young, density=density, length=length,
                             width=width, height=height)
    lam_m = torch.linalg.eigvalsh(0.5 * (m + m.t())).clamp(min=0.0)
    inv = torch.diag(lam_m.rsqrt())
    mu = torch.clamp(torch.linalg.eigvalsh(inv @ (0.5 * (k + k.t())) @ inv), min=0.0).sqrt()
    scale = float(mu.max())
    return {
        "nex": nex,
        "rigid_first_six": [float(v) for v in mu[:6]],
        "smallest_elastic": float(mu[6]),
        "rigid_max_over_max": float(mu[5]) / scale,
        "first_elastic_over_max": float(mu[6]) / scale,
    }


def lumping_report(nodes=None, cells=None, *, density: float = 7850.0) -> dict:
    """HRZ vs row-sum lumping on a unit cube: positivity, total mass, corner mass.

    The lumped vector stores three entries per node (one per displacement
    component), so the *per-component* total is ``row[0::3].sum()`` and it must
    equal the element mass ``rho V``.
    """
    if nodes is None:
        nodes, cells = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    x = nodes[cells]
    element_mass = float(hex20_mass_totals(x, density).sum())
    hrz = hex20_lumped_mass(x, density, scheme="hrz")[0]
    row = hex20_lumped_mass(x, density, scheme="row")[0]
    return {
        "element_mass": element_mass,
        "hrz_min": float(hrz.min()),
        "hrz_min_over_node_avg": float(hrz.min()) / (element_mass / 20.0),
        "hrz_component_total_ratio": float(hrz[0::3].sum()) / element_mass,
        "row_min": float(row.min()),
        "row_negative_component_count": int((row < 0).sum()),
        "row_component_total_ratio": float(row[0::3].sum()) / element_mass,
        "row_corner_mass": float(row[0]),
        "row_corner_over_element_mass": float(row[0]) / element_mass,
        "row_edge_mass": float(row[3 * 8]),
        "row_edge_over_element_mass": float(row[3 * 8]) / element_mass,
    }


def rule_error_table(*, nodes=None, cells=None, density: float = 7850.0) -> list:
    """Mass-matrix error and total-mass error for Gauss orders 1..4."""
    if nodes is None:
        nodes, cells = structured_hex20_mesh(1.0, 2.0, 3.0, 1, 1, 1)
    x = nodes[cells]
    exact = reference_mass(x, density)
    exact_t = torch.as_tensor(exact, dtype=x.dtype)
    exact_tot = float(reference_volume(x).sum() * density)
    rows = []
    for order in (1, 2, 3, 4):
        block = hex20_mass(x, density, rule=hex_gauss_rule(order, dtype=x.dtype)).reshape(
            len(x), 20, 3, 20, 3
        )
        block = block[:, :, 0, :, 0].reshape(len(x), 20, 20)
        rel = float((block - exact_t).norm() / exact_t.norm())
        tot = float(hex20_mass_totals(x, density, rule=hex_gauss_rule(order, dtype=x.dtype)).sum())
        rows.append({"order": order, "rel_frobenius": rel,
                     "total_mass_rel": (tot - exact_tot) / exact_tot})
    return rows


def distortion_scope(*, density: float = 7850.0) -> dict:
    """Where the order-3 rule stops being exact: affine vs non-planar distortion."""
    nodes, cells = structured_hex20_mesh(1.0, 2.0, 3.0, 1, 1, 1)
    base = nodes[cells].clone()

    def rel(block, ref):
        block = block.reshape(len(block), 20, 3, 20, 3)[:, :, 0, :, 0].reshape(len(block), 20, 20)
        return float((block - ref).norm() / ref.norm())

    out = {}
    # (a) affine shear (parallelepiped): det J constant -> rule stays exact
    shear = base.clone()
    shear[..., 0] = base[..., 0] + 0.4 * base[..., 2]
    ref = torch.as_tensor(reference_mass(shear, density), dtype=base.dtype)
    out["affine_shear_rule3_reference_rel"] = rel(hex20_mass(shear, density), ref)
    # (b) single-corner perturbation: faces cease to be planar -> integrand not polynomial
    bent = base.clone()
    bent[0, 6, 2] += 0.3
    ref = torch.as_tensor(reference_mass(bent, density), dtype=base.dtype)
    out["nonplanar_rule3_reference_rel"] = rel(hex20_mass(bent, density), ref)
    ref4 = hex20_mass(bent, density, rule=hex_gauss_rule(4, dtype=base.dtype))
    out["nonplanar_rule4_reference_rel"] = rel(ref4, ref)
    out["nonplanar_rule3_vs_rule4_rel"] = float(
        (hex20_mass(bent, density).reshape(20, 3, 20, 3)[:, 0, :, 0]
         - ref4.reshape(20, 3, 20, 3)[:, 0, :, 0]).norm()
        / ref4.reshape(20, 3, 20, 3)[:, 0, :, 0].norm()
    )
    return out


def curved_element_residual(*, bulge: float = 0.2, density: float = 7850.0) -> dict:
    """Curved HEX20: the order-3 rule is *not* exact -- measure the residual."""
    nodes, cells = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    x = nodes[cells].clone()
    # push one vertical mid-edge node inward on the (x=1, y=1) face -> curved faces
    x[0, 14, 2] += bulge  # node 14: between corners 2 and 6 (edge (2,6))
    ref = torch.as_tensor(reference_mass(x, density), dtype=x.dtype)
    out = {}
    for order in (3, 4, 5):
        block = hex20_mass(x, density, rule=hex_gauss_rule(order, dtype=x.dtype)).reshape(
            len(x), 20, 3, 20, 3
        )[:, :, 0, :, 0].reshape(len(x), 20, 20)
        out[f"order{order}"] = {
            "rel_frobenius": float((block - ref).norm() / ref.norm()),
        }
    vol = float(reference_volume(x)[0]) * density
    out["total_mass_order3_rel"] = (
        float(hex20_mass_totals(x, density).sum()) - vol
    ) / vol
    out["bulge"] = bulge
    out["volume"] = vol / density
    return out


def build_evidence() -> dict:
    """Assemble every number quoted in the documentation (single source of truth)."""
    nodes, cells = structured_hex20_mesh(1.0, 2.0, 3.0, 1, 1, 1)
    x = nodes[cells]
    density = 7850.0
    full = hex20_mass(x, density)[0]
    block = full.reshape(20, 3, 20, 3)[:, 0, :, 0]
    ref = torch.as_tensor(reference_mass(x, density), dtype=x.dtype)
    energy_v = rigid_velocity_energy(full, (0.7, -1.1, 2.3))
    volume = float(reference_volume(x)[0])
    energy_v_exact = 0.5 * density * volume * (0.7 ** 2 + 1.1 ** 2 + 2.3 ** 2)
    inertia = box_inertia(density * volume, (1.0, 2.0, 3.0))
    centre = torch.tensor([0.5, 1.0, 1.5], dtype=x.dtype)
    rot = [rigid_rotation_energy(full[None], x, w, about=centre)
           for w in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
    rot_exact = [0.5 * float(inertia[i, i]) for i in range(3)]

    evidence = {
        "schema": "tensorfem-hex20-mass-evidence/1",
        "element": "HEX20 serendipity brick, 60 DOF",
        "default_rule": "tensor-product Gauss order 3 (27 points) on [-1,1]^3",
        "straight_element": {
            "volume": volume,
            "volume_rel_error": (float(hex20_mass_totals(x, density).sum()) / (density * volume)) - 1.0,
            "mass_vs_reference_frobenius": float((block - ref).norm() / ref.norm()),
            "mass_vs_reference_max_abs": float((block - ref).abs().max()),
            "symmetry_max_abs": float((block - block.transpose(-1, -2)).abs().max()),
            "row_sum_total_mass": float(block.sum()),
        },
        "kinetic_energy": {
            "translation_fe": energy_v,
            "translation_exact": energy_v_exact,
            "translation_rel_error": (energy_v - energy_v_exact) / energy_v_exact,
            "rotation_fe": rot,
            "rotation_exact": rot_exact,
            "rotation_rel_error": [(a - b) / b for a, b in zip(rot, rot_exact)],
        },
        "rules": rule_error_table(),
        "curved": curved_element_residual(),
        "lumping": lumping_report(),
        "distortion": distortion_scope(),
        "free_free": free_free_modes(2),
        "rod_full3d": [axial_modes(n) for n in (2, 4, 8)],
        "rod_constrained": [constrained_rod_modes(n) for n in (2, 4, 8, 16)],
    }
    observed = []
    for prev, cur in zip(evidence["rod_constrained"], evidence["rod_constrained"][1:]):
        e0 = abs(prev["rows"][0]["rel_error"])
        e1 = abs(cur["rows"][0]["rel_error"])
        observed.append(math.log2(e0 / e1) if e1 > 0 else float("inf"))
    evidence["rod_constrained_observed_order"] = observed
    observed3d = []
    for prev, cur in zip(evidence["rod_full3d"], evidence["rod_full3d"][1:]):
        e0 = abs(prev["rows"][0]["rel_error"])
        e1 = abs(cur["rows"][0]["rel_error"])
        observed3d.append(math.log2(e0 / e1) if e1 > 0 else float("inf"))
    evidence["rod_full3d_observed_order"] = observed3d
    return evidence


if __name__ == "__main__":  # pragma: no cover - thin CLI
    import json

    print(json.dumps(build_evidence(), indent=2, default=float))