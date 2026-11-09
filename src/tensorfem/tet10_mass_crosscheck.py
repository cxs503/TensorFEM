"""Independent oracles for the TET10 inertia operator.

``tensorfem.quadratic_solid.tet10_consistent_mass`` ships the quadratic-tetrahedron mass matrix, and a mass
matrix is the kind of object where "symmetric, positive and the total mass matches"
proves almost nothing: a wrong node ordering, a wrong Jacobian or an under-integrated
rule all reproduce ``rho V``.  Every check below is therefore rebuilt from a second,
deliberately separate code path:

* :func:`reference_mass` integrates ``N_i N_j`` **exactly** in barycentric
  coordinates (closed-form simplex monomial integrals, no quadrature at all).  It is
  valid for the affine tetrahedron, which is exactly the case where the shipped rule
  is *supposed* to be exact -- so any deviation is a bug, not discretisation.
* :func:`rigid_rotation_energy` compares the mass-weighted kinetic energy of a rigid
  rotation with the closed-form tetrahedron inertia tensor, pinning the *spatial
  distribution* of mass (sum-only checks cannot see a corner/mid-side mixup).
* :func:`free_free_modes` checks that the six rigid-body modes of the assembled
  ``(K, M)`` pair are numerically zero relative to the elastic spectrum.
* :func:`axial_rod_modes` compares full 3-D TET10 rod eigenfrequencies with
  ``omega_n = (2n-1) pi / (2L) * sqrt(E/rho)`` for the consistent and the HRZ-lumped
  mass, which is where the two answer different questions (accurate modes vs. a
  cheap diagonal).
* :func:`rule_order_study` and :func:`keast_mass_error` measure what the shipped
  degree-three five-point Keast rule does to the consistency integral instead of
  repeating "it is under-integrated".
* :func:`lumping_report` measures the row-sum trap (negative vertex masses).

Nothing here is imported by the production solver path.  All numbers quoted in
``docs/tet10-consistent-mass.md`` are produced by :func:`build_evidence` (via
``scripts/audit_tet10_mass.py``), so prose and code cannot drift.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import torch

from .quadratic_solid import (
    tet10_consistent_mass,
    tet10_lumped_mass as _shipped_lumped_mass,
    tet10_mass_quadrature,
    tet10_quadrature,
    tet10_shape,
    tet10_stiffness,
    tet10_volume,
    tet4_to_tet10,
)
from .solid3d import elasticity_matrix_3d, hex_to_tet_mesh, structured_hex_mesh

__all__ = [
    "SHAPE_COEFFS",
    "monomial_integral",
    "reference_mass_scalar",
    "tet10_from_corners",
    "block_total",
    "reference_mass",
    "reference_volume",
    "tetra_inertia",
    "rigid_velocity_energy",
    "rigid_rotation_energy",
    "reference_mass_error",
    "rule_order_study",
    "keast_mass_error",
    "lumping_report",
    "free_free_modes",
    "axial_rod_modes",
    "curved_element_study",
    "build_evidence",
]

# Local ordering of the ten barycentric coordinates ``l0..l3``; matches
# ``quadratic_solid.tet10_shape``: corners 0-3, then edges (0,1),(1,2),(2,0),
# (0,3),(1,3),(2,3).
_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _poly_mul(a, b):
    out = {}
    for ea, ca in a.items():
        for eb, cb in b.items():
            key = tuple(x + y for x, y in zip(ea, eb))
            out[key] = out.get(key, Fraction(0)) + ca * cb
    return out


def _shape_polys():
    """Barycentric coefficients: corner ``i`` is ``2 l_i^2 - l_i``, edge ``(a,b)`` is ``4 l_a l_b``."""
    polys = []
    for i in range(4):
        e2 = [0, 0, 0, 0]
        e2[i] = 2
        e1 = [0, 0, 0, 0]
        e1[i] = 1
        polys.append({tuple(e2): Fraction(2), tuple(e1): Fraction(-1)})
    for a, b in _EDGES:
        e = [0, 0, 0, 0]
        e[a] += 1
        e[b] += 1
        polys.append({tuple(e): Fraction(4)})
    return polys


#: ``{exponent tuple (l0,l1,l2,l3): coefficient}`` for the ten TET10 shapes.
SHAPE_COEFFS = tuple(_shape_polys())


def monomial_integral(exponents) -> float:
    """``int_ref prod l_i^a_i dV`` over the unit tetrahedron (volume 1/6)."""
    exponents = tuple(int(e) for e in exponents)
    if len(exponents) != 4 or any(e < 0 for e in exponents):
        raise ValueError("expect four non-negative barycentric exponents")
    num = 1
    for e in exponents:
        num *= math.factorial(e)
    return 6.0 * num / math.factorial(sum(exponents) + 3)


def reference_mass_scalar() -> np.ndarray:
    """Exact ``int_ref N_i N_j dV`` (``10x10``) on the reference tetrahedron."""
    out = np.zeros((10, 10))
    for i in range(10):
        for j in range(i, 10):
            total = 0.0
            prod = _poly_mul(SHAPE_COEFFS[i], SHAPE_COEFFS[j])
            for exps, coeff in prod.items():
                total += float(coeff) * monomial_integral(exps)
            out[i, j] = out[j, i] = total
    return out


def reference_volume(x) -> np.ndarray:
    """Exact volume of straight-edged tets from the corner-node Jacobian."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None]
    j = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=-1)
    return np.abs(np.linalg.det(j)) / 6.0


def reference_mass(x, density) -> np.ndarray:
    """Exact consistent mass ``[e,30,30]`` for affine (straight) TET10 elements."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[None]
    rho = np.asarray(density, dtype=float)
    rho = np.full(len(x), float(rho)) if rho.ndim == 0 else rho
    scalar = reference_mass_scalar()
    vol = reference_volume(x)
    out = np.zeros((len(x), 30, 30))
    block = np.kron(scalar, np.eye(3))
    for e in range(len(x)):
        out[e] = rho[e] * vol[e] * block
    return out


def tetra_inertia(x, density=1.0) -> np.ndarray:
    """Closed-form inertia tensor of a straight tetrahedron about its centroid.

    ``int x x^T dV = V/20 (sum_a v_a v_a^T + S S^T)`` with ``S = sum_a v_a`` is exact
    because ``x`` is linear in the barycentric coordinates.
    """
    x = np.asarray(x, dtype=float).reshape(4, 3)
    vol = float(reference_volume(x)[0])
    s = x.sum(axis=0)
    a = vol / 20.0 * (x.T @ x + np.outer(s, s))
    c = s / 4.0
    cxx = a - vol * np.outer(c, c)
    return float(density) * (np.trace(cxx) * np.eye(3) - cxx)


def rigid_velocity_energy(block, velocity, nodes: int = 10) -> float:
    """``0.5 * rho int |v|^2 dV`` for a constant velocity field."""
    block = torch.as_tensor(block, dtype=torch.float64)
    v = torch.as_tensor(velocity, dtype=block.dtype).expand(nodes, 3).reshape(-1)
    return float(0.5 * v @ block @ v)


def rigid_rotation_energy(block, x, omega, about=None, nodes: int = 10) -> float:
    """``0.5 * omega^T J omega`` of a rigid rotation about ``about``."""
    block = torch.as_tensor(block, dtype=torch.float64)
    x = torch.as_tensor(x, dtype=block.dtype).reshape(nodes, 3)
    centre = x.new_zeros(3) if about is None else torch.as_tensor(about, dtype=x.dtype)
    w = torch.as_tensor(omega, dtype=block.dtype)
    r = x - centre
    vel = torch.cross(w.expand_as(r), r, dim=1).reshape(-1)
    return float(0.5 * vel @ block @ vel)


def reference_mass_error(x, density) -> dict:
    """Frobenius / max-abs / diagonal error of the shipped mass vs the exact one."""
    x = torch.as_tensor(x, dtype=torch.float64)
    m = tet10_consistent_mass(x, density).numpy()
    ref = reference_mass(x.numpy(), density)
    fro = np.linalg.norm(m - ref) / np.linalg.norm(ref)
    diag_scale = np.max(np.abs(np.diagonal(ref, axis1=-2, axis2=-1)))
    return {
        "frobenius_relative": float(fro),
        "max_abs_relative_to_diagonal": float(np.max(np.abs(m - ref)) / diag_scale),
        "total_mass_relative": float(abs(m.sum(-1).sum(-1).sum() - ref.sum(-1).sum(-1).sum())
                                     / ref.sum(-1).sum(-1).sum()),
        "symmetry": float(np.max(np.abs(m - np.transpose(m, (0, 2, 1)))) / np.linalg.norm(ref)),
    }


def _single_tet(scale=1.0):
    """A deliberately irregular (yet straight-edged) tetrahedron."""
    return torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.05, -0.02],
        [0.15, 0.9, 0.05],
        [-0.05, 0.1, 1.1],
    ], dtype=torch.float64) * scale


def tet10_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """Upgrade one straight TET4 to TET10 with exact mid-edge nodes."""
    nodes, cells = tet4_to_tet10(torch.as_tensor(corners, dtype=torch.float64),
                                 torch.tensor([[0, 1, 2, 3]]))
    return nodes[cells[0]]


def rule_order_study() -> list[dict]:
    """Consistent-mass error vs Duffy order on an affine element (exactness ladder)."""
    x = tet10_from_corners(_single_tet())[None]
    ref = reference_mass(x.numpy(), 1.0)
    rows = []
    for order in (1, 2, 3, 4, 5, 6, 8):
        pts, w = tet10_mass_quadrature(order)
        m = tet10_consistent_mass(x, 1.0, rule=(pts, w)).numpy()
        rows.append({
            "order": order,
            "points": int(len(w)),
            "frobenius_relative": float(np.linalg.norm(m - ref) / np.linalg.norm(ref)),
        })
    return rows


def keast_mass_error() -> dict:
    """What the shipped degree-three five-point rule does to the consistency integral."""
    x = tet10_from_corners(_single_tet())[None]
    ref = reference_mass(x.numpy(), 1.0)
    pts, w = tet10_quadrature()
    m = tet10_consistent_mass(x, 1.0, rule=(pts, w)).numpy()
    diag = np.diagonal(m, axis1=-2, axis2=-1)
    rdiag = np.diagonal(ref, axis1=-2, axis2=-1)
    return {
        "frobenius_relative": float(np.linalg.norm(m - ref) / np.linalg.norm(ref)),
        "diagonal_relative_min": float(np.min(np.abs(diag - rdiag) / np.abs(rdiag))),
        "diagonal_relative_max": float(np.max(np.abs(diag - rdiag) / np.abs(rdiag))),
        "total_mass_relative": float(abs(m.sum() - ref.sum()) / ref.sum()),
        "zero_row_count": int(np.sum(np.all(np.abs(m[0]) < 1e-12, axis=1))),
    }


def lumping_report() -> dict:
    """Row-sum trap and HRZ mass preservation on the irregular element."""
    x = tet10_from_corners(_single_tet())[None]
    consistent = tet10_consistent_mass(x, 1.0)
    total = float(tet10_consistent_mass(x, 1.0).sum())
    row = torch.diagonal(consistent, dim1=-2, dim2=-1).clone()
    # row-sum = int N_i dV: recompute by summing the consistent rows
    row_sum = consistent.sum(-1)[0]
    hrz = _shipped_lumped_mass(x, 1.0)[0]
    if hrz.ndim == 2:
        hrz = torch.diagonal(hrz)
    return {
        "row_sum_vertex_masses": [float(v) for v in row_sum[:4]],
        "row_sum_negative_vertex_count": int((row_sum[:4] < 0).sum()),
        "row_sum_negative_components": int((row_sum < 0).sum()),
        "row_sum_total_relative_error": float(abs(float(row_sum.sum()) - total) / total),
        "hrz_min_entry": float(hrz.min()),
        "hrz_total_relative_error": float(abs(float(hrz.sum()) - total) / total),
        "consistent_diagonal_min": float(row.min()),
    }


def _generalized(k: torch.Tensor, m: torch.Tensor):
    """Generalized symmetric eigenproblem ``K phi = lambda M phi`` via Cholesky."""
    k = 0.5 * (k + k.t())
    m = 0.5 * (m + m.t())
    chol = torch.linalg.cholesky(m)
    reduced = torch.linalg.solve_triangular(chol, k, upper=False)
    reduced = torch.linalg.solve_triangular(chol, reduced.t(), upper=False).t()
    lam, vec = torch.linalg.eigh(0.5 * (reduced + reduced.t()))
    phi = torch.linalg.solve_triangular(chol, vec, upper=False)
    return lam, phi


def free_free_modes() -> dict:
    """Six rigid-body modes of a single-element ``(K, M)`` pair must be zero."""
    x = tet10_from_corners(_single_tet())
    d = elasticity_matrix_3d(torch.tensor([200e9], dtype=torch.float64),
                             torch.tensor([0.3], dtype=torch.float64))
    ke, _ = tet10_stiffness(x[None], d)
    me = tet10_consistent_mass(x[None], 7850.0)
    lam, _ = _generalized(ke[0], me[0])
    lam = torch.clamp(lam, min=0.0)
    elastic = float(lam[-1])
    rigid = [float(v) for v in lam[:6]]
    return {
        "rigid_relative_max": max(rigid) / elastic,
        "smallest_elastic_relative": float(lam[6]) / elastic,
        "rigid_eigenvalues_normalised": [v / elastic for v in rigid],
    }


def _rod_mesh(nex: int, *, length: float = 1.0, width: float = 0.06, ntrans: int = 1):
    nodes, hexes = structured_hex_mesh(length, width, width, nex, ntrans, ntrans)
    _, tets = hex_to_tet_mesh(nodes, hexes)
    return tet4_to_tet10(nodes, tets)


def axial_rod_modes(nex: int, *, modes: int = 3, young: float = 200e9,
                    poisson: float = 0.3, density: float = 7850.0,
                    length: float = 1.0, width: float = 0.06, ntrans: int = 1,
                    lumping: str | None = None, end: str = "clamp") -> dict:
    """Full 3-D fixed-free TET10 rod; axial modes compared with the rod theory.

    Axial modes are identified by the axial mass participation so that flexural and
    torsional modes cannot be mistaken for axial ones.
    """
    nodes, cells = _rod_mesh(nex, length=length, width=width, ntrans=ntrans)
    d = elasticity_matrix_3d(
        nodes.new_full((len(cells),), young), nodes.new_full((len(cells),), poisson)
    )
    ke, _ = tet10_stiffness(nodes[cells], d)
    me = tet10_consistent_mass(nodes[cells], density)
    if lumping is not None:
        lumped = _shipped_lumped_mass(nodes[cells], density)
        if lumped.ndim == 3:                      # shipped returns a 30x30 diagonal
            lumped = torch.diagonal(lumped, dim1=-2, dim2=-1)
        me = torch.diag_embed(lumped)
    ed = torch.stack(tuple(3 * cells + i for i in range(3)), 2).reshape(len(cells), 30).long()
    n_dofs = 3 * len(nodes)
    rows = ed[:, :, None].expand(-1, -1, 30).reshape(-1)
    cols = ed[:, None, :].expand(-1, 30, -1).reshape(-1)
    k = nodes.new_zeros((n_dofs, n_dofs)).index_put((rows, cols), ke.reshape(-1), accumulate=True)
    m = nodes.new_zeros((n_dofs, n_dofs)).index_put((rows, cols), me.reshape(-1), accumulate=True)
    left = torch.where(nodes[:, 0] == 0)[0]
    if end == "clamp":
        fixed = torch.stack((3 * left, 3 * left + 1, 3 * left + 2), 1).reshape(-1)
    elif end == "axial":
        fixed = (3 * left).long()
    else:
        raise ValueError("end must be 'clamp' or 'axial'")
    free = torch.ones(n_dofs, dtype=torch.bool)
    free[fixed] = False
    kr, mr = k[free][:, free], m[free][:, free]
    lam, phi = _generalized(kr, mr)
    lam = torch.clamp(lam, min=0.0)
    omega = lam.sqrt()
    xmask = (torch.arange(int(free.sum())) % 3) == 0
    phi_x = phi.clone()
    phi_x[~xmask] = 0.0
    num = torch.einsum("ij,ik,kj->j", phi_x, mr, phi_x)
    participation = num / torch.einsum("ij,ik,kj->j", phi, mr, phi)
    axial = [i for i in torch.argsort(participation, descending=True).tolist()
             if participation[i] > 0.5]
    axial = sorted(axial, key=lambda i: float(omega[i]))[:modes]
    exact = [(2 * n - 1) * math.pi / (2 * length) * math.sqrt(young / density)
             for n in range(1, modes + 1)]
    measured = [float(omega[i]) for i in axial]
    return {
        "nex": nex,
        "elements": int(len(cells)),
        "dofs": int(n_dofs),
        "lumping": lumping or "consistent",
        "ntrans": ntrans,
        "end": end,
        "frequencies": measured,
        "exact": exact,
        "relative_errors": [abs(m_ - e_) / e_ for m_, e_ in zip(measured, exact)],
    }


def block_total(m: torch.Tensor) -> float:
    """Element mass ``rho V`` read off one displacement component of the matrix.

    ``m`` is ``[e,30,30]``; ``sum_ij M[(i,a),(j,b)] = 3 rho V`` for ``a == b`` and
    ``0`` otherwise, so a single component block (``a = b = 0``) yields ``rho V``.
    """
    blocks = m.reshape(-1, 10, 3, 10, 3)
    return float(blocks[:, :, 0, :, 0].sum())


def _curved_element() -> torch.Tensor:
    """A genuinely curved TET10 (mid-edge nodes lifted by 8 % of the element size)."""
    x = tet10_from_corners(_single_tet())
    x[4] = x[4] + torch.tensor([0.0, 0.0, 0.09])   # edge (0,1)
    x[5] = x[5] + torch.tensor([0.0, 0.0, 0.09])   # edge (1,2)
    x[6] = x[6] + torch.tensor([0.0, 0.0, 0.09])   # edge (2,0)
    return x


def curved_element_study() -> dict:
    """Rule-order convergence of the element mass on a curved isoparametric geometry.

    The element volume is read off the first diagonal 3x3 block: ``sum(M) = 3 rho V``
    because every entry is repeated for the three displacement components.

    There is no closed form here (the integrand is rational), so the reference is the
    order-12 Duffy rule; ``min_det`` documents that the element stays well shaped.
    """
    x = _curved_element()
    reference_rule = tet10_mass_quadrature(12)
    ref = tet10_consistent_mass(x[None], 1.0, rule=reference_rule)
    rows = []
    for order in (2, 3, 4, 6):
        pts, w = tet10_mass_quadrature(order)
        m = tet10_consistent_mass(x[None], 1.0, rule=(pts, w))
        rows.append({
            "order": order,
            "points": int(len(w)),
            "frobenius_relative": float(torch.linalg.norm(m - ref) / torch.linalg.norm(ref)),
            "element_volume_relative": float(abs(block_total(m) - block_total(ref))
                                             / block_total(ref)),
        })
    points, weights = reference_rule
    dets = []
    for p_, w_ in zip(points, weights):
        _, dn = tet10_shape(p_)
        dets.append(float(torch.linalg.det(torch.einsum("eia,ib->eab", x[None], dn))[0]) * float(w_))
    return {
        "rows": rows,
        "element_volume_order12": float(block_total(ref)),
        "independent_det_sum": float(sum(dets)),
        "element_volume_affine_corners": float(reference_volume(x.numpy()[None])[0]),
    }


def build_evidence() -> dict:
    """Every number quoted in ``docs/tet10-consistent-mass.md``."""
    x1 = tet10_from_corners(_single_tet())[None]
    ref = reference_mass(x1.numpy(), 1.0)
    vol = float(reference_volume(x1.numpy())[0])
    consistent = tet10_consistent_mass(x1, 1.0)
    inertia = tetra_inertia(_single_tet().numpy(), density=1.0)
    centre = _single_tet().numpy().mean(axis=0)
    rot_cases = []
    for omega in ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]):
        w = np.asarray(omega, dtype=float)
        exact = 0.5 * float(w @ inertia @ w)
        measured = rigid_rotation_energy(consistent[0], x1[0], omega, about=centre)
        rot_cases.append({"omega": list(omega), "exact": exact, "measured": measured,
                          "relative_error": abs(measured - exact) / exact})
    rod = [axial_rod_modes(n) for n in (2, 4, 8)]
    rod_axial_only = [axial_rod_modes(n, poisson=0.0) for n in (2, 4, 8)]
    rod_lumped = [axial_rod_modes(n, poisson=0.0, lumping="hrz") for n in (2, 4, 8)]
    rod_lumped3d = [axial_rod_modes(n, lumping="hrz") for n in (2, 4, 8)]
    curved = curved_element_study()
    return {
        "element": {
            "volume": vol,
            "reference_mass_total": float(ref.sum(-1).sum(-1).sum()),
            "consistent_mass_total": float(consistent.sum()),
            "consistent_vs_reference": reference_mass_error(x1, 1.0),
            "duffy_weight_sum_error": float(tet10_mass_quadrature()[1].sum() - 1 / 6),
        },
        "rigid_body": {
            "translation": {
                "exact": 0.5 * 7.1 ** 2 + 0.5 * 3.0 ** 2,
                "measured": rigid_velocity_energy(consistent[0], [7.1, 3.0, 0.0]),
            },
            "rotation": rot_cases,
        },
        "free_free": free_free_modes(),
        "rule_orders": rule_order_study(),
        "keast": keast_mass_error(),
        "lumping": lumping_report(),
        "rod_consistent": rod,
        "rod_consistent_nu0": rod_axial_only,
        "rod_hrz": rod_lumped,
        "rod_hrz_nu0": rod_lumped,
        "rod_hrz_nu03": rod_lumped3d,
        "curved": curved,
    }