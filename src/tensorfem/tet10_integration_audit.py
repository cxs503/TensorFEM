"""Independent high-order quadrature oracle for the TET10 solid element.

Closes the verification debt recorded in ``docs/quadratic-solid-rank-audit.md`` §5.4
and ``docs/element-rank-deficiency-audit.md``: the shipped five-point Keast rule is
degree-three exact, but the quadratic-tetrahedron stiffness and consistent-mass
integrands exceed degree three as soon as the isoparametric geometry leaves affine
(distorted mid-edge nodes), so "the element does not lose rank" was never the same
claim as "the element is integrated accurately enough".

This module builds a reference that shares no quadrature weights and no
shape-function code with :mod:`tensorfem.quadratic_solid`:

* collapsed-cube (Duffy) transform ``r = u(1-v)``, ``s = uv(1-w)``, ``t = uvw`` with
  Jacobian ``u**2 * v``, so the unit tetrahedron becomes the unit cube;
* tensor-product Gauss-Legendre in ``u, v, w``, which is exact for polynomial
  integrands of sufficient one-dimensional order and converges fast for the rational
  integrands of a curved isoparametric element.

The oracle is *self-verified* before it is used as a benchmark: it reproduces closed
form monomial integrals :math:`\\int r^a s^b t^c \\, dV` (therefore the reference
volume) and it reproduces the shipped Keast stiffness to round-off on affine
geometry, where the two rules must agree because the integrand is constant.  Only
then is the Keast error measured on curved geometry, where the discrepancy is a real
integration deficiency rather than a bug in the reference.

All tensors are NumPy ``float64`` and the module keeps no global state; nothing here
is imported by the production solver path.
"""
from __future__ import annotations

from math import factorial
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "KEAST_POINTS",
    "KEAST_WEIGHTS",
    "affine_patch_energy",
    "integration_sweep",
    "duffy_rule",
    "duffy_self_verification",
    "elasticity_matrix",
    "element_mass",
    "element_stiffness",
    "mass_matrix_audit",
    "monomial_integral",
    "relative_frobenius",
    "strain_energy_audit",
    "strain_matrix",
    "tet10_geometry",
    "tet10_shape",
]

# --------------------------------------------------------------------------------------
# Shipped rule, restated locally (audit-only copy).
#
# This is the degree-three five-point Keast rule with weights already multiplied by the
# reference-tetrahedron volume, i.e. ``sum(weights) == 1/6``.  A gate in the test file
# asserts it equals ``tensorfem.quadratic_solid.tet10_quadrature`` exactly, so the copy
# cannot silently drift away from the production rule it audits.
# --------------------------------------------------------------------------------------
KEAST_POINTS = np.array(
    [
        (0.25, 0.25, 0.25),
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),  # barycentric (.5, 1/6, 1/6, 1/6) -> r=s=t=1/6
        (0.5, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 0.5, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 0.5),
    ],
    dtype=np.float64,
)
KEAST_WEIGHTS = np.array([-2.0 / 15.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0])


# --------------------------------------------------------------------------------------
# Reference-element shape functions (written independently of the production code).
# --------------------------------------------------------------------------------------
_REF_DERIVATIVES = np.array(
    [
        (-1.0, -1.0, -1.0),  # l0 = 1 - r - s - t
        (1.0, 0.0, 0.0),  # l1 = r
        (0.0, 1.0, 0.0),  # l2 = s
        (0.0, 0.0, 1.0),  # l3 = t
    ],
    dtype=np.float64,
)

#: Mid-edge node order used by :class:`tensorfem.quadratic_solid.Tet10Model`.
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def tet10_shape(rst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions and natural derivatives at points ``[..., 3]``.

    Returns ``(N, dN)`` with shapes ``[q, 10]`` and ``[q, 10, 3]``.  Vertex functions
    are the quadratic Lagrange polynomials in the barycentric coordinates and the six
    edge functions are ``4 * l_i * l_j``, matching the documented node ordering
    ``(0,1,2,3, 01,12,20,03,13,23)``.
    """
    rst = np.atleast_2d(np.asarray(rst, dtype=np.float64))
    r, s, t = rst[:, 0], rst[:, 1], rst[:, 2]
    l = np.stack((1.0 - r - s - t, r, s, t), axis=-1)  # [q,4]

    n_vertex = l * (2.0 * l - 1.0)
    n_edge = np.stack([4.0 * l[:, i] * l[:, j] for i, j in EDGES], axis=-1)
    n = np.concatenate((n_vertex, n_edge), axis=-1)

    dn_vertex = (4.0 * l - 1.0)[:, :, None] * _REF_DERIVATIVES[None, :, :]
    dn_edge = np.stack(
        [4.0 * (l[:, i, None] * _REF_DERIVATIVES[j] + l[:, j, None] * _REF_DERIVATIVES[i])
         for i, j in EDGES],
        axis=-2,
    )
    return n, np.concatenate((dn_vertex, dn_edge), axis=-2)


# --------------------------------------------------------------------------------------
# Duffy (collapsed-cube) reference rule.
# --------------------------------------------------------------------------------------
def duffy_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Legendre rule on the collapsed cube ``[0,1]^3``.

    ``order`` points per direction, ``order**3`` points in total.  Weights already
    include the collapse Jacobian ``u**2 v`` but *not* the element Jacobian, so
    ``weights.sum() == 1/6`` (the reference-tetrahedron volume).
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    u, w = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (u + 1.0)
    w = 0.5 * w
    uu, vv, ww = np.meshgrid(u, u, u, indexing="ij")
    wu, wv, ww_ = np.meshgrid(w, w, w, indexing="ij")
    r = uu * (1.0 - vv)
    s = uu * vv * (1.0 - ww)
    t = uu * vv * ww
    points = np.stack((r.ravel(), s.ravel(), t.ravel()), axis=-1)
    weights = (wu * wv * ww_ * uu**2 * vv).ravel()
    return points, weights


def monomial_integral(exponents: Sequence[int]) -> float:
    """Closed form ``int_tet r^a s^b t^c dV`` over ``{r,s,t >= 0, r+s+t <= 1}``."""
    a, b, c = exponents
    return factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)


def duffy_self_verification(*, max_degree: int = 6,
                            orders: Iterable[int] = (2, 3, 4, 6, 8, 12, 16)) -> list[dict]:
    """Check the oracle against closed forms before it is trusted as a benchmark.

    For each order the largest total degree whose monomials are all integrated to
    ``1e-14`` relative accuracy is reported, plus the measured reference volume error.
    A correct implementation must reach machine precision from order >= 4 and keep it
    as the order grows; a collapse-Jacobian slip shows up immediately as a constant
    bias.
    """
    records: list[dict] = []
    for order in orders:
        points, weights = duffy_rule(order)
        measure = monomial_integral((0, 0, 0))
        volume_error = abs(weights.sum() - measure) / measure
        exact_through = -1
        worst = 0.0
        for degree in range(max_degree + 1):
            ok = True
            for a in range(degree + 1):
                for b in range(degree - a + 1):
                    c = degree - a - b
                    reference = monomial_integral((a, b, c))
                    measured = float(np.sum(weights * points[:, 0] ** a * points[:, 1] ** b
                                            * points[:, 2] ** c))
                    err = abs(measured - reference) / max(abs(reference), measure)
                    worst = max(worst, err)
                    if err > 1e-14:
                        ok = False
            if not ok:
                break
            exact_through = degree
        records.append(
            {
                "order": order,
                "points": int(points.shape[0]),
                "volume_relative_error": float(volume_error),
                "exact_through_total_degree": int(exact_through),
                "worst_monomial_relative_error": float(worst),
            }
        )
    return records


# --------------------------------------------------------------------------------------
# Element integrations.
# --------------------------------------------------------------------------------------
def elasticity_matrix(young: float, poisson: float) -> np.ndarray:
    """Isotropic 3-D elasticity matrix, strain ordering ``(xx, yy, zz, xy, yz, zx)``."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    d = np.zeros((6, 6), dtype=np.float64)
    d[:3, :3] = lam * np.ones((3, 3)) + np.diag(2.0 * mu * np.ones(3))
    d[3, 3] = d[4, 4] = d[5, 5] = mu
    return d


def _point_b(nodes: np.ndarray, dn: np.ndarray) -> tuple[np.ndarray, float]:
    """Strain matrix and Jacobian determinant at one integration point."""
    jac = nodes.T @ dn  # [3,3]: J[a,b] = d x_a / d xi_b
    det = float(np.linalg.det(jac))
    grad = dn @ np.linalg.inv(jac)  # [10,3]: dN_i / dx_a
    b = np.zeros((6, 30), dtype=np.float64)
    b[0, 0::3] = grad[:, 0]
    b[1, 1::3] = grad[:, 1]
    b[2, 2::3] = grad[:, 2]
    b[3, 0::3] = grad[:, 1]
    b[3, 1::3] = grad[:, 0]
    b[4, 1::3] = grad[:, 2]
    b[4, 2::3] = grad[:, 1]
    b[5, 0::3] = grad[:, 2]
    b[5, 2::3] = grad[:, 0]
    return b, det


def strain_matrix(nodes: np.ndarray, rst: np.ndarray) -> np.ndarray:
    """Strain matrix at natural points ``rst`` for a single element (``[q,6,30]``)."""
    _, dn = tet10_shape(rst)
    return np.stack([_point_b(nodes, dn[q])[0] for q in range(len(dn))], axis=0)


def _require_positive_jacobian(nodes: np.ndarray, points: np.ndarray) -> None:
    _, dn = tet10_shape(points)
    dets = np.array([np.linalg.det(nodes.T @ dn[q]) for q in range(len(dn))])
    if dets.min() <= 0.0:
        raise ValueError("TET10 geometry has a non-positive Jacobian at an integration point")


def element_stiffness(nodes: np.ndarray, d: np.ndarray, points: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    """``B^T D B detJ`` integrated with the supplied rule for one element."""
    nodes = np.asarray(nodes, dtype=np.float64)
    _require_positive_jacobian(nodes, points)
    _, dn = tet10_shape(points)
    ke = np.zeros((30, 30), dtype=np.float64)
    for q in range(len(weights)):
        b, det = _point_b(nodes, dn[q])
        ke += weights[q] * det * (b.T @ d @ b)
    return ke


def element_mass(nodes: np.ndarray, rho: float, points: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """Consistent mass matrix ``rho * N_i N_j detJ`` for one element."""
    nodes = np.asarray(nodes, dtype=np.float64)
    _require_positive_jacobian(nodes, points)
    n, dn = tet10_shape(points)
    dets = np.array([np.linalg.det(nodes.T @ dn[q]) for q in range(len(dn))])
    return (n * (weights * dets)[:, None]).T @ n * rho


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """``||a - b||_F / ||b||_F``."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------------------
# Geometries.
# --------------------------------------------------------------------------------------
_REGULAR = np.array(
    [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], dtype=np.float64
)
_DISTORTED = np.array(
    [(0.0, 0.0, 0.0), (1.20, 0.10, -0.05), (0.10, 1.10, 0.05), (0.05, -0.10, 1.15)],
    dtype=np.float64,
)


#: Alternating mid-edge sign pattern and direction, copied in spirit from the HEX20
#: curvature audit so the two quadratic solid families are measured on the same scale.
_EDGE_SIGNS = (1.0, -1.0, 0.5, -0.5, 0.7, -0.7)
_WARP_DIRECTION = np.array([0.08, -0.06, 0.05], dtype=np.float64)
_WARP_DIRECTION /= np.linalg.norm(_WARP_DIRECTION)


def tet10_geometry(kind: str = "regular", curvature: float = 0.0,
                   vertices: np.ndarray | None = None) -> np.ndarray:
    """Ten node coordinates of one element.

    ``kind`` selects the vertex set and the mid-edge treatment:

    * ``"regular"`` - unit right tetrahedron, mid-edge nodes at the exact midpoints;
    * ``"distorted"`` - a well-shaped but non-affine-friendly vertex set, straight edges;
    * ``"curved"`` - the distorted vertices plus alternating mid-edge offsets of
      ``curvature * mean_edge_length`` along a fixed unit direction (the same
      alternating-sign convention as the HEX20 curvature audit, so ``curvature`` is a
      fraction of the element edge length in both families);
    * ``"warped"`` - the distorted vertices plus pseudo-random mid-edge offsets of
      ``curvature * mean_edge_length`` (fixed seed), used to check that the conclusions
      do not depend on one hand-picked perturbation pattern.

    The vertex set is identical for every ``curvature`` value of a given ``kind``, so
    rows of a sweep differ only by the mid-edge displacement.
    """
    if vertices is None:
        if kind in ("regular",):
            vertices = _REGULAR
        elif kind in ("distorted", "curved", "warped"):
            vertices = _DISTORTED
        else:
            raise ValueError(f"unknown TET10 geometry kind {kind!r}")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (4, 3):
        raise ValueError("vertices must have shape [4,3]")

    edges = [(vertices[j] - vertices[i]) for i, j in EDGES]
    mean_edge = float(np.mean([np.linalg.norm(e) for e in edges]))
    mids = []
    for index, (i, j) in enumerate(EDGES):
        mid = 0.5 * (vertices[i] + vertices[j])
        if curvature:
            if kind == "curved":
                offset = _EDGE_SIGNS[index] * _WARP_DIRECTION
            elif kind == "warped":
                rng = np.random.default_rng(1000 + index)
                offset = rng.normal(size=3)
                offset /= np.linalg.norm(offset)
            else:
                offset = np.zeros(3)
            mid = mid + curvature * mean_edge * offset
        mids.append(mid)
    return np.vstack((vertices, np.array(mids, dtype=np.float64)))


# --------------------------------------------------------------------------------------
# Measurements.
# --------------------------------------------------------------------------------------
def integration_sweep(*, kind: str = "curved",
                      curvatures: Iterable[float] = (0.0, 0.02, 0.05, 0.08, 0.11, 0.15),
                      reference_order: int = 24, young: float = 200.0e9,
                      poisson: float = 0.3, rho: float = 7850.0) -> list[dict]:
    """Keast error versus mid-edge curvature, against the high-order Duffy reference.

    ``reference_order=24`` means ``24**3 = 13824`` reference points per element; the
    half-order value is recorded alongside so the reference is auditable rather than
    assumed, and the straight-edge production stiffness is compared to it in the test
    suite.
    """
    d = elasticity_matrix(young, poisson)
    ref_points, ref_weights = duffy_rule(reference_order)
    half_points, half_weights = duffy_rule(max(8, reference_order // 2))
    records: list[dict] = []
    for alpha in curvatures:
        nodes = tet10_geometry(kind if kind != "curved" or alpha else "distorted",
                               curvature=alpha)
        keast = element_stiffness(nodes, d, KEAST_POINTS, KEAST_WEIGHTS)
        reference = element_stiffness(nodes, d, ref_points, ref_weights)
        half = element_stiffness(nodes, d, half_points, half_weights)
        keast_mass = element_mass(nodes, rho, KEAST_POINTS, KEAST_WEIGHTS)
        reference_mass = element_mass(nodes, rho, ref_points, ref_weights)
        records.append(
            {
                "kind": kind,
                "curvature": float(alpha),
                "stiffness_keast_vs_reference": relative_frobenius(keast, reference),
                "mass_keast_vs_reference": relative_frobenius(keast_mass, reference_mass),
                "reference_half_vs_full": relative_frobenius(half, reference),
                "affine_patch_energy_error": affine_patch_energy(nodes, d)["relative_error"],
                "det_jacobian_min": _min_det(nodes, ref_points),
                "volume_ratio_to_straight": (
                    float(np.sum(ref_weights * _dets_at(nodes, ref_points)))
                    / float(np.sum(ref_weights * _dets_at(
                        tet10_geometry("distorted"), ref_points)))
                ),
            }
        )
    return records


def _min_det(nodes: np.ndarray, points: np.ndarray) -> float:
    _, dn = tet10_shape(points)
    return float(min(np.linalg.det(nodes.T @ dn[q]) for q in range(len(dn))))


def affine_patch_energy(nodes: np.ndarray, d: np.ndarray,
                        strain: Sequence[float] = (1e-3, -2e-4, 5e-4, 3e-4, 0.0, 1e-4),
                        *, reference_order: int = 24) -> dict:
    """Strain energy of an exactly representable affine field, Keast vs reference.

    The quadratic interpolation reproduces a linear displacement field exactly, so the
    strain is the prescribed constant vector and the energy integrand reduces to
    ``0.5 * eps:D:eps * detJ``.  The comparison therefore isolates how well each rule
    integrates the *volume distortion* ``detJ`` of a curved isoparametric element,
    separately from the ``B``-matrix variation measured by the stiffness Frobenius test.
    """
    eps = np.asarray(strain, dtype=np.float64)
    energy_density = 0.5 * float(eps @ d @ eps)
    ref_points, ref_weights = duffy_rule(reference_order)
    keast_volume = float(np.sum(KEAST_WEIGHTS * _dets_at(nodes, KEAST_POINTS)))
    reference_volume = float(np.sum(ref_weights * _dets_at(nodes, ref_points)))
    keast_energy = energy_density * keast_volume
    reference_energy = energy_density * reference_volume
    return {
        "keast_energy": keast_energy,
        "reference_energy": reference_energy,
        "relative_error": abs(keast_energy - reference_energy) / abs(reference_energy),
        "keast_volume": keast_volume,
        "reference_volume": reference_volume,
    }


def _dets_at(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    n, dn = tet10_shape(points)
    del n
    return np.array([np.linalg.det(nodes.T @ dn[q]) for q in range(len(dn))])


def strain_energy_audit(*, kind: str = "curved",
                        curvatures: Iterable[float] = (0.0, 0.05, 0.08, 0.11, 0.15),
                        reference_order: int = 24, young: float = 200.0e9,
                        poisson: float = 0.3) -> list[dict]:
    """Strain energy of two prescribed nodal fields, Keast vs reference.

    The stiffness-matrix Frobenius error is an abstract norm; energy error is what an
    energy-based acceptance check actually sees.  Two fields are used: an affine
    displacement field (constant strain, exactly representable by the quadratic
    interpolation) and a bending field ``u_x = -y z, u_y = x z, u_z = 0`` style warp
    that produces a genuinely varying strain inside the element.  The affine row is a
    sanity control: its energy must be integrated to round-off because it reduces to
    ``detJ`` integration, which isolates the ``B``-variation as the error source.
    """
    d = elasticity_matrix(young, poisson)
    ref_points, ref_weights = duffy_rule(reference_order)
    records: list[dict] = []
    for alpha in curvatures:
        nodes = tet10_geometry(kind if alpha else "distorted", curvature=alpha)
        x = nodes
        affine = (0.7 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2]).tolist()
        affine = np.zeros(30)
        affine[0::3] = 1e-3 * x[:, 0] - 2e-4 * x[:, 1] + 5e-4 * x[:, 2]
        affine[1::3] = 3e-4 * x[:, 0] + 1e-3 * x[:, 1] - 1e-4 * x[:, 2]
        affine[2::3] = -2e-4 * x[:, 0] + 4e-4 * x[:, 1] + 8e-4 * x[:, 2]
        bending = np.zeros(30)
        bending[0::3] = -1e-3 * x[:, 1] * x[:, 2]
        bending[1::3] = 1e-3 * x[:, 0] * x[:, 2]
        bending[2::3] = 2e-3 * x[:, 0] * x[:, 1]
        row = {"kind": kind, "curvature": float(alpha)}
        for label, field in (("affine", affine), ("bending", bending)):
            k_keast = element_stiffness(nodes, d, KEAST_POINTS, KEAST_WEIGHTS)
            k_reference = element_stiffness(nodes, d, ref_points, ref_weights)
            e_keast = 0.5 * float(field @ k_keast @ field)
            e_reference = 0.5 * float(field @ k_reference @ field)
            row[f"{label}_energy_keast"] = e_keast
            row[f"{label}_energy_reference"] = e_reference
            row[f"{label}_energy_relative_error"] = abs(e_keast - e_reference) / abs(e_reference)
        records.append(row)
    return records


def mass_matrix_audit(*, curvatures: Iterable[float] = (0.0, 0.05, 0.11),
                      reference_order: int = 24, rho: float = 7850.0,
                      kind: str = "curved") -> list[dict]:
    """Consistent-mass error of the shipped rule (production has no TET10 mass yet).

    Keast is degree-three exact while ``N_i N_j detJ`` is quartic on a straight element,
    so a non-zero error is expected even for affine geometry.  The number is reported
    as a forward-looking constraint for whoever implements ``tet10_mass``: the shipped
    rule cannot be reused unchanged there.
    """
    ref_points, ref_weights = duffy_rule(reference_order)
    records: list[dict] = []
    for alpha in curvatures:
        nodes = tet10_geometry(kind if alpha else "distorted", curvature=alpha)
        keast = element_mass(nodes, rho, KEAST_POINTS, KEAST_WEIGHTS)
        reference = element_mass(nodes, rho, ref_points, ref_weights)
        total_keast = float(keast.sum())
        total_reference = float(reference.sum())
        records.append(
            {
                "curvature": float(alpha),
                "relative_frobenius_error": relative_frobenius(keast, reference),
                "diagonal_relative_error_max": float(
                    np.max(np.abs(np.diag(keast) - np.diag(reference)) / np.diag(reference))
                ),
                "total_mass_relative_error": abs(total_keast - total_reference) / total_reference,
            }
        )
    return records