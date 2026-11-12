"""Fictitious-crack (cohesive-crack) model driven by FE influence functions.

Reproduction of the method of

    A. Hillerborg, M. Modeer and P.-E. Petersson, "Analysis of crack formation
    and crack growth in concrete by means of fracture mechanics and finite
    elements", *Cement and Concrete Research* **6** (1976) 773--781.

The paper's contribution is the *fictitious crack model*: the process zone of a
concrete crack is smeared into a cohesive traction--separation relation
``sigma = f(w)`` acting on the crack faces, with the fracture energy
``G_F = int_0^inf f(w) dw`` as the governing material parameter, while the bulk
stays linear elastic.  The numerical scheme of the paper is an
**influence-function (flexibility) method**: the crack opening is a superposition
of the opening produced by the external load and the openings produced by the
cohesive tractions themselves,

    ``w(x) = w^P(x) - int_{a0}^{a} C(x, xi) sigma(xi) d xi``

with ``C`` the influence (compliance) coefficients of the linear-elastic body,
obtained from a finite-element solution.  Only the cohesive tractions are
unknowns; the mesh and the solid stiffness matrix never change.

This module re-derives that method for the symmetric mode-I problem, so that the
nonlinearity lives in a small dense system (tens of unknowns) instead of a
global tangent-stiffness iteration.  For a symmetric three-point-bend specimen
the mode-I crack plane is the symmetry plane, so a **half model is exact**: the
symmetry plane is cut, the normal traction there is the cohesive traction, and
``u_x = 0`` holds on the ligament above the admissible crack path.  Symmetry also
makes the shear traction on that plane vanish identically, so the interface
carries only the normal (mode-I) law -- no zero-thickness element, no duplicated
node line, and no loss of physics.

Why this is not a duplicate of :mod:`tensorfem.cohesive_bending`: that module is
a global Newton solver on the full Q4 mesh with a pre-meshed cohesive element
line spanning the *whole* depth of an *un-notched* beam.  Here (i) the mesh
carries a real notch of any depth ``a0/W`` (so an ``a/W`` series costs nothing),
(ii) the nonlinear system is solved by crack-mouth-opening (CMOD) control on a
small dense flexibility system which stays non-singular through the load peak --
no stalling in the ductile (large process zone) regime -- and (iii) the energy
balance is closed separately against the solid FE energy, the interface secant
energy and the constitutive dissipation.

Independent reference quantities used for verification (nothing is compared
against this model's own output):

* handbook stress-intensity factor of a single-edge notched bend specimen
  (Srawley & Gross 1967; Tada, *The Stress Analysis of Cracks Handbook*, case
  2.16; ASTM E399), used for the linear-elastic (LEFM) asymptote;
* the closed-form area of the softening law, ``int f dw = G_F``;
* Maxwell--Betti reciprocity of the influence matrix;
* the ligament strength criterion for the small-specimen asymptote;
* collapse of the size effect in ``W / l_ch`` with Hillerborg's characteristic
  length ``l_ch = E G_F / f_t^2``.

All quantities are SI (m, Pa, J/m^2, N, J).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "ExponentialSoftening",
    "BilinearSoftening",
    "SENBGeometry",
    "InfluenceModel",
    "build_influence_model",
    "BendPath",
    "solve_cmod_control",
    "senb_handbook_shape_factor",
    "senb_handbook_stress_intensity",
    "senb_lefm_peak_load",
    "strength_limit_load",
    "elastic_stress_intensity_from_compliance",
    "SizeEffectPoint",
    "run_size_effect",
]

# 2x2 Gauss rule for the Q4 continuum elements.
_H = 1.0 / math.sqrt(3.0)
_Q4_GP = ((-_H, -_H), (_H, -_H), (_H, _H), (-_H, _H))

#: traction below this fraction of f_t counts as "fully separated".
_SEPARATED_FRACTION = 1e-6


# ---------------------------------------------------------------------------
# softening laws
# ---------------------------------------------------------------------------
class _SofteningLaw:
    """Common protocol: a monotone traction--separation envelope.

    ``initial_stiffness`` is the penalty stiffness of the *intact* interface (it
    stands in for "not yet cracked" material).  The envelope is
    ``sigma = k0 * w`` for ``w <= w_e`` and a monotone softening branch beyond
    it, with the total area under the envelope exactly ``fracture_energy``.
    """

    kind = "?"

    def traction(self, w):
        raise NotImplementedError

    def slope(self, w):
        """``d sigma / d w`` (positive on the elastic branch, negative in softening)."""
        raise NotImplementedError

    def opening(self, sigma):
        """Inverse of :meth:`traction` (single valued: the envelope is monotone)."""
        raise NotImplementedError

    def dissipated(self, w):
        """``int_0^w sigma dw`` -- fracture work per unit area."""
        raise NotImplementedError

    def total_area(self, n=400001):
        """Exact-area check by quadrature of the envelope up to full separation."""
        w = np.linspace(0.0, self.separation_opening, n)
        return float(np.trapezoid(self.traction(w), w))


@dataclass(frozen=True)
class ExponentialSoftening(_SofteningLaw):
    """Hillerborg's exponential softening, ``sigma = f_t exp(-(w-w_e)/w0)``.

    ``w0 = (G_F - f_t^2/(2 k0)) / f_t`` so that the area under the envelope is
    *exactly* ``G_F`` (elastic triangle ``f_t^2/(2 k0)`` plus tail ``f_t w0``).
    """

    tensile_strength: float
    fracture_energy: float
    initial_stiffness: float
    kind: str = field(default="exponential", init=False)

    def __post_init__(self):
        for name in ("tensile_strength", "fracture_energy", "initial_stiffness"):
            value = float(getattr(self, name))
            if not (value > 0.0):
                raise ValueError(f"{name} must be positive, got {value}")
        if self.fracture_energy <= 0.5 * self.tensile_strength**2 / self.initial_stiffness:
            raise ValueError("initial_stiffness too low: elastic area exceeds G_F")

    @property
    def elastic_opening(self) -> float:
        return self.tensile_strength / self.initial_stiffness

    @property
    def w0(self) -> float:
        soft = self.fracture_energy - 0.5 * self.tensile_strength**2 / self.initial_stiffness
        return soft / self.tensile_strength

    @property
    def separation_opening(self) -> float:
        return self.elastic_opening + self.w0 * math.log(1.0 / _SEPARATED_FRACTION)

    def traction(self, w):
        w = np.asarray(w, dtype=float)
        we = self.elastic_opening
        soft = self.tensile_strength * np.exp(-np.clip((w - we) / self.w0, 0.0, 700.0))
        return np.where(w <= we, self.initial_stiffness * w, soft)

    def slope(self, w):
        w = np.asarray(w, dtype=float)
        we = self.elastic_opening
        soft = -(self.tensile_strength / self.w0) * np.exp(
            -np.clip((w - we) / self.w0, 0.0, 700.0)
        )
        return np.where(w <= we, self.initial_stiffness, soft)

    def opening(self, sigma):
        sigma = np.asarray(sigma, dtype=float)
        ft = self.tensile_strength
        soft = self.elastic_opening + self.w0 * np.log(ft / np.clip(sigma, 1e-300, None))
        return np.where(sigma >= ft, sigma / self.initial_stiffness, soft)

    def dissipated(self, w):
        w = np.asarray(w, dtype=float)
        we = self.elastic_opening
        soft = 0.5 * self.tensile_strength * we + self.tensile_strength * self.w0 * (
            1.0 - np.exp(-np.clip((w - we) / self.w0, 0.0, 700.0))
        )
        return np.where(w <= we, 0.5 * self.initial_stiffness * w**2, soft)


@dataclass(frozen=True)
class BilinearSoftening(_SofteningLaw):
    """Bilinear softening (the piecewise-linear envelope common in FE codes).

    Supplied so the reproduction can quantify the *shape* sensitivity of the
    cohesive law at fixed ``f_t`` and ``G_F`` (the 1976 paper measured its
    ``sigma-w`` curve, i.e. the shape is a measured input, not a free choice).
    """

    tensile_strength: float
    fracture_energy: float
    initial_stiffness: float
    kind: str = field(default="bilinear", init=False)

    def __post_init__(self):
        for name in ("tensile_strength", "fracture_energy", "initial_stiffness"):
            value = float(getattr(self, name))
            if not (value > 0.0):
                raise ValueError(f"{name} must be positive, got {value}")
        if self.fracture_energy <= 0.5 * self.tensile_strength**2 / self.initial_stiffness:
            raise ValueError("initial_stiffness too low: elastic area exceeds G_F")

    @property
    def elastic_opening(self) -> float:
        return self.tensile_strength / self.initial_stiffness

    @property
    def failure_opening(self) -> float:
        soft = self.fracture_energy - 0.5 * self.tensile_strength**2 / self.initial_stiffness
        return self.elastic_opening + 2.0 * soft / self.tensile_strength

    @property
    def separation_opening(self) -> float:
        return self.failure_opening

    def traction(self, w):
        w = np.asarray(w, dtype=float)
        we, wf = self.elastic_opening, self.failure_opening
        soft = self.tensile_strength * (wf - np.clip(w, None, wf)) / (wf - we)
        return np.where(w <= we, self.initial_stiffness * w, np.where(w < wf, soft, 0.0))

    def slope(self, w):
        w = np.asarray(w, dtype=float)
        we, wf = self.elastic_opening, self.failure_opening
        soft = np.full_like(w, -self.tensile_strength / (wf - we))
        return np.where(w <= we, self.initial_stiffness, np.where(w < wf, soft, 0.0))

    def opening(self, sigma):
        sigma = np.asarray(sigma, dtype=float)
        we, wf = self.elastic_opening, self.failure_opening
        soft = we + (wf - we) * (1.0 - sigma / self.tensile_strength)
        return np.where(sigma >= self.tensile_strength, sigma / self.initial_stiffness, soft)

    def dissipated(self, w):
        w = np.asarray(w, dtype=float)
        we, wf = self.elastic_opening, self.failure_opening
        dw = np.clip(w - we, 0.0, wf - we)
        soft = 0.5 * self.tensile_strength * we + self.tensile_strength * (
            dw - 0.5 * dw**2 / (wf - we)
        )
        return np.where(w <= we, 0.5 * self.initial_stiffness * w**2,
                        np.where(w < wf, soft, self.fracture_energy))


# ---------------------------------------------------------------------------
# geometry and handbook (LEFM) references
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SENBGeometry:
    """Single-edge notched bend specimen (three-point bending, span ``S``).

    ``depth`` is ``W``, ``thickness`` is ``B``, ``span`` is ``S`` (support
    distance), ``notch`` is the initial notch depth ``a0`` and ``cut_fraction``
    is where the admissible crack path ends; the ligament above it stays intact,
    which keeps the half model connected.  The notch depth actually used is
    snapped to the nearest symmetry-plane node and reported by
    :class:`InfluenceModel`.
    """

    depth: float
    span: float
    thickness: float
    notch: float
    cut_fraction: float = 0.9
    young: float = 30.0e9
    poisson: float = 0.2

    def __post_init__(self):
        if not (0.0 < self.notch < self.depth):
            raise ValueError("notch must lie in (0, W)")
        if not (self.notch < self.cut_fraction * self.depth < self.depth):
            raise ValueError("need a0 < cut_fraction * W < W")

    @property
    def cut(self) -> float:
        return self.cut_fraction * self.depth

    @property
    def alpha(self) -> float:
        """Nominal notch depth ratio ``a0/W`` (before node snapping)."""
        return self.notch / self.depth

    @property
    def modified_modulus(self) -> float:
        """Plane-stress modulus ``E' = E`` (the Q4 model is plane stress)."""
        return self.young


def senb_handbook_shape_factor(alpha: float) -> float:
    """Srawley--Gross shape factor of a three-point-bend SENB specimen, ``S = 4W``.

    ``K_I = (P S)/(B W^1.5) * f(alpha)`` with

        ``f(alpha) = 3 sqrt(alpha) [1.99 - alpha(1-alpha)(2.15 - 3.93 alpha
        + 2.7 alpha^2)] / [2 (1 + 2 alpha) (1 - alpha)^1.5]``

    (Srawley & Gross 1967; Tada, *Stress Analysis of Cracks Handbook*, case 2.16;
    ASTM E399 SENB).  Tabulated for ``0.05 < alpha < 0.6``; the ``alpha -> 0``
    limit reproduces the edge-crack asymptote ``K = 1.122 sigma_b sqrt(pi a)``.
    """
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    num = 3.0 * math.sqrt(a) * (1.99 - a * (1.0 - a) * (2.15 - 3.93 * a + 2.7 * a * a))
    den = 2.0 * (1.0 + 2.0 * a) * (1.0 - a) ** 1.5
    return num / den


def senb_handbook_stress_intensity(load, geom: SENBGeometry, notch=None) -> float:
    """Handbook ``K_I`` of a point-loaded SENB specimen (``S = 4W`` assumed)."""
    a = geom.notch if notch is None else notch
    if abs(geom.span - 4.0 * geom.depth) > 1e-9 * geom.depth:
        raise ValueError("handbook shape factor is tabulated for S = 4W only")
    return (
        load
        * geom.span
        / (geom.thickness * geom.depth**1.5)
        * senb_handbook_shape_factor(a / geom.depth)
    )


def senb_lefm_peak_load(geom: SENBGeometry, fracture_energy, notch=None) -> float:
    """LEFM peak load: ``K_I(P_max) = K_Ic`` with ``K_Ic = sqrt(E' G_F)`` (plane stress)."""
    k_ic = math.sqrt(geom.modified_modulus * fracture_energy)
    a = geom.notch if notch is None else notch
    shape = senb_handbook_shape_factor(a / geom.depth)
    return k_ic * geom.thickness * geom.depth**1.5 / (geom.span * shape)


def strength_limit_load(geom: SENBGeometry, tensile_strength, notch=None) -> float:
    """Net-section modulus-of-rupture load: the small-specimen (strength) limit.

    A cohesive bend specimen whose process zone spans the whole ligament fails
    when the *net-section* bending stress reaches ``f_t``, i.e. when
    ``M = f_t B (W - a0)^2 / 6`` (modulus of rupture of the ligament).  With the
    three-point-bend moment ``M = P S / 4``,

        ``P_str = (2/3) f_t B (W - a0)^2 / S``,

    which is the classical flexural limit ``(2/3) B W^2 f_t / S`` for ``a0 -> 0``.
    """
    a = geom.notch if notch is None else notch
    return 2.0 / 3.0 * tensile_strength * geom.thickness * (geom.depth - a) ** 2 / geom.span


# ---------------------------------------------------------------------------
# influence model (half model of the symmetric three-point bend)
# ---------------------------------------------------------------------------
@dataclass
class InfluenceModel:
    """Linear-elastic influence data of the cracked half body."""

    geometry: SENBGeometry
    law: _SofteningLaw
    nodes: np.ndarray            # (n_node, 2) coordinates of the half body
    elements: np.ndarray         # (n_elem, 4) connectivity
    left_nodes: np.ndarray       # node ids on the symmetry plane, bottom to top
    left_y: np.ndarray           # their y coordinates
    active_mask: np.ndarray      # cohesive tractions allowed (a0 < y < cut)
    area: np.ndarray             # (n_left,) tributary fracture area per node
    C: np.ndarray                # (n_left, n_left) opening per unit traction [m/Pa]
    a_load: np.ndarray           # (n_left,) opening per unit total load [m/N]
    d_load: float                # platen deflection per unit total load [m/N]
    d_traction: np.ndarray       # (n_left,) platen deflection per unit traction
    dof_left_x: np.ndarray
    platen_dofs: np.ndarray
    platen_weights: np.ndarray
    n_dof: int
    effective_notch: float
    effective_cut: float
    n_nodes_y: int
    _free: np.ndarray = None
    _lu: object = None
    _K: sp.csr_matrix = None
    f_unit: np.ndarray = None

    # -- derived ------------------------------------------------------------
    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum())

    @property
    def active_y(self) -> np.ndarray:
        return self.left_y[self.active_mask]

    @property
    def characteristic_length(self) -> float:
        law = self.law
        return self.geometry.modified_modulus * law.fracture_energy / law.tensile_strength**2

    def force_compliance(self) -> np.ndarray:
        """Opening per unit *force* pair, ``S_ij = C_ij / A_j`` (symmetric by Maxwell--Betti)."""
        safe = np.where(self.area > 0.0, self.area, 1.0)
        return self.C / safe[None, :]

    def reciprocity_error(self) -> float:
        """``max|S - S^T| / max|S|`` with ``S`` the opening-per-unit-force matrix.

        Maxwell--Betti reciprocity holds exactly for the linear-elastic half body;
        the raw ``C`` is only symmetric when all tributary areas are equal, so the
        column scaling by ``A_j`` is the physically meaningful form.
        """
        s = self.force_compliance()
        scale = float(np.max(np.abs(s)))
        return float(np.max(np.abs(s - s.T)) / scale) if scale > 0 else 0.0

    def mouth_index(self) -> int:
        """Index (in ``left_nodes``) of the crack-mouth node, ``y = 0``."""
        return 0

    # -- linear solves ------------------------------------------------------
    def platen_displacement(self, load: float, tractions: np.ndarray) -> float:
        """Platen (mid-span) displacement in ``y`` (negative = downward)."""
        sigma = self._full_traction_vector(tractions)
        return float(load * self.d_load + np.dot(sigma, self.d_traction))

    def platen_deflection(self, load: float, tractions: np.ndarray) -> float:
        """Platen displacement magnitude, positive in the direction of the load."""
        return -self.platen_displacement(load, tractions)

    def solid_displacement(self, load: float, tractions: np.ndarray) -> np.ndarray:
        f = load * self.f_unit
        sigma = self._full_traction_vector(tractions)
        for node, s, a in zip(self.left_nodes, sigma, self.area):
            if s != 0.0:
                f[self.dof_left_x[node]] -= s * a
        u = np.zeros(self.n_dof)
        u[self._free] = self._lu.solve(f[self._free])
        return u

    def openings(self, load: float, tractions: np.ndarray) -> np.ndarray:
        """Crack-face opening at every symmetry-plane node (below the cut)."""
        sigma = self._full_traction_vector(tractions)
        return load * self.a_load - self.C @ sigma

    def _full_traction_vector(self, tractions: np.ndarray) -> np.ndarray:
        sigma = np.zeros_like(self.left_y)
        sigma[self.active_mask] = tractions
        return sigma



def _solve_small(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Gaussian elimination with partial pivoting for small dense systems.

    Deliberately hand-written: on a many-core host the multithreaded BLAS entry
    overhead dominates a ``n ~ 40`` LAPACK call (measured ~5 ms per call, versus
    ~0.08 ms single-threaded), and this solve sits inside the state iteration.
    """
    m = np.array(mat, dtype=float, copy=True)
    b = np.array(rhs, dtype=float, copy=True)
    nb = b.size
    for k in range(nb - 1):
        pivot_row = k + int(np.argmax(np.abs(m[k:, k])))
        if pivot_row != k:
            m[[k, pivot_row], :] = m[[pivot_row, k], :]
            b[[k, pivot_row]] = b[[pivot_row, k]]
        pivot = m[k, k]
        if pivot == 0.0:
            raise np.linalg.LinAlgError("singular system in _solve_small")
        factors = m[k + 1 :, k] / pivot
        m[k + 1 :, :] -= np.outer(factors, m[k, :])
        b[k + 1 :] -= factors * b[k]
    x = np.zeros(nb)
    for k in range(nb - 1, -1, -1):
        x[k] = (b[k] - m[k, k + 1 :] @ x[k + 1 :]) / m[k, k]
    return x


def _dN2(xi: float, eta: float) -> np.ndarray:
    """Shape-function derivatives of the Q4 element in natural coordinates."""
    return 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )


def _q4_stiffness(coords: np.ndarray, dmat: np.ndarray, thickness: float) -> np.ndarray:
    """Q4 plane-stress stiffness of one element (2x2 Gauss, explicit 2x2 inverse).

    The 2x2 Jacobian is inverted in closed form on purpose: on this platform
    ``numpy.linalg.solve`` on a 2x2 system costs milliseconds (LAPACK/BLAS call
    overhead), which dominates the whole analysis if it sits in the element loop.
    """
    ke = np.zeros((8, 8))
    for xi, eta in _Q4_GP:
        dN = _dN2(xi, eta)
        jac = dN @ coords
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        if det <= 0.0:
            raise ValueError("degenerate Q4 element")
        jac_inv = np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det
        dndx = jac_inv @ dN
        b = np.zeros((3, 8))
        b[0, 0::2] = dndx[0]
        b[1, 1::2] = dndx[1]
        b[2, 0::2] = dndx[1]
        b[2, 1::2] = dndx[0]
        ke += (b.T @ dmat @ b) * det * thickness
    return ke


def _q4_stiffness_batch(coords: np.ndarray, dmat: np.ndarray, thickness: float) -> np.ndarray:
    """Vectorised Q4 stiffness of ``n`` elements at once, shape ``(n, 8, 8)``.

    Identical algebra to :func:`_q4_stiffness` (asserted in the tests); only the
    loop over elements is removed so that a full refinement study stays cheap.
    """
    n = coords.shape[0]
    ke = np.zeros((n, 8, 8))
    for xi, eta in _Q4_GP:
        dN = _dN2(xi, eta)
        jac = np.einsum("ai,eij->eaj", dN, coords)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        if np.any(det <= 0.0):
            raise ValueError("degenerate Q4 element")
        jac_inv = np.empty_like(jac)
        jac_inv[:, 0, 0] = jac[:, 1, 1]
        jac_inv[:, 0, 1] = -jac[:, 0, 1]
        jac_inv[:, 1, 0] = -jac[:, 1, 0]
        jac_inv[:, 1, 1] = jac[:, 0, 0]
        jac_inv /= det[:, None, None]
        dndx = np.einsum("eab,bi->eai", jac_inv, dN)
        b = np.zeros((n, 3, 8))
        b[:, 0, 0::2] = dndx[:, 0, :]
        b[:, 1, 1::2] = dndx[:, 1, :]
        b[:, 2, 0::2] = dndx[:, 1, :]
        b[:, 2, 1::2] = dndx[:, 0, :]
        ke += np.einsum("eai,ab,ebj->eij", b, dmat, b) * det[:, None, None] * thickness
    return ke


def _plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    c = young / (1.0 - poisson**2)
    return c * np.array(
        [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1 - poisson)]]
    )


def build_influence_model(
    geom: SENBGeometry,
    law: _SofteningLaw,
    *,
    nodes_y: int = 48,
    platen_fraction: float = 0.05,
) -> InfluenceModel:
    """Assemble the half body, factorise it once and extract the influence matrix.

    Mesh: structured Q4 grid on ``x in [S/2, S]``, ``y in [0, W]`` with the
    symmetry-plane node column at ``x = S/2``.  Boundary conditions: ``u_x = 0``
    on the symmetry plane above the admissible crack path (ligament) and
    ``u_y = 0`` at ``x = S, y = 0`` (roller).

    ``C[:, j]`` is the opening at node ``i`` caused by a unit *traction* on node
    ``j``; the pair of opposite tractions on the two crack faces of the full
    specimen is accounted for by the factor two between the half-model
    displacement and the physical crack opening.  ``a_load`` is the opening per
    unit *total* load, so a point on the symmetry plane has
    ``w_i = load * a_load_i - sum_j C_ij sigma_j``.
    """
    W, S, B = geom.depth, geom.span, geom.thickness
    ny = int(nodes_y)
    nx = max(4, int(round(2.4 * ny * (S - 0.5 * W) / W)))
    dy = W / ny
    dx = (S - 0.5 * W) / nx

    xs = 0.5 * W + dx * np.arange(nx + 1)
    ys = dy * np.arange(ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    n_y = ny + 1

    def nid(i, j):
        return i * n_y + j

    elements = np.empty((nx * ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            elements[i * ny + j] = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]

    left_nodes = np.array([nid(0, j) for j in range(n_y)], dtype=int)
    left_y = nodes[left_nodes, 1]

    cut_node = int(round(geom.cut_fraction * ny))
    cut_node = max(2, min(ny - 1, cut_node))
    notch_node = int(round(geom.notch / dy))
    notch_node = max(1, min(cut_node - 1, notch_node))
    effective_notch = notch_node * dy
    effective_cut = cut_node * dy

    active_mask = (np.arange(n_y) >= notch_node) & (np.arange(n_y) <= cut_node - 1)
    ligament_mask = np.arange(n_y) >= cut_node

    n_dof = 2 * nodes.shape[0]
    dof_left_x = 2 * left_nodes

    dmat = _plane_stress_matrix(geom.young, geom.poisson)
    k_local = _q4_stiffness_batch(nodes[elements], dmat, B)

    rows, cols, vals = [], [], []
    for e, el in enumerate(elements):
        dofs = np.empty(8, dtype=int)
        dofs[0::2] = 2 * el
        dofs[1::2] = 2 * el + 1
        ke = k_local[e]
        rows.append(np.repeat(dofs, 8))
        cols.append(np.tile(dofs, 8))
        vals.append(ke.ravel())
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dof, n_dof),
    ).tocsr()

    fixed = list(dof_left_x[ligament_mask]) + [2 * nid(nx, 0) + 1]
    fixed = np.unique(np.array(fixed, dtype=int))
    free = np.setdiff1d(np.arange(n_dof), fixed)
    lu = spla.splu(K[free][:, free].tocsc())

    def solve_free(rhs):
        full = np.zeros(n_dof)
        full[free] = lu.solve(rhs[free])
        return full

    # platen: distributed load on a small patch at the symmetry plane, total 1 N
    n_platen = max(1, int(round(platen_fraction * ny)))
    platen_nodes = np.array([nid(0, n_y - 1 - k) for k in range(n_platen + 1)], dtype=int)
    wts = np.full(len(platen_nodes), dx)
    wts[0] = wts[-1] = 0.5 * dx
    wts = wts / wts.sum()
    platen_dofs = 2 * platen_nodes + 1
    f_unit = np.zeros(n_dof)
    f_unit[platen_dofs] = -0.5 * wts  # half body carries half of the total load

    u_unit = solve_free(f_unit)
    a_load = 2.0 * u_unit[dof_left_x]
    d_load = float(np.dot(wts, u_unit[platen_dofs]))

    # tributary fracture area per symmetry-plane node (half body = physical crack area)
    area = np.zeros(n_y)
    for j in range(0, cut_node):
        h = dy
        if j == 0 or j == cut_node - 1:
            h = 0.5 * dy
        area[j] = B * h

    C = np.zeros((n_y, n_y))
    d_traction = np.zeros(n_y)
    for j in range(n_y):
        if area[j] == 0.0:
            continue
        f = np.zeros(n_dof)
        f[dof_left_x[j]] -= area[j]  # unit tension (1 Pa) on the half body
        uj = solve_free(f)
        C[:, j] = -2.0 * uj[dof_left_x]
        d_traction[j] = uj[platen_dofs] @ wts

    return InfluenceModel(
        geometry=geom,
        law=law,
        nodes=nodes,
        elements=elements,
        left_nodes=left_nodes,
        left_y=left_y,
        active_mask=active_mask,
        area=area,
        C=C,
        a_load=a_load,
        d_load=d_load,
        d_traction=d_traction,
        dof_left_x=dof_left_x,
        platen_dofs=platen_dofs,
        platen_weights=wts,
        n_dof=n_dof,
        effective_notch=effective_notch,
        effective_cut=effective_cut,
        n_nodes_y=ny,
        _free=free,
        _lu=lu,
        _K=K,
        f_unit=f_unit,
    )

# ---------------------------------------------------------------------------
# CMOD-controlled nonlinear solution
# ---------------------------------------------------------------------------
@dataclass
class BendPath:
    """Result of a CMOD-controlled bend analysis."""

    cmod: np.ndarray
    load: np.ndarray
    deflection: np.ndarray
    opening: np.ndarray            # (n_step, n_active) [m]
    traction: np.ndarray           # (n_step, n_active) [Pa]
    dissipated: np.ndarray         # constitutive dissipation [J]
    solid_energy: np.ndarray       # full-specimen strain energy [J]
    interface_energy: np.ndarray   # recoverable interface energy [J]
    external_work: np.ndarray      # int P d(delta) [J]
    balance: np.ndarray            # (W_ext - 2 U_solid - U_if - D) / W_ext
    process_zone: np.ndarray       # damaged length along the crack path [m]
    newton_iterations: np.ndarray
    converged: np.ndarray
    peak_load: float
    peak_index: int
    max_recoil: float              # worst negative opening increment (history check)

    @property
    def cmod_at_peak(self) -> float:
        return float(self.cmod[self.peak_index])

    def summary(self) -> dict:
        return {
            "peak_load_n": float(self.peak_load),
            "cmod_at_peak_m": self.cmod_at_peak,
            "steps": int(len(self.cmod)),
            "converged_steps": int(self.converged.sum()),
            "max_recoil_m": float(self.max_recoil),
            "final_dissipated_j": float(self.dissipated[-1]),
            "max_abs_balance": float(np.max(np.abs(self.balance))),
        }


def _w_scale(law: _SofteningLaw) -> float:
    if hasattr(law, "w0"):
        return float(law.w0)
    return float(law.failure_opening - law.elastic_opening)


def solve_cmod_control(
    model: InfluenceModel,
    *,
    cmod_max: float | None = None,
    n_steps: int = 48,
    tol: float = 1e-10,
    max_iter: int = 200,
    cmod_min_fraction: float = 1e-2,
) -> BendPath:
    """Trace the load--CMOD curve with the crack-mouth opening as control variable.

    Formulation.  The cohesive tractions ``sigma`` (one per symmetry-plane node
    above the notch) and the total load ``P`` are the unknowns; the interface
    carries the *secant* stiffness ``k_s(kappa) = f(kappa)/kappa`` of its own
    history variable ``kappa = max opening`` (irreversible damage), so the
    equation set is **linear** for a frozen state:

        ``(C sigma)_i + sigma_i / k_s,i = P a_i``            (i in the process zone)
        ``(C sigma)_0 - P a_0 = -delta``                     (prescribed mouth opening)

    and the state is updated by ``kappa <- max(kappa, sigma/k_s)`` until it stops
    moving.  This is the classical secant (initial-stiffness) iteration of
    cohesive-crack analysis; it needs no tangent at the load peak, which is where
    a load-controlled Newton iteration stalls.  The update is under-relaxed
    (Aitken-style halving) if it diverges.

    Physical interpretation of the secant state: points with ``kappa <= w_e`` are
    still intact (``k_s = k0``, the penalty stiffness); points beyond ``w_e``
    soften on the envelope and unload towards the origin, as a damage model does.
    """
    law = model.law
    ft = law.tensile_strength
    w_scale = _w_scale(law)
    p_ref = ft * model.geometry.thickness * model.geometry.depth

    act = np.nonzero(model.active_mask)[0]
    n = act.size
    C = model.C
    a_load = model.a_load
    area = model.area[act]
    mouth = model.mouth_index()
    Caa = C[np.ix_(act, act)]
    Ca = a_load[act]
    Cm = C[mouth, act]
    am = float(a_load[mouth])
    dy_node = float(model.left_y[1] - model.left_y[0])
    w_elastic = law.elastic_opening

    if cmod_max is None:
        cmod_max = 12.0 * w_scale
    cmod_targets = np.geomspace(cmod_min_fraction * w_scale, cmod_max, n_steps)

    D_hat = Caa * ft / w_scale
    b_scaled = Ca * p_ref / w_scale
    Dm_hat = Cm * ft / w_scale
    bm_scaled = am * p_ref / w_scale
    newton_tol = 1e-11

    kappa = np.zeros(n)
    s_out = np.zeros((n_steps, n))
    w_out = np.zeros((n_steps, n))
    load_out = np.zeros(n_steps)
    defl_out = np.zeros(n_steps)
    cmod_out = np.zeros(n_steps)
    diss_out = np.zeros(n_steps)
    solid_out = np.zeros(n_steps)
    iface_out = np.zeros(n_steps)
    work_out = np.zeros(n_steps)
    zone_out = np.zeros(n_steps)
    iters_out = np.zeros(n_steps, dtype=int)
    conv_out = np.zeros(n_steps, dtype=bool)

    max_recoil = 0.0
    w_prev = np.zeros(n)

    for step, delta in enumerate(cmod_targets):
        omega = 1.0
        omega_prev = 1.0
        dk_prev = None
        converged = False
        kappa_prev = kappa.copy()
        kappa_trial = kappa.copy()
        for it in range(max_iter):
            k_sec = np.where(
                kappa_trial > 0.0,
                law.traction(kappa_trial) / np.maximum(kappa_trial, 1e-300),
                law.initial_stiffness,
            )
            k_sec = np.maximum(k_sec, 1e-12 * law.initial_stiffness)
            mat = np.empty((n + 1, n + 1))
            mat[:n, :n] = Caa
            mat[np.diag_indices(n)] += 1.0 / k_sec
            mat[:n, n] = -Ca
            mat[n, :n] = Cm
            mat[n, n] = -am
            rhs = np.zeros(n + 1)
            rhs[n] = -delta
            sol = _solve_small(mat, rhs)
            sigma, load = sol[:n], float(sol[n])
            w = sigma / k_sec
            kappa_new = np.maximum(kappa_trial, w)
            dk = kappa_new - kappa_trial
            change = float(np.max(np.abs(dk))) / w_scale
            if change < tol:
                kappa_trial = kappa_new
                converged = True
                break
            # Aitken dynamic relaxation (Delta^2), clipped to a sane range
            if dk_prev is not None:
                diff = dk - dk_prev
                den = float(np.dot(diff, diff))
                if den > 0.0:
                    omega = -omega_prev * float(np.dot(dk_prev, diff)) / den
                    omega = min(max(omega, 0.05), 2.0)
            dk_prev = dk.copy()
            omega_prev = omega
            kappa_trial = kappa_trial + omega * dk
            kappa_trial = np.maximum(kappa_trial, kappa_prev)

        # secant-consistent opening at the trial state (committed only at the end)
        k_sec = np.where(
            kappa_trial > 0.0,
            law.traction(kappa_trial) / np.maximum(kappa_trial, 1e-300),
            law.initial_stiffness,
        )
        k_sec = np.maximum(k_sec, 1e-12 * law.initial_stiffness)
        mat = np.empty((n + 1, n + 1))
        mat[:n, :n] = Caa
        mat[np.diag_indices(n)] += 1.0 / k_sec
        mat[:n, n] = -Ca
        mat[n, :n] = Cm
        mat[n, n] = -am
        rhs = np.zeros(n + 1)
        rhs[n] = -delta
        sol = _solve_small(mat, rhs)
        sigma = sol[:n]
        load = float(sol[n])
        w = sigma / k_sec

        # safeguarded Newton polish on the exact envelope: accepted only if it
        # lowers the residual of the (CMOD-constrained) system
        def _residual(what_v, p_v):
            s_hat = law.traction(what_v * w_scale) / ft
            r = np.empty(n + 1)
            r[:n] = p_v * b_scaled - D_hat @ s_hat - what_v
            r[n] = p_v * bm_scaled - Dm_hat @ s_hat - delta / w_scale
            return r

        what0 = w / w_scale
        p0 = load / p_ref
        r0 = _residual(what0, p0)
        nrm0 = float(np.linalg.norm(r0, np.inf))
        what, p, nrm = what0, p0, nrm0
        for it_n in range(30):
            if nrm < newton_tol:
                break
            d_hat_v = law.slope(what * w_scale) * w_scale / ft
            jac = np.empty((n + 1, n + 1))
            jac[:n, :n] = -D_hat * d_hat_v[None, :]
            jac[np.diag_indices(n)] -= 1.0
            jac[:n, n] = b_scaled
            jac[n, :n] = -Dm_hat * d_hat_v
            jac[n, n] = bm_scaled
            dx = _solve_small(jac, -r0 if nrm == nrm0 else -_residual(what, p))
            cur = _residual(what, p)
            alpha = 1.0
            accepted = False
            for _ in range(20):
                trial = np.clip(what + alpha * dx[:n], 0.0, None)
                p_trial = p + alpha * dx[n]
                r_try = _residual(trial, p_trial)
                if np.linalg.norm(r_try, np.inf) <= (1.0 - 1e-5 * alpha) * np.linalg.norm(cur, np.inf):
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
            what, p = trial, p_trial
            nrm = float(np.linalg.norm(_residual(what, p), np.inf))
        if nrm > nrm0:  # polish made it worse -> keep the secant solution
            what, p, nrm = what0, p0, nrm0
        else:
            sigma = law.traction(what * w_scale)
            load = p * p_ref
            w = what * w_scale
        converged = converged or (nrm < 1e3 * newton_tol)
        # the history variable is committed only by a converged step, so a failed
        # step can never corrupt the following ones
        kappa = np.maximum(kappa_prev, w) if converged else kappa_prev
        it_total = it + it_n + 1
        iters_out[step] = it_total
        conv_out[step] = converged
        s_out[step] = sigma
        w_out[step] = w
        load_out[step] = load
        defl_out[step] = -model.platen_displacement(load, sigma)  # positive downward
        cmod_out[step] = float(model.openings(load, sigma)[mouth])
        diss_out[step] = float(np.sum(area * (law.dissipated(w) - 0.5 * sigma * w)))
        iface_out[step] = float(np.sum(area * (0.5 * sigma * w)))

        u = model.solid_displacement(load, sigma)
        f = load * model.f_unit
        for k, (s, a) in enumerate(zip(sigma, area)):
            f[model.dof_left_x[act[k]]] -= s * a
        solid_out[step] = 2.0 * 0.5 * float(np.dot(u, f))

        damaged = w > w_elastic
        zone_out[step] = float((np.nonzero(damaged)[0].max() + 1) * dy_node) if damaged.any() else 0.0

        if step == 0:
            work_out[step] = 0.5 * load * defl_out[step]
        else:
            work_out[step] = work_out[step - 1] + 0.5 * (
                load + load_out[step - 1]
            ) * (defl_out[step] - defl_out[step - 1])

        max_recoil = max(max_recoil, float(np.max(np.clip(w_prev - w, 0.0, None))))
        w_prev = w

    balance = (work_out - solid_out - iface_out - diss_out) / np.maximum(work_out, 1e-30)
    peak_index = int(np.argmax(load_out))
    return BendPath(
        cmod=cmod_out,
        load=load_out,
        deflection=defl_out,
        opening=w_out,
        traction=s_out,
        dissipated=diss_out,
        solid_energy=solid_out,
        interface_energy=iface_out,
        external_work=work_out,
        balance=balance,
        process_zone=zone_out,
        newton_iterations=iters_out,
        converged=conv_out,
        peak_load=float(load_out[peak_index]),
        peak_index=peak_index,
        max_recoil=max_recoil,
    )


# ---------------------------------------------------------------------------
# linear-elastic (LEFM) consistency of the influence data
# ---------------------------------------------------------------------------
def elastic_stress_intensity_from_compliance(
    model: InfluenceModel,
    notch_depth: float,
    *,
    delta: float | None = None,
) -> dict:
    """``K_I`` of the *traction-free* notched body, from the model's own compliance.

    The ligament above ``notch_depth`` is tied rigidly (``w = 0``), which turns
    the flexibility system into ``C_tt sigma = P a_t``; the load-point deflection
    then follows from reciprocity and the energy-release rate
    ``G = P^2/2 dC/da`` gives ``K_I = sqrt(E' G)``.  This is an independent route
    to the stress-intensity factor (compliance derivative, not a near-tip field
    fit) and is compared with the Srawley--Gross handbook value.
    """
    geom = model.geometry
    dy = float(model.left_y[1] - model.left_y[0])
    if delta is None:
        delta = 0.02 * geom.depth

    act = np.nonzero(model.active_mask)[0]

    def compliance(a):
        tied = act[model.left_y[act] >= a - 1e-9 * geom.depth]
        if tied.size < 2:
            raise ValueError("no tied ligament nodes for this notch depth")
        a_used = float(model.left_y[tied[0]])
        c_tt = model.C[np.ix_(tied, tied)]
        rhs = model.a_load[tied]
        sigma = _solve_small(c_tt, rhs)
        return -float(model.d_load + np.dot(sigma, model.d_traction[tied])), a_used

    c_minus, a_minus = compliance(notch_depth - delta)
    c_plus, a_plus = compliance(notch_depth + delta)
    dc_da = (c_plus - c_minus) / (a_plus - a_minus)
    # G = P^2/2 dC/da for a unit load; K = sqrt(E' G)
    g = 0.5 * dc_da
    k_model = math.sqrt(geom.modified_modulus * g)
    k_handbook = senb_handbook_stress_intensity(1.0, geom, notch=a_plus)
    return {
        "notch_m": a_plus,
        "alpha": a_plus / geom.depth,
        "compliance_m_per_n": 0.5 * (c_minus + c_plus),
        "dc_da_m_per_n_per_m": dc_da,
        "K_from_compliance_pa_sqrt_m": k_model,
        "K_handbook_pa_sqrt_m": k_handbook,
        "rel_error_pct": 100.0 * (k_model - k_handbook) / k_handbook,
    }


# ---------------------------------------------------------------------------
# size-effect driver
# ---------------------------------------------------------------------------
@dataclass
class SizeEffectPoint:
    depth: float
    thickness: float
    span: float
    notch: float
    alpha: float
    brittleness: float             # W / l_ch
    peak_load: float
    lefm_peak: float
    strength_peak: float
    ratio_lefm: float
    ratio_strength: float
    cmod_at_peak: float
    process_zone: float
    steps_converged: int
    steps: int
    max_abs_balance: float
    dissipated_final: float

    def as_dict(self) -> dict:
        return {
            "depth_m": self.depth,
            "thickness_m": self.thickness,
            "span_m": self.span,
            "notch_m": self.notch,
            "alpha": self.alpha,
            "W_over_lch": self.brittleness,
            "peak_load_n": self.peak_load,
            "lefm_peak_load_n": self.lefm_peak,
            "strength_peak_load_n": self.strength_peak,
            "peak_over_lefm": self.ratio_lefm,
            "peak_over_strength": self.ratio_strength,
            "cmod_at_peak_m": self.cmod_at_peak,
            "process_zone_m": self.process_zone,
            "steps": self.steps,
            "steps_converged": self.steps_converged,
            "max_abs_energy_balance": self.max_abs_balance,
            "dissipated_final_j": self.dissipated_final,
        }


def run_size_effect(
    *,
    young: float = 30.0e9,
    poisson: float = 0.2,
    tensile_strength: float = 3.0e6,
    fracture_energy: float = 100.0,
    initial_stiffness_factor: float = 1.0e3,
    depths: tuple[float, ...] = (0.0833, 0.1667, 0.3333, 0.6667, 1.3333, 2.6667, 5.3333),
    alpha: float = 0.2,
    span_ratio: float = 4.0,
    cut_fraction: float = 0.9,
    nodes_y: int = 48,
    n_steps: int = 48,
    law_kind: str = "exponential",
    progress=None,
) -> list[SizeEffectPoint]:
    """Run the cohesive bend analysis over a geometrically similar size series.

    Everything is scaled by the specimen depth, so ``W / l_ch`` is the only
    dimensionless group that changes (``l_ch = E G_F / f_t^2``); the penalty
    stiffness scales as ``k0 = factor * E / W`` so that the elastic interface
    compliance is resolution independent.
    """
    law_cls = {"exponential": ExponentialSoftening, "bilinear": BilinearSoftening}[law_kind]
    out: list[SizeEffectPoint] = []
    for depth in depths:
        k0 = initial_stiffness_factor * young / depth
        law = law_cls(tensile_strength=tensile_strength,
                      fracture_energy=fracture_energy,
                      initial_stiffness=k0)
        geom = SENBGeometry(
            depth=float(depth),
            span=float(span_ratio * depth),
            thickness=float(depth),
            notch=float(alpha * depth),
            cut_fraction=cut_fraction,
            young=young,
            poisson=poisson,
        )
        model = build_influence_model(geom, law, nodes_y=nodes_y)
        path = solve_cmod_control(model, n_steps=n_steps)
        l_ch = geom.modified_modulus * fracture_energy / tensile_strength**2
        lefm = senb_lefm_peak_load(geom, fracture_energy)
        strength = strength_limit_load(geom, tensile_strength)
        out.append(
            SizeEffectPoint(
                depth=float(depth),
                thickness=float(depth),
                span=float(span_ratio * depth),
                notch=model.effective_notch,
                alpha=model.effective_notch / depth,
                brittleness=float(depth / l_ch),
                peak_load=path.peak_load,
                lefm_peak=float(lefm),
                strength_peak=float(strength),
                ratio_lefm=path.peak_load / lefm,
                ratio_strength=path.peak_load / strength,
                cmod_at_peak=path.cmod_at_peak,
                process_zone=float(path.process_zone[path.peak_index]),
                steps=len(path.cmod),
                steps_converged=int(path.converged.sum()),
                max_abs_balance=float(np.max(np.abs(path.balance))),
                dissipated_final=float(path.dissipated[-1]),
            )
        )
        if progress is not None:
            progress(out[-1])
    return out
