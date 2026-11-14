"""Ice-belt local structure: elastoplastic progressive collapse of stiffened plating.

Why this module exists
----------------------
``tensorfem.plate_mindlin`` supplies an *elastic* Mindlin plate and
``tensorfem.smith_progressive_collapse`` a *closed-form* panel
stress-shortening curve anchored on the Faulkner effective-width law.  Neither
answers the question an ice-class structural assessment actually asks: *what
local ice pressure does this ice-belt plate field carry before it collapses, and
does the deformation path conserve energy?*  That needs

* a layered **elastoplastic** plate element (von Mises + isotropic hardening,
  plane stress enforced at every through-thickness layer),
* **moderate-rotation** (von Karman) kinematics so that the peak of the
  load-shortening curve -- not first yield -- can be located,
* eccentric **Tee stiffeners** sharing plate nodes (axial + own bending +
  column geometric stiffness), and
* incremental/iterative solution control (Newton load / displacement /
  arc-length) with an auditable energy ledger.

This module is that surface.  It is a *verification face*: every capability is
tied to an executable anchor (see :func:`run_verification_suite`), and no
literature table value is transcribed -- rule coefficients are caller supplied
and carry a provenance string.

Limit loads are located with :func:`solve_arc_length`, a modified Riks
continuation (spherical, load-factor weighted, Crisfield root selection with a
continuation guard); the load/displacement controlled solvers are kept as the
independent corroboration of every peak, and the energy ledger is recomputed
from the stored path rather than accumulated incrementally.

Model content and provenance
----------------------------
* ``SteelGrade`` -- Ludwik hardening ``sigma_y(p) = sigma_0 + K p**n``.  AH36 /
  DH36: ``sigma_0 = 355 MPa`` and ``sigma_u = 490 MPa`` are the specified
  minimum values of the normal-strength grades, ``E = 206 GPa``, ``nu = 0.3``.
  ``n = 0.5`` is a documented engineering choice and ``K`` is **calibrated in
  module** so the curve passes through ``(p = 0.15, sigma_u)``; the anchor state
  is therefore reproduced by construction and only deviations are model content.
* ``IACSPressureArea`` -- ``p(A) = p0 (A/A0)**-0.3``.  The exponent ``-0.3`` is
  the documented IACS local pressure-area exponent; the class anchor ``p0`` is
  **caller supplied** (no rule table offline, nothing embedded).
* Plate element -- 4-node flat shell, 5 DOF/node ``(u, v, w, psi_x, psi_y)``;
  membrane and bending at 2x2 Gauss, transverse shear at one centre point with
  ``kappa = 5/6`` (selective reduced integration, the plane-stress analogue of
  MITC4 on regular meshes); 5 Gauss layers through the thickness so a plastic
  hinge can form inside one element.
* ``K_T`` = finite-differenced (hence consistent) algorithmic material tangent
  plus the von Karman geometric stiffness ``K_sigma(N)``.  The ``K_L ==
  K_sigma`` symmetrisation of the moderate-rotation tangent is the standard
  engineering approximation and is stated, not hidden.

Deliberate scope limits
-----------------------
Small strain, moderate rotations, no strain-rate / temperature effect, no
through-thickness residual-stress field, no crack or brittle screening, no
contact, and no stiffener web local buckling.  The anchor set is chosen so that
local buckling is not the governing mechanism of the FE case.  Numbers and the
withheld-anchor list are in ``docs/ice-belt-progressive-collapse.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------------------- #
# material
# --------------------------------------------------------------------------- #
SIGMA_U_REF = 4.90e8   # specified tensile strength, normal-strength hull steel
P_REF = 0.15           # 5d gauge plastic strain at the tensile strength
NU_STEEL = 0.30
E_STEEL = 2.06e11
N_HARDENING = 0.50     # Ludwik exponent: documented engineering choice


@dataclass(frozen=True)
class SteelGrade:
    """Ludwik hardening ``sigma_y(p) = sigma_0 + K p**n`` with elastic data."""

    name: str
    young: float
    poisson: float
    sigma_0: float
    K: float
    n: float
    sigma_u: float
    provenance: str

    @property
    def lam(self) -> float:
        return self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))

    @property
    def mu(self) -> float:
        return self.young / (2 * (1 + self.poisson))

    def plane_stress_matrix(self) -> np.ndarray:
        c = self.young / (1 - self.poisson ** 2)
        return c * np.array([[1.0, self.poisson, 0.0],
                             [self.poisson, 1.0, 0.0],
                             [0.0, 0.0, 0.5 * (1 - self.poisson)]])

    def yield_stress(self, pbar) -> np.ndarray:
        p = np.maximum(np.asarray(pbar, dtype=float), 1e-12)
        return self.sigma_0 + self.K * np.power(p, self.n)

    def hardening_slope(self, pbar, cap: float = 1.0e12) -> np.ndarray:
        p = np.maximum(np.asarray(pbar, dtype=float), 1e-12)
        return np.minimum(self.K * self.n * np.power(p, self.n - 1.0), cap)

    def plastic_work_increment(self, p_old, p_new) -> np.ndarray:
        """Exact ``int sigma_y dp`` for isotropic Ludwik hardening."""
        po = np.maximum(np.asarray(p_old, dtype=float), 0.0)
        pn = np.maximum(np.asarray(p_new, dtype=float), 0.0)
        return (self.sigma_0 * (pn - po)
                + self.K / (self.n + 1.0)
                * (np.power(pn, self.n + 1.0) - np.power(po, self.n + 1.0)))

    def uniaxial_stress(self, strain) -> np.ndarray:
        """Uniaxial reference curve implied by the hardening law."""
        eps = np.asarray(strain, dtype=float)
        e = self.sigma_0 / self.young
        out = self.young * eps
        plastic = eps > e
        if np.any(plastic):
            p = eps[plastic] - self.sigma_0 / self.young - self.sigma_0 / self.young * 0.0
            p = eps[plastic] - self.sigma_0 / self.young
            p = np.maximum(p, 0.0)
            out = out.copy()
            out[plastic] = self.sigma_0 + self.K * np.power(p, self.n)
        return out


def calibrate_user_K(sigma_0: float, sigma_u: float = SIGMA_U_REF,
                     p_ref: float = P_REF, n: float = N_HARDENING) -> float:
    """``K`` that puts the Ludwik curve through ``(p_ref, sigma_u)``."""
    if sigma_u <= sigma_0:
        raise ValueError("sigma_u must exceed sigma_0")
    return (sigma_u - sigma_0) / p_ref ** n


def _grade(name: str, sigma_0: float, sigma_u: float, note: str) -> SteelGrade:
    return SteelGrade(
        name=name, young=E_STEEL, poisson=NU_STEEL, sigma_0=sigma_0,
        K=calibrate_user_K(sigma_0, sigma_u), n=N_HARDENING, sigma_u=sigma_u,
        provenance=(
            f"sigma_0={sigma_0 / 1e6:.1f} MPa, sigma_u={sigma_u / 1e6:.1f} MPa are the "
            "specified minima of the grade; K is calibrated in module so the Ludwik "
            f"curve passes through (p={P_REF}, sigma_u); n={N_HARDENING} is a documented "
            f"engineering choice (no tensile database offline). {note}"
        ))


AH36 = _grade("AH36", 3.55e8, SIGMA_U_REF,
              "AH36/DH36 differ in toughness, not in the strength data used here.")
DH36 = _grade("DH36", 3.55e8, SIGMA_U_REF,
              "Identical strength calibration to AH36 by design; the grade difference "
              "is Charpy toughness, which this module does not model.")


def elastic_perfectly_plastic(sigma_0: float = 3.55e8, young: float = E_STEEL,
                              poisson: float = NU_STEEL) -> SteelGrade:
    """``K = 0`` specialisation used for the plastic-hinge anchor."""
    return SteelGrade(name="EPP", young=young, poisson=poisson, sigma_0=sigma_0,
                      K=0.0, n=N_HARDENING, sigma_u=sigma_0,
                      provenance="ideal elastic-perfectly-plastic reference material")


# --------------------------------------------------------------------------- #
# ice load
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IACSPressureArea:
    """``p(A) = p0 * (A/A0)**-exponent`` local ice pressure."""

    p0: float
    area_ref: float = 1.0
    exponent: float = 0.3
    provenance: str = (
        "pressure-area power form with the documented IACS local-pressure exponent; "
        "the anchor p0 is caller supplied (rule table not available offline)")

    def pressure(self, area):
        a = np.asarray(area, dtype=float)
        if np.any(a <= 0.0):
            raise ValueError("load area must be positive")
        if self.p0 <= 0.0 or self.area_ref <= 0.0:
            raise ValueError("p0 and area_ref must be positive")
        if self.exponent < 0.0:
            raise ValueError("exponent must be non-negative")
        return self.p0 * np.power(a / self.area_ref, -self.exponent)

    def total_force(self, area):
        a = np.asarray(area, dtype=float)
        return self.pressure(a) * a


def pressure_preset(ice_class: str) -> IACSPressureArea:
    """Refuses to invent a rule anchor: the class ``p0`` must be supplied."""
    raise KeyError(
        f"no offline tabulated p0 for ice class {ice_class!r}; construct "
        "IACSPressureArea(p0=..., provenance=...) from the applicable rule edition")


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
DOF_PER_NODE = 5
I_U, I_V, I_W, I_RX, I_RY = 0, 1, 2, 3, 4

_G = 1.0 / math.sqrt(3.0)
_GP2 = ((-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G))
_GAUSS_Z = (-0.906179845938664, -0.538469310105683, 0.0,
            0.538469310105683, 0.906179845938664)
_GAUSS_Z_W = (0.236926885056189, 0.478628670499366, 0.568888888888889,
              0.478628670499366, 0.236926885056189)
SHEAR_CORRECTION = 5.0 / 6.0
HOURGLASS_SHARE = 0.02   # share of the 2x2 shear operator used to stabilise the 1-point rule


def _shapes(xi: float, eta: float):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dxi, deta



def _mitc4_shear_operators(coords) -> np.ndarray:
    """MITC4 assumed transverse-shear operators ``(4, 2, 20)`` for one element.

    Covariant shear strains are sampled at the four mid-side tying points and
    interpolated, then transformed to physical components with ``J^-1``.  This
    removes both shear locking and the hourglass mechanisms of the 1-point rule.
    """
    x, y = coords[:, 0], coords[:, 1]
    ties = ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))   # A, B, C, D
    A_xi = np.zeros((4, 20))
    A_eta = np.zeros((4, 20))
    for k, (xi, eta) in enumerate(ties):
        N, dxi, deta = _shapes(xi, eta)
        xs_xi, ys_xi = dxi @ x, dxi @ y
        xs_eta, ys_eta = deta @ x, deta @ y
        for i in range(4):
            A_xi[k, DOF_PER_NODE * i + I_W] = dxi[i]
            A_xi[k, DOF_PER_NODE * i + I_RX] = N[i] * xs_xi
            A_xi[k, DOF_PER_NODE * i + I_RY] = N[i] * ys_xi
            A_eta[k, DOF_PER_NODE * i + I_W] = deta[i]
            A_eta[k, DOF_PER_NODE * i + I_RX] = N[i] * xs_eta
            A_eta[k, DOF_PER_NODE * i + I_RY] = N[i] * ys_eta
    out = np.zeros((4, 2, 20))
    for q, (xi, eta) in enumerate(_GP2):
        N, dxi, deta = _shapes(xi, eta)
        jac = np.array([[dxi @ x, dxi @ y], [deta @ x, deta @ y]])
        inv = np.linalg.inv(jac)
        cov_xi = 0.5 * (1.0 - eta) * A_xi[0] + 0.5 * (1.0 + eta) * A_xi[2]
        cov_eta = 0.5 * (1.0 - xi) * A_eta[3] + 0.5 * (1.0 + xi) * A_eta[1]
        out[q, 0] = inv[0, 0] * cov_xi + inv[0, 1] * cov_eta
        out[q, 1] = inv[1, 0] * cov_xi + inv[1, 1] * cov_eta
    return out


def structured_mesh(length_x: float, length_y: float, nx: int, ny: int):
    """Structured Q4 mesh; node ``(i, j)`` has id ``j*(nx+1)+i``."""
    if nx < 1 or ny < 1:
        raise ValueError("nx, ny must be >= 1")
    xs = np.linspace(0.0, length_x, nx + 1)
    ys = np.linspace(0.0, length_y, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([gx.ravel(order="F"), gy.ravel(order="F")])
    elems = np.empty((nx * ny, 4), dtype=int)
    k = 0
    for i in range(nx):
        for j in range(ny):
            n0 = j * (nx + 1) + i
            elems[k] = (n0, n0 + 1, n0 + 1 + (nx + 1), n0 + (nx + 1))
            k += 1
    return nodes, elems

# --------------------------------------------------------------------------- #
# layered J2 plane-stress return map (vectorised over points)
# --------------------------------------------------------------------------- #
@dataclass
class PointState:
    """Per-integration-point plastic state (arrays of length ``n_points``)."""

    eps_p: np.ndarray      # (n, 4) tensor components xx, yy, zz, xy
    pbar: np.ndarray       # (n,)   equivalent plastic strain

    @staticmethod
    def zeros(n: int) -> "PointState":
        return PointState(eps_p=np.zeros((n, 4)), pbar=np.zeros(n))

    def copy(self) -> "PointState":
        return PointState(self.eps_p.copy(), self.pbar.copy())


def _deviatoric_voigt(s):
    p = (s[..., 0] + s[..., 1] + s[..., 2]) / 3.0
    d = s.copy()
    d[..., 0] -= p
    d[..., 1] -= p
    d[..., 2] -= p
    return d


def _von_mises_from_voigt(s):
    d = _deviatoric_voigt(s)
    j2 = 0.5 * (d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) + d[..., 3] ** 2
    return np.sqrt(np.maximum(3.0 * j2, 0.0))


def _stress_from_elastic_strain(grade: SteelGrade, strain):
    lam, mu = grade.lam, grade.mu
    tr = strain[..., 0] + strain[..., 1] + strain[..., 2]
    s = np.empty_like(strain)
    s[..., 0] = lam * tr + 2 * mu * strain[..., 0]
    s[..., 1] = lam * tr + 2 * mu * strain[..., 1]
    s[..., 2] = lam * tr + 2 * mu * strain[..., 2]
    s[..., 3] = 2 * mu * strain[..., 3]
    return s


def _return_map_3d(grade: SteelGrade, strain_total, state: PointState):
    """Radial return with isotropic hardening -> ``(stress, state, dWp)``."""
    trial_elastic = strain_total - state.eps_p
    sigma_trial = _stress_from_elastic_strain(grade, trial_elastic)
    q_trial = _von_mises_from_voigt(sigma_trial)
    plastic = (q_trial - grade.yield_stress(state.pbar)) > 0.0

    sigma = sigma_trial.copy()
    eps_p = state.eps_p.copy()
    pbar = state.pbar.copy()
    dWp = np.zeros_like(pbar)
    if not np.any(plastic):
        return sigma, PointState(eps_p, pbar), dWp

    G = grade.mu
    idx = np.where(plastic)[0]
    qt = np.maximum(q_trial[idx], 1e-30)
    st = sigma_trial[idx]
    dt = _deviatoric_voigt(st)
    p_old = pbar[idx]
    dp = np.zeros_like(qt)
    for _ in range(80):
        res = qt - 3.0 * G * dp - grade.yield_stress(p_old + dp)
        slope = -3.0 * G - grade.hardening_slope(p_old + dp, cap=4.0 * G)
        slope = np.where(np.abs(slope) < 1e-30, -3.0 * G, slope)
        step = res / slope
        dp = np.maximum(dp - step, 0.0)
        if np.all(np.abs(step) <= 1e-15 + 1e-13 * np.maximum(dp, 1e-30)):
            break
    scale = (3.0 * G * dp) / qt
    sigma[idx] = st - scale[:, None] * dt
    n_dir = (1.5 / qt)[:, None] * dt
    eps_p[idx] = state.eps_p[idx] + dp[:, None] * n_dir
    pbar[idx] = p_old + dp
    dWp[idx] = grade.plastic_work_increment(p_old, p_old + dp)
    return sigma, PointState(eps_p, pbar), dWp


def plane_stress_return(grade: SteelGrade, strain_2d, state: PointState,
                        ez0=None, iterations: int = 40, atol: float | None = None):
    """Return map with ``sigma_zz = 0`` enforced by a scalar loop on ``eps_zz``.

    ``strain_2d`` is ``(n, 3)`` with engineering shear ``gxy``.  Only the points
    still violating ``sigma_zz = 0`` are iterated (elastic points converge in one
    pass), and the update uses the exact elastic derivative ``lambda + 2 mu``.
    Returns ``(stress_2d, state, dWp, eps_zz)``.
    """
    strain_2d = np.atleast_2d(np.asarray(strain_2d, dtype=float))
    n = strain_2d.shape[0]
    ez = np.zeros(n) if ez0 is None else np.array(ez0, dtype=float)
    if atol is None:
        amp = float(np.max(np.abs(strain_2d))) if strain_2d.size else 0.0
        atol = 1e-9 * max(grade.young * amp, 1.0) + 1e-6
    sigma4 = np.zeros((n, 4))
    eps_p = state.eps_p.copy()
    pbar = state.pbar.copy()
    dWp = np.zeros(n)
    active = np.arange(n)
    slope = grade.lam + 2.0 * grade.mu

    def assemble(idx, ez_vec):
        out = np.empty((idx.size, 4))
        out[:, 0] = strain_2d[idx, 0]
        out[:, 1] = strain_2d[idx, 1]
        out[:, 2] = ez_vec
        out[:, 3] = 0.5 * strain_2d[idx, 2]
        return out

    for _ in range(iterations):
        if active.size == 0:
            break
        sub = PointState(state.eps_p[active], state.pbar[active])
        sig, new_state, dw = _return_map_3d(grade, assemble(active, ez[active]), sub)
        sigma4[active] = sig
        eps_p[active] = new_state.eps_p
        pbar[active] = new_state.pbar
        dWp[active] = dw
        szz = sig[:, 2]
        keep = np.abs(szz) > atol
        if not np.any(keep):
            break
        ez[active[keep]] -= szz[keep] / slope
        active = active[keep]
    stress_2d = np.column_stack([sigma4[:, 0], sigma4[:, 1], sigma4[:, 3]])
    return stress_2d, PointState(eps_p, pbar), dWp, ez


def plane_stress_tangent(grade: SteelGrade, strain_2d, state: PointState,
                         ez=None, h: float = 1e-8):
    """Finite-difference algorithmic tangent ``d sigma / d eps`` -> (n, 3, 3)."""
    strain_2d = np.atleast_2d(np.asarray(strain_2d, dtype=float))
    scale = max(float(np.max(np.abs(strain_2d))) if strain_2d.size else 0.0, 1e-6)
    step = h * scale
    n = strain_2d.shape[0]
    C = np.zeros((n, 3, 3))
    for k in range(3):
        pert = strain_2d.copy()
        pert[:, k] += step
        sp, _, _, _ = plane_stress_return(grade, pert, state, ez0=ez)
        pert[:, k] -= 2.0 * step
        sm, _, _, _ = plane_stress_return(grade, pert, state, ez0=ez)
        C[:, :, k] = (sp - sm) / (2.0 * step)
    return 0.5 * (C + np.transpose(C, (0, 2, 1)))


# --------------------------------------------------------------------------- #
# stiffener
# --------------------------------------------------------------------------- #
@dataclass
class Stiffener:
    """Eccentric Tee stiffener sharing the plate nodes of one grid line."""

    node_line: np.ndarray
    direction: str          # 'x' or 'y'
    area: float             # tee cross-section area [m^2]
    inertia: float          # own second moment about its own centroid [m^4]
    offset: float           # centroid distance below plate mid-plane [m], >0
    name: str = "stiffener"


def stiffener_triplets(model: "PlateModel", stiff: Stiffener, axial_force: float = 0.0,
                      include_elastic: bool = True):
    """Elastic + geometric triplets of one stiffener (shared plate nodes)."""
    rows, cols, data = [], [], []
    line = np.asarray(stiff.node_line, dtype=int)
    if stiff.area <= 0.0 or stiff.inertia < 0.0:
        raise ValueError("stiffener area must be positive and inertia non-negative")
    if stiff.direction == "x":
        axial_i, rot_i = I_U, I_RX
    elif stiff.direction == "y":
        axial_i, rot_i = I_V, I_RY
    else:
        raise ValueError("stiffener direction must be 'x' or 'y'")
    e = stiff.offset
    for a in range(line.size - 1):
        n1, n2 = int(line[a]), int(line[a + 1])
        length = float(np.linalg.norm(model.nodes[n2] - model.nodes[n1]))
        if length <= 0.0:
            raise ValueError("zero-length stiffener segment")
        dof = np.array([DOF_PER_NODE * n1 + axial_i, DOF_PER_NODE * n2 + axial_i,
                        DOF_PER_NODE * n1 + rot_i, DOF_PER_NODE * n2 + rot_i])
        if include_elastic:
            B_ax = np.array([-1.0, 1.0, e, -e]) / length
            B_b = np.array([0.0, 0.0, -1.0, 1.0]) / length
            Ke = stiff.area * model.grade.young * length * np.outer(B_ax, B_ax) \
                + stiff.inertia * model.grade.young * length * np.outer(B_b, B_b)
            rows.append(np.repeat(dof, 4))
            cols.append(np.tile(dof, 4))
            data.append(Ke.ravel())
        if axial_force != 0.0:
            wd = np.array([DOF_PER_NODE * n1 + I_W, DOF_PER_NODE * n2 + I_W])
            Kg = (axial_force / length) * np.array([[1.0, -1.0], [-1.0, 1.0]])
            rows.append(np.repeat(wd, 2))
            cols.append(np.tile(wd, 2))
            data.append(Kg.ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class PlateModel:
    """Layered elastoplastic von Karman plate with optional eccentric stiffeners."""

    def __init__(self, nodes, elements, thickness: float, grade: SteelGrade,
                 n_layers: int = 5):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.thickness = float(thickness)
        self.grade = grade
        self.n_layers = int(n_layers)
        if self.n_layers not in (3, 5):
            raise ValueError("n_layers must be 3 or 5 (supported Gauss rules)")
        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.elements.shape[0]
        self.ndof = DOF_PER_NODE * self.n_nodes
        self.stiffeners: list[Stiffener] = []
        self.von_karman = True
        self.ez_iterations = 16
        self._build_operators()
        self.imp_x = np.zeros(self.n_points)
        self.imp_y = np.zeros(self.n_points)

    # -- operators -----------------------------------------------------------
    def _build_operators(self):
        M, nl, nq = self.n_elem, self.n_layers, len(_GP2)
        self.npt_elem = nq * nl
        npt = M * self.npt_elem
        self.n_points = npt
        self.wcol = np.array([DOF_PER_NODE * i + I_W for i in range(4)])
        self.edof = (self.elements[:, :, None] * DOF_PER_NODE
                     + np.arange(DOF_PER_NODE)[None, None, :]).reshape(M, 20)
        self.pt_elem = np.repeat(np.arange(M), self.npt_elem)
        self.dNdx = np.zeros((M, nq, 4))
        self.dNdy = np.zeros((M, nq, 4))
        self.shapeN = np.zeros((M, nq, 4))
        self.detj = np.zeros((M, nq))
        self.B_layer = np.zeros((npt, 3, 20))
        self.layer_weight = np.zeros(npt)
        self.layer_z = np.zeros(npt)
        self.Bs_gp = np.zeros((M, nq, 2, 20))
        if nl == 5:
            zz, zw = np.array(_GAUSS_Z), np.array(_GAUSS_Z_W)
        elif nl == 3:
            zz = np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)])
            zw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        else:
            raise ValueError("n_layers must be 3 or 5")
        for e in range(M):
            coords = self.nodes[self.elements[e]]
            self.Bs_gp[e] = _mitc4_shear_operators(coords)
            for q, (xi, eta) in enumerate(_GP2):
                N, dxi, deta = _shapes(xi, eta)
                jac = np.array([[dxi @ coords[:, 0], dxi @ coords[:, 1]],
                                [deta @ coords[:, 0], deta @ coords[:, 1]]])
                detj = float(np.linalg.det(jac))
                if detj <= 0.0:
                    raise ValueError(f"element {e} has a non-positive Jacobian")
                inv = np.linalg.inv(jac)
                dNdx = inv[0, 0] * dxi + inv[1, 0] * deta
                dNdy = inv[0, 1] * dxi + inv[1, 1] * deta
                self.dNdx[e, q] = dNdx
                self.dNdy[e, q] = dNdy
                self.shapeN[e, q] = N
                self.detj[e, q] = detj
                Bm = np.zeros((3, 20))
                Bb = np.zeros((3, 20))
                for i in range(4):
                    Bm[0, DOF_PER_NODE * i + I_U] = dNdx[i]
                    Bm[1, DOF_PER_NODE * i + I_V] = dNdy[i]
                    Bm[2, DOF_PER_NODE * i + I_U] = dNdy[i]
                    Bm[2, DOF_PER_NODE * i + I_V] = dNdx[i]
                    Bb[0, DOF_PER_NODE * i + I_RX] = dNdx[i]
                    Bb[1, DOF_PER_NODE * i + I_RY] = dNdy[i]
                    Bb[2, DOF_PER_NODE * i + I_RX] = dNdy[i]
                    Bb[2, DOF_PER_NODE * i + I_RY] = dNdx[i]
                for lz in range(nl):
                    k = e * self.npt_elem + q * nl + lz
                    z = 0.5 * self.thickness * zz[lz]
                    self.B_layer[k] = Bm + z * Bb
                    self.layer_z[k] = z
                    self.layer_weight[k] = detj * 0.5 * self.thickness * zw[lz]
        self.layer_weight_nominal = self.layer_weight / np.repeat(
            self.detj.reshape(M * nq), nl)
        self.pt_dNdx = np.repeat(self.dNdx.reshape(M * nq, 4), nl, axis=0)
        self.pt_dNdy = np.repeat(self.dNdy.reshape(M * nq, 4), nl, axis=0)
        self.shear_modulus = self.grade.mu * SHEAR_CORRECTION * self.thickness
        self._row20 = np.repeat(self.edof[:, :, None], 20, axis=2)
        self._col20 = np.repeat(self.edof[:, None, :], 20, axis=1)
        self._edof_w = self.edof[:, self.wcol]
        self._row4 = np.repeat(self._edof_w[:, :, None], 4, axis=2)
        self._col4 = np.repeat(self._edof_w[:, None, :], 4, axis=1)

    # -- imperfection --------------------------------------------------------
    def set_imperfection(self, w0_nodal) -> None:
        """Initial out-of-flatness: enters as an initial slope ``w0_x, w0_y``.

        With ``w_total = w0 + w`` the von Karman membrane strain is
        ``w0_x w_x + 0.5 w_x**2``, so the flat state stays an equilibrium and
        the imperfection drives the buckling-only branch -- the standard
        panel-collapse idealisation (no 3-D geometry needed).
        """
        w0 = np.asarray(w0_nodal, dtype=float)
        if w0.size != self.n_nodes:
            raise ValueError("imperfection field must have one value per node")
        w0_e = w0[self.elements]
        gx = np.einsum("mqi,mi->mq", self.dNdx, w0_e)
        gy = np.einsum("mqi,mi->mq", self.dNdy, w0_e)
        self.imp_x = np.repeat(gx.reshape(-1), self.n_layers)
        self.imp_y = np.repeat(gy.reshape(-1), self.n_layers)

    # -- kinematics ----------------------------------------------------------
    def layer_kinematics(self, u):
        """Return ``(ue, Baug, strain)`` with von Karman membrane terms."""
        ue = u[self.edof]
        ue_pt = ue[self.pt_elem]
        w_pt = ue_pt[:, self.wcol]
        wx = np.einsum("pi,pi->p", self.pt_dNdx, w_pt)
        wy = np.einsum("pi,pi->p", self.pt_dNdy, w_pt)
        Baug = self.B_layer.copy()
        if self.von_karman:
            wxs = wx + self.imp_x
            wys = wy + self.imp_y
            Baug[:, 0, self.wcol] += wxs[:, None] * self.pt_dNdx
            Baug[:, 1, self.wcol] += wys[:, None] * self.pt_dNdy
            Baug[:, 2, self.wcol] += (wxs[:, None] * self.pt_dNdy
                                      + wys[:, None] * self.pt_dNdx)
        strain = np.einsum("pij,pj->pi", Baug, ue_pt)
        if self.von_karman:
            # B_nl = w,x * dN/dx is the derivative of 0.5 w,x**2, so the discrete
            # quadratic strain is half of what B_nl . w returns.
            strain[:, 0] -= 0.5 * wx ** 2
            strain[:, 1] -= 0.5 * wy ** 2
            strain[:, 2] -= wx * wy
        return ue, Baug, strain

    def shear_stiffness_element(self):
        """MITC4 transverse-shear stiffness ``(M, 20, 20)``."""
        return self.shear_modulus * np.einsum(
            "mqki,mqkj,mq->mij", self.Bs_gp, self.Bs_gp, self.detj)

    def shear_force_element(self, ue):
        gamma = np.einsum("mqij,mj->mqi", self.Bs_gp, ue)
        return self.shear_modulus * np.einsum(
            "mqij,mqi,mq->mj", self.Bs_gp, gamma, self.detj)

    def shear_energy(self, ue) -> float:
        gamma = np.einsum("mqij,mj->mqi", self.Bs_gp, ue)
        return 0.5 * self.shear_modulus * float(
            np.einsum("mqi,mqi,mq->", gamma, gamma, self.detj))

    def shear_strain(self, ue):
        """Mean transverse shear strain ``(M, 2)`` (diagnostic)."""
        return np.einsum("mqij,mj->mqi", self.Bs_gp, ue).mean(axis=1)

    # -- constitutive --------------------------------------------------------
    def stress_update(self, strain, state: PointState, ez, need_tangent: bool,
                      ez_iterations: int | None = None):
        ez_iterations = self.ez_iterations if ez_iterations is None else ez_iterations
        stress, new_state, dWp, ez_new = plane_stress_return(
            self.grade, strain, state, ez0=ez, iterations=ez_iterations)
        C = None
        if need_tangent:
            C = plane_stress_tangent(self.grade, strain, state, ez=ez_new)
        return stress, new_state, dWp, ez_new, C

    # -- assembly ------------------------------------------------------------
    def internal_force(self, ue, Baug, stress, shear):
        f_pt = stress * self.layer_weight[:, None]
        f = np.einsum("pij,pi->pj", Baug, f_pt).reshape(
            self.n_elem, self.npt_elem, 20).sum(axis=1)
        f += self.shear_force_element(ue)
        return f

    def assembled_internal_force(self, ue, Baug, stress, shear):
        return self.assemble_vector(self.internal_force(ue, Baug, stress, shear))

    def membrane_resultants(self, stress):
        """``N`` (3 components) at every Gauss point of every element."""
        N = np.einsum("p,pi->pi", self.layer_weight_nominal, stress)
        return N.reshape(self.n_elem, self.dNdx.shape[1], self.n_layers, 3).sum(axis=2)

    def geometric_stiffness(self, N_gp):
        """von Karman / initial-stress matrix on the ``w`` DOFs -> (M, 4, 4)."""
        Kg = np.zeros((self.n_elem, 4, 4))
        for q in range(self.dNdx.shape[1]):
            dnx, dny = self.dNdx[:, q], self.dNdy[:, q]
            nxx, nyy, nxy = N_gp[:, q, 0], N_gp[:, q, 1], N_gp[:, q, 2]
            w = self.detj[:, q]
            for i in range(4):
                for j in range(4):
                    Kg[:, i, j] += w * (nxx * dnx[:, i] * dnx[:, j]
                                        + nyy * dny[:, i] * dny[:, j]
                                        + nxy * (dnx[:, i] * dny[:, j]
                                                 + dny[:, i] * dnx[:, j]))
        return Kg

    def elastic_matrices(self):
        """Linear elastic (small-displacement) stiffness blocks for eigen work."""
        M = self.n_elem
        C = self.grade.plane_stress_matrix()
        Cb = np.broadcast_to(C, (self.n_points, 3, 3))
        tmp = np.einsum("pik,pkl->pil", Cb, self.B_layer)
        tmp *= self.layer_weight[:, None, None]
        Ke = np.einsum("pij,pil->pjl", self.B_layer, tmp).reshape(
            M, self.npt_elem, 20, 20).sum(axis=1)
        Ke += self.shear_stiffness_element()
        return Ke

    def assemble(self, Ke, Kgw=None, stiffener_terms=None):
        rows = [self._row20.ravel()]
        cols = [self._col20.ravel()]
        data = [Ke.ravel()]
        if Kgw is not None:
            rows.append(self._row4.ravel())
            cols.append(self._col4.ravel())
            data.append(Kgw.ravel())
        if stiffener_terms is not None:
            srows, scols, sdata = stiffener_terms
            if srows.size:
                rows.append(srows)
                cols.append(scols)
                data.append(sdata)
        K = sp.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof))
        return K.tocsr()

    def assemble_vector(self, felem):
        f = np.zeros(self.ndof)
        np.add.at(f, self.edof.ravel(), felem.ravel())
        return f

    def stiffener_terms(self, axial_forces=None, include_elastic: bool = True):
        if not self.stiffeners:
            return None
        forces = axial_forces if axial_forces is not None else [0.0] * len(self.stiffeners)
        r, c, d = [], [], []
        for stiff, force in zip(self.stiffeners, forces):
            tri = stiffener_triplets(self, stiff, axial_force=force,
                                     include_elastic=include_elastic)
            r.append(tri[0]); c.append(tri[1]); d.append(tri[2])
        return (np.concatenate(r), np.concatenate(c), np.concatenate(d))


# --------------------------------------------------------------------------- #
# boundary conditions and linear algebra helpers
# --------------------------------------------------------------------------- #
def node_dofs(model: PlateModel, node_id, component: int) -> int:
    return DOF_PER_NODE * int(node_id) + int(component)


def edge_nodes(model: PlateModel, side: str, tol: float = 1e-12):
    """Node ids on an edge: ``'x0' | 'x1' | 'y0' | 'y1'``."""
    x, y = model.nodes[:, 0], model.nodes[:, 1]
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if side == "x0":
        return np.where(np.abs(x - xmin) <= tol)[0]
    if side == "x1":
        return np.where(np.abs(x - xmax) <= tol)[0]
    if side == "y0":
        return np.where(np.abs(y - ymin) <= tol)[0]
    if side == "y1":
        return np.where(np.abs(y - ymax) <= tol)[0]
    raise ValueError("side must be one of x0, x1, y0, y1")


def scatter_load(model: PlateModel, nodal_values, component: int):
    """Assemble a nodal load vector from a ``{node: value}`` mapping."""
    f = np.zeros(model.ndof)
    for node, value in nodal_values.items():
        f[DOF_PER_NODE * int(node) + component] += float(value)
    return f


def uniform_edge_load(model: PlateModel, side: str, total_force: float, component=1):
    """Consistent nodal load for a uniformly distributed edge/wall force."""
    nodes = edge_nodes(model, side)
    coords = model.nodes[nodes]
    if side in ("y0", "y1"):
        span = coords[:, 0]
    else:
        span = coords[:, 1]
    order = np.argsort(span)
    nodes, span = nodes[order], span[order]
    values = {}
    if nodes.size == 1:
        values[int(nodes[0])] = total_force
        return scatter_load(model, values, component)
    for k in range(nodes.size - 1):
        dL = span[k + 1] - span[k]
        share = 0.5 * total_force * dL / (span[-1] - span[0])
        values[int(nodes[k])] = values.get(int(nodes[k]), 0.0) + share
        values[int(nodes[k + 1])] = values.get(int(nodes[k + 1]), 0.0) + share
    return scatter_load(model, values, component)


def patch_pressure_load(model: PlateModel, pressure: float, patch_area: float,
                        centre=None):
    """Uniform patch pressure on elements inside a centred rectangular patch.

    The patch area is distributed over the elements whose centroid lies inside
    the rectangle of area ``patch_area`` centred on the plate (aspect preserved),
    and the residual area mismatch is compensated by a single factor so that
    ``sum(f_w) == pressure * patch_area`` exactly.
    """
    if patch_area <= 0.0:
        raise ValueError("patch_area must be positive")
    centroid = model.nodes[model.elements].mean(axis=1)
    x, y = model.nodes[:, 0], model.nodes[:, 1]
    cx = 0.5 * (x.min() + x.max()) if centre is None else float(centre[0])
    cy = 0.5 * (y.min() + y.max()) if centre is None else float(centre[1])
    lx, ly = x.max() - x.min(), y.max() - y.min()
    if patch_area > lx * ly:
        raise ValueError("patch area exceeds the plate area")
    ratio = math.sqrt(patch_area / (lx * ly))
    hx, hy = 0.5 * ratio * lx, 0.5 * ratio * ly
    inside = ((np.abs(centroid[:, 0] - cx) <= hx + 1e-12)
              & (np.abs(centroid[:, 1] - cy) <= hy + 1e-12))
    if not np.any(inside):
        raise ValueError("patch area is smaller than one element")
    f = np.zeros(model.ndof)
    elem_f = np.zeros((model.n_elem, 20))
    cover = 0.0
    for e in np.where(inside)[0]:
        coords = model.nodes[model.elements[e]]
        area = 0.0
        N, dxi, deta = _shapes(0.0, 0.0)
        xq, yq = coords[:, 0], coords[:, 1]
        area = 0.25 * ((xq @ np.array([-1, 1, 1, -1]))) * ((yq @ np.array([-1, -1, 1, 1])))
        area = abs(area)
        cover += area
        for i in range(4):
            elem_f[e, DOF_PER_NODE * i + I_W] = pressure * area / 4.0
    f = model.assemble_vector(elem_f)
    total = f[I_W::DOF_PER_NODE].sum()
    target = pressure * patch_area
    if abs(total) > 0.0:
        f *= target / total
    return f


def free_dofs(ndof: int, fixed) -> np.ndarray:
    mask = np.ones(ndof, dtype=bool)
    mask[np.asarray(sorted(fixed), dtype=int)] = False
    return np.where(mask)[0]


# --------------------------------------------------------------------------- #
# nonlinear solution control
# --------------------------------------------------------------------------- #
@dataclass
class SolutionPath:
    lam: list = field(default_factory=list)
    u: list = field(default_factory=list)
    centre_w: list = field(default_factory=list)
    end_shortening: list = field(default_factory=list)
    axial_force: list = field(default_factory=list)
    converged: list = field(default_factory=list)
    iterations: list = field(default_factory=list)
    external_work: float = 0.0
    plastic_work: float = 0.0
    elastic_energy: float = 0.0
    state: PointState | None = None
    ez: np.ndarray | None = None
    trial_lam: float | None = None      # first load factor that failed to converge
    soft_convergences: int = 0          # states accepted on a stalled residual
    arc_stalled: bool = False           # path stopped on an arc-length stall

    def energy_residual(self) -> float:
        scale = max(abs(self.external_work), 1e-30)
        return (self.external_work - self.elastic_energy - self.plastic_work) / scale


def _centre_node(model: PlateModel) -> int:
    x, y = model.nodes[:, 0], model.nodes[:, 1]
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    return int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))


def _strain_energy(stress, elastic_strain, weight) -> float:
    return float(0.5 * np.einsum("pi,pi,p->", stress, elastic_strain, weight))


class _Increment:
    """Bookkeeping of one converged increment."""

    def __init__(self):
        self.state = None
        self.ez = None
        self.dWp = 0.0


def _point_plastic_strain(state: PointState) -> np.ndarray:
    """Plastic strain in the 3-component engineering convention."""
    out = np.empty((state.eps_p.shape[0], 3))
    out[:, 0] = state.eps_p[:, 0]
    out[:, 1] = state.eps_p[:, 1]
    out[:, 2] = 2.0 * state.eps_p[:, 3]
    return out


def strain_energy(model: PlateModel, strain, state: PointState, shear,
                  ue=None) -> float:
    """``int 0.5 sigma : eps_e dV`` (elastic part of the internal energy)."""
    stress, _, _, _, _ = model.stress_update(strain, state, None, need_tangent=False)
    elastic = strain - _point_plastic_strain(state)
    energy = 0.5 * np.einsum("pi,pi,p->", stress, elastic, model.layer_weight)
    if ue is not None:
        energy += model.shear_energy(ue)
    return float(energy)


def _finish_path(model: PlateModel, path: SolutionPath, fext) -> SolutionPath:
    """Fill the energy ledger from the stored history (no incremental drift)."""
    if not path.u:
        return path
    w_ext = 0.0
    lam_prev, u_prev = 0.0, np.zeros(model.ndof)
    for lam, u in zip(path.lam, path.u):
        w_ext += 0.5 * (lam + lam_prev) * float(fext @ (u - u_prev))
        lam_prev, u_prev = lam, u
    path.external_work = w_ext
    u = path.u[-1]
    ue, Baug, strain = model.layer_kinematics(u)
    path.elastic_energy = strain_energy(model, strain, path.state,
                                        model.shear_strain(ue), ue=ue)
    return path


def solve_load_control(model: PlateModel, fext, fixed, lam_steps,
                       tol: float = 1e-6, max_iter: int = 30,
                       max_backtracks: int = 6) -> SolutionPath:
    """Incremental load control with consistent-Newton iterations."""
    free = free_dofs(model.ndof, fixed)
    fext = np.array(fext, dtype=float)
    fext[np.asarray(sorted(fixed), dtype=int)] = 0.0
    u = np.zeros(model.ndof)
    state = PointState.zeros(model.n_points)
    ez = np.zeros(model.n_points)
    path = SolutionPath()
    centre = _centre_node(model)
    damped = 0.0

    for lam in lam_steps:
        u_step = u.copy()
        converged = False
        for it in range(max_iter):
            ue, Baug, strain = model.layer_kinematics(u)
            stress, new_state, dWp, ez_new, C = model.stress_update(
                strain, state, ez, need_tangent=True)
            shear = model.shear_strain(ue)
            fint = model.assembled_internal_force(ue, Baug, stress, shear)
            res = fint - lam * fext
            ref = max(np.linalg.norm(lam * fext[free]),
                      1e-8 * np.linalg.norm(fint[free]), 1e-30)
            if np.linalg.norm(res[free]) <= tol * ref:
                converged = True
                break
            N = model.membrane_resultants(stress)
            tmp = np.einsum("pik,pkl->pil", C, Baug) * model.layer_weight[:, None, None]
            Ke = np.einsum("pij,pil->pjl", Baug, tmp).reshape(
                model.n_elem, model.npt_elem, 20, 20).sum(axis=1)
            Ke += model.shear_stiffness_element()
            Kgw = model.geometric_stiffness(N) if model.von_karman else None
            forces = stiffener_axial_forces(model, N)
            K = model.assemble(Ke, Kgw, model.stiffener_terms(forces))
            lu = spla.splu(K[free][:, free].tocsc())
            du = lu.solve(-res[free])
            u[free] = u[free] + du
        if not converged:
            u = u_step
            break
        state, ez = new_state, ez_new
        path.lam.append(float(lam))
        path.u.append(u.copy())
        path.centre_w.append(float(u[DOF_PER_NODE * centre + I_W]))
        path.converged.append(True)
        path.iterations.append(it + 1)
        path.plastic_work += float(np.sum(dWp * model.layer_weight))
    path.state = state
    path.ez = ez
    path.damped_iterations = damped
    return _finish_path(model, path, fext)


def solve_displacement_control(model: PlateModel, fixed, prescribed, steps,
                               tol: float = 1e-6, max_iter: int = 30,
                               max_backtracks: int = 6) -> SolutionPath:
    """Displacement control: ``prescribed`` maps dof -> list of step values."""
    prescribed = {int(k): np.asarray(v, dtype=float) for k, v in prescribed.items()}
    n_steps = len(steps)
    fixed = list(fixed) + list(prescribed.keys())
    free = free_dofs(model.ndof, fixed)
    pdofs = np.array(sorted(prescribed.keys()), dtype=int)
    u = np.zeros(model.ndof)
    state = PointState.zeros(model.n_points)
    ez = np.zeros(model.n_points)
    path = SolutionPath()
    centre = _centre_node(model)
    damped = 0.0

    for s in range(n_steps):
        target = u.copy()
        for dof, values in prescribed.items():
            target[dof] = values[min(s, values.size - 1)]
        u[pdofs] = target[pdofs]
        converged = False
        for it in range(max_iter):
            ue, Baug, strain = model.layer_kinematics(u)
            stress, new_state, dWp, ez_new, C = model.stress_update(
                strain, state, ez, need_tangent=True)
            shear = model.shear_strain(ue)
            fint = model.assembled_internal_force(ue, Baug, stress, shear)
            ref = max(np.linalg.norm(fint[pdofs]), 1e-8 * np.linalg.norm(fint[free]), 1e-30)
            if np.linalg.norm(fint[free]) <= tol * ref:
                converged = True
                break
            N = model.membrane_resultants(stress)
            tmp = np.einsum("pik,pkl->pil", C, Baug) * model.layer_weight[:, None, None]
            Ke = np.einsum("pij,pil->pjl", Baug, tmp).reshape(
                model.n_elem, model.npt_elem, 20, 20).sum(axis=1)
            Ke += model.shear_stiffness_element()
            Kgw = model.geometric_stiffness(N) if model.von_karman else None
            forces = stiffener_axial_forces(model, N)
            K = model.assemble(Ke, Kgw, model.stiffener_terms(forces))
            Kff = K[free][:, free].tocsc()
            lu = spla.splu(Kff)
            du = lu.solve(-fint[free])
            u[free] = u[free] + du
        if not converged:
            break
        state, ez = new_state, ez_new
        path.lam.append(float(steps[s]))
        path.u.append(u.copy())
        path.centre_w.append(float(u[DOF_PER_NODE * centre + I_W]))
        path.axial_force.append(float(fint[pdofs].sum()))
        path.converged.append(True)
        path.iterations.append(it + 1)
        path.plastic_work += float(np.sum(dWp * model.layer_weight))
    path.state = state
    path.ez = ez
    path.damped_iterations = damped
    path.fext = None
    if path.u:
        ue, Baug, strain = model.layer_kinematics(path.u[-1])
        path.elastic_energy = strain_energy(model, strain, state,
                                            model.shear_strain(ue), ue=ue)
    return path


def solve_arc_length(model: PlateModel, fext, fixed, *, n_steps: int = 80,
                     dlam0: float = 0.02, dlam_max: float = 0.08,
                     tol: float = 1e-6, max_iter: int = 30,
                     growth: float = 1.15, shrink: float = 0.5,
                     max_retry: int = 10, stop_ratio: float | None = 0.75,
                     verbose: bool = False) -> SolutionPath:
    """Modified Riks arc-length continuation of ``fint(u) - lam * fext = 0``.

    Constraint (spherical, load-factor weighted, ``kappa = |du_f|**2``)::

        |u - u_step|**2 + kappa (lam - lam_step)**2 = dl**2

    where ``du_f = K**-1 fext`` is the tangential load direction, so the two
    terms are both measured in displacement units and the step size ``dl`` is
    dimensionless-consistent.  ``dlam0`` sets the first step (its load-factor
    part) and the step is adapted from the Newton iteration count.

    The corrector solves the quadratic arc-length constraint exactly and picks
    the root that (i) keeps the increment on the same side as the previous
    *converged* increment (continuation guard) and (ii) is closest to the
    tangential prediction -- the standard Crisfield root selection, which is
    what makes a limit point reachable without knowing it in advance.

    Scope, stated honestly: this continuation locates and brackets the
    **limit load** (the peak of the equilibrium path).  On the ice-belt and
    plate-field paths used here, steps past the peak lose the tangent
    (``K_T`` is the symmetrised ``K_L ~ K_sigma`` form) and the step size
    collapses, so the *post-peak branch is not traced*; the returned path ends
    at the last converged pre-peak state and the trial load above it is the
    bracket.  ``path.soft_convergences`` counts states accepted only because the
    residual stalled (always zero for the runs in this module).
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if dlam0 <= 0.0:
        raise ValueError("dlam0 must be positive")
    if dlam_max < dlam0:
        raise ValueError("dlam_max must not be smaller than dlam0")
    fixed = np.asarray(sorted(set(int(d) for d in fixed)), dtype=int)
    free = free_dofs(model.ndof, fixed)
    fext = np.array(fext, dtype=float)
    fext[fixed] = 0.0
    u = np.zeros(model.ndof)
    state = PointState.zeros(model.n_points)
    ez = np.zeros(model.n_points)
    path = SolutionPath()
    path.lam.append(0.0)
    path.u.append(u.copy())
    path.soft_convergences = 0
    lam = 0.0

    def evaluate(un):
        ue, Baug, strain = model.layer_kinematics(un)
        stress, new_state, dWp, ez_new, C = model.stress_update(
            strain, state, ez, need_tangent=True)
        fint = model.assembled_internal_force(ue, Baug, stress,
                                              model.shear_strain(ue))
        N = model.membrane_resultants(stress)
        tmp = np.einsum("pik,pkl->pil", C, Baug) * model.layer_weight[:, None, None]
        Ke = np.einsum("pij,pil->pjl", Baug, tmp).reshape(
            model.n_elem, model.npt_elem, 20, 20).sum(axis=1)
        Ke += model.shear_stiffness_element()
        Kgw = model.geometric_stiffness(N) if model.von_karman else None
        K = model.assemble(Ke, Kgw, model.stiffener_terms(
            stiffener_axial_forces(model, N)))
        return fint, K, new_state, dWp, ez_new

    fint, K, _, _, _ = evaluate(u)
    lu = spla.splu(K[free][:, free].tocsc())
    du_f = np.zeros(model.ndof)
    du_f[free] = lu.solve(fext[free])
    kappa = float(du_f @ du_f)
    if kappa <= 0.0:
        raise ValueError("reference load has no component in the free subspace")
    dl = dlam0 * math.sqrt(2.0 * kappa)      # first step: |du|^2 + kappa dlam^2
    dlam_prev = dlam0
    best, since_best = 0.0, 0
    path.trial_lam = None
    retries = 0
    for _ in range(n_steps):
        u_step, lam_step = u.copy(), lam
        du_prev = path.u[-1] - path.u[-2] if len(path.u) > 1 else None
        dlam_pred = math.copysign(dl / math.sqrt(2.0 * kappa), dlam_prev)
        if abs(dlam_pred) > dlam_max:       # keep the peak from being stepped over
            dlam_pred = math.copysign(dlam_max, dlam_pred)
            dl = abs(dlam_pred) * math.sqrt(2.0 * kappa)
        u = u + dlam_pred * du_f
        lam = lam + dlam_pred
        converged, iters, last = False, 0, dlam_pred
        for it in range(max_iter):
            fint, K, new_state, dWp, ez_new = evaluate(u)
            res = fint - lam * fext
            ref = max(np.linalg.norm(lam * fext[free]),
                      1e-8 * np.linalg.norm(fint[free]), 1e-30)
            if np.linalg.norm(res[free]) <= tol * ref:
                converged = True
                break
            lu = spla.splu(K[free][:, free].tocsc())
            du_f = np.zeros(model.ndof)
            du_f[free] = lu.solve(fext[free])
            du_r = np.zeros(model.ndof)
            du_r[free] = lu.solve(-res[free])
            c = u - u_step
            dlam_c = lam - lam_step
            a = float(du_f @ du_f) + kappa
            b = 2.0 * (float((c + du_r) @ du_f) + kappa * dlam_c)
            cc = float((c + du_r) @ (c + du_r)) + kappa * dlam_c ** 2 - dl * dl
            disc = b * b - 4.0 * a * cc
            if disc < 0.0:
                break                       # no intersection: step too large
            root = math.sqrt(disc)
            roots = ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))
            target = (-(float(c @ du_f) + kappa * dlam_c) / a if it == 0 else last)
            candidates = roots
            if du_prev is None:
                # First step: there is no converged increment to guard against, and on
                # the symmetric zero-displacement start (du_r == 0) the two roots are
                # d = 0 and d = -2*dlam_pred, exactly equidistant from ``target``, so
                # the tie can be broken onto the mirror branch: the path then descends
                # at negative load, keeps converging (the elastic response is odd in
                # (u, lam)), and every limit load comes out negative.  Restrict the
                # choice to roots that keep the load increment in the predicted
                # direction (Crisfield's sign rule, applied to lam instead of u).
                forward = [d for d in roots
                           if float(dlam_c + d) * float(dlam_pred) > 0.0]
                if forward:
                    candidates = tuple(forward)
            dsel = min(candidates, key=lambda d: abs(d - target))
            if du_prev is not None:
                ahead = [d for d in candidates
                         if float((c + du_r + d * du_f) @ du_prev) > 0.0]
                if ahead:
                    dsel = min(ahead, key=lambda d: abs(d - target))
            u = u + du_r + dsel * du_f
            lam = lam + dsel
            last, iters = dsel, it + 1
        if not converged:
            u, lam = u_step, lam_step
            dl *= shrink
            retries += 1
            if retries >= max_retry:
                path.trial_lam = float(lam + dlam_pred)
                break
            continue
        retries = 0
        state, ez = new_state, ez_new
        path.lam.append(float(lam))
        path.u.append(u.copy())
        path.centre_w.append(float(u[DOF_PER_NODE * _centre_node(model) + I_W]))
        path.converged.append(True)
        path.iterations.append(iters + 1)
        path.plastic_work += float(np.sum(dWp * model.layer_weight))
        dlam_prev = lam - lam_step
        if verbose:
            print(f"step {len(path.lam) - 2:3d} lam={lam:8.4f} "
                  f"dlam={dlam_prev:+.4f} dl={dl:.2e} it={iters + 1}")
        if iters <= 3:
            dl *= growth
        elif iters >= 12:
            dl *= shrink
        if stop_ratio is not None and len(path.lam) > 6 \
                and lam < stop_ratio * max(path.lam[1:]):
            break
        if lam > best * (1.0 + 1.0e-4):
            best, since_best = lam, 0
        else:
            since_best += 1
        if len(path.lam) > 12 and since_best >= 12 \
                and abs(dlam_prev) < 0.05 * dlam0:
            # asymptotic approach: the last 12 steps bought < 0.01 % of load
            break

    path.state = state
    path.ez = ez
    # |dlam| collapsing towards zero is the signature of an asymptotic approach
    # to a limit point: the last state is a lower bound, not a bracketed peak.
    path.arc_stalled = bool(len(path.lam) > 6 and since_best >= 12)
    return _finish_path(model, path, fext)


def arc_length_status(limit: dict) -> str:
    """Qualitative state of an arc-length peak: bracketed / stalled / ascending."""
    if limit.get("bracket") is not None:
        return "bracketed"
    if limit.get("arc_stalled"):
        return "stalled-approaching-limit"
    return "still-ascending"


def arc_length_limit(path: SolutionPath) -> dict:
    """Peak load factor of an arc-length path and the pre-peak bracket."""
    if len(path.lam) < 2:
        raise ValueError("arc-length path carries no converged step")
    lams = np.asarray(path.lam[1:], dtype=float)
    peak = float(lams.max())
    top = int(np.argmax(lams))
    return {"peak_load_factor": peak,
            "peak_step": top + 1,
            "steps_converged": int(lams.size),
            "post_peak_steps": int(lams.size - top - 1),
            "trial_lam": path.trial_lam,
            "bracket": (peak, path.trial_lam) if path.trial_lam is not None
            else None,
            "monotone": bool(np.all(np.diff(lams) >= 0.0)),
            "arc_stalled": bool(getattr(path, "arc_stalled", False))}


def plate_field_collapse_arc_case(span_x: float = 1.6, width: float = 0.8,
                                  thickness: float = 0.018, nx: int = 16,
                                  ny: int = 8, imperfection_ratio=None,
                                  grade: SteelGrade | None = None,
                                  dlam0: float = 0.02,
                                  n_steps: int = 60) -> dict:
    """C8: plate-field ultimate under end compression by Riks continuation.

    The load is a uniform compressive traction on ``x = span_x`` with the
    reference level ``sigma_0`` (so ``lam`` is directly ``sigma/sigma_0``), the
    loaded edges are held ``w = 0`` (frame lines) and the imperfection is the
    first elastic buckling mode scaled to ``0.1 beta**2 t`` -- the Faulkner
    reference state, i.e. the same state the Smith curve is anchored on.
    """
    grade = elastic_perfectly_plastic() if grade is None else grade
    beta = width / thickness * math.sqrt(grade.sigma_0 / grade.young)
    if imperfection_ratio is None:
        imperfection_ratio = 0.1 * beta ** 2
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)
    fixed = simply_supported_edges(model)
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] ** 2 + model.nodes[:, 1] ** 2))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))
    _, _, mode = buckling_eigenvalue(model, 1.0e6, fixed, direction="x",
                                     return_modes=True)
    shape = mode[I_W::DOF_PER_NODE]
    shape = shape / np.max(np.abs(shape))
    model.set_imperfection(shape * imperfection_ratio * thickness)
    fext = uniform_edge_load(model, "x1", -grade.sigma_0 * width * thickness,
                             component=I_U)
    path = solve_arc_length(model, fext, fixed, n_steps=n_steps, dlam0=dlam0)
    limit = arc_length_limit(path)
    from .smith_progressive_collapse import PlateStrip
    strip = PlateStrip(length=span_x, width=width, thickness=thickness,
                       young=grade.young, poisson=grade.poisson,
                       yield_stress=grade.sigma_0,
                       imperfection_ratio=imperfection_ratio,
                       residual_ratio=0.0)
    sigma_fem = limit["peak_load_factor"] * grade.sigma_0
    sigma_smith = float(strip.ultimate_stress)
    from .plate_stiffener_ultimate import paik_thayamballi_plate_ultimate
    sigma_paik = paik_thayamballi_plate_ultimate(
        beta=beta, sigma_y=grade.sigma_0, a_over_b=span_x / width)
    out = {"beta": beta, "imperfection_ratio": imperfection_ratio,
           "w0": imperfection_ratio * thickness, "mesh": [nx, ny],
           "span": [span_x, width], "thickness": thickness,
           "sigma_fem_peak": sigma_fem, "sigma_smith": sigma_smith,
           "sigma_paik": float(sigma_paik),
           "rel_error": sigma_fem / sigma_smith - 1.0,
           "rel_error_paik": sigma_fem / float(sigma_paik) - 1.0,
           "reference_spread": float(sigma_smith / sigma_paik - 1.0),
           "rel_error_elastic_plastic_probe": None,
           "energy_residual": path.energy_residual(),
           "plastic_work": path.plastic_work,
           "elastic_energy": path.elastic_energy,
           "external_work": path.external_work,
           "mean_iterations": float(np.mean(path.iterations)),
           "peak_deflection": path.centre_w[int(np.argmax(path.lam[1:]))],
           "path": {"lam": list(path.lam[1:]), "w": list(path.centre_w)},
           "limit": limit, "limit_status": arc_length_status(limit)}
    return out


def ice_belt_limit_pressure_arc_case(nx: int = 24, ny: int = 8,
                                     span_x: float = 3.0, width: float = 0.5,
                                     thickness: float = 0.016,
                                     p0: float = 2.0e6, area_ref: float = 1.0,
                                     dlam0: float = 0.03,
                                     n_steps: int = 45) -> dict:
    """C9: ice-belt limit pressure by Riks continuation past load control.

    Same bay, load and material as :func:`ice_belt_case`; the reference load is
    the IACS pressure at the bay area, so ``lam * pressure_at_area`` is the
    uniform patch pressure at collapse.
    """
    ice = IACSPressureArea(p0=p0, area_ref=area_ref)
    load_area = span_x * width
    pressure = float(ice.pressure(load_area))
    grade = elastic_perfectly_plastic()
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)
    model.von_karman = False
    unit = uniform_pressure_load(model, pressure)
    fixed = inplane_free_bcs(model)
    path = solve_arc_length(model, unit, fixed, n_steps=n_steps, dlam0=dlam0)
    limit = arc_length_limit(path)
    mp = grade.sigma_0 * thickness ** 2 / 4.0
    p_code = 8.0 * mp / width ** 2
    collapse = limit["peak_load_factor"] * pressure
    out = {"pressure_at_area": pressure, "load_area": load_area,
           "limit_pressure": collapse, "code_pressure": p_code,
           "code_basis": "strip yield-line 8*Mp/b^2 with Mp=sigma_0*t^2/4",
           "rel_error": collapse / p_code - 1.0,
           "mesh": [nx, ny], "thickness": thickness,
           "energy_residual": path.energy_residual(),
           "plastic_work": path.plastic_work,
           "limit": limit, "limit_status": arc_length_status(limit),
           "path": {"lam": list(path.lam[1:]), "w": list(path.centre_w)},
           "provenance": ice.provenance}
    return out


def stiffener_axial_forces(model: PlateModel, N_gp):
    """Axial force in every stiffener from the membrane resultants."""
    if not model.stiffeners:
        return []
    centroid = model.nodes[model.elements].mean(axis=1)
    out = []
    for stiff in model.stiffeners:
        line_xy = model.nodes[np.asarray(stiff.node_line, dtype=int)].mean(axis=0)
        if stiff.direction == "x":
            sel = np.abs(centroid[:, 1] - line_xy[1]) < 1e-9
            comp = 0
        else:
            sel = np.abs(centroid[:, 0] - line_xy[0]) < 1e-9
            comp = 1
        if not np.any(sel):
            out.append(0.0)
            continue
        n_mean = float(N_gp[sel, :, comp].mean())
        out.append(n_mean * stiff.area / model.thickness)
    return out


def buckling_eigenvalue(model: PlateModel, sigma_ref: float, fixed,
                        direction: str = "x", n_modes: int = 4,
                        return_modes: bool = False):
    """Smallest buckling multiplier for a uniform membrane stress ``sigma_ref``.

    ``sigma_ref > 0`` means compression.  ``K_sigma`` uses ``N = -sigma_ref * t``.
    """
    free = free_dofs(model.ndof, fixed)
    ue = np.zeros((model.n_elem, 20))
    Baug = model.B_layer.reshape(model.n_elem, model.npt_elem, 3, 20)[:, 0] \
        if False else None
    C = model.grade.plane_stress_matrix()
    Cfull = np.broadcast_to(C, (model.n_points, 3, 3))
    tmp = np.einsum("pik,pkl->pil", Cfull, model.B_layer) * model.layer_weight[:, None, None]
    Ke = np.einsum("pij,pil->pjl", model.B_layer, tmp).reshape(
        model.n_elem, model.npt_elem, 20, 20).sum(axis=1)
    Ke += model.shear_stiffness_element()
    stress_ref = np.zeros((model.n_points, 3))
    if direction == "x":
        stress_ref[:, 0] = -sigma_ref
    elif direction == "y":
        stress_ref[:, 1] = -sigma_ref
    else:
        raise ValueError("direction must be 'x' or 'y'")
    N_gp = model.membrane_resultants(stress_ref)
    Kgw = model.geometric_stiffness(N_gp)
    forces = [(-sigma_ref * stiff.area) if (stiff.direction == direction) else 0.0
              for stiff in model.stiffeners]
    K0 = model.assemble(Ke, None, model.stiffener_terms(None))[free][:, free].tocsc()
    Kg = model.assemble(np.zeros((model.n_elem, 20, 20)), Kgw,
                        model.stiffener_terms(forces, include_elastic=False)
                        )[free][:, free].tocsc()
    lu = spla.splu(K0)

    def matvec(v):
        return lu.solve(-(Kg @ v))

    op = spla.LinearOperator((free.size, free.size), matvec=matvec, dtype=float)
    k = min(n_modes, free.size - 2)
    vals, vecs = spla.eigs(op, k=k, which="LM",
                           return_eigenvectors=True)
    keep = np.abs(vals) > 0
    vals, vecs = vals[keep], vecs[:, keep]
    lam = 1.0 / np.real(vals)
    order = np.argsort(lam)
    lam_sorted = lam[order]
    modes = np.real(vecs[:, order])
    positive = lam_sorted > 0
    if not np.any(positive):
        raise RuntimeError("no positive buckling multiplier found")
    first = float(lam_sorted[positive][0])
    if return_modes:
        mode = np.zeros(model.ndof)
        mode[free] = modes[:, np.where(positive)[0][0]]
        return first, lam_sorted, mode
    return first, lam_sorted


# --------------------------------------------------------------------------- #
# reference solutions (independent of the FE)
# --------------------------------------------------------------------------- #
def navier_ssss_centre(span_x: float, span_y: float, thickness: float, young: float,
                       poisson: float, pressure: float, terms: int = 199,
                       position=None) -> float:
    """Kirchhoff Navier deflection of a simply supported plate (default centre)."""
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    x, y = (0.5 * span_x, 0.5 * span_y) if position is None else position
    odd = np.arange(1, int(terms) + 1, 2, dtype=float)
    mm, nn = np.meshgrid(odd, odd, indexing="ij")
    sx = np.sin(mm * math.pi * x / span_x)
    sy = np.sin(nn * math.pi * y / span_y)
    total = float(np.sum(sx * sy / (mm * nn * (mm ** 2 / span_x ** 2
                                               + nn ** 2 / span_y ** 2) ** 2)))
    return 16.0 * pressure / (math.pi ** 6 * rigidity) * total


def csr_plate_buckling_stress(span_x: float, width: float, thickness: float,
                              young: float, poisson: float,
                              buckling_coefficient: float | None = None) -> float:
    """CSR/rule elastic plate buckling stress; ``k(a/b)`` exact when not given."""
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    if buckling_coefficient is None:
        alpha = span_x / width
        k = min((m / alpha + alpha / m) ** 2 for m in range(1, 40))
    else:
        k = float(buckling_coefficient)
    return k * math.pi ** 2 * rigidity / (width ** 2 * thickness)


def euler_column_stress(panel_length: float, plate_width: float, plate_thickness: float,
                        stiffener_area: float, stiffener_inertia: float,
                        stiffener_offset: float, young: float) -> dict:
    """Rule-format column stress of a stiffener with attached plating."""
    a_p = plate_width * plate_thickness
    a_tot = a_p + stiffener_area
    ybar = stiffener_area * stiffener_offset / a_tot
    i_tot = (stiffener_inertia + stiffener_area * (stiffener_offset - ybar) ** 2
             + plate_width * plate_thickness ** 3 / 12.0 + a_p * ybar ** 2)
    sigma_e = math.pi ** 2 * young * i_tot / (a_tot * panel_length ** 2)
    return {"area_total": a_tot, "inertia_total": i_tot, "ybar": ybar,
            "second_moment": i_tot, "euler_stress": sigma_e}


def analytic_moment_curvature(grade: SteelGrade, kappa: float, thickness: float,
                              slices: int = 4001) -> float:
    """Moment of a rectangular section from the 1-D hardening law (no FE)."""
    half = 0.5 * thickness
    z = np.linspace(0.0, half, slices)
    sigma = grade.uniaxial_stress(kappa * z)
    return float(abs(2.0 * np.trapezoid(sigma * z, z)))


def fenves_twice_elastic_slope(loads, displacements,
                               elastic_slope: float | None = None) -> float | None:
    """ASME twice-elastic-slope limit load (a documented code procedure).

    ``elastic_slope`` should be the slope of the purely elastic response; when
    omitted the first secant is used (only valid if the first step is elastic).
    """
    q = np.asarray(loads, dtype=float)
    d = np.asarray(displacements, dtype=float)
    if q.size < 3:
        return None
    slope = elastic_slope if elastic_slope is not None else (
        q[0] / d[0] if d[0] != 0.0 else None)
    if slope is None or not np.isfinite(slope):
        return None
    line = slope / 2.0 * d
    diff = q - line
    for k in range(1, q.size):
        if diff[k - 1] > 0.0 and diff[k] <= 0.0:
            t = diff[k - 1] / (diff[k - 1] - diff[k])
            return float(q[k - 1] + t * (q[k] - q[k - 1]))
    return None


# --------------------------------------------------------------------------- #
# load helpers
# --------------------------------------------------------------------------- #
def element_areas(model: PlateModel) -> np.ndarray:
    coords = model.nodes[model.elements]
    x, y = coords[:, :, 0], coords[:, :, 1]
    return 0.5 * np.abs(np.einsum("mi,mi->m", x, np.roll(y, -1, axis=1))
                        - np.einsum("mi,mi->m", y, np.roll(x, -1, axis=1)))


def uniform_pressure_load(model: PlateModel, pressure: float) -> np.ndarray:
    """Consistent (lumped) transverse load for a uniform pressure."""
    area = element_areas(model)
    f = np.zeros(model.ndof)
    for e in range(model.n_elem):
        for i in range(4):
            f[model.edof[e, DOF_PER_NODE * i + I_W]] += pressure * area[e] / 4.0
    return f


def section_moment(model: PlateModel, u, state: PointState, ez, element: int) -> float:
    """Bending moment resultant ``int sigma_x z dz`` (per unit width) of one element."""
    ue, Baug, strain = model.layer_kinematics(u)
    stress, _, _, _, _ = model.stress_update(strain, state, ez, need_tangent=False)
    base = element * model.npt_elem
    pts = np.arange(base, base + model.npt_elem)
    gauss_index = (pts - base) // model.n_layers
    n_inplane = model.npt_elem // model.n_layers
    wz = model.layer_weight[pts] / model.detj[element, gauss_index]
    return float(np.sum(stress[pts, 0] * model.layer_z[pts] * wz) / n_inplane)


def simply_supported_edges(model: PlateModel):
    """``w = 0`` on all four edges (simply supported plate)."""
    dofs = []
    for side in ("x0", "x1", "y0", "y1"):
        dofs += [DOF_PER_NODE * int(n) + I_W for n in edge_nodes(model, side)]
    return sorted(set(dofs))


def inplane_free_bcs(model: PlateModel, fixed_w_edges=True, corner=(0, 0)):
    """Simply supported out-of-plane, in-plane traction free (no membrane action)."""
    fixed = simply_supported_edges(model) if fixed_w_edges else []
    node = int(np.argmin((model.nodes[:, 0] - model.nodes[corner[0], 0]) ** 2
                         + (model.nodes[:, 1] - model.nodes[corner[1], 1]) ** 2))
    x, y = model.nodes[:, 0], model.nodes[:, 1]
    other = int(np.argmin(np.abs(x - x.max()) + np.abs(y - y.min())))
    fixed += [DOF_PER_NODE * node + I_U, DOF_PER_NODE * node + I_V,
              DOF_PER_NODE * other + I_V]
    return sorted(set(fixed))


# --------------------------------------------------------------------------- #
# section helpers
# --------------------------------------------------------------------------- #
def tee_section(web_height: float, web_thickness: float, flange_width: float,
                flange_thickness: float, plate_thickness: float) -> dict:
    """Tee stiffener below a plate: area, own inertia and centroid offset."""
    a_web = web_height * web_thickness
    a_flange = flange_width * flange_thickness
    area = a_web + a_flange
    z_web = -(0.5 * plate_thickness + 0.5 * web_height)
    z_flange = -(0.5 * plate_thickness + web_height + 0.5 * flange_thickness)
    centroid = (a_web * z_web + a_flange * z_flange) / area
    i_own = (web_thickness * web_height ** 3 / 12.0 + a_web * (z_web - centroid) ** 2
             + flange_width * flange_thickness ** 3 / 12.0
             + a_flange * (z_flange - centroid) ** 2)
    return {"area": area, "inertia": i_own, "offset": -centroid,
            "web_area": a_web, "flange_area": a_flange}


def line_nodes(model: PlateModel, direction: str, coordinate: float, tol=1e-9):
    """Ordered node ids on a grid line."""
    if direction == "x":
        sel = np.abs(model.nodes[:, 1] - coordinate) <= tol
        order = np.argsort(model.nodes[sel, 0])
    elif direction == "y":
        sel = np.abs(model.nodes[:, 0] - coordinate) <= tol
        order = np.argsort(model.nodes[sel, 1])
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return np.where(sel)[0][order]


# --------------------------------------------------------------------------- #
# verification cases
# --------------------------------------------------------------------------- #
def elastic_plate_case(n: int = 16, span_x: float = 1.0, span_y: float = 0.8,
                       thickness: float = 0.005, pressure: float = 1.0e4) -> dict:
    """C1: elastic SSSS plate centre deflection against the Navier series."""
    nodes, elems = structured_mesh(span_x, span_y, n, max(1, int(round(n * span_y / span_x))))
    model = PlateModel(nodes, elems, thickness, AH36)
    model.von_karman = False
    load = uniform_pressure_load(model, pressure)
    path = solve_load_control(model, load, inplane_free_bcs(model), [1.0],
                              tol=1e-9, max_iter=25)
    reference = navier_ssss_centre(span_x, span_y, thickness, E_STEEL, NU_STEEL, pressure)
    w_fe = path.centre_w[-1]
    return {"w_fe": w_fe, "w_navier": reference,
            "rel_error": w_fe / reference - 1.0,
            "mesh": [n, max(1, int(round(n * span_y / span_x)))],
            "thickness": thickness, "span": [span_x, span_y],
            "iterations": path.iterations}


def plastic_hinge_case(ratios=(0.5, 1.0, 2.0), layers: int = 5) -> dict:
    """C2: moment-curvature of a strip in cylindrical bending vs the 1-D law."""
    length, width, thickness = 0.2, 0.05, 0.1
    nodes, elems = structured_mesh(length, width, 8, 4)
    x, y = nodes[:, 0], nodes[:, 1]
    lower = int(np.argmin(x + y))
    upper = int(np.argmin(x + (y.max() - y)))
    fixed = [DOF_PER_NODE * n + I_U for n in range(len(nodes))]
    fixed += [DOF_PER_NODE * lower + I_W, DOF_PER_NODE * lower + I_V,
              DOF_PER_NODE * upper + I_W]
    rows = []
    for grade in (elastic_perfectly_plastic(), AH36):
        model = PlateModel(nodes, elems, thickness, grade, n_layers=layers)
        model.von_karman = False
        kappa_y = 2.0 * grade.sigma_0 / (grade.young * thickness)
        for ratio in ratios:
            kappa = ratio * kappa_y
            theta = 0.5 * kappa * length
            steps = np.linspace(0.0, theta, 21)[1:]
            prescribed = {}
            for n in edge_nodes(model, "x0"):
                prescribed[DOF_PER_NODE * int(n) + I_RX] = steps
            for n in edge_nodes(model, "x1"):
                prescribed[DOF_PER_NODE * int(n) + I_RX] = -steps
            path = solve_displacement_control(model, fixed, prescribed, steps,
                                              tol=1e-9, max_iter=20)
            converged = len(path.u) == len(steps)
            row = {"grade": grade.name, "kappa_ratio": ratio, "converged": converged}
            if converged:
                moment = abs(section_moment(model, path.u[-1], path.state,
                                            path.ez, 4))
                reference = analytic_moment_curvature(grade, kappa, thickness)
                row.update({"moment_fe": moment, "moment_1d": reference,
                            "rel_error": moment / reference - 1.0})
            rows.append(row)
    errors = [abs(r["rel_error"]) for r in rows if r["converged"]]
    return {"rows": rows, "max_abs_error": max(errors) if errors else None,
            "full_plastic_moment": elastic_perfectly_plastic().sigma_0 * thickness ** 2 / 4.0,
            "all_converged": all(r["converged"] for r in rows)}


def plate_buckling_case(span_x: float = 1.2, width: float = 0.8,
                        thickness: float = 0.01, nx: int = 24, ny: int = 16) -> dict:
    """C3: elastic plate-field buckling against the CSR ``k(a/b)`` formula."""
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, AH36)
    model.von_karman = False
    fixed = simply_supported_edges(model)
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] ** 2 + model.nodes[:, 1] ** 2))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))
    sigma_ref = 1.0e6
    multiplier, _ = buckling_eigenvalue(model, sigma_ref, fixed, direction="x")
    sigma_fe = multiplier * sigma_ref
    sigma_csr = csr_plate_buckling_stress(span_x, width, thickness, E_STEEL, NU_STEEL)
    return {"sigma_fe": sigma_fe, "sigma_code": sigma_csr,
            "rel_error": sigma_fe / sigma_csr - 1.0, "mesh": [nx, ny],
            "thickness": thickness, "span": [span_x, width]}


def stiffened_panel_case(web_height: float = 0.25, web_thickness: float = 0.010,
                         flange_width: float = 0.10, flange_thickness: float = 0.015,
                         panel_length: float = 8.0, width: float = 0.8,
                         thickness: float = 0.03, nx: int = 40, ny: int = 6) -> dict:
    """C4 side (a): stiffener + attached plating column against the rule Euler."""
    section = tee_section(web_height, web_thickness, flange_width,
                          flange_thickness, thickness)
    nodes, elems = structured_mesh(panel_length, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, AH36)
    model.von_karman = False
    model.stiffeners.append(Stiffener(
        node_line=line_nodes(model, "x", 0.0), direction="x",
        area=section["area"], inertia=section["inertia"],
        offset=section["offset"], name="tee"))
    fixed = [DOF_PER_NODE * int(n) + I_W for n in edge_nodes(model, "x0")]
    fixed += [DOF_PER_NODE * int(n) + I_W for n in edge_nodes(model, "x1")]
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] + model.nodes[:, 1]))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))
    multiplier, _ = buckling_eigenvalue(model, 1.0e6, fixed, direction="x")
    column = euler_column_stress(panel_length, width, thickness, section["area"],
                                 section["inertia"], section["offset"], E_STEEL)
    return {"sigma_fe": multiplier * 1.0e6, "sigma_column": column["euler_stress"],
            "rel_error": multiplier * 1.0e6 / column["euler_stress"] - 1.0,
            "converged": bool(multiplier > 0.0), "section": section,
            "inertia_total": column["inertia_total"], "mesh": [nx, ny]}


def ice_belt_case(nx: int = 24, ny: int = 8, span_x: float = 3.0,
                  width: float = 0.5, thickness: float = 0.016,
                  p0: float = 2.0e6, area_ref: float = 1.0,
                  steps: int = 24, layer_material=None,
                  von_karman: bool = False) -> dict:
    """C5: ice-belt plate field collapsing under the IACS pressure-area load."""
    ice = IACSPressureArea(p0=p0, area_ref=area_ref)
    load_area = span_x * width
    pressure = float(ice.pressure(load_area))
    grade = elastic_perfectly_plastic() if layer_material is None else layer_material
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)
    model.von_karman = bool(von_karman)
    unit = uniform_pressure_load(model, pressure)
    lam = np.linspace(0.0, 1.0, steps + 1)[1:]
    fixed = inplane_free_bcs(model)
    path = solve_load_control(model, unit, fixed, lam, tol=1e-7, max_iter=30)
    # purely elastic reference slope (below first yield) for the TES construction
    elastic = PlateModel(nodes, elems, thickness, AH36, n_layers=model.n_layers)
    elastic.von_karman = model.von_karman
    probe = solve_load_control(elastic, unit, fixed, [0.05], tol=1e-9, max_iter=10)
    elastic_slope = (0.05 * pressure) / probe.centre_w[-1] if probe.centre_w else None
    loads = [l * pressure for l in path.lam]
    tes = fenves_twice_elastic_slope(loads, path.centre_w, elastic_slope)
    lower = loads[-1] if loads else None
    upper = (lam[len(path.lam)] * pressure
             if len(path.lam) < len(lam) else None)
    collapse = tes if tes is not None else lower
    mp = grade.sigma_0 * thickness ** 2 / 4.0
    p_code = 8.0 * mp / width ** 2
    out = {"pressure_at_area": pressure, "load_area": load_area,
           "p0": p0, "collapse_pressure": collapse,
           "tes_pressure": tes,
           "collapse_lower_bound": lower, "collapse_upper_trial": upper,
           "collapse_basis": ("twice-elastic-slope" if tes is not None else
                              "highest converged load (statically admissible "
                              "state -> lower bound on the limit load)"),
           "code_pressure": p_code,
           "code_basis": "strip yield-line 8*Mp/b^2 with Mp=sigma_0*t^2/4",
           "mp": mp, "steps_converged": len(path.lam), "steps_requested": steps,
           "energy_residual": path.energy_residual(),
           "elastic_slope": elastic_slope,
           "deflection": path.centre_w[-1], "mesh": [nx, ny],
           "thickness": thickness, "grade": grade.name,
           "von_karman": model.von_karman,
           "provenance": ice.provenance}
    if collapse is not None:
        out["rel_error"] = collapse / p_code - 1.0
    return out


def plate_field_collapse_case(span_x: float = 1.6, width: float = 0.8,
                              thickness: float = 0.018, nx: int = 24, ny: int = 12,
                              steps: int = 30, imperfection_ratio=None,
                              grade: SteelGrade = AH36) -> dict:
    """C5: FEM peak stress of a plate field under end shortening vs the Smith law."""
    beta = width / thickness * math.sqrt(grade.sigma_0 / grade.young)
    if imperfection_ratio is None:
        imperfection_ratio = 0.1 * beta ** 2
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, thickness, grade)
    fixed = simply_supported_edges(model)
    fixed += [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x0")]
    corner = int(np.argmin(model.nodes[:, 0] ** 2 + model.nodes[:, 1] ** 2))
    fixed = sorted(set(fixed + [DOF_PER_NODE * corner + I_V]))
    _, _, mode = buckling_eigenvalue(model, 1.0e6, fixed, direction="x",
                                     return_modes=True)
    w = mode[I_W::DOF_PER_NODE]
    w = w / np.max(np.abs(w))
    model.set_imperfection(w * imperfection_ratio * thickness)
    loaded = [DOF_PER_NODE * int(n) + I_U for n in edge_nodes(model, "x1")]
    total = 0.004 * span_x
    increments = np.linspace(0.0, total, steps + 1)[1:]
    prescribed = {dof: -increments for dof in loaded}
    path = solve_displacement_control(model, fixed, prescribed, increments,
                                      tol=1e-7, max_iter=25)
    area = width * thickness
    forces = np.abs(np.asarray(path.axial_force)) if path.axial_force else np.zeros(0)
    strains = increments[:len(path.u)] / span_x
    stress = forces / area
    peak = float(np.max(stress)) if stress.size else None
    from .smith_progressive_collapse import PlateStrip
    strip = PlateStrip(length=span_x, width=width, thickness=thickness,
                       young=grade.young, poisson=grade.poisson,
                       yield_stress=grade.sigma_0,
                       imperfection_ratio=imperfection_ratio,
                       residual_ratio=0.0)
    smith = float(strip.ultimate_stress)
    out = {"beta": beta, "imperfection_ratio": imperfection_ratio,
           "w0": imperfection_ratio * thickness,
           "sigma_fe_peak": peak, "sigma_smith": smith,
           "steps_requested": steps, "steps_completed": len(path.u),
           "mesh": [nx, ny], "thickness": thickness, "span": [span_x, width]}
    if peak is not None:
        out["rel_error"] = peak / smith - 1.0
    return out


def _displacement_path_from_imperfection(model, fixed, prescribed, increments,
                                        u0, state0=None, ez0=None,
                                        tol=1e-8, max_iter=25):
    """Displacement control started from an out-of-flat initial field."""
    prescribed = {int(k): np.asarray(v, dtype=float) for k, v in prescribed.items()}
    dofs = list(fixed) + list(prescribed.keys())
    free = free_dofs(model.ndof, dofs)
    pdofs = np.array(sorted(prescribed.keys()), dtype=int)
    u = np.array(u0, dtype=float)
    for dof in dofs:
        u[dof] = 0.0
    state = PointState.zeros(model.n_points) if state0 is None else state0.copy()
    ez = np.zeros(model.n_points) if ez0 is None else np.array(ez0, dtype=float)
    path = SolutionPath()
    centre = _centre_node(model)
    for s in range(len(increments)):
        for dof, values in prescribed.items():
            u[dof] = values[min(s, values.size - 1)]
        converged = False
        for it in range(max_iter):
            ue, Baug, strain = model.layer_kinematics(u)
            stress, new_state, dWp, ez_new, C = model.stress_update(
                strain, state, ez, need_tangent=True)
            fint = model.assembled_internal_force(
                ue, Baug, stress, model.shear_strain(ue))
            ref = max(np.linalg.norm(fint[pdofs]),
                      1e-8 * np.linalg.norm(fint[free]), 1e-30)
            if np.linalg.norm(fint[free]) <= tol * ref:
                converged = True
                break
            N = model.membrane_resultants(stress)
            tmp = np.einsum("pik,pkl->pil", C, Baug) * model.layer_weight[:, None, None]
            Ke = np.einsum("pij,pil->pjl", Baug, tmp).reshape(
                model.n_elem, model.npt_elem, 20, 20).sum(axis=1)
            Ke += model.shear_stiffness_element()
            K = model.assemble(Ke, model.geometric_stiffness(N) if model.von_karman else None,
                               model.stiffener_terms(stiffener_axial_forces(model, N)))
            du = spla.spsolve(K[free][:, free].tocsc(), -fint[free])
            u[free] = u[free] + du
        if not converged:
            break
        state, ez = new_state, ez_new
        path.lam.append(float(increments[s]))
        path.u.append(u.copy())
        path.centre_w.append(float(u[DOF_PER_NODE * centre + I_W]))
        path.axial_force.append(float(fint[pdofs].sum()))
        path.converged.append(True)
        path.iterations.append(it + 1)
        path.plastic_work += float(np.sum(dWp * model.layer_weight))
    path.state = state
    path.ez = ez
    return path


def mesh_convergence_case(meshes=((16, 6), (24, 9), (32, 12)), fraction: float = 0.6,
                          span_x: float = 3.0, width: float = 0.5,
                          thickness: float = 0.016, steps: int = 8) -> dict:
    """C6: h-refinement convergence of the ice-belt deflection at a fixed load."""
    mp = elastic_perfectly_plastic().sigma_0 * thickness ** 2 / 4.0
    reference_pressure = fraction * 8.0 * mp / width ** 2
    rows = []
    for nx, ny in meshes:
        nodes, elems = structured_mesh(span_x, width, nx, ny)
        model = PlateModel(nodes, elems, thickness, elastic_perfectly_plastic())
        model.von_karman = False
        load = uniform_pressure_load(model, reference_pressure)
        lam = np.linspace(0.0, 1.0, steps + 1)[1:]
        path = solve_load_control(model, load, inplane_free_bcs(model), lam,
                                  tol=1e-8, max_iter=25)
        rows.append({"mesh": [nx, ny], "deflection": path.centre_w[-1],
                     "elements": nx * ny, "steps": len(path.lam)})
    finest = rows[-1]["deflection"]
    for row in rows:
        row["rel_to_finest"] = row["deflection"] / finest - 1.0
    successive = [abs(rows[k + 1]["deflection"] / rows[k]["deflection"] - 1.0)
                  for k in range(len(rows) - 1)]
    for k, value in enumerate(successive):
        rows[k]["change_on_refinement"] = value
    return {"reference_pressure": reference_pressure, "rows": rows,
            "successive_changes": successive,
            "last_refinement_change": successive[-1],
            "coarsest_deviation": abs(rows[0]["rel_to_finest"]),
            "max_rel_spread": successive[-1],
            "finest_mesh": list(meshes[-1])}


def run_verification_suite(*, fast: bool = True) -> dict:
    """C1-C7 of the ice-belt progressive-collapse verification face."""
    ice_mesh = (24, 8) if fast else (32, 12)
    ice_steps = 30 if fast else 60
    cv_steps = 4 if fast else 8
    checks = {}
    checks["C1_elastic_plate_navier"] = elastic_plate_case(16)
    checks["C2_plastic_hinge_moment"] = plastic_hinge_case()
    checks["C3_plate_field_buckling"] = plate_buckling_case(24, 16)
    checks["C4_plate_field_collapse"] = plate_field_collapse_case(
        nx=16 if fast else 24, ny=8 if fast else 12,
        steps=100 if fast else 140)
    checks["C5_stiffened_panel_column"] = stiffened_panel_case(nx=40, ny=6)
    checks["C6_ice_belt_limit_pressure"] = ice_belt_case(
        nx=ice_mesh[0], ny=ice_mesh[1], steps=ice_steps)
    checks["C7_mesh_convergence"] = mesh_convergence_case(
        meshes=((16, 6), (24, 9), (32, 12), (40, 15)) if not fast
        else ((16, 6), (24, 9), (32, 12)))
    checks["C8_arc_length_plate_field"] = plate_field_collapse_arc_case(
        nx=16 if fast else 24, ny=8 if fast else 12, n_steps=140 if fast else 180)
    checks["C9_arc_length_ice_limit"] = ice_belt_limit_pressure_arc_case(
        nx=16 if fast else 24, ny=6 if fast else 8,
        n_steps=90 if fast else 150)
    tolerances = {"C1_elastic_plate_navier": 0.01,
                  "C2_plastic_hinge_moment": 0.03,
                  "C3_plate_field_buckling": 0.05,
                  "C4_plate_field_collapse": 0.10,
                  "C5_stiffened_panel_column": 0.05,
                  "C6_ice_belt_limit_pressure": 0.15,
                  "C7_mesh_convergence": 0.03,
                  "C8_arc_length_plate_field": 0.10,
                  "C9_arc_length_ice_limit": 0.15,
                  "C10_arc_length_vs_paik": 0.05}
    verdicts = {}
    c1 = checks["C1_elastic_plate_navier"]
    verdicts["C1_elastic_plate_navier"] = abs(c1["rel_error"]) <= tolerances[
        "C1_elastic_plate_navier"]
    c2 = checks["C2_plastic_hinge_moment"]
    verdicts["C2_plastic_hinge_moment"] = (c2["all_converged"]
                                           and c2["max_abs_error"] is not None
                                           and c2["max_abs_error"] <= tolerances[
                                               "C2_plastic_hinge_moment"])
    c3 = checks["C3_plate_field_buckling"]
    verdicts["C3_plate_field_buckling"] = abs(c3["rel_error"]) <= tolerances[
        "C3_plate_field_buckling"]
    c4 = checks["C4_plate_field_collapse"]
    verdicts["C4_plate_field_collapse"] = bool(
        c4.get("rel_error") is not None
        and abs(c4["rel_error"]) <= tolerances["C4_plate_field_collapse"]
        and c4["steps_completed"] == c4["steps_requested"])
    c5 = checks["C5_stiffened_panel_column"]
    verdicts["C5_stiffened_panel_column"] = abs(c5["rel_error"]) <= tolerances[
        "C5_stiffened_panel_column"]
    c6 = checks["C6_ice_belt_limit_pressure"]
    verdicts["C6_ice_belt_limit_pressure"] = bool(
        c6.get("rel_error") is not None
        and abs(c6["rel_error"]) <= tolerances["C6_ice_belt_limit_pressure"])
    c7 = checks["C7_mesh_convergence"]
    verdicts["C7_mesh_convergence"] = c7["max_rel_spread"] <= tolerances[
        "C7_mesh_convergence"]
    c8 = checks["C8_arc_length_plate_field"]
    verdicts["C8_arc_length_plate_field"] = bool(
        abs(c8["energy_residual"]) <= 0.02 and abs(c8["rel_error"]) <= 0.10)
    c9 = checks["C9_arc_length_ice_limit"]
    verdicts["C9_arc_length_ice_limit"] = bool(
        abs(c9["energy_residual"]) <= 0.02 and abs(c9["rel_error"]) <= 0.15)
    verdicts["C10_arc_length_vs_paik"] = bool(
        abs(c8["energy_residual"]) <= 0.02 and abs(c8["rel_error_paik"]) <= 0.05)
    energy = abs(checks["C6_ice_belt_limit_pressure"]["energy_residual"])
    verdicts["energy_ledger"] = energy <= 0.02
    return {"checks": checks, "tolerances": tolerances, "verdicts": verdicts,
            "energy_ledger_residual": energy,
            "passed": all(verdicts.values()),
            "fast": fast}
