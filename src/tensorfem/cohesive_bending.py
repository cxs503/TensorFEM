"""Zero-thickness cohesive interface elements for mode-I crack growth in bending.

This module provides a complete, self-contained 2-D plane-stress simulation of
progressive mode-I fracture of a brittle plate/beam in three-point bending:

* :class:`BilinearCohesiveLaw` -- an irreversible bilinear traction--separation
  law (peak traction ``sigma_f``, fracture energy ``G_f``) whose initial
  stiffness is tied to the material's cohesive length
  ``l_ch = E G_f / sigma_f**2``;
* :func:`build_three_point_bend` -- a structured bilinear (Q4) mesh of the beam
  with a *duplicated* node line on the symmetry plane, i.e. a pre-meshed
  cohesive interface spanning the whole depth;
* :func:`solve_cmod_control` -- a Newton--Raphson path follower driven by the
  **crack-mouth opening displacement (CMOD)**, which stays monotone through the
  load peak and through post-peak softening.  Crack-mouth controlled bend tests
  are the standard way to trace a specimen past its peak; load-point
  displacement control would fold back (snap-back) for a brittle specimen.
  The applied load enters through a platen whose stroke is an unknown of the
  system, so the load is the stroke *reaction* and the stroke may decrease
  post-peak while the CMOD keeps growing.

The crack front is *not* prescribed: only the interface plane is pre-meshed
(standard cohesive-element practice, and exact here because the mode-I crack
plane of a symmetric three-point bend is the symmetry plane).  Damage initiates
at the tensile face, the interface elements soften in Mode I, and the crack
front height, the load--displacement curve and the dissipated fracture energy
are returned as history arrays.

Energy accounting is deliberately redundant:

* ``dissipated_path`` -- trapezoidal integration of ``t_n d(delta_n)`` over the
  accepted load steps (a path integral of the constitutive response);
* ``dissipated_envelope`` -- closed-form area under the bilinear envelope at the
  committed maximum opening of every Gauss point;
* ``external_work`` / ``strain_energy`` -- global balance
  ``W_ext - U_elastic - W_diss ~ 0``.

All quantities are SI (m, Pa, J/m^2, N).

Conventions
-----------
``x`` is the span direction, ``y`` the depth direction.  The interface line is
vertical at ``x = span/2`` with unit normal ``+x``, so the Mode-I opening is
``u_x(right) - u_x(left) > 0``.  DOF order is ``(u_x, u_y)`` per node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "BilinearCohesiveLaw",
    "ThreePointBendModel",
    "CohesiveBendResult",
    "build_three_point_bend",
    "solve_cmod_control",
    "solve_arclength",
    "flexural_strength_load",
    "timoshenko_midspan_deflection",
    "plane_stress_matrix",
]

# 2-point Gauss rule on the interface parametric coordinate s in [-1, 1].
_GP_S = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
_GP_W = (1.0, 1.0)
N_GP = len(_GP_S)

#: relative floor applied to the cohesive tangent (numerics only).
_TANGENT_FLOOR = 1e-6

# 2x2 Gauss rule for the Q4 continuum elements.
_H = 1.0 / math.sqrt(3.0)
_Q4_POINTS = ((-_H, -_H), (_H, -_H), (_H, _H), (-_H, _H))


# ---------------------------------------------------------------------------
# cohesive law
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BilinearCohesiveLaw:
    """Irreversible bilinear Mode-I cohesive law with elastic Mode-II penalty.

    Parameters
    ----------
    normal_stiffness : float
        Penalty stiffness of the interface normal springs [Pa/m].
    tangential_stiffness : float
        Penalty stiffness of the interface tangential springs [Pa/m].  The
        tangential response stays linear elastic (Mode-I model).
    peak_traction : float
        Tensile strength ``sigma_f`` [Pa].
    fracture_energy : float
        Mode-I fracture energy ``G_f`` [J/m^2].

    The envelope is linear from ``(0, 0)`` to ``(delta_0, sigma_f)`` with
    ``delta_0 = sigma_f / K`` and then linear to ``(delta_f, 0)`` with
    ``delta_f = 2 G_f / sigma_f``, so the envelope area equals ``G_f`` exactly.
    Compression keeps the undamaged stiffness and never advances damage.
    """

    normal_stiffness: float
    tangential_stiffness: float
    peak_traction: float
    fracture_energy: float

    # -- construction ------------------------------------------------------
    @classmethod
    def from_material(
        cls,
        young_modulus: float,
        tensile_strength: float,
        fracture_energy: float,
        *,
        penalty_ratio: float = 50.0,
    ) -> "BilinearCohesiveLaw":
        """Build the law from physical parameters and the cohesive length.

        The normal penalty stiffness is ``K = penalty_ratio * E / l_ch`` with
        ``l_ch = E G_f / sigma_f**2``, hence ``delta_0 / delta_f =
        1 / (2 penalty_ratio)``: the elastic part of the envelope is a
        controlled fraction of the fracture opening and refining the interface
        mesh does not change the shape of the law.
        """
        if young_modulus <= 0.0 or tensile_strength <= 0.0 or fracture_energy <= 0.0:
            raise ValueError("material parameters must be positive")
        if penalty_ratio < 10.0:
            raise ValueError("penalty_ratio must be >= 10 to keep delta_0 << delta_f")
        l_ch = young_modulus * fracture_energy / tensile_strength**2
        stiffness = penalty_ratio * young_modulus / l_ch
        return cls(
            normal_stiffness=stiffness,
            tangential_stiffness=stiffness,
            peak_traction=float(tensile_strength),
            fracture_energy=float(fracture_energy),
        )

    # -- derived quantities ------------------------------------------------
    @property
    def damage_opening(self) -> float:
        """Opening at which damage starts, ``delta_0 = sigma_f / K``."""
        return self.peak_traction / self.normal_stiffness

    @property
    def failure_opening(self) -> float:
        """Opening at which the traction vanishes, ``delta_f = 2 G_f / sigma_f``."""
        return 2.0 * self.fracture_energy / self.peak_traction

    @property
    def cohesive_length(self) -> float:
        """Cohesive length implied by the penalty stiffness ``l_ch = E G_f/sigma_f^2``.

        Only meaningful for a law created by :meth:`from_material`, where
        ``E = K l_ch / penalty_ratio`` is folded into ``K``.
        """
        return 2.0 * self.normal_stiffness * self.fracture_energy ** 2 / self.peak_traction**3

    # -- constitutive response --------------------------------------------
    def damage(self, kappa: np.ndarray) -> np.ndarray:
        """Damage variable ``d(kappa)`` for a (trial or committed) history."""
        kappa = np.asarray(kappa, dtype=float)
        d0, df = self.damage_opening, self.failure_opening
        d = np.zeros_like(kappa)
        mid = (kappa > d0) & (kappa < df)
        if np.any(mid):
            d[mid] = df * (kappa[mid] - d0) / (kappa[mid] * (df - d0))
        d[kappa >= df] = 1.0
        return np.clip(d, 0.0, 1.0)

    def damage_slope(self, kappa: np.ndarray) -> np.ndarray:
        """``dd/dkappa`` inside the softening branch, zero elsewhere."""
        kappa = np.asarray(kappa, dtype=float)
        d0, df = self.damage_opening, self.failure_opening
        slope = np.zeros_like(kappa)
        mid = (kappa > d0) & (kappa < df)
        if np.any(mid):
            slope[mid] = df * d0 / (kappa[mid] ** 2 * (df - d0))
        return slope

    def envelope_area(self, kappa: np.ndarray) -> np.ndarray:
        """Energy dissipated per unit area at history ``kappa`` [J/m^2]."""
        kappa = np.asarray(kappa, dtype=float)
        d0, df = self.damage_opening, self.failure_opening
        area = np.zeros_like(kappa)
        rising = (kappa > 0.0) & (kappa <= d0)
        area[rising] = 0.5 * self.normal_stiffness * kappa[rising] ** 2
        soft = (kappa > d0) & (kappa < df)
        if np.any(soft):
            area[soft] = (
                0.5 * self.normal_stiffness * d0**2
                + 0.5 * self.peak_traction * (kappa[soft] - d0)
            )
        area[kappa >= df] = self.fracture_energy
        return area

    def traction(self, delta_n: float, delta_t: float, kappa: float):
        """Traction and consistent tangent for one trial opening state.

        Returns ``(t_n, t_t, c_nn, c_tt, kappa_new)``.  ``kappa_new`` is the
        updated history variable; compression never advances it and unloading
        keeps the secant stiffness ``(1-d)K`` of the historical maximum.
        """
        k_trial = max(kappa, delta_n)
        if delta_n > 0.0:
            d = float(self.damage(np.asarray([k_trial]))[0])
            t_n = (1.0 - d) * self.normal_stiffness * delta_n
            loading = delta_n >= k_trial
            if loading and self.damage_opening < k_trial < self.failure_opening:
                slope = float(self.damage_slope(np.asarray([k_trial]))[0])
                c_nn = (1.0 - d) * self.normal_stiffness - self.normal_stiffness * delta_n * slope
            else:
                c_nn = (1.0 - d) * self.normal_stiffness
        else:
            k_trial = kappa
            t_n = self.normal_stiffness * delta_n
            c_nn = self.normal_stiffness
        t_t = self.tangential_stiffness * delta_t
        return t_n, t_t, c_nn, self.tangential_stiffness, k_trial


# ---------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------
@dataclass
class ThreePointBendModel:
    """Structured Q4 mesh of a beam with a cohesive interface on the mid-span."""

    nodes: np.ndarray  # (nn, 2)
    elements: np.ndarray  # (ne, 4)
    interface: np.ndarray  # (ns, 4) = [left_bottom, left_top, right_bottom, right_top]
    thickness: float
    law: BilinearCohesiveLaw
    young_modulus: float
    poisson_ratio: float
    span: float
    depth: float
    dirichlet: dict  # dof index -> prescribed value
    load_dofs: np.ndarray  # dof indices carrying the reference load
    load_weights: np.ndarray  # reference load per dof [N]
    mouth_dofs: tuple  # (right bottom u_x dof, left bottom u_x dof)

    @property
    def n_dofs(self) -> int:
        return 2 * len(self.nodes)

    @property
    def n_segments(self) -> int:
        return len(self.interface)

    @property
    def segment_heights(self) -> np.ndarray:
        """Vertical extent of every interface segment [m]."""
        y = self.nodes[self.interface[:, :2], 1]
        return np.abs(y[:, 1] - y[:, 0])

    @property
    def segment_centres(self) -> np.ndarray:
        """Centroid of every interface segment, shape ``(ns, 2)``."""
        left = self.nodes[self.interface[:, :2]]
        right = self.nodes[self.interface[:, 2:]]
        left_mid = 0.5 * (left[:, 0, :] + left[:, 1, :])
        right_mid = 0.5 * (right[:, 0, :] + right[:, 1, :])
        return 0.5 * (left_mid + right_mid)

    def reference_load_vector(self) -> np.ndarray:
        """Reference load vector: unit total downward force on the top patch."""
        f = np.zeros(self.n_dofs)
        f[self.load_dofs] = self.load_weights
        return f

    def free_dofs(self) -> np.ndarray:
        mask = np.ones(self.n_dofs, dtype=bool)
        mask[list(self.dirichlet)] = False
        return np.flatnonzero(mask)


def plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    """Isotropic plane-stress constitutive matrix (engineering shear)."""
    return (
        young
        / (1.0 - poisson**2)
        * np.array(
            [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]
        )
    )


def _q4_stiffness(coords: np.ndarray, dmat: np.ndarray, thickness: float) -> np.ndarray:
    """Batched fully integrated Q4 element stiffness ``(ne, 8, 8)``."""
    ne = len(coords)
    ke = np.zeros((ne, 8, 8))
    for xi, eta in _Q4_POINTS:
        natural = np.array(
            [
                [-(1 - eta), -(1 - xi)],
                [(1 - eta), -(1 + xi)],
                [(1 + eta), (1 + xi)],
                [-(1 + eta), (1 - xi)],
            ]
        ) / 4.0
        jac = np.einsum("eia,ib->eab", coords, natural)
        det = np.linalg.det(jac)
        if np.any(det <= 0.0):
            raise ValueError("Q4 element has non-positive Jacobian")
        deriv = np.einsum("ib,ebc->eic", natural, np.linalg.inv(jac))
        b = np.zeros((ne, 3, 8))
        b[:, 0, 0::2] = deriv[:, :, 0]
        b[:, 1, 1::2] = deriv[:, :, 1]
        b[:, 2, 0::2] = deriv[:, :, 1]
        b[:, 2, 1::2] = deriv[:, :, 0]
        ke += np.einsum("eai,ab,ebj->eij", b, dmat, b) * (det * thickness)[:, None, None]
    return ke


def build_three_point_bend(
    *,
    span: float,
    depth: float,
    thickness: float,
    young_modulus: float,
    poisson_ratio: float,
    law: BilinearCohesiveLaw,
    nx: int = 60,
    ny: int = 80,
    load_patch_fraction: float = 0.25,
) -> ThreePointBendModel:
    """Build the beam mesh with a cohesive interface on the mid-span plane.

    Parameters
    ----------
    span, depth, thickness : float
        Beam geometry [m]; ``span`` is the support span, ``depth`` the beam
        height (bending direction), ``thickness`` the out-of-plane width.
    nx : int
        Total number of Q4 elements along the span (must be even: the interface
        line splits the mesh into two ``nx/2`` blocks).
    ny : int
        Number of Q4 elements through the depth.  The cohesive interface has
        ``ny`` linear segments, so the crack-front resolution is ``depth/ny``.
    load_patch_fraction : float
        Width of the loaded patch on the top face as a fraction of ``depth``.
        A finite patch avoids the singular point-load field at the crack mouth
        while leaving the mid-span bending moment unchanged.
    """
    if nx % 2 != 0:
        raise ValueError("nx must be even so that the interface lies on a node line")
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be >= 2")
    nxh = nx // 2
    nrx = nx - nxh
    x_left = np.linspace(0.0, span / 2.0, nxh + 1)
    x_right = np.linspace(span / 2.0, span, nrx + 1)
    y = np.linspace(0.0, depth, ny + 1)

    left = np.stack(np.meshgrid(x_left, y, indexing="xy"), axis=-1).reshape(-1, 2)
    right = np.stack(np.meshgrid(x_right, y, indexing="xy"), axis=-1).reshape(-1, 2)
    nodes = np.vstack((left, right))
    offset = left.shape[0]

    elements = []
    for j in range(ny):
        for i in range(nxh):
            n0 = j * (nxh + 1) + i
            elements.append((n0, n0 + 1, n0 + nxh + 2, n0 + nxh + 1))
    for j in range(ny):
        for i in range(nrx):
            n0 = offset + j * (nrx + 1) + i
            elements.append((n0, n0 + 1, n0 + nrx + 2, n0 + nrx + 1))
    elements = np.asarray(elements, dtype=np.int64)

    interface = []
    for j in range(ny):
        lb = j * (nxh + 1) + nxh
        lt = (j + 1) * (nxh + 1) + nxh
        rb = offset + j * (nrx + 1)
        rt = offset + (j + 1) * (nrx + 1)
        interface.append((lb, lt, rb, rt))
    interface = np.asarray(interface, dtype=np.int64)

    # supports: pin bottom-left corner, roller at bottom-right corner
    dirichlet = {0: 0.0, 1: 0.0}
    support_right = offset + nrx  # node (span, 0) of the right block
    dirichlet[2 * support_right + 1] = 0.0

    top = np.concatenate(
        [
            np.arange(ny * (nxh + 1), (ny + 1) * (nxh + 1)),
            offset + np.arange(ny * (nrx + 1), (ny + 1) * (nrx + 1)),
        ]
    )
    x_top = nodes[top, 0]
    patch = np.abs(x_top - span / 2.0) <= 0.5 * load_patch_fraction * depth
    loaded = top[patch]
    weights = np.full(len(loaded), -1.0 / len(loaded))
    load_dofs = 2 * loaded + 1

    # crack-mouth pair: the bottom nodes of the two blocks at x = span/2
    mouth = (2 * (offset + 0), 2 * (0 * (nxh + 1) + nxh))

    return ThreePointBendModel(
        nodes=nodes,
        elements=elements,
        interface=interface,
        thickness=thickness,
        law=law,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        span=span,
        depth=depth,
        dirichlet=dirichlet,
        load_dofs=load_dofs,
        load_weights=weights,
        mouth_dofs=mouth,
    )


# ---------------------------------------------------------------------------
# interface element
# ---------------------------------------------------------------------------
def _assemble(
    model: ThreePointBendModel,
    u: np.ndarray,
    kappa: np.ndarray,
    elastic_stiffness: sp.csr_matrix,
    edofs: np.ndarray,
):
    """Internal force and tangent of the whole model at displacement ``u``."""
    ns = model.n_segments
    law = model.law
    force = np.zeros((ns, 8))
    tangent = np.zeros((ns, 8, 8))
    t_n = np.zeros((ns, N_GP))
    t_t = np.zeros((ns, N_GP))
    delta_n = np.zeros((ns, N_GP))
    delta_t = np.zeros((ns, N_GP))
    kappa_trial = np.zeros((ns, N_GP))
    weights = np.zeros((ns, N_GP))

    for s in range(ns):
        lb, lt, rb, rt = model.interface[s]
        xl = model.nodes[[lb, lt]]
        diff = xl[1] - xl[0]
        ds = math.hypot(diff[0], diff[1])
        if ds <= 0.0:
            raise ValueError("degenerate interface segment")
        normal = np.array([diff[1], -diff[0]]) / ds
        tvec = np.array([diff[0], diff[1]]) / ds
        dofs = edofs[s]
        ul = u[dofs[:4]].reshape(2, 2)
        ur = u[dofs[4:]].reshape(2, 2)
        for g, (sv, wv) in enumerate(zip(_GP_S, _GP_W)):
            shape = np.array([(1.0 - sv) / 2.0, (1.0 + sv) / 2.0])
            rel = shape @ (ur - ul)
            dn = float(rel @ normal)
            dt = float(rel @ tvec)
            tn, tt, cnn, ctt, kk = law.traction(dn, dt, float(kappa[s, g]))
            w = 0.5 * ds * wv * model.thickness
            b = np.zeros((2, 8))
            b[0, 0:2] = -shape[0] * normal
            b[0, 2:4] = -shape[1] * normal
            b[0, 4:6] = shape[0] * normal
            b[0, 6:8] = shape[1] * normal
            b[1, 0:2] = -shape[0] * tvec
            b[1, 2:4] = -shape[1] * tvec
            b[1, 4:6] = shape[0] * tvec
            b[1, 6:8] = shape[1] * tvec
            force[s] += (b.T @ np.array([tn, tt])) * w
            # numerical tangent regularisation: fully failed Gauss points have
            # zero normal stiffness, which makes the bordered system singular.
            # The floor only modifies the tangent, never the residual, so the
            # converged solution is unaffected.
            cnn_reg = max(cnn, _TANGENT_FLOOR * law.normal_stiffness)
            ctt_reg = max(ctt, _TANGENT_FLOOR * law.tangential_stiffness)
            tangent[s] += (b.T @ np.diag([cnn_reg, ctt_reg]) @ b) * w
            t_n[s, g] = tn
            t_t[s, g] = tt
            delta_n[s, g] = dn
            delta_t[s, g] = dt
            kappa_trial[s, g] = kk
            weights[s, g] = w

    f_int = elastic_stiffness @ u
    rows = (edofs[:, :, None] * np.ones((1, 1, 8), dtype=np.int64)).reshape(-1)
    cols = (edofs[:, None, :] * np.ones((1, 8, 1), dtype=np.int64)).reshape(-1)
    coh_k = sp.coo_matrix(
        (tangent.reshape(-1), (rows, cols)), shape=(model.n_dofs, model.n_dofs)
    ).tocsr()
    np.add.at(f_int, edofs.reshape(-1), force.reshape(-1))
    return {
        "force": f_int,
        "stiffness": (elastic_stiffness + coh_k).tocsr(),
        "t_n": t_n,
        "t_t": t_t,
        "delta_n": delta_n,
        "delta_t": delta_t,
        "kappa_trial": kappa_trial,
        "weights": weights,
    }


def _interface_nodes(model: ThreePointBendModel) -> np.ndarray:
    return np.stack(
        [
            2 * model.interface[:, 0],
            2 * model.interface[:, 0] + 1,
            2 * model.interface[:, 1],
            2 * model.interface[:, 1] + 1,
            2 * model.interface[:, 2],
            2 * model.interface[:, 2] + 1,
            2 * model.interface[:, 3],
            2 * model.interface[:, 3] + 1,
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# CMOD-controlled Newton path follower
# ---------------------------------------------------------------------------
@dataclass
class CohesiveBendResult:
    """History of a CMOD-controlled three-point bend simulation."""

    cmod: np.ndarray
    load: np.ndarray  # total applied load [N]
    deflection: np.ndarray  # load-point deflection, downward positive [m]
    crack_length: np.ndarray  # fully failed height on the interface [m]
    process_zone_length: np.ndarray  # tip-inclusive height [m]
    dissipated_envelope: np.ndarray
    dissipated_path: np.ndarray
    external_work: np.ndarray
    strain_energy: np.ndarray
    damage_profile: np.ndarray  # (ns,) final damage
    kappa_profile: np.ndarray  # (ns,) final max opening
    segment_heights: np.ndarray
    segment_centres: np.ndarray
    peak_load: float
    peak_deflection: float
    peak_index: int
    newton_iterations: np.ndarray
    converged: bool
    stop_reason: str
    law: BilinearCohesiveLaw
    meta: dict = field(default_factory=dict)


def _newton(
    model: ThreePointBendModel,
    u: np.ndarray,
    delta: float,
    cmod: float,
    kappa: np.ndarray,
    *,
    elastic_stiffness: sp.csr_matrix,
    edofs: np.ndarray,
    free: np.ndarray,
    platen: np.ndarray,
    g_row: sp.csr_matrix,
    rtol: float,
    atol: float,
    max_iter: int,
    debug: bool = False,
):
    """Newton iteration for the bordered platen/CMOD system.

    Unknowns are the free displacements ``u[free]`` plus the platen stroke
    ``delta`` (so that ``u[platen] = -delta``).  The extra equation is the
    crack-mouth opening constraint ``u_rx - u_lx = cmod``.  The load is *not* an
    unknown: it is the reaction of the stroke, ``P = -sum(f_int[platen])``.
    """
    n_free = len(free)
    mouth = model.mouth_dofs

    def merit(force, gap):
        scale = max(abs(reaction_ref[0]), 1.0)
        cscale = max(abs(cmod), 1e-30)
        return 0.5 * (float(force @ force) / scale**2 + (gap / cscale) ** 2)

    reaction_ref = [1.0]
    for it in range(max_iter):
        out = _assemble(model, u, kappa, elastic_stiffness, edofs)
        reaction = -float(np.sum(out["force"][platen]))
        reaction_ref[0] = reaction
        residual = out["force"][free]
        g = float(u[mouth[0]] - u[mouth[1]] - cmod)
        r_norm = float(np.max(np.abs(residual))) if n_free else 0.0
        if debug:
            print(f"    it={it} |R|={r_norm:.3e} g={g:.3e} P={reaction:.4f}")
        if r_norm <= atol + rtol * max(abs(reaction), 1.0) and abs(g) <= 1e-8 * max(abs(cmod), 1e-30):
            return u, delta, out, reaction, it, True
        stiffness = out["stiffness"]
        kff = stiffness[free][:, free].tocsc()
        column = -np.asarray(stiffness[:, platen].sum(axis=1)).reshape(-1)[free]
        bordered = sp.bmat(
            [[kff, sp.csr_matrix(column[:, None])], [g_row, None]], format="csc"
        )
        rhs = np.concatenate((-residual, np.array([-g])))
        try:
            solution = spla.spsolve(bordered, rhs)
        except Exception:  # pragma: no cover - defensive
            return u, delta, out, reaction, it, False
        if not np.all(np.isfinite(solution)):
            return u, delta, out, reaction, it, False
        du = solution[:n_free]
        d_delta = float(solution[n_free])
        # backtracking line search on the merit function (damage-induced
        # stiffness jumps make the full Newton step unreliable)
        current = merit(residual, g)
        alpha = 1.0
        accepted = False
        for _ in range(7):
            u_try = u.copy()
            u_try[free] += alpha * du
            u_try[platen] = -(delta + alpha * d_delta)
            out_try = _assemble(model, u_try, kappa, elastic_stiffness, edofs)
            res_try = out_try["force"][free]
            g_try = float(u_try[mouth[0]] - u_try[mouth[1]] - cmod)
            trial = merit(res_try, g_try)
            if trial < current * (1.0 - 1e-4 * alpha) or trial < 1e-20 * current:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return u, delta, out, reaction, it, False
        u = u_try
        delta += alpha * d_delta
        out_final = out_try
    out = _assemble(model, u, kappa, elastic_stiffness, edofs)
    reaction = -float(np.sum(out["force"][platen]))
    return u, delta, out, reaction, max_iter, False


def solve_cmod_control(
    model: ThreePointBendModel,
    *,
    cmod_max: float | None = None,
    initial_step: float | None = None,
    growth: float = 1.3,
    max_steps: int = 400,
    rtol: float = 1e-5,
    atol: float = 1e-3,
    max_iter: int = 40,
    stop_load_fraction: float = 0.02,
    load_tolerance: float = 0.05,
    verbose: bool = False,
    debug: bool = False,
) -> CohesiveBendResult:
    """Trace the three-point bend response under crack-mouth opening control.

    The load is transmitted by a platen whose stroke ``Delta`` is an *unknown*
    (``u_y = -Delta`` on the loaded patch); the path parameter is the
    crack-mouth opening ``CMOD``.  The two closure equations are

    ``R  = f_int(u) = 0``  on the free dofs (no free external force),
    ``g  = u_x(right mouth) - u_x(left mouth) - CMOD = 0``,

    with unknowns ``(u_free, Delta)`` -- a square system.  The applied load is
    the stroke reaction ``P = -sum(f_int[platen])``.  Driving ``CMOD`` keeps the
    solution on the equilibrium path through the peak and through post-peak
    softening, where the stroke ``Delta`` decreases (snap-back) and full
    load-point displacement control has no solution.
    """
    law = model.law
    dmat = plane_stress_matrix(model.young_modulus, model.poisson_ratio)
    coords = model.nodes[model.elements]
    ke = _q4_stiffness(coords, dmat, model.thickness)
    edofs_e = np.stack([2 * model.elements + i for i in (0, 1)], axis=2).reshape(
        len(model.elements), 8
    )
    rows = np.repeat(edofs_e, 8, axis=1).reshape(-1)
    cols = np.tile(edofs_e, (1, 8)).reshape(-1)
    elastic_stiffness = sp.coo_matrix(
        (ke.reshape(-1), (rows, cols)), shape=(model.n_dofs, model.n_dofs)
    ).tocsr()

    edofs = _interface_nodes(model)
    platen = np.asarray(model.load_dofs, dtype=np.int64)
    fixed = np.asarray(list(model.dirichlet), dtype=np.int64)
    free = np.setdiff1d(np.arange(model.n_dofs), np.concatenate((fixed, platen)))
    n_free = len(free)
    dof_index = -np.ones(model.n_dofs, dtype=np.int64)
    dof_index[free] = np.arange(n_free)
    mouth = model.mouth_dofs
    row = np.zeros((1, n_free))
    row[0, dof_index[mouth[0]]] = 1.0
    row[0, dof_index[mouth[1]]] = -1.0
    g_row = sp.csr_matrix(row)

    u = np.zeros(model.n_dofs)
    delta = 0.0
    kappa = np.zeros((model.n_segments, N_GP))
    heights = model.segment_heights
    centres = model.segment_centres

    if initial_step is None:
        initial_step = 0.5 * law.damage_opening
    if cmod_max is None:
        cmod_max = 60.0 * law.failure_opening

    keys = (
        "cmod", "load", "deflection", "crack_length", "process_zone_length",
        "dissipated_envelope", "dissipated_path", "external_work", "strain_energy",
        "newton_iterations",
    )
    history = {k: [] for k in keys}
    path_diss = 0.0
    ext_work = 0.0
    prev = None
    cmod = 0.0
    step = initial_step
    stop_reason = "max_steps"
    converged_all = True
    peak_so_far = 0.0
    iters_log = []

    def commit(out, cmod_value, load_value, u_value, kappa_new, stroke):
        nonlocal path_diss, ext_work, prev
        weights = out["weights"]
        area = law.envelope_area(kappa_new)
        diss_env = float(np.sum(weights * area))
        if prev is not None:
            path_diss += float(
                np.sum(prev["weights"] * 0.5 * (prev["t_n"] + out["t_n"]) * (out["delta_n"] - prev["delta_n"]))
            )
        u_el = 0.5 * float(u_value @ (elastic_stiffness @ u_value))
        d = law.damage(np.where(out["delta_n"] > 0.0, np.maximum(kappa_new, out["delta_n"]), kappa_new))
        coh_strain = float(
            np.sum(weights * 0.5 * ((1.0 - d) * law.normal_stiffness * out["delta_n"] ** 2
                                    + law.tangential_stiffness * out["delta_t"] ** 2))
        )
        strain = u_el + coh_strain
        if prev is not None:
            ext_work += 0.5 * (prev["load"] + load_value) * (stroke - prev["stroke"])
        failed = kappa_new.max(axis=1) >= law.failure_opening
        crack = float(np.sum(heights[failed]))
        zone = float(np.sum(heights * np.clip(area.max(axis=1) / law.fracture_energy, 0.0, 1.0)))
        history["cmod"].append(cmod_value)
        history["load"].append(load_value)
        history["deflection"].append(stroke)
        history["crack_length"].append(crack)
        history["process_zone_length"].append(zone)
        history["dissipated_envelope"].append(diss_env)
        history["dissipated_path"].append(path_diss)
        history["external_work"].append(ext_work)
        history["strain_energy"].append(strain)
        history["newton_iterations"].append(0)
        prev = {
            "delta_n": out["delta_n"].copy(),
            "t_n": out["t_n"].copy(),
            "weights": weights.copy(),
            "load": load_value,
            "stroke": stroke,
        }
        return {"crack": crack, "zone": zone, "diss_env": diss_env, "strain": strain}

    for _ in range(max_steps):
        target = min(cmod + step, cmod_max)
        u_new, delta_new, out, load_new, iters, ok = _newton(
            model, u.copy(), delta, target, kappa,
            elastic_stiffness=elastic_stiffness, edofs=edofs, free=free, platen=platen,
            g_row=g_row, rtol=rtol, atol=atol, max_iter=max_iter, debug=debug,
        )
        if not ok:
            step *= 0.4
            iters_log.append(iters)
            if step < 1e-11 * max(law.failure_opening, 1e-30):
                stop_reason = "newton_stall"
                converged_all = False
                break
            continue
        kappa_committed = np.where(out["delta_n"] > 0.0, np.maximum(kappa, out["delta_n"]), kappa)
        state = commit(out, target, load_new, u_new, kappa_committed, delta_new)
        kappa = kappa_committed
        u, delta, load_value = u_new, delta_new, load_new
        cmod = target
        iters_log.append(iters)
        peak_so_far = max(peak_so_far, load_value)
        if verbose:
            print(
                f"  cm={target:.4e} P={load_value:10.1f} delta={delta:.4e} "
                f"a={state['crack']:.4f} it={iters}"
            )
        if len(history["load"]) >= 2:
            d_load = abs(history["load"][-1] - history["load"][-2])
            d_cmod = history["cmod"][-1] - history["cmod"][-2]
            if d_cmod > 0.0 and d_load > 0.0:
                step = min(step * growth, load_tolerance * max(load_value, peak_so_far) * d_cmod / d_load)
        # the step must keep making progress even where the load is flat
        # (crack arrest / stick-slip), otherwise the path stalls
        step = max(step, 0.02 * cmod, 1e-9 * law.failure_opening)
        if cmod >= cmod_max - 1e-30:
            stop_reason = "cmod_limit"
            break
        if state["crack"] >= model.depth * (1.0 - 1e-9) or state["zone"] >= model.depth * (1.0 - 1e-9):
            stop_reason = "complete_fracture"
            break
        if peak_so_far > 0.0 and load_value < stop_load_fraction * peak_so_far:
            stop_reason = "residual_ligament"
            break

    load = np.asarray(history["load"], dtype=float)
    deflection = np.asarray(history["deflection"], dtype=float)
    cmod_hist = np.asarray(history["cmod"], dtype=float)
    peak_index = int(np.argmax(load)) if len(load) else 0
    peak_load = float(load[peak_index]) if len(load) else 0.0
    peak_deflection = float(deflection[peak_index]) if len(load) else 0.0
    if 0 < peak_index < len(load) - 1:
        c0, c1, c2 = cmod_hist[peak_index - 1], cmod_hist[peak_index], cmod_hist[peak_index + 1]
        p0, p1, p2 = load[peak_index - 1], load[peak_index], load[peak_index + 1]
        denom = (c0 - c1) * (c0 - c2) * (c1 - c2)
        if abs(denom) > 0.0:
            a = ((c2 - c1) * p0 + (c0 - c2) * p1 + (c1 - c0) * p2) / denom
            b = ((c1**2 - c2**2) * p0 + (c2**2 - c0**2) * p1 + (c0**2 - c1**2) * p2) / denom
            if a < 0.0:
                c_ref = -b / (2.0 * a)
                if min(c0, c2) <= c_ref <= max(c0, c2):
                    peak_load = float(a * c_ref**2 + b * c_ref + (p1 - a * c1**2 - b * c1))
                    d0_, d1_, d2_ = (deflection[peak_index - 1], deflection[peak_index],
                                     deflection[peak_index + 1])
                    a_d = ((c2 - c1) * d0_ + (c0 - c2) * d1_ + (c1 - c0) * d2_) / denom
                    b_d = ((c1**2 - c2**2) * d0_ + (c2**2 - c0**2) * d1_ + (c0**2 - c1**2) * d2_) / denom
                    peak_deflection = float(a_d * c_ref**2 + b_d * c_ref + (d1_ - a_d * c1**2 - b_d * c1))
    damage = law.damage(kappa.max(axis=1))
    return CohesiveBendResult(
        cmod=cmod_hist,
        load=load,
        deflection=deflection,
        crack_length=np.asarray(history["crack_length"], dtype=float),
        process_zone_length=np.asarray(history["process_zone_length"], dtype=float),
        dissipated_envelope=np.asarray(history["dissipated_envelope"], dtype=float),
        dissipated_path=np.asarray(history["dissipated_path"], dtype=float),
        external_work=np.asarray(history["external_work"], dtype=float),
        strain_energy=np.asarray(history["strain_energy"], dtype=float),
        damage_profile=damage,
        kappa_profile=kappa.max(axis=1),
        segment_heights=heights,
        segment_centres=centres,
        peak_load=peak_load,
        peak_deflection=peak_deflection,
        peak_index=peak_index,
        newton_iterations=np.asarray(iters_log),
        converged=converged_all,
        stop_reason=stop_reason,
        law=law,
    )

def _assemble_reference(model: ThreePointBendModel):
    """Elastic stiffness matrix (CSR), element dof table and unit load vector."""
    dmat = plane_stress_matrix(model.young_modulus, model.poisson_ratio)
    ke = _q4_stiffness(model.nodes[model.elements], dmat, model.thickness)
    edofs_e = np.stack([2 * model.elements + i for i in (0, 1)], axis=2).reshape(
        len(model.elements), 8
    )
    rows = np.repeat(edofs_e, 8, axis=1).reshape(-1)
    cols = np.tile(edofs_e, (1, 8)).reshape(-1)
    stiffness = sp.coo_matrix(
        (ke.reshape(-1), (rows, cols)), shape=(model.n_dofs, model.n_dofs)
    ).tocsr()
    fref = model.reference_load_vector()
    return stiffness, fref


def solve_arclength(
    model: ThreePointBendModel,
    *,
    arclength: float | None = None,
    max_steps: int = 600,
    max_iter: int = 60,
    rtol: float = 1e-5,
    atol: float = 1e-3,
    growth: float = 1.25,
    stop_load_fraction: float = 0.02,
    stop_crack_fraction: float = 0.995,
    verbose: bool = False,
) -> CohesiveBendResult:
    """Trace the three-point bend response with a spherical arc-length control.

    Unknowns are ``(u_free, Lambda)`` with the proportional load
    ``Lambda * f_ref`` (``f_ref`` is a unit total downward force on the loaded
    patch), and the path constraint is

    ``|du|^2 + psi dLambda^2 = ds^2``

    with ``psi = (delta_char / Lambda_char)^2`` taken from the state at the start
    of each step.  The load is a *force* multiplier here, which is why the
    constraint - and not a prescribed displacement - carries the path: this is
    what allows the solution to pass limit points (peak load) and snap-back
    (post-peak softening of a brittle specimen), where neither load control nor
    load-point displacement control has a solution.
    """
    law = model.law
    elastic_stiffness, fref = _assemble_reference(model)
    edofs = _interface_nodes(model)
    fixed = np.asarray(list(model.dirichlet), dtype=np.int64)
    free = np.setdiff1d(np.arange(model.n_dofs), fixed)
    n_free = len(free)
    f_free = fref[free]

    u = np.zeros(model.n_dofs)
    lam = 0.0
    kappa = np.zeros((model.n_segments, N_GP))
    heights = model.segment_heights
    centres = model.segment_centres

    if arclength is None:
        arclength = 0.25 * law.damage_opening
    ds = float(arclength)

    keys = (
        "load", "deflection", "crack_length", "process_zone_length",
        "dissipated_envelope", "dissipated_path", "external_work", "strain_energy",
    )
    history = {k: [] for k in keys}
    path_diss = 0.0
    ext_work = 0.0
    prev = None
    iters_log = []
    converged_all = True
    stop_reason = "max_steps"
    peak_so_far = 0.0
    direction = 1.0

    def record(out, load_value, u_value, kappa_new):
        nonlocal path_diss, ext_work, prev
        weights = out["weights"]
        area = law.envelope_area(kappa_new)
        diss_env = float(np.sum(weights * area))
        if prev is not None:
            path_diss += float(
                np.sum(prev["weights"] * 0.5 * (prev["t_n"] + out["t_n"]) * (out["delta_n"] - prev["delta_n"]))
            )
        u_el = 0.5 * float(u_value @ (elastic_stiffness @ u_value))
        d = law.damage(np.where(out["delta_n"] > 0.0, np.maximum(kappa_new, out["delta_n"]), kappa_new))
        coh_strain = float(
            np.sum(weights * 0.5 * ((1.0 - d) * law.normal_stiffness * out["delta_n"] ** 2
                                    + law.tangential_stiffness * out["delta_t"] ** 2))
        )
        deflection = float(u_value @ fref)
        if prev is not None:
            ext_work += 0.5 * (prev["load"] + load_value) * (deflection - prev["deflection"])
        failed = kappa_new.max(axis=1) >= law.failure_opening
        crack = float(np.sum(heights[failed]))
        zone = float(np.sum(heights * np.clip(area.max(axis=1) / law.fracture_energy, 0.0, 1.0)))
        history["load"].append(load_value)
        history["deflection"].append(deflection)
        history["crack_length"].append(crack)
        history["process_zone_length"].append(zone)
        history["dissipated_envelope"].append(diss_env)
        history["dissipated_path"].append(path_diss)
        history["external_work"].append(ext_work)
        history["strain_energy"].append(u_el + coh_strain)
        prev = {
            "delta_n": out["delta_n"].copy(),
            "t_n": out["t_n"].copy(),
            "weights": weights.copy(),
            "load": load_value,
            "deflection": deflection,
        }
        return {"crack": crack, "zone": zone}

    for _ in range(max_steps):
        # ---- predictor: follow the tangent in the previous direction --------
        out = _assemble(model, u, kappa, elastic_stiffness, edofs)
        kff = out["stiffness"][free][:, free].tocsc()
        try:
            lu = spla.splu(kff)
        except Exception:  # pragma: no cover - defensive
            stop_reason = "singular_tangent"
            converged_all = False
            break
        du_ii = lu.solve(f_free)
        norm_ii = float(np.linalg.norm(du_ii))
        if norm_ii <= 0.0:
            stop_reason = "zero_tangent_load"
            converged_all = False
            break
        load_char = max(abs(lam), 1.0)
        disp_char = max(float(-u @ fref), law.damage_opening)
        psi = (disp_char / load_char) ** 2
        d_lam = direction * ds / math.sqrt(norm_ii**2 + psi)
        du = d_lam * du_ii
        u[free] += du
        lam += d_lam
        du_total = du.copy()
        d_lam_total = d_lam

        converged = False
        iters = 0
        for it in range(max_iter):
            iters = it + 1
            out = _assemble(model, u, kappa, elastic_stiffness, edofs)
            residual = out["force"][free] - lam * f_free
            r_norm = float(np.max(np.abs(residual)))
            if verbose:
                print(f"    step {len(history['load'])} it={it} |R|={r_norm:.3e} lam={lam:.4f}")
            if r_norm <= atol + rtol * max(abs(lam), 1.0):
                converged = True
                break
            try:
                lu = spla.splu(out["stiffness"][free][:, free].tocsc())
            except Exception:  # pragma: no cover - defensive
                break
            du_i = lu.solve(-residual)
            du_ii = lu.solve(f_free)
            a = float(du_ii @ du_ii) + psi
            b = 2.0 * (float((du_total + du_i) @ du_ii) + psi * d_lam_total)
            c0 = float((du_total + du_i) @ (du_total + du_i)) + psi * d_lam_total**2 - ds**2
            disc = b * b - 4.0 * a * c0
            if a <= 0.0 or disc < 0.0:
                break
            root = math.sqrt(disc)
            r1 = (-b + root) / (2.0 * a)
            r2 = (-b - root) / (2.0 * a)
            d_lam_corr = r1 if abs(r1 - d_lam_total) <= abs(r2 - d_lam_total) else r2
            du_i = du_i + d_lam_corr * du_ii
            u[free] += du_i
            lam += d_lam_corr
            du_total += du_i
            d_lam_total += d_lam_corr
        if not converged:
            # retry the step with a smaller arc length from the same state
            u[free] -= du_total
            lam -= d_lam_total
            ds *= 0.5
            iters_log.append(iters)
            if ds < 1e-12 * law.damage_opening:
                stop_reason = "arclength_stall"
                converged_all = False
                break
            continue

        kappa_new = np.where(out["delta_n"] > 0.0, np.maximum(kappa, out["delta_n"]), kappa)
        state = record(out, lam, u, kappa_new)
        kappa = kappa_new
        iters_log.append(iters)
        peak_so_far = max(peak_so_far, lam)
        if verbose:
            print(
                f"  step {len(history['load'])}: P={lam:9.2f} N "
                f"delta={history['deflection'][-1] * 1e3:8.4f} mm "
                f"a={state['crack'] * 1e3:6.2f} mm ds={ds:.3e}"
            )
        if d_lam_total < 0.0:
            direction = -1.0 if lam > 0.0 else 1.0
        # Crisfield step-size adaptation
        ds *= (6.0 / max(iters, 1)) ** 0.5 if iters > 6 else (1.0 if iters >= 3 else growth)
        if state["crack"] >= model.depth * stop_crack_fraction:
            stop_reason = "complete_fracture"
            break
        if state["zone"] >= model.depth * stop_crack_fraction:
            stop_reason = "complete_fracture"
            break
        if peak_so_far > 0.0 and lam < stop_load_fraction * peak_so_far:
            stop_reason = "residual_ligament"
            break

    load = np.asarray(history["load"], dtype=float)
    deflection = np.asarray(history["deflection"], dtype=float)
    peak_index = int(np.argmax(load)) if len(load) else 0
    peak_load = float(load[peak_index]) if len(load) else 0.0
    peak_deflection = float(deflection[peak_index]) if len(load) else 0.0
    damage = law.damage(kappa.max(axis=1))
    return CohesiveBendResult(
        cmod=np.zeros_like(load),
        load=load,
        deflection=deflection,
        crack_length=np.asarray(history["crack_length"], dtype=float),
        process_zone_length=np.asarray(history["process_zone_length"], dtype=float),
        dissipated_envelope=np.asarray(history["dissipated_envelope"], dtype=float),
        dissipated_path=np.asarray(history["dissipated_path"], dtype=float),
        external_work=np.asarray(history["external_work"], dtype=float),
        strain_energy=np.asarray(history["strain_energy"], dtype=float),
        damage_profile=damage,
        kappa_profile=kappa.max(axis=1),
        segment_heights=heights,
        segment_centres=centres,
        peak_load=peak_load,
        peak_deflection=peak_deflection,
        peak_index=peak_index,
        newton_iterations=np.asarray(iters_log),
        converged=converged_all,
        stop_reason=stop_reason,
        law=law,
    )

# ---------------------------------------------------------------------------
# analytical anchors
# ---------------------------------------------------------------------------
def flexural_strength_load(
    *, span: float, depth: float, thickness: float, tensile_strength: float
) -> float:
    """Euler--Bernoulli limit load of an un-notched three-point bend beam.

    ``sigma_max = 1.5 P L / (b d^2) = sigma_f`` gives
    ``P = 2 b d^2 sigma_f / (3 L)``.  This is the analytical anchor for a
    *brittle* cohesive law (``l_ch / depth -> 0``), where the peak load
    coincides with first damage at the tensile face.
    """
    return 2.0 * thickness * depth**2 * tensile_strength / (3.0 * span)


def timoshenko_midspan_deflection(
    *,
    span: float,
    depth: float,
    thickness: float,
    young_modulus: float,
    poisson_ratio: float,
    load: float,
    shear_correction: float = 5.0 / 6.0,
) -> float:
    """Mid-span deflection of a three-point bend beam incl. shear deformation."""
    area = thickness * depth
    inertia = thickness * depth**3 / 12.0
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    bending = load * span**3 / (48.0 * young_modulus * inertia)
    shear = load * span / (4.0 * shear_correction * shear_modulus * area)
    return bending + shear