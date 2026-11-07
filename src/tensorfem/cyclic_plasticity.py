"""Cyclic plasticity at the material point: Armstrong-Frederick kinematic hardening.

This module is the *verifiable* cyclic-plasticity companion of
:mod:`tensorfem.j2_elastoplasticity` (the linear-isotropic-hardening J2 surface,
mission IN14-D4 / m4951).  It adds the three ingredients that a fatigue /
shakedown / ratcheting assessment needs and that the monotonic surface
explicitly refuses to claim:

1. **Nonlinear kinematic hardening (Armstrong-Frederick)**

   .. math::  \\dot{\\boldsymbol X} = \\tfrac{2}{3} C \\dot{\\boldsymbol\\varepsilon}^p
              - \\gamma \\boldsymbol X \\dot p ,\\qquad
              \\dot{\\boldsymbol\\varepsilon}^p = \\dot p\\,
              \\sqrt{\\tfrac32}\\,\\boldsymbol n

   with the J2 yield function written *exactly* as in the mission statement,
   ``f = ||s - X||_F - sqrt(2/3) * R(p)`` (Frobenius norm; ``||s||_F =
   sqrt(2/3) sigma_eq`` for uniaxial stress, hence ``f = 0 <=> sigma_eq =
   R(p)``).  Isotropic hardening is the linear/Voce mixture
   ``R(p) = sigma_y0 + H p + Q_inf (1 - exp(-b p))``.

2. **Implicit return mapping with the back stress inside the yield function.**
   Backward Euler in the *fully implicit* AF form
   ``X^{n+1} = (X^n + sqrt(2/3) C dp n) / (1 + gamma dp)`` makes the return
   mapping collapse onto one scalar, monotonically decreasing equation in
   ``dp`` (derived in :func:`integrate`)::

       Phi(dp) = || s_tr - X^n/(1+gamma dp) ||_F
                 - (mu + sqrt(2/3) C/(1+gamma dp)) dp
                 - sqrt(2/3) R(p^n + dp) = 0,        mu = sqrt(6) G.

   This is *not* the "X freeze" simplification: it reproduces the classical
   ``q = q_tr - 3G dp - (2/3)C dp`` result in the proportional limit and stays
   exact for non-proportional steps (``X^n`` need not be parallel to the trial
   direction).  Because the discretisation is fully implicit, ``X`` is
   contractive, which is what makes a stabilised hysteresis loop and a
   well-defined ratcheting rate exist at all.

3. **The analytic algorithmic tangent** including the ``dX/deps`` coupling
   (:func:`consistent_tangent`), linearised analytically from the same scalar
   equation -- so a Newton loop on a global FE system keeps its quadratic
   convergence in the presence of kinematic hardening.

Analytic anchors (all closed form, all machine checked in
``tests/test_cyclic_plasticity.py`` / ``scripts/cyclic_plasticity_verification.py``):

* stabilised uniaxial strain-controlled loop (``C/``gamma`` = saturation back
  stress ``X_sat``)::

      alpha_a = (3/2) X_11 = (C/gamma) tanh(gamma eps_p^a)
      sigma_a = alpha_a + sigma_y0 + R_sat
      eps_p^a = eps_a - sigma_a / E              (scalar root, 2 unknowns)

* dissipated energy per stabilised cycle::

      A = 4 sigma_y0 eps_p^a + (4C/gamma)(gamma eps_p^a - tanh(gamma eps_p^a))

* Bauschinger drop of the reverse yield stress (magnitude, measured against the
  virgin yield stress of the same material)::

      dsigma_B = sigma_y0 - |sigma_rev| = alpha_a

* stabilised ratcheting rate under uniaxial stress control (``P = P_f - P_r``)::

      A_ = sigma_m + sigma_a - sigma_y0,   B_ = sigma_m - sigma_a + sigma_y0
      d eps_p / cycle = (1/gamma) [ ln((C - gamma B_)/(C - gamma A_))
                                   - ln((C + gamma A_)/(C + gamma B_)) ]

Conventions are shared with :mod:`tensorfem.j2_elastoplasticity`: symmetric
``3x3`` tensors, Voigt order ``[xx, yy, zz, xy, yz, zx]`` with **engineering**
shear strains in the tangent columns.

Scope / limitations: infinitesimal strain, associative flow, isotropic
elasticity, one Armstrong-Frederick back stress (no multiple Chaboche terms),
``gamma >= 0`` (linear Prager kinematic hardening is the ``gamma = 0`` limit),
no rate/temperature dependence, no damage, no consistent ``C -> 0``
degeneration of the *tangent* of the m4951 module unless ``C == 0`` exactly
(see :func:`degenerate_check`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .plasticity import J2State, j2_consistent_tangent, update_j2

Tensor = torch.Tensor

__all__ = [
    "AFMaterial",
    "CyclicState",
    "VOIGT_PAIRS",
    "sqrt_2_3",
    "sqrt_3_2",
    "deviator",
    "voigt_from_tensor",
    "strain_from_voigt",
    "elastic_tangent",
    "yield_radius",
    "yield_slope",
    "integrate",
    "consistent_tangent",
    "uniaxial_cycle",
    "backstress_amplitude",
    "loop_peak_stress",
    "loop_dissipation",
    "bauschinger_drop",
    "ratcheting_rate",
    "degenerate_check",
    "uniaxial_stress_cycle",
    "strain_cycle",
    "stress_cycle",
]

_DTYPE = torch.float64
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))
_SF23 = math.sqrt(2.0 / 3.0)
_SF32 = math.sqrt(3.0 / 2.0)


def sqrt_2_3() -> float:
    """``sqrt(2/3)`` -- Frobenius <-> von Mises conversion for the yield radius."""
    return _SF23


def sqrt_3_2() -> float:
    """``sqrt(3/2)`` -- Frobenius <-> von Mises conversion for equivalent stress."""
    return _SF32


# --------------------------------------------------------------------------- #
# Material / state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AFMaterial:
    """Isotropic elasticity + J2 + Armstrong-Frederick kinematic + iso hardening.

    Parameters
    ----------
    young, poisson :
        Isotropic elasticity (small strain).
    yield_stress :
        Initial yield stress ``sigma_y0`` (> 0).
    kinematic_modulus :
        Armstrong-Frederick ``C`` (>= 0).  ``C = 0`` switches the kinematic
        branch off and must degenerate onto :mod:`tensorfem.plasticity`.
    kinematic_gamma :
        Armstrong-Frederick recall parameter ``gamma`` (>= 0).  The saturation
        back stress is ``X_sat = C/gamma`` (``3/2`` of it uniaxially).
    hardening :
        Linear isotropic hardening modulus ``H`` (>= 0).
    saturation, saturation_rate :
        Voce isotropic saturation ``Q_inf (1 - exp(-b p))`` (both >= 0).
        ``Q_inf = 0`` gives the purely linear m4951 isotropic law.
    """

    young: float
    poisson: float
    yield_stress: float
    kinematic_modulus: float = 0.0
    kinematic_gamma: float = 0.0
    hardening: float = 0.0
    saturation: float = 0.0
    saturation_rate: float = 0.0

    def __post_init__(self) -> None:
        if not (self.young > 0.0):
            raise ValueError("young must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must satisfy -1 < nu < 0.5")
        if not (self.yield_stress > 0.0):
            raise ValueError("yield_stress must be positive")
        if self.kinematic_modulus < 0.0:
            raise ValueError("kinematic_modulus C must be non-negative")
        if self.kinematic_gamma < 0.0:
            raise ValueError("kinematic_gamma gamma must be non-negative")
        if self.hardening < 0.0:
            raise ValueError("hardening H must be non-negative (no softening)")
        if self.saturation < 0.0 or self.saturation_rate < 0.0:
            raise ValueError("Voce saturation parameters must be non-negative")
        if self.saturation > 0.0 and self.saturation_rate <= 0.0:
            raise ValueError("saturation needs a positive saturation_rate")

    # -- elastic constants ---------------------------------------------------
    @property
    def shear(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def saturation_backstress(self) -> float:
        """``X_sat = C/gamma`` (Frobenius component); ``inf`` if ``gamma == 0``."""
        if self.kinematic_gamma <= 0.0:
            return math.inf if self.kinematic_modulus > 0.0 else 0.0
        return self.kinematic_modulus / self.kinematic_gamma

    def isotropic_saturated(self) -> float:
        """``sigma_y0 + Q_inf`` -- the limit yield radius (``H`` grows forever)."""
        return self.yield_stress + self.saturation


@dataclass(frozen=True)
class CyclicState:
    """Committed material state: plastic strain, back stress, accumulated ``p``."""

    plastic_strain: Tensor
    backstress: Tensor
    alpha: Tensor

    @staticmethod
    def zero(dtype=_DTYPE, device=None) -> "CyclicState":
        return CyclicState(
            plastic_strain=torch.zeros((3, 3), dtype=dtype, device=device),
            backstress=torch.zeros((3, 3), dtype=dtype, device=device),
            alpha=torch.zeros((), dtype=dtype, device=device),
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def deviator(tensor: Tensor) -> Tensor:
    """Deviatoric part ``t - tr(t)/3 I``."""
    return tensor - torch.trace(tensor) / 3.0 * torch.eye(
        3, dtype=tensor.dtype, device=tensor.device)


def voigt_from_tensor(tensor: Tensor) -> Tensor:
    """Pack a symmetric 3x3 tensor into ``[xx, yy, zz, xy, yz, zx]`` (no factors)."""
    if tuple(tensor.shape) != (3, 3):
        raise ValueError("expected a 3x3 tensor")
    return torch.stack([tensor[i, j] for i, j in VOIGT_PAIRS])


def strain_from_voigt(vector: Tensor) -> Tensor:
    """Unpack ``[xx, yy, zz, gamma_xy, gamma_yz, gamma_zx]`` into a 3x3 strain."""
    if tuple(vector.shape) != (6,):
        raise ValueError("expected a 6-vector")
    half = 0.5 * vector
    out = torch.zeros((3, 3), dtype=vector.dtype, device=vector.device)
    out[0, 0], out[1, 1], out[2, 2] = vector[0], vector[1], vector[2]
    out[0, 1] = out[1, 0] = half[3]
    out[1, 2] = out[2, 1] = half[4]
    out[2, 0] = out[0, 2] = half[5]
    return out


def elastic_tangent(material: AFMaterial, dtype=_DTYPE) -> Tensor:
    """Isotropic elastic 6x6 tangent (engineering shear columns)."""
    K, G = material.bulk, material.shear
    tangent = torch.zeros((6, 6), dtype=dtype)
    for i in range(3):
        for j in range(3):
            tangent[i, j] = K + (2.0 * G if i == j else 0.0) - 2.0 * G / 3.0
    tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G
    return tangent


def yield_radius(material: AFMaterial, p: float) -> float:
    """``R(p) = sigma_y0 + H p + Q_inf (1 - exp(-b p))``."""
    return (material.yield_stress + material.hardening * p
            + material.saturation * (1.0 - math.exp(-material.saturation_rate * p)))


def yield_slope(material: AFMaterial, p: float) -> float:
    """``dR/dp`` -- linear part plus Voce exponential tail."""
    return (material.hardening
            + material.saturation * material.saturation_rate
            * math.exp(-material.saturation_rate * p))


# --------------------------------------------------------------------------- #
# Return mapping
# --------------------------------------------------------------------------- #
def _flat(tensor: Tensor):
    """3x3 tensor -> 9 Python floats (row major), for the float-arithmetic core."""
    return tensor.reshape(9).tolist()


def _contract(u, v) -> float:
    """Full double contraction ``u : v`` of two 9-float row-major tensors."""
    return sum(a * b for a, b in zip(u, v))


def _norm(u) -> float:
    return math.sqrt(sum(a * a for a in u))


def _scalar_phi(material, s_tr, back_tr, p_old, dp, mu, c_sf23, gamma):
    """Scalar consistency function ``Phi(dp)`` (monotone decreasing in ``dp``)."""
    a = 1.0 / (1.0 + gamma * dp)
    q_w = 0.0
    for s, x in zip(s_tr, back_tr):
        w = s - a * x
        q_w += w * w
    q_w = math.sqrt(q_w)
    phi = (q_w - (mu + c_sf23 * a) * dp
           - _SF23 * yield_radius(material, p_old + dp))
    return phi, q_w, a


def _to_tensor(values, dtype, device) -> Tensor:
    return torch.tensor(values, dtype=dtype, device=device).reshape(3, 3)


def integrate(material: AFMaterial, strain: Tensor, state: CyclicState):
    """Implicit return mapping with Armstrong-Frederick back stress.

    Derivation (fully implicit backward Euler, ``a = 1/(1+gamma dp)``,
    ``mu = sqrt(6) G``)::

        s        = s_tr - mu dp n
        X^{n+1}  = a (X^n + sqrt(2/3) C dp n)
        eta      = s - X = [s_tr - a X^n] - [mu + sqrt(2/3) C a] dp n
        => (q_F + B dp) n = w,  w = s_tr - a X^n,  B = mu + sqrt(2/3) C a
        => n = w/||w||_F  and  q_F = ||w||_F - B dp            (exact, any X^n)

    so the yield condition ``||eta||_F = sqrt(2/3) R(p + dp)`` becomes the single
    scalar equation solved here by a safeguarded Newton iteration (analytic
    derivative, bisection fallback).  ``X^n`` need not be parallel to the trial
    direction, so non-proportional steps are exact too.

    The scalar core runs in Python floats (torch is only used at the boundaries);
    the branch taken is the one that was actually integrated, so
    :func:`consistent_tangent` can be called with the returned ``dp``.
    Returns ``(stress, new_state, dp)``.
    """
    if tuple(strain.shape) != (3, 3):
        raise ValueError("strain must be a 3x3 tensor")
    if tuple(state.plastic_strain.shape) != (3, 3) or tuple(state.backstress.shape) != (3, 3):
        raise ValueError("state tensors must be 3x3")
    dtype, device = strain.dtype, strain.device
    G, K = material.shear, material.bulk
    C, gamma = material.kinematic_modulus, material.kinematic_gamma
    mu = math.sqrt(6.0) * G
    c_sf23 = _SF23 * C

    e_tr = _flat(strain - state.plastic_strain)
    trace_e = e_tr[0] + e_tr[4] + e_tr[8]
    third = trace_e / 3.0
    two_g = 2.0 * G
    s_tr = [two_g * (value - (third if k in (0, 4, 8) else 0.0))
            for k, value in enumerate(e_tr)]
    back_tr = _flat(state.backstress)
    p_old = float(state.alpha)

    q_trial = _norm([s - x for s, x in zip(s_tr, back_tr)])
    radius_old = yield_radius(material, p_old)
    pressure = K * trace_e
    if q_trial - _SF23 * radius_old <= 0.0:  # elastic step
        stress = [s_tr[k] + (pressure if k in (0, 4, 8) else 0.0) for k in range(9)]
        return _to_tensor(stress, dtype, device), state, torch.zeros((), dtype=dtype, device=device)

    # --- initial guess: linearised predictor (exact for the m4951 limit) -----
    slope_old = yield_slope(material, p_old)
    dp = max((q_trial - _SF23 * radius_old) / (mu + c_sf23 + _SF23 * slope_old), 1e-30)

    # --- bracket the root: Phi(0) > 0 > Phi(hi) ------------------------------
    lo = 0.0
    hi = max(2.0 * dp, q_trial / mu)
    for _ in range(200):
        phi_hi, _, _ = _scalar_phi(material, s_tr, back_tr, p_old, hi, mu, c_sf23, gamma)
        if phi_hi <= 0.0:
            break
        hi *= 2.0
    else:  # pragma: no cover - cannot happen for admissible parameters
        raise RuntimeError("failed to bracket the plastic multiplier")

    tol = 1e-15 * max(q_trial, 1.0)
    for _ in range(100):
        phi, q_w, a = _scalar_phi(material, s_tr, back_tr, p_old, dp, mu, c_sf23, gamma)
        if abs(phi) <= tol:
            break
        if phi > 0.0:
            lo = dp
        else:
            hi = dp
        if not (q_w > 0.0):  # pragma: no cover
            raise RuntimeError("degenerate plastic direction")
        n = [(s - a * x) / q_w for s, x in zip(s_tr, back_tr)]
        dq = gamma * a * a * _contract(n, back_tr)  # d||w||/ddp  (a decreases with dp)
        dBdp = mu + c_sf23 * (a - gamma * a * a * dp)  # d(B dp)/ddp
        dphi = dq - dBdp - _SF23 * yield_slope(material, p_old + dp)
        candidate = dp - phi / dphi if dphi != 0.0 else 0.5 * (lo + hi)
        if not (lo < candidate < hi) or not math.isfinite(candidate):
            candidate = 0.5 * (lo + hi)
        if abs(candidate - dp) <= 1e-17 * max(1.0, dp):
            dp = candidate
            break
        dp = candidate

    a = 1.0 / (1.0 + gamma * dp)
    w_l = [s - a * x for s, x in zip(s_tr, back_tr)]
    q_w = _norm(w_l)
    n = [w / q_w for w in w_l]
    scale = _SF32 * dp
    plastic = _flat(state.plastic_strain)
    new_plastic = [plastic[k] + scale * n[k] for k in range(9)]
    new_back = [a * (back_tr[k] + c_sf23 * dp * n[k]) for k in range(9)]
    stress = [s_tr[k] - mu * dp * n[k] + (pressure if k in (0, 4, 8) else 0.0)
              for k in range(9)]
    new_state = CyclicState(plastic_strain=_to_tensor(new_plastic, dtype, device),
                            backstress=_to_tensor(new_back, dtype, device),
                            alpha=state.alpha + dp)
    return (_to_tensor(stress, dtype, device), new_state,
            torch.as_tensor(dp, dtype=dtype, device=device))


def consistent_tangent(material: AFMaterial, strain: Tensor, state: CyclicState,
                       dp: Tensor) -> Tensor:
    """Analytic algorithmic tangent ``d sigma / d eps`` for :func:`integrate`.

    Linearises the scalar consistency equation *and* the back-stress update along
    the branch that was actually integrated (elastic or plastic)::

        dp   = (n : ds_tr) / D
        D    = mu + sqrt(2/3) C (a - gamma a^2 dp) + sqrt(2/3) R'(p+dp)
               + gamma a^2 (n : X^n)
        ds   = ds_tr - mu (n dp + dp dn),   dn = (I - n (x) n) : dw / q_w
        dw   = ds_tr + gamma a^2 X^n dp,    dsigma = K tr(deps) I + ds

    Voigt order ``[xx, yy, zz, xy, yz, zx]`` with engineering shear columns
    (``d eps_ij = d gamma_ij / 2``), so it can be dropped straight into a global
    stiffness matrix.
    """
    if tuple(strain.shape) != (3, 3):
        raise ValueError("strain must be a 3x3 tensor")
    dtype, device = strain.dtype, strain.device
    G, K = material.shear, material.bulk
    C, gamma = material.kinematic_modulus, material.kinematic_gamma
    mu = math.sqrt(6.0) * G
    c_sf23 = _SF23 * C
    dpf = float(dp)

    if dpf == 0.0:  # elastic branch
        tangent = torch.zeros((6, 6), dtype=dtype, device=device)
        for i in range(3):
            for j in range(3):
                tangent[i, j] = K + (2.0 * G if i == j else 0.0) - 2.0 * G / 3.0
        tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G
        return tangent

    e_tr = _flat(strain - state.plastic_strain)
    trace_e = e_tr[0] + e_tr[4] + e_tr[8]
    third = trace_e / 3.0
    two_g = 2.0 * G
    s_tr = [two_g * (value - (third if k in (0, 4, 8) else 0.0))
            for k, value in enumerate(e_tr)]
    x_l = _flat(state.backstress)
    p_old = float(state.alpha)
    a = 1.0 / (1.0 + gamma * dpf)
    w_l = [s - a * x for s, x in zip(s_tr, x_l)]
    q_w = _norm(w_l)
    n_l = [w / q_w for w in w_l]
    gaa = gamma * a * a
    D = (mu + c_sf23 * (a - gaa * dpf)
         + _SF23 * yield_slope(material, p_old + dpf) - gaa * _contract(n_l, x_l))

    tangent = [[0.0] * 6 for _ in range(6)]
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        # deviatoric elastic trial response to the k-th Voigt strain direction
        ds_tr = [0.0] * 9
        if i == j:
            for p in range(3):
                for q in range(3):
                    ds_tr[3 * p + q] = two_g * ((1.0 if (p == i and q == j) else 0.0)
                                                - (1.0 / 3.0 if p == q else 0.0))
        else:
            ds_tr[3 * i + j] += G  # 2G * (1/2) on the two symmetric entries
            ds_tr[3 * j + i] += G
        d_dp = _contract(n_l, ds_tr) / D
        dw = [v + gaa * x * d_dp for v, x in zip(ds_tr, x_l)]  # dw/ddp = +gamma a^2 X^n
        n_dw = _contract(n_l, dw)
        dn = [(v - nv * n_dw) / q_w for v, nv in zip(dw, n_l)]
        ds = [v - mu * (nv * d_dp + dpf * dv) for v, nv, dv in zip(ds_tr, n_l, dn)]
        trace_part = K if i == j else 0.0
        for m, (p, q) in enumerate(VOIGT_PAIRS):
            tangent[m][k] = ds[3 * p + q] + (trace_part if p == q else 0.0)
    return torch.tensor(tangent, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
# Uniaxial driver (strain controlled, sigma_22 = sigma_33 = 0)
# --------------------------------------------------------------------------- #
def uniaxial_cycle(material: AFMaterial, strain_path, state: CyclicState | None = None,
                   dtype=_DTYPE, tol: float = 1.0e-9, max_iter: int = 60,
                   collect_states: bool = False):
    """Integrate a strain-controlled uniaxial history with ``sigma_22 = sigma_33 = 0``.

    ``strain_path`` is a sequence of axial strains ``eps_11`` (the path is
    traversed in order; make it non-monotone for cycles).  The lateral strain is
    found at every increment with a Newton loop driven by the analytic tangent,
    so the returned stresses satisfy the uniaxial-stress condition to ``tol``
    (relative to the current axial stress).

    Returns a dict of equal-length lists: ``eps`` (axial), ``sigma`` (axial),
    ``lateral`` (``eps_22 = eps_33``), ``p`` (accumulated plastic strain),
    ``backstress`` (``X_11``), ``alpha_back`` (``1.5 X_11``), ``dp`` (increment),
    ``tangent_axial`` (``d sigma_11 / d eps_11`` on the uniaxial-stress branch)
    and ``iterations`` (lateral Newton iterations).
    """
    state = state or CyclicState.zero(dtype=dtype)
    recorded = {key: [] for key in ("eps", "sigma", "lateral", "p", "backstress",
                                    "alpha_back", "dp", "tangent_axial", "iterations")}
    if collect_states:
        #: committed state *before* each increment (index k), needed to re-solve the
        #: elastic/plastic boundary inside increment k during the yield-point search
        recorded["states"] = []
    lateral = -material.poisson * 0.0
    for eps_11 in strain_path:
        eps_11 = float(eps_11)
        target = torch.zeros((3, 3), dtype=dtype)
        target[0, 0] = eps_11
        target[1, 1] = target[2, 2] = lateral
        # Newton on the lateral strain so that sigma_22 == 0
        if collect_states:
            recorded["states"].append(state)
        it = 0
        for it in range(1, max_iter + 1):
            stress, new_state, dp = integrate(material, target, state)
            residual = float(stress[1, 1])
            scale = max(abs(float(stress[0, 0])), 1.0)
            if abs(residual) <= tol * scale:
                break
            tangent = consistent_tangent(material, target, state, dp)
            # d sigma_22 / d eps_lateral = Ct[yy,yy] + Ct[yy,zz]
            slope = float(tangent[1, 1] + tangent[1, 2])
            if abs(slope) < 1e-30:  # pragma: no cover
                raise RuntimeError("singular lateral tangent")
            lateral -= residual / slope
            target[1, 1] = target[2, 2] = lateral
        else:  # pragma: no cover - Newton is quasi-linear here
            raise RuntimeError(f"lateral Newton failed at eps_11={eps_11}")

        tangent = consistent_tangent(material, target, state, dp)
        stress, new_state, dp = integrate(material, target, state)
        residual = float(stress[1, 1])
        scale = max(abs(float(stress[0, 0])), 1.0)
        if abs(residual) > 10.0 * tol * scale:
            raise RuntimeError("uniaxial condition violated after Newton")
        # uniaxial-stress condensed axial modulus
        axial = float(tangent[0, 0]
                      - (tangent[0, 1] + tangent[0, 2]) * (tangent[1, 0] + tangent[2, 0])
                      / (tangent[1, 1] + tangent[1, 2] + tangent[2, 1] + tangent[2, 2]))
        recorded["eps"].append(eps_11)
        recorded["sigma"].append(float(stress[0, 0]))
        recorded["lateral"].append(lateral)
        recorded["p"].append(float(new_state.alpha))
        recorded["backstress"].append(float(new_state.backstress[0, 0]))
        recorded["alpha_back"].append(1.5 * float(new_state.backstress[0, 0]))
        recorded["dp"].append(float(dp))
        recorded["tangent_axial"].append(axial)
        recorded["iterations"].append(it)
        state = new_state
    recorded["state"] = state
    return recorded


def cyclic_history(initial: float, amplitude: float, cycles: int, steps_per_half: int):
    """Triangular history with clean per-cycle bounds (both ends at ``+amplitude``).

    Returns ``(path, bounds)`` where ``bounds[k] = (start, end)`` indexes one
    closed cycle: it begins at the tensile peak and returns to it, so the net
    quantity accumulated over ``bounds[k]`` is the per-cycle increment.
    ``initial`` is the first (strain or stress) value before the ramp to the peak.
    """
    path = [initial + (amplitude - initial) * (i + 1) / steps_per_half
            for i in range(steps_per_half)]
    bounds = []
    for _ in range(cycles):
        start = len(path)
        for i in range(2 * steps_per_half):
            path.append(amplitude - 2.0 * amplitude * (i + 1) / (2 * steps_per_half))
        for i in range(2 * steps_per_half):
            path.append(-amplitude + 2.0 * amplitude * (i + 1) / (2 * steps_per_half))
        bounds.append((start, len(path)))
    return path, bounds


def strain_cycle(amplitude: float, cycles: int, steps_per_half: int):
    """Triangular strain history ``0 -> +a -> -a -> +a ...`` (``cycles`` loops)."""
    path, _ = cyclic_history(0.0, amplitude, cycles, steps_per_half)
    return path


def stress_cycle(mean: float, amplitude: float, cycles: int, steps_per_half: int = 200):
    """Triangular stress history ``mean -> +a -> -a -> +a ...`` (absolute stresses)."""
    path, _ = cyclic_history(mean, amplitude, cycles, steps_per_half)
    return [mean + value for value in path]


# --------------------------------------------------------------------------- #
# Closed-form anchors
# --------------------------------------------------------------------------- #
def backstress_amplitude(material: AFMaterial, plastic_amplitude: float) -> float:
    """``alpha_a = (3/2) X_11`` amplitude of the stabilised loop: ``(C/gamma) tanh(gamma eps_p^a)``."""
    C, gamma = material.kinematic_modulus, material.kinematic_gamma
    if C == 0.0:
        return 0.0
    if gamma == 0.0:  # Prager limit
        return C * plastic_amplitude
    return (C / gamma) * math.tanh(gamma * plastic_amplitude)


def loop_peak_stress(material: AFMaterial, strain_amplitude: float,
                     state_saturated: bool = True, p_at_peak: float | None = None,
                     tol: float = 1e-14) -> dict:
    """Closed-form peak/trough stress and plastic strain amplitude of a stabilised loop.

    Solves the two scalar equations
    ``sigma_a = alpha_a(eps_p^a) + R(p_peak)`` and
    ``eps_p^a = eps_a - sigma_a/E`` by a safeguarded Newton iteration on
    ``eps_p^a``.  ``p_at_peak`` (if given) is the accumulated plastic strain at
    the tensile peak -- needed only when ``R`` has not saturated (``H > 0`` or
    ``Q_inf exp(-b p)`` still significant); the default uses
    ``sigma_y0 + Q_inf`` (fully saturated isotropic hardening).
    """
    E = material.young
    if p_at_peak is None or state_saturated:
        radius = material.yield_stress + material.saturation
    else:
        radius = yield_radius(material, p_at_peak)
    lo, hi = 0.0, strain_amplitude

    def resid(eps_p_a: float) -> float:
        sigma_a = backstress_amplitude(material, eps_p_a) + radius
        return eps_p_a - (strain_amplitude - sigma_a / E)

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        value = resid(mid)
        if value > 0.0:  # too much plastic strain -> sigma too large
            hi = mid
        else:
            lo = mid
    eps_p_a = 0.5 * (lo + hi)
    alpha_a = backstress_amplitude(material, eps_p_a)
    sigma_a = alpha_a + radius
    return {
        "plastic_amplitude": eps_p_a,
        "backstress_amplitude": alpha_a,
        "stress_amplitude": sigma_a,
        "peak": sigma_a,
        "trough": -sigma_a,
        "radius": radius,
    }


def loop_dissipation(material: AFMaterial, strain_amplitude: float,
                     plastic_amplitude: float | None = None) -> float:
    """Closed-form energy per stabilised cycle.

    ``A = 4 sigma_y eps_p^a + (4C/gamma^2)(gamma eps_p^a - tanh(gamma eps_p^a))``;
    the second term vanishes as ``gamma -> 0`` (the Prager limit is a parallelogram
    of area ``4 sigma_y eps_p^a``).  Requires ``eps_p^a`` from
    :func:`loop_peak_stress` when the material also has isotropic hardening.
    """
    if plastic_amplitude is None:
        plastic_amplitude = loop_peak_stress(material, strain_amplitude)["plastic_amplitude"]
    C, gamma = material.kinematic_modulus, material.kinematic_gamma
    base = 4.0 * (material.yield_stress + material.saturation) * plastic_amplitude
    if C == 0.0:
        return base
    if gamma == 0.0:  # Prager: the loop is a parallelogram of height 2 sigma_y
        return base
    g = gamma * plastic_amplitude
    # (4C/gamma^2)(gamma eps_p^a - tanh(gamma eps_p^a)) -- see the module docstring
    return base + (4.0 * C / (gamma * gamma)) * (g - math.tanh(g))


def bauschinger_drop(material: AFMaterial, strain_amplitude: float) -> dict:
    """Closed-form Bauschinger data of the stabilised loop.

    ``dsigma_B = sigma_y0 - |sigma_rev| = alpha_a`` with
    ``sigma_rev = alpha_a - R_sat`` (reverse yield stress of the stabilised loop).
    Also returns ``ratio = dsigma_B / sigma_y0`` used as the detectability gate.
    """
    anchor = loop_peak_stress(material, strain_amplitude)
    alpha_a = anchor["backstress_amplitude"]
    sigma_rev = alpha_a - anchor["radius"]
    return {
        "drop": alpha_a,
        "radius": anchor["radius"],
        "reverse_yield": sigma_rev,
        "peak": anchor["peak"],
        "ratio": alpha_a / anchor["radius"],
        "plastic_amplitude": anchor["plastic_amplitude"],
    }


def ratcheting_rate(material: AFMaterial, mean_stress: float, stress_amplitude: float,
                    yield_stress: float | None = None) -> float:
    """Closed-form stabilised plastic strain increment per uniaxial stress cycle.

    ``A_ = sigma_m + sigma_a - sigma_y`` and ``B_ = sigma_m - sigma_a + sigma_y``
    are the back-stress endpoints of the steady cycle; both branches integrate
    analytically to logarithms.  Requires ``max(|A_|, |B_|) < C/gamma`` (the
    Armstrong-Frederick saturation capacity) -- otherwise no steady ratcheting
    cycle exists and ``inf`` is returned.
    """
    C, gamma = material.kinematic_modulus, material.kinematic_gamma
    if C <= 0.0:
        return 0.0  # no back stress -> no ratcheting (Prager/AF absent)
    radius = material.yield_stress if yield_stress is None else float(yield_stress)
    upper = mean_stress + stress_amplitude - radius
    lower = mean_stress - stress_amplitude + radius
    capacity = C / gamma if gamma > 0.0 else math.inf
    if max(abs(upper), abs(lower)) >= capacity:
        return math.inf
    if gamma == 0.0:  # linear Prager kinematic hardening: no ratcheting
        return 0.0
    forward = math.log((C - gamma * lower) / (C - gamma * upper))
    reverse = math.log((C + gamma * upper) / (C + gamma * lower))
    return (forward - reverse) / gamma


def degenerate_check(material: AFMaterial, strain: Tensor,
                     state: CyclicState | None = None) -> dict:
    """``C -> 0`` degeneration onto the m4951 isotropic kernel (bit-level)."""
    state = state or CyclicState.zero(dtype=strain.dtype)
    legacy = J2State(plastic_strain=state.plastic_strain.clone(),
                     alpha=state.alpha.clone())
    sigma_legacy, legacy_state, dgamma = update_j2(
        strain, material.young, material.poisson, material.yield_stress,
        material.hardening, legacy)
    af = AFMaterial(young=material.young, poisson=material.poisson,
                    yield_stress=material.yield_stress, kinematic_modulus=0.0,
                    kinematic_gamma=0.0, hardening=material.hardening)
    sigma_af, af_state, dp = integrate(af, strain, state)
    tangent_af = consistent_tangent(af, strain, state, dp)
    tangent_legacy = j2_consistent_tangent(strain, af.young, af.poisson,
                                           af.hardening, legacy, dgamma)
    scale = max(float(sigma_legacy.abs().max()), 1.0)
    scale_t = max(float(tangent_legacy.abs().max()), 1.0)
    return {
        "stress_error": float((sigma_af - sigma_legacy).abs().max()) / scale,
        "dp_error": abs(float(dp) - float(dgamma)) / max(abs(float(dgamma)), 1e-300),
        "plastic_error": float((af_state.plastic_strain
                                - legacy_state.plastic_strain).abs().max()) / max(
            float(legacy_state.plastic_strain.abs().max()), 1e-300),
        "alpha_error": abs(float(af_state.alpha) - float(legacy_state.alpha))
        / max(abs(float(legacy_state.alpha)), 1e-300),
        "tangent_error": float((tangent_af - tangent_legacy).abs().max()) / scale_t,
        "stress": sigma_af,
        "state": af_state,
        "tangent": tangent_af,
    }

def _uniaxial_strain_tensor(eps_11: float, lateral: float, dtype=_DTYPE) -> Tensor:
    tensor = torch.zeros((3, 3), dtype=dtype)
    tensor[0, 0] = eps_11
    tensor[1, 1] = tensor[2, 2] = lateral
    return tensor


def _axial_modulus(tangent: Tensor) -> float:
    """Condensed ``d sigma_11 / d eps_11`` of the uniaxial-stress branch."""
    den = float(tangent[1, 1] + tangent[1, 2] + tangent[2, 1] + tangent[2, 2])
    if abs(den) < 1e-30:  # pragma: no cover
        raise RuntimeError("singular lateral block in the tangent")
    return float(tangent[0, 0]
                 - (tangent[0, 1] + tangent[0, 2]) * (tangent[1, 0] + tangent[2, 0]) / den)


def solve_uniaxial_point(material: AFMaterial, state: CyclicState, eps_11: float,
                         lateral: float = 0.0, dtype=_DTYPE, tol: float = 1.0e-12,
                         max_iter: int = 40):
    """One strain-controlled uniaxial-stress point: lateral Newton at fixed state.

    Returns ``(sigma_11, new_state, dp, lateral)`` with ``sigma_22 = 0`` enforced
    to ``tol`` (relative).  Used by the cyclic drivers and by the yield-point
    bisection in the verification script.
    """
    for _ in range(max_iter):
        strain = _uniaxial_strain_tensor(eps_11, lateral, dtype=dtype)
        stress, new_state, dp = integrate(material, strain, state)
        residual = float(stress[1, 1])
        scale = max(abs(float(stress[0, 0])), material.yield_stress)
        if abs(residual) <= tol * scale:
            return float(stress[0, 0]), new_state, dp, lateral
        tangent = consistent_tangent(material, strain, state, dp)
        slope = float(tangent[1, 1] + tangent[1, 2])
        lateral -= residual / slope
    raise RuntimeError(f"lateral Newton failed at eps_11={eps_11}")  # pragma: no cover


def uniaxial_stress_cycle(material: AFMaterial, stress_path, state: CyclicState | None = None,
                          dtype=_DTYPE, tol: float = 1.0e-10, max_iter: int = 60):
    """Integrate a stress-controlled uniaxial history (``sigma_22 = sigma_33 = 0``).

    Two unknown strains ``(eps_11, eps_22 = eps_33)`` are solved per increment by
    a 2x2 Newton loop with the analytic tangent; used for ratcheting.
    Returns a dict with ``sigma``, ``eps``, ``eps_plastic`` (``eps_p_11``),
    ``plastic_path`` (accumulated ``p``), ``dp``, ``backstress``,
    ``alpha_back`` and ``iterations``.
    """
    state = state or CyclicState.zero(dtype=dtype)
    keys = ("sigma", "eps", "eps_plastic", "plastic_path", "dp", "backstress",
            "alpha_back", "iterations")
    recorded = {key: [] for key in keys}
    eps_11 = 0.0
    lateral = 0.0
    for sigma_target in stress_path:
        sigma_target = float(sigma_target)
        it = 0
        for it in range(1, max_iter + 1):
            strain = _uniaxial_strain_tensor(eps_11, lateral, dtype=dtype)
            stress, new_state, dp = integrate(material, strain, state)
            r1 = float(stress[0, 0]) - sigma_target
            r2 = float(stress[1, 1])
            scale = max(abs(sigma_target), material.yield_stress, 1.0)
            if abs(r1) <= tol * scale and abs(r2) <= tol * scale:
                break
            tangent = consistent_tangent(material, strain, state, dp)
            j11 = float(tangent[0, 0])
            j12 = float(tangent[0, 1] + tangent[0, 2])
            j21 = float(tangent[1, 0])
            j22 = float(tangent[1, 1] + tangent[1, 2])
            det = j11 * j22 - j12 * j21
            if not math.isfinite(det) or abs(det) < 1e-12 * max(abs(j11 * j22),
                                                               abs(j12 * j21)):
                # degenerate Jacobian: fall back to a damped steepest-descent step
                d_eps = r1 / max(abs(j11), 1e-30)
                d_lat = r2 / max(abs(j22), 1e-30)
            else:
                d_eps = -(j22 * r1 - j12 * r2) / det
                d_lat = -(-j21 * r1 + j11 * r2) / det
            # backtracking: the trial solve is only mildly nonlinear, but a bad
            # iterate must never be amplified (bounded plastic multiplier)
            scale_u = max(abs(eps_11), abs(lateral), 1e-8)
            factor = 1.0
            while (not math.isfinite(d_eps) or not math.isfinite(d_lat)
                   or abs(d_eps) > 10.0 * scale_u or abs(d_lat) > 10.0 * scale_u):
                factor *= 0.5
                if factor < 1e-6:  # pragma: no cover
                    raise RuntimeError("stress Newton step rejected")
                d_eps *= 0.5
                d_lat *= 0.5
            eps_11 += d_eps
            lateral += d_lat
        else:  # pragma: no cover
            raise RuntimeError(f"stress Newton failed at sigma={sigma_target}")
        strain = _uniaxial_strain_tensor(eps_11, lateral, dtype=dtype)
        stress, new_state, dp = integrate(material, strain, state)
        recorded["sigma"].append(float(stress[0, 0]))
        recorded["eps"].append(eps_11)
        recorded["eps_plastic"].append(float(new_state.plastic_strain[0, 0]))
        recorded["plastic_path"].append(float(new_state.alpha))
        recorded["dp"].append(float(dp))
        recorded["backstress"].append(float(new_state.backstress[0, 0]))
        recorded["alpha_back"].append(1.5 * float(new_state.backstress[0, 0]))
        recorded["iterations"].append(it)
        state = new_state
    recorded["state"] = state
    return recorded
