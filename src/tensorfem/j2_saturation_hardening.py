"""J2 elastoplasticity with *non-linear* isotropic hardening (Voce saturation).

The repository already owns a linear-hardening radial-return kernel
(:mod:`tensorfem.plasticity`, :mod:`tensorfem.j2_elastoplasticity`).  That
kernel closes the return mapping in one algebraic step because the yield stress
is linear in the equivalent plastic strain.  Real metal tests, however, show a
*decelerating* hardening branch, classically modelled by a Voce/saturation law

.. math::
    \\sigma_y(\\alpha) = \\sigma_{y0} + H\\,\\alpha + Q\\,(1 - e^{-b\\alpha}),

which makes the backward-Euler return mapping *implicit*: the incremental
plastic multiplier :math:`\\Delta\\gamma` is the root of

.. math::
    f(\\Delta\\gamma) = q_{tr} - 3G\\,\\Delta\\gamma
        - \\sigma_y(\\alpha_n + \\Delta\\gamma) = 0 .

Because :math:`f' = -3G - H(\\alpha) < 0` the root is unique, but it has no
closed form.  This module supplies

1. :class:`HardeningLaw` — the yield-stress law above (Voce when ``H = 0``,
   pure linear when ``Q = 0``) with fail-closed validation;
2. :func:`return_map` — safeguarded local Newton (with bisection bracket) for
   the implicit multiplier, plus a posteriori residuals of every constitutive
   identity;
3. :func:`consistent_tangent` — the algorithmic tangent obtained by
   differentiating the *discrete* return map, evaluated through the scalar
   implicit-function derivative :math:`\\mathrm{d}\\Delta\\gamma/\\mathrm{d}q_{tr}
   = 1/(3G + H(\\alpha_{n+1}))`; it is verified against finite differences of
   :func:`return_map` in the test-suite;
4. :func:`uniaxial_stress_path` — a stress-driven uniaxial driver that solves the
   lateral strain with the consistent tangent, and :func:`uniaxial_reference`,
   the closed-form 1-D solution :math:`\\sigma = \\sigma_y(\\alpha)`,
   :math:`\\varepsilon = \\sigma/E + \\alpha` obtained by an independent scalar
   root solve.

Conventions: symmetric ``3x3`` strain/stress tensors; Voigt order
``[xx, yy, zz, xy, yz, zx]`` with **engineering** shear strains
``gamma_ij = 2 eps_ij``.  Infinitesimal strain, isotropic elasticity, J2 yield,
isotropic (Voce + linear) hardening only: kinematic/combined hardening, rate
and thermal effects, damage and finite strain are **not** claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor

__all__ = [
    "HardeningLaw",
    "EPState",
    "ReturnResult",
    "VOIGT_INDEX",
    "voigt_map",
    "elastic_stiffness",
    "return_map",
    "return_map_residuals",
    "consistent_tangent",
    "uniaxial_reference",
    "uniaxial_stress_path",
]

VOIGT_INDEX = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


# --------------------------------------------------------------------------- #
# hardening law
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HardeningLaw:
    """Isotropic hardening ``sigma_y(a) = s0 + H a + Q (1 - exp(-b a))``."""

    yield_stress: float = 250.0e6
    linear: float = 0.0
    saturation: float = 0.0
    saturation_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.yield_stress > 0.0:
            raise ValueError("yield_stress must be positive")
        if self.linear < 0.0:
            raise ValueError("negative hardening modulus is outside this module's contract")
        if self.saturation < 0.0:
            raise ValueError("saturation stress must be non-negative")
        if self.saturation > 0.0 and not self.saturation_rate > 0.0:
            raise ValueError("a non-zero saturation stress requires saturation_rate > 0")
        if self.saturation_rate < 0.0:
            raise ValueError("saturation_rate must be non-negative")
        if self.saturation > 0.0 and self.linear > 0.0:
            # allowed, but the caller should know the tail is linear
            pass

    def sigma_y(self, alpha: Tensor | float) -> Tensor:
        a = torch.as_tensor(alpha, dtype=torch.float64) if not isinstance(alpha, Tensor) else alpha
        value = a.new_tensor(self.yield_stress) + self.linear * a
        if self.saturation != 0.0:
            value = value + self.saturation * (-torch.expm1(-self.saturation_rate * a))
        return value

    def slope(self, alpha: Tensor | float) -> Tensor:
        """``d sigma_y / d alpha`` (the hardening modulus at ``alpha``)."""
        a = torch.as_tensor(alpha, dtype=torch.float64) if not isinstance(alpha, Tensor) else alpha
        value = a.new_zeros(a.shape) + self.linear
        if self.saturation != 0.0:
            value = value + self.saturation * self.saturation_rate * torch.exp(
                -self.saturation_rate * a
            )
        return value

    @property
    def saturated_stress(self) -> float:
        """``sigma_y(inf)``; ``inf`` when the law keeps hardening linearly."""
        if self.linear > 0.0:
            return math.inf
        return self.yield_stress + self.saturation


@dataclass(frozen=True)
class EPState:
    """Committed material state: plastic strain tensor and equivalent plastic strain."""

    plastic_strain: Tensor
    alpha: Tensor

    @classmethod
    def virgin(cls, dtype=torch.float64, device=None) -> "EPState":
        return cls(torch.zeros((3, 3), dtype=dtype, device=device),
                   torch.zeros((), dtype=dtype, device=device))


@dataclass(frozen=True)
class ReturnResult:
    stress: Tensor
    state: EPState
    dgamma: Tensor
    trial_equivalent_stress: Tensor
    plastic: bool
    iterations: int
    converged: bool
    history: tuple = ()


def _as_voigt(t: Tensor) -> Tensor:
    return torch.stack([t[i, j] for i, j in VOIGT_INDEX])


def _from_voigt(v: Tensor) -> Tensor:
    t = v.new_zeros((3, 3))
    for k, (i, j) in enumerate(VOIGT_INDEX):
        t[i, j] = v[k] / (2.0 if k >= 3 else 1.0)
        t[j, i] = t[i, j]
    return t


voigt_map = (_as_voigt, _from_voigt)


def elastic_stiffness(young: float, poisson: float, dtype=torch.float64) -> Tensor:
    """Isotropic ``6x6`` stiffness in engineering-shear Voigt order."""
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    c = torch.zeros((6, 6), dtype=dtype)
    c[:3, :3] = lam
    c[0, 0] += 2.0 * mu
    c[1, 1] += 2.0 * mu
    c[2, 2] += 2.0 * mu
    c[3, 3] = c[4, 4] = c[5, 5] = mu
    return c


# --------------------------------------------------------------------------- #
# return mapping
# --------------------------------------------------------------------------- #
def return_map(
    strain: Tensor,
    state: EPState,
    law: HardeningLaw,
    young: float,
    poisson: float,
    *,
    tolerance: float = 1e-14,
    max_iterations: int = 60,
) -> ReturnResult:
    """Backward-Euler J2 radial return with implicit (non-linear) hardening.

    The scalar equation for ``dgamma`` is solved by Newton with a bisection
    safeguard on ``[0, q_tr / 3G]`` (``sigma_y >= 0``), so the iteration cannot
    leave the admissible bracket even for a nearly perfectly plastic law.
    """
    if strain.shape != (3, 3):
        raise ValueError("strain must be 3x3")
    if not young > 0.0 or not (-1.0 < poisson < 0.5):
        raise ValueError("invalid elastic constants")
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    eye = torch.eye(3, dtype=strain.dtype, device=strain.device)
    eps_e = strain - state.plastic_strain
    theta = torch.trace(eps_e)
    s_trial = 2.0 * G * (eps_e - theta / 3.0 * eye)
    q_trial = torch.sqrt(1.5 * torch.sum(s_trial * s_trial))
    flow_dir = torch.where(q_trial > 0.0, 1.5 * s_trial / q_trial, torch.zeros_like(s_trial))
    f_trial = q_trial - law.sigma_y(state.alpha)

    if float(f_trial) <= 0.0:  # elastic step: no state change
        stress = K * theta * eye + s_trial
        return ReturnResult(stress, state, torch.zeros((), dtype=strain.dtype, device=strain.device),
                            q_trial, False, 0, True)

    q = float(q_trial)
    lo, hi = 0.0, q / (3.0 * G)  # f(lo) > 0 >= f(hi) because sigma_y > 0
    dg = min(max(float(f_trial) / (3.0 * G + float(law.slope(state.alpha))), 1e-30), hi)
    iterations, converged, history = 0, False, []
    scale = max(q, 1.0)
    for iterations in range(1, max_iterations + 1):
        residual = q - 3.0 * G * dg - float(law.sigma_y(state.alpha + dg))
        history.append(abs(residual))
        if abs(residual) <= tolerance * scale:
            converged = True
            break
        if residual > 0.0:
            lo = dg
        else:
            hi = dg
        slope = 3.0 * G + float(law.slope(state.alpha + dg))
        candidate = dg + residual / slope
        dg = candidate if lo < candidate < hi else 0.5 * (lo + hi)
    alpha_new = state.alpha + float(dg)
    sigma_y_new = float(law.sigma_y(alpha_new))
    # radial return: s = s_trial - 2G dgamma * flow_dir  with  q = sigma_y
    shrink = sigma_y_new / q
    s_new = shrink * s_trial
    stress = K * theta * eye + s_new
    new_state = EPState(state.plastic_strain + dg * flow_dir,
                        state.alpha + dg)
    return ReturnResult(stress, new_state, torch.as_tensor(dg, dtype=strain.dtype),
                        q_trial, True, iterations, converged, tuple(history))


def return_map_residuals(
    strain: Tensor, state: EPState, law: HardeningLaw, young: float, poisson: float,
    result: ReturnResult | None = None,
) -> dict[str, float]:
    """Re-derive the discrete constitutive identities from the returned state.

    Nothing here trusts the return mapping: every entry is recomputed from the
    *returned* ``(stress, plastic_strain, alpha)`` triple.  Stresses are
    normalised by ``sigma_y0 + E ||eps||``-scale quantities and plastic strains
    by ``dgamma``, so a single threshold (the test-suite uses ``1e-10``) is
    meaningful for elastic and plastic steps alike.
    """
    result = result or return_map(strain, state, law, young, poisson)
    G = young / (2.0 * (1.0 + poisson))
    eye = torch.eye(3, dtype=strain.dtype)
    eps_e_n = strain - state.plastic_strain
    theta = torch.trace(eps_e_n)
    s_trial = 2.0 * G * (eps_e_n - theta / 3.0 * eye)
    q_trial = float(torch.sqrt(1.5 * torch.sum(s_trial * s_trial)))
    s = result.stress - torch.trace(result.stress) / 3.0 * eye
    q = torch.sqrt(1.5 * torch.sum(s * s))
    scale = float(law.yield_stress) + young * float(torch.linalg.vector_norm(strain))
    dg = float(result.dgamma)
    if dg == 0.0:
        return {"yield": float(abs(q_trial - float(law.sigma_y(state.alpha)))) / scale,
                "elastic_step": 1.0, "hardening_law": 0.0, "flow_rule": 0.0,
                "incompressibility": 0.0, "scalar_consistency": 0.0,
                "radial_return": 0.0}
    flow = 1.5 * s_trial / q_trial
    d_eps_p = result.state.plastic_strain - state.plastic_strain
    return {
        "yield": float(abs(q - law.sigma_y(result.state.alpha))) / scale,
        "hardening_law": abs(float(result.state.alpha) - float(state.alpha) - dg) / dg,
        "flow_rule": float(torch.linalg.vector_norm(d_eps_p - dg * flow)) / dg,
        "incompressibility": float(abs(torch.trace(result.state.plastic_strain))) / dg,
        "scalar_consistency": abs(q_trial - 3.0 * G * dg
                                 - float(law.sigma_y(result.state.alpha))) / scale,
        "radial_return": float(torch.linalg.vector_norm(
            s - (float(law.sigma_y(result.state.alpha)) / q_trial) * s_trial)) / scale,
    }


# --------------------------------------------------------------------------- #
# algorithmic (consistent) tangent
# --------------------------------------------------------------------------- #
def _voigt_basis(dtype=torch.float64):
    """Symmetric tensors ``E_b`` such that ``d eps = E_b d v_b`` for Voigt ``v``.

    Rows/columns use engineering shear strains, hence the ``1/2`` on the shear
    modes: an increment ``d gamma_xy`` means ``d eps_xy = d gamma_xy / 2``.
    """
    basis = []
    for k, (i, j) in enumerate(VOIGT_INDEX):
        e = torch.zeros((3, 3), dtype=dtype)
        if k < 3:
            e[i, j] = 1.0
        else:
            e[i, j] = e[j, i] = 0.5
        basis.append(e)
    return basis


def consistent_tangent(
    strain: Tensor, state: EPState, law: HardeningLaw, young: float, poisson: float,
    result: ReturnResult | None = None,
) -> Tensor:
    """Algorithmic tangent ``d sigma / d eps`` (``6x6``, engineering shear).

    Linearised form of the discrete radial return (Simo & Hughes, Box 3.2,
    generalised to a state-dependent hardening modulus):

    ``C = K 1x1 + 2G beta I_dev - (9 G^2 / q_tr^2) [1/A - dgamma/q_tr] s_tr x s_tr``

    with ``beta = 1 - 3 G dgamma / q_tr`` and ``A = 3G + H(alpha_{n+1})`` the
    derivative of the local scalar residual with respect to the plastic
    multiplier.  For the elastic branch the elastic stiffness is returned.
    """
    result = result or return_map(strain, state, law, young, poisson)
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    eye = torch.eye(3, dtype=strain.dtype)
    if not result.plastic:
        return elastic_stiffness(young, poisson, dtype=strain.dtype)

    eps_e = strain - state.plastic_strain
    theta = torch.trace(eps_e)
    s_trial = 2.0 * G * (eps_e - theta / 3.0 * eye)
    q_tr = float(torch.sqrt(1.5 * torch.sum(s_trial * s_trial)))
    dg = float(result.dgamma)
    beta = 1.0 - 3.0 * G * dg / q_tr
    alpha_new = float(result.state.alpha)
    A = 3.0 * G + float(law.slope(alpha_new))
    coefficient = (9.0 * G ** 2 / q_tr ** 2) * (1.0 / A - dg / q_tr)

    def operator(e: Tensor) -> Tensor:
        trace = torch.trace(e)
        dev = e - trace / 3.0 * eye
        return (K * trace * eye + 2.0 * G * beta * dev
                - coefficient * torch.sum(s_trial * e) * s_trial)

    basis = _voigt_basis(strain.dtype)
    return torch.stack([_as_voigt(operator(e)) for e in basis], dim=1)


# --------------------------------------------------------------------------- #
# 1-D stress-driven drivers
# --------------------------------------------------------------------------- #
def uniaxial_reference(
    total_strain: float, law: HardeningLaw, young: float, *, tolerance: float = 1e-15
) -> dict[str, float]:
    """Closed-form uniaxial state: ``sigma = sigma_y(alpha)``, ``eps = sigma/E + alpha``.

    ``alpha`` is obtained from the monotone scalar equation
    ``E (eps - alpha) - sigma_y(alpha) = 0`` by bisection + Newton polish, i.e.
    independently of the tensor return mapping under test.
    """
    elastic_limit = law.yield_stress / young
    if total_strain <= elastic_limit:
        return {"alpha": 0.0, "stress": young * total_strain,
                "plastic": 0.0 if total_strain < elastic_limit else 0.0}
    lo, hi = 0.0, total_strain - law.yield_stress / young
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        g = young * (total_strain - mid) - float(law.sigma_y(torch.tensor(mid, dtype=torch.float64)))
        if g > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tolerance * max(total_strain, 1e-30):
            break
    alpha = 0.5 * (lo + hi)
    stress = float(law.sigma_y(torch.tensor(alpha, dtype=torch.float64)))
    return {"alpha": alpha, "stress": stress, "plastic": 1.0}


def uniaxial_stress_path(
    strain_axial: Tensor, law: HardeningLaw, young: float, poisson: float, *,
    tolerance: float = 1e-12, max_iterations: int = 50,
) -> dict[str, object]:
    """Impose ``eps_11`` and relax ``eps_22 = eps_33`` until ``sigma_22 = sigma_33 = 0``.

    A genuinely multiaxial tangent exercise: the two lateral strains are the
    unknowns of a 2x2 Newton loop that is driven by the *consistent* tangent
    (:func:`consistent_tangent`), so the measured quadratic convergence is
    evidence that the tangent matches the return map.
    """
    path = torch.as_tensor(strain_axial, dtype=torch.float64).reshape(-1)
    state = EPState.virgin(dtype=torch.float64)
    stresses, alphas, lateral, iterations = [], [], [], []
    for eps11 in path:
        eps22 = torch.zeros((), dtype=torch.float64)
        eps33 = torch.zeros((), dtype=torch.float64)
        for count in range(1, max_iterations + 1):
            strain = torch.stack((
                torch.stack((eps11, torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64))),
                torch.stack((torch.zeros((), dtype=torch.float64), eps22, torch.zeros((), dtype=torch.float64))),
                torch.stack((torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.float64), eps33)),
            ))
            result = return_map(strain, state, law, young, poisson)
            residual = torch.stack((result.stress[1, 1], result.stress[2, 2]))
            reference = float(law.yield_stress)
            if float(torch.max(torch.abs(residual))) <= tolerance * reference:
                break
            tangent = consistent_tangent(strain, state, law, young, poisson, result)
            block = tangent[1:3, 1:3]
            delta = torch.linalg.solve(block, residual)
            eps22 = eps22 - delta[0]
            eps33 = eps33 - delta[1]
        else:
            raise RuntimeError("lateral relaxation did not converge")
        stresses.append(float(result.stress[0, 0]))
        alphas.append(float(result.state.alpha))
        lateral.append((float(eps22), float(eps33)))
        iterations.append(count)
        state = result.state
    return {"strain": path, "stress": torch.tensor(stresses, dtype=torch.float64),
            "alpha": torch.tensor(alphas, dtype=torch.float64),
            "lateral_strain": lateral, "iterations": iterations}