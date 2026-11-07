"""J2 (von Mises) small-strain elastoplasticity: minimum verifiable surface.

This module is the *material-point* entry point for classical metal plasticity:
linear isotropic hardening, associative flow rule, backward-Euler radial return
mapping and the closed-form algorithmic (consistent) tangent.  It deliberately
reuses the return-mapping kernel in :mod:`tensorfem.plasticity` (single source of
truth) and adds the three things a qualification surface needs:

1. **Fail-closed material validation** — non-positive Young's modulus or yield
   stress, non-physical Poisson ratio and a *negative* hardening modulus are
   rejected at construction time (a softening modulus is outside this module's
   contract: it is not a local well-posedness claim, see Limitations).
2. **A posteriori residuals** — after every return mapping the yield condition,
   the radial-return stress identity, the flow rule, the hardening law and the
   complementarity condition are re-evaluated from the *returned* state, so the
   integration can be audited without trusting the algorithm that produced it.
3. **Analytic anchors** — a uniaxial-stress driver (lateral strains solved with
   the consistent tangent so that ``sigma_22 = sigma_33 = 0``) reproducing the
   piecewise closed form (elastic slope ``E``, plastic slope ``EH/(E+H)``) and a
   pure-shear / general-direction yield probe compared with the analytic yield
   surface ``q = sigma_y``.

Conventions: strains and stresses are symmetric ``3x3`` tensors; Voigt order is
``[xx, yy, zz, xy, yz, zx]`` with **engineering** shear strains
``gamma_ij = 2 eps_ij``, so the returned ``6x6`` tangent maps
``d[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]`` to
``d[sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_zx]``.

Scope: infinitesimal strain, isotropic linear elasticity, J2 yield surface,
linear isotropic hardening, monotonic and non-proportional proportional-step
paths.  Cyclic hardening, kinematic/combined hardening, rate dependence,
thermal softening, finite strain and damage are **not** claimed here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from .plasticity import J2State, j2_consistent_tangent, update_j2

Tensor = torch.Tensor

__all__ = [
    "J2Material",
    "J2Return",
    "J2State",
    "VOIGT_PAIRS",
    "strain_from_voigt",
    "voigt_from_tensor",
    "deviator",
    "equivalent_stress",
    "elastic_tangent",
    "radial_return",
    "return_map_residuals",
    "uniaxial_stress_path",
    "yield_probe",
    "pure_shear_yield_point",
    "verify_minimum_surface",
]

#: Voigt index pairs for the symmetric 3x3 <-> 6 convention used everywere here.
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))

_DTYPE = torch.float64
_TINY = 1.0e-300


# --------------------------------------------------------------------------- #
# Tensor / Voigt plumbing
# --------------------------------------------------------------------------- #
def _eye(dtype=_DTYPE, device=None) -> Tensor:
    return torch.eye(3, dtype=dtype, device=device)


def voigt_from_tensor(tensor: Tensor) -> Tensor:
    """Pack a symmetric 3x3 strain/stress tensor into ``[xx,yy,zz,xy,yz,zx]``.

    The *strain* caller is responsible for the engineering-shear factor; this
    helper is a pure index gather and is used for stresses and for the tangent
    outer products, where no factor is wanted.
    """
    if tuple(tensor.shape) != (3, 3):
        raise ValueError("expected a 3x3 tensor")
    return torch.stack([tensor[i, j] for i, j in VOIGT_PAIRS])


def strain_from_voigt(vector: Tensor) -> Tensor:
    """Unpack ``[exx,eyy,ezz,gxy,gyz,gzx]`` (engineering shear) into a 3x3 tensor."""
    if tuple(vector.shape) != (6,):
        raise ValueError("expected a length-6 Voigt vector")
    e = torch.zeros((3, 3), dtype=vector.dtype, device=vector.device)
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        e[i, j] = e[j, i] = vector[k] if i == j else 0.5 * vector[k]
    return e


def deviator(tensor: Tensor) -> Tensor:
    """Deviatoric part of a symmetric 3x3 tensor."""
    return tensor - torch.trace(tensor) / 3.0 * _eye(tensor.dtype, tensor.device)


def equivalent_stress(tensor: Tensor) -> Tensor:
    """von Mises equivalent stress ``q = sqrt(3/2 s:s)`` of a stress tensor."""
    s = deviator(tensor)
    return torch.sqrt(1.5 * torch.sum(s * s))


def elastic_tangent(young: float, poisson: float, *, dtype=_DTYPE) -> Tensor:
    """Isotropic 6x6 elasticity matrix in the Voigt convention above."""
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    Id = _eye(dtype)
    C = torch.zeros((6, 6), dtype=dtype)
    C[:3, :3] = (K - 2.0 * G / 3.0) * torch.ones((3, 3), dtype=dtype) + 2.0 * G * Id
    C[3, 3] = C[4, 4] = C[5, 5] = G
    return C


# --------------------------------------------------------------------------- #
# Material and result records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class J2Material:
    """Isotropic linear-elastic / J2 plastic material with linear hardening.

    ``hardening`` is ``H = d sigma_y / d alpha`` with ``alpha`` the equivalent
    plastic strain (``d alpha = d gammap``).  ``H = 0`` is perfect plasticity;
    ``H < 0`` (softening) is rejected because the local return mapping then
    stops being unique and the module makes no claim about it.
    """

    young: float
    poisson: float
    yield_stress: float
    hardening: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("young", self.young),
            ("poisson", self.poisson),
            ("yield_stress", self.yield_stress),
            ("hardening", self.hardening),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.young <= 0.0:
            raise ValueError(f"young's modulus must be positive, got {self.young!r}")
        if not -1.0 < self.poisson < 0.5:
            raise ValueError(
                f"poisson ratio must lie in (-1, 0.5) for a positive-definite "
                f"isotropic elasticity tensor, got {self.poisson!r}"
            )
        if self.yield_stress <= 0.0:
            raise ValueError(f"yield stress must be positive, got {self.yield_stress!r}")
        if self.hardening < 0.0:
            raise ValueError(
                "hardening modulus must be non-negative (softening is outside "
                f"this module's contract), got {self.hardening!r}"
            )

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        return self.young / (3.0 * (1.0 - 2.0 * self.poisson))

    @property
    def plastic_tangent_modulus(self) -> float:
        """Uniaxial elastoplastic tangent ``E H / (E + H)`` (``0`` if ``H == 0``)."""
        return self.young * self.hardening / (self.young + self.hardening)

    @property
    def elastic_tangent(self) -> Tensor:
        return elastic_tangent(self.young, self.poisson)

    def virgin_state(self, *, dtype=_DTYPE) -> J2State:
        return J2State(torch.zeros((3, 3), dtype=dtype), torch.zeros((), dtype=dtype))


@dataclass(frozen=True)
class J2Return:
    """Outcome of one backward-Euler radial return at a material point."""

    stress: Tensor  # 3x3 Cauchy stress
    state: J2State  # committed state for the next increment
    tangent: Tensor  # 6x6 algorithmic (consistent) tangent
    dgamma: float  # plastic multiplier increment
    trial_equivalent_stress: float  # q of the trial (elastic predictor) stress
    trial_yield_function: float  # f_trial = q_trial - sigma_y - H alpha_old
    elastic: bool
    residual: Mapping[str, float]  # a posteriori residuals (dimensionless)

    @property
    def stress_voigt(self) -> Tensor:
        return voigt_from_tensor(self.stress)

    @property
    def equivalent_stress(self) -> float:
        return float(equivalent_stress(self.stress))

    @property
    def equivalent_plastic_strain(self) -> float:
        return float(self.state.alpha)

    @property
    def max_residual(self) -> float:
        return max(self.residual.values(), default=0.0)


# --------------------------------------------------------------------------- #
# Return mapping + audit
# --------------------------------------------------------------------------- #
def _check_strain(strain: Tensor) -> Tensor:
    if tuple(strain.shape) != (3, 3):
        raise ValueError("strain must be a 3x3 symmetric tensor")
    if not torch.isfinite(strain).all():
        raise ValueError("strain contains non-finite entries")
    scale = float(strain.abs().max())
    if scale > 0.0:
        asymmetry = float((strain - strain.T).abs().max()) / scale
        if asymmetry > 1.0e-12:
            raise ValueError(f"strain must be symmetric (relative asymmetry {asymmetry:.3e})")
    return strain


def return_map_residuals(
    strain: Tensor, material: J2Material, state: J2State, result: J2Return | None = None
) -> dict[str, float]:
    """Re-derive the radial-return identities from the *returned* state.

    Every entry is dimensionless: stresses are normalised by
    ``sigma_y + E ||eps||``-scale quantities, plastic strains by ``dgamma``.
    For an exactly integrated step the plastic-branch residuals are at the level
    of double-precision round-off (the qualification gate is ``< 1e-12``).
    """
    strain = _check_strain(strain)
    if result is None:
        raise ValueError("result is required (call radial_return first)")
    stress, new_state, dgamma = result.stress, result.state, result.dgamma
    stress_scale = max(material.yield_stress, float(strain.abs().max()) * material.young, _TINY)

    # Elastic predictor from the *committed* state.
    elastic_strain = strain - state.plastic_strain
    q_trial = equivalent_stress(2.0 * material.shear_modulus * elastic_strain)
    s_trial = 2.0 * material.shear_modulus * deviator(elastic_strain)
    f_trial = float(q_trial) - (material.yield_stress + material.hardening * float(state.alpha))

    # 1) yield condition on the returned state
    q_new = equivalent_stress(stress)
    f_new = float(q_new) - (material.yield_stress + material.hardening * float(new_state.alpha))
    yield_residual = abs(f_new) / stress_scale if dgamma > 0.0 else max(f_new, 0.0) / stress_scale

    # 2) radial-return stress identity: s_new = s_trial - 3 G dgamma s_trial / q_trial
    if dgamma > 0.0 and float(q_trial) > 0.0:
        flow_dir_trial = 1.5 * s_trial / q_trial
        predicted = s_trial - 2.0 * material.shear_modulus * dgamma * flow_dir_trial
        stress_residual = float((deviator(stress) - predicted).abs().max()) / stress_scale
    else:
        stress_residual = float((deviator(stress) - s_trial).abs().max()) / stress_scale

    # 3) consistency parameter: dgamma (3G + H) = f_trial
    consistency = (
        abs(dgamma * (3.0 * material.shear_modulus + material.hardening) - max(f_trial, 0.0))
        / stress_scale
    )

    # 4) associative flow rule: d eps_p = dgamma * 1.5 s_new / q_new
    d_plastic = new_state.plastic_strain - state.plastic_strain
    if dgamma > 0.0 and float(q_new) > 0.0:
        n_new = 1.5 * deviator(stress) / q_new
        flow = float((d_plastic - dgamma * n_new).abs().max()) / max(dgamma, _TINY)
    else:
        flow = float(d_plastic.abs().max()) / max(stress_scale / material.young, _TINY)

    # 5) hardening law: d alpha = dgamma
    d_alpha = float(new_state.alpha) - float(state.alpha)
    hardening = abs(d_alpha - dgamma) / max(dgamma, 1.0e-30)

    # 6) complementarity: dgamma >= 0, f <= 0
    complementarity = max(-dgamma, f_new) / stress_scale

    return {
        "yield": yield_residual,
        "stress_identity": stress_residual,
        "consistency": consistency,
        "flow_rule": flow,
        "hardening": hardening,
        "complementarity": complementarity,
    }


def radial_return(strain: Tensor, material: J2Material, state: J2State | None = None) -> J2Return:
    """Backward-Euler radial return + closed-form consistent tangent.

    ``state`` is the committed (start-of-increment) plastic state; ``None``
    means a virgin material.  The returned ``stress`` and ``state`` are the
    end-of-increment values; ``tangent`` is the algorithmic tangent belonging to
    the branch actually integrated (elastic branch when ``dgamma == 0``).
    """
    if not isinstance(material, J2Material):
        raise TypeError("material must be a J2Material")
    strain = _check_strain(strain)
    state = material.virgin_state(dtype=strain.dtype) if state is None else state
    if tuple(state.plastic_strain.shape) != (3, 3):
        raise ValueError("committed plastic strain must be 3x3")

    stress, new_state, dgamma = update_j2(
        strain, material.young, material.poisson, material.yield_stress, material.hardening, state
    )
    tangent = j2_consistent_tangent(
        strain, material.young, material.poisson, material.hardening, state, dgamma
    )
    elastic_strain = strain - state.plastic_strain
    q_trial = float(equivalent_stress(2.0 * material.shear_modulus * elastic_strain))
    f_trial = q_trial - (material.yield_stress + material.hardening * float(state.alpha))
    result = J2Return(
        stress=stress,
        state=new_state,
        tangent=tangent,
        dgamma=float(dgamma),
        trial_equivalent_stress=q_trial,
        trial_yield_function=f_trial,
        elastic=bool(float(dgamma) == 0.0),
        residual={},
    )
    residuals = return_map_residuals(strain, material, state, result)
    return J2Return(
        stress=result.stress,
        state=result.state,
        tangent=result.tangent,
        dgamma=result.dgamma,
        trial_equivalent_stress=result.trial_equivalent_stress,
        trial_yield_function=result.trial_yield_function,
        elastic=result.elastic,
        residual=residuals,
    )


# --------------------------------------------------------------------------- #
# Analytic anchors
# --------------------------------------------------------------------------- #
def uniaxial_stress_path(
    material: J2Material, strains, *, tolerance: float = 1e-11, max_iterations: int = 80
) -> dict:
    """Drive a uniaxial-**stress** path ``sigma_22 = sigma_33 = 0``.

    For each imposed axial strain ``eps_11`` the two equal lateral strains are
    solved with a Newton iteration that uses the consistent tangent (so the
    tangent is exercised, not bypassed).  Returns total strain, axial stress,
    lateral strain, equivalent plastic strain and the *secant* and *incremental*
    axial tangent moduli, which for linear isotropic hardening must reproduce

    * elastic branch: ``d sigma / d eps = E``
    * plastic branch: ``d sigma / d eps = E H / (E + H)``
    * transition at ``eps = sigma_y / E``
    """
    strains = [float(e) for e in strains]
    if not strains:
        raise ValueError("at least one axial strain is required")
    state = material.virgin_state()
    out = {
        "strain": [],
        "stress": [],
        "lateral_strain": [],
        "equivalent_plastic_strain": [],
        "elastic": [],
    }
    for eps11 in strains:
        x = -material.poisson * eps11  # elastic lateral guess
        result = None
        for _ in range(max_iterations):
            strain = torch.zeros((3, 3), dtype=_DTYPE)
            strain[0, 0] = eps11
            strain[1, 1] = strain[2, 2] = x
            result = radial_return(strain, material, state)
            residual = float(result.stress[1, 1])
            if abs(residual) <= tolerance * material.yield_stress:
                break
            # R depends on x through eps_yy and eps_zz simultaneously.
            dR = float(result.tangent[1, 1]) + float(result.tangent[1, 2])
            if abs(dR) < _TINY:
                raise RuntimeError("degenerate lateral-strain Newton derivative")
            x -= residual / dR
        else:  # pragma: no cover - defensive
            raise RuntimeError("lateral-strain Newton did not converge")
        if abs(float(result.stress[1, 1])) > 1e-8 * material.yield_stress:
            raise RuntimeError("sigma_22 did not reach zero (uniaxial constraint)")
        state = result.state
        out["strain"].append(eps11)
        out["stress"].append(float(result.stress[0, 0]))
        out["lateral_strain"].append(x)
        out["equivalent_plastic_strain"].append(float(result.state.alpha))
        out["elastic"].append(result.elastic)
    sigma = out["stress"]
    eps = out["strain"]
    out["incremental_modulus"] = [
        (sigma[k + 1] - sigma[k]) / (eps[k + 1] - eps[k])
        for k in range(len(eps) - 1)
        if eps[k + 1] != eps[k]
    ]
    return out


def yield_probe(material: J2Material, strain_direction: Tensor, *, iterations: int = 200) -> dict:
    """Bisect the elastic scaling factor at which ``q(sigma) = sigma_y``.

    The elastic stress is linear in the strain scale, so the bisection converges
    to the analytic onset scale; the reported ``equivalent_stress`` at onset is
    compared directly with the analytic yield surface value ``sigma_y``.
    """
    direction = torch.as_tensor(strain_direction, dtype=_DTYPE)
    _check_strain(direction)
    C = material.elastic_tangent
    eps_v = voigt_from_tensor(direction)
    eps_v = eps_v.clone()
    eps_v[3:] *= 2.0  # engineering shear
    stress_v = C @ eps_v
    stress_dir = _stress_tensor_from_voigt(stress_v)
    q_unit = float(equivalent_stress(stress_dir))
    if q_unit <= 0.0:
        raise ValueError("strain direction produces no deviatoric stress")
    analytic_scale = material.yield_stress / q_unit
    lo, hi = 0.0, 2.0 * analytic_scale
    state = material.virgin_state()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        result = radial_return(mid * direction, material, state)
        if result.equivalent_stress < material.yield_stress:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    result = radial_return(scale * direction, material, state)
    q_onset = result.equivalent_stress
    return {
        "scale": scale,
        "analytic_scale": analytic_scale,
        "scale_relative_error": abs(scale - analytic_scale) / analytic_scale,
        "equivalent_stress": q_onset,
        "yield_surface_relative_error": abs(q_onset - material.yield_stress)
        / material.yield_stress,
        "dgamma": result.dgamma,
    }


def _stress_tensor_from_voigt(vector: Tensor) -> Tensor:
    s = torch.zeros((3, 3), dtype=vector.dtype, device=vector.device)
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        s[i, j] = s[j, i] = vector[k]
    return s


def pure_shear_yield_point(material: J2Material) -> dict:
    """Yield of a pure-shear state ``eps_xy = gamma/2``.

    Analytic values: ``tau_y = sigma_y / sqrt(3)``, ``gamma_y = sigma_y / (sqrt(3) G)``.
    """
    direction = torch.zeros((3, 3), dtype=_DTYPE)
    direction[0, 1] = direction[1, 0] = 0.5
    probe = yield_probe(material, direction)
    gamma_numeric = probe["scale"]
    gamma_analytic = material.yield_stress / (math.sqrt(3.0) * material.shear_modulus)
    tau_numeric = material.shear_modulus * gamma_numeric  # tau = G gamma
    tau_analytic = material.yield_stress / math.sqrt(3.0)
    return {
        "gamma_y": gamma_numeric,
        "gamma_y_analytic": gamma_analytic,
        "gamma_relative_error": abs(gamma_numeric - gamma_analytic) / gamma_analytic,
        "tau_y": tau_numeric,
        "tau_y_analytic": tau_analytic,
        "tau_relative_error": abs(tau_numeric - tau_analytic) / tau_analytic,
        "yield_surface_relative_error": probe["yield_surface_relative_error"],
    }


# --------------------------------------------------------------------------- #
# Self-verification (programmatic, no hand-copied numbers)
# --------------------------------------------------------------------------- #
def verify_minimum_surface(material: J2Material | None = None) -> dict:
    """Run the whole minimum-surface verification and return a JSON-ready dict."""
    material = material or J2Material(
        young=210.0e9, poisson=0.3, yield_stress=300.0e6, hardening=10.0e9
    )
    E, H = material.young, material.hardening
    eps_y = material.yield_stress / E
    E_t = material.plastic_tangent_modulus

    # -- uniaxial piecewise anchor -------------------------------------------
    strains = [0.25 * eps_y, 0.5 * eps_y, 0.75 * eps_y, eps_y] + [
        2.0 * eps_y,
        4.0 * eps_y,
        6.0 * eps_y,
        8.0 * eps_y,
    ]
    path = uniaxial_stress_path(material, strains)
    elastic_moduli = path["incremental_modulus"][:2]
    plastic_moduli = path["incremental_modulus"][4:]
    elastic_error = max(abs(m - E) / E for m in elastic_moduli)
    plastic_error = max(abs(m - E_t) / E_t for m in plastic_moduli)

    # -- closed-form stress at each strain -----------------------------------
    closed_form = []
    for eps in strains:
        if eps <= eps_y:
            closed_form.append(E * eps)
        else:
            closed_form.append(material.yield_stress + E_t * (eps - eps_y))
    stress_error = max(abs(s - c) / c for s, c in zip(path["stress"], closed_form))

    # -- pure shear / yield surface -----------------------------------------
    shear = pure_shear_yield_point(material)
    directions = {
        "uniaxial": torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=_DTYPE)),
        "pure_shear": torch.tensor(
            [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=_DTYPE
        ),
        "equibiaxial_shear": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]], dtype=_DTYPE
        ),
        "combined": torch.tensor(
            [[0.8, 0.3, 0.0], [0.3, -0.2, 0.1], [0.0, 0.1, -0.6]], dtype=_DTYPE
        ),
    }
    surface = {name: yield_probe(material, direction) for name, direction in directions.items()}
    surface_error = max(v["yield_surface_relative_error"] for v in surface.values())

    # -- a posteriori residuals over a non-proportional path -----------------
    #   tension, shear, reversal of the shear, then proportional reloading.
    path_strains = [
        torch.diag(torch.tensor([2.0e-3, -0.5e-3, -0.5e-3], dtype=_DTYPE)),
        torch.tensor(
            [[2.0e-3, 0.9e-3, 0.0], [0.9e-3, -0.5e-3, 0.0], [0.0, 0.0, -0.5e-3]], dtype=_DTYPE
        ),
        torch.tensor(
            [[2.0e-3, -0.9e-3, 0.0], [-0.9e-3, -0.5e-3, 0.0], [0.0, 0.0, -0.5e-3]], dtype=_DTYPE
        ),
        torch.tensor(
            [[2.4e-3, -1.2e-3, 0.2e-3], [-1.2e-3, -0.4e-3, 0.1e-3], [0.2e-3, 0.1e-3, -0.6e-3]],
            dtype=_DTYPE,
        ),
    ]
    state = material.virgin_state()
    residual_max = 0.0
    residual_by_component: dict[str, float] = {}
    for strain in path_strains:
        result = radial_return(strain, material, state)
        residual_max = max(residual_max, result.max_residual)
        for key, value in result.residual.items():
            residual_by_component[key] = max(residual_by_component.get(key, 0.0), value)
        state = result.state

    # -- consistent tangent vs central difference ---------------------------
    # Plastic branch: committed state is the end of the first path step.
    plastic_state = radial_return(path_strains[0], material, material.virgin_state()).state
    tangent_error = _tangent_difference_error(material, plastic_state, path_strains[3], 1e-7)

    return {
        "material": {
            "young": E,
            "poisson": material.poisson,
            "yield_stress": material.yield_stress,
            "hardening": H,
            "shear_modulus": material.shear_modulus,
            "bulk_modulus": material.bulk_modulus,
            "yield_strain": eps_y,
            "plastic_tangent_modulus": E_t,
        },
        "uniaxial": {
            "strains": path["strain"],
            "stress": path["stress"],
            "closed_form_stress": closed_form,
            "equivalent_plastic_strain": path["equivalent_plastic_strain"],
            "elastic_modulus_numeric": elastic_moduli,
            "plastic_modulus_numeric": plastic_moduli,
            "elastic_modulus_relative_error": elastic_error,
            "plastic_modulus_relative_error": plastic_error,
            "stress_relative_error": stress_error,
        },
        "pure_shear": shear,
        "yield_surface": {name: value for name, value in surface.items()},
        "yield_surface_max_relative_error": surface_error,
        "return_map": {
            "max_residual": residual_max,
            "residual_by_component": residual_by_component,
            "steps": len(path_strains),
        },
        "consistent_tangent_relative_error": tangent_error,
    }


def _tangent_difference_error(
    material: J2Material, state: J2State, strain: Tensor, h: float
) -> float:
    """Relative error of the consistent tangent against a central difference."""
    base = radial_return(strain, material, state)
    C = base.tangent
    worst = 0.0
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        direction = torch.zeros((3, 3), dtype=_DTYPE)
        if i == j:
            direction[i, i] = 1.0
        else:
            direction[i, j] = direction[j, i] = 0.5
        plus = radial_return(strain + h * direction, material, state).stress_voigt
        minus = radial_return(strain - h * direction, material, state).stress_voigt
        fd = (plus - minus) / (2.0 * h)
        column = C[:, k]
        # Compare only meaningful directions (avoid amplifying round-off of
        # near-zero entries): use a global scale for the column.
        scale = max(float(column.abs().max()), _TINY)
        worst = max(worst, float((fd - column).abs().max()) / max(scale, 1e-30))
    return worst
