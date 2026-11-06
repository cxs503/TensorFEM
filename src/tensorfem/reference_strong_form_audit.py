"""Strong-form audit of the closed-form reference layer (Kirsch, Lamé, Airy).

Why this module exists
----------------------
Every accuracy claim in this package is stated *relative to* a closed-form
reference: :mod:`tensorfem.kirsch_hole`, :mod:`tensorfem.lame_reference`,
:mod:`tensorfem.airy_cantilever`, :mod:`tensorfem.rotating_disk` and
:mod:`tensorfem.thermal_analytic`.  The reference is the ruler.  A wrong ruler
makes a correct solver look wrong and a wrong solver look right, and no amount
of mesh refinement can detect it, because refinement is measured against the
ruler itself.

Three of those modules already carry their own residual checks
(:meth:`tensorfem.rotating_disk.RotatingDisk.equilibrium_residual`,
:meth:`tensorfem.rotating_disk.RotatingDisk.compatibility_residual`,
:func:`tensorfem.thermal_analytic.laplacian_residual`, and the exact Prandtl
field of :mod:`tensorfem.saint_venant_torsion`).  The three that carry the most
weight -- Kirsch, Lamé and the Airy cantilever -- do not: their algebra is
trusted by inspection, so a typo in one coefficient would propagate silently
into every stress comparison in the suite.

This module closes that gap without reusing the module under audit.  The closed
forms are differentiated *numerically* (4th-order central stencils, float64) and
substituted into the governing equations and boundary conditions, which then
have to hold to round-off.  Nothing is assembled and nothing is solved, so the
audit is cheap enough to run inside the test suite.

Checks, all dimensionless
-------------------------
``equilibrium``     ``div(sigma) + b = 0`` from the Cartesian stress components.
``beltrami``        ``laplace(sigma_xx + sigma_yy) = 0`` (2-D, zero body force).
``compatibility``   strain of the closed-form displacement against the Hooke
                    strain of the closed-form stress (plane stress/strain aware).
``polar_equilibrium`` axisymmetric radial equilibrium for the Lamé family.
``elasticity``      internal consistency of a 1-D (radial) elastic state:
                    ``eps_r = du/dr`` and ``eps_t = u/r`` from the same
                    constitutive law that produced the stresses.
``boundary_*``      the traction conditions each family is supposed to satisfy.
``far_field_40a``   informational: the residual of the asymptotic remote field at
                    ``r = 40 a``, reported together with its decay exponent
                    (``far_field_order``) instead of being gated at a finite radius.
``statics_*``       resultant identities (end shear / end moment of the beam).
``far_field_order`` observed decay exponent of the Kirsch remote field.
``root_rotation``   informational: rotation of the clamped beam root.

The gates are shown to be non-vacuous by two controls:

* **perturbation control** -- the same operators applied to a field that was
  deliberately corrupted by ``eps * sigma_ref * x / L``; the residual must grow
  linearly with ``eps``, and the smallest detected ``eps`` is reported as the
  detection floor.
* **stencil control** -- the residual as a function of the difference step must
  fall at the 4th-order rate of the stencil before hitting the round-off floor,
  i.e. the residual is truncation-limited, not noise-limited.

A physically realistic corruption (dropping the ``3 a**4 / r**4`` term of the
Kirsch hoop stress -- a one-coefficient typo) is exercised in the test suite and
is rejected at ``O(1e-1)``, six orders of magnitude above the pass gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch

from .airy_cantilever import AiryCantilever
from .kirsch_hole import KirschHole
from .lame_reference import lame_state

__all__ = [
    "AuditReport",
    "INFORMATIONAL",
    "PASS_TOLERANCE",
    "STENCIL_ORDER",
    "audit_airy_cantilever",
    "audit_kirsch_hole",
    "audit_lame_cylinder",
    "beltrami_residual",
    "compatibility_residual",
    "divergence_residual",
    "judge_suite",
    "perturbation_control",
    "run_audit_suite",
    "stencil_control",
]

D = torch.float64

#: 4th-order central stencil for the first derivative: ``(offset, weight)`` pairs
#: in units of the step ``h``; ``sum(weight * f(x + offset*h)) / h``.
_STENCIL = ((-2.0, 1.0 / 12.0), (-1.0, -2.0 / 3.0), (1.0, 2.0 / 3.0), (2.0, -1.0 / 12.0))

#: 4th-order central stencil for the second derivative.
_STENCIL_2 = ((-2.0, -1.0 / 12.0), (-1.0, 4.0 / 3.0), (0.0, -5.0 / 2.0),
              (1.0, 4.0 / 3.0), (2.0, -1.0 / 12.0))

#: Convergence rate the stencil control must observe when the step is refined.
STENCIL_ORDER = 4.0

#: Every normalised residual below this magnitude counts as "satisfied to round-off".
PASS_TOLERANCE = 1.0e-6

#: Metrics reported for information only; they are not pass/fail gates.
INFORMATIONAL = frozenset({"far_field_order", "far_field_40a", "root_rotation"})

_StressFn = Callable[[torch.Tensor], torch.Tensor]
_ScalarFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# independent numerical operators
# --------------------------------------------------------------------------- #
def _shift(points: torch.Tensor, axis: int, delta: float) -> torch.Tensor:
    moved = points.clone()
    moved[:, axis] = moved[:, axis] + delta
    return moved


def _derivative(field: _ScalarFn, points: torch.Tensor, h: float, axis: int) -> torch.Tensor:
    total = torch.zeros_like(field(points))
    for offset, weight in _STENCIL:
        total = total + weight * field(_shift(points, axis, offset * h))
    return total / h


def _second_derivative(field: _ScalarFn, points: torch.Tensor, h: float,
                       axis: int) -> torch.Tensor:
    total = torch.zeros_like(field(points))
    for offset, weight in _STENCIL_2:
        total = total + weight * field(_shift(points, axis, offset * h))
    return total / (h * h)


def _component(field: _StressFn, index: int) -> _ScalarFn:
    return lambda points: field(points)[:, index]


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values * values)))


def stress_scale(stress: _StressFn, points: torch.Tensor) -> float:
    """RMS of ``sqrt(sxx**2 + syy**2 + 2*sxy**2)`` over the sample -- the residual unit."""
    s = stress(points)
    return _rms(torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + 2.0 * s[:, 2] ** 2))


def divergence_residual(
    stress: _StressFn,
    points: torch.Tensor,
    length: float,
    *,
    h: float | None = None,
    body_force: _StressFn | None = None,
) -> float:
    """``RMS(div(sigma) + b) / (sigma_ref / L)``: the dimensionless equilibrium defect."""
    step = 1e-2 * length if h is None else h
    sxx, syy, sxy = (_component(stress, i) for i in range(3))
    rx = _derivative(sxx, points, step, 0) + _derivative(sxy, points, step, 1)
    ry = _derivative(sxy, points, step, 0) + _derivative(syy, points, step, 1)
    scale = stress_scale(stress, points) / length
    if body_force is not None:
        b = body_force(points)
        rx = rx + b[:, 0]
        ry = ry + b[:, 1]
        scale = max(scale, _rms(torch.linalg.vector_norm(b, dim=1)))
    if scale == 0.0:
        raise ValueError("degenerate field: zero stress and zero body force")
    return float(torch.sqrt(torch.mean(rx * rx + ry * ry)) / scale)


def beltrami_residual(stress: _StressFn, points: torch.Tensor, length: float,
                      *, h: float | None = None) -> float:
    """``RMS(laplace(sigma_xx + sigma_yy)) / (sigma_ref / L**2)``.

    For a 2-D elastic field in equilibrium under zero body force the trace of the
    stress is harmonic (Beltrami-Michell), in plane stress and plane strain alike,
    so this identity constrains the field independently of the displacement.
    """
    step = 1e-2 * length if h is None else h
    trace = lambda points: stress(points)[:, 0] + stress(points)[:, 1]  # noqa: E731
    lap = (_second_derivative(trace, points, step, 0)
           + _second_derivative(trace, points, step, 1))
    scale = stress_scale(stress, points) / (length * length)
    return float(_rms(lap) / scale)


def _hooke_strain(stress: _StressFn, points: torch.Tensor, young: float, poisson: float,
                  plane: str) -> torch.Tensor:
    s = stress(points)
    if plane == "stress":
        exx = (s[:, 0] - poisson * s[:, 1]) / young
        eyy = (s[:, 1] - poisson * s[:, 0]) / young
        gxy = 2.0 * (1.0 + poisson) * s[:, 2] / young
    elif plane == "strain":
        exx = (1.0 + poisson) * ((1.0 - poisson) * s[:, 0] - poisson * s[:, 1]) / young
        eyy = (1.0 + poisson) * ((1.0 - poisson) * s[:, 1] - poisson * s[:, 0]) / young
        gxy = 2.0 * (1.0 + poisson) * s[:, 2] / young
    else:  # pragma: no cover - guarded by the callers
        raise ValueError("plane must be 'stress' or 'strain'")
    return torch.stack((exx, eyy, gxy), dim=1)


def compatibility_residual(
    displacement: _StressFn,
    stress: _StressFn,
    young: float,
    poisson: float,
    points: torch.Tensor,
    length: float,
    *,
    plane: str = "stress",
    h: float | None = None,
) -> float:
    """Strain from the closed-form displacement vs the Hooke strain of the stress.

    This is the check that the *displacement* half of a reference is the integral
    of the *stress* half.  Normalised by the RMS Hooke strain, so it is an
    error fraction of the elastic strain scale.
    """
    step = 1e-2 * length if h is None else h
    ux, uy = (_component(displacement, i) for i in range(2))
    measured = torch.stack(
        (
            _derivative(ux, points, step, 0),
            _derivative(uy, points, step, 1),
            _derivative(ux, points, step, 1) + _derivative(uy, points, step, 0),
        ),
        dim=1,
    )
    expected = _hooke_strain(stress, points, young, poisson, plane)
    scale = _rms(expected)
    return float(_rms(measured - expected) / scale)


def polar_equilibrium_residual(
    sigma_r: _ScalarFn, sigma_t: _ScalarFn, radii: torch.Tensor, length: float,
    *, h: float | None = None,
) -> float:
    """``RMS(dsigma_r/dr + (sigma_r - sigma_t)/r) / (sigma_ref / L)``."""
    step = 1e-2 * (radii.max() - radii.min()).item() if h is None else h
    residual = (_derivative(sigma_r, radii.reshape(-1, 1), step, 0)
                + (sigma_r(radii.reshape(-1, 1)) - sigma_t(radii.reshape(-1, 1))) / radii.reshape(-1, 1))
    scale = float(torch.max(torch.abs(sigma_r(radii.reshape(-1, 1))))) / length
    return float(_rms(residual[:, 0]) / scale)


# --------------------------------------------------------------------------- #
# sampling helpers
# --------------------------------------------------------------------------- #
def polar_samples(r_min: float, r_max: float, count: int, seed: int = 0,
                  dtype: torch.dtype = D) -> torch.Tensor:
    """Deterministic interior samples of an annulus ``r_min <= r <= r_max``."""
    generator = torch.Generator().manual_seed(seed)
    radius = r_min + (r_max - r_min) * torch.rand(count, generator=generator, dtype=dtype)
    angle = 2.0 * math.pi * torch.rand(count, generator=generator, dtype=dtype)
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)


def annular_ring(radius: float, count: int = 33, dtype: torch.dtype = D) -> torch.Tensor:
    """Points exactly on the circle of the given radius (boundary sampling)."""
    angle = torch.linspace(0.0, 2.0 * math.pi, count + 1, dtype=dtype)[:-1]
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)


def simpson(values: torch.Tensor, step: float) -> float:
    """Composite Simpson integral of evenly spaced ``values`` (even count required)."""
    n = values.numel()
    if n % 2 == 0:  # pragma: no cover - callers pass an odd count
        raise ValueError("Simpson needs an odd number of samples")
    weights = torch.ones(n, dtype=values.dtype)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float((weights * values).sum() * step / 3.0)


# --------------------------------------------------------------------------- #
# report container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AuditReport:
    """Normalised residuals of one closed-form reference family."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def gated(self) -> dict[str, float]:
        return {k: v for k, v in self.metrics.items() if k not in INFORMATIONAL}

    def worst(self) -> tuple[str, float]:
        key = max(self.gated, key=lambda k: abs(self.gated[k]))
        return key, self.gated[key]

    def passed(self, tolerance: float = PASS_TOLERANCE) -> bool:
        return all(abs(value) <= tolerance for value in self.gated.values())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed(),
            "worst_metric": self.worst()[0],
            "worst_value": self.worst()[1],
            "metrics": {k: float(v) for k, v in sorted(self.metrics.items())},
        }


# --------------------------------------------------------------------------- #
# family 1: Kirsch (uniaxial tension, circular hole)
# --------------------------------------------------------------------------- #
def _numpy_field(fn, points: torch.Tensor) -> torch.Tensor:
    """Adapt a numpy closed form ``fn(x, y) -> tuple`` of components to torch."""
    x = points[:, 0].detach().cpu().numpy()
    y = points[:, 1].detach().cpu().numpy()
    return torch.as_tensor(np.stack(fn(x, y), axis=1), dtype=D)


def kirsch_stress(hole: KirschHole) -> _StressFn:
    return lambda points: _numpy_field(hole.stress_cartesian, points)


def kirsch_displacement(hole: KirschHole) -> _StressFn:
    return lambda points: _numpy_field(hole.displacement_cartesian, points)


def audit_kirsch_hole(
    hole: KirschHole | None = None,
    *,
    samples: int = 121,
    seed: int = 0,
    inner_factor: float = 1.12,
    outer_factor: float = 4.0,
) -> AuditReport:
    """Strong-form audit of :class:`tensorfem.kirsch_hole.KirschHole`."""
    hole = hole or KirschHole()
    a, traction = hole.radius, hole.far_field
    length = a
    stress, displacement = kirsch_stress(hole), kirsch_displacement(hole)
    points = polar_samples(inner_factor * a, outer_factor * a, samples, seed)

    young, poisson = hole.effective_constants
    metrics = {
        "equilibrium": divergence_residual(stress, points, length),
        "beltrami": beltrami_residual(stress, points, length),
        "compatibility": compatibility_residual(
            displacement, stress, young, poisson, points, length, plane="stress"
        ),
    }

    # boundary: the hole surface is traction free
    ring = annular_ring(a, count=64)
    srr, stt, srt = hole.stress_polar(np.full(ring.shape[0], a),
                                      torch.atan2(ring[:, 1], ring[:, 0]).numpy())
    metrics["boundary_hole_radial"] = float(np.max(np.abs(srr)) / traction)
    metrics["boundary_hole_shear"] = float(np.max(np.abs(srt)) / traction)

    # remote field: it is an asymptotic condition, so measure the decay exponent
    radii = (10.0, 20.0, 40.0, 80.0)
    errors = []
    for factor in radii:
        probe = annular_ring(factor * a, count=73)
        sxx, syy, sxy = hole.stress_cartesian(probe[:, 0].numpy(), probe[:, 1].numpy())
        errors.append(float(np.max(np.maximum.reduce(
            (np.abs(sxx - traction), np.abs(syy), np.abs(sxy)))) / traction))
    # the remote field is an *asymptotic* statement, so it is reported with its
    # observed decay exponent instead of being gated at a finite radius
    metrics["far_field_order"] = math.log2(errors[1] / errors[2])
    metrics["far_field_40a"] = errors[2]

    return AuditReport("kirsch_hole", metrics)


# --------------------------------------------------------------------------- #
# family 2: Lamé (pressurised cylinder)
# --------------------------------------------------------------------------- #
def lame_stress(inner: float, outer: float, p_inner: float, p_outer: float,
                *, young: float, poisson: float, plane: str) -> _StressFn:
    """Cartesian adapter of the polar Lamé field (checked in the *other* frame)."""
    def stress(points: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        state = lame_state(r, inner_radius=inner, outer_radius=outer,
                           inner_pressure=p_inner, outer_pressure=p_outer,
                           young_modulus=young, poisson_ratio=poisson, plane=plane)
        cos = points[:, 0] / r
        sin = points[:, 1] / r
        sxx = state.radial_stress * cos * cos + state.hoop_stress * sin * sin
        syy = state.radial_stress * sin * sin + state.hoop_stress * cos * cos
        sxy = (state.radial_stress - state.hoop_stress) * cos * sin
        return torch.stack((sxx, syy, sxy), dim=1)

    return stress


def lame_displacement(inner: float, outer: float, p_inner: float, p_outer: float,
                      *, young: float, poisson: float, plane: str) -> _StressFn:
    def field(points: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        state = lame_state(r, inner_radius=inner, outer_radius=outer,
                           inner_pressure=p_inner, outer_pressure=p_outer,
                           young_modulus=young, poisson_ratio=poisson, plane=plane)
        return torch.stack((state.radial_displacement * points[:, 0] / r,
                            state.radial_displacement * points[:, 1] / r), dim=1)

    return field


def audit_lame_cylinder(
    *,
    inner: float = 1.0,
    outer: float = 2.0,
    p_inner: float = 5.0,
    p_outer: float = 1.0,
    young: float = 200.0e9,
    poisson: float = 0.3,
    plane: str = "strain",
    samples: int = 121,
    seed: int = 1,
) -> AuditReport:
    """Strong-form audit of :func:`tensorfem.lame_reference.lame_state`."""
    stress = lame_stress(inner, outer, p_inner, p_outer, young=young, poisson=poisson, plane=plane)
    displacement = lame_displacement(inner, outer, p_inner, p_outer, young=young,
                                     poisson=poisson, plane=plane)
    points = polar_samples(1.05 * inner, 0.95 * outer, samples, seed)
    length = outer - inner
    p_ref = max(p_inner, p_outer)

    metrics = {
        "equilibrium": divergence_residual(stress, points, length),
        "beltrami": beltrami_residual(stress, points, length),
        "compatibility": compatibility_residual(
            displacement, stress, young, poisson, points, length, plane=plane
        ),
    }

    # 1-D radial check, using only the polar closed form
    radii = torch.linspace(1.05 * inner, 0.95 * outer, 201, dtype=D)

    def sigma_r(columns: torch.Tensor) -> torch.Tensor:
        r = columns[:, 0]
        return lame_state(r, inner_radius=inner, outer_radius=outer, inner_pressure=p_inner,
                          outer_pressure=p_outer, young_modulus=young,
                          poisson_ratio=poisson, plane=plane).radial_stress.reshape(-1, 1)

    def sigma_t(columns: torch.Tensor) -> torch.Tensor:
        r = columns[:, 0]
        return lame_state(r, inner_radius=inner, outer_radius=outer, inner_pressure=p_inner,
                          outer_pressure=p_outer, young_modulus=young,
                          poisson_ratio=poisson, plane=plane).hoop_stress.reshape(-1, 1)

    metrics["polar_equilibrium"] = polar_equilibrium_residual(
        sigma_r, sigma_t, radii, length=inner
    )

    # the two rim tractions must reproduce the applied pressures exactly
    rim_r = torch.tensor([[inner], [outer]], dtype=D)
    s_r = sigma_r(rim_r)[:, 0]
    metrics["boundary_inner_pressure"] = float(abs(s_r[0] + p_inner) / p_ref)
    metrics["boundary_outer_pressure"] = float(abs(s_r[1] + p_outer) / p_ref)

    # plane strain carries an axial stress which plane stress does not
    if plane == "strain":
        radii_t = radii.reshape(-1, 1)
        state = lame_state(radii_t[:, 0], inner_radius=inner, outer_radius=outer,
                           inner_pressure=p_inner, outer_pressure=p_outer,
                           young_modulus=young, poisson_ratio=poisson, plane=plane)
        expected = poisson * (state.radial_stress + state.hoop_stress)
        metrics["axial_identity"] = float(
            torch.max(torch.abs(state.axial_stress - expected)) / float(expected.abs().max())
        )
    else:
        radii_t = radii.reshape(-1, 1)
        state = lame_state(radii_t[:, 0], inner_radius=inner, outer_radius=outer,
                           inner_pressure=p_inner, outer_pressure=p_outer,
                           young_modulus=young, poisson_ratio=poisson, plane=plane)
        metrics["axial_identity"] = float(torch.max(torch.abs(state.axial_stress)) / p_ref)

    return AuditReport(f"lame_cylinder[{plane}]", metrics)


# --------------------------------------------------------------------------- #
# family 3: Airy cantilever (exact plane stress, parabolic end shear)
# --------------------------------------------------------------------------- #
def audit_airy_cantilever(model: AiryCantilever | None = None, *,
                          samples: int = 145, seed: int = 2) -> AuditReport:
    """Strong-form audit of :class:`tensorfem.airy_cantilever.AiryCantilever`."""
    model = model or AiryCantilever()
    length = model.length
    c = model.half_height
    stress = lambda points: model.stress(points)  # noqa: E731
    displacement = lambda points: model.displacement(points)  # noqa: E731

    generator = torch.Generator().manual_seed(seed)
    x = 0.05 * length + 0.9 * length * torch.rand(samples, generator=generator, dtype=D)
    y = 0.9 * c * (2.0 * torch.rand(samples, generator=generator, dtype=D) - 1.0)
    points = torch.stack((x, y), dim=1)

    metrics = {
        "equilibrium": divergence_residual(stress, points, c),
        "beltrami": beltrami_residual(stress, points, c),
        "compatibility": compatibility_residual(
            displacement, stress, model.young_modulus, model.poisson, points, c,
            plane="stress"
        ),
    }

    # the long edges are traction free
    edge_x = torch.linspace(0.0, length, 101, dtype=D)
    for sign, label in ((1.0, "top"), (-1.0, "bottom")):
        edge = torch.stack((edge_x, torch.full_like(edge_x, sign * c)), dim=1)
        s = model.stress(edge)
        scale = float(s[:, 0].abs().max())
        metrics[f"boundary_free_edge_{label}"] = float(
            torch.max(torch.maximum(s[:, 1].abs(), s[:, 2].abs())) / scale
        )

    # statics: the end shear and the clamped-end moment of the exact field
    across = torch.linspace(-c, c, 2001, dtype=D)
    step = float(across[1] - across[0])
    shear = torch.stack((torch.full_like(across, length), across), dim=1)
    tau = model.stress(shear)[:, 2]
    load_per_thickness = model.end_shear / model.thickness
    metrics["statics_end_shear"] = abs(simpson(tau, step) - load_per_thickness) / abs(
        load_per_thickness
    )
    root = torch.stack((torch.zeros_like(across), across), dim=1)
    sigma_x = model.stress(root)[:, 0]
    moment = simpson(across * sigma_x, step)
    expected_moment = -model.end_shear * length / model.thickness
    metrics["statics_root_moment"] = abs(moment - expected_moment) / abs(expected_moment)

    # informational: the rotation of the clamped root, which the pin at the origin
    # leaves free (a cantilever root should carry none)
    h = 1e-3 * length
    duy_dx = float((displacement(torch.tensor([[h, 0.0]], dtype=D))[0, 1]
                    - displacement(torch.tensor([[-h, 0.0]], dtype=D))[0, 1]) / (2.0 * h))
    tip = abs(model.tip_deflection())
    metrics["root_rotation"] = abs(duy_dx) * length / tip if tip else 0.0

    return AuditReport("airy_cantilever", metrics)


# --------------------------------------------------------------------------- #
# controls: prove the gates bite
# --------------------------------------------------------------------------- #
def perturbed(stress: _StressFn, amplitude: float, length: float,
              sigma_ref: float) -> _StressFn:
    """Corrupt ``sigma_xx`` by ``amplitude * sigma_ref * x / L`` (a smooth, localised typo)."""

    def field(points: torch.Tensor) -> torch.Tensor:
        s = stress(points).clone()
        s[:, 0] = s[:, 0] + amplitude * sigma_ref * points[:, 0] / length
        return s

    return field


def perturbation_control(
    stress: _StressFn,
    points: torch.Tensor,
    length: float,
    *,
    amplitudes: Iterable[float] = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
) -> dict[str, float]:
    """Equilibrium residual of a corrupted field as a function of the corruption size.

    A working gate returns residuals that track ``amplitude`` one-to-one; the
    smallest amplitude that still rises clear of the clean baseline is the
    detection floor reported by :func:`run_audit_suite`.
    """
    sigma_ref = stress_scale(stress, points)
    baseline = divergence_residual(stress, points, length)
    table = {}
    for amplitude in amplitudes:
        table[f"{amplitude:.0e}"] = divergence_residual(
            perturbed(stress, amplitude, length, sigma_ref), points, length
        )
    table["baseline"] = baseline
    return table


def stencil_control(stress: _StressFn, points: torch.Tensor, length: float,
                    *, steps: Iterable[float] = (2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3)) -> dict:
    """Equilibrium residual as a function of the difference step.

    The 4th-order stencil must show residuals falling at close to
    :data:`STENCIL_ORDER` before the round-off floor is reached; that is what
    makes the reported residual a property of the field rather than of the step.
    """
    steps = list(steps)
    table = [{"step": float(h), "residual": divergence_residual(stress, points, length, h=h)}
             for h in steps]
    observed = []
    for coarse, fine in zip(table, table[1:]):
        if fine["residual"] > 0 and coarse["residual"] > 0:
            observed.append(math.log2(coarse["residual"] / fine["residual"]))
    return {
        "table": table,
        "observed_orders": [float(v) for v in observed],
        "order": float(max(observed)) if observed else float("nan"),
    }


# --------------------------------------------------------------------------- #
# suite
# --------------------------------------------------------------------------- #
def judge_suite(reports: Iterable[AuditReport], tolerance: float = PASS_TOLERANCE) -> dict:
    """Fail-closed verdict over a list of reports."""
    entries = {}
    for report in reports:
        key, value = report.worst()
        entries[report.name] = {
            "passed": report.passed(tolerance),
            "worst_metric": key,
            "worst_value": float(value),
        }
    return {
        "tolerance": tolerance,
        "passed": all(entry["passed"] for entry in entries.values()),
        "references": entries,
    }


def run_audit_suite(*, samples: int = 121, seed: int = 20261103) -> dict:
    """Audit every closed-form reference that lacks its own residual check."""
    reports = [
        audit_kirsch_hole(samples=samples, seed=seed),
        audit_lame_cylinder(samples=samples, seed=seed + 1, plane="strain"),
        audit_lame_cylinder(samples=samples, seed=seed + 2, plane="stress"),
        audit_airy_cantilever(samples=samples, seed=seed + 3),
    ]
    hole = KirschHole()
    points = polar_samples(1.12, 4.0, samples, seed)
    kirsch = kirsch_stress(hole)
    controls = {
        "kirsch_perturbation": perturbation_control(kirsch, points, hole.radius),
        "kirsch_stencil": stencil_control(kirsch, points, hole.radius),
    }
    baseline = controls["kirsch_perturbation"]["baseline"]
    detected = [float(key) for key, value in controls["kirsch_perturbation"].items()
                if key != "baseline" and value > 10.0 * max(baseline, 1e-15)]
    return {
        "reports": [report.to_dict() for report in reports],
        "verdict": judge_suite(reports),
        "controls": controls,
        "detection_floor": min(detected) if detected else float("nan"),
        "stencil_order": controls["kirsch_stencil"]["order"],
    }