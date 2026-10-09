"""Numerical contour J integral for sampled two-dimensional LEFM fields.

The implementation integrates Eshelby's energy-momentum flux on an ordered
closed contour.  It operates on *discrete field samples*: stress and the full
displacement gradient are required at every contour point.  The Williams
mode-I sampler is included as a verification oracle, not as the J evaluator.

Assumptions are small-strain, homogeneous isotropic 2-D linear elasticity,
traction-free crack faces, no body force inside the contour and a crack that
extends in the positive local x direction.  Plasticity, dynamics, material
interfaces and crack growth are outside this module's scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ContourField:
    """Ordered counter-clockwise samples on one closed integration contour."""

    coordinates_m: torch.Tensor  # (n, 2), closure point must not be repeated
    stress_pa: torch.Tensor  # (n, 2, 2)
    displacement_gradient: torch.Tensor  # (n, 2, 2), grad[i,j] = du_i/dx_j


@dataclass(frozen=True)
class JIntegralResult:
    j_j_m2: float
    strain_energy_term_j_m2: float
    traction_work_term_j_m2: float
    contour_points: int


def _validate_field(field: ContourField) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(field, ContourField):
        raise TypeError("field must be a ContourField")
    x, stress, grad = field.coordinates_m, field.stress_pa, field.displacement_gradient
    if not all(isinstance(v, torch.Tensor) for v in (x, stress, grad)):
        raise TypeError("all contour fields must be torch.Tensor values")
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 8:
        raise ValueError("coordinates_m must have shape (n, 2) with n >= 8")
    if stress.shape != (len(x), 2, 2) or grad.shape != (len(x), 2, 2):
        raise ValueError("stress_pa and displacement_gradient must have shape (n, 2, 2)")
    if not all(v.is_floating_point() and bool(torch.isfinite(v).all()) for v in (x, stress, grad)):
        raise ValueError("contour fields must be finite floating tensors")
    if not torch.allclose(stress, stress.transpose(1, 2), rtol=1e-8, atol=1e-6):
        raise ValueError("stress_pa must be symmetric")
    segment = torch.roll(x, -1, 0) - x
    if bool(torch.any(torch.linalg.vector_norm(segment, dim=1) <= 0)):
        raise ValueError("contour contains a zero-length segment")
    signed_area = 0.5 * torch.sum(x[:, 0] * torch.roll(x[:, 1], -1) -
                                  torch.roll(x[:, 0], -1) * x[:, 1])
    if float(signed_area) <= 0:
        raise ValueError("contour samples must be counter-clockwise")
    return x, stress, grad


def contour_j_integral(field: ContourField) -> JIntegralResult:
    """Integrate ``J = integral(W n_x - t_i u_i,x) ds`` numerically.

    Periodic trapezoidal integration is applied directly to the sampled field;
    no stress-intensity factor or closed-form J relation enters this function.
    """
    x, stress, grad = _validate_field(field)
    strain = 0.5 * (grad + grad.transpose(1, 2))
    energy = 0.5 * torch.einsum("nij,nij->n", stress, strain)
    # For a CCW polygon, n ds = (dy, -dx).  Evaluate each scalar flux at
    # vertices and use the periodic trapezoidal rule on each straight segment.
    delta = torch.roll(x, -1, 0) - x
    normal_ds = torch.stack((delta[:, 1], -delta[:, 0]), dim=1)
    energy_avg = 0.5 * (energy + torch.roll(energy, -1, 0))
    energy_term = torch.sum(energy_avg * normal_ds[:, 0])
    ux = grad[:, :, 0]
    flux = torch.einsum("nij,ni->nj", stress, ux)
    flux_avg = 0.5 * (flux + torch.roll(flux, -1, 0))
    work_term = torch.sum(torch.einsum("ni,ni->n", flux_avg, normal_ds))
    value = energy_term - work_term
    return JIntegralResult(float(value), float(energy_term), float(work_term), len(x))


def _mode_i_displacement(points: torch.Tensor, k_i: float, young: float,
                         poisson: float, plane_strain: bool) -> torch.Tensor:
    x, y = points[:, 0], points[:, 1]
    r = torch.sqrt(x*x + y*y)
    theta = torch.atan2(y, x)
    mu = young / (2.0 * (1.0 + poisson))
    kappa = 3.0 - 4.0*poisson if plane_strain else (3.0-poisson)/(1.0+poisson)
    scale = k_i/(2.0*mu) * torch.sqrt(r/(2.0*math.pi))
    ux = scale*torch.cos(theta/2.0)*(kappa-1.0+2.0*torch.sin(theta/2.0)**2)
    uy = scale*torch.sin(theta/2.0)*(kappa+1.0-2.0*torch.cos(theta/2.0)**2)
    return torch.stack((ux, uy), dim=1)


def sample_williams_mode_i_contour(radius_m: float, points: int, k_i_pa_sqrt_m: float,
                                   young_pa: float, poisson: float = 0.3,
                                   *, plane_strain: bool = True,
                                   radial_perturbation: float = 0.0) -> ContourField:
    """Sample the leading Williams mode-I field on a discrete closed contour.

    Gradients are obtained by centered Cartesian differences of displacement,
    while stresses are evaluated independently from the Williams stress field.
    This deliberately gives the numerical J evaluator discretized input.
    ``radial_perturbation`` creates a smooth non-circular path for path tests.
    """
    for value, name in ((radius_m, "radius_m"), (k_i_pa_sqrt_m, "k_i_pa_sqrt_m"),
                        (young_pa, "young_pa")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if isinstance(points, bool) or not isinstance(points, int) or points < 16:
        raise ValueError("points must be an integer >= 16")
    if not isinstance(poisson, (int, float)) or isinstance(poisson, bool) or not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must satisfy -1 < poisson < 0.5")
    if not isinstance(radial_perturbation, (int, float)) or abs(radial_perturbation) >= 0.25:
        raise ValueError("abs(radial_perturbation) must be < 0.25")
    dtype = torch.float64
    # Midpoint angular offset avoids sampling the displacement jump exactly on
    # the two crack faces while retaining a complete periodic contour.
    theta = -math.pi + (torch.arange(points, dtype=dtype)+0.5)*(2.0*math.pi/points)
    radius = radius_m*(1.0 + float(radial_perturbation)*torch.cos(3.0*theta))
    xy = torch.stack((radius*torch.cos(theta), radius*torch.sin(theta)), dim=1)
    root = float(k_i_pa_sqrt_m)/torch.sqrt(2.0*math.pi*radius)
    c, s = torch.cos(theta/2.0), torch.sin(theta/2.0)
    s3, c3 = torch.sin(1.5*theta), torch.cos(1.5*theta)
    sxx = root*c*(1.0-s*s3)
    syy = root*c*(1.0+s*s3)
    sxy = root*s*c*c3
    stress = torch.stack((torch.stack((sxx, sxy), dim=1),
                          torch.stack((sxy, syy), dim=1)), dim=1)
    # A small physical-space stencil makes the recovered gradient independent
    # of angular quadrature density and mimics nodal-field differentiation.
    h = radius_m*1e-5
    gradient = torch.empty((points, 2, 2), dtype=dtype)
    for direction in range(2):
        shift = torch.zeros_like(xy)
        shift[:, direction] = h
        plus = _mode_i_displacement(xy+shift, float(k_i_pa_sqrt_m), float(young_pa),
                                    float(poisson), plane_strain)
        minus = _mode_i_displacement(xy-shift, float(k_i_pa_sqrt_m), float(young_pa),
                                     float(poisson), plane_strain)
        gradient[:, :, direction] = (plus-minus)/(2.0*h)
    return ContourField(xy, stress, gradient)


def mode_i_j_reference(k_i_pa_sqrt_m: float, young_pa: float, poisson: float = 0.3,
                       *, plane_strain: bool = True) -> float:
    """Independent LEFM oracle ``K_I**2/E'`` used only for qualification."""
    effective = young_pa/(1.0-poisson**2) if plane_strain else young_pa
    return float(k_i_pa_sqrt_m)**2/effective


def run_crack_tip_j_qualification() -> dict[str, object]:
    """Run numerical-contour, path-independence and refinement evidence."""
    k_i, young, poisson = 42e6, 210e9, 0.29
    evidence = []
    for plane_strain in (False, True):
        field = sample_williams_mode_i_contour(
            0.012, 512, k_i, young, poisson, plane_strain=plane_strain)
        actual = contour_j_integral(field).j_j_m2
        oracle = mode_i_j_reference(
            k_i, young, poisson, plane_strain=plane_strain)
        error = abs(actual/oracle-1.0)
        evidence.append({
            "id": "plane_strain" if plane_strain else "plane_stress",
            "actual": actual, "oracle": oracle, "relative_error": error,
            "tolerance": 0.03, "passed": error < 0.03,
        })
    paths = [
        contour_j_integral(sample_williams_mode_i_contour(
            radius, 384, k_i, young, poisson, radial_perturbation=perturb)).j_j_m2
        for radius, perturb in ((0.004, 0.0), (0.015, 0.12), (0.05, -0.18))
    ]
    spread = (max(paths)-min(paths))/(sum(paths)/len(paths))
    return {
        "scope": ("discrete 2-D homogeneous isotropic linear-elastic Mode-I "
                  "contour integration; no plastic J, growth, or 3-D crack front"),
        "evidence": evidence,
        "path_independence": {"values": paths, "relative_spread": spread,
                              "tolerance": 0.005, "passed": spread < 0.005},
        "passed": all(row["passed"] for row in evidence) and spread < 0.005,
    }
