"""Exact plane-stress solution for a rotating annular disc (Michel's problem).

Why this module exists
----------------------
Every verifier run in this repository up to now loaded the continuum by *edge
tractions* (Cook membrane, Kirsch plate, curved bar) or by point loads.  No
merged check exercised the **body-force load path**: inertia, gravity and
centrifugal loads are assembled as a consistent nodal vector, and a wrong
lumping rule or a missing ``r`` factor in the integrand is invisible to a
traction-loaded test.

The rotating disc is the cheapest problem that closes that gap *and* has an
exact closed form: for an annular disc of inner radius ``a``, outer radius
``b``, uniform thickness ``t``, density ``rho`` and angular velocity ``omega``
the axisymmetric plane-stress field is Michel's solution

.. math::

   \\sigma_r = A - \\frac{B}{r^2} - \\frac{3+\\nu}{8}\\rho\\omega^2 r^2, \\qquad
   \\sigma_\\theta = A + \\frac{B}{r^2} - \\frac{1+3\\nu}{8}\\rho\\omega^2 r^2,

with the two free-surface conditions ``sigma_r(a) = sigma_r(b) = 0`` giving

.. math::

   B = \\frac{3+\\nu}{8}\\rho\\omega^2 a^2 b^2, \\qquad
   A = \\frac{3+\\nu}{8}\\rho\\omega^2 (a^2+b^2),

and the radial displacement

.. math::

   u_r(r) = \\frac{r}{E}\\Big[(1-\\nu)A + \\frac{(1+\\nu)B}{r^2}
            - \\frac{1-\\nu^2}{8}\\rho\\omega^2 r^2\\Big].

The solution is *plane stress* (``sigma_z = 0``), which is exactly the
idealisation :class:`tensorfem.continuum.ContinuumModel` implements, so the FE
comparison is like-for-like with no 3-D correction.  The finite-element side is
a quarter annulus with symmetry cuts, exactly the sector geometry already used
by :func:`tensorfem.lame_reference.sector_q4_mesh`.

Ship relevance: propeller hubs, shaft flanges, pump impellers, flywheels and
turbine rotors are all rotating discs with a bore, and the burst margin is set
by the bore hoop stress ``sigma_theta(a)`` -- the quantity this module returns
in closed form (`bore_hoop_stress`).

All functions are pure ``torch`` (float64) so the module has no SciPy
dependency; the benchmark and the tests self-validate the reference against the
governing ODE, compatibility, the free surfaces, the global force balance and
the solid-disc limit before any FE number is quoted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "RotatingDiscState",
    "disc_coefficients",
    "rotating_disc_state",
    "solid_rotating_disc_state",
    "bore_hoop_stress",
    "sector_body_force_resultant",
    "sector_hoop_traction_resultant",
    "centrifugal_nodal_loads",
]


@dataclass(frozen=True)
class RotatingDiscState:
    """Exact plane-stress state sampled at ``radius``."""

    radius: torch.Tensor
    radial_stress: torch.Tensor
    hoop_stress: torch.Tensor
    radial_displacement: torch.Tensor
    coefficients: tuple[float, float]

    @property
    def shear_stress(self) -> torch.Tensor:
        """``tau_rtheta`` -- identically zero for an axisymmetric field."""
        return torch.zeros_like(self.radial_stress)


def _check_disc_parameters(inner_radius: float, outer_radius: float, density: float, poisson: float) -> None:
    if not math.isfinite(outer_radius) or outer_radius <= 0.0:
        raise ValueError("outer_radius must be finite and positive")
    if not (0.0 <= inner_radius < outer_radius):
        raise ValueError("require 0 <= inner_radius < outer_radius")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")


def disc_coefficients(
    *,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    poisson: float,
) -> tuple[float, float]:
    """Return Michel's ``(A, B)`` from the two traction-free free surfaces.

    ``inner_radius == 0`` is the solid-disc limit, where ``B`` must vanish (a
    ``1/r^2`` term would be singular at the centre); the closed form is returned
    directly instead of solving a degenerate 2x2 system.
    """
    _check_disc_parameters(inner_radius, outer_radius, density, poisson)
    prefactor = (3.0 + poisson) / 8.0 * density * omega**2
    if inner_radius == 0.0:
        return prefactor * outer_radius**2, 0.0
    return (
        prefactor * (inner_radius**2 + outer_radius**2),
        prefactor * inner_radius**2 * outer_radius**2,
    )


def rotating_disc_state(
    radius,
    *,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    young: float,
    poisson: float,
    radius_tolerance: float = 0.0,
) -> RotatingDiscState:
    """Exact annular-disc state; ``inner_radius`` must be strictly positive.

    ``radius_tolerance`` widens the accepted band to ``[a - tol, b + tol]``.
    It exists because the centroid of a *chordal* Q4 on a polar mesh lies up to
    one sagitta inside the bore circle: on a mesh with ``n_theta`` divisions the
    innermost centroid radius can be ``a (dtheta/2)^2 / 2`` smaller than ``a``,
    and the benchmark must be able to evaluate the exact field there.  The
    default (``0``) keeps the strict check for ordinary use.
    """
    _check_disc_parameters(inner_radius, outer_radius, density, poisson)
    if inner_radius <= 0.0:
        raise ValueError("annular state requires inner_radius > 0; use solid_rotating_disc_state")
    if not math.isfinite(young) or young <= 0.0:
        raise ValueError("young must be finite and positive")
    r = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
    if bool((r <= 0.0).any()):
        raise ValueError("sample radii must be positive")
    if radius_tolerance < 0.0:
        raise ValueError("radius_tolerance must be non-negative")
    tolerance = 1e-12 * outer_radius + radius_tolerance
    if bool((r < inner_radius - tolerance).any()) or bool((r > outer_radius + tolerance).any()):
        raise ValueError("sample radii must lie inside [inner_radius, outer_radius]")

    a_coef, b_coef = disc_coefficients(
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        density=density,
        omega=omega,
        poisson=poisson,
    )
    rw2 = density * omega**2
    sigma_r = a_coef - b_coef / r**2 - (3.0 + poisson) / 8.0 * rw2 * r**2
    sigma_t = a_coef + b_coef / r**2 - (1.0 + 3.0 * poisson) / 8.0 * rw2 * r**2
    u_r = r / young * (
        (1.0 - poisson) * a_coef
        + (1.0 + poisson) * b_coef / r**2
        - (1.0 - poisson**2) / 8.0 * rw2 * r**2
    )
    return RotatingDiscState(r, sigma_r, sigma_t, u_r, (a_coef, b_coef))


def solid_rotating_disc_state(
    radius,
    *,
    outer_radius: float,
    density: float,
    omega: float,
    young: float,
    poisson: float,
) -> RotatingDiscState:
    """Michel's solid-disc solution, regular at ``r = 0``.

    ``sigma_r = (3+nu)/8 rho omega^2 (b^2 - r^2)``,
    ``sigma_theta = (3+nu)/8 rho omega^2 (b^2 - (1+3nu)/(3+nu) r^2)``,
    ``u_r = (1-nu)/(8E) rho omega^2 r [(3+nu) b^2 - (1+nu) r^2]``.
    """
    _check_disc_parameters(0.0, outer_radius, density, poisson)
    if not math.isfinite(young) or young <= 0.0:
        raise ValueError("young must be finite and positive")
    r = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
    if bool((r < 0.0).any()):
        raise ValueError("sample radii must be non-negative")
    if bool((r > outer_radius * (1.0 + 1e-12)).any()):
        raise ValueError("sample radii must not exceed outer_radius")
    rw2 = density * omega**2
    prefactor = (3.0 + poisson) / 8.0 * rw2
    sigma_r = prefactor * (outer_radius**2 - r**2)
    sigma_t = prefactor * (outer_radius**2 - (1.0 + 3.0 * poisson) / (3.0 + poisson) * r**2)
    u_r = (1.0 - poisson) / (8.0 * young) * rw2 * r * (
        (3.0 + poisson) * outer_radius**2 - (1.0 + poisson) * r**2
    )
    return RotatingDiscState(r, sigma_r, sigma_t, u_r, (prefactor * outer_radius**2, 0.0))


def bore_hoop_stress(
    *,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    poisson: float,
) -> float:
    """Closed-form hoop stress at the bore -- the disc burst driver.

    ``sigma_theta(a) = (3+nu)/4 rho omega^2 b^2 + (1-nu)/4 rho omega^2 a^2``,
    which tends to ``2 x`` the centre stress of the solid disc as ``a -> 0``
    (the classical factor-of-two stress concentration of a central hole).
    """
    _check_disc_parameters(inner_radius, outer_radius, density, poisson)
    rw2 = density * omega**2
    return (3.0 + poisson) / 4.0 * rw2 * outer_radius**2 + (1.0 - poisson) / 4.0 * rw2 * inner_radius**2


def sector_body_force_resultant(
    *,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    thickness: float,
    sector_angle: float = math.pi / 2.0,
) -> float:
    """Total ``x``-component of the centrifugal load on an annular sector.

    ``int int rho omega^2 x dA = rho omega^2 sin(alpha) (b^3 - a^3)/3`` times the
    thickness; it must be balanced by the reaction on the symmetry cut.
    """
    if not (0.0 <= inner_radius < outer_radius):
        raise ValueError("require 0 <= inner_radius < outer_radius")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")
    if not (0.0 < sector_angle <= 2.0 * math.pi):
        raise ValueError("sector_angle must lie in (0, 2*pi]")
    return (
        density * omega**2 * math.sin(sector_angle) * (outer_radius**3 - inner_radius**3) / 3.0 * thickness
    )


def sector_hoop_traction_resultant(
    *,
    inner_radius: float,
    outer_radius: float,
    density: float,
    omega: float,
    young: float,
    poisson: float,
    thickness: float,
    samples: int = 200001,
) -> float:
    """``t * int_a^b sigma_theta dr`` -- the membrane force on a symmetry cut.

    Equals the ``x`` body-force resultant of the quarter sector by radial
    equilibrium, so it is an independent check on the reaction at the cut.
    """
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")
    if samples < 3:
        raise ValueError("samples must be >= 3")
    radius = torch.linspace(inner_radius, outer_radius, samples, dtype=torch.float64)
    state = rotating_disc_state(
        radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        density=density,
        omega=omega,
        young=young,
        poisson=poisson,
    )
    return thickness * float(torch.trapezoid(state.hoop_stress, radius))


_GAUSS = ((-1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
          (1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
          (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
          (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)))


def centrifugal_nodal_loads(
    nodes,
    cells,
    *,
    thickness: float,
    density: float,
    omega: float,
    consistent: bool = True,
) -> torch.Tensor:
    """Assemble the centrifugal body load ``b = rho omega^2 x`` on Q4 cells.

    ``consistent=True`` uses the exact 2x2 Gauss integral
    ``f_a = int_e N_a rho omega^2 x t dA`` (exact for bilinear ``N`` and linear
    ``x``).  ``consistent=False`` is the equal-split lumping of the same
    element load, kept so the benchmark can price the lumping error.

    The returned vector has shape ``[2 * n_nodes]`` with ``[u_x, u_y]``
    interleaving, matching :class:`tensorfem.continuum.ContinuumModel.forces`.
    """
    if not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be finite and positive")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    coords_all = torch.as_tensor(nodes, dtype=torch.float64)
    connectivity = torch.as_tensor(cells, dtype=torch.long)
    if coords_all.ndim != 2 or coords_all.shape[1] != 2:
        raise ValueError("nodes must have shape [n_nodes, 2]")
    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise ValueError("Q4 connectivity must have shape [n_cells, 4]")
    if connectivity.numel() and (
        int(connectivity.min()) < 0 or int(connectivity.max()) >= coords_all.shape[0]
    ):
        raise ValueError("invalid node in element connectivity")

    coords = coords_all[connectivity]
    rw2 = density * omega**2 * thickness
    n_cells = connectivity.shape[0]
    element_load = torch.zeros(n_cells, 4, 2, dtype=torch.float64)

    shape = []
    gradients = []
    for xi, eta in _GAUSS:
        shape.append(torch.tensor(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ], dtype=torch.float64) * 0.25)
        gradients.append((
            torch.tensor([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=torch.float64) * 0.25,
            torch.tensor([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=torch.float64) * 0.25,
        ))

    if consistent:
        for (n_vec, (d_xi, d_eta)) in zip(shape, gradients):
            j11 = torch.einsum("a,ea->e", d_xi, coords[:, :, 0])
            j12 = torch.einsum("a,ea->e", d_xi, coords[:, :, 1])
            j21 = torch.einsum("a,ea->e", d_eta, coords[:, :, 0])
            j22 = torch.einsum("a,ea->e", d_eta, coords[:, :, 1])
            det_j = j11 * j22 - j12 * j21
            if bool((det_j <= 0.0).any()):
                raise ValueError("non-positive Jacobian: cells must be counter-clockwise and non-degenerate")
            x_gp = torch.einsum("a,ead->ed", n_vec, coords)
            element_load += (
                n_vec.reshape(1, 4, 1)
                * det_j.reshape(-1, 1, 1)
                * rw2
                * x_gp.reshape(-1, 1, 2)
            )
    else:
        quad_area = (coords[:, 0, 0] * (coords[:, 1, 1] - coords[:, 3, 1])
                     + coords[:, 1, 0] * (coords[:, 2, 1] - coords[:, 0, 1])
                     + coords[:, 2, 0] * (coords[:, 3, 1] - coords[:, 1, 1])
                     + coords[:, 3, 0] * (coords[:, 0, 1] - coords[:, 2, 1])) * 0.5
        if bool((quad_area <= 0.0).any()):
            raise ValueError("non-positive cell area: cells must be counter-clockwise")
        element_load += (
            (quad_area.reshape(-1, 1, 1) / 4.0) * rw2 * coords
        )

    forces = torch.zeros(2 * coords_all.shape[0], dtype=torch.float64)
    for local in range(4):
        forces.index_add_(0, 2 * connectivity[:, local], element_load[:, local, 0])
        forces.index_add_(0, 2 * connectivity[:, local] + 1, element_load[:, local, 1])
    return forces