"""Sound radiated by a vibrating circular cylinder: J&F kernel, window closure, ESM.

Why this module exists
----------------------
``plate_acoustics`` / ``plate_radiation_chain`` close the structure-acoustic
chain for a *flat* baffled panel.  A stiffened hull section is, however, curved:
the radiating surface is a finite circular cylinder, and the classical anchor for
a cylinder mode is the **Junger & Feit** modal radiation resistance of an infinite
cylinder (Junger & Feit, *Sound, Structures and Their Interaction*, 2nd ed.,
ch. 8, "radiation from cylindrical radiators").  Nothing in this repository could
evaluate it, so the cylinder branch of the structure-acoustic chain had no
reference quantity at all.

Three numerically independent evaluations of the radiated power ``Pi`` for the
*same* prescribed normal-velocity field ``v_n(x, theta)`` on the cylinder surface
``r = R``, ``0 <= x <= L`` are supplied here:

1. :func:`infinite_cylinder_conductance` -- the **analytic kernel**.  For the
   surface velocity ``v = V cos(n theta) exp(i k_x x)`` the exterior pressure is
   the outgoing solution ``p = P H_n^(1)(kappa r) cos(n theta) exp(i k_x x)``,
   ``kappa = sqrt(k^2 - k_x^2)``, and the surface condition fixes ``P``.  The
   per-unit-length conductance follows in closed form.  This *is* the J&F modal
   radiation resistance written as a conductance (only its real part is needed
   here; the imaginary part would be the modal added mass).

2. :func:`finite_cylinder_mode_power` -- the **wavenumber-window closure**.  The
   J&F closed form is per axial wavenumber; a simply supported finite mode is
   ``sin(m pi x/L)``, whose axial spectrum is analytic.  Integrating the analytic
   kernel against that spectrum (the standard finite-panel closure) gives the
   power of the finite mode.  As ``L -> infinity`` with ``m pi/L`` fixed it must
   reproduce (1); that limit is pinned in ``tests/test_cylinder_radiation.py``.

3. :func:`equivalent_source_power` -- the **numerical radiation integral**.  A
   boundary-integral (exterior Helmholtz) solve without a surface mesh: monopole
   sources sit on a retracted interior surface, their strengths are fitted to the
   prescribed normal velocity in a Tikhonov-regularised least-squares sense
   (equivalent-source / source-simulation method, Koopmann & Fahnline) and the
   total power follows from the closed-form monopole cross-power
   ``j0(k r_ij)/(8 pi rho c)``.  The end caps enter as *rigid* (zero normal
   velocity) matching points, so the closed body is the geometry a test article
   would present.  The constant of the cross-power formula is pinned on the exact
   pulsating-sphere power ``Pi = 2 pi R^2 rho c |V|^2 (kR)^2/(1+(kR)^2)``.

Route (3) carries curvature and finite length exactly (to source-fit
convergence); route (2) carries them through the analytic kernel.  Their
difference is a genuine measure of the end-cap / finite-length physics that the
J&F closure ignores -- it is reported, never tuned away.

Structural side
---------------
:func:`donnell_frequencies` supplies the closed-form three-root Donnell anchor and
:func:`shell_radial_amplitudes` projects a 6-DOF/node Shell4 eigenvector (from
:mod:`tensorfem.cylindrical_shell_modal`) onto the surface normal.  The cylinder
quantity verified here is the *radiation operator*: it is a property of the
surface velocity field and the frequency, not of any particular eigenvalue, so the
gates are evaluated with exactly prescribed modal velocity fields and the FEM shell
model enters through the driven response and the mode-shape cross-check.  The shell
*eigenvalue* problem of that element is deliberately not claimed as verified --
Donnell overestimates the low-``n`` inextensional branch (measured factor ~10 at
``n = 2``) and the element's own mesh sequence is not anchored yet; that is
recorded as an open item, not papered over.

Conventions
-----------
Peak amplitudes, ``exp(-i omega t)``, normal velocity positive along the outward
normal.  Radiation efficiency ``sigma = Pi / (rho c S <v^2>)`` with ``S`` the
radiating (lateral) area and ``<v^2>`` the surface mean square normal velocity.

References
----------
* Junger, M. C. & Feit, D. (1986). *Sound, Structures and Their Interaction*,
  2nd ed., MIT Press, ch. 8.
* Koopmann, G. H. & Fahnline, J. B. (1997). *Designing Quiet Structures: A Sound
  Power Minimization Approach*, Academic Press.
* Leissa, A. W. (1973). *Vibration of Shells*, NASA SP-288 (Donnell equations).
* Fahy, F. & Gardonio, P. (2007). *Sound and Structural Vibration*, 2nd ed.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from scipy.special import hankel1

__all__ = [
    "compact_body_power",
    "shell_radial_amplitudes",
    "CylinderRadiationResult",
    "EquivalentSourceResult",
    "axial_spectrum",
    "conductance_sweep",
    "cylinder_mode_field",
    "cylinder_surface_grid",
    "donnell_frequencies",
    "equivalent_source_power",
    "finite_cylinder_mode_power",
    "fourier_cylinder_mode",
    "infinite_cylinder_conductance",
    "infinite_cylinder_efficiency",
    "monopole_power_matrix",
]

_RHO_AIR = 1.225
_C_AIR = 343.0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    result = float(value)
    if not (-1.0 < result < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    return result


def _count(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result != value:
        raise ValueError(f"{name} must be an integer")
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


# ---------------------------------------------------------------------------
# Analytic anchor 1: Donnell shell roots (three branches per (m, n))
# ---------------------------------------------------------------------------

def donnell_frequencies(
    m: int,
    n: int,
    *,
    radius: float,
    length: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
) -> tuple[float, float, float]:
    """The three Donnell natural frequencies (Hz) of the mode ``(m, n)``.

    Donnell-Mushtari shell theory with simply supported (diaphragm) ends gives the
    nondimensional eigenproblem ``det|C - Omega^2 I| = 0`` with

    ``C11 = lam^2 + (1-nu)/2 n^2``, ``C12 = (1+nu)/2 lam n``, ``C13 = nu lam``,
    ``C22 = (1-nu)/2 lam^2 + n^2``, ``C23 = n``,
    ``C33 = 1 + kappa (lam^2 + n^2)^2``,

    ``lam = m pi R/L``, ``kappa = t^2/(12 R^2)`` and
    ``omega^2 = Omega^2 E / (rho (1-nu^2) R^2)`` (Leissa 1973, NASA SP-288).

    Roots are returned in ascending frequency order.  Measured against the
    classical inextensional estimate, the lowest of the three is Donnell's
    *membrane-type* branch: Donnell kinematics (no transverse shear, and the hoop
    term ``C33 = 1 + ...`` not cancelled by the in-plane coupling) overestimate the
    true inextensional (bending) branch of a thin shell by up to an order of
    magnitude at ``n <= 2``.  The roots are therefore used here only as a
    *branch-structure* anchor, never as the shell frequency reference; an accurate
    low-``n`` shell eigenvalue reference needs Fluegge/Goldenveizer kinematics.
    """
    m = _count("m", m)
    n = _count("n", n, minimum=0)
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    density = _positive("density", density)
    poisson = _poisson(poisson)

    lam = m * math.pi * radius / length
    kappa = thickness ** 2 / (12.0 * radius ** 2)
    nu = poisson
    c11 = lam ** 2 + 0.5 * (1.0 - nu) * n ** 2
    c12 = 0.5 * (1.0 + nu) * lam * n
    c13 = nu * lam
    c22 = 0.5 * (1.0 - nu) * lam ** 2 + n ** 2
    c23 = float(n)
    c33 = 1.0 + kappa * (lam ** 2 + n ** 2) ** 2
    matrix = torch.tensor(
        [[c11, c12, c13], [c12, c22, c23], [c13, c23, c33]], dtype=torch.float64
    )
    omega_sq = torch.linalg.eigvalsh(matrix)
    scale = young / (density * (1.0 - nu ** 2) * radius ** 2)
    frequencies = [
        math.sqrt(max(float(value), 0.0) * scale) / (2.0 * math.pi)
        for value in omega_sq.tolist()
    ]
    return (frequencies[0], frequencies[1], frequencies[2])


# ---------------------------------------------------------------------------
# Analytic kernel: infinite-cylinder (Junger & Feit) modal conductance
# ---------------------------------------------------------------------------

def _kappa(k: float, k_axial: float) -> complex:
    """Radial wavenumber with the outgoing (``val > 0``) / decaying branch."""
    value = k * k - k_axial * k_axial
    if value >= 0.0:
        return complex(math.sqrt(value), 0.0)
    return complex(0.0, math.sqrt(-value))


def infinite_cylinder_conductance(
    n: int,
    k: float,
    k_axial: float,
    *,
    radius: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
) -> float:
    """Per-unit-length radiation conductance of one infinite-cylinder mode.

    Surface velocity ``v = V cos(n theta) exp(i k_x x)``; exterior pressure
    ``p = P H_n^(1)(kappa r) cos(n theta) exp(i k_x x)`` with
    ``P = -i omega rho V / (kappa H_n^(1)'(kappa R))``.  Then

    ``Pi' = -(1/2) Re[P V*] R int_0^{2 pi} cos^2(n theta) dtheta
          = (1/2) Re[-P/V] R pi (1 + delta_{n0}) |V|^2``.

    Returns ``G = Pi'/|V|^2`` in ``kg/(m s)``.  ``G >= 0`` to machine precision,
    which the tests use as a sign check.
    """
    n = _count("n", n, minimum=0)
    k = _positive("k", k)
    radius = _positive("radius", radius)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    omega = k * c
    kap = _kappa(k, float(k_axial))
    x = kap * radius
    h_n = hankel1(n, x)
    h_n_prime = 0.5 * (hankel1(n - 1, x) - hankel1(n + 1, x))
    ratio = -1j * omega * rho * h_n / (kap * h_n_prime)      # P / V
    return float((-0.5 * ratio.real) * radius * math.pi * (1.0 + (n == 0)))


def infinite_cylinder_efficiency(
    n: int,
    k: float,
    k_axial: float,
    *,
    radius: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
) -> float:
    """``sigma = Pi'/(rho c 2 pi R <v^2>_theta)`` with ``<v^2> = |V|^2/2``."""
    conductance = infinite_cylinder_conductance(
        n, k, k_axial, radius=radius, rho=rho, c=c
    )
    return conductance / (0.5 * rho * c * 2.0 * math.pi * radius)


def conductance_sweep(
    n: int,
    ks: Sequence[float],
    *,
    radius: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
    k_axial: float = 0.0,
) -> torch.Tensor:
    """``sigma_inf(k)`` of a ring mode (``k_x = 0`` by default) over ``ks``."""
    values = [
        infinite_cylinder_efficiency(n, float(k), k_axial, radius=radius, rho=rho, c=c)
        for k in ks
    ]
    return torch.tensor(values, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Finite-length closure: axial wavenumber spectrum of a sine mode
# ---------------------------------------------------------------------------

def axial_spectrum(
    k_axial: torch.Tensor, length: float, half_waves: int
) -> torch.Tensor:
    """``int_0^L sin(m pi x/L) exp(-i k_x x) dx`` in closed form.

    With ``q = m pi/L`` the antiderivative is ``q (1 - (-1)^m exp(-i k_x L)) /
    (q^2 - k_x^2)``, with the removable-singularity limit ``-i L/2`` at
    ``k_x = q``.
    """
    length = _positive("length", length)
    m = _count("half_waves", half_waves)
    q = m * math.pi / length
    discriminant = q * q - k_axial * k_axial
    numerator = q * (1.0 - (-1.0) ** m * torch.exp(-1j * k_axial * length))
    threshold = 1e-10 * q * q
    nonsingular = torch.abs(discriminant) > threshold
    safe = torch.where(nonsingular, discriminant, torch.ones_like(discriminant))
    value = numerator / safe
    limit = torch.full(k_axial.shape, -0.5j * length, dtype=torch.complex128)
    return torch.where(nonsingular, value, limit)


@dataclass(frozen=True)
class CylinderRadiationResult:
    """Radiated power / efficiency of one finite-cylinder mode."""

    m: int
    n: int
    frequency: float
    ka: float
    power: float
    sigma: float
    mean_square_velocity: float
    label: str = field(default="")

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.label or f'mode({self.m},{self.n})'} f={self.frequency:.1f} Hz "
            f"ka={self.ka:.3f} sigma={self.sigma:.4e}"
        )


def finite_cylinder_mode_power(
    m: int,
    n: int,
    *,
    radius: float,
    length: float,
    frequency: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
    velocity_amplitude: float = 1.0,
    samples: int = 8001,
    half_width: float = 12.0,
) -> CylinderRadiationResult:
    """Power of a finite simply supported cylinder mode (window closure).

    ``v(x, theta) = V sin(m pi x/L) cos(n theta)`` on ``0 <= x <= L``.  Since the
    infinite-cylinder eigenfunctions are orthogonal in ``k_x``,

    ``Pi = (1/2 pi) int dk_x G_n(k_x) |V(k_x)|^2``

    with ``G_n`` from :func:`infinite_cylinder_conductance`.  The integration band
    is ``|k_x| <= max(k, m pi/L) + half_width * 2 pi/L`` (the mode spectrum decays
    like ``1/k_x^2`` and the conductance is evanescent outside it); ``samples``
    points resolve the spectral peak of width ``~ pi/L``.  Both controls are
    convergence-checked in the tests.
    """
    m = _count("m", m)
    n = _count("n", n, minimum=0)
    radius = _positive("radius", radius)
    length = _positive("length", length)
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    velocity_amplitude = float(velocity_amplitude)
    samples = _count("samples", samples, minimum=201)

    k = 2.0 * math.pi * frequency / c
    k_peak = m * math.pi / length
    span = max(k, k_peak) + half_width * 2.0 * math.pi / length
    grid = torch.linspace(-span, span, samples, dtype=torch.float64)
    spacing = float(grid[1] - grid[0])
    spectrum = axial_spectrum(grid, length, m)
    conductance = torch.tensor(
        [
            infinite_cylinder_conductance(
                n, k, float(kx), radius=radius, rho=rho, c=c
            )
            for kx in grid.tolist()
        ],
        dtype=torch.float64,
    )
    power = float((conductance * spectrum.abs() ** 2).sum()) * spacing / (2.0 * math.pi)
    power *= velocity_amplitude ** 2
    # <sin^2> = 1/2 and <cos^2(n theta)> = 1/2 (n >= 1) or 1 (n = 0).
    mean_square = velocity_amplitude ** 2 * (0.25 if n > 0 else 0.5)
    area = 2.0 * math.pi * radius * length
    sigma = (
        power / (rho * c * area * mean_square) if mean_square > 0.0 else 0.0
    )
    return CylinderRadiationResult(
        m=m,
        n=n,
        frequency=frequency,
        ka=k * radius,
        power=power,
        sigma=sigma,
        mean_square_velocity=mean_square,
        label=f"window({m},{n})",
    )


# ---------------------------------------------------------------------------
# Numerical radiation integral: equivalent-source (source simulation) method
# ---------------------------------------------------------------------------

def monopole_power_matrix(
    source_positions: torch.Tensor, k: float, rho: float, c: float
) -> torch.Tensor:
    """Power matrix ``R_ij = j0(k r_ij)/(8 pi rho c)`` of free-space monopoles.

    For peak monopole strengths ``q_i`` (defined by ``p = sum_i q_i
    exp(ik r)/(4 pi r)``) the radiated power is ``Pi = q^H R q``.  One source gives
    ``|q|^2/(8 pi rho c)``, i.e. the classical ``rho c k^2 |Q|^2/(8 pi)`` with
    ``q = i omega rho Q``.
    """
    k = _positive("k", k)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    if source_positions.dim() != 2 or source_positions.shape[1] != 3:
        raise ValueError("source_positions must be an (n, 3) array")
    diff = source_positions[:, None, :] - source_positions[None, :, :]
    r = torch.linalg.norm(diff, dim=2)
    kr = k * r
    small = kr < 1e-6
    safe = torch.where(small, torch.ones_like(kr), kr)
    j0 = torch.where(small, 1.0 - kr ** 2 / 6.0, torch.sin(safe) / safe)
    return j0 / (8.0 * math.pi * rho * c)


@dataclass(frozen=True)
class EquivalentSourceResult:
    """Outcome of one equivalent-source exterior solve."""

    power: float
    sigma: float
    normal_velocity_residual: float
    condition_number: float
    n_sources: int
    n_match_points: int
    k: float


def equivalent_source_power(
    points: torch.Tensor,
    normals: torch.Tensor,
    normal_velocity: torch.Tensor,
    *,
    frequency: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
    sources: Optional[torch.Tensor] = None,
    shrink: float = 0.6,
    regularization: float = 1e-6,
    area_weights: Optional[torch.Tensor] = None,
) -> EquivalentSourceResult:
    """Radiated power of a prescribed normal-velocity field by the ESM.

    ``points``/``normals`` are ``(N, 3)``; ``normal_velocity`` is ``(N,)`` (peak
    amplitude, may be complex).  ``sources`` defaults to the surface contracted
    towards the body centroid by ``shrink`` (valid for a convex body such as a
    capped cylinder).  ``regularization`` is the Tikhonov parameter relative to
    the largest eigenvalue of the normal matrix.  ``area_weights`` (optional) are
    used only to form the mean square velocity and hence ``sigma``.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if normals.shape != points.shape:
        raise ValueError("normals must have the same shape as points")
    points = points.double()
    normals = normals.double()
    velocity = normal_velocity.reshape(-1)
    if velocity.numel() != points.shape[0]:
        raise ValueError("normal_velocity must have one entry per point")
    k = 2.0 * math.pi * frequency / c
    omega = 2.0 * math.pi * frequency
    n_match = points.shape[0]

    if sources is None:
        centroid = points.mean(dim=0, keepdim=True)
        sources = centroid + shrink * (points - centroid)
    else:
        sources = sources.double()
        if sources.dim() != 2 or sources.shape[1] != 3:
            raise ValueError("sources must be an (M, 3) array")
    n_sources = sources.shape[0]

    diff = points[:, None, :] - sources[None, :, :]            # (N, M, 3)
    r = torch.linalg.norm(diff, dim=2).clamp_min(1e-12)        # (N, M)
    projection = (diff * normals[:, None, :]).sum(dim=2) / r   # n . (x-s)/r
    ik_r = 1j * k * r
    green = torch.exp(ik_r) / (4.0 * math.pi * r)
    transfer = (1j / (omega * rho)) * (ik_r - 1.0) * green * projection / r

    target = velocity.to(torch.complex128)
    gram = transfer.conj().T @ transfer
    rhs = transfer.conj().T @ target
    scale = float(torch.linalg.eigvalsh(gram.real.double()).abs().max())
    lam = max(regularization, 1e-12) * (scale if scale > 0.0 else 1.0)
    system = gram + lam * torch.eye(n_sources, dtype=torch.complex128)
    strengths = torch.linalg.solve(system, rhs)

    residual = float(
        torch.linalg.norm(transfer @ strengths - target)
        / torch.linalg.norm(target).clamp_min(1e-300)
    )
    values = torch.linalg.svdvals(transfer).abs()
    condition = float(values.max() / values.min().clamp_min(1e-300))

    power_matrix = monopole_power_matrix(sources.double(), k, rho, c)
    power = float(
        (strengths.conj() @ power_matrix.to(torch.complex128) @ strengths).real
    )

    msq = float("nan")
    sigma = float("nan")
    if area_weights is not None:
        weights = area_weights.reshape(-1).double()
        msq = float(((velocity.abs() ** 2) * weights).sum() / weights.sum())
        area = float(weights.sum())
        if msq > 0.0:
            sigma = power / (rho * c * area * msq)
    return EquivalentSourceResult(
        power=power,
        sigma=sigma,
        normal_velocity_residual=residual,
        condition_number=condition,
        n_sources=n_sources,
        n_match_points=n_match,
        k=k,
    )


# ---------------------------------------------------------------------------
# Cylinder surface sampling and mode identification
# ---------------------------------------------------------------------------

def cylinder_surface_grid(
    *,
    radius: float,
    length: float,
    n_axial: int,
    n_circ: int,
    with_caps: bool = True,
    cap_rings: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surface points/normals/weights of a capped cylinder (axis ``+x``).

    The lateral surface is a ``n_axial x n_circ`` grid (trapezoidal weights); with
    ``with_caps`` the two rigid end discs contribute ``cap_rings`` radial rings of
    ``n_circ`` points each (outer radius ``0.9 R``, at ``x = 0`` and ``x = L``,
    normals ``-x``/``+x``).  Cap points carry zero weight and zero velocity: they
    are *rigid*, present only as matching points of the exterior solve.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    n_axial = _count("n_axial", n_axial, minimum=2)
    n_circ = _count("n_circ", n_circ, minimum=3)

    x = torch.linspace(0.0, length, n_axial, dtype=torch.float64)
    theta = torch.linspace(0.0, 2.0 * math.pi, n_circ + 1, dtype=torch.float64)[:-1]
    xx, tt = torch.meshgrid(x, theta, indexing="ij")
    flat_x = xx.reshape(-1)
    flat_t = tt.reshape(-1)
    lateral = torch.stack(
        (flat_x, radius * torch.cos(flat_t), radius * torch.sin(flat_t)), dim=1
    )
    lateral_normals = torch.stack(
        (torch.zeros_like(flat_t), torch.cos(flat_t), torch.sin(flat_t)), dim=1
    )
    dx = float(x[1] - x[0])
    dtheta = 2.0 * math.pi / n_circ
    # Trapezoidal rule along the axis: the two end rows count half.  Using the
    # full width everywhere over-counts the area by n_axial/(n_axial-1) (5 % at the
    # 21-row grid used by the benchmark), which propagates straight into sigma.
    axial_weight = torch.full_like(x, dx)
    axial_weight[0] = 0.5 * dx
    axial_weight[-1] = 0.5 * dx
    lateral_weights = (
        axial_weight.repeat_interleave(n_circ) * (radius * dtheta)
    )

    points = [lateral]
    normals = [lateral_normals]
    weights = [lateral_weights]
    if with_caps:
        cap_rings = _count("cap_rings", cap_rings)
        radii = radius * torch.linspace(0.25, 0.9, cap_rings, dtype=torch.float64)
        rr, tc = torch.meshgrid(radii, theta, indexing="ij")
        flat_r = rr.reshape(-1)
        flat_c = tc.reshape(-1)
        y = flat_r * torch.cos(flat_c)
        z = flat_r * torch.sin(flat_c)
        for x_end, sign in ((0.0, -1.0), (length, 1.0)):
            cap = torch.stack(
                (torch.full_like(y, x_end), y, z), dim=1
            )
            cap_normals = torch.stack(
                (
                    torch.full_like(y, sign),
                    torch.zeros_like(y),
                    torch.zeros_like(y),
                ),
                dim=1,
            )
            points.append(cap)
            normals.append(cap_normals)
            weights.append(torch.zeros_like(y))
    return (
        torch.cat(points, dim=0),
        torch.cat(normals, dim=0),
        torch.cat(weights, dim=0),
        torch.tensor([n_axial, n_circ], dtype=torch.long),
    )


def cylinder_mode_field(
    x: torch.Tensor, theta: torch.Tensor, amplitude: float, m: int, n: int
) -> torch.Tensor:
    """``amplitude * sin(m pi x/L) cos(n theta)`` on flattened ``(x, theta)``."""
    m = _count("m", m)
    n = _count("n", n, minimum=0)
    return amplitude * torch.sin(m * math.pi * x) * torch.cos(n * theta)


def fourier_cylinder_mode(
    x: torch.Tensor,
    theta: torch.Tensor,
    radial_amplitude: torch.Tensor,
    *,
    length: float,
    axial_modes: int,
    circ_modes: int,
) -> tuple[dict[tuple[int, int], float], tuple[int, int], float]:
    """Least-squares identification of a nodal radial field in the sine basis.

    Basis: ``sin(p pi x/L) cos(q theta)`` and ``sin(p pi x/L) sin(q theta)`` for
    ``p = 1..axial_modes``, ``q = 0..circ_modes`` (the ``sin`` partner of ``q = 0``
    is identically zero and dropped).  Returns
    ``(coeffs, (p*, q*), captured_fraction)`` where ``coeffs`` is keyed by
    ``("cos"|"sin", p, q)`` (the two partners of one ``(p, q)`` are independent
    degrees of freedom and must not share a key), ``(p*, q*)`` is the dominant
    harmonic pair and the fraction is the share of the fitted energy carried by
    that single component times the share of the nodal field the fit captures.
    """
    length = _positive("length", length)
    axial_modes = _count("axial_modes", axial_modes)
    circ_modes = _count("circ_modes", circ_modes)
    x = x.reshape(-1).double()
    theta = theta.reshape(-1).double()
    values = radial_amplitude.reshape(-1).double()

    columns: list[torch.Tensor] = []
    labels: list[tuple[int, int]] = []
    keys: list[tuple[str, int, int]] = []
    for p in range(1, axial_modes + 1):
        axial = torch.sin(p * math.pi * x / length)
        for q in range(0, circ_modes + 1):
            columns.append(axial * torch.cos(q * theta))
            labels.append((p, q))
            keys.append(("cos", p, q))
            if q > 0:
                columns.append(axial * torch.sin(q * theta))
                labels.append((p, q))
                keys.append(("sin", p, q))
    design = torch.stack(columns, dim=1)
    coeffs, *_ = torch.linalg.lstsq(design, values.unsqueeze(1))
    coeffs = coeffs.reshape(-1)
    energy = coeffs ** 2
    total = float(energy.sum())
    mapping = {
        key: float(value) for key, value in zip(keys, coeffs.tolist())
    }
    sigma_names = torch.stack(
        [design[:, i] * coeffs[i] for i in range(len(labels))], dim=1
    ).sum(dim=1)
    captured = float(
        torch.linalg.norm(sigma_names) / torch.linalg.norm(values).clamp_min(1e-300)
    ) ** 2
    dominant = labels[int(torch.argmax(energy))]
    share = float(energy.max() / total) if total > 0.0 else 0.0
    return mapping, dominant, share * captured

# ---------------------------------------------------------------------------
# Compact-body (low ka) asymptote: monopole + dipole radiation
# ---------------------------------------------------------------------------

def compact_body_power(
    points: torch.Tensor,
    normals: torch.Tensor,
    normal_velocity: torch.Tensor,
    area_weights: torch.Tensor,
    *,
    frequency: float,
    rho: float = _RHO_AIR,
    c: float = _C_AIR,
) -> tuple[float, float, float]:
    """Low-``ka`` (compact radiator) power asymptote ``Pi_mono + Pi_dip``.

    For a body small compared with the wavelength the exterior field is the
    monopole + dipole multipole pair, so

    ``Pi = rho c k^2 |Q|^2/(8 pi) + rho c k^4 |D|^2/(24 pi)``,

    with the volume velocity ``Q = int v_n dS`` and the dipole moment
    ``D = int v_n r dS`` (peak amplitudes; the ``4 pi/3`` solid-angle average of
    ``|r_hat . D|^2`` is already inserted).  Modes with both ``Q = 0`` and
    ``D = 0`` (e.g. antisymmetric axial modes of a symmetric body) need the
    quadrupole term and are *not* covered -- the returned ``Q`` and ``|D|`` let a
    caller detect that case instead of trusting a null asymptote.

    Returns ``(power, Q, |D|)``.
    """
    frequency = _positive("frequency", frequency)
    rho = _positive("rho", rho)
    c = _positive("c", c)
    points = points.double()
    velocity = normal_velocity.reshape(-1).to(torch.complex128)
    weights = area_weights.reshape(-1).double()
    k = 2.0 * math.pi * frequency / c
    weighted = velocity * weights
    volume_velocity = weighted.sum()
    dipole = (points.to(torch.complex128) * weighted[:, None]).sum(dim=0)
    q2 = float((volume_velocity.conj() * volume_velocity).real)
    power = (
        rho * c * k ** 2 * q2 / (8.0 * math.pi)
        + rho * c * k ** 4 * float((dipole.conj() * dipole).real.sum())
        / (24.0 * math.pi)
    )
    return power, float(torch.abs(volume_velocity)), float(torch.linalg.norm(dipole))


# ---------------------------------------------------------------------------
# Shell FEM mode -> radial surface velocity
# ---------------------------------------------------------------------------

def shell_radial_amplitudes(
    nodes: torch.Tensor, node_theta: torch.Tensor, mode: torch.Tensor
) -> torch.Tensor:
    """Radial normal amplitude of a 6-DOF/node cylindrical shell eigenvector.

    The Shell4 element of :mod:`tensorfem.cylindrical_shell_modal` builds its
    local triad as ``e_r = (0, cos theta, -sin theta)`` for the *bending* (index 2,
    "w") degree of freedom; on the mesh generated by
    :func:`structured_cylindrical_mesh` (``y = R cos theta``, ``z = R sin theta``)
    that direction is ``e_w = (0, sin theta, cos theta)``.  A thin-shell bending
    mode therefore carries 98-99 % of its displacement along ``e_w`` (measured; the
    remaining 1-2 % is the in-plane membrane component), and the acoustic normal
    velocity is that projection.  The ``theta``-phase of the projection is a rigid
    rotation of the mode, which cannot change the radiated power.

    Returns ``(n_nodes,)`` radial amplitudes (the mode's overall sign is fixed by a
    positive mean).
    """
    nodes = nodes.double()
    theta = node_theta.reshape(-1).double()
    mode = mode.reshape(-1).double()
    if mode.numel() != 6 * nodes.shape[0]:
        raise ValueError("mode must have 6 entries per node")
    displacement = mode.reshape(-1, 6)[:, 0:3]
    direction = torch.stack(
        (torch.zeros_like(theta), torch.sin(theta), torch.cos(theta)), dim=1
    )
    radial = (displacement * direction).sum(dim=1)
    if float(radial.sum()) < 0.0:
        radial = -radial
    return radial
