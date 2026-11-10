"""Two-way plate/fluid coupling: radiation impedance, coupled response, power balance.

The gap this module fills
-------------------------
``tensorfem.plate_acoustics`` answers "given a surface velocity, what is the
radiated power?" and ``tensorfem.plate_radiation_chain`` answers "given a *dry*
eigenvector, what is the radiation efficiency?".  Neither closes the loop: the
acoustic field pushes back on the plate, and the structure-acoustic *coupling*
is precisely that feedback.  This module implements it for a simply supported
rectangular plate in an infinite rigid baffle:

1. **Radiating half-space operator** (:func:`half_space_kernel`).  The on-surface
   pressure of a baffled planar source is

   .. math:: p(y) = \\frac{i\\omega\\rho}{2\\pi}\\int_S v(y')\\frac{e^{ikR}}{R}\\,dS',

   which is singular at ``R -> 0``.  The kernel is split as
   ``e^{ikR}/R = 1/R + (e^{ikR} - 1)/R``: the ``1/R`` part of the *own* panel is
   integrated analytically (:func:`_self_panel_integral`), the remainder is
   smooth and is handled by the panel rule.  That split is what lets the
   pressure route and the Rayleigh route agree instead of both carrying the same
   quadrature bias.

2. **Radiation impedance in the sine-mode basis**
   (:func:`radiation_impedance_matrix`).  ``Z = i\\omega\\rho W^T G W`` with ``G``
   the singular kernel and ``W = diag(panel areas)``; ``Re Z`` is the radiated
   power operator, ``Im Z / \\omega`` the added mass.

3. **Two-way coupled forced response** (:func:`coupled_modal_response`).  The
   partitioned fixed point

   .. math:: q^{(k+1)} = D^{-1}\\left(F - \\Phi^T W p(v^{(k)})\\right),
             \\qquad D_m = K_m - \\omega^2 m_m

   is iterated with Aitken dynamic relaxation, and the converged iterate is
   checked against the direct solve of ``(D + Z) q = F``.  Two residuals are
   reported: the interface-pressure residual
   ``||p^{k+1} - p^k|| / ||p^{k+1}||`` and the coupled-equation residual.

4. **Power conservation**.  For an undamped plate the coupled solution obeys
   ``P_in = P_rad`` exactly (multiply the modal equation by ``conj(i omega q)``
   and take the real part).  That identity is a free, extremely sharp check of
   the whole chain -- it fails at once if the sign of the pressure reaction, the
   area weights or the ``i omega rho`` factor is wrong.

Conventions: peak (not RMS) amplitudes, ``e^{-i omega t}``, half-space
``z > 0`` above the plate, plate normal velocity ``v > 0`` into the fluid.

Known limitation (measured, not hidden)
---------------------------------------
The discretised radiation resistance is positive semi-definite only up to the
panel-quadrature error: its most negative eigenvalue is about ``-3e-3`` of the
largest one on a 24x16 panel grid and shrinks roughly as ``h^2``
(``-1.3e-3`` at 32x32, ``-8e-4`` at 48x32).  For a *resolved* velocity field --
any single sine mode, or a coupled response dominated by low modes -- the
radiated power still matches the Rayleigh integral to 0.02-0.2 %, so the
efficiency is reliable in absolute terms to ~1e-3.  When the true power is a
near-cancellation of modal contributions (a strongly phase-mixed multi-mode
response at a frequency between wet resonances) the residue of the quadrature
error can decide the sign of a power of that size, and the reported efficiency
must not be trusted below ~1e-3.  The power-balance identity ``P_in = P_rad``
does *not* suffer from this: it is algebraic and holds to solver precision at
any grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

from ..acoustic_radiation import _gauss_legendre
from .levy_modes import PlateModalConfig, _positive

RHO_WATER = 1000.0
C_WATER = 1500.0
RHO_AIR = 1.225
C_AIR = 343.0


# ---------------------------------------------------------------------------
# Surface quadrature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadiationGrid:
    """Uniform midpoint panel grid over the wetted plate face."""

    x: torch.Tensor
    y: torch.Tensor
    weight_x: torch.Tensor
    weight_y: torch.Tensor
    panel_x: float
    panel_y: float

    @classmethod
    def uniform(
        cls, length: float, width: float, panels_x: int, panels_y: int
    ) -> "RadiationGrid":
        length = _positive("length", length)
        width = _positive("width", width)
        if int(panels_x) < 1 or int(panels_y) < 1:
            raise ValueError("panels_x and panels_y must be >= 1")
        nx, ny = int(panels_x), int(panels_y)
        hx, hy = length / nx, width / ny
        x = (torch.arange(nx, dtype=torch.float64) + 0.5) * hx
        y = (torch.arange(ny, dtype=torch.float64) + 0.5) * hy
        return cls(
            x=x,
            y=y,
            weight_x=torch.full((nx,), hx, dtype=torch.float64),
            weight_y=torch.full((ny,), hy, dtype=torch.float64),
            panel_x=hx,
            panel_y=hy,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.numel(), self.x.numel())

    @property
    def surface(self) -> float:
        return float(self.weight_x.sum() * self.weight_y.sum())

    @property
    def weights(self) -> torch.Tensor:
        """Flat panel areas, shape ``(N,)``, ordered ``y`` fastest."""
        return (self.weight_y[:, None] * self.weight_x[None, :]).reshape(-1)

    @property
    def points(self) -> torch.Tensor:
        """Flat panel centres, shape ``(N, 2)``, ordered ``y`` fastest."""
        yy, xx = torch.meshgrid(self.y, self.x, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    def to_grid(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape a flat ``(N,)`` field to ``(ny, nx)``."""
        return flat.reshape(self.shape)


@lru_cache(maxsize=64)
def _self_panel_integral(panel_x: float, panel_y: float) -> float:
    r"""``int_panel dS'/R`` for the point's own rectangular panel -- closed form.

    The integrand is singular at the panel centre, so Gauss quadrature converges
    only algebraically there (8x8 nodes are 6 % low, 32x32 still 2 % low).  The
    integral has an exact elementary form, obtained by integrating ``asinh`` by
    parts:

    .. math:: \int_{rect} \frac{dS'}{R}
              = 2\left[h_x \operatorname{asinh}\frac{h_y}{h_x}
                     + h_y \operatorname{asinh}\frac{h_x}{h_y}\right],

    which reduces to ``4 h asinh(1) = 3.5255 h`` for a square of side ``h``.  Using
    the closed form removes the ``O(h)``-converging quadrature error from the
    *diagonal* of the kernel, which is what makes the discretised radiation
    resistance indefinite on coarse grids.
    """
    if panel_x <= 0.0 or panel_y <= 0.0:
        raise ValueError("panel sides must be positive")
    return 2.0 * (
        panel_x * math.asinh(panel_y / panel_x)
        + panel_y * math.asinh(panel_x / panel_y)
    )


@lru_cache(maxsize=64)
def _self_panel_mean_radius(panel_x: float, panel_y: float, order: int = 24) -> float:
    """``(1/A) int_panel R dS`` -- the first moment that corrects the own panel.

    Used for the ``O(k^2)`` term of ``(e^{ikR} - 1)/R``: the Taylor series
    ``ik - (k R)/2 + ...`` integrated over the panel adds ``-(k^2/2) int R dS``,
    which is a *real* (resistive) correction.  Dropping it leaves a self-panel
    error of order ``k^2 h^2``, which is precisely what makes the discretised
    radiation resistance slightly indefinite on coarse grids.
    """
    nodes, weights = _gauss_legendre(int(order))
    gx = 0.5 * panel_x * nodes
    gy = 0.5 * panel_y * nodes
    wx = 0.5 * panel_x * weights
    wy = 0.5 * panel_y * weights
    r = torch.sqrt(gx[None, :] ** 2 + gy[:, None] ** 2)
    area = panel_x * panel_y
    return float(((wy[:, None] * wx[None, :]) * r).sum() / area)


def _pairwise_distance(points: torch.Tensor) -> torch.Tensor:
    dx = points[:, None, 0] - points[None, :, 0]
    dy = points[:, None, 1] - points[None, :, 1]
    return torch.sqrt(dx * dx + dy * dy)


def half_space_kernel(
    grid: RadiationGrid,
    *,
    frequency: float,
    sound_speed: float = C_AIR,
    area_weighted: bool = False,
) -> torch.Tensor:
    """Singular-free discretisation of ``exp(i k R) / (2 pi R)``.

    Returns the symmetric ``(N, N)`` complex kernel.  With ``area_weighted`` the
    columns carry the panel areas, so the pressure follows from a single
    matrix-vector product ``p = i omega rho K v``; the unweighted form is what
    the radiation impedance needs in order to stay symmetric under the Galerkin
    product ``W^T G W``.
    """
    frequency = _positive("frequency", frequency)
    k = 2.0 * math.pi * frequency / _positive("sound_speed", sound_speed)
    points = grid.points
    r = _pairwise_distance(points)
    mask = torch.eye(r.shape[0], dtype=torch.bool)
    r_safe = torch.where(mask, torch.ones_like(r), r)
    kernel = torch.exp(1j * k * r_safe) / (2.0 * math.pi * r_safe)
    # Own-panel term: analytic 1/R integral plus the smooth remainder ~ i k * area
    area = grid.panel_x * grid.panel_y
    self_term = (
        _self_panel_integral(grid.panel_x, grid.panel_y)
        + 1j * k * area
        - 0.5 * k * k * _self_panel_mean_radius(grid.panel_x, grid.panel_y) * area
    ) / (2.0 * math.pi)
    kernel = torch.where(mask, torch.full_like(kernel, complex(self_term)), kernel)
    if area_weighted:
        kernel = kernel * grid.weights[None, :]
    return kernel


def surface_pressure(
    grid: RadiationGrid,
    velocity: torch.Tensor,
    *,
    frequency: float,
    rho: float = C_AIR,
    sound_speed: float = C_AIR,
    kernel: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """On-surface acoustic pressure of a baffled plate (complex, peak)."""
    velocity = torch.as_tensor(velocity).to(torch.complex128).reshape(-1)
    if velocity.numel() != grid.points.shape[0]:
        raise ValueError("velocity length must match the grid")
    if kernel is None:
        kernel = half_space_kernel(
            grid, frequency=frequency, sound_speed=sound_speed, area_weighted=True
        )
    omega = 2.0 * math.pi * _positive("frequency", frequency)
    # ``e^{-i omega t}``: a positive normal velocity radiates a positive pressure
    # phasor, hence the leading minus (the same sign that makes
    # ``0.5 Re int p v* dS`` positive for a real velocity field).
    return -1j * omega * _positive("rho", rho) * (kernel @ velocity)


def radiated_power_from_pressure(
    pressure: torch.Tensor, velocity: torch.Tensor, grid: RadiationGrid
) -> float:
    """``P = 1/2 Re int p v* dS`` (peak amplitudes, time averaged)."""
    return float(
        0.5 * (grid.weights * pressure * torch.conj(velocity)).real.sum()
    )


# ---------------------------------------------------------------------------
# Simply supported sine-mode basis
# ---------------------------------------------------------------------------

def sine_basis(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    half_waves_x: int = 6,
    half_waves_y: int = 6,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Sine-mode matrix ``Phi`` of shape ``(N, nm)`` plus the ``(m, n)`` labels."""
    if int(half_waves_x) < 1 or int(half_waves_y) < 1:
        raise ValueError("half_waves_x and half_waves_y must be >= 1")
    points = grid.points
    columns: list[torch.Tensor] = []
    labels: list[tuple[int, int]] = []
    for m in range(1, int(half_waves_x) + 1):
        for n in range(1, int(half_waves_y) + 1):
            columns.append(
                torch.sin(m * math.pi * points[:, 0] / config.length)
                * torch.sin(n * math.pi * points[:, 1] / config.width)
            )
            labels.append((m, n))
    return torch.stack(columns, dim=1), labels


def sine_frequencies(
    config: PlateModalConfig, *, half_waves_x: int = 6, half_waves_y: int = 6
) -> torch.Tensor:
    """Dry simply supported frequencies (Hz) for the same mode ordering."""
    values = []
    scale = math.sqrt(config.rigidity / config.mass_per_area)
    for m in range(1, int(half_waves_x) + 1):
        for n in range(1, int(half_waves_y) + 1):
            k2 = (m * math.pi / config.length) ** 2 + (n * math.pi / config.width) ** 2
            values.append(k2 * scale / (2.0 * math.pi))
    return torch.tensor(values, dtype=torch.float64)


def modal_mass(config: PlateModalConfig, grid: RadiationGrid) -> torch.Tensor:
    """Modal masses ``rho h int phi^2 dS`` of the sine basis."""
    basis, _ = sine_basis(grid, config)
    return config.mass_per_area * ((basis ** 2) * grid.weights[:, None]).sum(dim=0)


def point_force_modal(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    location: tuple[float, float],
    amplitude: float = 1.0,
    half_waves_x: int = 6,
    half_waves_y: int = 6,
) -> torch.Tensor:
    """Modal force vector of a harmonic point force ``F(t) = A cos(omega t)``."""
    x0, y0 = location
    if not (0.0 <= x0 <= config.length and 0.0 <= y0 <= config.width):
        raise ValueError("point force location must lie on the plate")
    basis, _ = sine_basis(
        grid, config, half_waves_x=half_waves_x, half_waves_y=half_waves_y
    )
    points = grid.points
    # Nearest panel: the sine modes are smooth, so the panel value is the mode value
    distance = (points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2
    index = int(torch.argmin(distance))
    return float(amplitude) * basis[index, :]


def radiation_impedance_matrix(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    frequency: float,
    rho: float = C_AIR,
    sound_speed: float = C_AIR,
    half_waves_x: int = 6,
    half_waves_y: int = 6,
    kernel: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Modal radiation impedance ``Z = omega^2 rho Phi^T W G W Phi`` (complex).

    Modal fluid force convention: ``f_fluid = Z q`` for the modal displacement
    phasor ``q`` (``e^{-i omega t}``, velocity phasor ``-i omega q``).  ``Re Z``
    is the radiation damping operator and ``Im Z / omega^2`` the added mass.
    """
    basis, _ = sine_basis(
        grid, config, half_waves_x=half_waves_x, half_waves_y=half_waves_y
    )
    if kernel is None:
        kernel = half_space_kernel(grid, frequency=frequency, sound_speed=sound_speed)
    omega = 2.0 * math.pi * _positive("frequency", frequency)
    weighted = (basis * grid.weights[:, None]).to(torch.complex128)
    return omega ** 2 * _positive("rho", rho) * (weighted.T @ kernel @ weighted)


# ---------------------------------------------------------------------------
# Two-way coupled forced response
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoupledResponse:
    """Converged two-way coupled response at one excitation frequency."""

    frequency: float
    modal_amplitudes: torch.Tensor
    modal_velocity: torch.Tensor
    surface_velocity: torch.Tensor
    interface_pressure: torch.Tensor
    radiated_power: float
    input_power: float
    interface_residual: float
    equation_residual: float
    iterations: int
    residual_history: tuple[float, ...]
    direct_solution_gap: float
    labels: tuple[tuple[int, int], ...] = ()
    surface_area: float = 0.0
    mean_square_velocity: float = 0.0
    rho: float = RHO_AIR
    sound_speed: float = C_AIR

    @property
    def power_balance_error(self) -> float:
        """``|P_in - P_rad| / |P_in|`` -- zero for an undamped coupled solution."""
        if abs(self.input_power) < 1e-300:
            raise ZeroDivisionError("input power is zero")
        return abs(self.input_power - self.radiated_power) / abs(self.input_power)

    @property
    def radiation_efficiency(self) -> float:
        """``sigma = P_rad / (rho c S <v^2>)`` from the coupled solution."""
        denominator = (
            self.rho * self.sound_speed * self.surface_area * self.mean_square_velocity
        )
        if denominator <= 0.0:
            raise ZeroDivisionError("mean-square velocity is zero")
        return self.radiated_power / denominator


def coupled_modal_response(
    grid: RadiationGrid,
    config: PlateModalConfig,
    *,
    frequency: float,
    rho: float = C_AIR,
    sound_speed: float = C_AIR,
    half_waves_x: int = 6,
    half_waves_y: int = 6,
    force: Optional[torch.Tensor] = None,
    point_force: Optional[tuple[tuple[float, float], float]] = None,
    max_iterations: int = 400,
    tolerance: float = 1e-10,
    initial_relaxation: float = 0.5,
) -> CoupledResponse:
    """Partitioned two-way coupling of a baffled plate with its own sound field.

    The structural unknown is the vector of sine-mode amplitudes ``q``; the fluid
    unknown is the on-surface pressure.  One fixed-point sweep is
    "structure -> velocity -> pressure -> modal fluid force -> structure", and
    the sweep is accelerated with Aitken dynamic relaxation (the raw fixed point
    diverges in water, where the added mass is of the order of the plate mass).
    """
    frequency = _positive("frequency", frequency)
    omega = 2.0 * math.pi * frequency
    basis, labels = sine_basis(
        grid, config, half_waves_x=half_waves_x, half_waves_y=half_waves_y
    )
    if force is None:
        if point_force is None:
            raise ValueError("supply either force or point_force")
        location, amplitude = point_force
        force = point_force_modal(
            grid,
            config,
            location=location,
            amplitude=amplitude,
            half_waves_x=half_waves_x,
            half_waves_y=half_waves_y,
        )
    force = torch.as_tensor(force).to(torch.complex128).reshape(-1)
    if force.numel() != basis.shape[1]:
        raise ValueError("force length must equal the number of modes")

    dry = sine_frequencies(
        config, half_waves_x=half_waves_x, half_waves_y=half_waves_y
    )
    modal_mass = config.mass_per_area * ((basis ** 2) * grid.weights[:, None]).sum(0)
    stiffness = modal_mass * (2.0 * math.pi * dry) ** 2
    dynamic = stiffness - omega ** 2 * modal_mass
    if bool(torch.any(dynamic.abs() < 1e-12 * stiffness.abs())):
        raise ValueError("excitation frequency coincides with a dry resonance")

    kernel = half_space_kernel(grid, frequency=frequency, sound_speed=sound_speed)
    basis_c = basis.to(torch.complex128)
    weighted = (basis * grid.weights[:, None]).to(torch.complex128)
    impedance = omega ** 2 * _positive("rho", rho) * (weighted.T @ kernel @ weighted)

    def velocity_of(q: torch.Tensor) -> torch.Tensor:
        return -1j * omega * (basis_c @ q)

    def pressure_of(q: torch.Tensor) -> torch.Tensor:
        return -1j * omega * rho * (kernel @ (grid.weights * velocity_of(q)))

    def fluid_modal_force(q: torch.Tensor) -> torch.Tensor:
        return -(weighted.T @ pressure_of(q))

    # Added-mass splitting: the diagonal of the radiation impedance is the
    # dominant (mass-like) part of the coupling, especially in water.  Solving
    # it inside the sweep turns the raw fixed point -- whose spectral radius in
    # water is ~ mu/(1+mu) ~ 0.8 and which therefore crawls or diverges -- into
    # a well conditioned iteration on the *mutual* impedance only.
    diagonal_impedance = torch.diagonal(impedance)
    preconditioner = dynamic.to(torch.complex128) - diagonal_impedance

    q = torch.zeros_like(force)
    previous: Optional[torch.Tensor] = None
    previous_residual: Optional[torch.Tensor] = None
    relaxation = float(initial_relaxation)
    if not (0.0 < relaxation <= 2.0):
        raise ValueError("initial_relaxation must be in (0, 2]")
    history: list[float] = []
    residual = math.inf
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        raw = (force + (impedance @ q - diagonal_impedance * q)) / preconditioner
        residual_vector = raw - q
        norm = float(torch.linalg.vector_norm(residual_vector))
        residual = norm / max(float(torch.linalg.vector_norm(raw)), 1e-300)
        history.append(residual)
        if residual <= tolerance:
            q = q + relaxation * residual_vector
            break
        if previous is not None and previous_residual is not None:
            delta_q = q - previous
            delta_r = residual_vector - previous_residual
            denominator = float((torch.conj(delta_r) * delta_r).real.sum())
            if denominator > 0.0:
                candidate = -relaxation * float(
                    (torch.conj(delta_q) * delta_r).real.sum()
                ) / denominator
                if math.isfinite(candidate):
                    relaxation = min(max(candidate, 0.05), 1.0)
        previous = q.clone()
        previous_residual = residual_vector.clone()
        q = q + relaxation * residual_vector

    direct = torch.linalg.solve(
        torch.diag(dynamic.to(torch.complex128)) - impedance, force
    )
    gap = float(torch.linalg.vector_norm(q - direct)) / max(
        float(torch.linalg.vector_norm(direct)), 1e-300
    )

    modal_velocity = -1j * omega * q
    velocity = velocity_of(q)
    pressure = pressure_of(q)
    input_power = float(0.5 * (force * torch.conj(modal_velocity)).real.sum())
    radiated = radiated_power_from_pressure(pressure, velocity, grid)
    equation = torch.diag(dynamic.to(torch.complex128)) @ q - impedance @ q - force
    equation_residual = float(torch.linalg.vector_norm(equation)) / max(
        float(torch.linalg.vector_norm(force)), 1e-300
    )
    return CoupledResponse(
        frequency=frequency,
        modal_amplitudes=q,
        modal_velocity=modal_velocity,
        surface_velocity=velocity,
        interface_pressure=pressure,
        radiated_power=radiated,
        input_power=input_power,
        interface_residual=residual,
        equation_residual=equation_residual,
        iterations=iterations,
        residual_history=tuple(history),
        direct_solution_gap=gap,
        labels=tuple(labels),
        surface_area=grid.surface,
        mean_square_velocity=float(
            (grid.weights * (velocity.abs() ** 2)).sum() / grid.surface
        ),
        rho=rho,
        sound_speed=sound_speed,
    )