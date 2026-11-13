"""1D elastic rod wave propagation against the d'Alembert exact solution.

Case-pool anchor ``CASE-DG-DALEMBERT-ELASTIC-1D`` (ref: Graff 1975, *Wave Motion
in Elastic Solids*, 1D rod limit; acceptance: stress profile ``sigma(x, t)``
within 2 % of the d'Alembert two-travelling-wave solution, metric "no physical
dispersion").

Physical model
--------------
Axial motion of a linear elastic rod obeys the wave equation

    u_tt = c^2 u_xx,      c = sqrt(E / rho),

and the exact solution for a rod at rest carrying an initial velocity field
v0(x) is d'Alembert's two-travelling-wave form

    u(x, t)  = 1/(2c) * integral_{x-ct}^{x+ct} v0(s) ds
    sigma    = E u_x = (rho c / 2) * [ v0(x+ct) - v0(x-ct) ].

A 1D rod has **no physical dispersion**: the stress profile is the sum of two
pulses that translate rigidly at speed ``c`` without change of shape.  Whatever a
finite element model gets wrong about that profile is therefore *numerical*
dispersion of the discrete operator, which is what this module measures.

Boundary conditions enter through the method of images, so the reference stays
closed-form and never consumes an FE solve:

* ``free``  (traction-free ends, u_x = 0): even periodic extension about the
  ends, period 2L;
* ``fixed`` (clamped ends, u = 0): odd periodic extension, i.e. a reflection
  that flips tension into compression.

Finite element side
-------------------
2-node linear bar, consistent (or lumped) mass, explicit central difference with
the time step derived from the exact discrete spectrum, fail-closed on
stability.  Element-midpoint stress is compared with the exact solution at the
same abscissae.

This module is deliberately independent of :mod:`tensorfem.explicit_dynamics`
(that path requires a *lumped* mass matrix); here the consistent-mass pencil is
solved by a banded direct solve each step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf

__all__ = [
    "STEEL",
    "GaussianPulse",
    "rod_wave_speed",
    "bar_element_matrices",
    "assemble_bar",
    "stable_time_step",
    "discrete_omega_max",
    "numerical_phase_speed_ratio",
    "measure_mode_frequency",
    "dalembert_components",
    "exact_rod_state",
    "exact_rod_stress",
    "solve_bar_wave",
    "compare_stress_profile",
    "convergence_study",
    "run_case",
]

# Reference steel of the case (idealised rod, no lateral inertia).
STEEL: Mapping[str, float] = {
    "young": 210.0e9,   # Pa
    "density": 7850.0,  # kg/m^3
    "area": 1.0,        # m^2 (sets force amplitude only, not c)
}


# ---------------------------------------------------------------------------
# material, pulse, exact reference
# ---------------------------------------------------------------------------
def rod_wave_speed(young: float, density: float) -> float:
    """Bar wave speed ``c = sqrt(E/rho)`` (Graff 1975, rod limit)."""
    if young <= 0 or density <= 0:
        raise ValueError("young and density must be positive")
    return math.sqrt(young / density)


@dataclass(frozen=True)
class GaussianPulse:
    """Initial axial velocity pulse ``v0(x) = A exp(-((x-x0)/w)^2)`` [m/s].

    ``w`` is the e-folding half width: the support is effectively
    ``|x - x0| < 3 w`` and the dominant wavelength is about ``2 pi w``.
    """

    amplitude: float = 1.0
    center: float = 3.0
    width: float = 0.25

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("pulse width must be positive")

    def velocity(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.amplitude * np.exp(-(((x - self.center) / self.width) ** 2))

    def integral(self, a, b) -> np.ndarray:
        """``int_a^b v0(s) ds`` in closed form (error function)."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        w = self.width
        scale = self.amplitude * w * math.sqrt(math.pi) / 2.0
        return scale * (erf((b - self.center) / w) - erf((a - self.center) / w))


def _image_range(length: float, coords: np.ndarray, pulse) -> range:
    """Image indices ``k`` whose copies of the pulse can still contribute."""
    width = float(getattr(pulse, "width", 0.0) or 0.0)
    margin = max(2.0 * length, 8.0 * width)
    reach = float(np.max(np.abs(coords))) + margin
    half = int(math.ceil(reach / (2.0 * length))) + 1
    return range(-half, half + 1)


def _extended_values(x, *, length: float, pulse: GaussianPulse, boundary: str) -> np.ndarray:
    """Periodic image extension of ``v0`` (even for free ends, odd for fixed)."""
    if boundary not in {"free", "fixed"}:
        raise ValueError("boundary must be 'free' or 'fixed'")
    if length <= 0:
        raise ValueError("length must be positive")
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for k in _image_range(length, x, pulse):
        shift = 2.0 * k * length
        out = out + pulse.velocity(x + shift)
        mirrored = pulse.velocity(shift - x)
        out = out - mirrored if boundary == "fixed" else out + mirrored
    return out


def _extended_integral(a, b, *, length: float, pulse: GaussianPulse, boundary: str) -> np.ndarray:
    """``int_a^b v_tilde(s) ds`` for the image extension ``v_tilde``."""
    if boundary not in {"free", "fixed"}:
        raise ValueError("boundary must be 'free' or 'fixed'")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if length <= 0:
        raise ValueError("length must be positive")
    out = np.zeros(np.broadcast(a, b).shape, dtype=float)
    for k in _image_range(length, np.concatenate([a.ravel(), b.ravel()]), pulse):
        shift = 2.0 * k * length
        forward = pulse.integral(a + shift, b + shift)
        backward = pulse.integral(shift - b, shift - a)
        out = out - backward if boundary == "fixed" else out + backward
        out = out + forward
    return out


def dalembert_components(x, t: float, *, pulse, young: float, density: float):
    """Split the infinite-rod stress into its two travelling waves.

    ``sigma(x, t) = sigma_right(x, t) + sigma_left(x, t)`` with

    * ``sigma_right(x, t) = -(rho c / 2) v0(x - c t)``,
    * ``sigma_left(x, t)  = +(rho c / 2) v0(x + c t)``.

    A positive initial velocity launched to the right is compressive
    (``u_x = -v0/c`` for ``u = F(x-ct)``), which fixes the signs.

    Each component translates *rigidly* at ``c`` -- this is the quantitative
    statement of "no physical dispersion" in a 1D rod.
    """
    c = rod_wave_speed(young, density)
    x = np.asarray(x, dtype=float)
    amplitude = density * c / 2.0
    right = -amplitude * np.asarray(pulse.velocity(x - c * t), dtype=float)
    left = amplitude * np.asarray(pulse.velocity(x + c * t), dtype=float)
    return {"right": right, "left": left}


def exact_rod_state(
    x,
    t: float,
    *,
    pulse: GaussianPulse,
    young: float,
    density: float,
    length: float | None = None,
    boundary: str = "free",
) -> dict[str, np.ndarray]:
    """Closed-form d'Alembert displacement / strain / stress at ``(x, t)``.

    ``length=None`` selects the infinite-rod solution (valid while no reflection
    has reached the probe); a finite ``length`` switches on the image solution
    for ``free`` (even) or ``fixed`` (odd) ends.
    """
    c = rod_wave_speed(young, density)
    x = np.asarray(x, dtype=float)
    if length is None:
        values = pulse.velocity
        primitive = pulse.integral
    else:
        values = lambda s: _extended_values(s, length=length, pulse=pulse, boundary=boundary)  # noqa: E731
        primitive = lambda a, b: _extended_integral(a, b, length=length, pulse=pulse, boundary=boundary)  # noqa: E731

    displacement = primitive(x - c * t, x + c * t) / (2.0 * c)
    strain = (values(x + c * t) - values(x - c * t)) / (2.0 * c)
    return {"displacement": displacement, "strain": strain, "stress": young * strain}


def exact_rod_stress(x, t: float, **kwargs) -> np.ndarray:
    """Convenience wrapper returning only the stress field."""
    return exact_rod_state(x, t, **kwargs)["stress"]


# ---------------------------------------------------------------------------
# finite element model
# ---------------------------------------------------------------------------
def bar_element_matrices(length: float, young: float, density: float, area: float):
    """2-node linear bar: ``K_e = EA/L [[1,-1],[-1,1]]``, ``M_e = rho A L/6 [[2,1],[1,2]]``."""
    if length <= 0:
        raise ValueError("element length must be positive")
    stiffness = young * area / length * np.array([[1.0, -1.0], [-1.0, 1.0]])
    mass = density * area * length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    return stiffness, mass


def assemble_bar(
    n_elem: int,
    length: float,
    young: float,
    density: float,
    area: float = 1.0,
    *,
    mass_form: str = "consistent",
) -> dict[str, object]:
    """Assemble the tridiagonal bar system (stiffness and mass diagonals)."""
    if n_elem < 1:
        raise ValueError("n_elem must be >= 1")
    if mass_form not in {"consistent", "lumped"}:
        raise ValueError("mass_form must be 'consistent' or 'lumped'")
    h = length / n_elem
    n = n_elem + 1
    stiff, mass = bar_element_matrices(h, young, density, area)

    dk = np.full(n, 2.0 * stiff[0, 0])
    dk[0] = dk[-1] = stiff[0, 0]
    ok = np.full(n - 1, stiff[0, 1])
    if mass_form == "consistent":
        dm = np.full(n, 2.0 * mass[0, 0])
        dm[0] = dm[-1] = mass[0, 0]
        om = np.full(n - 1, mass[0, 1])
    else:
        dm = np.full(n, density * area * h / 2.0)
        om = np.zeros(n - 1)
    nodes = np.linspace(0.0, length, n)
    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    return {
        "dm": dm, "om": om, "dk": dk, "ok": ok,
        "h": h, "nodes": nodes, "midpoints": midpoints,
        "n_elem": n_elem, "mass_form": mass_form,
    }


def numerical_phase_speed_ratio(theta: float, *, mass_form: str = "consistent") -> float:
    """``c_num / c`` of the semi-discrete bar for the dimensionless wave number ``theta = k h``.

    Consistent mass: ``omega^2 = (6 c^2/h^2)(1-cos theta)/(2+cos theta)``, i.e.
    ``c_num/c = 1 + theta^2/24 + O(theta^4)`` (slightly *faster* than ``c``, so
    the numerical pulse arrives early).  Lumped mass:
    ``omega^2 = (2 c^2/h^2)(1-cos theta)``, i.e. ``c_num/c = 1 - theta^2/24``
    (slightly *slower*).  Either way the error is O(theta^2) numerical dispersion
    of the discrete operator -- a discretisation error, not physics.
    """
    if theta < 0:
        raise ValueError("theta must be non-negative")
    if theta == 0.0:
        return 1.0
    if mass_form == "consistent":
        omega2 = 6.0 * (1.0 - math.cos(theta)) / (2.0 + math.cos(theta))
    elif mass_form == "lumped":
        omega2 = 2.0 * (1.0 - math.cos(theta))
    else:
        raise ValueError("mass_form must be 'consistent' or 'lumped'")
    return math.sqrt(max(omega2, 0.0)) / theta


def stable_time_step(
    h: float,
    young: float,
    density: float,
    *,
    mass_form: str = "consistent",
    safety: float = 0.5,
) -> tuple[float, float]:
    """Return ``(dt, omega_max)`` for the central-difference scheme.

    ``omega_max = sqrt(12) c / h`` (consistent mass) or ``2 c / h`` (lumped mass),
    i.e. the extreme of the discrete spectrum at ``theta = pi``.
    """
    if h <= 0:
        raise ValueError("h must be positive")
    if not 0.0 < safety < 1.0:
        raise ValueError("safety must lie in (0, 1)")
    c = rod_wave_speed(young, density)
    factor = math.sqrt(12.0) if mass_form == "consistent" else 2.0
    omega_max = factor * c / h
    return safety * 2.0 / omega_max, omega_max


def discrete_omega_max(
    dm: np.ndarray,
    om: np.ndarray,
    dk: np.ndarray,
    ok: np.ndarray,
    *,
    iterations: int = 400,
    tol: float = 1e-10,
) -> float:
    """Largest discrete frequency ``omega_max`` of the assembled (K, M) pencil.

    Power iteration on ``M^{-1} K`` (banded direct solve, no LAPACK
    eigensolver, deterministic start on the sawtooth vector that carries the
    highest-frequency mode).  The free-free end rows shift ``omega_max`` by a
    few percent relative to the infinite-lattice estimate ``sqrt(12) c / h`` for
    very coarse meshes, so the central-difference step is bounded from the
    *assembled* system.
    """
    if dm.size != dk.size or om.size != dm.size - 1:
        raise ValueError("inconsistent tridiagonal assembly")
    ab = _banded(dm, om)
    index = np.arange(dm.size)
    x = np.where(index % 2 == 0, 1.0, -1.0) + 0.05 * np.cos(index * 1.7)
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iterations):
        y = _tri_matvec(dk, ok, x)
        z = solve_banded((1, 1), ab, y, check_finite=False)
        mass_z = _tri_matvec(dm, om, z)
        denom = float(z @ mass_z)
        if denom <= 0:
            break
        new_lam = float(z @ _tri_matvec(dk, ok, z)) / denom
        norm = float(np.linalg.norm(z))
        if norm == 0:
            break
        x = z / norm
        if abs(new_lam - lam) <= tol * max(abs(new_lam), 1.0):
            lam = new_lam
            break
        lam = new_lam
    return math.sqrt(max(lam, 0.0))


def _banded(diagonal: np.ndarray, off: np.ndarray) -> np.ndarray:
    """LAPACK band storage for a symmetric tridiagonal matrix (kl = ku = 1)."""
    ab = np.zeros((3, diagonal.size))
    ab[0, 1:] = off
    ab[1, :] = diagonal
    ab[2, :-1] = off
    return ab


def _tri_matvec(diagonal: np.ndarray, off: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = diagonal * vector
    out[:-1] += off * vector[1:]
    out[1:] += off * vector[:-1]
    return out


def solve_bar_wave(
    *,
    n_elem: int,
    length: float,
    young: float,
    density: float,
    pulse,
    end_time: float,
    area: float = 1.0,
    boundary: str = "free",
    mass_form: str = "consistent",
    cfl_safety: float = 0.5,
    snapshot_times: Sequence[float] | None = None,
    probe_x: float | None = None,
    energy_samples: int = 400,
) -> dict[str, object]:
    """Explicit central-difference bar transient (consistent mass by default).

    ``M (u_{n+1} - 2 u_n + u_{n-1}) = -dt^2 K u_n`` is stable for
    ``dt <= 2/omega_max`` of the *assembled* pencil, and is second-order accurate
    in time.  Displacement, element-midpoint stress, probe stress history and the
    mechanical energy history are returned.
    """
    if boundary not in {"free", "fixed"}:
        raise ValueError("boundary must be 'free' or 'fixed'")
    if end_time <= 0:
        raise ValueError("end_time must be positive")
    system = assemble_bar(n_elem, length, young, density, area, mass_form=mass_form)
    dm, om, dk, ok = (system[key] for key in ("dm", "om", "dk", "ok"))
    h = float(system["h"])
    nodes = np.asarray(system["nodes"])
    midpoints = np.asarray(system["midpoints"])

    omega_analytic = stable_time_step(h, young, density, mass_form=mass_form, safety=cfl_safety)[1]
    omega_max = max(discrete_omega_max(dm, om, dk, ok), 0.5 * omega_analytic)
    dt_max = cfl_safety * 2.0 / omega_max
    n_steps = max(1, int(math.ceil(end_time / dt_max)))
    dt = end_time / n_steps
    if dt > 2.0 / omega_max * (1.0 + 1e-12):
        raise ValueError("adjusted time step violates the central-difference limit")

    # Dirichlet ends are *eliminated* from the solve (clamping the free-free
    # solution afterwards would violate interior equilibrium by the reaction
    # terms and produce a spurious boundary layer).
    active = slice(1, -1) if boundary == "fixed" else slice(None)
    dm_a, om_a, dk_a, ok_a = dm[active], om[active], dk[active], ok[active]
    ab = _banded(dm_a, om_a)
    nodal_velocity = np.asarray(pulse.velocity(nodes), dtype=float)
    u_prev = np.zeros_like(dm_a)             # u_0 = 0 (rod at rest)
    u_curr = dt * nodal_velocity[active]     # u_1 = u_0 + dt v_0 (a_0 = 0)
    dt2 = dt * dt

    def expand(state: np.ndarray) -> np.ndarray:
        if boundary != "fixed":
            return state
        full = np.zeros(nodes.size, dtype=state.dtype)
        full[1:-1] = state
        return full

    times = np.arange(n_steps + 1) * dt
    snapshots: dict[float, dict[str, np.ndarray]] = {}
    wanted = sorted(float(t) for t in (snapshot_times or [end_time]))
    if wanted and wanted[-1] > end_time * (1.0 + 1e-9):
        raise ValueError("snapshot time exceeds end_time")
    pending = list(wanted)

    probe_index = None
    if probe_x is not None:
        if not 0.0 <= float(probe_x) <= length:
            raise ValueError("probe_x must lie inside the rod")
        probe_index = int(np.argmin(np.abs(midpoints - float(probe_x))))
    probe_t = np.zeros(n_steps + 1)
    probe_sigma = np.zeros(n_steps + 1)

    def element_stress(full_state: np.ndarray) -> np.ndarray:
        return young * (full_state[1:] - full_state[:-1]) / h

    def record(step: int, state: np.ndarray) -> None:
        full = expand(state)
        stress = element_stress(full)
        if probe_index is not None:
            probe_t[step] = times[step]
            probe_sigma[step] = stress[probe_index]
        while pending and times[step] >= pending[0] - 1e-15:
            target = pending.pop(0)
            snapshots[target] = {
                "time": np.array(times[step]),
                "x": midpoints.copy(),
                "stress": stress.copy(),
                "displacement": full.copy(),
            }

    energy_stride = max(1, n_steps // max(1, energy_samples))
    energy_t, energy_ke, energy_se = [], [], []

    record(0, u_prev)
    if n_steps >= 1:
        record(1, u_curr)
    for step in range(1, n_steps):
        rhs = (
            2.0 * _tri_matvec(dm_a, om_a, u_curr)
            - _tri_matvec(dm_a, om_a, u_prev)
            - dt2 * _tri_matvec(dk_a, ok_a, u_curr)
        )
        u_next = solve_banded((1, 1), ab, rhs, check_finite=False)
        if step % energy_stride == 0 or step == n_steps - 1:
            velocity = (u_next - u_prev) / (2.0 * dt)
            energy_t.append(times[step])
            energy_ke.append(0.5 * float(velocity @ _tri_matvec(dm_a, om_a, velocity)))
            energy_se.append(0.5 * float(u_curr @ _tri_matvec(dk_a, ok_a, u_curr)))
        u_prev, u_curr = u_curr, u_next
        record(step + 1, u_curr)

    energy_initial = 0.5 * float(
        nodal_velocity[active] @ _tri_matvec(dm_a, om_a, nodal_velocity[active])
    )
    return {
        "nodes": nodes,
        "midpoints": midpoints,
        "h": h,
        "dt": dt,
        "n_steps": n_steps,
        "omega_max": omega_max,
        "omega_max_analytic": omega_analytic,
        "omega_max_ratio": omega_max / omega_analytic,
        "stability_margin": dt * omega_max / 2.0,
        "mass_form": mass_form,
        "boundary": boundary,
        "end_time": end_time,
        "snapshots": {t: snapshots[t] for t in sorted(snapshots)},
        "probe": {
            "x": None if probe_index is None else float(midpoints[probe_index]),
            "t": probe_t,
            "stress": probe_sigma,
        },
        "energy": {
            "t": np.asarray(energy_t),
            "kinetic": np.asarray(energy_ke),
            "strain": np.asarray(energy_se),
            "total": np.asarray(energy_ke) + np.asarray(energy_se),
        },
        "energy_initial": energy_initial,
    }


# ---------------------------------------------------------------------------
# comparison and case driver
# ---------------------------------------------------------------------------
def _refine_peak(abscissa: np.ndarray, values: np.ndarray, index: int) -> float:
    """Sub-sample location of an extremum by a parabola through three samples."""
    i = int(index)
    n = values.size
    if i <= 0 or i >= n - 1:
        return float(abscissa[i])
    y0, y1, y2 = (abs(float(values[i - 1])), abs(float(values[i])), abs(float(values[i + 1])))
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(abscissa[i])
    delta = 0.5 * (y0 - y2) / denom
    return float(abscissa[i] + delta * (abscissa[i + 1] - abscissa[i]))


def measure_mode_frequency(
    *,
    n_elem: int,
    mode_index: int,
    length: float,
    young: float,
    density: float,
    area: float = 1.0,
    mass_form: str = "consistent",
    cfl_safety: float = 0.125,
    periods: float = 6.0,
) -> dict[str, float]:
    """Time-domain frequency of a single discrete mode vs the closed-form value.

    A free-free rod started with ``v0 = cos(mode_index * pi x / L)`` oscillates at
    one discrete frequency.  The measured frequency (zero-crossing count of the
    probe stress history) is an *independent* measurement of the same quantity
    that :func:`numerical_phase_speed_ratio` predicts, so agreement ties the
    observed profile error to numerical dispersion rather than to a reference bug.
    """
    if not 1 <= mode_index < n_elem:
        raise ValueError("mode_index must satisfy 1 <= mode_index < n_elem")
    k = mode_index * math.pi / length
    c = rod_wave_speed(young, density)

    class _Mode:
        def velocity(self, x):
            return np.cos(k * np.asarray(x, dtype=float))

        def integral(self, a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            return (np.sin(k * b) - np.sin(k * a)) / k

    h = length / n_elem
    theta = k * h
    omega_discrete = numerical_phase_speed_ratio(theta, mass_form=mass_form) * k * c
    period = 2.0 * math.pi / omega_discrete
    probe = length / n_elem * (mode_index * 0 + 0.5) if False else length * 0.5
    run = solve_bar_wave(
        n_elem=n_elem, length=length, young=young, density=density,
        pulse=_Mode(), end_time=periods * period, area=area, mass_form=mass_form,
        cfl_safety=cfl_safety, probe_x=probe,
    )
    t = np.asarray(run["probe"]["t"])
    sigma = np.asarray(run["probe"]["stress"])
    # zero crossings of the probe signal, with linear interpolation in time
    crossings = []
    for i in range(1, t.size):
        if sigma[i - 1] * sigma[i] < 0.0:
            frac = sigma[i - 1] / (sigma[i - 1] - sigma[i])
            crossings.append(float(t[i - 1] + frac * (t[i] - t[i - 1])))
    if len(crossings) < 5:
        raise RuntimeError("too few zero crossings to measure the frequency")
    span = crossings[-1] - crossings[0]
    half_periods = len(crossings) - 1
    omega_measured = math.pi * half_periods / span
    return {
        "mode_index": float(mode_index),
        "theta": float(theta),
        "omega_exact": float(k * c),
        "omega_discrete_formula": float(omega_discrete),
        "omega_measured": float(omega_measured),
        "formula_vs_measured_rel": float(abs(omega_measured - omega_discrete) / omega_discrete),
        "c_num_over_c_formula": float(numerical_phase_speed_ratio(theta, mass_form=mass_form)),
        "c_num_over_c_exact": 1.0,
        "h": float(h),
        "dt": float(run["dt"]),
    }


def compare_stress_profile(
    *,
    n_elem: int,
    length: float,
    young: float,
    density: float,
    pulse: GaussianPulse,
    end_time: float,
    snapshot_time: float | None = None,
    boundary: str = "free",
    area: float = 1.0,
    mass_form: str = "consistent",
    cfl_safety: float = 0.5,
) -> dict[str, object]:
    """Compare FEM midpoint stress with d'Alembert at one instant.

    Returns the two profiles plus the case metrics: relative L2 error, relative
    L-infinity error (scaled by the exact peak), peak amplitude ratio, peak
    abscissa error in pulse widths, and the discrete/numerical-wave-speed ratio
    implied by the mesh (``theta`` at the dominant wavelength).
    """
    t_star = float(snapshot_time if snapshot_time is not None else end_time)
    run = solve_bar_wave(
        n_elem=n_elem, length=length, young=young, density=density, pulse=pulse,
        end_time=t_star, area=area, boundary=boundary, mass_form=mass_form,
        cfl_safety=cfl_safety, snapshot_times=[t_star],
    )
    snap = run["snapshots"][t_star]
    x = np.asarray(snap["x"])
    sigma_fem = np.asarray(snap["stress"])
    exact = exact_rod_state(
        x, t_star, pulse=pulse, young=young, density=density,
        length=length, boundary=boundary,
    )["stress"]

    diff = sigma_fem - exact
    exact_norm = float(np.linalg.norm(exact))
    rel_l2 = float(np.linalg.norm(diff) / exact_norm) if exact_norm > 0 else float("inf")
    peak_exact = float(np.max(np.abs(exact)))
    peak_fem = float(np.max(np.abs(sigma_fem)))
    rel_linf = float(np.max(np.abs(diff)) / peak_exact) if peak_exact > 0 else float("inf")
    peak_ratio = peak_fem / peak_exact if peak_exact > 0 else float("inf")
    x_lead = pulse.center + rod_wave_speed(young, density) * t_star
    window = np.abs(x - x_lead) <= 3.0 * pulse.width
    if np.count_nonzero(window) < 5:
        window = np.ones_like(x, dtype=bool)
    index = np.flatnonzero(window)
    if index.size < 5:
        index = np.arange(x.size)
    i_exact = int(index[np.argmax(np.abs(exact[index]))])
    i_fem = int(index[np.argmax(np.abs(sigma_fem[index]))])
    x_peak_exact = _refine_peak(x, exact, i_exact)
    x_peak_fem = _refine_peak(x, sigma_fem, i_fem)
    lead_exact = float(abs(exact[i_exact]))
    lead_fem = float(abs(sigma_fem[i_fem]))

    # reflected lobe: free ends keep the sign, clamped ends invert it
    c_wave = rod_wave_speed(young, density)
    x_reflected = c_wave * t_star - pulse.center
    reflected = {"x": float(x_reflected), "present": bool(0.0 <= x_reflected <= length)}
    if reflected["present"]:
        ref_mask = np.abs(x - x_reflected) <= 3.0 * pulse.width
        if np.any(ref_mask):
            ref_idx = np.flatnonzero(ref_mask)
            j_exact = int(ref_idx[np.argmax(np.abs(exact[ref_idx]))])
            j_fem = int(ref_idx[np.argmax(np.abs(sigma_fem[ref_idx]))])
            reflected.update({
                "sign_exact": float(np.sign(exact[j_exact])),
                "sign_fem": float(np.sign(sigma_fem[j_fem])),
                "amplitude_ratio": float(abs(sigma_fem[j_fem]) / abs(exact[j_exact]))
                if exact[j_exact] != 0 else float("inf"),
            })
    theta = 2.0 * math.pi * float(run["h"]) / (2.0 * math.pi * pulse.width)
    correlation = float(np.dot(sigma_fem, exact) / (np.linalg.norm(sigma_fem) * exact_norm)) if exact_norm > 0 else 0.0

    energy = run["energy"]
    total = np.asarray(energy["total"])
    reference = float(total[0]) if total.size else 0.0
    energy_drift = float(np.max(np.abs(total - reference)) / reference) if reference > 0 else float("inf")

    return {
        "x": x,
        "stress_fem": sigma_fem,
        "stress_exact": exact,
        "rel_l2": rel_l2,
        "rel_linf": rel_linf,
        "peak_ratio": peak_ratio,
        "lead_amplitude_ratio": lead_fem / lead_exact if lead_exact > 0 else float("inf"),
        "peak_shift_widths": abs(x_peak_fem - x_peak_exact) / pulse.width,
        "x_peak_lead": x_lead,
        "x_peak_exact": x_peak_exact,
        "x_peak_fem": x_peak_fem,
        "reflected": reflected,
        "correlation": correlation,
        "h": float(run["h"]),
        "dt": float(run["dt"]),
        "n_steps": int(run["n_steps"]),
        "theta_dominant": theta,
        "c_num_ratio": numerical_phase_speed_ratio(theta, mass_form=mass_form),
        "stability_margin": float(run["stability_margin"]),
        "energy_drift": energy_drift,
        "run": run,
    }


def convergence_study(
    n_list: Sequence[int],
    *,
    length: float,
    young: float,
    density: float,
    pulse: GaussianPulse,
    end_time: float,
    boundary: str = "free",
    area: float = 1.0,
    mass_form: str = "consistent",
    cfl_safety: float = 0.5,
) -> list[dict[str, object]]:
    """Run :func:`compare_stress_profile` over a mesh sequence with observed orders."""
    rows = []
    for n in n_list:
        result = compare_stress_profile(
            n_elem=n, length=length, young=young, density=density, pulse=pulse,
            end_time=end_time, boundary=boundary, area=area, mass_form=mass_form,
            cfl_safety=cfl_safety,
        )
        result.pop("run")
        rows.append(result)
    for fine, coarse in zip(rows[1:], rows[:-1]):
        h_ratio = float(coarse["h"]) / float(fine["h"])
        e_ratio = float(coarse["rel_l2"]) / float(fine["rel_l2"])
        fine["observed_order"] = math.log(e_ratio) / math.log(h_ratio) if e_ratio > 0 else float("nan")
    if rows:
        rows[0]["observed_order"] = None
    return rows


def run_case(
    *,
    mesh_sequence: Sequence[int] = (200, 400, 800, 1600),
    length: float = 16.0,
    young: float = STEEL["young"],
    density: float = STEEL["density"],
    area: float = STEEL["area"],
    pulse: GaussianPulse = GaussianPulse(amplitude=1.0, center=5.0, width=0.5),
    end_time: float = 1.5e-3,
    tolerance: float = 0.02,
    probe_offset_ratio: float = 0.5,
    timestep_safety_sequence: Sequence[float] = (0.5, 0.25, 0.125),
) -> dict[str, object]:
    """Run the full case and return tables, verdict and physical cross-checks.

    Contents: (1) h-convergence of the stress profile for free ends, (2) the same
    for clamped ends (odd image reflection, tension -> compression), (3) exact
    translation invariance (no physical dispersion), (4) discrete dispersion law
    measured in the time domain, (5) pulse arrival time at a probe, (6) energy
    conservation, (7) explicit time-step sensitivity.
    """
    c = rod_wave_speed(young, density)
    rows = convergence_study(
        mesh_sequence, length=length, young=young, density=density, pulse=pulse,
        end_time=end_time,
    )
    finest = rows[-1]
    fixed_rows = convergence_study(
        tuple(n for n in mesh_sequence if n >= max(200, mesh_sequence[-1] // 2)),
        length=length, young=young, density=density, pulse=pulse,
        end_time=end_time, boundary="fixed",
    )

    # (3) exact solution has no physical dispersion: each travelling wave keeps
    #     its shape, i.e. is a rigid translation of itself between two instants.
    x_probe = np.linspace(0.5, length - 0.5, 4001)
    t1, t2 = 0.2e-3, 0.6e-3
    dt_probe = t2 - t1
    reference = dalembert_components(x_probe, t1, pulse=pulse, young=young, density=density)
    later = dalembert_components(x_probe, t2, pulse=pulse, young=young, density=density)
    right_shifted = dalembert_components(
        x_probe - c * dt_probe, t1, pulse=pulse, young=young, density=density
    )["right"]
    left_shifted = dalembert_components(
        x_probe + c * dt_probe, t1, pulse=pulse, young=young, density=density
    )["left"]
    scale = max(float(np.max(np.abs(reference["right"]))), 1e-30)
    translation_error = float(
        max(np.max(np.abs(later["right"] - right_shifted)), np.max(np.abs(later["left"] - left_shifted)))
        / scale
    )
    # the decomposition must reproduce the full d'Alembert stress field
    sigma_sum = reference["right"] + reference["left"]
    sigma_full = exact_rod_stress(x_probe, t1, pulse=pulse, young=young, density=density)
    decomposition_error = float(np.max(np.abs(sigma_sum - sigma_full)) / max(scale, 1e-30))

    # (5) discrete dispersion law measured in the time domain.
    dispersion = []
    for n_elem, mode_index in ((100, 4), (100, 16), (200, 32)):
        dispersion.append(
            measure_mode_frequency(
                n_elem=n_elem, mode_index=mode_index, length=length,
                young=young, density=density, area=area,
            )
        )

    # (6) probe arrival time on the finest mesh (sub-sample refined peak).
    probe_x = pulse.center + probe_offset_ratio * (length - pulse.center)
    run = solve_bar_wave(
        n_elem=int(mesh_sequence[-1]), length=length, young=young, density=density,
        pulse=pulse, end_time=end_time, area=area, probe_x=probe_x,
        snapshot_times=[end_time], energy_samples=200,
    )
    t_history = np.asarray(run["probe"]["t"])
    sigma_history = np.asarray(run["probe"]["stress"])
    t_peak_fem = _refine_peak(t_history, sigma_history, int(np.argmax(np.abs(sigma_history))))
    arrival = (float(run["probe"]["x"]) - pulse.center) / c
    arrival_error = abs(t_peak_fem - arrival) / arrival if arrival <= end_time else float("nan")

    energy = run["energy"]
    total = np.asarray(energy["total"])
    energy_drift = float(np.max(np.abs(total - total[0])) / total[0]) if total.size and total[0] > 0 else float("inf")

    # (7) explicit time-step sensitivity at a fixed mesh.
    timestep_rows = []
    for safety in timestep_safety_sequence:
        result = compare_stress_profile(
            n_elem=int(mesh_sequence[-2]), length=length, young=young, density=density,
            pulse=pulse, end_time=end_time, cfl_safety=float(safety),
        )
        timestep_rows.append({
            "cfl_safety": float(safety),
            "rel_l2": float(result["rel_l2"]),
            "dt": float(result["dt"]),
            "n_steps": int(result["n_steps"]),
        })

    def _clean(row: Mapping[str, object]) -> dict[str, object]:
        return {k: v for k, v in row.items() if k not in {"x", "stress_fem", "stress_exact", "run"}}

    return {
        "case": "CASE-DG-DALEMBERT-ELASTIC-1D",
        "parameters": {
            "young_Pa": young, "density_kg_m3": density, "area_m2": area,
            "wave_speed_m_s": c, "length_m": length,
            "pulse_amplitude_m_s": pulse.amplitude, "pulse_center_m": pulse.center,
            "pulse_width_m": pulse.width, "end_time_s": end_time,
            "boundary_free": "u_x = 0 (method of images: even extension)",
            "boundary_fixed": "u = 0 (method of images: odd extension)",
        },
        "rows_free": [_clean(row) for row in rows],
        "rows_fixed": [_clean(row) for row in fixed_rows],
        "finest": _clean(finest),
        "verdict": "pass" if float(finest["rel_l2"]) < tolerance else "fail",
        "tolerance": tolerance,
        "translation_error": translation_error,
        "decomposition_error": decomposition_error,
        "dispersion": dispersion,
        "probe": {
            "x_m": float(run["probe"]["x"]), "arrival_theory_s": arrival,
            "t_peak_fem_s": t_peak_fem, "arrival_rel_error": arrival_error,
        },
        "energy_drift": energy_drift,
        "timestep_rows": timestep_rows,
        "run": run,
    }
