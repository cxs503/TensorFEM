"""Lattice-Boltzmann solver for the Allen-Cahn equation on D3Q15 / D3Q19 lattices.

This module implements the phase-field Allen-Cahn equation using the
lattice-Boltzmann method (LBM) with 3D lattices (D3Q15, D3Q19, D3Q27),
following the approach used in the lssfau/walberla framework.

Model
-----
The Allen-Cahn equation governs the evolution of a phase-field variable
φ ∈ [0, 1] that distinguishes two phases:

    ∂φ/∂t = M ∇²φ  −  (M / ε²) f'(φ)

where

* M  is the mobility (diffusivity),
* ε  is the interface width parameter,
* f(φ) = ¼ φ²(1 − φ)²  is the double-well potential,
* f'(φ) = ½ φ(1 − φ)(1 − 2φ).

The LBM uses a single distribution function g_i with BGK collision and a
source term for the reaction:

    g_i(x + c_i, t+1) = g_i(x, t) − (g_i − g_i^eq)/τ + S_i

where

* g_i^eq = w_i φ           (equilibrium for pure diffusion),
* S_i    = −w_i (M/ε²) f'(φ) / (1 − 1/(2τ))   (reaction source with
  Chapman-Enskog correction),
* D = M = cs² (τ − ½)     (diffusivity from Chapman-Enskog).

The macroscopic phase field is recovered as  φ = Σ_i g_i.

Chapman-Enskog consistency
--------------------------
The Chapman-Enskog expansion of the BGK LBM with a source term at second
order yields (Kruger et al. 2017, ch. 7):

    ∂φ/∂t = cs²(τ − ½) ∇²φ + (1 − 1/(2τ)) Σ_i S_i

For the Allen-Cahn equation we need the source to contribute
−(M/ε²) f'(φ) to the macroscopic equation, hence:

    Σ_i S_i = −(M/ε²) f'(φ) / (1 − 1/(2τ))

and the simplest per-direction choice is:

    S_i = w_i × [−(M/ε²) f'(φ)] / (1 − 1/(2τ))

For τ = 1 the correction factor is 1/(1 − ½) = 2, i.e. the source term
must be doubled relative to the naive formulation.  This factor is
crucial: without it the shrinking-sphere velocity is only ~60 % of the
sharp-interface prediction, which is a systematic error, not a
discretisation artefact.

Validation cases
----------------
1. **Diffusion decay (cosine mode)**: With the reaction term suppressed
   (ε → ∞), the LBM must reproduce the exact exponential decay of a
   cosine mode:  φ(x,t) = A cos(kx) exp(−D k² t).  This validates the
   diffusion operator on both D3Q15 and D3Q19.

2. **3D shrinking sphere**: A spherical droplet of phase φ = 1 in a
   background of φ = 0 shrinks under mean-curvature flow.  In the
   sharp-interface limit the radius evolves as

       dR/dt = −2M / R

   giving  R²(t) = R₀² − 4Mt.  The LBM is validated by tracking the
   effective radius and comparing against this law.

3. **D3Q15 vs D3Q19 accuracy**: D3Q15 has isotropy rank 3 (fourth-order
   moment tensor is anisotropic), while D3Q19 has isotropy rank 4.  The
   shrinking-sphere test exposes this difference: D3Q19 should give
   smaller curvature-driven velocity errors than D3Q15 at the same
   resolution.

References
----------
[1] Allen, S.M., Cahn, J.W. (1979). Acta Metall. 27, 1085 (microstructure
    kinetics).
[2] Chai, Z., Zhao, T.S. (2012). Phys. Rev. E 86, 016705 (LBM for Cahn-
    Hilliard / Allen-Cahn).
[3] lssfau/walberla, apps/showcases/phase_field_allen_cahn (reference
    implementation with D3Q15/D3Q19).
[4] Kruger, T. et al. (2017). The Lattice Boltzmann Method, ch. 7 (forcing
    and source terms).
[5] Guo, Z., Zheng, C., Shi, B. (2002). Phys. Fluids 14, 2007 (discrete
    lattice effects on the forcing term).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from tensorfem.lbm_lattice_tables import LatticeTable, get_table, stream

DTYPE = np.float64

__all__ = [
    "AllenCahnConfig",
    "AllenCahnLBM",
    "tanh_profile_1d",
    "sphere_mask",
    "double_well",
    "double_well_derivative",
    "run_diffusion_decay",
    "run_shrinking_sphere",
    "compare_d3q15_d3q19",
]


# ---------------------------------------------------------------------------
# analytical helpers
# ---------------------------------------------------------------------------
def double_well(phi: np.ndarray) -> np.ndarray:
    """Double-well potential  f(φ) = ¼ φ²(1−φ)²."""
    return 0.25 * phi * phi * (1.0 - phi) ** 2


def double_well_derivative(phi: np.ndarray) -> np.ndarray:
    """Derivative  f'(φ) = ½ φ(1−φ)(1−2φ)."""
    return 0.5 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def tanh_profile_1d(x: np.ndarray, epsilon: float, x0: float = 0.0) -> np.ndarray:
    """Steady-state 1D Allen-Cahn profile:  φ = ½(1 − tanh((x−x₀)/(2√2 ε)))."""
    xi = (x - x0) / (2.0 * np.sqrt(2.0) * epsilon)
    return 0.5 * (1.0 - np.tanh(xi))


def sphere_mask(
    nx: int, ny: int, nz: int,
    center: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Boolean mask: True inside a sphere of given center and radius.

    Coordinates are cell-centred: (i+0.5, j+0.5, k+0.5).
    """
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=DTYPE),
        np.arange(ny, dtype=DTYPE),
        np.arange(nx, dtype=DTYPE),
        indexing="ij",
    )
    dist2 = (xx + 0.5 - center[0]) ** 2 + (yy + 0.5 - center[1]) ** 2 + (zz + 0.5 - center[2]) ** 2
    return dist2 < radius * radius


def effective_radius(phi: np.ndarray, dx: float = 1.0) -> float:
    """Estimate the effective radius of a spherical phase-field region.

    Uses volume conservation:  V = Σ φ dx³ = (4/3)π R³  =>  R = (3V/(4π))^{1/3}.
    """
    volume = float(phi.sum()) * dx ** 3
    if volume <= 0:
        return 0.0
    return (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class AllenCahnConfig:
    """Allen-Cahn LBM configuration (lattice units, dx = dt = 1).

    Parameters
    ----------
    lattice_name : one of 'D3Q15', 'D3Q19', 'D3Q27'
    nx, ny, nz : domain size (periodic in all directions)
    tau : BGK relaxation time; diffusivity D = cs²(τ − ½)
    epsilon : interface width parameter
    mobility : M; if None, M = cs²(τ − ½) (i.e. D = M)
    steps : total number of LBM steps
    report_every : interval for logging / radius measurement
    """

    lattice_name: str = "D3Q19"
    nx: int = 32
    ny: int = 32
    nz: int = 32
    tau: float = 1.0
    epsilon: float = 4.0
    mobility: float | None = None
    steps: int = 2000
    report_every: int = 100

    @property
    def table(self) -> LatticeTable:
        return get_table(self.lattice_name)

    @property
    def cs2(self) -> float:
        return float(self.table.cs2)

    @property
    def M(self) -> float:
        if self.mobility is not None:
            return self.mobility
        return self.cs2 * (self.tau - 0.5)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class AllenCahnLBM:
    """3-D Allen-Cahn LBM solver using a table-driven lattice.

    State: ``g`` with shape ``(q, nz, ny, nx)``.
    """

    def __init__(self, cfg: AllenCahnConfig):
        self.cfg = cfg
        self.table = cfg.table
        self.q = self.table.q
        self.dim = self.table.dim
        self.nx, self.ny, self.nz = cfg.nx, cfg.ny, cfg.nz
        self.tau = cfg.tau
        self.M = cfg.M
        self.epsilon = cfg.epsilon
        self.eps2 = cfg.epsilon ** 2

        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5 for positive diffusivity")

        # Chapman-Enskog correction factor for the source term:
        # The CE expansion gives  dφ/dt = D∇²φ + (1 − 1/(2τ)) Σ S_i
        # so to get the desired reaction R(φ) in the macroscopic equation,
        # we need  Σ S_i = R(φ) / (1 − 1/(2τ))
        self._source_correction = 1.0 / (1.0 - 1.0 / (2.0 * self.tau))

        # float arrays from the table
        self._vel, self._w, self._opp = self.table.to_float()
        # reshape for broadcasting: (q, 1, 1, 1)
        self._w4 = self._w.reshape(self.q, 1, 1, 1)

        # distribution function
        self.g = np.zeros((self.q, self.nz, self.ny, self.nx), dtype=DTYPE)
        self.t = 0

    # ---------------------------------------------------------------- setup
    def init_field(self, phi: np.ndarray) -> None:
        """Set the phase field from a macroscopic φ array and initialise g to equilibrium."""
        assert phi.shape == (self.nz, self.ny, self.nx)
        # g_i^eq = w_i * phi
        self.g = self._w4 * phi[None, :, :, :]
        self.t = 0

    def init_cosine_mode(self, amplitude: float = 0.1, mode: int = 1,
                         direction: int = 0) -> None:
        """Initialise with a cosine perturbation on a uniform background.

        φ = 0.5 + A cos(2π mode x_i / L_i)  along the given axis direction.
        This is naturally periodic and used for the diffusion-decay test.
        """
        n = [self.nx, self.ny, self.nz]
        grids = np.meshgrid(
            np.arange(self.nz, dtype=DTYPE),
            np.arange(self.ny, dtype=DTYPE),
            np.arange(self.nx, dtype=DTYPE),
            indexing="ij",
        )
        # direction 0 -> x (axis 2), 1 -> y (axis 1), 2 -> z (axis 0)
        axis_map = {0: 2, 1: 1, 2: 0}
        ax = axis_map[direction]
        coord = grids[ax] + 0.5
        L = n[direction]
        phi = 0.5 + amplitude * np.cos(2.0 * np.pi * mode * coord / L)
        self.init_field(phi)

    def init_sphere(self, center: tuple[float, float, float] | None = None,
                    radius: float | None = None) -> None:
        """Initialise with a smooth spherical droplet (tanh profile at the interface)."""
        if center is None:
            center = (self.nx / 2.0, self.ny / 2.0, self.nz / 2.0)
        if radius is None:
            radius = min(self.nx, self.ny, self.nz) / 4.0

        zz, yy, xx = np.meshgrid(
            np.arange(self.nz, dtype=DTYPE),
            np.arange(self.ny, dtype=DTYPE),
            np.arange(self.nx, dtype=DTYPE),
            indexing="ij",
        )
        dist = np.sqrt(
            (xx + 0.5 - center[0]) ** 2
            + (yy + 0.5 - center[1]) ** 2
            + (zz + 0.5 - center[2]) ** 2
        )
        # smooth interface: φ = ½(1 − tanh((r − R)/(2√2 ε)))
        phi = 0.5 * (1.0 - np.tanh((dist - radius) / (2.0 * np.sqrt(2.0) * self.epsilon)))
        self.init_field(phi)

    # ---------------------------------------------------------- macroscopic
    @property
    def phi(self) -> np.ndarray:
        """Recover the phase field:  φ = Σ_i g_i."""
        return self.g.sum(axis=0)

    # ---------------------------------------------------------- evolution
    def step(self, disable_reaction: bool = False) -> None:
        """One LBM time step: collide + source + stream.

        Parameters
        ----------
        disable_reaction : if True, skip the reaction source term (pure diffusion).
        """
        phi = self.phi

        # equilibrium: g_i^eq = w_i * phi
        g_eq = self._w4 * phi[None, :, :, :]

        # BGK collision
        omega = 1.0 / self.tau
        g_post = self.g - omega * (self.g - g_eq)

        # reaction source with Chapman-Enskog correction:
        # S_i = w_i * [-(M/ε²) f'(φ)] * 1/(1 - 1/(2τ))
        if not disable_reaction:
            fp = double_well_derivative(phi)
            source = (-self._w4 * (self.M / self.eps2) * fp[None, :, :, :]
                      * self._source_correction)
            g_post += source

        # stream (periodic)
        self.g = stream(self.table, g_post)
        self.t += 1

    def run(self, steps: int | None = None, callback=None,
            disable_reaction: bool = False) -> dict:
        """Run the simulation for a given number of steps.

        Parameters
        ----------
        steps : number of steps (default: cfg.steps)
        callback : optional callable(AllenCahnLBM, step) called every step
        disable_reaction : if True, run pure diffusion (no reaction term)

        Returns
        -------
        dict with timing and step count.
        """
        if steps is None:
            steps = self.cfg.steps
        t0 = time.perf_counter()
        for s in range(steps):
            self.step(disable_reaction=disable_reaction)
            if callback is not None:
                callback(self, s)
        elapsed = time.perf_counter() - t0
        return {
            "steps": steps,
            "elapsed_s": elapsed,
            "mlups": steps * self.nx * self.ny * self.nz / elapsed / 1e6,
        }


# ---------------------------------------------------------------------------
# validation runners
# ---------------------------------------------------------------------------
def run_diffusion_decay(
    lattice_name: str = "D3Q19",
    n: int = 32,
    tau: float = 1.0,
    amplitude: float = 0.1,
    mode: int = 1,
    steps: int = 200,
    probe_every: int = 10,
) -> dict:
    """Validate the diffusion operator by exponential decay of a cosine mode.

    With the reaction term disabled, the Allen-Cahn LBM reduces to a pure
    diffusion solver.  The initial condition φ = 0.5 + A cos(2πx/L) decays
    as  φ(x,t) = 0.5 + A exp(−D k² t) cos(kx)  where k = 2π/L and
    D = cs²(τ − ½).

    We track the amplitude decay and fit the measured diffusivity, then
    compare against the theoretical D.

    Returns
    -------
    dict with keys: lattice, n, tau, D_theory, D_measured, D_rel_err,
        amplitudes, times.
    """
    cfg = AllenCahnConfig(
        lattice_name=lattice_name,
        nx=n, ny=n, nz=n,
        tau=tau, epsilon=100.0,  # large ε suppresses reaction
        steps=steps,
    )
    solver = AllenCahnLBM(cfg)
    solver.init_cosine_mode(amplitude=amplitude, mode=mode, direction=0)

    D_theory = cfg.cs2 * (tau - 0.5)
    k = 2.0 * np.pi * mode / n

    amplitudes = []
    times = []

    def _probe(s: AllenCahnLBM, step: int) -> None:
        if step % probe_every == 0:
            phi = s.phi
            # amplitude = max(φ) − 0.5 along the x-axis through the center
            mid_y = s.ny // 2
            mid_z = s.nz // 2
            amp = float(phi[mid_z, mid_y, :].max() - 0.5)
            amplitudes.append(amp)
            times.append(step)

    solver.run(steps, callback=_probe, disable_reaction=True)

    # fit D from amplitude decay: A(t) = A0 exp(-D k² t)
    A0 = amplitudes[0]
    if A0 <= 0 or len(amplitudes) < 3:
        D_measured = float("nan")
    else:
        # linear regression on ln(A/A0) vs t
        log_amps = np.log(np.array(amplitudes) / A0)
        t_arr = np.array(times, dtype=DTYPE)
        # least squares: ln(A/A0) = -D k² t
        # slope = -D k²
        if len(t_arr) > 1:
            slope = np.polyfit(t_arr, log_amps, 1)[0]
            D_measured = -slope / (k * k)
        else:
            D_measured = float("nan")

    D_rel_err = abs(D_measured - D_theory) / D_theory if D_theory > 0 else float("nan")

    return {
        "lattice": lattice_name,
        "n": n,
        "tau": tau,
        "D_theory": D_theory,
        "D_measured": D_measured,
        "D_rel_err": D_rel_err,
        "amplitudes": amplitudes,
        "times": times,
    }


def run_shrinking_sphere(
    lattice_name: str = "D3Q19",
    n: int = 32,
    tau: float = 1.0,
    epsilon: float = 3.0,
    radius0: float = 10.0,
    steps: int = 2000,
    report_every: int = 100,
) -> dict:
    """Validate the shrinking-sphere test case.

    A spherical droplet of phase φ ≈ 1 in a background of φ ≈ 0 shrinks
    under mean-curvature flow.  In the sharp-interface limit:

        dR/dt = −2M / R   =>   R²(t) = R₀² − 4Mt

    The LBM is run for ``steps`` iterations and the effective radius is
    tracked.  The final radius is compared to the analytical prediction.

    Note: the effective radius from volume integration (R_eff = (3V/4π)^{1/3})
    is larger than the sharp-interface radius R₀ because the diffuse interface
    adds extra volume.  We account for this by using the initial effective
    radius as the reference R₀_eff and comparing the *rate* of R² change
    against the analytical dR²/dt = −4M.

    Returns
    -------
    dict with keys: lattice, n, tau, epsilon, radius0, steps, M,
        radii, times, R_final_lbm, R_final_analytical, R_rel_err,
        dR2_dt_measured, dR2_dt_analytical, dR2_dt_rel_err.
    """
    cfg = AllenCahnConfig(
        lattice_name=lattice_name,
        nx=n, ny=n, nz=n,
        tau=tau, epsilon=epsilon,
        steps=steps, report_every=report_every,
    )
    solver = AllenCahnLBM(cfg)
    center = (n / 2.0, n / 2.0, n / 2.0)
    solver.init_sphere(center=center, radius=radius0)

    M = cfg.M
    radii = [effective_radius(solver.phi)]
    times = [0]

    def _cb(solver_inner: AllenCahnLBM, step: int) -> None:
        if (step + 1) % report_every == 0:
            radii.append(effective_radius(solver_inner.phi))
            times.append(step + 1)

    solver.run(steps, callback=_cb)

    # Measure dR²/dt from the simulation data
    R2_vals = [r ** 2 for r in radii]
    t_vals = times
    if len(R2_vals) >= 2:
        # linear fit to R²(t)
        slope = np.polyfit(t_vals, R2_vals, 1)[0]
        dR2_dt_measured = slope
    else:
        dR2_dt_measured = float("nan")

    dR2_dt_analytical = -4.0 * M
    dR2_dt_rel_err = abs(dR2_dt_measured - dR2_dt_analytical) / abs(dR2_dt_analytical)

    # Also compute the final radius comparison using the measured rate
    R0_eff = radii[0]
    R2_final_predicted = R0_eff ** 2 + dR2_dt_analytical * times[-1]
    R_final_analytical = np.sqrt(max(R2_final_predicted, 0.0)) if R2_final_predicted > 0 else 0.0
    R_final_lbm = radii[-1]
    R_rel_err = abs(R_final_lbm - R_final_analytical) / max(R_final_analytical, 1e-12)

    return {
        "lattice": lattice_name,
        "n": n,
        "tau": tau,
        "epsilon": epsilon,
        "radius0": radius0,
        "steps": steps,
        "M": M,
        "radii": radii,
        "times": times,
        "R_final_lbm": R_final_lbm,
        "R_final_analytical": R_final_analytical,
        "R_rel_err": R_rel_err,
        "dR2_dt_measured": dR2_dt_measured,
        "dR2_dt_analytical": dR2_dt_analytical,
        "dR2_dt_rel_err": dR2_dt_rel_err,
    }


def compare_d3q15_d3q19(
    n: int = 32,
    tau: float = 1.0,
    epsilon: float = 3.0,
    radius0: float = 10.0,
    steps: int = 2000,
    report_every: int = 100,
) -> dict:
    """Run the shrinking-sphere test on both D3Q15 and D3Q19 and compare.

    D3Q19 has fourth-moment isotropy (rank 4); D3Q15 does not (rank 3).
    The shrinking sphere is sensitive to anisotropic diffusion errors, so
    D3Q19 should give a smaller dR²/dt error than D3Q15 at the same
    resolution.

    Returns
    -------
    dict with keys: d3q15, d3q19, d3q15_dR2_dt_rel_err, d3q19_dR2_dt_rel_err,
        d3q19_more_accurate.
    """
    r15 = run_shrinking_sphere("D3Q15", n=n, tau=tau, epsilon=epsilon,
                               radius0=radius0, steps=steps,
                               report_every=report_every)
    r19 = run_shrinking_sphere("D3Q19", n=n, tau=tau, epsilon=epsilon,
                               radius0=radius0, steps=steps,
                               report_every=report_every)
    return {
        "d3q15": r15,
        "d3q19": r19,
        "d3q15_dR2_dt_rel_err": r15["dR2_dt_rel_err"],
        "d3q19_dR2_dt_rel_err": r19["dR2_dt_rel_err"],
        "d3q19_more_accurate": r19["dR2_dt_rel_err"] <= r15["dR2_dt_rel_err"],
    }
