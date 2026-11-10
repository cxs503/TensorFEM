"""Fluid-structure interaction (FSI) wet modal analysis for submerged structures.

This module owns the *fluid side* of a wet-vibration eigenvalue problem

.. math::

    \left(K - \omega^2 (M_s + M_a)\right)\phi = 0

in the low-frequency (incompressible) fluid limit, and it verifies the fluid
loading operators against closed-form references.  Two anchors are implemented:

1. **Submerged thin cylindrical shell** -- Junger & Feit (1986) modal added
   mass, ``L/D = 10``, ``R/t = 100``.
2. **Submerged simply supported rectangular panel** -- the exact low-frequency
   baffled-panel added mass (Rayleigh integral, ``k a << 1``) with the classical
   asymptotic law ``m_a = rho_f / k_mn``.

Geometry convention: cylinder axis along ``x``, radius ``R``, length ``L``,
circumferential wave number ``n >= 1``, axial half-wave number ``m >= 1``.

Theory (cylinder)
-----------------
For an external incompressible fluid the displacement potential ``psi``
(``dpsi/dn = w`` on the wetted surface) of the mode
``w = cos(n theta) sin(m pi x / L)`` is, in the slender-body / thin-shell
approximation,

.. math:: \psi = A K_n(k_m r) \cos n\theta \sin(m\pi x/L), \quad k_m = m\pi/L

with ``K_n`` the modified Bessel function that decays as ``r -> infinity``
(use ``I_n`` for a fluid-filled shell).  Normal-velocity continuity at ``r = R``
and the unsteady Bernoulli relation ``p = -rho_f dphi/dt`` give the added mass
per unit wetted area

.. math::

    m_a(m,n) = \rho_f R \, C(m,n), \qquad
    C(m,n) = \frac{-K_n(x)}{x K_n'(x)}, \quad x = k_m R

(``C = I_n/(x I_n')`` for internal fluid).  As ``x -> 0`` both branches tend to
``1/n``, which reproduces the exact two-dimensional result ``m_a = rho_f R/n``
(rigid-translation limit ``n = 1``: ``rho_f pi R^2`` per unit length).
Projecting on the mode shape, ``int w^2 dA = pi R L / 2``, the modal added-mass
ratio is

.. math::

    \mu(m,n) = \frac{M_a}{M_s} = \frac{\rho_f}{\rho_s}\frac{R}{t} C(m,n),
    \qquad \frac{f_{wet}}{f_{dry}} = \frac{1}{\sqrt{1 + \mu}}

Theory (plate)
--------------
For a panel in an infinite rigid baffle with negligible acoustic
compressibility (``k_0 a << 1``) the fluid reaction is the Rayleigh
single-layer operator with kernel ``1/(2\pi|x-y|)``.  In wave-number space the
added mass of mode ``phi`` is

.. math::

    M_a = \frac{\rho_f}{(2\pi)^2}\iint
          \frac{|\tilde\phi(k_x,k_y)|^2}{\sqrt{k_x^2+k_y^2}}\,dk_x\,dk_y

which tends to the classical asymptotic ``m_a = rho_f / k_mn``,
``k_mn = pi sqrt((m/a)^2 + (n/b)^2)``, for high modal wave numbers.

Numerical fluid domain
----------------------
The analytic Bessel result is cross-checked by an independent finite-difference
solution of the *same* incompressible field equation for circumferential
harmonic ``n``,

.. math::

    \frac{\partial^2\psi}{\partial r^2} + \frac{1}{r}
    \frac{\partial\psi}{\partial r} - \frac{n^2}{r^2}\psi
    + \frac{\partial^2\psi}{\partial x^2} = 0

on an ``(r, x)`` grid with ``dpsi/dr = w(x)`` on the wetted wall.  Two end
conditions are supported: ``baffle`` (the wetted wall is continued rigidly past
the shell ends -- the idealisation behind the Bessel formula) and
``rigid_caps`` (the shell is closed by rigid end plates, so no axial flux
escapes).  Their difference quantifies the finite-length end effect at
``L/D = 10``.

Scope limits
------------
* Linear, small strain, isotropic, uniform thickness; no damping, prestress or
  mean flow.
* Fluid is incompressible (``k_0 a << 1``, well below the coincidence
  frequency); acoustic radiation damping is **not** modelled.
* Axisymmetric (``n = 0``) shell modes are rejected: a net-volume-changing
  radial mode has no incompressible solution and needs a compressible model.
* The cylinder reference uses Donnell thin-shell theory for ``n >= 2``; the
  ``n = 1`` beam branch is outside Donnell's accuracy and is not adjudicated.

References
----------
* Junger, M. C. & Feit, D. (1986). *Sound and Structural Vibration*, 2nd ed.
  Wiley, Ch. 4-5 (modal added mass of cylindrical shells).
* Gorman, D. J. & Horacek, J. (1990). "Free vibration analysis of a completely
  free rectangular plate." *J. Sound Vib.* 139(1), 1-17.  Dry/in-vacuo
  benchmark; see ``docs/fsi-wet-modal-dual-criteria.md`` for why it cannot
  anchor a *wet* case.
* Davies, H. G. (1971). "Low frequency random excitation of water-loaded
  rectangular plates." *J. Sound Vib.* 15(1), 107-126.
* Fahy, F. (1985). *Sound and Structural Vibration*. Academic Press, Sec. 4
  (baffled-panel radiation mass, ``m_a = rho/k_mn``).
* Blevins, R. D. (1979). *Formulas for Natural Frequency and Mode Shape_.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

__all__ = [
    "cylinder_fluid_loading_coefficient",
    "cylinder_modal_added_mass_ratio",
    "wet_frequency_ratio",
    "donnell_shell_frequency_hz",
    "analytic_submerged_cylinder",
    "IncompressiblePotential",
    "solve_incompressible_potential",
    "numerical_local_added_mass_per_area",
    "numerical_cylinder_added_mass_ratio",
    "CylinderWetModalResult",
    "solve_submerged_cylinder_wet_modal",
    "plate_modal_wavenumber",
    "plate_asymptotic_added_mass_per_area",
    "plate_exact_modal_added_mass",
    "plate_panel_added_mass_ratio",
    "PlateWetModalResult",
    "plate_mode_shape_check",
    "solve_submerged_plate_wet_modal",
    "FRFResult",
    "submerged_plate_frf",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return value


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _poisson(value: float) -> float:
    value = float(value)
    if not (-1.0 < value < 0.5):
        raise ValueError(f"poisson must lie in (-1, 0.5), got {value!r}")
    return value


# ---------------------------------------------------------------------------
# 1. Closed-form fluid loading (Junger & Feit)
# ---------------------------------------------------------------------------


def cylinder_fluid_loading_coefficient(
    radius: float,
    length: float,
    n_axial: int,
    n_circ: int,
    *,
    exterior: bool = True,
) -> float:
    """Dimensionless modal fluid-loading coefficient ``C(m, n)``.

    ``m_a(m, n) = rho_f * R * C(m, n)`` is the added mass per unit wetted area,
    ``C = -K_n(x)/(x K_n'(x))`` for an external fluid and ``C = I_n(x)/(x
    I_n'(x))`` for an internal fluid, with ``x = m*pi*R/L``.  Both branches tend
    to ``1/n`` as ``x -> 0``.

    Parameters
    ----------
    radius, length : float
        Shell radius and length [m].
    n_axial : int
        Axial half-wave number ``m >= 1``.
    n_circ : int
        Circumferential wave number ``n >= 1``.
    exterior : bool
        ``True`` for a submerged shell, ``False`` for a fluid-filled shell.

    Raises
    ------
    RuntimeError
        If scipy (needed for the Bessel functions) is not installed.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    n_axial = _count("n_axial", n_axial)
    n_circ = _count("n_circ", n_circ)
    try:
        from scipy.special import iv, ivp, kv, kvp
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "tensorfem.fsi_wet_modal requires scipy for the modified Bessel "
            "functions of the Junger & Feit fluid loading; install "
            "'scipy>=1.11' (the 'dev' extra) to use this module"
        ) from exc

    x = n_axial * math.pi * radius / length
    order = float(n_circ)
    if exterior:
        value = -float(kv(order, x)) / (x * float(kvp(order, x)))
    else:
        value = float(iv(order, x)) / (x * float(ivp(order, x)))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"non-positive fluid loading coefficient {value!r} for "
            f"m={n_axial}, n={n_circ}"
        )
    return value


def cylinder_modal_added_mass_ratio(
    radius: float,
    length: float,
    thickness: float,
    rho_fluid: float,
    rho_struct: float,
    n_axial: int,
    n_circ: int,
    *,
    exterior: bool = True,
) -> float:
    """Modal added-mass ratio ``mu = M_a / M_s`` of a thin cylindrical shell.

    ``mu = (rho_f/rho_s) (R/t) C(m, n)`` follows from projecting the added mass
    per unit area on ``w = cos(n theta) sin(m pi x / L)``: the fluid and the
    structural generalised mass carry the same ``int w^2 dA = pi R L / 2``.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    coefficient = cylinder_fluid_loading_coefficient(
        radius, length, n_axial, n_circ, exterior=exterior
    )
    return float(rho_fluid / rho_struct * radius / thickness * coefficient)


def wet_frequency_ratio(added_mass_ratio: float) -> float:
    """``f_wet / f_dry = 1 / sqrt(1 + mu)`` for one added-mass ratio."""
    added_mass_ratio = float(added_mass_ratio)
    if not math.isfinite(added_mass_ratio) or added_mass_ratio < 0.0:
        raise ValueError("added_mass_ratio must be finite and >= 0")
    return 1.0 / math.sqrt(1.0 + added_mass_ratio)


# ---------------------------------------------------------------------------
# 2. Donnell thin-shell dry frequency (analytic structural reference)
# ---------------------------------------------------------------------------


def donnell_shell_frequency_hz(
    radius: float,
    length: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
    n_axial: int = 1,
    n_circ: int = 2,
) -> float:
    """Simply supported cylindrical shell frequency, Donnell-Mushtari-Vlasov.

    ``omega^2 = E/(rho R^2)[lambda^4/(lambda^2+n^2)^2 + k(lambda^2+n^2)^2]``
    with ``lambda = m pi R/L`` and ``k = t^2/(12 R^2 (1-nu^2))``.

    Unlike :func:`tensorfem.cylindrical_shell_modal.dmv_angular_frequency` the
    membrane term ``lambda^4/(lambda^2+n^2)^2`` is retained for ``n = 1`` as
    well; that helper drops it, which underestimates the ``n = 1`` branch by
    about a factor of 8 on an ``L/R = 20`` shell.  The ``n = 1`` branch is still
    excluded from the FSI adjudication because thin-shell theory is inaccurate
    there.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    density = _positive("density", density)
    poisson = _poisson(poisson)
    n_axial = _count("n_axial", n_axial)
    n_circ = _count("n_circ", n_circ)

    lam = n_axial * math.pi * radius / length
    order = float(n_circ)
    kappa = thickness ** 2 / (12.0 * radius ** 2 * (1.0 - poisson ** 2))
    denominator = (lam ** 2 + order ** 2) ** 2
    omega_sq = young / (density * radius ** 2) * (
        lam ** 4 / denominator + kappa * denominator
    )
    if omega_sq <= 0.0:
        raise ValueError("non-positive omega^2 for the requested mode")
    return math.sqrt(omega_sq) / (2.0 * math.pi)


def analytic_submerged_cylinder(
    *,
    radius: float,
    length: float,
    thickness: float,
    rho_fluid: float,
    rho_struct: float,
    young: float,
    poisson: float,
    modes: Sequence[tuple[int, int]],
) -> dict:
    """Closed-form (Donnell + Junger & Feit) wet modal prediction.

    Returns lists ``dry_hz``, ``wet_hz``, ``mu`` and ``ratio`` for the
    requested ``(m, n)`` modes.
    """
    dry: list[float] = []
    wet: list[float] = []
    ratios: list[float] = []
    mus: list[float] = []
    for m, n in modes:
        f_dry = donnell_shell_frequency_hz(
            radius, length, thickness, young, poisson, rho_struct, m, n
        )
        mu = cylinder_modal_added_mass_ratio(
            radius, length, thickness, rho_fluid, rho_struct, m, n
        )
        ratio = wet_frequency_ratio(mu)
        dry.append(f_dry)
        mus.append(mu)
        ratios.append(ratio)
        wet.append(f_dry * ratio)
    return {"dry_hz": dry, "wet_hz": wet, "mu": mus, "ratio": ratios}# ---------------------------------------------------------------------------
# 3. Numerical incompressible fluid domain (finite differences on (r, x))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IncompressiblePotential:
    """Incompressible displacement potential on the wetted wall.

    ``wall_potential`` is ``psi`` at ``r = R`` where ``dpsi/dr =
    wall_velocity``; ``x`` is the axial grid.
    """

    x: np.ndarray
    wall_potential: np.ndarray
    wall_velocity: np.ndarray
    r: np.ndarray
    field: np.ndarray
    radius: float
    n_circ: int
    end_condition: str
    n_radial: int
    n_axial: int
    radial_extent: float


def solve_incompressible_potential(
    radius: float,
    n_circ: int,
    wall_velocity: Callable[[np.ndarray], np.ndarray],
    *,
    x_min: float,
    x_max: float,
    wall_x_min: Optional[float] = None,
    wall_x_max: Optional[float] = None,
    radial_extent: float = 6.0,
    n_radial: int = 160,
    n_axial_grid: int = 400,
    end_condition: str = "baffle",
) -> IncompressiblePotential:
    """Solve the incompressible fluid field around a cylindrical wall.

    The displacement potential is ``psi(r, x, theta) = Psi(r, x) cos(n theta)``
    and satisfies ``Psi_rr + Psi_r/r - n^2 Psi/r^2 + Psi_xx = 0`` with

    * ``dPsi/dr = wall_velocity(x)`` on ``r = R`` for ``wall_x_min <= x <=
      wall_x_max`` (zero flux elsewhere: rigid baffle continuation),
    * ``dPsi/dr + (n/r) Psi = 0`` on the outer boundary ``r = R (1 +
      radial_extent)`` -- the exact far-field decay of the circumferential
      harmonic ``r^{-n}`` (a Dirichlet condition there would truncate the field
      and bias the added mass low),
    * ``end_condition = "baffle"``: ``Psi = 0`` at ``x_min`` / ``x_max``;
      ``end_condition = "rigid_caps"``: ``dPsi/dx = 0`` at both axial ends
      (rigid end plates, no axial flux escape).

    The discrete system is a second-order finite-difference stencil on an
    ``(r, x)`` grid, solved with a sparse direct solver.
    """
    radius = _positive("radius", radius)
    n_circ = _count("n_circ", n_circ)
    if end_condition not in ("baffle", "rigid_caps"):
        raise ValueError("end_condition must be 'baffle' or 'rigid_caps'")
    x_min = float(x_min)
    x_max = float(x_max)
    if not (x_min < x_max):
        raise ValueError("x_min must be < x_max")
    wall_x_min = x_min if wall_x_min is None else float(wall_x_min)
    wall_x_max = x_max if wall_x_max is None else float(wall_x_max)
    if not (wall_x_min < wall_x_max):
        raise ValueError("wall_x_min must be < wall_x_max")
    radial_extent = _positive("radial_extent", radial_extent)
    n_radial = _count("n_radial", n_radial)
    n_axial_grid = _count("n_axial_grid", n_axial_grid)

    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "tensorfem.fsi_wet_modal requires scipy for the sparse "
            "incompressible fluid solve; install 'scipy>=1.11' (the 'dev' "
            "extra) to use this module"
        ) from exc

    r_out = radius * (1.0 + radial_extent)
    r = np.linspace(radius, r_out, n_radial + 1)
    x = np.linspace(x_min, x_max, n_axial_grid + 1)
    dr = float(r[1] - r[0])
    dx = float(x[1] - x[0])
    nr, nx = r.size, x.size
    n_unknown = nr * nx

    velocity = np.zeros(nx)
    inside = (x >= wall_x_min - 1e-12) & (x <= wall_x_max + 1e-12)
    velocity[inside] = np.asarray(wall_velocity(x[inside]), dtype=float)

    def index(i: int, j: int) -> int:
        return i * nx + j

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_unknown)

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        vals.append(value)

    for i in range(nr):
        for j in range(nx):
            k = index(i, j)
            if i == 0:  # wetted wall: normal flux, 2nd-order one-sided stencil
                add(k, k, -3.0 / (2.0 * dr))
                add(k, index(1, j), 4.0 / (2.0 * dr))
                add(k, index(2, j), -1.0 / (2.0 * dr))
                rhs[k] = velocity[j]
            elif i == nr - 1:  # outer boundary: far-field decay dpsi/dr = -n psi/r
                rr = float(r[i])
                order = float(n_circ) / rr
                add(k, k, 3.0 / (2.0 * dr) + order)
                add(k, index(i - 1, j), -4.0 / (2.0 * dr))
                add(k, index(i - 2, j), 1.0 / (2.0 * dr))
            elif j == 0 or j == nx - 1:
                if end_condition == "rigid_caps" and j == 0:
                    add(k, index(i, 1), 1.0 / dx)
                    add(k, k, -1.0 / dx)
                elif end_condition == "rigid_caps" and j == nx - 1:
                    add(k, index(i, nx - 2), 1.0 / dx)
                    add(k, k, -1.0 / dx)
                else:
                    add(k, k, 1.0)
            else:
                rr = float(r[i])
                add(k, index(i + 1, j), 1.0 / dr ** 2 + 1.0 / (2.0 * dr * rr))
                add(k, index(i - 1, j), 1.0 / dr ** 2 - 1.0 / (2.0 * dr * rr))
                add(k, index(i, j + 1), 1.0 / dx ** 2)
                add(k, index(i, j - 1), 1.0 / dx ** 2)
                add(k, k, -2.0 / dr ** 2 - 2.0 / dx ** 2 - n_circ ** 2 / rr ** 2)

    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown))
    solution = spla.spsolve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("incompressible potential solve failed (non-finite)")
    psi = solution.reshape(nr, nx)

    return IncompressiblePotential(
        x=x,
        wall_potential=np.ascontiguousarray(psi[0, :]),
        r=r,
        field=psi,
        wall_velocity=velocity,
        radius=radius,
        n_circ=n_circ,
        end_condition=end_condition,
        n_radial=n_radial,
        n_axial=int(x.size),
        radial_extent=radial_extent,
    )


def numerical_local_added_mass_per_area(
    potential: IncompressiblePotential,
    rho_fluid: float,
    x_probe: Sequence[float],
) -> np.ndarray:
    """Local added mass per unit area ``-rho_f psi / w`` at the probe points.

    For an axially harmonic wall velocity ``w = sin(k x)`` the local reaction of
    the Bessel (infinite-cylinder) solution is ``m_a = rho_f R C(k, n)``, so this
    is a pointwise comparison of the numerical fluid field against the closed
    form -- sharper than comparing integrals.  Sign: ``psi < 0`` where ``w > 0``
    because the fluid is pushed outward, hence the minus sign (``p = rho_f
    omega^2 psi``).
    """
    rho_fluid = _positive("rho_fluid", rho_fluid)
    values: list[float] = []
    for probe in np.atleast_1d(np.asarray(x_probe, dtype=float)):
        j = int(np.argmin(np.abs(potential.x - probe)))
        w = float(potential.wall_velocity[j])
        if abs(w) < 1e-12:
            values.append(float("nan"))
            continue
        values.append(float(-rho_fluid * potential.wall_potential[j] / w))
    return np.asarray(values)


def numerical_cylinder_added_mass_ratio(
    *,
    radius: float,
    length: float,
    thickness: float,
    rho_fluid: float,
    rho_struct: float,
    n_axial: int,
    n_circ: int,
    end_condition: str = "baffle",
    radial_extent: float = 6.0,
    n_radial: int = 160,
    n_axial_grid: int = 400,
    axial_padding: float = 2.0,
) -> dict:
    """Modal added-mass ratio of the *finite* shell from the numerical field.

    The wall velocity is the shell mode ``sin(m pi x / L)`` on ``0 <= x <= L``.
    With ``int cos^2(n theta) dtheta = pi`` the modal added mass is ``M_a =
    -rho_f pi R \\int psi w dx`` and the ratio uses the thin-shell generalised
    mass ``M_s = rho_s t pi R L / 2``.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    n_axial = _count("n_axial", n_axial)
    n_circ = _count("n_circ", n_circ)
    axial_padding = float(axial_padding)
    if axial_padding < 0.0:
        raise ValueError("axial_padding must be >= 0")

    k_m = n_axial * math.pi / length

    def velocity(z: np.ndarray) -> np.ndarray:
        return np.sin(k_m * z)

    if end_condition == "rigid_caps":
        x_min, x_max = 0.0, length
    else:
        pad = axial_padding * radius
        x_min, x_max = -pad, length + pad

    potential = solve_incompressible_potential(
        radius,
        n_circ,
        velocity,
        x_min=x_min,
        x_max=x_max,
        wall_x_min=0.0,
        wall_x_max=length,
        radial_extent=radial_extent,
        n_radial=n_radial,
        n_axial_grid=n_axial_grid,
        end_condition=end_condition,
    )
    shell = (potential.x >= 0.0) & (potential.x <= length)
    xs = potential.x[shell]
    psi_wall = potential.wall_potential[shell]
    w = np.sin(k_m * xs)
    modal_added_mass = -rho_fluid * math.pi * radius * float(
        np.trapezoid(psi_wall * w, xs)
    )
    structural_mass = rho_struct * thickness * math.pi * radius * length / 2.0
    return {
        "added_mass_ratio": modal_added_mass / structural_mass,
        "modal_added_mass": modal_added_mass,
        "structural_generalized_mass": structural_mass,
        "end_condition": end_condition,
        "n_radial": potential.n_radial,
        "n_axial_grid": potential.n_axial,
    }


# ---------------------------------------------------------------------------
# 4. Submerged cylinder: dual-criteria wet modal result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CylinderWetModalResult:
    """Dual-criteria wet modal result for a submerged cylindrical shell."""

    mode_numbers: tuple[tuple[int, int], ...]
    dry_hz_analytic: tuple[float, ...]
    wet_hz_analytic: tuple[float, ...]
    mu_analytic: tuple[float, ...]
    ratio_analytic: tuple[float, ...]
    mu_numerical_baffle: tuple[float, ...]
    mu_numerical_rigid_caps: tuple[float, ...]
    wet_hz_numerical: tuple[float, ...]
    radius: float
    length: float
    thickness: float
    local_coefficient_error: float

    def summary(self) -> dict:
        """JSON-serialisable view of the result."""
        return {
            "mode_numbers": [list(mode) for mode in self.mode_numbers],
            "dry_hz_analytic": list(self.dry_hz_analytic),
            "wet_hz_analytic": list(self.wet_hz_analytic),
            "mu_analytic": list(self.mu_analytic),
            "ratio_analytic": list(self.ratio_analytic),
            "mu_numerical_baffle": list(self.mu_numerical_baffle),
            "mu_numerical_rigid_caps": list(self.mu_numerical_rigid_caps),
            "wet_hz_numerical": list(self.wet_hz_numerical),
            "local_coefficient_max_rel_error": self.local_coefficient_error,
            "radius": self.radius,
            "length": self.length,
            "thickness": self.thickness,
        }


def solve_submerged_cylinder_wet_modal(
    *,
    radius: float = 1.0,
    length: float = 20.0,
    thickness: float = 0.01,
    rho_fluid: float = 1025.0,
    rho_struct: float = 7850.0,
    young: float = 210.0e9,
    poisson: float = 0.3,
    modes: Sequence[tuple[int, int]] = ((1, 2), (1, 3), (2, 2), (1, 4), (2, 3)),
    radial_extent: float = 6.0,
    n_radial: int = 160,
    n_axial_grid: int = 400,
    local_check_mode: tuple[int, int] = (1, 2),
) -> CylinderWetModalResult:
    """Wet modal analysis of a submerged cylindrical shell with two criteria.

    Criterion A (:func:`analytic_submerged_cylinder`): Donnell dry frequency +
    Junger & Feit closed-form added mass.

    Criterion B (:func:`numerical_cylinder_added_mass_ratio`): the added mass of
    the same mode obtained from a finite-difference solution of the incompressible
    field around the *finite* shell, for both end idealisations.

    In addition ``local_coefficient_error`` reports the pointwise agreement
    ``-rho_f psi / w`` of the numerical field against ``rho_f R C(m, n)`` for
    ``local_check_mode`` -- the sharpest comparison of the two fluid models
    because it removes any integral/normalisation differences.
    """
    radius = _positive("radius", radius)
    length = _positive("length", length)
    thickness = _positive("thickness", thickness)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    if not modes:
        raise ValueError("at least one (m, n) mode is required")
    for m, n in modes:
        _count("n_axial", m)
        _count("n_circ", n)

    analytic = analytic_submerged_cylinder(
        radius=radius,
        length=length,
        thickness=thickness,
        rho_fluid=rho_fluid,
        rho_struct=rho_struct,
        young=young,
        poisson=poisson,
        modes=modes,
    )

    mu_baffle: list[float] = []
    mu_caps: list[float] = []
    wet_numerical: list[float] = []
    for (m, n), f_dry in zip(modes, analytic["dry_hz"]):
        baffle = numerical_cylinder_added_mass_ratio(
            radius=radius,
            length=length,
            thickness=thickness,
            rho_fluid=rho_fluid,
            rho_struct=rho_struct,
            n_axial=m,
            n_circ=n,
            end_condition="baffle",
            radial_extent=radial_extent,
            n_radial=n_radial,
            n_axial_grid=n_axial_grid,
        )
        caps = numerical_cylinder_added_mass_ratio(
            radius=radius,
            length=length,
            thickness=thickness,
            rho_fluid=rho_fluid,
            rho_struct=rho_struct,
            n_axial=m,
            n_circ=n,
            end_condition="rigid_caps",
            radial_extent=radial_extent,
            n_radial=n_radial,
            n_axial_grid=n_axial_grid,
            axial_padding=0.0,
        )
        mu_baffle.append(baffle["added_mass_ratio"])
        mu_caps.append(caps["added_mass_ratio"])
        wet_numerical.append(f_dry * wet_frequency_ratio(baffle["added_mass_ratio"]))

    # Pointwise fluid-loading check with an axially harmonic wall velocity.
    m_check, n_check = local_check_mode
    k_check = m_check * math.pi / length
    probes = np.array([0.25, 0.5, 0.75]) * length
    potential = solve_incompressible_potential(
        radius,
        n_check,
        lambda z: np.sin(k_check * z),
        x_min=-3.0 * radius,
        x_max=length + 3.0 * radius,
        wall_x_min=-3.0 * radius,
        wall_x_max=length + 3.0 * radius,
        radial_extent=radial_extent,
        n_radial=n_radial,
        n_axial_grid=n_axial_grid,
    )
    local = numerical_local_added_mass_per_area(potential, rho_fluid, probes)
    reference = rho_fluid * radius * cylinder_fluid_loading_coefficient(
        radius, length, m_check, n_check
    )
    local_error = float(np.nanmax(np.abs(local - reference)) / reference)

    return CylinderWetModalResult(
        mode_numbers=tuple((int(m), int(n)) for m, n in modes),
        dry_hz_analytic=tuple(analytic["dry_hz"]),
        wet_hz_analytic=tuple(analytic["wet_hz"]),
        mu_analytic=tuple(analytic["mu"]),
        ratio_analytic=tuple(analytic["ratio"]),
        mu_numerical_baffle=tuple(mu_baffle),
        mu_numerical_rigid_caps=tuple(mu_caps),
        wet_hz_numerical=tuple(wet_numerical),
        radius=radius,
        length=length,
        thickness=thickness,
        local_coefficient_error=local_error,
    )# ---------------------------------------------------------------------------
# 5. Submerged plate: exact, panel-discretised and asymptotic added masses
# ---------------------------------------------------------------------------


def plate_modal_wavenumber(
    length: float, width: float, m: int, n: int
) -> float:
    """Modal wave number ``k_mn = pi sqrt((m/a)^2 + (n/b)^2)`` of an SS panel."""
    length = _positive("length", length)
    width = _positive("width", width)
    m = _count("m", m)
    n = _count("n", n)
    return math.pi * math.hypot(m / length, n / width)


def plate_asymptotic_added_mass_per_area(
    length: float,
    width: float,
    m: int,
    n: int,
    rho_fluid: float,
) -> float:
    """Classical baffled-panel asymptotic added mass ``m_a = rho_f / k_mn``.

    Valid for modal wave numbers well above the acoustic wave number
    (``k_mn >> k_0``) and an infinite baffle (Fahy 1985; Davies 1971).
    """
    rho_fluid = _positive("rho_fluid", rho_fluid)
    return rho_fluid / plate_modal_wavenumber(length, width, m, n)


def _sine_fourier_transform(k: np.ndarray, span: float, order: int) -> np.ndarray:
    """``\\int_0^span sin(order*pi*x/span) exp(-i k x) dx`` (stable form)."""
    alpha = order * math.pi / span
    k = np.asarray(k, dtype=float)
    denominator = alpha ** 2 - k ** 2
    numerator = alpha * (1.0 - (-1.0) ** order * np.exp(-1j * k * span))
    result = numerator / denominator
    # Removable singularity at k = +/- alpha: exact limit is -i*span/2.
    singular = np.abs(denominator) < 1e-12 * alpha ** 2
    if np.any(singular):
        result = np.where(singular, -0.5j * span, result)
    return result


def plate_exact_modal_added_mass(
    a: float,
    b: float,
    m: int,
    n: int,
    rho_fluid: float,
    *,
    n_wave: int = 192,
    n_theta: int = 256,
    wave_extent: float = 60.0,
) -> float:
    """Exact incompressible baffled-panel modal added mass of mode ``(m, n)``.

    Evaluates

    .. math:: M_a = \\frac{\\rho_f}{(2\\pi)^2}\\iint
              \\frac{|\\tilde\\phi(k_x,k_y)|^2}{|k|}\\,dk_x\\,dk_y

    in polar wave-number coordinates (so the ``1/|k|`` kernel is absorbed by the
    ``k``-measure and the integrand is bounded).  This is the *exact* linear
    incompressible fluid reaction for a simply supported panel in an infinite
    rigid baffle, i.e. the reference the asymptotic ``rho_f/k_mn`` law and the
    panel discretisation are judged against.
    """
    a = _positive("a", a)
    b = _positive("b", b)
    m = _count("m", m)
    n = _count("n", n)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    n_wave = _count("n_wave", n_wave)
    n_theta = _count("n_theta", n_theta)
    wave_extent = _positive("wave_extent", wave_extent)

    k_modal = plate_modal_wavenumber(a, b, m, n)
    k_max = wave_extent * k_modal
    nodes, weights = np.polynomial.legendre.leggauss(n_wave)
    k = 0.5 * k_max * (nodes + 1.0)
    w_k = 0.5 * k_max * weights
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    d_theta = 2.0 * math.pi / n_theta

    kx = k[:, None] * np.cos(theta)[None, :]
    ky = k[:, None] * np.sin(theta)[None, :]
    phi_hat = _sine_fourier_transform(kx, a, m) * _sine_fourier_transform(ky, b, n)
    integrand = (np.abs(phi_hat) ** 2).sum(axis=1) * d_theta
    return float(rho_fluid / (2.0 * math.pi) ** 2 * np.sum(integrand * w_k))


def _panel_geometry(length: float, width: float, nx: int, ny: int) -> tuple:
    """Element centroids and areas of a uniform ``nx`` x ``ny`` panel mesh."""
    nx = _count("nx", nx)
    ny = _count("ny", ny)
    dx = length / nx
    dy = width / ny
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    areas = np.full(nx * ny, dx * dy)
    return centres, areas


def plate_panel_added_mass_ratio(
    length: float,
    width: float,
    m: int,
    n: int,
    rho_fluid: float,
    rho_struct: float,
    thickness: float,
    *,
    nx: int = 16,
    ny: int = 16,
) -> float:
    """Modal added-mass ratio from the panel-discretised Rayleigh operator.

    The fluid operator is :func:`tensorfem.hydro_added_mass.dipole_added_mass_matrix`
    (piecewise-constant panels, exact equal-area patch self term).  The mode is
    sampled at element centroids, so the panel mesh acts as the fluid
    discretisation and convergence in ``nx`` is a numerical check.
    """
    from .hydro_added_mass import dipole_added_mass_matrix

    length = _positive("length", length)
    width = _positive("width", width)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    thickness = _positive("thickness", thickness)
    centres, areas = _panel_geometry(length, width, nx, ny)
    mode = np.sin(m * math.pi * centres[:, 0] / length) * np.sin(
        n * math.pi * centres[:, 1] / width
    )
    operator = dipole_added_mass_matrix(
        torch.as_tensor(centres), torch.as_tensor(areas), rho_fluid
    ).numpy()
    # The operator already carries the rho/(2 pi) scale and the A_i A_j pair
    # weights, so the modal sum is phi^T M_a phi directly.
    modal_added_mass = float(mode @ operator @ mode)
    structural = rho_struct * thickness * length * width / 4.0
    return modal_added_mass / structural


@dataclass(frozen=True)
class PlateWetModalResult:
    """Dual-criteria wet modal result for a submerged simply supported panel."""

    mode_numbers: tuple[tuple[int, int], ...]
    dry_hz_navier: tuple[float, ...]
    dry_hz_fe: tuple[float, ...]
    wet_hz: tuple[float, ...]
    mu_exact: tuple[float, ...]
    mu_panel: tuple[float, ...]
    mu_asymptotic: tuple[float, ...]
    length: float
    width: float
    thickness: float
    panel_mesh: tuple[int, int]

    def summary(self) -> dict:
        """JSON-serialisable view of the result."""
        return {
            "mode_numbers": [list(mode) for mode in self.mode_numbers],
            "dry_hz_navier": list(self.dry_hz_navier),
            "dry_hz_fe": list(self.dry_hz_fe),
            "wet_hz": list(self.wet_hz),
            "mu_exact": list(self.mu_exact),
            "mu_panel": list(self.mu_panel),
            "mu_asymptotic": list(self.mu_asymptotic),
            "length": self.length,
            "width": self.width,
            "thickness": self.thickness,
            "panel_mesh": list(self.panel_mesh),
        }


def navier_plate_frequency_hz(
    length: float,
    width: float,
    thickness: float,
    young: float,
    poisson: float,
    density: float,
    m: int = 1,
    n: int = 1,
) -> float:
    """Simply supported Kirchhoff plate frequency (Navier closed form)."""
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    density = _positive("density", density)
    poisson = _poisson(poisson)
    m = _count("m", m)
    n = _count("n", n)
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    omega = (
        math.pi ** 2
        * math.sqrt(rigidity / (density * thickness))
        * ((m / length) ** 2 + (n / width) ** 2)
    )
    return omega / (2.0 * math.pi)


def solve_submerged_plate_wet_modal(
    *,
    length: float = 1.0,
    width: float = 1.0,
    thickness: float = 0.005,
    rho_fluid: float = 1025.0,
    rho_struct: float = 7850.0,
    young: float = 210.0e9,
    poisson: float = 0.3,
    modes: Sequence[tuple[int, int]] = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3)),
    panel_mesh: tuple[int, int] = (24, 24),
    fe_mesh: tuple[int, int] = (12, 12),
) -> PlateWetModalResult:
    """Wet modal analysis of a submerged simply supported panel, three criteria.

    * ``mu_exact`` -- :func:`plate_exact_modal_added_mass` (exact incompressible
      baffled-panel reaction, numerical wave-number integral).
    * ``mu_panel`` -- panel-discretised Rayleigh operator
      (:func:`plate_panel_added_mass_ratio`).
    * ``mu_asymptotic`` -- classical ``rho_f/k_mn`` law expressed as a ratio.

    ``dry_hz_navier`` is the closed-form simply supported frequency and
    ``dry_hz_fe`` the Mindlin finite-element value on ``fe_mesh`` (structural
    discretisation check, independent of the fluid).
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    if not modes:
        raise ValueError("at least one (m, n) mode is required")

    from .plate_vibration import solve_plate_vibration

    fe_modes = max(m for m, _ in modes) * max(n for _, n in modes) + 2
    try:
        fe = solve_plate_vibration(
            fe_mesh[0],
            fe_mesh[1],
            length=length,
            width=width,
            young=young,
            poisson=poisson,
            thickness=thickness,
            density=rho_struct,
            modes=fe_modes,
        )
        fe_frequencies = [float(v) for v in fe.frequencies_hz]
    except Exception:  # pragma: no cover - defensive, FE mesh too coarse
        fe_frequencies = []

    dry_navier: list[float] = []
    dry_fe: list[float] = []
    wet: list[float] = []
    mu_exact: list[float] = []
    mu_panel: list[float] = []
    mu_asym: list[float] = []
    for m, n in modes:
        f_dry = navier_plate_frequency_hz(
            length, width, thickness, young, poisson, rho_struct, m, n
        )
        dry_navier.append(f_dry)
        dry_fe.append(
            min(fe_frequencies, key=lambda value: abs(value - f_dry))
            if fe_frequencies
            else float("nan")
        )
        structural = rho_struct * thickness * length * width / 4.0
        mu_e = plate_exact_modal_added_mass(length, width, m, n, rho_fluid) / structural
        mu_p = plate_panel_added_mass_ratio(
            length,
            width,
            m,
            n,
            rho_fluid,
            rho_struct,
            thickness,
            nx=panel_mesh[0],
            ny=panel_mesh[1],
        )
        m_a_asym = plate_asymptotic_added_mass_per_area(
            length, width, m, n, rho_fluid
        )
        mu_exact.append(mu_e)
        mu_panel.append(mu_p)
        mu_asym.append(m_a_asym / (rho_struct * thickness))
        wet.append(f_dry * wet_frequency_ratio(mu_e))

    return PlateWetModalResult(
        mode_numbers=tuple((int(m), int(n)) for m, n in modes),
        dry_hz_navier=tuple(dry_navier),
        dry_hz_fe=tuple(dry_fe),
        wet_hz=tuple(wet),
        mu_exact=tuple(mu_exact),
        mu_panel=tuple(mu_panel),
        mu_asymptotic=tuple(mu_asym),
        length=length,
        width=width,
        thickness=thickness,
        panel_mesh=(int(panel_mesh[0]), int(panel_mesh[1])),
    )


def plate_mode_shape_check(
    *,
    length: float = 1.0,
    width: float = 0.8,
    thickness: float = 0.005,
    young: float = 210.0e9,
    poisson: float = 0.3,
    rho_struct: float = 7850.0,
    elements_x: int = 12,
    elements_y: int = 12,
    m: int = 1,
    n: int = 1,
) -> dict:
    """Compare a Mindlin FE dry mode shape with the analytic sine mode.

    The eigenvector closest (in frequency) to the Navier value of ``(m, n)`` is
    correlated with ``sin(m pi x/a) sin(n pi y/b)`` through the modal assurance
    criterion ``MAC = (phi . psi)^2 / (|phi|^2 |psi|^2)`` and a scaled maximum
    deviation.  Use a *rectangular* panel to keep the two lowest modes of a
    square panel separated.
    """
    from .plate_vibration import solve_plate_vibration

    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    young = _positive("young", young)
    rho_struct = _positive("rho_struct", rho_struct)
    poisson = _poisson(poisson)
    elements_x = _count("elements_x", elements_x)
    elements_y = _count("elements_y", elements_y)
    m = _count("m", m)
    n = _count("n", n)

    result = solve_plate_vibration(
        elements_x, elements_y, length=length, width=width, young=young,
        poisson=poisson, thickness=thickness, density=rho_struct,
        modes=max(m, n) * 4,
    )
    target = navier_plate_frequency_hz(
        length, width, thickness, young, poisson, rho_struct, m, n
    )
    frequencies = [float(value) for value in result.frequencies_hz]
    index = min(range(len(frequencies)), key=lambda i: abs(frequencies[i] - target))
    nodes = result.nodes
    node_ids = torch.arange(nodes.shape[0])
    phi = result.modes[3 * node_ids, index]
    x = nodes[node_ids, 0]
    y = nodes[node_ids, 1]
    analytic = torch.sin(m * math.pi * x / length) * torch.sin(
        n * math.pi * y / width
    )
    if float(torch.dot(phi, analytic)) < 0.0:  # eigenvector sign is arbitrary
        phi = -phi
    phi_norm = phi / torch.linalg.vector_norm(phi)
    psi_norm = analytic / torch.linalg.vector_norm(analytic)
    mac = float(torch.dot(phi_norm, psi_norm) ** 2)
    deviation = float(
        (phi / phi.abs().max() - analytic / analytic.abs().max()).abs().max()
    )
    return {
        "m": m,
        "n": n,
        "fe_index": int(index),
        "fe_hz": frequencies[index],
        "navier_hz": target,
        "frequency_error": frequencies[index] / target - 1.0,
        "mac": mac,
        "max_scaled_deviation": deviation,
    }


# ---------------------------------------------------------------------------
# 6. Frequency response function of a submerged panel (modal superposition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FRFResult:
    """Frequency response function of a submerged panel."""

    frequencies_hz: torch.Tensor
    displacement: torch.Tensor  # complex modal displacement per unit force
    dry_frequencies_hz: tuple[float, ...]
    wet_frequencies_hz: tuple[float, ...]
    added_mass_ratios: tuple[float, ...]
    generalized_masses: tuple[float, ...]
    damping_ratio: float
    force_location: tuple[float, float]
    response_location: tuple[float, float]
    participation: tuple[float, ...]

    def amplitude(self) -> torch.Tensor:
        """``|H(f)|`` [m/N]."""
        return torch.abs(self.displacement)

    def peak_frequencies_hz(self) -> list[float]:
        """Frequencies of the local maxima of ``|H|`` on the sampled grid."""
        amplitude = self.amplitude()
        peaks: list[float] = []
        for i in range(1, amplitude.numel() - 1):
            if amplitude[i] > amplitude[i - 1] and amplitude[i] >= amplitude[i + 1]:
                peaks.append(float(self.frequencies_hz[i]))
        return peaks

    def resonance_peaks_hz(self, window: float = 0.05) -> list[float]:
        """Wet resonance frequencies, refined by a parabolic fit.

        For every wet natural frequency the ``|H|`` maximum inside
        ``+/- window * f_wet`` is located and refined with a three-point
        parabolic fit, which removes the frequency-grid bias of
        :meth:`peak_frequencies_hz` and makes the peak position comparable with
        ``wet_frequencies_hz`` at the 1e-3 level.
        """
        amplitude = self.amplitude()
        frequencies = self.frequencies_hz
        peaks: list[float] = []
        for f_wet in self.wet_frequencies_hz:
            band = (frequencies > f_wet * (1.0 - window)) & (
                frequencies < f_wet * (1.0 + window)
            )
            if not bool(band.any()):
                peaks.append(float("nan"))
                continue
            indices = torch.nonzero(band, as_tuple=False).flatten()
            local = amplitude[indices]
            best = int(torch.argmax(local))
            index = int(indices[best])
            if 0 < index < amplitude.numel() - 1:
                y0 = float(amplitude[index - 1])
                y1 = float(amplitude[index])
                y2 = float(amplitude[index + 1])
                denominator = y0 - 2.0 * y1 + y2
                offset = 0.0
                if denominator < -1e-300:  # genuine local maximum
                    offset = max(-0.5, min(0.5, 0.5 * (y0 - y2) / denominator))
                step = float(frequencies[1] - frequencies[0])
                peaks.append(float(frequencies[index]) + offset * step)
            else:  # pragma: no cover - band clipped at the grid ends
                peaks.append(float(frequencies[index]))
        return peaks


def _element_average(
    values: torch.Tensor, elements: torch.Tensor, count: int
) -> torch.Tensor:
    return values[elements].mean(dim=1)


def submerged_plate_frf(
    *,
    length: float = 1.0,
    width: float = 1.0,
    thickness: float = 0.005,
    rho_fluid: float = 1025.0,
    rho_struct: float = 7850.0,
    young: float = 210.0e9,
    poisson: float = 0.3,
    elements_x: int = 12,
    elements_y: int = 12,
    n_modes: int = 9,
    damping_ratio: float = 0.01,
    force_location: tuple[float, float] = (0.5, 0.5),
    response_location: tuple[float, float] = (0.5, 0.5),
    freq_range: tuple[float, float] = (1.0, 200.0),
    n_freq_points: int = 4000,
    dtype: torch.dtype = torch.float64,
) -> FRFResult:
    """Frequency response of a submerged simply supported panel.

    Modal superposition with the *wet* eigenpairs::

        H(f) = sum_k phi_k(x_f) phi_k(x_r) /
               (m_k (omega_k^2 - omega^2 + 2 i zeta omega_k omega))

    * structure: Mindlin Q4 finite elements
      (:func:`tensorfem.plate_vibration.solve_plate_vibration`), mode shapes
      interpolated at the force/response points with the element shape
      functions of the containing element;
    * fluid: panel-discretised Rayleigh added mass
      (:func:`tensorfem.hydro_added_mass.dipole_added_mass_matrix`) projected on
      the mass-normalised mode shapes, ``m_k^{wet} = m_k + phi_k^T M_a phi_k``
      and ``omega_k^{wet} = omega_k / sqrt(m_k^{wet}/m_k)``.

    ``damping_ratio`` is a constant modal loss factor (the fluid model carries no
    radiation damping, see the module scope limits).
    """
    length = _positive("length", length)
    width = _positive("width", width)
    thickness = _positive("thickness", thickness)
    rho_fluid = _positive("rho_fluid", rho_fluid)
    rho_struct = _positive("rho_struct", rho_struct)
    young = _positive("young", young)
    poisson = _poisson(poisson)
    elements_x = _count("elements_x", elements_x)
    elements_y = _count("elements_y", elements_y)
    n_modes = _count("n_modes", n_modes)
    damping_ratio = float(damping_ratio)
    if not (0.0 < damping_ratio < 1.0):
        raise ValueError("damping_ratio must lie in (0, 1)")
    n_freq_points = _count("n_freq_points", n_freq_points)
    f_min, f_max = float(freq_range[0]), float(freq_range[1])
    if not (0.0 < f_min < f_max):
        raise ValueError("freq_range must satisfy 0 < f_min < f_max")

    from .hydro_added_mass import dipole_added_mass_matrix
    from .plate_vibration import (
        q4_plate_mass_matrix,
        solve_plate_vibration,
    )

    result = solve_plate_vibration(
        elements_x,
        elements_y,
        length=length,
        width=width,
        young=young,
        poisson=poisson,
        thickness=thickness,
        density=rho_struct,
        modes=n_modes,
    )
    nodes = result.nodes
    elements = result.elements
    modes = result.modes

    # Consistent mass matrix (re-assembled to obtain the generalized masses).
    ndof = 3 * nodes.shape[0]
    mass = torch.zeros((ndof, ndof), dtype=dtype)
    for connectivity in elements:
        element_mass = q4_plate_mass_matrix(
            nodes[connectivity], rho_struct, thickness
        )
        dofs = torch.stack(
            (3 * connectivity, 3 * connectivity + 1, 3 * connectivity + 2), dim=1
        ).reshape(-1)
        mass[dofs[:, None], dofs] += element_mass

    # Panel fluid operator on element centroids.
    centroids = nodes[elements].mean(dim=1)
    areas = torch.zeros(elements.shape[0], dtype=dtype)
    for i, connectivity in enumerate(elements):
        xy = nodes[connectivity]
        areas[i] = abs(
            (xy[1, 0] - xy[0, 0]) * (xy[3, 1] - xy[0, 1])
            - (xy[3, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        )
    operator = dipole_added_mass_matrix(centroids, areas, rho_fluid)

    transverse = torch.arange(0, ndof, 3)
    generalized_mass: list[float] = []
    added_mass_ratio: list[float] = []
    wet_omega: list[float] = []
    dry_frequencies: list[float] = []
    wet_frequencies: list[float] = []
    for k in range(n_modes):
        phi = modes[:, k]
        m_k = float(phi @ mass @ phi)
        if not math.isfinite(m_k) or m_k <= 0.0:
            raise ValueError("non-positive generalised mass for a mode")
        phi_w = phi[transverse]
        phi_panel = _element_average(phi_w, elements, elements.shape[0])
        added = float(phi_panel @ operator @ phi_panel)
        mu = added / m_k
        omega_dry = 2.0 * math.pi * float(result.frequencies_hz[k])
        omega_wet = omega_dry / math.sqrt(1.0 + mu)
        generalized_mass.append(m_k)
        added_mass_ratio.append(mu)
        wet_omega.append(omega_wet)
        dry_frequencies.append(float(result.frequencies_hz[k]))
        wet_frequencies.append(omega_wet / (2.0 * math.pi))

    # Shape functions of the containing element give phi(x_f), phi(x_r).
    def interpolate(point: tuple[float, float]) -> torch.Tensor:
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= length and 0.0 <= y <= width):
            raise ValueError("force/response point outside the panel")
        values = torch.zeros(n_modes, dtype=dtype)
        for i, connectivity in enumerate(elements):
            xy = nodes[connectivity]
            x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
            y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
            if x0 - 1e-12 <= x <= x1 + 1e-12 and y0 - 1e-12 <= y <= y1 + 1e-12:
                xi = 2.0 * (x - x0) / (x1 - x0) - 1.0
                eta = 2.0 * (y - y0) / (y1 - y0) - 1.0
                shape = torch.tensor(
                    [
                        0.25 * (1 - xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 + eta),
                        0.25 * (1 - xi) * (1 + eta),
                    ],
                    dtype=dtype,
                )
                for k in range(n_modes):
                    values[k] = shape @ (modes[transverse, k][connectivity] / math.sqrt(
                        generalized_mass[k]
                    ))
                return values
        raise ValueError("could not locate the containing element")

    phi_force = interpolate(force_location)
    phi_response = interpolate(response_location)
    participation = tuple(abs(float(phi_force[k] * phi_response[k])) for k in range(n_modes))

    frequencies = torch.linspace(f_min, f_max, n_freq_points, dtype=dtype)
    omega = 2.0 * math.pi * frequencies
    displacement = torch.zeros(n_freq_points, dtype=torch.complex128)
    for k in range(n_modes):
        omega_k = wet_omega[k]
        denominator = (
            omega_k ** 2 - omega ** 2 + 2j * damping_ratio * omega_k * omega
        )
        displacement += (phi_force[k] * phi_response[k]).to(torch.complex128) / denominator

    return FRFResult(
        frequencies_hz=frequencies,
        displacement=displacement,
        dry_frequencies_hz=tuple(dry_frequencies),
        wet_frequencies_hz=tuple(wet_frequencies),
        added_mass_ratios=tuple(added_mass_ratio),
        generalized_masses=tuple(generalized_mass),
        damping_ratio=damping_ratio,
        force_location=(float(force_location[0]), float(force_location[1])),
        response_location=(float(response_location[0]), float(response_location[1])),
        participation=participation,
    )