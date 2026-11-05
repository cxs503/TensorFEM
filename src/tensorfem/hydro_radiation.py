"""Finite-frequency (finite-``ka``) baffled-panel radiation impedance.

Motivation
----------
``tensorfem.hydro_added_mass`` supplies the *in-phase* (``k -> 0``) Rayleigh
operator ``G = 1 / (2 pi r)`` and states its own scope explicitly: "at higher
modes / higher frequency the full ``exp(-j k r) / (2 pi r)`` kernel with the
radiation damping term must be used".  This module is that continuation.

With the harmonic convention ``exp(+j w t)``, a plane surface ``S`` embedded in
an infinite rigid baffle and moving with normal velocity ``u(x)`` radiates the
surface pressure

    p(x) = j w rho  int_S  u(y) G(x - y) dy,        G = exp(-j k r) / (2 pi r),

and the *radiation impedance* seen by the surface is the double integral

    Z = F / u = j w rho  int_S int_S  u(x) u(y) G(x - y) dy dx / u^2.

Its real part is the radiation resistance (the physical origin of added
damping) and ``Z / (j w)`` is the complex added mass; ``Re(Z / (j w))`` reduces
to the in-phase added mass used by ``hydro_added_mass`` as ``k -> 0``.

The point-patch singularity is integrable and is replaced by its exact value on
an equal-area circular patch,

    int_disc(kernel) dA = (1 - exp(-j k a)) / (j k),     a = sqrt(A / pi),

whose ``k -> 0`` limit is ``a`` -- the self term already used by
``hydro_added_mass.dipole_added_mass_matrix``.

Exact reference anchor
----------------------
For a circular piston of radius ``a`` in an infinite rigid baffle the impedance
is known in closed form (Rayleigh 1945 Sec. 302; Kinsler & Frey):

    Z = rho c pi a^2 [ R1(2 k a) + j X1(2 k a) ],
    R1(x) = 1 - 2 J1(x) / x,
    X1(x) = 2 H1(x) / x,          H1 = Struve function of order 1,

with the low-frequency limits ``R1(x) ~ x^2 / 8`` (resistance ``~ O(k^2)``) and
``X1(x) ~ 4 x / (3 pi)`` so that ``Im(Z) / w -> (8/3) rho a^3``, i.e. exactly
the piston added mass anchored in ``hydro_added_mass``.  Both special functions
are implemented here (power series) so that the module keeps the package's pure
``torch`` dependency set; the series are cross-checked against SciPy in
``tests/test_hydro_radiation.py``.

All quantities are SI (m, kg, s).  The impedance sums are accumulated in row
blocks so that a few thousand panels need only ``O(N)`` memory.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "bessel_j1",
    "struve_h1",
    "piston_resistance_factor",
    "piston_reactance_factor",
    "piston_impedance_exact",
    "baffled_kernel_matrix",
    "planar_radiation_impedance",
    "modal_radiation_impedance",
    "effective_added_mass",
]

DEFAULT_RHO_WATER = 1025.0
DEFAULT_SOUND_SPEED = 1500.0

#: Radius of convergence of the Struve H1 power series is the whole plane, but
#: cancellation limits double precision; for |z| beyond this the caller should
#: use the asymptotic form (not needed for ship panel acoustics).
_STRUVE_MAX_ARG = 12.0


def _as_tensor(value, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def _check_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _check_wavenumber(k: float) -> float:
    k = float(k)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("k must be finite and non-negative")
    return k


def bessel_j1(x, *, dtype=torch.float64, max_terms: int = 200) -> torch.Tensor:
    """Bessel function of the first kind, order 1, by its power series.

    ``J1(x) = sum_m (-1)^m (x/2)^(2m+1) / (m! (m+1)!)``.  Accurate to double
    precision over the range of acoustic wavenumbers used here (``|x| <= 30``).
    """
    x = _as_tensor(x, dtype)
    half = 0.5 * x
    term = half.clone()          # m = 0 term: (x/2) / (0! 1!)
    total = term.clone()
    half_sq = half * half
    for m in range(1, max_terms):
        term = term * (-half_sq) / (m * (m + 1))
        total = total + term
        if bool(torch.all(torch.abs(term) <= 1e-17 * (torch.abs(total) + 1e-300))):
            break
    return total


def struve_h1(z, *, dtype=torch.float64, max_terms: int = 300) -> torch.Tensor:
    """Struve function of order 1, ``H1``, by its power series.

    ``H1(z) = sum_m (-1)^m (z/2)^(2m+2) / [Gamma(m+3/2) Gamma(m+5/2)]``
    (Abramowitz & Stegun 12.1.5).  The leading term is ``2 z^2 / (3 pi)``, which
    is what turns ``X1(x) = 2 H1(x) / x`` into ``4x/(3 pi)`` and therefore the
    exact piston added mass ``(8/3) rho a^3``.
    """
    z = _as_tensor(z, dtype)
    if bool(torch.any(torch.abs(z) > _STRUVE_MAX_ARG)):
        raise ValueError(
            f"struve_h1 power series is limited to |z| <= {_STRUVE_MAX_ARG}"
        )
    half = 0.5 * z
    term = half * half / (math.gamma(1.5) * math.gamma(2.5))   # m = 0
    total = term.clone()
    half_sq = half * half
    for m in range(1, max_terms):
        term = term * (-half_sq) / ((m + 0.5) * (m + 1.5))
        total = total + term
        if bool(torch.all(torch.abs(term) <= 1e-17 * (torch.abs(total) + 1e-300))):
            break
    return total


def piston_resistance_factor(x, *, dtype=torch.float64) -> torch.Tensor:
    """Rayleigh resistance factor ``R1(x) = 1 - 2 J1(x) / x`` (``R1(0) = 0``)."""
    x = _as_tensor(x, dtype)
    safe = torch.where(torch.abs(x) < 1e-12, torch.ones_like(x), x)
    factor = 1.0 - 2.0 * bessel_j1(safe, dtype=dtype) / safe
    return torch.where(torch.abs(x) < 1e-12, x * x / 8.0, factor)


def piston_reactance_factor(x, *, dtype=torch.float64) -> torch.Tensor:
    """Rayleigh reactance factor ``X1(x) = 2 H1(x) / x`` (``X1(0) = 0``)."""
    x = _as_tensor(x, dtype)
    safe = torch.where(torch.abs(x) < 1e-12, torch.ones_like(x), x)
    factor = 2.0 * struve_h1(safe, dtype=dtype) / safe
    return torch.where(torch.abs(x) < 1e-12, 4.0 * x / (3.0 * math.pi), factor)


def piston_impedance_exact(
    radius: float,
    k: float,
    rho_water: float = DEFAULT_RHO_WATER,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> complex:
    """Exact radiation impedance of a circular piston in an infinite baffle.

    ``Z = rho c pi a^2 [R1(2ka) + j X1(2ka)]``, valid for all ``k a``.
    Returns a Python ``complex`` in N s/m (mechanical impedance).
    """
    radius = _check_positive("radius", radius)
    rho = _check_positive("rho_water", rho_water)
    c = _check_positive("sound_speed", sound_speed)
    k = _check_wavenumber(k)
    x = 2.0 * k * radius
    area = math.pi * radius * radius
    scale = rho * c * area
    resistance = float(piston_resistance_factor(x))
    reactance = float(piston_reactance_factor(x))
    return complex(scale * resistance, scale * reactance)


def baffled_kernel_matrix(
    centres: torch.Tensor,
    areas: torch.Tensor,
    k: float,
    rho_water: float = DEFAULT_RHO_WATER,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    chunk: int = 256,
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Dense complex radiation operator ``M(k)`` for a planar baffled surface.

    ``M[i, j] = rho A_i A_j exp(-j k r_ij) / (2 pi r_ij)`` for ``i != j`` and the
    exact equal-area patch value

    ``M[i, i] = rho A_i (1 - exp(-j k a_i)) / (j k)``,  ``a_i = sqrt(A_i/pi)``,

    so that ``Z = j w sum(M)`` is the surface radiation impedance.  The matrix is
    complex **symmetric** (not Hermitian): reciprocity holds for the Green
    function, the ``exp(-j k r)`` propagation factor is symmetric in ``i, j``.

    Use :func:`planar_radiation_impedance` when only the double sum is needed --
    it avoids materialising the ``N x N`` matrix.
    """
    centres, areas, rho, _c, k = _validate_inputs(
        centres, areas, k, rho_water, sound_speed, dtype=torch.float64
    )
    n = centres.shape[0]
    chunk = max(1, int(chunk))
    real_dtype = torch.float64
    matrix = torch.zeros((n, n), dtype=torch.complex128)
    scale = rho / (2.0 * math.pi)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = centres[start:stop, None, 0] - centres[None, :, 0]
        dy = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy)
        phase = torch.cos(k * dist) - 1j * torch.sin(k * dist)
        block = (areas[start:stop, None] * areas[None, :]) * phase / dist
        matrix[start:stop, :] = scale * block.to(torch.complex128)
    # Exact self-patch term, replacing the (integrable) 1/r singularity.
    radii = torch.sqrt(areas / math.pi)
    if k == 0.0:
        diag = (rho * areas * radii).to(torch.complex128)
    else:
        jk = 1j * k
        diag = rho * areas * (1.0 - torch.exp(-jk * radii)) / jk
    idx = torch.arange(n)
    matrix[idx, idx] = diag
    _ = real_dtype
    return matrix


def _validate_inputs(centres, areas, k, rho_water, sound_speed, *, dtype):
    centres = _as_tensor(centres, dtype)
    areas = _as_tensor(areas, dtype)
    if centres.ndim != 2 or centres.shape[1] != 2:
        raise ValueError("centres must have shape (N, 2)")
    n = centres.shape[0]
    if areas.shape != (n,):
        raise ValueError("areas must have shape (N,) matching centres")
    if bool((areas <= 0).any()):
        raise ValueError("all panel areas must be positive")
    rho = _check_positive("rho_water", rho_water)
    c = _check_positive("sound_speed", sound_speed)
    k = _check_wavenumber(k)
    return centres, areas, rho, c, k


def planar_radiation_impedance(
    centres: torch.Tensor,
    areas: torch.Tensor,
    k: float,
    rho_water: float = DEFAULT_RHO_WATER,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    chunk: int = 512,
) -> complex:
    """Radiation impedance ``Z = j w sum_ij M_ij(k)`` of a baffled plane surface.

    Uniform normal velocity (piston-like) motion.  Memory is ``O(N)``: only row
    blocks of the influence matrix are ever held, so a few thousand panels fit
    comfortably.
    """
    centres, areas, rho, c, k = _validate_inputs(
        centres, areas, k, rho_water, sound_speed, dtype=torch.float64
    )
    n = centres.shape[0]
    chunk = max(1, int(chunk))
    scale = rho / (2.0 * math.pi)
    total = 0.0 + 0.0j
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = centres[start:stop, None, 0] - centres[None, :, 0]
        dy = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy)
        phase = torch.cos(k * dist) - 1j * torch.sin(k * dist)
        weighted = (areas[start:stop, None] * areas[None, :]) * phase / dist
        # Remove the diagonal before summing: replaced by the exact patch value.
        rows = torch.arange(stop - start)
        weighted[rows, rows + start] = 0.0
        block_sum = complex(weighted.sum().to(torch.complex128).item())
        total += scale * block_sum
    radii = torch.sqrt(areas / math.pi)
    if k == 0.0:
        self_terms = rho * areas * radii
    else:
        jk = 1j * k
        self_terms = rho * areas * (1.0 - torch.exp(-jk * radii)) / jk
    total += complex(self_terms.sum().to(torch.complex128).item())
    omega = k * c
    return 1j * omega * total


def modal_radiation_impedance(
    centres: torch.Tensor,
    areas: torch.Tensor,
    values: torch.Tensor,
    k: float,
    rho_water: float = DEFAULT_RHO_WATER,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    chunk: int = 512,
) -> complex:
    """Radiation impedance of a prescribed normal-velocity distribution.

    ``Z = j w rho sum_ij phi_i A_i phi_j A_j exp(-j k r_ij) / (2 pi r_ij)``
    with ``phi`` the velocity shape values.  ``phi = 1`` recovers
    :func:`planar_radiation_impedance` (piston); a panel mode is handled by
    ``hydro_added_mass.sine_mode_values``.  ``Re(Z)`` is the modal radiation
    resistance, i.e. the added-damping term that the in-phase operator omits.
    """
    centres, areas, rho, c, k = _validate_inputs(
        centres, areas, k, rho_water, sound_speed, dtype=torch.float64
    )
    values = _as_tensor(values, dtype=torch.float64)
    n = centres.shape[0]
    if values.shape != (n,):
        raise ValueError("values must have shape (N,) matching centres")
    chunk = max(1, int(chunk))
    scale = rho / (2.0 * math.pi)
    weighted_area = areas * values
    total = 0.0 + 0.0j
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dx = centres[start:stop, None, 0] - centres[None, :, 0]
        dy = centres[start:stop, None, 1] - centres[None, :, 1]
        dist = torch.sqrt(dx * dx + dy * dy)
        phase = torch.cos(k * dist) - 1j * torch.sin(k * dist)
        block = (weighted_area[start:stop, None] * weighted_area[None, :]) * phase / dist
        rows = torch.arange(stop - start)
        block[rows, rows + start] = 0.0
        total += scale * complex(block.sum().to(torch.complex128).item())
    radii = torch.sqrt(areas / math.pi)
    if k == 0.0:
        self_terms = rho * weighted_area * weighted_area * radii / areas
    else:
        jk = 1j * k
        self_terms = (
            rho * weighted_area * weighted_area / areas
            * (1.0 - torch.exp(-jk * radii)) / jk
        )
    total += complex(self_terms.sum().to(torch.complex128).item())
    omega = k * c
    return 1j * omega * total


def effective_added_mass(impedance: complex, omega: float) -> float:
    """In-phase added mass implied by an impedance: ``Re(Z / (j w))``.

    As ``k -> 0`` this tends to the low-frequency added mass that
    ``hydro_added_mass`` computes directly, so the two modules can be cross
    checked on the same geometry.
    """
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be positive and finite")
    return float((impedance / (1j * omega)).real)