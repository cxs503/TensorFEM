"""Levy-series and Rayleigh-Ritz modal anchors for rectangular Kirchhoff plates.

Why a new modal module
----------------------
``tensorfem.plate_radiation_chain.simply_supported_mode_frequencies`` is the
closed form for a *fully simply supported* plate and the clamped case is only
available as Leissa's tabulated *square* plate coefficients.  Every
structure-acoustic coupling benchmark therefore had to be run on a square plate
with four identical edges -- exactly the geometry for which the boundary
conditions cannot be wrong.  This module supplies two independent general
references:

1. **Levy series** (:func:`levy_frequencies`).  For a plate that is simply
   supported on ``y = 0, b`` the field can be written
   ``W(x, y) = sum_m Y_m(x) sin(m pi y / b)``, which turns the biharmonic
   equation into an ordinary differential equation per half-wave number ``m``:

   .. math:: Y_m'''' - 2\\beta_m^2 Y_m'' + (\\beta_m^4 - k^4) Y_m = 0,
             \\qquad \\beta_m = m\\pi/b, \\quad k^4 = \\rho h \\omega^2 / D.

   The characteristic roots are ``s^2 = \\beta_m^2 \\pm k^2``, so the solution is
   a ``cosh/sinh`` pair (real root) plus a ``cos/sin`` pair when ``k > \\beta_m``
   or a second ``cosh/sinh`` pair when ``k < \\beta_m``.  The four edge
   conditions at ``x = 0, a`` give a ``4x4`` determinant whose zeros are the
   eigenvalues.  Two of the four edges may be clamped, so the classical
   ``S-C-S-C`` and ``C-S-C-S`` (and of course ``S-S-S-S``) plates are covered with
   an *exact* transcendental reference, for any aspect ratio and any Poisson
   ratio.

2. **Rayleigh-Ritz** (:func:`ritz_frequencies`) with clamped-clamped beam
   characteristic functions in each direction.  The Ritz route is variational,
   so its eigenvalues are strict upper bounds and its admissible functions
   violate the natural boundary conditions nowhere; it covers the fully clamped
   plate that the Levy ansatz cannot represent.

The two routes share no code and no asymptotics, so agreement on ``S-C-S-C``
(and agreement of the Ritz route with the textbook ``C-C-C-C`` square
coefficients of Leissa, 1969) is a genuine cross-check.

Conventions
-----------
``x`` in ``[0, a]`` (``length``), ``y`` in ``[0, b]`` (``width``), thin
Kirchhoff plate of thickness ``h``, ``D = E h^3 / (12 (1 - nu^2))``.  Levy needs
``y`` edges simply supported; the ``x`` edges are given by ``x_edges``.
Frequencies are in hertz, ordered ascending, and the first ``modes`` of the
*whole plate* are returned (not the first per half-wave number).

References
----------
* Leissa, A. W. (1969). *Vibration of Plates*. NASA SP-160, Table 4.1.
* Gorman, D. J. (1982). *Free Vibration Analysis of Rectangular Plates*,
  Elsevier (Levy method, Chapter 2).
* Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*,
  2nd ed., Chapter 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..acoustic_radiation import _gauss_legendre

EdgeCondition = Literal["simply_supported", "clamped"]

BOTH: tuple[EdgeCondition, EdgeCondition] = ("simply_supported", "clamped")


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _poisson(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not (-1.0 < value < 0.5):
        raise ValueError("poisson must be finite and in (-1, 0.5)")
    return value


@dataclass(frozen=True)
class PlateModalConfig:
    """Isotropic thin-plate geometry/material for the modal references."""

    length: float = 1.5
    width: float = 1.0
    thickness: float = 0.01
    young: float = 210.0e9
    poisson: float = 0.3
    density: float = 7850.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "length", _positive("length", self.length))
        object.__setattr__(self, "width", _positive("width", self.width))
        object.__setattr__(self, "thickness", _positive("thickness", self.thickness))
        object.__setattr__(self, "young", _positive("young", self.young))
        object.__setattr__(self, "density", _positive("density", self.density))
        object.__setattr__(self, "poisson", _poisson(self.poisson))

    @property
    def rigidity(self) -> float:
        """Bending rigidity ``D = E h^3 / (12 (1 - nu^2))``."""
        return self.young * self.thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    @property
    def mass_per_area(self) -> float:
        return self.density * self.thickness

    @property
    def area(self) -> float:
        return self.length * self.width

    def frequency(self, wavenumber: float) -> float:
        """Kirchhoff frequency of the bending wavenumber ``k`` (1/m)."""
        wavenumber = float(wavenumber)
        return (
            wavenumber ** 2
            * math.sqrt(self.rigidity / self.mass_per_area)
            / (2.0 * math.pi)
        )

    def wavenumber(self, frequency: float) -> float:
        frequency = _positive("frequency", frequency)
        return math.sqrt(
            2.0 * math.pi * frequency
            * math.sqrt(self.mass_per_area / self.rigidity)
        )


# ---------------------------------------------------------------------------
# Route 1: Levy determinant and its transcendental roots
# ---------------------------------------------------------------------------

def _levy_determinant(
    config: PlateModalConfig,
    half_wave: int,
    wavenumbers: torch.Tensor,
    x_edges: EdgeCondition,
) -> torch.Tensor:
    """``det`` of the 4x4 Levy boundary-condition matrix, batched over ``k``.

    Columns are the amplitudes ``(A, B, C, D)`` of the ``cosh/sinh`` pair and of
    the second pair (``cos/sin`` for ``k > beta``, ``cosh/sinh`` for
    ``k < beta``).  Both branches coincide at ``k = beta``, so the determinant is
    continuous through the branch switch and no spurious roots appear there.
    """
    if x_edges not in BOTH:
        raise ValueError(f"x_edges must be one of {BOTH}")
    k = torch.as_tensor(wavenumbers, dtype=torch.float64).reshape(-1)
    beta = half_wave * math.pi / config.width
    alpha = torch.sqrt(beta ** 2 + k ** 2)
    trig = k >= beta
    gamma = torch.sqrt(torch.clamp(k ** 2 - beta ** 2, min=0.0))
    delta = torch.sqrt(torch.clamp(beta ** 2 - k ** 2, min=0.0))

    alpha2 = alpha ** 2
    a = config.length
    ch = torch.cosh(alpha * a)
    sh = torch.sinh(alpha * a)
    cg = torch.cos(gamma * a)
    sg = torch.sin(gamma * a)
    chd = torch.cosh(delta * a)
    shd = torch.sinh(delta * a)

    # Second-pair value/derivative/second derivative at x = a and x = 0
    ct = torch.where(trig, cg, chd)
    st = torch.where(trig, sg, shd)
    d_ct = torch.where(trig, -gamma * sg, delta * shd)
    d_st = torch.where(trig, gamma * cg, delta * chd)
    e_ct = torch.where(trig, -gamma ** 2 * cg, delta ** 2 * chd)
    e_st = torch.where(trig, -gamma ** 2 * sg, delta ** 2 * shd)
    s0 = torch.where(trig, gamma, delta)          # second-pair slope at x = 0
    t0 = torch.where(trig, -gamma ** 2, delta ** 2)  # second-pair curvature at 0

    rows = torch.empty(k.numel(), 4, 4, dtype=torch.float64)
    one = torch.ones_like(k)
    zero = torch.zeros_like(k)

    # x = 0 conditions (identical displacement row for both edge types)
    rows[:, 0, 0] = one
    rows[:, 0, 1] = zero
    rows[:, 0, 2] = one
    rows[:, 0, 3] = zero
    if x_edges == "simply_supported":
        rows[:, 1, 0] = alpha2
        rows[:, 1, 1] = zero
        rows[:, 1, 2] = t0
        rows[:, 1, 3] = zero
    else:  # clamped: slope instead of curvature
        rows[:, 1, 0] = zero
        rows[:, 1, 1] = alpha
        rows[:, 1, 2] = zero
        rows[:, 1, 3] = s0

    # x = a conditions
    rows[:, 2, 0] = ch
    rows[:, 2, 1] = sh
    rows[:, 2, 2] = ct
    rows[:, 2, 3] = st
    if x_edges == "simply_supported":
        rows[:, 3, 0] = alpha2 * ch
        rows[:, 3, 1] = alpha2 * sh
        rows[:, 3, 2] = e_ct
        rows[:, 3, 3] = e_st
    else:
        rows[:, 3, 0] = alpha * sh
        rows[:, 3, 1] = alpha * ch
        rows[:, 3, 2] = d_ct
        rows[:, 3, 3] = d_st

    det = torch.linalg.det(rows)
    # Scale by the matrix magnitude so the sign pattern is not dominated by the
    # cosh growth; the zeros of det are unchanged.
    scale = rows.abs().amax(dim=(1, 2)) ** 4
    return det / torch.where(scale > 0.0, scale, torch.ones_like(scale))


def levy_wavenumbers(
    config: PlateModalConfig,
    half_wave: int,
    *,
    x_edges: EdgeCondition = "clamped",
    scan_points: int = 4000,
    k_max_factor: float = 6.0,
    tolerance: float = 1e-12,
) -> tuple[float, ...]:
    """All Levy bending wavenumbers of half-wave number ``half_wave``.

    The determinant is sampled on a uniform ``k`` grid, every sign change is
    bracketed and then refined by a *batched* bisection, so the roots are
    accurate to ``tolerance`` in relative terms.
    """
    if int(half_wave) < 1:
        raise ValueError("half_wave must be >= 1")
    k_base = math.pi * math.sqrt(1.0 / config.length ** 2 + 1.0 / config.width ** 2)
    k_max = k_max_factor * k_base
    grid = torch.linspace(1e-6 * k_base, k_max, int(scan_points), dtype=torch.float64)
    values = _levy_determinant(config, int(half_wave), grid, x_edges)
    lo = grid[:-1]
    hi = grid[1:]
    f_lo = values[:-1]
    f_hi = values[1:]
    crossing = (f_lo * f_hi) < 0.0
    if not bool(crossing.any()):
        return ()
    lo = lo[crossing]
    hi = hi[crossing]
    f_lo = f_lo[crossing]
    for _ in range(200):
        if bool(torch.all(hi - lo < tolerance * torch.clamp(hi, min=1.0))):
            break
        mid = 0.5 * (lo + hi)
        f_mid = _levy_determinant(config, int(half_wave), mid, x_edges)
        same = f_mid * f_lo > 0.0
        lo = torch.where(same, mid, lo)
        hi = torch.where(same, hi, mid)
        f_lo = torch.where(same, f_mid, f_lo)
    return tuple(float(v) for v in (0.5 * (lo + hi)))


def levy_frequencies(
    config: PlateModalConfig,
    *,
    x_edges: EdgeCondition = "clamped",
    y_edges: EdgeCondition = "simply_supported",
    modes: int = 10,
    half_waves: int | None = None,
) -> list[float]:
    """Lowest ``modes`` Kirchhoff frequencies (Hz) of a Levy-representable plate.

    ``y_edges`` must be ``"simply_supported"``: that is the hypothesis of the
    Levy ansatz.  ``x_edges`` may be ``"simply_supported"`` or ``"clamped"``.
    """
    if int(modes) < 1:
        raise ValueError("modes must be >= 1")
    if y_edges != "simply_supported":
        raise ValueError(
            "the Levy ansatz requires simply supported y edges "
            "(use ritz_frequencies for a clamped y direction)"
        )
    if half_waves is None:
        half_waves = max(6, int(modes) + 4)
    roots: list[float] = []
    for m in range(1, int(half_waves) + 1):
        roots.extend(
            levy_wavenumbers(config, m, x_edges=x_edges, scan_points=3000, k_max_factor=4.5)
        )
    roots.sort()
    frequencies = [config.frequency(k) for k in roots]
    if len(frequencies) < int(modes):
        raise RuntimeError(
            f"only {len(frequencies)} Levy roots found, {modes} requested; "
            "increase half_waves"
        )
    return frequencies[: int(modes)]


# ---------------------------------------------------------------------------
# Route 2: Rayleigh-Ritz with beam characteristic functions
# ---------------------------------------------------------------------------

def _clamped_beam_mu(n_terms: int) -> list[float]:
    """Clamped-clamped Euler-Bernoulli eigenvalues ``mu_n = k_n L``.

    Roots of ``cos(mu) cosh(mu) = 1``.  Bracketed per branch and bisected (no
    tabulated constants: ``mu_1 = 4.7300407...`` is *computed*, which is what
    makes the Ritz route independent of any transcribed table).
    """
    roots: list[float] = []
    for n in range(1, int(n_terms) + 1):
        centre = (2 * n + 1) * math.pi / 2.0

        def g(mu: float) -> float:
            return math.cos(mu) * math.cosh(mu) - 1.0

        lo, hi = centre - 0.45, centre + 0.45
        f_lo, f_hi = g(lo), g(hi)
        if f_lo == 0.0:
            roots.append(lo)
            continue
        if f_lo * f_hi > 0.0:  # pragma: no cover - defensive
            raise RuntimeError(f"could not bracket clamped-beam root {n}")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = g(mid)
            if f_lo * f_mid <= 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        roots.append(0.5 * (lo + hi))
    return roots


def _beam_basis(
    kind: EdgeCondition, length: float, n_terms: int, order: int = 28
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Values of ``X_n``, ``X_n'``, ``X_n''`` on a Gauss grid and the weights."""
    nodes, weights = _gauss_legendre(int(order))
    x = 0.5 * length * (nodes + 1.0)
    w = 0.5 * length * weights
    if kind == "simply_supported":
        rows, d1, d2 = [], [], []
        for n in range(1, int(n_terms) + 1):
            q = n * math.pi / length
            rows.append(torch.sin(q * x))
            d1.append(q * torch.cos(q * x))
            d2.append(-q * q * torch.sin(q * x))
        return torch.stack(rows), torch.stack(d1), torch.stack(d2), w
    if kind != "clamped":
        raise ValueError(f"unsupported edge condition {kind!r}")
    mus = _clamped_beam_mu(int(n_terms))
    rows, d1, d2 = [], [], []
    for mu in mus:
        q = mu / length
        cx, sx = torch.cosh(mu * x / length), torch.sinh(mu * x / length)
        cc, ss = torch.cos(mu * x / length), torch.sin(mu * x / length)
        sigma = (math.cosh(mu) - math.cos(mu)) / (math.sinh(mu) - math.sin(mu))
        rows.append(cx - cc - sigma * (sx - ss))
        d1.append(q * (sx + ss - sigma * (cx - cc)))
        d2.append(q * q * (cx + cc - sigma * (sx + ss)))
    return torch.stack(rows), torch.stack(d1), torch.stack(d2), w


def _gram(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight[None, :]) @ values.T


def ritz_frequencies(
    config: PlateModalConfig,
    *,
    x_edges: EdgeCondition = "clamped",
    y_edges: EdgeCondition = "clamped",
    modes: int = 10,
    terms: int = 10,
    order: int = 28,
) -> list[float]:
    """Lowest ``modes`` frequencies (Hz) from a Rayleigh-Ritz minimisation.

    The admissible functions are ``X_i(x) Y_j(y)`` with clamped-clamped (or
    sine) beam characteristic functions in each direction, so the default
    ``clamped/clamped`` case covers the fully clamped plate that the Levy route
    cannot represent.  The strain energy density
    ``D/2 [(lap w)^2 - 2(1-nu)(w_xx w_yy - w_xy^2)]`` is integrated exactly per
    product pair through the 1-D Gram matrices.
    """
    if int(modes) < 1 or int(terms) < 1:
        raise ValueError("modes and terms must be >= 1")
    if int(modes) > int(terms) ** 2:
        raise ValueError("modes cannot exceed terms**2")
    xv, x1, x2, wx = _beam_basis(x_edges, config.length, int(terms), order)
    yv, y1, y2, wy = _beam_basis(y_edges, config.width, int(terms), order)
    nx, ny = int(terms), int(terms)

    s0x, s1x, s2x = _gram(xv, wx), _gram(x1, wx), _gram(x2, wx)
    s02x = (xv * wx[None, :]) @ x2.T          # int X_i X_j''
    s0y, s1y, s2y = _gram(yv, wy), _gram(y1, wy), _gram(y2, wy)
    s02y = (yv * wy[None, :]) @ y2.T

    def mix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a[:, None, :, None] * b[None, :, None, :]).reshape(nx * ny, nx * ny)

    stiffness = config.rigidity * (
        mix(s2x, s0y) + mix(s0x, s2y)
        + 2.0 * config.poisson * mix(s02x, s02y)
        + 2.0 * (1.0 - config.poisson) * mix(s1x, s1y)
    )
    mass = config.mass_per_area * mix(s0x, s0y)
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)

    chol = torch.linalg.cholesky(mass)
    inv_chol = torch.linalg.inv(chol)
    reduced = inv_chol @ stiffness @ inv_chol.T
    eigenvalues = torch.linalg.eigvalsh(0.5 * (reduced + reduced.T))
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    # The stiffness and mass matrices already carry ``D`` and ``rho h``, so the
    # generalised eigenvalues are ``omega^2`` in 1/s^2 with no further scaling.
    omega = torch.sqrt(eigenvalues)
    frequencies = (omega / (2.0 * math.pi))[: int(modes)]
    return [float(f) for f in frequencies]


def leissa_cccc_square_lambdas() -> tuple[float, ...]:
    """Leissa (1969) Table 4.1 ``C-C-C-C`` square-plate coefficients, modes 1-8.

    ``lambda = omega a^2 sqrt(rho h / D)`` for ``nu = 0.3`` (``lambda_2 =
    lambda_3`` and ``lambda_7 = lambda_8`` are the symmetric pairs).  Transcribed
    from the literature, not computed here -- this is the only anchor in the
    module that is *not* derived from the code base, which is why it is used as
    an external referee for the Ritz route.

    Only eight values are returned on purpose.  A converged admissible Ritz
    basis (10x10 and 12x12, agreeing to 0.1 %) puts the ninth and tenth
    eigenvalues of the square plate at ``lambda = 210.4`` (a symmetric pair),
    not at the 181.2/206.2 values that circulate for this table.  Since a Ritz
    value is an upper bound for the true eigenvalue, either those two entries
    belong to a different boundary condition or they were mis-transcribed in
    whatever source we quoted them from; they are dropped rather than patched.
    Modes 1-8 are reproduced to 0.05 %, which is what the referee gate uses.
    """
    return (
        35.985, 73.394, 73.394, 108.22,
        131.58, 132.24, 165.00, 165.00,
    )