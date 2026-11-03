"""Hetényi analytical reference solutions for beams/plates on Winkler foundation.

Reference
---------
M. Hetényi, *Beams on Elastic Foundation*, Univ. of Michigan Press, 1946.
For the plate solution: Timoshenko & Woinowsky-Krieger, *Theory of Plates and
Shells*, 2nd ed., §62 (plate on elastic foundation, Kelvin functions).

For an *infinite* beam on a Winkler foundation the governing equation is

    EI w'''' + k_b w = q

where *k_b* is the foundation coefficient per unit length (N/m²).  Defining

    λ = (k_b / 4EI)^(1/4)          (characteristic wavenumber, 1/m)
    L_c = 1/λ = (4EI / k_b)^(1/4)  (characteristic length, m)

the four Hetényi shape functions are

    A(ξ) = exp(-ξ)·(cos ξ + sin ξ)
    B(ξ) = exp(-ξ)·(cos ξ - sin ξ)
    C(ξ) = exp(-ξ)·cos ξ
    D(ξ) = exp(-ξ)·sin ξ

For a concentrated force *P* at the origin on an infinite beam:

    w(x) = P·λ / (2·k_b) · A(λ·|x|)
    M(x) = P / (4·λ) · B(λ·|x|)

For *n* concentrated loads {P_i at x_i} the solution follows by superposition.

For a *uniformly distributed* load of intensity *q₀* over [a, b], integrating
the point-load Green's function exactly gives (with C(ξ) = exp(-ξ)·cos ξ)

    w(x) = q₀/(2·k_b) · [G(x-a) - G(x-b)],   G(d) = sgn(d)·(1 - C(λ·|d|)).

For a *finite, simply supported* beam the Fourier-sine series is exact:

    w(x) = Σ_n [q_n / (EI·(nπ/L)⁴ + k_b)]·sin(nπx/L).

For an *infinite plate* on a Winkler foundation under an axisymmetric point load
the exact Kelvin-function solution is provided by
:func:`infinite_plate_point_load`.

All reference solutions in this module are pinned by independent numerical
oracles (Green's-function quadrature, Hankel quadrature, finite-difference beam
solve) in ``tests/test_foundation_reference_audit.py``; the audit that produced
those oracles is written up in ``docs/hetanyi-reference-audit.md``.
"""
from __future__ import annotations

import math
import torch


# ---------------------------------------------------------------------------
# Hetényi shape functions
# ---------------------------------------------------------------------------

def hetenyi_A(xi: torch.Tensor) -> torch.Tensor:
    """A(ξ) = exp(-ξ)·(cos ξ + sin ξ)"""
    return torch.exp(-xi) * (torch.cos(xi) + torch.sin(xi))


def hetenyi_B(xi: torch.Tensor) -> torch.Tensor:
    """B(ξ) = exp(-ξ)·(cos ξ - sin ξ)"""
    return torch.exp(-xi) * (torch.cos(xi) - torch.sin(xi))


def hetenyi_C(xi: torch.Tensor) -> torch.Tensor:
    """C(ξ) = exp(-ξ)·cos ξ"""
    return torch.exp(-xi) * torch.cos(xi)


def hetenyi_D(xi: torch.Tensor) -> torch.Tensor:
    """D(ξ) = exp(-ξ)·sin ξ"""
    return torch.exp(-xi) * torch.sin(xi)


# ---------------------------------------------------------------------------
# Infinite-beam solutions
# ---------------------------------------------------------------------------

def infinite_beam_point_load(
    x_eval: torch.Tensor,
    x_load: float,
    P: float,
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Deflection of an infinite beam under one concentrated load.

    Parameters
    ----------
    x_eval : (N,) tensor of evaluation positions.
    x_load : position of the point load.
    P : magnitude of the point load (positive = downward).
    lam : λ = (k_b / 4EI)^(1/4).
    k_b : foundation coefficient per unit length (N/m²).

    Returns
    -------
    w : (N,) tensor of deflections.
    """
    dtype = x_eval.dtype
    dx = torch.abs(x_eval - x_load)
    xi = lam * dx
    return torch.as_tensor(P * lam / (2.0 * k_b), dtype=dtype) * hetenyi_A(xi)


def infinite_beam_multiple_point_loads(
    x_eval: torch.Tensor,
    loads: list[tuple[float, float]],
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Superposition of point loads on an infinite beam.

    Parameters
    ----------
    loads : list of (x_position, P_magnitude) tuples.
    """
    w = torch.zeros_like(x_eval)
    for x_p, P in loads:
        w = w + infinite_beam_point_load(x_eval, x_p, P, lam, k_b)
    return w


def infinite_beam_uniform_load(
    x_eval: torch.Tensor,
    q0: float,
    x_start: float,
    x_end: float,
    lam: float,
    k_b: float,
) -> torch.Tensor:
    """Deflection under a uniformly distributed load q0 over [x_start, x_end].

    The closed form is the exact integral of the point-load Green's function
    ``P·λ/(2k_b)·A(λ|x-s|)`` over the loaded interval.  With ``C(ξ)=exp(-ξ)cos ξ``,

        ∫_a^b A(λ|x-s|) ds = (1/λ)·[G(x-a) - G(x-b)],   G(d) = sgn(d)·(1 - C(λ|d|))

    hence

        w(x) = q0/(2·k_b) · [G(x-a) - G(x-b)].

    The result decays to zero in both far fields (the net load is finite) and
    satisfies ``EI w'''' + k_b w = q(x)`` pointwise with ``q(x)=q0`` inside
    ``[x_start, x_end]`` and zero outside.  Both properties are pinned by
    ``tests/test_foundation_reference_audit.py``.

    Notes
    -----
    An earlier revision returned
    ``(q0/k_b)·[1 - ½(A(λ|x-a|)sgn(x-a) - A(λ|x-b|)sgn(x-b))]``.  That expression
    is *not* the Green's-function integral: it leaves an O(q0) residual in the
    governing ODE and tends to ``q0/k_b`` instead of zero in the far field
    (relative error above 40x at four characteristic lengths).  See
    ``docs/hetanyi-reference-audit.md``.
    """
    dtype = x_eval.dtype

    def _G(d: torch.Tensor) -> torch.Tensor:
        return torch.sign(d) * (1.0 - hetenyi_C(lam * torch.abs(d)))

    delta_a = x_eval - torch.as_tensor(x_start, dtype=dtype)
    delta_b = x_eval - torch.as_tensor(x_end, dtype=dtype)
    return torch.as_tensor(q0 / (2.0 * k_b), dtype=dtype) * (_G(delta_a) - _G(delta_b))


# ---------------------------------------------------------------------------
# Kelvin functions (exact infinite plate on a Winkler foundation)
# ---------------------------------------------------------------------------

#: Argument above which the ascending series is replaced by the asymptotic form.
#: At the switch both branches are accurate to about 1e-7 relative (the series
#: loses digits as exp(Re z) grows, the asymptotic series truncation error falls
#: as 1/x^6), which keeps the function continuous across the branch.
_KEI_ASYMPTOTIC_SWITCH = 14.0

#: Coefficients a_k of the K0(z) ~ sqrt(pi/2z) e^-z sum_k a_k/z^k expansion.
_K0_ASYMPTOTIC_COEFFS = (
    1.0,
    -1.0 / 8.0,
    9.0 / (2.0 * 64.0),
    -225.0 / (6.0 * 512.0),
    11025.0 / (24.0 * 4096.0),
    -893025.0 / (120.0 * 32768.0),
)


def _kelvin_k0_series(z: torch.Tensor, n_terms: int = 160) -> torch.Tensor:
    """Complex K0(z) from the ascending series (|arg z| < pi).

        K0(z) = -(ln(z/2) + gamma)*I0(z) + sum_{k>=1} (z^2/4)^k/(k!)^2 * H_k
        I0(z) = sum_{k>=0} (z^2/4)^k/(k!)^2

    ``n_terms=160`` keeps the float64 result accurate to about 1e-7 for
    ``|z| <= 14``; beyond that the cancellation between I0 and the sum grows
    like ``exp(|Re z|)`` and :func:`kelvin_kei` switches to the asymptotic form.
    """
    q = z * z / 4.0
    term = torch.ones_like(q)
    i0 = torch.ones_like(q)
    acc = torch.zeros_like(q)
    harmonic = 0.0
    for k in range(1, n_terms + 1):
        term = term * q / (k * k)
        harmonic += 1.0 / k
        i0 = i0 + term
        acc = acc + term * harmonic
    gamma = 0.5772156649015329
    return -(torch.log(z / 2.0) + gamma) * i0 + acc


def kelvin_kei(x: torch.Tensor) -> torch.Tensor:
    """Kelvin function ``kei(x) = Im K0(x*exp(i*pi/4))`` for ``x >= 0``.

    ``kei(0) = -pi/4``; the function decays like ``exp(-x/sqrt2)`` with an
    oscillating, sign-changing tail.  Small/medium arguments use the ascending
    series of K0, large arguments the six-term asymptotic expansion

        kei(x) ~ -sqrt(pi/(2x))*exp(-x/sqrt2)*[cos/sin combination],

    evaluated exactly via the complex expansion rather than a hand-fitted phase.
    The function is accurate to about 1e-7 relative over the whole range and is
    cross-checked against an independent Hankel quadrature of the plate problem
    in ``tests/test_foundation_reference_audit.py``.
    """
    dtype = x.dtype
    xd = x.to(torch.float64)
    out = torch.empty_like(xd)
    zero = xd == 0.0
    small = (xd <= _KEI_ASYMPTOTIC_SWITCH) & (~zero)
    if bool(zero.any()):
        out[zero] = -math.pi / 4.0
    if bool(small.any()):
        xs = xd[small]
        z = xs.to(torch.complex128) * complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
        out[small] = torch.imag(_kelvin_k0_series(z))
    large = (~small) & (~zero)
    if bool(large.any()):
        xl = xd[large]
        z = xl.to(torch.complex128) * complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
        series = torch.zeros_like(z)
        for k, coeff in enumerate(_K0_ASYMPTOTIC_COEFFS):
            series = series + coeff / z ** k
        asym = torch.sqrt(math.pi / (2.0 * z)) * torch.exp(-z) * series
        out[large] = torch.imag(asym)
    return out.to(dtype)


def infinite_plate_point_load(
    r: torch.Tensor,
    P: float,
    D: float,
    k: float,
) -> torch.Tensor:
    """Exact deflection of an infinite Kirchhoff plate on a Winkler foundation.

    Solves ``D·∇⁴w + k·w = P·δ(r)/(2πr)`` in closed form.  With the
    characteristic length ``l = (D/k)^(1/4)``,

        w(r) = -(P/(2·π·k·l²))·kei(r/l),

    so the centre deflection is ``w(0) = P/(8·√(D·k))``.  Unlike the exponential
    fit previously used inside ``plate_mindlin.hetenyi_reference``, this solution
    changes sign and has the (physical) shallow uplift ring around the load.

    Parameters
    ----------
    r : (N,) tensor of radial distances from the load.
    P : point load (positive = downward).
    D : plate flexural rigidity per unit width, ``E·h³/(12(1-ν²))``.
    k : Winkler foundation modulus (pressure per unit deflection).
    """
    l = (D / k) ** 0.25
    xi = r / torch.as_tensor(l, dtype=r.dtype)
    return -(P / (2.0 * math.pi * k * l * l)) * kelvin_kei(xi)


# ---------------------------------------------------------------------------
# Finite-beam series solution (simply supported, exact eigenfunction expansion)
# ---------------------------------------------------------------------------

def finite_beam_series(
    x_eval: torch.Tensor,
    beam_length: float,
    lam: float,
    k_b: float,
    EI: float,
    loads: list[tuple[float, float]],
    boundary: str = 'simply_supported',
    n_terms: int = 50,
) -> torch.Tensor:
    """Truncated-series solution for a finite, simply supported beam on Winkler foundation.

    Uses a Fourier-sine-series expansion.  For a beam of length L on elastic
    foundation, the deflection under arbitrary loading can be expressed as

        w(x) = Σ_n [ q_n / (EI·(nπ/L)⁴ + k_b) ] · sin(nπx/L)

    where q_n are the Fourier-sine coefficients of the load distribution.

    For concentrated loads P_i at x_i:

        q_n = (2/L) · Σ_i P_i · sin(nπ·x_i/L)

    The sine basis satisfies ``w = w'' = 0`` at both ends, so this expansion is
    the *exact* solution for simply supported ends (truncated at ``n_terms``).
    It is **not** a free-end solution: ``boundary='free'`` used to be accepted
    silently and returned the simply supported result.  Free ends are now
    rejected instead of being mislabelled; for a beam long compared with
    ``1/λ`` use :func:`infinite_beam_point_load`, whose end effects are
    exponentially small.  A genuine free-end Hetényi end-conditioning solution
    remains an open gap (see ``docs/hetanyi-reference-audit.md``).

    Parameters
    ----------
    beam_length : total beam length L.
    boundary : only ``'simply_supported'`` (aliases ``'simply-supported'``,
        ``'ss'``, ``'pinned'``) is implemented.
    n_terms : number of Fourier terms (truncation).

    Returns
    -------
    w : (N,) tensor of deflections.
    """
    if boundary not in ('simply_supported', 'simply-supported', 'ss', 'pinned'):
        raise ValueError(
            "finite_beam_series only implements simply supported ends, got "
            f"boundary={boundary!r}.  Free-end Hetényi end-conditioning is not implemented; "
            "use infinite_beam_point_load / infinite_beam_uniform_load for beams that are long "
            "compared with the characteristic length 1/lambda."
        )
    L = beam_length
    dtype = x_eval.dtype
    w = torch.zeros_like(x_eval)
    for n in range(1, n_terms + 1):
        npi = n * math.pi
        # Fourier coefficient of load
        qn = torch.tensor(0.0, dtype=dtype)
        for x_p, P in loads:
            qn = qn + P * torch.sin(npi * torch.as_tensor(x_p, dtype=dtype) / L)
        qn = 2.0 / L * qn
        # Modal stiffness (lam is accepted for signature stability; the modal
        # stiffness EI·(nπ/L)⁴ + k_b already encodes the foundation)
        k_n = EI * (npi / L) ** 4 + k_b
        w = w + (qn / k_n) * torch.sin(npi * x_eval / L)
    return w


# ---------------------------------------------------------------------------
# Plate-strip reduction (wide plate → beam equivalence)
# ---------------------------------------------------------------------------

def plate_strip_parameters(
    young: float,
    poisson: float,
    thickness: float,
    foundation_coefficient: float,
    strip_width: float = 1.0,
) -> tuple[float, float, float]:
    """Convert plate properties to equivalent beam-on-foundation parameters.

    For a plate strip of unit width:

        D_plate = E·h³ / (12·(1-ν²))    (flexural rigidity per unit width)
        k_b = k · b                      (foundation force per unit length)
        λ = (k / (4·D_plate))^(1/4)

    Parameters
    ----------
    young : E
    poisson : ν
    thickness : h
    foundation_coefficient : k (N/m³)
    strip_width : width of the strip (m), default 1.

    Returns
    -------
    EI_eff, k_b, lam
    """
    D_plate = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    EI_eff = D_plate * strip_width
    k_b = foundation_coefficient * strip_width
    lam = (foundation_coefficient / (4.0 * D_plate)) ** 0.25
    return EI_eff, k_b, lam