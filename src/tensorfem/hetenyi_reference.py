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
from dataclasses import dataclass

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
    exponentially small.  For free, clamped or guided ends use
    :func:`finite_beam_bvp`, which implements Hetényi's end-conditioning solution
    exactly (see ``docs/hetanyi-free-end-reference.md``).

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


# ---------------------------------------------------------------------------
# Finite beam on a Winkler foundation: general end conditions (Hetényi BVP)
# ---------------------------------------------------------------------------
#
# :func:`finite_beam_series` is exact for simply supported ends only.  The
# routines below close that documented gap with Hetényi's end-conditioning
# solution of the same ODE ``EI w'''' + k_b w = q`` for arbitrary end
# combinations (free / simply supported / clamped / guided).
#
# Working in the dimensionless state
#
#     z(xi) = [w, w'/lam, w''/lam^2, w'''/lam^3],   xi = lam*x,  lam^4 = k_b/(4EI)
#
# the homogeneous ODE becomes the constant-coefficient system
#
#     dz/dxi = A z,   A = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]]
#
# so any state propagates exactly, ``z(xi) = P(xi) z(0)`` with
# ``P(xi) = exp(A xi)``.  ``P`` is elementary: with the four products
#
#     u1 = cosh(xi)cos(xi),  u2 = cosh(xi)sin(xi),
#     u3 = sinh(xi)cos(xi),  u4 = sinh(xi)sin(xi)
#
# the rows of ``exp(A xi)`` follow from ``u1'' = -2u4``, ``u2'' = 2u3``,
# ``u3'' = -2u2``, ``u4'' = 2u1`` and their derivatives; this is the matrix
# form of Hetényi's end-conditioning coefficient table (Hetényi 1946, ch. II).
#
# Because ``P`` contains ``exp(+lam x)``, a single-ended propagation loses
# accuracy once ``lam*L`` is large (the boundary-value system becomes
# exponentially ill-conditioned).  The solver therefore uses *multiple
# shooting*: the beam is subdivided so that every propagation span is at most
# ``max_span_xi / lam``, and the state at every node is an unknown of a banded
# block system.  The conditioning is then independent of ``lam*L``.
#
# Sign conventions agree with the rest of this module: ``w`` positive
# downward, ``q`` positive downward, ``M = -EI w''`` (so that the infinite
# beam gives ``M = P/(4 lam) B(lam x)``).  The shear is ``V = -EI w'''`` so
# that a downward point load ``P`` produces the physical jump ``Delta V = -P``.

# Component indices of the dimensionless state used by each end condition:
#   free            : M = 0, V = 0   -> w''  = 0, w''' = 0
#   simply supported: w = 0, M = 0   -> w    = 0, w''  = 0
#   clamped         : w = 0, w' = 0  -> w    = 0, w'   = 0
#   guided (symm.)  : w' = 0, V = 0  -> w'   = 0, w''' = 0
_END_CONDITION_ROWS: dict[str, tuple[int, int]] = {
    "free": (2, 3),
    "simply_supported": (0, 2),
    "clamped": (0, 1),
    "guided": (1, 3),
}

_END_CONDITION_ALIASES: dict[str, str] = {
    "free": "free", "free-free": "free", "f": "free",
    "simply_supported": "simply_supported", "simply-supported": "simply_supported",
    "simply supported": "simply_supported", "ss": "simply_supported",
    "pinned": "simply_supported", "simple": "simply_supported",
    "clamped": "clamped", "clamped-clamped": "clamped", "fixed": "clamped",
    "c": "clamped",
    "guided": "guided", "symmetry": "guided",
}

#: End conditions accepted by :func:`finite_beam_bvp`.
FINITE_BEAM_END_CONDITIONS: tuple[str, ...] = tuple(_END_CONDITION_ROWS)

_STATE_BASIS_AT_ORIGIN = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, -2.0, 0.0],
    ],
    dtype=torch.float64,
)
_STATE_BASIS_AT_ORIGIN_INV = torch.linalg.inv(_STATE_BASIS_AT_ORIGIN)


def _normalise_end_condition(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _END_CONDITION_ALIASES:
        raise ValueError(
            f"unsupported end condition {name!r}; expected one of "
            f"{sorted(set(_END_CONDITION_ALIASES.values()))}"
        )
    return _END_CONDITION_ALIASES[key]


def _state_basis_matrix(xi: torch.Tensor) -> torch.Tensor:
    """Rows are the four homogeneous solutions and their xi-derivatives."""
    u1 = torch.cosh(xi) * torch.cos(xi)
    u2 = torch.cosh(xi) * torch.sin(xi)
    u3 = torch.sinh(xi) * torch.cos(xi)
    u4 = torch.sinh(xi) * torch.sin(xi)
    return torch.stack(
        [
            torch.stack([u1, u2, u3, u4]),
            torch.stack([u3 - u2, u1 + u4, u1 - u4, u2 + u3]),
            torch.stack([-2.0 * u4, 2.0 * u3, -2.0 * u2, 2.0 * u1]),
            torch.stack([-2.0 * (u2 + u3), 2.0 * (u1 - u4), -2.0 * (u1 + u4), 2.0 * (u3 - u2)]),
        ]
    )


def beam_state_propagator(xi, *, dtype=torch.float64) -> torch.Tensor:
    """Exact propagator ``P`` with ``z(xi) = P(xi) @ z(0)``.

    ``P(xi) = exp(A xi)`` for the dimensionless Winkler-beam system, evaluated
    in closed form from the ``u1..u4`` products rather than by integrating the
    ODE, so the accuracy is limited only by floating-point round-off.  It is
    compared against ``torch.matrix_exp`` in the unit tests.
    """
    xi_t = torch.as_tensor(xi, dtype=dtype)
    if torch.any(xi_t < 0):
        raise ValueError("xi must be non-negative")
    return _state_basis_matrix(xi_t) @ _STATE_BASIS_AT_ORIGIN_INV.to(dtype)


@dataclass(frozen=True)
class FiniteBeamSolution:
    """Exact field of a finite beam on a Winkler foundation.

    Attributes
    ----------
    x : evaluation abscissae (m).
    deflection : ``w`` positive downward (m).
    slope : ``dw/dx`` (rad).
    moment : ``M = -EI w''`` (N m).
    shear : ``V = -EI w'''`` (N); a downward point load makes ``V`` jump by ``-P``.
    foundation_reaction : ``k_b w`` (N/m), positive downward.
    state_at_origin : dimensionless state at ``x = 0`` (after any jump there).
    end_conditions : (``end_a``, ``end_b``) after alias normalisation.
    system_condition : condition number of the multiple-shooting system.
    nodes : shooting nodes (m).
    """

    x: torch.Tensor
    deflection: torch.Tensor
    slope: torch.Tensor
    moment: torch.Tensor
    shear: torch.Tensor
    foundation_reaction: torch.Tensor
    state_at_origin: torch.Tensor
    end_conditions: tuple[str, str]
    system_condition: float
    nodes: torch.Tensor


def finite_beam_bvp(
    x_eval: torch.Tensor,
    *,
    beam_length: float,
    lam: float,
    k_b: float,
    EI: float,
    point_loads: list[tuple[float, float]] | tuple = (),
    distributed_loads: list[tuple[float, float, float]] | tuple = (),
    end_a: str = "free",
    end_b: str = "free",
    max_span_xi: float = 1.0,
) -> FiniteBeamSolution:
    """Exact finite beam on a Winkler foundation for arbitrary end conditions.

    Solves ``EI w'''' + k_b w = q`` on ``[0, L]`` with two boundary conditions
    at each end using Hetényi's end-conditioning method in matrix form, i.e. the
    exact propagator of the dimensionless state plus a linear boundary-value
    solve.  This is the reference solution that :func:`finite_beam_series`
    could not provide for free or clamped ends.

    Parameters
    ----------
    x_eval : (N,) evaluation abscissae inside ``[0, L]``.
    beam_length : ``L`` (m).
    lam : ``(k_b / 4EI)^(1/4)`` (1/m).
    k_b : foundation coefficient per unit length (N/m^2).
    EI : bending stiffness (N m^2).
    point_loads : list of ``(x, P)``, ``P`` positive downward (N); positions
        must lie strictly inside the span.
    distributed_loads : list of ``(x0, x1, q)`` with ``q`` positive downward
        (N/m); overlapping segments are summed.
    end_a, end_b : end condition at ``x = 0`` / ``x = L``: ``'free'``,
        ``'simply_supported'``, ``'clamped'`` or ``'guided'`` (aliases
        ``'ss'``, ``'pinned'``, ``'fixed'``, ``'symmetry'`` are accepted).
    max_span_xi : largest ``lam * (node spacing)`` used by the multiple
        shooting; the default 1.0 keeps the linear system well conditioned.

    Returns
    -------
    :class:`FiniteBeamSolution`
    """
    L = float(beam_length)
    lam = float(lam)
    k_b = float(k_b)
    EI = float(EI)
    for name, value in (("beam_length", L), ("lam", lam), ("k_b", k_b), ("EI", EI)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if not math.isfinite(max_span_xi) or max_span_xi <= 0.0:
        raise ValueError(f"max_span_xi must be positive and finite, got {max_span_xi!r}")
    end_a_c = _normalise_end_condition(end_a)
    end_b_c = _normalise_end_condition(end_b)

    dtype = x_eval.dtype
    loads = [(float(x), float(P)) for x, P in point_loads]
    dists = [(float(x0), float(x1), float(q)) for x0, x1, q in distributed_loads]
    for x, _ in loads:
        if not (0.0 < x < L):
            raise ValueError(f"point load at x={x!r} must lie strictly inside (0, {L})")
    for x0, x1, _ in dists:
        if not (0.0 <= x0 < x1 <= L):
            raise ValueError(f"distributed load span ({x0!r}, {x1!r}) must be inside [0, {L}]")
    if torch.any(x_eval < 0) or torch.any(x_eval > L):
        raise ValueError("x_eval must lie inside [0, beam_length]")

    # --- shooting nodes: load stations plus uniform subdivision -------------
    base = sorted({0.0, L} | {x for x, _ in loads} | {x0 for x0, _, _ in dists} | {x1 for _, x1, _ in dists})
    nodes: list[float] = [base[0]]
    for a, b in zip(base[:-1], base[1:]):
        n = max(1, int(math.ceil(lam * (b - a) / max_span_xi)))
        for j in range(1, n + 1):
            nodes.append(b if j == n else a + (b - a) * j / n)
    n_seg = len(nodes) - 1

    jump = torch.zeros((n_seg + 1, 4), dtype=dtype)
    for x, P in loads:
        idx = min(range(len(nodes)), key=lambda i: abs(nodes[i] - x))
        jump[idx, 3] = jump[idx, 3] + P / (EI * lam ** 3)

    seg_q = []
    for i in range(n_seg):
        mid = 0.5 * (nodes[i] + nodes[i + 1])
        seg_q.append(sum(q for x0, x1, q in dists if x0 <= mid <= x1))
    zp = [torch.tensor([q / k_b, 0.0, 0.0, 0.0], dtype=dtype) for q in seg_q]
    props = [beam_state_propagator(lam * (nodes[i + 1] - nodes[i]), dtype=dtype) for i in range(n_seg)]

    # --- multiple-shooting system: unknowns are the states at the nodes ----
    n_unknown = 4 * (n_seg + 1)
    matrix = torch.zeros((n_unknown, n_unknown), dtype=dtype)
    rhs = torch.zeros(n_unknown, dtype=dtype)
    row = 0
    for comp in _END_CONDITION_ROWS[end_a_c]:
        matrix[row, comp] = 1.0
        row += 1
    for i in range(n_seg):
        block_lo = 4 * i
        nxt = 4 * (i + 1)
        matrix[row:row + 4, block_lo:block_lo + 4] = -props[i]
        matrix[row:row + 4, nxt:nxt + 4] = torch.eye(4, dtype=dtype)
        rhs[row:row + 4] = (-props[i] @ zp[i] + zp[i]) + jump[i + 1]
        row += 4
    for comp in _END_CONDITION_ROWS[end_b_c]:
        matrix[row, 4 * n_seg + comp] = 1.0
        row += 1
    system_condition = float(torch.linalg.cond(matrix))
    solution = torch.linalg.solve(matrix, rhs)
    states = solution.reshape(n_seg + 1, 4)

    # --- evaluate the field -----------------------------------------------
    w = torch.empty_like(x_eval)
    slope = torch.empty_like(x_eval)
    wpp = torch.empty_like(x_eval)
    wppp = torch.empty_like(x_eval)
    node_t = torch.as_tensor(nodes[1:-1], dtype=dtype)
    idx = torch.searchsorted(node_t, x_eval, right=True)
    for e, (xe, k) in enumerate(zip(x_eval.tolist(), idx.tolist())):
        span = max(0.0, xe - nodes[k])
        prop = beam_state_propagator(lam * span, dtype=dtype)
        zx = prop @ (states[k] - zp[k]) + zp[k]
        w[e] = zx[0]
        slope[e] = lam * zx[1]
        wpp[e] = lam * lam * zx[2]
        wppp[e] = lam ** 3 * zx[3]
    return FiniteBeamSolution(
        x=x_eval.clone(),
        deflection=w,
        slope=slope,
        moment=-EI * wpp,
        shear=-EI * wppp,
        foundation_reaction=k_b * w,
        state_at_origin=states[0].clone(),
        end_conditions=(end_a_c, end_b_c),
        system_condition=system_condition,
        nodes=torch.as_tensor(nodes, dtype=dtype),
    )


def foundation_reaction_integral(solution: FiniteBeamSolution) -> float:
    """Trapezoidal ``integral k_b w dx`` of a :class:`FiniteBeamSolution`.

    For a beam free at both ends this equals the total applied load, so it is an
    exact global-equilibrium oracle (trapezoidal quadrature is the only error
    source).
    """
    return float(torch.trapezoid(solution.foundation_reaction, solution.x))
