"""Analytical reference solutions for plate-on-elastic-foundation verification.

This module collects closed-form and series solutions used to validate the
TensorFEM plate + foundation implementations.  All functions are pure
(python/math only, no torch dependency) so they can serve as independent
oracles.

References
----------
[1] Hetényi, M., "Beams on Elastic Foundation", Univ. Michigan Press, 1946.
[2] Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells", 2nd ed.
[3] Kerr, A.D., "Elastic and Viscoelastic Foundation Models", JAM 1964.
[4] Vlasov, V.Z., "Beams, Plates and Shells on Elastic Foundations", 1960.
[5] Selvadurai, A.P.S., "Elastic Analysis of Soil-Foundation Interaction", 1979.
"""
from __future__ import annotations

import math


def navier_ss_plate_uniform(
    q: float,
    a: float,
    b: float,
    D: float,
    k_foundation: float = 0.0,
    M: int = 150,
    N: int = 150,
) -> float:
    """Navier double-series for simply-supported plate under uniform load.

    w(x,y) = Σ_m Σ_n  q_mn / (D·((mπ/a)²+(nπ/b)²)² + k) · sin(mπx/a)·sin(nπy/b)

    For uniform load q:  q_mn = 16q/(π²mn)  for m,n odd; 0 otherwise.

    Parameters
    ----------
    q : float  – load intensity (Pa)
    a, b : float  – plate dimensions (m)
    D : float  – flexural rigidity Eh³/(12(1-ν²))
    k_foundation : float  – Winkler/Kerr spring coefficient (N/m³)
    M, N : int  – number of odd terms per direction

    Returns
    -------
    w_center : float  – center deflection (m)
    """
    w = 0.0
    for m in range(1, 2 * M + 1, 2):
        for n in range(1, 2 * N + 1, 2):
            qmn = 16.0 * q / (math.pi ** 2 * m * n)
            am = m * math.pi / a
            bn = n * math.pi / b
            denom = D * (am ** 2 + bn ** 2) ** 2 + k_foundation
            # sin(mπ/2)·sin(nπ/2) for odd m,n
            sm = 1.0 if (m % 4 == 1) else -1.0
            sn = 1.0 if (n % 4 == 1) else -1.0
            w += qmn / denom * sm * sn
    return w


def navier_ss_plate_point_load(
    P: float,
    a: float,
    b: float,
    D: float,
    k_foundation: float = 0.0,
    x0: float = None,
    y0: float = None,
    M: int = 150,
    N: int = 150,
) -> float:
    """Navier series for simply-supported plate under center point load.

    For load P at (x0, y0), the Fourier coefficient is:
    q_mn = 4P/(ab) · sin(mπx0/a) · sin(nπy0/b)

    Returns center deflection w(a/2, b/2).
    """
    if x0 is None:
        x0 = a / 2
    if y0 is None:
        y0 = b / 2
    w = 0.0
    for m in range(1, 2 * M + 1, 2):
        for n in range(1, 2 * N + 1, 2):
            am = m * math.pi / a
            bn = n * math.pi / b
            denom = D * (am ** 2 + bn ** 2) ** 2 + k_foundation
            sm_load = math.sin(m * math.pi * x0 / a)
            sn_load = math.sin(n * math.pi * y0 / b)
            qmn = 4.0 * P / (a * b) * sm_load * sn_load
            sm_eval = 1.0 if (m % 4 == 1) else -1.0
            sn_eval = 1.0 if (n % 4 == 1) else -1.0
            w += qmn / denom * sm_eval * sn_eval
    return w


def kerr_ss_plate_uniform_approx(
    q: float,
    a: float,
    b: float,
    D: float,
    k: float,
    Gp: float,
    M: int = 150,
    N: int = 150,
) -> float:
    """Approximate Navier-type series for Kerr foundation.

    The Kerr foundation governing equation for a plate is:
    D·∇⁴w + k·w - Gp·∇²w = q

    For simply-supported boundary conditions the Navier approach gives:
    w_mn = q_mn / (D·(α_m²+β_n²)² + Gp·(α_m²+β_n²) + k)

    Note: This is the Vlasov-type approximation where the shear layer
    contributes Gp·(α²+β²) to the denominator.

    Parameters
    ----------
    q : load intensity
    a, b : plate dimensions
    D : flexural rigidity
    k : Winkler spring coefficient
    Gp : Kerr shear parameter
    """
    w = 0.0
    for m in range(1, 2 * M + 1, 2):
        for n in range(1, 2 * N + 1, 2):
            qmn = 16.0 * q / (math.pi ** 2 * m * n)
            am = m * math.pi / a
            bn = n * math.pi / b
            lap = am ** 2 + bn ** 2
            denom = D * lap ** 2 + Gp * lap + k
            sm = 1.0 if (m % 4 == 1) else -1.0
            sn = 1.0 if (n % 4 == 1) else -1.0
            w += qmn / denom * sm * sn
    return w


def kirchhoff_ss_center(
    q: float,
    a: float,
    b: float,
    D: float,
    M: int = 150,
    N: int = 150,
) -> float:
    """Classical Kirchhoff thin-plate center deflection (no foundation)."""
    return navier_ss_plate_uniform(q, a, b, D, k_foundation=0.0, M=M, N=N)


def fsdt_ss_center_deflection(
    q: float,
    a: float,
    b: float,
    D: float,
    h: float,
    G: float,
    k_normal: float = 0.0,
    g_shear: float = 0.0,
    kappa: float = 5.0 / 6.0,
    M: int = 150,
    N: int = 150,
) -> float:
    """Exact first-order-shear-deformation (Mindlin-Timoshenko) centre deflection.

    Double-sine solution of the simply supported plate on a two-parameter
    foundation, ``w(x,y) = sum W_mn sin(m pi x/a) sin(n pi y/b)`` with
    ``lam = (m pi/a)^2 + (n pi/b)^2`` and shear rigidity ``S = kappa G h``:

        W_mn = q_mn / ( k_normal + g_shear lam + D lam^2 / (1 + D lam / S) )

    For uniform load ``q_mn = 16 q / (pi^2 m n)`` (odd ``m, n``).  Dropping the
    ``1 / (1 + D lam / S)`` factor returns the Kirchhoff Navier series, so the
    difference is exactly the transverse-shear deflection
    ``sum q_mn / (S lam)``.  This replaces the earlier hard-coded
    ``nu = 0.3``/``C_s = 0.384`` estimate, which was only valid for square
    plates with ``nu = 0.3`` (see the P9263-N01 verification note).

    Pure python (no torch) so it stays an independent oracle.
    """
    if a <= 0.0 or b <= 0.0 or h <= 0.0:
        raise ValueError("plate geometry must be positive")
    S = kappa * G * h
    if S <= 0.0:
        raise ValueError("shear rigidity must be positive")
    w = 0.0
    for m in range(1, 2 * M + 1, 2):
        am = m * math.pi / a
        sm = 1.0 if (m % 4 == 1) else -1.0
        for n in range(1, 2 * N + 1, 2):
            bn = n * math.pi / b
            lam = am ** 2 + bn ** 2
            qmn = 16.0 * q / (math.pi ** 2 * m * n)
            sn = 1.0 if (n % 4 == 1) else -1.0
            denom = (k_normal + g_shear * lam
                     + D * lam ** 2 / (1.0 + D * lam / S))
            w += qmn / denom * sm * sn
    return w


def mindlin_ss_center_correction(
    q: float,
    a: float,
    b: float,
    D: float,
    h: float,
    G: float,
    kappa: float = 5.0 / 6.0,
    M: int = 150,
    N: int = 150,
) -> float:
    """Exact Mindlin (FSDT) centre deflection of a simply supported plate.

    The transverse shear part is *not* an estimate any more: the returned value
    is the exact double-sine (Navier) solution of the first-order shear
    deformation plate, which for uniform load on a square plate is about
    ``q a^4 / (D) * 0.00406 * (1 + K)`` with
    ``K = sum q_mn / (S lam) / w_K``.

    Signature is kept for compatibility with ``test_plate_foundation_ladder``;
    ``D``, ``G`` and ``h`` over-determine the isotropic pair ``(E, nu)`` and are
    consistent only if ``nu = 1 - G h^3 / (6 D)``, which is what the previous
    implementation ignored (it hard-coded ``nu = 0.3`` and the coefficient
    ``0.384``).
    """
    return fsdt_ss_center_deflection(q, a, b, D, h, G, kappa=kappa, M=M, N=N)


def gci_order_of_convergence(f1, f2, f3, r12, r23):
    """Compute GCI and apparent order of convergence.

    Uses three solutions on systematically refined meshes.

    Parameters
    ----------
    f1, f2, f3 : float  – solutions on fine, medium, coarse meshes
    r12 : float  – refinement ratio h_coarse/h_medium
    r23 : float  – refinement ratio h_medium/h_fine

    Returns
    -------
    p : float  – apparent order
    GCI_fine : float  – Grid Convergence Index for fine mesh (%)
    """
    eps12 = abs(f2 - f1)
    eps23 = abs(f3 - f2)
    if eps12 == 0 or eps23 == 0:
        return 2.0, 0.0  # perfect convergence

    # Apparent order (Roache's method)
    s = 1.0  # sign of solution change
    if (f3 - f2) * (f2 - f1) > 0:
        s = 1.0
    else:
        s = -1.0

    q = math.log(r12 / r23) if r12 != r23 else math.log(2.0)
    p = abs(math.log(abs(eps23 / eps12)) / q) if eps12 > 0 and eps23 > 0 else 2.0
    p = max(p, 0.5)  # clamp to reasonable range

    # GCI for fine mesh (safety factor Fs = 1.25 for 3 grids)
    Fs = 1.25
    GCI_fine = Fs * abs((f2 - f1) / f1) * 100.0 / (r23 ** p - 1) if f1 != 0 else 0.0

    return p, GCI_fine
