"""Independent Ritz reference for plate buckling under non-uniform uniaxial stress.

The production Shell4 operator integrates a bilinear initial-stress matrix on a
structured finite-element mesh.  This module is a deliberately different
method for the same physical problem: a simply supported isotropic plate whose
buckling field is expanded in a double sine series, with the geometric energy
integrated in closed-form in ``x`` and by Gauss quadrature in ``y``.  It shares
no code with the finite-element path, so agreement between the two is an
independent-method check rather than a self-consistency test.

Pre-stress model (positive compression)::

    Nx(y) = N0 * [1 + (psi - 1) * y / b],      Ny = Nxy = 0

``psi = 1`` is uniform compression, ``psi = 0`` compresses the ``y = 0`` edge
only, ``psi = -1`` is pure in-plane bending. ``N0`` is the compressive
resultant at ``y = 0`` and is the eigenvalue returned by both paths.

Energy functional for a simply supported plate::

    Pi = D/2 * int (lap w)^2 - 1/2 * int Nx(y) (dw/dx)^2

with ``D = E t^3 / (12 (1 - nu^2))``.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _check(name: str, value: float, *, positive: bool = True) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite")
    if positive and x <= 0:
        raise ValueError(f"{name} must be positive")
    return x


def ritz_uniaxial_buckling(*, a: float, b: float, young: float, poisson: float,
                           thickness: float, psi: float, harmonics: int = 12,
                           quadrature: int = 240) -> dict:
    """Return the critical compressive resultant ``N0`` of the Ritz model.

    ``harmonics`` is the number of sine half-waves retained in each direction;
    ``quadrature`` is the Gauss-Legendre order used for the ``y`` integrals.
    """
    a = _check("a", a); b = _check("b", b)
    E = _check("young", young); t = _check("thickness", thickness)
    nu = _check("poisson", poisson, positive=False)
    psi = _check("psi", psi, positive=False)
    if not -1 < nu < 0.5:
        raise ValueError("poisson must be in (-1, 0.5)")
    if isinstance(harmonics, bool) or int(harmonics) != harmonics or harmonics < 1:
        raise ValueError("harmonics must be a positive integer")
    if isinstance(quadrature, bool) or int(quadrature) != quadrature or quadrature < 8:
        raise ValueError("quadrature must be an integer >= 8")

    M = int(harmonics)
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    modes = [(m, n) for m in range(1, M + 1) for n in range(1, M + 1)]
    size = len(modes)

    # Gauss-Legendre nodes on [0, b] and the linear stress factor there.
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature))
    y = 0.5 * b * (nodes + 1.0)
    w = 0.5 * b * weights
    g = 1.0 + (psi - 1.0) * y / b

    # Y-integral matrix I[n, n'] = int_0^b g(y) sin(n pi y/b) sin(n' pi y/b) dy.
    sines = np.sin(np.outer(np.pi * np.arange(1, M + 1) / b, y))      # (M, Q)
    Y = (sines * (w * g)) @ sines.T

    K = np.zeros((size, size))
    G = np.zeros((size, size))
    for idx, (m, n) in enumerate(modes):
        kx = (m * math.pi / a) ** 2
        ky = (n * math.pi / b) ** 2
        K[idx, idx] = D * a * b / 8.0 * (kx + ky) ** 2
        for jdx, (m2, n2) in enumerate(modes):
            if m2 != m:
                continue
            # 1/2 from the geometric energy, (a/2) from int cos^2.
            G[idx, jdx] = 0.5 * kx * (a / 2.0) * Y[n - 1, n2 - 1]

    # K is diagonal and positive, so the pencil is whitened exactly.
    kinv = 1.0 / np.sqrt(np.diag(K))
    B = kinv[:, None] * G * kinv[None, :]
    B = 0.5 * (B + B.T)
    mu, vec = np.linalg.eigh(B)
    mu_max = float(mu[-1])
    if not math.isfinite(mu_max) or mu_max <= 0:
        raise RuntimeError("pre-stress field carries no compressive energy")
    ncr = 1.0 / mu_max
    mode = vec[:, -1] / kinv                       # physical amplitudes
    dominant = modes[int(np.argmax(np.abs(mode)))]
    return {
        "a": a, "b": b, "psi": psi, "harmonics": M, "quadrature": int(quadrature),
        "flexural_rigidity": D, "critical_resultant": ncr,
        "buckling_coefficient": ncr * b ** 2 / (math.pi ** 2 * D),
        "dominant_halfwaves": dominant,
        "mode_amplitudes": mode.reshape(M, M),
        "method": "Ritz double-sine energy minimisation (independent of the FE path)",
    }


def ritz_uniaxial_convergence(*, harmonics: Sequence[int] = (6, 8, 10, 12, 16),
                              **kwargs) -> list:
    """Sweep the harmonic basis and report the successive relative changes."""
    rows = []
    previous = None
    for h in harmonics:
        out = ritz_uniaxial_buckling(harmonics=int(h), **kwargs)
        change = None if previous is None else abs(out["critical_resultant"] - previous) / out["critical_resultant"]
        rows.append({"harmonics": int(h), "critical_resultant": out["critical_resultant"],
                     "buckling_coefficient": out["buckling_coefficient"],
                     "successive_change": change})
        previous = out["critical_resultant"]
    return rows


def timoshenko_coefficient_table() -> dict:
    """Values used only as literature cross-checks, not as fitting targets.

    ``uniform`` is the closed-form Navier result for ``a/b = 1`` (``k = 4``).
    ``pure_bending`` is the classical value quoted for a square simply
    supported plate in pure in-plane bending (Timoshenko & Gere, *Theory of
    Elastic Stability*, 2nd ed., ch. 9).  It is reported for comparison; the
    reference used for the finite-element verdict is the Ritz solution above.
    """
    return {"uniform": 4.0, "pure_bending": 23.9}