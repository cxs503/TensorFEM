#!/usr/bin/env python
"""Independent plane-stress classical lamination theory (CLT) + ply failure criteria.

This module is deliberately written from the textbook expressions (Jones 1999,
*Mechanics of Composite Materials*, ch. 4-5; Tsai & Wu 1971) and imports **nothing
from** :mod:`tensorfem`, so that a comparison against TensorFEM is a comparison of
two code paths rather than a tautology.  The symbolic derivation that plays the
role of the anchor lives in :mod:`symbolic_anchor` and uses a different route
(tensor rotation + symbolic thickness integration) on purpose.

Contents
--------
``PlyProperties``       plane-stress engineering constants of one lamina.
``reduced_stiffness``   ``Q`` in material axes (plane stress).
``transformed_reduced`` ``Qbar(theta)`` by the classical power formulas.
``abd``                 ``A``, ``B``, ``D`` and the transverse-shear ``As``.
``clt_response``        ``(eps0, kappa)`` for a membrane load ``N`` (B != 0 allowed).
``ply_stress``          ply stresses in material axes for a strain state.
``tsai_wu`` / ``tsai_hill`` / ``max_stress``  failure indices (1 = first ply failure).

All quantities in SI (Pa, m); ``theta`` in degrees, measured from the x axis,
positive anticlockwise, ply 0 = bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Classical Tsai-Wu interaction parameter ``F12* = F12 / sqrt(F11 F22)``.
#: -1/2 is the value that makes the criterion coincide with a von Mises ellipse in
#: the (sigma1, sigma2) plane for an isotropic material (Tsai & Wu 1971); it is an
#: explicit *assumption* of the criterion, not a material constant, and it is the
#: single largest source of spread between published Tsai-Wu implementations.
F12_STAR = -0.5


@dataclass(frozen=True)
class PlyProperties:
    """Transversely isotropic plane-stress ply plus its five strength allowables (Pa)."""

    E1: float
    E2: float
    nu12: float
    G12: float
    G13: float
    G23: float
    Xt: float
    Xc: float
    Yt: float
    Yc: float
    S: float
    name: str = "ply"

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("name", "E1", "E2", "nu12", "G12", "G13", "G23",
                 "Xt", "Xc", "Yt", "Yc", "S")}


def reduced_stiffness(ply: PlyProperties) -> np.ndarray:
    """Plane-stress reduced stiffness ``Q`` (material axes), Voigt (11, 22, 12)."""
    nu21 = ply.nu12 * ply.E2 / ply.E1
    d = 1.0 - ply.nu12 * nu21
    return np.array([
        [ply.E1 / d, ply.nu12 * ply.E2 / d, 0.0],
        [ply.nu12 * ply.E2 / d, ply.E2 / d, 0.0],
        [0.0, 0.0, ply.G12],
    ])


def transformed_reduced(ply: PlyProperties, theta_deg: float) -> np.ndarray:
    """``Qbar`` by the classical power formulas (Jones 1999 eq. 4.31)."""
    Q = reduced_stiffness(ply)
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    Q11, Q22, Q12, Q66 = Q[0, 0], Q[1, 1], Q[0, 1], Q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    m3n, mn3 = m2 * m * n, m * n2 * n
    return np.array([
        [Q11 * m4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * n4,
         (Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4),
         (Q11 - Q12 - 2.0 * Q66) * m3n - (Q22 - Q12 - 2.0 * Q66) * mn3],
        [(Q11 + Q22 - 4.0 * Q66) * m2 * n2 + Q12 * (m4 + n4),
         Q11 * n4 + 2.0 * (Q12 + 2.0 * Q66) * m2 * n2 + Q22 * m4,
         (Q11 - Q12 - 2.0 * Q66) * mn3 - (Q22 - Q12 - 2.0 * Q66) * m3n],
        [(Q11 - Q12 - 2.0 * Q66) * m3n - (Q22 - Q12 - 2.0 * Q66) * mn3,
         (Q11 - Q12 - 2.0 * Q66) * mn3 - (Q22 - Q12 - 2.0 * Q66) * m3n,
         (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * m2 * n2 + Q66 * (m4 + n4)],
    ])


def transformed_shear(ply: PlyProperties, theta_deg: float) -> np.ndarray:
    """Transverse-shear stiffness block ``[[Q55, Q45], [Q45, Q44]]`` (xz, yz)."""
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    c2, s2, cs = m * m, n * n, m * n
    # Q45 = (G13 - G23) cs: the sign follows from the shear compliance
    # S45 = cs (1/G13 - 1/G23) written out with e1 = (c, s) at +theta, so that
    # Q = S^-1 flips it.  The opposite sign (G23 - G13) cs is a common slip; it was
    # caught here by the 3-D rotation route of TensorFEM and by an energy check.
    return np.array([
        [ply.G13 * c2 + ply.G23 * s2, (ply.G13 - ply.G23) * cs],
        [(ply.G13 - ply.G23) * cs, ply.G13 * s2 + ply.G23 * c2],
    ])


def interfaces(thicknesses: Sequence[float]) -> list:
    """Ply interfaces bottom -> top, midsurface at z = 0."""
    h = float(sum(thicknesses))
    z = [-0.5 * h]
    for t in thicknesses:
        z.append(z[-1] + float(t))
    return z


def abd(ply: PlyProperties, angles: Sequence[float], thicknesses: Sequence[float],
        shear_correction: float = 5.0 / 6.0) -> tuple:
    """``(A, B, D, As)`` with ``As`` already shear-corrected (xz, yz ordering)."""
    angles = [float(a) for a in angles]
    thicknesses = [float(t) for t in thicknesses]
    if len(angles) != len(thicknesses):
        raise ValueError("one thickness per angle")
    z = interfaces(thicknesses)
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    As = np.zeros((2, 2))
    for k, th in enumerate(angles):
        Qb = transformed_reduced(ply, th)
        Qs = transformed_shear(ply, th)
        z0, z1 = z[k], z[k + 1]
        A += Qb * (z1 - z0)
        B += 0.5 * Qb * (z1 ** 2 - z0 ** 2)
        D += Qb * (z1 ** 3 - z0 ** 3) / 3.0
        As += shear_correction * Qs * (z1 - z0)
    return A, B, D, As


def clt_response(ply: PlyProperties, angles: Sequence[float], thicknesses: Sequence[float],
                 N: Sequence[float], M: Sequence[float] = (0.0, 0.0, 0.0)) -> tuple:
    """Mid-surface strain and curvature for membrane ``N`` and moment ``M`` resultants."""
    A, B, D, _ = abd(ply, angles, thicknesses)
    K = np.block([[A, B], [B, D]])
    rhs = np.concatenate([np.asarray(N, dtype=float), np.asarray(M, dtype=float)])
    sol = np.linalg.solve(K, rhs)
    return sol[:3], sol[3:]


def ply_stress(ply: PlyProperties, theta_deg: float, eps0: Sequence[float],
               kappa: Sequence[float], z: float) -> np.ndarray:
    """Ply stress in material axes (sigma1, sigma2, tau12) at through-thickness ``z``."""
    Qb = transformed_reduced(ply, theta_deg)
    eps = np.asarray(eps0, dtype=float) + float(z) * np.asarray(kappa, dtype=float)
    sig = Qb @ eps
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    s11 = m * m * sig[0] + n * n * sig[1] + 2.0 * m * n * sig[2]
    s22 = n * n * sig[0] + m * m * sig[1] - 2.0 * m * n * sig[2]
    s12 = -m * n * sig[0] + m * n * sig[1] + (m * m - n * n) * sig[2]
    return np.array([s11, s22, s12])


# ── failure criteria (index = 1 at first ply failure) ───────────────────────
def tsai_wu(ply: PlyProperties, s1: float, s2: float, tau12: float,
            f12_star: float = F12_STAR) -> float:
    F1 = 1.0 / ply.Xt - 1.0 / ply.Xc
    F2 = 1.0 / ply.Yt - 1.0 / ply.Yc
    F11 = 1.0 / (ply.Xt * ply.Xc)
    F22 = 1.0 / (ply.Yt * ply.Yc)
    F66 = 1.0 / (ply.S * ply.S)
    F12 = f12_star * math.sqrt(F11 * F22)
    return (F1 * s1 + F2 * s2 + F11 * s1 * s1 + 2.0 * F12 * s1 * s2
            + F22 * s2 * s2 + F66 * tau12 * tau12)


def tsai_hill(ply: PlyProperties, s1: float, s2: float, tau12: float) -> float:
    X = ply.Xt if s1 >= 0.0 else ply.Xc
    Y = ply.Yt if s2 >= 0.0 else ply.Yc
    return ((s1 / X) ** 2 - s1 * s2 / (X * X) + (s2 / Y) ** 2
            + (tau12 / ply.S) ** 2)


def max_stress(ply: PlyProperties, s1: float, s2: float, tau12: float) -> float:
    f1 = abs(s1) / (ply.Xt if s1 >= 0.0 else ply.Xc)
    f2 = abs(s2) / (ply.Yt if s2 >= 0.0 else ply.Yc)
    return max(f1, f2, abs(tau12) / ply.S)


CRITERIA = {"tsai_wu": tsai_wu, "tsai_hill": tsai_hill, "max_stress": max_stress}


def symmetric(half_angles: Sequence[float]) -> list:
    """``[a/b]s`` from ``[a, b]``: mirror through the midsurface."""
    return [float(a) for a in half_angles] + [float(a) for a in reversed(half_angles)]


def fpf_load_factor(ply: PlyProperties, angles: Sequence[float], thicknesses: Sequence[float],
                    N: Sequence[float], criterion: str = "tsai_wu") -> dict:
    """Scale factor ``lam`` such that ``max_ply index(lam * N) = 1``, plus the hot ply.

    ``tsai_wu``/``tsai_hill`` are homogeneous of degree two in the stress and
    ``max_stress`` of degree one, so the factor follows in closed form -- the result
    is therefore independent of the arbitrary magnitude of the load case.
    """
    fn = CRITERIA[criterion]
    eps0, kappa = clt_response(ply, angles, thicknesses, N)
    worst = None
    for k, th in enumerate(angles):
        s = ply_stress(ply, th, eps0, kappa, 0.0)   # membrane load: z-independent
        idx = fn(ply, s[0], s[1], s[2])
        if worst is None or idx > worst["index_unit"]:
            worst = {"ply": k, "angle_deg": float(th), "index_unit": idx,
                     "stress_unit": s.tolist()}
    degree = 2 if criterion != "max_stress" else 1
    lam = (1.0 / worst["index_unit"]) ** (1.0 / degree)
    return {"load_factor": lam, "hot_ply": worst["ply"],
            "hot_angle_deg": worst["angle_deg"], "index_at_unit_load": worst["index_unit"],
            "stress_at_unit_load": worst["stress_unit"], "degree": degree}