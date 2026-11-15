#!/usr/bin/env python
"""Anchor generator: high-precision classical lamination theory reference values.

Why a second implementation at all
----------------------------------
The closure test in :mod:`run_closure` compares three code paths:

``ANCHOR``   this module -- 50-digit ``mpmath``, compliance-inversion ``Q`` and a
             transformation-matrix rotation ``Qbar = T^-1 Q R``, plus an exact
             thickness integration.  Different algebra *and* extended precision,
             so an implementation slip in the double-precision power-formula route
             shows up as a residue instead of being cancelled.
``IND``      :mod:`clt_independent` -- double-precision ``numpy``, textbook
             power formulas for ``Qbar`` (Jones 1999 eq. 4.31).
``TF``       :mod:`tensorfem.composite_laminate_bending` -- ``Laminate.abd()``,
             which reduces the *rotated 3-D* stiffness (``Q = C_ii - C_io
             C_oo^-1 C_oi``), i.e. a different reduction route from the classical
             plane-stress ``Qbar``; the residue therefore carries the modelling
             difference as well as round-off and is reported separately.

Honesty note
------------
These anchor numbers are **not** transcribed from a document: they are derived.
Their provenance is ``the formulation below + the declared inputs``, and they are
re-derivable by anyone who runs this script (the script hashes its own output).
Where an anchor is claimed to be *external* (a clause value, a published classic
solution) that is recorded separately with the retrieved document's sha256 in
``benchmarks/anchors/laminate-failure-criteria-anchor.json`` and its status is
graded item by item (several are ``WITHHELD``).

Usage::

    python benchmarks/composite_laminate_w30/symbolic_anchor.py [--digits 50]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import mpmath as mp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "anchor_values.json"

# ── declared inputs (identical on every path) ──────────────────────────────
#: IM7/8552 engineering constants, copied from the in-repo ply of TensorFEM
#: (``src/tensorfem/composite_laminate_bending.py``, ``IM7_8552_MATERIAL``); the
#: anchor and both solvers use exactly this set, so no comparison depends on it.
MATERIAL = {
    "name": "IM7/8552 carbon-epoxy (elastic constants: in-repo Ply3D)",
    "E1": "161.0e9", "E2": "11.38e9", "E3": "11.38e9",
    "nu12": "0.32", "nu13": "0.32", "nu23": "0.436",
    "G12": "5.17e9", "G13": "5.17e9", "G23": "3.98e9",
    # Strength allowables: DECLARED INPUT, provenance WITHHELD (see the anchor
    # ledger).  They enter anchor and solver identically, and every quantity that
    # is compared across paths is either independent of them (A, B, D, strains,
    # stresses) or exactly homogeneous in them (failure index, load factor), so a
    # wrong strength magnitude moves both sides by the same factor.
    "Xt": "2326.2e6", "Xc": "1200.1e6", "Yt": "62.3e6", "Yc": "199.8e6", "S": "92.3e6",
}
PLY_THICKNESS = "0.125e-3"

#: layup register: ``angles`` bottom -> top.  Two layups from the mission contract
#: plus two deliberately coupled ones (B != 0 / A16 != 0) for the root-cause table.
LAYUPS = {
    "L1_0_90_pm45_s": {"half": [0.0, 90.0, 45.0, -45.0], "symmetric": True,
                       "label": "[0/90/+-45]s"},
    "L2_pm30_0_s": {"half": [30.0, -30.0, 0.0], "symmetric": True,
                    "label": "[+-30/0]s"},
    "L3_coupled_B": {"half": [0.0, 45.0, -45.0, 90.0], "symmetric": False,
                     "label": "[0/45/-45/90] (unsymmetric, B != 0)"},
    "L4_coupled_A16": {"half": [0.0, 45.0], "symmetric": False,
                       "label": "[0/45] (unbalanced, A16 != 0)"},
}

#: in-plane load cases, N per unit width (N/m)
LOAD_CASES = {
    "LC1_uniaxial_tension": [1.0e6, 0.0, 0.0],
    "LC2_biaxial_tension": [5.0e5, 5.0e5, 0.0],
    "LC3_pure_shear": [0.0, 0.0, 5.0e5],
    "LC4_uniaxial_compression": [-1.0e6, 0.0, 0.0],
}


def _mp(x) -> "mp.mpf":
    return mp.mpf(str(x))


def angles_of(key: str) -> list:
    reg = LAYUPS[key]
    half = [float(a) for a in reg["half"]]
    return half + list(reversed(half)) if reg["symmetric"] else half


# ── anchor theory at working precision ─────────────────────────────────────
def compliance(ply) -> "mp.matrix":
    """3x3 plane-stress compliance ``S`` (engineering shear strain)."""
    S = mp.zeros(3, 3)
    S[0, 0] = 1 / _mp(ply["E1"])
    S[1, 1] = 1 / _mp(ply["E2"])
    S[0, 1] = S[1, 0] = -_mp(ply["nu12"]) / _mp(ply["E1"])
    S[2, 2] = 1 / _mp(ply["G12"])
    return S


def q_matrix(ply) -> "mp.matrix":
    """``Q = S^-1`` (inversion route, not the closed-form fractions)."""
    return mp.inverse(compliance(ply))


def rotation_matrices(theta_deg: float):
    """``T`` (stress) and ``R`` (engineering strain) for a rotation by ``theta``."""
    t = _mp(theta_deg) * mp.pi / 180
    c, s = mp.cos(t), mp.sin(t)
    T = mp.matrix([[c * c, s * s, 2 * c * s],
                   [s * s, c * c, -2 * c * s],
                   [-c * s, c * s, c * c - s * s]])
    R = mp.matrix([[c * c, s * s, c * s],
                   [s * s, c * c, -c * s],
                   [-2 * c * s, 2 * c * s, c * c - s * s]])
    return T, R


def qbar(ply, theta_deg: float) -> "mp.matrix":
    """``Qbar = T^-1 Q R`` -- transformation-matrix route."""
    T, R = rotation_matrices(theta_deg)
    return mp.inverse(T) * q_matrix(ply) * R


def qbar_shear(ply, theta_deg: float) -> "mp.matrix":
    """Transverse-shear block (uncorrected), ``[[Q55, Q45], [Q45, Q44]]``."""
    t = _mp(theta_deg) * mp.pi / 180
    c, s = mp.cos(t), mp.sin(t)
    g13, g23 = _mp(ply["G13"]), _mp(ply["G23"])
    return mp.matrix([[g13 * c * c + g23 * s * s, (g13 - g23) * c * s],
                      [(g13 - g23) * c * s, g13 * s * s + g23 * c * c]])


def abd_anchor(ply, angles, thicknesses, shear_correction="5/6"):
    z = [-sum(_mp(t) for t in thicknesses) / 2]
    for t in thicknesses:
        z.append(z[-1] + _mp(t))
    A, B, D, As = mp.zeros(3, 3), mp.zeros(3, 3), mp.zeros(3, 3), mp.zeros(2, 2)
    k = _mp(shear_correction)
    for i, th in enumerate(angles):
        Qb = qbar(ply, th)
        z0, z1 = z[i], z[i + 1]
        for r in range(3):
            for c in range(3):
                A[r, c] += Qb[r, c] * (z1 - z0)
                B[r, c] += Qb[r, c] * (z1 ** 2 - z0 ** 2) / 2
                D[r, c] += Qb[r, c] * (z1 ** 3 - z0 ** 3) / 3
        Qs = qbar_shear(ply, th)
        for r in range(2):
            for c in range(2):
                As[r, c] += k * Qs[r, c] * (z1 - z0)
    return A, B, D, As


def clt_solve(ply, angles, thicknesses, N):
    A, B, D, _ = abd_anchor(ply, angles, thicknesses)
    K = mp.zeros(6, 6)
    for r in range(3):
        for c in range(3):
            K[r, c] = A[r, c]
            K[r, 3 + c] = B[r, c]
            K[3 + r, c] = B[r, c]
            K[3 + r, 3 + c] = D[r, c]
    rhs = mp.matrix([_mp(v) for v in N] + [mp.mpf(0)] * 3)
    sol = mp.lu_solve(K, rhs)
    return [sol[i] for i in range(3)], [sol[3 + i] for i in range(3)]


def ply_stress_anchor(ply, theta_deg: float, eps0, kappa, z=0.0):
    Qb = qbar(ply, theta_deg)
    eps = mp.matrix([eps0[i] + _mp(z) * kappa[i] for i in range(3)])
    sig = Qb * eps
    T, _ = rotation_matrices(theta_deg)
    s = T * sig          # global -> material axes
    return s[0], s[1], s[2]


def tsai_wu_anchor(ply, s1, s2, t12, f12_star="-1/2"):
    Xt, Xc, Yt, Yc, S = (_mp(ply[k]) for k in ("Xt", "Xc", "Yt", "Yc", "S"))
    F1, F2 = 1 / Xt - 1 / Xc, 1 / Yt - 1 / Yc
    F11, F22, F66 = 1 / (Xt * Xc), 1 / (Yt * Yc), 1 / (S * S)
    F12 = _mp(f12_star) * mp.sqrt(F11 * F22)
    return (F1 * s1 + F2 * s2 + F11 * s1 ** 2 + 2 * F12 * s1 * s2
            + F22 * s2 ** 2 + F66 * t12 ** 2)


def tsai_hill_anchor(ply, s1, s2, t12):
    X = _mp(ply["Xt"]) if s1 >= 0 else _mp(ply["Xc"])
    Y = _mp(ply["Yt"]) if s2 >= 0 else _mp(ply["Yc"])
    return (s1 / X) ** 2 - s1 * s2 / X ** 2 + (s2 / Y) ** 2 + (t12 / _mp(ply["S"])) ** 2


def max_stress_anchor(ply, s1, s2, t12):
    f1 = abs(s1) / (_mp(ply["Xt"]) if s1 >= 0 else _mp(ply["Xc"]))
    f2 = abs(s2) / (_mp(ply["Yt"]) if s2 >= 0 else _mp(ply["Yc"]))
    return max(f1, f2, abs(t12) / _mp(ply["S"]))


CRIT = {"tsai_wu": tsai_wu_anchor, "tsai_hill": tsai_hill_anchor,
        "max_stress": max_stress_anchor}


def _f(x) -> float:
    return float(x)


def _mat(M) -> list:
    return [[_f(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]


# ── anchor bundle ──────────────────────────────────────────────────────────
def build(digits: int = 50) -> dict:
    mp.mp.dps = digits
    ply = MATERIAL
    layups = {}
    for key, reg in LAYUPS.items():
        angles = angles_of(key)
        th = [_mp(PLY_THICKNESS)] * len(angles)
        A, B, D, As = abd_anchor(ply, angles, th, shear_correction="1")
        lam = {}
        for lc, N in LOAD_CASES.items():
            eps0, kappa = clt_solve(ply, angles, th, N)
            plies = []
            for k, tdeg in enumerate(angles):
                s1, s2, t12 = ply_stress_anchor(ply, tdeg, eps0, kappa, 0.0)
                plies.append({
                    "ply": k, "angle_deg": float(tdeg),
                    "sigma1_Pa": _f(s1), "sigma2_Pa": _f(s2), "tau12_Pa": _f(t12),
                    "index": {name: _f(fn(ply, s1, s2, t12)) for name, fn in CRIT.items()},
                })
            worst = {name: max(range(len(plies)), key=lambda i: plies[i]["index"][name])
                     for name in CRIT}
            lam[lc] = {
                "N_N_per_m": N,
                "eps0": [_f(eps0[i]) for i in range(3)],
                "kappa_1_per_m": [_f(kappa[i]) for i in range(3)],
                "plies": plies,
                "fpf_ply": {name: {"ply": worst[name],
                                   "angle_deg": plies[worst[name]]["angle_deg"],
                                   "index": plies[worst[name]]["index"][name]}
                            for name in CRIT},
                "fpf_load_factor": {
                    name: (1.0 / plies[worst[name]]["index"][name]) ** (0.5 if name !=
                                                                      "max_stress" else 1.0)
                    for name in CRIT},
            }
        layups[key] = {
            "label": reg["label"], "angles_deg": angles,
            "ply_thickness_m": float(_mp(PLY_THICKNESS)),
            "A": _mat(A), "B": _mat(B), "D": _mat(D),
            "As_uncorrected": _mat(As),
            "As_k_5_6": _mat(As * _mp("5/6")),
            "load_cases": lam,
            "is_symmetric": abs(float(B[0, 0])) < 1e-300 and abs(float(B[0, 1])) < 1e-300,
        }
    # calibration identities: the index of a uniaxially loaded lamina is exactly 1
    calib = []
    singles = {
        "sigma1=+Xt": ("Xt", 1.0), "sigma1=-Xc": ("Xc", -1.0),
        "sigma2=+Yt": ("Yt", 1.0), "sigma2=-Yc": ("Yc", -1.0),
    }
    for label, (key, sign) in singles.items():
        val = _mp(ply[key]) * sign
        L = {"sigma1": val, "sigma2": mp.mpf(0), "tau12": mp.mpf(0)}
        for name, fn in CRIT.items():
            calib.append({"id": f"CALIB-{name}-{label}", "criterion": name,
                          "state": label, "index": _f(fn(ply, L["sigma1"], L["sigma2"],
                                                         L["tau12"])),
                          "exact_value": 1.0})
    tau = _mp(ply["S"])
    for name, fn in CRIT.items():
        calib.append({"id": f"CALIB-{name}-tau12=+S", "criterion": name,
                      "state": "tau12=+S",
                      "index": _f(fn(ply, mp.mpf(0), mp.mpf(0), tau)), "exact_value": 1.0})
    return {
        "schema": "w30-laminate-anchor/1",
        "generator": "benchmarks/composite_laminate_w30/symbolic_anchor.py",
        "precision": {"mpmath_dps": digits, "library": mp.__version__,
                      "python": platform.python_version()},
        "theory": {
            "route": "Q = S^-1 (compliance inversion); Qbar = T^-1 Q R "
                     "(transformation matrices); ABD by exact thickness integration",
            "plane_stress_Q_check": "compared against the closed-form power form in "
                                    "run_closure.py (route-independence check)",
            "F12_star": -0.5,
            "shear_correction": 5.0 / 6.0,
            "As_convention": "As_uncorrected is the raw integral of the rotated "
                             "transverse-shear stiffness (k = 1); As_k_5_6 applies "
                             "the FSDT shear correction factor",
        },
        "material": MATERIAL,
        "layups": layups,
        "calibration": calib,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=50)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    bundle = build(args.digits)
    # precision-stability check: rebuild at digits+30 and compare
    other = build(args.digits + 30)
    worst = 0.0
    # normalise by the matrix scale: several entries of a symmetric laminate are
    # exact zeros up to cancellation, so an element-wise denominator would report
    # 1.0 for pure round-off at 1e-37 against 1e7.
    for key, data in bundle["layups"].items():
        ref = max([abs(data[blk][r][c]) for blk in ("A", "B", "D")
                   for r in range(3) for c in range(3)]) or 1.0
        for blk in ("A", "B", "D"):
            for r in range(3):
                for c in range(3):
                    a, b = data[blk][r][c], other["layups"][key][blk][r][c]
                    worst = max(worst, abs(a - b) / ref)
    bundle["precision"]["stability_rel_max_between_dps"] = worst
    text = json.dumps(bundle, indent=1, sort_keys=True)
    Path(args.out).write_text(text)
    print(f"anchor written: {args.out}  sha256={hashlib.sha256(text.encode()).hexdigest()}")
    print(f"precision stability (dps {args.digits} vs {args.digits + 30}): {worst:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())