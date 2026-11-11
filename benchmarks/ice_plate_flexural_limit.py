#!/usr/bin/env python3
"""H3-ICE-PLATE-FEM: flexural (first-crack) limit load of a floating ice sheet.

Produces the acceptance evidence of the ice-plate bending + failure-criterion
anchor:

1. **Convergence** -- FE centre moment vs the exact Kelvin patch moment
   (:func:`tensorfem.levelset.plate_crack.patch_max_moment`) for four meshes.
2. **Limit load vs analytic plate theory** -- first-crack load
   ``P_f = sigma_f h^2 / (6 m_hat)`` from the FE moment field against the Kelvin
   reference and against the Westergaard/Nevel closed form
   ``(1+nu)(P/4 pi)(ln(l/a) + 0.6159)``.
3. **Nevel validity window** -- its own error against the exact solution, which
   is what limits the <10% acceptance window in ``a/l``.
4. **Failure location and mode** -- where the critical tensile fibre sits, which
   fibre it is, and the crack orientation implied by the polar moments.

Everything is written to ``benchmarks/artifacts/ice_plate_flexural_limit/``
(JSON evidence + PNG figures).  Runtime is ~2-3 min on one core; no cluster job.

Usage
-----
    python benchmarks/ice_plate_flexural_limit.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "benchmarks" / "artifacts" / "ice_plate_flexural_limit"
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.ice_plate_flexural_limit import (  # noqa: E402
    characteristic_moment_per_load,
    failure_mode,
    first_crack_load,
    nevel_moment_applicability,
    solve_circular_patch,
)
from tensorfem.levelset.plate_crack import patch_max_moment  # noqa: E402

# Ice sheet + water (same defaults as tensorfem.levelset.ice_plate_fracture.IceSheet)
SHEET = dict(
    young=7.0e9,        # Pa
    poisson=0.30,
    thickness=0.30,     # m
    foundation=1025.0 * 9.81,   # rho_w g, N/m^3
)
FLEXURAL_STRENGTH = 6.0e5   # Pa, ISO 19906 typical columnar freshwater ice
LENGTH_SCALE = None         # filled in main()


def _log(msg: str) -> None:
    print(msg, flush=True)


def convergence_table() -> list[dict]:
    """FE centre moment vs the Kelvin anchor on four meshes, two patch sizes."""
    rows = []
    ell = LENGTH_SCALE
    for frac in (0.3, 1.0):
        a = frac * ell
        anchor = patch_max_moment(1.0, a, _rigidity(), SHEET["foundation"], SHEET["poisson"])
        for cells, beta in ((40, 4.0), (80, 5.0), (120, 6.0), (160, 8.0)):
            t0 = time.time()
            sol = solve_circular_patch(patch_radius=a, cells=cells, beta=beta, **SHEET)
            specs = characteristic_moment_per_load(sol)
            rows.append(
                {
                    "radius_over_length": frac,
                    "cells": cells,
                    "beta": beta,
                    "centre_moment": specs["centre_moment"],
                    "kelvin_anchor": anchor,
                    "centre_moment_error": specs["centre_moment"] / anchor - 1.0,
                    "load_error": sol.total_load
                    / (math.pi * a ** 2 / 4.0)
                    - 1.0,
                    "element_size_at_centre": sol.metadata["element_size_at_centre"],
                    "seconds": round(time.time() - t0, 2),
                }
            )
            _log(
                f"  convergence a/l={frac} cells={cells:3d} beta={beta:>4} "
                f"M_c err={100 * rows[-1]['centre_moment_error']:+.3f}% "
                f"load err={100 * rows[-1]['load_error']:+.4f}%"
            )
    return rows


def limit_load_table(fracs=(0.1, 0.2, 0.3, 0.5, 1.0, 2.0)) -> list[dict]:
    rows = []
    for frac in fracs:
        row = first_crack_load(
            patch_radius=frac * LENGTH_SCALE,
            flexural_strength=FLEXURAL_STRENGTH,
            cells=80,
            beta=5.0,
            **SHEET,
        )
        row["radius_over_length"] = frac
        rows.append(row)
        _log(
            f"  limit load a/l={frac:>4}: FEM={row['first_crack_load_fem']/1e3:8.2f} kN "
            f"Kelvin={row['first_crack_load_kelvin']/1e3:8.2f} kN "
            f"Nevel={row['first_crack_load_nevel']/1e3:8.2f} kN "
            f"err(Kelvin)={100 * row['error_vs_kelvin']:+.3f}% "
            f"err(Nevel)={100 * row['error_vs_nevel']:+.3f}%"
        )
    return rows


def mode_table(fracs=(0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)) -> list[dict]:
    rows = []
    for frac in fracs:
        out = failure_mode(patch_radius=frac * LENGTH_SCALE, cells=80, beta=5.0, **SHEET)
        sol = out.pop("solution")
        row = {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
               for k, v in out.items()}
        row["radius_over_length"] = frac
        row["radius_over_length"] = frac
        row["uniform_pressure_moment_per_load"] = characteristic_moment_per_load(sol)["m_hat"]
        rows.append(row)
        _log(
            f"  mode a/l={frac:>4}: fibre={row['critical_fibre']:>6} "
            f"r_crit/a={row['critical_radius_over_patch']:.3f} "
            f"top/bottom={row['top_over_bottom']:.3f} "
            f"Mtheta/Mr={row['hoop_over_radial']:.3f} -> {row['mode']}"
        )
    return rows


def mode_mesh_check(fracs=(3.0, 4.0, 6.0)) -> list[dict]:
    """Mesh sensitivity of the critical location and of the polar ratio."""
    rows = []
    for frac in fracs:
        coarse = failure_mode(patch_radius=frac * LENGTH_SCALE, cells=80, beta=5.0, **SHEET)
        fine = failure_mode(patch_radius=frac * LENGTH_SCALE, cells=120, beta=6.0, **SHEET)
        coarse.pop("solution"), fine.pop("solution")
        rows.append(
            {
                "radius_over_length": frac,
                "cells": [80, 120],
                "critical_radius_over_patch": [
                    coarse["critical_radius_over_patch"], fine["critical_radius_over_patch"]
                ],
                "hoop_over_radial": [coarse["hoop_over_radial"], fine["hoop_over_radial"]],
                "top_over_bottom": [coarse["top_over_bottom"], fine["top_over_bottom"]],
                "mode": [coarse["mode"], fine["mode"]],
                "critical_fibre": [coarse["critical_fibre"], fine["critical_fibre"]],
            }
        )
        _log(
            f"  mode mesh check a/l={frac}: r_crit/a "
            f"{coarse['critical_radius_over_patch']:.3f} -> "
            f"{fine['critical_radius_over_patch']:.3f}, mode {coarse['mode']} -> {fine['mode']}"
        )
    return rows


def polar_profiles(fracs=(0.3, 3.0), rings: int = 60) -> dict[str, list[dict]]:
    """Ring-averaged ``M_r``, ``M_theta`` normalised by ``P`` (the shape evidence)."""
    profiles: dict[str, list[dict]] = {}
    for frac in fracs:
        a = frac * LENGTH_SCALE
        sol = solve_circular_patch(patch_radius=a, cells=120 if frac >= 2.0 else 80,
                                   beta=6.0 if frac >= 2.0 else 5.0, **SHEET)
        polar = sol.polar_fields()
        r = polar["r"] / sol.length_scale
        p_full = sol.full_load
        edges = np.linspace(0.0, 8.0, rings + 1)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (r >= lo) & (r < hi)
            if not np.any(m):
                continue
            rows.append(
                {
                    "r_over_length": float(0.5 * (lo + hi)),
                    "M_r_over_P": float(np.mean(polar["M_r"][m]) / p_full),
                    "M_theta_over_P": float(np.mean(polar["M_theta"][m]) / p_full),
                    "M_r_ring_scatter": float(np.ptp(polar["M_r"][m]) / max(
                        abs(float(np.mean(polar["M_r"][m]))), 1e-30)),
                }
            )
        profiles[f"{frac:g}"] = rows
        _log(f"  polar profile a/l={frac}: {len(rows)} rings")
    return profiles


def _rigidity() -> float:
    e, nu, h = SHEET["young"], SHEET["poisson"], SHEET["thickness"]
    return e * h ** 3 / (12.0 * (1.0 - nu ** 2))


def _figure(evidence: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rows = evidence["limit_load"]
    frac = [r["radius_over_length"] for r in rows]
    ax.plot(frac, [r["first_crack_load_fem"] / 1e3 for r in rows], "o-",
            label="FEM moment field + Rankine criterion", lw=2)
    ax.plot(frac, [r["first_crack_load_kelvin"] / 1e3 for r in rows], "s--",
            label="exact Kelvin patch moment", lw=1.4)
    ax.plot(frac, [r["first_crack_load_nevel"] / 1e3 for r in rows], "^:",
            label="Westergaard/Nevel closed form", lw=1.4)
    ax.axvspan(0.05, 0.5, color="tab:green", alpha=0.08,
               label="Nevel form within 2% of exact")
    ax.set_xscale("log")
    ax.set_xlabel("patch radius a / characteristic length l")
    ax.set_ylabel("first-crack load  $P_f$  [kN]")
    ax.set_title("Ice sheet first-crack (flexural limit) load, h=0.30 m, "
                 r"$\sigma_f$=0.6 MPa")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = ART / "limit_load_vs_radius.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=False)
    for k, (key, prof) in enumerate(evidence["polar_profiles"].items()):
        rr = [p["r_over_length"] for p in prof]
        ax[k].plot(rr, [p["M_r_over_P"] for p in prof], "-", label="$M_r$")
        ax[k].plot(rr, [p["M_theta_over_P"] for p in prof], "--", label="$M_\\theta$")
        ax[k].axvline(float(key), color="tab:red", ls=":", lw=1.2, label="patch edge $a$")
        ax[k].set_title(f"a/l = {key}")
        ax[k].set_xlabel("r / l")
        ax[k].set_ylabel("moment per unit load  [m]")
        ax[k].grid(alpha=0.3)
        ax[k].legend(fontsize=8)
    fig.suptitle("Polar moment distribution on the Winkler foundation "
                 "(load centre = peak $M_r=M_\\theta$ for small patches)")
    path = ART / "polar_moment_profiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, colour in ((0.3, "tab:blue"), (1.0, "tab:orange")):
        rows = [r for r in evidence["convergence"] if r["radius_over_length"] == key]
        ax.plot([r["cells"] for r in rows],
                [abs(r["centre_moment_error"]) * 100 for r in rows], "o-",
                color=colour, label=f"a/l = {key}")
    ax.set_xlabel("elements per side (quarter model)")
    ax.set_ylabel("|centre moment error| vs Kelvin anchor [%]")
    ax.set_yscale("log")
    ax.set_title("Mesh convergence of the FE patch moment")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = ART / "convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)
    return made


def main() -> int:
    global LENGTH_SCALE
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    LENGTH_SCALE = (
        _rigidity() / (SHEET["foundation"])
    ) ** 0.25

    _log("H3-ICE-PLATE-FEM | ice plate bending + first-crack criterion")
    _log(f"  D = {_rigidity():.6e} N m, l = (D/k)^1/4 = {LENGTH_SCALE:.4f} m, "
         f"sigma_f = {FLEXURAL_STRENGTH/1e6:.2f} MPa")

    _log("[1/4] convergence of the FE patch moment")
    convergence = convergence_table()
    _log("[2/4] first-crack limit load vs analytic plate theory")
    limit_load = limit_load_table()
    _log("[3/4] failure location and mode")
    modes = mode_table()
    _log("[4/4] polar moment profiles + mode mesh check")
    profiles = polar_profiles()
    mode_check = mode_mesh_check()

    evidence = {
        "case": "H3-ICE-PLATE-FEM",
        "what": "elastic-foundation ice plate bending with an independent "
                "bottom-surface tensile strength (first-crack) criterion",
        "sheet": {**SHEET, "flexural_strength": FLEXURAL_STRENGTH,
                  "rigidity": _rigidity(), "length_scale": LENGTH_SCALE},
        "convergence": convergence,
        "limit_load": limit_load,
        "nevel_applicability": nevel_moment_applicability(**SHEET),
        "modes": modes,
        "modes_mesh_check": mode_check,
        "polar_profiles": profiles,
        "external_anchor": {
            "kelvin_patch_moment": "tensorfem.levelset.plate_crack.patch_max_moment -- "
                                   "exact Kelvin-function centre moment of a circular "
                                   "patch on a Winkler foundation, "
                                   "M_c = (1+nu)/2 p l^2 X kei'(X)",
            "nevel_form": "Westergaard/Nevel closed form "
                          "(1+nu)(P/4 pi)(ln(l/a)+0.6159), implemented as "
                          "tensorfem.levelset.plate_crack.patch_max_moment_nevel "
                          "(Nevel 1969; Kerr 1979 review; ISO 19906:2010 flexural "
                          "strength criterion sigma = 6M/h^2)",
            "foundation_kernel": "tensorfem.foundation_plate_bending "
                                 "(GAP-FEM-WINKLER), used unchanged",
            "acceptance": "first-crack load within 10% of the analytic reference",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "repro_cmd": "python benchmarks/ice_plate_flexural_limit.py",
        "runtime_seconds": round(time.time() - t0, 1),
    }

    figures = _figure(evidence)
    evidence["figures"] = figures
    script = Path(__file__).resolve()
    evidence["script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
    try:
        evidence["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            check=False,
        ).stdout.strip()
        evidence["dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
            check=False,
        ).stdout.strip())
    except OSError:  # pragma: no cover
        pass

    out = ART / "evidence.json"
    out.write_text(json.dumps(evidence, indent=2))
    _log(f"wrote {out} ({out.stat().st_size} bytes) and {len(figures)} figures")
    _log(f"runtime {evidence['runtime_seconds']} s")

    worst = max(abs(r["error_vs_kelvin"]) for r in limit_load)
    worst_nevel = max(abs(r["error_vs_nevel"]) for r in limit_load
                      if r["radius_over_length"] <= 0.5)
    _log(f"PASS/FAIL | max |error vs Kelvin| = {100*worst:.3f}% "
         f"(a/l<=0.5 vs Nevel: {100*worst_nevel:.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())