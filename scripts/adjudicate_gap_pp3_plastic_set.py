"""Independent adjudication of GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

The anchor module :mod:`tensorfem.plate_patch_plastic_limit` ships its own
analytic references (Navier series, first-yield scan, yield-line / Ilyushin /
strip bounds).  Its unit tests compare FE against *those* functions, so a
shared algebra error would pass unnoticed.  This script closes that gap: every
reference value below is re-derived here from scratch (own series truncation,
own quadrature, own rigid-plastic work integrals) and only the *finite element*
solver is taken from the library.

Run one mesh level per process (results are written as JSON so the levels can
be fanned out over the cluster)::

    python scripts/adjudicate_gap_pp3_plastic_set.py --nx 8 --out out/nx8.json

Reported quantities per mesh level:

``elastic``  centre deflection and surface von Mises per unit patch pressure,
             compared against the independent Navier series;
``transition`` load factor at surface first yield, compared with
             ``sigma_y / (sigma_eq per unit pressure)`` -- both same-mesh and
             raw against the independent analytic first-yield pressure;
``collapse`` numerical limit load from load control, compared with the
             independent rigid-plastic bracket ``[strip_lb, min(yl, sine)]``;
``permanent`` residual centre deflection after unloading to zero.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.plate_patch_plastic_limit import run_patch_plastic_path  # noqa: E402

# --- anchor case (docs/gap_pp3_ice_struct_plastic_set_verification.md) -------
A = 1.0
B = 1.0
T = 0.02
E = 210.0e9
NU = 0.3
SY = 355.0e6          # DH36 at -20 C, see the low-temperature table
PATCH = (0.5 * A, 0.5 * B)


# ---------------------------------------------------------------------------
# Independent analytic references (deliberately not the module's functions)
# ---------------------------------------------------------------------------

def mp_independent(t: float = T, sy: float = SY) -> float:
    """Fully plastic moment per unit width, m_p = sigma_y t^2 / 4."""
    return sy * t * t / 4.0


def navier_centre(a: float = A, b: float = B, t: float = T, e: float = E,
                  nu: float = NU, patch: tuple[float, float] = PATCH,
                  terms: int = 81) -> dict:
    """Navier double series for a centrally patched SS plate, per unit pressure.

    Written directly from the load expansion
    ``q_mn = 4 q0 Δcos_x Δcos_y / (pi^2 m n)`` with
    ``Δcos_x = cos(m pi x1/a) - cos(m pi x2/a)``; moments are recovered from
    ``M = q_mn (alpha^2 + nu beta^2)/lamda^2 sin sin``.
    """
    d = e * t ** 3 / (12.0 * (1.0 - nu * nu))
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    w = mx = my = 0.0
    for m in range(1, terms + 1, 2):
        am = m * math.pi / a
        dx = math.cos(am * x1) - math.cos(am * x2)
        for n in range(1, terms + 1, 2):
            bn = n * math.pi / b
            dy = math.cos(bn * y1) - math.cos(bn * y2)
            qmn = 4.0 * dx * dy / (math.pi ** 2 * m * n)
            lam = am * am + bn * bn
            s = math.sin(am * 0.5 * a) * math.sin(bn * 0.5 * b)
            w += qmn / (d * lam * lam) * s
            mx += qmn * (am * am + nu * bn * bn) / (lam * lam) * s
            my += qmn * (bn * bn + nu * am * am) / (lam * lam) * s
    def mises(mxx: float, myy: float, t: float = t) -> float:
        sx, sy_ = 6.0 * mxx / (t * t), 6.0 * myy / (t * t)
        return math.sqrt(max(sx * sx - sx * sy_ + sy_ * sy_, 0.0))
    return {"w_per_pa": w, "Mx_per_pa": mx, "My_per_pa": my,
            "sigma_eq_per_pa": mises(mx, my)}


def first_yield_pressure_independent(a: float = A, b: float = B, t: float = T,
                                     e: float = E, nu: float = NU,
                                     sy: float = SY,
                                     patch: tuple[float, float] = PATCH,
                                     grid: int = 41, terms: int = 61) -> dict:
    """Smallest pressure that reaches sigma_y anywhere on the surface.

    The Navier field is sampled on a ``grid x grid`` node-centred lattice; the
    peak surface Mises per unit pressure ``S`` gives ``p_y = sigma_y / S``.
    """
    d = e * t ** 3 / (12.0 * (1.0 - nu * nu))
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    xs = [a * k / (grid - 1) for k in range(grid)]
    ys = [b * k / (grid - 1) for k in range(grid)]
    peak = 0.0
    where = (0.0, 0.0)
    for m in range(1, terms + 1, 2):
        am = m * math.pi / a
        dx = math.cos(am * x1) - math.cos(am * x2)
        for n in range(1, terms + 1, 2):
            bn = n * math.pi / b
            dy = math.cos(bn * y1) - math.cos(bn * y2)
            qmn = 4.0 * dx * dy / (math.pi ** 2 * m * n)
            lam = am * am + bn * bn
            cx = qmn * (am * am + nu * bn * bn) / (lam * lam)
            cy = qmn * (bn * bn + nu * am * am) / (lam * lam)
            si = [math.sin(am * x) for x in xs]
            sj = [math.sin(bn * y) for y in ys]
            for i in range(grid):
                for j in range(grid):
                    s = si[i] * sj[j]
                    mx = cx * s
                    my = cy * s
                    sx, syv = 6.0 * mx / (t * t), 6.0 * my / (t * t)
                    eq = math.sqrt(max(sx * sx - sx * syv + syv * syv, 0.0))
                    if eq > peak:
                        peak, where = eq, (xs[i], ys[j])
    return {"sigma_eq_per_pa": peak, "pressure": sy / peak, "at": where,
            "grid": grid, "terms": terms}


def _mechanism_volume_ratio(a: float, b: float, patch: tuple[float, float],
                            samples: int = 801) -> float:
    """``int_patch (1 - 2 max(|x|/a, |y|/b)) dA`` by midpoint quadrature.

    ``w = w0 (1 - 2 max(|x|/a, |y|/b))`` is the four-triangle pyramid whose
    kinks run from the plate centre to the corners; the normal-rotation jump
    across each kink gives the dissipation used below.
    """
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    dx = (x2 - x1) / samples
    dy = (y2 - y1) / samples
    total = 0.0
    for i in range(samples):
        x = x1 + (i + 0.5) * dx - 0.5 * a         # coordinates about the centre
        ax = abs(x) / a
        row = 0.0
        for j in range(samples):
            y = y1 + (j + 0.5) * dy - 0.5 * b
            row += 1.0 - 2.0 * max(ax, abs(y) / b)
        total += row * dx * dy
    return total


def yield_line_upper_bound_independent(a: float = A, b: float = B, t: float = T,
                                       sy: float = SY,
                                       patch: tuple[float, float] = PATCH) -> float:
    """Four-triangle pyramid mechanism, own quadrature.

    Dissipation: on the kink from the centre to a corner the normal rotation
    jumps by ``w0 (a/b + b/a)/L`` over length ``L = sqrt(a^2+b^2)/2``, so each
    of the four kinks dissipates ``m_p w0 (a/b + b/a)``.
    """
    internal = 4.0 * mp_independent(t, sy) * (a / b + b / a)
    vol = _mechanism_volume_ratio(a, b, patch)
    return internal / vol


def sine_upper_bound_independent(a: float = A, b: float = B, t: float = T,
                                 sy: float = SY,
                                 patch: tuple[float, float] = PATCH) -> float:
    """Double-sine mode ``w0 sin(pi x/a) sin(pi y/b)``, own integrals.

    ``W_int = 4 m_p a b w0 (1/a^2 + 1/b^2)`` (both curvatures same sign, so the
    Johansen dissipation is ``m_p (|k1| + |k2|)``); ``W_ext = q w0 (a/pi)(b/pi)
    dcos_x dcos_y``.
    """
    u, v = patch
    x1, x2 = 0.5 * (a - u), 0.5 * (a + u)
    y1, y2 = 0.5 * (b - v), 0.5 * (b + v)
    internal = 4.0 * mp_independent(t, sy) * a * b * (1.0 / (a * a) + 1.0 / (b * b))
    ext = ((a / math.pi) * (math.cos(math.pi * x1 / a) - math.cos(math.pi * x2 / a))
           * (b / math.pi) * (math.cos(math.pi * y1 / b) - math.cos(math.pi * y2 / b)))
    return internal / ext


def strip_lower_bound_independent(a: float = A, b: float = B, t: float = T,
                                  sy: float = SY,
                                  patch: tuple[float, float] = PATCH) -> float:
    """Two-way strip (statically admissible) field: ``8 m_p / (u(2a-u))`` etc."""
    mp = mp_independent(t, sy)
    u, v = patch
    return 8.0 * mp / (u * (2.0 * a - u)) + 8.0 * mp / (v * (2.0 * b - v))


# ---------------------------------------------------------------------------
# FE driver
# ---------------------------------------------------------------------------

def _interp_crossing(steps: list[dict], key: str, target: float) -> float | None:
    """Own first-crossing interpolation (independent of PatchPath properties)."""
    prev = None
    for s in steps:
        if prev is not None and prev[key] < target <= s[key]:
            frac = (target - prev[key]) / (s[key] - prev[key])
            return prev["factor"] + frac * (s["factor"] - prev["factor"])
        if prev is None and s[key] >= target:
            return s["factor"]
        prev = s
    return None


def run_level(nx: int, *, hardening: float, layers: int = 3,
              fractions: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85, 0.95,
                                              1.0, 1.05, 1.10, 1.15, 1.20),
              elastic_probe: float = 0.05, unload_steps: int = 4) -> dict:
    """One mesh level: elastic probe, ascending ramp, unload from the peak.

    Loads are prescribed as fractions of the *independent* double-sine upper
    bound, so the ramp brackets the rigid-plastic collapse window instead of an
    arbitrary multiple of the first-yield pressure.
    """
    py = first_yield_pressure_independent()
    ref = py["pressure"]
    sine = sine_upper_bound_independent()
    strip = strip_lower_bound_independent()
    yl = yield_line_upper_bound_independent()

    probe = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=SY,
                                   hardening=hardening, patch=PATCH,
                                   load_factors=[elastic_probe * sine],
                                   nx=nx, ny=nx, layers=layers)
    p0 = probe.loading[0]
    fe_sigma_per_pa = p0.max_surface_mises / p0.load_factor
    fe_w_per_pa = p0.centre_deflection / p0.load_factor

    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=SY,
                                  hardening=hardening, patch=PATCH,
                                  load_factors=[f * sine for f in fractions],
                                  nx=nx, ny=nx, layers=layers,
                                  unload_steps=unload_steps)
    loading = [{"factor": s.load_factor, "w": s.centre_deflection,
                "w_over_t": s.centre_deflection / T,
                "sigma": s.max_surface_mises, "alpha": s.max_alpha}
               for s in path.loading]
    unloading = [{"factor": s.load_factor, "w": s.centre_deflection}
                 for s in path.unloading]

    # incremental tangent stiffness (normalised by the elastic one) -- the
    # quantity that actually identifies the bending mechanism, whereas the
    # "last converged load factor" only reports the end of the ramp.
    k0 = None
    for i in range(1, len(loading)):
        dw = loading[i]["w"] - loading[i - 1]["w"]
        dp = loading[i]["factor"] - loading[i - 1]["factor"]
        if dw > 0.0 and dp > 0.0:
            k0 = dp / dw
            break
    tangent = []
    for i in range(1, len(loading)):
        dw = loading[i]["w"] - loading[i - 1]["w"]
        dp = loading[i]["factor"] - loading[i - 1]["factor"]
        if dw <= 0.0:
            tangent.append({"factor": loading[i]["factor"], "k": 0.0})
        else:
            tangent.append({"factor": loading[i]["factor"],
                            "k": (dp / dw) / k0 if k0 else None})
    knee = next((t["factor"] for t in tangent if t["k"] is not None and t["k"] < 0.05),
                None)
    last = loading[-1]["factor"] if loading else None
    nav = navier_centre()
    return {
        "nx": nx, "layers": layers, "hardening": hardening,
        "analytic": {
            "mp": mp_independent(),
            "py": ref,
            "sigma_eq_per_pa": py["sigma_eq_per_pa"],
            "w_per_pa_navier": nav["w_per_pa"],
            "yield_line_ub": yl,
            "sine_ub": sine,
            "strip_lb": strip,
        },
        "elastic": {"fe_w_per_pa": fe_w_per_pa, "fe_sigma_per_pa": fe_sigma_per_pa,
                    "same_mesh_py": SY / fe_sigma_per_pa,
                    "w_bias_pct": 100.0 * (fe_w_per_pa / nav["w_per_pa"] - 1.0),
                    "sigma_bias_pct": 100.0 * (fe_sigma_per_pa / py["sigma_eq_per_pa"] - 1.0)},
        "loading": loading, "unloading": unloading, "tangent": tangent,
        "ramp_max": fractions[-1] * sine,
        "collapse_last_converged": last,
        "collapse_knee": knee,
        "ramp_completed": bool(last is not None and last >= fractions[-1] * sine - 1e-9),
        "first_yield_fe": _interp_crossing(loading, "sigma", SY),
        "permanent": (unloading[-1]["w"] if unloading else None),
        "unload_converged": bool(len(unloading) == unload_steps),
        "peak_w": loading[-1]["w"] if loading else None,
        "module_props": {"first_yield_factor": path.first_yield_factor,
                         "permanent_deflection": path.permanent_deflection,
                         "collapse_factor": path.collapse_factor},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--hardening", type=float, default=1.0e6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = run_level(args.nx, hardening=args.hardening, layers=args.layers)
    text = json.dumps(out, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())