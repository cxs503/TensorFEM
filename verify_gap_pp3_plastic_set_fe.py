#!/usr/bin/env python
"""FE verification driver for GAP-PP-3-ICE-STRUCT-PLASTIC-SET (debt closure).

The adversarial audit (docs/AUDIT-GAP-PP-3-ICE-STRUCT-PLASTIC-SET.md) repaired
the reference solutions and the acceptance drivers (F1-F5) but left the
post-fix FE numbers as *not_run*: the 3-grid lateral matrix (transition +
limit + permanent deflection at 8x4/12x6/16x8) was submitted as cj3469 and
that job died with ModuleNotFoundError (rc=1) before producing anything.

This driver closes that debt.  It runs, on the repaired code path:

  1. Elastic->plastic transition (uniaxial + lateral) on 8x4/12x6/16x8
     against the analytical yield load                          (criterion < 5 %)
  2. Plastic limit load (lateral) on 8x4/12x6/16x8
     against the yield-line upper bound                         (criterion < 10 %)
  3. Three-grid convergence of limit_load and max_deflection
  4. Permanent-deflection protocol:
       - below first yield  -> permanent deflection exactly 0
       - above first yield  -> > 0 and monotone in peak load

Outputs (relative to the repo root):
  artifacts/gap_pp3_plastic_set_fe/results.json
  artifacts/gap_pp3_plastic_set_fe/fig_transition.png
  artifacts/gap_pp3_plastic_set_fe/fig_limit.png
  artifacts/gap_pp3_plastic_set_fe/fig_convergence.png
  artifacts/gap_pp3_plastic_set_fe/fig_permanent.png

Every number in results.json is produced by the code path under test;
nothing is hand-typed.  A subtask that fails is recorded with its exception,
never silently dropped.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
import torch

from tensorfem.ice_plate_plasticity import (
    _newton_load_control,
    compute_permanent_deflection,
    elastic_first_yield_pressure,
    extract_max_permanent_deflection,
    flow_stress_at_temperature,
    patch_load_plate_mesh,
    verify_elastic_plastic_transition,
    verify_mesh_convergence,
    verify_plastic_limit_load,
    yield_line_limit_pressure,
)
from tensorfem.layered_shell4_plasticity import LayeredShell4State

OUT = os.path.join(REPO, "artifacts", "gap_pp3_plastic_set_fe")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Case definition (representative ice-structure panel, same as the audit)
# ---------------------------------------------------------------------------
CASE = dict(
    a=2.0,            # plate length [m]
    b=1.0,            # plate width [m]
    t=0.020,          # thickness [m]
    grade="DH36",
    temperature=-40.0,
    young=206e9,
    poisson=0.3,
    hardening=1000e6,
    patch=(0.75, 1.25, 0.25, 0.75),  # 0.5 x 0.5 m centred patch
)
GRIDS = ((8, 4), (12, 6), (16, 8))

SY = flow_stress_at_temperature(CASE["grade"], CASE["temperature"])
PY = elastic_first_yield_pressure(
    CASE["a"], CASE["b"], CASE["t"], SY, CASE["poisson"])
P_UB = yield_line_limit_pressure(CASE["a"], CASE["b"], SY, CASE["t"])

results = {
    "case": {**CASE, "yield_stress_Pa": SY,
             "first_yield_pressure_Pa": PY,
             "yield_line_upper_bound_Pa": P_UB},
    "transition": {},
    "limit": {},
    "convergence": {},
    "permanent": {},
    "errors": {},
}


def _tag(loading, nx, ny):
    return f"{loading}_{nx}x{ny}"


def _run(fn, key, **kw):
    """Run one subtask; record the result or the exception, never crash."""
    t0 = time.time()
    try:
        res = fn(**kw)
        parts = key.split(".")
        results[parts[0]][".".join(parts[1:])] = res
        print(f"[ok] {key} in {time.time()-t0:.1f}s", flush=True)
        return res
    except Exception:
        results["errors"][key] = traceback.format_exc()
        print(f"[FAIL] {key}:\n{traceback.format_exc()}", flush=True)
        return None


# ---------------------------------------------------------------------------
# 1. Elastic->plastic transition, 3 grids x 2 loadings
# ---------------------------------------------------------------------------
for loading in ("uniaxial", "lateral"):
    for nx, ny in GRIDS:
        def _t(loading=loading, nx=nx, ny=ny):
            return verify_elastic_plastic_transition(
                CASE["a"], CASE["b"], CASE["t"], CASE["young"],
                CASE["poisson"], SY, CASE["hardening"], nx=nx, ny=ny,
                loading=loading,
                patch_x0=CASE["patch"][0], patch_x1=CASE["patch"][1],
                patch_y0=CASE["patch"][2], patch_y1=CASE["patch"][3])
        _run(_t, f"transition.{_tag(loading, nx, ny)}")

# ---------------------------------------------------------------------------
# 2. Plastic limit load (lateral), 3 grids
# ---------------------------------------------------------------------------
for nx, ny in GRIDS:
    def _l(nx=nx, ny=ny):
        return verify_plastic_limit_load(
            CASE["a"], CASE["b"], CASE["t"], CASE["young"], CASE["poisson"],
            SY, CASE["hardening"], nx=nx, ny=ny, loading="lateral",
            patch_x0=CASE["patch"][0], patch_x1=CASE["patch"][1],
            patch_y0=CASE["patch"][2], patch_y1=CASE["patch"][3])
    _run(_l, f"limit.{_tag('lateral', nx, ny)}")

# ---------------------------------------------------------------------------
# 3. Three-grid convergence (limit_load + max_deflection, lateral)
# ---------------------------------------------------------------------------
for quantity in ("limit_load", "max_deflection"):
    def _c(quantity=quantity):
        return verify_mesh_convergence(
            CASE["a"], CASE["b"], CASE["t"], CASE["young"], CASE["poisson"],
            SY, CASE["hardening"], mesh_configs=GRIDS,
            loading="lateral", quantity=quantity)
    _run(_c, f"convergence.{quantity}")

# ---------------------------------------------------------------------------
# 4. Permanent deflection protocol
# ---------------------------------------------------------------------------
def _load_to_factor(nx, ny, target_factor, n_steps=25):
    """Incrementally load the lateral patch case to target_factor (Pa of
    reference pressure); return (model, info, u, state, all_converged)."""
    model, info = patch_load_plate_mesh(
        CASE["a"], CASE["b"], CASE["t"], nx, ny,
        CASE["young"], CASE["poisson"], SY, CASE["hardening"],
        CASE["patch"][0], CASE["patch"][1], CASE["patch"][2], CASE["patch"][3])
    reference_load = info["reference_load"]
    fixed_dofs = info["fixed_dofs"]
    ndof = model.n_dofs
    device = model.nodes.device
    fixed = fixed_dofs.to(device=device, dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool, device=device)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    u = torch.zeros(ndof, dtype=model.nodes.dtype, device=device)
    state = LayeredShell4State.virgin(model)
    for step in range(1, n_steps + 1):
        load_factor = step / n_steps * target_factor
        external = reference_load * load_factor
        u_new, state_new, conv = _newton_load_control(
            model, external, u, state, free, max_iter=25)
        if not conv:
            return model, info, u, state, False
        u, state = u_new, state_new
    return model, info, u, state, True


# Peak pressures are scaled on the yield-line upper bound P_UB (not PY): the
# patch FE first-yield sits between PY and P_UB and is mesh-dependent, so a
# fraction of P_UB is a mesh-independent way to bracket the plastic regime.
PEAKS = (("below_yield_0p5", 0.5),
         ("at_upper_bound_1p0", 1.0),
         ("above_upper_bound_1p2", 1.2))

for nx, ny in GRIDS:
    for label, frac in PEAKS:
        def _p(nx=nx, ny=ny, label=label, frac=frac):
            model, info, u, state, conv = _load_to_factor(nx, ny,
                                                           frac * P_UB)
            if not conv:
                return {"converged": False, "peak_factor": frac,
                        "permanent_deflection_m": float("nan")}
            u_perm = compute_permanent_deflection(model, state,
                                                   info["fixed_dofs"])
            w = extract_max_permanent_deflection(u_perm, model.nodes)
            return {"converged": True, "peak_factor": frac,
                    "peak_pressure_Pa": frac * P_UB,
                    "permanent_deflection_m": w}
        _run(_p, f"permanent.{_tag('lateral', nx, ny)}.{label}")

# ---------------------------------------------------------------------------
# Serialise (dataclasses -> plain dicts)
# ---------------------------------------------------------------------------
def _plain(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _plain(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, torch.Tensor):
        return _plain(obj.tolist())
    return str(obj)


with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(_plain(results), f, indent=2)
print(f"[done] results -> {OUT}/results.json", flush=True)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
if not HAVE_MPL:
    print("[skip] matplotlib unavailable on this node; figures will be "
          "generated locally from results.json", flush=True)
    raise SystemExit(0)

def _mpa(p_pa):
    return p_pa / 1e6


def _get(section, key, field):
    try:
        return results[section][key][field]
    except (KeyError, TypeError):
        return float("nan")


# Fig 1: transition, FE vs analytical per grid
fig, ax = plt.subplots(figsize=(7, 4.5))
names = [f"{nx}x{ny}" for nx, ny in GRIDS]
fe_uni = [_get("transition", f"uniaxial_{n}", "fe_yield_load") / 1e6
           for n in names]
fe_lat = [_get("transition", f"lateral_{n}", "fe_yield_load") / 1e6
          for n in names]
x = range(len(names))
ax.bar([i - 0.2 for i in x], fe_uni, width=0.4, label="FE uniaxial")
ax.bar([i + 0.2 for i in x], fe_lat, width=0.4, label="FE lateral")
ax.axhline(_mpa(SY), color="k", ls="--", lw=1)
ax.text(0.02, _mpa(SY) * 1.01, f"analytical = {_mpa(SY):.0f} MPa",
        fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("yield load [MPa]")
ax.set_title("Elastic->plastic transition vs analytical yield load")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_transition.png"), dpi=140)
plt.close(fig)

# Fig 2: limit load, FE vs yield-line upper bound per grid
fig, ax = plt.subplots(figsize=(7, 4.5))
fe_lim = [_get("limit", f"lateral_{n}", "fe_limit") / 1e6 for n in names]
ax.bar([i - 0.2 for i in x], fe_lim, width=0.4, label="FE limit (hardening)")
ax.axhline(_mpa(P_UB), color="r", ls="--", lw=1)
ax.text(0.02, _mpa(P_UB) * 1.01,
        f"yield-line upper bound = {_mpa(P_UB):.3f} MPa",
        color="r", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylabel("limit pressure [MPa]")
ax.set_title("Plastic limit pressure vs yield-line upper bound")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_limit.png"), dpi=140)
plt.close(fig)

# Fig 3: 3-grid convergence
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, quantity in zip(axes, ("limit_load", "max_deflection")):
    conv = results["convergence"].get(quantity)
    if conv is None:
        ax.text(0.5, 0.5, "not run", ha="center")
        continue
    ne = [m["n_elements"] for m in conv["meshes"]]
    vals = [v / 1e6 if quantity == "limit_load" else v * 1e3
            for v in conv["values"]]
    ax.plot(ne, vals, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("n elements")
    ax.set_ylabel("limit [MPa]" if quantity == "limit_load"
                 else "max deflection [mm]")
    ax.set_title(f"{quantity} (passed={conv['passed']})")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=140)
plt.close(fig)

# Fig 4: permanent deflection vs peak load factor
fig, ax = plt.subplots(figsize=(7, 4.5))
for nx, ny in GRIDS:
    w = [_get("permanent", f"lateral_{nx}x{ny}.{l}", "permanent_deflection_m")
          for l, _ in PEAKS]
    ax.plot([f * _mpa(P_UB) for _, f in PEAKS], [w * 1e3 for w in w],
            marker="o", label=f"{nx}x{ny}")
ax.axvline(_mpa(P_UB), color="r", ls=":", lw=1)
ax.text(_mpa(P_UB) * 1.01, ax.get_ylim()[1] * 0.9,
        "yield-line upper bound", color="r", fontsize=8)
ax.set_xlabel("peak patch pressure [MPa]")
ax.set_ylabel("max permanent deflection [mm]")
ax.set_title("Permanent deflection after full unloading")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_permanent.png"), dpi=140)
plt.close(fig)

print("[done] figures written", flush=True)
