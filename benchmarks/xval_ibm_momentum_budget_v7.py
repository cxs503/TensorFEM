#!/usr/bin/env python3
"""XVAL v7 -- momentum budget, epsilon attribution and constraint audit (IBM).

Closes the debts of ``docs/xval_implicit_ibm_aitken_v6.md``:

* the "epsilon-conservation factor 2.46" (does it belong in the added mass?),
* the "decisive discriminator ... momentum budget ... NOT_RUN" note,
* the queued (never run) box-size/confinement shards.

Stages: ``operator`` (algebra, instant) | ``gain`` (direct-forcing gain x epsilon)
| ``shape`` (plate vs cylinder reference) | ``box`` (confinement) | ``aggregate``
| ``all``.

    python3 benchmarks/xval_ibm_momentum_budget_v7.py --stage all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    analytic_added_mass,
    marker_ring,
    plate_markers,
)
from tensorfem.ibm_aitken_implicit import IBMInterface, LBMFluid2D  # noqa: E402
from tensorfem.ibm_momentum_budget import (  # noqa: E402
    epsilon_amplification,
    ideal_direct_forcing_gain,
    one_step_force_budget,
    prescribed_plate_run,
)

ART = os.path.join(HERE, "xval_ibm_momentum_budget_v7_artifacts")
GAIN_LIFE = 2.0
GAIN_IDEAL = ideal_direct_forcing_gain()
GAINS = (2.0, 1.0, 0.5)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write(name, payload):
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, name), "w") as fh:
        json.dump(_clean(payload), fh, indent=1, sort_keys=True)
    return os.path.join(ART, name)


def _tag(**kw):
    return "_".join(f"{k}{kw[k]}" for k in sorted(kw)).replace(".", "p")


# ---------------------------------------------------------------------------
def operator_stage():
    rows = []
    for c in (4.0, 10.0, 20.0):
        for offset in (0.0, 0.25, 0.5):
            center = (80.0 + offset, 80.0)
            for eps_on in (True, False):
                ibm = IBMInterface(plate_markers(center, c), (160, 160), dx=1.0,
                                   correct_epsilon=eps_on)
                rec = epsilon_amplification(ibm)
                rec.update({"half_width": c, "offset": offset,
                            "correct_epsilon": eps_on})
                rows.append(rec)
    ring = IBMInterface(marker_ring((80.0, 80.0), 8.0), (160, 160), dx=1.0)
    ring_rec = epsilon_amplification(ring)
    ring_rec.update({"shape": "ring", "radius": 8.0})
    fluid = LBMFluid2D((32, 32))
    ibm = IBMInterface(plate_markers((16.0, 16.0), 4.0), (32, 32), dx=1.0)
    F = np.tile([0.0, 1.0], (len(ibm.positions), 1))
    budget = one_step_force_budget(fluid, ibm.spread(F))
    budget.update({
        "sum_F_ds": float((F[:, 1] * ibm.ds).sum()),
        "sum_F_eps_ds": float(np.dot(F[:, 1], ibm.eps_used * ibm.ds)),
    })
    payload = {"operator_rows": rows, "ring": ring_rec,
               "one_step_budget": budget, "ideal_gain": GAIN_IDEAL,
               "life_gain": GAIN_LIFE}
    _write("operator.json", payload)
    return payload


def _run_family(family, pos_builder, box, period, n_steps, amplitude, gains,
                modes, eps_flags, half_width=None):
    rows = []
    for gain, mode, eps_on in [(g, m, e) for g in gains for m in modes
                               for e in eps_flags]:
        t0 = time.time()
        rec = prescribed_plate_run(half_width=half_width or 0.0,
                                  lattice=(box, box), drive_period=period,
                                  amplitude=amplitude, n_steps=n_steps,
                                  gain=gain, gain_mode=mode,
                                  correct_epsilon=eps_on)
        rec["wall_s"] = time.time() - t0
        rec["family"] = family
        rec["shape"] = rec.pop("half_width", None)
        rows.append(rec)
        _write(f"{family}_{_tag(B=box, g=gain, m=mode, eps=int(eps_on))}.json", rec)
        print(f"  {family} gain={gain:.4f} mode={mode} eps={int(eps_on)} "
              f"m_force={rec['m_force']:.2f} m_mom={rec['m_momentum']:.2f} "
              f"res_guo={rec['residual_guo_max']:.3f} "
              f"res_raw={rec['residual_raw_max']:.2f} ({rec['wall_s']:.0f}s)",
              flush=True)
    return rows


def gain_stage(box=64, period=64.0, n_steps=1600, amplitude=0.2):
    return _run_family("gain", None, box, period, n_steps, amplitude,
                       GAINS, ("scalar", "operator"), (True, False),
                       half_width=4.0)


def shape_stage(box=64, period=64.0, n_steps=1600, amplitude=0.2):
    """Same drive, a cylinder instead of a plate: the canonical reference."""
    rows = []
    for radius in (5.0,):
        t0 = time.time()
        rec = _ring_run(radius=radius, box=box, period=period, n_steps=n_steps,
                        amplitude=amplitude, gain=GAIN_LIFE)
        rec["wall_s"] = time.time() - t0
        rec["family"] = "shape"
        rows.append(rec)
        _write(f"shape_{_tag(R=int(radius), B=box)}.json", rec)
        print(f"  ring r={radius} m_force={rec['m_force']:.2f} "
              f"m_mom={rec['m_momentum']:.2f} ref={math.pi * radius ** 2:.2f}",
              flush=True)
    return rows


def _ring_run(radius, box, period, n_steps, amplitude, gain):
    """Prescribed transverse drive of a marker ring (loaded cylinders)."""
    from tensorfem.ibm_added_mass_harmonic import harmonic_projection

    base = np.asarray(marker_ring((0.5 * box, 0.5 * box), radius))
    ibm = IBMInterface(base, (box, box), dx=1.0)
    fluid = LBMFluid2D((box, box), omega=1.8, rho0=1.0)
    w = 2.0 * math.pi / float(period)
    forces, mom = [], []
    n = len(ibm.positions)
    for k in range(1, int(n_steps) + 1):
        t = float(k)
        y = amplitude * math.sin(w * t)
        v = amplitude * w * math.cos(w * t)
        rho, u = fluid.macroscopic(include_force=False)
        ibm.move(base + np.array([0.0, y]))
        rho_i, mom_i = ibm.interpolate(rho, u)
        u_body = np.array([[0.0, v]] * n)
        f_lag = float(gain) * (rho_i[:, None] * (u_body - mom_i / rho_i[:, None]))
        fluid.step(force=ibm.spread(f_lag))
        wgt = ibm.eps_used * ibm.ds
        forces.append(-float(np.dot(f_lag[:, 1], wgt)))
        mm = (fluid.f[..., :, None] * fluid.D[None, None, :, :]).sum(axis=-2)
        mom.append(float(mm[..., 1].sum()))
    ts = np.arange(1, int(n_steps) + 1, dtype=float)
    pf = harmonic_projection(ts, np.asarray(forces), w, skip_cycles=2.0)
    pm = harmonic_projection(ts, np.asarray(mom), w, skip_cycles=2.0)
    amp = float(amplitude) * w
    return {"kind": "ring", "radius": float(radius), "lattice": [box, box],
            "drive_period": float(period), "gain": float(gain),
            "n_markers": n, "n_steps": int(n_steps),
            "m_force": float(pf["in_phase"] / (float(amplitude) * w ** 2)),
            "m_momentum": float(pm["quadrature"] / amp),
            "r2_force": float(pf["r2"]),
            "reference": float(math.pi * radius ** 2),
            "eps_amplification": float((ibm.eps_used * ibm.ds).sum() / ibm.ds.sum())}


def box_stage(period=64.0, n_steps=1600, amplitude=0.2):
    rows = []
    for box in (64, 128):
        rows += _run_family(f"box{box}", None, box, period, n_steps, amplitude,
                            (GAIN_LIFE,), ("scalar",), (True,), half_width=4.0)
    return rows


# ---------------------------------------------------------------------------
def load(pattern):
    rows = []
    if not os.path.isdir(ART):
        return rows
    for name in sorted(os.listdir(ART)):
        if name.startswith(pattern) and name.endswith(".json"):
            with open(os.path.join(ART, name)) as fh:
                rec = json.load(fh)
            if isinstance(rec, dict) and "m_force" in rec:
                rows.append(rec)
    return rows


def aggregate_stage():
    with open(os.path.join(ART, "operator.json")) as fh:
        operator = json.load(fh)
    gain_rows = load("gain_")
    shape_rows = load("shape_")
    box_rows = load("box")
    c = gain_rows[0]["shape"]
    lamb, smeared = analytic_added_mass(c), math.pi * (c + 1.5) ** 2

    def sel(rows, gain, mode="scalar", eps=True):
        for r in rows:
            if (abs(r["gain"] - gain) < 1e-9 and r["gain_mode"] == mode
                    and r["correct_epsilon"] == eps):
                return r
        return None

    life = sel(gain_rows, GAIN_LIFE)
    opmode = sel(gain_rows, GAIN_LIFE, "operator")
    eps_off = sel(gain_rows, GAIN_LIFE, "scalar", eps=False)
    summary = {
        "half_width": c, "gain_life": GAIN_LIFE,
        "reference_lamb": lamb, "reference_smeared": smeared,
        "operator": operator,
        "gain_rows": gain_rows, "shape_rows": shape_rows, "box_rows": box_rows,
        "archived": {k: life[k] for k in
                     ("m_force", "m_momentum", "residual_guo_max",
                      "residual_raw_max", "r2_force")} if life else None,
        "epsilon_off": {"m_force": eps_off["m_force"],
                        "delta_pct": 100.0 * (eps_off["m_force"] - life["m_force"])
                        / life["m_force"],
                        "residual_guo_max": eps_off["residual_guo_max"]}
        if (life and eps_off) else None,
        "operator_gain": {"m_force": opmode["m_force"],
                          "residual_guo_max": opmode["residual_guo_max"],
                          "delta_pct": 100.0 * (opmode["m_force"] - life["m_force"])
                          / life["m_force"]} if (life and opmode) else None,
        "force_momentum_split_pct": 100.0 * (life["m_force"] /
                                             life["m_momentum"] - 1.0) if life else None,
        "err_vs_lamb_pct": 100.0 * (life["m_force"] - lamb) / lamb if life else None,
        "err_vs_smeared_pct": 100.0 * (life["m_force"] - smeared) / smeared
        if life else None,
    }
    if shape_rows:
        r0 = shape_rows[0]
        summary["ring"] = {"m_force": r0["m_force"], "reference": r0["reference"],
                           "ratio": r0["m_force"] / r0["reference"],
                           "eps_amplification": r0["eps_amplification"]}
    _write("summary.json", summary)
    try:
        make_figure(summary)
    except Exception as exc:                       # pragma: no cover
        print(f"figure skipped: {exc}", flush=True)
    return summary


def make_figure(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gain_rows = summary["gain_rows"]
    shape_rows = summary["shape_rows"]
    box_rows = summary["box_rows"]
    lamb, smeared = summary["reference_lamb"], summary["reference_smeared"]
    c = summary["half_width"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for mode, eps_on, style in (("scalar", True, "o-"), ("scalar", False, "s--"),
                                ("operator", True, "^-")):
        sel = sorted([r for r in gain_rows
                      if r["gain_mode"] == mode and r["correct_epsilon"] == eps_on],
                     key=lambda r: r["gain"])
        if not sel:
            continue
        ax.loglog([r["gain"] for r in sel], [r["residual_guo_max"] for r in sel],
                  style, label=f"{mode}, eps {'on' if eps_on else 'off'}")
        ax.loglog([r["gain"] for r in sel], [r["residual_raw_max"] for r in sel],
                  ":", color=ax.lines[-1].get_color())
    ax.axvline(GAIN_LIFE, color="0.6", lw=1, ls=":")
    ax.set_xlabel("direct-forcing gain")
    ax.set_ylabel("max |u(marker) - u_body| / A$\\omega$")
    ax.set_title("(a) constraint error: Guo (solid) vs raw (dotted)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5)

    ax = axes[1]
    eps_flags = sorted({r["correct_epsilon"] for r in gain_rows})
    for eps_on, style in zip(eps_flags, ("o-", "s--")):
        sel = sorted([r for r in gain_rows
                      if r["gain_mode"] == "scalar"
                      and r["correct_epsilon"] == eps_on], key=lambda r: r["gain"])
        ax.semilogx([r["gain"] for r in sel], [r["m_force"] for r in sel], style,
                    label=f"force route, eps {'on' if eps_on else 'off'}")
        ax.semilogx([r["gain"] for r in sel], [r["m_momentum"] for r in sel], ":",
                    color=ax.lines[-1].get_color())
    ax.axhline(lamb, color="k", lw=1, ls="--", label=r"Lamb $\rho\pi c^2$")
    ax.axhline(smeared, color="0.5", lw=1, ls="-.", label=r"smeared $\rho\pi(c+1.5)^2$")
    ax.set_xlabel("direct-forcing gain")
    ax.set_ylabel("apparent added mass")
    ax.set_title(f"(b) plate c={c:g}$\\Delta$: gain dependence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5)

    ax = axes[2]
    if box_rows:
        boxes = sorted({r["lattice"][0] for r in box_rows})
        sel = [r for r in box_rows if r["gain_mode"] == "scalar"]
        xs = [r["lattice"][0] / c for r in sorted(sel, key=lambda r: r["lattice"][0])]
        ys = [r["m_force"] for r in sorted(sel, key=lambda r: r["lattice"][0])]
        if xs:
            ax.semilogx(xs, ys, "o-", label="plate, force route")
    if shape_rows:
        r0 = shape_rows[0]
        ax.plot([r0["lattice"][0] / r0["radius"]], [r0["m_force"]], "k*", ms=12,
                label=f"ring r={r0['radius']:g} (ref $\\rho\\pi a^2$={r0['reference']:.1f})")
    ax.axhline(lamb, color="k", lw=1, ls="--", label=r"Lamb $\rho\pi c^2$")
    ax.axhline(smeared, color="0.5", lw=1, ls="-.")
    ax.set_xlabel("box / body size  (confinement)")
    ax.set_ylabel("apparent added mass")
    ax.set_title("(c) confinement and shape")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5)

    fig.tight_layout()
    out = os.path.join(ART, "v7_momentum_budget.png")
    fig.savefig(out, dpi=140)
    print(f"figure -> {out}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("operator", "gain", "shape", "box", "aggregate",
                             "all"))
    ap.add_argument("--n-steps", type=int, default=1600)
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.stage in ("operator", "all"):
        print("operator ...", flush=True)
        operator_stage()
    if args.stage in ("gain", "all"):
        print("gain ...", flush=True)
        gain_stage(n_steps=args.n_steps)
    if args.stage in ("shape", "all"):
        print("shape ...", flush=True)
        shape_stage(n_steps=args.n_steps)
    if args.stage in ("box", "all"):
        print("box ...", flush=True)
        box_stage(n_steps=args.n_steps)
    if args.stage in ("aggregate", "all"):
        print("aggregate ...", flush=True)
        s = aggregate_stage()
        print(json.dumps({k: s[k] for k in
                          ("archived", "epsilon_off", "operator_gain", "ring",
                           "err_vs_lamb_pct", "err_vs_smeared_pct",
                           "force_momentum_split_pct") if k in s}, indent=2),
              flush=True)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()