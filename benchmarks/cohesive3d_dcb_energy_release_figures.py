#!/usr/bin/env python
"""Figures for the DCB energy-release-rate verification (m16698).

Reads ``benchmarks/cohesive3d_dcb_energy_release.json`` and writes two PNGs
under ``docs/assets/gap-fem-3d-cohesive/``:

* ``dcb_energy_release_pdelta.png``  -- load vs opening (P-delta curve);
* ``dcb_energy_release_rate.png``    -- cumulative interface work vs cracked
  area; the slope of that line *is* the energy release rate ``G``, compared
  with the ``G_Ic`` reference.

Zero hand-typed numbers: every axis value comes from the JSON artefact.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "assets" / "gap-fem-3d-cohesive"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    d = json.loads((HERE / "cohesive3d_dcb_energy_release.json").read_text())
    curve = d["curve"]
    disp = np.array([c["disp"] for c in curve]) * 1e3          # mm
    load = np.array([c["load"] for c in curve])                # N
    w_if = np.array([c["interface_work_J"] for c in curve])    # J
    a_fail = np.array([c["failed_area_m2"] for c in curve])    # m^2
    front = np.array([c["crack_front_m"] for c in curve]) * 1e3  # mm
    a0 = d["parameters"]["initial_crack_m"] * 1e3             # mm
    gic = d["parameters"]["G_Ic_J_m2"]
    crit = d["criterion"]
    win = d["window"]

    # --- figure 1: P-delta curve ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(disp, load, "-", color="#1f4e79", lw=2,
            label="FE (HEX8 + COH3D8)")
    ipk = int(np.argmax(load))
    ax.plot(disp[ipk], load[ipk], "o", color="#c00000", ms=7,
            label=f"peak {load[ipk]:.2f} N @ {disp[ipk]:.2f} mm")
    ax.set_xlabel("total opening  $\\delta$  [mm]")
    ax.set_ylabel("applied load  $P$  [N]")
    ax.set_title("DCB load--opening curve (front propagates past the peak)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "dcb_energy_release_pdelta.png")
    plt.close(fig)

    # --- figure 2: W_interface vs A_cracked (slope = G) --------------------
    ext = front - a0                                           # mm
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot(a_fail * 1e4, w_if * 1e3, "o-", color="#2e7d32", ms=4, lw=1.4,
             label="FE  $W_{interface}$ vs $A_{cracked}$")
    # reference line of slope G_Ic through the origin
    a_ref = np.array([0.0, a_fail.max()]) * 1e4
    ax.plot(a_ref, gic * a_ref * 1e-1, "--", color="#c00000", lw=1.6,
            label=f"$G_{{Ic}}$ = {gic:.0f} J/m$^2$ (slope)")
    # band of +/-10 %
    ax.fill_between(a_ref, gic * 0.9 * a_ref * 1e-1, gic * 1.1 * a_ref * 1e-1,
                   color="#c00000", alpha=0.08)
    ax.set_xlabel("cracked area  $A_{cracked}$  [$10^{-4}$ m$^2$]")
    ax.set_ylabel("interface work  $W_{interface}$  [mJ]")
    ax.set_title(
        f"Energy release rate: slope = {crit['got_local_J_m2']:.1f} J/m$^2$"
        f"  ({crit['rel_error_local']*100:+.1f} % vs $G_{{Ic}}$)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "dcb_energy_release_rate.png")
    plt.close(fig)
    print("wrote", OUT / "dcb_energy_release_pdelta.png")
    print("wrote", OUT / "dcb_energy_release_rate.png")


if __name__ == "__main__":
    main()
