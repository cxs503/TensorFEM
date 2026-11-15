#!/usr/bin/env python3
"""Background driver for the v8 clean-protocol sweeps (writes JSON progressively).

Protocol (fixed after the window/Mach study in the report, section 3):

* drive period 256 lattice steps *per physical unit of dx* -- i.e. ``period_phys
  = 256`` with the acoustic scaling ``dt = dx`` -- so the Mach number of the
  prescribed motion is 5e-3 and the drive is quasi-static relative to the box
  acoustic transit;
* six drive cycles recorded, the first three discarded (``skip_cycles = 3``);
* operator (``B^-1``) direct-forcing gain, which imposes the marker velocity to
  machine precision, so the measurement is not contaminated by constraint error.

Usage:  python3 benchmarks/run_v8_sweeps.py refine|csize
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.ibm_added_mass_refinement import lattice_added_mass  # noqa: E402

ART = os.path.join(HERE, "xval_ibm_added_mass_refinement_v8_artifacts")
PERIOD = 256.0
CYCLES = 6.0
SKIP = 3.0
AMP = 0.2


def run(tag, **kw):
    t0 = time.time()
    r = lattice_added_mass(period_phys=PERIOD, n_cycles=CYCLES, skip_cycles=SKIP,
                           amplitude=AMP, gain=2.0, gain_mode="operator", **kw)
    r["tag"] = tag
    r["wall_s"] = round(time.time() - t0, 2)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, tag + ".json"), "w") as fh:
        json.dump({k: (v if not isinstance(v, list) else
                       [x if isinstance(x, (int, float)) else x for x in v])
                   for k, v in r.items()}, fh, indent=1, sort_keys=True)
    print(f"{tag:22s} C_mom={r['C_momentum']:.4f} C_force={r['C_force']:.4f} "
          f"r2={r['r2_momentum']:.4f} constr={r['constraint_rel_max']:.1e} "
          f"({r['wall_s']:.0f}s)", flush=True)
    return r


def refine():
    for refine in (1, 2, 4):
        run(f"clean_refine_{refine}", refine=refine, c_phys=4.0, box_phys=64.0)


def csize():
    for c in (4.0, 8.0, 16.0):
        run(f"clean_c_{int(c)}", refine=1, c_phys=c, box_phys=16.0 * c)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "refine"
    {"refine": refine, "csize": csize}[what]()