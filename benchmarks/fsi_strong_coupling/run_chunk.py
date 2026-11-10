"""Chunked FSI-2 run with checkpointing (for time-boxed shell calls).

Each invocation advances the coupled solver by ``--steps`` lattice steps,
writes a checkpoint (fluid populations + modal state) and appends the recorded
history to a JSON file, so a long run can be produced in several short calls.

    python benchmarks/fsi_strong_coupling/run_chunk.py --steps 2500 --tag run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.fsi_lbm_strong_coupling import (  # noqa: E402
    FSI2Parameters,
    StrongCouplingFSI2,
)

OUT = os.path.join(ROOT, "artifacts", "fsi_strong_coupling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--u-lat", type=float, default=0.1)
    ap.add_argument("--n-modes", type=int, default=5)
    ap.add_argument("--zeta", type=float, default=0.02)
    ap.add_argument("--record-every", type=int, default=5)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args(argv)

    ckpt_path = os.path.join(OUT, f"checkpoint_{args.tag}.npz")
    hist_path = os.path.join(OUT, f"history_{args.tag}.json")

    fsi = StrongCouplingFSI2(FSI2Parameters(), cells_per_D=args.cells,
                             u_lat=args.u_lat, n_modes=args.n_modes,
                             zeta=args.zeta, max_sub_iterations=12,
                             tolerance=1e-4)
    history = {k: [] for k in ("t", "tip", "drag", "lift", "sub", "fres",
                               "fluid_ke", "struct_ke", "struct_pe",
                               "inlet_power", "outlet_power", "viscous",
                               "interface_power")}
    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path, allow_pickle=True)
        fsi.solver.f = data["f"]
        fsi.solver._prev_eff_solid = data["prev_solid"]
        fsi.beam.eta = data["eta"]
        fsi.beam.eta_dot = data["eta_dot"]
        fsi.beam.eta_ddot = data["eta_ddot"]
        fsi.beam._sync_full_state()
        fsi._t = float(data["t"])
        with open(hist_path, encoding="utf-8") as fh:
            history = json.load(fh)
        print("resumed at t=%.4f s" % fsi._t)

    hist = fsi.run(args.steps, record_every=args.record_every)
    for key, attr in (("t", "time"), ("tip", "tip_displacement"),
                      ("drag", "drag"), ("lift", "lift"),
                      ("sub", "sub_iterations"), ("fres", "force_residual"),
                      ("fluid_ke", "fluid_ke"), ("struct_ke", "struct_ke"),
                      ("struct_pe", "struct_pe"), ("inlet_power", "inlet_power"),
                      ("outlet_power", "outlet_power"),
                      ("viscous", "viscous_dissipation"),
                      ("interface_power", "interface_power")):
        history[key] += [float(x) for x in getattr(hist, attr)][: len(getattr(hist, attr))]

    np.savez_compressed(ckpt_path, f=fsi.solver.f,
                        prev_solid=fsi.solver._prev_eff_solid,
                        eta=fsi.beam.eta, eta_dot=fsi.beam.eta_dot,
                        eta_ddot=fsi.beam.eta_ddot, t=fsi._t)
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump(history, fh)
    tip = np.asarray(history["tip"])
    print("tag=%s t=%.3f s  samples=%d  tip[mm]=[%.2f, %.2f]  mean_sub=%.2f"
          % (args.tag, fsi._t, len(history["t"]), 1e3 * tip.min(), 1e3 * tip.max(),
             float(np.mean(history["sub"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())