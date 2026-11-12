#!/usr/bin/env python
"""Strength/mesh trend of the CZM DCB peak load: does it approach LEFM?"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry, build_dcb_model, solve_dcb, beam_peak_load,
)

torch.set_num_threads(8)
CASES = [dict(nx=120, tn=20e6), dict(nx=240, tn=50e6)]
rows = []
for c in CASES:
    kn = c["tn"] / 5e-6
    law = BKCohesive3DLaw(kn, kn / 2.5, c["tn"], 0.6 * c["tn"], 300.0, 800.0, 2.2)
    g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=3e-3,
                    initial_crack=30e-3, nx=c["nx"], ny=1, nz=2)
    m = build_dcb_model(g, law)
    t0 = time.time()
    r = solve_dcb(m, delta_max=2.5e-3, n_steps=80, tol=1e-8)
    pl = beam_peak_load(g.initial_crack, 300.0, g, root_rotation=True)
    pk = r.peak["load"]
    idx = int(np.argmax([p["load"] for p in r.curve]))
    rows.append(dict(nx=c["nx"], t_n0=c["tn"], peak_load=pk, lefm=pl,
                     rel_err=(pk - pl) / pl, peak_disp=r.curve[idx]["disp"],
                     cz_length=r.curve[idx]["crack_front_m"],
                     conv=r.stats["converged_all"], runtime=time.time() - t0))
    print(json.dumps(rows[-1]), flush=True)
(ROOT / "scratch" / "dcb_peak_trend.json").write_text(json.dumps(rows, indent=1))