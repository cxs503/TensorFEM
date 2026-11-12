#!/usr/bin/env python
"""Mesh/strength study: does the CZM DCB peak load converge to LEFM?"""
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
    DCBGeometry, build_dcb_model, solve_dcb, beam_peak_load, beam_serr,
)

torch.set_num_threads(8)
rows = []
for nx in (60,):
    for tn in (5e6, 10e6, 20e6, 40e6):
        law = BKCohesive3DLaw(tn / 5e-6, tn / 5e-6 / 2.5, tn, 0.6 * tn, 300.0, 800.0, 2.2)
        g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=3e-3,
                        initial_crack=30e-3, nx=nx, ny=1, nz=2)
        m = build_dcb_model(g, law)
        t0 = time.time()
        r = solve_dcb(m, delta_max=2.5e-3, n_steps=80, tol=1e-8)
        pl = beam_peak_load(g.initial_crack, 300.0, g, root_rotation=True)
        pk = r.peak["load"]
        idx = int(np.argmax([p["load"] for p in r.curve]))
        r2 = solve_dcb(m, delta_max=r.curve[idx]["disp"], n_steps=40, tol=1e-8)
        cz = (float(g.length - r2.iface_x[r2.damage > 1e-3].min()) - g.initial_crack
              if np.any(r2.damage > 1e-3) else 0.0)
        row = dict(nx=nx, t_n0=tn, delta_n0=tn / (tn / 5e-6), peak_load=pk,
                   lefm_peak=pl, rel_err=(pk - pl) / pl,
                   gI_at_fe_peak=beam_serr(pk, g.initial_crack, g, root_rotation=True),
                   cz_length=cz, conv=r.stats["converged_all"],
                   peak_disp=r.curve[idx]["disp"], runtime=time.time() - t0)
        rows.append(row)
        print(json.dumps(row), flush=True)
(ROOT / "scratch" / "dcb_strength_study.json").write_text(json.dumps(rows, indent=1))