#!/usr/bin/env python
"""Diagnose why the 3-D DCB specimen is ~5.6x too flexible vs beam theory.

Linear (undamaged) compliance only: solve the linear system with the initial
cohesive stiffness at a minuscule opening, so no damage happens and the
cohesive law contributes nothing but its penalty stiffness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry,
    beam_compliance,
    build_dcb_model,
)

torch.set_num_threads(1)


def linear_compliance(g, law, delta=1e-9):
    model = build_dcb_model(g, law)
    K = model.solid_stiffness()
    u = np.zeros(model.n_dofs)
    free = model.free_dofs
    # initial (undamaged) interface tangent
    _, K_coh, _, _ = model.interface_response(u, None)
    Kt = (K + K_coh)[free][:, free].tocsc()
    faced = np.concatenate((model.upper_load_dofs, model.lower_load_dofs))
    signs = np.concatenate((np.full(model.upper_load_dofs.size, 0.5),
                            np.full(model.lower_load_dofs.size, -0.5))) * delta
    u[faced] = signs
    rfc = K[free][:, faced]
    rhs = -(rfc @ u[faced])
    u[free] = spla.splu(Kt).solve(rhs)
    fint = K @ u + K_coh @ u
    load = abs(fint[model.upper_load_dofs].sum()) + abs(fint[model.lower_load_dofs].sum())
    return delta / load, model, u, fint


def main():
    law = BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 300.0, 800.0, 2.2)
    for (nx, ny, nz) in [(24, 1, 2), (24, 1, 4), (24, 1, 6), (48, 1, 4),
                         (24, 2, 4), (96, 1, 8)]:
        g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=3e-3,
                        initial_crack=30e-3, nx=nx, ny=ny, nz=nz)
        c, model, u, fint = linear_compliance(g, law)
        c_ref = beam_compliance(g.initial_crack, g, root_rotation=True, shear=True)
        c_plain = beam_compliance(g.initial_crack, g, root_rotation=False, shear=False)
        uu = u.reshape(-1, 3)
        x = model.nodes.numpy()[:, 0]
        nlow = model.n_lower
        # tip and crack-tip (x=30mm) vertical displacement of the upper arm
        sel_tip = (np.abs(x - g.length) < 1e-12)
        sel_ct = (np.abs(x - (g.length - g.initial_crack)) < 1e-12)
        uz_up_tip = uu[sel_tip, 2][uu[sel_tip, 2] > 0].mean()
        uz_up_ct = uu[sel_ct, 2].max()
        print(f"nx={nx} ny={ny} nz={nz:2d} C_fem={c:.4e} C_root={c_ref:.4e} "
              f"C_plain={c_plain:.4e} ratio_root={c/c_ref:6.3f} "
              f"ratio_plain={c/c_plain:6.3f} uz_tip={uz_up_tip:.3e} "
              f"uz_cracktip={uz_up_ct:.3e}")


if __name__ == "__main__":
    main()