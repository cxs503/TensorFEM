"""Adversarial probe: is the layered-Shell4 offset an element deficit, and is the
"elastic bias" contaminated by plasticity?  Pure-elastic (huge sigma_y) runs."""
import math, sys, time
import torch
from tensorfem.plate_patch_plastic_limit import (
    elastic_patch_plate, patch_plate_mesh, patch_pressure_forces,
    surface_extrapolated_mises, _centre_node,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State, solve_layered_shell4_increment,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
D = E * T ** 3 / (12 * (1 - NU ** 2))
Q = 1.0e5  # 0.1 MPa uniform -> w/t = 0.05, far below yield


def run(n, patch, q, layers=3, sy=1e15):
    model, info = patch_plate_mesh(A, B, T, n, n, young=E, poisson=NU,
                                   yield_stress=sy, hardening=1e10,
                                   layers=layers, patch=patch)
    st = LayeredShell4State.virgin(model)
    u0 = torch.zeros(model.n_dofs, dtype=torch.float64)
    step = solve_layered_shell4_increment(model, info["reference_load"], q,
                                          info["fixed_dofs"], st, u0,
                                          tolerance=1e-10, max_iterations=12)
    c = _centre_node(info)
    w = float(step.displacement[6 * c + 2])
    sig = max(surface_extrapolated_mises(step.stress, T), 1e-30)
    return w, sig


for label, patch in (("uniform", (A, B)), ("patch 0.5x0.5", (0.5 * A, 0.5 * B))):
    ref = elastic_patch_plate(A, B, T, E, NU, patch, Q, terms=61)
    wr, sr = ref["deflection"][0], ref["sigma_eq"][0]
    print(f"== {label}: analytic w={wr:.6e} (w/t={wr/T:.4f})  sigma={sr/1e6:.4f} MPa")
    for n in (4, 6, 8):
        t0 = time.time()
        w, s = run(n, patch, Q)
        print(f"  n={n:2d} w={w:.6e} w/ref={w/wr:+.4f}  sigma/ref={s/sr:+.4f}  ({time.time()-t0:.1f}s)", flush=True)