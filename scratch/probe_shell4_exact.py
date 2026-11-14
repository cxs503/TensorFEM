"""Probe 2: does exact through-thickness integration (dz^3/12 term) remove the
"element-level" bias, and is the bias the closed-form (1-1/n^2) mid-point deficit?"""
import torch
from tensorfem.plate_patch_plastic_limit import (
    elastic_patch_plate, patch_plate_mesh, surface_extrapolated_mises, _centre_node,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4State, solve_layered_shell4_increment,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
D = E * T ** 3 / (12 * (1 - NU ** 2))
Q = 1.0e5


def run(n, patch, q, layers, exact, sy=1e15):
    model, info = patch_plate_mesh(A, B, T, n, n, young=E, poisson=NU,
                                   yield_stress=sy, hardening=1e10,
                                   layers=layers, patch=patch)
    object.__setattr__(model, "exact_thickness", exact) if False else None
    model = model.__class__(nodes=model.nodes, elements=model.elements, young=E,
                            poisson=NU, thickness=T, yield_stress=sy,
                            hardening=1e10, layers=layers, exact_thickness=exact)
    st = LayeredShell4State.virgin(model)
    step = solve_layered_shell4_increment(model, info["reference_load"], q,
                                          info["fixed_dofs"], st,
                                          torch.zeros(model.n_dofs, dtype=torch.float64),
                                          tolerance=1e-10, max_iterations=12)
    c = _centre_node(info)
    return float(step.displacement[6 * c + 2]), max(surface_extrapolated_mises(step.stress, T), 1e-30)


for label, patch in (("uniform", (A, B)), ("patch", (0.5 * A, 0.5 * B))):
    ref = elastic_patch_plate(A, B, T, E, NU, patch, Q, terms=61)
    wr, sr = ref["deflection"][0], ref["sigma_eq"][0]
    print(f"== {label}: analytic w={wr:.6e} sigma={sr/1e6:.4f} MPa")
    for exact in (False, True):
        for layers in (3, 5, 9):
            w, s = run(8, patch, Q, layers, exact)
            print(f"  exact={int(exact)} n=8 layers={layers}: w/ref={w/wr:.4f} "
                  f"(pred (1-1/n^2)^-1={1/(1-1/layers**2):.4f})  sigma/ref={s/sr:.4f}", flush=True)
    for exact in (False, True):
        for n in (4, 6, 8, 12):
            w, s = run(n, patch, Q, 3, exact)
            print(f"  exact={int(exact)} layers=3 n={n}: w/ref={w/wr:.4f} sigma/ref={s/sr:.4f}", flush=True)