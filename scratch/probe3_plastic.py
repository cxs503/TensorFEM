"""Probe 3: plastic acceptance with exact through-thickness integration.
Usage: python probe3.py <n> <layers> <exact>
Prints: analytic first-yield pressure, same-mesh elastic first-yield, collapse."""
import sys, json
import torch
from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure, elastic_patch_plate, patch_plate_mesh,
    run_patch_plastic_path, surface_extrapolated_mises, _centre_node,
    yield_line_patch_upper_bound, strip_lower_bound, low_temperature_flow_stress,
)
from tensorfem.layered_shell4_plasticity import LayeredShell4State, solve_layered_shell4_increment

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
FY = low_temperature_flow_stress("DH36", -40.0).yield_stress
PATCH = (0.5 * A, 0.5 * B)
H = 2e10
n = int(sys.argv[1]); layers = int(sys.argv[2]); exact = bool(int(sys.argv[3]))

ref = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=21, terms=41)
upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)
lower = strip_lower_bound(A, B, T, FY, PATCH)

# same-mesh elastic surface stress per unit pressure (huge sigma_y -> no plasticity)
model, info = patch_plate_mesh(A, B, T, n, n, young=E, poisson=NU, yield_stress=1e15,
                               hardening=H, layers=layers, patch=PATCH,
                               exact_thickness=exact)
st = LayeredShell4State.virgin(model)
step = solve_layered_shell4_increment(model, info["reference_load"], 1.0, info["fixed_dofs"], st,
                                      torch.zeros(model.n_dofs, dtype=torch.float64),
                                      tolerance=1e-10, max_iterations=12)
sig_pa = surface_extrapolated_mises(step.stress, T)
elastic_first_yield = FY / sig_pa

p_y = ref["pressure"]
steps = [f * p_y for f in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)]
path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY, hardening=H,
                              patch=PATCH, load_factors=steps, nx=n, ny=n, layers=layers,
                              exact_thickness=exact) if False else None
print(json.dumps({"n": n, "layers": layers, "exact": exact,
                  "analytic_first_yield": p_y, "same_mesh_elastic_first_yield": elastic_first_yield,
                  "err_vs_analytic": elastic_first_yield / p_y - 1,
                  "upper": upper["pressure"], "lower": lower["pressure"]}))