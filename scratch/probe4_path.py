"""Probe 4: elastoplastic path (first yield, collapse, permanent set) for one mesh."""
import sys, json
from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure, run_patch_plastic_path,
    yield_line_patch_upper_bound, strip_lower_bound, low_temperature_flow_stress,
)
A = B = 1.0; T = 0.02; E, NU = 210e9, 0.3
FY = low_temperature_flow_stress("DH36", -40.0).yield_stress
PATCH = (0.5 * A, 0.5 * B); H = 2e10
n = int(sys.argv[1]); layers = int(sys.argv[2]); exact = bool(int(sys.argv[3]))
ref = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH, grid=21, terms=41)
upper = yield_line_patch_upper_bound(A, B, T, FY, PATCH)
lower = strip_lower_bound(A, B, T, FY, PATCH)
p_y = ref["pressure"]
steps = [f * p_y for f in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)]
path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY, hardening=H,
                              patch=PATCH, load_factors=steps, nx=n, ny=n, layers=layers,
                              unload_steps=3, exact_thickness=exact)
col = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY, hardening=1e6,
                             patch=PATCH, load_factors=[f * p_y for f in (0.6, 1.0, 1.4, 1.8, 2.4, 3.0)],
                             nx=n, ny=n, layers=layers, exact_thickness=exact)
collapse = col.collapse_factor or col.loading[-1].load_factor
print(json.dumps({"n": n, "layers": layers, "exact": exact,
                  "p_y_analytic": p_y, "upper": upper["pressure"], "lower": lower["pressure"],
                  "first_yield_factor": path.first_yield_factor,
                  "first_yield_err": path.first_yield_factor / p_y - 1,
                  "collapse_near_ideal": collapse,
                  "collapse_err_vs_upper": collapse / upper["pressure"] - 1,
                  "collapse_err_vs_upper_layering_corrected":
                      collapse * (1 - 1 / layers) / upper["pressure"] - 1,
                  "peak_deflection": path.loading[-1].centre_deflection,
                  "permanent_deflection": path.permanent_deflection}))