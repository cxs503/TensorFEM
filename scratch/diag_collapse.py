"""Diagnostic: why does the near-ideal FE collapse exceed the yield-line upper bound?"""
import torch
from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure, patch_plate_mesh, run_patch_plastic_path,
    yield_line_patch_upper_bound, strip_lower_bound, ilyushin_sine_upper_bound,
)

A = B = 1.0; T = 0.02; E, NU = 210e9, 0.3
FY = 355e6 + 0.48e6 * (20 + 40)   # DH36 @ -40C
FY = 383.8e6
PATCH = (0.5, 0.5)
py = analytic_first_yield_pressure(A, B, T, E, NU, FY, PATCH)
up = yield_line_patch_upper_bound(A, B, T, FY, PATCH)
lo = strip_lower_bound(A, B, T, FY, PATCH)
sine = ilyushin_sine_upper_bound(A, B, T, FY, PATCH)
print("first yield %.4f MPa  yield-line %.4f  strip lower %.4f  sine %.4f MPa"
      % (py["pressure"]/1e6, up["pressure"]/1e6, lo["pressure"]/1e6, sine["pressure"]/1e6))
print("m_p =", up["m_p"], "mechanism volume ratio", up["mechanism_volume"]/ (A*B))

for layers in (3, 5):
    factors = [f * py["pressure"] for f in (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4)]
    path = run_patch_plastic_path(A, B, T, young=E, poisson=NU, yield_stress=FY,
                                  hardening=1e6, patch=PATCH, load_factors=factors,
                                  nx=8, ny=8, layers=layers, exact_thickness=True)
    print(f"\n--- near-ideal hardening=1e6, layers={layers}, nx=8 ---")
    for s in path.loading:
        print("  p=%8.4f MPa (%4.2f x Py)  w=%8.4f mm  alpha_max=%.6f  sigma_surf=%.4f MPa (%.3f fy)"
              % (s.load_factor/1e6, s.load_factor/py["pressure"], s.centre_deflection*1e3,
                 s.max_alpha, s.max_surface_mises/1e6, s.max_surface_mises/FY))
    print("  collapse_factor:", path.collapse_factor, "limit_point:", path.limit_point_searched,
          "ramp_completed:", path.ramp_completed,
          "stiffness_ratio:", path.stiffness_ratio())