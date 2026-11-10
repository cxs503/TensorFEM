#!/usr/bin/env python3
"""Run all 6 verification criteria and output results as JSON."""
import json, math, sys
sys.path.insert(0, "src")
import torch
from tensorfem.hertz_cattaneo_mindlin import *

R_star = 5.0; E = 1e5; nu = 0.3
E_star = 1.0 / (2.0 * (1 - nu**2) / E)
delta = 0.025
mu = 0.3

P_ref = hertz_load_from_indentation(delta, R_star, E_star)
ref = hertz_two_sphere_reference(P_ref, R_star, 1e18, E, nu, E, nu)

results = {}

# E1: Contact radius ≤ 2% (3 mesh levels)
print("E1: Mesh convergence study...", flush=True)
study = mesh_convergence_study(
    mesh_levels=[10, 14, 18],
    R_star=R_star, E_star=E_star, nu=nu,
    indentation=delta, width=4.0, depth=4.0, grading=3.0,
    penalty_factor=100.0, al_max_iterations=60,
)
fine = study["runs"][-1]
results["E1_contact_radius"] = {
    "error": fine["contact_radius_error"],
    "threshold": 0.02,
    "pass": fine["contact_radius_error"] <= 0.02,
    "fe_value": fine["contact_radius_fe"],
    "ref_value": fine["contact_radius_ref"],
    "mesh_levels": [r["nr"] for r in study["runs"]],
    "errors_by_mesh": [r["contact_radius_error"] for r in study["runs"]],
}
print(f"  a_err={fine['contact_radius_error']:.4%} {'PASS' if results['E1_contact_radius']['pass'] else 'FAIL'}", flush=True)

# E2: Peak pressure ≤ 5%
results["E2_peak_pressure"] = {
    "error": fine["peak_pressure_error"],
    "threshold": 0.05,
    "pass": fine["peak_pressure_error"] <= 0.05,
    "fe_value": fine["peak_pressure_fe"],
    "ref_value": fine["peak_pressure_ref"],
}
print(f"  p0_err={fine['peak_pressure_error']:.4%} {'PASS' if results['E2_peak_pressure']['pass'] else 'FAIL'}", flush=True)

# E3: P-δ curve R² ≥ 0.99
print("E3: Load-displacement curve...", flush=True)
deltas = [0.004, 0.008, 0.012, 0.016, 0.020, 0.025]
curve = hertz_load_displacement_curve(
    indentations=deltas,
    R_star=R_star, E_star=E_star, nu=nu,
    nr=18, nz=18, grading=2.5, penalty_factor=100.0,
    al_max_iterations=60,
)
results["E3_load_displacement"] = {
    "r_squared": curve["r_squared"],
    "threshold": 0.99,
    "pass": curve["r_squared"] >= 0.99,
    "loads_fe": curve["loads_fe"],
    "loads_analytical": curve["loads_analytical"],
    "indentations": deltas,
}
print(f"  R²={curve['r_squared']:.6f} {'PASS' if results['E3_load_displacement']['pass'] else 'FAIL'}", flush=True)

# E4: Cattaneo-Mindlin stick/slip ≤ 5%
print("E4: Cattaneo-Mindlin...", flush=True)
Q = 0.5 * mu * P_ref
ref_cm = cattaneo_mindlin_reference(P_ref, Q, mu, ref.contact_radius, E, nu, E, nu)
sol = solve_hertz_contact(
    nr=18, nz=18, width=6.0, depth=6.0,
    R_star=R_star, E_star=E_star, nu=nu,
    indentation=delta, grading=3.0, penalty_factor=100.0,
    al_max_iterations=60,
)
cm_sol = solve_cattaneo_mindlin(
    contact_solution=sol, Q=Q, mu=mu,
    E1=E, nu1=nu, E2=E, nu2=nu,
)
cm_err = abs(cm_sol.stick_radius / ref_cm.stick_radius - 1.0)
results["E4_stick_slip"] = {
    "error": cm_err,
    "threshold": 0.05,
    "pass": cm_err <= 0.05,
    "fe_stick_radius": cm_sol.stick_radius,
    "ref_stick_radius": ref_cm.stick_radius,
    "contact_radius": ref.contact_radius,
    "Q_ratio": Q / (mu * P_ref),
}
print(f"  stick_err={cm_err:.4%} {'PASS' if results['E4_stick_slip']['pass'] else 'FAIL'}", flush=True)

# E5: Complementarity ≤ 1e-8
results["E5_complementarity"] = {
    "residual": sol.complementarity_residual,
    "threshold": 1e-8,
    "pass": sol.complementarity_residual <= 1e-8,
    "max_penetration": sol.max_penetration,
}
print(f"  comp={sol.complementarity_residual:.2e} {'PASS' if results['E5_complementarity']['pass'] else 'FAIL'}", flush=True)

# E6: Negative control
print("E6: Negative control...", flush=True)
nc = negative_control(nr=10, nz=10, R_star=R_star, E_star=E_star, nu=nu)
results["E6_negative_control"] = {
    "zero_indentation_all_zero": nc["zero_indentation"]["all_zero"],
    "separated_all_zero": nc["separated"]["all_zero"],
    "pass": nc["zero_indentation"]["all_zero"] and nc["separated"]["all_zero"],
    "zero_max_force": nc["zero_indentation"]["max_nodal_force"],
    "separated_max_force": nc["separated"]["max_nodal_force"],
}
print(f"  zero={nc['zero_indentation']['all_zero']} separated={nc['separated']['all_zero']} {'PASS' if results['E6_negative_control']['pass'] else 'FAIL'}", flush=True)

# Summary
all_pass = all(v["pass"] for v in results.values())
results["all_pass"] = all_pass
results["reference"] = {
    "P": P_ref, "a": ref.contact_radius, "p0": ref.peak_pressure,
    "R_star": R_star, "E_star": E_star, "nu": nu, "delta": delta,
}

print(f"\n{'ALL PASS' if all_pass else 'SOME FAIL'}", flush=True)
with open("artifacts/hertz_cattaneo_mindlin/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to artifacts/hertz_cattaneo_mindlin/results.json", flush=True)
