"""Independent field checks for the published rectangular Q4 benchmarks.

Beam stress is compared at element centres in the declared interior region
0.2 L <= x <= 0.8 L. This is an engineering beam reference, not a certificate
for clamp/end boundary stresses. Uniform membrane stress is checked everywhere.
"""
from __future__ import annotations

import math
import torch


def _field(case):
    mesh = case["mesh"]
    nx, ny = mesh["nx"], mesh["ny"]
    if nx < 2 or ny < 2:
        raise ValueError("field verification requires at least a 2x2 mesh")
    nodes = case["field"]["nodes"]
    stress = case["field"]["element_stress"]
    if len(nodes) != (nx + 1) * (ny + 1) or len(stress) != nx * ny:
        raise ValueError("incomplete displacement or stress field")
    n = torch.tensor([[p[k] for k in ("x", "y", "ux", "uy")] for p in nodes],
                     dtype=torch.float64)
    s = torch.tensor([[p[k] for k in ("x", "y", "sigma_x", "sigma_y", "tau_xy",
                                     "von_mises")] for p in stress], dtype=torch.float64)
    if not bool(torch.isfinite(n).all() and torch.isfinite(s).all()):
        raise ValueError("nonfinite displacement or stress field")
    # Do not accept duplicated/mislocated samples as a complete mesh.
    problem = case["problem"]
    x = torch.linspace(0., problem["length_m"], nx + 1, dtype=torch.float64)
    y = torch.linspace(-problem["height_m"] / 2, problem["height_m"] / 2,
                       ny + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = torch.meshgrid((y[:-1] + y[1:]) / 2, (x[:-1] + x[1:]) / 2,
                           indexing="ij")
    if not (torch.allclose(n[:, :2], torch.stack((xx.flatten(), yy.flatten()), 1),
                          atol=1e-12, rtol=1e-12) and
            torch.allclose(s[:, :2], torch.stack((cx.flatten(), cy.flatten()), 1),
                           atol=1e-12, rtol=1e-12)):
        raise ValueError("field coordinates do not match the declared mesh")
    recovered_vm = torch.sqrt(s[:, 2]**2 - s[:, 2]*s[:, 3] + s[:, 3]**2 + 3*s[:, 4]**2)
    if not torch.allclose(recovered_vm, s[:, 5], rtol=1e-10, atol=1e-6):
        raise ValueError("von Mises field does not match stress components")
    return n, s


def _relative_l2(computed, reference):
    return float(torch.linalg.vector_norm(computed-reference) /
                 torch.linalg.vector_norm(reference))


def verify_case(case, kind):
    """Recompute errors from complete fields; never trust stored pass flags."""
    nodes, stress = _field(case)
    p = case["problem"]
    L, H, t, E, P = (p[k] for k in ("length_m", "height_m", "thickness_m",
                                    "young_modulus_pa", "load_n"))
    if not all(math.isfinite(v) and v > 0 for v in (L, H, t, E)) or not P:
        raise ValueError("invalid reference parameters")
    right = nodes[nodes[:, 0] == L]
    displacement_field_error = None
    if kind == "bending":
        I = t * H**3 / 12
        ref_u = P * L**3 / (3 * E * I)
        tip = float(right[len(right)//2, 3])
        mask = (stress[:, 0] >= .2*L) & (stress[:, 0] <= .8*L)
        s = stress[mask]
        sx = -P * (L-s[:, 0]) * s[:, 1] / I
        shear = P / (2*I) * (H**2/4-s[:, 1]**2)
        vm = torch.sqrt(sx**2 + 3*shear**2)
        ref = torch.stack((sx, torch.zeros_like(sx), shear), 1)
        errors = {"sigma_x_l2": _relative_l2(s[:, 2], sx),
                  "tau_xy_l2": _relative_l2(s[:, 4], shear),
                  "stress_tensor_l2": _relative_l2(s[:, 2:5], ref),
                  "von_mises_l2": _relative_l2(s[:, 5], vm)}
        force = case["applied_load_y"]
        reaction = case["reaction_left_y"]
        scope = "element centres at 0.2L <= x <= 0.8L; clamp/end peaks excluded"
    elif kind == "membrane":
        ref_u = P * L / (E*t*H)
        tip = float(right[:, 2].mean())
        sx = torch.full_like(stress[:, 2], P/(t*H))
        ref = torch.stack((sx, torch.zeros_like(sx), torch.zeros_like(sx)), 1)
        errors = {"sigma_x_l2": _relative_l2(stress[:, 2], sx),
                  "stress_tensor_l2": _relative_l2(stress[:, 2:5], ref),
                  "von_mises_l2": _relative_l2(stress[:, 5], sx.abs())}
        ref_displacement = torch.stack((P*nodes[:, 0]/(E*t*H),
            -p["poisson"]*P*(nodes[:, 1]+H/2)/(E*t*H)), 1)
        displacement_field_error = _relative_l2(nodes[:, 2:4], ref_displacement)
        force = case["applied_load_x"]
        reaction = case["reaction_left_x"]
        scope = "all element centres; uniform membrane patch only"
    else:
        raise ValueError("unknown field reference")
    displacement_error = abs(tip-ref_u)/abs(ref_u)
    load_error = abs(force-P)/abs(P)
    balance = abs(reaction+force)/abs(P)
    if not all(math.isfinite(v) for v in (displacement_error, load_error, balance)):
        raise ValueError("nonfinite force/displacement evidence")
    passed = (displacement_error < .03 and max(errors.values()) < .03 and
              load_error < 1e-10 and balance < 1e-8 and
              (displacement_field_error is None or displacement_field_error < .03))
    return {"schema": "tensorfem.rectangular-field-verification/1.0",
            "scope": scope, "relative_error_lt": .03,
            "displacement_relative_error": displacement_error,
            "displacement_field_relative_error": displacement_field_error,
            "stress_errors": errors, "applied_load_relative_error": load_error,
            "force_balance_relative_error": balance, "passed": passed}


def verify_mesh_study(cases, kind):
    """Require three genuinely refined meshes and a passing final field check."""
    if len(cases) < 3:
        raise ValueError("three mesh levels required for a published study")
    checks = [verify_case(c, kind) for c in cases]
    for case, check in zip(cases, checks):
        if (case.get("verification") != check or
                case.get("status") != ("qualified" if check["passed"] else "blocked")):
            raise ValueError("stored field qualification does not match recomputed errors")
    for a, b in zip(cases, cases[1:]):
        if not (b["mesh"]["nx"] > a["mesh"]["nx"] and
                b["mesh"]["ny"] > a["mesh"]["ny"]):
            raise ValueError("mesh study must refine both directions")
        if b["problem"] != a["problem"]:
            raise ValueError("mesh study changes the physical problem")
    # Patch errors at roundoff have no meaningful convergence order.
    def trend(values):
        return all(b < a or max(a, b) < 1e-10 for a, b in zip(values, values[1:]))
    converged = trend([x["displacement_relative_error"] for x in checks])
    if kind == "membrane":
        converged = converged and trend([x["displacement_field_relative_error"] for x in checks])
    for metric in checks[0]["stress_errors"]:
        converged = converged and trend([x["stress_errors"][metric] for x in checks])
    return {"schema": "tensorfem.rectangular-mesh-study/1.0",
            "mesh_levels": len(cases), "scope": checks[-1]["scope"],
            "checks": checks, "error_decreases_or_roundoff": converged,
            "passed": bool(checks[-1]["passed"] and converged)}
