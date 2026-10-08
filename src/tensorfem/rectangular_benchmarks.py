"""Actual Q4 fields for the rectangular bending verification example."""
import torch

from .benchmark_fields import verify_case, verify_mesh_study
from .continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum


def solve_bending(nx, ny):
    L, H, t, E, nu, P = 1., .1, .012, 210e9, .3, -100.
    nodes, elements = rectangular_q4_mesh(L, H, nx, ny)
    forces = torch.zeros(2*len(nodes), dtype=torch.float64)
    right = torch.where(nodes[:, 0] == L)[0]
    forces[2*right+1] = P/ny
    forces[2*right[[0, -1]]+1] *= .5
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((2*left, 2*left+1), 1).reshape(-1)
    result = solve_continuum(ContinuumModel(nodes, elements, nodes.new_tensor(E),
                            nodes.new_tensor(nu), nodes.new_tensor(t), forces, fixed))
    centres = nodes[elements].mean(1)
    vm = torch.sqrt(result.stress[:, 0]**2-result.stress[:, 0]*result.stress[:, 1]
                    + result.stress[:, 1]**2 + 3*result.stress[:, 2]**2)
    tip = float(result.displacement[2*right[len(right)//2]+1])
    reference = P*L**3/(3*E*(t*H**3/12))
    case = {
        "mesh": {"nx": nx, "ny": ny, "nodes": len(nodes), "elements": len(elements)},
        "problem": {"length_m": L, "height_m": H, "thickness_m": t,
                    "young_modulus_pa": E, "poisson": nu, "load_n": P},
        "tip_displacement": tip, "reference_displacement": reference,
        "relative_error": abs(tip/reference-1),
        "applied_load_y": float(forces[2*right+1].sum()),
        "reaction_left_y": float(result.reaction[2*left+1].sum()),
        "field": {
            "nodes": [{"x": float(p[0]), "y": float(p[1]),
                       "ux": float(result.displacement[2*i]),
                       "uy": float(result.displacement[2*i+1])} for i, p in enumerate(nodes)],
            "element_stress": [{"x": float(p[0]), "y": float(p[1]),
                                "sigma_x": float(s[0]), "sigma_y": float(s[1]),
                                "tau_xy": float(s[2]), "von_mises": float(v)}
                               for p, s, v in zip(centres, result.stress, vm)]},
    }
    case["verification"] = verify_case(case, "bending")
    case["status"] = "qualified" if case["verification"]["passed"] else "blocked"
    return case


def bending_report():
    cases = [solve_bending(nx, ny) for nx, ny in ((20, 4), (40, 8), (80, 16))]
    study = verify_mesh_study(cases, "bending")
    return {"schema": "tensorfem.rectangular-bending-report/2.0",
            "title": "Rectangular Q4 cantilever bending",
            "physical_case_id": "rectangular-q4-cantilever",
            "independent_case": True,
            "reference": "Euler-Bernoulli displacement and interior beam stress",
            "cases": cases, "mesh_convergence": study,
            "status": "qualified" if study["passed"] else "blocked",
            "acceptance": {"relative_error_lt": .03,
                           "stress_scope": study["scope"],
                           "force_balance_relative_error_lt": 1e-8}}
