"""Run a reproducible real-Q4 FE stiffened-panel membrane benchmark.

The panel is represented by an equivalent orthotropic-free plate strip under
uniform longitudinal load; the longitudinal stiffener is included in the
effective thickness.  This is a deliberately auditable baseline for the
ship/marine benchmark chain (not an ultimate-strength claim).
"""
import argparse, json
from pathlib import Path
import torch
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum
from tensorfem.benchmark_fields import verify_case, verify_mesh_study


def solve_case(nx: int, ny: int):
    L, H, t_plate, t_stiff, E, nu, load = 2.0, 1.0, .010, .006, 210e9, .3, 1.0e6
    t = t_plate + t_stiff
    nodes, elements = rectangular_q4_mesh(L, H, nx, ny)
    forces = torch.zeros(2 * len(nodes), dtype=torch.float64)
    right = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(L)))[0]
    # Consistent edge traction integration (trapezoidal rule): ``ny`` edge
    # segments carry the prescribed *total* load.  Dividing by the number of
    # nodes and then halving the end nodes under-loads every mesh.
    forces[2 * right] = load / ny
    forces[2 * right[0]] *= .5; forces[2 * right[-1]] *= .5
    left = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(0.)))[0]
    # The analytical membrane solution permits Poisson contraction.  Constrain
    # u_x on the loaded-opposite edge and pin one u_y DOF only to remove rigid
    # translation; clamping every u_y on the edge would be a different problem.
    fixed = torch.cat((2*left, (2*left[:1] + 1)))
    result = solve_continuum(ContinuumModel(nodes, elements, nodes.new_tensor(E),
        nodes.new_tensor(nu), nodes.new_tensor(t), forces, fixed))
    tip = float(result.displacement[2*right,].mean())
    ref_tip = load * L / (E * t * H)
    ref_stress = load / (t * H)
    centers = nodes[elements].mean(1)
    vm = torch.sqrt(result.stress[:,0]**2 - result.stress[:,0]*result.stress[:,1]
                    + result.stress[:,1]**2 + 3*result.stress[:,2]**2)
    err = abs(tip-ref_tip)/abs(ref_tip)
    case = {"mesh":{"nx":nx,"ny":ny,"nodes":len(nodes),"elements":len(elements)},
      "problem": {"length_m": L, "height_m": H, "thickness_m": t,
                  "young_modulus_pa": E, "poisson": nu, "load_n": load},
      "tip_axial_displacement":tip,"reference_tip_displacement":ref_tip,
      "reference_axial_stress":ref_stress,"relative_error":err,
      "status":"qualified" if err < .03 else "blocked",
      "applied_load_x":float(forces[2*right].sum()),
      "reaction_left_x":float(result.reaction[2*left].sum()),
      "reaction_left_y":float(result.reaction[2*left+1].sum()),
      "field":{"nodes":[{"x":float(p[0]),"y":float(p[1]),"ux":float(result.displacement[2*i]),"uy":float(result.displacement[2*i+1])} for i,p in enumerate(nodes)],
               "element_stress":[{"x":float(p[0]),"y":float(p[1]),"sigma_x":float(s[0]),"sigma_y":float(s[1]),"tau_xy":float(s[2]),"von_mises":float(v)} for p,s,v in zip(centers,result.stress,vm)]}}
    case["verification"] = verify_case(case, "membrane")
    case["status"] = "qualified" if case["verification"]["passed"] else "blocked"
    return case

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    cases=[solve_case(10,5),solve_case(20,10),solve_case(40,20)]
    report={"schema":"tensorfem.marine-stiffened-panel-fe-benchmark/1.0","title":"Stiffened marine panel longitudinal membrane benchmark",
      "physical_case_id": "uniform-q4-membrane-patch", "independent_case": True,
      "problem":{"length_m":2.0,"width_m":1.0,"plate_thickness_m":.01,"stiffener_equivalent_thickness_m":.006,"young_modulus_pa":210e9,"poisson":.3,"axial_load_n":1e6},
      "reference":"uniform axial strain, sigma=P/(H*t_eff), u=PL/(EA)","cases":cases,
      "acceptance":{"relative_error_lt":.03,"required_fields":["displacement","stress","reaction","mesh_convergence"]}}
    report["mesh_convergence"] = verify_mesh_study(cases, "membrane")
    report["status"] = "qualified" if report["mesh_convergence"]["passed"] else "blocked"
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"output":str(a.output),"errors":[c["relative_error"] for c in cases],"statuses":[c["status"] for c in cases]}))
    if report["status"] != "qualified":
        raise SystemExit(1)
if __name__=='__main__': main()
