"""Real Q4 FE benchmark: idealised ship hull-girder longitudinal bending.

The hull girder is represented by a welded-equivalent rectangular section.  A
tip transverse load creates a transparent Euler--Bernoulli reference, while
the Q4 continuum solve supplies reactions, nodal displacement and element
von-Mises fields.  This is a verification benchmark, not a class-rule model.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum

E, NU, T, L, H, P = 210e9, .30, .012, 1.0, .10, -100.0

def solve(nx: int, ny: int) -> dict:
    nodes, elements = rectangular_q4_mesh(L, H, nx, ny)
    forces = torch.zeros(2 * len(nodes), dtype=torch.float64)
    right = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(L)))[0]
    forces[2 * right + 1] = P / ny
    forces[2 * right[[0, -1]] + 1] *= .5
    left = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(0.0)))[0]
    fixed = torch.stack((2 * left, 2 * left + 1), 1).reshape(-1)
    result = solve_continuum(ContinuumModel(nodes, elements, nodes.new_tensor(E),
        nodes.new_tensor(NU), nodes.new_tensor(T), forces, fixed))
    tip_node = right[len(right) // 2]
    tip = float(result.displacement[2 * tip_node + 1])
    ref = P * L**3 / (3 * E * (T * H**3 / 12))
    err = abs(tip / ref - 1.0)
    centres = nodes[elements].mean(1)
    vm = torch.sqrt(result.stress[:, 0]**2 - result.stress[:, 0]*result.stress[:, 1]
                    + result.stress[:, 1]**2 + 3*result.stress[:, 2]**2)
    return {
        "mesh": {"nx": nx, "ny": ny, "nodes": len(nodes), "elements": len(elements)},
        "tip_displacement": tip, "reference_displacement": ref,
        "relative_error": err, "status": "qualified" if err < .03 else "blocked",
        "reaction_left_y": float(result.reaction[2 * left + 1].sum()),
        "field": {
            "nodes": [{"x": float(p[0]), "y": float(p[1]),
                       "ux": float(result.displacement[2*i]),
                       "uy": float(result.displacement[2*i+1])}
                      for i, p in enumerate(nodes)],
            "element_stress": [{"x": float(p[0]), "y": float(p[1]),
                                "von_mises": float(v)} for p, v in zip(centres, vm)],
        },
        # Renderer-compatible stress samples; the full fields above remain
        # machine-readable for dedicated displacement/stress cloud exporters.
        "field_samples": [{"x": float(p[0]), "y": float(p[1]),
                            "stress": float(v)} for p, v in zip(centres, vm)],
    }

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); cases = [solve(20, 4), solve(40, 8)]
    report = {"schema": "tensorfem.marine-hull-girder-fe/1.0",
      "title": "Idealised ship hull-girder longitudinal bending: Q4 FE verification",
      "problem": "A welded-equivalent hull-girder strip in longitudinal bending.",
      "calculation_conditions": {"units": "SI", "E_Pa": E, "nu": NU, "thickness_m": T,
                                  "length_m": L, "depth_m": H, "tip_load_N": P,
                                  "boundary": "clamped transverse section at x=0"},
      "calculation_process": ["generate structured Q4 meshes", "assemble and solve linear continuum FE",
                               "recover element von-Mises stress and reactions",
                               "compare centre-line tip displacement with Euler-Bernoulli reference"],
      "reference": "Euler-Bernoulli hull-girder strip tip displacement",
      "cases": cases, "acceptance": {"relative_error_lt": .03,
          "missing_field": "blocked", "over_limit": "blocked"}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "statuses": [c["status"] for c in cases],
                      "errors": [c["relative_error"] for c in cases]}))
if __name__ == "__main__": main()
