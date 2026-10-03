"""Qualification contract for a real multi-element compressed shell panel.

The model builder in this module creates actual Shell4 meshes and material
point initial fields.  The qualification runner is deliberately fail-closed:
    until the full mesh/control matrix has executed peak and post-peak evidence,
    it reports ``blocked`` instead of substituting a strip, beam, or Koiter reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .layered_shell4_plasticity import LayeredShell4Model


@dataclass(frozen=True)
class PanelCase:
    """A stress-free imperfect Shell4 panel and its analysis contract."""

    model: LayeredShell4Model
    perfect_nodes: torch.Tensor
    fixed_dofs: torch.Tensor
    reference_load: torch.Tensor
    probe_dof: int
    edge_nodes: torch.Tensor
    mesh_divisions: int
    imperfection_amplitude: float


def _structured_mesh(divisions: int, length: float, width: float, *, dtype):
    xs = torch.linspace(0.0, length, divisions + 1, dtype=dtype)
    ys = torch.linspace(0.0, width, divisions + 1, dtype=dtype)
    nodes = torch.tensor([[float(x), float(y), 0.0] for y in ys for x in xs], dtype=dtype)
    elements = []
    stride = divisions + 1
    for j in range(divisions):
        for i in range(divisions):
            n = j * stride + i
            elements.append((n, n + 1, n + 1 + stride, n + stride))
    return nodes, torch.tensor(elements, dtype=torch.long)


def _residual_stress(nodes, elements, layers, width, amplitude):
    """Return a symmetric, zero-resultant longitudinal weld-stress surrogate."""
    gp = 1.0 / math.sqrt(3.0)
    natural = ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp))
    field = torch.zeros((len(elements), 4, layers, 3), dtype=nodes.dtype)
    for e, conn in enumerate(elements):
        xy = nodes[conn, :2]
        for q, (xi, eta) in enumerate(natural):
            shape = nodes.new_tensor(((1-xi)*(1-eta), (1+xi)*(1-eta),
                                      (1+xi)*(1+eta), (1-xi)*(1+eta))) / 4
            y = torch.dot(shape, xy[:, 1])
            # One complete cosine wave is self-balanced over the panel width.
            field[e, q, :, 0] = amplitude * torch.cos(2 * math.pi * y / width)
    return field


def build_panel_case(
    divisions: int,
    *,
    length: float = 1.0,
    width: float = 1.0,
    thickness: float = 0.01,
    young: float = 210.0e9,
    poisson: float = 0.3,
    yield_stress: float = 250.0e6,
    hardening: float = 2.1e9,
    layers: int = 5,
    imperfection_ratio: float = 0.1,
    residual_ratio: float = 0.2,
    dtype: torch.dtype = torch.float64,
) -> PanelCase:
    """Build a simply supported square plate under longitudinal compression.

    The reference load has unit resultant.  The left edge fixes longitudinal
    translation; all boundary nodes fix transverse translation, while one
    corner removes the remaining in-plane rigid translation.  Rotations are
    free.  The sine imperfection is part of the reference geometry and is
    therefore stress free.
    """
    if divisions < 2 or divisions % 2:
        raise ValueError("divisions must be an even integer >= 2")
    if min(length, width, thickness, young, yield_stress) <= 0:
        raise ValueError("panel geometry and material constants must be positive")
    if not (0 <= imperfection_ratio <= 1 and 0 <= residual_ratio < 1):
        raise ValueError("invalid initial-field ratio")
    perfect, elements = _structured_mesh(divisions, length, width, dtype=dtype)
    amplitude = imperfection_ratio * thickness
    imperfect = perfect.clone()
    imperfect[:, 2] = amplitude * torch.sin(math.pi * perfect[:, 0] / length) \
        * torch.sin(math.pi * perfect[:, 1] / width)
    residual = _residual_stress(perfect, elements, layers, width,
                                residual_ratio * yield_stress)
    model = LayeredShell4Model(imperfect, elements, young, poisson, thickness,
                               yield_stress, hardening, layers=layers,
                               residual_stress=residual)
    model.validate()

    tol = 8 * torch.finfo(dtype).eps * max(length, width)
    left = torch.nonzero(torch.abs(perfect[:, 0]) <= tol).flatten()
    right = torch.nonzero(torch.abs(perfect[:, 0] - length) <= tol).flatten()
    boundary = torch.nonzero(
        (torch.abs(perfect[:, 0]) <= tol) | (torch.abs(perfect[:, 0]-length) <= tol)
        | (torch.abs(perfect[:, 1]) <= tol) | (torch.abs(perfect[:, 1]-width) <= tol)
    ).flatten()
    fixed = set(int(6*n+2) for n in boundary)  # simply-supported transverse condition
    fixed.update(int(6*n) for n in left)
    fixed.add(1)  # one corner fixes the in-plane y rigid mode

    force = torch.zeros(model.n_dofs, dtype=dtype)
    # Trapezoidal edge weights give exactly unit compressive resultant.
    ordered = right[torch.argsort(perfect[right, 1])]
    weights = torch.ones(len(ordered), dtype=dtype)
    weights[[0, -1]] = 0.5
    weights /= weights.sum()
    force[6*ordered] = -weights
    centre = (divisions // 2) * (divisions + 1) + divisions // 2
    return PanelCase(model, perfect, torch.tensor(sorted(fixed), dtype=torch.long),
                     force, 6*centre+2, right, divisions, amplitude)


def classical_panel_references(case: PanelCase) -> dict[str, float]:
    """Independent elastic-buckling and squash-load bounds for the square plate."""
    x = case.perfect_nodes[:, 0]; y = case.perfect_nodes[:, 1]
    length = float(x.max()-x.min()); width = float(y.max()-y.min())
    model = case.model
    rigidity = model.young * model.thickness**3 / (12*(1-model.poisson**2))
    # Long plate/square simply-supported uniaxial compression, k=4.
    nx_critical = 4 * math.pi**2 * rigidity / width**2
    elastic_force = nx_critical * width
    squash_force = model.yield_stress * model.thickness * width
    return {
        "classical_elastic_buckling_force": elastic_force,
        "gross_section_squash_force": squash_force,
        "slenderness": math.sqrt(squash_force / elastic_force),
        "reference_aspect_ratio": length / width,
    }


def panel_contract(divisions=(4, 8, 12), arc_steps=(0.02, 0.01)) -> dict[str, object]:
    """Return the mesh/control matrix required for future qualification."""
    cases = [build_panel_case(n) for n in divisions]
    rows = []
    for case in cases:
        refs = classical_panel_references(case)
        rows.append({
            "divisions": case.mesh_divisions,
            "elements": len(case.model.elements),
            "nodes": len(case.model.nodes),
            "dofs": case.model.n_dofs,
            "imperfection_amplitude": case.imperfection_amplitude,
            "reference_load_resultant": float(-case.reference_load[0::6].sum()),
            **refs,
        })
    return {
        "schema": "tensorfem.marine-panel-ultimate-fe/1.0",
        "model": "real multi-element Shell4 simply-supported square plate",
        "mesh_matrix": rows,
        "arc_length_controls": list(arc_steps),
        "required_outputs": [
            "peak_force", "post_peak_force_at_1.5_peak_shortening",
            "centre_deflection", "yielded_fraction", "failure_mode",
            "free_dof_equilibrium_norm", "external_work",
            "recoverable_energy", "plastic_dissipation", "energy_residual",
        ],
        "acceptance": {
            "peak_mesh_change": 0.03,
            "peak_arc_step_change": 0.03,
            "post_peak_arc_step_change": 0.05,
            "relative_equilibrium_norm": 1e-6,
            "relative_energy_residual": 1e-4,
        },
        "reference_kind": "classical analytical bounds; not an external experiment",
    }


def run_panel_ultimate_qualification() -> dict[str, object]:
    """Fail closed until the declared panel mesh/control matrix is executed."""
    result = panel_contract()
    result.update({
        "status": "blocked",
        "passed": False,
        "blocking_capability": (
            "4/8/12 meshes and both arc-length controls require executed peak, "
            "post-peak, equilibrium, and energy-convergence evidence"
        ),
        "boundary": (
            "The meshes and initial fields are real FE inputs. No peak or post-peak "
            "claim is made, and no Koiter/strip/beam reduced model is substituted."
        ),
    })
    return result
