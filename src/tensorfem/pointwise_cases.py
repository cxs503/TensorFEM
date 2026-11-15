"""Cases stored in the pointwise reference-regression store.

Each builder returns ``(computed, analytic)``:

* ``computed`` is a :class:`~tensorfem.pointwise_regression.Snapshot` of values
  produced by the **production** solver path of this package (no re-implemented
  element, no hand-written matrices).  It is what the store holds and what a
  regression check recomputes.
* ``analytic`` is the same probe set with the closed-form field evaluated at the
  probe coordinates, or ``None`` when the case has no pointwise exact solution
  (the Cook membrane is distorted and has none).  ``None`` is a first-class
  answer: the case then only supports regression mode, and the harness must not
  pretend a "reference" exists.

The cases are chosen to cover the three situations a field regression has to
distinguish:

``q4-uniaxial-patch``
    Patch test: bilinear Q4 reproduces a linear displacement field exactly, so
    the computed field must match the closed form to round-off.  This pins the
    element, the assembly and the load vector; a failure here is an element
    defect, not a discretisation error.  It also exercises the zero-component
    rule (``sigma_yy = 0`` identically).

``thermal-harmonic-square``
    Steady conduction with Dirichlet data from the exact harmonic field
    ``T = sin(pi x) sinh(pi y)/sinh(pi)``.  Nodal temperatures converge at
    ``O(h^2)`` so the analytic gate is meaningful at a fixed mesh; the
    element-centre flux (the engineering quantity of interest) is captured for
    regression only.  This case pins the thermal assembly and the steady solve.

``cook-membrane-q4``
    The classical distorted Cook membrane: no pointwise exact solution, so the
    store is a pure regression baseline (drift detection) - the situation most
    large codes are actually in.  It pins the distorted-mapping path and the
    stress recovery.
"""
from __future__ import annotations

import datetime

import torch

from . import __version__
from .continuum import ContinuumModel, solve_continuum
from .continuum_benchmarks import cook_membrane
from .pointwise_regression import (
    Snapshot,
    build_snapshot,
    mesh_signature,
    sample_element_centres,
    sample_nodes,
)
from .thermal_analytic import (
    element_centre_gradients,
    harmonic_gradient,
    harmonic_temperature,
    solve_dirichlet,
    structured_quad_mesh,
)

__all__ = [
    "CASE_BUILDERS",
    "ANALYTIC_REFERENCES",
    "uniaxial_patch",
    "thermal_harmonic_square",
    "cook_membrane_q4",
]

def _environment() -> dict[str, object]:
    return {
        "tensorfem_version": __version__,
        "torch_version": torch.__version__,
        "torch_threads": int(torch.get_num_threads()),
        "dtype": "float64",
    }


def utc_now() -> str:
    """Capture timestamp, kept in ``meta`` only (never compared)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


_E = 210.0e9
_NU = 0.3
_THICKNESS = 0.01
_SIGMA0 = 1.0e6


# ---------------------------------------------------------------------------
# 1. uniaxial patch test (pointwise exact)
# ---------------------------------------------------------------------------
def _uniaxial_model(nx: int, ny: int) -> ContinuumModel:
    """Rectangular Q4 patch under uniform uniaxial traction ``sigma0``."""
    if nx < 2 or ny < 2:
        raise ValueError("the patch test needs at least a 2x2 mesh")
    length, height = 2.0, 1.0
    x = torch.linspace(0.0, length, nx + 1, dtype=torch.float64)
    y = torch.linspace(0.0, height, ny + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append((n0, n0 + 1, n0 + nx + 2, n0 + nx + 1))
    elements = torch.tensor(cells, dtype=torch.long)

    forces = torch.zeros(2 * len(nodes), dtype=torch.float64)
    right = torch.arange(nx, (ny + 1) * (nx + 1), nx + 1)
    edge = _SIGMA0 * _THICKNESS * height / ny            # consistent nodal load
    forces[2 * right] = edge
    forces[2 * right[[0, -1]]] *= 0.5                    # half load at the corners
    left = torch.arange(0, (ny + 1) * (nx + 1), nx + 1)
    # ux = 0 on x = 0 removes the x translation and the rotation; uy(0, 0) = 0
    # (dof 1, NOT dof 0 - a duplicate of the ux constraint would leave the
    # vertical translation free and make the system singular).
    fixed = [int(2 * n) for n in left] + [1]
    return ContinuumModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor(_E, dtype=torch.float64),
        poisson_ratio=torch.tensor(_NU, dtype=torch.float64),
        thickness=torch.tensor(_THICKNESS, dtype=torch.float64),
        forces=forces,
        fixed_dofs=torch.tensor(sorted(fixed), dtype=torch.long),
        element_type="q4",
        plane="stress",
    )


def _uniaxial_exact(points: torch.Tensor) -> dict[str, torch.Tensor]:
    x, y = points[:, 0], points[:, 1]
    return {
        "ux": _SIGMA0 * x / _E,
        "uy": -_NU * _SIGMA0 * y / _E,
        "sigma_xx": torch.full_like(x, _SIGMA0),
        "sigma_yy": torch.zeros_like(x),
        "tau_xy": torch.zeros_like(x),
    }


def _uniaxial_stress_exact(points: torch.Tensor) -> dict[str, torch.Tensor]:
    full = _uniaxial_exact(points)
    return {k: full[k] for k in ("sigma_xx", "sigma_yy", "tau_xy")}


def uniaxial_patch(nx: int = 4, ny: int = 4) -> tuple[Snapshot, Snapshot]:
    """Uniform uniaxial tension: exact linear field, reproduced by Q4."""
    model = _uniaxial_model(nx, ny)
    result = solve_continuum(model)
    nodes = model.nodes
    centres = nodes[model.elements].mean(dim=1)
    ux = result.displacement[0::2]
    uy = result.displacement[1::2]

    node_ids = [j * (nx + 1) + i for j in range(1, ny) for i in range(1, nx)]
    elem_ids = list(range(len(model.elements)))

    fields = {"ux": ux, "uy": uy}
    exact_nodes = _uniaxial_exact(nodes)
    probes_c = sample_nodes(nodes, fields, node_ids)
    probes_e = sample_element_centres(
        centres,
        {"sigma_xx": result.stress[:, 0], "sigma_yy": result.stress[:, 1],
         "tau_xy": result.stress[:, 2]},
        elem_ids,
    )
    probes_ca = sample_nodes(nodes, {k: exact_nodes[k] for k in ("ux", "uy")}, node_ids)
    probes_ea = sample_element_centres(centres, _uniaxial_stress_exact(centres), elem_ids)

    meta = {
        "case_key": "q4-uniaxial-patch",
        "solver": "tensorfem.continuum.solve_continuum (dense direct)",
        "discretisation": {"nx": nx, "ny": ny, "elements": int(len(model.elements)),
                           "nodes": int(len(nodes)), "dofs": int(model.n_dofs)},
        "parameters": {"length": 2.0, "height": 1.0, "young_modulus": _E,
                       "poisson_ratio": _NU, "thickness": _THICKNESS,
                       "traction": _SIGMA0, "plane": "stress"},
        "mesh_signature": mesh_signature(nodes, model.elements),
        "has_analytic_reference": True,
        "quantity_units": {"ux": "m", "uy": "m", "sigma_xx": "Pa", "sigma_yy": "Pa",
                           "tau_xy": "Pa"},
        # sigma_yy and tau_xy are identically zero, so their absolute floor is
        # taken from the applied traction instead of from (empty) reference data.
        "quantity_scales": {"ux": 0.0, "uy": 0.0, "sigma_xx": _SIGMA0,
                            "sigma_yy": _SIGMA0, "tau_xy": _SIGMA0},
        "environment": _environment(),
        "created_utc": utc_now(),
    }
    quantities = ("ux", "uy", "sigma_xx", "sigma_yy", "tau_xy")
    computed = build_snapshot("q4-uniaxial-patch", quantities, probes_c + probes_e, meta)
    analytic = build_snapshot("q4-uniaxial-patch", quantities, probes_ca + probes_ea,
                              dict(meta, source="closed-form uniaxial field"))
    return computed, analytic


# ---------------------------------------------------------------------------
# 2. thermal harmonic square
# ---------------------------------------------------------------------------
def thermal_harmonic_square(nx: int = 16, ny: int = 16,
                            stride: int = 2) -> tuple[Snapshot, Snapshot | None]:
    """Steady conduction with Dirichlet data taken from the exact harmonic field."""
    mesh = structured_quad_mesh(nx, ny)
    temperature = solve_dirichlet(mesh, harmonic_temperature)
    centres, gradient = element_centre_gradients(mesh, temperature)
    flux = -gradient            # k = 1, q = -k grad T

    step = max(stride, 1)
    node_ids = [j * (nx + 1) + i for j in range(step, ny, step)
                for i in range(step, nx, step)]
    elem_ids = [j * nx + i for j in range(step - 1, ny, step)
                for i in range(step - 1, nx, step)]

    valid_n = torch.ones(len(mesh.nodes), dtype=torch.bool)
    valid_n[mesh.boundary_nodes] = False
    node_ids = [i for i in node_ids if bool(valid_n[i])]
    if not node_ids:
        raise ValueError("probe stride left no interior thermal probes")

    exact_nodes = {"T": harmonic_temperature(mesh.nodes)}
    exact_flux = -harmonic_gradient(centres)
    fields = {"T": temperature}
    probes_c = sample_nodes(mesh.nodes, fields, node_ids)
    probes_e = sample_element_centres(centres, {"q_x": flux[:, 0], "q_y": flux[:, 1]},
                                      elem_ids)
    probes_ca = sample_nodes(mesh.nodes, exact_nodes, node_ids)
    probes_ea = sample_element_centres(centres, {"q_x": exact_flux[:, 0],
                                                 "q_y": exact_flux[:, 1]}, elem_ids)

    meta = {
        "case_key": "thermal-harmonic-square",
        "solver": "tensorfem.thermal.solve_steady_thermal (Dirichlet from exact data)",
        "discretisation": {"nx": nx, "ny": ny, "elements": int(len(mesh.elements)),
                           "nodes": int(len(mesh.nodes))},
        "parameters": {"lx": 1.0, "ly": 1.0, "conductivity": 1.0,
                       "field": "T = sin(pi x) sinh(pi y)/sinh(pi)"},
        "mesh_signature": mesh_signature(mesh.nodes, mesh.elements),
        "has_analytic_reference": True,
        "quantity_units": {"T": "K", "q_x": "W/m^2", "q_y": "W/m^2"},
        "environment": _environment(),
        "created_utc": utc_now(),
    }
    quantities = ("T", "q_x", "q_y")
    computed = build_snapshot("thermal-harmonic-square", quantities, probes_c + probes_e, meta)
    analytic = build_snapshot("thermal-harmonic-square", quantities,
                              probes_ca + probes_ea,
                              dict(meta, source="closed-form harmonic field"))
    return computed, analytic


# ---------------------------------------------------------------------------
# 3. Cook membrane (regression only: no pointwise exact solution)
# ---------------------------------------------------------------------------
def cook_membrane_q4(nx: int = 8, stride: int = 2) -> tuple[Snapshot, None]:
    """Classical Cook membrane on the production Q4 path; regression baseline only."""
    model, _ = cook_membrane(nx)
    result = solve_continuum(model)
    nodes = model.nodes
    elements = model.elements
    centres = nodes[elements].mean(dim=1)
    step = max(stride, 1)
    node_ids = list(range(0, len(nodes), step))
    elem_ids = list(range(0, len(elements), step))

    probes_c = sample_nodes(nodes, {"ux": result.displacement[0::2],
                                    "uy": result.displacement[1::2]}, node_ids)
    probes_e = sample_element_centres(
        centres,
        {"sigma_xx": result.stress[:, 0], "sigma_yy": result.stress[:, 1],
         "tau_xy": result.stress[:, 2]},
        elem_ids,
    )
    meta = {
        "case_key": "cook-membrane-q4",
        "solver": "tensorfem.continuum.solve_continuum (dense direct), Cook model",
        "discretisation": {"nx": nx, "elements": int(len(elements)),
                           "nodes": int(len(nodes)), "dofs": int(model.n_dofs)},
        "parameters": {"corners": [[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]],
                       "young_modulus": 1.0, "poisson_ratio": 1.0 / 3.0,
                       "thickness": 1.0, "loaded_edge_resultant": 1.0},
        "mesh_signature": mesh_signature(nodes, elements),
        "has_analytic_reference": False,
        "analytic_reference_reason": (
            "the Cook membrane is a distorted quadrilateral domain with no pointwise "
            "closed-form elasticity solution; the stored values are a regression "
            "baseline, not an accuracy reference"
        ),
        "quantity_units": {"ux": "m", "uy": "m", "sigma_xx": "Pa", "sigma_yy": "Pa",
                           "tau_xy": "Pa"},
        "environment": _environment(),
        "created_utc": utc_now(),
    }
    quantities = ("ux", "uy", "sigma_xx", "sigma_yy", "tau_xy")
    computed = build_snapshot("cook-membrane-q4", quantities, probes_c + probes_e, meta)
    return computed, None


CASE_BUILDERS = {
    "q4-uniaxial-patch": uniaxial_patch,
    "thermal-harmonic-square": thermal_harmonic_square,
    "cook-membrane-q4": cook_membrane_q4,
}


_ANALYTIC_CACHE: dict[str, Snapshot | None] = {}


def analytic_reference(case_id: str) -> Snapshot | None:
    """Closed-form field at the case probes, or ``None`` when none exists.

    Not stored on disk: it is derived from the closed forms above, so storing it
    would create a second, editable source of truth for the same numbers.
    """
    if case_id not in CASE_BUILDERS:
        raise KeyError(f"unknown pointwise case {case_id!r}")
    if case_id not in _ANALYTIC_CACHE:
        _ANALYTIC_CACHE[case_id] = CASE_BUILDERS[case_id]()[1]
    return _ANALYTIC_CACHE[case_id]



def analytic_case_ids() -> tuple[str, ...]:
    """Cases whose probe set has a closed-form counterpart (running this builds them)."""
    return tuple(cid for cid in CASE_BUILDERS
                 if CASE_BUILDERS[cid]()[0].meta.get("has_analytic_reference"))
