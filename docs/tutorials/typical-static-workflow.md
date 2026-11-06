# Tutorial: the typical TensorFEM analysis loop

This tutorial walks through the loop every TensorFEM study repeats: **define the
geometry → build and gate the mesh → assemble and solve → recover and verify the
fields → accept or reject on declared tolerances**. It is deliberately *runnable
and self-checking*: every `python` block below is executed by
`tests/test_tutorial_static_workflow.py`, the assertions inside the blocks must
hold, and the numbers recorded in the
[machine-checked fact table](#7-record-the-numbers) must equal the numbers the
code computes. A tutorial that drifts away from the code it shows fails the test
suite.

The physics is intentionally small so the whole loop finishes in a few seconds on
a laptop, but nothing in the structure changes for a real hull-panel model: the
same gates, the same "recompute, never trust a stored pass flag" rule, and the
same declared sampling region apply.

| Item | Value |
|---|---|
| Case | Rectangular Q4 cantilever strip, plane stress, distributed end load |
| Geometry | L = 1.0 m, H = 0.1 m, t = 0.012 m |
| Material | E = 210 GPa, ν = 0.30 |
| Load | P = −100 N vertical |
| Meshes | 10×2, 20×4, 40×8 Q4 (three refinement levels) |
| Independently known reference | Euler–Bernoulli tip deflection, interior beam stress |
| Acceptance | displacement error < 3 %, σ_x and τ_xy L2 error < 3 %, force imbalance < 1e-8 |
| Extra analysis axis | Euler–Bernoulli free vibration of the same section properties |

## 0. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest tests/test_tutorial_static_workflow.py -q
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` stops torch from probing optional accelerators at
import; the workflow runs on whatever device the tensors carry.

## 1. Preflight: precision and units

Two decisions are made before any solver call, because neither can be repaired
afterwards: the **dtype** and the **unit system**. Structural stiffness scales
with E ≈ 1e11 Pa, so a 3 % field gate is not reachable in float32; this tutorial
pins float64 and states SI units in the names of the constants.

```python
import torch

from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum
from tensorfem.modal import cantilever_beam_modes, cantilever_exact_angular_frequency

DTYPE = torch.float64
assert torch.finfo(DTYPE).eps < 1e-15, "this workflow requires double precision"

L, H, THICK = 1.0, 0.1, 0.012          # m: length, height, thickness
E, NU = 210e9, 0.30                    # Pa, -: steel, plane stress
LOAD = -100.0                          # N: vertical end load, pointing down
RHO = 7850.0                           # kg/m^3: only used by the modal step
```

## 2. Mesh generation and the mesh gate

No assembly happens until the mesh has been gated. The gate checks that every Q4
is counter-clockwise (positive signed area), that the mesh fills the declared
plate, and that no node is duplicated. These are cheap checks that catch the
classic "the answer is wrong because the mesh was wrong" failure before a single
element matrix is built.

```python
def build_mesh(nx, ny):
    """Structured Q4 mesh for the strip, plus the gate it must pass."""
    nodes, elements = rectangular_q4_mesh(L, H, nx, ny, dtype=DTYPE)
    corners = nodes[elements]                       # [n_elem, 4, 2]
    x, y = corners[..., 0], corners[..., 1]
    nxt = (1, 2, 3, 0)                              # shoelace, counter-clockwise
    area = 0.5 * sum(
        x[:, i] * y[:, nxt[i]] - x[:, nxt[i]] * y[:, i] for i in range(4)
    )
    report = {
        "nodes": int(nodes.shape[0]),
        "elements": int(elements.shape[0]),
        "min_area": float(area.min()),
        "total_area": float(area.sum()),
        "unique_nodes": int(torch.unique(nodes, dim=0).shape[0]),
    }
    assert bool((area > 0).all()), "Q4 connectivity must be counter-clockwise"
    assert abs(report["total_area"] - L * H) < 1e-12, "mesh does not fill the plate"
    assert report["unique_nodes"] == report["nodes"], "duplicate nodes in the mesh"
    assert int(elements.min()) >= 0 and int(elements.max()) < report["nodes"]
    return nodes, elements, report


nodes, elements, mesh_report = build_mesh(20, 4)
print(mesh_report)
assert mesh_report["nodes"] == 105 and mesh_report["elements"] == 80
```

## 3. Assembly, boundary conditions and solve

The load is applied on the loaded edge as a tributary distribution; the two corner
nodes take half weight, so the resultant equals `LOAD` exactly. The fixed edge
clamps both displacement components. `solve_continuum` assembles the global
stiffness, applies the Dirichlet data, solves the reduced system and returns
displacement, reaction, strain and stress in one object.

```python
def run_case(nx, ny):
    nodes, elements, mesh_report = build_mesh(nx, ny)
    right = torch.where(nodes[:, 0] == L)[0]
    left = torch.where(nodes[:, 0] == 0)[0]
    assert right.numel() == ny + 1 and left.numel() == ny + 1, "edge node sets incomplete"

    forces = torch.zeros(2 * nodes.shape[0], dtype=DTYPE)
    forces[2 * right + 1] = LOAD / ny                # ny tributary segments
    forces[2 * right[[0, -1]] + 1] *= 0.5            # half weight at the corners
    assert abs(float(forces.sum()) - LOAD) < 1e-12, "resultant must equal the applied load"

    fixed = torch.stack((2 * left, 2 * left + 1), 1).reshape(-1)
    model = ContinuumModel(
        nodes=nodes,
        elements=elements,
        young_modulus=nodes.new_tensor(E),
        poisson_ratio=nodes.new_tensor(NU),
        thickness=nodes.new_tensor(THICK),
        forces=forces,
        fixed_dofs=fixed,
    )
    return {
        "mesh": mesh_report,
        "nodes": nodes,
        "elements": elements,
        "left": left,
        "right": right,
        "applied": float(forces[2 * right + 1].sum()),
        "result": solve_continuum(model),
    }


case = run_case(20, 4)
print("max |displacement| [mm]:", float(case["result"].displacement.abs().max()) * 1e3)
```

## 4. Field recovery and verification

Verification recomputes every reported quantity from the raw nodal and element
fields. Stored pass flags are never trusted: if the recovery, the reference or the
comparison changes, the number changes with it.

Two scope rules matter and are fixed *before* the numbers are looked at:

* **Tip deflection** is sampled at the mid-height node of the loaded edge and
  compared with `δ = P L³ / (3 E I)`, `I = t H³ / 12`.
* **Stress** is compared only at element centres with `0.2L ≤ x ≤ 0.8L`. The clamp
  region (singular in 2-D elasticity) and the loaded edge (load introduction) are
  outside the declared region; including them would measure boundary-layer error,
  not beam-theory error.

```python
def relative_l2(computed, reference):
    return float(torch.linalg.norm(computed - reference) / torch.linalg.norm(reference))


def metrics(case):
    """Recompute displacement, stress and force-balance errors from raw fields."""
    nodes, elements, result = case["nodes"], case["elements"], case["result"]
    inertia = THICK * H**3 / 12                                  # m^4
    reference = LOAD * L**3 / (3 * E * inertia)                  # m
    right = case["right"]
    tip = float(result.displacement[2 * right[right.numel() // 2] + 1])

    centres = nodes[elements].mean(1)                            # exact for rectangular Q4
    band = (centres[:, 0] >= 0.2 * L) & (centres[:, 0] <= 0.8 * L)
    sigma_x = -LOAD * (L - centres[band, 0]) * centres[band, 1] / inertia
    tau_xy = LOAD / (2 * inertia) * (H**2 / 4 - centres[band, 1] ** 2)
    stress = result.stress[band]
    reaction = float(result.reaction[2 * case["left"] + 1].sum())

    return {
        "tip": tip,
        "reference": reference,
        "displacement_error": abs(tip / reference - 1),
        "sigma_x_l2": relative_l2(stress[:, 0], sigma_x),
        "tau_xy_l2": relative_l2(stress[:, 2], tau_xy),
        "applied": case["applied"],
        "reaction": reaction,
        "imbalance": abs(reaction + case["applied"]) / abs(case["applied"]),
    }


print(metrics(case))
```

## 5. Refinement study and the acceptance gate

One converged-looking number is not evidence. The same case is run on three
meshes; the gate then has to **reject** the coarse meshes and **accept** the
finest one, and the error sequence has to be monotone. A gate that accepts
everything is not a gate, so the rejection is asserted too.

```python
MESHES = ((10, 2), (20, 4), (40, 8))
CASES = {}
for nx, ny in MESHES:
    run = run_case(nx, ny)
    run["metrics"] = metrics(run)
    CASES[(nx, ny)] = run

LIMITS = {                       # declared before the numbers are compared
    "displacement_error": 3e-2,
    "sigma_x_l2": 3e-2,
    "tau_xy_l2": 3e-2,
    "imbalance": 1e-8,
}


def gate(m):
    failed = [name for name, limit in LIMITS.items() if not m[name] <= limit]
    return {"passed": not failed, "failed": failed}


print("| mesh | nodes/elements | tip deflection (mm) | disp. error | sigma_x L2 | tau_xy L2 | gate |")
print("|---|---:|---:|---:|---:|---:|---|")
for nx, ny in MESHES:
    m = CASES[(nx, ny)]["metrics"]
    verdict = gate(m)
    print(
        f"| {nx}x{ny} | {CASES[(nx, ny)]['mesh']['nodes']}/{CASES[(nx, ny)]['mesh']['elements']} "
        f"| {1e3 * m['tip']:.6f} | {100 * m['displacement_error']:.3f}% "
        f"| {100 * m['sigma_x_l2']:.3f}% | {100 * m['tau_xy_l2']:.3f}% "
        f"| {'qualified' if verdict['passed'] else 'blocked: ' + ','.join(verdict['failed'])} |"
    )

fine = CASES[(40, 8)]["metrics"]
assert gate(fine)["passed"], f"finest mesh must meet the declared gate: {gate(fine)['failed']}"
assert not gate(CASES[(20, 4)]["metrics"])["passed"], "a coarse mesh must be rejected, not excused"
assert not gate(CASES[(10, 2)]["metrics"])["passed"], "a coarse mesh must be rejected, not excused"
errors = [CASES[key]["metrics"]["displacement_error"] for key in MESHES]
assert errors[0] > errors[1] > errors[2], f"refinement must reduce the error, got {errors}"
assert fine["imbalance"] < 1e-8, "internal and external forces must balance"
```

The coarse rows stay `blocked` in the table. That is the point of the loop: the
refinement path is what turns a blocked number into a qualified one, and the
tutorial keeps the intermediate verdicts visible instead of hiding them.

## 6. Reusing the discretisation for a second analysis type

The same section properties drive a free-vibration check, which is the cheapest
independent second axis: the Euler–Bernoulli eigenvalues have a closed form
(roots of `cos(βL)cosh(βL) = -1`) and the finite-element frequencies approach them
from above with a consistent mass matrix. Nothing from the static solve is reused
except the section data, so agreement is real evidence and not a tautology.

```python
inertia = THICK * H**3 / 12
mass_per_length = RHO * THICK * H                 # kg/m
modal = cantilever_beam_modes(L, 40, E * inertia, mass_per_length, modes=3)
MODAL = []
for mode in range(3):
    exact = cantilever_exact_angular_frequency(L, E * inertia, mass_per_length, mode + 1)
    computed = float(modal.angular_frequencies[mode])
    MODAL.append(
        {
            "mode": mode + 1,
            "omega": computed,
            "exact": exact,
            "error": abs(computed / exact - 1),
            "hz": float(modal.frequencies_hz[mode]),
        }
    )
    assert abs(computed / exact - 1) < 1e-4, f"mode {mode + 1} missed its reference"

print("| mode | omega (rad/s) | reference (rad/s) | relative error | frequency (Hz) |")
print("|---|---:|---:|---:|---:|")
for entry in MODAL:
    print(
        f"| {entry['mode']} | {entry['omega']:.7f} | {entry['exact']:.7f} "
        f"| {entry['error']:.3e} | {entry['hz']:.6f} |"
    )
```

## 7. Record the numbers

The fact table below is the tutorial's contract with its test. The test re-runs
every block, compares each recorded value with the computed one at a relative
tolerance of 1e-6, and fails on any missing or drifted key. Values printed by the
tables above are rounded for readability and are not the checked ones; the checked
ones are here.

<!-- tutorial-facts:begin -->
| key | value |
|---|---|
| `mesh_10x2_nodes` | 33 |
| `mesh_10x2_elements` | 20 |
| `mesh_20x4_nodes` | 105 |
| `mesh_20x4_elements` | 80 |
| `mesh_40x8_nodes` | 369 |
| `mesh_40x8_elements` | 320 |
| `mesh_40x8_min_area_m2` | 0.0003125 |
| `reference_deflection_mm` | -0.158730159 |
| `tip_10x2_mm` | -0.112870290 |
| `tip_20x4_mm` | -0.144485040 |
| `tip_40x8_mm` | -0.155519679 |
| `displacement_error_10x2` | 0.288917168 |
| `displacement_error_20x4` | 0.0897442474 |
| `displacement_error_40x8` | 0.0202260243 |
| `sigma_x_l2_40x8` | 0.0249455880 |
| `tau_xy_l2_40x8` | 0.0121054282 |
| `applied_load_n` | -100 |
| `reaction_left_40x8_n` | 99.9999999987 |
| `modal_omega_1` | 524.9705586 |
| `modal_omega_2` | 3289.934767 |
| `modal_omega_3` | 9211.920891 |
| `modal_frequency_hz_1` | 83.55165939 |
<!-- tutorial-facts:end -->

`min_area` is recorded because it is a mesh-quality witness, not because it is a
result: on the 40×8 mesh every element has the same area by construction.

```python
FACTS = {}
for nx, ny in MESHES:
    entry = CASES[(nx, ny)]
    FACTS[f"mesh_{nx}x{ny}_nodes"] = float(entry["mesh"]["nodes"])
    FACTS[f"mesh_{nx}x{ny}_elements"] = float(entry["mesh"]["elements"])
    FACTS[f"tip_{nx}x{ny}_mm"] = 1e3 * entry["metrics"]["tip"]
    FACTS[f"displacement_error_{nx}x{ny}"] = entry["metrics"]["displacement_error"]
FACTS["mesh_40x8_min_area_m2"] = CASES[(40, 8)]["mesh"]["min_area"]
FACTS["reference_deflection_mm"] = 1e3 * fine["reference"]
FACTS["sigma_x_l2_40x8"] = fine["sigma_x_l2"]
FACTS["tau_xy_l2_40x8"] = fine["tau_xy_l2"]
FACTS["applied_load_n"] = fine["applied"]
FACTS["reaction_left_40x8_n"] = fine["reaction"]
for index, entry in enumerate(MODAL, start=1):
    FACTS[f"modal_omega_{index}"] = entry["omega"]
FACTS["modal_frequency_hz_1"] = MODAL[0]["hz"]

for key, value in FACTS.items():
    print(f"| `{key}` | {value:.10g} |")
```

## 8. Pitfalls this loop is built to catch

1. **Corner loads must be half-weighted.** A uniform `LOAD / ny` on all `ny + 1`
   edge nodes applies `(ny + 1) / ny` times the intended load. The force-balance
   assertion is what makes that visible.
2. **`nodes[elements].mean(1)` is the element centre only for a rectangle.** For a
   distorted Q4 the isoparametric centre is `0.25 * Σ x_i` at `(ξ, η) = (0, 0)`,
   which coincides with the node mean only for parallelograms. The stress band here
   is compared at node means, so the mesh must stay rectangular.
3. **Plane stress vs plane strain changes the stress error, not the displacement
   much.** The reference used is beam theory with `I = t H³ / 12`, i.e. plane
   stress. Running the same meshes with `plane="strain"` on a 12 mm plate compares
   against the wrong reference; the gate would rightly reject it.
4. **The dense solver in `solve_continuum` scales as N³.** At 40×8 that is 738
   equations and milliseconds; a 100k-DOF model needs the sparse path
   (`solve_sparse_static`) or domain decomposition, not a bigger dense matrix.
5. **Consistent-mass frequencies converge from above.** A single-element beam
   model is far off; the modal assertion is stated at 40 elements, where the
   amplitude error is ~1e-6. Stating a tolerance without the mesh it belongs to is
   how modal evidence gets overstated.
6. **Pin the BLAS pool on a shared host.** This case is milliseconds of work,
   but on a 128-core shared node the default torch thread pool spends seconds of
   scheduling per solve and the wall clock stops describing the solver. The test
   that executes this tutorial pins `torch.set_num_threads(1)` for the same
   reason `tests/test_element_rank_audit.py` does.
7. **Never accept a stored `"passed": True`.** Both gates here recompute from raw
   fields, which is why they survive geometry, load or reference edits.