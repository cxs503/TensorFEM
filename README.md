# TensorFEM

TensorFEM is a verification-first finite-element toolkit built on PyTorch for
engineering analysis, inverse problems, design optimization, and AI-assisted
digital twins. Stable capabilities are admitted only with executable
analytical or recognized benchmark evidence.

## Capability maturity (v0.4)

| Capability | Maturity | Verification |
|---|---|---|
| 2-D truss, frame, CST/Q4 and Timoshenko beam | stable | analytical solutions, patch tests and Cook membrane |
| modal, buckling, implicit and explicit dynamics | stable | beam/column/SDOF references and stability gates |
| TET4, HEX8, B-bar HEX8, TET10 and HEX20 solids | stable | affine patches, locking, distortion, modal and cantilever cases |
| sparse COO/CSR, CG/GMRES, Jacobi, multi-RHS, Dirichlet and MPC | stable core | dense equivalence, ill-conditioned/indefinite systems and independent 100k-DOF performance test |
| model database and Abaqus INP/Gmsh/VTK/HDF5 adapters | stable core | parser round trips and INP-to-solver-to-VTK workflow |
| adaptive nonlinear load steps, restart and plastic-contact coupling | stable core | finite-strain, elastoplastic/contact closed forms and rollback tests |
| TET4 J2 elastoplasticity | stable kernel | global uniaxial loading/unloading and consistent tangent |
| steady/transient thermal and sequential thermoelasticity | stable | conduction, convection, decay and thermal expansion references |
| plate, flat shell, finite-sliding 2-D contact and cohesive formulations | stable kernels | patch, energy, complementarity, friction and fracture-energy checks |
| cylindrical shell research formulation | experimental | Scordelis-Lo converges below 3%, but finite-angle rigid-body objectivity is not exact |
| 3-D surface/self-contact and general crack growth | experimental/planned | not advertised as stable |
| distributed sparse solvers and production CAD meshing | planned | not implemented |

The formal registry is fail-closed: each scalar entry records its source,
computed and reference values, relative error and tolerance. Passing requires
`error < tolerance <= 3%`. Zero-reference invariants and large performance
tests remain in the specialist test suite rather than being misrepresented as
relative-error benchmarks.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
tensorfem verify --output benchmark-evidence.json
tensorfem capabilities
pytest
```

The default registry is deliberately lightweight. Large sparse throughput,
mesh convergence, negative/fail-closed cases and experimental formulations are
covered independently under `tests/` and documented under `docs/`.

## Minimal static example

```python
from tensorfem.benchmarks import single_bar
from tensorfem import solve_linear_static

model, analytical_displacement = single_bar()
result = solve_linear_static(model)
print(result.displacement, result.axial_stress)
```

## Industrial data path

The solver-neutral `ModelDB` stores nodes, element blocks, sets, materials,
sections, boundary conditions, loads, steps, and output requests. Current
adapters support a qualified subset of Abaqus INP and Gmsh v2 ASCII input,
legacy VTK output, and optional HDF5 results. Unsupported records fail clearly;
format support should not be interpreted as full compatibility with the
corresponding commercial products.

## Scope and verification policy

TensorFEM is under active development and is not a substitute for engineering
judgement or code-required certification. Validation contracts, sources and
known limitations are recorded under `docs/`. The curved-shell implementation
is explicitly experimental until its recognized benchmark errors are below
the project threshold.
