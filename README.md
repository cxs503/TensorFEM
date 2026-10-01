# TensorFEM

TensorFEM is a differentiable structural finite-element toolkit built on
PyTorch. It targets verification-first engineering analysis, inverse problems,
design optimization, and AI-assisted digital twins.

## Capability maturity (v0.3)

| Capability | Status | Verification |
|---|---|---|
| 2-D truss and frame | stable | analytical bar and cantilever |
| CST/Q4 continuum | stable | patch tests and Cook membrane |
| Timoshenko beam | stable | deep/slender cantilevers |
| modal, Newmark dynamics, linear buckling | stable | analytical beam, SDOF and Euler columns |
| TET4/HEX8 solids | stable | patch, distortion and bar-mode cases |
| Mindlin plate | stable | sinusoidal-load thin/thick plate |
| geometric nonlinearity and plasticity | stable kernels | shallow arch and return mapping |
| frictionless contact | stable kernel | complementarity and penalty spring |
| flat shell and cohesive law | stable kernels | energy identities and fracture energy |
| general fracture, advanced contact, sparse/distributed solvers | planned | not advertised |

Stable scalar benchmarks are executed through a fail-closed registry. Every
entry records its source, computed and reference values, relative error and
tolerance; passing requires `error < tolerance <= 3%`. Zero-reference patch
tests remain explicit invariants and are not misreported as relative errors.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
tensorfem verify --output benchmark-evidence.json
pytest
```

```python
from tensorfem.benchmarks import single_bar
from tensorfem import solve_linear_static

model, analytical_displacement = single_bar()
result = solve_linear_static(model)
print(result.displacement, result.axial_stress)
```

## Verification roadmap

1. Timoshenko beam/frame elements
2. Cook's membrane and distorted-mesh convergence
3. J2 plasticity and geometric nonlinearity
4. Sparse solvers, batched models and distributed execution
5. Gmsh/meshio input and VTK/PyVista output

Numerical features are admitted only with analytical or recognized benchmark
evidence and mesh-convergence tests.

The current automated suite covers truss, frame, CST/Q4 continuum, vibration,
and Euler buckling. Run `pytest`; every formal accuracy case must satisfy
`relative_error < 3%`. Validation contracts are documented under `docs/`.
