# TensorFEM

TensorFEM is a differentiable structural finite-element toolkit built on
PyTorch. It targets verification-first engineering analysis, inverse problems,
design optimization, and AI-assisted digital twins.

## Current capability (v0.2)

- 2D linear-elastic truss elements
- Dense differentiable assembly (CPU/CUDA)
- Displacement boundary conditions and nodal forces
- Linear static solution
- Displacement, reaction, strain, stress, axial force and strain energy
- Analytical single-bar benchmark
- Three-bar structural example
- Automatic differentiation through material and section parameters
- 2D Euler--Bernoulli beam/frame elements and distributed loads
- CST and fully integrated Q4 plane stress/plane strain elements
- Consistent-mass Euler--Bernoulli beam modal analysis
- Linear eigenvalue buckling for beam-columns
- Fail-closed benchmark evidence with strict relative error below 3%

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
tensorfem benchmark
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
