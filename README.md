# TensorFEM

TensorFEM is a differentiable structural finite-element toolkit built on
PyTorch. It targets verification-first engineering analysis, inverse problems,
design optimization, and AI-assisted digital twins.

## Current capability (v0.1)

- 2D linear-elastic truss elements
- Dense differentiable assembly (CPU/CUDA)
- Displacement boundary conditions and nodal forces
- Linear static solution
- Displacement, reaction, strain, stress, axial force and strain energy
- Analytical single-bar benchmark
- Three-bar structural example
- Automatic differentiation through material and section parameters

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

1. Truss patch tests and multi-load cases
2. Euler-Bernoulli and Timoshenko beam/frame elements
3. CST/Q4 plane stress and Cook's membrane
4. Modal analysis and Euler buckling
5. J2 plasticity and geometric nonlinearity
6. Sparse solvers, batched models and distributed execution
7. Gmsh/meshio input and VTK/PyVista output

Numerical features are admitted only with analytical or recognized benchmark
evidence and mesh-convergence tests.
