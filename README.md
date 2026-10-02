# TensorFEM

TensorFEM is a verification-first finite-element toolkit built on PyTorch for
engineering analysis, inverse problems, design optimization, and AI-assisted
digital twins. Stable capabilities are admitted only with executable
analytical or recognized benchmark evidence.

## Capability maturity (v0.18)

| Capability | Maturity | Verification |
|---|---|---|
| 2-D truss, frame, CST/Q4 and Timoshenko beam | stable | analytical solutions, patch tests and Cook membrane |
| modal, buckling, implicit and explicit dynamics | stable | beam/column/SDOF references and stability gates |
| TET4, HEX8, B-bar HEX8, TET10 and HEX20 solids | stable | affine patches, locking, distortion, modal and cantilever cases |
| sparse COO/CSR, CG/GMRES, Jacobi, multi-RHS, Dirichlet and MPC | stable core | dense equivalence, ill-conditioned/indefinite systems and independent 100k-DOF performance test |
| additive-Schwarz domain decomposition | stable prototype | partition-independent coupled solves and 100k/1M topology benchmarks |
| non-diagonal structural-grid sparse benchmark | stable verification utility | exact discrete modes, dense cross-check and one-million-DOF run |
| model database and Abaqus INP/Gmsh/VTK/HDF5 adapters | stable core | parser round trips and INP-to-solver-to-VTK workflow |
| Gmsh v2/v4 mesh diagnostics and conversion | stable core | Jacobian, quality, topology and metadata fail-closed tests |
| adaptive nonlinear load steps, restart and plastic-contact coupling | stable core | finite-strain, elastoplastic/contact closed forms and rollback tests |
| spatial bar geometric-plastic-contact coupling | stable benchmark | independent force-law reference, rollback and restart checks |
| TET4 J2 elastoplasticity | stable kernel | global uniaxial loading/unloading and consistent tangent |
| multi-element small-strain TET4 J2 paths | stable core | 12-element loading/unloading, restart and rollback verification |
| total-Lagrangian TET4 hyperelasticity | stable core | 50% extension, finite-rotation objectivity, exact tangent and restart |
| steady/transient thermal and sequential thermoelasticity | stable | conduction, convection, decay and thermal expansion references |
| plate, flat shell, finite-sliding 2-D/3-D contact and cohesive formulations | stable kernels | patch, energy, complementarity, friction and fracture-energy checks |
| low-order surface contact | stable kernel | TRI3/QUAD4 tributary integration, resultant balance and topology gates |
| cylindrical shell research formulation | experimental | global nonlinear Newton/line search/restart works; wider nonlinear shell benchmarks remain |
| axisymmetric deformable Hertz model | stable benchmark | coupled Q4/contact pressure solution reaches all four 3% gates at 40x40 |
| multiplicative finite-strain J2 plasticity | stable prototype | objectivity, isochoric flow, dissipation, coaxial reference and restart gates |
| low-order frictionless Mortar contact | stable prototype | force/moment patch, master-slave symmetry and nonmatching convergence |
| Pinched Cylinder shell benchmark | qualified evidence | MacNeal-Harder 12x12 error 0.985%; shell family remains experimental |
| general nonproportional finite plasticity/self-contact | experimental | broader path benchmarks and full contact solve remain |
| adaptive nonproportional finite J2 integration | stable prototype | independent 400-substep reference, path dependence and restart gates |
| frictional Mortar and self-contact force update | stable prototype | Coulomb return, rollback, dissipation, force and moment balance |
| Hemispherical Shell with 18-degree hole | qualified evidence | MacNeal-Harder 12x12 error 0.690%; coarse locking evidence retained |
| global adaptive cyclic finite plasticity | stable prototype | multi-TET Newton, 500-substep reference, rollback and restart |
| vertex-triangle continuous collision detection | stable foundation | analytical TOI, no-tunnelling, impulse conservation and pair history |
| general nonlinear doubly-curved shell | experimental solver | objective energy/tangent, multi-element Newton and mesh convergence gates |
| AD finite-plasticity tangent | stable prototype | Richardson agreement 2.06e-10 and measured 6.75x speedup |
| edge-edge CCD and Coulomb impact | stable foundation | analytical TOI, deterministic broad phase, momentum and energy gates |
| large-rotation pure-bending shell | qualified evidence | 4x1 tip error 0.645% and moment error 2.56e-7 |
| local implicit finite-plasticity tangent | stable qualified kernel | dual-oracle error below 3%, near-repeated spectrum gate and 16.6x reference speedup |
| multi-contact complementarity impulses | stable foundation | PGS nonpenetration, complementarity, friction-cone and energy gates |
| Crisfield arc-length continuation | stable algorithm | analytical von Mises arch limit load error 0.0255% and restart verification |
| hybrid implicit/AD plastic tangent | stable strategy | implicit below 1% evidence gate; automatic AD fallback outside qualification domain |
| rigid contact graph with rotational inertia | stable foundation | conservation gates, deterministic islands and 1000-constraint benchmark |
| public ring-load snap-through shell | blocked qualification | missing tabulated oracle/material data and follower-load tangent; audit fails closed |
| global hybrid-tangent cyclic plasticity | stable prototype | per-point strategy, rollback/restart and about 24.5x path reduction on integration test |
| colored vectorized contact graph | stable foundation | serial agreement, conservation and 1024-constraint deterministic benchmark |
| follower shell pressure/load tangent | stable kernel | analytical resultant, rotational covariance and tangent error 3.02e-10 |
| taped adaptive-substep plastic tangent | stable prototype | 42 accepted updates, Richardson error 2.38e-10 and explicit retape gate |
| colored frictional contact graph | stable foundation | serial agreement, friction-cone/energy gates and 1024-constraint benchmark |
| contact-island worker scheduling | stable CPU foundation | deterministic 1/2/4-worker results, serialization and failure propagation |
| multi-GPU contact execution | planned/unqualified | current host reports no CUDA devices; no multi-GPU claim |
| Model-Step-Job-Result workflow | stable case workflow | deterministic IDs, state, replay/restart, checksums and failure records |
| hemisphere post-processing | stable case workflow | JSON/VTK/Markdown, probes, paths, extrema, reactions and balance gates |
| versioned benchmark evidence | stable infrastructure | schema, DOI, hashes, tamper detection, quick/full tiers and drift comparison |
| general engineering project workflow | stable initial schema | safe static dispatch for three shell cases, replay and diagnostics |
| ResultDB | stable core | field checksums, chunked HDF5 round trip, lazy subset queries and no-h5py fallback |
| classical shell evidence suite | stable verification suite | four sourced cases, 19 meshes, quick/full tiers and suite drift hash |
| general self-contact, mortar contact and crack growth | experimental/planned | not advertised as stable |
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
