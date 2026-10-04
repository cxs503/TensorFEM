# TensorFEM

TensorFEM is a verification-first finite-element toolkit built on PyTorch for
engineering analysis, inverse problems, design optimization, and AI-assisted
digital twins. Stable capabilities are admitted only with executable
analytical or recognized benchmark evidence.

## Capability maturity (v0.37)

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
| axisymmetric deformable Hertz model | qualified benchmark | deformable Q4 half-space pressure, load, radius, peak pressure and overlap converge below 3% at 48x48; rigid indenter and axisymmetric scope only |
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
| arbitrary-mesh project schema | stable initial subset | safe truss/TET4/HEX8 structural and LINE2/Q4 thermal JSON projects with deterministic jobs |
| multistep execution | stable initial subset | linear, thermal, modal and sequential thermoelastic steps with resume |
| ResultDB v2 | stable core | multistep/multiframe/IP fields, histories, lazy queries and atomic HDF5 publication |
| engineering case packages | stable examples | pressurized component, thin-wall panel and stopped connector with checkpoints, ResultDB, VTK and reports |
| public API and release gates | stable infrastructure | versioned 310-symbol manifest, drift audit and offline wheel/sdist installation smoke test |
| ship structural screening | stable benchmarks | hull-girder longitudinal bending and equivalent-orthotropic stiffened-panel analytical checks |
| box-barge hydrostatics | stable initial subset | displacement, KB/BM/KM/GM and small-angle restoring moment for intact rectangular hulls |
| marine environmental loads | stable initial subset | Airy-wave/Morison pile actions and audited TensorLBM-to-TensorFEM nodal force histories |
| marine qualification matrix | stable verification suite | 35 registered nonzero-reference quantities plus mesh/integration convergence and balance gates |
| TensorFEM-only marine structural gate | stable verification view | 15 scalar references and four DOI-traceable shell cases; TensorLBM/CFD excluded |
| marine buckling, section ultimate and fatigue primitives | qualified prototypes | Navier plate buckling, rectangular section yielding, hot-spot/S-N/Miner hand-oracle gates |
| marine postbuckling, progressive section and spectrum fatigue | qualified prototypes | reduced-order Koiter path, component-section equilibrium, rainflow and Paris-law gates |
| Shell4 buckling and structural integrity post-processing | qualified core/prototypes | real initial-stress Shell4 eigenproblem, ResultDB hotspot/SCL and LEFM gates |
| numerical crack-tip J integral | qualified Mode-I prototype | direct discrete contour integration, plane stress/strain K-J oracle, path-shape independence and quadrature convergence |
| initial imperfection and residual-stress transfer | qualified reduced-order integration | ResultDB v1/v2 ID/coordinate mapping, balance checks, round trip and field-sensitive elastic/plastic strip response |
| Shell4 arc-length residual/tangent integration | qualified integration prototype | real assembled corotational residual and autograd Hessian with equilibrium and step-size gates; no postbuckling claim |
| multi-facet Shell4 transactional state | qualified elastic prototype | measured imperfection in the stress-free reference, accepted generalized section state, exact tangent and rejected-step rollback |
| global finite-sliding contact equilibrium | qualified 2-D prototype | deformable node/polyline current-configuration search, Coulomb history, balanced global residual, algorithmic tangent and rollback |
| unified analysis workflow | stable initial subset | versioned ModelDB/AnalysisPlan, registered structural/thermal steps, deterministic jobs, atomic checkpoint/resume and ResultDB v2 |
| industrial P0 qualification | stable verification view | hashed executable report with explicit unqualified shell-plasticity, general 3-D mortar/self-contact and distributed-solver boundaries |
| layered Shell4 plasticity | qualified small-strain prototype | 2x2 in-plane and through-thickness points, condensed plane-stress J2, residual stress, unloading, tangent and transactional history gates |
| global 3-D surface contact equilibrium | qualified node-TRI3 prototype | current-facet search, barycentric master reactions, Coulomb history, objective motion, force/moment balance and rollback |
| external P0 benchmark contracts | stable evidence governance | qualified Bathe-Bolourchi elastic large rotation; shell snap-through and deformable Hertz FE remain explicitly blocked |
| industrial P0 phase-2 qualification | stable verification view | hashed layered-shell/contact report with finite-rotation plastic postbuckling and general mortar/self-contact kept unqualified |
| finite-rotation layered Shell4 arc integration | qualified integration primitive | corotational small-strain J2 layers, residual stress, imperfections, algorithmic tangent and transactional accept/reject history; not a finite-membrane-strain claim |
| marine panel ultimate-strength FE contract | partial execution evidence | 4x4 peak/post-peak and energy gates are established; complete 8x8/12x12 peak paths and cross-mesh convergence remain to execute |
| industrial P0 phase-3 qualification | stable verification view | hashed executable report qualifies the Shell4 integration primitive and axisymmetric Hertz scope while keeping panel ultimate and general 3-D double-deformable contact blocked |
| shallow-warp layered Shell4 geometry | qualified integration primitive | ordered best-fit midsurface, stress-free warped references, rigid-motion objectivity and explicit deep-fold rejection |
| marine panel qualification execution | stable execution foundation | dimensionless force/displacement scaling, resumable hashed cells and fail-closed timeouts; accepted pre-peak paths exist, but the declared peak/post-peak matrix remains incomplete |
| two-sided 3-D facet contact assembly | stable kernel | reference-area quadrature, deformable slave/master residual, exact active-set tangent, objectivity and force/moment balance; full 3-D Hertz remains blocked |
| deformable-to-deformable 3-D Hertz | blocked qualification | strict three-mesh/domain/pressure/equilibrium contract exists; curved solid coupling and complete convergence evidence remain |
| finite-rotation layered Shell4 consistent tangent | qualified integration primitive | one material integration per element, corotational material/geometric chain rule and directional errors below 4e-9 |
| transactional residual-stress equilibration | qualified panel foundation | initial free residual is relaxed from about 139 kN before continuation, with rollback and deterministic displacement/material-state hashes |
| dimensionally scaled shell arc continuation | qualified panel foundation | thickness-integrated rotation metric, scaled load coordinate, relative force/constraint gates, drilling-gauge exclusion and attempt diagnostics |
| marine panel first-point performance gate | qualified pre-peak foundation | real 2x2/4x4 first points in 3.26/12.89 s with balance near 1e-13; three-point paths pass, peak/post-peak remains blocked |
| matrix-free shell arc continuation | stable experimental solver | transactional finite-difference JVP, GMRES, frozen-tangent/Jacobi/block options and fail-fast initial-equilibrium diagnostics |
| sparse Shell4 tangent assembly | verified storage foundation | connectivity-graph COO tangent matches dense force and directional action; 4x4 storage gate passes, while production sparse throughput remains unqualified |
| normalized augmented arc predictor | verified continuation correction | displacement and scaled-load coordinates share one normalized metric with non-unit-load-scale and branch-direction regression gates |
| resumable panel long-path continuation | stable execution foundation | branch-direction restart, atomic hashed mechanical/energy checkpoints and dynamically extendable targets/checkpoint cadence |
| panel path energy and failure evidence | qualified observer | projected-facet stored energy is force-conjugate; coarse/fine 4x4 residuals pass the 0.025 J gate and show 24.98x second-order reduction, with versioned ledgers preventing stale resume |
| marine panel long-path localization | 4x4 peak/post-peak and energy qualified | fine ds=0.02 peaks at 508.999 kN on point 334 and completes 400 points with 66 post-peak points; the coarse/fine peak difference is 0.0615%, zero yield confirms an elastic geometric snap-back, while cross-mesh convergence remains open |
| large panel path scalability | qualified pre-peak foundation | dimensionally consistent 8x8/12x12 paths reach 340/20 points and 580.889/309.197 kN with equilibrium/energy gates passing; neither establishes a refined-mesh peak |
| refined panel peak-window scheduling | verified execution foundation | energy/metric-versioned immutable generations, stale-manifest replay protection and coarse/approach/peak scheduling support 8x8 and 12x12; refined-mesh descending branches remain open |
| panel 4/8/12 mesh-convergence gate | verified fail-closed evaluator | requires explicit peak and post-peak evidence on every mesh and rejects missing data without extrapolation; current status is blocked by 8x8/12x12 paths |
| optional sparse direct/ILU backend | verified internal adapter | fail-closed lazy SciPy integration matches dense 4x4/8x8 correctors, but is slower at current sizes and is not a production-performance claim |
| advisory nonlinear control policy | verified execution foundation | deterministic Newton/rejection/curvature/equilibrium/energy classifications, hashed restart chain and explicit recommendations; automatic solver switching remains disabled |
| marine stiffening and degradation screening | qualified analytical/numerical prototypes | smeared stiffener buckling, uniform-corrosion thickness sensitivity and self-balanced residual-stress first-yield errors remain below 0.08%; discrete tripping, pitting and collapse remain unqualified |
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
