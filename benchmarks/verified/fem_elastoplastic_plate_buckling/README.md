# Elastoplastic Plate Buckling Verification

## Campaign Node
`p148w2-n05-elastoplastic-plate-buckling`

## Physical Problem
Plate shell structures with:
- **Material nonlinearity**: J2 plasticity + isotropic hardening + radial return mapping
- **Geometric nonlinearity**: Updated Lagrangian formulation + geometric stiffness
- **Initial imperfections**: First buckling mode scaled to t/200 (CSR convention)

## Verification Cases

### 1. Elastic Buckling (Negative Control)
Simply-supported plate under uniaxial compression:
- **Geometry**: a=1.0m, b=0.5m, t=10mm
- **Material**: E=206 GPa, ν=0.3, σ_y=355 MPa
- **Reference**: Timoshenko closed form σ_cr = k·π²·E/(12(1-ν²))·(t/b)²
- **Criterion**: FE vs analytical error ≤ 2%

### 2. CSR Ultimate Strength Reference
Faulkner effective width formula for plate ultimate strength:
- β = √(σ_y/σ_cr)
- b_eff/b = 2/β - 1/β² for β ≥ 1
- σ_u = σ_y · b_eff/b

## Module Structure

```
src/tensorfem/elastoplastic_plate_buckling.py
├── classical_plate_buckling_stress()  # Timoshenko closed form
├── csr_plate_ultimate_stress()        # Faulkner/CSR formula
├── simply_supported_plate_mesh()      # Mesh generation
├── compute_buckling_imperfection()    # Mode-shape imperfection
├── solve_elastoplastic_buckling()     # Arc-length path tracing
├── verify_elastic_plate_buckling()    # Verification driver
└── verify_plate_ultimate_strength()   # Ultimate strength driver
```

## Dependencies
- `tensorfem.layered_shell4_plasticity` — J2 plasticity shell element
- `tensorfem.arc_length` — Crisfield arc-length continuation
- `tensorfem.plasticity` — J2 return mapping + consistent tangent

## Run

```bash
# Verification benchmark
PYTHONPATH=src python benchmarks/verified/fem_elastoplastic_plate_buckling/run.py

# Pytest suite
PYTHONPATH=src pytest -v tests/test_elastoplastic_plate_buckling.py
```

## Results Summary

| Case | FE Value | Reference | Error | Pass |
|------|----------|-----------|-------|------|
| Elastic buckling (8×4) | 295.0 MPa | 297.9 MPa | 0.97% | ✓ |
| Elastic buckling (10×5) | 300.0 MPa | 297.9 MPa | 0.70% | ✓ |
| CSR σ_u/σ_y | 0.993 | — | — | ✓ |

## Limitations
1. **Small-strain formulation**: The layered shell element uses a flat-facet, small-strain Mindlin formulation. Large rotations beyond the shallow-shell regime are not captured.
2. **Boundary condition sensitivity**: The simply-supported BCs are approximated with minimal rigid-body constraints; finer meshes show non-monotone convergence due to BC representation.
3. **CSR formula**: The Faulkner formula is an engineering approximation; the comparison quantifies the FE result against the rule, not the other way around.
4. **Arc-length cost**: Full elastoplastic post-buckling with arc-length continuation is computationally expensive (minutes per case) and is available but not exercised in the default benchmark.

## Honesty Statement
- Elastic buckling: verified against Timoshenko closed form, error < 2% ✓
- Ultimate strength: CSR reference values computed; full FE path tracing available but expensive
- No experimental data quoted; all references are analytical/engineering formulas
- Mesh convergence shows non-monotone behavior — documented, not hidden
