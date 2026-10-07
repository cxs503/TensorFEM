# Marine stiffening and degradation qualification

This qualification closes three narrowly defined pre-collapse sensitivities
with executable evidence and a 3% acceptance threshold.

| Case | Numerical model | Independent reference |
|---|---|---|
| Longitudinal stiffener | 32x32 central-difference plate eigenproblem with smeared longitudinal rigidity | continuous Navier orthotropic plate eigenvalue |
| Uniform corrosion | the same discrete eigenproblem at 0, 10, 20 and 30% thickness loss | classical `sigma_cr proportional to t^2` sensitivity and Navier eigenvalue |
| Residual stress | midpoint-fibre first-local-yield search for `sigma_r cos(2 pi x)` | continuum maximum at the plate edge, with 16/32/64/128-fibre convergence |

The residual field has zero membrane resultant. Its qualification result is
the reduction in *first local yield shortening*, not ultimate capacity.

Run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_marine_degradation_qualification.py
PYTHONPATH=src python examples/marine/stiffening_degradation.py
```

## Explicit boundary

This suite does not qualify discrete stiffener tripping, plate/stiffener weld
details, nonuniform pitting, welding heat transfer, shell collapse, fatigue,
or class-rule scantling. Those require real geometry, shell-mesh convergence
and experimental or classification-society evidence.
