# Shell4 tangent assembly performance qualification

## Function-level evidence

An 8x8 residual-stressed imperfect panel was profiled at its initial state.
Before this optimization, one reduced sparse tangent assembly took 5.176 s in
the profiler. `element_response` consumed 2.569 s in layered material
integration, including 1,280 `plane_stress_j2_update` calls and 8,960 nested
material-response calls. Corotational mapping and its geometric Hessian used
about 2.48 s. Thus sparse index assembly was not the bottleneck; repeated
finite-difference material tangents and AD kinematics were.

For a material point safely inside the elastic domain, the condensed J2 update
is exactly linear elastic. `plane_stress_j2_update` now returns the analytical
plane-stress matrix after performing the same stress/state update once. A
conservative yield-surface margin retains the numerical consistent tangent
near active-set changes and throughout plastic loading. This changes neither
the force calculation nor the trial/commit state contract.

After the change the identical profile took 3.525 s. Layered
`element_response` fell to 1.092 s and plane-stress updates to 0.713 s; the
remaining dominant cost is the corotational mapping/geometric Hessian AD.

## Correctness gates

- The analytical elastic tangent action matches a centered force difference
  to `2e-10` relative tolerance and preserves the virgin state exactly.
- Existing plastic consistent-tangent, Shell4 force/tangent-action,
  objectivity, sparse assembly, and rejected-state transaction tests pass.
- The real 8x8 first arc point remains `17,253.182558461 N` with displacement
  norm `0.008720980686687 m`. Relative differences from the qualified prior
  result are about `1.2e-12` and `2e-13`, far below 1%.
- The 80-point fully yielded plastic path remains at load factor
  `7.719914400896421`, displacement norm `0.123817012566696`, recoverable
  energy `0.430871136111123 J`, alpha sum `1.015320107085862`, and plastic
  strain norm `0.359180613734073`. Differences from the prior numerical
  tangent evidence are at floating-point roundoff. Alpha remains monotone and
  every integration point yields. Byte hashes differ and are not claimed as
  identical.

## Scaling

| case | prior assembly | analytical-elastic assembly | improvement |
|---|---:|---:|---:|
| 8x8 | 4.676 s | 3.216 s | 31.2% |
| 20x20 | 26.903 s | 17.757 s | 34.0% |
| 8x8 complete first arc point, SuperLU | 52.84 s | 38.11 s | 27.9% |

Sparse matrix topology and values are unchanged within tangent tolerances, so
matrix and factor storage do not increase. The improvement applies primarily
to elastic points; heavily plastic paths retain the more expensive numerical
algorithmic tangent by design. Further safe acceleration must target batched
corotational Jacobian/Hessian evaluation, which is now the largest measured
cost. A step-frozen tangent was separately rejected because it changed a real
panel path by more than 1%.

## Corotational AD optimization experiments

The remaining Jacobian and contracted Hessian were evaluated with both legacy
`torch.autograd.functional` vectorization and `torch.func` transforms. On a
representative imperfect facet, `jacrev` reduced the isolated mapping time
from 8.60 ms to 7.89 ms, but `torch.func.hessian` increased the geometric term
from 19.32 ms to 53.22 ms. Retaining the legacy Hessian and changing only the
Jacobian produced no repeatable assembly benefit: the 8x8 run regressed from
3.216 s to 4.092 s due partly to transform startup, while 20x20 changed only
from 17.757 s to 17.694 s (0.35%, within run noise). The experimental change
was reverted.

Reference-frame, projected-shape and Gauss-operator caching was also assessed.
Those quantities are immutable, but after the analytical material improvement
they account for much less time than the AD Jacobian/Hessian; a process-global
tensor-identity cache would introduce stale-model and lifecycle risks for a
small upper-bound gain. It was not implemented. `vmap` batching is not safely
applicable to the current SVD frame code because it contains data-dependent
normal orientation, degeneracy and finite-difference fallback branches.

The accepted optimization therefore remains the analytical elastic material
tangent. No corotational AD rewrite is merged: the tested alternatives either
slowed the Hessian, showed no reproducible whole-assembly gain, or could not
preserve the existing fail-closed geometry branches. Future work requires a
dedicated analytical polar-rotation derivative, including repeated-singular-
value handling, rather than replacing one AD frontend with another.
