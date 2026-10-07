# Safe hybrid finite-plastic tangent

Each point first computes adaptive two-half-step evidence and a local implicit
return. The implicit tangent is selected only when no recursive substeps are
needed and combined stress/state disagreement is below 1%; otherwise adaptive
AD is mandatory. Per-element strategy, error, path count and fallback reason
are returned. Both branches remain below 3% versus Richardson.

The gate is deliberately conservative and never extrapolates the local tangent
to large or strongly non-coaxial increments. It preserves committed-state
rollback, restart compatibility, plastic volume and dissipation checks. A
closed-form chain across an arbitrary adaptive substep tree remains future work.
