# Finite-rotation Shell4 arc backtracking qualification

The finite-rotation layered Shell4 corrector now accepts the explicit opt-in
`line_search="backtracking"`.  Its default remains fixed full Newton.  The
strategy never changes the predictor, tangent, arc metric, material model, or
Newton method.

Every candidate displacement is evaluated against the last committed
`LayeredShell4State`.  Candidate material states remain unreachable; only the
response evaluated at an accepted equilibrium is committed.  If no candidate
reduces the normalized equilibrium-plus-constraint merit, the attempted step
fails closed and follows the existing radius-reduction path.

Panel job identity includes the strategy only when it is enabled, preserving
all legacy/default checkpoint keys.  New backtracking checkpoints also record
the strategy in the immutable binary generation and reject a mismatched
resume.

## Controlled difficult Shell4

The regression is a real finite-rotation, three-layer J2 Shell4 cantilever
under transverse end loading.  A radius-0.8 step is intentionally limited to
15 corrector iterations, reproducing the large-rotation difficult behavior
without requiring the multi-hour 8x8 panel prefix.

| Corrector | Accepted | Rejected/failed | Result |
|---|---:|---:|---|
| Fixed full Newton | 0 | 1 | maximum iterations, no commit |
| Backtracking | 1 | 0 | accepted in 13 iterations |

Backtracking uses factors below one and agrees with a tighter-tolerance
backtracking reference to below 1% in load, displacement path, and recoverable
energy.  The caller-owned virgin material checkpoint remains byte-equivalent
after both the rejected fixed step and all line-search trials.

An easier two-step path verifies exact default compatibility and verifies that
one accepted step followed by restart matches an uninterrupted backtracking
path.  Automatic activation remains prohibited: the nonlinear controller can
recommend line search, but only the caller's explicit opt-in enables it.

