# Automatic step-only nonlinear control qualification

TensorFEM's nonlinear controller remains advisory by default.  The panel
chunk runner now has an explicit `automatic_step_control=True` mode which may
apply only the controller's bounded step-size factor after an accepted,
committed chunk.  It never changes the Newton method, line search, arc metric,
or equilibrium and energy tolerances.

Each automatic decision records the current, minimum, maximum, and applied
step sizes.  These values, the predecessor decision hash, and the operating
mode are covered by the decision SHA-256.  The mode also participates in the
panel job key, and the latest decision hash is stored in and verified against
the immutable binary checkpoint.  An advisory checkpoint therefore cannot be
silently resumed as an automatic run.

## v0.38 evidence

The von Mises shallow arch retains the fixed-control equilibrium path across
an interrupted/restarted run.  Its peak remains within the existing 3%
analytic benchmark tolerance.  With the qualified upper step bound active,
the automatic policy produces the identical path, so the peak and all sampled
path points differ by 0% (below the 1% qualification threshold).

A real 2x2 imperfect panel was also run for six one-point chunks with fixed
and automatic control (`normalized_arc_step=0.02`).  Both runs accepted six
of six attempts with no rejection, reached 145793.74750784395 N, produced
identical point histories, and passed every energy-balance gate.  Their job
keys differ, proving the control mode is part of checkpoint identity.  The
evidence is under `.qualification/v038-auto-controller/`.

The automatic path was then resumed from six to eight points and compared
with a fresh uninterrupted eight-point automatic run.  Point histories,
controller decision chains, and the terminal force (184503.55148166636 N)
were exactly equal.  This covers the controller state as well as the
mechanical state across restart.

The policy did not reduce iterations or rejected attempts on either benign
short path.  It therefore remains **experimental** and explicit opt-in.  This
qualification establishes path preservation, deterministic restart, energy
closure, and fail-closed identity; it does not claim a performance benefit.
