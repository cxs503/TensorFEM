# Automatic step control: real panel path qualification

The v0.39 qualification uses the real 2x2 imperfect marine panel, not a
synthetic residual sequence.  Automatic control is explicit opt-in and can
change only the continuation step.  Full Newton, disabled line search, and
the dimensionally scaled arc metric remain fixed.

The controlled path starts with normalized step 0.1 and uses one accepted
point per transaction.  It executes real scaling at points 17--20 after the
latest-increment energy gate crosses its tolerance.  Whole-path energy is
still reported unchanged; the controller uses an additional incremental gate
because a past quadrature defect cannot be repaired by later small steps.

At the automatic terminal shortening, linear interpolation on the fixed path
gives the following differences:

| Quantity | Relative difference |
|---|---:|
| Axial force | 0.122243% |
| Centre deflection | 0.001567% |
| External work | 0.036467% |
| Internal energy | 0.052674% |

All are below the 1% qualification boundary.  Restarting at point 10 and
continuing to point 20 produces exactly the same point history and controller
decision chain as an uninterrupted run.  The automatic run used 65 total
Newton iterations versus 66 for fixed control, with zero rejected attempts in
both runs.  Thus it reduced Newton work by one iteration (1.52%) but has not
yet demonstrated rejected-step reduction.

The machine evidence is stored under
`.qualification/v039-auto-long2/{automatic,automatic-restart,fixed}`.  The
automatic controller remains experimental: this case proves deterministic
path preservation and a small iteration reduction, not general performance.

