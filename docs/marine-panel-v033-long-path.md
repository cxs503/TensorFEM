# Marine panel v0.33 long-path evidence

The v0.33 work removes the execution barriers around long nonlinear panel
paths without promoting an unobserved ultimate load.  Arc continuation now
persists the previous scaled increment together with displacement, load factor
and complete layered J2 state.  Checkpoints are written atomically and their
SHA-256 is checked before a trusted local restart is loaded.

The 2x2 localization mesh completed 220 accepted points at normalized arc
step 0.1.  Its terminal force was 2.295110 MN, all material points had yielded,
and accepted equilibrium errors remained around 1e-13.  The load was still
increasing at the final point, with positive hardening after the near-plateau,
so this path does not provide a peak or descending branch.

Short-path scalability evidence was also obtained for the qualification
meshes: the 8x8 first point completed in 48.36 s, the 12x12 first point in
106.96 s, and the 8x8 three-point path completed in 80.03 s.  These are
pre-peak execution results, not ultimate-strength convergence evidence.

Energy reconstruction is based only on accepted states.  It integrates
external work, elastic and isotropic-hardening storage, and
`yield_stress * delta_alpha` plastic dissipation.  Checkpoints originating
before the energy ledger was introduced are marked `energy_prefix_complete =
false`; their partial ledger is never used as whole-path qualification.

The next qualification step is a 4x4 long path followed by 8x8 and 12x12
refinement around a confirmed peak.  A peak requires at least two accepted
points with continued shortening and a material load drop; a plateau or a
single reversal is insufficient.
