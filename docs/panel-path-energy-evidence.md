# Panel path energy and failure evidence

`panel_path_evidence` converts **accepted** finite-rotation layered-Shell4
points into auditable energy and failure records. It is an observer and cannot
commit a material state, so rejected predictor/corrector iterations never enter
the record.

For every accepted point it reports:

- trapezoidal external work, using the load factor and reference-load vector;
- recoverable elastic plus isotropic-hardening energy;
- cumulative plastic dissipation `sigma_y * Delta alpha`, integrated over all
  four in-plane points and all thickness layers;
- internal energy (recoverable-energy change plus plastic dissipation), absolute
  energy residual, and scale-independent relative residual;
- volume-weighted yielded fraction, transverse displacement and conjugate axial
  shortening.

The existing J2 state is sufficient: plastic strain reconstructs elastic stored
energy and accumulated equivalent plastic strain reconstructs both hardening
energy and associative J2 dissipation. No extra mutable history is needed.

A peak is the maximum accepted load. A post-peak branch is only confirmed when
a later accepted point has greater shortening and a configurable material load
drop (2% by default). Failure mode is then classified as elastic buckling,
material collapse, or interactive buckling/yielding from transverse deformation
and yielded volume. Without that evidence the label remains `prebuckling` or
`yielding_without_confirmed_peak`; a negative tangent alone is not a collapse
claim.

The energy residual is a path diagnostic, not an artificial correction. It
therefore exposes continuation-step integration error, inconsistent load work,
or an invalid accepted-state history rather than forcing closure by definition.
