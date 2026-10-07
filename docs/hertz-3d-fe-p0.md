# True 3-D Hertz FE P0

`hertz_3d_fe.solve_hertz_cap_block` is the first real three-dimensional
two-deformable-solid closure. A shallow spherical cap and an elastic block are
independently meshed with TET4 elements. Their remote faces are fixed at an
imposed mutual approach; no analytical or Winkler interface load is applied.
Neo-Hookean volume residuals and faceted Mortar contact share one Newton solve.

The initial 2/3 nonmatching mesh gives contact force `0.5075946445` against the
classical Hertz value `0.7326007326` (30.71% error), while converging to residual
`8.64e-13`, minimum `det(F)=0.999385`, and interface force imbalance
`1.11e-16`. This proves the real solver path but is **not** qualification.

The composite general-contact gate remains blocked. Contact-zone local mesh
refinement and a larger remote domain are required before the three-level
Hertz error can be tested against the 3% threshold and persisted as
`tensorfem.hertz-3d-qualification/1.0` evidence.
