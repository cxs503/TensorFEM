# Frictional Mortar consistent-tangent foundation

`assemble_frictional_mortar` supplies a two-sided residual and algorithmic
tangent for the existing integration-point Coulomb history. Candidate search,
active set and stick/slip return branch are frozen during one Newton
linearisation; projection coordinates, normals, penalty pressure and the
selected return map remain in the automatic-differentiation graph.

The nonmatching planar patch test checks the tangent against an independent
centred directional difference. Common rigid translation creates no slip;
interface force and moment balance and caller-owned history immutability are
also gated.

This is a consistent local/global-assembly foundation, not yet a qualified
curved frictional complete-Newton path. `general_surface_to_surface` remains
blocked until the curved double-deformable path uses this history and tangent
through accepted/rejected load increments and passes mesh/interchange gates.
