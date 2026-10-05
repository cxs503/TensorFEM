# Double-deformable frictional surface path

This path couples the frictional Mortar residual and frozen-return-branch
algorithmic tangent to independent three-direction foundation stiffness on
both surfaces. A nonmatching 2/3 faceted parabolic-cap/plane model is advanced
through normal closure, sticking shear and sliding shear load steps. Coulomb
history is committed only after Newton convergence; a forced failed increment
must preserve both displacement and integration-point history.

The qualification checks stick-to-slip transition, the sliding `|T|=mu*N`
oracle below 3%, two-sided force balance and the moment of the complete nodal
equilibrium residual. Rigid-motion objectivity and tangent consistency are
covered by the frictional-Mortar assembly qualification.

This is a real curved, nonmatching, double-deformable Newton path, but
the symmetric mode keeps separate forward/reverse integration histories with
explicit weights `0.5 + 0.5 = 1`. Residuals, tangents, stored energy and
dissipation use the same weights, so the physical interface is not counted
twice. Exchanging roles swaps the histories and reverses the applied relative
shear while preserving the converged result.

Until the opt-in mesh report passes, `general_surface_to_surface` remains
blocked. A passing report qualifies only the curved frictional small-strain
subset; finite-strain contact and self-contact remain outside the claim.

## Opt-in mesh qualification

The slow qualification runs complete 2/3, 3/4 and 4/5 nonmatching paths. The
normal closure step is compared with the axisymmetric Winkler oracle
`F=pi*keff*R*delta^2`; the sliding step is checked against `|T|=mu*N`.
It also gates monotone normal-load error, positive non-duplicated dissipation,
force balance, a complete role-exchanged path and rejected-step rollback.

Default tests skip the expensive 4/5 Hessians. Run explicitly with:

`TENSORFEM_RUN_SLOW_FRICTIONAL_SURFACE=1 PYTHONPATH=src python scripts/run_frictional_surface_mesh_qualification.py`
