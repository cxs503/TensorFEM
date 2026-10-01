# Experimental assumed-shear Shell4

This module tests a selective assumed-natural-strain idea without changing the
published Scordelis--Lo reference (`uz = -0.3024`).  It uses full integration
for normal membrane strains, a centre-tied constant membrane shear strain,
and the existing reduced-shear Mindlin bending field.

## Negative validation result

| mesh | computed `uz` | relative error |
|---:|---:|---:|
| 4 x 4 | -0.601653 | 98.96% |
| 8 x 8 | -0.465325 | 53.88% |
| 12 x 12 | -0.458582 | 51.65% |

For comparison, the original flat-facet formulation gives errors of 62.98%,
34.11%, and 40.80% on those meshes. Adding the diaphragm-axis rotation to the
end constraints changes the 12 x 12 displacement by less than 0.001%, so the
boundary rotation is not the source of the discrepancy.

The experiment is finite and balances the applied load, but does not converge
to the reference.  It therefore remains excluded from the stable benchmark
registry.  The evidence rejects the hypothesis that transverse/membrane shear
tying alone repairs the flat-facet formulation; a consistent curved-shell
membrane--bending interpolation is required before qualification.
