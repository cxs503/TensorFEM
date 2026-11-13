# TensorFEM Knowledge Governance Report

## Overview

- **Total anchors**: 36
- **Certified anchors**: 28
- **Contradicted anchors**: 3
- **Uncertified anchors**: 5
- **Total gates**: 325
- **Knowledge objects analysed**: 325
- **Duplicate candidates found**: 15
- **Cross-anchor contradictions**: 18
- **Entity clusters**: 2

## Entity Clusters (Potential Duplicates)

### Cluster `e696eb2a` (2 anchors)

Anchors: fem_beam_static, fem_structural_benchmarks

- **fem_beam_static ↔ fem_structural_benchmarks** (similarity=0.533)
  - L1.order_l2 ↔ B3.plate_order (sim=0.762): observed convergence order of the L2 deflection error…

### Cluster `4060d556` (2 anchors)

Anchors: fem_winkler_plate, fem_winkler_plate_k_sweep

- **fem_winkler_plate ↔ fem_winkler_plate_k_sweep** (similarity=0.527)
  - C2_equilibrium ↔ A_equilibrium_k=1000 (sim=1.000): Foundation reaction residual…
  - C2_equilibrium ↔ A_equilibrium_k=1e+04 (sim=1.000): Foundation reaction residual…
  - C2_equilibrium ↔ A_equilibrium_k=1.006e+04 (sim=1.000): Foundation reaction residual…

## Top Duplicate Candidates

| Anchor A | Anchor B | Similarity | Shared Gates |
|----------|----------|------------|--------------|
| fem_beam_static | fem_structural_benchmarks | 0.5333 | 1 |
| fem_winkler_plate | fem_winkler_plate_k_sweep | 0.5275 | 10 |
| fem_pinched_cylinder | fem_shell_scordelis_lo | 0.4828 | 10 |
| fem_cantilever | fem_pinched_cylinder | 0.4760 | 1 |
| fem_plate_static | fem_structural_benchmarks | 0.4582 | 4 |
| fem_cylinder_radiation | fem_kirsch_plate | 0.3990 | 1 |
| fem_beam_static | fem_plate_static | 0.3967 | 1 |
| fem_modal | fem_shell_modal | 0.3967 | 2 |
| fem_sent_j_integral | fem_sent_plane_strain | 0.3900 | 3 |
| fem_beam | fem_plate | 0.3875 | 1 |
| fem_pinched_cylinder | fem_plate_modal | 0.3800 | 1 |
| fem_cantilever | fem_cantilever_shear | 0.3687 | 2 |
| fem_kirsch_plate | fem_sphere_pressure_membrane | 0.3609 | 1 |
| fem_beam_static | fem_winkler_plate | 0.3550 | 1 |
| fem_plate_static | fem_winkler_plate | 0.3550 | 3 |

## Cross-Anchor Contradictions

- **Hard contradictions**: 10
- **Soft contradictions**: 8

### Hard Contradictions

- **fem_plate_static** (L2.ssss_centre_vs_navier_11) vs **fem_structural_benchmarks** (B1.centre_vs_timoshenko)
  - Quantity: SSSS centre deflection vs Navier 11x11 (N=40)
  - Values: 1.174701e-05 vs 3.610936e-03

- **fem_plate_static** (L2.ssss_centre_vs_navier_11) vs **fem_structural_benchmarks** (B2.centre_vs_navier)
  - Quantity: SSSS centre deflection vs Navier 11x11 (N=40)
  - Values: 1.174701e-05 vs 3.029707e-03

- **fem_plate_static** (L2.ssss_centre_vs_timoshenko) vs **fem_structural_benchmarks** (B1.centre_vs_timoshenko)
  - Quantity: SSSS centre deflection vs Timoshenko table 8 alpha=0.00406 (N=40)
  - Values: 5.795395e-04 vs 3.610936e-03

- **fem_plate_static** (L2.ssss_centre_vs_timoshenko) vs **fem_structural_benchmarks** (B2.centre_vs_navier)
  - Quantity: SSSS centre deflection vs Timoshenko table 8 alpha=0.00406 (N=40)
  - Values: 5.795395e-04 vs 3.029707e-03

- **fem_plate_static** (L2.ssss_centre_vs_navier_301) vs **fem_structural_benchmarks** (B1.centre_vs_timoshenko)
  - Quantity: SSSS centre deflection vs converged Navier series (N=40)
  - Values: 6.635595e-08 vs 3.610936e-03

- **fem_plate_static** (L2.ssss_centre_vs_navier_301) vs **fem_structural_benchmarks** (B2.centre_vs_navier)
  - Quantity: SSSS centre deflection vs converged Navier series (N=40)
  - Values: 6.635595e-08 vs 3.029707e-03

- **fem_plate_static** (L2.monotone_ssss) vs **fem_structural_benchmarks** (B1.centre_vs_timoshenko)
  - Quantity: SSSS centre-deflection error decreases monotonically
  - Values: 1.000000e+00 vs 3.610936e-03

- **fem_plate_static** (L2.monotone_ssss) vs **fem_structural_benchmarks** (B2.centre_vs_navier)
  - Quantity: SSSS centre-deflection error decreases monotonically
  - Values: 1.000000e+00 vs 3.029707e-03

- **fem_beam_static** (L1.cantilever_tip) vs **fem_structural_benchmarks** (A1.tip_deflection)
  - Quantity: cantilever tip deflection relative error (N=40)
  - Values: 6.455690e-11 vs 2.749004e-12

- **fem_beam_static** (L1.sections) vs **fem_cantilever** (geometry_extent)
  - Quantity: max relative error of the 5 section deflections (N=40)
  - Values: 2.539297e-07 vs 0.000000e+00

## Knowledge Object Registry Summary

| Anchor | Gates | Certified |
|--------|-------|-----------|
| fem_beam | 3 | ✓ |
| fem_beam_static | 6 | ✓ |
| fem_bending_patch | 17 | ✓ |
| fem_cantilever | 7 | ✓ |
| fem_cantilever_shear | 23 | ✓ |
| fem_curved_beam | 8 | ✓ |
| fem_curved_ring_solid | 8 | ✓ |
| fem_cylinder_radiation | 15 | ✓ |
| fem_hertz_contact | 7 | ✓ |
| fem_hertz_contact_loadcontrol | 20 | ✓ |
| fem_inverse_identification | 11 | ✓ |
| fem_kerr_plate | 7 | ✓ |
| fem_kirsch_plate | 14 | ✓ |
| fem_modal | 6 | ✓ |
| fem_morley_skew_plate | 9 | ✓ |
| fem_panel_buckling | 6 | ✓ |
| fem_pinched_cylinder | 6 | ✓ |
| fem_plate | 3 | ✓ |
| fem_plate_modal | 9 | ✓ |
| fem_plate_postbuckling | 6 | ✓ |
| fem_plate_static | 11 | ✓ |
| fem_sent_j_integral | 17 | ✗ |
| fem_sent_plane_strain | 10 | ✓ |
| fem_shell_modal | 4 | ✓ |
| fem_shell_scordelis_lo | 12 | ✓ |
| fem_sphere_pressure_membrane | 11 | ✓ |
| fem_structural_acoustics | 17 | ✗ |
| fem_structural_benchmarks | 12 | ✓ |
| fem_wave_dispersion | 10 | ✓ |
| fem_winkler_plate | 3 | ✓ |
| fem_winkler_plate_k_sweep | 27 | ✗ |

## Methodology

- **Entity resolution**: Jaccard + token overlap similarity on gate quantity descriptions, combined with anchor name similarity (weights 0.6/0.4 for gate, 0.7/0.3 for candidate scoring)
- **Contradiction detection**: Groups gates by normalised quantity tokens; flags pairs where relative value difference exceeds 10× the tighter threshold (hard) or 50% (soft)
- **Clustering**: Union-Find on duplicate candidate graph at similarity threshold 0.5

## Limitations

- Similarity is text-based; semantic equivalence (e.g. 'tip deflection' vs 'end displacement') may be missed
- Contradiction detection requires numeric values; gates without values are excluded
- Thresholds are tunable; results should be reviewed by domain experts before action
