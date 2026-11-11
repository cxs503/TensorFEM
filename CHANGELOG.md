## Unreleased

### 3-D cohesive interface: closed-form dissipated energy + imported-checkout guard
- **New capability** `BKCohesive3DLaw.dissipated_energy(separation, history)` and
  `Cohesive3DResponse.dissipated_energy`: the energy dissipated per unit reference
  interface area is now a closed-form state function of the history variable,
  `W_d(h) = int_0^h t_eff(s) ds - t_eff(h) h / 2` (the second term is what the secant
  unloading branch stores).  `cohesive_hex8_response` aggregates it into
  `diagnostics["dissipated_energy"]` [J] (per Gauss point:
  `dissipated_per_area` [J/m^2]), so an interface patch reports its own damage energy
  and a nonlinear solve can check `W_ext = W_strain + W_d` without integrating the
  traction history; delamination R-curves get `W_d / A` directly.
- **Verification (closed form vs two independent oracles)**: numerical
  `int t . d(delta)` along a monotonic ray for 7 mode mixes → max deviation `5.0e-6`;
  `W_d(a_f) == G_c(direction)` for mode I / II / III / 45° / 3-D → `5.7e-16`;
  element `dissipated_energy == G_c * A` for **both** kinematics → `1.1e-16`;
  rigid-rotation invariance (corotational) → `2.3e-16`; elastic branch exactly 0.
  The verification table grows from 18 PASS to **24 PASS / 0 FAIL / 2 KNOWN_LIMIT**.
- **Measured non-proportional path** (previously an unquantified scope limit):
  open to `1.5 a_0(I)` then shear to failure gives `W_path = 1200.249 J/m^2` vs the
  closed form `1199.754 J/m^2` (residual `4.1e-4`) and stays below `G_IIc = 1200`.
  The limitation is now a measured row instead of a sentence.
- **Defect fix · verification provenance**: `benchmarks/cohesive3d_verification_tables.py`
  imported `tensorfem` from whatever the shared `site-packages` resolved to.  In a
  multi-worktree host that was *another* checkout, where the run died inside a bug
  already fixed on main (`RuntimeError: size of tensor a (3) must match tensor b (16)`);
  a table generated that way would describe the wrong code.  The script now puts its own
  `../src` first on `sys.path` and writes a `provenance` block
  (`tensorfem_module_file`, `imported_from_this_checkout`, `git_commit`, `git_dirty`)
  into the JSON, warning loudly if the module still comes from elsewhere.
- **Tests** `tests/test_cohesive3d.py`: 11 new tests (path-integral oracle, `G_c` at
  failure for 5 directions, elasticity/monotonicity/saturation, unloading independence,
  compression, element-level `G_c A`, rotation invariance, bounded non-proportional
  path).

### Foundation: Winkler `k` sweep (3 decades) vs Hetényi — GAP-FEM-WINKLER hardening
- **New benchmark** `benchmarks/fem_winkler_plate_k_sweep.py`: the existing
  `fem_winkler_plate` anchor pins one foundation modulus (ρ_w g) and sweeps the mesh;
  a wrong units/quadrature/factor-of-`l` in `K_w = k ∫ Nᵀ N dA` survives no test at a
  single `k`, because `k l⁴ = D` binds every length scale together. This benchmark
  moves `k` instead. Family A: infinite plate (quarter-symmetry Q4 Mindlin, MITC) under a
  central point load vs Hetényi's kei solution, `k` = 1e3 / 1e4 / 10055.25 / 1e5 / 1e6 N/m³
  (3 decades); family B: finite simply supported EB beam (L = 20 m) under a central point
  and a uniform load vs the exact Fourier-sine solution, `k_b` = 1e4 / 1e5 / 1e6 N/m²;
  family C: finite simply supported plate under uniform pressure vs the Navier series,
  `k` = 1e4 / 1e5 / 1e6 N/m³.
- **Numbers** (`benchmarks/verified/fem_winkler_plate_k_sweep/result.json`, 27/27 gates
  pass): plate centre deflection error 0.0918 / 0.0701 / 0.0701 / 0.0021 / 0.2124 % against
  `w(0)=P/(8√(Dk))`; profile error at r/l = 0.5/1/2/3 ≤ 0.0785 % (normalised by `w(0)`,
  since kei crosses zero); foundation equilibrium residual ≤ 1.4e-08 %; beam sup-norm error
  ≤ 1.94e-04 %; finite-plate centre error ≤ 0.1533 %. All three chains are inside the
  stricter 3 % reading, not just the 5 % gate.
- **Documented model boundary**: thick-plate regime `t/l > 0.1` diverges from the Kirchhoff
  oracle by O((t/l)²) — 150 mm ice plate at `k = 1e6` gives 3.54 → 4.43 → 4.84 % for
  n = 20/40/60 (error *growing* towards the converged Mindlin value, i.e. model difference,
  not implementation error). Family A therefore uses a 50 mm steel plate (`t/l ≤ 0.04`).
- **Tests** `tests/test_fem_winkler_plate_k_sweep.py`: 18 passed (fast lane) + 1 slow
  full-sweep gate; one of them is a mutation probe (element `k` x1.25 -> 10.9 % error ->
  gate breached), so the acceptance chain is not inert.
- **Report** `docs/foundation-plate-winkler-k-sweep.md` (rendered from the result JSON by
  `scripts/winkler_k_sweep_doc.py`, zero hand-copied numbers).

### Torsion: multi-cell closed sections (double-hull box girder) and the shared-wall flow
- **New module** `tensorfem.multicell_torsion`: finite-difference Prandtl oracle with one
  unknown constant per compartment (circulation conditions `contour_i(dphi/dn) ds = 2 G theta A_i`
  eliminating them in one `n_cell` linear system), the classical multi-cell Bredt solution
  (shared-wall flow `q_i - q_j`), and the merged HEX8 path on a conforming block mesh of the
  same section.  This is the gap written down in `docs/box-girder-torsion-validation.md` §5
  ("未覆盖的仍是缺口：多闭室箱梁").
- **Numbers** (`benchmarks/multicell_box_torsion_evidence.json`): single-cell degeneracy
  bit-identical to `box_torsion` (`J_T` rel diff 0.0); circulation residual <= 3.1e-15;
  energy vs torque identities within 1.19 % at `h = 0.0125`; HEX8 vs FD
  -0.55 % (single) / -0.92 % (symmetric 2-cell, 12555 dof) / -0.81 % (asymmetric 2-cell) /
  -0.70 % (3-cell), all inside the 3 % gate; shared-web shear flow 0.204 vs Bredt 0.205
  (0.5 %).  Symmetric double hull: the web carries 6.9 % of the outer-wall flow
  (Bredt: exactly zero) and adds only +0.84 % to `J`.  Bredt truncation 10.83 % -> 4.99 % ->
  2.40 % -> 0.94 % for `t = 0.2/0.1/0.05/0.02` (first order); "independent cells" naive sum
  is -20 % to -37 % (design-office trap quantified).
- **Tests** `tests/test_multicell_torsion.py`: 17 passed.
- **Report** `docs/multicell-box-torsion-validation.md` (rendered from the evidence JSON by
  `scripts/render_multicell_box_torsion.py`), figures `docs/assets/multicell-box-torsion/`.

### Fracture: level-set crack propagation and floating ice-sheet bending fracture (p360w1-n02)
- **New package** `tensorfem.levelset` (5 modules): Sussman signed-distance
  re-initialisation with a Godunov upwind Hamiltonian and a first-order band-edge
  fallback (max `||grad phi|-1|` = 2.6e-4, gate < 1 %), semi-Lagrangian /
  upwind level-set advection, Adalsteinsson-Sethian velocity extension,
  marching-squares zero contour with edge-key topology, Erdogan-Sih maximum
  hoop-stress and maximum energy-release-rate kink criteria, the
  Kirchhoff/plane-stress-analogy bending-Griffith relation
  `G = pi c^2 M^2 a/(D h)`, a Mindlin plate + Winkler foundation finite element
  with an explicit (node-duplicated) through crack and virtual-crack-extension
  energy release rate, and a level-set driven growth loop for a floating ice
  sheet.
- **Numbers** (`artifacts/levelset_ice_plate_benchmark.json`,
  `artifacts/levelset_vce_convergence.json`): G vs analytic bending Griffith
  -2.6 % at `h_e = a/8` (gate < 10 %); incidence fracture load 11.73 kN vs the
  analytic small-crack 10.60 kN (+10.7 %, gate < 20 %); uncracked patch-load
  centre moment vs the exact Kelvin solution +1.07 % (and the Westergaard
  constant 0.6159 = ln 2 - gamma + 1/2 confirmed, the 4 pi prefactor refuted);
  mixed-mode level-set path deviation 7.0 % / RMS 2.1 % (experimental arm
  WITHHELD: no digitised dataset available); unstable crack growth with G/G_c
  rising 1.04 -> 1.43.
- **Tests** `tests/test_levelset_fracture.py`: 21 passed.
- **Report** `docs/levelset-ice-plate-fracture.md`,
  figures `docs/assets/levelset-ice-fracture/`.
- **Performance note**: `torch.linalg.inv` costs ~17 ms per call on this host, so
  the rectangular Mindlin element is re-implemented in numpy
  (`tensorfem.levelset.plate_crack.rect_mindlin_stiffness`) and asserted equal to
  `tensorfem.plate.q4_mindlin_stiffness` to 1e-16 relative.

### Plasticity: three-case qualification of the J2 surface (p337w1-n07)
- **New verification surface** (`tensorfem.j2_elastoplastic_three_case`): one material
  card (E = 210 GPa, nu = 0.3, sigma_y = 250 MPa, H = 1000 MPa) driven through
  (A) a displacement-controlled TET4 tensile bar (1/8 model, 48 tets), (B) a
  material-point pure-shear path plus a TET4 homogeneous-shear patch test, and
  (C) a plane-strain axisymmetric thick-walled cylinder, against oracles the code
  under test never sees (bilinear card + independent scalar bisection for A, closed
  form tau(gamma) with tau_y = sigma_y/sqrt(3) for B, Hill 1950 + quadrature u_r(r)
  for C).
- **Numbers** (`artifacts/j2_elastoplastic_three_case/report.json`): sigma(eps)
  3.8e-15, sigma_y read back from the FE curve 250.0000 MPa (3.6e-14), tau_y
  144.337567 MPa vs 250/sqrt(3) (2.1e-16), pi-plane 6-direction deviation 1.2e-16;
  r_p(p) worst 1.37% (band 5%) over five contained-plastic levels p/k = 1.00..1.35,
  u_r(r) worst 2.08% (band 10%); Roache GCI on u_r(a) over n_r = 16/32/64 <= 8.6e-4.
  The residual ~2% offset against Hill is traced to the classical plane-strain
  sigma_z inconsistency for nu != 1/2, not to the mesh (GCI 0.1%).
- **Honest gaps**: no ASTM E8 dataset exists in the repository, so the experimental
  chord is *withheld* (scored against the analytic card instead); the campaign's
  literal p = 0..100 MPa band is elastic for this card (p_y = 108.25 MPa) and is
  reported as an elastic check; the H = 1000 MPa cylinder has no closed form and is
  verified by mesh refinement only.
- 11 pytest cases (`tests/test_j2_elastoplastic_three_case.py`, end-to-end case in the
  slow lane); report `docs/j2-elastoplastic-three-case.md`.

### FSI: partitioned vortex-lattice/FE flutter verification (P88W1-N07)
- **New time-domain fluid solver** (`tensorfem.unsteady_vlm`): 2-D unsteady vortex
  lattice for a flat plate -- cosine panels, quarter-chord bound vortices, wake shed as
  finite vortex-**sheet** segments (implicit Kelvin closure) and unsteady Bernoulli
  loads. Verified against Theodorsen (1935): steady lift exact, harmonic lift/moment
  amplitude within 4.29% and phase within 2.11 deg for
  `k = 0.05..0.5`, pitch and plunge.
- **New frequency-domain reference** (`tensorfem.theodorsen_pk`): real Hankel-function
  `C(k)`, Theodorsen lift/moment matrices and a p-k continuation (complex `k = -i p b/U`
  with mode tracking) giving the honest flutter boundary. The legacy
  `tensorfem.theodorsen` flutter routines are documented as MOCK (its
  `theodorsen_critical_velocity(20)` returns 8.9443 against the p-k 2.2096).
- **New partitioned FSI driver** (`tensorfem.fsi_flutter_vlm`): explicit staggered
  coupling of the vortex lattice with a 2-DOF modal structure whose frequencies come
  from the TensorFEM FE modal solve (cantilever plate 1.0x0.2x0.01 m aluminium in air:
  `mu = 140.3`, `sigma = 0.106`,
  `b*omega_alpha = 48.75 m/s`).
- **Evidence** (`docs/fsi-flutter-vlm-N07.md`, `artifacts/fsi_flutter_vlm_n07/`):
  time-domain `U_cr` within 1.89% of the p-k reference for mu = 5, 10, 20, 50; flutter
  frequency within 1.66% at 1.05 U_cr; supercritical growth rates within
  7.7%; mass-ratio exponent
  0.4973 (0.54% from 1/2 for mu >= 200);
  negative control rho_f -> 0 within 0.80% of the sqrt(mu) law.
  13 pytest cases (`tests/test_fsi_flutter_vlm.py`).

### Plasticity / verification: non-linear (Voce) hardening and the Hill elastic-plastic cylinder
- **Implicit J2 return mapping with non-linear isotropic hardening**
  (`tensorfem.j2_saturation_hardening`): yield law
  `sigma_y(alpha) = sigma_y0 + H alpha + Q (1 - exp(-b alpha))` (Voce/linear/perfect
  as special cases), so the incremental multiplier has no closed form; it is solved by
  Newton inside the analytic bracket `[0, q_tr/(3G)]` (`f' = -(3G+H) < 0`). The
  consistent tangent generalises the linear-hardening operator to
  `C = K 1x1 + 2G beta I_dev - (9G^2/q_tr^2)[1/A - dg/q_tr] s_tr x s_tr` with
  `A = 3G + H(alpha_{n+1})`, verified against central finite differences of the
  implemented return map (3.1e-10 relative). Audited numbers: uniaxial
  stress-strain vs the closed-form 1-D solution 1.4e-13 over `0 <= eps <= 2%` for four
  hardening laws, post-mapping identities (yield, flow rule, hardening law,
  incompressibility, scalar consistency) `<= 1.2e-15`, isochoric plastic flow
  `|tr(eps_p)|/dg = 1e-16`, local Newton <= 4 iterations.
  The existing `tensorfem.plasticity` / `j2_elastoplasticity` surface covers *linear*
  hardening only, so the Voce branch (and the implicit solve it forces) is new.
- **Hill 1950 elastic-plastic thick-walled cylinder** (`tensorfem.thick_cylinder_plasticity`):
  plane-strain closed form (plastic radius `p(c) = 2k[ln(c/a) + (b^2-c^2)/(2b^2)]`,
  stresses in both zones, `p_y = k(1-a^2/b^2)`, `p_p = 2k ln(b/a)`, `u_r` by
  compatibility quadrature) plus a plane-strain axisymmetric Q4 model with `u_z = 0`
  at every node (exact `eps_z = 0`, exact cylindrical geometry) and consistent bore
  pressure loads. Campaign gate: plastic radius within 3% of the closed form for
  `n_r = 8..32` at `p/k = 1.0, 1.15` -> measured worst **0.34%** (Lamé-fit estimator;
  model-free bracket 1.8% worst); radial displacement at the bore 0.30%; hoop/radial
  stress <= 2.8% of `p`; yield residual at every yielded Gauss point 0.0; global Newton
  <= 7 iterations with a quadratic residual history. The element itself is qualified
  separately against the elastic Lamé field at `0.5 p_y` (sigma_theta 0.90%,
  `u_r(a)` 0.021% at `n_r = 32`). `sigma_z` differs from the classical formula by up to
  13.8% of `p` at the interface, where Hill's incompressible-plastic-zone assumption
  makes `sigma_z` jump while the FE field stays continuous — reported as an
  approximation in the closed form, not an FE defect.
- Tests `tests/test_j2_saturation_hardening.py`, `tests/test_thick_cylinder_plasticity.py`
  (42 cases, ~22 s); report generator
  `scripts/plasticity_hill_thick_cylinder_report.py`; documentation
  `docs/thick-cylinder-von-mises-hill.md`; artefact
  `artifacts/plasticity_hill/plasticity_hill_report.json` + six figures; platform
  progress report delivered as report id 10562
  (`scripts/plasticity_hill_platform_report.py`, payload rebuilt from the artefact).

### Ice-belt structures / arc-length continuation
- **Riks arc-length limit-load continuation on the layered elastoplastic plate**
  (`tensorfem.ice_belt_collapse.solve_arc_length`, `arc_length_limit`,
  `arc_length_status`, `plate_field_collapse_arc_case`,
  `ice_belt_limit_pressure_arc_case`): spherical load-factor-weighted constraint
  with Crisfield root selection and a continuation guard, `K_T` from the
  finite-difference consistent tangent plus the von Kármán geometric stiffness.
  Verified anchors: linear-elastic exactness 8.9e-17, peak vs load control
  +0.3 %, energy ledger 3e-4, h-refinement -1.6 %/-0.5 %.
  Plate-field peak 256.4 MPa at 32x16 (-15.3 % vs Faulkner-Smith -> criterion
  FAIL, -0.4 % vs Paik-Thayamballi); ice-belt plastic limit 897.5 kPa at 48x16
  (+23.5 % vs the strip yield line, twice-elastic-slope design limit 826 kPa
  stays inside the 15 % gate). Docs `docs/ice-belt-arc-length.md`, benchmark
  `benchmarks/ice_belt_arc_length.py`, artefact
  `artifacts/ice_belt_collapse/arc_length.json`.

### Solids / verification
- **TET10 mass cross-check without quadrature** (`tensorfem.tet10_mass_crosscheck`,
  `tests/test_tet10_mass_crosscheck.py`): the shipped `tet10_consistent_mass` was
  verified against a 24^3 Duffy reference in `test_tet10_mass.py`; this adds a
  method-independent closed-form reference (barycentric monomial integrals, no
  quadrature at all) and reproduces the operator to 1.10e-15 Frobenius with the
  analytic row sums `-rho V/20` (vertex) / `+rho V/5` (mid-edge). New measurements:
  rigid-rotation energy vs the closed-form tetrahedron inertia tensor (<=1.2e-15,
  three axes), single-element free-free rigid eigenvalues 4.67e-17 relative to the
  first elastic mode, `order=4` collapsed rule as the first exact order
  (2->2.92e-1, 3->3.89e-2, 4->1.1e-15), the row-sum lumping trap (negative vertex
  masses, 12 negative components) and HRZ positivity. Two residual debts are
  quantified instead of hidden: the rod eigenfrequency anchor used with the
  constrained modulus is a 1-D model substitution, not a 3-D verification (with
  `nu = 0` the same mesh converges at high order, 1.4e-5 at 4 elements; with
  `nu = 0.3` it plateaus near 0.5 % at ~O(h) because lateral contraction at the
  fully clamped end dominates), and HRZ lumping is ~50x less accurate than the
  consistent matrix on the same mesh. 10 new gates.

### Solids / inertia
- **HEX20 consistent & lumped mass** (`tensorfem.hex20_mass`, `hex20_lumped_mass`,
  `hex20_mass_totals`, `assemble_hex20_mass`, `hex_gauss_rule`,
  `tensorfem.hex20_mass_audit`, `scripts/audit_hex20_mass.py`,
  `docs/hex20-consistent-mass.md`): HEX20 shipped stiffness only
  (`grep -n mass src/tensorfem/hex20.py` was empty), so every vibration/transient
  claim on the workhorse quadratic brick was unsupported. Default 3x3x3 Gauss is
  exact for affine/straight geometry: mass matrix vs an independent NumPy
  serendipity oracle (own node table, own central-difference Jacobian) 1.07e-10
  Frobenius, total mass 8.4e-11, rigid translation *and* three-axis rotation
  kinetic energy 8.4e-11 vs the analytic cuboid inertia, six free-free rigid
  modes 6.45e-10 relative. Eigenfrequency evidence: 1-D constrained rod
  6.45e-6 error at 16 elements with observed order 4.00, full 3-D rod axial mode
  0.44 % at 8 elements (sqrt(E/rho), lateral contraction resolved). Documented
  boundaries: the *total mass is exact for every Gauss order* while the order-2
  matrix is 13.17 % wrong; affine shear stays exact but non-planar faces give
  7.3e-4 and curved mid-edge 0.182 % (order 4 recovers 1.2e-10); row-sum lumping
  is structurally indefinite (corner mass = -rho*V/8, 24 negative components) so
  HRZ is the default. 13 new tests including three negative controls.

### Solids / integration accuracy
- **TET10 integration-accuracy audit** (`tensorfem.tet10_integration_audit`,
  `scripts/audit_tet10_integration.py`, `docs/tet10-integration-accuracy.md`): the
  outstanding "Keast is only degree-three exact, accuracy loss unquantified" debt is
  closed with an independent high-order reference (Duffy collapsed-cube tensor-product
  Gauss, `24**3` points/element) that first verifies itself against closed-form
  monomial integrals (<=1e-16) and against the production torch implementation
  (<=2.7e-16, five geometries) before judging the shipped rule. Straight-edged
  distortion: 3.64e-15 (the rule is exact on affine geometry, so the existing TET10
  cantilever benchmark is unaffected -- its 2.44 % Timoshenko residual is
  discretisation, its element-level quadrature error is 1.94e-15). Curved isoparametric
  edges: 0.428 % / 2.551 % / 3.292 % / 7.970 % for mid-edge offsets of
  0.05 / 0.11 / 0.12 / 0.15 edge lengths -- the 3 % gate is first crossed at 0.12 (still
  a well-shaped element, min detJ 0.32), and a random-warp control reproduces the same
  magnitude, so the deficiency is not a hand-picked pattern. Affine-field strain energy
  is exact to <=3.1e-15 at every curvature, isolating the `B`-matrix sampling (not detJ)
  as the error source. The consistent mass matrix cannot be integrated by the shipped
  rule at all (Frobenius 19.7 %, diagonal 70.8 %-81.9 %) -- a forward constraint for a
  future `tet10_mass`. `tet10_stiffness(x, d, rule=(points, weights))` adds an opt-in
  high-order hook whose default path is bit-for-bit unchanged; `duffy_rule(6)` (216
  points) reduces the curved error to 3.1e-06. 21 gates in
  `tests/test_tet10_integration_audit.py`.

### Vibration / acoustics
- **Plate structure-acoustic radiation chain** (`tensorfem.plate_acoustics`,
  `tensorfem.plate_radiation_chain`): the Q4 Mindlin eigenpairs of a simply-supported or
  clamped plate are turned into a sampled surface-velocity field (bilinear interpolation on
  a Gauss grid, unit-RMS normalisation) and fed to three *numerically independent*
  evaluations of the baffled half-space radiated power -- the 4-D Rayleigh surface integral,
  a far-field boundary-integral extrapolation of the directivity, and the closed-form
  angular spectrum of a sine mode -- plus the radiation efficiency
  `sigma = P/(rho c S <v^2>)` in the harmonic (rms) convention, the coincidence frequency
  `c^2/(1.8 c_L h)`, the equal-modal-energy band average and its asymptotics. Evidence
  (`artifacts/plate_radiation_chain/`, 4 figures, `scripts/plate_radiation_chain_benchmark.py`):
  simply-supported eigenfrequencies within 1.71 % of Kirchhoff (5 modes, 24x24), clamped
  within 2.68 % of Leissa's CCCC table, FEM-driven sigma within 0.14 % of the analytical
  Rayleigh value, the three acoustic routes agreeing to 7e-16 at `ka` from 0.4 to 55,
  fixed-mode low-frequency exponent 1.9446 against the classical 2, `sigma(10 f_c) =
  1.00003`, piston limit 1.00085. **Honest FAIL**: the narrow-band radiation-efficiency peak
  sits at 1.0607 `f_c`, i.e. 6.07 % from `c^2/(1.8 c_L h)` (criterion was 5 %); a discrete
  finite panel can only sample the coincidence singularity from a distance, and the 1/3-octave
  average is worse (1.2 `f_c`). Audit note: the existing `acoustic_radiation` module mixes
  the peak-squared and rms velocity conventions, so its Leppington branch is a factor two
  above the exact Rayleigh result far above coincidence (left unchanged, documented in
  `docs/plate_radiation_chain.md`). 25 tests in `tests/test_plate_radiation_chain.py`.

### Marine structures / ultimate strength
- **Panel moment--curvature collapse surface** (`tensorfem.panel_mchi_qualification`): a
  Smith-method integration that re-solves the axial equilibrium of a stiffened-panel
  cross-section at every curvature increment (the neutral axis migrates instead of being
  frozen at its elastic value), assigns each fibre its own limiting mode and optional
  softening branch so a fibre past its limit unloads while the curvature grows, and reports
  a verification ledger: work `int M dchi` against fibre internal energy, increment
  refinement, monotonicity to the peak and peak uniqueness. A *plateau* fibre law has no
  limit point by construction and the collapse object says so (`saturating`), so an
  asymptotic plastic moment cannot be reported as an ultimate point. Cross-verification
  surface: beam eigenvalue vs Euler, finite-difference plate eigenvalue vs the `k = 4`
  anchor (24x24 error 0.13 %), and the elasto-plastic Shell4 Riks unit cell
  (`stiffened_panel_unit_fe`) vs the section chain. Measured: energy residual second-order
  (1.0e-5 at 200 increments, 7.8e-7 at 800), `M_u` step-size independent to 1e-4, section
  inertia within 0.97 % of `StiffenedPanel.column_inertia`, Euler eigenvalue to 1e-8 at 64
  elements. **Honest FAIL**: the executed unit-cell Riks paths rise monotonically to
  0.97--0.99 sigma_y without reaching a limit point, so their values are lower bounds and
  the panel `sigma_u` vs FE criterion (15 %) is not met (28.5 % on the tee bay, 7.2 % on the
  ice-belt panel); the gap is attributed to the symmetry-plane rotational restraint of the
  unit cell against the simply supported edges assumed by the Faulkner plate limit. An
  elasto-plastic shell *bending* `M`--`chi` path is not executed and that sub-item is
  WITHHELD. Evidence `artifacts/panel_mchi_qualification/{evidence,fe_traces}.json`,
  figures `docs/assets/panel_mchi_qualification/`, note
  `docs/panel-mchi-ultimate-strength-validation.md`, generator
  `benchmarks/panel_mchi_qualification.py`, tests
  `tests/test_panel_mchi_qualification.py`.

### Adaptivity / verification
- **1D h-adaptive continuation with closed-form error** (`tensorfem.adaptive_1d`): the P1
  two-point boundary value problem on `(0,1)` with two analytic solutions (`u = sin(pi x)` and
  `u = x^0.6`, singular flux at `x = 0`), an analytic load vector (no quadrature on the singular
  element), a **closed-form discrete energy-norm error** per element, an L2-projection flux
  recovery with a closed-form element indicator, Doerfler marking and bisection refinement.
  Measured uniform orders `-0.9722` vs theory `-1` (2.8 %) and `-0.0977` vs theory
  `-(alpha-1/2) = -0.1` (2.3 %); the adaptive loop reaches `-0.5027` and needs **>= 39x** fewer
  unknowns than the uniform ladder at equal accuracy on the singular problem, while on the smooth
  problem it is a negative control (2x more unknowns than uniform). Rejection gates: non-monotone
  meshes, empty markings, out-of-range marking fractions, `alpha` outside `(0.5, 1)`, non-finite
  flux, and a zero exact error whose effectivity is undefined. No closed-form rate is claimed for
  the adaptive ladder: the textbook optimal grading (`gamma = 2/(2 alpha - 1)`) exhausts double
  precision at `N >= 64` and that limit is recorded in the grading scan.
- **2D Q4 analytic-order audit** (`tensorfem.adaptive_poisson_2d`): scalar Poisson on `(0,1)^2`
  with the polynomial bubble `u = 256 x^2(1-x)^2 y^2(1-y)^2` (zero boundary trace, exact
  quadrature for stiffness, load, errors and indicators), a vectorised gradient/recovery path
  guarded by a per-element `J^-1 dN/dxi` cross-check, and a recovered-gradient ZZ indicator.
  Energy order `-0.4848` and L2 order `-0.9704` against theory `-0.5` / `-1.0` (3.0 % at the
  finest rung pair); effectivity `theta = 1.0014` at 1521 unknowns, i.e. the estimator is
  asymptotically exact on a smooth field, which localises the bias measured on the singular
  Williams field to the *field*, not the estimator. The averaged recovery is only mildly
  superconvergent (`-0.72`, reported; SPR lives in `adaptive_zz`).
  Evidence `artifacts/adaptive_continuation/{study_1d,study_2d,grading_scan}.json` + `tables.md`,
  figures `docs/assets/adaptive_continuation/`, generator
  `scripts/adaptive_continuation_study.py`, note `docs/adaptive-continuation-order-cost.md`,
  tests `tests/test_adaptive_1d.py` + `tests/test_adaptive_poisson_2d.py`.
  Uniform meshes only in 2D; the local-refinement loop with hanging nodes remains
  `tensorfem.adaptive_zz`.

### Contact
- **Two-block frictionless mortar segment integration**
  (`tensorfem.mortar_segment_contact3d`, `tensorfem.mortar_two_block`,
  `tensorfem.mortar_segment_transfer`): the existing facet-quadrature kernel
  integrated every slave facet at its own quadrature points regardless of the
  overlap with a master facet. The new kernel clips each slave facet against its
  master facets in the master plane, fan-triangulates the clipped convex polygon
  and integrates the penalty potential with a three-point rule per segment
  triangle, distributing the traction to both surfaces with their own shape
  functions (non-matching interfaces, no node pairing). Gates:
  identical-block patch test reproduces the analytic uniform traction `p` to
  ~1e-10 at four refinement levels (area exact, equilibrium 1e-11, penetration
  `p/eps_n`); a prescribed smooth gap field converges at second order
  (L2 6.4e-2 -> 4.7e-3, fitted rate 1.97) against the analytic integral; the
  mismatched-punch interface distribution is declared withheld (penalty and mesh
  dependent, classical elastic edge singularity) while its integral is exact.
  Rejection gates: `eps_n <= 0`, initial interpenetration, invalid connectivity,
  too small a search radius. Evidence `artifacts/mortar-two-block/evidence.json`,
  tests `tests/test_mortar_segment_contact3d.py`, note
  `docs/mortar-two-block-segment-validation.md`. Frictional mortar with finite
  sliding remains roadmap.

### CI / tooling
- **main 红修复：lint 门的 format 维度**（`docs/lint-gate-scope.md` §Line endings）：
  `main` 在 `f5be5cf`（#253 合并）红在 `lint (advisory)`。根因是新增的
  `tests/test_smoke_m4622.py` 缺文件末尾换行——钉住的 `E4/E7/E9/F` 规则集下
  `ruff check` 对行尾无话可说（`All checks passed!`, exit 0），唯一的红信号是 job 的
  第二条命令 `ruff format --check`（`Would reformat`, exit 1）。修复=补末尾换行（1 字节）；
  并补两个守护测试：封闭负控（临时仓里无末尾换行的新文件：linter 过、formatter 必红）
  与本地回放（对当前改动跑 CI 的两条命令，浅克隆/ruff 越界时 skip），使该类红在
  presubmit 即可复现。证据 `artifacts/lint-gate/evidence-m4636.json`
  （before=`f5be5cf` 格式不洁 → after 全绿）。
- **Lint gate made deterministic**（`docs/lint-gate-determinism.md`）：`main` 自
  ruff 0.17 发布起持续红。根因不在代码——CI 的 lint 只判定本改动触及的文件，
  而 `[tool.ruff]` 未写 `select`，规则集继承 ruff 的隐式默认；ruff 0.17 把
  `I/UP/RUF/C4/PL` 并入默认，0.17.4 于是在同一批文件上报 39 个 finding（全部落在
  新并入的族，`E4/E7/E9/F` 为 0），而 0.15.0 对同一批文件 `All checks passed`；
  且 dev 依赖只写 `ruff>=0.6` 下限，CI 不同天解析到 0.15.0 / 0.17.3 / 0.17.4 三种
  版本，同一类代码得到相反结论。修复=`select = ["E4","E7","E9","F"]` 显式钉规则集 +
  `ruff>=0.15,<0.16` 钉 minor（0.15.x 同时是 loong64 有轮子的最新版，runner 与
  702 节点跑同一二进制）+ 3 个守护测试（含负控：对旧 pyproject 断言必失败）。
  39 个 finding 清单与"新族按需并入"的流程记在文档 §2/§5。

### Verification
- **Armstrong-Frederick 循环塑性最小面**（`docs/cyclic-plasticity-af-anchor.md`，P16-N2/m5197，
  补 m4951 明确 withhold 的循环加载缺口）：新增 `tensorfem.cyclic_plasticity`——AF 非线性随动
  强化 `dX=(2/3)C dε^p − γX dp` + 线性/Voce 等向强化，屈服函数按任务书写成
  `f=‖s−X‖_F−√(2/3)R(p)`；**全隐式**返回映射（`X^{n+1}=a(X^n+√(2/3)CΔp n)`，a=1/(1+γΔp)）
  收敛为单个单调标量方程（含 `w=s_tr−aX^n` 解析解，非比例步也精确，不做 X 冻结近似），
  一致切线含 `dX/dε` 耦合项（对中心差分 ≤1.5e-10）。四个闭式锚逐项对齐：稳定滞回环峰/谷
  `α_a=(C/γ)tanh(γε^p_a)`（≤0.16%，5 应变幅×2 循环数）、每循环耗散
  `4Rε^p_a+(4C/γ²)(γε^p_a−tanh γε^p_a)`（≤0.26%）、Bauschinger 反向屈服下降量 =α_a（≤0.69%，
  5 幅值）、棘轮率闭式 `(1/γ)[ln((C−γα_min)/(C−γα_max))−ln((C+γα_max)/(C+γα_min))]`（≤2.6%，
  20 循环×2 均值×2 幅值；Richardson 极限 0.02%）。`C→0` 逐位退化回 m4951 核（≤2.5e-16），
  `H_k=0` 负控使 C3 判据 100% FAIL。27 个 pytest + m4951 回归 16 个全绿，工件
  `artifacts/cyclic_plasticity/{verification.json,*.png}`。

- **J2 弹塑性最小面**（`docs/j2-plasticity-minimal-surface.md`，IN14-D4/m4951，补 m605
  「无弹塑性」缺口）：新增 `tensorfem.j2_elastoplasticity`（材料点 API），在既有
  `plasticity.update_j2` / `j2_consistent_tangent` 核上补出可验证面——材料参数封闭拒收
  （H<0、σy≤0、E≤0、ν∉(-1,0.5)、NaN、非对称/非 3×3 应变）、返回映射六项事后残差
  （屈服/应力恒等式/一致性参数/流动法则/硬化律/互补条件，实测 ≤3.7e-16）、单轴分段解析锚
  （弹性斜率 E、塑性斜率 EH/(E+H)，误差 ~1e-16/~6e-15）、纯剪与四个偏量方向屈服点对
  解析屈服面 q=σy（误差 ~2e-16，门槛 <1%）、一致切线对中心差分（4.2e-10）。
  16 个 pytest（含 7 个负例），工件 `artifacts/j2_minimal/{verification.json,fig_*.png}`。
- **均布压力板的精度复核**（`docs/plate-uniform-load-verification.md`）：已合并的 Q4 Mindlin
  板路径此前只对一条与第一阶模态同形的正弦载荷复核过，无法发现载荷向量、高阶模态刚度、
  剪切分解或边界条件错误。新增 `tensorfem.plate_uniform_load`（精确 Navier 双级数 +
  可分离 Rayleigh-Ritz 固支参考，Ritz 以正弦模态自检到 0.008 %）与
  `scripts/verify_plate_uniform_load.py`：简支/固支、$t/a=0.01/0.10$、均布压力下
  中心挠度、中心弯矩全部 < 3 %（最差 0.4886 %）；夹支边峰值弯矩不认证。
  3 个负控（4×4 粗网格保留失败、固支对简支参考、厚板对 Kirchhoff 参考）按设计失败。

- Added `tensorfem.prestressed_modal`: stress-stiffened (prestressed) free vibration for Euler--Bernoulli beams and Q4 Mindlin plates, `(K + N*Kg) phi = omega^2 M phi`, with the linear buckling load of the same discretisation reported and enforced fail-closed. Verified against the exact pinned-beam law `omega_n^2 = (n pi/L)^2 (EI (n pi/L)^2 + N)/rho A`, the exact Navier prestressed-plate law and the Navier buckling resultant (plate 2.13%/1.34% at 8x8/10x10, beam 3.5e-5 at 24 elements), plus an exact discrete identity that puts the fundamental at zero to 4e-10 at the reported critical load; the 4x4 plate mesh fails the 3% gate at 9.13% and is recorded as a coarse-mesh failure. Evidence `benchmarks/prestressed_modal_evidence.json`, generator `benchmarks/prestressed_modal_verification.py`, write-up `docs/prestressed-modal-verification.md`, gate `tests/test_prestressed_modal.py` (16 tests).

## 2026-11-11 -- independent review of `stiffened_panel_ultimate`

* **fix:** `simply_supported_plate_series` returned the centre deflection a
  factor `E` too large in SI units (`D/E` used as `D`); the series now takes
  `young` (default 2.06e11) and `plate_rigidity` is the physical rigidity `D`.
  Bending moments unchanged.
* **fix:** `flange_local_buckling_stress` now uses the outstand
  `(flange_width - web_thickness)/2` (new `flange_outstand` property) instead of
  the total flange width; `sigma_cr` was under-predicted by about four, costing
  up to 20 % of the panel ultimate for flanges with `b_f/t_f` in 18...37.
* **new:** `scripts/verify_stiffened_panel_ultimate_independent.py`
  (FD plate solver + numerically integrated yield-line virtual work +
  re-implemented component limits) with evidence in
  `artifacts/stiffened_panel_ultimate_review/`, `docs/stiffened-panel-ultimate-independent-review.md`
  and `tests/test_stiffened_panel_ultimate_independent_review.py`.
* **reported:** yield-line selector step of 53.8 % at `a/b = 1.5` (conservative,
  left as a quantified gap).


## 2026-11-04 大倾角稳性：盒形驳 GZ 曲线与 KN 横交曲线（精确截面几何）

- 新增 `tensorfem.marine_large_angle_gz`：补齐 `marine_hydrostatics` 文档里声明的
  「不支持大倾角 GZ / 横交曲线」欠账。浸水剖面用凸多边形裁剪 + shoelace 质心精确求值，
  等排水量（`Area = B T`）二分求水线偏置，`GZ = xB cos + (zB - KG) sin`；
  支持 0–90°、任意 KG、`KN` 横交曲线族与动稳性积分。
- 验证锚（`tests/test_marine_large_angle_gz.py`）：墙侧公式
  `sin(phi)(GM + BM tan^2 phi / 2)` 机精度复现（带内最大残差
  1.33e-15 m，相对 8.3e-16）；90° 精确几何
  `xB = B/2 - B T/(2D)`、`GZ = D/2 - KG`；初始斜率回到 `GM`（Richardson 1e-6）；
  排水体积守恒 1e-12；独立第二套列式（水线高度截断原函数 + 20 万点积分）在三个几何区
  一致到 1e-8；动稳性积分对闭式 `GM(1-cos) + BM(sec+cos-2)/2` 达 1e-6 且二阶收敛。
- 发现并记录：闭式甲板入水角只在墙侧假设 `h = T` 下成立，当 `D > 2T` 时龙骨先出水，
  真实甲板入水角晚于估计（基准驳闭式估计 33.69°，几何求解首个 0.5° 采样点 37.00°），必须用几何求解器而非闭式。
- 证据/图：`benchmarks/large_angle_gz_reference.json`（0.5° 步长全表）、
  `docs/assets/marine_large_angle_gz.png`、`docs/marine-large-angle-stability.md`；
  由 `scripts/plot_large_angle_gz.py` 生成，并有防漂移测试钉住。

## 2026-11-03 Cook 膜应力审计证据收尾

## 2026-11-04 — plane-strain B-bar Q4 (near-incompressible locking cure)

- `src/tensorfem/continuum_bbar.py`: mean-dilatation (B-bar) plane-strain Q4, the
  counterpart of `solid3d_bbar`. Deviatoric part integrated at 2x2 Gauss (no hourglass),
  volumetric part averaged; plane stress / non-Q4 refused explicitly.
- Qualification against the exact plane-strain Lame cylinder, `nu = 0.4999`, 306 DOF:
  displacement 0.1236 %, radial stress 0.5999 %, mean stress 0.0182 %, flat in `nu`,
  observed order 2.00. The merged fully integrated Q4 stays fail-closed at 61.9 % / 155.2 %
  and is still 38.5 % at 2450 DOF.
- `scripts/audit_plane_strain_bbar_locking.py`, `benchmarks/plane_strain_bbar_locking.json`,
  `docs/plane-strain-bbar-locking.md`, `tests/test_continuum_bbar.py`.
- `incompressible_locking.plane_strain_sweep` gains `formulations=("standard", "bbar")`
  and `plane_strain_case` also reports hoop and mean stress; default behaviour unchanged.

- 重新生成 `benchmarks/cook_membrane_stress_reference.json`：旧文件是单元 Jacobian 转置缺陷修复前的产物，
  冒记 `recovery_cross_check = Infinity` 与「加载边型线分歧」；现行代码下交叉校验 2.5e-14 通过，
  加载边型线 Q4/Q8 最大偏差 0.1087（占参考位移 0.454%），两项遗留疑问闭合。
- 新增存盘场量测：单元中心 von Mises vs 独立 Q8 参考，n=64 中位 0.137%、p95 0.467%、
  尺度归一化最大 2.910%（scoped 通过），n=32 为 4.787%（负控保留）。
- 任意点判据仍未达 3%（n=64 最大 6.375%，观测阶 0.983，外推需 n≈122）；生产 Q4 路径与
  `stress_accuracy_qualified=False` 均未改动。
- 新增 `tests/test_cook_stress_audit_closure.py` 与证据一致性断言，防止证据再次与代码漂移。

## 0.46.0

- Audited the Winkler-foundation analytical references against independent oracles (Green's-function quadrature, Hankel quadrature, finite-difference beam solve, equilibrium and exact centre-deflection identities). Findings and corrected error tables are in `docs/hetanyi-reference-audit.md`; the numbers are regenerated by `benchmarks/foundation_reference_audit.py`.
- Fixed `infinite_beam_uniform_load`: the previous expression was not the integral of the point-load Green's function (residual of order `q0` in the governing ODE; `q0/k_b` instead of zero in both far fields). The corrected closed form matches the Green's-function quadrature to 4.7e-8 and satisfies the ODE to the finite-difference truncation level.
- Replaced the fabricated `plate_mindlin.hetenyi_reference` exponential fit with the exact Kelvin-function solution `w = -P/(2*pi*k*l^2)*kei(r/l)`; the centre deflection reproduces `P/(8*sqrt(D*k))`, Hankel-quadrature agreement is 5.1e-5, and the physical uplift ring (sign change at `r/l = 3.96`) is recovered. The rejected fit deviated by up to 2211%.
- `finite_beam_series` now fails closed for `boundary='free'`: the sine expansion is exact only for simply supported ends and the previously ignored `boundary` argument silently returned a differently-bounded solution. Free-end end-conditioning remains an explicit open gap; the default is now `simply_supported`.
- Added 22 library-level verification anchors in `tests/test_foundation_reference_audit.py`, with self-validated Bessel/Kelvin oracles so the tests need no SciPy.

# Changelog

All notable user-visible changes are recorded here.

## Unreleased
- Added the first pressure-load verification of the metric-consistent cylindrical Shell4 element: an exact four-constant axisymmetric shell solution (`D w'''' + (E t / R^2) w = p`, clamped--clamped, open ends) for a 1.5 m x 15 mm steel cylinder under 1.0 MPa internal pressure with beta*L = 10.28. `tensorfem.cylindrical_pressure_reference` carries the closed form (displacement, moment, shear, hoop force; derivatives by automatic differentiation), self-validated against the governing ODE (residual 8.2e-16), the clamped end conditions (1.4e-12), the semi-infinite limit `|M_x(0)| = p/(2 beta^2)` (1.0e-4), the classical two-layer superposition (5.6e-5) and the pressure-work/strain-energy identity (5.3e-9), plus a consistent 2x2-Gauss pressure load and the production-element driver. At 32x16 elements (3168 dofs) the radial displacement L2 error is 1.41 %, the peak 1.64 %, the plateau 1.70 % and the axial moment L2 0.64 % with |M_x(0)| at 0.11 %, so the field-level 3 % gate passes; the node adjacent to the clamped edge is reported as NOT qualified (5.58 %) and an 8-facet circumferential mesh is rejected (5.82 %), fixing a mesh requirement instead of hiding it. The edge bending stress (181.6 MPa on top of the 100.0 MPa membrane stress) is the ship-relevant number: a `pR/t` screen understates the clamped-edge surface stress by 2.8x. Evidence in `benchmarks/cylindrical_pressure_boundary_layer_evidence.json`, report in `docs/cylindrical-pressure-boundary-layer.md`, gate `tests/test_cylindrical_pressure_boundary_layer.py` (14 tests, including a negative control that rejects a 10 % thickness error in the reference).

- Closed the quadratic-solid rank-deficiency debt and corrected a mislabelled evidence row: the
  library audit had recorded the 20-node brick as `hex20 (2x2x2 integration)` with zero spurious
  modes, but the shipped `hex20_stiffness` integrates with the exact-order 3x3x3 rule, and the
  measured 8-point rule gives **six** zero-energy mechanisms, so the old label and conclusion could
  not both stand. New `audit_quadratic_solids`, `audit_hex20_mesh_scan`, `hex20_stiffness_with_rule`
  (orders 2-5, `order=3` locked to production to round-off), `hex20_reduced_stiffness`,
  `quadrature_exactness` and `hex20_quadrature_convergence`; TET10 now enters the library-level
  audit, which had skipped the quadratic tetrahedron entirely. Results: the shipped rule has exactly
  the 6 rigid-body modes on regular, distorted and genuinely curved bricks (0 spurious) and
  reproduces a 64-point reference to 5.5e-16 on the regular brick; the 8-point rule has 6
  mechanisms on all three geometries and is 7.65%-8.84% too soft (measured against a converged
  rule, above the 3% gate); its mechanisms do not survive conforming assembly (warped and jittered
  free-free meshes keep a 6-dimensional null space). TET10 is rank sufficient with straight and
  curved edges. The Keast degree-3 rule under-integrates curved quadratic tetrahedra - quantified
  accuracy loss remains open, "rank sufficient" is explicitly not "integration accurate".
  Evidence `docs/benchmarks/tutorials/quadratic-solid-rank-audit.json`, write-up
  `docs/quadratic-solid-rank-audit.md`, gate `tests/test_quadratic_solid_rank_audit.py` (10 tests,
  including the 8-point rule as negative control).

- Audited the linear righting arm of `tensorfem.marine_hydrostatics` against a new
  independent large-angle oracle (`tensorfem.marine_large_angle_stability`): the exact
  wall-sided arm is rebuilt from immersed-section geometry (bisected flotation
  equilibrium plus exact half-plane polygon clipping), never from the textbook formula.
  Seven machine anchors close at 1e-12..1e-15, including the free-surface closure
  `h0 = T*cos(phi)`, the waterplane chord `B/cos(phi)`, the closed-form trapezoid
  centroid, the wall-sided arm, the dynamic-stability integral and the lurch angle of a
  near-neutral section. Findings: the merged `GM*sin(phi)` arm is 2.09% off at 10
  degrees for the shipped proportions but 3.27% for a low-`GM` barge of the same
  geometry, so the documented "<3% over the declared range" holds only for
  `GM >= 0.5027*BM`; and the textbook wall-sided closed form is valid only inside
  `min(phi_deck, atan(2*T/B))`, where the bilge-surfacing limit is the binding one for
  deep narrow sections (5.1% divergence just past it). Evidence
  `benchmarks/marine_large_angle_stability_evidence.json`, report
  `docs/marine-large-angle-stability-audit.md`, gate
  `tests/test_marine_large_angle_stability.py` (19 tests).

- Closed the body-force verification debt of the continuum path. Every previous quantitative check of `tensorfem.continuum` loaded the model through edge tractions or point loads, so the consistent assembly `int N_a b dV` used by inertia, gravity and centrifugal loads had no reference value anywhere in the repository. New `tensorfem.rotating_disc_reference` carries Michel's exact plane-stress solution for a rotating annular disc (stress, displacement, bore hoop stress, sector resultants) and `centrifugal_nodal_loads` (2x2 Gauss consistent assembly plus the equal-split variant). `benchmarks/rotating_disc_reference.py` verifies the production Q4 plane-stress path on a quarter annulus: at 24x24 the hoop-stress relative L2 is 0.24 % (worst point 0.47 %), radial stress 0.11 % of the hoop peak, radial displacement 0.47 %, with observed orders 1.4-2.7 and a symmetry-cut reaction matching `t*int sigma_theta dr` to 2e-3. Findings: (i) the bore hoop stress is a post-processed quantity -- two-point linear extrapolation underestimates it by 6.7 % on a uniform 12x24 mesh and by 1.07 % even after radial grading, while a three-point quadratic extrapolation on the graded mesh reaches 0.81 %, so an ungraded burst margin is unsafe; (ii) the assembled load resultant matches an independent centroid-triangulated integral over the discretised polygon to 1e-16 but the exact annular sector only to O(dtheta^2), because the chordal mesh domain is smaller -- the two effects are reported separately; (iii) equal-split lumping is *not* detectable by field comparison for this smooth load (0.16 % error), which is recorded as a capability boundary rather than a pass. Three negative controls (20 % omega error, radius-free body force, plane strain) each break the 3 % gate. Evidence `benchmarks/rotating_disc_michel_evidence.json`, write-up `docs/rotating-disc-michel-verification.md`, gate `tests/test_rotating_disc_reference.py` (17 tests).

- Added an independent Saint-Venant torsion verification of the 3-D solid paths, the one elementary load case that no stored HEX8/HEX20/TET4/TET10 check had ever exercised. `tensorfem.torsion_bar` carries the closed-form series (torsion constant, Prandtl stress function, shear stress, warping), self-validated against the Timoshenko coefficient table (beta = 0.141, alpha = 0.208), the governing PDE, the traction-free boundary, `J = 2 int psi dA` and the thin-strip limit, plus a driver that twists a prism with an exact end torque and recovers the twist rate from the interior window. HEX20 reaches 0.118% twist-rate error at 4515 dofs and 1.47% at 783 dofs; fully integrated HEX8 needs 8019 dofs for 1.34%. The boundary peak of |tau| is reported as WITHHELD because centroid sampling cannot reach the lateral surface. Evidence in `benchmarks/saint_venant_torsion_evidence.json`, report in `docs/saint-vanant-torsion.md`.

## Curved-bar pure bending verified against the exact Airy solution

* New `tensorfem.curved_beam_reference`: exact three-constant Airy solution for
  pure bending of a rectangular curved bar (neutral axis `r_n = h/ln(r_o/r_i)`,
  eccentricity `e = r_bar - r_n`), its compatible displacement field, consistent
  end-face tractions and symmetry/gauge constraints.
* `benchmarks/curved_beam_pure_bending.py` verifies the production Q4
  plane-stress path on curved, bending-dominated geometry: hoop and radial
  stress at element centres, nodal displacement, rebuilt resultant moment,
  recovered neutral axis and mesh-convergence orders, with ten machine-checked
  anchors. At 32x64 (4290 dofs) the hoop-stress L2 error is 0.087 % and the
  worst point 0.55 %; observed orders are 1.95 / 1.76 / 2.00.
* Documents two findings: the textbook Winkler-Bach formula is an approximation
  (it neglects `sigma_r`, which reaches 13.8 % of `sigma_theta` here, so the
  closed form is 0.70 % off the exact solution at `r_o/r_i = 2`), and the
  discrete field keeps an O(h) parasitic shear (1.7e-3 of the hoop peak at the
  finest mesh) in a problem whose exact shear is identically zero.
* `docs/curved-beam-pure-bending.md` and three figures are generated from the
  computed evidence; `tests/test_curved_beam_pure_bending.py` (11 tests) locks
  the reference's self-validation and the FE accuracy.

- Closed the transient-conduction verification debt: `solve_transient_thermal` was only checked against a semi-discrete single-element formula (`tau = rho cp L^2/(3k)`), which cannot detect a mis-scaled capacity, a wrong element length, or a mislabelled time-integration order. New anchors compare the merged solver with the Carslaw & Jaeger Fourier series (relative L2 1.11e-3 at Fo=0.10, 32 LINE2 elements, 3% gate) and with the single sine mode, giving observed orders 2.00 (space), 0.98 (backward Euler, expected 1) and 2.00 (Crank-Nicolson, expected 2). Cross-checks: Q4 strip reproduces LINE2 to 4.4e-14, adiabatic mean temperature conserved to 1.2e-14, sum(C) equals `rho*cp*A*L` to 1e-15 and `K@1 = 0` exactly. Evidence `benchmarks/thermal_transient_reference_evidence.json`, generator `benchmarks/thermal_transient_reference.py`, write-up `docs/thermal-transient-reference-verification.md`, gate `tests/test_thermal_transient_reference.py` (includes a negative control that rejects a 20% diffusivity error).

## Curved-bar pure bending verified against the exact Airy solution

* New `tensorfem.curved_beam_reference`: exact three-constant Airy solution for
  pure bending of a rectangular curved bar (neutral axis `r_n = h/ln(r_o/r_i)`,
  eccentricity `e = r_bar - r_n`), its compatible displacement field, consistent
  end-face tractions and symmetry/gauge constraints.
* `benchmarks/curved_beam_pure_bending.py` verifies the production Q4
  plane-stress path on curved, bending-dominated geometry: hoop and radial
  stress at element centres, nodal displacement, rebuilt resultant moment,
  recovered neutral axis and mesh-convergence orders, with ten machine-checked
  anchors. At 32x64 (4290 dofs) the hoop-stress L2 error is 0.087 % and the
  worst point 0.55 %; observed orders are 1.95 / 1.76 / 2.00.
* Documents two findings: the textbook Winkler-Bach formula is an approximation
  (it neglects `sigma_r`, which reaches 13.8 % of `sigma_theta` here, so the
  closed form is 0.70 % off the exact solution at `r_o/r_i = 2`), and the
  discrete field keeps an O(h) parasitic shear (1.7e-3 of the hoop peak at the
  finest mesh) in a problem whose exact shear is identically zero.
* `docs/curved-beam-pure-bending.md` and three figures are generated from the
  computed evidence; `tests/test_curved_beam_pure_bending.py` (11 tests) locks
  the reference's self-validation and the FE accuracy.

- Added independent two-dimensional analytical verification of the Q4 heat-conduction
  path: a harmonic field on a Cartesian and on a sheared mesh, a logarithmic radial
  field on a curvilinear annulus, a two-dimensional transient decay and element-centre
  heat-flux recovery, all closed-form. On the finest meshes the relative errors are
  4.85e-4 / 5.56e-4 / 1.55e-4 / 6.02e-4 and the transient rate error 0.144%, each with
  observed second-order convergence, so the previous 3% thermal acceptance rested on a
  linear patch test that could not detect mapping or quadrature defects. Numbers are
  regenerated by `benchmarks/thermal_2d_analytic.py`; the note is
  `docs/thermal-2d-analytic-verification.md` and the 18 fail-closed gates are in
  `tests/test_thermal_2d_analytic.py`.

- Added a finite-width (Howland/Heywood) verification of the same production Q4 plane-stress path, closing the gap recorded by the Kirsch unit: quarter-plate starburst mesh with arc-length boundary nodes (the load/free-edge corner is always a node) and uniform tension on the loaded end. Net-section factors `K_tn` for `d/W = 0.1..0.5` land within 0.04-0.98% of Heywood's fit and 0.01-0.65% of the Peterson/Howland table at 3362 dofs; Richardson extrapolation tightens this to 0.01-0.18%. The `d/W -> 0` limit returns Kirsch's 3 to 0.053% and static equilibrium closes to 2.4e-13. The report also documents the net- vs gross-section convention trap (a factor up to 2 at `d/W = 0.5`). Evidence in `benchmarks/finite_width_hole_reference.json`, write-up in `docs/finite-width-hole-howland.md`. Scope: isotropic thin-plate plane stress only; orthotropic, cracked and interacting holes remain open.

- Added an independent Kirsch circular-hole verification of the production Q4 plane-stress path: closed-form stress/displacement reference validated against equilibrium, traction-free hole, strain compatibility and plane-strain compliance oracles; quarter-annulus case with exact outer-arc tractions. On 578-4802 dofs the nodal displacement L2 error is 3.2e-3 to 3.8e-4, the pointwise hoop-stress error 1.23% to 0.19%, and the stress concentration factor 2.9955 to 3.0030 against the exact 3 (0.15% / 0.10%), so the 3% gate passes at every level. Evidence in `benchmarks/kirsch_hole_reference.json`, write-up in `docs/kirsch-hole-stress-concentration.md`. Scope: 2-D plane stress Q4 only; shell/3-D stress concentration and finite-width (Howland) corrections remain open.

- Add actual CPU/CUDA FVM wall pressure and viscous point-load transfer, separate explicit sampling clocks, SI/ownership/traction validation, exact replay and independent NumPy force/moment/work audits. This is a one-way carrier interface, not flexible FSI qualification.

- Add an atomic macroscopic energy-budget gate for unforced local rigid-fluid feedback. Actual first-node conversions are rejected with complete rollback and exact restart; the moving-boundary physics remains unqualified.

- Add actual LBM motion feedback through a translating Q4 carrier, local boundary and global reference runs, complete restart and independent raw audits. Time/EOS sensitivity (46.34%) and unforced energy growth fail; physical wet icebreaking remains unqualified.

- Add location-preserving point-force transfer and motion return for reference Q4 carriers, with actual calibrated-wet marker-force and exact replay evidence.

- Add recorded LBM–DEM resultant replay on the appended SUBOFF linear shell, midpoint energy/restart checks, and second-round multiphysics acceptance records.

- Added current Q4 surface force/velocity transfer with offset couples, explicit 2-D/3-D frame embedding, ownership checks and a vehicle-only linear hull coupling receiver. Published synthetic sail-load fields, timestep and complete restart audits. Coordinated actual LBM–DEM and independent FVM evidence; full 3-D icebreaking remains unqualified.

- Added welded equivalent SUBOFF sail/fins and configurable ice thickness, open water, precracks, boundaries, buoyancy, added water inertia, drag and conservative fragment diagnostics. Published six actual runs, field/hash audits and four report formats. Time-step sensitivity passes; structural mesh sensitivity remains above 3%, and full breaking-ice physics is unqualified.

- Added an experimental SUBOFF upward ice-impact case using numerical outer geometry exported from TensorLBM, closed quadrilateral pole caps, a declared steel shell/equipment mass, physical rotary inertia, deformable ice tiles, irreversible cohesive seams and dynamic contact. Added raw fields, mesh/time/contact/ice-grid sensitivity runs and energy/impulse audits. Physical breaking-ice accuracy remains unqualified.
- Corrected the Mindlin derivative transformation on skew elements; retained a rigid-tilt regression and replaced invalid hemisphere response baselines.
- Corrected the Cook membrane reference probe to the loaded-edge midpoint.
- Added real-field benchmark reports and Markdown/HTML/PDF/Word exports, with independent reference formulas, coarse failures, mesh convergence and hashed artifacts.
- Added full 3-D HEX8 spherical pressure validation against Lamé elasticity and exact Mindlin plate displacement, moment, shear and bending stress fields.
- Strengthened spherical radial stress and hull-girder fibre stress qualification with pointwise maximum errors, alongside the existing L2 checks. Shell stress oracles for Cook and the hole hemisphere remain incomplete; TensorFEM 1.0 is not qualified.
- Synchronized the public and packaged API manifests for five existing experimental Hertz mesh exports.

## 0.45.0

- Added matching and curved nonmatching finite-strain, double-deformable friction paths with adaptive continuation, positive-Jacobian, Coulomb, dissipation, master/slave interchange and transactional rollback gates. This remains a bounded subset: deformable-to-deformable Hertz is blocked, so the general surface-contact composite is not qualified.
- Separated generation-scheduler committed and attempt heads. A wall-time failure is recorded in the immutable attempt chain without replacing the last valid manifest/checkpoint; readiness follows committed events and fails closed on detached or modified state.
- Added independent hashed nonlinear-Shell4 robustness evidence and a unified v1 evidence aggregator. Release evidence is produced by executed regression/API/offline-install commands with atomically stored logs and verified output hashes rather than hand-authored pass flags.
- Added local panel peak-step refinement and strict 4/8/12 aggregation contracts. The required refined peak/post-peak matrix is still incomplete, so panel convergence and TensorFEM 1.0 remain blocked.
- Preserved qualification boundaries: sparse scalability remains a memory-only claim, deformable Hertz remains blocked, and bounded contact subsets are not promoted to general contact.

## 0.44.0

- Added a consistent two-sided frictional Mortar assembly and a transactional curved, nonmatching stick-slip path with force, moment, objectivity, tangent and rollback gates. The slow curved two-pass qualification remains fail-closed, and general surface-to-surface contact remains blocked.
- Added a hash-verified 10,882-active-DOF Shell4 sparse release report. ILU-GMRES retains sub-1% accuracy and reduces storage 16.9% versus SuperLU and 90.7% versus dense, but remains slower; the machine-readable claim is therefore `memory_only`.
- Added explicit legacy sparse-cache migration by full tangent reassembly, coefficient/action comparison and atomic SHA-256 publication. Unknown or modified caches and qualification reports fail closed before promotion.
- Added generation-scheduled panel supervision and readiness evidence with event-chain, checkpoint and status integrity. Refined panel paths remain in progress, so no peak, post-peak convergence or TensorFEM 1.0 claim is made.
- Preserved the v1 boundary: the curved slow contact suite has not qualified, general contact remains blocked, and the 4/8/12 panel matrix is still generating.

## 0.43.0

- Added a SHA-chained, one-generation-at-a-time panel scheduler with per-point wall limits and stop-on-failure semantics. The migrated 8x8 path advances from point 659 to 680 at 1.0111 MN and 2.03125% yielded integration points; 12x12 advances from 25 to 30 at 414.48 kN. Both remain monotone, so no peak is claimed.
- Added a conservative analytic elastic plane-stress material tangent with automatic fallback near yield and in plasticity. Shell4 assembly improves by 31-34% and a real 8x8 first point by 27.9%, while an 80-point fully yielded path retains load, displacement, energy and plastic history to roundoff.
- Added curved, nonmatching, symmetric two-pass Mortar infrastructure and a default fail-closed smoke gate. Independent 4/5 one-pass brackets are within 1.75% of a Winkler paraboloid oracle, but their raw role bias exceeds 3%; the expensive full two-pass qualification remains opt-in and is not yet promoted.
- Kept TensorFEM 1.0 blocked: the 4/8/12 panel paths still lack peaks/post-peak convergence, and the curved two-pass contact suite has not completed every interchange and objectivity gate.

## 0.42.0

- Added immutable, hash-audited panel solver-strategy migration. The 8x8 generation-658 mechanical state and energy ledger migrate exactly from fixed Newton to a distinct backtracking identity, after which point 659 accepts in three Newton iterations at 969.736 kN with equilibrium and energy gates passing.
- Extended the two-compliant-surface contact closure to nonmatching 1/2, 2/3 and 3/4 QUAD4 meshes. Fine-grid oracle error reaches 0.1515% and a complete swapped-discretization solve differs by 0.564%, while curved, frictional and production segmentation contact remain blocked.
- Qualified Shell4 sparse storage beyond the 1.0 contract threshold: a 42x42 model has 10,882 active DOFs, 1.33e-6 solution error and 1.34e-10 true residual, using 87.65 MB versus 105.54 MB SuperLU and 947.34 MB dense.
- Recorded and rejected a frozen-tangent optimization after it changed a real 8x8 first-point load by about 49%. Assembly remains 98.2% of the measured 42x42 ILU path, so runtime acceleration remains open despite the storage qualification.

## 0.41.0

- Integrated opt-in backtracking into the transactional finite-rotation layered Shell4 arc path and panel checkpoint identity. A yielding Shell4 step rejected by fixed Newton converges in 13 iterations and remains within 1% of a strict reference without leaking trial material history.
- Added a multi-integration-point, two-compliant-surface Newton closure. Matching QUAD4 contact errors converge from 2.436% to 0.360% with force/moment balance, objectivity, master/slave interchange and rollback gates; curved, nonmatching and frictional contact remain unqualified.
- Added native SciPy ILU-GMRES and auditable preconditioner refresh. The 20x20, 2544-DOF gate stays below 1% error and uses 9.286 MB versus 12.652 MB for SuperLU and 51.775 MB dense, but no runtime crossover is observed.
- Added a hashed, fail-closed TensorFEM 1.0 readiness contract. It prevents package health or bounded prototypes from being promoted to 1.0 before general contact, 10k-DOF Shell4 sparse evidence and explicit 4/8/12 peak/post-peak convergence are present.

## 0.40.0

- Added opt-in backtracking line search to arc-length continuation with fail-closed merit descent and checkpoint identity. A difficult public von Mises arch step is accepted instead of rejected, matches a high-iteration reference below 1e-12 and retains the 0.0255% analytic peak error.
- Added a two-compliant-body 3-D contact Newton closure. Its reaction matches an independent series-compliance oracle to machine precision and passes force, moment, objectivity, master/slave interchange and failed-step rollback gates, while general curved, frictional and double-sided mortar contact remain unqualified.
- Added a fully sparse ILU-GMRES route for Shell4 Newton and bordered arc-length solves. A 12x12 gate remains below 1% error while reducing matrix-plus-factor storage by 22.3% versus SuperLU and 67.3% versus dense; measured runtime remains slightly slower, so no speedup is claimed.
- Preserved the recoverable 8x8 panel qualification checkpoint while its first post-yield continuation point remains computationally difficult; no peak or mesh-convergence claim is inferred from an in-progress solve.

## 0.39.0

- Qualified dense/SuperLU equivalence through an 80-point finite-rotation Shell4 path with all integration points yielded. Load, displacement, recoverable/hardening energy and plastic-history differences are around 1e-15; the measured small case remains slower and uses more peak RSS with SuperLU, so no acceleration claim is made.
- Qualified opt-in automatic step control on a 20-transaction real panel path containing four actual step changes. Interpolated force, displacement, external-work and internal-energy differences remain below 0.123%, interrupted restart is exact, and Newton iterations fall from 66 to 65 without a rejected-step improvement.
- Added reduced-order lateral-torsional buckling and sharp circular-pit screening. The independent spatial/oracle errors converge to 0.0518% and 0.139%, while coupled plate-stiffener collapse, weld failure, cracks and postbuckling remain explicitly excluded.
- Extended the corrected-metric 12x12 panel path from 20 to 25 accepted points and 369.285 kN with equilibrium and energy gates passing. It remains elastic and does not establish an ultimate-load peak; the longer 8x8 continuation remains a recoverable background qualification.

## 0.38.0

- Added an explicit SuperLU route to the finite-rotation Shell4 increment and full arc-length path. Real 2x2 and 4x4 prefixes match dense loads and displacements far below 1%, with auditable sparse factorization diagnostics and no speedup claim.
- Added opt-in, bounded automatic step-size control with deterministic decision hashes and checkpoint identity. Fixed and automatic short arch/panel paths agree, and interrupted/resumed execution matches a fresh run; Newton, line search and arc-metric switching remain disabled.
- Added reduced-order marine qualifications for individual compatible line stiffeners and a smooth nonuniform-corrosion field. Both have independent analytical Ritz oracles and executable 3% gates, with tripping, weld failure, isolated pitting and ultimate collapse explicitly excluded.
- Advanced the corrected-metric 4x4 panel path to 600 monotone points and 1.0095 MN, and the 8x8 path to first yield at point 638 and 967.73 kN at point 658. Neither establishes a peak, so the former 509 kN coarse result is retained only as invalidated legacy evidence and mesh convergence remains fail-closed.

## 0.37.0

- Added a deterministic advisory-only nonlinear controller that classifies accepted chunks from Newton contraction, rejected attempts, path curvature, equilibrium and energy evidence; restart verifies a SHA-256 decision chain and no unqualified algorithm switch is activated.
- Added a unified optional sparse linear-solver adapter with factor reuse, multi-RHS support and fill/time/memory/residual diagnostics. Existing SuperLU passes real 4x4/8x8 linear accuracy checks but does not yet qualify full nonlinear acceleration.
- Added executable marine screening qualifications for smeared longitudinal stiffening, uniform corrosion thickness sensitivity and self-equilibrated residual-stress first yield, with errors of 0.0755%, machine precision and 0.0110% respectively.
- Continued the dimensionally consistent 8x8 panel path beyond 800 kN while retaining equilibrium and energy gates; its continued monotone response invalidates transfer of the former coarse 4x4 peak and keeps mesh convergence fail-closed.
- Started a fresh dimensionally consistent 4x4 path so future 4/8/12 comparisons use the same continuation metric and checkpoint semantics.

## 0.36.0

- Replaced the dimensionally invalid shell continuation norm with an opt-in thickness-integrated metric: bending rotations use `t^2/12`, drilling does not consume arc length, and all DOFs remain in equilibrium.
- Diagnosed the 8x8 480.2 kN plateau as drilling-gauge drift (99.9934% of the old increment), then advanced the corrected path to 580.889 kN at point 340 with equilibrium and energy gates passing; the path remains monotone, so no peak is claimed.
- Added immutable generation checkpoints and atomic manifest pointers, with fault injection proving that interruption between binary publication and manifest commit preserves the prior valid generation.
- Added hash-verified controlled metric migration and retained provenance for the 8x8 and 12x12 paths; the corrected 12x12 prefix reached 20 points and 309.197 kN.
- Added an optional, fail-closed SciPy SuperLU/ILU adapter without a hard dependency. Real 4x4/8x8 corrector comparisons qualify accuracy but not speed, so production sparse acceleration remains unclaimed.
- Preserved the fail-closed 4/8/12 convergence status: refined meshes have not reached their peaks and currently contradict the former assumption that the 4x4 peak transfers directly.

## 0.35.0

- Fixed the panel stored-energy observer to use the same projected facet basis as the Shell4 internal force. The 4x4 coarse/fine terminal residuals are now 0.001526/0.0000611 J, both below the 0.025 J gate, with the expected 24.98x second-order reduction under fivefold step refinement.
- Versioned the energy definition in checkpoint identities so an older non-conjugate energy ledger cannot be silently resumed.
- Completed the 4x4 fine path to 400 accepted points, preserving the 508.999 kN elastic geometric peak and extending the snap-back evidence to 66 post-peak points.
- Generalized peak-window scheduling to even refined meshes, added a fail-closed 4/8/12 convergence evaluator, and executed a real 12x12 first point with equilibrium and energy gates passing.
- Advanced the corrected 8x8 path to ten recoverable points and fixed stale-manifest replay after checkpoint-identity migrations.
- Recorded negative sparse-solver qualification evidence: current GMRES preconditioners do not reliably complete the nonlinear corrector, so no production sparse speedup is claimed.

## 0.34.0

- Added connectivity-graph sparse Shell4 tangent assembly with dense-equivalent force/action checks and a measured 4x4 storage gate; this establishes the storage/assembly foundation, not production-scale distributed solving.
- Corrected the augmented arc-length predictor so displacement and load coordinates use one normalized metric, with regression coverage for non-unit load scales and branch direction.
- Added a resumable 8x8 peak-window scheduler with hashed one-point checkpoints and conservative coarse/approach/peak step selection; a real three-point 8x8 window has executed, but that mesh has not yet reached a peak or descending branch.
- Confirmed a 4x4 elastic geometric peak and accepted post-peak points across normalized arc steps 0.10 and 0.02. The fine path peaks at 508.999 kN (point 334) and ends at 507.570 kN after 355 accepted points, a 0.28072% drop; its peak differs from the 508.686 kN coarse result by 0.06156%. The run stopped at its 240 s wall limit rather than solver failure, with two rejected late attempts, zero yielded material and continuous accepted history. Qualification remains blocked because the 360-point target was not completed, the terminal absolute energy residual is 0.107968 J and cross-mesh convergence is unverified.
- Added cross-chunk accepted-state energy ledgers and a dimensionful absolute/relative energy gate while retaining raw near-zero relative residuals for audit.
- Kept marine-panel peak/post-peak strength qualification, complete 4/8/12 mesh-control convergence and production sparse throughput explicitly unqualified.

## 0.33.0

- Added branch-preserving, atomically hashed long-path checkpoints whose target length and persistence cadence can change without recomputing a validated prefix.
- Added accepted-state panel energy evidence covering recoverable and hardening energy, plastic dissipation, external work, balance, yielded volume and conservative failure-mode classification.
- Optimized force-only corotational mapping and large-system linear solves; 8x8 and 12x12 first points now complete in 48.36 and 106.96 seconds.
- Traced a real 2x2 panel through 220 accepted points to 2.295 MN and 100% yielded material. The path remains monotone, so no peak/post-peak qualification is claimed.
- Kept incomplete legacy energy prefixes explicitly marked and excluded from energy qualification.

## 0.32.0

- Replaced repeated finite-rotation material reintegration with a chain-rule tangent containing the transformed local algorithmic tangent and corotational geometric Hessian.
- Added transactional initial residual-stress equilibration, a scaled load coordinate and dimensionally consistent continuation convergence checks.
- Added auditable 2x2/4x4 marine-panel first-point gates; accepted points complete in 3.26/12.89 seconds and short three-point pre-peak paths converge with near-machine balance.
- Added a matrix-free Crisfield/GMRES continuation option with fail-fast initial-equilibrium checks, frozen-tangent and block/Jacobi preconditioning paths, rollback and detailed termination statistics.
- Kept marine-panel peak, plastic post-peak and 4/8/12 convergence qualification explicitly blocked pending long-path execution.

## 0.31.0

- Added guarded best-fit midsurface projection for shallow-warp layered Shell4 facets, fixing stress-free sinusoidal panel imperfections while rejecting deeply folded elements.
- Accelerated finite-rotation layered-shell evaluation by using an AD kinematic Jacobian and skipping nested material tangents during outer residual differentiation, with identical force and tangent evidence.
- Added a dimensionally normalized, resumable marine-panel qualification executor; real 2x2 and 4x4 trials expose the remaining first-step tangent performance blocker instead of overstating peak/post-peak capability.
- Added an energy-consistent two-sided deformable 3-D facet-contact residual and exact active-set tangent with objectivity and force/moment audits.
- Added strict fail-closed three-dimensional deformable Hertz qualification gates; complete curved-solid convergence evidence remains blocked.

## 0.30.0

- Added a finite-rotation corotational Shell4 wrapper with layered plane-stress J2 plasticity, stress-free imperfections, residual stress, numerical algorithmic tangent and transactional Crisfield continuation.
- Expanded the deformable axisymmetric Hertz qualification to pressure-field L2 error, force balance, penalty overlap, three contact-zone meshes and a half-space domain-size study; all declared fine-grid errors remain below 3%.
- Added a real multi-element marine-panel ultimate-strength input and acceptance contract while keeping peak/post-peak qualification blocked until its complete mesh/control matrix is executed.
- Added a hashed industrial P0 phase-3 report that separates qualified integration primitives and axisymmetric Hertz evidence from unqualified panel ultimate strength and general 3-D deformable-to-deformable contact.

## 0.29.0

- Added a real small-strain layered Shell4 with 2x2 in-plane integration, through-thickness plane-stress J2 points, residual stress, unloading and transactional material history.
- Added a global low-order 3-D node-to-TRI3 contact solve with current-configuration search, barycentric master reactions, Coulomb history, objective motion and force/moment audits.
- Added machine-readable external benchmark contracts: Bathe-Bolourchi large-rotation bending is qualified while shallow-shell snap-through and deformable Hertz FE remain blocked.
- Added a hashed industrial P0 phase-2 report preserving explicit boundaries around finite-rotation plastic postbuckling and general mortar/self-contact.

## 0.28.0

- Added a unified ModelDB-to-AnalysisPlan-to-Job-to-ResultDB v2 vertical slice with strict schema migration, registered kernels, deterministic IDs, atomic failure records and resume.
- Added a real multi-facet geometrically nonlinear Shell4 continuation path with measured reference imperfections, committed generalized section state and rejected-step rollback.
- Added a transactional 2-D global finite-sliding contact solve with deformable-master search, objective Coulomb history, balanced residual, algorithmic tangent and rollback.
- Added a hashed industrial P0 qualification report that keeps shell integration-point plasticity, general 3-D mortar/self-contact and distributed production solving explicitly unqualified.

## 0.27.0

- Connected the Crisfield continuation solver to an assembled corotational Shell4 residual and consistent autograd tangent, with discrete-equilibrium and step-size gates.
- Added direct numerical Mode-I contour J integration with plane stress/strain, path-independence and refinement evidence.
- Added fail-closed ResultDB v1/v2 import of imperfection and residual-stress fields and complete-field reduced strip response paths.
- Removed the undeclared NumPy dependency from the imperfect plastic strip and made ResultDB checksums work with the declared PyTorch-only core.
- Expanded the formal benchmark registry to 55 passing nonzero reference quantities while retaining explicit postbuckling and crack-growth boundaries.

## 0.26.0

- Added a real Shell4 initial-stress geometric matrix and eigenbuckling qualification.
- Added a reduced imperfection/residual-stress/plastic strip convergence and energy gate.
- Added ResultDB field-path, SCL, hot-spot, Mode-I K/J, and toughness post-processing gates.
- Expanded the formal benchmark registry to 53 passing reference quantities.

## 0.25.0

- Added uniaxial/biaxial prestress buckling and imperfect reduced-order postbuckling qualification.
- Added multi-component hull-girder progressive-yielding equilibrium and event auditing.
- Added path hot-spot extraction, rainflow counting, spectrum Miner damage, and Paris crack growth.
- Expanded the formal benchmark registry to 45 passing reference quantities.

## 0.24.0

- Added local isotropic and smeared-stiffener plate-buckling convergence qualifications.
- Added a section-fiber hull-girder yielding and full-plastic-moment benchmark.
- Added weld hot-spot, S-N, thickness-correction, and Miner fatigue assessment primitives.
- Expanded the formal registry to 37 benchmarks while keeping TensorLBM out of scope.

## 0.23.0

- Added a TensorFEM-only marine structural qualification report with no TensorLBM or CFD dependency.
- Combined 15 scalar references with four DOI-traceable classical shell cases.
- Made local plate buckling, ultimate hull strength, and fatigue explicit unqualified gaps.

## 0.22.0

- Promoted eight marine reference quantities into the fail-closed formal benchmark registry.
- Added hull-girder mesh/reaction and Morison quadrature-convergence qualification gates.
- Added an explicit marine validation matrix and documented unqualified capability boundaries.

## 0.21.0

- Added verified hull-girder and equivalent-orthotropic stiffened-panel screening cases.
- Added rectangular-barge hydrostatics, load cases, and intact small-angle stability checks.
- Added Airy/Morison marine load generation and an audited TensorLBM-to-TensorFEM force-history contract.

## 0.20.0

- Expanded safe mesh projects with HEX8 structural and LINE2/Q4 steady thermal elements.
- Added three complete engineering case packages with reproducible deliverables.
- Froze and audited the candidate v1 public API and offline release artifacts.

## 0.19.0

- Added a versioned public-API manifest and drift gate without removing existing exports.
- Added staged deprecation primitives and classified beta, experimental, and internal-candidate exports.
- Added offline wheel/sdist build and clean-environment installation smoke checks.

Earlier releases predate the formal changelog; their history remains available in Git.

## 2026-11-03 (m2430)

- 修复审计模块 `cook_stress_reference` 的单元算子缺陷：物理梯度漏用逆转置 Jacobian（Q8/Q4）、
  Q4 父坐标导数表错项且 measure 偏大 16 倍。常应变能量恒等式由 +747% 回到 1.0e-13，审计 Q8
  与生产 Q4 的 Cook 膜加载边型线由相差 6.8 倍收敛到 <0.3%，m2125 留下的分歧闭环。
- 新增 `tests/test_cook_oracle_operator.py`（10 项，含约定无关 patch/能量/FD 判据与 fail-closed 断言）。
- 生产 Q4 路径未改动；Cook 膜逐点应力仍为 5.02%（n=64）> 3%，`stress_accuracy_qualified` 保持 False。
## 2026-11-03 波浪谱与短期统计（耐波链条补全 + 双重验证）
- 新增 `tensorfem.wave_spectrum`：Pierson-Moskowitz 与 JONSWAP 谱（均以 `Hs = 4√m0` 归一化）、
  谱矩 m0/m1/m2（`t = ln(ω/ωp)` 换元 + 嵌套复合 Gauss-Legendre，收敛层自证）、
  `Hs/Tm/Tz/Tc`、Rayleigh 超越概率、精确与渐近两种短期极值、`irfft` 时域记录生成，
  以及 `equivalent_design_wave` 把谱接进已有 `AiryWave`/Morison 载荷通道。
  此前库里只能给规则波，海况无法表达。
- 验证锚点见 `docs/marine-wave-spectrum-verification.md`（数字由 `scripts/build_wave_spectrum_doc.py`
  从 `artifacts/wave_spectrum_verification/validation_report.json` 渲染，零手抄）：
  PM 谱矩 vs 解析闭式解最差 3.3e-16；JONSWAP(γ=1) 与 PM 逐点为 0 差；α vs 文献
  `5.061(Hs/Tp²)²(1-0.287 ln γ)` 最差 0.285%；`P(H>Hs) = exp(-2)`；时域 Tz 与全谱 `2π√(m0/m2)`
  差 3~4%（限带效应），与同限带谱差 1.7~2.2%。
- 如实记录的失败项：单条记录最大波高 vs Rayleigh `H_mp` 均值低 5%，单种子散度 -15.8%~+11.2%，
  超 3% 门限；窄带/Rayleigh 对极值只是保守上界，不作认证结论。PM 的 m4 发散使 `Tc` 无定义，库中返回 `None`。
- 新增 `tests/test_wave_spectrum.py`（14 项，含 fail-closed 反例：非法 Hs/Tp/γ、非 2 的幂
  `n_samples`、`m4` 发散、非法 `period`）。
