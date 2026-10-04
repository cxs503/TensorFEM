# Changelog

All notable user-visible changes are recorded here.

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
