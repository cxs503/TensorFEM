# Changelog

All notable user-visible changes are recorded here.

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
