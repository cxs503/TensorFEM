# Changelog

All notable user-visible changes are recorded here.

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
