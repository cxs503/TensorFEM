# Changelog

All notable user-visible changes are recorded here.

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
