# External P0 benchmark evidence contracts

This evidence layer records two independently identifiable external cases.  It
distinguishes a passed large-rotation case from two blocked capability gaps.
The machine contract is implemented in
`tensorfem.external_benchmark_contracts`; `require_qualified()` fails whenever
required evidence is absent.

## Qualified nonlinear shell: large-rotation pure bending

The complete runnable case is associated with K.-J. Bathe and S. Bolourchi,
“Large displacement analysis of three-dimensional beam structures,”
*International Journal for Numerical Methods in Engineering* 14 (1979),
961–986, [DOI 10.1002/nme.1620140703](https://doi.org/10.1002/nme.1620140703).
Its DOI metadata was checked on 2026-10-03.  The contract stores only citation
facts and an independently derived circular-elastica oracle.

The `10 × 1 × 0.1` strip has `E=1.2e6`, `nu=0`, a clamped root and a prescribed
90-degree end rotation.  The exact circular solution gives
`x_tip=z_tip=20/pi=6.366197724` and `M=15.707963268`.  The existing 1-, 2-, and
4-element sequence converges monotonically; its 4-element vector tip error is
0.645% and moment error is below the 3% gate.  The contract links directly to
the solver, test and validation document.  This qualifies large-rotation,
linear-elastic, displacement-controlled pure bending only—it is not evidence
of limit-point or post-buckling capability.

## Nonlinear shell: clamped shallow spherical shell under ring load

The source is E. E. Karataş and R. F. Yükseler, “Snap-through Buckling of
Shallow Spherical Shells under Ring Loads,” *Teknik Dergi* 32(2), 2021,
10695–10716, [DOI 10.18400/tekderg.565095](https://doi.org/10.18400/tekderg.565095).
The publisher article metadata and PDF were checked on 2026-10-03.

The paper identifies the experimental-comparison geometry (`R=254 mm`,
`t=0.3810 mm`, `eta=0.0618`, ring diameter `25.4 mm`) and a clamped edge.
The limit coordinates are figures rather than a numerical table, however, and
the complete material constants for the selected curve are not available as a
machine-auditable dataset.  TensorFEM also lacks the configuration-dependent
ring-load residual and consistent tangent.  Pixel digitisation is explicitly
excluded from the `<3%` release gate, so the contract remains **blocked**.

## 3-D contact: Hertz sphere on elastic half-space

The primary source is H. Hertz, “Ueber die Berührung fester elastischer
Körper,” *Journal für die reine und angewandte Mathematik* 92, 1882, 156–171,
[DOI 10.1515/crll.1882.92.156](https://doi.org/10.1515/crll.1882.92.156).
Its bibliographic metadata was checked through the DOI record on 2026-10-03.
Only bibliographic facts and independently evaluated classical equations are
stored; no publisher scan or typesetting is redistributed.

The contract fixes a reproducible SI case: `F=1000 N`, `R=0.01 m`, and both
bodies have `E=210 GPa`, `nu=0.3`.  Its analytical values are
`E*=115.3846153846 GPa`, `a=0.402072576 mm`,
`delta=0.016166236 mm`, and `p0=2.953469443 GPa`.

These values qualify the analytical oracle only.  Current TensorFEM contact
helpers do not provide a global deformable sphere/half-space FE solution with
pressure integration, domain-size convergence and contact-zone mesh
convergence.  The 3-D contact contract therefore also remains **blocked**.

## Promotion gate

A case may change to `qualified` only when:

1. all contract fields and an unambiguous external oracle are present;
2. every `missing_evidence` item is closed;
3. result files identify solver version, mesh, increments and units;
4. at least three meshes (and, where relevant, two continuation step sizes or
   domain sizes) demonstrate convergence; and
5. each registered reference quantity has relative error strictly below 3%.

Until then the contracts are useful backlog and provenance records, not
entries in the supported benchmark registry.
