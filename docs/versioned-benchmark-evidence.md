# Versioned benchmark cases and traceable evidence

## Schema

`benchmark_cases.py` complements the existing scalar `benchmark_registry`
without changing its public entry point. A versioned case must record:

- stable case ID, semantic case version and schema version;
- authors, title, publication, year and DOI;
- geometry, material, boundary conditions and loads;
- ordered mesh sequence;
- reference quantity, physical unit and sign policy;
- relative-error strategy and strict tolerance no larger than 3%;
- maturity and explicit capability boundaries;
- an allow-listed executable runner and quick recorded values;
- the meshes that qualify the advertised capability.

Missing DOI, physical definition, unit, boundaries, runner, mesh values or a
valid tolerance fails closed. `to_legacy_evidence` adapts one qualifying row to
the existing `BenchmarkEvidence` object.

## Evidence integrity and tiers

Evidence JSON uses schema `tensorfem.benchmark-evidence/2.0`. Canonical JSON
SHA-256 hashes protect both the complete case definition and result archive.
Reading a modified value without resealing the archive fails. Archives contain
no volatile timestamp, so identical inputs produce identical hashes.

- `quick`: validates schema/hashes and checks recorded, previously reproduced
  values. This is the default CI tier.
- `full`: reruns every mesh through `hemisphere_with_hole`. It is enabled in
  tests with `TENSORFEM_FULL_BENCHMARKS=1` and is intended for release/nightly
  qualification.

The complete 4/6/8/12/16/20/24 run remains opt-in so additional future expensive cases do not accumulate
in the ordinary test gate.

## 18-degree-hole hemisphere evidence

Source: MacNeal and Harder, *Finite Elements in Analysis and Design* 1 (1985),
3--20, DOI `10.1016/0168-874X(85)90003-4`. The reference loaded-equator radial
displacement magnitude is `0.0924`.

Case version 2.0.0 supersedes the old skew-gradient numerical baseline.
Meshes 4/6/8/12/16/20/24 give errors 9.938%, 6.859%, 5.200%, 3.682%,
3.040%, 2.693%, 2.473%. Only 20 and 24 are formal default-factor response
qualification meshes. All coarse failures remain in the archive. Parameter
robustness and complete FE fields are separate from this scalar archive;
see [the complete report](benchmarks/tutorials/hemisphere_hole_full.md).

## Drift comparison

`compare_archives` first verifies both hashes and case IDs, then compares every
common mesh. Drift is relative to the baseline computed value and has its own
strict threshold up to 3%. The report records baseline/current case versions,
per-mesh values, drift and pass flag. It detects numerical regression even
when both versions still happen to lie within the literature tolerance.

The case boundary remains a linear, small-strain, quarter-symmetry projected
MITC-like Q4 shell; it does not qualify general nonlinear doubly-curved shells.

```bash
PYTHONWARNINGS=error PYTHONPATH=src python -m pytest -q tests/test_benchmark_cases.py
TENSORFEM_FULL_BENCHMARKS=1 PYTHONWARNINGS=error PYTHONPATH=src \
  python -m pytest -q tests/test_benchmark_cases.py
```
