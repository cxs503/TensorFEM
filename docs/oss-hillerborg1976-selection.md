# OSS / method survey and selection -- REPRO-FEM-HILLERBORG1976

Second-round requirement: survey mature methods first, sediment the selection as reusable
knowledge, then implement. This file is the sediment (the platform tables `oss_inventory` /
`oss_solutions` / `negative_knowledge_ledger` are **not writable by this tenant**:
`psql` reports `permission denied for table oss_inventory|oss_solutions|negative_knowledge_ledger`
and the REST surface has no `/oss/*` route (404). The row that would have been inserted is kept
below verbatim in JSON so a privileged run can load it.)

## 0. Retrieval reach (honest)

| surface | attempted | result |
|---|---|---|
| `SELECT * FROM oss_inventory WHERE feature_slug ILIKE '%HILLERBORG1976%'` | psql | permission denied |
| cluster `~/oss-cache` | `ls ~/oss-cache` | only LBM repos locally (`Maarten-vd-Sande_lbm`, `hschwane`, `pylbm`); cluster listing job queued behind 300+ jobs |
| `GET /api/v1/oss/*` | REST | 404 (no such route in the platform OpenAPI) |
| `negative_knowledge` search | `GET /api/v1/tensor/negative-knowledge/search?q=TensorFEM` | 0 hits |
| prior round | cross-audit `docs/board_audit/m15268/AUDIT_m15144.md` | read; its three findings drive this round |

## 1. Method genealogy (what mature codes do)

| # | method | key points | standard checks | anchor |
|---|---|---|---|---|
| M1 | Fictitious/cohesive crack with **influence functions** (this round) | bulk linear elastic, unknowns = cohesive tractions only, `w = w^P - C sigma` | LEFM asymptote, energy balance, a/W series at no re-meshing cost | Hillerborg 1976 (this anchor) |
| M2 | Zero-thickness cohesive interface elements + global Newton | penalty stiffness `k0`, tangent at the peak is singular defensively | patch test, objectivity | Abaqus Theory 4.5.5; repo `cohesive3d.py`, `cohesive_bending.py` |
| M3 | LEFM handbook calibration | `K_I = PS/(BW^1.5) f(a/W)` (Srawley-Gross) | shape factor limits | Tada handbook case 2.16; ASTM E399 |
| M4 | Size-effect / brittleness number | `l_ch = E G_F/f_t^2`, `W/l_ch` controls the brittle-ductile transition | collapse of the dimensionless peak load | Hillerborg 1976; Bazant size effect |
| M5 | Work-of-fracture (RILEM) back-calculation | `G_F = W_frac/(B(W-a0))` | energy closure | RILEM 50-FMC |

## 2. Selection and why not the others

* **Selected M1 + M3 + M4 + M5**, implemented as `tensorfem.fictitious_crack`. M1 removes the two
  failure modes of round 1 at once: the notch becomes a mask on the influence system (so the `a/W`
  series exists at all) and the nonlinearity lives in a small dense system that is non-singular at
  the peak (so CMOD control cannot stall).
* Not M2 alone: the repo already has it (`cohesive_bending`, `cohesive3d`), and the round-1 audit
  showed that a load-controlled global Newton on it stalls for ductile concrete; keeping it as a
  cross-check would need the notch parameter it does not have.
* Not phase field: different theory, different energy functional, out of scope for a 1976
  reproduction.
* Not a pure LEFM re-derivation: M3 is used only as an *independent* reference, never as the model.

## 3. Migration plan (as executed)

1. `src/tensorfem/fictitious_crack.py`: laws + influence model + CMOD control + LEFM helpers.
2. `benchmarks/hillerborg1976_repro.py`: size series, a/W series, collocation, penalty and
   law-shape sensitivity, machine-checked criteria -> `artifacts/hillerborg1976/repro.json`.
3. `tests/test_fictitious_crack.py`: 15 invariants (no stored-output comparisons).
4. `docs/repro-hillerborg1976.md`: six-section report generated from the JSON.

## 4. Sediment row for `oss_solutions` (blocked by permission)

```json
{
 "repo": "TensorFEM",
 "feature_slug": "REPRO-FEM-HILLERBORG1976",
 "tag": "fictitious-crack-influence-function",
 "method": "Hillerborg 1976 fictitious crack model solved by FE influence functions with CMOD-controlled secant iteration + safeguarded Newton polish",
 "files": "src/tensorfem/fictitious_crack.py, benchmarks/hillerborg1976_repro.py, tests/test_fictitious_crack.py",
 "conventions": "mode-I only, plane stress, symmetry half model, penalty stiffness k0 = factor*E/W, crack path pre-cut to 0.9W",
 "transfer": "any mode-I cohesive-crack problem whose crack path is known: replace the geometry builder, keep the influence/mouth-control machinery",
 "evidence": "artifacts/hillerborg1976/repro.json; peak-load drift <= 1.32% over 2x collocation; energy balance <= 2.93e+33"
}
```

