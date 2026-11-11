# j2_plastic_membrane

Plane-stress J2 (von Mises) elastoplastic membrane qualification for
`tensorfem.j2_plane_stress` — the plane-stress half of the repository's J2 family
(the 3-D kernel in `tensorfem.plasticity` is plane-strain by construction).

## What is verified

| id | check | threshold | anchor |
|---|---|---|---|
| C1 | uniaxial stress vs closed form (incl. plastic slope `E H/(E+H)`) | 1e-10 | analytic card |
| C2 | `sigma_zz = 0` after the plane-stress return | 1e-9 | plane-stress reduction |
| C3 | plastic points on `q = sigma_y + H alpha` | 1e-8 | consistency condition |
| C4 | batched kernel vs `plasticity.update_j2` (independent code path) | 1e-7 | single source of truth |
| C5 | Cook membrane elastic deflection vs 23.96 | 1 % | Simo & Rifai (1990) DOI:10.1002/nme.1620290802 |
| C6 | `W_ext = U + W_p` on the assembled elastoplastic Cook path | 1e-3 | exact hardening integral |
| C7 | incremental plastic work vs closed form `sigma_y a + H a^2/2` | 1e-12 | exact integral |
| C8 | Cook elastoplastic mesh independence (n=24 vs n=32) | 5 % | h-refinement |
| C9 | notched specimen inside the plane-stress limit-analysis bound | `P <= 2/sqrt(3)(sigma_y + H alpha_max) t w` | limit analysis |
| C10 | ABAQUS literature baseline < 10 % | — | **WITHHELD** (no offline ABAQUS dataset) |

## Reproduce

```bash
python benchmarks/j2_plastic_membrane/run.py           # full run (~7 min)
python benchmarks/j2_plastic_membrane/run.py --quick   # smoke (~30 s, same criteria)
python -m pytest tests/test_j2_plane_stress.py -q      # 10 invariants (~2 min)
```

Outputs: `artifacts/j2_plastic_membrane/report.json` (all numbers + criteria),
`fig1..fig5*.png`, and `docs/j2-plastic-membrane.md` rendered from the JSON.

## Honest limitation

C10 is withheld: the ABAQUS verification dataset is not reachable from this
worktree and no published elastoplastic Cook / notch table is quoted here that
could be verified. Round-2 input is recorded in `report.json` under
`failure_reason`.