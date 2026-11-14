# MacNeal (1985) Slender-Beam Shear-Locking — Verification Anchor

## Physics

A simply-supported beam of length `L = 10` and unit square cross-section
(`h = 1`, unit width) under a central point load `P = 1`, with `E = 1`,
`ν = 0.3` (slenderness `L/h = 10`).  The beam is discretised as a two-
dimensional strip of four-node quadrilaterals (`n` elements along the axis,
one through the thickness).  This is the slender-beam test of

> MacNeal, R. H. (1985). "A simple quadrilateral, 4-node shell element with
> enhanced interpolation." *Finite Elements in Analysis and Design* 5(2):113-118.

The case probes **transverse-shear locking**: a low-order element that
interpolates `w` and the rotation with the same order and integrates the shear
at full order cannot represent the vanishing shear strain of the thin limit, so
it stiffens and underpredicts the deflection.

## Oracle

| Quantity | Value | Source |
|---|---|---|
| Centre deflection (Euler-Bernoulli) | 250.0000 | closed form `P L³/(48 E I)`, `I = b h³/12` — the MacNeal yardstick |
| Centre deflection (Timoshenko) | 257.8000 | closed form `P L³/(48 E I) + P L/(4 k A G)`, `k = 5/6` — exact Mindlin at `L/h = 10` |

Both references are **evaluated in `run.py`** (zero hand-copying); the paper's
own tabulated numbers are not available in this environment, so grading is
against the closed forms the paper uses as its yardstick.

## Element

**Q4 Mindlin/Reissner** (`tensorfem.plate.q4_mindlin_stiffness`), run as a 2-D
strip (`θ_y = 0`).  Bending rigidity set to the *beam* value `E h³/12` (unit
width, no `1-ν²` plate factor); shear rigidity `k G h`, `k = 5/6`.  Three shear
blocks are compared (same 2×2 bending rule in all):

* `full` — shear at the full 2×2 rule (equal-order, **locking** case; added for
  this benchmark);
* `sri` — one-point selective reduced integration (locking-free);
* `mitc` — assumed natural (Dvorkin-Bathe) shear field (locking-free).

## Mesh Ladder

`n = 2, 4, 8, 16` elements along the axis (all even, so `x = L/2` is a node).

## Results (finest level, `n = 16`)

| Treatment | δ (got) | err vs Timoshenko | err vs Euler-Bernoulli |
|---|---|---|---|
| full | 229.1148 | −11.13 % ❌ | −8.35 % ❌ |
| sri | 256.8234 | −0.38 % ✅ | +2.73 % ✅ |
| mitc | 256.8234 | −0.38 % ✅ | +2.73 % ✅ |

The equal-order `full` rule is the least accurate (overall factor 0.125 from
`n=2` to `n=16`) — the textbook locking signature.  `sri` and `mitc` agree to
machine precision and converge at a steady 0.25/doubling (≈ O(h²), limited by
the point-load kink).  **Caveat, quantified in the next section:** at `n=16`
the equal-order rule is still *pre-asymptotic* (its local rate is only 1.58), so
this ladder alone cannot say whether locking costs an order of accuracy or
"only" a constant — `locking_rate_study.py` settles that.

## Asymptotic rate: locking is an error-constant defect (not a rate defect)

`run.py` stops at `n = 16`, where the equal-order error ratio per doubling is
still climbing (1.33, 2.00, 3.00).  `locking_rate_study.py` extends the ladder
to `n = 256` with a **pre-registered** protocol (decision, competing hypotheses
`H_rate`/`H_const`, thresholds, stopping rule and confounders are written in the
module docstring *before* the ladder was run).

| Treatment | local rate `n=8→16` | asymptotic `p∞` (`n=64…256`) | `n` for 5 % | error vs Timoshenko at `n=256` |
|---|---|---|---|---|
| full | 1.58 | **1.995** | **24.5** | −0.049 % |
| sri | 2.00 | 2.000 | 4.40 | −0.0015 % |
| mitc | 2.00 | 2.000 | 4.40 | −0.0015 % |

* `H_const` **accepted** (`p∞(full) = 1.995 ≥ 1.8`, `|p∞(full) − p∞(sri)| = 0.005
  ≤ 0.2`); `H_rate` **rejected** (`p∞(full) = 1.995 > 1.5`).
  Shear locking therefore multiplies the error **constant** — the equal-order
  rule keeps the same `O(h²)` rate and does converge to the Timoshenko answer
  (−0.049 % at `n = 256`); it never caps accuracy.
* The measured mesh penalty is `24.5 / 4.40 = 5.57`, and the error-constant
  ratio at `n = 256` is `|e_full|/|e_sri| = 33.0`; the prediction
  `√33.0 = 5.75` reproduces the measured penalty to −3.1 %.  That consistency
  is the quantitative statement of "locking costs mesh, not accuracy".
* The 5 % anchor tolerance is met by `full` only at `n ≈ 24.5`, i.e. ~5.6× the
  mesh the locking-free rules need — so the anchor's `n = 16` verdict (`full`
  FAIL, `sri`/`mitc` PASS) stays valid *as a mesh-cost statement*, and must not
  be read as "the equal-order element is unusable/wrong".

## Gates

| Gate | Criterion | Actual | Verdict |
|---|---|---|---|
| G1 | sri/mitc finest ≤ 5 % of Timoshenko | 0.38 % | ✅ PASS |
| G2 | sri/mitc finest ≤ 5 % of Euler-Bernoulli | 2.73 % | ✅ PASS |
| G3 | full rule least accurate (locking) | 11.13 % > 0.38 % | ✅ PASS |
| G4 | sri/mitc error non-increasing | 24.2→6.1→1.5→0.4 % | ✅ PASS |

**Overall: PASS** (the locking-free rules meet the < 5 % tolerance; the `full`
rule's failure is the expected, documented locking behaviour).

## Reproduce

```bash
PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/run.py
PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/validate.py
PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/locking_rate_study.py   # 14 s
python -m pytest tests/test_macneal_beam_shear_locking.py -q                        # 6 s
```

## Files

| File | Content |
|---|---|
| `run.py` | Driver (assembles, solves, grades, plots) |
| `validate.py` | Machine gate (18 checks, re-derives every verdict) |
| `make_report.py` | Builds `report.json` (numbers injected from `results.json`) |
| `results.json` | Every number (machine-generated) |
| `report.json` | Paper-grade report with embedded figures |
| `locking_rate_study.py` | Pre-registered `n = 2…256` asymptotic-rate study |
| `locking_rate.json` | That study's numbers, decision and gate block |
| `../../tests/test_macneal_beam_shear_locking.py` | CI gates: references, locking, rate study |
| `docs/assets/macneal_beam/*.png` | Convergence, error-decay, rate and field figures |
