## 6. Errata (2026-11-13, m17127): why `rebuild_from_dump_candidate` was the wrong remedy for most rows

§2 classed 12 anchors as `rebuild_from_dump_candidate` — "one renderer + zero solve".
That classification is about *data availability* and it is correct, but it hid a
**path-class defect** that made the remedy unachievable as written:

> The platform push harness (`tp_pusher.py`) drops every `*.png` that is not under
> `docs/` (`_EXCLUDE2 = re.compile(r'\.(db|sqlite3?|png|log)$')`, with a narrow
> `v5.8 F3` exemption: *"docs/ 下的交付图（png/svg）放行"*). A figure committed at
> `benchmarks/verified/<anchor>/<name>.png` therefore **never reaches GitHub**, so the
> certificate that names it stays `E_ARTIFACT_MISSING` no matter how green the local
> working tree is.

Measured on this checkout (`git ls-files`, 2026-11-13):

| measurement | value |
|---|---:|
| tracked `*.png` under `benchmarks/` | **0** |
| tracked `*.png` under `docs/` | 484 |
| anchors whose certificate names a `*.png` outside `docs/` | 15 (all currently missing) |
| such names in total | 40 |

The `fem_shell_modal` repair of §4 was merged with its four PNGs still at
`benchmarks/verified/fem_shell_modal/*.png`; the push dropped them, `main` kept the
renderer, the provenance stamp and the test, and
`tests/test_shell_modal_anchor_certificate.py` has been **red on `main`** ever since
(`certificate names artifacts that are not on disk`). The repairable-by-commit story of
§4 was true locally and false in the repository of record.

### 6.1 Corrected remedy: `rebuild_from_dump_into_docs`

1. `FIGURE_DIR = <repo>/docs/assets/verified/<anchor>/` — figures are *documentation*,
   which is exactly the class the push gate exempts.
2. The certificate names **repo-relative** paths (`_resolve()` in
   `tools/verified_anchor_audit.py` tries `REPO_ROOT / raw` before `anchor_dir / raw`,
   so repo-relative names resolve and are tracked-checkable).
3. `figure_provenance.json` stays next to `result.json` and keys its `figures` map by
   the same repo-relative paths, plus a `figure_dir` field.
4. A test asserts the figure directory is under `docs/`, so the mistake cannot be
   re-introduced quietly (`tests/test_shell_modal_anchor_certificate.py`,
   `tests/test_anchor_figure_gate_safety.py`).

Worked example, `fem_shell_modal` (m17127):

| metric | before | after |
|---|---:|---:|
| certified anchors (`tools/verified_anchor_audit.py`) | 11 | **12** |
| contradicted anchors | 17 | **16** |
| `tests/test_shell_modal_anchor_certificate.py` | 1 failed | **9 passed, 0 failed** |
| `tests/test_verified_anchor_certificates.py` | 2 failed (unregistered + stale) | **1 failed (unregistered, pre-existing)** |
| stale debt entries (`debt-diff`) | `['fem_shell_scordelis_lo']` | `[]` |

The re-render is byte-identical to the hashes the original repair recorded
(`9300f090… beam`, `a395c4a2… plate`, `b86c1d59… shell-vs-DMV`, `71ff83f6… audit`),
which is the useful part of this errata: **the pixels were always right; only the path
was doomed.** The honest verdict of the anchor is unchanged — it is still a FAIL
(G3 50.36 % vs 15 %, G4 eight rotation-only mechanisms in the first ten modes); this
repair restores the evidence chain, not the accuracy.

### 6.2 What is still open

* 10 anchors remain unregistered debt (`fem_beam_static`, `fem_cantilever`,
  `fem_hertz_contact`, `fem_kirsch_plate`, `fem_panel_buckling`, `fem_plate_modal`,
  `fem_plate_postbuckling`, `fem_plate_static`, `fem_structural_benchmarks`,
  `fem_winkler_plate`), so `tools/verified_anchor_audit.py --check` still exits 1.
  Each needs §6.1 applied to its own driver; that is 10 independent missions, not one.
* `fem_curved_beam` and `fem_wave_dispersion` name figures outside `docs/` but are not
  registered debt for other reasons; they are frozen in
  `tests/test_anchor_figure_gate_safety.py::FROZEN_OFFENDERS` (shrink-only).
* The push gate itself was not changed: `benchmarks/**/*.png` remains unwritable by
  design. If the platform ever wants anchor figures as repository objects rather than
  documentation, that is an owner-level change to `tp_pusher._EXCLUDE2`, not a repo fix.
* §5 stays true: this errata re-runs no solver and re-measures no physics.