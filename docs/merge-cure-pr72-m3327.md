# PR #72 (`agent/m3327`) merge cure: three-way union with content-loss audit

## 1. Situation

PR #72 adds a linearly-varying edge-compression plate buckling qualification
(`src/tensorfem/plate_bending_buckling.py`, its benchmark driver, evidence JSON,
validation report and 30 tests). The branch was cut from merge-base
`dff97e7ff`; `origin/main` then advanced 6 commits (`14ca2fd`, merges of
#35/#51/#52 plus a main rebuild), so GitHub reported `mergeable=false`.

Because a mechanical overlay of one side has silently dropped content in this
repository before, the cure was done as a **content-union audit**, not just as
"git said the merge was clean".

## 2. Method

1. Merge `agent/m3327` (`1f548955b`) into `origin/main` (`14ca2fdc1`)
   in the worktree; `git merge --no-commit` reported **one** auto-merged file.
2. Assert three mechanical invariants against the merge result:
   * every file changed **only** by the branch must be byte-identical to the
     branch version (7 files);
   * every file changed **only** by main must be byte-identical to the main
     version (28 files);
   * every file changed by **both** sides must contain both sides' additions.
3. Cross-check with a tree diff: the merged tree differs from `origin/main` in
   exactly the branch's 8 files and from the branch in exactly
   main's 29 files, so the only file that differs from *both*
   parents is `CHANGELOG.md` — the legitimate union.
4. Run the full test suite on the merged tree and on an exported `origin/main`
   tree under identical conditions (`pytest -n 12`, `PYTHONPATH` pinned to each
   tree's `src`, `OMP_NUM_THREADS=2`).

## 3. Conflicts and resolutions

**Conflict files: 1** (`CHANGELOG.md`). Git auto-merged it because the two sides
insert at disjoint anchors (branch at the top of the file, main at the
`## Unreleased` block); the auto-merge was then *verified*, not trusted:

| Check | Result |
|---|---|
| branch marker `板面内线性变化压缩屈曲` present exactly once | True |
| main marker `finite-width (Howland/Heywood) verification` present exactly once | True |
| merged file differs from both parents (true union, not a fast-forward of one side) | True |
| lossless | **True** |

All other files are single-sided and were taken verbatim:

Branch-only files (7):
| `benchmarks/plate_bending_buckling.py` |
| `benchmarks/plate_bending_buckling_evidence.json` |
| `docs/assets/plate_bending_buckling.png` |
| `docs/benchmark-qualification-status.md` |
| `docs/marine-plate-linear-stress-buckling-validation.md` |
| `src/tensorfem/plate_bending_buckling.py` |
| `tests/test_plate_bending_buckling.py` |

Main-only files (28), including main's deletion of
`.scratch/proto_ritz.py`:
| `.scratch/proto_ritz.py` |
| `benchmarks/combined_panel_buckling.json` |
| `benchmarks/finite_width_hole_reference.json` |
| `benchmarks/near_incompressible_locking.py` |
| `benchmarks/near_incompressible_locking_evidence.json` |
| `docs/assets/combined_panel_buckling/convergence.png` |
| `docs/assets/combined_panel_buckling/interaction.png` |
| `docs/assets/combined_panel_buckling/shear_mode.png` |
| `docs/assets/finite_width_hole/finite_width_L_sensitivity.png` |
| `docs/assets/finite_width_hole/finite_width_convergence.png` |
| `docs/assets/finite_width_hole/finite_width_hole_edge_profile.png` |
| `docs/assets/finite_width_hole/finite_width_ktn_vs_ratio.png` |
| `docs/assets/finite_width_hole/finite_width_sigma_xx_field.png` |
| `docs/assets/near_incompressible_locking/error_vs_poisson.png` |
| `docs/assets/near_incompressible_locking/refinement_and_2d_debt.png` |
| `docs/assets/near_incompressible_locking/stress_profile_nu_4999.png` |
| `docs/finite-width-hole-howland.md` |
| `docs/kirsch-hole-stress-concentration.md` |
| `docs/marine-plate-combined-loading-verification.md` |
| `docs/near-incompressible-locking.md` |
| `scripts/audit_combined_panel_buckling.py` |
| `scripts/audit_finite_width_hole.py` |
| `src/tensorfem/finite_width_hole.py` |
| `src/tensorfem/incompressible_locking.py` |
| `src/tensorfem/marine_plate_combined_buckling.py` |
| `tests/test_finite_width_hole.py` |
| `tests/test_incompressible_locking.py` |
| `tests/test_marine_plate_combined_buckling.py` |

## 4. Test evidence

| Tree | passed | failed | skipped | xpassed |
|---|---|---|---|---|
| baseline `origin/main` `14ca2fdc1` | 1389 | 2 | 6 | 1 |
| merged `5f8b6d736` | 1419 | 2 | 6 | 1 |

The two failures are **identical in both trees** and pre-existing:

- `tests/test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure`
- `tests/test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors`

* The release-quality failure is environmental: `torch` emits a `pynvml`
  `FutureWarning` at import, which the test turns fatal via
  `PYTHONWARNINGS=error`; it reproduces on unmodified `origin/main`.
* The friction-contact failure is a numerical non-convergence
  (`Coulomb ratio=1.00002`, residual 5.0) at the same load step (index 7) on both
  trees.

The suite grew by exactly the 30 new buckling tests (merged total
1428 =
baseline total + 30); failure and skip counts are unchanged, so every new
test passes: **merged ≥ baseline, with no regression**.

## 5. SHAs

| ref | commit |
|---|---|
| `agent/m3327` before cure | `1f548955b84706f0be4d5196ee938d81d8ae5b57` |
| `origin/main` at cure time | `14ca2fdc1fd06467f3edb33db1d9e549b6a901a2` |
| merge commit (both refs) | `5f8b6d736a4b865a1516afcc0669402e56f634c3` |

## 6. Caveats

* GitHub was unreachable from this sandbox (TCP 443 timeout to github.com), so
  the refreshed commit could not be pushed from here; the local `agent/m3327`
  ref is moved to the merge commit and the platform publish path carries it.
* The environment has an editable-install `.pth` pointing at another agent's
  worktree, so every test run pins `PYTHONPATH` explicitly; unpinned runs would
  silently import foreign code.
* Two heavy single-file runs (friction contact, buckling) exceed the 2-minute
  shell budget standalone; their verdicts come from the parallel full-suite run.
