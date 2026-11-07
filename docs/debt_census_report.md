# TensorFEM device-readiness debt census

Status: **delivered** · tool `tools/debt_census.py` · data `artifacts/fem_debt_census.json`
· schema `fem-debt-census/1` (port of TensorFVM `fvm-debt-census/1`, same 口径)
· figure `artifacts/figs/fem_debt_overview.png`

## 1. Why this exists

TensorFEM is a differentiable finite-element library built on PyTorch. Running it
end-to-end on an accelerator (NPU / GPU) is not decided by whether the solver
"works", but by whether the code can *stay* on the device: a module that only
knows NumPy can never run there, a `.cpu()`/`.numpy()` round trip serialises the
device queue, a tensor constructor without `device=` materialises on the default
device, and a Python `for` loop is executed by the interpreter on the host.

This census quantifies those four classes over `src/` so the porting effort can be
ranked instead of guessed. It is a **static** measurement: it counts constructs
that *block or impede* device residence, not runtime time-share.

## 2. Method (identical to the TensorFVM original)

Population: every `*.py` below `src/`, excluding `tests/`, `test/`, `examples/`,
`example/`, `__pycache__` subtrees (host-only by policy). Files are visited in
sorted order; every emitted list is sorted; no timestamp, absolute path or
environment value enters the JSON; floats are rounded. `source_digest` is a
content hash of the scanned bytes.

### D1 `numpy_only` (`numpy_only_count`)

Modules that import **no** device-capable array framework (`torch`, `torch_npu`,
`torch_sdaa`, `torch_mlu`, `jax`, `tensorflow`, `paddle`, `mindspore`, `cupy`)
but do import `numpy`. Such a module cannot run on an accelerator at all; it is
the hardest debt class. Pure-Python helper modules that import neither are
reported separately as `python_only` and kept **out** of D1.

### D2 `sync_points`

`<expr>.cpu()` plus `<expr>.numpy()` call sites — tensor→host synchronising round
trips that serialise the device queue. `.item()` / `.tolist()` /
`.synchronize()` are also synchronising but reported separately in
`breakdown.sync_extended`, because the task metric is the cpu/numpy pair.

### D3 `no_device_constructs`

Tensor-constructor call sites (module-level `torch.tensor/as_tensor/from_numpy/
frombuffer/zeros/ones/full/empty/arange/linspace/logspace/eye/rand/randn/randint/
randperm/normal/bernoulli/scalar_tensor/zeros_like/ones_like/full_like/
empty_like`) with (a) no `device=` keyword, (b) no `**kwargs` spread and (c) no
device inheritance. `*_like` constructors inherit from their first argument and
are therefore **not** debt (`breakdown.device_inherited_like`). Constructors
forwarding `**opts`/`**kwargs` go to `device_unverified_spread`: static analysis
cannot prove whether the caller supplies a device. `from_numpy`/`frombuffer` are
tracked in the `no_device_host_boundary` sub-bucket (they are always CPU by
definition).

### D4 `host_loops`

Python `for` / `async for` / `while` statement nodes. Any interpreter-level loop
runs on the host. Loops whose body contains an indexed store (`a[i] = ...`,
`a[i] += ...`) or an accumulator `.append/.extend/.add/.update(...)` are also
counted in `breakdown.vectorizable_loops` — those are the per-index loops that
must become tensor ops for a device port. `while` loops (solver time/iteration
loops) land in `breakdown.host_loops_by_kind.while`.

### Coverage / honest-FAIL rule

`ast_coverage_pct = 100 * files_parsed / files_scanned`. Files that fail to
parse/decode are listed **in full** in `uncovered_files` with the failure reason,
and the report must not round that away. Below 50 % coverage the full
`uncovered_files` table is the deliverable and the run is **not** a FAIL — it is
an honest incomplete measurement. **Here coverage is 100 %, so the table is
empty by measurement, not by suppression.**

## 3. Results — the four numbers

| metric | value | denominator |
|---|---:|---|
| `numpy_only` (D1) | **12** modules | 256 scanned |
| `sync_points` (D2) | **76** sites | 65 675 source lines |
| `no_device_constructs` (D3) | **1 186** sites | 1 943 tensor-factory calls |
| `host_loops` (D4) | **1 447** loops | 65 675 source lines |

Population and coverage:

| field | value |
|---|---:|
| `files_scanned` | 256 |
| `files_parsed` | 256 |
| `files_excluded` (inside scanned roots) | 0 |
| `ast_coverage_pct` | 100.00 |
| `uncovered_files` | `[]` (none) |
| `source_digest` | `sha256:fd31eec934aaa1d5aa50e347f59dc33256ec7241e50ea162939cfe80c4e5802e` |

Module composition and breakdowns:

| breakdown | value |
|---|---|
| device-capable modules (torch imported) | 202 |
| `numpy_only` modules | 12 (7 596 lines, 11.6 % of the tree) |
| `python_only` host helpers (no numpy, no framework) | 42 |
| `sync_points` cpu / numpy | 35 / 41 |
| `sync_extended` item / tolist / synchronize | 42 / 316 / 0 |
| tensor factories: total | 1 943 |
| — `device_explicit` | 507 (26.1 %) |
| — `device_inherited_like` | 250 (12.9 %) |
| — `device_unverified_spread` | 0 |
| — `no_device_plain` | 1 171 |
| — `no_device_host_boundary` (`from_numpy`/`frombuffer`) | 15 |
| `vectorizable_loops` (indexed store / append) | 1 049 (72.5 % of all loops) |
| loops inside `numpy_only` modules | 128 |
| sync points per 1 000 lines | 1.16 |

D3 by factory (top): `tensor` 546, `zeros` 215, `as_tensor` 132, `arange` 108,
`linspace` 85, `ones` 35, `eye` 17, `full` 16, `from_numpy` 15, `empty` 11,
`rand` 6.

D4 by iterable kind: `builtin:range` 488, `host_sequence` 338, `literal` 183,
`builtin:enumerate` 156, `builtin:zip` 68, `host_index` 60, `method:items` 46,
`call` 40, `method:tolist` 27, `while` 27, `builtin:sorted` 9,
`method:values` 4, `builtin:list` 1.

`numpy_only` modules (D1, full list):

```
src/tensorfem/adaptive_1d.py                 src/tensorfem/plate_bending_buckling.py
src/tensorfem/adaptive_poisson_2d.py         src/tensorfem/plate_buckling_hermite.py
src/tensorfem/cook_probe_protocol.py         src/tensorfem/plate_ritz_reference.py
src/tensorfem/cook_stress_reference.py       src/tensorfem/plate_shear_buckling.py
src/tensorfem/marine_corrosion_ice_ultimate.py  src/tensorfem/plate_shear_buckling_spectral.py
src/tensorfem/plate_varying_buckling_reference.py  src/tensorfem/rainflow_spectrum_reference.py
```

## 4. Where the debt actually is

Top modules by D3 (constructors that materialise on the default device):

| module | lines | no_device | sync | loops | device= |
|---|---:|---:|---:|---:|---:|
| `ice_panel_fsi.py` | 758 | 61 | 0 | 20 | 0 |
| `benchmark_registry.py` | 116 | 54 | 0 | 0 | 0 |
| `thermal_2d_reference.py` | 557 | 37 | 0 | 8 | 0 |
| `thermal_transient_reference.py` | 456 | 34 | 0 | 6 | 0 |
| `engineering_cases.py` | 120 | 30 | 0 | 2 | 0 |
| `element_rank_audit.py` | 897 | 25 | 0 | 35 | 6 |
| `suboff_ice_collision.py` | 446 | 25 | 0 | 19 | 0 |
| `saint_venant_torsion.py` | 535 | 23 | 0 | 2 | 0 |
| `suboff_ice_simulation.py` | 278 | 23 | 0 | 14 | 0 |
| `finite_strain_contact3d.py` | 420 | 22 | 0 | 22 | 0 |
| `rotating_disk.py` | 526 | 21 | 0 | 11 | 0 |
| `ice_sheet.py` | 184 | 20 | 0 | 14 | 0 |

Top modules by D2 (host round trips):

| module | sync_points | lines |
|---|---:|---:|
| `kirsch_hole.py` | 10 | 704 |
| `reference_strong_form_audit.py` | 7 | 650 |
| `finite_width_hole.py` | 6 | 399 |
| `hemisphere_hex20_oracle.py` | 6 | 345 |
| `cylindrical_shell_buckling.py` | 5 | 323 |
| `sparse_direct.py` | 5 | 291 |
| `result_db.py` | 4 | 172 |
| `coupled_3d_demo.py` | 3 | 132 |

Only 27 of 256 modules contain any `sync_point`, and 127 of 256 contain at least
one D3 site; the remaining 129 are contamination-free for D3.

Debt concentration by module role (filename classification; each module is
assigned to the first matching category, so the rows sum to 1 186):

| role | modules | D3 | share of 1 186 |
|---|---:|---:|---:|
| reference / oracle / analytic solutions | 17 | 190 | 16.0 % |
| benchmark / example / demo | 29 | 166 | 14.0 % |
| audit / qualification / evidence | 19 | 69 | 5.8 % |
| marine / ice / fatigue | 32 | 230 | 19.4 % |
| core kernels and the rest | 159 | 531 | 44.8 % |

Core element/kernel modules are largely device-clean and device-explicit —
`shell4.py` (0 D3, 10 `device=`), `contact.py` (0/3), `solid3d.py` (4/8),
`hex20.py` (2/6), `assembly.py` (0/1), `solvers.py` (0/3) — while the two
densest offenders are tiny composition modules, `frame_benchmarks.py`
(17 D3 in 31 lines) and `benchmark_registry.py` (54 D3 in 116 lines), both pure
table builders.

**Root cause amplifier.** No module in `src/` or `tests/` calls
`torch.set_default_device(...)`, and there is no shared `resolve_device()`
helper. Consequently a D3 site is not a stylistic nit: with a CUDA/NPU
`device` argument threaded into the public API, an internal `torch.tensor(...)`
or `torch.zeros(...)` still lands on CPU and the next kernel operation either
fails or silently round-trips. This is why D3 is 1 186 and not ~100.

## 5. Verification

1. **Determinism (machine criterion).** Two consecutive runs from the same
   working directory produce byte-identical output; a third run from a different
   working directory with an absolute `--repo-root` produces the same bytes:

   ```
   $ python3 tools/debt_census.py --out /tmp/cen_a.json --quiet
   $ python3 tools/debt_census.py --out /tmp/cen_b.json --quiet
   $ cmp /tmp/cen_a.json /tmp/cen_b.json            # identical (diff = 0)
   $ cd /tmp && python3 $REPO/tools/debt_census.py --repo-root $REPO --out /tmp/cen_c.json
   $ cmp /tmp/cen_a.json /tmp/cen_c.json            # identical (diff = 0)
   sha256 = d2e76a5f91bc1a0fcd5b747281c69285dde93e5886a0295c675298dcf04460d4
   ```

2. **Independent recount (tokenize).** The script re-counts `.cpu()`/`.numpy()`
   call sites with `tokenize` (no AST) and reports the deviation:
   AST 35/41 vs tokenize 35/41 → **deviation 0.0 %** for both.

3. **Independent re-implementation.** A second, separately written AST pass
   (different structure, definitions read from the text above, not copied)
   reproduces all four numbers exactly: `numpy_only=12`, `sync_points=76`,
   `no_device_constructs=1186`, `host_loops=1447`, `files=256`.

4. **Regression tests.** `tests/test_debt_census.py` (7 tests, ~12 s, unmarked =
   fast lane) pins: frozen FVM-compatible schema key order and `schema_version`;
   the four metrics on a synthetic fixture whose expected values are enumerated
   construct by construct (including the `*_like` inheritance and `**kwargs`
   spread exclusions); `uncovered_files` completeness on an unparsable module
   (coverage 66.67 %); byte-identical double run via the CLI from two different
   working directories; and a smoke scan of the real tree with the
   `parsed + uncovered == scanned` invariant.

5. **Cross-implementation sanity.** `grep`-based line counts agree in magnitude
   but not exactly with AST counts, as expected: 199 files match
   `^\s*(import|from)\s+torch` while the AST finds 202 framework modules — the
   difference is three files using `import time,torch` style combined imports
   (`ad_plastic_tangent.py`, `implicit_plastic_tangent.py`, `mesh_project.py`).
   The census is AST-based precisely so it does not inherit that grep blind spot.

## 6. Parity with the TensorFVM original

`git diff` against TensorFVM `tools/debt_census.py` is **6 changed lines**, all
identity/plumbing, no metric semantics:

| line | FVM | FEM |
|---|---|---|
| docstring title | `for TensorFVM` | `for TensorFEM` |
| docstring doc pointer | `docs/p8/FVM_DEBT_REPORT.md` | `docs/debt_census_report.md` |
| usage example | `artifacts/p8/fvm_debt.json` | `artifacts/fem_debt_census.json` |
| `SCHEMA_VERSION` | `fvm-debt-census/1` | `fem-debt-census/1` |
| `repo` field | `TensorFVM` | `TensorFEM` |
| `--out` default | `artifacts/p8/fvm_debt.json` | `artifacts/fem_debt_census.json` |

Top-level JSON keys, breakdown structure, thresholds, exclusion set and rounding
are identical, so FVM and FEM numbers are directly comparable.

## 7. How to read the numbers (and what they do not mean)

* **1 186 D3 sites is an upper bound on the porting work, not 1 186 bugs.**
  A `torch.tensor` inside diagnostic/report code that is immediately printed or
  written to JSON is harmless. The concentration analysis (55.2 % of D3 outside the core-kernel bucket:
  reference 16.0 %, benchmark 14.0 %, audit 5.8 %, marine/ice 19.4 %) says the plumbing is where the work is, and the first mechanical
  fix — routing every factory through a `device=` derived from the inputs — is
  worth more than rewriting the reference oracles.
* **1 049 vectorizable loops (72.5 %)** are the batch-vectorisation backlog; 128
  of the 1 447 loops live in the 12 `numpy_only` modules, i.e. inside the D1
  debt itself. Assembly/element loops over `range(n_elem)` dominate
  (`builtin:range` 488).
* **316 `.tolist()`** calls are asserted by D2 to be out of scope. They still
  synchronise at runtime; if a device port starts today, they are the next
  latency item after the 76 in-scope sync points.
* **Static analysis cannot see** `torch.empty_like(x)` where `x` came from CPU
  upstream, nor a device routed through a dataclass/config object; conversely a
  `*_like` call is trusted to inherit. D3 therefore measures *declared* device
  intent, not runtime device residence. A runtime census (`torch.cuda`/`npu`
  memory watch during a benchmark case) is the complementary measurement and is
  out of scope here.
* The 12 `numpy_only` modules are almost all analytic reference solutions
  (Ritz/spectral plate buckling, Cook probe, rainflow) whose CPU-only nature may
  be *intentional*: for oracles, NumPy is the point. They are still D1 because
  the rule is structural, not intentional — do not delete the metric, classify
  the module.

## 8. Limitations

1. Static, single-tree measurement: excludes `tests/` and `examples/` by policy,
   so test-side host code is invisible (and irrelevant for the shipping path).
2. Attribute-based call detection requires the framework import to be visible in
   the same module (`torch`, `torch.<alias>`, or a direct `from torch import
   tensor`). A framework reached through a re-export (`tensorfem.some_helper.torch`)
   is not recognised — none observed here.
3. `host_loops` counts statement nodes, not executed iterations; a loop over a
   3-element tuple and a loop over 10⁶ elements count the same. Use the
   `vectorizable_loops` sub-count for hot-loop triage.
4. The scrape is text/AST only; decorators, `eval`/`exec`, Cython and generated
   code are outside the population.

## 9. Reproduce

```
python3 tools/debt_census.py                          # -> artifacts/fem_debt_census.json
python3 tools/debt_census.py --quiet                  # one-line summary
python3 tools/debt_census_figure.py                   # -> artifacts/figs/fem_debt_overview.png
python3 -m pytest tests/test_debt_census.py -q        # 7 passed
```

Regenerate both artifacts in this order to keep the figure consistent with the
JSON (`source_digest` lets a reviewer confirm they came from the same tree).