# Shell4 formulation matrix: what the "in-plane mechanism" actually is

Status: **rank accounting corrected, membrane consequence measured, one open
item dissolved** · anchor proposal `GAP-FEM-SHELL4-INPLANE-RANK` ·
study `benchmarks/verified/fem_shell4_formulation_matrix/study.py` ·
data `benchmarks/verified/fem_shell4_formulation_matrix/results.json`
(schema `fem-shell4-formulation-matrix/1`) ·
figure `benchmarks/verified/fem_shell4_formulation_matrix/formulation_matrix.png` ·
library `src/tensorfem/shell4_rank.py` (`facet_rank_accounting`,
`drilling_gauge_mode`), `src/tensorfem/shell_membrane_benchmark.py` ·
tests `tests/test_shell4_rank_accounting.py`,
`tests/test_shell_membrane_benchmark.py`

## 1. Case

The merged Scordelis–Lo diagnosis
(`docs/CASE-FEM-SCORDELIS-LO-shear-rank-diagnosis.md`) reported that the flat
facet Shell4 keeps *"one spurious mechanism, purely in-plane
(`w = r_x = r_y = 0`)"* under every transverse-shear rule and left this open
item:

> *"One in-plane spurious mechanism survives every shear rule. It is
> orthogonal to the bending response measured here and its effect on
> membrane-dominated problems is **not** quantified by this study — open item."*

That sentence is a claim about a **defect** and it is testable, because a
zero-energy direction is either (a) a mechanism that an assembled load can
excite, or (b) a gauge that no assembly can represent. The two have opposite
consequences, and they can be separated with exact integer linear algebra plus
a membrane-dominated benchmark — the missing half of the benchmark matrix,
since every curved-shell benchmark in the repository is bending dominated.

Geometry and parameters of the two benchmarks used here:

| | Scordelis–Lo roof | pressurised cylinder |
|---|---|---|
| quantity | midspan free-edge deflection | mean radial expansion |
| reference | `w = −0.3024` (Scordelis & Lo 1960; MacNeal & Harder 1985) | `w = pR²/(E t)` (exact) |
| geometry | L 50, R 25, 2×40°, t 0.25 | R 100, L/2 50, t 1 (half ring, 2 symmetry planes) |
| material | E 4.32×10⁸, ν 0 | E 1000, ν 0.3 |
| load | vertical q = 90 | internal pressure p = 1 |
| supports | rigid end diaphragms | `u_z = 0` on both generators, `u_x = 0` on the x = 0 ring, one `u_y` anchor |
| dominated by | bending + membrane | pure membrane (`σ_θ = pR/t = 100`) |

## 2. Method

1. **A seventh zero-energy direction that is not a mechanism.**
   `u = 0`, `r = n` — a constant section rotation about the facet normal with
   no displacement — is resisted by nothing: the membrane block sees
   `u = 0`, the plate block does not see a rotation about the normal, and the
   difference-only drilling stabilisation has zero discrete Laplacian on a
   constant normal rotation. `drilling_gauge_mode` builds that vector and
   projects the six rigid-body modes out of it; `facet_rank_accounting`
   classifies the null space against **six rigid + one gauge** instead of six,
   so the reported spurious count is the number of directions that can
   actually carry a load path.
2. **Why it cannot be communicated.** Zero energy of the gauge requires the
   nodal rotation to be *parallel to its own facet's normal*. Two adjacent
   facets do not share a normal, so at the shared nodes the rotation would have
   to be parallel to two different directions — only `r = 0` survives, which is
   a rigid mode. Hence the gauge exists per facet and vanishes in any
   assembly; a flat mesh (all normals equal) is the sole exception, where it
   collapses into one global mode together with the in-plane rigid rotation.
   This is an exact algebraic statement, not a numerical observation.
3. **Cross-check with an independently written tool.**
   `tensorfem.element_rank_audit` was written separately and *already* expects
   seven zero modes per flat facet; its assembled-mesh scan is used unchanged.
4. **Membrane benchmark.** The pressurised cylinder is solved on a half ring
   cut on two symmetry planes so that **no support restrains the radial
   expansion being measured**. Because one `u_y` anchor removes the last rigid
   translation by letting the ring slide, the raw `u · e_r` reading is not the
   expansion; two gauge-invariant measures are used instead and must agree:
   a per-ring least-squares fit `u = t + w e_r`, and the deformed arc length
   `ε = L'/L₀ − 1` with `w = R ε`.

## 3. Results

### 3.1 Element rank accounting (1×1 m facet, E = 200 GPa, ν = 0.3, t = 0.01)

| configuration | zero modes | rigid | **gauge** | **spurious (gauge-aware)** | legacy 6-mode count |
|---|---:|---:|---:|---:|---:|
| `shell4` + plate `sri` | 9 | 6 | 1 | **2** | 3 |
| `shell4` + plate `mitc` | 7 | 6 | 1 | **0** | 1 |
| `shell4_mitc` + plate `sri` | 9 | 6 | 1 | **2** | 3 |
| `shell4_mitc` + plate `mitc` | 7 | 6 | 1 | **0** | 1 |

`‖K g‖ / ‖K‖_max = 1.4×10⁻¹⁶` for the gauge under every configuration: it is
exactly zero-energy, and the legacy column over-counts by exactly one, which is
the gauge. The two genuine mechanisms left by the one-point transverse-shear
rule are the classical plate hourglasses (`‖w‖ > 10⁻³`); the legacy "purely
in-plane mechanism" of the MITC variant has overlap `1.000` with the gauge
vector and is a re-labelling of it.

### 3.2 Assembled flat meshes (independent tool, expecting 7)

| configuration | n = 2 | n = 3 | n = 4 | null dim vs expected |
|---|---:|---:|---:|---|
| plate `sri` (both membrane variants) | 1 | 1 | 1 | 8 vs 7 |
| plate `mitc` (both membrane variants) | 0 | 0 | 0 | 7 vs 7 |

The gauge does not survive assembly; one global plate hourglass does, until the
transverse-shear block is rank sufficient.

### 3.3 Scordelis–Lo roof (bending dominated, reference −0.3024)

| elements/edge | `shell4`+`sri` | `shell4`+`mitc` | `shell4_mitc`+`sri` | `shell4_mitc`+`mitc` |
|---|---:|---:|---:|---:|
| 8 | −0.405557 (**+34.11 %**) | −0.284874 (−5.80 %) | −0.465325 (+53.88 %) | −0.316256 (+4.58 %) |
| 16 | −0.437542 (**+44.69 %**) | −0.295310 (**−2.34 %**) | −0.457580 (+51.32 %) | −0.305232 (**+0.94 %**) |
| 24 | −0.447801 (+48.08 %) | −0.300713 (−0.56 %) | — | — |

(8×8 and 16×16 rows are produced by `study.py` into `results.json`; the 24×24
rows are the same ladder re-run during this study and reproduce the merged
report to the printed digits.)

Non-monotone and plateauing for the one-point rule; monotone and inside the
3 % gate from 16×16 for the assumed-shear rule. **The membrane block is not
neutral here**: reducing the in-plane shear (`shell4_mitc`) shifts the same
faceted ladder from −2.34 % to +0.94 % at 16×16 and from +34 % to +54 % under
the one-point transverse rule, i.e. the two knobs interact and only the
combination `shell4_mitc` + plate `mitc` is inside 1 % at 16×16.

### 3.4 Pressurised cylinder (membrane dominated, exact `w = 10`, `u_x(L/2) = −1.5`)

Identical for **all four** configurations (displacements agree to <10⁻¹²
absolute):

| facets/half ring | w (fit) | error | w (arc length) | `u_x(L/2)` | axial error | facet ripple |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 9.80785 | −1.92 % | 9.80785 | −1.47118 | −1.92 % | <10⁻⁹ |
| 16 | 9.95185 | −0.48 % | 9.95185 | −1.49278 | −0.48 % | <10⁻⁹ |
| 24 | 9.97859 | −0.21 % | 9.97859 | −1.49679 | −0.21 % | <10⁻⁹ |
| 32 | 9.98795 | −0.12 % | 9.98795 | −1.49819 | −0.12 % | <10⁻⁹ |
| 48 | 9.99465 | −0.05 % | 9.99465 | −1.49920 | −0.05 % | <10⁻⁹ |

The facet-scale ripple is **below 10⁻⁹ relative** in every run: the discrete solution
is exactly "uniform expansion plus the support-induced rigid slide", and the
remaining error is the faceting error of the ring (the isosceles-trapezoid
facet is not exactly representable by a bilinear field). Both membrane
expansion and axial contraction (`−ν ε_θ`, a Poisson-coupling check that a
wrong plane-stress assumption would break) are inside 3 % from 16 facets.

## 4. Figures

`formulation_matrix.png`, three panels generated by the study script:
(a) spurious zero-energy modes per configuration, legacy vs gauge-aware;
(b) Scordelis–Lo error ladder against the 3 % gate;
(c) pressurised-cylinder error ladder against the 3 % gate (all four curves
coincide, which is the result).

## 5. Notes / limits

- **The corrected statement.** The merged report's "3 spurious / 1 spurious"
  is "2 real hourglasses + 1 gauge / 0 + gauge". The failure of the default is
  *not* about an in-plane mechanism; it is the transverse-shear block, as the
  same report's ladder already showed.
- **The open item is void, not merely small.** The in-plane reduction cannot
  affect a membrane state, because that state carries no in-plane engineering
  shear and no plate field: both knobs are null operators on it. Measured:
  identical displacement fields for the four configurations. The remaining
  membrane error is faceting, and it is *not* a formulation property.
- The membrane *block* is not a no-op in general: it moves the Scordelis–Lo
  answer (see §3.3), so "the in-plane reduction is unnecessary" holds for
  membrane states, not for bending-dominated faceted shells.
- A membrane-dominated problem that *does* carry non-uniform in-plane shear
  (e.g. a curved shear panel) would discriminate the membrane blocks. This
  study does not construct one, so "the membrane reduction is unnecessary" is
  supported only for states whose in-plane shear is constant or zero — stated
  honestly as a limitation, not as a general claim.
- The cylinder uses `E t /(h²)`-scale numbers (R 100, t 1) rather than steel
  values: at `t/R = 0.01` the membrane and bending stiffnesses differ by ~10⁴
  and a `V`-scaled model would only move the round-off floor, not the reported
  convergence.
- Capacity: all runs are single-process; the largest solve in the study is
  3750 dofs (Scordelis–Lo 24×24) with a dense factorisation, no cluster job.
- `shell4_stiffness` now accepts a planar float32 facet of O(100) size; the
  coplanarity guard previously had a fixed 1e-10 relative bound that is below
  float32 resolution at that scale (measured residual 1.6×10⁻⁷ = float32 eps ×
  coordinate) and rejected legitimate geometry. Guard fixed, regression test
  added; float64 behaviour is unchanged.

## 6. Error determination

| quantity | value | reference | error | < 3 %? |
|---|---|---:|---:|---|
| Scordelis–Lo, `shell4`+`mitc`, 16×16 | −0.295310 | −0.3024 | 2.34 % | **yes** |
| Scordelis–Lo, `shell4_mitc`+`mitc`, 16×16 | −0.305232 | −0.3024 | 0.94 % | **yes** |
| Scordelis–Lo, `shell4`+`sri`, 16×16 (default) | −0.437542 | −0.3024 | 44.69 % | **no** |
| cylinder `w`, any configuration, 16 facets | 9.95185 | 10 | −0.48 % | **yes** |
| cylinder `w`, any configuration, 48 facets | 9.99465 | 10 | −0.05 % | **yes** |
| cylinder `u_x(L/2)`, any configuration, 16 facets | −1.49278 | −1.5 | −0.48 % | **yes** |
| spurious modes, `sri` / `mitc` (gauge-aware) | 2 / 0 | 0 | exact integers | structural |
| gauge zero-energy residual | 1.4×10⁻¹⁶ | 0 | — | structural |