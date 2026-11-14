# GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE — round 3: eccentric ice patch

Anchor: `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE` — 加筋板架在冰压斑 + 侧压组合下的极限强度
(ledger status checked at start of work: `open`). Repository: TensorFEM.
Round 2 built the rigid-plastic grillage mechanism with **the ice patch centred on
the representative grillage**; this round removes that assumption and, in doing
so, finds and fixes a real defect in the orthotropic plate response that the
centred assumption had been hiding.

## 1. Why the patch position is the next question

The Marine Structures reviewer question that round 2 answered was "panel stress →
structural limit". The follow-up question is "does the answer depend on *where*
the ice patch sits?". Round 2 could not answer it: `grillage_plastic_collapse`
hard-coded the patch rectangle as centred (`patch = (0.5 - lx/2/L, ...)`), and
`plate_patch_response` evaluated its Navier series on **odd modes only**, a basis
that forces the plate response to be symmetric about the plate mid-lines. Both
place the patch at the centre — and the second one silently makes *any* patch
look centred, so even a user who moved the patch would have got a symmetric
answer.

Ice patches are not centred structures: a level-ice floe or a ridge keel hits the
shell wherever it hits. Two things therefore had to be answered together:

1. what is the grillage ultimate for an off-centre patch, and
2. is the centred patch the worst case (i.e. was round 2 conservative)?

## 2. Method

**Closed-form limit load for an eccentric patch (new, independent).**
`eccentric_patch_collapse_force` in `grillage_plastic_collapse`: for a simply
supported span, a patch of length `c` centred at `a` carrying a total force `F`
(line load `q = F/c`),

```
x_a = a - c/2,  R_A = q c (L-a)/L,  x_s = x_a + R_A/q
M_max/q = (x_s^2 - x_a^2)/2                  if x_s <= a + c/2   (shear-free section in patch)
M_max/q = R_A/q * x_b - c^2/2,  x_b = a+c/2  otherwise            (maximum at patch edge)
F_u = c M_p / (M_max/q)
```

`a = L/2` reproduces the central-patch value `8 M_p/(2L-c)` and `c → 0`
reproduces the point-load value `M_p L/(a(L-a))` — both are used as gates, so the
formula is not free-floating. For a determinate span one hinge is a mechanism, so
`F_u` is the collapse force exactly; no elastic property enters.

**Even modes in the plate series (defect fix).** `plate_patch_response` now uses
modes `1 … 2*terms` instead of the odd modes `1, 3, … 2*terms-1`. The odd modes
are a subset of the new set, and the even modes vanish identically for a patch
centred on the mid-lines, so every round-2 (centred / uniformly loaded) number is
reproduced unchanged: all 21 pre-existing gates still pass with identical values
to 7 significant digits, and the case-study ultimate moves by 4e-6 relative. Peak
sampling stays on `linspace(0, span, sample)` with `sample = 21`.

**Independent finite-difference reference (new).**
`tensorfem.orthotropic_plate_fd.ssss_orthotropic_plate_fd` solves
`D1 w,xxxx + 2 D3 w,xxyy + D2 w,yyyy = q` on a uniform grid with the 13-point
biharmonic stencil, simply supported edges by antisymmetric ghost nodes
(`w_(-1) = -w_1`, i.e. `w = w,nn = 0`) and the patch load integrated **exactly**
over each cell. It shares no code and no method with the series solution, which
is the point: a wrong mode set is invisible from inside the series. The FD solver
is itself checked against the Timoshenko value for a uniformly loaded SSSS square
plate (`w_max = 0.00406 q a^4/D`, `M_x = 0.0479 q a^2`): +0.06 % / −0.04 %.

**Grillage extension.** `grillage_plastic_collapse(..., patch_centre=(cx, cy))`
now places the patch anywhere on the two system spans (validated: the patch must
lie inside the plate), feeds the eccentric rectangle to the load-share integral
and the eccentric interval to both member lines, and reports the interval and
centre in the result. `patch_position_map` sweeps patch position × coexisting
compression into a machine-readable table.

## 3. Parameters (unchanged round-2 case study)

| parameter | value |
| --- | --- |
| frame spacing / bay spacing | 2.1 m / 0.35 m |
| grillage length × frame span | 8.40 m × 2.80 m (4 spans) |
| plate t / sigma_y | 20 mm / 355 MPa |
| stiffener | Tee 250×10 + 80×12, tributary 0.35 m |
| frame | Tee 550×10 + 120×14, effective plate 0.84 m |
| M_p stiffener / frame | 0.189 / 0.905 MN·m |
| ice patch | 1.05 m × 0.35 m (c/L_span = 0.5 on the frame span) |
| hydrostatic / design pressure | 0.10 MPa / 4.0 MPa (caller supplied) |
| coexisting compression | 0.00 / 0.30 sigma_y |

## 4. Results

### 4.1 Eccentric patch vs exact closed form (frame span, n = 64 elements)

| a/L | hinge solver q_u [N/m] | closed form q_u [N/m] | error | hinges |
| --- | --- | --- | --- | --- |
| 0.30 | 1 803 206 | 1 803 779 | −0.032 % | 1 |
| 0.50 | 1 514 770 | 1 515 175 | −0.027 % | 1 |
| 0.70 | 1 803 206 | 1 803 779 | −0.032 % | 1 |

Symmetry `a ↔ L−a` holds to 1e-11. The limit force is **minimum at `a/L = 0.5`**
for the determinate span (closed form and hinge solver agree; gates V13a/V13b).

### 4.2 Plate response: Navier (fixed basis) vs finite difference

| patch centre x/L | w_max Navier | w_max FD (nx = 240) | error | M_x,max Navier | M_x,max FD | error |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 7.48908e-10 | 7.48980e-10 | −0.010 % | 0.064871 | 0.064963 | −0.142 % |
| 0.50 | 7.38952e-10 | 7.39044e-10 | −0.012 % | 0.063728 | 0.063724 | +0.006 % |

### 4.3 Why the fix matters (defect witness, patch at x/L = 0.3)

| quantity | FD reference | fixed basis | error | odd-only basis (pre-round-3) | error |
| --- | --- | --- | --- | --- | --- |
| w_max [m] | 7.4638e-10 | 7.4602e-10 | −0.049 % | 3.7361e-10 | **−49.9 %** |
| M_x,max | 0.063972 | 0.063940 | −0.050 % | 0.030302 | **−52.6 %** |

The old basis underestimates the local plate response to an eccentric patch by
~50 % (it forces the peak to midspan). Gate V15 fails if this ever regresses.

### 4.4 Grillage ultimate vs patch position (`patch_position_map.json`)

| patch centre x/L | axial ratio | ultimate [MPa] | governing mode | margin / design | load share x | stiffener limit [MPa] | frame limit [MPa] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25 | 0.0 | 2.2743 | plate_local_bending | 0.569 | 0.1623 | 125.69 | 6.687 |
| 0.25 | 0.3 | 2.1648 | plate_local_bending | 0.541 | 0.1623 | 114.37 | 6.687 |
| 0.35 | 0.0 | 2.2743 | plate_local_bending | 0.569 | 0.1587 | 38.16 | 6.659 |
| 0.35 | 0.3 | 2.1648 | plate_local_bending | 0.541 | 0.1587 | 34.72 | 6.659 |
| 0.50 | 0.0 | 2.2743 | plate_local_bending | 0.569 | 0.1673 | 142.39 | 6.728 |
| 0.50 | 0.3 | 2.1648 | plate_local_bending | 0.541 | 0.1673 | 129.56 | 6.728 |
| 0.65 | 0.0 | 2.2743 | plate_local_bending | 0.569 | 0.1587 | 38.16 | 6.659 |
| 0.65 | 0.3 | 2.1648 | plate_local_bending | 0.541 | 0.1587 | 34.72 | 6.659 |
| 0.75 | 0.0 | 2.2743 | plate_local_bending | 0.569 | 0.1623 | 122.04 | 6.687 |
| 0.75 | 0.3 | 2.1648 | plate_local_bending | 0.541 | 0.1623 | 111.05 | 6.687 |

Three findings, none of them assumed in advance:

* **The reported ultimate is insensitive to patch position** (2.2743 / 2.1648 MPa,
  identical across the sweep) because the plate field governs for this
  scantling — so the round-2 headline number stands for any patch position.
* **But the round-2 patch position is the *least* severe one for the stiffener
  line.** `x/L = 0.5` of the 8.40 m grillage is 4.20 m = 2 × 2.10 m, i.e. the
  patch sits *on* an interior frame; the stiffener line then reaches 142 MPa.
  Near a stiffener span's mid-span (`x/L = 0.35`, x = 2.94 m) the same line
  collapses at 34.7 MPa — a factor of 4. The margin to the plate-field limit
  therefore shrinks from 63× to 15×, still not governing, so the conclusion is
  unchanged, but the "worst span" wording of round 2 was not conservative for
  the stiffener line.
* **Off-centre patches shift load to the frames.** The orthotropic share moves
  from `f_x = 0.1673` (centred) to `0.1587` (x/L = 0.35/0.65), and since the frame
  system carries the low limit pressure the grillage plastic pressure *falls*
  off-centre (6.728 → 6.659 MPa). At grillage level, centred is therefore **not**
  the worst case — the opposite of the single-line result. Both facts are locked
  into `tests/test_grillage_patch_eccentricity.py`.

## 5. Verification (33 gates, 0 failed)

```
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification artifacts/grillage_plastic_collapse
PYTHONPATH=src python -m pytest tests/test_grillage_patch_eccentricity.py \
      tests/test_grillage_plastic_collapse.py tests/test_grillage_ice_ultimate.py -q
```

New gates (all pre-declared before the run):

| gate | got | reference | error | allowed |
| --- | --- | --- | --- | --- |
| V11a_x30_eccentric_patch | 1.80321e6 | 1.80378e6 | −0.032 % | 0.5 % |
| V11b_x50_eccentric_patch | 1.51477e6 | 1.51517e6 | −0.027 % | 0.5 % |
| V11c_x70_eccentric_patch | 1.80321e6 | 1.80378e6 | −0.032 % | 0.5 % |
| V12a_closed_form_central_patch | 1.59093e6 | 1.59093e6 | 0.000 % | 1e-12 |
| V12b_closed_form_point_load | 1.53885e6 | 1.53885e6 | 0.000 % | 1e-5 |
| V13a_worst_centre_closed_form | 1.59093e6 | 1.59093e6 | 0.000 % | 1e-9 |
| V13b_worst_centre_solver | 1.51356e6 | 1.51356e6 | 0.000 % | 0.5 % |
| V14a_x25_plate_deflection | 7.48908e-10 | 7.48980e-10 | −0.010 % | 2 % |
| V14a_x25_plate_moment | 0.0648712 | 0.0649635 | −0.142 % | 2 % |
| V14b_x50_plate_deflection | 7.38952e-10 | 7.39044e-10 | −0.012 % | 2 % |
| V14b_x50_plate_moment | 0.0637281 | 0.0637245 | +0.006 % | 2 % |
| V15_odd_only_basis_witness | 1 (pass) | 1 | 0.000 % | 0 |

Tests: `tests/test_grillage_patch_eccentricity.py` 17 tests, all pass; the 10
pre-existing grillage tests still pass after the mode-set change.

## 6. Error verdict against the platform criterion

| key quantity | reference | error | < 3 % ? |
| --- | --- | --- | --- |
| eccentric-patch collapse force (a/L = 0.3/0.5/0.7) | exact rigid-plastic closed form | ≤ 0.032 % | **yes** |
| plate w_max, M_x,max (eccentric patch) | independent FD solve | ≤ 0.142 % | **yes** |
| centred-patch results (round 2) | unchanged | ≤ 4e-6 relative | **yes** |
| FD reference itself | Timoshenko SSSS square plate | 0.06 % / 0.04 % | **yes** |
| grillage plastic pressure, n = 16 vs n = 8 | mesh self-consistency | 2.15 % | yes (gate 5 %) |

No gate compares the solver with itself except the two identity limits of the new
closed form (V12a/V12b), which are labelled as such.

## 7. Pitfalls found on the way

* **The FD check caught a bug in the FD solver first.** The `_FOURTH` stencil was
  originally applied with `dj` for the y direction; because that tuple has
  `dj = 0` everywhere, all five y-terms landed on the centre node and summed to
  zero, giving a plate 1.29× too soft (caught immediately by the Timoshenko
  check, not by the Navier comparison — which is exactly why the reference needs
  its own reference).
* **Difference of squares.** `(x_s^2 - x_a^2)/2` loses ~6 digits in the
  `c → 0` point-load limit; the factored form `(x_s-x_a)(x_s+x_a)/2` is used.
* **`sample` matters once the field is asymmetric.** The peak sampling grid of
  the series solution was fine at 11 points while the field was forced symmetric;
  21 is now the default and the peaks are resolved to <0.5 %.
* **Grid inference from `terms`.** Doubling the mode set quadruples the series
  cost; the verification run went from 15 s to ~110 s and the patch-position
  sweep is the bulk of it, so it runs at 6 elements/span.

## 8. Limits (unchanged + new)

Rigid-plastic material, monotonic loading, hinges at element boundaries only, no
plate buckling / web tripping / residual stress / initial deflection inside the
member mechanism (they enter only through the plate-field limit), quasi-static
patch. New limits: the patch is axis-aligned (no rotated patch), the support
lines are still rigid (girder flexibility not modelled), and the load share for
an eccentric patch is still the energy share of the *smeared* orthotropic plate —
the FD gate validates the plate response, not the smearing.

## 9. Reproduction

```
PYTHONPATH=src python -m tensorfem.orthotropic_plate_fd                     # FD self-check
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification artifacts/grillage_plastic_collapse
PYTHONPATH=src python -m pytest tests/test_grillage_patch_eccentricity.py -q
```

Artifacts: `artifacts/grillage_plastic_collapse/grillage_plastic_verification.json`,
`patch_position_map.json`, `fig_grillage_eccentric_patch.png`,
`fig_grillage_patch_position_map.png`, `fig_grillage_odd_only_witness.png`,
`fig_grillage_plastic_collapse.png`, `fig_grillage_plastic_sensitivity.png`.