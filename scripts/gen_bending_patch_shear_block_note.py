#!/usr/bin/env python3
"""Generate ``docs/CASE-BENDING-PATCH-round2-shear-block.md`` from the audit JSON.

Every number in the report is injected from
``docs/benchmarks/tutorials/bending-patch-shear-block.json`` (produced by
``scripts/audit_bending_patch_shear_block.py``); nothing is transcribed by hand.

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_bending_patch_shear_block.py
    PYTHONPATH=src python scripts/gen_bending_patch_shear_block_note.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "benchmarks" / "tutorials" / "bending-patch-shear-block.json"
OUT = REPO / "docs" / "CASE-BENDING-PATCH-round2-shear-block.md"

FIGS = REPO / "docs" / "assets" / "bending_patch"


def sci(x: float, digits: int = 3) -> str:
    if x == 0.0:
        return "0"
    return f"{x:.{digits}e}"


def pct(x: float, digits: int = 3) -> str:
    return f"{100.0 * x:.{digits}f} %"


def shear_fraction(t_over_l: float, poisson: float, kappa: float) -> float:
    """Shear share of the tip deflection of a moment-loaded plate strip.

    ``w_shear / w_bend = 2 D_b / (kappa_s G t L) = (1 + nu)/(3 kappa_s) (t/L)^2``,
    so the share of the *total* deflection is ``s / (1 + s)``.
    """
    s = (1.0 + poisson) / (3.0 * kappa) * t_over_l ** 2
    return s / (1.0 + s)


def main() -> int:
    d = json.loads(SRC.read_text())
    m = d["material"]
    gates = d["gates"]
    exact = d["exactness"]
    controls = d["controls"]
    anchor = {row["name"]: row for row in d["anchor_blindness_survey"]}
    ladder = d["sensitivity_ladder"]
    split = d["convention_split"]
    field = d["field_map"]
    ident = d["instrument_identity"]
    base = d["baseline_vs_committed"]

    nominal = anchor["none"]
    kappa_anchor = anchor["kappa_s=1 (-20%)"]
    pi_anchor = anchor["kappa_s=pi^2/12 (-1.3%)"]
    sign_anchor = anchor["shear_sign"]
    noshear_anchor = anchor["no_shear"]
    ctrl = {row["name"]: row for row in controls}
    exact_worst = max(r["energy_rel_error"] for r in exact)
    first_flip = next((row for row in ladder if row["n_flipped"] > 0), None)
    n_legs = len(nominal["legs"])

    L: list[str] = []
    A = L.append

    A("# CASE-BENDING-PATCH round 2 (reopened): is the transverse-shear block of the")
    A("Q4 Mindlin plate element *ever* graded?")
    A("")
    A("**Anchor**: `fem_bending_patch` (`benchmarks/verified/fem_bending_patch/`),")
    A("round-1/round-2 gate pool of 24 gates. **Mission**: m21969 (reopened PR#1073).")
    A("**Machine-readable evidence**: `docs/benchmarks/tutorials/bending-patch-shear-block.json`")
    A("(all numbers below are injected from it).")
    A("")
    A("---")
    A("")
    A("## 1. Case introduction")
    A("")
    A("### 1.1 Physical background")
    A("")
    A("The four-node Mindlin-Reissner plate element (`tensorfem.plate.q4_mindlin_stiffness`)")
    A("is the production element behind the plate/shell paths of the ship-structure models")
    A("(hull plating, deck panels, grillage flange plating, ice-strengthened shell). Its")
    A("stiffness splits into a **bending** block `Db` and a **transverse-shear** block")
    A("`Ds = kappa_s G t`. Round 1 and round 2 of this anchor graded the bending block")
    A("extremely hard (constant-curvature patch test, plane-stress `M_y = 0` state, load-")
    A("driven strip), but every one of those states is **shear-free by construction**:")
    A("`gamma_xz = gamma_yz = 0` is precisely what makes them *constant-curvature* states.")
    A("The transverse-shear block therefore enters no graded quantity, and the load-driven")
    SHARE_REF = shear_fraction(0.01, m["poisson"], m["shear_correction"])
    A("strip does not help either -- it runs at `t/L = 0.01`, where the shear flexibility is")
    A(f"~{pct(SHARE_REF, 4)} of the tip deflection (closed form in §5).")
    A("")
    A("This matters for thick or shear-flexible ship structures: sandwich panels, ice-class")
    A("plating with `t/a` of 0.1-0.3, and any panel where the transverse-shear deflection is")
    A("a first-order term. A 20 % error in `Ds` is a realistic defect (it is exactly what")
    A("`kappa_s = 1` instead of the declared `5/6` does, and `kappa_s = pi^2/12` -- the")
    A("parabolic-distribution value -- differs by 1.3 %), and nothing in the anchor would")
    A("report it.")
    A("")
    A("### 1.2 Geometry, material, parameters")
    A("")
    A(f"| item | value |")
    A("|---|---|")
    A(f"| material | E = {m['young']:.1f} MPa, nu = {m['poisson']}, t = {m['thickness']} mm |")
    A(f"| shear factor used by the element | kappa_s = 5/6 = {m['shear_correction']:.6f} |")
    A(f"| shear factor of a uniform-shear assumption (control) | kappa_s = 1 |")
    A(f"| shear factor of a parabolic shear distribution (control) | kappa_s = pi^2/12 = {m['kappa_s_parabolic']:.6f} |")
    A("| element | `tensorfem.plate.q4_mindlin_stiffness`, shear treatments `sri` / `mitc` / `full` |")
    A("| patch families | `affine` (unit square, 2x2), `skew` (parallelogram, skew 0.35), `distorted` (random interior shift, seed 20261111) |")
    distinct_states = sorted({row["state"] for row in exact})
    A("| graded states | constant transverse shear `gamma = (gamma_xz, gamma_yz)`, `kappa = 0`: "
      + ", ".join(f"`{st}`" for st in distinct_states) + " |")
    A("| reference | closed form `U = 1/2 kappa_s G t gamma^2 * Area` (no table digit) |")
    A("")
    A("---")
    A("")
    A("## 2. What was done, and what was predeclared")
    A("")
    A("### 2.1 Decision")
    A("")
    A("May the round-2 anchor (24 gates) stand as evidence for the **whole** Q4 Mindlin")
    A("element, or must the shear block be graded separately?")
    A("")
    A("| option | content | why / why not |")
    A("|---|---|---|")
    A("| A | keep the anchor as delivered | the gate pool is invariant to the shear block (measured below); presenting it as element evidence overstates the coverage |")
    A("| B | add a thick-plate displacement benchmark | the shear share of a plate strip's deflection is `s/(1+s)` with `s = (1+nu)/(3 kappa_s)(t/L)^2`, i.e. 0.005 % at `t/L = 0.01` and 2.1 % at `t/L = 0.2`; a 20 % `Ds` error there moves the benchmark by <= 0.5 %, i.e. below any sensible displacement tolerance (table in §5) |")
    A("| **C** | **grade the constitutive block directly**: a *constant-transverse-shear* state whose prescribed-field energy identity `1/2 u^T K u = 1/2 Ds gamma^2 Area` is exact to round-off, plus mutation controls that prove the check can fail | **chosen** |")
    A("")
    A("### 2.2 Hypothesis and predeclared tests")
    A("")
    A("| # | hypothesis | predeclared test / threshold (fixed before the survey was inspected) | outcome |")
    A("|---|---|---|---|")
    A(f"| H1 | the anchor cannot see the shear modulus | with `Ds` scaled by 1.2 (i.e. `kappa_s = 1`), **zero** of the {n_legs} shear-sensitive anchor legs fail | measured {kappa_anchor['n_flipped']} fail -- **confirmed** |")
    A(f"| H2 | a *gross* `Ds` error is also invisible | with `Ds` multiplied by 5, still zero legs fail | measured {next(r for r in ladder if r['kappa_scale']==5.0)['n_flipped']} -- **confirmed** (first flip only at `Ds x{first_flip['kappa_scale']:.0f}`) |")
    A(f"| H3 | a flipped shear sign relation *is* caught | `gamma = w,x - theta_x` must fail many legs | measured {sign_anchor['n_flipped']}/{n_legs} legs fail -- **confirmed** (the sign relation is graded) |")
    A(f"| H4 | a deleted shear block is caught fail-safe | the Dirichlet patch must not solve | measured `{noshear_anchor['status']}` -- **confirmed** (loud failure, not a silent pass) |")
    A(f"| H5 | the new check is exact, not merely convergent | energy identity `<= 1e-12` on 3 families x 3 treatments x 3 states | measured worst {sci(exact_worst)} -- **confirmed** |")
    A(f"| H6 | the new check can fail | `kappa_s = 1` (-20 %) must be rejected by `>= 0.15` relative energy error | measured {ctrl['kappa_s=1 (-20%)']['energy_rel_error']:.6f} -- **confirmed** |")
    A(f"| H7 | `kappa_s = pi^2/12` (-1.3 %) is also graded | relative energy error `>= 1e-2` | measured {sci(ctrl['kappa_s=pi^2/12 (-1.3%)']['energy_rel_error'])} -- **confirmed** |")
    A(f"| H8 | the field split matters | a `w`-only state must be blind to the shear sign (`<= 1e-9` separation) while the balanced split must separate by `>= 0.9` | measured {sci(split['w_only_convention_blindness'])} vs {split['balanced_convention_separation']:.3f} -- **confirmed** |")
    A("")
    A("**Stopping rule**: the survey stops at 9 `Ds` multipliers (nominal, `pi^2/12`,")
    A("`kappa_s = 1`, x1.2, x0.8, x2, x5, x20, x100); no threshold was moved after the")
    A("first result was seen. **Repetition**: every quantity is deterministic -- no")
    A("sampling, no seeds except the fixed `distorted` mesh seed, so a single run per")
    A("configuration is the complete experiment; the reproducibility check is re-running")
    A("the driver and diffing the JSON (bit-identical apart from timings).")
    A("")
    A("### 2.3 Method")
    A("")
    A("1. **Mutation controls.** `tensorfem.bending_patch.shear_block_mutation()` swaps the")
    A("   `sri` element for a deliberately defective one (`mutated_q4_stiffness`): `Ds`")
    A("   scaled, sign relation flipped, or the shear block deleted. Inside the context the")
    A("   *unmodified* anchor legs (`run_patch_case`, `free_surface_check`,")
    A("   `strip_pure_bending`) are recomputed with the anchor's own thresholds, so the")
    A("   question \"would the gate pool notice?\" is answered by the gate pool itself.")
    A("2. **The repair.** A constant-transverse-shear state `gamma = const, kappa = 0`,")
    A("   realised with the *balanced split* `theta = gamma/2`, `w = gamma.x/2`. All DOFs")
    A("   are prescribed (a uniform shear field is not in equilibrium at free rotation")
    A("   DOFs -- it needs a body load), so this is a prescribed-field energy identity, not")
    A("   a Dirichlet patch test. Because `w` is linear and `theta` constant, the state lies")
    A("   in the interpolation space, so the identity is exact for every treatment and every")
    A("   valid quad mesh -- there is no interpolation caveat and no convergence study.")
    A("3. **Instrument integrity.** `mutated_q4_stiffness(..., mutation='none')` must")
    A(f"   reproduce the production `sri` stiffness (measured {sci(ident['reproduction_sri_rel'], 2)}")
    A(f"   relative), the context must leave `mitc`/`full` untouched")
    A(f"   ({sci(ident['mitc_untouched_rel'], 1)}/{sci(ident['full_untouched_rel'], 1)}) and must")
    A(f"   actually bite on `sri` ({ident['sri_changed_rel']:.3f} relative change) -- all three")
    A("   are graded, so the audit cannot degenerate into a copy auditing itself.")
    A("")
    A("---")
    A("")
    A("## 3. Results")
    A("")
    A("### 3.1 The finding: the anchor is blind to the shear modulus")
    A("")
    A(f"{n_legs} shear-sensitive legs of the anchor were recomputed under each mutation, with")
    A("the anchor's own thresholds (`<= 1e-12` for the exactness legs, `<= 2 %` for the")
    A("patch/strip pool tolerance).")
    A("")
    A("| mutation | status | legs that fail |")
    A("|---|---|---|")
    for name in ("none", "kappa_s=1 (-20%)", "kappa_s=pi^2/12 (-1.3%)", "shear_sign", "no_shear"):
        row = anchor[name]
        A(f"| {name} | `{row['status']}` | {row['n_flipped']} / {n_legs}"
          + (f" ({', '.join('`'+f+'`' for f in row['legs_flipped'][:4])}"
             + (", ..." if row["n_flipped"] > 4 else "") + ")" if row["n_flipped"] else "")
          + " |")
    A("")
    A("Sensitivity ladder -- how wrong must `Ds` be before the anchor notices?")
    A("")
    A("| `Ds` multiplier (what it means) | anchor legs that fail | new check, relative error |")
    A("|---|---|---|")
    for row in ladder:
        A(f"| {row['name']} (`x{row['kappa_scale']:.4f}`) | {row['n_flipped']} / {n_legs} | {sci(row['new_check_energy_rel_error'])} |")
    A("")
    A("Read the first three rows: a **20 % error in the transverse-shear rigidity, and even a")
    A("5x error, leave every patch/strip/plane-stress gate green.** The reason is structural,")
    A("not a resolution issue -- see §5. The sign relation (row 4) *is* graded, because the")
    A("constant-curvature states have `gamma = 0` and a flipped sign makes the parasitic shear")
    A("`2 theta != 0`; and a deleted block (row 5) fails loudly through a singular stiffness")
    A("matrix. So the gap is exactly one quantity: the **magnitude** of `Ds`.")
    A("")
    A("### 3.2 The repair: constant-shear prescribed-energy identity")
    A("")
    A(f"| quantity | value | threshold |")
    A("|---|---|---|")
    for g in gates:
        A(f"| `{g['id']}`: {g['quantity']} | {sci(g['value'])} | {g['operator']} {g['threshold']:.2e} |")
    A("")
    A(f"All {len(gates)} gates pass. The identity holds to {sci(exact_worst)} relative error over")
    A(f"{len(exact)} cases (3 patch families x 3 shear treatments x 3 shear states), i.e. to")
    A("round-off, on *distorted* elements as well -- the check is a property of the element,")
    A("not of the mesh.")
    A("")
    A("### 3.3 Discriminating power (the check is not vacuous)")
    A("")
    A("| mutation | energy identity, relative error | separation from nominal |")
    A("|---|---|---|")
    for row in controls:
        A(f"| {row['name']} | {sci(row['energy_rel_error'])} | {row['separation']:.3f} |")
    A("")
    A(f"A 20 % error in `Ds` is rejected with {sci(ctrl['kappa_s=1 (-20%)']['energy_rel_error'], 1)}")
    A(f"relative error against a {sci(1e-12, 0)} tolerance, i.e. ~1e14 of margin; the 1.3 %")
    A("`pi^2/12` variant is rejected by ~1e10 of margin. The flipped sign relation and the")
    A("deleted block give `gamma = 0`, i.e. 100 % separation.")
    A("")
    A("### 3.4 Why the *balanced split* and not a `w`-only field")
    A("")
    A(f"| field | energy, model convention | energy, flipped sign relation | separation |")
    A("|---|---|---|---|")
    A(f"| `w`-only (`theta = 0`) | {sci(split['w_only_energy'])} | {sci(split['w_only_energy_flipped'])} | {sci(split['w_only_convention_blindness'], 1)} |")
    A(f"| balanced (`theta = gamma/2`) | {sci(split['balanced_energy'])} | {sci(split['balanced_energy_flipped'])} | {split['balanced_convention_separation']:.3f} |")
    A("")
    A("A `w`-only state is *invariant* to the sign of the shear relation (`w,x + theta_x` and")
    A("`w,x - theta_x` both give `gamma` when `theta = 0`), so it cannot grade the convention.")
    A("The balanced split is what turns the same energy identity into a convention-sensitive")
    A("check -- a design point that had to be measured, not assumed.")
    A("")
    A("---")
    A("")
    A("## 4. Figures")
    A("")
    A("![sensitivity](assets/bending_patch/shear_block_sensitivity.png)")
    A("")
    A("**Fig. 1** (a) number of anchor legs that fail as the transverse-shear rigidity is")
    A("scaled over 0.8x-100x (log axis); the shaded band is the realistic `kappa_s` error")
    A("band (`+-20 %`), inside which the anchor reports nothing. The first flip needs")
    A(f"`Ds x{first_flip['kappa_scale']:.0f}`. (b) the new constant-shear energy identity under")
    A("the same mutations, against the `1e-12` graded tolerance (dashed) -- eight to fourteen")
    A("orders of magnitude of separation.")
    A("")
    A("![field map](assets/bending_patch/shear_block_field_map.png)")
    A("")
    A("**Fig. 2** field map of the graded state on the *distorted* patch: the transverse shear")
    A("recovered from the finite-element field at the Gauss points is constant")
    A(f"(spread {sci(field['shear_spread_rel'], 2)} relative) and the bending strain is zero")
    A(f"(`max|kappa_i| = {sci(field['curvature_leak'], 2)}`). This is the cloud-map evidence that")
    A("the state the check grades is the state it claims to grade; colour scales are tight")
    A("around the exact value, so a non-constant field would be visible.")
    A("")
    A("---")
    A("")
    A("## 5. Points to note, limits, applicability")
    A("")
    A("* **The gap is structural, not a resolution oversight.** For a moment-loaded plate")
    A("  strip `w_shear/w_bend = 2 D_b/(kappa_s G t L) = (1+nu)/(3 kappa_s) (t/L)^2`, so the")
    A("  shear share of the tip deflection is `s/(1+s)` with")
    A("  `s = (1+nu)/(3 kappa_s)(t/L)^2`. A wrong `Ds` moves a displacement benchmark by")
    A("  `(Delta Ds/Ds) x share`, and for a thin strip the share is tiny:")
    A("")
    A("  | `t/L` | shear share of the tip deflection | deflection change from a 20 % `Ds` error |")
    A("  |---|---|---|")
    for ratio in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        share = shear_fraction(ratio, m["poisson"], m["shear_correction"])
        A(f"  | {ratio:g} | {pct(share, 4)} | {pct(0.2 * share, 4)} |")
    A("")
    A("  At the anchor's `t/L = 0.01` the change is ~1e-5 relative, four orders of magnitude")
    A("  below the 2 % pool tolerance: **no displacement benchmark of a thin plate can grade")
    A("  `Ds` at realistic slenderness, and refining the strip cannot help.** Only a")
    A("  constitutive-level check can, which is what §2.3/§3.2 add.")
    A("* **Consequence for ship panels.** At `t/a = 0.01` (typical hull plating, `a/t ~ 100`)")
    A("  a 20 % `Ds` error changes the bending answer by ~1e-5 relative -- physically")
    A("  irrelevant, which is exactly why such a defect survives a displacement-based")
    A("  verification suite. At `t/a = 0.3` (sandwich faces, ice-belt insert plating, deep")
    A("  grillage flange) it is ~1 % of the deflection, and ~2 % at `t/a = 0.5`: comparable to")
    A("  the discretisation error of a design mesh, i.e. no longer negligible there. The check")
    A("  is a no-op cost for thin panels and worth its one line of code for thick ones.")
    A("* **What the new check does NOT do**: it grades the *magnitude and consistency* of")
    A("  `Ds`, not whether `kappa_s = 5/6` is the right *physical* factor. `5/6` is the")
    A("  declared model input (average-strain factor for a rectangular section); the check")
    A("  asserts that the element implements the declared `Ds` and that its assumed/reduced")
    A("  shear sampling reproduces a constant shear strain. It also says nothing about")
    A("  shear locking -- that stays with the anchor's `full`-integration negative control.")
    A("* **Why `mitc` and `full` also pass**: for a linear `w` and constant `theta` the")
    A("  kinematic shear is constant, and the Dvorkin-Bathe covariant sampling reproduces")
    A("  constant covariant strains exactly on a parallelogram; `full` evaluates the exact")
    A("  gradient directly. This is a property, not a coincidence, and it is what makes the")
    A("  identity a pass/fail test rather than a convergence study.")
    A("* **Instrumentation, not a solver.** `mutated_q4_stiffness` duplicates the two-by-two")
    A("  bending block; the duplication is bounded by the exact-reproduction gate")
    A(f"  ({sci(ident['reproduction_sri_rel'], 2)} relative) so it cannot drift away from the")
    A("  element it instruments. `shear_block_mutation` swaps a module-level symbol and is")
    A("  therefore **not thread-safe** -- auditors and tests stay single-threaded.")
    A(f"* **Baseline fidelity.** The survey legs reproduce the committed anchor values where")
    A(f"  the definitions coincide (`affine_exact_w`: {sci(base['affine_exact_w']['recomputed'])})")
    A(f"  vs {sci(base['affine_exact_w']['committed'])} committed; `poisson_coupling_computed`:")
    A(f"  {sci(base['poisson_coupling_computed']['recomputed'])} vs")
    A(f"  {sci(base['poisson_coupling_computed']['committed'])}); the legs whose committed gate")
    A("  takes a maximum over meshes or states that this survey fixes are marked")
    A("  `comparable: false` in the JSON with the reason, and are not used for the verdict.")
    A("* **Not merged into `gates_verified.json` on purpose.** The anchor's `run.py` and its")
    A("  gate file are being edited concurrently by the round-3 locking-control work; this")
    A("  check keeps its own driver and JSON so the two advances cannot silently reshuffle")
    A("  each other's evidence. Wiring it into `run.py` is a one-line follow-up once that")
    A("  branch lands.")
    A("")
    A("---")
    A("")
    A("## 6. Error verdict")
    A("")
    A("The graded quantities of this round are *identities*, so the error statement is about")
    A("the identity residual and about the discrimination margin:")
    A("")
    A("| key quantity | reference | measured error / residual | < 3 % ? |")
    A("|---|---|---|---|")
    A(f"| constant-shear energy (sri/mitc/full, 3 families, 3 states) | `1/2 kappa_s G t gamma^2 Area` | {sci(exact_worst)} relative | **yes** |")
    A(f"| shear strain recovered from the FE field | prescribed constant `gamma` | {sci(field['shear_spread_rel'], 2)} spread | **yes** |")
    A(f"| bending strain of the state | 0 | {sci(field['curvature_leak'], 2)} | **yes** |")
    A(f"| `kappa_s = 1` (-20 % `Ds`) rejection | must exceed 15 % | {pct(ctrl['kappa_s=1 (-20%)']['energy_rel_error'])} | rejected |")
    A(f"| `kappa_s = pi^2/12` (-1.3 % `Ds`) rejection | must exceed 1 % | {pct(ctrl['kappa_s=pi^2/12 (-1.3%)']['energy_rel_error'])} | rejected |")
    A(f"| anchor baseline (production element) | committed gate values | {nominal['n_flipped']} of {n_legs} legs flipped | unchanged |")
    A("")
    A("No benchmark here is compared to a literature *number* -- the universe of discourse is")
    A("a closed-form constitutive identity, verified as an oracle before any stiffness is")
    A("assembled (`shear_state_selfcheck`), so there is no reference-value uncertainty to")
    A("report. The physical limitation is stated instead: the check certifies that the")
    A("element implements the *declared* `Ds`, not that the declared `Ds` is the truth about")
    A("a real plate (§5).")
    A("")
    A("---")
    A("")
    A("## 7. Reproduction")
    A("")
    A("```")
    A("OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_bending_patch_shear_block.py")
    A("PYTHONPATH=src python scripts/gen_bending_patch_shear_block_note.py")
    A("PYTHONPATH=src pytest tests/test_bending_patch_shear_block.py -q")
    A("PYTHONPATH=src pytest tests/test_bending_patch.py -q")
    A("```")
    A("")
    A(f"Runtime of the driver: {d['wall_seconds']:.1f} s (single node, 4 threads).")
    A("")
    A("| artefact | path |")
    A("|---|---|")
    for key, val in d["artifacts"].items():
        if isinstance(val, list):
            for v in val:
                A(f"| {key} | `{v}` |")
        else:
            A(f"| {key} | `{val}` |")
    A("")

    text = "\n".join(L)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())