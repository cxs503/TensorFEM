# Phase-field fracture (AT2) in TensorFEM — implementation and verification face

Module: `src/tensorfem/phase_field_fracture.py` · suite: `benchmarks/phase_field_fracture/run_suite.py`
figures: `benchmarks/phase_field_fracture/make_figures.py` · artifacts: `artifacts/phase_field_fracture/`
tests: `tests/test_phase_field_fracture.py`

## 1 Case introduction (physics, geometry, parameters)

TensorFEM's structural path carried linear elasticity, cohesive zones and J2 plasticity but **no
diffuse damage evolution**. This node adds the regularised variational fracture (AT2) model, which
tracks crack initiation, propagation and kinking *without* any a-priori crack path:

E(u, d) = ∫ g(d) ψ⁺(ε) dV + ∫ ψ⁻(ε) dV + ∫ G_c [ d²/(2l) + (l/2)|∇d|² ] dV ,
  g(d) = (1−d)² + k,  ψ⁺ = ½K⟨tr ε⟩² + μ ε_dev:ε_dev,  ψ⁻ = ½K⟨−tr ε⟩²  (Amor split, plane strain)
  H(x,t) = max_{s≤t} ψ⁺(ε(x,s))  (history variable ⇒ ḋ ≥ 0)

All runs are 2-D plane strain, E = 1000 MPa, ν = 0.3, G_c = 1.0e-3 N/mm (K_Ic = √(G_c E′) = 1.048
MPa√mm), thickness 1 mm, Q4 elements, 2×2 Gauss.

| case | geometry | settings |
|---|---|---|
| SENT / K_I (C1) | W = 1, 2H = 2, notch a = 0.3W, sharp slit, uniform traction σ = 1 | nx = 60, h = 1/60, Δa = h |
| SENT propagation (C4–C6) | W = 1, 2H = 2, a = 0.3W, prescribed end displacement | h = l/2, Δδ = 2e-5, 30 steps |
| SE(B) three-point bend (C2) | S = 4W, W = 1, a = 0.45W, δ-controlled | h = 0.04, l = 0.04 |
| inclined crack (C3) | 1×1 plate, crack at β = 45°, a = 0.35 | h = 0.02, l = 0.04 |

## 2 Method / tool chain

* Q4 kinematics reuse the production conventions (`continuum._b_matrix` order); the element
  stiffness of an undamaged element reproduces `continuum.q4_stiffness` to 1e-6 relative
  (regression test), so the new module is anchored to the existing plane kernel.
* Residuals `R_u = ∫Bᵀ(g σ⁺ + σ⁻) − f`, `R_d = ∫[(G_c/l)d + G_c l ∇d·∇N + g′(d) H]N` with the
  **consistent (non-symmetric) tangent**; verified against central differences of the residual
  (relative error 3e-8, `test_consistent_tangent_matches_finite_differences`).
* Drivers: monolithic Newton (line search, history floor `max(H, ψ⁺)`) and staggered
  (alternate minimisation). Global sparse LU (SciPy SuperLU), PyTorch element vectorisation.
* C1 uses a **true slit** (void element row, `void_elements`) and the potential-energy
  finite-difference method `G = −(1/t) dΠ/da` with central differences of width h.
* Post-processing: reactions → load, ASTM E399 shape factor for SE(B), Tada shape factor for
  SENT, maximum-tangential-stress kink angle, energy ledger
  `W_ext = ∫[gψ⁺+ψ⁻]dV + ∫G_c γ dV` (fracture energy referenced to the pre-existing notch).

## 3 Results (all numbers from `artifacts/phase_field_fracture/*.json`)

| index | quantity | computed | reference | deviation | verdict |
|---|---|---|---|---|---|
| C1 | K_I (SENT, a/W = 0.3, nx = 60) | 1.6217 MPa√mm | 1.6115 (Tada) | **+0.63 %** | **PASS (≤3 %)** |
| C1 mesh | nx = 20 / 30 / 40 | 2.58 / 1.49 / 1.04 % | – | monotone O(h) | PASS |
| C2 | SE(B) fracture load → K | K = 0.363 | K_Ic = 1.048 | **−65.3 %** | **FAIL (≤5 % not met)** |
| C4 | G_app from the critical load (l = 0.08 / 0.05 / 0.032) | 1.0273e-3 | G_c = 1.0e-3 | **+2.73 %** (all l) | partial (see §5) |
| C5 | energy ledger, last step, l = 0.032 (stable path) | 0.17 % | ≤2 % | – | PASS for the stable run |
| C5 | energy ledger, l = 0.08 / 0.05 (global failure, snap-back) | >100 % | ≤2 % | – | FAIL |
| C6 | history irreversibility violations (all runs) | **0** | 0 | – | PASS |
| C7 | Newton residual slope | not captured | ≈2 | – | **not demonstrated** |
| C3 | kink angle (β = 45°) | not resolvable (0 ridge nodes) | MTS −53.13° | – | **FAIL / not determined** |

C1 detail: U and f·u of the three notch lengths (a−h, a, a+h) give a central-difference
dΠ/da = −2.393e-3 N/mm ⇒ G = 2.393e-3 N/mm ⇒ K = √(G E′) = 1.6217 vs 1.6115.
The single-element-thick slit (odd row count so that y = 0 is an element centre line) removes the
+1.2 % bias that a two-element-thick slit produces.

C4 detail: the displacement-controlled SENT reaches crack initiation at δ_c = 6.0e-4 mm for every
l, i.e. σ_c = 0.659 MPa, G_app = K(σ_c)²/E′ = 1.027e-3 = 1.027 G_c (+2.7 %). The deviation is
**identical for all three l** (difference < 0.1 %), so the *apparent toughness is l-independent*
at this resolution: the AT2 model is already in its sharp-interface regime. The literal reading of
C4 (monotone shrinking deviation with l → 0) is therefore **not demonstrated**; what is
demonstrated is l-independence within 0.1 % with a +2.7 % offset.

## 4 Figures

`artifacts/phase_field_fracture/figures/`
* `sent_load_displacement.png` — P–δ of the SENT for the three band widths (peak ≈ 0.11 N at
  δ = 6e-4, post-peak softening; the widest band softens earliest).
* `l_convergence.png` — G_app/G_c versus l with the deviation bars.
* `newton_history.png` — residual histories per load step (staggered driver, see §5/C7).
* `damage_inclined_crack.png` — damage contour of the inclined-crack specimen (produced only if
  the C3 run completed).

## 5 Pitfalls, limitations, honest statement

1. **C2 was measured wrongly, not necessarily computed wrongly.** The SE(B) case reports the load
   at which the damage at the notch-tip neighbourhood exceeds 0.6; with the prescribed fully
   damaged notch the tip node is *already* d = 1, so the criterion fires on the first small
   increment and the recorded "fracture load" is an initiation artefact (65 % low). A correct
   peak-load identification needs (a) an evolving-tip detection based on d ahead of the notch and
   (b) a load-step refinement near the peak. Reported as **honest FAIL**; the machinery
   (geometries, ASTM post-processing, energy ledger) is in place.
2. **C7 not measured.** The propagation runs use the staggered driver, whose record stores the
   iteration count, not the residual norms; the monolithic Newton residual history was not stored
   for a converged step in this window, so quadratic convergence is **not claimed**. The tangent
   itself *is* verified to be consistent (finite-difference test), which is the prerequisite.
3. **Energy ledger** is exact for the stable path (0.17 % at l = 0.032) but the trapezoidal work
   integration breaks once the specimen fails globally (snap-back at l = 0.08/0.05): the reaction
   drops by a factor 5 between two increments. Smaller increments (or arc-length control) are
   required there.
4. **Plane strain only.** The Amor volumetric–deviatoric split is implemented with ε_zz = 0 so the
   energy identity holds exactly; plane stress (ship plating) needs the stress-based split and is
   not provided.
5. **History variable uses the split of the current strain** (Amor) with the tension-only driving
   force; compression damage is excluded by construction, so compressive crushing (ice–structure
   contact) is out of scope.
6. AT2 has no intrinsic strength parameter: for l comparable to the notch length the model
   predicts a much lower peak load than LEFM (this is why the SENT l study uses l ≤ 0.08 with
   a = 0.3).
7. Symmetry was avoided (full models) because a half model puts the crack on a free boundary and
   halves the dissipated surface energy.

## 6 Error verdict (explicit)

* **C1 — is the error < 3 %?** **Yes: +0.63 %** (K_I 1.6217 vs 1.6115, Tada finite-width SENT).
* **C2 — within 5 %?** **No: −65.3 %** (honest FAIL, cause in §5.1).
* **C3 — within 15°?** **No: the damage ridge test found no node with d > 0.7 in the 0.02–0.25 band ahead of the initial tip** (the β = 45° run finished 30 increments and the specimen failed globally, d_max = 1, so the path measure degenerated). Reported as honest FAIL; the kink post-processing and the MTS reference (−53.13°) are implemented and reproduce the textbook value.
* **C4 — monotone convergence to G_c?** Deviation +2.73 % at every l (l-independence < 0.1 %), so
  the convergence is uniform rather than monotonically shrinking → **partially met, not claimed as
  full pass**.
* **C5 — ledger ≤ 2 %?** **Yes for the stable run (0.17 %)**; no for the snap-back runs.
* **C6 — irreversibility violations = 0?** **Yes (0 violations in all 90 converged states).**
* **C7 — slope ≈ 2?** **Not measured** → not claimed.

## 7 Capability statement

New capability: `phase_field_fracture` (AT2 damage, history irreversibility, monolithic
Newton + staggered solvers, slit/void crack representation, energy ledger, SENT and SE(B)
post-processing). Verified anchors: Q4 stiffness regression, tangent consistency, K_I ≤ 1 %,
irreversibility, stable-path energy balance. Unverified/open: SE(B) peak load, kink angle,
Newton order, frictional/compressive damage, plane stress, 3-D.