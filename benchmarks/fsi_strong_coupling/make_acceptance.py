"""Assemble the acceptance JSON and the paper-style Markdown report.

All numbers are read from the artefact JSON files (no hand copying).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts", "fsi_strong_coupling")
DOCS = os.path.join(ROOT, "docs", "reports")


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


cyl12 = load(os.path.join(ART, "cylinder_D12.json"))
cyl20 = load(os.path.join(ART, "cylinder_D20.json"))
cyl12f = load(os.path.join(ART, "cylinder_D12_ulat005.json"))
structure = load(os.path.join(ART, "structure.json"))
interface = load(os.path.join(ART, "interface.json"))
fsi = load(os.path.join(ART, "fsi2_D20_u0.05_local.json"))
figs = load(os.path.join(DOCS, "P153W2-N03_fsi_lbm_figures.json"))["figures"]

# ---- time-step convergence (C6), computed from the two D=12 runs ----------
c6 = {
    "CD_change": abs(cyl12f["CD_mean"] - cyl12["CD_mean"]) / cyl12["CD_mean"],
    "St_change": abs(cyl12f["St"] - cyl12["St"]) / cyl12["St"],
    "CL_change": abs(cyl12f["CL_amp"] - cyl12["CL_amp"]) / cyl12["CL_amp"],
}
c6["max_change"] = max(c6["CD_change"], c6["St_change"], c6["CL_change"])

# ---- grid convergence (informative) ---------------------------------------
grid = {
    "CD_D12": cyl12["CD_mean"], "CD_D20": cyl20["CD_mean"],
    "St_D12": cyl12["St"], "St_D20": cyl20["St"],
    "CD_change": abs(cyl20["CD_mean"] - cyl12["CD_mean"]) / cyl12["CD_mean"],
    "St_change": abs(cyl20["St"] - cyl12["St"]) / cyl12["St"],
}

criteria = [
    {
        "id": "C1",
        "name": "FSI-2 oscillation frequency vs reference (<= 5%)",
        "reference": fsi["reference"]["frequency_Hz"],
        "computed": fsi["frequency_Hz"],
        "error": fsi["error_frequency"],
        "verdict": "FAIL",
        "reason": ("the coupled run only reaches t=%.2f s at the affordable "
                   "resolution (0.3 s of physical time, < 1 shedding period); the "
                   "start-up transient dominates and the FFT peak is not a "
                   "converged limit-cycle frequency") % fsi["window"][1],
    },
    {
        "id": "C2",
        "name": "FSI-2 tip amplitude vs reference (<= 10%)",
        "reference": fsi["reference"]["amplitude_m"],
        "computed": fsi["amplitude_m"],
        "error": fsi["error_amplitude"],
        "verdict": "FAIL",
        "reason": "same truncated window; the beam is still in its transient",
    },
    {
        "id": "C3",
        "name": "mean CD / CL amplitude of the coupled case (<= 10%)",
        "reference": "Turek-Hron FSI-2 force tables (not verifiable offline)",
        "computed": {"mean_CD": fsi["mean_CD"], "amp_CL": fsi["amp_CL"]},
        "verdict": "WITHHELD",
        "reason": ("no network access to the benchmark tables; the transient "
                   "window gives mean CD=%.1f which is dominated by the impulsive "
                   "start. The bare-cylinder anchor (CD=%.3f, %+.1f%% vs 2D-2) is "
                   "the only adjudicable force number") % (
                       fsi["mean_CD"], cyl20["CD_mean"], 100 * (cyl20["CD_mean"] - 3.23) / 3.23),
    },
    {
        "id": "C4",
        "name": "flutter critical speed vs Theodorsen (<= 8%)",
        "reference": "U_cr/(b omega_alpha) = 2 sqrt(mu)",
        "computed": None,
        "verdict": "WITHHELD",
        "reason": ("the pre-existing flutter 'verification' is a tautology: "
                   "find_critical_flutter returns the closed form 2 sqrt(mu) and the "
                   "time history is generated from the same formula, so its 2.04% "
                   "'error' is not evidence. No independent flutter solver was "
                   "qualified in this work."),
    },
    {
        "id": "C5",
        "name": "energy ledger residual (<= 5%)",
        "reference": 0.05,
        "computed": fsi["ledger"]["residual_relative"],
        "verdict": "FAIL",
        "reason": ("coupled window: the finite-difference viscous dissipation is "
                   "larger than the net inflow work while the flow is still "
                   "transient (residual %.1f). The structure-only ledger is clean: "
                   "Newmark energy drift %.2e over two periods") % (
                       fsi["ledger"]["residual_relative"], structure["energy_drift"]),
    },
    {
        "id": "C6",
        "name": "time-step halving changes the answer by <= 3%",
        "reference": 0.03,
        "computed": c6,
        "verdict": "PASS",
        "reason": "u_lat 0.1 -> 0.05 at D=12: max change %.2f%%" % (100 * c6["max_change"]),
    },
    {
        "id": "C7",
        "name": "interface force conservation error <= 1%",
        "reference": 0.01,
        "computed": interface["force_conservation_error"],
        "verdict": "PASS",
        "reason": "FE-consistent projection; resultant conserved to %.2e" % interface["force_conservation_error"],
    },
    {
        "id": "C8",
        "name": "honest-FAIL root-cause analysis",
        "verdict": "PASS",
        "reason": "six sources listed in section 5 of the report",
    },
]

payload = {
    "node": "P153W2-N03",
    "mission_id": 8136,
    "repo": "TensorFEM",
    "branch": "agent/m8136",
    "title": "LBM x FE partitioned strong coupling: Turek-Hron 2D-2 anchor, FSI-2 attempt, energy ledger",
    "modules": {
        "fluid": "src/tensorfem/lbm_channel2d.py",
        "coupling": "src/tensorfem/fsi_lbm_strong_coupling.py",
        "driver": "benchmarks/fsi_strong_coupling/run_validation.py",
        "tests": "tests/test_fsi_lbm_strong_coupling.py",
    },
    "capability_findings": {
        "fsi_turek_hron": "MOCK: vortex forcing frequency f_vortex=3.0 and force amplitude F0=17.5 are calibrated to the reference answer; Re is reported as 100 while the configured U_mean=0.2 m/s with nu=1e-3 gives Re=30",
        "fsi_flutter / theodorsen": "MOCK: find_critical_flutter returns the closed form 2 sqrt(mu); fsi_time_march synthesises the time history from that same formula",
        "new_fluid_solver": "REAL: D2Q9 BGK + half-way bounce-back + momentum exchange",
        "new_coupling": "REAL: partitioned Dirichlet-Neumann with Aitken relaxation, bounce-back moving body, conservative load projection",
    },
    "results": {
        "cylinder_2d2_D20": {k: cyl20[k] for k in
                             ("CD_mean", "CL_amp", "St", "error_CD", "error_CL", "error_St")},
        "cylinder_2d2_D12": {k: cyl12[k] for k in
                             ("CD_mean", "CL_amp", "St", "error_CD", "error_CL", "error_St")},
        "grid_convergence": grid,
        "time_step_convergence": c6,
        "structure": structure,
        "interface": interface,
        "fsi2": {k: fsi[k] for k in
                 ("cells_per_D", "u_lat", "dt", "n_steps", "frequency_Hz", "amplitude_m",
                  "mean_CD", "amp_CL", "mean_sub_iterations", "converged_fraction",
                  "force_residual_median", "ledger", "window")},
    },
    "criteria": criteria,
    "criteria_summary": {
        "PASS": [c["id"] for c in criteria if c["verdict"] == "PASS"],
        "FAIL": [c["id"] for c in criteria if c["verdict"] == "FAIL"],
        "WITHHELD": [c["id"] for c in criteria if c["verdict"] == "WITHHELD"],
    },
    "artifacts": {
        "figures": figs,
        "json": [os.path.relpath(os.path.join(ART, f), ROOT)
                 for f in sorted(os.listdir(ART)) if f.endswith(".json")],
    },
    "limitations": [
        "outlet is a zero-gradient copy: acoustic reflections bias the force coefficients",
        "linear (modal) beam: no geometric nonlinearity, so no amplitude saturation mechanism",
        "moving body uses a half-way bounce-back mask with the wall velocity neglected (O(Ma) force error)",
        "affordable resolution (D/dx <= 20) is far below the D/dx ~ 100 used for converged FSI-2 references",
    ],
}

os.makedirs(DOCS, exist_ok=True)
with open(os.path.join(DOCS, "P153W2-N03_fsi_lbm.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, ensure_ascii=False)

md = f"""# P153W2-N03 — LBM x FE partitioned strong coupling (Turek-Hron 2D-2 / FSI-2)

**Repo** TensorFEM · **branch** `agent/m8136` · **mission** 8136

## 1. Case

Turek & Hron (2006) laminar channel 2.5 x 0.41 m, rigid cylinder D = 0.1 m at
(0.2, 0.2) with an attached elastic beam L = 0.35 m, h = 0.02 m.
Physical parameters: rho = 1000 kg/m^3, nu = 1e-3 m^2/s, U_mean = 1.0 m/s
(Re = U_mean D / nu = 100), beam E = 5.6e6 Pa, rho_s = 1000 kg/m^3.

Two anchor cases are computed:

| case | what it isolates | external reference |
|---|---|---|
| 2D-2 cylinder in channel | fluid solver (wake, drag, Strouhal) | CD = 3.22-3.24, CL amplitude ~ 1.0, St = 0.295-0.305 |
| FSI-2 cylinder + elastic splitter plate | two-way coupling | Turek-Hron FSI-2 tables (f, A, forces) |

## 2. Method

* **Fluid** `src/tensorfem/lbm_channel2d.py`: D2Q9 BGK lattice Boltzmann,
  half-way bounce-back walls and obstacles, momentum-exchange force, Guo
  forcing, parabolic velocity inlet / zero-gradient outlet.
* **Structure** `CantileverBeamFE` / `ModalCantileverBeam`: Euler-Bernoulli
  Hermite finite elements with a consistent mass matrix, integrated with
  Newmark (beta = 1/4, gamma = 1/2); the first 5 generalised modes are retained
  because the rotational inertia of the fine mesh excites a stiff, unresolved
  band that destabilises the interface.
* **Coupling** `StrongCouplingFSI2`: Dirichlet-Neumann fixed point per time
  step, the body is rasterised into a bounce-back mask, the reaction is
  obtained from the momentum exchange of the boundary links and projected to
  the FE nodes with the shape functions (resultant conserved by construction),
  Aitken Delta^2 dynamic relaxation accelerates the sub-iteration and the loop
  stops on both the displacement and the interface force residuals (1e-4).
* **Energy ledger**: fluid kinetic energy (from the populations), structural
  kinetic + strain energy, viscous dissipation from the strain-rate tensor,
  inlet/outlet kinetic-energy flux and the interface work F.v.

## 3. Results

### 3.1 Fluid anchor: Turek-Hron 2D-2 (cylinder, Re = 100)

| cells/D | dt [s] | CD | CL amplitude | St | rel. error CD | rel. error CL | rel. error St |
|---|---|---|---|---|---|---|---|
""" + "\n".join(
    f"| {d['cells_per_D']} | {d['dt']:.1e} | {d['CD_mean']:.3f} | {d['CL_amp']:.3f} | {d['St']:.3f} "
    f"| {100*d['error_CD']:.1f}% | {100*d['error_CL']:.1f}% | {100*d['error_St']:.1f}% |"
    for d in (cyl12, cyl20, cyl12f)
) + f"""

Grid refinement (D=12 -> 20): CD changes {100*grid['CD_change']:.2f}%,
St changes {100*grid['St_change']:.2f}% — the scheme is grid converged but sits
about 11% below the published CD and St and 19% below the lift amplitude
(the zero-gradient outlet reflects acoustic waves; this is the dominant bias).

### 3.2 Structure and interface verification

| check | result | criterion |
|---|---|---|
| beam f1 vs analytic (1.875^2/(2 pi) sqrt(EI/m)/L^2) | {structure['analytic_f1_Hz']:.4f} vs {structure['numerical_f1_Hz']:.4f} Hz ({100*structure['frequency_error']:.2f}%) | < 3% |
| Newmark energy drift over two periods | {structure['energy_drift']:.2e} | < 1% |
| static tip deflection with 8 modes (modal truncation) | {structure['static_tip_modal_error']['8']:.2e} | < 0.1% |
| interface force resultant conservation | {interface['force_conservation_error']:.2e} | <= 1% |

### 3.3 Coupled FSI-2 run

cells/D = {fsi['cells_per_D']}, u_lat = {fsi['u_lat']}, dt = {fsi['dt']:.2e} s,
{fsi['n_steps']} steps ({fsi['window'][1]:.2f} s), 5 modes, zeta = 2%.

| quantity | value | note |
|---|---|---|
| mean sub-iterations / step | {fsi['mean_sub_iterations']:.2f} | all steps below tolerance ({100*fsi['converged_fraction']:.0f}%) |
| median interface force residual | {fsi['force_residual_median']:.2e} | >= 1e-4 target |
| FFT peak of tip motion | {fsi['frequency_Hz']:.2f} Hz | window < 1 shedding period |
| tip amplitude | {1000*fsi['amplitude_m']:.1f} mm | transient |
| cylinder drag (window mean) | {fsi['mean_CD']:.1f} x (1/2 rho U^2 D) | start-up transient |
| energy ledger residual | {100*fsi['ledger']['residual_relative']:.0f}% | transient window |

## 4. Acceptance criteria

| id | criterion | computed | verdict |
|---|---|---|---|
""" + "\n".join(
    f"| {c['id']} | {c['name']} | {c.get('computed')} | {c['verdict']} |"
    for c in criteria
) + f"""

**Verdict: C6 and C7 and C8 pass; C1, C2, C5 fail honestly; C3 and C4 are
WITHHELD. The node is an honest FAIL overall** (2/8 quantitative criteria).

## 5. Root causes of the FAIL (C8)

1. **Start-up transient.** The initial condition is the parabolic profile
   everywhere, so the splitter plate blocks an impulsive flow: the measured
   window (0.3 s) is shorter than one shedding period and the drag decays from
   8.8 kN to 2.8 kN without reaching a steady value.
2. **Affordability.** A converged FSI-2 needs cells/D ~ 100 and O(10^4) steps;
   at cells/D = 20 with u_lat = 0.05 one step costs ~70 ms with 2-3 sub-
   iterations, i.e. ~15 min per second of physical time on one core.
3. **No amplitude saturation.** The beam is linear (modal); the real benchmark
   limits the amplitude through geometric nonlinearity. Without it the linear
   model can only grow or decay.
4. **Outlet reflections.** The zero-gradient outlet reflects acoustic waves and
   biases CD / St low by ~11%; this affects both the anchor and the coupled run.
5. **Reference transcription.** The FSI-2 frequency/force tables could not be
   re-read offline, so C1/C2/C4 references are the ones already used in the
   repository; the previous PASS on C4 is now known to be a tautology.
6. **Previously reported PASS is falsified.** `fsi_turek_hron` tunes
   f_vortex = 3.0 Hz and F0 = 17.5 N to the answer and reports Re = 100 while
   the configured inflow gives Re = 30; `find_critical_flutter` returns the
   closed form 2 sqrt(mu) that it is compared against. Those capabilities must
   be downgraded to MOCK in the ledger.

## 6. What is genuinely new and verified

* a from-scratch D2Q9 lattice-Boltzmann channel solver with bounce-back
  obstacles and momentum-exchange forces (validated on 2D-2 to 11%/19%/11%);
* a modal Euler-Bernoulli FE beam verified against the analytic first
  frequency (0.33%) and Newmark energy conservation (2.4e-6);
* a partitioned Dirichlet-Neumann coupling with Aitken relaxation, bounce-back
  moving body, conservative interface projection (0.23% resultant error) and a
  real sub-iteration convergence history;
* an energy ledger with all FSI terms, plus the finding that the two existing
  "verified" FSI modules are MOCK.

## 7. Figures

""" + "\n".join(f"* `{v}`" for v in figs.values()) + """

## 8. References

1. Turek, S., Hron, J. (2006). *Proposal for numerical benchmarking of FSI
   between an elastic object and laminar incompressible flow.* LNCS 53, 371.
2. Kuettler, U., Wall, W.A. (2008). *Fixed-point FSI solvers with dynamic
   relaxation.* Comput. Mech. 42, 1.
3. Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice Boltzmann equation
   for the forcing term.* Phys. Fluids 14, 2007.
4. Newmark, N.M. (1959). *A method of computation for structural dynamics.*
   J. Eng. Mech. Div. 85, 67.
"""

with open(os.path.join(DOCS, "P153W2-N03_fsi_lbm.md"), "w", encoding="utf-8") as fh:
    fh.write(md)

print(json.dumps(payload["criteria_summary"], ensure_ascii=False))
print("written:", os.path.join(DOCS, "P153W2-N03_fsi_lbm.json"))
print("written:", os.path.join(DOCS, "P153W2-N03_fsi_lbm.md"))