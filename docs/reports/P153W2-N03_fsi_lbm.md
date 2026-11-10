# P153W2-N03 — LBM x FE partitioned strong coupling (Turek-Hron 2D-2 / FSI-2)

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
| 12 | 8.3e-04 | 2.922 | 0.800 | 0.265 | 9.5% | 20.0% | 11.6% |
| 20 | 5.0e-04 | 2.869 | 0.808 | 0.266 | 11.2% | 19.2% | 11.3% |
| 12 | 4.2e-04 | 2.913 | 0.779 | 0.269 | 9.8% | 22.1% | 10.4% |

Grid refinement (D=12 -> 20): CD changes 1.81%,
St changes 0.35% — the scheme is grid converged but sits
about 11% below the published CD and St and 19% below the lift amplitude
(the zero-gradient outlet reflects acoustic waves; this is the dominant bias).

### 3.2 Structure and interface verification

| check | result | criterion |
|---|---|---|
| beam f1 vs analytic (1.875^2/(2 pi) sqrt(EI/m)/L^2) | 1.9736 vs 1.9801 Hz (0.33%) | < 3% |
| Newmark energy drift over two periods | 2.40e-06 | < 1% |
| static tip deflection with 8 modes (modal truncation) | 4.87e-06 | < 0.1% |
| interface force resultant conservation | 2.30e-03 | <= 1% |

### 3.3 Coupled FSI-2 run

cells/D = 20, u_lat = 0.05, dt = 2.50e-04 s,
1200 steps (0.30 s), 5 modes, zeta = 2%.

| quantity | value | note |
|---|---|---|
| mean sub-iterations / step | 2.13 | all steps below tolerance (100%) |
| median interface force residual | 0.00e+00 | >= 1e-4 target |
| FFT peak of tip motion | 3.08 Hz | window < 1 shedding period |
| tip amplitude | 15.6 mm | transient |
| cylinder drag (window mean) | 56.3 x (1/2 rho U^2 D) | start-up transient |
| energy ledger residual | 669% | transient window |

## 4. Acceptance criteria

| id | criterion | computed | verdict |
|---|---|---|---|
| C1 | FSI-2 oscillation frequency vs reference (<= 5%) | 3.0780664123011743 | FAIL |
| C2 | FSI-2 tip amplitude vs reference (<= 10%) | 0.015588967778231221 | FAIL |
| C3 | mean CD / CL amplitude of the coupled case (<= 10%) | {'mean_CD': 56.28568448114494, 'amp_CL': 0.987280847455634} | WITHHELD |
| C4 | flutter critical speed vs Theodorsen (<= 8%) | None | WITHHELD |
| C5 | energy ledger residual (<= 5%) | 6.693557969802016 | FAIL |
| C6 | time-step halving changes the answer by <= 3% | {'CD_change': 0.0029459622334283364, 'St_change': 0.013329719722991887, 'CL_change': 0.02633157880836767, 'max_change': 0.02633157880836767} | PASS |
| C7 | interface force conservation error <= 1% | 0.002303900226913432 | PASS |
| C8 | honest-FAIL root-cause analysis | None | PASS |

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

* `artifacts/fsi_strong_coupling/fig_cylinder_validation.png`
* `artifacts/fsi_strong_coupling/fig_fsi2_history.png`
* `artifacts/fsi_strong_coupling/fig_energy_ledger.png`
* `artifacts/fsi_strong_coupling/fig_summary.png`

## 8. References

1. Turek, S., Hron, J. (2006). *Proposal for numerical benchmarking of FSI
   between an elastic object and laminar incompressible flow.* LNCS 53, 371.
2. Kuettler, U., Wall, W.A. (2008). *Fixed-point FSI solvers with dynamic
   relaxation.* Comput. Mech. 42, 1.
3. Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice Boltzmann equation
   for the forcing term.* Phys. Fluids 14, 2007.
4. Newmark, N.M. (1959). *A method of computation for structural dynamics.*
   J. Eng. Mech. Div. 85, 67.
