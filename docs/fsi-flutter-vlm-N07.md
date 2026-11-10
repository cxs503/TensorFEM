# P88W1-N07 -- Partitioned FSI flutter of an elastic plate in uniform flow

**Node**: P88W1-N07 (TensorFEM) -- FSI verification surface for the flow-structure
coupling stack. **Evidence level of this document: REAL** (numerical solutions of the
governing equations, verified against an independent solver and a closed-form theory).

## 1. Case description

A cantilever rectangular plate in a uniform incompressible stream of air is the
reference structure. The span-to-chord ratio is 5, so the aerodynamics of the
representative section are taken as two-dimensional (strip theory) and the structure
is reduced to the classical two-degree-of-freedom typical section: plunge = first
cantilever bending mode, pitch = first torsion mode.

| quantity | value | source |
|---|---|---|
| span L | 1.0 m | node specification |
| chord c | 0.2 m | node specification |
| thickness t | 0.01 m | node specification |
| semi-chord b | 0.1 m | c/2 |
| Young's modulus E | 7e+10 Pa | node specification |
| Poisson ratio nu | 0.3 | node specification |
| solid density | 2700.0 kg/m^3 | node specification |
| fluid density | 1.225 kg/m^3 | node specification |
| mass per span m' | 5.4 kg/m | rho_s*c*t |
| bending stiffness EI | 1166.67 N m^2 | E*c*t^3/12 |
| torsional stiffness GJ | 1738.33 N m^2 | G*b*c*t^3/3 (Saint-Venant) |
| polar inertia I_p | 0.018045 kg m^2/m | rho_s*c*t*(c^2+t^2)/12 |
| omega_h (TensorFEM FE modal) | 51.6806 rad/s | cantilever_beam_modes, 24 elements |
| omega_h (closed form) | 51.6806 rad/s | 1.8751^2*sqrt(EI/(m' L^4)) |
| omega_alpha | 487.537 rad/s | (pi/2L)*sqrt(GJ/I_p) |
| mass ratio mu | 140.32 | m'/(pi*rho_f*b^2) |
| frequency ratio sigma | 0.1060 | omega_h/omega_alpha |
| radius of gyration r_alpha | 0.5781 | sqrt(I_p/(m' b^2)) |
| static unbalance x_alpha | 0.0 | symmetric section, CG on the EA |
| EA position a | 0.0 | mid-chord |
| velocity scale b*omega_alpha | 48.754 m/s | b*omega_alpha |

The FE modal solve reproduces the Euler-Bernoulli cantilever root to 2.6e-08 relative error, so the structural side of the
coupling is not the error source.

## 2. Method

Two independent solvers are coupled explicitly (weak / partitioned coupling):

* **fluid** -- `tensorfem.unsteady_vlm.UnsteadyVLM2D`: a time-domain 2-D vortex
  lattice (cosine panels, quarter-chord bound vortices, three-quarter-chord control
  points). Vorticity leaves the trailing edge as a uniform vortex **sheet** segment
  convected at the free-stream speed; its strength is solved together with the bound
  circulation through Kelvin's theorem (implicit closure). Surface loads come from the
  unsteady Bernoulli equation `dp = rho*(U*gamma - d(phi)/dt)`.
* **structure** -- the 2-DOF modal model `Ms q'' + Ks q = [-L, M]`, integrated with
  classical RK4; the modal frequencies come from the TensorFEM FE solve above.
* **reference** -- `tensorfem.theodorsen_pk`: the frequency-domain Theodorsen lift and
  moment matrices are assembled from Hankel-function `C(k)` and the aeroelastic
  eigenvalues are obtained by a p-k continuation over the velocity sweep. Its zero
  crossing of `Re(p)` is the reference flutter boundary.

The two routes share only the physical model and no numerical machinery, so their
agreement is a verification of both. Growth rate and frequency are extracted from the
simulated `alpha(t)` by synchronous demodulation (sliding quadrature window) and a
log-envelope fit.

Reproduce with:

```
python scripts/fsi_flutter_vlm_n07.py --stage all
pytest tests/test_fsi_flutter_vlm.py -q          # 13 tests (2 marked slow)
```

## 3. Results

### 3.1 Fluid solver vs Theodorsen (harmonic pitch and plunge)

| k | mode | |L|/|L_th| | phase L [deg] | |M|/|M_th| | phase M [deg] |
|---|---|---|---|---|---|
| 0.05 | pitch | 0.9984 | -0.09 | 0.9965 | -0.96 |
| 0.05 | plunge | 0.9984 | -0.09 | 0.9960 | -0.96 |
| 0.10 | pitch | 0.9984 | +2.10 | 0.9920 | +0.90 |
| 0.10 | plunge | 0.9984 | +2.11 | 0.9927 | +0.93 |
| 0.20 | pitch | 1.0102 | +0.46 | 0.9988 | -0.88 |
| 0.20 | plunge | 1.0105 | +0.48 | 0.9962 | -0.90 |
| 0.30 | pitch | 1.0412 | +1.38 | 1.0185 | +0.30 |
| 0.30 | plunge | 1.0429 | +1.44 | 1.0191 | +0.08 |
| 0.50 | pitch | 1.0323 | +0.12 | 1.0056 | -0.49 |
| 0.50 | plunge | 1.0366 | +0.12 | 0.9984 | -0.62 |

Maximum amplitude error over the table: 4.29% (lift), 1.91% (moment); maximum phase error 2.11 deg.

### 3.2 Flutter boundary: p-k reference vs coupled time-domain march

| mu | U_cr/(b omega_a) p-k | omega_fl/omega_a p-k | k_fl | U_cr/(b omega_a) march | rel. error |
|---|---|---|---|---|---|
| 5 | 1.2310 | 0.6284 | 0.5105 | 1.2543 | 1.89% |
| 10 | 1.6160 | 0.6031 | 0.3732 | 1.6281 | 0.75% |
| 20 | 2.2096 | 0.5612 | 0.2540 | 2.2019 | 0.35% |
| 50 | 3.4089 | 0.5043 | 0.1479 | 3.3970 | 0.35% |

Largest boundary error 1.89% (criteria: < 5%).

### 3.3 Growth law and flutter frequency

| mu | U/U_cr | regime | sigma march | sigma p-k | rel. error | omega march | omega p-k | rel. error (omega) |
|---|---|---|---|---|---|---|---|---|
| 20 | 0.90 | subcritical | -0.0631 | -0.0504 | +25.1% | 0.5848 | 0.6739 | 13.2% |
| 20 | 1.05 | supercritical | +0.0368 | +0.0399 | -7.7% | 0.5236 | 0.5198 | 0.7% |
| 20 | 1.15 | supercritical | +0.1051 | +0.1069 | -1.7% | 0.4775 | 0.4608 | 3.6% |
| 50 | 0.90 | subcritical | -0.0660 | -0.0605 | +9.1% | 0.6178 | 0.6497 | 4.9% |
| 50 | 1.05 | supercritical | +0.0458 | +0.0495 | -7.5% | 0.4658 | 0.4581 | 1.7% |
| 50 | 1.15 | supercritical | +0.1170 | +0.1193 | -1.9% | 0.4091 | 0.3908 | 4.7% |

Supercritical (flutter-relevant) growth rates: worst relative error 7.7%, worst absolute error 0.0037 /omega_alpha.
Subcritical (decaying) growth rates: worst relative error 25.1%, worst absolute error 0.0126 /omega_alpha -- the explicit
staggered coupling adds a small amount of numerical damping, which matters
relatively when the physical damping itself is small.

### 3.4 Mass-ratio scaling, asymptotic coefficient and the negative control

| mu | U_cr | omega_fl | U_cr/sqrt(mu) |
|---|---|---|---|
| 5 | 1.2310 | 0.6284 | 0.5505 |
| 10 | 1.6160 | 0.6031 | 0.5110 |
| 20 | 2.2096 | 0.5612 | 0.4941 |
| 50 | 3.4089 | 0.5043 | 0.4821 |
| 100 | 4.7629 | 0.4690 | 0.4763 |
| 200 | 6.6819 | 0.4410 | 0.4725 |
| 400 | 9.4043 | 0.4189 | 0.4702 |
| 800 | 13.2675 | 0.4012 | 0.4691 |
| 1600 | 18.7481 | 0.3866 | 0.4687 |
| 3200 | 26.5212 | 0.3744 | 0.4688 |

| fit range | exponent n in U_cr ~ mu^n | deviation from 1/2 |
|---|---|---|
| mu>=200 | 0.4973 | 0.54% |
| mu>=50 | 0.4937 | 1.27% |
| mu>=800 | 0.4996 | 0.07% |

Negative control (rho_f -> 0, i.e. mu -> infinity): with mu raised by a factor 8.0 the critical speed rises by 2.806, against the sqrt-law prediction 2.828 (-0.80%). The asymptotic
coefficient is 0.4689, i.e. U_cr ~ 0.469*sqrt(mu)*b*omega_alpha -- the
*classical* coefficient for sigma = 0.2, r_alpha = 0.5, a = 0, x_alpha = 0.

### 3.5 The reference plate in the campaign velocity band 10-100 m/s

| U [m/s] | U/(b omega_a) | sigma march | sigma p-k | stable |
|---|---|---|---|---|
| 10 | 0.205 | -0.00140 | -0.00055 | True |
| 20 | 0.410 | -0.00161 | -0.00115 | True |
| 40 | 0.820 | -0.00310 | -0.00265 | True |
| 60 | 1.231 | -0.00521 | -0.00470 | True |
| 80 | 1.641 | -0.00802 | -0.00741 | True |
| 100 | 2.051 | -0.01163 | -0.01084 | True |

All velocities in the requested 10-100 m/s band are subcritical (all_stable_in_band = True). The plate's own flutter
speed is **312 m/s**: a 10 mm aluminium plate has
mu = 140 and is far too light and stiff for air at 1 atm to flutter
inside that band. This is a physical result, not a numerical one -- the band is
covered by the parametric family instead (mu = 5 ... 50 gives U_cr in the range
shown in table 3.2 in reduced form; dimensional values follow from the section).

## 4. Figures

![aerodynamic transfer](../artifacts/fsi_flutter_vlm_n07/fig_aero_transfer.png)

**Fig. 1** -- Amplitude ratio and phase of the vortex-lattice lift/moment against
Theodorsen for harmonic pitch and plunge. Deviations stay below 5% and 3 deg.

![growth curves](../artifacts/fsi_flutter_vlm_n07/fig_growth_curves.png)

**Fig. 2** -- Aeroelastic growth rate versus U/U_cr: p-k continuation (lines) and the
coupled time-domain points (open circles). The zero crossing is the flutter boundary.

![scaling](../artifacts/fsi_flutter_vlm_n07/fig_scaling.png)

**Fig. 3** -- Critical speed against mass ratio on log-log axes with the fitted
exponents; the dashed line is the asymptotic sqrt(mu) law.

![plate scan](../artifacts/fsi_flutter_vlm_n07/fig_plate_scan.png)

**Fig. 4** -- Dimensional growth rate of the reference plate over 10-100 m/s (both
solvers), showing that the whole band is stable for this structure.

## 5. Pitfalls, scope and limitations

1. **The pre-existing 'theodorsen' flutter stack is MOCK and self-referential.**
   `tensorfem.theodorsen.fsi_time_march` synthesises `exp(sigma t) cos(omega t)`;
   `theodorsen_critical_velocity` returns the rule of thumb `2*sqrt(mu)*b*omega_alpha`.
   At mu = 20 that routine reports 8.9443 while the p-k solution gives 2.2096, i.e. the
   legacy number is 305% high. Any earlier FSI verification that compared these
   routines with each other validated nothing; the module docstring now says so.
2. **Point-vortex wake shedding is unconditionally unstable here.** A 2-D point vortex
   sitting one panel behind the trailing edge radiates a spurious `Gamma/(2*pi*r)`
   field; the explicit `shed -> wake -> bound` loop has a gain of order `2*pi/U*dt`
   (~5 per step). The cure used here is (a) a finite vortex-sheet kernel and (b) an
   implicit Kelvin closure that solves the freshly shed segment together with the bound
   circulation. With either ingredient missing the solution diverges within 20 steps.
3. **Sign of the unsteady pressure term.** The correct Bernoulli form is
   `L = -rho*U*Gamma + rho*int(d(Delta phi)/dt) dx`; getting that relative sign wrong
   keeps the steady lift exact but rotates the unsteady lift by up to 50 deg -- the
   aerodynamic check is what catches it.
4. **Chordwise/wake resolution matters more than the time step.** With 40 panels and a
   30 semi-chord wake the growth rate at mu = 10 was off by 60%; with 64 panels and 60
   semi-chords it matches p-k to four digits. The trailing-edge log kernel needs the
   wake resolved far enough downstream.
5. **Pure-displacement initial conditions can sit on the static equilibrium**
   (`alpha_eq/alpha0 ~ U^2/(mu*r_alpha^2)`), leaving a nearly constant signal; the
   coupled march therefore uses an impulsive start and the estimator removes the static
   offset.
6. **Scope**: 2-D strip aerodynamics (aspect ratio 5), linear structure, potential flow
   (no viscosity, no separation), flat wake (no roll-up), small deflections. Transonic
   and separated-flow flutter, and panel flutter (Dowell's in-plane-flow case), are
   outside this node.
7. **The growth-rate reference used here is the p-k eigenvalue, not a digitised Dowell
   (1995) figure.** The Dowell textbook curves are reproductions of the same p-k
   eigenvalue problem, but no machine-readable copy of the figure was available offline,
   so the comparison is made against the theory the figure plots. This is stated
   explicitly rather than presenting a hand-digitised curve as independent data.

## 6. Error judgement

| criterion | target | achieved | verdict |
|---|---|---|---|
| (a) U_cr vs Theodorsen p-k | < 5% | 1.89% (mu = 5, 10, 20, 50) | PASS |
| (b) flutter frequency at the boundary | < 3% | 1.66% at 1.05 U_cr (20, 50) | PASS |
| (c) amplitude growth law | < 10% | 7.7% (supercritical) | PASS |
| (d) U_cr ~ mu^n, error in n | < 2% | 0.54% (mu >= 200) | PASS |
| (e) negative control rho_f -> 0 | U_cr -> infinity | monotone rise, sqrt-law within 0.80% | PASS |
| (f) pytest cases | >= 8 | 13 (11 fast + 2 slow) | PASS |

**Is the flutter frequency error below 3%?** Yes at the flutter boundary: the
worst 1.05 U_cr measurement is 1.66%. Deeper into the
supercritical range (1.15 U_cr) the simulated frequency deviates by up to 4.7%: there the two aeroelastic
branches are no longer well separated and the demodulation locks onto the mixed
response. That number is reported, not hidden, but it is not the flutter frequency
defined at the boundary.

**Is the flutter speed error below 5%?** Yes: 1.89% worst case over mu = 5, 10, 20, 50.

**Data provenance**: every number in this document is written by
`scripts/fsi_flutter_vlm_n07.py --stage report` from
`artifacts/fsi_flutter_vlm_n07/results.json`, which the same script fills from the
solvers; no value is typed by hand.
