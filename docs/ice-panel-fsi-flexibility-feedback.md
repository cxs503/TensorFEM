# Elastic hull-panel / ice-contact flexibility feedback (FSI)

Module: `tensorfem.ice_panel_fsi` · driver: `scripts/run_ice_panel_fsi_case.py`
· tests: `tests/test_ice_panel_fsi.py` · artifacts: `artifacts/ice_panel_fsi/`

## 1. Question

An ice belt panel bay is loaded by ice that advances against the shell. Two
questions are answered with one coupled model:

* does the **panel flexibility** change the contact load it receives (rigid
  versus elastic bay, identical forcing)?
* does the flexibility change the **ice failure mode** (crushing versus
  bending) and therefore feed back on the load?

## 2. Model

**Panel.** Rectangular bay `span_x × span_y × t` (default 0.75 m × 2.4 m ×
25 mm, E = 206 GPa, ν = 0.30, ρ = 7850 kg/m³, four edges clamped at the frames
and stringers; simply supported is available). Discretised with the four-node
Mindlin-Reissner element of `tensorfem.plate.q4_mindlin_stiffness`
(selective reduced integration on shear). The assembled stiffness is condensed
once into a contact operator, so the coupled time loop only evaluates

```
w_i = sum_j R_ij F_j          (contact deflection)
se  = 0.5 F^T R F             (panel strain energy)
```

with `R = P K_ff^-1 L` (`P` interpolates `w` at the contact points, `L` holds
the consistent unit loads).

**Ice.** A row of `n` independent DEM blocks (default 12 over 2.4 m, 0.20 m of
contact line each, 0.1 m × 0.20 m contact patch, 233 kg per block). Each block
carries an elastic-perfectly-plastic crush law in series with the panel
compliance:

```
p  = min(E_ice * delta_el / L_c , p_c(v)) ,  delta_el = delta - plastic
```

`L_c = (D_ice / (rho_w g))^{1/4}` is the buoyancy characteristic length, i.e.
the length over which the sheet transfers the force into the far field, so the
contact elastic stiffness is `E_ice / L_c` rather than the (≈16× too stiff)
`E_ice / h`. The crushing cap is velocity dependent,
`p_c = sigma_c * clamp((v_ref/v)^0.3, 0.5, 2.0)`, an elastic unloading /
plastic return is applied through the accumulated offset `plastic`, and the
crushing work is booked as plastic work.

**Two feedback loops.** Every time step is a fixed point iteration:

1. penetration from the **deformed** surface, `delta = advance - w - gap`;
2. crush pressure from `delta_el` and the exact coupled gain
   `p = (I + K A R)^-1 K A delta_el / A` (a naive fixed point diverges, the
   gain `K A R_ii` exceeds one at mid-bay);
3. **ice bending cap**: the ice sheet is a plate on a buoyancy foundation
   along the contact line, `D_ice w'''' + rho_w g w = p(y)`, solved periodically
   over one bay pitch by FFT. The map `sigma = B p` (patch-averaged, signed) is
   built once; a uniform reaction gives `B p = 0` exactly, so only the
   *variation* of the panel reaction bends the ice. Where `|sigma| > sigma_f`
   the pressure is capped (`bending_capped_pressure`), which is the failure
   mode conversion;
4. panel deflection from the capped pressure, back to (1).

Block motion is explicit symplectic Euler with `dt = 0.02 T_ice`, and the panel
is quasi-static.

## 3. Verification and results

Static verification (uniform pressure 0.1 MPa, `uniform_load_centre_deflection`):

| bay | t | FE centre w | plate theory | error |
|---|---|---|---|---|
| 1.0 × 1.0 SS | 10 mm | 2.1475e-2 m | 2.1547e-2 m | **-0.33 %** |
| 0.75 × 2.4 SS | 10 mm | 2.0425e-2 m | 2.0840e-2 m | **-1.99 %** |
| 0.75 × 2.4 SS | 25 mm | 1.3107e-3 m | 1.3376e-3 m | **-2.01 %** |

Reference = Kirchhoff Navier double series plus the first-mode Mindlin shear
amplification `w * lam D /(kappa G h)`. Clamped bays are checked by ratio:
0.31 for the square (textbook 0.00126/0.00406 = 0.31) and 0.20 for the
aspect-3.2 bay (one-way strip limit 0.20). Acceptance (a), < 10 %, is met with
a factor five margin.

Representative ice impact (v = 0.3 m/s, 233 kg blocks, KE = 125.8 J):

| quantity | rigid bay | elastic bay (25 mm) |
|---|---|---|
| peak contact load | 174.2 kN | **139.2 kN** |
| peak pressure | 0.66 MPa | 0.66 MPa |
| max panel deflection | 0 | 0.41 mm |
| ice blocks failed in bending | 0/12 | 0/12 |
| energy closure error | 0.012 % | 0.021 % |

**Peak-load reduction 20.0 %** — inside the 10–30 % band reported for
compliant structures (Hughes 1993; Kim 2003). The rigid peak is energy limited,
`F = sqrt(2 E_kin K_ice)`, not cap limited: 174 kN against a crush-cap load of
351 kN, so the mechanism is genuinely the strain-energy sharing between the bay
and the ice.

Plating-thickness sweep (same forcing):

| t | 12 mm | 15 mm | 20 mm | 25 mm | 30 mm | 40 mm | 50 mm |
|---|---|---|---|---|---|---|---|
| peak reduction | 63.3 % | 49.2 % | 31.3 % | 20.0 % | 13.2 % | 6.4 % | 3.5 % |
| panel f1 | 133 Hz | 165 Hz | 220 Hz | 275 Hz | 328 Hz | 435 Hz | 538 Hz |
| closure error | 2.2 % | 0.59 % | 0.08 % | 0.02 % | 0.01 % | 0.02 % | 0.02 % |

The reduction falls monotonically with plating thickness and follows
`1 - 1/sqrt(1 + K_ice/K_panel)`: the flexible bay is soft in series with the
ice and absorbs part of the impact energy elastically.

Failure-mode conversion (high severity: 4 t blocks, KE = 2.16 MJ):

| bay | peak reduction | ice bending failure |
|---|---|---|
| 12 mm | 68.7 % | **66.7 %** of blocks |
| 25 mm | 17.1 % | 100 % at peak, 4/12 cumulative |
| 40 mm | 2.1 % | **0 %** |

A thin bay produces a strongly non-uniform reaction, the ice sheet cannot
follow that gradient without exceeding `sigma_f`, and the bending-capped blocks
drop the load further; a stiff bay reacts almost uniformly (`B p -> 0`) and the
ice fails by crushing. Energy closure for this dissipative case is 4.2 %
(crushing 1680 J of 2160 J kinetic, remainder returned kinetic + panel strain).

Figures: `fig1` load history and pressure profile, `fig2` deflection field and
the thickness trend against the 10–30 % band, `fig3` peak load and bending
fraction versus thickness, `fig4` energy budget and severity comparison.

## 4. Notes, limits, scope

* **Not a calibrated sea-ice model.** Crushing is elastic-perfectly-plastic
  with a velocity-dependent cap, the blocks are translationally uncoupled (a
  block row), and the pressure–area effect is available but off by default
  (`pressure_area_exponent = 0`). `tensordem` is not installed in this
  environment, so the DEM lives here behind a small documented contract; a real
  DEM back end can replace `run_ice_panel_case`'s block loop.
* **Quasi-static panel.** Panel inertia is neglected. The bay's Rayleigh
  frequency is 275 Hz (period 3.6 ms) while the impact lasts a few ms, so the
  static assumption is optimistic: a dynamically responding bay stores less
  energy, which would *raise* the elastic peak and reduce the 20 % figure. The
  number should be read as an upper bound on the reduction.
* Contact pressure magnitude scales with the assumed contact patch (0.1 m);
  the certified quantity is the relative rigid/elastic difference and the
  energy closure, not an absolute design ice pressure.
* No water pressure on the shell, no added mass, no plasticity or buckling of
  the panel (linear elastic only, so the bay stays in the small-deflection
  regime: 0.4 mm on a 750 mm span).
* The bending operator idealises the ice as a strip on a buoyancy foundation
  periodic over one bay pitch. For a vertical side shell the surrounding ice is
  an in-plane support that is stiffer than buoyancy, so the bending stress is an
  upper bound and the bending-limited pressure a lower bound.