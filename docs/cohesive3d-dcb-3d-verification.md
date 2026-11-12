# 3-D cohesive element: width-resolved DCB verification (GAP-FEM-3D-COHESIVE)

Generated 2026-11-12T10:27:40.732502+00:00 by `benchmarks/cohesive3d_dcb_3d.py` on `loongarch64` (torch 2.7.1, python 3.11.6).

## 1. Specimen and method

Two HEX8 adherends (production `tensorfem.solid3d.hex8_stiffness`) are bonded by a
pre-meshed zero-thickness COH3D8 ligament (production `tensorfem.cohesive3d`, BK
mixed-mode law). The interface nodes are duplicated, the specimen is clamped at
`x=0` and opened symmetrically at `x=L` (upper `+delta/2`, lower `-delta/2`), so
`P` is the **total** load and the classical DCB relations apply:
`C = 4a^3/(E b h^3) + a/(kGbh)` and `G_I = 6P^2a^2/(E b^2 h^3)`, both optionally
with the Williams/ASTM D5528 root rotation `a -> a + Delta`,
`Delta = h sqrt(E/(11G) (3 - 2 (Gamma/(1+Gamma))^2))`, `Gamma = 1.18 sqrt(E/G)`.

| quantity | value |
|---|---|
| adherend E, nu | 7e+10 Pa, 0.3 |
| arm thickness h | 3.00 mm |
| k_n, k_s | 2.4e+14, 1e+14 Pa/m |
| t_n0, t_s0 | 120.0, 80.0 MPa |
| G_Ic | 300 J/m^2 |
| cohesive length E G_Ic / t_n0^2 | 1.46 mm |

## 2. Criteria (independent reference in the same row)

| check | reference | computed | rel. error | tolerance | status |
|---|---|---|---|---|---|
| `compliance_bracket` | 2.35674e-05 | 2.18415e-05 | -7.32 % | |err| <= 10% vs root-rotation form (bracket in JSON) | PASS |
| `peak_load_vs_lefm` | 28.6994 | 0.412059 | -98.56 % | |err| <= 5% | FAIL |
| `energy_balance` | 0 | 0.493586 | +49.36 % | <= 1% | FAIL |
| `interface_work_vs_GIc` | 300 | nan | +nan % | <= 10% | FAIL |
| `peak_ratio_two_cracks` | 0.728101 | 4.68161 | +542.99 % | |err| <= 2% | FAIL |
| `width_objectivity` | 30.0852 | 11.51 | -61.74 % | |err| <= 8% (transitional b/h, 3-D anticlastic effect) | FAIL |

## 3. Propagation runs

### a0_reference

mesh `nx=100, ny=1, nz=3` (dx = 0.60 mm, b/h = 1.0), a0 = 30.0 mm, 4848 dofs, 50 interface elements, 9 accepted steps, 578 Newton iterations, 62 s.

* peak load **0.412 N** vs LEFM(root rotation) 28.699 N (-98.56 %) and plain beam 30.741 N (-98.66 %).
* energy balance `|W_ext - U_solid - W_interface| / W_ext` = 49.359 %.
* interface work per failed area = nan J/m^2 (G_Ic = 300 J/m^2).
* crack front across the width: ['0.0300'] m.

### a0_long

mesh `nx=100, ny=1, nz=3` (dx = 0.60 mm, b/h = 1.0), a0 = 42.0 mm, 4848 dofs, 30 interface elements, 5 accepted steps, 570 Newton iterations, 50 s.

* peak load **0.088 N** vs LEFM(root rotation) 20.896 N (-99.58 %) and plain beam 21.958 N (-99.60 %).
* energy balance `|W_ext - U_solid - W_interface| / W_ext` = 48.179 %.
* interface work per failed area = nan J/m^2 (G_Ic = 300 J/m^2).
* crack front across the width: ['0.0420'] m.

### wide_ny3

mesh `nx=100, ny=3, nz=3` (dx = 0.60 mm, b/h = 6.0), a0 = 30.0 mm, 9696 dofs, 150 interface elements, 40 accepted steps, 80 Newton iterations, 12 s.

* peak load **11.510 N** vs LEFM(root rotation) 172.196 N (-93.32 %) and plain beam 184.445 N (-93.76 %).
* energy balance `|W_ext - U_solid - W_interface| / W_ext` = 49.969 %.
* interface work per failed area = nan J/m^2 (G_Ic = 300 J/m^2).
* crack front across the width: ['0.0300', '0.0300', '0.0300'] m.

## 4. Figures

![load_opening](../docs/assets/cohesive3d-dcb-3d/load_opening.png)
![rcurve](../docs/assets/cohesive3d-dcb-3d/rcurve.png)
![damage_field](../docs/assets/cohesive3d-dcb-3d/damage_field.png)

## 5. Honest limitations

* Small-strain kinematics: the arms rotate < 3 deg at the load peak, but
  large-displacement DCB tests (ASTM D5528 asks for a large-displacement
  correction) are outside this benchmark.
* The Williams root rotation is itself an approximation of the crack-tip
  root flexibility; the FEM value is bracketed by the plain and the
  root-rotation closed forms, and the bracket is reported instead of a
  single number.
* No friction/contact on the crack faces, no cyclic degradation.
