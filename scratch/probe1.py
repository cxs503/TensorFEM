import sys, time, math
sys.path.insert(0, "/srv/tensor/agent/work/m21707/src")
import numpy as np
import tensorfem.patch_from_contact as P
import tensorfem.ice_panel_patch_contact as IC

A = 1.0; T = 0.01; E = 2.06e11; NU = 0.3
half = 0.2; force = 4.0e4
p0 = force / (4 * half * half)
xs = np.linspace(0.5 - half, 0.5 + half, 9); ys = xs.copy()
XX, YY = np.meshgrid(xs, ys)
uni_data = P.ContactPressureData(node_positions=np.column_stack([XX.ravel(), YY.ravel()]),
                                 normal_pressures=np.full(XX.size, p0), contact_area=0.16)
uni_patch = IC.uniform_patch_pressure(total_force=force, centre=(0.5, 0.5), half_x=half, half_y=half,
                                      panel=(A, A), grid=33)
nav = IC.navier_rect_moments(uni_patch, A, A, young=E, poisson=NU, thickness=T, x=0.5, y=0.5)
nav_mx = 6 * abs(nav["moment_x"]) / T**2 / 1e6
print("navier Mx MPa", nav_mx)
for nx in (16, 24, 32, 48):
    t0 = time.time()
    mine, res = P.contact_pressure_to_plate_load(uni_data, nx, A, E, NU, T, 0.0)
    st = P.peak_stresses(mine)
    nodes, el, f, info = IC.patch_from_contact(uni_patch, nx, nx, A, A)
    sol = IC.solve_rect_panel(nx, nx, A, A, young=E, poisson=NU, thickness=T, load=f)
    st_ic = sol.peak_stress()
    print(nx, "mine %.3f (%.3f%%) ic %.3f (%.3f%%)  t=%.1fs" % (
        st["sigma_x_MPa"], 100 * (st["sigma_x_MPa"] / nav_mx - 1),
        st_ic["sigma_Mx_MPa"], 100 * (st_ic["sigma_Mx_MPa"] / nav_mx - 1), time.time() - t0))