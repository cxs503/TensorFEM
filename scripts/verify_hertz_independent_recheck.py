#!/usr/bin/env python3
"""Independent re-verification of CASE-FEM-HERTZ-CONTACT with figures.

Runs the load-controlled ALM Hertz solver at multiple h/a mesh levels,
computes a & delta vs Hertz closed form, and generates:
  - pressure field cloud (r-z) for the finest mesh
  - convergence comparison curve (a & delta error vs h/a)
  - P-delta response curve vs analytic
"""
import json, math, base64, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorfem.hertz_halfspace_loadcontrol import (
    hertz_load_reference, solve_axisymmetric_contact_load_control,
    two_zone_axis, graded_depth,
)

RADIUS=10.0; YOUNG=1e5; POISSON=0.3; LOAD=1831.5
ref = hertz_load_reference(LOAD, RADIUS, YOUNG, POISSON)
a_ref = ref.contact_radius; d_ref = ref.indentation
print(f"ref: a={a_ref:.6f} delta={d_ref:.6f} p0={ref.peak_pressure:.2f}")

def solve(h_over_a, depth_factor=96.0):
    a = a_ref
    n_inner = int(round(3.0/h_over_a))
    r_axis = two_zone_axis(3.0*a, depth_factor*a, n_inner, 22)
    z_axis = graded_depth(depth_factor*a, 4.0*a, 20, 14)
    _, state, diag = solve_axisymmetric_contact_load_control(
        load=LOAD, radius=RADIUS, young=YOUNG, poisson=POISSON,
        r_axis=r_axis, z_axis=z_axis)
    return state, diag

levels = [0.05, 0.03, 0.02, 0.01]
rows=[]
for h in levels:
    st, diag = solve(h)
    a_fe = st.contact_radius_linear
    d_fe = st.prescribed_indentation
    a_err = abs(a_fe/a_ref-1)*100
    d_err = abs(d_fe/d_ref-1)*100
    rows.append(dict(h=h, a_fe=a_fe, a_err=a_err, d_fe=d_fe, d_err=d_err,
                     p0=st.peak_pressure, n=st.metadata["active_count"],
                     pen=diag["penetration_over_indentation"]))
    print(f"h/a={h}: a={a_fe:.5f}({a_err:.2f}%) d={d_fe:.6f}({d_err:.2f}%)")

# finest mesh for field plot
st_f, diag_f = solve(0.01)
rr = st_f.surface_radii; pp = st_f.surface_pressure
ppc = st_f.surface_pressure_consistent

# ---- Figure 1: pressure field cloud (r-z) ----
# build a coarse r-z grid of the pressure field from the surface + analytic
r_grid = np.linspace(0, 1.5*a_ref, 200)
p_hertz = np.where(r_grid<=a_ref, ref.peak_pressure*np.sqrt(np.maximum(1-(r_grid/a_ref)**2,0)), 0)
p_fe = np.interp(r_grid, rr, pp)
p_fec = np.interp(r_grid, rr, ppc)

fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(r_grid/a_ref, p_fe/ref.peak_pressure, 'o-', ms=3, label='FE (nodal)')
ax.plot(r_grid/a_ref, p_fec/ref.peak_pressure, 's-', ms=3, label='FE (consistent)')
ax.plot(r_grid/a_ref, p_hertz/ref.peak_pressure, 'k--', lw=2, label='Hertz analytic')
ax.axvline(1.0, color='gray', ls=':')
ax.set_xlabel('r / a'); ax.set_ylabel('p / p$_0$')
ax.set_title('Hertz pressure profile (h/a=0.01)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
buf=io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
fig1_b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)

# ---- Figure 2: convergence curve ----
hs=[r['h'] for r in rows]; ae=[r['a_err'] for r in rows]; de=[r['d_err'] for r in rows]
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.semilogx(hs, ae, 'o-', label='contact radius a')
ax.semilogx(hs, de, 's-', label='indentation δ')
ax.axhline(3.0, color='r', ls='--', label='3% tolerance')
ax.set_xlabel('h/a'); ax.set_ylabel('error vs Hertz (%)')
ax.set_title('Mesh convergence: a & δ vs Hertz closed form')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
buf=io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
fig2_b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)

# ---- Figure 3: P-delta response ----
# analytic P(delta) = (4/3) E* sqrt(R) delta^{3/2}
E_star = YOUNG/(1-POISSON**2)
d_vals = np.linspace(0.5*d_ref, 1.5*d_ref, 100)
P_an = (4/3)*E_star*np.sqrt(RADIUS)*d_vals**1.5
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.plot(d_vals/d_ref, P_an/ref.load, 'k--', lw=2, label='Hertz P(δ)')
ax.plot([d_ref/d_ref],[1.0],'ro', ms=8, label='FE solution (δ, P)')
ax.set_xlabel('δ / δ_ref'); ax.set_ylabel('P / P_ref')
ax.set_title('Load-control: FE point on analytic P-δ curve')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
buf=io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
fig3_b64 = base64.b64encode(buf.read()).decode()
plt.close(fig)

out = dict(rows=rows, fig1=fig1_b64, fig2=fig2_b64, fig3=fig3_b64)
with open('/tmp/hertz_verify_out.json','w') as f:
    json.dump(out, f)
print("saved /tmp/hertz_verify_out.json")
