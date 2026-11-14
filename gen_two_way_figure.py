"""Figure for docs/CASE-FEM-ELASTIC-RESTRAINT-TWO-WAY-PEAK.md.

Data = uniform-load band of the manuscript panel, measured with
``scratch/probe_two_way_peak.py`` in this worktree (uniform load uses the smooth
load path, so the numbers are unaffected by the patch-integration fix).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BETA = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]                       # beta_x = beta_y
EXTRAP_GLOBAL = [378.56, 348.97, 307.94, 277.84, 237.53, 251.25]   # MPa
SPRING_EDGE = [0.0, 21.51, 51.55, 96.46, 171.18, 326.05]
CONSISTENT = [347.38, 325.52, 302.24, 277.84, 237.53, 326.05]
ONE_WAY_SPRING = [-0.22, -0.06]   # not plotted; see the case doc tables

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(BETA, CONSISTENT, 'o-', lw=2, label='consistent peak (reported)')
ax.plot(BETA, EXTRAP_GLOBAL, 's--', label='extrapolated global max')
ax.plot(BETA, SPRING_EDGE, 'd-.', label=r'spring edge max  $M_n=c_r\theta_n$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel(r'embedding coefficient $\beta_x=\beta_y$')
ax.set_ylabel('von Mises bending stress [MPa]')
ax.set_title('0.8 m x 0.7 m x 22 mm panel, uniform 1 MPa\n'
             'peak-stress definition vs fixity (nx=16)')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
os.makedirs('docs/figures', exist_ok=True)
fig.savefig('docs/figures/elastic_restraint_two_way_peak.png', dpi=150)
print('wrote docs/figures/elastic_restraint_two_way_peak.png')
