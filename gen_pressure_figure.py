#!/usr/bin/env python3
"""Generate Hertz pressure profile comparison figure."""
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorfem.hertz_cattaneo_mindlin import (
    hertz_two_sphere_reference,
    hertz_pressure_profile,
    hertz_load_from_indentation,
)

D = torch.float64

R1 = R2 = 10.0
E = 1e5
nu = 0.3
R_star = 1.0 / (1.0 / R1 + 1.0 / R2)
E_star = 1.0 / (2.0 * (1 - nu**2) / E)

indentation = 0.025
P = hertz_load_from_indentation(indentation, R_star, E_star)
ref = hertz_two_sphere_reference(P, R1, R2, E, nu, E, nu)

# Analytical pressure profile
r = torch.linspace(0, ref.contact_radius * 1.5, 200, dtype=D)
p = hertz_pressure_profile(r, ref.contact_radius, ref.peak_pressure)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(r.numpy() / ref.contact_radius, p.numpy() / ref.peak_pressure, 'b-', linewidth=2)
ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='Contact edge r=a')
ax.set_xlabel('r / a', fontsize=12)
ax.set_ylabel('p / p₀', fontsize=12)
ax.set_title('Hertz Contact Pressure Profile', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.5)
ax.set_ylim(-0.05, 1.1)
plt.tight_layout()
plt.savefig('/root/tp-agent/work/m6635/artifacts/hertz_pressure_profile.png', dpi=150)
plt.close()
print("Saved: artifacts/hertz_pressure_profile.png")
