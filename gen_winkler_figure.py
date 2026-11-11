"""Generate comparison figure for GAP-FEM-WINKLER verification report."""
import math
import base64
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorfem.plate_mindlin import (
    hetenyi_reference,
    solve_plate_point_load,
)

# Parameters (ice plate)
E = 9.0e9
nu = 0.33
t = 0.5
k_f = 1.0e4
P = 1.0e5

D = E * t**3 / (12 * (1 - nu**2))
l_char = (D / k_f) ** 0.25
w_het_0 = P / (8 * math.sqrt(D * k_f))

# Solve FEM
side = 10 * l_char
nx = 16
result = solve_plate_point_load(
    nx=nx, side=side, young=E, poisson=nu, thickness=t,
    winkler_k=k_f, point_load=P,
)

# Extract centerline deflection
nodes = result.nodes
center_y = side / 2
mask = (nodes[:, 1] - center_y).abs() < (side / nx)
line_indices = mask.nonzero(as_tuple=False).flatten()
line_nodes = nodes[line_indices]
order = torch.argsort(line_nodes[:, 0])
x_vals = (line_nodes[order, 0].numpy() - side/2) / l_char  # normalize by l
w_fem_vals = result.displacement[3 * line_indices[order]].numpy()

# Hetényi reference
r_vals = np.linspace(0, 4 * l_char, 200)
w_het_vals = np.array([hetenyi_reference(r, 0.0, P, E, nu, t, k_f) for r in r_vals])
r_norm = r_vals / l_char

# Find center node for annotation
center_idx = torch.argmin((nodes[:, 0] - side/2)**2 + (nodes[:, 1] - side/2)**2).item()
w_fem_center = result.displacement[3 * center_idx].item()
rel_err = abs(w_fem_center - w_het_0) / abs(w_het_0) * 100

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Deflection profile comparison
ax1.plot(r_norm, w_het_vals / w_het_0, 'b-', linewidth=2, label='Hetényi (analytical)')
ax1.plot(x_vals, w_fem_vals / w_het_0, 'ro', markersize=4, label=f'FEM (nx={nx})', alpha=0.7)
ax1.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
ax1.set_xlabel('Radial distance / characteristic length (r/l)', fontsize=11)
ax1.set_ylabel('Normalized deflection w/w(0)', fontsize=11)
ax1.set_title('Deflection Profile: FEM vs Hetényi Analytical', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.2, 4.2)

# Right: Convergence study
nx_values = [4, 6, 8, 12, 16]
errors = []
for nx_val in nx_values:
    res = solve_plate_point_load(
        nx=nx_val, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, point_load=P,
    )
    ci = torch.argmin((res.nodes[:, 0] - side/2)**2 + (res.nodes[:, 1] - side/2)**2).item()
    w_fem = res.displacement[3 * ci].item()
    err = abs(w_fem - w_het_0) / abs(w_het_0) * 100
    errors.append(err)

ax2.semilogy(nx_values, errors, 'bs-', linewidth=2, markersize=8)
ax2.axhline(y=5, color='r', linewidth=1.5, linestyle='--', label='5% acceptance')
ax2.axhline(y=1, color='g', linewidth=1.5, linestyle='--', label='1% target')
ax2.set_xlabel('Elements per side (nx)', fontsize=11)
ax2.set_ylabel('Relative error (%)', fontsize=11)
ax2.set_title('Mesh Convergence: Center Deflection Error', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(nx_values)

plt.suptitle(f'GAP-FEM-WINKLER Verification: Ice Plate on Winkler Foundation\n'
             f'E={E/1e9:.0f} GPa, t={t:.1f} m, k={k_f:.0e} N/m³, '
             f'l={l_char:.2f} m, Center error = {rel_err:.2f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()

# Save to buffer
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

print(f'Figure generated: {len(png_base64)} bytes base64')
print(f'Center deflection FEM: {w_fem_center:.6e} m')
print(f'Hetényi analytical: {w_het_0:.6e} m')
print(f'Relative error: {rel_err:.2f}%')

# Save to file for report
with open('/root/tp-agent/work/m13782/artifacts/winkler_hetenyi_comparison.png', 'wb') as f:
    buf.seek(0)
    f.write(buf.read())

print('Figure saved to artifacts/winkler_hetenyi_comparison.png')
