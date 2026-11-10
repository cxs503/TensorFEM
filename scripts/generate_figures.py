"""Generate verification figures for composite laminate buckling."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'src')

from tensorfem.composite_laminate_buckling import (
    PlyProperty, LaminateLayup, compute_laminate_abd,
    navier_clt_buckling, fsdt_buckling, parametric_ply_angle_sweep,
    thick_plate_ratio_scan, AS4_9310_E1, AS4_9310_E2, AS4_9310_NU12, AS4_9310_G12
)

# AS4/9310 carbon/epoxy
ply = PlyProperty(E1=AS4_9310_E1, E2=AS4_9310_E2, nu12=AS4_9310_NU12, G12=AS4_9310_G12)

# Figure 1: Ply angle parametric sweep
print("Generating Figure 1: Ply angle sweep...")
fig1, ax1 = plt.subplots(figsize=(10, 6))
angles = list(range(0, 91, 5))
sweep_Ncr = []
for theta in angles:
    layup = LaminateLayup(ply_props=ply, angles_deg=[theta, -theta], ply_thickness=0.125e-3)
    stiff = compute_laminate_abd(layup)
    res = navier_clt_buckling(stiff, 0.3, 0.3, m_max=10, n_max=10)
    sweep_Ncr.append(res.N_cr / 1e3)  # Convert to kN/m

ax1.plot(angles, sweep_Ncr, 'b-o', linewidth=2, markersize=6)
ax1.set_xlabel('Ply Angle θ (degrees)', fontsize=12)
ax1.set_ylabel('Critical Load N_cr (kN/m)', fontsize=12)
ax1.set_title('Buckling Load vs Ply Angle for [θ/-θ]s Laminate\nAS4/9310 Carbon/Epoxy, 300×300mm Plate', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 90)
fig1.tight_layout()
fig1.savefig('artifacts/fig1_ply_angle_sweep.png', dpi=150, bbox_inches='tight')
print(f"  Saved: artifacts/fig1_ply_angle_sweep.png")

# Figure 2: FSDT/CLT ratio vs h/a
print("Generating Figure 2: FSDT/CLT ratio...")
fig2, ax2 = plt.subplots(figsize=(10, 6))
h_over_a = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
result = thick_plate_ratio_scan(ply, [0, 90], a=0.2, b=0.2, h_over_a_range=h_over_a)
ratios = [r * 100 for r in result["ratio_FSDT_CLT"]]  # Convert to percentage

ax2.plot(h_over_a, ratios, 'r-s', linewidth=2, markersize=8)
ax2.set_xlabel('Thickness-to-Length Ratio h/a', fontsize=12)
ax2.set_ylabel('FSDT/CLT Ratio (%)', fontsize=12)
ax2.set_title('FSDT Correction for [0/90]s Laminate\nEffect of Transverse Shear Deformation', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='CLT (no shear)')
ax2.legend()
fig2.tight_layout()
fig2.savefig('artifacts/fig2_fsdt_clt_ratio.png', dpi=150, bbox_inches='tight')
print(f"  Saved: artifacts/fig2_fsdt_clt_ratio.png")

# Figure 3: Layup comparison
print("Generating Figure 3: Layup comparison...")
fig3, ax3 = plt.subplots(figsize=(10, 6))
layups = [
    ('[0/90]s', [0, 90]),
    ('[±45]2s', [45, -45]),
    ('[0/90]2s', [0, 90, 0, 90]),
]
Ncr_CLT = []
Ncr_FSDT = []
for name, angles in layups:
    layup = LaminateLayup(ply_props=ply, angles_deg=angles, ply_thickness=0.125e-3)
    stiff = compute_laminate_abd(layup)
    res_clt = navier_clt_buckling(stiff, 0.3, 0.3, m_max=10, n_max=10)
    res_fsd = fsdt_buckling(stiff, 0.3, 0.3, m_max=10, n_max=10)
    Ncr_CLT.append(res_clt.N_cr / 1e3)
    Ncr_FSDT.append(res_fsd.N_cr / 1e3)

x = np.arange(len(layups))
width = 0.35
bars1 = ax3.bar(x - width/2, Ncr_CLT, width, label='CLT (Kirchhoff)', color='steelblue')
bars2 = ax3.bar(x + width/2, Ncr_FSDT, width, label='FSDT (Mindlin)', color='coral')

ax3.set_xlabel('Laminate Layup', fontsize=12)
ax3.set_ylabel('Critical Load N_cr (kN/m)', fontsize=12)
ax3.set_title('Buckling Load Comparison: CLT vs FSDT\n300×300mm Plate, AS4/9310', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels([name for name, _ in layups])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

fig3.tight_layout()
fig3.savefig('artifacts/fig3_layup_comparison.png', dpi=150, bbox_inches='tight')
print(f"  Saved: artifacts/fig3_layup_comparison.png")

print("\nAll figures generated successfully!")
