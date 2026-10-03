"""TensorFEM-only variable-amplitude fatigue and crack-growth example."""
from tensorfem.marine_fatigue_advanced import (
    CrackGrowthBlock,
    ParisLaw,
    PathStressSample,
    hot_spot_stress_from_path_pa,
    paris_crack_growth,
    rainflow_cycles,
    spectrum_miner_damage,
)
from tensorfem.marine_fatigue_qualification import SNLogLogCurve

path = tuple(
    PathStressSample(x, 150e6 - 4e9 * x)
    for x in (0.0, 0.003, 0.007, 0.012, 0.016)
)
hot_spot_pa = hot_spot_stress_from_path_pa(path, thickness_m=0.01)

cycles = rainflow_cycles((0.0, 80e6, -20e6, 120e6, 0.0))
damage = spectrum_miner_damage(cycles, SNLogLogCurve(100e6, 2e6, 3.0))

growth = paris_crack_growth(
    0.005,
    (CrackGrowthBlock(80e6, 10_000), CrackGrowthBlock(120e6, 2_000)),
    ParisLaw(coefficient_c=1e-24, exponent_m=2.0),
    critical_crack_m=0.02,
)

print(f"hot spot: {hot_spot_pa / 1e6:.3f} MPa")
print(f"rainflow cycles: {len(cycles)}, Miner damage: {damage.damage:.6g}")
print(f"final crack: {growth.final_crack_m * 1e3:.3f} mm")
