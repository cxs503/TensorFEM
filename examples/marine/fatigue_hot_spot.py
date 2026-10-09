"""TensorFEM-only weld hot-spot and fatigue assessment example (SI units)."""
from tensorfem.marine_fatigue_qualification import (
    SNLogLogCurve,
    StressBlock,
    hot_spot_stress_pa,
    miner_damage,
)

hot_spot = hot_spot_stress_pa((132e6, 105e6), (0.004, 0.010), method="linear")
curve = SNLogLogCurve(reference_stress_pa=100e6, reference_cycles=2e6, slope_m=3.0)
result = miner_damage((StressBlock(hot_spot, 100_000), StressBlock(100e6, 500_000)), curve)
print({"hot_spot_stress_pa": hot_spot, "miner_damage": result.damage, "passed": result.passed})
