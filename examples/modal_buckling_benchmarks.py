"""Run the canonical beam vibration and Euler column buckling benchmarks."""

from tensorfem.buckling import euler_pinned_critical_load, pinned_pinned_column_buckling
from tensorfem.modal import cantilever_beam_modes, cantilever_exact_angular_frequency


def report(name: str, computed: float, exact: float) -> None:
    print(f"{name}: computed={computed:.8g}, exact={exact:.8g}, error={abs(computed/exact-1):.4%}")


modal = cantilever_beam_modes(1.0, 8, 210e9 * 8e-8, 7850 * 4e-4, 3)
modal_exact = cantilever_exact_angular_frequency(1.0, 210e9 * 8e-8, 7850 * 4e-4)
report("cantilever omega_1 [rad/s]", modal.angular_frequencies[0].item(), modal_exact)

buckling = pinned_pinned_column_buckling(3.0, 8, 200e9 * 6e-6, 3)
buckling_exact = euler_pinned_critical_load(3.0, 200e9 * 6e-6)
report("pinned column P_cr [N]", buckling.load_factors[0].item(), buckling_exact)
