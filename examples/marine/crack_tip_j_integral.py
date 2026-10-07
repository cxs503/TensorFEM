"""Numerical path-independence check for a mode-I crack-tip field."""
from tensorfem.crack_tip_j_integral import (
    contour_j_integral, mode_i_j_reference, sample_williams_mode_i_contour,
)

k_i, young, poisson = 40e6, 210e9, 0.3
reference = mode_i_j_reference(k_i, young, poisson)
print(f"K-based reference: {reference:.6f} J/m^2")
for radius in (0.005, 0.02, 0.05):
    field = sample_williams_mode_i_contour(radius, 512, k_i, young, poisson)
    result = contour_j_integral(field)
    error = abs(result.j_j_m2-reference)/reference
    print(f"r={radius:.3f} m: J={result.j_j_m2:.6f} J/m^2, error={error:.3%}")
