# Explicit central-difference validation

The explicit linear-dynamics kernel requires a positive diagonal lumped mass.
It computes the model-specific limit `dt_crit=2/omega_max` and rejects a time
increment above `stability_safety * dt_crit`. This is a fail-closed contract,
not merely a warning.

Validation includes an undamped SDOF oscillator over five periods. With 100
increments per period, peak history error and total-energy drift must each be
strictly below 3%. A two-element fixed-free axial rod additionally checks the
first frequency against `omega_1=(pi/(2L))*sqrt(E/rho)` and requires <3%.

The scope is currently linear, undamped small displacement. Contact impact,
hourglass control, mass scaling and nonlinear internal-force updates require a
later explicit nonlinear step and are not advertised by this kernel.
