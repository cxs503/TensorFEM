"""Structure-acoustic coupling verification surface (plate <-> fluid <-> sound).

This package fills the structure-acoustic *coupling* gap left open by
:mod:`tensorfem.plate_acoustics` (acoustic half space given a velocity field)
and :mod:`tensorfem.plate_radiation_chain` (FEM eigenpairs -> radiation
efficiency).  Neither of them contains

* an independent *analytic* modal anchor for plates that are not fully simply
  supported (:mod:`tensorfem.struct_acoustics.levy_modes`),
* the fluid *reaction* on the structure -- a two-way coupled response with an
  interface residual and a power-balance identity
  (:mod:`tensorfem.struct_acoustics.radiation_coupling`),
* a free-surface (underwater, Lloyd's mirror) radiation problem
  (:mod:`tensorfem.struct_acoustics.free_surface`), or
* an acoustic-analogy (FW-H / Kirchhoff data surface) route to the radiated
  power that can be cross-checked against the Rayleigh route
  (:mod:`tensorfem.struct_acoustics.fwh_analogy`),
* an exterior monopole (pulsating sphere) solved with a radial Helmholtz finite
  element model and an *exact* spherical radiation condition
  (:mod:`tensorfem.struct_acoustics.pulsating_sphere`),
* the far field of a baffled circular piston / plate from direct numerical
  Rayleigh quadrature, anchored by the exact on-axis integral and the Airy
  directivity (:mod:`tensorfem.struct_acoustics.plate_far_field`), and
* the sound-power / vibration-velocity level law ``Lw = Lv + C`` with a derived
  ``C`` plus the energy balance of a fluid-loaded breathing shell
  (:mod:`tensorfem.struct_acoustics.radiation_scale_law`).

The modules are deliberately independent code paths: each one is verified
against the others and against closed forms rather than against a stored
snapshot of this repository's output.
"""

from .cylinder_radiation import (
    CylinderRadiationResult,
    EquivalentSourceResult,
    axial_spectrum,
    compact_body_power,
    cylinder_mode_field,
    cylinder_surface_grid,
    donnell_frequencies,
    equivalent_source_power,
    finite_cylinder_mode_power,
    fourier_cylinder_mode,
    infinite_cylinder_conductance,
    infinite_cylinder_efficiency,
    monopole_power_matrix,
    shell_radial_amplitudes,
)
from .levy_modes import (
    PlateModalConfig,
    leissa_cccc_square_lambdas,
    levy_frequencies,
    levy_wavenumbers,
    ritz_frequencies,
)
from .radiation_coupling import (
    C_WATER,
    RHO_WATER,
    CoupledResponse,
    RadiationGrid,
    coupled_modal_response,
    half_space_kernel,
    modal_mass,
    point_force_modal,
    radiated_power_from_pressure,
    radiation_impedance_matrix,
    sine_basis,
    sine_frequencies,
    surface_pressure,
)
from .free_surface import (
    FreeSurfaceField,
    free_surface_directivity,
    free_surface_power,
    free_surface_power_ratio_analytic,
    lloyd_null_cosines,
)
from .plate_far_field import (
    PistonPowerCheck,
    angle_sweep_error,
    baffled_piston_far_field,
    bessel_j1,
    far_field_power_from_directivity,
    node_cos_theta_far_field,
    on_axis_exact,
    piston_directivity,
    piston_power_check,
    piston_radiation_efficiency,
    rayleigh_far_field,
)
from .pulsating_sphere import (
    PulsatingSphereSolution,
    RadialMesh,
    assemble_radial_helmholtz,
    convergence_study,
    plane_wave_form_relative_error,
    plane_wave_power,
    plane_wave_power_relative_error,
    plane_wave_pressure,
    pressure as monopole_pressure,
    radiated_power as monopole_radiated_power,
    radiation_efficiency as monopole_radiation_efficiency,
    solve_pulsating_sphere,
    surface_impedance as monopole_surface_impedance,
)
from .radiation_scale_law import (
    BreathingShell,
    CoupledBreathingSphere,
    SurfaceSourceLaw,
    coupled_breathing_sphere,
    energy_balance_sweep,
    impedance_convergence,
    monopole_surface_source,
    piston_surface_source,
    scale_law_offset_db,
    scale_law_residual_db,
    scaling_slope_db,
    sound_power_level,
    spherical_shell_breathing_parameters,
    velocity_level,
)
from .fwh_analogy import (
    KirchhoffSurfaceResult,
    fwh_far_field_power,
    kirchhoff_data_surface_power,
)

__all__ = [
    "PlateModalConfig",
    "leissa_cccc_square_lambdas",
    "levy_frequencies",
    "levy_wavenumbers",
    "ritz_frequencies",
    "C_WATER",
    "RHO_WATER",
    "CoupledResponse",
    "RadiationGrid",
    "coupled_modal_response",
    "half_space_kernel",
    "modal_mass",
    "point_force_modal",
    "radiated_power_from_pressure",
    "radiation_impedance_matrix",
    "sine_basis",
    "sine_frequencies",
    "surface_pressure",
    "FreeSurfaceField",
    "free_surface_directivity",
    "free_surface_power",
    "free_surface_power_ratio_analytic",
    "lloyd_null_cosines",
    "KirchhoffSurfaceResult",
    "fwh_far_field_power",
    "kirchhoff_data_surface_power",
    "PistonPowerCheck",
    "angle_sweep_error",
    "baffled_piston_far_field",
    "bessel_j1",
    "far_field_power_from_directivity",
    "node_cos_theta_far_field",
    "on_axis_exact",
    "piston_directivity",
    "piston_power_check",
    "piston_radiation_efficiency",
    "rayleigh_far_field",
    "PulsatingSphereSolution",
    "RadialMesh",
    "assemble_radial_helmholtz",
    "convergence_study",
    "plane_wave_form_relative_error",
    "plane_wave_power",
    "plane_wave_power_relative_error",
    "plane_wave_pressure",
    "monopole_pressure",
    "monopole_radiated_power",
    "monopole_radiation_efficiency",
    "solve_pulsating_sphere",
    "monopole_surface_impedance",
    "BreathingShell",
    "CoupledBreathingSphere",
    "SurfaceSourceLaw",
    "coupled_breathing_sphere",
    "energy_balance_sweep",
    "impedance_convergence",
    "monopole_surface_source",
    "piston_surface_source",
    "scale_law_offset_db",
    "scale_law_residual_db",
    "scaling_slope_db",
    "sound_power_level",
    "spherical_shell_breathing_parameters",
    "velocity_level",
]
