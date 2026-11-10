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
  (:mod:`tensorfem.struct_acoustics.fwh_analogy`).

The modules are deliberately independent code paths: each one is verified
against the others and against closed forms rather than against a stored
snapshot of this repository's output.
"""

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
]
