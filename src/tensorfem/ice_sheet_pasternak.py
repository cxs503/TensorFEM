"""Level ice sheet on a Pasternak foundation: flexural response and the
local -> global failure-mode transition.

Physical setting
----------------
A level ice sheet resting on water is classically idealised as a plate on a
Winkler bed of stiffness ``k = rho_w g`` (buoyancy) with plate rigidity
``D = E h^3 / (12 (1 - nu^2))``.  That Winkler idealisation cannot represent any
*shear* transfer inside the foundation.  Two situations where shear transfer is
real and often dominant:

* ice over a rubble / grounded-ice or soil subgrade (ice roads, landfast ice
  over shoals, ice landing pads) -- the granular layer transmits shear, modelled
  by a Pasternak (two-parameter) bed of shear modulus ``G_s`` [N/m];
* ice in a rubble field: neighbouring blocks push on each other, again a
  shear-carrying bed.

The sheet then obeys ``D laplacian^4 w - G_s laplacian^2 w + k w = q``, whose
1-D strip form ``D w'''' - G_s w'' + k w = q`` is exactly the beam problem of
:mod:`tensorfem.pasternak_beam` with ``EI -> D``.  This module composes that
solver (and the pre-existing plate-level Pasternak closed form in
:mod:`tensorfem.foundation_plate_bending`) into ice-engineering quantities:

* the characteristic lengths of the sheet,
* the flexural (first-crack) failure load under a patch load,
* the local contact-crushing load,
* the foundering (buoyancy) bound of a finite sheet,
* the resulting failure-mode map and the local -> global transition.

Everything is reported **per metre of sheet width** (1-D strip idealisation,
patch spanning the unit width).  Loads are N/m, moments N m/m.

Failure model (declared, not calibrated)
----------------------------------------
Three idealised mechanisms compete and the smallest load governs:

``local_crushing``  contact crushing  ``P = sigma_c * 2a``
``global_flexure``  first-crack bending ``P = P_ref M_cap / M_max`` from the
                    strip FE solution, ``M_cap = sigma_f h^2 / 6``
``foundering``      sheet awash      ``P = rho_w g (1 - rho_i/rho_w) h L``

That local-crushing vs global-bending ranking is the classical ice
bearing-capacity dichotomy (Michel 1978; Kerr 1976; Sanderson 1988).  The
mechanism ranking is physical; ``sigma_f``, ``sigma_c`` are *inputs* and are not
calibrated against a sea-ice test series here.  Rate effects and the
strain-rate dependence of ice strength are out of scope.

Limitations
-----------
* 1-D strip idealisation (unit width); no 2-D Pasternak failure surface.  The
  plate-level Pasternak closed form (:func:`plate_peak_deflection`) is used only
  as an implementation cross-check.
* Bilateral foundation (no tensionless lift-off), first-crack linear-elastic
  criterion rather than crack propagation; static loading, no added mass and no
  hydrodynamic coupling.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .pasternak_beam import (
    PasternakBeamModel,
    pasternak_characteristic_lengths,
    power_law_exponent,
    solve_pasternak_beam,
)

FLOAT = torch.float64

__all__ = [
    "IceSheetProperties",
    "ice_plate_rigidity",
    "buoyancy_stiffness",
    "ice_sheet_characteristic_lengths",
    "strip_patch_fem",
    "flexural_failure_load",
    "local_crushing_load",
    "foundering_load",
    "failure_mode",
    "transition_scan",
    "branch_fit",
    "plate_peak_deflection",
]


@dataclass(frozen=True)
class IceSheetProperties:
    """Elastic and strength properties of a level ice sheet (SI units)."""

    thickness: float = 1.0            # h [m]
    young: float = 5.0e9              # E [Pa]
    poisson: float = 0.3
    ice_density: float = 917.0        # [kg/m^3]
    water_density: float = 1025.0     # [kg/m^3]
    gravity: float = 9.81             # [m/s^2]
    flexural_strength: float = 0.5e6  # sigma_f [Pa]
    crushing_strength: float = 2.0e6  # sigma_c [Pa]

    def __post_init__(self):
        if self.thickness <= 0.0:
            raise ValueError("ice thickness must be positive")
        if self.young <= 0.0 or not -1.0 < self.poisson < 0.5:
            raise ValueError("invalid elastic constants")
        if not 0.0 < self.ice_density < self.water_density:
            raise ValueError("ice must be lighter than water to float")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if self.flexural_strength <= 0.0 or self.crushing_strength <= 0.0:
            raise ValueError("strengths must be positive")

    @property
    def rigidity(self) -> float:
        return ice_plate_rigidity(self.young, self.poisson, self.thickness)

    @property
    def buoyancy(self) -> float:
        return buoyancy_stiffness(self.water_density, self.gravity)

    @property
    def freeboard(self) -> float:
        """Emergent freeboard ``(1 - rho_i/rho_w) h`` [m]."""
        return (1.0 - self.ice_density / self.water_density) * self.thickness

    @property
    def moment_capacity(self) -> float:
        """First-crack moment capacity per unit width ``sigma_f h^2 / 6``."""
        return self.flexural_strength * self.thickness**2 / 6.0


def ice_plate_rigidity(young: float, poisson: float, thickness: float) -> float:
    """Plate rigidity ``D = E h^3 / (12 (1 - nu^2))`` [N m]."""
    if young <= 0.0 or thickness <= 0.0:
        raise ValueError("young's modulus and thickness must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson ratio must lie in (-1, 0.5)")
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def buoyancy_stiffness(water_density: float, gravity: float) -> float:
    """Winkler stiffness of the buoyant support ``k = rho_w g`` [N/m^3]."""
    if water_density <= 0.0 or gravity <= 0.0:
        raise ValueError("water density and gravity must be positive")
    return water_density * gravity


def ice_sheet_characteristic_lengths(sheet: IceSheetProperties,
                                     pasternak: float = 0.0) -> dict:
    """Characteristic lengths of the ice sheet for a bed shear ``G_s`` [N/m].

    ``l_ice = (D/k)^{1/4}`` (classical ice-sheet characteristic length,
    ~14.6 m for 1 m of sea ice), ``l_winkler = (4D/k)^{1/4}`` (the Hetenyi
    penetration length, i.e. ``P/(2 k w(0))``, ~20.7 m),
    ``l_shear = 1/Re lambda_fast`` (-> ``sqrt(D/G_s)``), ``l_soft = 1/Re
    lambda_slow`` (-> ``sqrt(G_s/k)``), ``l_dominant = max(l_shear, l_soft)`` and
    ``l_effective`` (the Winkler bed giving the same peak penetration).
    """
    lengths = pasternak_characteristic_lengths(sheet.rigidity, sheet.buoyancy,
                                               pasternak)
    lengths["l_ice"] = (sheet.rigidity / sheet.buoyancy) ** 0.25
    lengths["l_dominant"] = max(lengths["l_shear"], lengths["l_soft"])
    lengths["pasternak"] = float(pasternak)
    return lengths


# ---------------------------------------------------------------------------
# strip (unit width) finite element model of the sheet
# ---------------------------------------------------------------------------
def _strip_model(sheet: IceSheetProperties, length: float, n_elements: int,
                 pasternak: float) -> PasternakBeamModel:
    return PasternakBeamModel(length=length, n_elements=n_elements, E=1.0,
                              Iz=sheet.rigidity, k=sheet.buoyancy,
                              shear=pasternak, width=1.0)


def _auto_elements(sheet: IceSheetProperties, length: float,
                   contact_half_length: float, pasternak: float) -> int:
    """Mesh that resolves the shortest active length scale ~12 elements across."""
    lengths = ice_sheet_characteristic_lengths(sheet, pasternak)
    shortest = min(2.0 * contact_half_length, lengths["l_winkler"],
                   lengths["l_shear"])
    return int(min(max(round(12.0 * length / shortest), 200), 4000))


def strip_patch_fem(sheet: IceSheetProperties, *, length: float,
                    n_elements: int | None = None, pasternak: float = 0.0,
                    contact_half_length: float = 0.5,
                    load: float = 1.0e3) -> dict:
    """FEM response of a unit-width ice strip under a central patch load.

    ``load`` [N/m] is spread uniformly over ``2 * contact_half_length`` at the
    centre (consistent Hermite nodal loads on each patch node's tributary
    length), so no point-load singularity is introduced.  The ends are free and
    the bed supplies buoyancy springs plus the shear layer.
    """
    if contact_half_length <= 0.0:
        raise ValueError("contact half length must be positive")
    if not 0.0 < 2.0 * contact_half_length <= length:
        raise ValueError("the contact patch must fit inside the strip")
    if n_elements is None:
        n_elements = _auto_elements(sheet, length, contact_half_length, pasternak)
    model = _strip_model(sheet, length, n_elements, pasternak)
    h = model.h
    centre = 0.5 * length
    n0 = int(round((centre - contact_half_length) / h))
    n1 = int(round((centre + contact_half_length) / h))
    n0 = max(n0, 0)
    n1 = min(max(n1, n0 + 1), n_elements)
    tributary = torch.zeros(model.n_elements + 1, dtype=FLOAT)
    for e in range(n0, n1):
        tributary[e] += 0.5 * h
        tributary[e + 1] += 0.5 * h
    contact_length = float(tributary.sum())
    if contact_length <= 0.0:
        raise RuntimeError("empty contact patch")
    q = load / contact_length
    nodal = {int(i): float(q * tributary[i])
             for i in range(model.n_elements + 1) if tributary[i] > 0.0}
    sol = solve_pasternak_beam(model, nodal_loads=nodal, diagnostics=True)
    sol.update({"load": load, "contact_half_length": contact_half_length,
                "contact_length": contact_length, "line_load": q,
                "n_elements": n_elements, "mesh": h,
                "pasternak": float(pasternak)})
    return sol


def flexural_failure_load(sheet: IceSheetProperties, *, sheet_length: float,
                          pasternak: float = 0.0,
                          contact_half_length: float = 0.5,
                          n_elements: int | None = None,
                          reference_load: float = 1.0e3) -> dict:
    """First-crack (flexural) load of the ice strip, by linear scaling [N/m].

    The model is linear, so the load that brings the peak sagging moment to the
    first-crack capacity ``sigma_f h^2 / 6`` is
    ``P_flex = P_ref M_cap / M_max(P_ref)``.
    """
    sol = strip_patch_fem(sheet, length=sheet_length, n_elements=n_elements,
                          pasternak=pasternak,
                          contact_half_length=contact_half_length,
                          load=reference_load)
    m_max = float(sol["M"].max())
    if m_max <= 0.0:
        raise RuntimeError("no sagging moment recovered from the strip model")
    scale = sheet.moment_capacity / m_max
    return {
        "load": reference_load * scale,
        "moment_capacity": sheet.moment_capacity,
        "moment_peak": m_max,
        "moment_per_load": m_max / reference_load,
        "peak_location": float(sol["nodes"][int(sol["M"].argmax())]),
        "peak_deflection": float(sol["w"].abs().max()),
        "deflection_per_load": float(sol["w"].abs().max()) / reference_load,
        "foundation_reaction": sol["reaction"]["total"],
        "n_elements": sol["n_elements"],
        "mesh": sol["mesh"],
    }


def local_crushing_load(sheet: IceSheetProperties, *,
                        contact_half_length: float) -> float:
    """Contact crushing load per metre of width ``sigma_c * 2a`` [N/m]."""
    if contact_half_length <= 0.0:
        raise ValueError("contact half length must be positive")
    return sheet.crushing_strength * 2.0 * contact_half_length


def foundering_load(sheet: IceSheetProperties, *, sheet_length: float) -> float:
    """Buoyancy (foundering) bound of a free-floating strip ``rho_w g fb L``.

    The sheet can carry no more per metre of width than the weight of the water
    displaced by its emergent freeboard before it goes awash.
    """
    if sheet_length <= 0.0:
        raise ValueError("sheet length must be positive")
    return sheet.water_density * sheet.gravity * sheet.freeboard * sheet_length


def failure_mode(sheet: IceSheetProperties, *, sheet_length: float,
                 pasternak: float = 0.0, contact_half_length: float = 0.5,
                 n_elements: int | None = None) -> dict:
    """Rank the three idealised mechanisms and report the governing one."""
    flex = flexural_failure_load(sheet, sheet_length=sheet_length,
                                 pasternak=pasternak,
                                 contact_half_length=contact_half_length,
                                 n_elements=n_elements)
    local = local_crushing_load(sheet,
                                contact_half_length=contact_half_length)
    buoy = foundering_load(sheet, sheet_length=sheet_length)
    loads = {"local_crushing": local, "global_flexure": flex["load"],
             "foundering": buoy}
    mode = min(loads, key=loads.get)
    lengths = ice_sheet_characteristic_lengths(sheet, pasternak)
    return {
        "mode": mode,
        "critical_load": loads[mode],
        "loads": loads,
        "flexure": flex,
        "l_c": lengths["l_dominant"],
        "l_winkler": lengths["l_winkler"],
        "l_ratio": lengths["l_dominant"] / sheet_length,
        "sheet_length": sheet_length,
        "contact_half_length": contact_half_length,
        "contact_ratio": contact_half_length / lengths["l_dominant"],
        "pasternak": float(pasternak),
        "n_elements": flex["n_elements"],
    }


def transition_scan(sheet: IceSheetProperties, *, sheet_lengths,
                    pasternak_values=(0.0,), contact_half_length: float = 0.5,
                    n_elements: int | None = None) -> list[dict]:
    """Scan the failure mode over ``(sheet length, bed shear)``.

    Each record is a :func:`failure_mode` dictionary; the abscissa of the
    reported transition curve is ``l_c / L`` with ``l_c`` the dominant Pasternak
    decay length.
    """
    records = []
    for pasternak in pasternak_values:
        for length in sheet_lengths:
            records.append(failure_mode(sheet, sheet_length=float(length),
                                        pasternak=float(pasternak),
                                        contact_half_length=contact_half_length,
                                        n_elements=n_elements))
    records.sort(key=lambda rec: rec["l_ratio"])
    return records


def branch_fit(records: list[dict], mode: str, *, x_key: str = "l_ratio",
               y_key: str = "critical_load") -> dict:
    """Power-law fit of one failure branch ``y = c x^m`` over its own records."""
    xs, ys = [], []
    for rec in records:
        if rec["mode"] != mode:
            continue
        x, y = rec[x_key], rec[y_key]
        if x > 0.0 and y > 0.0:
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        raise ValueError(f"fewer than two records on the {mode} branch")
    return power_law_exponent(xs, ys)


def plate_peak_deflection(sheet: IceSheetProperties, *,
                          load: float = 1.0e5, pasternak: float = 0.0) -> float:
    """Peak deflection of an *infinite plate* on the same buoyancy + shear bed.

    Corroboration against the pre-existing plate-level implementation
    (:func:`tensorfem.foundation_plate_bending.
    infinite_plate_two_parameter_point_load`); this is an independent
    cross-check of the 2-D Pasternak response, not a new solver.
    """
    from .foundation_plate_bending import infinite_plate_two_parameter_point_load

    value = infinite_plate_two_parameter_point_load(
        torch.zeros(1, dtype=FLOAT), load, sheet.rigidity, sheet.buoyancy,
        pasternak)
    return float(value[0])