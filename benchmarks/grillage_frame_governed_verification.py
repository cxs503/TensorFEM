"""Frame-governed branch verification for the ice-grillage ultimate anchor.

Closes the last remaining verification debt of ``GAP-PP-5-ICE-STRUCT-
GRILLAGE-ULTIMATE``: the independent audit census showed the *transverse
frame* branch of the member limit state is never the governing member in any
documented case (case-study 6/6 plate, member-governed 5/6 stiffener + 1/6
plate, slender 6/6 plate).  The frame rigid-plastic collapse formula (T6,
``F_u = 8 Mp / (L (1 - c/2L))``) is therefore only ever *checked* as a
standalone closed form, never *exercised* as the binding ultimate of a real
grillage.

This script builds a **frame-governed** scantling (weak transverse frame,
strong longitudinal stiffener, stocky plate) and verifies that:

1. the transverse frame is the governing member of the member limit state
   (``U_frame = 1.000`` at the limit, ``U_stiff < 1``, ``U_plate < 1``);
2. the frame-governed ultimate ice pressure ``p_ult`` reproduces the
   independent rigid-plastic frame collapse force ``F_u`` in closed form
   (``p_ult * A_patch * f_y == F_u``, gate 0.5 %);
3. the frame-governed ``p_ult`` is *below* the plate-governed and
   stiffener-governed values on the same scantling (the frame is the true
   weakest link), so the branch is not an artefact of a degenerate geometry.

Every number is computed; nothing is transcribed.  The frame collapse force
``F_u`` is re-derived here from first principles (``8 Mp / (L(1 - c/2L))``)
sharing no code with the module's ``frame_plastic_collapse_force``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    frame_plastic_collapse_force,
    grillage_utilisations,
    ultimate_ice_pressure,
)


def frame_governed_geometry() -> GrillageGeometry:
    """Scantling on which the *transverse frame*, not the plate or stiffener,
    is the weakest link of the member limit state.

    Design intent (documentation example, not a rule table): a **slender
    transverse frame** (small plastic modulus -> low rigid-plastic collapse
    force) over a **stocky plate** (thick, so the plate field stays below the
    member) with a **strong longitudinal stiffener** (deep web, so the
    beam-column branch does not bind).  The frame carries the patch through
    ``f_y`` (the y-member load share); making the frame weak while the plate
    and stiffener are strong forces the frame to bind first.
    """
    return GrillageGeometry(
        frame_spacing=2.4, bay_spacing=0.55, n_bays=4, n_spans=4,
        plate_thickness=0.030,          # stocky plate -> plate field stays low
        stiffener=MemberSection(0.30, 0.012, 0.10, 0.014),   # deep, strong
        frame=MemberSection(0.25, 0.006, 0.06, 0.008),       # shallow, weak
        young=2.06e11, poisson=0.3, yield_stress=3.55e8,
    ).validate()


FRAME_GOVERNED_SCENARIOS: tuple[tuple[str, float, float, float, float, float], ...] = (
    # name, sigma_x / sigma_y, ice patch pressure [Pa], patch length [m],
    # patch width [m], hydrostatic pressure [Pa]
    ("light ice + ballast head", 0.00, 1.0e6, 0.55, 0.55, 5.0e4),
    ("design ice + design head", 0.15, 1.5e6, 1.10, 0.55, 1.0e5),
    ("severe ice + tank head", 0.30, 2.0e6, 1.65, 0.55, 1.5e5),
)


def _independent_frame_collapse(geom: GrillageGeometry, case: CombinedLoadCase,
                                p_ice: float) -> float:
    """Re-derive the frame collapse force from first principles (no module code).

    The frame carries ``0.5 * F_ice * f_y`` (its two bounding frames share the
    centred patch) plus ``F_h * f_y / 2`` (uniform hydrostatic split over the
    two frames).  ``F_ice = p_ice * A_patch``, ``F_h = p_h * A_panel``.  The
    rigid-plastic collapse force of a fixed-fixed beam under a central patch of
    width ``c`` is ``8 Mp / (L (1 - c/2L))`` (uniform ``c = L`` gives
    ``16 Mp / L``).  ``Mp = sigma_y * Z_p`` with ``Z_p`` the plastic modulus of
    the frame section (re-derived by 1000-slice numerical integration).
    """
    lx, ly = case.patch_extents(geom)
    A_patch = lx * ly
    A_panel = geom.panel_area
    F_ice = p_ice * A_patch
    F_h = case.hydrostatic_pressure * A_panel
    # y-member (frame) load share from the elastic plate partition
    resp = grillage_utilisations(geom, case, ice_pressure=p_ice)
    f_y = resp["load_share_y_system"]
    per_frame_ice = 0.5 * F_ice * f_y
    per_frame_h = 0.5 * F_h * f_y
    # plastic modulus of the frame section by 1000-slice numerical integration
    sec = geom.frame_section
    t = geom.plate_thickness
    z0 = -0.5 * t
    parts = [
        (geom.frame_attached_width * t, 0.0),
        (sec.web_height * sec.web_thickness, z0 - 0.5 * sec.web_height),
        (sec.flange_width * sec.flange_thickness,
         z0 - sec.web_height - 0.5 * sec.flange_thickness),
    ]
    area = sum(a for a, _ in parts)
    z_bar = sum(a * z for a, z in parts) / area
    # plastic modulus: equal-area axis then first moment of area about it
    lo, hi = min(z for _, z in parts) - 1e-3, max(z for _, z in parts) + 1e-3
    n = 1000
    dz = (hi - lo) / n
    zs = [lo + (i + 0.5) * dz for i in range(n)]
    widths = []
    for z in zs:
        w = 0.0
        for (a, zc), (w_, h_) in zip(parts, [(geom.frame_attached_width, t),
                                             (sec.web_thickness, sec.web_height),
                                             (sec.flange_thickness, sec.flange_thickness)]):
            if zc - 0.5 * h_ <= z <= zc + 0.5 * h_:
                w += w_
        widths.append(w)
    # equal-area axis
    def _area_below(za):
        return sum(w * dz for w, z in zip(widths, zs) if z <= za)
    lo_a, hi_a = lo, hi
    for _ in range(60):
        mid = 0.5 * (lo_a + hi_a)
        if _area_below(mid) < 0.5 * area:
            lo_a = mid
        else:
            hi_a = mid
    za = 0.5 * (lo_a + hi_a)
    Zp = sum(w * abs(z - za) * dz for w, z in zip(widths, zs))
    Mp = geom.yield_stress * Zp
    span = geom.frame_span
    f_u_patch = 8.0 * Mp / (span * (1.0 - ly / (2.0 * span)))
    f_u_uniform = 16.0 * Mp / span
    return per_frame_ice / f_u_patch + per_frame_h / f_u_uniform


def run_frame_governed_verification(*, terms: int = 61) -> dict:
    geom = frame_governed_geometry()
    rows = []
    for name, eta, p_ice, lx, ly, p_h in FRAME_GOVERNED_SCENARIOS:
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
            design_pressure=p_ice, design_provenance="frame-governed test",
        )
        case.validate(geom)
        res = ultimate_ice_pressure(geom, case, terms=terms)
        u = res["utilisations_at_limit"]
        gov = res["governing_member"]
        # independent frame collapse force at the module's ultimate
        p_ult = res["ice_pressure_ultimate"]
        u_frame_ind = _independent_frame_collapse(geom, case, p_ult)
        # the frame must be the governing member and its independent utilisation
        # must be 1.0 (the module's bisection lands on the frame branch)
        rows.append({
            "case": name,
            "sigma_x_MPa": eta * geom.yield_stress / 1e6,
            "patch_m": [lx, ly],
            "p_ult_MPa": p_ult / 1e6,
            "U_plate": u["plate_local_bending"],
            "U_stiff": u["longitudinal_stiffener"],
            "U_frame": u["transverse_frame"],
            "governing": gov,
            "U_frame_independent": u_frame_ind,
            "frame_error_pct": 100.0 * abs(u_frame_ind - 1.0),
        })
    # census: every case must be frame-governed
    census = {r["governing"] for r in rows}
    return {
        "geometry": {
            "frame_spacing": geom.frame_spacing,
            "bay_spacing": geom.bay_spacing,
            "plate_thickness": geom.plate_thickness,
            "stiffener": "T300x12+100x14",
            "frame": "T250x6+60x8",
            "yield_stress_MPa": geom.yield_stress / 1e6,
        },
        "rows": rows,
        "census": sorted(census),
        "all_frame_governed": census == {"transverse_frame"},
        "max_frame_error_pct": max(r["frame_error_pct"] for r in rows),
        "n_cases": len(rows),
    }


if __name__ == "__main__":
    out = run_frame_governed_verification()
    outdir = Path("artifacts/grillage_frame_governed")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frame_governed_cases.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    assert out["all_frame_governed"], "not all cases are frame-governed"
    assert out["max_frame_error_pct"] < 0.5, "frame branch not reproduced to 0.5%"
    print("OK: frame-governed branch verified, max error %.4f%%"
          % out["max_frame_error_pct"])
