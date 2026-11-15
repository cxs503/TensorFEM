import dataclasses, math
from tensorfem.grillage_ice_ultimate import (case_study_geometry, GrillageGeometry,
    CombinedLoadCase, grillage_utilisations)
from tensorfem.grillage_beam_grid import beam_grid_case_study

g = case_study_geometry()
for nsp, widths in ((4,(0.35,0.70,1.05)),(3,(0.35,0.70,1.05))):
    geom = dataclasses.replace(g, n_spans=nsp)
    for w in widths:
        case = CombinedLoadCase(ice_pressure=6e6, patch_length=w, patch_width=0.35)
        lx, ly = case.patch_extents(geom)
        span_x = geom.n_spans*geom.frame_spacing
        rect = (0.5-0.5*lx/span_x, 0.5+0.5*lx/span_x, 0.5-0.5*ly/geom.frame_span, 0.5+0.5*ly/geom.frame_span)
        u = grillage_utilisations(geom, case)
        # unit-pressure beam grid
        res = beam_grid_case_study(geom, sub=8, load_split=0.5, patch=rect, pressure=1.0)
        p=6e6
        ms = res["moment_x_max"]*p; mf = res["moment_y_max"]*p
        mpf = geom.yield_stress*geom.frame_section.plastic_modulus
        print(f"n_spans={nsp} w={w:.2f} smeared: Mst={u['per_stiffener_ice_force']:.0f}N Mfr={u['per_frame_ice_force']:.0f}N Ufr={u['transverse_frame']:.3f} Mst={u['m_plate']:.0f}"
              f" | grid: Mst={ms/1e3:.1f}kNm Mfr={mf/1e3:.1f}kNm Ufr={mf/mpf:.3f} mp={mpf/1e6:.3f}MNm fx_energy={res['share_x_energy']:.4f}")
