"""Executable qualification evidence for the initial marine capability set."""
from __future__ import annotations

import math

from .benchmark_registry import BenchmarkEvidence, _evidence
from .marine_hydrodynamics import AiryWave, MorisonMember, morison_base_actions, morison_quarter_phase_oracle
from .marine_hydrostatics import SCHEMA, analyze_box_barge
from .marine_structures import solve_hull_girder_uniform_load, stiffened_panel_sine_benchmark


def run_marine_benchmarks() -> tuple[BenchmarkEvidence, ...]:
    """Run eight traceable, non-zero-reference ship/offshore benchmarks."""
    out: list[BenchmarkEvidence] = []

    girder = solve_hull_girder_uniform_load(
        length=120.0, young=2.1e11, area=5.0, inertia=180.0,
        still_water_load=1.4e6, wave_load=0.9e6, elements=24,
    )
    beam_source = "Euler-Bernoulli simply-supported uniform-load solution: w=5qL^4/(384EI), M=qL^2/8"
    out.append(_evidence("marine.hull_girder.deflection", "ship hull girder",
                         "midship deflection", "m", beam_source,
                         girder.computed_midship_deflection, girder.reference_midship_deflection))
    out.append(_evidence("marine.hull_girder.moment", "ship hull girder",
                         "midship bending moment", "N m", beam_source,
                         girder.computed_midship_moment, girder.reference_midship_moment))

    panel = stiffened_panel_sine_benchmark(
        length=4.0, width=2.0, plate_thickness=0.014, plate_young=2.1e11,
        plate_poisson=0.3, stiffener_spacing=0.5, stiffener_area=0.003,
        stiffener_young=2.1e11, stiffener_eccentricity=0.11,
        pressure_amplitude=1.2e5, quadrature_order=18,
    )
    out.append(_evidence("marine.stiffened_panel.navier", "stiffened ship panel",
                         "centre deflection", "m",
                         "Navier one-term simply-supported orthotropic plate under sinusoidal pressure",
                         panel.computed_center_deflection, panel.reference_center_deflection))

    case = {
        "schema": SCHEMA, "name": "qualification-barge", "units": "SI",
        "geometry": {"length": 60.0, "beam": 18.0, "depth": 5.0},
        "fluid": {"density": 1025.0, "gravity": 9.80665},
        "condition": {"mode": "specified_draft", "draft": 2.5, "kg": 3.0,
                      "heel_angle_deg": 6.0},
    }
    hydro = analyze_box_barge(case)
    volume_ref = 60.0 * 18.0 * 2.5
    gm_ref = 2.5 / 2.0 + 18.0**2 / (12.0 * 2.5) - 3.0
    moment_ref = 1025.0 * volume_ref * 9.80665 * gm_ref * math.sin(math.radians(6.0))
    hydro_source = "Archimedes displacement and wall-sided box-barge KB+BM-KG initial-stability identities"
    out.append(_evidence("marine.hydrostatics.displacement", "upright hydrostatics",
                         "displacement volume", "m^3", hydro_source,
                         hydro.displacement_volume, volume_ref))
    out.append(_evidence("marine.hydrostatics.gm", "intact initial stability",
                         "transverse metacentric height", "m", hydro_source, hydro.gm, gm_ref))
    out.append(_evidence("marine.hydrostatics.restoring", "intact initial stability",
                         "small-angle restoring moment", "N m", hydro_source,
                         hydro.restoring_moment, moment_ref))

    wave = AiryWave(height=3.0, period=9.0, depth=25.0)
    member = MorisonMember(diameter=1.2, drag_coefficient=1.05, inertia_coefficient=2.0)
    time = -math.pi / (2.0 * wave.omega)
    actions = morison_base_actions(wave, member, 0.0, time,
                                   current_velocity=0.6, quadrature_order=32)
    oracle = morison_quarter_phase_oracle(wave, member, current_velocity=0.6)
    morison_source = "Airy linear-wave kinematics and DNV-RP-C205 Morison slender-member load equation"
    out.append(_evidence("marine.morison.base_shear", "offshore wave-current loading",
                         "fixed-pile base shear", "N", morison_source,
                         actions["base_shear"], oracle["base_shear"]))
    out.append(_evidence("marine.morison.overturning", "offshore wave-current loading",
                         "fixed-pile overturning moment", "N m", morison_source,
                         actions["overturning_moment"], oracle["overturning_moment"]))
    return tuple(out)
