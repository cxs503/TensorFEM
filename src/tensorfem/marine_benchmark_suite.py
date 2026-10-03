"""Executable qualification evidence for the initial marine capability set."""
from __future__ import annotations

import math

from .benchmark_registry import BenchmarkEvidence, _evidence
from .marine_hydrodynamics import AiryWave, MorisonMember, morison_base_actions, morison_quarter_phase_oracle
from .marine_hydrostatics import SCHEMA, analyze_box_barge
from .marine_structures import solve_hull_girder_uniform_load, stiffened_panel_sine_benchmark
from .marine_plate_buckling import run_local_plate_buckling_qualification
from .hull_girder_ultimate import run_hull_girder_ultimate_benchmark
from .marine_fatigue_qualification import run_fatigue_qualification
from .marine_plate_postbuckling import run_plate_postbuckling_qualification
from .hull_girder_progressive import run_progressive_hull_girder_benchmark
from .marine_fatigue_advanced import run_advanced_fatigue_qualification
from .shell4_buckling import run_shell4_buckling_qualification
from .marine_imperfect_strip import run_imperfect_strip_qualification
from .marine_hotspot_fracture import run_hotspot_fracture_qualification


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

    buckling = run_local_plate_buckling_qualification()
    buckling_source = "Navier simply-supported orthotropic plate uniaxial-buckling eigenvalue"
    for row in buckling["cases"]:
        out.append(_evidence(f"marine.plate_buckling.{row['case']}", "local plate buckling",
                             "critical compressive line load", "N/m", buckling_source,
                             row["loads"][-1], row["reference_load"]))

    ultimate = run_hull_girder_ultimate_benchmark()
    ultimate_source = "Closed-form rectangular ideal-elastic-plastic section bending solution"
    out.append(_evidence("marine.hull_ultimate.initial_yield", "hull-girder section yielding",
                         "initial yield moment", "N m", ultimate_source,
                         ultimate["initial_yield"]["computed_moment"],
                         ultimate["initial_yield"]["reference_moment"]))
    out.append(_evidence("marine.hull_ultimate.full_plastic", "hull-girder section yielding",
                         "full plastic moment", "N m", ultimate_source,
                         ultimate["full_plastic"]["computed_moment"],
                         ultimate["full_plastic"]["reference_moment"]))

    fatigue = run_fatigue_qualification()
    fatigue_source = "Independent weld-toe extrapolation, power-law S-N, and Palmgren-Miner hand calculations"
    quantities = {
        "hot_spot.linear": ("linear hot-spot stress", "Pa"),
        "hot_spot.quadratic": ("quadratic hot-spot stress", "Pa"),
        "sn.single_block": ("constant-amplitude fatigue life", "cycles"),
        "miner.multi_block": ("cumulative fatigue damage", "1"),
    }
    for row in fatigue["evidence"]:
        quantity, unit = quantities[row["id"]]
        out.append(_evidence(f"marine.fatigue.{row['id']}", "weld fatigue assessment",
                             quantity, unit, fatigue_source, row["actual"], row["oracle"]))

    postbuckling = run_plate_postbuckling_qualification()
    post_source = "Continuous Navier prestress eigenvalue and bracketed von Karman-Koiter equilibrium branch"
    for row in postbuckling["prestress_eigenbuckling"]:
        out.append(_evidence(f"marine.prestress_buckling.{row['case']}",
                             "prestressed plate buckling", "critical load factor", "1",
                             post_source, row["factors"][-1], row["reference_factor"]))
    path = postbuckling["imperfect_postbuckling"]
    out.append(_evidence("marine.postbuckling.imperfect_path", "imperfect plate postbuckling",
                         "final modal amplitude", "thickness", post_source,
                         path["final_amplitudes"][-1], path["reference_final_amplitude"]))

    progressive = run_progressive_hull_girder_benchmark()
    endpoint = progressive["curve"][-1]
    out.append(_evidence("marine.hull_progressive.moment", "multi-component hull-girder yielding",
                         "high-curvature section moment", "N m",
                         "Closed-form mirrored-component elastic-perfect-plastic section sum",
                         endpoint["moment"], endpoint["reference_moment"]))

    advanced = run_advanced_fatigue_qualification()
    advanced_source = "Independent path extrapolation, ASTM E1049-style rainflow, Miner, and Paris-law calculations"
    advanced_quantities = {
        "hot_spot.path_linear": ("path-extrapolated hot-spot stress", "Pa"),
        "rainflow.triangle_count": ("rainflow cycle count", "cycles"),
        "miner.variable_amplitude": ("variable-amplitude damage", "1"),
        "paris.m2_closed_form": ("Paris-law final crack length", "m"),
    }
    for row in advanced["evidence"]:
        quantity, unit = advanced_quantities[row["id"]]
        out.append(_evidence(f"marine.fatigue_advanced.{row['id']}",
                             "advanced structural fatigue", quantity, unit,
                             advanced_source, row["actual"], row["oracle"]))

    shell_buckling = run_shell4_buckling_qualification()
    out.append(_evidence("marine.shell4_buckling.navier", "Shell4 initial-stress buckling",
                         "critical compressive line load", "N/m",
                         "Navier simply-supported isotropic plate critical load",
                         shell_buckling["meshes"][-1]["value"], shell_buckling["exact"]))

    strip = run_imperfect_strip_qualification()
    strip_units = ("amplitude", "membrane force", "external work")
    for index, quantity in enumerate(strip_units):
        out.append(_evidence(f"marine.imperfect_strip.{quantity.replace(' ', '_')}",
                             "imperfect residual-stress plastic strip", quantity, "normalized",
                             "1280-step/512-fibre independently refined strip reference",
                             getattr(strip["final"], quantity.replace(" ", "_")),
                             strip["oracle"]["final_response"][index]))

    fracture = run_hotspot_fracture_qualification()
    fracture_units = {"hot_spot": "Pa", "scl_membrane": "Pa",
                      "mode_i_k": "Pa sqrt(m)", "mode_i_j": "J/m^2"}
    for row in fracture["evidence"]:
        out.append(_evidence(f"marine.fracture.{row['id']}", "hot-spot and LEFM post-processing",
                             row["id"], fracture_units[row["id"]],
                             "Affine field/SCL and finite-width Mode-I direct analytical identities",
                             row["actual"], row["oracle"]))
    return tuple(out)
