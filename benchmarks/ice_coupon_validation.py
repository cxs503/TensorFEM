"""Held-out ice coupon comparisons. Recompute from the repository root."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from tensorfem.cohesive_bending import (
    BilinearCohesiveLaw,
    build_three_point_bend,
    solve_cmod_control,
)
from tensorfem.ice_bend_initiation import first_crack
from tensorfem.ice_bend_stroke import stroke_control
from tensorfem.ice_bend_transient import rupture
from tensorfem.ice_indentation import spherical_bearing_load, spherical_bearing_work
from tensorfem.indentation_plasticity import J2Material, run_indentation_cycles

DATA = Path(__file__).parent / "data" / "ice_coupons"
OUT = Path("artifacts/ice_coupons")
SPAN = 0.18
# Engineering screening thresholds chosen for this study, not source uncertainty.
PEAK_TOL = 0.10
CURVE_TOL = 0.10
WORK_TOL = 0.10


def bending_material(specimens):
    calibration = []
    for row in specimens:
        if row["role"] == "calibration":
            width, depth, load, delta = [
                float(row[k]) for k in ("width_m", "depth_m", "peak_N", "peak_deflection_m")
            ]
            calibration.append(
                [
                    load * SPAN**3 / (4 * width * depth**3 * delta),
                    1.5 * load * SPAN / (width * depth**2),
                ]
            )
    return np.mean(calibration, axis=0)


def bending():
    with (DATA / "bending.csv").open() as file:
        specimens = list(csv.DictReader(file))
    young, strength = bending_material(specimens)
    rows = []
    for row in specimens:
        if row["role"] != "holdout":
            continue
        configs = [(24, 16, 50, 0.4)]
        if row["id"] == "05":
            configs += [
                (12, 8, 50, 0.4),
                (36, 24, 50, 0.4),
                (24, 16, 25, 0.4),
                (24, 16, 100, 0.4),
                (24, 16, 50, 0.2),
                (24, 16, 50, 0.8),
            ]
        for nx, ny, penalty, increment in configs:
            law = BilinearCohesiveLaw.from_material(young, strength, 1.5, penalty_ratio=penalty)
            model = build_three_point_bend(
                span=SPAN,
                depth=float(row["depth_m"]),
                thickness=float(row["width_m"]),
                young_modulus=young,
                poisson_ratio=0.3,
                law=law,
                nx=nx,
                ny=ny,
            )
            res = solve_cmod_control(model, initial_step=increment * law.damage_opening)
            force_error = abs(res.peak_load / float(row["peak_N"]) - 1)
            disp_error = abs(res.peak_deflection / float(row["peak_deflection_m"]) - 1)
            rows.append(
                dict(
                    id=row["id"],
                    nx=nx,
                    ny=ny,
                    penalty_ratio=penalty,
                    initial_cmod_in_damage_openings=increment,
                    converged=bool(res.converged),
                    stop_reason=res.stop_reason,
                    peak_N=res.peak_load,
                    peak_deflection_m=res.peak_deflection,
                    peak_error=force_error,
                    deflection_error=disp_error,
                    fracture_dissipation_J=float(res.dissipated_path[-1]),
                    external_work_J=float(res.external_work[-1]),
                    final_crack_height_m=float(res.crack_length[-1]),
                    force_N=res.load.tolist(),
                    deflection_m=res.deflection.tolist(),
                    measured_peak_N=float(row["peak_N"]),
                    measured_peak_deflection_m=float(row["peak_deflection_m"]),
                    endpoints_pass=bool(res.converged and max(force_error, disp_error) < PEAK_TOL),
                )
            )
    return dict(
        calibration_ids=["02", "03", "04"],
        held_out_ids=["05", "10"],
        E_Pa=young,
        strength_Pa=strength,
        Gf_input_J_m2=1.5,
        endpoint_tolerance=PEAK_TOL,
        crack_plane="prescribed at midspan; location cannot be independently predicted by this mesh",
        measured_fracture_work_J=None,
        measured_full_curves=None,
        physical_validation_pass=False,
        rows=rows,
    )


def stroke():
    with (DATA / "bending.csv").open() as file:
        specimens = list(csv.DictReader(file))
    young, strength = bending_material(specimens)
    rows = []
    for sample in specimens:
        if sample["role"] != "holdout":
            continue
        configs = [(24, 16, 50, 320)]
        if sample["id"] == "05":
            configs += [
                (12, 8, 50, 320),
                (36, 24, 50, 320),
                (24, 16, 25, 320),
                (24, 16, 100, 320),
                (24, 16, 50, 160),
                (24, 16, 50, 640),
            ]
        for nx, ny, penalty, steps in configs:
            law = BilinearCohesiveLaw.from_material(young, strength, 1.5, penalty_ratio=penalty)
            model = build_three_point_bend(
                span=SPAN,
                depth=float(sample["depth_m"]),
                thickness=float(sample["width_m"]),
                young_modulus=young,
                poisson_ratio=0.3,
                law=law,
                nx=nx,
                ny=ny,
            )
            row = stroke_control(model, max_stroke=0.0016, increments=steps)
            measured, deflection = float(sample["peak_N"]), float(sample["peak_deflection_m"])
            row.update(
                id=sample["id"],
                nx=nx,
                ny=ny,
                penalty_ratio=penalty,
                measured_peak_N=measured,
                measured_peak_deflection_m=deflection,
                force_error=abs(row["peak_N"] / measured - 1),
                deflection_error=abs(row["peak_deflection_m"] / deflection - 1),
            )
            rows.append(row)
    return dict(
        peak_tolerance=PEAK_TOL,
        rows=rows,
        physical_validation_pass=all(
            r["converged"] and max(r["force_error"], r["deflection_error"]) < PEAK_TOL for r in rows
        ),
    )


def transient():
    with (DATA / "bending.csv").open() as file:
        specimens = list(csv.DictReader(file))
    young, strength = bending_material(specimens)
    rows = []
    for sample in specimens:
        if sample["role"] != "holdout":
            continue
        configs = [(24, 16, 100, 2e-7)]
        if sample["id"] == "05":
            configs += [
                (12, 8, 100, 2e-7),
                (36, 24, 100, 2e-7),
                (24, 16, 50, 2e-7),
                (24, 16, 200, 2e-7),
                (24, 16, 100, 1e-7),
                (24, 16, 100, 3e-7),
            ]
        for nx, ny, penalty, dt in configs:
            law = BilinearCohesiveLaw.from_material(young, strength, 1.5)
            model = build_three_point_bend(
                span=SPAN,
                depth=float(sample["depth_m"]),
                thickness=float(sample["width_m"]),
                young_modulus=young,
                poisson_ratio=0.3,
                law=law,
                nx=nx,
                ny=ny,
            )
            row = rupture(model, punch_ratio=penalty, fixed_dt=dt)
            row.update(
                id=sample["id"],
                nx=nx,
                ny=ny,
                measured_peak_N=float(sample["peak_N"]),
                measured_peak_deflection_m=float(sample["peak_deflection_m"]),
            )
            row["force_error"] = abs(row["peak_N"] / row["measured_peak_N"] - 1)
            row["deflection_error"] = abs(
                row["peak_deflection_m"] / row["measured_peak_deflection_m"] - 1
            )
            row["numerical_pass"] = bool(
                row["postpeak_stop"] == "sustained_postpeak_unloading"
                and row["energy_balance_rel"] < 0.01
            )
            rows.append(row)
    return dict(
        energy_tolerance=0.01,
        peak_tolerance=PEAK_TOL,
        rows=rows,
        physical_validation_pass=all(
            r["numerical_pass"] and max(r["force_error"], r["deflection_error"]) < PEAK_TOL
            for r in rows
        ),
    )


def initiation():
    with (DATA / "bending.csv").open() as file:
        specimens = list(csv.DictReader(file))
    young, strength = bending_material(specimens)
    rows = []
    for row in specimens:
        if row["role"] != "holdout":
            continue
        configs = [(24, 16, 100, 0.001)]
        if row["id"] == "05":
            configs += [
                (12, 8, 100, 0.001),
                (36, 24, 100, 0.001),
                (24, 16, 50, 0.001),
                (24, 16, 200, 0.001),
                (24, 16, 100, 0.0005),
                (24, 16, 100, 0.002),
            ]
        for nx, ny, penalty, dt in configs:
            law = BilinearCohesiveLaw.from_material(young, strength, 1.5)
            model = build_three_point_bend(
                span=SPAN,
                depth=float(row["depth_m"]),
                thickness=float(row["width_m"]),
                young_modulus=young,
                poisson_ratio=0.3,
                law=law,
                nx=nx,
                ny=ny,
            )
            result = first_crack(model, dt=dt, penalty_ratio=penalty)
            result.update(
                id=row["id"],
                nx=nx,
                ny=ny,
                penalty_ratio=penalty,
                measured_peak_N=float(row["peak_N"]),
                measured_deflection_m=float(row["peak_deflection_m"]),
            )
            rows.append(result)
    return dict(stroke_speed_m_s=0.0025, density_kg_m3=917, rows=rows)


def prydz():
    # Displacement-driven elastic response at an independently observed failure
    # stroke. The same nominal geometry is used for all three unlabelled traces.
    data = np.genfromtxt(
        DATA / "prydz_curves.csv", delimiter=",", names=True, dtype=None, encoding="utf8"
    )
    rows = []
    for ice in ("columnar", "mixed", "snow"):
        points = data[data["ice_type"] == ice]
        force, disp = points["force_N"], points["deflection_m"]
        peak = int(force.argmax())
        f, h = force[: peak + 1], disp[: peak + 1]
        used = (h >= 0.1 * h[-1]) & (h <= 0.4 * h[-1])
        stiffness = float((h[used] @ f[used]) / (h[used] @ h[used]))
        pred = stiffness * h
        measured_pre = float(trapezoid(f, h))
        measured_total = float(trapezoid(force, disp))
        w = 0.5 * stiffness * h[-1] ** 2
        rows.append(
            dict(
                ice_type=ice,
                stiffness_N_m=stiffness,
                peak_N=float(f[-1]),
                failure_deflection_m=float(h[-1]),
                predicted_N_at_failure=float(pred[-1]),
                heldout_nrmse=float(
                    np.sqrt(np.mean((pred[h > 0.4 * h[-1]] - f[h > 0.4 * h[-1]]) ** 2)) / f[-1]
                ),
                measured_prepeak_work_J=measured_pre,
                measured_trace_work_J=measured_total,
                predicted_elastic_work_J=w,
                prepeak_work_error=abs(w / measured_pre - 1),
                abrupt_failure_total_work_error=abs(w / measured_total - 1),
            )
        )
    return dict(
        input="measured displacement and 10–40% of the loading stroke for stiffness",
        limitation="work under the plotted force-deflection path; failure stroke is prescribed; "
        "machine compliance, fragment kinetic energy and fracture energy are not separated",
        rows=rows,
    )


def indentation():
    data = np.genfromtxt(DATA / "indentation.csv", delimiter=",", names=True)
    force, depth = data["force_N"], data["depth_m"]
    peak = int(np.argmax(force))
    f, h = force[: peak + 1], depth[: peak + 1]
    # Use force as the independent variable only on the monotonically loaded branch.
    order = np.argsort(f)
    cal_depth = float(np.interp(20.0, f[order], h[order]))

    def run(yield_stress, peak_h, size, penalty, steps):
        return run_indentation_cycles(
            nr=size,
            nz=size,
            width=0.008,
            depth=0.008,
            grading=2.0,
            radius=0.0016,
            peak_indentation=peak_h,
            load_steps=steps,
            unload_steps=steps,
            material=J2Material(0.75e9, 0.33, yield_stress),
            penalty_factor=penalty,
            strict_unloading=False,
        ).cycles[0]

    from scipy.optimize import brentq

    def calibration(y):
        cycle = run(y, cal_depth, 16, 100, 10)
        if not cycle.converged:
            raise ValueError("unconverged calibration")
        return cycle.steps[-1].load - 20.0

    yield_stress = brentq(calibration, 1e6, 150e6, xtol=2500.0)
    measured_work = float(trapezoid(force, depth))
    rows = []
    for size, penalty, steps in [
        (16, 100, 20),
        (12, 100, 20),
        (24, 100, 20),
        (16, 50, 20),
        (16, 200, 20),
        (16, 100, 10),
        (16, 100, 40),
    ]:
        cycle = run(yield_stress, float(h[-1]), size, penalty, steps)
        depth_fe, force_fe = cycle.loading_curve()
        valid = (f >= 30) & (f <= 78)
        predicted = np.interp(h[valid], np.r_[0, depth_fe], np.r_[0, force_fe])
        nrmse = float(np.sqrt(np.mean((predicted - f[valid]) ** 2)) / f.max())
        peak_error = abs(force_fe[-1] / f.max() - 1)
        work_error = abs(cycle.external_work / measured_work - 1)
        rows.append(
            dict(
                size=size,
                penalty_factor=penalty,
                load_steps=steps,
                peak_N=float(force_fe[-1]),
                converged=cycle.converged,
                force_N=force_fe.tolist(),
                depth_m=depth_fe.tolist(),
                heldout_load_nrmse=nrmse,
                peak_rel_error=peak_error,
                cycle_work_J=cycle.external_work,
                work_error=work_error,
                residual_depth_m=cycle.residual_depth,
                closure_error=cycle.energy_closure_corrected,
                release_plastic_flow=cycle.release_plastic_flow,
                passed=bool(
                    cycle.converged
                    and nrmse < CURVE_TOL
                    and peak_error < PEAK_TOL
                    and work_error < WORK_TOL
                ),
            )
        )
    # The published pressure-area relation gives a separate constitutive limit.
    # Fit exactly the same 20 N point as the J2 model; retain both verdicts.
    radius = 0.0016
    hardness = 20.0 / spherical_bearing_load(cal_depth, radius, 1.0)
    predicted = np.array([spherical_bearing_load(x, radius, hardness) for x in h[valid]])
    predicted_peak = spherical_bearing_load(h[-1], radius, hardness)
    predicted_work = spherical_bearing_work(h[-1], radius, hardness)
    bearing = dict(
        hardness_Pa=hardness,
        heldout_load_nrmse=float(np.sqrt(np.mean((predicted - f[valid]) ** 2)) / f.max()),
        peak_rel_error=abs(predicted_peak / f.max() - 1),
        work_error=abs(predicted_work / measured_work - 1),
        peak_N=predicted_peak,
        loading_work_J=predicted_work,
        assumption="fully plastic spherical cap, rigid unloading; no unloading curve or creep prediction",
    )
    bearing["passed"] = bool(
        bearing["heldout_load_nrmse"] < CURVE_TOL
        and bearing["peak_rel_error"] < PEAK_TOL
        and bearing["work_error"] < WORK_TOL
    )
    return dict(
        radius_m=0.0016,
        speed_m_s=3.8e-6,
        measured_peak_N=float(f.max()),
        measured_net_work_J=measured_work,
        calibration_load_N=20.0,
        calibration_depth_m=cal_depth,
        heldout_load_range_N=[30, 78],
        yield_stress_Pa=yield_stress,
        E_Pa=0.75e9,
        nu=0.33,
        bearing_limit=bearing,
        thresholds=dict(curve=CURVE_TOL, peak=PEAK_TOL, work=WORK_TOL),
        rows=rows,
        physical_validation_pass=all(r["passed"] for r in rows),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=["bending", "indentation", "initiation", "prydz", "stroke", "transient"]
    )
    case = parser.parse_args().case
    result = {
        "bending": bending,
        "indentation": indentation,
        "initiation": initiation,
        "stroke": stroke,
        "transient": transient,
        "prydz": prydz,
    }[case]()
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / (case + ".json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(case, result.get("physical_validation_pass"), len(result["rows"]))
