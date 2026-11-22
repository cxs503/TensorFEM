"""Independently score same-step Ren force and deflection, using the Python stdlib."""

import argparse
import csv
import json
import math
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/data/ice_coupons/bending.csv"
CONTROLS = ROOT / "artifacts/ice_coupons/ren_controls.json"
CALIBRATION = ("02", "03", "04")
HOLDOUT = ("05", "10")
SPAN = 0.180
LIMIT = Decimal("0.10")


def positive(x):
    x = float(x)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("expected a finite positive observation")
    return x


def endpoint(predicted, measured):
    # Decimal handles equality at the original, strict 10% boundary.
    delta = Decimal(str(positive(predicted))) / Decimal(str(positive(measured))) - 1
    return {"signed_relative_error": float(delta), "pass": abs(delta) < LIMIT}


def read_table(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    specimens = {r["id"]: r for r in rows}
    if len(rows) != len(specimens) or set(specimens) != set(CALIBRATION + HOLDOUT):
        raise ValueError("expected the five unique Ren specimens")
    for i, row in specimens.items():
        if row["role"] != ("calibration" if i in CALIBRATION else "holdout"):
            raise ValueError(f"unexpected role for {i}")
    return specimens


def beam_check(specimens):
    derived = {}
    # Hypothetical independent rounding of both dimensions to the last printed
    # millimetre. This range is a sensitivity scenario, not measurement error.
    half_unit = 0.0005
    rounded = {}
    for i, row in specimens.items():
        b, h, force, disp = (
            positive(row[k]) for k in ("width_m", "depth_m", "peak_N", "peak_deflection_m")
        )
        sigma = 1.5 * force * SPAN / (b * h**2)
        if i in HOLDOUT:
            rounded[i] = [
                0.9 * 1.5 * force * SPAN / ((b + half_unit) * (h + half_unit) ** 2),
                1.1 * 1.5 * force * SPAN / ((b - half_unit) * (h - half_unit) ** 2),
            ]
        derived[i] = dict(
            apparent_strength_Pa=sigma,
            endpoint_secant_E_Pa=force * SPAN**3 / (4 * b * h**3 * disp),
            nominal_strain_rate_per_s=6 * h * 0.0025 / SPAN**2,
            strength_limits_Pa=[0.9 * sigma, 1.1 * sigma],
        )
    a, b = sorted(derived[i]["apparent_strength_Pa"] for i in HOLDOUT)
    return dict(
        specimens=derived,
        rounding_scenario=dict(
            half_unit_m=half_unit,
            strength_ranges_Pa=rounded,
            common_limits_Pa=[
                max(v[0] for v in rounded.values()),
                min(v[1] for v in rounded.values()),
            ],
        ),
        strength_gap_Pa=0.9 * b - 1.1 * a,
        minimax_shared_strength_Pa=2 * a * b / (a + b),
        minimum_worst_peak_relative_error=(b - a) / (a + b),
    )


def score_row(row, measured, block_size):
    frames = row["block_peaks"]
    count = row["frame_count"]
    if (
        count != row["steps"] + 1
        or count < 2
        or len(frames) != math.ceil(count / block_size)
        or any(len(f) != 4 or not all(map(math.isfinite, f)) for f in frames)
        or any(a[0] >= b[0] for a, b in zip(frames, frames[1:]))
    ):
        raise ValueError("incomplete, unordered or nonfinite peak observations")
    stroke, force, patch, disp = max(frames, key=lambda frame: frame[1])
    fp = endpoint(force, measured["peak_N"])
    dp = endpoint(disp, measured["peak_deflection_m"])
    if not math.isclose(
        force,
        positive(row["punch_stiffness_N_m"]) * max(0, stroke - patch),
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise ValueError("peak force and paired displacements violate the punch law")
    external = positive(row["external_work_J"])
    total = sum(
        row[k]
        for k in (
            "stored_energy_J",
            "kinetic_energy_J",
            "fracture_dissipation_J",
            "contact_energy_J",
        )
    )
    balance = abs(external + row["initial_kinetic_J"] - total) / external
    if not math.isfinite(balance):
        raise ValueError("nonfinite energy balance")
    numerical = row["postpeak_stop"] == "sustained_postpeak_unloading" and balance < 0.01
    return dict(
        id=row["id"],
        case=row["case"],
        peak_N=force,
        sample_deflection_m=disp,
        punch_stroke_m=stroke,
        patch_deflection_m=patch,
        force=fp,
        deflection=dp,
        energy_balance_rel=balance,
        numerical_pass=numerical,
        joint_pass=numerical and fp["pass"] and dp["pass"],
    )


def score(data, specimens):
    if (
        data["calibration_ids"] != list(CALIBRATION)
        or data["holdout_ids"] != list(HOLDOUT)
        or data["span_m"] != SPAN
        or data["block_size"] <= 0
        or data["frame_columns"]
        != ["punch_stroke_m", "force_N", "patch_deflection_m", "top_midspan_m"]
    ):
        raise ValueError("unexpected split, span or observation protocol")
    beam = beam_check(specimens)
    strength, young = (
        sum(beam["specimens"][i][k] for i in CALIBRATION) / len(CALIBRATION)
        for k in ("apparent_strength_Pa", "endpoint_secant_E_Pa")
    )
    if not (
        math.isclose(positive(data["strength_Pa"]), strength)
        and math.isclose(positive(data["E_Pa"]), young)
    ):
        raise ValueError("shared parameters differ from the calibration-only beam estimates")
    rows = []
    pairs = {}
    for row in data["rows"]:
        i, case = row["id"], row["case"]
        group = pairs.setdefault(case, set())
        if i not in HOLDOUT or i in group:
            raise ValueError(f"unexpected or duplicate specimen {i} for {case}")
        if case == "baseline":
            expected = dict(
                width_m=float(specimens[i]["width_m"]),
                depth_m=float(specimens[i]["depth_m"]),
                E_scale=1.0,
                strength_Pa=strength,
                Gf_J_m2=1.5,
                speed_m_s=0.0025,
                nx=24,
                ny=16,
                punch_ratio=100,
                dt_s=2e-7,
            )
            if (
                any(not math.isclose(row["inputs"][k], v, abs_tol=0.0) for k, v in expected.items())
                or row["inputs"]["integrate_load_patch"]
                or not math.isclose(positive(row["E_Pa"]), young)
                or row["contact_mode"] != "ratio"
            ):
                raise ValueError(f"baseline protocol mismatch for specimen {i}")
        group.add(i)
        rows.append(score_row(row, specimens[i], data["block_size"]))
    if "baseline" not in pairs or any(ids != set(HOLDOUT) for ids in pairs.values()):
        raise ValueError("each configuration needs both holdouts, including baseline")
    baseline = [r for r in rows if r["case"] == "baseline"]
    return dict(
        force_and_deflection_limit_exclusive=float(LIMIT),
        beam=beam,
        baseline_pass_count=sum(r["joint_pass"] for r in baseline),
        baseline_joint_pass=all(r["joint_pass"] for r in baseline),
        rows=rows,
    )


def score_depth_study(data, specimens):
    """Check that both grids series reproduce their archived baselines with exactly one variable."""
    result = score(data, specimens)
    by_key = {(r["id"], r["case"]): r for r in data["rows"]}
    scored = {(r["id"], r["case"]): r for r in result["rows"]}
    depths = (16, 32, 64)
    expected = {"baseline", *(f"depth_{n}" for n in depths)}
    if {case for _, case in by_key} != expected:
        raise ValueError("depth study requires baseline and all three depth grids")
    controls = []
    for i in HOLDOUT:
        base = by_key[i, "baseline"]
        for ny in depths:
            case = f"depth_{ny}"
            row, evaluation = by_key[i, case], scored[i, case]
            if (
                row["inputs"] != base["inputs"] | {"ny": ny, "dt_s": 1e-7}
                or row["contact_mode"] != "fixed_stiffness"
                or any(
                    row[k] != base[k]
                    for k in ("E_Pa", "punch_stiffness_N_m", "loaded_x_count", "cohesive_length_m")
                )
            ):
                raise ValueError(f"confounded depth study for {i} at grid {ny}")
            controls.append(
                dict(
                    id=i,
                    ny=ny,
                    segment_to_characteristic_length=row["inputs"]["depth_m"]
                    / (ny * positive(row["cohesive_length_m"])),
                    peak_N=evaluation["peak_N"],
                    deflection_m=evaluation["sample_deflection_m"],
                    numerical_pass=evaluation["numerical_pass"],
                    joint_pass=evaluation["joint_pass"],
                )
            )
    result["through_depth_controls"] = controls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=CONTROLS)
    parser.add_argument("--measurements", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="exit 1 for a failed baseline gate")
    parser.add_argument(
        "--through-depth",
        action="store_true",
        help="also verify the three-grid through-depth protocol",
    )
    args = parser.parse_args()
    scorer = score_depth_study if args.through_depth else score
    result = scorer(json.loads(args.controls.read_text()), read_table(args.measurements))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in result["rows"]:
        if r["case"] == "baseline":
            print(
                f"{r['id']}: {r['peak_N']:.3f} N, {1e3 * r['sample_deflection_m']:.4f} mm, "
                f"joint={r['joint_pass']}"
            )
    print(f"Baseline passed {result['baseline_pass_count']}/{len(HOLDOUT)}")
    return int(args.check and not result["baseline_joint_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
