"""Paired, one-factor transient controls for the Ren three-point bend specimens."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from ice_coupon_validation import DATA, SPAN, bending_material
from tensorfem.cohesive_bending import BilinearCohesiveLaw, build_three_point_bend
from tensorfem.ice_bend_transient import rupture

OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/ice_coupons/ren_controls.json"
STRENGTH_MPA = (0.70, 0.80, 0.85, 0.90, 0.95, 1.10, 1.20)
BLOCK_SIZE = 8192


class BlockPeaks:
    """One synchronous frame per block. The global maximum survives reduction."""

    def __init__(self):
        self.frames = []
        self.count = 0

    def __call__(self, stroke, force, displacement, midspan):
        if self.count % BLOCK_SIZE == 0:
            self.frames.append([stroke, force, displacement, midspan])
        elif force > self.frames[-1][1]:
            self.frames[-1] = [stroke, force, displacement, midspan]
        self.count += 1


def cases(other):
    yield "baseline", {}
    for mpa in STRENGTH_MPA:
        yield f"strength_{mpa:.2f}", {"strength_Pa": mpa * 1e6}
    for factor in (0.5, 2.0):
        yield f"Gf_x{factor:g}", {"Gf_J_m2": 1.5 * factor}
        yield f"speed_x{factor:g}", {"speed_m_s": 0.0025 * factor}
        yield f"punch_x{factor:g}", {"punch_ratio": 100 * factor}
    for factor in (0.75, 1.25):
        yield f"E_x{factor:g}", {"E_scale": factor}
    yield "other_width", {"width_m": float(other["width_m"])}
    yield "other_depth", {"depth_m": float(other["depth_m"])}
    for nx, ny in [(12, 8), (36, 24)]:
        yield f"mesh_{nx}x{ny}", {"nx": nx, "ny": ny}
    for dt in (1e-7, 3e-7):
        yield f"dt_{dt * 1e6:.1f}us", {"dt_s": dt}
    # Compare the same pressure footprint at three resolutions. The reference
    # for each comparison across quadrature rules is its matching legacy grid.
    for nx, ny in [(12, 8), (24, 16), (36, 24)]:
        yield f"integrated_{nx}x{ny}", {"nx": nx, "ny": ny, "integrate_load_patch": True}


def depth_cases():
    # Baseline anchors the punch and independent scorer to the original protocol.
    yield "baseline", {}
    for ny in (16, 32, 64):
        yield f"depth_{ny}", {"ny": ny, "dt_s": 1e-7}


def run_pair(sample, other, young, law, *, through_depth=False):
    defaults = dict(
        width_m=float(sample["width_m"]),
        depth_m=float(sample["depth_m"]),
        E_scale=1.0,
        strength_Pa=law.peak_traction,
        Gf_J_m2=law.fracture_energy,
        speed_m_s=0.0025,
        nx=24,
        ny=16,
        punch_ratio=100,
        dt_s=2e-7,
        integrate_load_patch=False,
    )
    mesh_punch = {}
    for name, override in depth_cases() if through_depth else cases(other):
        p = defaults | override
        modulus = young * p["E_scale"]
        interface = replace(
            law, young_modulus=modulus, peak_traction=p["strength_Pa"], fracture_energy=p["Gf_J_m2"]
        )
        model = build_three_point_bend(
            span=SPAN,
            depth=p["depth_m"],
            thickness=p["width_m"],
            young_modulus=modulus,
            poisson_ratio=0.3,
            law=interface,
            nx=p["nx"],
            ny=p["ny"],
            integrate_load_patch=p["integrate_load_patch"],
        )
        # Use the archive's compliance-normalised contact for the legacy
        # mesh/dt/contact controls. Hold dimensional Kp fixed for the material,
        # geometry and speed studies so changing E or width cannot retune Kp.
        normalized = name == "baseline" or name.startswith(("mesh_", "dt_", "punch_"))
        # On each grid swap only the projection of the pressure patch.
        grid = (p["nx"], p["ny"])
        reference_grid = (24, 16) if through_depth else grid
        fixed = None if normalized else mesh_punch[reference_grid]
        observer = BlockPeaks()
        result = rupture(
            model,
            fixed_dt=p["dt_s"],
            speed=p["speed_m_s"],
            punch_ratio=p["punch_ratio"],
            punch_stiffness_N_m=fixed,
            observe_step=observer,
        )
        if name == "baseline" or name.startswith("mesh_"):
            mesh_punch[grid] = result["punch_stiffness_N_m"]
        peak = max(observer.frames, key=lambda frame: frame[1])
        if (
            observer.count != result["steps"] + 1
            or peak[:3] != max(result["history"], key=lambda f: f[1])
            or peak[3] != result["peak_midspan_top_deflection_m"]
        ):
            raise RuntimeError(f"incomplete or inconsistent force observations for {name}")
        keys = (
            "steps",
            "step_fraction",
            "postpeak_stop",
            "external_work_J",
            "initial_kinetic_J",
            "stored_energy_J",
            "kinetic_energy_J",
            "fracture_dissipation_J",
            "contact_energy_J",
            "failed_area_fraction",
            "punch_stiffness_N_m",
        )
        yield dict(
            id=sample["id"],
            case=name,
            inputs=p,
            E_Pa=modulus,
            cohesive_length_m=interface.cohesive_length,
            loaded_x_count=len(set(model.nodes[model.load_dofs // 2, 0])),
            contact_mode="ratio" if normalized else "fixed_stiffness",
            **{k: result[k] for k in keys},
            frame_count=observer.count,
            block_peaks=observer.frames,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--through-depth",
        action="store_true",
        help="vary only the depth mesh at fixed dt and absolute punch stiffness",
    )
    args = parser.parse_args()
    output = args.output or (
        OUTPUT.with_name("ren_depth_controls.json") if args.through_depth else OUTPUT
    )
    with (DATA / "bending.csv").open() as stream:
        samples = {row["id"]: row for row in csv.DictReader(stream)}
    calibration = [samples[i] for i in ("02", "03", "04")]
    young, strength = bending_material(calibration)
    law = BilinearCohesiveLaw.from_material(young, strength, 1.5)
    rows = []
    for sample, other in [("05", "10"), ("10", "05")]:
        for row in run_pair(
            samples[sample], samples[other], young, law, through_depth=args.through_depth
        ):
            rows.append(row)
            print(sample, row["case"], flush=True)
    meta = dict(
        calibration_ids=[r["id"] for r in calibration],
        holdout_ids=["05", "10"],
        span_m=SPAN,
        density_kg_m3=917,
        poisson_ratio=0.3,
        load_patch_fraction=0.25,
        E_Pa=young,
        strength_Pa=strength,
        normal_stiffness_Pa_m=law.normal_stiffness,
        tangential_stiffness_Pa_m=law.tangential_stiffness,
        block_size=BLOCK_SIZE,
        frame_columns=["punch_stroke_m", "force_N", "patch_deflection_m", "top_midspan_m"],
    )
    # One reproducible record per line keeps comparisons of reruns local.
    output.write_text(
        json.dumps(meta, indent=2)[:-2]
        + ',\n  "rows": [\n'
        + ",\n".join("    " + json.dumps(r, allow_nan=False) for r in rows)
        + "\n  ]\n}\n"
    )


if __name__ == "__main__":
    main()
