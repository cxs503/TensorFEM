"""H1-BROKEN-ICE-DEM study driver: floatation, heave, wedge contact, channel clearing.

Writes ``artifacts/broken_ice_floe_field/study.json`` plus three figures used by
the published verification report.  Every number in the report is read back from
this JSON (no hand transcription).

Run:  python scripts/broken_ice_floe_field_study.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from tensorfem.broken_ice_floe_field import (
    FloeFieldConfig,
    archimedes_equilibrium,
    heave_natural_frequency,
    measure_heave_period,
    simulate_floe_field,
    simulate_free_float,
    simulate_wedge_floe_sideways,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "broken_ice_floe_field"
OUT.mkdir(parents=True, exist_ok=True)

DT_FIELD = 0.018  # max dt allowed by validate() for the 0.4 m floe is 0.019


def floatation_sweep() -> list[dict]:
    rows = []
    for ratio in (0.85, 0.88, 0.90, 0.92, 0.95):
        water_density = 1025.0
        ice_density = ratio * water_density
        cfg = FloeFieldConfig(ice_density=ice_density, water_density=water_density, dt=0.005)
        out = simulate_free_float(
            0.6, radius=2.5, config=cfg, damping_ratio=0.8, t_end=25.0
        )
        analytic = archimedes_equilibrium(0.6, ice_density, water_density)
        rows.append(
            {
                "density_ratio": ratio,
                "ice_density": ice_density,
                "freeboard_analytic": analytic.freeboard,
                "freeboard_measured": out["freeboard_measured"],
                "draft_analytic": analytic.draft,
                "draft_measured": out["draft_measured"],
                "rel_error": out["relative_error"],
            }
        )
    return rows


def heave_sweep() -> list[dict]:
    rows = []
    for thickness in (0.3, 0.5, 0.8, 1.0, 1.2):
        out = measure_heave_period(thickness, radius=2.5, offset=0.002, t_end=16.0)
        rows.append(
            {
                "thickness": thickness,
                "period_analytic": out["period_analytic"],
                "period_measured": out["period_measured"],
                "rel_error": out["relative_error"],
                "n_periods": out["n_periods"],
                "omega_analytic": heave_natural_frequency(thickness),
            }
        )
    return rows


def wedge_sweep() -> list[dict]:
    rows = []
    for angle in (15.0, 20.0, 30.0, 40.0, 50.0):
        out = simulate_wedge_floe_sideways(angle, speed=1.0, t_end=15.0)
        rows.append({"half_angle_deg": angle, **out})
    return rows


def clearing_sweep() -> tuple[list[dict], list[dict], list[dict]]:
    speeds = [0.5, 1.0, 1.5, 2.0, 3.0]
    speed_rows = []
    canonical = None
    for speed in speeds:
        cfg = FloeFieldConfig(structure_speed=speed, dt=DT_FIELD)
        result = simulate_floe_field(cfg)
        if speed == 1.5:
            canonical = result
        speed_rows.append(
            {
                "speed": speed,
                **{k: v for k, v in result.corridor.items() if k != "half_width"},
                "half_width": result.corridor["half_width"],
                "max_penetration": result.max_penetration,
                "n_floes": cfg.n_floes,
                "froude_like": speed / math.sqrt(cfg.gravity * float(cfg.thickness_range[1])),
            }
        )
    angle_rows = []
    for angle in (20.0, 30.0, 40.0):
        cfg = FloeFieldConfig(bow_half_angle_deg=angle, structure_speed=1.5, dt=DT_FIELD)
        result = simulate_floe_field(cfg)
        angle_rows.append(
            {
                "half_angle_deg": angle,
                "half_width": result.corridor["half_width"],
                "clearing_rate_centroid": result.corridor["clearing_rate_centroid"],
                "clearing_rate_coverage": result.corridor["clearing_rate_coverage"],
                "mean_lateral_displacement": result.corridor["mean_lateral_displacement"],
                "dragged_forward_fraction": result.corridor["dragged_forward_fraction"],
                "residual_fraction": result.corridor["residual_fraction"],
                "n_corridor_initial": result.corridor["n_corridor_initial"],
            }
        )
    friction_rows = []
    for mu in (0.0, 0.3):
        cfg = FloeFieldConfig(friction=mu, structure_speed=1.5, dt=DT_FIELD)
        result = simulate_floe_field(cfg)
        friction_rows.append(
            {
                "friction": mu,
                "clearing_rate_centroid": result.corridor["clearing_rate_centroid"],
                "clearing_rate_coverage": result.corridor["clearing_rate_coverage"],
                "mean_lateral_displacement": result.corridor["mean_lateral_displacement"],
                "dragged_forward_fraction": result.corridor["dragged_forward_fraction"],
                "residual_fraction": result.corridor["residual_fraction"],
                "n_corridor_initial": result.corridor["n_corridor_initial"],
            }
        )
    if canonical is not None:
        torch.save(
            {
                "positions0": canonical.positions0,
                "positions": canonical.positions,
                "radii": canonical.radii,
                "heave": canonical.heave,
            },
            OUT / "canonical_field.pt",
        )
    return speed_rows, angle_rows, friction_rows


def main() -> dict:
    t0 = time.time()
    payload: dict = {"dt_field": DT_FIELD}
    payload["floatation"] = floatation_sweep()
    payload["heave"] = heave_sweep()
    payload["wedge"] = wedge_sweep()
    speeds, angles, frictions = clearing_sweep()
    payload["clearing_speed"] = speeds
    payload["clearing_bow_angle"] = angles
    payload["clearing_friction"] = frictions
    payload["runtime_s"] = time.time() - t0
    (OUT / "study.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()