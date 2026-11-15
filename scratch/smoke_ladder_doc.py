"""Smoke-run the ladder aggregation + document generator on synthetic artifacts.

Guards the *generator* (not the physics): as soon as more than one mesh landed,
the old monotonicity check indexed a record dict with ``1`` and raised
``KeyError`` -- exactly at the moment the three-mesh ladder became available.
"""
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path("/srv/tensor/agent/work/m21966")
TMP = ROOT / "scratch" / "ladder_smoke"
TMP.mkdir(parents=True, exist_ok=True)
for stale in TMP.glob("*.json"):
    stale.unlink()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ladder = load_module("ladder_drv", ROOT / "scripts" / "patch_plate_plastic_mesh_ladder.py")
figs = load_module("ladder_fig", ROOT / "scripts" / "patch_plate_plastic_ladder_figures.py")
ladder.ART = TMP
figs.ART = TMP
figs.DOCS = TMP

BOUNDS = {"ilyushin_sine": 1.5151816676552382e6, "yield_line": 1.842240e6,
          "strip_lower": 0.8187733333333334e6, "johansen_twist": 2.379982e6}
PY = 0.8692336468059256e6


def record(nx, limit, first_yield_err, perm, geometry="free", w_scale=1.0):
    path = [{"p": 0.2 * PY * i, "p_over_py": 0.2 * i,
             "w": 0.2 * i * PY * 4.4e-8 * w_scale, "alpha": 0.0,
             "moment_ratio": min(1.0, 0.5 * i), "sigma_surface_over_fy": 0.5 * i}
            for i in range(1, 16)]
    return {
        "nx": nx, "layers": 4, "geometry": geometry,
        "section_capacity": {"ratio": 0.98},
        "references": {"bounds": BOUNDS, "p_yield_analytic": PY,
                       "yield_stress": 383.8e6, "grade": "DH36",
                       "temperature_c": -40.0, "hardening": 1e6,
                       "m_p": 38380.0, "patch": [0.5, 0.5], "a": 1.0,
                       "b": 1.0, "t": 0.02},
        "limit_estimates": {"tight": {"limit_load": limit, "spread": 0.05,
                                      "measurements": {}},
                            "loose": {"limit_load": limit, "spread": 0.12,
                                      "measurements": {}}},
        "first_yield": PY * (1 + first_yield_err),
        "first_yield_error_vs_analytic": first_yield_err,
        "ramp_completed": True, "ramp_truncated": False,
        "measured_limit_load": None, "limit_point_searched": False,
        "peak_factor": 2.6 * PY, "permanent_deflection": perm,
        "permanent_set_valid": True, "unload_converged_steps": 3,
        "unload_failure": None, "elapsed_s": 42.0, "path": path,
        "errors": {k: limit / v - 1.0 for k, v in BOUNDS.items()},
    }


for nx, limit, err, perm in ((8, 1.1997e6, 0.008, 5.2e-3),
                             (12, 1.16e6, 0.004, 5.0e-3),
                             (16, 1.14e6, 0.003, 4.9e-3)):
    (TMP / f"patch_plate_plastic_ladder_nx{nx}_free.json").write_text(
        json.dumps(record(nx, limit, err, perm), indent=2))
(TMP / "patch_plate_plastic_ladder_nx8_membrane.json").write_text(
    json.dumps(record(8, 2.4e6, 0.008, 9e-3, geometry="membrane"), indent=2))

assert ladder.analyse() == 0
doc = figs.main()
print("generated:", doc)
text = (TMP / "CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md").read_text()
assert "3-level mesh convergence" in text
assert "KeyError" not in text
print(text[text.index("## 6."):text.index("## 7.")])