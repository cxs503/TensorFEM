"""Staged driver for the JOUR-DIFFLBM forcing benchmark (keeps each shell call
under the 10 min local budget; the merged output equals the single-shot run)."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfem import lbm_forcing as F

SCRATCH = pathlib.Path(__file__).resolve().parent
ART = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "JOUR-DIFFLBM-REF-VERIFY"
NU = 0.05
NY_SWEEP = 34


def part_a() -> dict:
    rows = []
    for ny in (10, 18, 34):
        for scheme in ("guo", "naive"):
            rows.append(F._march(ny, NU, scheme).as_dict())
    rows.append(F._march(66, NU, "guo").as_dict())
    return {"rows": rows}


def part_b() -> dict:
    sweep = []
    for nu_i in (0.05, 0.2, 0.5):
        tau = 0.5 + 3.0 * nu_i
        entry = {"nu": nu_i, "tau": tau, "amplitude_law": 1.0 + 0.5 / tau}
        for scheme in ("guo", "naive"):
            row = F._march(NY_SWEEP, nu_i, scheme)
            entry[scheme] = row.as_dict()
            entry[f"{scheme}_amplitude_ratio"] = row.u_max_got / row.u_max_ref
        sweep.append(entry)
    prod = F._production_row(NY_SWEEP, NU)
    return {"tau_sweep": sweep, "production": prod.as_dict()}


def merge() -> dict:
    a = json.loads((SCRATCH / "stageA.json").read_text())
    b = json.loads((SCRATCH / "stageB.json").read_text())
    rows = a["rows"]
    result = {
        "reference": F.SCHEME_REFERENCE.bib_line(),
        "reference_doi": F.SCHEME_REFERENCE.doi,
        "reference_registry_key": "guo2002_forcing",
        "reference_status": F.SCHEME_REFERENCE.status,
        "case": "force-driven plane Poiseuille flow (periodic in x, "
                "half-way bounce-back walls at y=1 and y=ny-1)",
        "analytic": "u(y) = F_x (y - 1)(ny - 1 - y) / (2 rho nu)",
        "resolutions": [10, 18, 34],
        "rows": rows,
        "tau_sweep": b["tau_sweep"],
        "production": b["production"],
        "convergence_order": F._convergence_order(
            [F.BenchRow(**{k: r[k] for k in ("solver", "scheme", "ny", "nu",
                                             "force_x", "u_max_ref", "u_max_got",
                                             "rel_err_u_max", "rel_l2_profile",
                                             "steps", "converged")},
                          extra={kk: vv for kk, vv in r.items()
                                 if kk not in ("solver", "scheme", "ny", "nu",
                                               "force_x", "u_max_ref",
                                               "u_max_got", "rel_err_u_max",
                                               "rel_l2_profile", "steps",
                                               "converged")})
             for r in rows]),
        "driver": "benchmarks/m22839_stage.py (staged to keep each shell call "
                  f"under the local budget; identical to "
                  "jour_difflbm_ref_verify.py's forcing leg)",
    }
    result["gate"] = {
        "criterion": "Guo scheme: rel_err_u_max < 3 % and rel_l2 < 1 % at the "
                     "finest grid, and smaller than the uncorrected scheme at "
                     "the same grid",
        "guo_finest": F._finest_from_rows(rows, "guo"),
        "naive_finest": F._finest_from_rows(rows, "naive"),
        "production": b["production"],
        "passed": None,
    }
    result["gate"]["passed"] = F.benchmark_passed(result)
    return result


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "a":
        (SCRATCH / "stageA.json").write_text(json.dumps(part_a(), indent=2))
        print("stage A done")
    elif what == "b":
        (SCRATCH / "stageB.json").write_text(json.dumps(part_b(), indent=2))
        print("stage B done")
    else:
        out = merge()
        ART.mkdir(parents=True, exist_ok=True)
        (ART / "forcing_poiseuille.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({
            "rows": [(r["scheme"], r["solver"], r["ny"],
                      round(r["rel_err_u_max"], 6),
                      round(r["rel_l2_profile"], 6)) for r in out["rows"]],
            "order": out["convergence_order"]["mean_order"],
            "production": {k: out["production"][k]
                           for k in ("ny", "rel_err_u_max", "rel_l2_profile",
                                     "steps", "converged")},
            "gate_passed": out["gate"]["passed"],
        }, indent=2))