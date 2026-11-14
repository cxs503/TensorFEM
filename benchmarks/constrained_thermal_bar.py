"""Evidence driver for ``CASE-THERMAL-BAR`` (constrained thermal stress, TensorFEM).

One run produces every artefact of the case; nothing in the report is typed by
hand:

* ``benchmarks/constrained_thermal_bar_evidence.json`` -- parameters, the
  closed-form reference with its provenance hash, every computed number, the
  predeclared gates and their verdicts;
* ``docs/constrained-thermal-bar-verification.md`` -- the six-section report,
  formatted from that dictionary;
* ``docs/assets/constrained_thermal_*.png`` -- field cloud + reference overlay,
  axial-stress profile, h-convergence, sensitivity panel.

Run::

    PYTHONPATH=src python benchmarks/constrained_thermal_bar.py

The systems are small (largest is a 4x4x4 hex block, 375 dofs), so the torch
thread pool is capped.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import torch                          # noqa: E402

torch.set_num_threads(int(os.environ.get("TENSORFEM_THREADS", "4")))

from tensorfem import constrained_thermal_bar as ctb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
EVIDENCE = ROOT / "benchmarks" / "constrained_thermal_bar_evidence.json"
REPORT = DOCS / "constrained-thermal-bar-verification.md"

DT = 100.0          # temperature rise of the governing case [K]
T_REF = 20.0        # stress-free reference temperature [degC]
LENGTH = 1.0        # bar length [m]
SIDE = 0.20         # hex specimen side [m]
BAR_LEVELS = (1, 2, 4, 8, 16)
HEX_LEVELS = (1, 2, 3, 4)
NU_LEVELS = (0.0, 0.2, 0.3, 0.45)
DT_LEVELS = (25.0, 50.0, 100.0, 150.0)
GRADED_BAR_LEVELS = (2, 4, 8, 16, 32)     # axial elements, 1-D path
GRADED_HEX_LEVELS = (2, 4, 8, 16)         # axial elements, 3-D path
GRADED_SIDE = 0.10                        # lateral extent of the graded block [m]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "dtype": "torch.float64",
        "torch_threads": torch.get_num_threads(),
        "git_sha": _git_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "determinism": "direct dense solves, no sampling, no seed; reruns are bit-identical",
    }


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def bar_study(steel) -> list[dict]:
    """1-D FEM (bar element) vs ``-E alpha dT``, h-refinement on the 1-D path."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    ref = ctb.restrained_uniaxial_stress(E, alpha, DT)
    rows = []
    for n in BAR_LEVELS:
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=LENGTH)
        sigma = float(res.sigma.mean())
        rows.append({
            "n_elem": n,
            "h_m": LENGTH / n,
            "sigma_Pa": sigma,
            "sigma_MPa": sigma / 1e6,
            "reference_Pa": ref,
            "reference_MPa": ref / 1e6,
            "rel_error": ctb.rel_error(sigma, ref),
            "pass_task_tol": abs(ctb.rel_error(sigma, ref)) < ctb.TASK_REL_TOL,
            "pass_roundoff": abs(ctb.rel_error(sigma, ref)) < ctb.ROUNDOFF_REL_TOL,
            "sigma_spread_over_elements": float(res.sigma.max() - res.sigma.min()),
            "max_abs_displacement": float(res.displacement.abs().max()),
            "reaction_sum_N": float(res.reaction.sum()),
        })
    return rows


def bar_quadratic_study(steel) -> dict:
    """Genuine h-convergence: ``T = T_ref + dT (x/L)**2``, both ends fixed.

    The exact solution has a constant axial force ``N = -E A alpha dT / 3``, but
    linear elements interpolate the quadratic ``T`` only to ``O(h^2)``, so the
    element stress carries a real discretisation error -- this is the study that
    measures an order of accuracy instead of round-off.
    """
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    ref = ctb.restrained_uniaxial_stress(E, alpha, DT) / 3.0
    rows, errors, hs = [], [], []
    for n in (2, 4, 8, 16, 32):
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=LENGTH,
                            temperature="quadratic")
        sigma = float(res.sigma.mean())
        err = abs(ctb.rel_error(sigma, ref))
        rows.append({
            "n_elem": n, "h_m": LENGTH / n,
            "sigma_Pa": sigma, "reference_Pa": ref,
            "abs_rel_error": err,
            "u_rms_error_m": float(torch.sqrt(((res.displacement
                                                - res.exact_displacement) ** 2).mean())),
        })
        errors.append(err)
        hs.append(LENGTH / n)
    order = ctb.observed_order(errors, hs)
    return {
        "reference_Pa": ref,
        "reference_expression": "-E alpha dT / 3 (constant axial force, quadratic T)",
        "rows": rows,
        "observed_order": order,
        "theory_order": 2.0,
        "order_gate": abs(order - 2.0) <= ctb.ORDER_TOL,
    }


def hex_study(steel) -> list[dict]:
    """3-D hex8 path over the restraint family, h-refinement in every direction."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    scale = E * alpha * DT
    rows = []
    for n in HEX_LEVELS:
        mesh = ctb.hex_mesh(n, (SIDE, SIDE, SIDE))
        for restraint in ("uniaxial", "biaxial", "triaxial", "free"):
            res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint=restraint)
            s = res.sigma.mean(dim=0)
            sxx, syy, szz = float(s[0]), float(s[1]), float(s[2])
            shear = float(s[3:].abs().max())
            if restraint == "free":
                err, ref = abs(sxx) / scale, 0.0
                gate = err < ctb.ROUNDOFF_REL_TOL
                lateral = [abs(syy) / scale, abs(szz) / scale]
            else:
                err = ctb.rel_error(sxx, res.exact_sigma)
                ref = res.exact_sigma
                gate = abs(err) < min(ctb.TASK_REL_TOL, ctb.ROUNDOFF_REL_TOL)
                lateral = [ctb.rel_error(syy, res.exact_lateral[0])
                           if res.exact_lateral[0] != 0.0 else abs(syy) / scale,
                           ctb.rel_error(szz, res.exact_lateral[1])
                           if res.exact_lateral[1] != 0.0 else abs(szz) / scale]
            rows.append({
                "n_elem_per_side": n,
                "n_nodes": mesh.n_nodes,
                "n_elements": mesh.n_elements,
                "h_m": SIDE / n,
                "restraint": restraint,
                "sigma_xx_Pa": sxx,
                "sigma_yy_Pa": syy,
                "sigma_zz_Pa": szz,
                "max_shear_Pa": shear,
                "reference_Pa": ref,
                "rel_error_xx": err,
                "lateral_measure": lateral,
                "max_abs_displacement_m": float(res.displacement.abs().max()),
                "pass_task_tol": abs(err) < ctb.TASK_REL_TOL,
                "pass_roundoff": gate,
            })
    return rows


def nu_sensitivity(steel) -> dict:
    """``sigma_xx`` of the uniaxially restrained block must not depend on ``nu``."""
    from dataclasses import replace

    mesh = ctb.hex_mesh(2, (SIDE, SIDE, SIDE))
    rows = []
    for nu in NU_LEVELS:
        res = ctb.solve_hex(mesh, replace(steel, nu=nu), DT, T_ref=T_REF,
                            restraint="uniaxial")
        sxx = float(res.sigma.mean(dim=0)[0])
        rows.append({
            "nu": nu,
            "sigma_xx_Pa": sxx,
            "sigma_xx_MPa": sxx / 1e6,
            "reference_Pa": res.exact_sigma,
            "reference_MPa": res.exact_sigma / 1e6,
            "rel_error": ctb.rel_error(sxx, res.exact_sigma),
        })
    drift = max(abs(r["sigma_xx_Pa"] - rows[0]["sigma_xx_Pa"]) for r in rows)
    rel = drift / abs(rows[0]["sigma_xx_Pa"])
    return {"rows": rows, "max_drift_Pa": drift, "max_drift_rel": rel,
            "invariance_gate": rel < ctb.ROUNDOFF_REL_TOL}


def dt_sensitivity(steel) -> dict:
    """``sigma`` must be linear in ``dT`` with slope ``-E alpha``."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    rows = []
    for dT in DT_LEVELS:
        res = ctb.solve_bar(4, E, alpha, dT, T_ref=T_REF, length=LENGTH)
        sigma = float(res.sigma.mean())
        ref = ctb.restrained_uniaxial_stress(E, alpha, dT)
        rows.append({
            "dT_K": dT,
            "sigma_Pa": sigma,
            "sigma_over_dT_Pa_per_K": sigma / dT,
            "reference_Pa": ref,
            "rel_error": ctb.rel_error(sigma, ref),
        })
    slope = (rows[-1]["sigma_Pa"] - rows[0]["sigma_Pa"]) / (rows[-1]["dT_K"] - rows[0]["dT_K"])
    return {"rows": rows, "slope_Pa_per_K": slope,
            "analytic_slope_Pa_per_K": -E * alpha,
            "slope_rel_error": ctb.rel_error(slope, -E * alpha)}


def graded_study(steel) -> dict:
    """Spatially varying ``dT`` -- the rung on which the 3% gate actually bites.

    Physics: ``T(x) = T_ref + dT (x/L)**2`` on a fully restrained bar with constant
    ``E``, ``alpha``.  Compatibility fixes the axial force, so the continuum answer
    is *constant*:

        sigma_xx = -E alpha <dT>,      <dT> = dT/3.

    Both FE paths build the thermal load from one temperature per element, which
    for an equally spaced grid is the trapezoid mean of the nodal field:
    ``<dT>_trap = dT (1/3 + 1/(6 n**2))``.  The code therefore has to reproduce

        sigma_xx = -E alpha <dT>_trap  =  sigma_exact (1 + 1/(2 n**2))

    to round-off (a bug in the thermal load or in the restraint set would break
    this), and it approaches ``sigma_exact`` at second order (a temperature
    interpolation error, nothing else).  Relative bias ``1/(2 n**2)`` exceeds the
    3% acceptance gate for ``n <= 4`` and is below it for ``n >= 5``: these two
    statements are predeclared in :mod:`tensorfem.constrained_thermal_bar`
    (``GRADED_COARSE_MAX`` / ``GRADED_FINE_MIN``) and the observed pattern is
    reported against them, not fitted to them.

    The 3-D rung is the first configuration in this case in which the hex8
    thermo-elastic path is driven by a *non-uniform* temperature field (before
    this, every element saw the same ``dT``, so the element-temperature sampling
    was never exercised).
    """
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)

    bar_rows = []
    for n in GRADED_BAR_LEVELS:
        res = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=LENGTH,
                            temperature="quadratic")
        sigma = float(res.sigma.mean())
        refs = ctb.graded_field_stress(E, alpha, DT, n, LENGTH)
        bar_rows.append({
            "n_elem_axial": n,
            "h_m": LENGTH / n,
            "sigma_Pa": sigma,
            "model_Pa": refs["element_average_Pa"],
            "continuum_Pa": refs["continuum_Pa"],
            "model_rel_error": ctb.rel_error(sigma, refs["element_average_Pa"]),
            "continuum_rel_error": ctb.rel_error(sigma, refs["continuum_Pa"]),
            "bias_rel": refs["bias_rel"],
            "predicted_bias_rel": refs["predicted_bias_rel"],
            "pass_task_tol": abs(ctb.rel_error(sigma, refs["continuum_Pa"]))
                             < ctb.TASK_REL_TOL,
        })

    hex_rows = []
    for nx in GRADED_HEX_LEVELS:
        mesh = ctb.graded_mesh(nx, LENGTH, GRADED_SIDE)
        res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint="uniaxial",
                            temp_profile="graded")
        sxx = float(res.sigma.mean(dim=0)[0])
        refs = ctb.graded_field_stress(E, alpha, DT, nx, LENGTH)
        hex_rows.append({
            "n_elem_axial": nx,
            "n_elem": mesh.n_elements,
            "n_dof": 3 * mesh.n_nodes,
            "h_m": LENGTH / nx,
            "sigma_Pa": sxx,
            # independent analytic model (derived from the nodal geometry, not from
            # the solver's own element temperatures) + the solver's own copy of it
            "model_Pa": refs["element_average_Pa"],
            "code_model_Pa": res.model_sigma,
            "continuum_Pa": refs["continuum_Pa"],
            "model_rel_error": ctb.rel_error(sxx, refs["element_average_Pa"]),
            "code_model_rel_error": ctb.rel_error(sxx, res.model_sigma),
            "continuum_rel_error": ctb.rel_error(sxx, refs["continuum_Pa"]),
            "element_spread_rel": float(res.sigma[:, 0].max() - res.sigma[:, 0].min())
                                  / abs(sxx),
            "lateral_max_over_axial": float(res.sigma[:, 1:].abs().max()) / abs(sxx),
            "bias_rel": refs["bias_rel"],
            "predicted_bias_rel": refs["predicted_bias_rel"],
            "pass_task_tol": abs(ctb.rel_error(sxx, res.exact_sigma)) < ctb.TASK_REL_TOL,
        })

    def _orders(rows):
        return ctb.observed_order([abs(r["continuum_rel_error"]) for r in rows],
                                  [r["h_m"] for r in rows])

    def _bite(rows):
        return ([r["n_elem_axial"] for r in rows if not r["pass_task_tol"]],
                [r["n_elem_axial"] for r in rows if r["pass_task_tol"]])

    bar_fail, bar_pass = _bite(bar_rows)
    hex_fail, hex_pass = _bite(hex_rows)
    bias_id = max(abs(r["bias_rel"] - r["predicted_bias_rel"])
                  for r in bar_rows + hex_rows)
    model_roundoff = max(abs(r["model_rel_error"]) for r in bar_rows + hex_rows)
    o1, o3 = _orders(bar_rows), _orders(hex_rows)
    coarse_ok = all(n <= ctb.GRADED_COARSE_MAX for n in bar_fail + hex_fail)
    fine_ok = all(n >= ctb.GRADED_FINE_MIN for n in bar_pass + hex_pass)
    return {
        "bar_1d": bar_rows,
        "hex_3d": hex_rows,
        "bias_identity_max_rel": bias_id,
        "model_roundoff_max": model_roundoff,
        "observed_order_1d": o1,
        "observed_order_3d": o3,
        "task_gate_fail_levels_1d": bar_fail,
        "task_gate_pass_levels_1d": bar_pass,
        "task_gate_fail_levels_3d": hex_fail,
        "task_gate_pass_levels_3d": hex_pass,
        "predeclared_coarse_max": ctb.GRADED_COARSE_MAX,
        "predeclared_fine_min": ctb.GRADED_FINE_MIN,
        "model_gate": model_roundoff < ctb.ROUNDOFF_REL_TOL and bias_id < 1e-12,
        "order_gate": (abs(o1 - 2.0) < ctb.ORDER_TOL
                       and abs(o3 - 2.0) < ctb.ORDER_TOL),
        "discrimination_gate": coarse_ok and fine_ok,
        "note": ("the 3% gate is informative only here: on the uniform rung the "
                 "discrete answer is exact, so no tolerance below round-off can bite"),
    }


def gate_strength_probe(steel) -> dict:
    """Do the gates actually catch a defect, and which gate catches which?

    The round-1 audit of this case observed that the governing rung cannot exercise
    any tolerance (the discrete answer is exact).  This probe turns that remark into
    a measurement: two *predeclared* physics corruptions are injected into an
    in-memory copy of the module (never into the working tree) and the same
    quantities are recomputed:

    ``A`` drop the thermal strain from the stress recovery (``sigma = D eps``);
    ``B`` sample the element temperature at one node instead of averaging it.

    ``A`` must trip the uniform-rung gates.  ``B`` is *invisible* to every uniform
    gate -- the uniform field makes any sampling equal -- and is caught only by the
    graded rung added in this revision.  That is the operational content of "the
    graded rung is the one where the tolerance bites": it catches a defect the
    pre-existing gates pass.
    """
    import importlib.util
    import sys
    import tempfile

    src_text = Path(ctb.__file__).read_text()
    mutants = {
        "A_drop_thermal_strain_in_stress_recovery": (
            'sigma = torch.einsum("ekl,egl->egk", D, eps - eps_th).mean(dim=1)',
            'sigma = torch.einsum("ekl,egl->egk", D, eps).mean(dim=1)'),
        "B_sample_element_temperature_at_one_node": (
            "T_elem = T_node[mesh.elements].mean(dim=1)",
            "T_elem = T_node[mesh.elements][:, 0]"),
    }
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    ref = ctb.restrained_uniaxial_stress(E, alpha, DT)
    graded_model = ctb.graded_field_stress(E, alpha, DT, 4, LENGTH)["element_average_Pa"]

    def _measure(module):
        uni = module.solve_hex(module.hex_mesh(2, (SIDE, SIDE, SIDE)), steel, DT,
                              T_ref=T_REF, restraint="uniaxial")
        free = module.solve_hex(module.hex_mesh(2, (SIDE, SIDE, SIDE)), steel, DT,
                                T_ref=T_REF, restraint="free")
        gr = module.solve_hex(module.graded_mesh(4, LENGTH, GRADED_SIDE), steel, DT,
                              T_ref=T_REF, restraint="uniaxial", temp_profile="graded")
        return {
            "uniform_rel_error": abs(ctb.rel_error(float(uni.sigma.mean(dim=0)[0]), ref)),
            "free_control_rel": float(free.sigma.abs().max()) / abs(ref),
            "graded_rel_error_vs_model": abs(ctb.rel_error(
                float(gr.sigma.mean(dim=0)[0]), graded_model)),
        }

    with tempfile.TemporaryDirectory() as tmp:
        def _load(name: str, text: str):
            path = Path(tmp) / f"{name}.py"
            path.write_text(text)
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            # ``@dataclass`` resolves annotations through ``sys.modules``, so the
            # mutant has to be registered before it is executed.
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.modules.pop(name, None)
            return mod

        baseline = _measure(ctb)
        rows = []
        for name, (old, new) in mutants.items():
            if old not in src_text:  # pragma: no cover - mutation target vanished
                raise RuntimeError(f"mutation target not found for {name}")
            mod = _load(name, src_text.replace(old, new, 1))
            got = _measure(mod)
            caught = [k for k, v in got.items() if v > ctb.ROUNDOFF_REL_TOL]
            rows.append({
                "mutation": name,
                "metrics": got,
                "caught_by": caught,
                "caught": bool(caught),
            })

    by_name = {r["mutation"]: r for r in rows}
    a = by_name["A_drop_thermal_strain_in_stress_recovery"]
    b = by_name["B_sample_element_temperature_at_one_node"]
    uniform_gates = ("uniform_rel_error", "free_control_rel")
    a_caught = any(k in a["caught_by"] for k in uniform_gates)
    b_invisible_on_uniform = all(b["metrics"][k] < ctb.ROUNDOFF_REL_TOL
                                 for k in uniform_gates)
    b_caught_on_graded = "graded_rel_error_vs_model" in b["caught_by"]
    return {
        "baseline": baseline,
        "mutants": rows,
        "predeclared_expectation": {
            "A": "caught by a uniform-rung gate",
            "B": "invisible to every uniform-rung gate, caught by the graded rung",
        },
        "A_caught_on_uniform": a_caught,
        "B_invisible_on_uniform": b_invisible_on_uniform,
        "B_caught_on_graded": b_caught_on_graded,
        "gate": a_caught and b_invisible_on_uniform and b_caught_on_graded,
    }


def temperature_dependent_case(tdep_steel) -> dict:
    """EH36 with its real ``dE/dT`` / ``dalpha/dT``: reference at the element temperature."""
    res = ctb.solve_hex(ctb.hex_mesh(2, (SIDE, SIDE, SIDE)), tdep_steel, DT,
                        T_ref=T_REF, restraint="uniaxial")
    sxx = float(res.sigma.mean(dim=0)[0])
    T_hot = T_REF + DT
    frozen = ctb.restrained_uniaxial_stress(tdep_steel.E(T_REF),
                                            tdep_steel.alpha(T_REF), DT)
    return {
        "steel": tdep_steel.name,
        "T_hot_C": T_hot,
        "E_at_T_Pa": float(tdep_steel.E(T_hot)),
        "alpha_at_T_per_K": float(tdep_steel.alpha(T_hot)),
        "sigma_xx_Pa": sxx,
        "reference_Pa": res.exact_sigma,
        "rel_error": ctb.rel_error(sxx, res.exact_sigma),
        "reference_convention": "-E(T) alpha(T) dT with E, alpha at the element temperature",
        "constant_property_value_Pa": frozen,
        "frozen_property_rel_error": ctb.rel_error(frozen, res.exact_sigma),
        "pass_task_tol": abs(ctb.rel_error(sxx, res.exact_sigma)) < ctb.TASK_REL_TOL,
    }
# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_field(steel, path: Path) -> None:
    """sigma_xx cloud on the restrained block, with the analytic value overlaid."""
    mesh = ctb.hex_mesh(4, (SIDE, SIDE, SIDE))
    res = ctb.solve_hex(mesh, steel, DT, T_ref=T_REF, restraint="uniaxial")
    X = mesh.nodes[mesh.elements].mean(dim=1)
    sxx = res.sigma[:, 0].numpy() / 1e6
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2),
                           gridspec_kw={"width_ratios": [1.25, 1.0]})
    sc = ax[0].scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=sxx, s=42,
                       marker="s", cmap="coolwarm")
    fig.colorbar(sc, ax=ax[0], label=r"$\sigma_{xx}$ [MPa]")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]  (all z layers projected)")
    ax[0].set_title(f"FEM $\\sigma_{{xx}}$, uniaxial restraint, dT={DT:g} K")
    uniq = np.unique(np.round(sxx, 9))
    ax[1].axhline(res.exact_sigma / 1e6, color="k", ls="--",
                  label=r"analytic $-E\alpha\Delta T$")
    ax[1].plot(np.arange(len(uniq)), uniq, "o", color="C3",
               label="distinct FEM element values")
    ax[1].set_xlabel("distinct element value index")
    ax[1].set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax[1].set_title(f"spread over {mesh.n_elements} elements: "
                    f"{float(uniq.max() - uniq.min()):.2e} MPa")
    ax[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_profile(steel, path: Path) -> None:
    """Axial stress along the bar: uniform and quadratic dT, FEM vs closed form."""
    E, alpha = steel.E(T_REF), steel.alpha(T_REF)
    n = 8
    uni = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=LENGTH)
    quad = ctb.solve_bar(n, E, alpha, DT, T_ref=T_REF, length=LENGTH,
                         temperature="quadratic")
    x = np.linspace(0.0, LENGTH, n + 1)
    xc = 0.5 * (x[:-1] + x[1:])
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.0))
    for res, lab in ((uni, "uniform $\\Delta T$"), (quad, "quadratic $T(x)$")):
        ax[0].step(xc, res.sigma.numpy() / 1e6, where="mid", label=f"FEM, {lab}")
        ax[0].axhline(res.exact_sigma / 1e6, ls="--", lw=1.0,
                      label=f"analytic, {lab}: {res.exact_sigma / 1e6:.4f} MPa")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel(r"element $\sigma_{xx}$ [MPa]")
    ax[0].set_title(f"fully restrained bar, dT={DT:g} K, {n} elements")
    ax[0].legend(fontsize=7)
    xf = np.linspace(0.0, LENGTH, 400)
    ax[1].plot(xf, alpha * DT * (xf ** 3 / (3.0 * LENGTH ** 2) - xf / 3.0) * 1e3,
               "k-", lw=1.2, label="exact $u(x)$, quadratic $T$")
    ax[1].plot(x, quad.displacement.numpy() * 1e3, "o", ms=4, label="FEM nodal $u$")
    ax[1].plot(x, uni.displacement.numpy() * 1e3, "s", ms=4,
               label="uniform $\\Delta T$: $u\\equiv 0$")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("axial displacement [mm]")
    ax[1].set_title("displacement field (end restraints)")
    ax[1].legend(fontsize=8)
    # panel 3: why the graded rung is the one that can bite the 3% gate
    ns = np.array(sorted(GRADED_BAR_LEVELS))
    ax[2].loglog(ns, 1.0 / (2.0 * ns ** 2), "o-", label=r"trapezoid bias $1/(2n^2)$")
    ax[2].axhline(ctb.TASK_REL_TOL, color="C3", ls=":", lw=1.2,
                  label="task gate 3%")
    bite = ns[1.0 / (2.0 * ns ** 2) > ctb.TASK_REL_TOL]
    ax[2].plot(bite, 1.0 / (2.0 * bite ** 2), "rx", ms=9,
               label="gate bites (n = " + ", ".join(str(int(v)) for v in bite) + ")")
    ax[2].set_xlabel("axial elements n")
    ax[2].set_ylabel("relative bias of $\\sigma_{xx}$")
    ax[2].set_title("quadratic $T(x)$: element-average bias vs 3% gate")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_convergence(quad: dict, bar_rows, graded: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hs = [r["h_m"] for r in quad["rows"]]
    es = [r["abs_rel_error"] for r in quad["rows"]]
    ax.loglog(hs, es, "o-", label=f"quadratic $T(x)$ (order $\\approx$ {quad['observed_order']:.2f})")
    ax.loglog([r["h_m"] for r in bar_rows],
              [max(abs(r["rel_error"]), 1e-18) for r in bar_rows], "s--",
              label="uniform $\\Delta T$ (round-off level)")
    g1 = graded["bar_1d"]
    g3 = graded["hex_3d"]
    ax.loglog([r["h_m"] for r in g1],
              [abs(r["continuum_rel_error"]) for r in g1], "d-", color="C2",
              label=f"graded $T(x)$, 1-D (order $\\approx$ {graded['observed_order_1d']:.2f})")
    ax.loglog([r["h_m"] for r in g3],
              [abs(r["continuum_rel_error"]) for r in g3], "^-.", color="C4",
              label=f"graded $T(x)$, 3-D hex8 (order $\\approx$ {graded['observed_order_3d']:.2f})")
    ax.loglog([r["h_m"] for r in g1],
              [max(abs(r["model_rel_error"]), 1e-18) for r in g1], ":", color="0.4",
              label="graded vs element-average model (round-off)")
    ax.axhline(ctb.ROUNDOFF_REL_TOL, color="0.4", lw=1.0, label="round-off gate 1e-8")
    ax.axhline(ctb.TASK_REL_TOL, color="C3", ls=":", lw=1.2, label="task gate 3%")
    ax.loglog(hs, es[0] * (np.array(hs) / hs[0]) ** 2.0, "k--", lw=0.9, alpha=0.6,
              label="$O(h^2)$ reference slope")
    ax.set_xlabel("h = L / n_elem [m]")
    ax.set_ylabel("relative error of the axial stress")
    ax.set_title("h-convergence of the restrained-bar stress")
    ax.annotate("3% gate bites for n <= "
                f"{int(ctb.GRADED_COARSE_MAX)}",
                xy=(LENGTH / ctb.GRADED_COARSE_MAX, ctb.TASK_REL_TOL),
                xytext=(0.03, 0.06), textcoords="axes fraction",
                fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_sensitivity(hex_rows, nu, dts, tdep, path: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    labels = ["uniaxial", "biaxial", "triaxial", "free"]
    n = 2
    sel = [r for r in hex_rows if r["n_elem_per_side"] == n]
    x = np.arange(len(labels))
    fem = [r["sigma_xx_Pa"] / 1e6 for r in sel]
    ref = [r["reference_Pa"] / 1e6 for r in sel]
    ax[0].bar(x - 0.18, fem, 0.36, label="FEM")
    ax[0].bar(x + 0.18, ref, 0.36, label="analytic")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax[0].set_title(f"restraint family (3-D hex8, {n}$^3$ elements)")
    ax[0].legend(fontsize=8)

    ax[1].plot([r["nu"] for r in nu["rows"]], [r["sigma_xx_MPa"] for r in nu["rows"]],
               "o-", label=r"FEM $\sigma_{xx}$")
    ax[1].axhline(nu["rows"][0]["reference_MPa"], color="k", ls="--",
                  label=r"analytic $-E\alpha\Delta T$")
    ax[1].set_xlabel(r"Poisson ratio $\nu$")
    ax[1].set_ylabel(r"$\sigma_{xx}$ [MPa]")
    ax[1].set_title(f"uniaxial: $\\nu$-invariance, drift {nu['max_drift_rel']:.1e}")
    ax[1].legend(fontsize=8)

    ax[2].plot([r["dT_K"] for r in dts["rows"]], [r["sigma_Pa"] / 1e6 for r in dts["rows"]],
               "o-", label="FEM")
    ax[2].plot([r["dT_K"] for r in dts["rows"]],
               [r["reference_Pa"] / 1e6 for r in dts["rows"]], "k--", label="analytic")
    ax[2].set_xlabel(r"$\Delta T$ [K]")
    ax[2].set_ylabel(r"$\sigma$ [MPa]")
    ax[2].set_title(f"$\\Delta T$ linearity: slope err {dts['slope_rel_error']:.1e}\n"
                    f"EH36 T-dependent case: rel. err {tdep['rel_error']:.1e}")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# evidence + report
# ---------------------------------------------------------------------------
def build_evidence() -> dict:
    steel = ctb.constant_steel("EH36", T_ref=T_REF)
    tdep = ctb.default_steel("EH36", T_ref=T_REF)
    E, alpha, nu = steel.E(T_REF), steel.alpha(T_REF), steel.nu

    bar_rows = bar_study(steel)
    quad = bar_quadratic_study(steel)
    hex_rows = hex_study(steel)
    nu_rows = nu_sensitivity(steel)
    dt_rows = dt_sensitivity(steel)
    tdep_case = temperature_dependent_case(tdep)
    graded = graded_study(steel)
    probe = gate_strength_probe(steel)

    ref = ctb.restrained_uniaxial_stress(E, alpha, DT)
    derivation = {r: ctb.derivation_residual(E, alpha, DT, nu, r)
                  for r in ("uniaxial", "biaxial", "triaxial")}
    restrained = [r for r in hex_rows if r["restraint"] != "free"]
    free_rows = [r for r in hex_rows if r["restraint"] == "free"]

    gates = {
        "task_rel_tol": ctb.TASK_REL_TOL,
        "roundoff_rel_tol": ctb.ROUNDOFF_REL_TOL,
        "order_tol": ctb.ORDER_TOL,
        "bar_uniform_max_abs_rel_error": max(abs(r["rel_error"]) for r in bar_rows),
        "bar_quadratic_observed_order": quad["observed_order"],
        "hex_max_abs_rel_error_xx": max(abs(r["rel_error_xx"]) for r in restrained),
        "hex_free_control_measure_max": max(max(r["lateral_measure"]) for r in free_rows),
        "hex_lateral_measure_max": max(max(r["lateral_measure"]) for r in hex_rows),
        "nu_max_drift_rel": nu_rows["max_drift_rel"],
        "dT_slope_rel_error": abs(dt_rows["slope_rel_error"]),
        "graded_model_roundoff_max": graded["model_roundoff_max"],
        "graded_bias_identity_max_rel": graded["bias_identity_max_rel"],
        "graded_observed_order_1d": graded["observed_order_1d"],
        "graded_observed_order_3d": graded["observed_order_3d"],
        "graded_task_gate_fail_levels_1d": graded["task_gate_fail_levels_1d"],
        "graded_task_gate_pass_levels_1d": graded["task_gate_pass_levels_1d"],
        "graded_task_gate_fail_levels_3d": graded["task_gate_fail_levels_3d"],
        "graded_task_gate_pass_levels_3d": graded["task_gate_pass_levels_3d"],
        "graded_predeclared_coarse_max": graded["predeclared_coarse_max"],
        "graded_predeclared_fine_min": graded["predeclared_fine_min"],
        "gate_strength_A_caught_on_uniform": probe["A_caught_on_uniform"],
        "gate_strength_B_invisible_on_uniform": probe["B_invisible_on_uniform"],
        "gate_strength_B_caught_on_graded": probe["B_caught_on_graded"],
    }
    verdict = {
        "task_gate_3pct": (gates["bar_uniform_max_abs_rel_error"] < ctb.TASK_REL_TOL
                           and gates["hex_max_abs_rel_error_xx"] < ctb.TASK_REL_TOL
                           and tdep_case["pass_task_tol"]),
        "roundoff_gate": (gates["bar_uniform_max_abs_rel_error"] < ctb.ROUNDOFF_REL_TOL
                          and gates["hex_max_abs_rel_error_xx"] < ctb.ROUNDOFF_REL_TOL
                          and gates["hex_free_control_measure_max"] < ctb.ROUNDOFF_REL_TOL
                          and nu_rows["invariance_gate"]),
        "convergence_order_gate": quad["order_gate"],
        "derivation_gate": max(abs(v) for v in derivation.values()) < 1e-15,
        "graded_model_gate": graded["model_gate"],
        "graded_order_gate": graded["order_gate"],
        "graded_discrimination_gate": graded["discrimination_gate"],
        "gate_strength_gate": probe["gate"],
    }
    verdict["all_pass"] = all(verdict.values())

    return {
        "case": "CASE-THERMAL-BAR",
        "repo": "TensorFEM",
        "title": "Constrained (fully restrained) thermal stress: sigma = E alpha dT",
        "parameters": {
            "dT_K": DT,
            "T_ref_C": T_REF,
            "bar_length_m": LENGTH,
            "bar_area_m2": 1.0,
            "hex_side_m": SIDE,
            "hex_levels": list(HEX_LEVELS),
            "bar_levels": list(BAR_LEVELS),
            "nu_levels": list(NU_LEVELS),
            "dT_levels_K": list(DT_LEVELS),
            "E_Pa": E,
            "E_GPa": E / 1e9,
            "alpha_per_K": alpha,
            "nu": nu,
            "dimensionless_groups": {
                "alpha_dT": alpha * DT,
                "dT_over_T_ref": DT / T_REF,
                "nu": nu,
                "sigma_over_E": abs(ref) / E,
                "slenderness_L_over_side": LENGTH / SIDE,
                "note": ("linear elastic, uniform dT -> no length scale enters, so the "
                         "dimensionless answer is sigma/(E alpha dT) = -1 (uniaxial), "
                         "-1/(1-nu) (biaxial), -1/(1-2nu) (triaxial), 0 (free)"),
            },
        },
        "reference": {
            "expression": "sigma_xx = -E alpha dT   (axial restraint, lateral free)",
            "value_Pa": ref,
            "value_MPa": ref / 1e6,
            "source": ("closed-form linear thermoelasticity (Hooke's law with "
                       "eps_th = alpha dT I); the classical result of Timoshenko & "
                       "Goodier, Theory of Elasticity (thermal-stress article) and "
                       "Boley & Weiner, Theory of Thermal Stresses, Ch. 1. No page "
                       "number is claimed: this sandbox has no library access, so the "
                       "formula is re-derived from the constitutive law in "
                       "`derivation_residual` (residual reported below)"),
            "derivation_residual": derivation,
            "material": ctb.material_digest(steel),
        },
        "environment": environment(),
        "studies": {
            "bar_uniform": bar_rows,
            "bar_quadratic_convergence": quad,
            "hex_restraint_family": hex_rows,
            "nu_sensitivity": nu_rows,
            "dt_sensitivity": dt_rows,
            "temperature_dependent_EH36": tdep_case,
            "graded_temperature": graded,
            "gate_strength": probe,
        },
        "gates": gates,
        "verdict": verdict,
    }


def render_report(ev: dict) -> str:
    p, r, st, g, v = (ev["parameters"], ev["reference"], ev["studies"],
                      ev["gates"], ev["verdict"])
    dg = r["material"]
    L: list[str] = []
    a = L.append
    a("# CASE-THERMAL-BAR — constrained thermal stress (TensorFEM)")
    a("")
    a("> Machine-generated by `benchmarks/constrained_thermal_bar.py`; every number "
      "below is formatted from `benchmarks/constrained_thermal_bar_evidence.json`. "
      "None is typed by hand.")
    a("")
    a("## 1. Problem and reference")
    a("")
    a("A bar / block is **prevented from expanding** when heated. The elastic answer "
      "is a closed form, which makes this the anchor every industrial thermal-stress "
      "model is calibrated on:")
    a("")
    a("```")
    a("sigma_xx = -E alpha dT   (axial restraint, lateral faces free)")
    a(f"         = -{p['E_Pa']:.6e} Pa * {p['alpha_per_K']:.6e} /K * {p['dT_K']:g} K")
    a(f"         = {r['value_Pa']:.6e} Pa = {r['value_MPa']:.4f} MPa")
    a("```")
    a("")
    a("`dT > 0` (heating against restraint) gives compression, hence the minus sign; "
      "`dT < 0` gives tension.")
    a("")
    a("**Reference provenance (zero hand-copy).** The target value is *computed*, not "
      "transcribed: the closed form evaluated with "
      f"`tensorfem.polar_thermal_stress.steel_from_anchors('{dg['name']}')` frozen at "
      f"`T_ref = {dg['T_ref_C']:g} degC` (E = {dg['E_ref_Pa']/1e9:.4f} GPa, "
      f"alpha = {dg['alpha_ref_per_K']:.6e} /K, nu = {dg['nu']:g}). The anchor table is "
      f"hashed into the evidence (`anchor_table_sha256` = `{dg['anchor_table_sha256']}`) "
      "and its own caveat string travels with it:")
    a("")
    a(f"> {dg['anchor_source']}")
    a("")
    a("The formula is re-derived inside the run from Hooke's law with a thermal strain, "
      "so no citation has to be trusted:")
    a("")
    a("| restraint | Hooke residual `(sigma - nu*sigma_lat)/E + alpha dT` |")
    a("|---|---|")
    for k, val in r["derivation_residual"].items():
        a(f"| {k} | {val:.3e} |")
    a("")
    a("`derivation_gate` (< 1e-15): " + ("**PASS**" if v["derivation_gate"] else "**FAIL**"))
    a("")
    a("No Timoshenko & Goodier / Boley & Weiner page number is quoted: there is no "
      "library access in this sandbox and the case forbids fabricated citations, so the "
      "reference is verified by re-derivation plus a hashed material table instead.")
    a("")
    a("## 2. Method and implementation")
    a("")
    a("* **1-D path (reuse, no new solver):** `tensorfem.thermal.solve_thermoelastic_bar` "
      "— sequential thermoelasticity on the two-node bar element "
      "(`f_th = E A alpha dT [-1, +1]`), both ends fixed.")
    a("* **3-D path (reuse):** the hex8 thermo-elastic assembly of "
      "`tensorfem.polar_thermal_stress` (`assemble_elasticity` + "
      "`assemble_thermal_load` + `solve_dirichlet`, dense reference assembly, "
      "`float64`), driven by a new *restraint-set* helper "
      "(`constrained_thermal_bar.restraint_dofs`) exposing the uniaxial / biaxial / "
      "triaxial / free families on one mesh. The repo's existing thermal gates covered "
      "only the two extremes (clamped biaxial and fully free); the uniaxial "
      "configuration — the one this case anchors — was untested.")
    a("* **Stress recovery:** `sigma = D (B u - eps_th)` at the 8 Gauss points, "
      "averaged per element (the `solve_panel_stress` convention).")
    a("* **Controls / discriminators:** the *free* case must give `sigma ~ 0` (it "
      "separates a real thermoelastic solve from one that silently drops the thermal "
      "load); the *nu sweep* must leave `sigma_xx` unchanged (uniaxial restraint has no "
      "Poisson coupling in the restrained direction); the *restraint family* must "
      "reproduce `-E alpha dT`, `-E alpha dT/(1-nu)`, `-E alpha dT/(1-2nu)` and `0`.")
    a("* **Graded rung (new in this revision):** the same two paths driven by a "
      "*spatially varying* temperature field `T(x) = T_ref + dT (x/L)^2` — "
      "`solve_bar(temperature='quadratic')` in 1-D and "
      "`solve_hex(temp_profile='graded')` in 3-D. Until now every element in the hex8 "
      "path saw the same `dT`, so the element-temperature sampling was never "
      "exercised; on this rung it is, and it is the only rung on which the 3% "
      "acceptance gate can fail (see §4.5).")
    a("* **Determinism:** direct dense solves, no sampling, no seed; reruns are "
      "bit-identical.")
    a("")
    a("## 3. Parameters (dimensionless groups included)")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a(f"| dT | {p['dT_K']:g} K |")
    a(f"| T_ref | {p['T_ref_C']:g} degC |")
    a(f"| E (frozen at T_ref) | {p['E_Pa']:.6e} Pa ({p['E_GPa']:.4f} GPa) |")
    a(f"| alpha (frozen at T_ref) | {p['alpha_per_K']:.6e} /K |")
    a(f"| nu | {p['nu']:g} |")
    a(f"| bar length / area | {p['bar_length_m']:g} m / {p['bar_area_m2']:g} m^2 |")
    a(f"| hex specimen side | {p['hex_side_m']:g} m |")
    a(f"| 1-D levels n_elem | {p['bar_levels']} |")
    a(f"| 3-D levels n^3 | {p['hex_levels']} |")
    dim = p["dimensionless_groups"]
    a(f"| alpha dT (free thermal strain) | {dim['alpha_dT']:.6e} |")
    a(f"| dT / T_ref | {dim['dT_over_T_ref']:.4f} |")
    a(f"| sigma / E | {dim['sigma_over_E']:.6e} |")
    a(f"| L / side | {dim['slenderness_L_over_side']:g} |")
    a("")
    a(f"{dim['note']}.")
    a("")
    a("## 4. Results: reference vs computed")
    a("")
    a("### 4.1 1-D bar, uniform dT (the anchor case)")
    a("")
    a("| n_elem | h [m] | sigma FEM [MPa] | reference [MPa] | signed rel. error | <3% | round-off <1e-8 |")
    a("|---|---|---|---|---|---|---|")
    for row in st["bar_uniform"]:
        a(f"| {row['n_elem']} | {row['h_m']:.4g} | {row['sigma_MPa']:.6f} | "
          f"{row['reference_MPa']:.6f} | {row['rel_error']:.2e} | "
          f"{'✓' if row['pass_task_tol'] else '✗'} | "
          f"{'✓' if row['pass_roundoff'] else '✗'} |")
    a("")
    b_last = st["bar_uniform"][-1]
    a(f"The FEM stress is uniform over the elements (max-min spread on the "
      f"{b_last['n_elem']}-element mesh = {b_last['sigma_spread_over_elements']:.2e} Pa) "
      f"and the nodal displacement is identically zero "
      f"(`max|u| = {b_last['max_abs_displacement']:.2e} m`), as the two end restraints "
      "require; the reaction sum is "
      f"`{b_last['reaction_sum_N']:.3e} N` (self-equilibrated pair).")
    a("")
    a("### 4.2 3-D hex8 block, restraint family (n^3 elements)")
    a("")
    a("| restraint | n | nodes | sigma_xx FEM [MPa] | sigma_xx ref [MPa] | rel. err. | "
      "sigma_yy [MPa] | sigma_zz [MPa] | lateral measure | max shear [Pa] | <3% |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in st["hex_restraint_family"]:
        lat = row["lateral_measure"]
        a(f"| {row['restraint']} | {row['n_elem_per_side']}^3 | {row['n_nodes']} | "
          f"{row['sigma_xx_Pa']/1e6:.6f} | {row['reference_Pa']/1e6:.6f} | "
          f"{row['rel_error_xx']:.2e} | {row['sigma_yy_Pa']/1e6:.3e} | "
          f"{row['sigma_zz_Pa']/1e6:.3e} | ({lat[0]:.1e}, {lat[1]:.1e}) | "
          f"{row['max_shear_Pa']:.2e} | {'✓' if row['pass_task_tol'] else '✗'} |")
    a("")
    a("`lateral measure` = relative error against the analytic lateral stress where that "
      "reference is non-zero, and `|sigma_lat|/(E alpha dT)` where it is zero (the "
      "free-expansion control). The `free` rows are the control: `sigma ~ 0` to "
      "round-off proves the thermal load is present and that the restraint set, not the "
      "load, produces the stress.")
    a("")
    a("### 4.3 h-convergence of the stress (genuine order study)")
    a("")
    a("Uniform dT is nodally exact, so it only measures round-off. A quadratic "
      "temperature field `T(x) = T_ref + dT (x/L)^2` is *not* representable by linear "
      "elements, so its stress error is a real discretisation error. Exact solution: "
      "constant axial force, "
      f"`sigma = -E alpha dT/3 = {st['bar_quadratic_convergence']['reference_Pa']/1e6:.6f} MPa`.")
    a("")
    a("| n_elem | h [m] | sigma FEM [Pa] | reference [Pa] | abs rel. error | u RMS error [m] |")
    a("|---|---|---|---|---|---|")
    for row in st["bar_quadratic_convergence"]["rows"]:
        a(f"| {row['n_elem']} | {row['h_m']:.4g} | {row['sigma_Pa']:.6e} | "
          f"{row['reference_Pa']:.6e} | {row['abs_rel_error']:.3e} | "
          f"{row['u_rms_error_m']:.3e} |")
    a("")
    a(f"Observed order = **{st['bar_quadratic_convergence']['observed_order']:.3f}** "
      f"(theory {st['bar_quadratic_convergence']['theory_order']:.1f}, gate "
      f"±{g['order_tol']}) -> "
      f"{'✓' if st['bar_quadratic_convergence']['order_gate'] else '✗'}")
    a("")
    a("### 4.4 Parameter sensitivity")
    a("")
    a("**Poisson ratio** (a uniaxially restrained block must be nu-invariant):")
    a("")
    a("| nu | sigma_xx FEM [MPa] | reference [MPa] | rel. error |")
    a("|---|---|---|---|")
    for row in st["nu_sensitivity"]["rows"]:
        a(f"| {row['nu']:g} | {row['sigma_xx_MPa']:.6f} | {row['reference_MPa']:.6f} | "
          f"{row['rel_error']:.2e} |")
    a("")
    a(f"Max drift over the sweep = {st['nu_sensitivity']['max_drift_rel']:.2e} -> "
      f"{'✓ invariant' if st['nu_sensitivity']['invariance_gate'] else '✗ not invariant'}")
    a("")
    a("**dT level** (stress must be linear in dT with slope `-E alpha`):")
    a("")
    a("| dT [K] | sigma FEM [MPa] | reference [MPa] | rel. error | sigma/dT [Pa/K] |")
    a("|---|---|---|---|---|")
    for row in st["dt_sensitivity"]["rows"]:
        a(f"| {row['dT_K']:g} | {row['sigma_Pa']/1e6:.6f} | {row['reference_Pa']/1e6:.6f} | "
          f"{row['rel_error']:.2e} | {row['sigma_over_dT_Pa_per_K']:.6e} |")
    a("")
    a(f"Endpoint slope = {st['dt_sensitivity']['slope_Pa_per_K']:.6e} Pa/K vs analytic "
      f"{st['dt_sensitivity']['analytic_slope_Pa_per_K']:.6e} Pa/K "
      f"(rel. err. {st['dt_sensitivity']['slope_rel_error']:.2e}).")
    a("")
    tdep = st["temperature_dependent_EH36"]
    a("**Temperature-dependent material** (EH36 with its anchor slopes; reference "
      "evaluated with `E(T), alpha(T)` at the element temperature, which is the code's "
      "own convention):")
    a("")
    a(f"* E({tdep['T_hot_C']:g} degC) = {tdep['E_at_T_Pa']/1e9:.4f} GPa, "
      f"alpha = {tdep['alpha_at_T_per_K']:.6e} /K")
    a(f"* FEM sigma_xx = {tdep['sigma_xx_Pa']/1e6:.6f} MPa vs reference "
      f"{tdep['reference_Pa']/1e6:.6f} MPa -> rel. err. {tdep['rel_error']:.2e} "
      f"({'✓' if tdep['pass_task_tol'] else '✗'})")
    a(f"* for contrast, freezing the properties would give "
      f"{tdep['constant_property_value_Pa']/1e6:.6f} MPa, i.e. "
      f"{abs(tdep['frozen_property_rel_error'])*100:.1f}% away: that is a "
      "property-evaluation effect, not a discretisation error. It is why the governing "
      "case freezes the properties — the 3% gate must be about the FEM, not about the "
      "material model.")
    a("")
    gr = st["graded_temperature"]
    gs = st["gate_strength"]
    a("### 4.5 Graded dT: the rung on which the 3% gate actually bites")
    a("")
    a("Uniform `dT` is nodally exact, so §4.1/§4.2 can only demonstrate round-off: no "
      "acceptance tolerance, however tight, can *discriminate* there. This rung makes "
      "the tolerance informative. The field `T(x) = T_ref + dT (x/L)^2` is fully "
      "restrained at both ends, so compatibility fixes one constant axial force and the "
      "continuum answer is constant:")
    a("")
    a("```")
    a("sigma_xx = -E alpha <dT>,   <dT> = dT/3     (both ends fixed, constant E, alpha)")
    a("```")
    a("")
    a("Both FE paths build the thermal load from **one temperature per element** (the "
      "average of the nodal values, the convention of "
      "`thermal.solve_thermoelastic_bar`). For an equally spaced grid that average is "
      "the trapezoid mean of the nodal interpolant, `<dT>_trap = dT (1/3 + 1/(6 n^2))`, "
      "so the code must reproduce")
    a("")
    a("```")
    a(f"sigma_xx = -E alpha <dT>_trap = sigma_exact (1 + 1/(2 n^2))"
      f"      [both paths, to round-off]")
    a("```")
    a("")
    a("and approach `sigma_exact` at second order. The bias is *interpolation of the "
      "temperature field*, not a quadrature defect (trapezoid is exact for the linear "
      "interpolant); `1/(2 n^2)` is derived in "
      "`constrained_thermal_bar.graded_bias`.")
    a("")
    a("**1-D path** (`solve_thermoelastic_bar`, quadratic `T`):")
    a("")
    a("| n_elem | h [m] | sigma FEM [Pa] | element-average model [Pa] | continuum [Pa] | rel. err vs model | rel. err vs continuum | bias (analytic) | <3% vs continuum? |")
    a("|---|---|---|---|---|---|---|---|---|")
    for row in gr["bar_1d"]:
        a(f"| {row['n_elem_axial']} | {row['h_m']:.4g} | {row['sigma_Pa']:.6e} | "
          f"{row['model_Pa']:.6e} | {row['continuum_Pa']:.6e} | "
          f"{row['model_rel_error']:.3e} | {abs(row['continuum_rel_error']):.4%} | "
          f"{row['bias_rel']:+.3e} | {'YES' if row['pass_task_tol'] else '**NO**'} |")
    a("")
    a("**3-D path** (hex8, uniaxial restraint, graded nodal `T`, lateral 2x2):")
    a("")
    a("| nx | elements | dofs | sigma_xx FEM [Pa] | element-average model [Pa] | continuum [Pa] | rel. err vs model | rel. err vs continuum | element spread | max lateral / axial | <3% vs continuum? |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in gr["hex_3d"]:
        a(f"| {row['n_elem_axial']} | {row['n_elem']} | {row['n_dof']} | "
          f"{row['sigma_Pa']:.6e} | {row['model_Pa']:.6e} | "
          f"{row['continuum_Pa']:.6e} | {row['model_rel_error']:.3e} | "
          f"{abs(row['continuum_rel_error']):.4%} | {row['element_spread_rel']:.2e} | "
          f"{row['lateral_max_over_axial']:.2e} | "
          f"{'YES' if row['pass_task_tol'] else '**NO**'} |")
    a("")
    a(f"* element-average model (both paths): max rel. error "
      f"{g['graded_model_roundoff_max']:.2e} -> gate "
      f"{ctb.ROUNDOFF_REL_TOL:.0e}: "
      f"**{'PASS' if v['graded_model_gate'] else 'FAIL'}**")
    a(f"* bias identity against `1/(2 n^2)`: max |diff| "
      f"{g['graded_bias_identity_max_rel']:.2e}")
    a(f"* observed order: 1-D {g['graded_observed_order_1d']:.3f}, 3-D "
      f"{g['graded_observed_order_3d']:.3f} (theory 2, gate ±{g['order_tol']}) -> "
      f"**{'PASS' if v['graded_order_gate'] else 'FAIL'}**")
    a(f"* task gate (<3% vs the continuum) fails at n = "
      f"{g['graded_task_gate_fail_levels_1d']} (1-D) / "
      f"{g['graded_task_gate_fail_levels_3d']} (3-D) and passes at n = "
      f"{g['graded_task_gate_pass_levels_1d']} (1-D) / "
      f"{g['graded_task_gate_pass_levels_3d']} (3-D); predeclared window was "
      f"fail for n <= {g['graded_predeclared_coarse_max']}, pass for n >= "
      f"{g['graded_predeclared_fine_min']} -> "
      f"**{'PASS' if v['graded_discrimination_gate'] else 'FAIL'}**")
    a("")
    a("So the 3% tolerance is *discriminating on this rung and not on the anchor rung*: "
      "it is a statement about the mesh, not decoration. That is the honest answer to "
      "the round-1 audit remark that the governing rung cannot exercise any tolerance "
      "(F3) — it cannot, and this rung is where the pool criterion does work.")
    a("")
    a("Two practical consequences, both visible in the table above:")
    a("")
    a("* a coarse graded mesh **over-predicts the restrained compression by "
      "`1/(2 n^2)`** (12.5% at n = 2, 3.1% at n = 4): for a real hull structure with a "
      "steep thermal gradient, four elements across the gradient is *outside* the 3% "
      "band, and the error is on the safe (over-stressed) side;")
    a("* `sigma_xx` is uniform across the block to round-off "
      f"({min(r['element_spread_rel'] for r in gr['hex_3d']):.1e} worst case) while the "
      "lateral components near the clamped ends reach "
      f"{max(r['lateral_max_over_axial'] for r in gr['hex_3d']):.0%} of `|sigma_xx|` — "
      "a lateral boundary layer that the 1-D model does not represent. The graded rung "
      "therefore validates `sigma_xx` only, and says so.")
    a("")
    a("### 4.6 Gate-strength probe: which gate catches which defect")
    a("")
    a("A tolerance is only worth its number if something can fail it. Two physics "
      "corruptions are injected into an **in-memory copy** of the module (the working "
      "tree is untouched, the mutants are not committed) and the same three quantities "
      "are recomputed:")
    a("")
    a("| quantity | baseline (rel.) | " + " | ".join(m["mutation"] for m in gs["mutants"]) + " |")
    a("|---|---|" + "---|" * len(gs["mutants"]))
    for key, label in (("uniform_rel_error", "uniform rung, sigma_xx vs closed form"),
                       ("free_control_rel", "free control, max abs(sigma) / (E alpha dT)"),
                       ("graded_rel_error_vs_model", "graded rung, sigma_xx vs element-average model")):
        cells = []
        for m in gs["mutants"]:
            val = m["metrics"][key]
            cells.append(f"{val:.3e}" + (" **caught**" if key in m["caught_by"] else ""))
        a(f"| {label} | {gs['baseline'][key]:.3e} | " + " | ".join(cells) + " |")
    a("")
    a("| mutation | predeclared expectation | observed |")
    a("|---|---|---|")
    for m in gs["mutants"]:
        pre = gs["predeclared_expectation"][m["mutation"][0]]
        obs = ", ".join(m["caught_by"]) if m["caught_by"] else "not caught"
        a(f"| `{m['mutation']}` | {pre} | {obs} |")
    a("")
    a(f"* mutation A caught on the uniform gates: "
      f"**{'PASS' if gs['A_caught_on_uniform'] else 'FAIL'}**")
    a(f"* mutation B invisible to every uniform gate (no gate can see it before this "
      f"revision): **{'PASS' if gs['B_invisible_on_uniform'] else 'FAIL'}**")
    a(f"* mutation B caught by the graded rung: "
      f"**{'PASS' if gs['B_caught_on_graded'] else 'FAIL'}** "
      f"-> gate-strength verdict: "
      f"**{'PASS' if v['gate_strength_gate'] else 'FAIL'}**")
    a("")
    a("Mutation B is the operational justification of §4.5: sampling the element "
      "temperature at a node instead of averaging it is invisible on a uniform field "
      "(every node holds the same temperature) and is caught by the graded rung at "
      f"{[m['metrics']['graded_rel_error_vs_model'] for m in gs['mutants'] if m['mutation'].startswith('B')][0]:.1%} "
      "of the axial stress. Without the round-2 rung this case would record that "
      "defect as a pass.")
    a("")
    a("## 5. Notes, limitations, applicability")
    a("")
    a("* The case is **one-dimensional in its physics**: uniform dT, linear elasticity, "
      "no gradient, no transient, no plasticity, no contact. It probes the thermal-strain "
      "load path, the restraint bookkeeping and the stress recovery — not a temperature "
      "field solution. Gradient (bending) and transient entries live elsewhere "
      "(`free_plate_gradient_stress`, `thermal_transient_reference`).")
    a("* **Uniaxial restraint is a numerical idealisation.** `u_x = 0` on the two faces "
      "with free lateral faces is exact for this anchor, but a heated plate restrained by "
      "a colder frame physically sits between the uniaxial and biaxial limits; the "
      "restraint-family table exists to make that bracket explicit instead of claiming "
      "one boundary condition is 'right'.")
    a("* The `u_y`, `u_z` point constraints that remove the rigid-body modes are outside "
      "the load path, so they do not perturb the stress: the free control reaches "
      f"{g['hex_free_control_measure_max']:.1e}, i.e. round-off.")
    a("* Because uniform dT is nodally exact, the governing rung **cannot** detect any "
      "discretisation error, of any order: the only thing it can catch is a round-off "
      "level defect (wrong thermal load, wrong restraint, wrong stress recovery). That "
      "is what §4.3 and §4.5 are for, and §4.5 quantifies the mesh at which the 3% "
      "criterion starts to bite (n = 5 and up, from the analytic bias `1/(2 n^2)`). A "
      "solver can pass the anchor rung and still be wrong on gradients — stated so the "
      "gate is not over-read.")
    a("* **Element-temperature convention is a modelling choice, not an error.** The "
      "thermal load uses one temperature per element (the average of its nodes). On a "
      "graded field this biases the restrained stress by `+1/(2 n^2)` relative; the "
      "bias is second order, so mesh refinement removes it, but a coarse mesh result "
      "should be read as the element-average answer, which is why §4.5 reports both "
      "references instead of only the continuum one.")
    a("* Material data are the transcribed EH36 anchors of `polar_thermal_stress` (with "
      "its ±3% band and its own source caveat), frozen at `T_ref`. The 3% verdict is "
      "FEM-vs-closed-form, i.e. a numerical-verification statement, **not** a claim about "
      "the accuracy of EH36 steel data.")
    a("")
    a("## 6. Error verdict")
    a("")
    a("| key quantity | computed | reference | error | <3%? |")
    a("|---|---|---|---|---|")
    a(f"| 1-D bar sigma_xx (n={b_last['n_elem']}, uniform dT) | {b_last['sigma_MPa']:.6f} MPa | "
      f"{b_last['reference_MPa']:.6f} MPa | {abs(b_last['rel_error']):.2e} | "
      f"{'YES' if b_last['pass_task_tol'] else 'NO'} |")
    for name in ("uniaxial", "biaxial", "triaxial"):
        row = [r for r in st["hex_restraint_family"] if r["restraint"] == name][-1]
        a(f"| 3-D hex8 sigma_xx ({name}, {row['n_elem_per_side']}^3) | "
          f"{row['sigma_xx_Pa']/1e6:.6f} MPa | {row['reference_Pa']/1e6:.6f} MPa | "
          f"{abs(row['rel_error_xx']):.2e} | {'YES' if row['pass_task_tol'] else 'NO'} |")
    a(f"| free-expansion control (sigma must vanish) | "
      f"{g['hex_free_control_measure_max']:.2e} x (E alpha dT) | 0 | "
      f"{g['hex_free_control_measure_max']:.2e} | YES (round-off) |")
    a(f"| nu sweep drift of sigma_xx | {g['nu_max_drift_rel']:.2e} | 0 | "
      f"{g['nu_max_drift_rel']:.2e} | YES |")
    g1 = st["graded_temperature"]["bar_1d"][-1]
    g3 = st["graded_temperature"]["hex_3d"][-1]
    a(f"| graded T(x), 1-D, n={g1['n_elem_axial']} | {g1['sigma_Pa']/1e6:.6f} MPa | "
      f"{g1['continuum_Pa']/1e6:.6f} MPa | {abs(g1['continuum_rel_error']):.2e} | "
      f"{'YES' if g1['pass_task_tol'] else 'NO'} |")
    a(f"| graded T(x), 3-D hex8, nx={g3['n_elem_axial']} | {g3['sigma_Pa']/1e6:.6f} MPa | "
      f"{g3['continuum_Pa']/1e6:.6f} MPa | {abs(g3['continuum_rel_error']):.2e} | "
      f"{'YES' if g3['pass_task_tol'] else 'NO'} |")
    a(f"| graded T(x), coarse mesh n={st['graded_temperature']['bar_1d'][0]['n_elem_axial']} "
      f"(gate must bite) | {st['graded_temperature']['bar_1d'][0]['sigma_Pa']/1e6:.6f} MPa | "
      f"{st['graded_temperature']['bar_1d'][0]['continuum_Pa']/1e6:.6f} MPa | "
      f"{abs(st['graded_temperature']['bar_1d'][0]['continuum_rel_error']):.2e} | "
      f"**NO** (by design, see §4.5) |")
    a("")
    a("**Answer to the case question: yes, every governing stress component agrees with "
      f"the closed form to better than 3% — in fact to at most "
      f"{max(g['bar_uniform_max_abs_rel_error'], g['hex_max_abs_rel_error_xx']):.2e} "
      "relative, i.e. float64 round-off.** The pool gate (<3%) is met with a margin of "
      "about eleven orders of magnitude, so the informational content of the case is not "
      "the margin but the fact that *all four restraint families* and the free control "
      "land on their own analytic branch: a wrong thermal load, a wrong restraint or a "
      "wrong stress-recovery convention would break at least one of them. The graded "
      "rung is the exception to the round-off margin and deliberately so: on a "
      "non-uniform temperature field the answer carries an O(h^2) interpolation bias, "
      "the 3% gate is met only from n = 5 upwards, and the coarse-mesh cells of §4.5 "
      "are reported as failures.")
    a("")
    a("| gate | value | verdict |")
    a("|---|---|---|")
    for key, val in v.items():
        a(f"| {key} | {val} | {'PASS' if val else 'FAIL'} |")
    a("")
    a("## 7. Reproduce")
    a("")
    a("```bash")
    a("PYTHONPATH=src python benchmarks/constrained_thermal_bar.py      # evidence + report + figures")
    a("PYTHONPATH=src python -m pytest tests/test_constrained_thermal_bar.py -q")
    a("```")
    a("")
    a("Figures: `docs/assets/constrained_thermal_field.png`, "
      "`..._profile.png`, `..._convergence.png`, `..._sensitivity.png`.")
    a("")
    return "\n".join(L)


def main() -> dict:
    DOCS.mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    ev = build_evidence()
    steel = ctb.constant_steel("EH36", T_ref=T_REF)

    fig_field(steel, ASSETS / "constrained_thermal_field.png")
    fig_profile(steel, ASSETS / "constrained_thermal_profile.png")
    fig_convergence(ev["studies"]["bar_quadratic_convergence"],
                    ev["studies"]["bar_uniform"],
                    ev["studies"]["graded_temperature"],
                    ASSETS / "constrained_thermal_convergence.png")
    fig_sensitivity(ev["studies"]["hex_restraint_family"],
                    ev["studies"]["nu_sensitivity"],
                    ev["studies"]["dt_sensitivity"],
                    ev["studies"]["temperature_dependent_EH36"],
                    ASSETS / "constrained_thermal_sensitivity.png")

    ev["figures"] = ["constrained_thermal_field.png", "constrained_thermal_profile.png",
                     "constrained_thermal_convergence.png",
                     "constrained_thermal_sensitivity.png"]
    EVIDENCE.write_text(json.dumps(ev, indent=2) + "\n")
    REPORT.write_text(render_report(ev))
    return ev


if __name__ == "__main__":
    evidence = main()
    print(json.dumps(evidence["gates"], indent=2))
    print(json.dumps(evidence["verdict"], indent=2))
    print("reference sigma [MPa]:", evidence["reference"]["value_MPa"])