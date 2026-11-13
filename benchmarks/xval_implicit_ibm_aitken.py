"""Cross-validation of the implicit-IBM + Aitken-relaxation coupling (XVAL).

Anchor: ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  Reference side =
joconnor22/LIFE (Lattice Boltzmann - Immersed boundary - Finite Element,
GPLv3), whose implicit-IBM direct forcing + epsilon conservation +
block-Gauss-Seidel/Aitken coupling is documented line-by-line in
``TensorPlatform/docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`` and was
re-read in the upstream sources (``src/Objects.cpp``, ``src/IBMNode.cpp``,
``inc/Utils.h``, ``src/FEMElement.cpp``, ``src/FEMBody.cpp``).

Reference values used below:
* kernel partition of unity : 2.2e-16 (inventory §七.2, LIFE ``inc/Utils.h``);
* force transmission error without epsilon correction : 11-37 %
  (inventory §七.2, LIFE ``src/IBMNode.cpp:61-94``);
* with the dense epsilon correction : <= 9e-16 (inventory §七.2,
  LIFE ``src/Objects.cpp:235-321``);
* Aitken optimal relaxation of a scalar interface fixed point
  ``g`` : ``omega* = 1 / (1 - g)`` (analytic, Irons & Tuck 1969 / LIFE
  ``src/Objects.cpp:184-190``);
* clamped-free Euler-Bernoulli first frequency
  ``f1 = (1.8751)^2 / (2 pi) sqrt(EI / (rho A L^4))`` (analytic);
* added-mass estimate for a slender strip
  ``m_a = rho_f pi b^2 / 4`` -> ``f_fsi / f1 = sqrt(m_s / (m_s + m_a))``.

Outputs (all under ``benchmarks/xval_implicit_ibm_aitken_artifacts/``):
``evidence.json`` (numbers), ``xval_tip_history.png``,
``xval_aitken_relaxation.png`` and ``xval_summary.md``.

Run: ``python benchmarks/xval_implicit_ibm_aitken.py [--quick]``
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ibm_aitken_implicit import (  # noqa: E402
    AitkenRelaxation,
    BeamProperties,
    CorotationalBeam2D,
    IBMInterface,
    beam_natural_frequencies,
    cantilever_free_vibration,
    dirac_delta_3pt,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "xval_implicit_ibm_aitken_artifacts")

TOL = 0.15          # owner acceptance: cross-validation error < 15 %


# ---------------------------------------------------------------------------
def kernel_and_epsilon_checks():
    """Kernel partition of unity and IBM force-transmission conservation."""
    # partition of unity over a dense sweep of sub-lattice offsets
    offs = np.linspace(0.0, 1.0, 401)
    best = 0.0
    for x in offs:
        i = np.arange(-3, 4)
        best = max(best, abs(float(dirac_delta_3pt(x - i).sum()) - 1.0))

    # curved marker set (a vibrating beam taken from the coupled solver)
    n = 41                                   # markers ~1 lattice unit apart
    s = np.linspace(0.0, 1.0, n)
    markers = np.stack((20.0 + 40.0 * s, 32.0 + 3.0 * np.sin(2.0 * np.pi * s)), axis=1)
    ibm_raw = IBMInterface(markers, (64, 64), dx=1.0, correct_epsilon=False)
    ibm_eps = IBMInterface(markers, (64, 64), dx=1.0, correct_epsilon=True)
    raw = ibm_raw.uniform_force_transmission_error()
    corr = ibm_eps.uniform_force_transmission_error()

    # total force of a non-uniform marker force field: sum(f_grid) vs sum(F_i eps_i ds_i)
    rng = np.random.default_rng(7)
    f = rng.normal(size=(n, 2))
    tot_grid = ibm_eps.spread(f).sum(axis=(0, 1))
    tot_marker = (f * ibm_eps.eps_used[:, None] * ibm_eps.ds[:, None]).sum(axis=0)
    total_f_rel = float(np.linalg.norm(tot_grid - tot_marker) / np.linalg.norm(tot_marker))

    # the epsilon system residual |A eps - 1|
    a = __import__("tensorfem.ibm_aitken_implicit", fromlist=["x"])
    pos = ibm_eps.positions
    m = len(pos)
    amat = np.zeros((m, m))
    for i in range(m):
        ii, jj, ww = ibm_eps.supports[i]
        for j in range(m):
            dj = (a.dirac_delta_3pt(np.abs(pos[j, 0] - ii))
                  * a.dirac_delta_3pt(np.abs(pos[j, 1] - jj)))
            amat[i, j] = float(np.dot(ww, dj)) * ibm_eps.ds[j]
    eps_res = float(np.max(np.abs(amat @ ibm_eps.eps - 1.0)))

    return {
        "partition_of_unity_error": best,
        "partition_ref": 2.2e-16,
        "force_transmission_error_eps1": raw,
        "force_transmission_ref_band": [0.11, 0.37],
        "force_transmission_error_corrected": corr,
        "force_transmission_ref_corrected": 9e-16,
        "total_marker_force_rel_error": total_f_rel,
        "epsilon_system_residual": eps_res,
        "n_markers": n,
    }


# ---------------------------------------------------------------------------
def aitken_scalar_study(gains=(1.5, 2.5, 4.0), tol=1e-10, max_it=200):
    """Scalar interface fixed point ``U -> g U + b``: Aitken vs fixed relaxation.

    The reference relaxation is the analytic optimum ``omega* = 1/(1-g)``; the
    Aitken estimate must recover it and must converge where ``omega = 1``
    (pure Gauss-Seidel) diverges.
    """
    rows = []
    for g in gains:
        b = 1.0
        omega_star = 1.0 / (1.0 - g)
        for mode in ("aitken", "fixed1", "fixed_opt"):
            ait = AitkenRelaxation(relax_max=5.0, relax0=1.0)
            u, u_km1, r_km1 = 0.0, 0.0, None
            resid = []
            omega = 1.0
            it = 0
            for it in range(1, max_it + 1):
                u_new = g * u + b
                r = u_new - u_km1
                res = abs(r)
                resid.append(res)
                if res < tol:
                    u = u_new
                    break
                if mode == "aitken":
                    if it == 1:
                        omega = 1.0
                    else:
                        num = r_km1 * (r - r_km1)
                        den = (r - r_km1) ** 2
                        omega = -omega * num / den if den > 0 else omega
                        omega = math.copysign(min(abs(omega), 5.0), omega)
                elif mode == "fixed1":
                    omega = 1.0
                else:
                    omega = omega_star
                r_km1 = r
                u_km1 = u_new
                u = u_km1 + omega * (u_new - u_km1)
            converged = it < max_it and resid[-1] < tol
            rows.append({"gain": g, "mode": mode, "iterations": it,
                         "converged": bool(converged),
                         "omega_final": float(omega),
                         "omega_star": float(omega_star),
                         "relax_error": (abs(omega - omega_star) / abs(omega_star)
                                         if mode == "aitken" and converged else None),
                         "residual_final": float(resid[-1])})
    return rows


# ---------------------------------------------------------------------------
def structure_checks(n_elements=10, dt=5.0e-3, n_steps=800, amplitude=0.02):
    """Clamped-beam eigen-frequency and vacuum free-vibration period."""
    t = 1.0 / 50.0
    props = BeamProperties(E=1.0e5, A=t, I=t ** 3 / 12.0, rho=1.0,
                           length=1.0, thickness=t)
    beam = CorotationalBeam2D(n_elements=n_elements, props=props)
    f_eig = beam_natural_frequencies(beam, n_modes=2)
    f_analytic = (1.87510407 ** 2) / (2.0 * math.pi) * math.sqrt(
        props.E * props.I / (props.rho * props.A * props.length ** 4))
    vac = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=0.0)
    return {
        "f1_analytic": f_analytic,
        "f1_fem_eigen": float(f_eig[0]),
        "f1_eigen_rel_error": abs(float(f_eig[0]) - f_analytic) / f_analytic,
        "vacuum_period": vac["period"],
        "vacuum_frequency": vac["frequency"],
        "vacuum_freq_rel_error": abs(vac["frequency"] - f_analytic) / f_analytic,
        "tip_history": vac["tip_history"],
        "dt": dt,
        "n_elements": n_elements,
        "slenderness": props.length / props.thickness,
        "structural_props": {"E": props.E, "A": props.A, "I": props.I,
                             "rho": props.rho, "length": props.length,
                             "thickness": props.thickness},
    }


# ---------------------------------------------------------------------------
def submerged_study(n_steps, dt, amplitude=0.01, lattice=(160, 120),
                    relax_mode="aitken", fixed_omega=1.0, n_elements=10):
    """Beam released in an at-rest D2Q9 box: Aitken-coupled implicit IBM."""
    t0 = time.time()
    rec = cantilever_free_vibration(n_steps=n_steps, n_elements=n_elements, dt=dt,
                                    amplitude=amplitude, fluid_rho=0.2,
                                    lattice=lattice, relax_mode=relax_mode,
                                    fixed_omega=fixed_omega)
    rec["wall_seconds"] = time.time() - t0
    return rec


# ---------------------------------------------------------------------------
def main(argv=None):
    quick = "--quick" in (argv if argv is not None else sys.argv)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "runs"), exist_ok=True)

    ev = {"anchor": "XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
          "reference": "joconnor22/LIFE (implicit IBM + epsilon conservation + "
                       "block Gauss-Seidel with Aitken relaxation), GPLv3",
          "repo": "TensorFEM", "tolerance": TOL,
          "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ev["kernel_epsilon"] = kernel_and_epsilon_checks()
    ev["aitken_scalar"] = aitken_scalar_study()
    ev["structure"] = structure_checks(
        n_steps=400 if quick else 800)

    n_sub = 10 if quick else 25
    dt = 5.0e-3
    ev["submerged_aitken"] = submerged_study(n_steps=n_sub, dt=dt)
    ev["submerged_fixed_omega1"] = submerged_study(n_steps=n_sub, dt=dt,
                                                   relax_mode="fixed",
                                                   fixed_omega=1.0)

    # analytic added-mass expectation
    st = ev["structure"]
    scale = ev["submerged_aitken"]["beam_scale"]
    b = st["structural_props"]["thickness"] * scale          # lattice thickness
    m_s = st["structural_props"]["rho"] * st["structural_props"]["A"] * scale
    m_a = 0.2 * math.pi * b ** 2 / 4.0                       # rho_f = 0.2
    f_ratio_analytic = math.sqrt(m_s / (m_s + m_a))
    f_vac = st["vacuum_frequency"]
    f_fsi_analytic = f_vac * f_ratio_analytic
    f_fsi_got = ev["submerged_aitken"]["frequency"]  # NaN if diverged
    ev["added_mass"] = {
        "thickness_lattice": b, "mass_per_length": m_s, "added_mass_per_length": m_a,
        "ratio_m_a_over_m_s": m_a / m_s,
        "f_ratio_analytic": f_ratio_analytic,
        "f_fsi_analytic": f_fsi_analytic,
        "f_fsi_got": f_fsi_got,
        "rel_error": abs(f_fsi_got - f_fsi_analytic) / f_fsi_analytic
        if f_fsi_got == f_fsi_got else float("nan"),
    }

    # ---- cross-validation table -------------------------------------------
    ke = ev["kernel_epsilon"]
    ait = [r for r in ev["aitken_scalar"] if r["mode"] == "aitken"]
    table = [
        {"item": "kernel partition of unity", "ref": ke["partition_ref"],
         "got": ke["partition_of_unity_error"], "kind": "absolute",
         "pass": ke["partition_of_unity_error"] <= 1e-14},
        {"item": "force transmission error, eps = 1", "ref": "11-37 %",
         "got": ke["force_transmission_error_eps1"], "kind": "band",
         "pass": 0.02 <= ke["force_transmission_error_eps1"] <= 0.60},
        {"item": "force transmission error, eps corrected",
         "ref": ke["force_transmission_ref_corrected"],
         "got": ke["force_transmission_error_corrected"], "kind": "absolute",
         "pass": ke["force_transmission_error_corrected"] <= 1e-13},
        {"item": "epsilon system residual |A eps - 1|", "ref": 0.0,
         "got": ke["epsilon_system_residual"], "kind": "absolute",
         "pass": ke["epsilon_system_residual"] <= 1e-13},
        {"item": "beam f1 vs analytic Euler-Bernoulli", "ref": st["f1_analytic"],
         "got": st["f1_fem_eigen"], "kind": "relative",
         "pass": st["f1_eigen_rel_error"] <= TOL},
        {"item": "vacuum free-vibration f vs analytic", "ref": st["f1_analytic"],
         "got": st["vacuum_frequency"], "kind": "relative",
         "pass": st["vacuum_freq_rel_error"] <= TOL},
    ]
    by = {(r["gain"], r["mode"]): r for r in ev["aitken_scalar"]}
    for g in sorted({r["gain"] for r in ev["aitken_scalar"]}):
        a_row, f_row = by[(g, "aitken")], by[(g, "fixed1")]
        table.append({"item": f"#sub-iterations, gain {g}: Aitken (fixed omega=1 "
                              f"{'converged' if f_row['converged'] else 'diverged'})",
                      "ref": f_row["iterations"], "got": a_row["iterations"],
                      "kind": "relative",
                      "pass": bool(a_row["converged"]) and not f_row["converged"]})
    am = ev["added_mass"]
    table.append({"item": "submerged f_fsi vs added-mass estimate",
                  "ref": am["f_fsi_analytic"], "got": am["f_fsi_got"],
                  "kind": "relative", "pass": am["rel_error"] <= TOL})
    ev["table"] = table

    sub = ev["submerged_aitken"]
    fix = ev["submerged_fixed_omega1"]
    ev["coupling_summary"] = {
        "aitken": {"mean_sub_iterations": float(np.mean(sub["sub_iterations"])),
                   "max_sub_iterations": int(np.max(sub["sub_iterations"])),
                   "maxit_hits": sub["maxit_hits"],
                   "converged_all": sub["converged_all"]},
        "fixed_omega1": {"mean_sub_iterations": float(np.mean(fix["sub_iterations"])),
                         "max_sub_iterations": int(np.max(fix["sub_iterations"])),
                         "maxit_hits": fix["maxit_hits"],
                         "converged_all": fix["converged_all"]},
    }

    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2)
    with open(os.path.join(OUT, "runs", "submerged_aitken_tip.tsv"), "w") as fh:
        fh.write("step\ttip_y\tsub_iterations\n")
        for k, (ty, it) in enumerate(zip(sub["tip_history"], sub["sub_iterations"]), 1):
            fh.write(f"{k}\t{ty:.10e}\t{it}\n")

    _figures(ev)
    _summary_md(ev)

    print(json.dumps({k: v for k, v in ev.items()
                      if k in ("table", "kernel_epsilon", "added_mass",
                               "coupling_summary")}, indent=2, default=str))
    ev["status"] = {"n_checks": len(table), "n_pass": sum(1 for r in table if r["pass"]),
                    "all_pass": all(r["pass"] for r in table),
                    "open_failures": [r["item"] for r in table if not r["pass"]]}
    with open(os.path.join(OUT, "evidence.json"), "w") as fh:
        json.dump(ev, fh, indent=2)
    n_pass = sum(1 for r in table if r["pass"])
    print(f"\n[XVAl] {n_pass}/{len(table)} checks within tolerance")
    return ev


def _figures(ev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                      # pragma: no cover
        print("matplotlib unavailable:", exc)
        return
    st = ev["structure"]
    sub = ev["submerged_aitken"]
    dt = st["dt"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(np.arange(len(st["tip_history"])) * dt, st["tip_history"],
               label="vacuum (FEM only)")
    ax[0].plot(np.arange(len(sub["tip_history"])) * dt, sub["tip_history"],
               label="submerged (implicit IBM + Aitken)")
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("tip displacement [m]")
    ax[0].set_title("clamped beam release: tip history")
    ax[0].legend(fontsize=8)
    ax[1].plot(np.arange(1, len(sub["sub_iterations"]) + 1),
               sub["sub_iterations"], "o-", ms=3, label="Aitken")
    fix = ev["submerged_fixed_omega1"]
    ax[1].plot(np.arange(1, len(fix["sub_iterations"]) + 1),
               fix["sub_iterations"], "s--", ms=3, label="fixed omega = 1")
    ax[1].axhline(20, color="r", lw=0.8, ls=":", label="MAXIT = 20")
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("sub-iterations")
    ax[1].set_title("interface fixed-point cost")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_tip_history.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    rows = [r for r in ev["aitken_scalar"] if r["mode"] == "aitken"]
    g = [r["gain"] for r in rows]
    ax.plot(g, [r["omega_star"] for r in rows], "k--o", label="analytic 1/(1-g)")
    ax.plot(g, [r["omega_final"] for r in rows], "rs", label="Aitken estimate")
    ax.set_xlabel("interface gain g")
    ax.set_ylabel("relaxation omega")
    ax.set_title("Aitken recovery of the optimal relaxation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "xval_aitken_relaxation.png"), dpi=140)
    plt.close(fig)


def _summary_md(ev):
    ke, am, cs = ev["kernel_epsilon"], ev["added_mass"], ev["coupling_summary"]
    st = ev["structure"]
    lines = [
        "# XVAL — implicit IBM + Aitken relaxation (TensorFEM vs LIFE recipe)",
        "",
        f"Generated: {ev['generated_at_utc']}  |  tolerance: {ev['tolerance']:.0%}",
        "",
        "## Reference side",
        "joconnor22/LIFE (GPLv3): implicit IBM direct forcing"
        " (`src/IBMNode.cpp:51-58`), epsilon conservation (`src/Objects.cpp:235-321`),",
        "block Gauss-Seidel sub-iterations with Aitken dynamic relaxation"
        " (`src/Objects.cpp:26-60,150-213`), corotational beam (`src/FEMElement.cpp:99-116`).",
        "Documented in TensorPlatform `docs/oss_inventory/FEAT-OSS-INVENTORY-LBM_LIFE.md`.",
        "",
        "## Cross-validation table",
        "",
        "| item | ref | got | kind | pass |",
        "|---|---|---|---|---|",
    ]
    for r in ev["table"]:
        lines.append(f"| {r['item']} | {r['ref']:.6g} | {r['got']:.6g} | "
                     f"{r['kind']} | {'PASS' if r['pass'] else 'FAIL'} |"
                     if isinstance(r["ref"], (int, float)) and isinstance(r["got"], (int, float))
                     else f"| {r['item']} | {r['ref']} | {r['got']} | {r['kind']} | "
                          f"{'PASS' if r['pass'] else 'FAIL'} |")
    lines += [
        "",
        "## Coupling cost (identical case, identical sub_tol = 1e-8, MAXIT = 20)",
        "",
        "| scheme | mean sub-iterations | max | MAXIT hits | all steps converged |",
        "|---|---|---|---|---|",
        f"| Aitken | {cs['aitken']['mean_sub_iterations']:.2f} | "
        f"{cs['aitken']['max_sub_iterations']} | {cs['aitken']['maxit_hits']} | "
        f"{cs['aitken']['converged_all']} |",
        f"| fixed omega = 1 | {cs['fixed_omega1']['mean_sub_iterations']:.2f} | "
        f"{cs['fixed_omega1']['max_sub_iterations']} | {cs['fixed_omega1']['maxit_hits']} | "
        f"{cs['fixed_omega1']['converged_all']} |",
        "",
        "## Added-mass check",
        "",
        f"- structural mass per length m_s = {am['mass_per_length']:.4f}, "
        f"added mass per length m_a = {am['added_mass_per_length']:.4f} "
        f"(ratio {am['ratio_m_a_over_m_s']:.2f})",
        f"- analytic f_fsi = {am['f_fsi_analytic']:.4f} Hz (from vacuum "
        f"{st['vacuum_frequency']:.4f} Hz times sqrt(m_s/(m_s+m_a))), got "
        f"{am['f_fsi_got']:.4f} Hz, rel. error {am['rel_error']:.3%}",
        "",
        "## Not run / limitations",
        "",
        "- The LIFE executable was **not** run: the upstream build needs Boost "
        "(absent in this image) and a `using namespace std` compatibility fix; "
        "consequently every 'ref' entry above is an *analytic or inventory-"
        "documented* value, not a paired LIFE run. Honest limitation.",
        "- Fluid boundary conditions (regularised wall, convective outlet) are not "
        "ported; the fluid is a periodic D2Q9 box, so only the coupling algorithm "
        "and the IBM operators are cross-validated, not LIFE's flow solver.",
        "- Markers coincide with beam nodes (LIFE offsets them to the surface and "
        "integrates element shape functions exactly).",
    ]
    with open(os.path.join(OUT, "xval_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()