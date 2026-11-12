#!/usr/bin/env python3
"""Independent probe for CASE-SPHERE-PRESSURE (board audit m15807).

Purpose: decide whether the 7.3e-7 membrane-stress "FEM error" reported by
mission m15703 is produced by an actual finite-element solve, or is an
analytic value echoed back into the report.  Nothing here imports the case
module's comparison code: the stiffness assembly / solve is exercised through
``tensorfem.axisym_shell`` and every derived quantity is recomputed with
formulas written out in this file.

Checks
  P1  equilibrium residual   ||K u - f|| / ||f||  on the free DOFs
  P2  energy identity        1/2 u.f  vs  N^2 (1-nu)/(E t) * A   (half model)
  P3  stress from nodal w    sigma = E w / (r (1-nu))  at the equator node
  P4  falsification: chord meridian and full (locking) shear integration must
      move the answer; an echoed analytic value would not react
"""
from __future__ import annotations

import json
import math

from tensorfem.axisym_shell import AxisymShell
from tensorfem.axisym_shell_benchmarks import sphere_nodes

R, T, E, NU, P = 10.0, 0.04, 6.825e7, 0.3, 1.0
FACETS = 8
DOFS_BC = {}


def build(facets=FACETS, exact_geometry=True, shear="reduced"):
    nodes = sphere_nodes(R, facets)
    m = AxisymShell(nodes, young=E, poisson=NU, thickness=T, pressure=P,
                    meridian_radius=R if exact_geometry else None,
                    shear_integration=shear)
    bc = {m.dof(0, "u"): 0.0, m.dof(0, "beta"): 0.0,
          m.dof(facets, "u"): 0.0, m.dof(facets, "beta"): 0.0}
    u = m.solve(bc)
    return m, u, bc, nodes


def equilibrium_residual(m, u, bc):
    r = m.K @ u - m.f
    free = [i for i in range(m.ndof) if i not in bc]
    rn = math.sqrt(float(sum(r[i] ** 2 for i in free)))
    fn = math.sqrt(float(sum(m.f[i] ** 2 for i in free)))
    return rn / fn if fn else float("nan"), rn


def main():
    m, u, bc, nodes = build()
    out = {"case": {"R": R, "t": T, "E": E, "nu": NU, "p": P, "facets": FACETS}}

    # -- P1 equilibrium ----------------------------------------------------
    res_rel, res_abs = equilibrium_residual(m, u, bc)
    out["P1_equilibrium"] = {"Ku_minus_f_over_f_free_dofs": res_rel,
                             "absolute_n": res_abs, "ndof": m.ndof,
                             "free_dofs": m.ndof - len(bc)}

    # -- P2 energy identity (independent derivation) ------------------------
    n_membrane = P * R / 2.0                 # equilibrium of half sphere
    eps = n_membrane * (1.0 - NU) / (E * T)  # Hooke, equi-biaxial
    area_half = 2.0 * math.pi * R ** 2       # symmetry half model
    U_ref = n_membrane * eps * area_half     # (1/2)*(N e + N e) * A
    W_fem = 0.5 * float(u @ m.f)
    out["P2_energy"] = {
        "strain_energy_ref_j": U_ref,
        "fem_half_u_dot_f_j": W_fem,
        "rel_diff": abs(W_fem - U_ref) / U_ref,
        "note": "U = N^2(1-nu)/(E t) * A/1 (half model); W = 1/2 u.f",
    }

    # -- P3 stress from nodal radial displacement only ---------------------
    w_pole = float(u[m.dof(0, "w")])
    w_eq = float(u[m.dof(FACETS, "w")])
    r_eq = nodes[-1][0]  # = R
    sigma_indep = E * w_eq / (r_eq * (1.0 - NU))   # sigma_theta = E w /(r(1-nu))
    sigma_ref = P * R / (2.0 * T)
    out["P3_stress_from_nodal_w"] = {
        "w_pole_m": w_pole,
        "w_equator_m": w_eq,
        "w_nonuniform_rel": abs(w_eq - w_pole) / w_pole,
        "sigma_independent_pa": sigma_indep,
        "sigma_reference_pR_2t_pa": sigma_ref,
        "rel_error_independent": abs(sigma_indep - sigma_ref) / sigma_ref,
        "library_reported_rel_error_at_facets8": 7.309107154185312e-07,
    }

    # -- P4 falsification: the number must react to element physics --------
    probes = {}
    _, u_chord, _, _ = build(exact_geometry=False)
    w_chord = float(u_chord[m.dof(0, "w")])
    sig_chord = E * w_chord / (R * (1.0 - NU))
    probes["chord_meridian"] = {
        "w_pole_m": w_chord,
        "sigma_from_w_pa": sig_chord,
        "rel_error": abs(sig_chord - sigma_ref) / sigma_ref,
    }
    _, u_full, _, _ = build(shear="full")
    w_full = float(u_full[m.dof(0, "w")])
    sig_full = E * w_full / (R * (1.0 - NU))
    probes["full_shear_integration"] = {
        "w_pole_m": w_full,
        "sigma_from_w_pa": sig_full,
        "rel_error": abs(sig_full - sigma_ref) / sigma_ref,
        "identical_to_reduced": w_full == w_pole,
    }
    _, u_alt, _, _ = build(facets=16)
    w_alt = float(u_alt[m.dof(0, "w")])
    probes["facets16"] = {"w_pole_m": w_alt,
                          "rel_error": abs(w_alt - w_pole) / w_pole}
    out["P4_sensitivity"] = probes
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()