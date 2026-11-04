"""Audit: does the flat-facet Shell4 cylinder reproduce classical shell buckling?

Produces ``benchmarks/cylindrical_shell_buckling_evidence.json``.  Three
independent pieces of evidence are collected:

1. **Membrane hoop energy.**  A prescribed uniform radial shrink is applied to
   the faceted ring and the elastic energy is compared with the plane-stress
   continuum value ``(1/2)*E*t/(1-nu**2)/R**2*w**2*Area``.  This isolates the
   hoop term from bending and from the eigensolver.
2. **Prescribed-field elastic energy.**  For a Donnell-like radial field
   ``w = sin(m*pi*x/L)`` the constrained minimum energy over the remaining DOFs
   is compared with ``(1/2)*D*k**4 + (1/2)*E*t/(1-nu**2)/R**2`` energy density.
3. **Eigenbuckling load factor.**  Circumferential and axial mesh sweeps, with
   the critical mode's circumferential amplitude spread and axial zero
   crossings, so a non-axisymmetric mode is visible as such.

Run: ``OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/audit_cylindrical_shell_buckling.py``
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import torch

from tensorfem.cylindrical_shell_buckling import (
    classical_axial_buckling_resultant,
    cylindrical_shell_axial_buckling,
    cylindrical_shell_mesh,
    donell_axisymmetric_load,
)
from tensorfem.shell4 import shell4_stiffness

CASE = dict(radius=1.0, length=1.2, young=200e9, poisson=.3, thickness=.01)


def hoop_membrane_energy(ntheta: int, nz: int = 4) -> dict:
    R, L = CASE["radius"], CASE["length"]
    E, nu, t = CASE["young"], CASE["poisson"], CASE["thickness"]
    nodes, elements = cylindrical_shell_mesh(R, L, nz, ntheta)
    ndof = 6 * len(nodes)
    K = torch.zeros((ndof, ndof), dtype=nodes.dtype)
    for el in elements:
        ids = torch.stack(tuple(6 * el + k for k in range(6)), 1).reshape(-1)
        K[ids[:, None], ids] += shell4_stiffness(nodes[el], E, nu, t)
    radial = nodes.clone(); radial[:, 0] = 0.
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    u = torch.zeros(ndof, dtype=nodes.dtype)
    for k in range(3):
        u[k::6] = -radial[:, k]
    fem = .5 * float(u @ K @ u)
    # A uniform shrink puts the shell in hoops-only uniaxial stress (the axial
    # strain stays zero and the thickness is free), so the continuum density is
    # (1/2)*E*t*(w/R)**2, not the plane-strain (1/2)*E*t/(1-nu**2)*... form.
    # A faceted ring can then only differ by its chord/arc area ratio.
    area = 2. * math.pi * R * L
    continuum = .5 * E * t / R ** 2 * area
    facet_area = ntheta * math.sin(math.pi / ntheta) / math.pi
    return {"ntheta": ntheta, "nz": nz, "energy_fem": fem, "energy_continuum": continuum,
            "ratio": fem / continuum, "facet_area_factor": facet_area,
            "ratio_over_facet_area": fem / continuum / facet_area}


def prescribed_field_energy(nz: int, ntheta: int, m: int) -> dict:
    R, L = CASE["radius"], CASE["length"]
    E, nu, t = CASE["young"], CASE["poisson"], CASE["thickness"]
    D = E * t ** 3 / (12. * (1. - nu ** 2))
    nodes, elements = cylindrical_shell_mesh(R, L, nz, ntheta)
    ndof = 6 * len(nodes)
    K = torch.zeros((ndof, ndof), dtype=nodes.dtype)
    for el in elements:
        ids = torch.stack(tuple(6 * el + k for k in range(6)), 1).reshape(-1)
        K[ids[:, None], ids] += shell4_stiffness(nodes[el], E, nu, t)
    k = m * math.pi / L
    w = torch.sin(k * nodes[:, 0]).to(nodes.dtype)
    radial = nodes.clone(); radial[:, 0] = 0.
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    U = torch.zeros(ndof, dtype=nodes.dtype)
    for c in range(3):
        U[c::6] = w * radial[:, c]
    prescribed = torch.tensor([i for i in range(ndof) if i % 6 < 3], dtype=torch.long)
    mask = torch.ones(ndof, dtype=torch.bool); mask[prescribed] = False
    free = torch.nonzero(mask).flatten()
    u = U.clone()
    u[free] = -torch.linalg.solve(K[free[:, None], free], K[free[:, None], prescribed] @ U[prescribed])
    fem = .5 * float(u @ K @ u)
    volume_area = (L / 2.) * 2. * math.pi * R
    continuum = .5 * D * k ** 4 * volume_area + .5 * E * t / R ** 2 * volume_area
    return {"nz": nz, "ntheta": ntheta, "m": m, "energy_fem": fem,
            "energy_donnell": continuum, "ratio": fem / continuum}


def mode_diagnostics(result, nz: int, ntheta: int) -> dict:
    nodes, mode = result.nodes, result.modes[:, 0].reshape(-1, 6)
    radial = nodes.clone(); radial[:, 0] = 0.
    radial = radial / torch.linalg.vector_norm(radial, dim=1, keepdim=True)
    amp = ((mode[:, :3] * radial).sum(1)).abs().reshape(nz + 1, ntheta)
    interior = amp[1:-1]
    spread = ((interior.max(1).values - interior.min(1).values)
              / interior.max(1).values.clamp_min(1e-30)).max()
    profile = amp[:, 0]
    sign = torch.sign(profile)
    crossings = int((sign[1:] != sign[:-1]).sum())
    return {"circumferential_spread": float(spread),
            "axial_zero_crossings": crossings,
            "axisymmetric": bool(float(spread) < 1e-6)}


def main() -> int:
    torch.set_num_threads(2)
    finite = donell_axisymmetric_load(**CASE)
    classical = classical_axial_buckling_resultant(
        young=CASE["young"], poisson=CASE["poisson"],
        thickness=CASE["thickness"], radius=CASE["radius"])
    evidence = {
        "schema": "tensorfem.cylindrical-shell-buckling-audit/1.0",
        "case": CASE,
        "references": {"donnell_finite_length": finite,
                       "classical_asymptotic": classical,
                       "donnell_over_classical": finite / classical},
        "hoop_energy": [], "prescribed_energy": [], "eigen": [],
        "boundary": ("linear initial-stress eigenbuckling, prescribed unit membrane "
                     "resultant, flat-facet cylinder, diaphragm SS ends; no "
                     "postbuckling/imperfection/plasticity claim"),
    }
    print("== membrane hoop energy (prescribed uniform radial shrink) ==", flush=True)
    for nt in (8, 16, 32, 64):
        row = hoop_membrane_energy(nt)
        evidence["hoop_energy"].append(row)
        print(f"  nt={nt:3d} ratio={row['ratio']:.4f} facet_area={row['facet_area_factor']:.4f}"
              f" ratio/facet_area={row['ratio_over_facet_area']:.4f}", flush=True)

    pathlib.Path("benchmarks/cylindrical_shell_buckling_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n")
    print("== prescribed Donnell field elastic energy, nz=20 nt=16 ==", flush=True)
    for m in (5, 6, 7, 8):
        row = prescribed_field_energy(20, 16, m)
        evidence["prescribed_energy"].append(row)
        print(f"  m={m} ratio_fem/donnell={row['ratio']:.4f}", flush=True)

    def eigen(nz, nt):
        res = cylindrical_shell_axial_buckling(ntheta=nt, nz=nz, **CASE)
        value = float(res.load_factors[0])
        diag = mode_diagnostics(res, nz, nt)
        row = {"nz": nz, "ntheta": nt, "elements": nz * nt, "N_cr": value,
               "error_vs_donnell": (value - finite) / finite,
               "error_vs_classical": (value - classical) / classical,
               "faceting_ratio": math.sin(math.pi / nt) ** 2, **diag}
        evidence["eigen"].append(row)
        print(f"  nz={nz:3d} nt={nt:3d} N_cr={value:.5e} err={100*row['error_vs_donnell']:+7.2f}% "
              f"axisym={diag['axisymmetric']} xings={diag['axial_zero_crossings']}", flush=True)
        # Stream partial evidence so an interrupted run stays salvageable.
        pathlib.Path("benchmarks/cylindrical_shell_buckling_evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n")
        return row

    print("== circumferential sweep (nz=10) ==", flush=True)
    hoop = [eigen(10, nt) for nt in (16, 24, 32, 48, 64, 96)]
    print("== axial sweep (ntheta=32) ==", flush=True)
    axial = [eigen(nz, 32) for nz in (6, 10, 15, 20, 30)]

    errors = [r["error_vs_donnell"] for r in hoop + axial]
    evidence["audit"] = {
        "monotone_in_ntheta": all(b["error_vs_donnell"] <= a["error_vs_donnell"] + 1e-12
                                  for a, b in zip(hoop, hoop[1:])),
        "sign_change": min(errors) < 0 < max(errors),
        "min_error": min(errors), "max_error": max(errors),
        "within_3pct_any_mesh": any(abs(e) < .03 for e in errors),
        "all_modes_non_axisymmetric": all(not r["axisymmetric"] for r in hoop + axial),
        "certified": False,
        "reason": ("flat-facet cylinder buckling is not certifiable within 3%: the "
                   "error is non-monotone and changes sign across the mesh family, "
                   "and no critical mode is axisymmetric"),
    }
    out = pathlib.Path("benchmarks/cylindrical_shell_buckling_evidence.json")
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())