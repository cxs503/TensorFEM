#!/usr/bin/env python3
"""Scordelis-Lo: attribute the cylindrical-shell4 equilibrium residual.

Question this study answers
---------------------------
``docs/CASE-FEM-SCORDELIS-LO-ROOF.md`` passes 11/11 gates but leaves one item
explicitly open: the metric-consistent cylindrical shell4 integrates the *exact*
arc area, yet the diaphragm reaction sum misses the applied vertical resultant by
``8.4e-3`` at 4x4 / ``5.3e-4`` at 16x16, while the flat-facet family is exact to
machine precision.  The merged report says "reported rather than explained (open
item)".

That is an attribution debt, and it is mechanical: for a linear static solve with
homogeneous essential boundary conditions the exact algebraic identity

    sum_z(reaction) - (-1^T f) = (K 1_z)^T u

holds for *any* assembled K, where ``1_z`` is the unit global z-translation
vector and ``f`` the assembled load.  So the residual is nonzero if and only if
``K 1_z != 0``, i.e. if and only if the element does not annihilate the
*interpolated* rigid vertical translation.  This study measures that quantity.

Derivation of the expected magnitude
------------------------------------
For a unit vertical translation of one cylindrical facet, the local nodal fields
are ``v = -sin(theta)`` (tangential) and ``w = cos(theta)`` (normal), with
``s = R theta``.  Bilinear interpolation at the facet centroid is off by
``-(h^2/8) f''`` and the interpolated tangent derivative by ``-(h^2/24) f'''``, so

    eps_s   = v,s + w/R   = (h^2 cos)/(8 R^3) - (h^2 cos)/(24 R^3)  = h^2 cos/(12 R^3)
    gamma_sz = w,s - v/R  = (h^2 sin)/(8 R^3) - (h^2 sin)/(24 R^3)  = h^2 sin/(12 R^3)

with ``h`` the facet pitch.  The spurious strain therefore scales as ``h^2/R^3``,
the spurious energy density as ``h^4/R^6``, and the reaction residual as the
rigid-mode projection of that strain, i.e. ``O(h^2)``.  Falsifiable statements:

  P1  algebra: ``reaction/applied - 1 == (K 1_z)^T u / (-1^T f)`` to round-off,
      so no other mechanism (load lumping, reaction extraction, ring indexing)
      contributes to the residual.
  P2  a single facet does not annihilate ``1_z``: ``1_z^T K 1_z > 0``, with the
      spurious strain falling as ``pitch^2`` at fixed R.
  P3  it is a curvature effect: at fixed pitch the energy density falls as
      ``R^-6``; ``K 1_x = 0`` exactly, because the axial translation has no
      ``sin(theta)`` modulation and *is* represented by the bilinear map.
  P4  the flat-facet family is exact (``< 1e-9``) on the same problem, so the
      residual is a curved-element property and not a load/reaction bug.
  P5  the closed form above predicts the measured spurious strain to better than
      15 %.

Run: ``PYTHONPATH=src:benchmarks/verified python \
benchmarks/verified/fem_shell_scordelis_lo/reaction_residual_study.py``
Outputs ``reaction_residual_study.{json,png}`` next to this file.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from tensorfem.cylindrical_shell4 import cylindrical_shell4_stiffness
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical
from tensorfem.shell_benchmarks import scordelis_lo_roof

HERE = Path(__file__).resolve().parent
OUT = HERE / "reaction_residual_study.json"
PNG = HERE / "reaction_residual_study.png"

SCHEMA = "fem-shell-scordelis-reaction-residual/1"
LADDER = (4, 8, 16)
DTHETA_REF = math.radians(2.0)
PITCH_REF = 25.0 * DTHETA_REF                      # pitch of one 2 deg facet at R=25
PITCHES = (PITCH_REF / 2.0, PITCH_REF, 2.0 * PITCH_REF, 4.0 * PITCH_REF)
CURV_PITCH = 2.0 * PITCH_REF                       # keeps the R sweep above round-off
RADII = (25.0, 50.0, 100.0, 200.0)
CENTER_ANGLE = 0.25                                # rad, inside the roof's 0..0.698 band
#: A facet energy is trusted only if it clears 100x the eps*sum|K_ij| round-off,
#: i.e. the strain is then good to about 0.5 %.  Below that the point is reported
#: as floor-limited instead of being fitted as physics.
ROUND_OFF_SNR = 100.0

# Scordelis-Lo material data, identical to the merged case
E, NU, T, Q, R0, L0, HALF = 4.32e8, 0.0, 0.25, 90.0, 25.0, 50.0, math.radians(40.0)
REFERENCE_W = -0.3024


def _loglog_slope(xs, ys) -> float:
    """Least-squares slope of log(y) against log(x)."""
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(xs)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def _rigid_translation(n_dof: int, component: int) -> torch.Tensor:
    v = torch.zeros(n_dof, dtype=torch.float64)
    v[component::6] = 1.0
    return v


def predicted_strain(pitch: float, radius: float, center: float) -> float:
    """Closed-form RMS spurious strain of the interpolated vertical translation."""
    e_pred = pitch ** 2 / (12.0 * radius ** 3)
    return e_pred * math.sqrt(math.cos(center) ** 2 / (1.0 - NU ** 2)
                              + (5.0 / 12.0) * math.sin(center) ** 2 / (1.0 + NU))


# ------------------------------------------------------------------ ladder ---
def level(nx: int) -> dict:
    """One Scordelis-Lo solve, audited down to the algebraic identity."""
    started = time.time()
    res = scordelis_lo_cylindrical(nx, nx)
    K = res.stiffness
    if K is None:  # pragma: no cover - contract guard
        raise RuntimeError("scordelis_lo_cylindrical must expose 'stiffness'")
    u, r = res.displacement, res.reaction
    f = K @ u - r                      # assembled load, recovered from the solve
    applied = -float(f.sum())          # total applied vertical resultant
    nt = nx
    ring = [i * (nt + 1) + j for i in (0, nx) for j in range(nt + 1)]
    reaction = float(r[[6 * n + 2 for n in ring]].sum())

    one_z = _rigid_translation(u.numel(), 2)
    one_x = _rigid_translation(u.numel(), 0)
    k_one_z = K @ one_z
    k_one_x = K @ one_x
    area = L0 * R0 * 2.0 * HALF
    # 1_z^T K 1_z is twice the strain energy of the *interpolated* unit vertical
    # translation; divided by E*t*A it is its mean squared strain.
    spur_energy_1z = float(one_z @ k_one_z)
    strain_rms = math.sqrt(max(spur_energy_1z, 0.0) / (E * T * area))
    k_scale = float(K.abs().max())
    return {
        "mesh": "%dx%d" % (nx, nt),
        "dofs": int(u.numel()),
        "applied_vertical": applied,
        "reaction_vertical": reaction,
        "equilibrium_ratio": reaction / applied,
        "residual_rel": reaction / applied - 1.0,
        # P1: the reaction residual must equal the rigid-mode projection
        "rigidmode_identity": float(k_one_z @ u) / applied,
        "identity_difference": abs(reaction / applied - 1.0 - float(k_one_z @ u) / applied),
        "norm_K_1z": float(k_one_z.norm()),
        "norm_K_1x": float(k_one_x.norm()),
        "norm_K_scale": k_scale,
        "spurious_force_over_load": float(k_one_z.norm()) / applied,
        "spurious_energy_1z": spur_energy_1z,
        "spurious_strain_rms": strain_rms,
        "predicted_spurious_strain": predicted_mesh_strain(nx),
        "predicted_strain_ratio": strain_rms / predicted_mesh_strain(nx),
        "probe_displacement": float(res.probe_displacement),
        "probe_relative_error": float(res.probe_displacement) / REFERENCE_W - 1.0,
        "seconds": round(time.time() - started, 2),
    }


def predicted_mesh_strain(nx: int) -> float:
    """Area-weighted RMS spurious strain of the whole roof mesh.

    The assembled quantity is ``1_z^T K 1_z = sum_e 1_z^T K_e 1_z``, so the mesh
    RMS strain is the area-weighted mean of the per-facet predictions, with the
    facet arc pitch ``h_s = R dtheta``.
    """
    nt = nx
    dtheta = 2.0 * HALF / nt
    h_s = R0 * dtheta
    total = 0.0
    for j in range(nt):
        center = -HALF + dtheta * (j + 0.5)
        total += predicted_strain(h_s, R0, center) ** 2
    return math.sqrt(total / nt)


# -------------------------------------------------------------- flat check ---
def flat_facet_control(nx: int, shear: str = "mitc") -> dict:
    """Same problem, flat-facet Shell4: the residual must be at round-off."""
    res = scordelis_lo_roof(nx, nx, plate_shear=shear)
    nt = nx
    ring = [i * (nt + 1) + j for i in (0, nx) for j in range(nt + 1)]
    reaction = float(res.reaction[[6 * n + 2 for n in ring]].sum())
    area = 0.0
    for element in res.elements:
        xyz = res.nodes[element]
        area += 0.5 * float(torch.linalg.vector_norm(torch.linalg.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])))
        area += 0.5 * float(torch.linalg.vector_norm(torch.linalg.cross(xyz[2] - xyz[0], xyz[3] - xyz[0])))
    applied = Q * area
    return {
        "configuration": "flat facet, plate_shear=%s" % shear,
        "mesh": "%dx%d" % (nx, nt),
        "equilibrium_ratio": reaction / applied,
        "residual_rel": reaction / applied - 1.0,
    }


# ------------------------------------------------------------ single facet ---
def facet_energy(pitch: float, radius: float, center: float = CENTER_ANGLE) -> dict:
    """Spurious energy of the interpolated unit vertical translation, one facet."""
    dtheta = pitch / radius
    dx = pitch
    angles = (center - dtheta / 2.0, center - dtheta / 2.0,
              center + dtheta / 2.0, center + dtheta / 2.0)
    xs = (-dx / 2.0, dx / 2.0, dx / 2.0, -dx / 2.0)
    x_theta = torch.tensor(list(zip(xs, angles)), dtype=torch.float64)
    K = cylindrical_shell4_stiffness(x_theta, radius, E, NU, T)
    one_z = _rigid_translation(24, 2)
    one_x = _rigid_translation(24, 0)
    area = pitch * pitch
    energy = float(one_z @ K @ one_z)
    strain = math.sqrt(max(energy, 0.0) / (E * T * area))
    pred = predicted_strain(pitch, radius, center)
    # The energy is a difference of O(E t h) stiffness entries; its round-off is
    # eps * sum|K_ij| over the translation block.  Report the signal-to-noise so
    # that floor-limited points cannot be mistaken for physics.
    block = float(one_z @ K.abs() @ one_z)
    noise = torch.finfo(torch.float64).eps * block
    snr = (energy / noise) if noise > 0 else float("inf")
    return {
        "translation_block_abs_sum": block,
        "energy_snr": snr,
        "round_off_limited": bool(snr < ROUND_OFF_SNR),
        "pitch": pitch,
        "radius": radius,
        "center_angle": center,
        "dtheta": dtheta,
        "pitch_over_radius": dtheta,
        "spurious_energy_density": 0.5 * energy / area,
        "spurious_strain_rms": strain,
        "predicted_strain": pred,
        "predicted_strain_membrane": pitch ** 2 * math.cos(center) / (12.0 * radius ** 3),
        "strain_over_predicted": (strain / pred) if pred > 0.0 else None,
        "norm_K_1z": float((K @ one_z).norm()),
        "norm_K_1x": float((K @ one_x).norm()),
        "energy_1x": float(one_x @ K @ one_x),
    }


# ------------------------------------------------------------------- study ---
def run() -> dict:
    ladder = [level(nx) for nx in LADDER]
    flat = [flat_facet_control(nx) for nx in (4, 8)]
    pitch_sweep = [facet_energy(p, R0) for p in PITCHES]
    curvature_sweep = [facet_energy(CURV_PITCH, r) for r in RADII]
    axis_center = facet_energy(PITCH_REF, R0, center=0.0)
    floor_probe = facet_energy(PITCH_REF / 8.0, R0)

    residuals = [abs(l["residual_rel"]) for l in ladder]
    order_residual = None
    if all(v > 0 for v in residuals):
        order_residual = -_loglog_slope(list(LADDER), residuals)

    pitch_ok = [p for p in pitch_sweep if not p["round_off_limited"]]
    order_pitch = _loglog_slope([p["pitch"] for p in pitch_ok],
                                [p["spurious_energy_density"] for p in pitch_ok])
    curv_ok = [c for c in curvature_sweep if not c["round_off_limited"]]
    floor_limited = [c["radius"] for c in curvature_sweep if c["round_off_limited"]]
    order_curvature = _loglog_slope([c["radius"] for c in curv_ok],
                                    [c["spurious_energy_density"] for c in curv_ok])
    ratios = [p["strain_over_predicted"] for p in pitch_ok]
    worst_prediction = max(abs(v - 1.0) for v in ratios)
    mesh_ratios = [l["predicted_strain_ratio"] for l in ladder]
    worst_mesh_prediction = max(abs(v - 1.0) for v in mesh_ratios)

    gates = [
        {
            "gate": "algebra.rigidmode_identity",
            "quantity": "max |residual - (K 1_z)^T u / (-1^T f)| over 4/8/16",
            "value": max(l["identity_difference"] for l in ladder),
            "bound": "< 1e-10 (relative to the applied load)",
            "passed": max(l["identity_difference"] for l in ladder) < 1e-10,
            "note": "P1: the residual is *exactly* the rigid-mode projection; no "
                    "load-lumping, reaction-extraction or ring-indexing term",
        },
        {
            "gate": "attribution.spurious_rigid_translation",
            "quantity": "min over ladder of spurious RMS strain under unit vertical translation",
            "value": min(l["spurious_strain_rms"] for l in ladder),
            "bound": "> 0, i.e. K 1_z != 0",
            "passed": all(l["spurious_energy_1z"] > 0.0 for l in ladder),
            "note": "P2: the assembled stiffness does not annihilate the "
                    "interpolated vertical translation",
        },
        {
            "gate": "attribution.residual_order",
            "quantity": "observed order of |residual| over 4/8/16",
            "value": order_residual,
            "bound": ">= 1.8 (h^2 from bilinear interpolation error)",
            "passed": order_residual is not None and order_residual >= 1.8,
            "note": "the residual contracts at the interpolation-error rate, so "
                    "it is a discretisation limit and not a persistent bias",
        },
        {
            "gate": "attribution.pitch_scaling",
            "quantity": "log-log slope of spurious energy density vs facet pitch at R = 25",
            "value": order_pitch,
            "bound": "|slope - 4| <= 0.4 (energy ~ strain^2 ~ pitch^4 at fixed R)",
            "passed": len(pitch_ok) >= 3 and abs(order_pitch - 4.0) <= 0.4,
            "note": "P2: the spurious strain itself falls as pitch^2; only the "
                    "points above the element round-off floor are fitted",
        },
        {
            "gate": "attribution.curvature_scaling",
            "quantity": "log-log slope of spurious energy density vs radius at fixed pitch",
            "value": order_curvature,
            "bound": "|slope + 6| <= 0.6 (strain ~ pitch^2 / R^3)",
            "passed": len(curv_ok) >= 3 and abs(order_curvature + 6.0) <= 0.6,
            "note": "P3: curvature-driven; a flat element cannot show this.  The "
                    "larger radii fall to the element round-off floor and are "
                    "excluded from the fit rather than reported as physics",
        },
        {
            "gate": "attribution.axial_translation_exact",
            "quantity": "max over ladder of |K 1_x| / |K|_max",
            "value": max(l["norm_K_1x"] / l["norm_K_scale"] for l in ladder),
            "bound": "< 1e-12",
            "passed": max(l["norm_K_1x"] / l["norm_K_scale"] for l in ladder) < 1e-12,
            "note": "P3 control: the axial translation *is* representable by the "
                    "bilinear map (no sin(theta) modulation) and it is exact",
        },
        {
            "gate": "control.flat_facet_exact",
            "quantity": "max |reaction/applied - 1|, flat-facet Shell4 on the same problem",
            "value": max(abs(c["residual_rel"]) for c in flat),
            "bound": "< 1e-9",
            "passed": max(abs(c["residual_rel"]) for c in flat) < 1e-9,
            "note": "P4: flat facets reproduce a rigid translation exactly; the "
                    "residual is a curved-element property, not a load bug",
        },
        {
            "gate": "attribution.prediction_facet",
            "quantity": "max |measured / derived spurious strain - 1| over the single-facet pitch sweep",
            "value": worst_prediction,
            "bound": "<= 5 % (derivation drops only O(dtheta^2) terms)",
            "passed": all(abs(v - 1.0) <= 0.05 for v in ratios),
            "note": "P5: eps_m = e cos(theta_c), gamma_s = e sin(theta_c) with "
                    "e = pitch^2/(12 R^3) predicts the measured spurious strain; "
                    "evaluated on the points above the round-off floor",
        },
        {
            "gate": "attribution.prediction_mesh",
            "quantity": "max |measured / derived mesh spurious strain - 1| over 4/8/16",
            "value": worst_mesh_prediction,
            "bound": "<= 5 %",
            "passed": all(abs(v - 1.0) <= 0.05 for v in mesh_ratios),
            "note": "the same closed form, integrated over the 16..256 facets of "
                    "the assembled roof (the quantity that actually drives the "
                    "reaction residual)",
        },
        {
            "gate": "accuracy.unchanged",
            "quantity": "w error of the cylindrical element at 16x16",
            "value": ladder[-1]["probe_relative_error"],
            "bound": "|error| < 3 %",
            "passed": abs(ladder[-1]["probe_relative_error"]) < 0.03,
            "note": "the attribution does not change the published accuracy "
                    "verdict; the residual is an order below the displacement error",
        },
    ]
    payload = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": "attribute the cylindrical-shell4 reaction residual registered as "
                    "an open item in docs/CASE-FEM-SCORDELIS-LO-ROOF.md",
        "problem": {
            "case": "Scordelis-Lo cylindrical roof",
            "reference_w": REFERENCE_W,
            "load_pa": Q,
            "young": E,
            "poisson": NU,
            "thickness": T,
            "radius": R0,
            "length": L0,
            "half_angle_deg": math.degrees(HALF),
        },
        "prediction": {
            "formula": "eps_m = e cos(theta_c), gamma_s = e sin(theta_c), "
                       "e = pitch^2/(12 R^3); "
                       "strain_rms^2 = eps_m^2/(1-nu^2) + (5/12) gamma_s^2/(1+nu)",
            "derivation": "unit vertical translation gives local v = -sin(theta), "
                          "w = cos(theta). Bilinear interpolation at the centroid is "
                          "off by -(h^2/8) f'' and the interpolated tangent derivative "
                          "by -(h^2/24) f''', hence eps_s = h^2 cos/(12 R^3) and "
                          "gamma_sz = h^2 sin/(12 R^3).",
        },
        "ladder": ladder,
        "flat_control": flat,
        "pitch_sweep": pitch_sweep,
        "curvature_sweep": curvature_sweep,
        "curvature_floor_limited_radii": floor_limited,
        "pitch_points_above_floor": len(pitch_ok),
        "axis_center_facet": axis_center,
        "round_off_floor_probe": floor_probe,
        "round_off_snr_threshold": ROUND_OFF_SNR,
        "orders": {
            "residual": order_residual,
            "pitch_energy": order_pitch,
            "curvature_energy": order_curvature,
        },
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "conclusion": (
            "The residual is the rigid-mode projection (K 1_z)^T u / (-1^T f): the "
            "metric-consistent cylindrical shell4 does not annihilate the interpolated "
            "vertical translation, because the tangential component of that translation "
            "varies as sin(theta) and a bilinear facet reproduces it only to O(pitch^2). "
            "The spurious strain is eps,gamma = pitch^2 (cos,sin)/(12 R^3), the residual "
            "contracts as h^%.2f and the flat-facet family is exact.  Open item closed: "
            "a discretisation limit of the curved metric, not a load or "
            "reaction-extraction defect." % (order_residual if order_residual else float("nan"))
        ),
    }
    return payload


# ------------------------------------------------------------------ figure ---
def figure(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax = axes[0]
    nmax = [float(l["mesh"].split("x")[0]) for l in payload["ladder"]]
    res = [abs(l["residual_rel"]) for l in payload["ladder"]]
    ident = [abs(l["rigidmode_identity"]) for l in payload["ladder"]]
    ax.loglog(nmax, res, "o-", label="measured |residual|")
    ax.loglog(nmax, ident, "x", label="|(K·1$_z$)$^T$u / (−1$^T$f)|")
    h2 = [res[0] * (nmax[0] / n) ** 2 for n in nmax]
    ax.loglog(nmax, h2, "k--", label="h$^2$ reference")
    for n, r in zip(nmax, res):
        ax.annotate("%.2e" % r, (n, r), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xlabel("elements per edge")
    ax.set_ylabel("|reaction / applied − 1|")
    ax.set_title("(a) residual contracts as h$^2$ (order %.2f)" % payload["orders"]["residual"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    pitch = [p["pitch"] for p in payload["pitch_sweep"]]
    energy = [p["spurious_energy_density"] for p in payload["pitch_sweep"]]
    ax.loglog(pitch, energy, "s-", label="measured (R = 25)")
    p4 = [energy[0] * (p / pitch[0]) ** 4 for p in pitch]
    ax.loglog(pitch, p4, "k--", label="pitch$^4$ reference")
    ax.set_xlabel("facet pitch [m]")
    ax.set_ylabel("spurious energy density [N/m]")
    ax.set_title("(b) single facet, strain ~ pitch$^2$ (slope %.2f)" % payload["orders"]["pitch_energy"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    radii = [c["radius"] for c in payload["curvature_sweep"]]
    en = [c["spurious_energy_density"] for c in payload["curvature_sweep"]]
    ok = [(c["radius"], c["spurious_energy_density"]) for c in payload["curvature_sweep"]
          if not c["round_off_limited"]]
    bad = [(c["radius"], c["spurious_energy_density"]) for c in payload["curvature_sweep"]
           if c["round_off_limited"]]
    ax.loglog([p[0] for p in ok], [p[1] for p in ok], "^-", label="measured (pitch fixed)")
    inv = [ok[0][1] * (ok[0][0] / r) ** 6 for r, _ in ok]
    ax.loglog([p[0] for p in ok], inv, "k--", label="R$^{-6}$ reference")
    if bad:
        ax.loglog([p[0] for p in bad], [abs(p[1]) for p in bad], "x", color="0.5",
                  label="below round-off floor (excluded)")
    ax.set_xlabel("cylinder radius R [m]")
    ax.set_ylabel("spurious energy density [N/m]")
    ax.set_title("(c) curvature-driven (slope %.2f)" % payload["orders"]["curvature_energy"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Scordelis-Lo: origin of the cylindrical shell4 reaction residual "
                 "(rigid-translation interpolation defect)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG, dpi=140)
    plt.close(fig)


def main() -> int:
    started = time.time()
    payload = run()
    payload["wall_seconds"] = round(time.time() - started, 1)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    figure(payload)
    print("%s: %s (%d/%d gates, %.1f s)" % (
        SCHEMA, "PASS" if payload["all_passed"] else "FAIL",
        sum(1 for g in payload["gates"] if g["passed"]), len(payload["gates"]),
        payload["wall_seconds"]))
    for g in payload["gates"]:
        value = g["value"]
        print("  %-40s %-14s %s" % (g["gate"],
                                     ("%.4g" % value) if isinstance(value, (int, float)) else str(value),
                                     "ok" if g["passed"] else "FAIL"))
    print(payload["conclusion"])
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())