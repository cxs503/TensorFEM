"""Machine-checked verification tables for the 3-D cohesive interface element.

Every row is a *closed-form* statement about ``tensorfem.cohesive3d`` -- no
stored baseline numbers, no tolerance tuned after looking at the output.  The
script writes two artefacts next to itself:

* ``benchmarks/cohesive3d_verification_tables.json`` -- machine readable, one
  record per check with ``check``, ``oracle``, ``measure``, ``value``,
  ``tolerance``, ``status``;
* ``benchmarks/cohesive3d_verification_tables.md`` -- the same table rendered
  for ``docs/cohesive3d-interface-element.md``.

The tables cover the two kinematics of the element:

``small_strain`` (the original formulation, reference-frame separation) is
verified only for *infinitesimal* interface rotation.  Its finite rigid-rotation
error is recorded as a ``KNOWN_LIMIT`` row with a measured number instead of
being hidden -- a rigid rotation of ``theta`` in a fixed frame produces a
spurious traction of order ``k_s g theta``.

``corotational`` (current mid-surface frame, reference-area quadrature) is
verified for arbitrary rigid rotation: zero traction under rigid motion, exact
rotation objectivity, exact consistent tangent (finite differences in the
softening branch), invariant dissipation under rigid rotation, and mesh
objectivity of an assembled patch.

Usage::

    PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py

Exit status is 0 when every ``PASS`` row is inside its predeclared tolerance
(``KNOWN_LIMIT`` rows never fail the run -- they are measurements, not claims).
"""

from __future__ import annotations

import json
import math
import pathlib

import torch

from tensorfem.cohesive3d import (
    BKCohesive3DLaw,
    assemble_cohesive_interface,
    cohesive_hex8_response,
    interface_area,
)
from tensorfem.mixed_mode_cohesive import BenzeggaghKenaneLaw

HERE = pathlib.Path(__file__).resolve().parent
JSON_PATH = HERE / "cohesive3d_verification_tables.json"
MD_PATH = HERE / "cohesive3d_verification_tables.md"

D = torch.float64
torch.set_default_dtype(D)

# Structural adhesive, SI units (identical to tests/test_cohesive3d.py).
K_N, K_S = 1.0e12, 4.0e11
T_N0, T_S0 = 5.0e6, 3.0e6
G_IC, G_IIC = 500.0, 1200.0
ETA = 1.6

SIZE, GAP = 1.0, 0.1


def new_law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(K_N, K_S, T_N0, T_S0, G_IC, G_IIC, ETA)


def reference_law() -> BenzeggaghKenaneLaw:
    return BenzeggaghKenaneLaw(K_N, K_S, T_N0, T_S0, G_IC, G_IIC, ETA)


def flat_element(size: float = SIZE, gap: float = GAP) -> torch.Tensor:
    s, g = size, gap
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [s, s, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, g],
            [s, 0.0, g],
            [s, s, g],
            [0.0, s, g],
        ],
        dtype=D,
    )


def rotation(axis_angle) -> torch.Tensor:
    """Rodrigues rotation matrix (axis_angle = angle * unit axis)."""
    vec = torch.as_tensor(axis_angle, dtype=D)
    theta = torch.linalg.vector_norm(vec)
    if float(theta) == 0.0:
        return torch.eye(3, dtype=D)
    k = vec / theta
    skew = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=D
    )
    return torch.eye(3, dtype=D) + math.sin(float(theta)) * skew + (1.0 - math.cos(float(theta))) * (skew @ skew)


def patch_mesh(n: int):
    """``n x n`` interface patch: top face offset by GAP along +z."""
    xs = [i / n for i in range(n + 1)]
    nodes, index = [], {}
    for j in range(n + 1):
        for i in range(n + 1):
            for z in (0.0, GAP):
                index[(i, j, z)] = len(nodes)
                nodes.append([xs[i], xs[j], z])
    conn = []
    for j in range(n):
        for i in range(n):
            conn.append(
                [
                    index[(i, j, 0.0)],
                    index[(i + 1, j, 0.0)],
                    index[(i + 1, j + 1, 0.0)],
                    index[(i, j + 1, 0.0)],
                    index[(i, j, GAP)],
                    index[(i + 1, j, GAP)],
                    index[(i + 1, j + 1, GAP)],
                    index[(i, j + 1, GAP)],
                ]
            )
    return torch.tensor(nodes, dtype=D), torch.tensor(conn)


def record(check, oracle, measure, value, tolerance, status="PASS"):
    return {
        "check": check,
        "oracle": oracle,
        "measure": measure,
        "value": float(value),
        "tolerance": float(tolerance),
        "status": status,
    }


def ok(value: float, tolerance: float) -> str:
    return "PASS" if value <= tolerance else "FAIL"


# --------------------------------------------------------------------------- #
# Constitutive law
# --------------------------------------------------------------------------- #


def check_law_reduces_to_2d():
    law, ref = new_law(), reference_law()
    worst = 0.0
    for ratio in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (3.0, 4.0), (1.0, 3.0)):
        direction = torch.tensor([ratio[0], ratio[1], 0.0], dtype=D)
        direction = direction / torch.linalg.vector_norm(direction)
        for amp in (1e-7, 1e-6, 5e-6, 2e-5, 1e-4):
            three = law.envelope_traction(amp * direction)
            two = ref.proportional_response(
                torch.tensor([amp], dtype=D), torch.tensor(ratio, dtype=D)
            ).traction[0]
            for component in range(2):
                a, b = float(three[component]), float(two[component])
                scale = max(abs(b), 1e-3 * T_N0)
                worst = max(worst, abs(a - b) / scale)
    return record(
        "law_reduces_to_2d_bk",
        "3-D law == merged 2-D BK law on delta_s2 = 0",
        "max scaled traction difference over 5 mode-mix ratios x 5 amplitudes",
        worst,
        1e-12,
        ok(worst, 1e-12),
    )


def integrated_energy(law, direction, steps=20000):
    """int_0^{a_f} t . d(delta) along a proportional path (trapezoid)."""
    c = torch.as_tensor(direction, dtype=D)
    c = c / torch.linalg.vector_norm(c)
    a_f = float(law.failure_amplitude(c))
    grid = torch.linspace(0.0, a_f, steps + 1, dtype=D)[1:].unsqueeze(-1) * c
    traction = law.envelope_traction(grid)
    along = (traction * c).sum(-1)
    step = a_f / steps
    return float(along.sum() * step)


def check_law_energy():
    law = new_law()
    worst, detail = 0.0, {}
    for name, direction, target in (
        ("mode_I", [1.0, 0.0, 0.0], G_IC),
        ("mode_II", [0.0, 1.0, 0.0], G_IIC),
        ("mode_III", [0.0, 0.0, 1.0], G_IIC),
        ("mixed_45", [0.7071, 0.0, 0.7071], None),
    ):
        value = integrated_energy(law, direction)
        target = target if target is not None else float(law.critical_energy(torch.tensor([direction], dtype=D))[0])
        rel = abs(value - target) / target
        detail[name] = (value, target, rel)
        worst = max(worst, rel)
    return [
        record(
            f"law_energy_{name}",
            "int t . d(delta) along the envelope == G_c(direction)",
            f"{name}: measure {value:.6f} vs {target:.6f} J/m^2",
            rel,
            1e-5,
            ok(rel, 1e-5),
        )
        for name, (value, target, rel) in detail.items()
    ]


def check_law_initiation():
    law = new_law()
    worst = 0.0
    for phi in torch.linspace(0.0, math.pi / 2.0, 11, dtype=D):
        direction = torch.tensor([[math.cos(float(phi)), math.sin(float(phi)), 0.0]], dtype=D)
        onset = float(law.onset_amplitude(direction))
        traction = law.envelope_traction(onset * direction)[0]
        ratio = (float(traction[0]) / T_N0) ** 2 + (float(traction[1]) / T_S0) ** 2
        worst = max(worst, abs(ratio - 1.0))
    return record(
        "law_initiation_quadratic",
        "(t_n/t_n0)^2 + (t_s/t_s0)^2 == 1 at damage onset",
        "max |quadratic criterion - 1| over 11 mode-mix angles",
        worst,
        1e-12,
        ok(worst, 1e-12),
    )


def check_law_tangent():
    law = new_law()
    worst = 0.0
    for phi in torch.linspace(0.0, math.pi / 2.0, 7, dtype=D):
        direction = torch.tensor([math.cos(float(phi)), math.sin(float(phi)), 0.0], dtype=D)
        for amp in (1e-7, 1e-6, 5e-6, 2e-5, 8e-5):
            sep = (amp * direction).reshape(1, 3)
            analytic = law.update(sep).tangent[0]

            def traction(x, sep=sep):
                return law.update(sep + x.reshape(1, 3)).traction.reshape(3)

            numeric = torch.autograd.functional.jacobian(traction, torch.zeros(3, dtype=D))
            scale = float(analytic.abs().max())
            worst = max(worst, float((analytic - numeric).abs().max()) / scale)
    return record(
        "law_tangent_vs_ad",
        "analytic dt/d(delta) == autograd Jacobian of the traction",
        "max relative tangent difference over 35 (angle, amplitude) points",
        worst,
        1e-10,
        ok(worst, 1e-10),
    )


# --------------------------------------------------------------------------- #
# Element
# --------------------------------------------------------------------------- #


def check_element_patch():
    law = new_law()
    coords = flat_element()
    area = float(interface_area(coords))
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 2e-5
    force, _, _, _ = cohesive_hex8_response(coords, u, law)
    expected = float(law.envelope_traction(torch.tensor([[2e-5, 0.0, 0.0]], dtype=D))[0, 0]) * area
    actual = float(force[4:, 2].sum())
    rel = abs(actual - expected) / abs(expected)
    return record(
        "element_patch_small_strain",
        "uniform opening: reaction on the top face == t * A (unit area)",
        f"measure {actual:.6f} N vs {expected:.6f} N",
        rel,
        1e-12,
        ok(rel, 1e-12),
    )


def rotation_error(kinematics: str, angles):
    law = new_law()
    coords = flat_element()
    scale = T_N0 * float(interface_area(coords))
    worst = 0.0
    for angle in angles:
        rot = rotation([angle, 0.0, 0.0])
        u = coords @ (rot.T - torch.eye(3, dtype=D))  # rigid rotation in a fixed frame
        force = cohesive_hex8_response(coords, u, law, kinematics=kinematics)[0]
        worst = max(worst, float(force.abs().max()) / scale)
    return worst


def check_rigid_rotation():
    angles = [0.01, 0.1, 0.3, 0.7, 1.4, 3.0]
    coro = rotation_error("corotational", angles)
    small = rotation_error("small_strain", angles)
    return [
        record(
            "element_rigid_rotation_small_strain",
            "rigid rotation of the body in a fixed frame must give zero traction",
            f"max |f| / (t_n0 A) over theta = {angles} (reference-frame separation)",
            small,
            0.0,
            "KNOWN_LIMIT",
        ),
        record(
            "element_rigid_rotation_corotational",
            "rigid rotation of the body in a fixed frame must give zero traction",
            f"max |f| / (t_n0 A) over theta = {angles} (current-frame separation)",
            coro,
            1e-10,
            ok(coro, 1e-10),
        ),
    ]


def check_rotation_objectivity():
    law = new_law()
    coords = flat_element()
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = 3e-6
    u[4:, 0] = 1e-6
    rot = rotation([0.5, -0.2, 0.9])
    f0 = cohesive_hex8_response(coords, u, law, kinematics="corotational")[0]
    f_rot = cohesive_hex8_response(coords @ rot.T, u @ rot.T, law, kinematics="corotational")[0]
    rel = float((f_rot - f0 @ rot.T).norm() / f0.norm())
    return record(
        "element_rotation_objectivity_corotational",
        "f(R X, R u) == R f(X, u) for a rotated element geometry",
        "relative Frobenius difference of the reaction under R (0.5, -0.2, 0.9)",
        rel,
        1e-9,
        ok(rel, 1e-9),
    )


def softened_state():
    """(coords, displacement, law) at a damaged, rotated interface state."""
    law = new_law()
    coords = flat_element()
    rot = rotation([0.3, 0.1, 0.2])
    u = coords @ (rot.T - torch.eye(3, dtype=D))
    u[4:, 2] += 2e-5
    return coords, u, law


def check_tangent_symmetry():
    coords, _, law = softened_state()
    # elastic state: rigid rotation plus a 2e-6 m opening (well below the onset
    # separation t_n0 / k_n = 5e-6 m), so no damage has occurred.
    u = coords @ (rotation([0.3, 0.1, 0.2]).T - torch.eye(3, dtype=D))
    u[4:, 2] += 2e-6
    _, tangent, _, _ = cohesive_hex8_response(coords, u, law, kinematics="corotational")
    rel = float((tangent - tangent.T).abs().max() / tangent.abs().max())
    return record(
        "corotational_elastic_tangent_symmetry",
        "tangent is the Hessian of a potential in the elastic regime => symmetric",
        "max |K - K^T| / max |K|",
        rel,
        1e-12,
        ok(rel, 1e-12),
    )


def check_tangent_finite_difference():
    coords, u, law = softened_state()
    _, tangent, _, diagnostics = cohesive_hex8_response(coords, u, law, kinematics="corotational")
    damage = float(diagnostics["damage"].max())
    eps = 1e-9
    numeric = torch.zeros(24, 24, dtype=D)
    for j in range(24):
        delta = torch.zeros(24, dtype=D)
        delta[j] = eps
        plus = cohesive_hex8_response(
            coords, (u.reshape(-1) + delta).reshape(8, 3), law, kinematics="corotational"
        )[0].reshape(-1)
        minus = cohesive_hex8_response(
            coords, (u.reshape(-1) - delta).reshape(8, 3), law, kinematics="corotational"
        )[0].reshape(-1)
        numeric[:, j] = (plus - minus) / (2.0 * eps)
    rel = float((tangent - numeric).norm() / numeric.norm())
    return record(
        "corotational_tangent_vs_finite_difference",
        "d f / d u == central differences of the internal force",
        f"relative Frobenius difference at damage {damage:.3f} (central differences, eps = 1e-9 m)",
        rel,
        1e-6,
        ok(rel, 1e-6),
    )


def dissipation(kinematics: str, angle: float, steps: int = 40):
    """Work of a monotonic opening applied along the rotated normal."""
    law = new_law()
    coords = flat_element()
    rot = rotation([angle, 0.0, 0.0])
    base = coords @ (rot.T - torch.eye(3, dtype=D))
    normal = rot @ torch.tensor([0.0, 0.0, 1.0], dtype=D)
    stroke = 2.2e-4
    work, history, previous = 0.0, None, 0.0
    for i in range(1, steps + 1):
        magnitude = stroke * i / steps
        u = base.clone()
        u[4:8] += magnitude * normal
        force, _, history, _ = cohesive_hex8_response(coords, u, law, history, kinematics=kinematics)
        current = float((force[4:8] @ normal).sum())
        work += 0.5 * (current + previous) * (stroke / steps)
        previous = current
    return work


def check_dissipation():
    works = {angle: dissipation("corotational", angle) for angle in (0.0, 0.4, 1.2)}
    base = works[0.0]
    spread = max(abs(value - base) for value in works.values()) / abs(base)
    law = new_law()
    rel_target = abs(base - G_IC * 1.0) / (G_IC * 1.0)
    small = dissipation("small_strain", 0.4)
    return [
        record(
            "corotational_dissipation_rotation_invariance",
            "dissipated work under monotonic opening is independent of rigid rotation",
            "max |W(theta) - W(0)| / W(0) for theta = 0.0, 0.4, 1.2 rad",
            spread,
            1e-12,
            ok(spread, 1e-12),
        ),
        record(
            "corotational_dissipation_vs_fracture_energy",
            "dissipated work == G_Ic * A (trapezoid integration, 40 steps)",
            f"measure {base:.4f} J vs {G_IC:.1f} J",
            rel_target,
            5e-3,
            ok(rel_target, 5e-3),
        ),
        record(
            "small_strain_dissipation_at_rotation",
            "same opening at theta = 0.4 rad, reference-frame separation (known limit)",
            f"measure {small:.1f} J vs {G_IC:.1f} J",
            abs(small - G_IC) / G_IC,
            0.0,
            "KNOWN_LIMIT",
        ),
    ]


def check_mesh_objectivity():
    law = new_law()
    expected = float(law.envelope_traction(torch.tensor([[2e-6, 0.0, 0.0]], dtype=D))[0, 0]) * 1.0
    worst = 0.0
    for n in (1, 2, 4):
        coords, conn = patch_mesh(n)
        u = torch.zeros(coords.shape[0], 3, dtype=D)
        u[coords[:, 2] > 0, 2] = 2e-6
        force, _, _ = assemble_cohesive_interface(coords, conn, u, law, kinematics="corotational")
        reaction = float(force.reshape(-1, 3)[coords[:, 2] > 0, 2].sum())
        worst = max(worst, abs(reaction - expected) / expected)
    return record(
        "corotational_mesh_objectivity",
        "n x n patches (n = 1, 2, 4) of the same unit interface give the same reaction",
        "max relative reaction difference from t * A",
        worst,
        1e-11,
        ok(worst, 1e-11),
    )


def check_assembly_consistency():
    law = new_law()
    coords, conn = patch_mesh(2)
    u = torch.zeros(coords.shape[0], 3, dtype=D)
    u[coords[:, 2] > 0, 2] = 3e-6
    force, tangent, histories = assemble_cohesive_interface(coords, conn, u, law, kinematics="corotational")
    manual_force = torch.zeros_like(force)
    manual_tangent = torch.zeros_like(tangent)
    for e, conn_e in enumerate(conn):
        dofs = (3 * conn_e[:, None] + torch.arange(3)[None, :]).reshape(-1)
        f_e, k_e, _, _ = cohesive_hex8_response(
            coords[conn_e], u[conn_e], law, histories[e], kinematics="corotational"
        )
        manual_force.index_add_(0, dofs, f_e.reshape(-1))
        rows = dofs[:, None].expand(-1, dofs.numel())
        cols = dofs[None, :].expand(dofs.numel(), -1)
        manual_tangent.index_put_((rows, cols), k_e, accumulate=True)
    worst = max(
        float((force - manual_force).abs().max() / manual_force.abs().max()),
        float((tangent - manual_tangent).abs().max() / manual_tangent.abs().max()),
    )
    return record(
        "corotational_assembly_consistency",
        "dense patch assembly == sum of element contributions (force and tangent)",
        "max relative difference, 2 x 2 patch",
        worst,
        1e-12,
        ok(worst, 1e-12),
    )


def check_compression():
    law = new_law()
    coords = flat_element()
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 2] = -1e-8  # closing
    force, _, history, diagnostics = cohesive_hex8_response(coords, u, law, kinematics="corotational")
    expected = -K_N * 1e-8 * float(interface_area(coords))
    rel = abs(float(force[4:, 2].sum()) - expected) / abs(expected)
    damaged = float(diagnostics["damage"].max())
    value = max(rel, damaged)
    return record(
        "corotational_compression_undamaged",
        "closing uses the penalty stiffness k_n and never damages the interface",
        f"reaction {float(force[4:, 2].sum()):.6f} N vs {expected:.6f} N, damage {damaged:.1e}",
        value,
        1e-8,
        ok(value, 1e-8),
    )


def build_tables():
    rows = []
    rows.append(check_law_reduces_to_2d())
    rows.extend(check_law_energy())
    rows.append(check_law_initiation())
    rows.append(check_law_tangent())
    rows.append(check_element_patch())
    rows.extend(check_rigid_rotation())
    rows.append(check_rotation_objectivity())
    rows.append(check_tangent_symmetry())
    rows.append(check_tangent_finite_difference())
    rows.extend(check_dissipation())
    rows.append(check_mesh_objectivity())
    rows.append(check_assembly_consistency())
    rows.append(check_compression())
    return rows


def render_markdown(rows):
    lines = [
        "| check | oracle (closed form) | measured | tolerance | status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{check}` | {oracle} | {measure} | {tol} | {status} |".format(
                check=row["check"],
                oracle=row["oracle"],
                measure=row["measure"],
                tol=("n/a" if row["status"] == "KNOWN_LIMIT" else f"{row['tolerance']:.0e}"),
                status=("**KNOWN_LIMIT**" if row["status"] == "KNOWN_LIMIT" else row["status"]),
            )
        )
    return "\n".join(lines)


def main() -> int:
    rows = build_tables()
    payload = {
        "module": "tensorfem.cohesive3d",
        "parameters": {
            "normal_stiffness_Pa_per_m": K_N,
            "shear_stiffness_Pa_per_m": K_S,
            "normal_strength_Pa": T_N0,
            "shear_strength_Pa": T_S0,
            "mode_i_energy_J_per_m2": G_IC,
            "mode_ii_energy_J_per_m2": G_IIC,
            "bk_eta": ETA,
        },
        "checks": rows,
        "passed": sum(1 for row in rows if row["status"] == "PASS"),
        "failed": sum(1 for row in rows if row["status"] == "FAIL"),
        "known_limits": sum(1 for row in rows if row["status"] == "KNOWN_LIMIT"),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    MD_PATH.write_text(render_markdown(rows) + "\n")
    print(render_markdown(rows))
    print(
        f"\n{payload['passed']} PASS, {payload['failed']} FAIL, "
        f"{payload['known_limits']} KNOWN_LIMIT -> {JSON_PATH.name}, {MD_PATH.name}"
    )
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())