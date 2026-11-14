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
    PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py --check
    PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py --check --quick

The write mode rewrites both artefacts and exits 0 when every ``PASS`` row is
inside its predeclared tolerance (``KNOWN_LIMIT`` rows never fail the run --
they are measurements, not claims).

``--check`` does not write: it re-derives the rows from *this* checkout and asks
whether the committed artefacts are a faithful, current description of the code.
That makes the tables a *gate* rather than a published claim, on three
independent axes:

* ``checks_digest`` -- SHA-256 of ``parameters + checks`` stored in the JSON.  A
  hand-edited number (or a half-merged regeneration) no longer hashes to the
  stored digest, so the tamper is visible without re-running anything.
* ``source_files`` -- repo-relative SHA-256 of every module the numbers depend
  on.  When the code moves and the table does not, ``--check`` reports the stale
  file and exits 3 (this is how the drift described in
  ``docs/cohesive3d-interface-element.md`` section 6 was found: the committed
  ``C_fem`` of the ``a = 40 mm`` specimen was ``4.715234e-05`` while the current
  code gives ``4.713796e-05``, a 3e-4 relative change nobody noticed).
* the numbers themselves -- every committed ``value`` and ``status`` is compared
  against a fresh run within ``atol + rtol * max(|committed|, |fresh|)``.

``--quick`` runs the rows that cost milliseconds and skips the four multi-second
groups (``corotational_tangent_vs_finite_difference``,
``corotational_dissipation_*`` / ``small_strain_dissipation_at_rotation``,
``nonproportional_path_dissipation``, ``corotational_mesh_objectivity``); it is
the mode the default test suite can afford (see
``tests/test_cohesive3d_verification_tables_artifact.py``).

Exit status: 0 reproducible, 1 numeric/status drift or a ``FAIL`` row, 2 artefact
missing, 3 stale source hash or digest mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent

# Import ``tensorfem`` from *this* checkout.  A shared site-packages tree can
# hold an editable install that points at another worktree (measured: without
# this the script imported a foreign checkout and crashed on a bug that is
# already fixed here), which would make the table a statement about somebody
# else's code.  ``provenance`` below records what was actually imported.
_SRC = REPO / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from tensorfem.cohesive3d import (  # noqa: E402
    BKCohesive3DLaw,
    assemble_cohesive_interface,
    cohesive_hex8_response,
    interface_area,
)
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry,
    beam_compliance,
    build_dcb_model,
    opening_prescription,
)
from tensorfem.mixed_mode_cohesive import BenzeggaghKenaneLaw  # noqa: E402
import tensorfem  # noqa: E402

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


#: Every file whose content decides a number in the table.  Repo-relative so the
#: artefact does not leak host paths and can be checked from any checkout.
SOURCE_FILES = (
    "src/tensorfem/cohesive3d.py",
    "src/tensorfem/cohesive3d_dcb.py",
    "src/tensorfem/mixed_mode_cohesive.py",
    "benchmarks/cohesive3d_verification_tables.py",
)

#: Rows produced by the four groups that cost seconds; skipped by ``--quick``.
SLOW_ROWS = frozenset({
    "corotational_tangent_vs_finite_difference",
    "corotational_dissipation_rotation_invariance",
    "corotational_dissipation_vs_fracture_energy",
    "small_strain_dissipation_at_rotation",
    "nonproportional_path_dissipation",
    "corotational_mesh_objectivity",
})


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _git(*args: str):
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), *args],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance() -> dict:
    """What code produced these numbers: repo-relative identity, not a host path."""
    module_file = pathlib.Path(tensorfem.__file__).resolve()
    try:
        relative = module_file.relative_to(REPO.resolve())
        inside = True
    except (OSError, ValueError):  # pragma: no cover - foreign checkout
        relative = pathlib.Path(module_file.name)
        inside = REPO.resolve() in module_file.parents
    return {
        "tensorfem_module_file": relative.as_posix(),
        "imported_from_this_checkout": bool(inside),
        "source_files": {
            rel: sha256_file(REPO / rel)
            for rel in SOURCE_FILES
            if (REPO / rel).is_file()
        },
        # Advisory only: the table is regenerated from a working tree that holds
        # the uncommitted regeneration itself, so ``git status`` is dirty by
        # construction and carries no information.  Content identity -- what the
        # gate actually enforces -- is ``source_files`` above.  ``git_head``
        # therefore names the parent of the commit that lands the table.
        "git_head": _git("rev-parse", "HEAD"),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
    }


def checks_digest(parameters: dict, checks) -> str:
    """SHA-256 of the numbers themselves: any hand edit changes the digest."""
    canonical = json.dumps(
        {"parameters": parameters, "checks": list(checks)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def stale_sources(payload: dict, *, repo: pathlib.Path | None = None) -> list:
    """Declared source hashes that no longer match the checkout."""
    repo = REPO if repo is None else pathlib.Path(repo)
    declared = (payload.get("provenance") or {}).get("source_files") or {}
    if not declared:
        # An artefact written before provenance existed cannot be tied to any code.
        return [{"file": "<provenance>", "declared": None,
                 "current": "source_files missing: the artefact describes no checkout"}]
    stale = []
    for rel, digest in sorted(declared.items()):
        target = repo / rel
        current = sha256_file(target) if target.is_file() else None
        if current != digest:
            stale.append({"file": rel, "declared": digest, "current": current})
    return stale


def compare_rows(committed, fresh, *, rtol: float = 1e-8, atol: float = 1e-10) -> dict:
    """Committed rows vs a fresh run, within ``atol + rtol * max(|a|, |b|)``."""
    by_name = {row["check"]: row for row in committed}
    diffs = []
    for row in fresh:
        name = row["check"]
        old = by_name.get(name)
        if old is None:
            diffs.append({"check": name, "field": "check", "on_disk": None, "fresh": name})
            continue
        if old.get("status") != row["status"]:
            diffs.append({"check": name, "field": "status",
                          "on_disk": old.get("status"), "fresh": row["status"]})
        a, b = float(old.get("value", "nan")), float(row["value"])
        if not abs(a - b) <= atol + rtol * max(abs(a), abs(b)):
            diffs.append({"check": name, "field": "value", "on_disk": a, "fresh": b})
        if float(old.get("tolerance", "nan")) != float(row["tolerance"]):
            diffs.append({"check": name, "field": "tolerance",
                          "on_disk": old.get("tolerance"), "fresh": row["tolerance"]})
    return {"n_diffs": len(diffs), "diffs": diffs}


def markdown_rows(text: str) -> list:
    """``(check, status)`` per table line of the rendered markdown, in order."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append((cells[0].strip("`"), cells[-1].strip("*")))
    return rows


def check_artifact(
    path=JSON_PATH,
    *,
    fresh=None,
    quick: bool = False,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    md_path=None,
) -> dict:
    """Is the committed table a faithful, current description of this checkout?"""
    path = pathlib.Path(path)
    md_path = pathlib.Path(MD_PATH if md_path is None else md_path)
    if not path.is_file():
        return {"ok": False, "exit_code": 2, "artifact": path.name, "quick": quick,
                "reason": f"artifact missing: {path.name}", "comparison": None,
                "stale": [], "digest_ok": None, "markdown_ok": None}
    on_disk = json.loads(path.read_text())
    committed = on_disk.get("checks") or []
    parameters = on_disk.get("parameters") or {}
    rows = build_tables(quick=quick) if fresh is None else list(fresh)
    comparison = compare_rows(committed, rows, rtol=rtol, atol=atol)
    stale = stale_sources(on_disk)
    digest_ok = on_disk.get("checks_digest") == checks_digest(parameters, committed)
    markdown_ok: bool | None = None
    if md_path.is_file():
        expected = [(row["check"], "KNOWN_LIMIT" if row["status"] == "KNOWN_LIMIT"
                     else row["status"]) for row in committed]
        markdown_ok = markdown_rows(md_path.read_text()) == expected
    failed_rows = [row["check"] for row in rows if row["status"] == "FAIL"]
    ok = (
        comparison["n_diffs"] == 0
        and not stale
        and bool(digest_ok)
        and markdown_ok is not False
        and not failed_rows
        and int(on_disk.get("failed", 0)) == 0
    )
    exit_code = 0
    if not ok:
        exit_code = 3 if (stale or not digest_ok) else 1
    return {
        "ok": ok,
        "exit_code": exit_code,
        "artifact": path.name,
        "quick": quick,
        "rows_compared": len(rows),
        "digest_ok": bool(digest_ok),
        "markdown_ok": markdown_ok,
        "comparison": comparison,
        "stale": stale,
        "failed_rows": failed_rows,
    }


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


def check_law_batch_consistency():
    """The batched (vectorised assembly) update must equal independent point calls.

    The element assembly evaluates the law on all Gauss points of the patch at
    once (``4 * n_elements`` points), so the batch path -- not the single-point
    path -- is what the solver actually uses.  A regression in the batch
    broadcasting of the BK mixing weight was invisible to every single-point
    check (and silently order dependent when the batch happened to have three
    points), so it is pinned here by an explicit pointwise oracle.
    """
    law = new_law()
    directions = torch.tensor(
        [[1.0, 0.3, 0.0], [0.2, 1.0, 0.0], [0.8, 0.5, 0.3], [0.0, 1.0, 1.0]], dtype=D
    )
    directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    onset = law.onset_amplitude(directions)
    failure = law.failure_amplitude(directions)
    separation = (0.5 * (onset + failure))[:, None] * directions
    separation = separation.repeat(4, 1)  # 16 points == a 4-element patch
    history = torch.zeros(separation.shape[0], dtype=D)
    batch = law.update(separation, history)
    worst = 0.0
    for i in range(separation.shape[0]):
        one = law.update(separation[i], history[i].reshape(()))
        worst = max(
            worst,
            float((batch.traction[i] - one.traction).abs().max() / one.traction.abs().max()),
            float((batch.tangent[i] - one.tangent).abs().max() / one.tangent.abs().max()),
        )
    damaged = float(batch.damage.max())
    value = worst if damaged > 0.0 else 1.0  # a batch that never softens proves nothing
    return record(
        "law_batch_matches_pointwise",
        "batched update (16 softening points) == 16 independent point calls",
        f"max relative difference {worst:.1e} at damage {damaged:.3f}",
        value,
        1e-12,
        ok(value, 1e-12),
    )


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


# --------------------------------------------------------------------------- #
# Dissipated-energy accounting of the closed-form state function
# --------------------------------------------------------------------------- #


def ray(angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    return torch.tensor([math.cos(theta), math.sin(theta), 0.0], dtype=D)


def path_work(law, separations) -> float:
    """Trapezoidal ``int t . d(delta)`` along an explicit separation path."""
    work, history, previous = 0.0, None, None
    for separation in separations:
        response = law.update(separation, history)
        history = response.history
        if previous is not None:
            work += 0.5 * float(((previous[1] + response.traction) * (separation - previous[0])).sum())
        previous = (separation, response.traction)
    return work


def check_dissipation_closed_form():
    """Closed-form dissipated energy vs the numerical work of the traction law."""
    law = new_law()
    worst = 0.0
    lines = []
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0):
        direction = ray(angle)
        failure = float(law.failure_amplitude(direction))
        amplitudes = torch.linspace(0.0, failure, 2001, dtype=D)
        separation = amplitudes[:, None] * direction[None, :]
        traction = law.update(separation).traction
        integrated = float(torch.trapezoid((traction * direction).sum(-1), amplitudes))
        closed = float(law.dissipated_energy(direction * failure))
        worst = max(worst, abs(closed - integrated) / integrated)
        lines.append(f"{angle:.0f}deg {closed:.4f}")
    return record(
        "law_dissipation_vs_path_integral",
        "closed-form W_d(a_f) == int t . d(delta) along a monotonic ray (7 mixities)",
        "max relative difference, " + "; ".join(lines[:3]) + " ...",
        worst,
        1e-5,
        ok(worst, 1e-5),
    )


def check_dissipation_equals_fracture_energy():
    law = new_law()
    worst = 0.0
    for direction in (
        torch.tensor([1.0, 0.0, 0.0], dtype=D),
        torch.tensor([0.0, 1.0, 0.0], dtype=D),
        torch.tensor([0.0, 0.0, 1.0], dtype=D),
        torch.tensor([1.0, 1.0, 0.0], dtype=D),
        torch.tensor([1.0, 1.0, 1.0], dtype=D),
    ):
        unit = direction / torch.linalg.vector_norm(direction)
        failure = float(law.failure_amplitude(unit))
        closed = float(law.dissipated_energy(unit * failure))
        critical = float(law.critical_energy(unit))
        worst = max(worst, abs(closed - critical) / critical)
    return record(
        "law_dissipation_equals_fracture_energy",
        "W_d(a_f) == G_c(direction), BK mixed mode, modes I / II / III / 45deg / 3-D",
        f"max relative difference from G_c, 5 directions, worst {worst:.2e}",
        worst,
        1e-12,
        ok(worst, 1e-12),
    )


def check_dissipation_monotone():
    law = new_law()
    direction = ray(30.0)
    onset = float(law.onset_amplitude(direction))
    failure = float(law.failure_amplitude(direction))
    critical = float(law.critical_energy(direction))
    elastic = float(law.dissipated_energy(direction[None, :] * torch.linspace(0.0, onset, 33, dtype=D)[:, None]).abs().max())
    amplitudes = torch.linspace(onset, failure * 1.5, 400, dtype=D)
    values = law.dissipated_energy(amplitudes[:, None] * direction[None, :])
    drops = float(torch.clamp(-torch.diff(values), min=0.0).max())
    bound = max(float(values.max()) - critical, 0.0) / critical
    value = max(drops / critical, bound, elastic / critical)
    return record(
        "law_dissipation_monotone_bounded",
        "W_d == 0 on the elastic branch, non-decreasing, saturating at G_c",
        f"elastic max {elastic:.1e} J/m^2, worst decrease {drops:.1e}, overshoot {bound:.1e}",
        value,
        1e-12,
        ok(value, 1e-12),
    )


def check_element_dissipation():
    law = new_law()
    coords = flat_element()
    area = float(interface_area(coords))
    failure = float(law.failure_amplitude(torch.tensor([1.0, 0.0, 0.0], dtype=D)))
    worst = 0.0
    measures = []
    for kinematics in ("small_strain", "corotational"):
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = failure
        _, _, _, diagnostics = cohesive_hex8_response(
            coords, u, law, kinematics=kinematics
        )
        energy = float(diagnostics["dissipated_energy"])
        worst = max(worst, abs(energy - G_IC * area) / (G_IC * area))
        measures.append(f"{kinematics} {energy:.6f} J")
    return record(
        "element_dissipation_equals_gc_times_area",
        "element diagnostics ``dissipated_energy`` == G_Ic * A at complete opening",
        ", ".join(measures) + f" vs {G_IC:.1f} J",
        worst,
        1e-12,
        ok(worst, 1e-12),
    )


def check_element_dissipation_rotation():
    law = new_law()
    coords = flat_element()
    values = []
    for angle in (0.0, 0.3, 1.2):
        rot = rotation([0.0, 0.0, angle])
        u = torch.zeros(8, 3, dtype=D)
        u[4:, 2] = 1e-4
        _, _, _, diagnostics = cohesive_hex8_response(
            coords @ rot.T, u @ rot.T, law, kinematics="corotational"
        )
        values.append(float(diagnostics["dissipated_energy"]))
    spread = (max(values) - min(values)) / max(values)
    return record(
        "element_dissipation_rotation_invariance",
        "element ``dissipated_energy`` independent of a rigid rotation (corotational)",
        "max relative spread over theta = 0.0, 0.3, 1.2 rad",
        spread,
        1e-14,
        ok(spread, 1e-14),
    )


def check_nonproportional_dissipation():
    """Mode-mix rotation at constant gap: measured, not assumed.

    The law is a history-of-maximum-separation model, so a path that rotates
    the mode mix is not integrated exactly by the state function.  The row
    quantifies the residual against the numerically integrated work and the
    bound ``G_c`` of the direction the path ends in.
    """
    law = new_law()
    peak = 1.5 * float(law.onset_amplitude(ray(0.0)))
    shear = float(law.failure_amplitude(ray(90.0)))
    path = [ray(0.0) * peak * (k / 200.0) for k in range(201)]
    path += [torch.tensor([peak, shear * k / 400.0, 0.0], dtype=D) for k in range(1, 401)]
    work = path_work(law, path)
    # The closed form is evaluated on the *current* state, so the history of
    # the last path point is recovered by replaying the path.
    history = None
    for separation in path:
        response = law.update(separation, history)
        history = response.history
    closed = float(response.dissipated_energy)
    residual = abs(closed - work) / work
    overshoot = max(work - G_IIC, 0.0) / G_IIC
    value = max(residual, overshoot)
    return record(
        "nonproportional_path_dissipation",
        "open to 1.5 a_0(I) then shear to failure: 0 <= W - G_IIc and W_d ~= W_path",
        f"W_path {work:.3f} J/m^2 vs G_IIc {G_IIC:.1f}, closed form {closed:.3f} "
        f"(residual {residual:.1e})",
        value,
        1e-3,
        ok(value, 1e-3),
    )


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


def _linear_dcb_compliance(g) -> float:
    """``delta / P`` of the intact DCB specimen from the tangent stiffness.

    The loaded face is *prescribed* (``+delta/2`` / ``-delta/2`` per node), so
    the linear solve is ``K_ff u_f = -K_fc u_c`` and the load is the sum of both
    arm reactions -- identical to the production path follower.  Circular
    import note: ``numpy``/``scipy`` are imported here because the rest of this
    file is pure ``torch`` closed-form work.
    """
    import numpy as np
    import scipy.sparse.linalg as spla

    model = build_dcb_model(g, BKCohesive3DLaw(K_N, K_S, T_N0, T_S0, G_IC, G_IIC, ETA))
    k_solid = model.solid_stiffness()
    u = np.zeros(model.n_dofs)
    _, k_coh, _, _ = model.interface_response(u, None)
    k_all = k_solid + k_coh
    free = model.free_dofs
    faced = np.concatenate((model.upper_load_dofs, model.lower_load_dofs))
    delta = 1e-8
    u[faced] = delta * opening_prescription(model)[faced]
    u[free] = spla.spsolve(k_all[free][:, free].tocsc(),
                           -(k_all[free][:, faced] @ u[faced]))
    fint = k_all @ u
    load = abs(float(fint[model.upper_load_dofs].sum()))
    load += abs(float(fint[model.lower_load_dofs].sum()))
    return delta / load


def check_specimen_compliance():
    """Specimen-level DCB compliance calibration (ASTM D5528 compliance method).

    Three rows, one per initial crack length.  The oracle is the closed-form
    Williams-root-rotation Timoshenko beam ``C(a)``; the FE value has to be
    *bracketed* by the plain beam and the root-rotation beam with 10 % margin,
    because the root rotation itself is only known asymptotically.  A
    mesh-dependent opening prescription (the historical
    ``delta / n_loaded_dofs`` defect) breaks every row by a factor ~6-20.
    """
    rows = []
    arm = 3e-3
    for crack in (20e-3, 30e-3, 40e-3):
        g = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=arm,
                        initial_crack=crack, nx=24, ny=1, nz=2)
        c_fem = _linear_dcb_compliance(g)
        c_plain = beam_compliance(crack, g, root_rotation=False, shear=True)
        c_root = beam_compliance(crack, g, root_rotation=True, shear=True)
        # worst-case relative distance to the bracket [c_plain, 1.10 * c_root]
        if c_fem < c_plain:
            err = (c_plain - c_fem) / c_plain
        elif c_fem > 1.10 * c_root:
            err = (c_fem - 1.10 * c_root) / c_root
        else:
            err = 0.0
        rows.append(
            record(
                f"specimen_compliance_a{crack*1e3:.0f}mm",
                f"bracket [plain {c_plain:.6e}, 1.10x root rotation {1.10*c_root:.6e}] m/N",
                f"C_fem = {c_fem:.6e} m/N ({100*(c_fem-c_root)/c_root:+.1f}% vs root rotation)",
                err,  # 0.0 inside the bracket, distance outside it otherwise
                0.0,
                "PASS" if err == 0.0 else "FAIL",
            )
        )
    return rows


def build_tables(quick: bool = False):
    """Every published row, in order.  ``quick`` drops the ``SLOW_ROWS`` groups."""
    rows = []
    rows.append(check_law_reduces_to_2d())
    rows.extend(check_law_energy())
    rows.append(check_law_initiation())
    rows.append(check_law_tangent())
    rows.append(check_law_batch_consistency())
    rows.append(check_element_patch())
    rows.extend(check_rigid_rotation())
    rows.append(check_rotation_objectivity())
    rows.append(check_tangent_symmetry())
    if not quick:
        rows.append(check_tangent_finite_difference())
        rows.extend(check_dissipation())
    rows.append(check_dissipation_closed_form())
    rows.append(check_dissipation_equals_fracture_energy())
    rows.append(check_dissipation_monotone())
    rows.append(check_element_dissipation())
    rows.append(check_element_dissipation_rotation())
    if not quick:
        rows.append(check_nonproportional_dissipation())
        rows.append(check_mesh_objectivity())
    rows.append(check_assembly_consistency())
    rows.append(check_compression())
    rows.extend(check_specimen_compliance())
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


def build_payload(*, quick: bool = False) -> dict:
    rows = build_tables(quick=quick)
    parameters = {
        "normal_stiffness_Pa_per_m": K_N,
        "shear_stiffness_Pa_per_m": K_S,
        "normal_strength_Pa": T_N0,
        "shear_strength_Pa": T_S0,
        "mode_i_energy_J_per_m2": G_IC,
        "mode_ii_energy_J_per_m2": G_IIC,
        "bk_eta": ETA,
    }
    return {
        "module": "tensorfem.cohesive3d",
        "parameters": parameters,
        "provenance": provenance(),
        "checks": rows,
        "passed": sum(1 for row in rows if row["status"] == "PASS"),
        "failed": sum(1 for row in rows if row["status"] == "FAIL"),
        "known_limits": sum(1 for row in rows if row["status"] == "KNOWN_LIMIT"),
        "checks_digest": checks_digest(parameters, rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare with the committed artefacts instead of writing them")
    parser.add_argument("--quick", action="store_true",
                        help="skip the multi-second rows (SLOW_ROWS); cheap enough for the test suite")
    parser.add_argument("--out", type=pathlib.Path, default=JSON_PATH, help="artefact path")
    parser.add_argument("--rtol", type=float, default=1e-8, help="relative tolerance for --check")
    parser.add_argument("--atol", type=float, default=1e-10, help="absolute tolerance for --check")
    args = parser.parse_args(argv)

    if args.check:
        report = check_artifact(args.out, quick=args.quick, rtol=args.rtol, atol=args.atol)
        if report.get("comparison") is None:
            print(report["reason"])
            print("remedy: python benchmarks/cohesive3d_verification_tables.py   # regenerate")
            return report["exit_code"]
        for diff in report["comparison"]["diffs"]:
            print(f"DIFF {diff['check']}.{diff['field']}: "
                  f"on_disk={diff['on_disk']!r} fresh={diff['fresh']!r}")
        for row in report["stale"]:
            print(f"STALE source {row['file']}: declared {row['declared']} current {row['current']}")
        if not report["digest_ok"]:
            print("DIGEST MISMATCH: checks_digest does not hash the stored parameters+checks")
        if report["markdown_ok"] is False:
            print("MARKDOWN MISMATCH: the rendered table does not match the JSON rows")
        for name in report.get("failed_rows") or []:
            print(f"FAILING ROW: {name}")
        n = len(report["stale"])
        print(f"{report['artifact']} ({'quick' if report['quick'] else 'full'}): "
              f"{'REPRODUCIBLE' if report['ok'] else 'NOT REPRODUCIBLE'} "
              f"({report['comparison']['n_diffs']} leaf differences, {n} stale source(s), "
              f"{report['rows_compared']} rows compared)")
        if not report["ok"]:
            print("remedy: python benchmarks/cohesive3d_verification_tables.py   # regenerate")
        return report["exit_code"]

    payload = build_payload()
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(render_markdown(payload["checks"]) + "\n")
    prov = payload["provenance"]
    if not prov["imported_from_this_checkout"]:
        print(
            f"WARNING: tensorfem imported from {prov['tensorfem_module_file']}, "
            "which is outside this checkout; the table describes that checkout."
        )
    print(render_markdown(payload["checks"]))
    print(
        f"\n{payload['passed']} PASS, {payload['failed']} FAIL, "
        f"{payload['known_limits']} KNOWN_LIMIT -> "
        f"{args.out.name}, {args.out.with_suffix('.md').name}"
    )
    print(f"checks_digest: {payload['checks_digest']}")
    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())