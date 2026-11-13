"""Certificate figures must live where the push gate will actually carry them.

`tp_pusher.py` (the platform's push harness) builds its commit from
``git diff --name-status`` and drops every path matching
``_EXCLUDE2 = \\.(db|sqlite3?|png|log)$`` **unless** it starts with ``docs/``
(the ``v5.8 F3`` exemption: "docs/ 下的交付图（png/svg）放行").  A figure
committed next to its anchor — ``benchmarks/verified/<anchor>/<name>.png`` — is
therefore never carried to GitHub, so the certificate that names it can never be
satisfied there, no matter how green the local working tree is.

Measured on this checkout (2026-11-13, `git ls-files`):

* tracked ``*.png`` under ``benchmarks/``: 0
* tracked ``*.png`` under ``docs/``: 484

The consequence is visible in ``python tools/verified_anchor_audit.py``: most of
the ``E_ARTIFACT_MISSING`` debt is exactly this path-class mistake.  The repair
proven on ``fem_shell_modal`` (mission m16923, re-landed with figures under
``docs/assets/verified/fem_shell_modal/``) is the template.

This test freezes the *current* offender set so the class cannot grow silently:
a new anchor that names a non-``docs/`` figure fails here, and a repaired anchor
must be deleted from the baseline (shrink-only, same discipline as
``benchmarks/verified/anchor_certificate_debt.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified"

#: Anchors whose certificates still name figures the push gate will drop.
#: Shrink-only: repairing one means deleting its row, never adding a new one.
FROZEN_OFFENDERS: dict[str, int] = {}  # emptied 2026-11-13: every row repaired or stale


def _declared_names(gates_path: Path) -> list[str]:
    payload = json.loads(gates_path.read_text())
    names: list[str] = []
    for value in (payload.get("artifacts") or {}).values():
        if isinstance(value, str):
            names.append(value)
        else:
            names.extend(str(item) for item in value)
    return names


def _offenders() -> dict[str, int]:
    found: dict[str, int] = {}
    for gates in sorted(VERIFIED.glob("*/gates_verified.json")):
        outside = [
            name for name in _declared_names(gates)
            if name.endswith(".png") and not name.startswith("docs/")
        ]
        if outside:
            found[gates.parent.name] = len(outside)
    return found


def test_no_new_anchor_names_a_figure_the_push_gate_drops():
    found = _offenders()
    added = {name: count for name, count in found.items()
             if name not in FROZEN_OFFENDERS}
    assert added == {}, (
        f"certificates name figures outside docs/, which the push gate strips: {added}. "
        "Render the figures into docs/assets/verified/<anchor>/ instead "
        "(see benchmarks/verified/fem_shell_modal/run_verification.py)."
    )


def test_frozen_offender_set_is_shrink_only():
    stale = {name: count for name, count in FROZEN_OFFENDERS.items()
             if name not in _offenders()}
    assert stale == {}, (
        f"these anchors no longer name non-docs figures, delete them from "
        f"FROZEN_OFFENDERS: {stale}"
    )
    grown = {name: (FROZEN_OFFENDERS[name], count)
             for name, count in _offenders().items() if count > FROZEN_OFFENDERS[name]}
    assert grown == {}, (
        f"an anchor added more unshippable figures (frozen, now): {grown}"
    )


def test_repaired_anchors_are_gate_safe():
    """The rebuilt anchors are the worked example; every named figure must ship.

    History: this test used ``fem_shell_modal`` as its example, but that anchor
    has since moved its figures to ``figures_rendered_on_demand`` (the
    ``tools/certificate_renderables.py`` policy) and no longer names a ``docs/``
    PNG at all, so the assertion was vacuous-by-failure on ``main`` before the
    rebuild.  The anchors below name real, committed ``docs/`` figures that are
    rendered from their own ``result.json`` by
    ``tools/anchor_figure_rebuild.py`` -- the property this test exists to keep.
    """
    for anchor in ("fem_inverse_identification", "fem_sent_j_integral",
                   "fem_winkler_plate_k_sweep"):
        gates = VERIFIED / anchor / "gates_verified.json"
        figures = [name for name in _declared_names(gates) if name.endswith(".png")]
        assert figures, f"{anchor}: the repaired anchor must still name its figures"
        assert all(name.startswith("docs/") for name in figures), (anchor, figures)
        assert all((ROOT / name).is_file() for name in figures), (
            f"{anchor}: every named figure must exist at its repo-relative path"
        )
