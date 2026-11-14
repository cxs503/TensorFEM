"""Regression: the arc-length continuation must not start on the negative load branch.

Defect (found while exercising the ice-belt plate field for the grillage
ultimate-strength cross-check): on the symmetric zero-displacement start the two
Crisfield roots are ``d = 0`` and ``d = -2*dlam_pred``, which are *exactly*
equidistant from ``target``.  The ``min(..., key=|d - target|)`` tie-break then
sometimes selected the mirror root; the whole path was traced at negative load
factor, kept converging (the elastic response is odd in ``(u, lam)``), reported
``still-ascending`` and returned a *negative* ultimate pressure.  Any consumer
that takes ``peak_load_factor * pressure`` therefore silently gets a negative
collapse pressure -- or, worse, an apparently "safe" negative margin.

The fix restricts the first step to roots that keep the load increment in the
predicted direction (Crisfield's sign rule applied to ``lam`` instead of ``u``).
These tests pin both the repair and the previously working case.
"""
from __future__ import annotations

import numpy as np

from tensorfem.ice_belt_collapse import (
    PlateModel,
    arc_length_limit,
    arc_length_status,
    elastic_perfectly_plastic,
    inplane_free_bcs,
    structured_mesh,
    uniform_pressure_load,
    solve_arc_length,
)


def _path(span_x: float, width: float, nx: int, ny: int, pressure: float,
          n_steps: int = 6, dlam0: float = 0.02):
    grade = elastic_perfectly_plastic(sigma_0=355.0e6)
    nodes, elems = structured_mesh(span_x, width, nx, ny)
    model = PlateModel(nodes, elems, 0.020, grade)
    model.von_karman = True
    fixed = inplane_free_bcs(model)
    fext = uniform_pressure_load(model, pressure)
    return solve_arc_length(model, fext, fixed, n_steps=n_steps, dlam0=dlam0)


def test_slender_strip_starts_on_positive_load_branch():
    """The narrow bay strip (2.1 x 0.35 m) that exposed the defect."""
    path = _path(2.1, 0.35, 20, 4, 2.165e6)
    lam = np.asarray(path.lam[1:], dtype=float)
    assert lam.size >= 3
    assert np.all(lam > 0.0), f"load factor went negative: {lam.tolist()}"
    assert abs(lam[0] - 0.02) <= 0.02 + 1e-9
    assert np.all(np.diff(lam) > 0.0)
    lim = arc_length_limit(path)
    assert lim["peak_load_factor"] > 0.0
    assert arc_length_status(lim) in {"still-ascending", "limit-point", "post-peak"}


def test_square_panel_branch_unchanged():
    """The square panel that always worked must be bit-for-bit unaffected."""
    path = _path(2.0, 2.0, 12, 12, 5.0e6, n_steps=6, dlam0=0.02)
    lam = np.asarray(path.lam[1:], dtype=float)
    assert lam.size >= 3
    assert np.all(lam > 0.0)
    assert np.all(np.diff(lam) > 0.0)


def test_first_step_sign_guard_is_general_not_geometry_specific():
    """Three aspect ratios and two step sizes all start ascending."""
    for span_x, width, nx, ny in ((2.1, 0.35, 20, 4), (0.7, 0.7, 8, 8), (1.4, 0.5, 12, 5)):
        for dlam0 in (0.01, 0.04):
            path = _path(span_x, width, nx, ny, 1.0e6, n_steps=4, dlam0=dlam0)
            lam = np.asarray(path.lam[1:], dtype=float)
            assert np.all(lam > 0.0), (span_x, width, dlam0, lam.tolist())
            assert np.all(np.diff(lam) > 0.0), (span_x, width, dlam0, lam.tolist())