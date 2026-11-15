"""v9: the implicit-IBM added-mass excess survives refinement with a finite
limit (closes the v8 open item).

Reads the ``cj3404``/``cj3405`` artefacts (refinement ``dx = 1, 1/2, 1/4`` at
fixed ``c_phys = 4``; body-size ``c = 4, 8, 16`` at ``dx = 1``) and asserts:

* the coefficient drops monotonically as the lattice is refined (``dx -> 0``);
* the ``dx -> 0`` linear limit is finite and < 1.3 (i.e. the excess does NOT
  vanish, but is bounded well below the coarse ``dx = 1`` value 2.79);
* the coupled frequency ``f/f_vac`` at the limit (``m_a/m_s = 0.6``) is within
  15 % of the thin-plate reference -- the anchor's acceptance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "benchmarks" / "xval_ibm_added_mass_refinement_v8_artifacts"

TOL = 0.15
MASS_RATIO = 0.6


def _load(tag):
    return json.loads((ART / f"{tag}.json").read_text())


def _f_ratio(C):
    return math.sqrt(1.0 / (1.0 + MASS_RATIO * C))


def test_refinement_monotone_drop():
    ref = [_load(f"clean_refine_{r}") for r in (1, 2, 4)]
    C = [r["C_momentum"] for r in ref]
    assert C[0] > C[1] > C[2], "coefficient must drop as dx -> 0"
    assert C[0] > 2.5 and C[2] < 1.7, "coarse vs fine must bracket"


def test_refinement_finite_limit():
    ref = [_load(f"clean_refine_{r}") for r in (1, 2, 4)]
    dx = np.array([r["dx"] for r in ref])
    C = np.array([r["C_momentum"] for r in ref])
    coef, *_ = np.linalg.lstsq(np.vstack([np.ones_like(dx), dx]).T, C,
                               rcond=None)
    C_inf = float(coef[0])
    # finite, non-vanishing, but bounded well below the coarse dx=1 value
    assert 1.0 < C_inf < 1.3, f"C_inf={C_inf:.3f} must be finite in (1, 1.3)"
    # the excess does NOT vanish (C_inf > 1) -- a genuine discrete-IBM artefact


def test_coupled_frequency_within_15():
    ref = [_load(f"clean_refine_{r}") for r in (1, 2, 4)]
    dx = np.array([r["dx"] for r in ref])
    C = np.array([r["C_momentum"] for r in ref])
    coef, *_ = np.linalg.lstsq(np.vstack([np.ones_like(dx), dx]).T, C,
                               rcond=None)
    C_inf = float(coef[0])
    fr = _f_ratio(C_inf)
    fr_ref = _f_ratio(1.0)
    err = abs(fr_ref - fr) / fr_ref
    assert err < TOL, f"coupled-freq err {err:.3f} must be < 15 %"


def test_force_momentum_consistency():
    ref = [_load(f"clean_refine_{r}") for r in (1, 2, 4)]
    for r in ref:
        rel = abs(r["C_force"] - r["C_momentum"]) / r["C_momentum"]
        assert rel < 0.01, f"force/momentum must agree < 1 % (got {rel:.4f})"


def test_constraint_exact():
    ref = [_load(f"clean_refine_{r}") for r in (1, 2, 4)]
    for r in ref:
        assert r["constraint_rel_max"] < 1e-12, "B^-1 gain must be exact"
