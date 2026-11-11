"""Regression guards for the 3-D DCB path follower (``tensorfem.cohesive3d_dcb``).

Two defects were found while trying to produce the specimen-level verification
artefacts for feature-ledger anchor *GAP-FEM-3D-COHESIVE*; each one made the
benchmark physically meaningless or non-terminating:

1. **Prescribed load dofs were left in the free set.**  The opening of a DCB is
   imposed by displacement control, so the loaded face must be constrained.
   While those dofs were free, Newton drove their residual to zero: the reported
   reaction load was identically zero, the specimen never opened, the damage
   variable never left zero and every peak-load criterion collapsed to a
   relative error of -1.
2. **The follower never terminated.**  The loop condition was
   ``while delta < delta_max * (1 + 1e-12)``, which stays true after the final
   accepted step (``delta == delta_max``); the run then appended zero-length
   steps for ever ("the DCB benchmark never finishes", the historical reason no
   run artefact existed).

The tests below pin *behaviour*, not the implementation: a nonzero reaction that
grows with the imposed opening, and termination exactly at ``delta_max``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import DCBGeometry, build_dcb_model, solve_dcb

TORCH_DTYPE = torch.float64


def _law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(1e12, 4e11, 5e6, 3e6, 300.0, 800.0, 2.2)


def _geometry(**kwargs) -> DCBGeometry:
    params = dict(
        length=60e-3,
        width=3e-3,
        arm_thickness=3e-3,
        initial_crack=30e-3,
        nx=12,
        ny=1,
        nz=2,
    )
    params.update(kwargs)
    return DCBGeometry(**params)


@pytest.fixture(scope="module")
def model():
    return build_dcb_model(_geometry(), _law())


def test_loaded_dofs_are_prescribed_not_free(model):
    """The displacement-controlled face must be constrained, not solved for."""
    load_dofs = np.concatenate((model.upper_load_dofs, model.lower_load_dofs))
    assert load_dofs.size > 0
    assert np.all(np.isin(load_dofs, model.fixed_dofs))
    assert not np.intersect1d(load_dofs, model.free_dofs).size


def test_reaction_load_is_nonzero_and_grows_with_opening(model):
    """A pre-damage DCB must carry load; the old code reported exactly zero."""
    small = solve_dcb(model, delta_max=2.0e-6, n_steps=4, tol=1e-8)
    large = solve_dcb(model, delta_max=4.0e-6, n_steps=4, tol=1e-8)
    p_small = small.peak["load"]
    p_large = large.peak["load"]
    assert p_small > 0.0 and p_large > 0.0
    # linear elastic branch: doubling the imposed opening doubles the load
    assert p_large == pytest.approx(2.0 * p_small, rel=5e-2)
    # ... and the specimen really is opening / the ligament stays elastic
    assert large.peak["max_damage"] == pytest.approx(0.0, abs=1e-12)


def test_follower_terminates_at_delta_max(model):
    """Regression for the ``delta < delta_max*(1+1e-12)`` infinite loop."""
    delta_max = 3.0e-6
    result = solve_dcb(model, delta_max=delta_max, n_steps=3, tol=1e-8)
    assert result.stats["converged_all"] is True
    assert result.stats["reach_delta"] == pytest.approx(delta_max, rel=1e-9)
    # a bounded number of committed steps: no zero-length spin at the end
    assert result.stats["steps"] <= 100 * 3
    assert len(result.curve) == result.stats["steps"]
    # the last committed step is the target, and it appears exactly once
    tops = [c for c in result.curve if c["disp"] == pytest.approx(delta_max, rel=1e-12)]
    assert len(tops) == 1