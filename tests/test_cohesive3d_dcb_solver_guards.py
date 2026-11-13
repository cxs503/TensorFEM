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

import scipy.sparse.linalg as spla

from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import (
    DCBGeometry,
    build_dcb_model,
    opening_prescription,
    solve_dcb,
)

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

def _linear_compliance(g: DCBGeometry) -> float:
    """``delta / P`` of the intact specimen from the tangent stiffness."""
    m = build_dcb_model(g, _law())
    k_solid = m.solid_stiffness()
    u = np.zeros(m.n_dofs)
    _, k_coh, _, _ = m.interface_response(u, None)
    k_all = k_solid + k_coh
    free = m.free_dofs
    faced = np.concatenate((m.upper_load_dofs, m.lower_load_dofs))
    delta = 1e-8
    u[faced] = delta * opening_prescription(m)[faced]
    u[free] = spla.spsolve(k_all[free][:, free].tocsc(),
                           -(k_all[free][:, faced] @ u[faced]))
    fint = k_all @ u
    load = abs(float(fint[m.upper_load_dofs].sum()))
    load += abs(float(fint[m.lower_load_dofs].sum()))
    return delta / load


def test_prescribed_opening_is_per_node_not_per_dof():
    """Defect 3: the opening was prescribed as ``delta / n_loaded_dofs``.

    Displacement control must translate *every* node of the loaded face by
    ``+delta/2`` (upper) / ``-delta/2`` (lower); dividing by the number of
    loaded dofs made the applied opening -- and therefore ``delta / P`` -- scale
    with the mesh, which is what made the DCB specimen look 6x (``nz=2``) and
    20x (``nz=4``) too compliant.
    """
    m = build_dcb_model(_geometry(), _law())
    presc = opening_prescription(m)
    assert np.allclose(presc[m.upper_load_dofs], 0.5)
    assert np.allclose(presc[m.lower_load_dofs], -0.5)
    opening = presc[m.upper_load_dofs].mean() - presc[m.lower_load_dofs].mean()
    assert opening == pytest.approx(1.0, rel=1e-15)
    # mesh independence: nz=4 carries 20/12 more loaded dofs than nz=2, so the
    # pre-fix prescription would have inflated the compliance by 20/12 = 1.67x
    c2 = _linear_compliance(_geometry(nz=2))
    c4 = _linear_compliance(_geometry(nz=4))
    assert c4 / c2 == pytest.approx(1.0, abs=0.05)


def test_path_follower_load_matches_the_linear_tangent():
    """``delta / P`` from the nonlinear follower == the linear elastic value.

    A single regression pin for the whole chain: prescribed opening -> reaction
    -> compliance.  The pre-fix code returned a value inflated by the number of
    loaded dofs (12 here).
    """
    g = _geometry()
    m = build_dcb_model(g, _law())
    delta_max = 2.0e-6
    r = solve_dcb(m, delta_max=delta_max, n_steps=2, tol=1e-8)
    c_follower = r.stats["reach_delta"] / r.peak["load"]
    assert r.peak["max_damage"] == pytest.approx(0.0, abs=1e-12)
    assert c_follower / _linear_compliance(g) == pytest.approx(1.0, rel=0.02)


# --------------------------------------------------------------------------- #
# Defects 5-7 (m17614): clamp face, penalty-conditioned Newton, admissible state
# --------------------------------------------------------------------------- #
def _bench_law() -> BKCohesive3DLaw:
    """The specimen law of ``benchmarks/cohesive3d_dcb_3d.py`` (G_Ic = 300 J/m^2)."""
    return BKCohesive3DLaw(2.4e14, 1.0e14, 120.0e6, 80.0e6, 300.0, 800.0, 2.2)


def _narrow() -> DCBGeometry:
    return DCBGeometry(
        length=60e-3,
        width=3e-3,
        arm_thickness=3e-3,
        initial_crack=30e-3,
        nx=100,
        ny=1,
        nz=3,
    )


def test_clamp_face_is_fully_fixed():
    """Defect 5: the clamp multiplied the *mask* by 3 instead of the node index.

    ``3 * clamp.numpy()[:, None]`` is ``3`` for every clamped node and ``0``
    otherwise (``3 * True == 3``, ``3 * False == 0``), so the fixed set was
    ``{0, 1, 2, 3, 4, 5}`` -- the six dofs of the first two nodes -- instead of
    the 48 dofs of the 16-node clamp face.  The specimen was held by two nodes,
    which produced a spurious near-null mode (``|lambda|_min = 4.7e-11`` against
    ``|lambda|_max = 2.3e8``) along which Newton drifted even with a converged
    residual (measured ``max|u|`` growing from 1e-4 m to 1.5 m in six steps).
    """
    m = build_dcb_model(_geometry(), _law())
    clamp_nodes = np.flatnonzero((m.nodes[:, 0] == 0.0).numpy())
    assert clamp_nodes.size == 2 * 2 * 3  # both arms, ny+1 columns, nz+1 layers
    expected = (3 * clamp_nodes[:, None] + np.arange(3)).reshape(-1)
    assert np.all(np.isin(expected, m.fixed_dofs))
    # every clamped dof is constrained: no free dof on the clamp face
    assert not np.intersect1d(expected, m.free_dofs).size
    # ... and the clamp is not just the first two nodes (the pre-fix signature)
    assert not np.array_equal(
        np.sort(m.fixed_dofs)[:6], np.arange(6)
    ) or m.fixed_dofs.size > 6


def test_tangent_has_no_spurious_near_null_mode():
    """The clamped-free assembly must not be numerically singular.

    With the pre-fix clamp the smallest eigenvalue of the free-free tangent was
    ``4.7e-11`` (condition number ``5e18``): a *linear* solve survived it, but a
    Newton path follower drifted along the mode.  The threshold ``10`` is two
    orders above the measured ``4.7e-11`` and two below the physical smallest
    eigenvalue of the *test* mesh (``702``; the reference specimen gives
    ``37.5``), so it separates the two regimes without pinning a mesh-dependent
    value.
    """
    m = build_dcb_model(_geometry(), _law())
    k_solid = m.solid_stiffness()
    _, k_coh, _, _ = m.interface_response(np.zeros(m.n_dofs), None)
    free = m.free_dofs
    a = (k_solid + k_coh)[free][:, free].tocsc()
    smallest = spla.eigsh(a, k=1, sigma=0.0, which="LM", return_eigenvectors=False)
    assert float(np.min(np.abs(smallest))) > 10.0


def test_follower_crosses_cohesive_onset():
    """Defect 6: unconditional Newton + an unreachable absolute tolerance froze
    the follower at ``delta = 9e-6 m`` (6 % of the peak opening) with
    ``converged_all = False`` and zero damage.

    The follower must now march *through* the cohesive onset (damage appears in
    the ligament) with a converged, energy-consistent path.  On the reference
    specimen the onset is at ``delta ~ 5.2e-4 m``, so a 5.5e-4 m window is
    enough -- and cheap.
    """
    g = _narrow()
    m = build_dcb_model(g, _bench_law())
    delta_max = 5.5e-4
    r = solve_dcb(m, delta_max=delta_max, n_steps=22, tol=1e-8)
    assert r.stats["converged_all"] is True
    assert r.stats["reach_delta"] == pytest.approx(delta_max, rel=1e-9)
    assert r.peak["max_damage"] > 0.0  # the cohesive law really is active
    assert abs(r.stats["energy_balance_rel"]) < 1e-3
    # the load must still be on the rising branch and of the right order
    assert 20.0 < r.peak["load"] < 30.0


def test_follower_never_commits_an_inadmissible_state():
    """Defect 7: a relative equilibrium tolerance can be met by a diverged state.

    Before the admissibility screen, ``solve_dcb(delta_max=1e-3)`` committed a
    state with ``max|u| = 1 m`` and reported ``final_strain_energy = -13.5 J``
    (negative for a positive-semidefinite ``K_solid``) plus a 3000x energy
    imbalance.  Every committed state must now be bounded and positive-energy.
    """
    g = _narrow()
    m = build_dcb_model(g, _bench_law())
    delta_max = 6.0e-4
    r = solve_dcb(m, delta_max=delta_max, n_steps=24, tol=1e-8)
    assert np.max(np.abs(r.displacement)) <= 1.0e2 * delta_max
    for c in r.curve:
        assert np.isfinite(c["strain_energy"])
        assert c["strain_energy"] >= 0.0
        assert np.isfinite(c["load"])
    assert r.stats["final_strain_energy"] >= 0.0
    assert r.stats["final_external_work"] > 0.0
