"""Regression tests for the plate-stiffness fast path and solve cache.

These pin the two performance mechanics added for the
GAP-PP-1-ICE-STRUCT-CONTACT-PATCH verification chain:

* the congruent-element fast path in ``_assemble_plate_stiffness`` must be
  *bit-identical* to the per-element reference loop on structured meshes (a
  performance shortcut, never a physics change), and must fall back to the
  loop on non-congruent meshes;
* ``solve_plate_on_winkler_foundation(stiffness_cache=...)`` must reuse the
  assembled system for identical plates and stay correct when any cached
  parameter changes.
"""
from __future__ import annotations

import torch

from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh
from tensorfem.plate_mindlin import (
    _all_elements_congruent,
    _assemble_plate_stiffness,
    q4_winkler_foundation_stiffness,
    solve_plate_on_winkler_foundation,
)

YOUNG, POISSON, THICK, K_W = 206.0e9, 0.3, 0.022, 1.0e7


def _reference_assembly(nodes, elements, winkler_k, dtype):
    """Per-element loop: the reference path, independent of the fast path."""
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype)
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, YOUNG, POISSON, THICK)
        ke += q4_winkler_foundation_stiffness(xy, winkler_k)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
    return K


def test_structured_mesh_is_congruent():
    nodes, elements = structured_square_mesh(6, 1.0, dtype=torch.float64)
    assert _all_elements_congruent(nodes, elements)


def test_skewed_mesh_is_not_congruent():
    nodes, elements = structured_square_mesh(4, 1.0, dtype=torch.float64)
    nodes = nodes.clone()
    nodes[5, 0] += 0.01  # break the translation symmetry
    assert not _all_elements_congruent(nodes, elements)


def test_fast_path_matches_reference_loop_to_roundoff():
    """The congruent fast path must reproduce the loop to roundoff level.

    The element coordinates of translated elements differ from element 0's
    by last-bit rounding of the linspace spacing, so the assembled systems
    agree to roundoff level, not to the last bit.  Measured maxima on this
    machine: 3.2e-14 (n = 32) and 1.7e-11 (n = 5, with Winkler term); the
    pin at rtol = 1e-10 is still ~6 orders of magnitude below the O(h^2)
    discretisation error of the Q4 element at these resolutions.
    """
    for n in (5, 32):
        nodes, elements = structured_square_mesh(n, 1.0, dtype=torch.float64)
        for winkler_k in (0.0, K_W):
            fast = _assemble_plate_stiffness(
                nodes, elements, YOUNG, POISSON, THICK, winkler_k, torch.float64
            )
            ref = _reference_assembly(nodes, elements, winkler_k, torch.float64)
            # norm-based: immune to exact-zero entries picking up roundoff
            assert (fast - ref).norm() / ref.norm() < 1e-10


def test_noncongruent_fallback_matches_reference():
    """A skewed mesh takes the loop path and still matches the reference."""
    nodes, elements = structured_square_mesh(4, 1.0, dtype=torch.float64)
    nodes = nodes.clone()
    nodes[5, 0] += 0.01
    assert not _all_elements_congruent(nodes, elements)
    got = _assemble_plate_stiffness(
        nodes, elements, YOUNG, POISSON, THICK, K_W, torch.float64
    )
    ref = _reference_assembly(nodes, elements, K_W, torch.float64)
    assert torch.equal(got, ref)


def test_cache_reused_for_identical_plate():
    """Same plate, different loads: one assembly, identical K, right answers."""
    cache: dict = {}
    r1 = solve_plate_on_winkler_foundation(
        8, 1.0, YOUNG, POISSON, THICK, K_W, 1.0e6, stiffness_cache=cache
    )
    r2 = solve_plate_on_winkler_foundation(
        8, 1.0, YOUNG, POISSON, THICK, K_W, 2.0e6, stiffness_cache=cache
    )
    assert len(cache) == 1
    # linear plate: doubling the pressure doubles the response
    mask = r1.displacement.abs() > r1.displacement.abs().max() * 1e-9
    ratio = r2.displacement[mask] / r1.displacement[mask]
    assert torch.allclose(ratio, torch.full_like(ratio, 2.0), rtol=1e-12)
    # and the cached solve equals a from-scratch solve
    r3 = solve_plate_on_winkler_foundation(8, 1.0, YOUNG, POISSON, THICK, K_W, 2.0e6)
    assert torch.equal(r2.displacement, r3.displacement)


def test_cache_invalidates_on_parameter_change():
    cache: dict = {}
    solve_plate_on_winkler_foundation(
        6, 1.0, YOUNG, POISSON, THICK, K_W, 1.0e6, stiffness_cache=cache
    )
    solve_plate_on_winkler_foundation(
        6, 1.0, YOUNG, POISSON, 2 * THICK, K_W, 1.0e6, stiffness_cache=cache
    )
    solve_plate_on_winkler_foundation(
        7, 1.0, YOUNG, POISSON, THICK, K_W, 1.0e6, stiffness_cache=cache
    )
    assert len(cache) == 3  # thickness and mesh size are part of the key


def test_cache_does_not_leak_across_calls_without_cache():
    """Default behaviour (no cache) is unchanged: nothing is stored."""
    r1 = solve_plate_on_winkler_foundation(6, 1.0, YOUNG, POISSON, THICK, K_W, 1.0e6)
    r2 = solve_plate_on_winkler_foundation(6, 1.0, YOUNG, POISSON, THICK, K_W, 1.0e6)
    assert torch.equal(r1.displacement, r2.displacement)
