"""Regression gates for the m20474 independent recomputation of CASE-CANTILEVER-TIP.

The module under test (``docs/audit/CASE-CANTILEVER-TIP-m20474/independent_cantilever.py``)
shares no code with ``tensorfem``; these tests therefore protect the *audit*, i.e.
they fail if

1. the serendipity shape functions or their derivatives drift (checked against
   central differences and against partition of unity -- a wrong Q8 derivative
   still produces plausible-looking deflections),
2. the mesh builder reintroduces isolated (zero-stiffness) nodes, which make the
   global matrix singular,
3. the independent oracles stop agreeing with the closed form,
4. the continuum model stops reproducing the closed-form tip deflection, or
5. the cross-implementation agreement with the repository's own Q8 anchor
   (``benchmarks/verified/fem_cantilever``) is lost, which would mean the
   repository evidence has drifted away from what an independent implementation
   computes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docs" / "audit" / "CASE-CANTILEVER-TIP-m20474" / "independent_cantilever.py"


def _load():
    spec = importlib.util.spec_from_file_location("independent_cantilever", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ic = _load()


# --------------------------------------------------------------------------- #
# Shape functions / mesh integrity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("xi,eta", [(-0.4, 0.7), (0.9, -0.9), (-0.7, -0.2), (0.1, 0.1)])
def test_q8_derivatives_match_finite_differences(xi, eta):
    h = 1e-6
    N, dN = ic.q8_shape(xi, eta)
    Npx, _ = ic.q8_shape(xi + h, eta)
    Nmx, _ = ic.q8_shape(xi - h, eta)
    Npy, _ = ic.q8_shape(xi, eta + h)
    Nmy, _ = ic.q8_shape(xi, eta - h)
    fd = np.column_stack([(Npx - Nmx) / (2 * h), (Npy - Nmy) / (2 * h)])
    assert np.abs(fd - dN).max() < 1e-7
    assert abs(N.sum() - 1.0) < 1e-12          # partition of unity
    assert np.abs(dN.sum(axis=0)).max() < 1e-12  # constant field -> zero gradient


def test_q8_lattice_nodes_all_carry_stiffness():
    """A Q8 lattice has interior points that belong to no element; they must be dropped."""
    coords, conn, lattice = ic.q8_structured_mesh(3, 2, 1.0, 0.01)
    assert conn.max() < coords.shape[0]
    # every retained node is referenced by at least one element
    assert set(np.unique(conn)) == set(range(coords.shape[0]))
    # interior lattice points that no element uses are marked -1
    assert (lattice == -1).any()


def test_q8_global_matrix_is_not_singular_after_clamping():
    r = ic.q8_plane_stress_tip_deflection(8, 2)
    assert np.isfinite(r["tip_deflection_m"])
    assert r["tip_deflection_m"] > 0.0


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #
def test_symbolic_oracle_is_the_closed_form():
    o = ic.symbolic_oracle()
    assert o["symbolic_residual"] == "0"
    assert o["castigliano_integral"] == pytest.approx(o["delta_euler_bernoulli"], rel=1e-12)
    props = ic.section_properties()
    expected = ic.P * ic.L ** 3 / (3.0 * props["EI"])
    assert o["delta_euler_bernoulli"] == pytest.approx(expected, rel=1e-13)


def test_quadrature_oracle_agrees_with_closed_form():
    o = ic.symbolic_oracle()
    q = ic.quadrature_oracle()
    assert q == pytest.approx(o["delta_euler_bernoulli"], rel=1e-9)


def test_shear_term_and_kappa_sensitivity():
    o = ic.symbolic_oracle()
    assert 0.0 < o["shear_fraction_five_sixths"] < 1e-3      # L/h = 100
    # Cowper's exact rectangle value is stiffer than 5/6 -> smaller shear term
    assert o["shear_term_m"]["cowper_rect"] < o["shear_term_m"]["five_sixths"]
    kappas = ic.poisson_correction_factors(0.3)
    assert kappas["cowper_rect"] == pytest.approx(10 * 1.3 / (12 + 3.3), rel=1e-12)


# --------------------------------------------------------------------------- #
# Independent beam FE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_hermite_beam_is_exact_on_the_cubic_solution(n):
    r = ic.hermite_beam_tip_deflection(n)
    ref = ic.symbolic_oracle()["delta_euler_bernoulli"]
    assert abs(r["tip_deflection_m"] - ref) / ref < 1e-9


# --------------------------------------------------------------------------- #
# Independent continuum Q8 ladder
# --------------------------------------------------------------------------- #
def test_q8_continuum_converges_inside_the_anchor_band():
    rows = [ic.q8_plane_stress_tip_deflection(*m) for m in ((8, 1), (16, 2), (32, 4), (64, 4))]
    errs = [r["abs_relative_error_vs_timoshenko"] for r in rows]
    assert all(e < 0.03 for e in errs), errs
    # one-sided and monotone: the continuum model is stiff from below
    assert all(np.diff(errs) < 0.0), errs
    assert all(r["rel_relative_error_vs_euler_bernoulli"] < 0.0 for r in rows)


def test_end_load_idealisation_is_not_a_discriminator_at_l_over_h_100():
    modes = {m: ic.q8_plane_stress_tip_deflection(32, 4, load_mode=m)["tip_deflection_m"]
             for m in ("parabolic", "uniform", "point")}
    ref = max(modes.values())
    for v in modes.values():
        assert abs(v - ref) / ref < 1e-3


def test_tip_load_resultant_equals_the_applied_force():
    # a wrong consistent-load vector is a classic silent failure; the builder raises
    # if the resultant does not match, so a successful solve already asserts it
    r = ic.q8_plane_stress_tip_deflection(16, 2, load_mode="uniform")
    assert np.isfinite(r["tip_deflection_m"])
    with pytest.raises(ValueError):
        ic.q8_plane_stress_tip_deflection(16, 3, load_mode="point")  # odd n_cross


# --------------------------------------------------------------------------- #
# Cross-implementation agreement with the repository anchor
# --------------------------------------------------------------------------- #
def test_cross_implementation_agrees_with_the_repository_q8_anchor():
    repo = ROOT / "benchmarks/verified/fem_cantilever/result.json"
    if not repo.exists():
        pytest.skip("repository anchor evidence not present")
    documented = {(lvl["axial_elements"], lvl["depth_elements"]): lvl["tip_deflection"]
                  for lvl in json.loads(repo.read_text())["q8_levels"]}
    case = ic.REPO_Q8_CASE
    props = ic.section_properties(case["length"], case["width"], case["height"], case["e"])
    ref_eb = case["load"] * case["length"] ** 3 / (3.0 * props["EI"])
    for (n_axial, n_cross) in ((40, 4), (80, 8)):
        if (n_axial, n_cross) not in documented:
            continue
        r = ic.q8_plane_stress_tip_deflection(
            n_axial, n_cross, length=case["length"], height=case["height"],
            thick=case["width"], e=case["e"], nu=case["nu"],
            load_per_width=case["load"] / case["width"])
        decl = documented[(n_axial, n_cross)]
        assert abs(r["tip_deflection_m"] - decl) / decl < 5e-3
        # both implementations must sit on the same side of the closed form
        assert (r["tip_deflection_m"] - ref_eb) * (decl - ref_eb) > 0.0