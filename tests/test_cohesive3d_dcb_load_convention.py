"""Load-convention and energy-balance gates for the 3-D DCB driver.

Defect repaired here (honest-FAIL item of anchor ``GAP-FEM-3D-COHESIVE``:
"energy balance 49.9 % gap" and "peak load +21 % vs LEFM").

``tensorfem.cohesive3d_dcb`` reports the **total** reaction ``P = 2 P_arm`` of the
two symmetrically opened arms, but two consumers used *per-arm* prefactors:

* the external-work accumulator integrated ``0.5 P d(delta)`` and skipped the
  first increment, so ``W_ext`` came out ~1.88x the true work (measured
  energy-balance residual 0.467 of ``W_ext``);
* ``beam_serr`` used the per-arm Irwin-Kies prefactor ``1/2`` on the total load,
  inflating ``G_I`` by exactly 2x and under-predicting the LEFM peak load by
  ``sqrt(2)``.

The gates below use oracles that do not share the defect: the strain energy of
the *linear solve* (``0.5 u^T K_solid u``, no load convention involved), the
conjugacy identity ``U = P delta / 4``, and the convention-free fixed-grip
energy release rate ``G = -(1/b) dU/da`` measured on three FE crack lengths.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfem.cohesive3d_dcb import (
    DCBGeometry,
    beam_compliance,
    beam_peak_load,
    beam_serr,
    build_dcb_model,
    solve_dcb,
    williams_root_rotation,
)

#: slim adherends (a/h ~ 38, b/h = 3) so Euler-Bernoulli bending dominates and
#: the beam-theory G oracle is meaningful.
E, NU = 70.0e9, 0.30
L, B, H = 60.0e-3, 3.0e-3, 1.0e-3
DELTA = 1.0e-8


def _geometry(initial_crack: float, nx: int = 120) -> DCBGeometry:
    return DCBGeometry(
        length=L,
        width=B,
        arm_thickness=H,
        young_modulus=E,
        poisson_ratio=NU,
        initial_crack=initial_crack,
        nx=nx,
        ny=1,
        nz=2,
    )


def _elastic(initial_crack: float, n_steps: int = 4):
    """Undamaged (linear) run: returns (geometry, U_solid, reported load, stats)."""
    g = _geometry(initial_crack)
    r = solve_dcb(build_dcb_model(g), delta_max=DELTA, n_steps=n_steps, max_iter=30)
    assert r.stats["converged_all"]
    return g, r.stats["final_strain_energy"], r.curve[-1]["load"], r.stats


def test_elastic_external_work_closes_on_strain_energy():
    """Linear-elastic regime: ``|W_ext - U_solid - W_interface| / W_ext < 0.5 %``.

    Before the repair the residual sat at 0.467 (2x work, minus the skipped
    first increment); afterwards it is limited only by the trapezoidal
    interface-work bookkeeping (~1e-4).
    """
    for crack in (20e-3, 30e-3, 40e-3):
        _, energy, load, stats = _elastic(crack)
        assert load > 0.0
        assert stats["energy_balance_rel"] < 5.0e-3
        assert stats["final_external_work"] == pytest.approx(energy, rel=5.0e-3)


def test_reported_load_is_total_of_both_arms_and_conjugate_to_half_opening():
    """``U == (P/2) * delta / 2`` for the reported (total) load ``P``.

    Per-arm force ``P/2`` moving ``delta/2`` gives ``U = P delta / 4``; using the
    per-arm prefactor on ``P`` (the pre-repair behaviour) is off by 2x.
    """
    for crack in (20e-3, 30e-3, 40e-3):
        _, energy, load, _ = _elastic(crack)
        assert energy == pytest.approx(0.25 * load * DELTA, rel=5.0e-3)


def test_beam_serr_matches_convention_free_fe_energy_release_rate():
    """``G = -(1/b) dU/da|_delta`` vs the repaired beam formula.

    Fixed-grip energy release rate needs no load convention, so it is the oracle
    for :func:`beam_serr`.  The pre-repair ``6 P^2 a^2 / (E b^2 h^3)`` is a
    factor 2 higher and is rejected below (measured ``G_fe / G_old`` ~ 0.43).
    """
    cracks = (40e-3, 38e-3, 36e-3)
    runs = [_elastic(a) for a in cracks]
    a = np.array(cracks)
    energy = np.array([r[1] for r in runs])
    load = np.array([r[2] for r in runs])

    g_fe = -(energy[2] - energy[0]) / (a[2] - a[0]) / B
    a_mid, p_mid = a[1], load[1]
    assert g_fe > 0.0
    g_beam = beam_serr(p_mid, a_mid, _geometry(a_mid), modulus=E, root_rotation=True)
    g_old = 6.0 * p_mid**2 * a_mid**2 / (E * B**2 * H**3)

    assert 0.75 < g_fe / g_beam < 1.05  # residual: bonded-ligament root stiffness
    assert not 0.75 < g_fe / g_old < 1.05  # pre-repair formula decisively rejected


def test_beam_peak_load_inverts_beam_serr_and_is_sqrt2_of_old_formula():
    """``beam_serr(peak) == Gc`` and the repaired peak is ``sqrt(2)`` x the old."""
    toughness = 300.0
    for crack in (20e-3, 30e-3, 40e-3):
        g = _geometry(crack)
        peak = beam_peak_load(crack, toughness, g, root_rotation=True)
        assert beam_serr(peak, crack, g, root_rotation=True) == pytest.approx(
            toughness, rel=1.0e-12
        )
        a_eff = crack + williams_root_rotation(H, E, g.shear_modulus)
        old = B * H**1.5 * np.sqrt(toughness * E / 6.0) / a_eff
        assert peak == pytest.approx(np.sqrt(2.0) * old, rel=1.0e-12)


def test_compliance_full_energy_identity():
    """``U == P^2 C / 4`` with ``C = delta / P``; FE compliance vs beam theory.

    The FE specimen is *stiffer* than the ideal cantilever crack model because
    the bonded ligament carries load behind the crack front, hence
    ``C_fe <= C_beam`` (18 % low on this mesh) -- documented, not a defect.
    """
    crack = 30e-3
    g, energy, load, _ = _elastic(crack)
    c_fe = DELTA / load
    assert energy == pytest.approx(load**2 * c_fe / 4.0, rel=5.0e-3)
    c_beam = beam_compliance(crack, g, root_rotation=True, shear=True)
    assert 0.75 * c_beam < c_fe <= c_beam