"""Interface convergence certificate of the implicit-IBM/Aitken loop.

The sub-iteration loop of :class:`tensorfem.ibm_aitken_implicit.ImplicitIBMFSI`
relaxes a residual vector whose components are **not** commensurable: a
corotational beam node carries two translations (length) and one rotation
(radians).  These tests pin down

* the DOF weights of :func:`tensorfem.ibm_aitken_implicit.dof_scale_vector`,
* the equivalence of ``residual_measure="legacy_isotropic"`` with the exact
  pre-fix expression (so archived numbers can be reproduced),
* the unit invariance of the ``dof_scaled`` residual and the failure of the
  legacy form to be invariant (its angular tolerance grows with the length
  unit / with ``ref_length``),
* the unit invariance of the Aitken Delta^2 estimate once the same weights are
  applied inside the dot products.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_implicit import (
    AitkenRelaxation,
    BeamProperties,
    CorotationalBeam2D,
    IBMInterface,
    ImplicitIBMFSI,
    LBMFluid2D,
    dof_scale_vector,
)


def _beam(n_elements=4, length=8.0):
    props = BeamProperties(E=1e5, A=0.02, I=0.02 ** 3 / 12.0, rho=1.0,
                           length=length, thickness=0.02)
    return CorotationalBeam2D(n_elements=n_elements, props=props)


# ---------------------------------------------------------------------------
# DOF weights
# ---------------------------------------------------------------------------
def test_beam_declares_three_dof_kind_pattern():
    beam = _beam(n_elements=5)
    assert beam.dof_kind.shape == (beam.n_dof,)
    assert list(beam.dof_kind[:6]) == [0, 0, 1, 0, 0, 1]


def test_scale_vector_weights_lengths_and_angles():
    beam = _beam(length=32.0)
    w = dof_scale_vector(beam, ref_length=32.0, ref_angle=1.0)
    assert w.shape == (beam.n_dof,)
    tr = beam.dof_kind == 0
    rot = beam.dof_kind == 1
    assert np.allclose(w[tr], 1.0 / 32.0)
    assert np.allclose(w[rot], 1.0)


def test_scale_vector_falls_back_to_lengths_without_dof_kind():
    class Body:
        n_dof = 3

    w = dof_scale_vector(Body(), ref_length=4.0)
    assert np.allclose(w, 0.25)


def test_scale_vector_reference_angle_is_respected():
    beam = _beam()
    w = dof_scale_vector(beam, ref_length=2.0, ref_angle=57.29577951308232)
    assert np.allclose(w[beam.dof_kind == 1], 1.0 / 57.29577951308232)


# ---------------------------------------------------------------------------
# Residual measures
# ---------------------------------------------------------------------------
def _fsi(measure, n_elements=4, length=8.0, lattice=(24, 24), ref_length=None):
    beam = _beam(n_elements=n_elements, length=length)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=0.2)
    return ImplicitIBMFSI(beam, fluid, dt=1e-3, dx=1.0,
                          ref_length=length if ref_length is None else ref_length,
                          sub_tol=1e-10, max_sub_iter=5,
                          residual_measure=measure)


def _re_reference(fsi, ref_length):
    """Re-normalise an existing driver (only ``interface_residual`` is exercised)."""
    fsi.ref_length = float(ref_length)
    fsi.dof_scale = dof_scale_vector(fsi.beam, fsi.ref_length, fsi.ref_angle)
    return fsi


def test_invalid_measure_is_rejected():
    with pytest.raises(ValueError):
        _fsi("rms")


def test_legacy_measure_reproduces_pre_fix_expression_exactly():
    fsi = _fsi("legacy_isotropic")
    rng = np.random.default_rng(7)
    r = rng.standard_normal(fsi.beam.n_dof)
    res, _ = fsi.interface_residual(r)
    legacy = np.linalg.norm(r) / (fsi.ref_length * math.sqrt(len(fsi.beam.free)))
    assert res == pytest.approx(legacy, rel=1e-15)


def test_dof_scaled_measure_matches_split_norm():
    fsi = _fsi("dof_scaled")
    rng = np.random.default_rng(11)
    r = np.zeros(fsi.beam.n_dof)
    r[fsi.beam.free] = rng.standard_normal(len(fsi.beam.free))
    res, (res_t, res_rot) = fsi.interface_residual(r)
    free = fsi.beam.free
    kind = fsi.beam.dof_kind[free]
    n_t = int((kind == 0).sum())
    n_r = int((kind == 1).sum())
    expect = math.sqrt(n_t * res_t ** 2 + n_r * res_rot ** 2) / math.sqrt(n_t + n_r)
    assert res == pytest.approx(expect, rel=1e-15)


def test_rotation_tolerance_grows_with_ref_length_in_legacy_measure():
    """Legacy reports an angular residual ``ref_length`` times too small.

    For a pure rotation residual the two measures differ by exactly
    ``ref_length`` -- so a legacy ``sub_tol`` accepts an angular error of
    ``sub_tol * ref_length`` (radians), and the criterion depends on the length
    unit.
    """
    sub_tol = 1e-4
    for length in (1.0, 32.0, 72.0):
        fsi = _re_reference(_fsi("legacy_isotropic"), length)
        free = fsi.beam.free
        n = len(free)
        n_rot = int((fsi.beam.dof_kind[free] == 1).sum())
        # angular residual chosen so that the legacy measure lands exactly on tol
        m = sub_tol * length * math.sqrt(n / n_rot)
        r = np.zeros(fsi.beam.n_dof)
        r[free[fsi.beam.dof_kind[free] == 1]] = m
        res_legacy, (_, res_rot) = fsi.interface_residual(r)
        assert res_rot == pytest.approx(m, rel=1e-12)
        assert res_legacy == pytest.approx(sub_tol, rel=1e-12)
        fsi.residual_measure = "dof_scaled"
        res_scaled, _ = fsi.interface_residual(r)
        # the true (dimensionless) residual, and the factor the legacy miss
        assert res_scaled / res_legacy == pytest.approx(length, rel=1e-12)


def test_dof_scaled_residual_is_invariant_under_length_unit_change():
    """Same physical state in mm instead of m: ``dof_scaled`` unchanged."""
    rng = np.random.default_rng(3)
    length_m = 32.0
    fsi_m = _re_reference(_fsi("dof_scaled"), length_m)
    free = fsi_m.beam.free
    rot = fsi_m.beam.dof_kind[free] == 1
    r_m = np.zeros(fsi_m.beam.n_dof)
    r_m[free] = rng.standard_normal(len(free))
    # rotation-dominated residual: the failure mode of the legacy measure
    r_m[free[~rot]] *= 1e-6
    # angles are unit-independent, translations scale with the length unit
    r_mm = r_m.copy()
    r_mm[free[~rot]] *= 1000.0

    res_m, _ = fsi_m.interface_residual(r_m)
    fsi_mm = _re_reference(_fsi("dof_scaled"), length_m * 1000.0)
    res_mm, _ = fsi_mm.interface_residual(r_mm)
    assert res_mm == pytest.approx(res_m, rel=1e-15)

    fsi_m.residual_measure = "legacy_isotropic"
    fsi_mm.residual_measure = "legacy_isotropic"
    leg_m, _ = fsi_m.interface_residual(r_m)
    leg_mm, _ = fsi_mm.interface_residual(r_mm)
    # the legacy form compares radians against a length: 1000x looser in mm
    assert leg_mm / leg_m == pytest.approx(1e-3, rel=1e-3)


# ---------------------------------------------------------------------------
# Aitken Delta^2 in a mixed-DOF residual space
# ---------------------------------------------------------------------------
def _omega_sequence(gains, scale_vec, s=1.0, n_iter=25, use_scale=True,
                    omega_max=1.0):
    """Delta^2 on a diagonal 2-DOF fixed point, optionally re-unitised by ``s``.

    ``scale_vec`` are the DOF weights in the *original* units.  A length-unit
    change multiplies the length-type component of the residual by ``s`` (a
    diagonal similarity of the fixed point), so a scale-invariant estimate must
    return the same sequence for every ``s``.
    """
    S = np.diag([s, 1.0])
    G = np.diag(gains)
    Gi = np.linalg.inv(S) @ G @ S
    b = S @ np.array([1.0, 1.0])
    # the DOF weights of the re-unitised coordinates: a length reference of
    # s*ref_length is exactly what a user working in the new unit would pass
    w_s = scale_vec / np.diag(S)
    ait = AitkenRelaxation(relax_max=omega_max, relax0=1.0)
    u = np.zeros(2)
    r_prev = None
    omega = 1.0
    omega_applied = float("nan")
    out = []
    for it in range(1, n_iter + 1):
        u_new = Gi @ u + b
        r_k = u_new - u
        if np.linalg.norm(r_k) < 1e-12:
            break
        if it >= 2:
            omega = ait.omega_for(r_prev, r_k, omega_prev=omega_applied,
                                  scale=(w_s if use_scale else None))
        else:
            omega = 1.0
        omega_applied = omega
        out.append(omega)
        r_prev = r_k.copy()
        u = u + omega * r_k
    return np.array(out)


def test_aitken_omega_is_unit_invariant_only_with_dof_scale():
    # a mixed-DOF fixed point: translation gain 1.05 (strong added mass) plus a
    # weaker rotational gain.  Both gains are < 2, so the Delta^2 factor sits
    # inside the +-1 clip and the estimate is exercised on every iteration.
    gains = (1.05, 0.50)
    w = np.array([1.0 / 32.0, 1.0])          # ref_length = 32, angles in rad
    base = _omega_sequence(gains, w, s=1.0, use_scale=True)
    for s in (1e3, 1e-3):
        scaled = _omega_sequence(gains, w, s=s, use_scale=True)
        n = min(len(base), len(scaled))
        assert n > 3
        assert np.allclose(base[:n], scaled[:n], rtol=0, atol=1e-12)

    legacy = _omega_sequence(gains, w, s=1.0, use_scale=False)
    legacy_mm = _omega_sequence(gains, w, s=1e3, use_scale=False)
    n = min(len(legacy), len(legacy_mm))
    # same physics, length unit 1000x smaller: the unweighted quotient flips the
    # sign of the relaxation factor
    assert np.max(np.abs(legacy[:n] - legacy_mm[:n])) > 0.5


def test_aitken_scale_does_not_change_single_dof_behaviour():
    """A uniform weight is a no-op for the quotient (scalar / single-unit case)."""
    r_prev = np.array([0.4, -0.2, 0.1])
    r_k = np.array([0.25, -0.05, 0.02])
    a = AitkenRelaxation(relax_max=1e6, relax0=0.7)
    b = AitkenRelaxation(relax_max=1e6, relax0=0.7)
    w_a = a.omega_for(r_prev, r_k, omega_prev=0.7)
    w_b = b.omega_for(r_prev, r_k, omega_prev=0.7, scale=np.full(3, 1.0 / 32.0))
    assert w_a == pytest.approx(w_b, rel=1e-15)


# ---------------------------------------------------------------------------
# Real driver behaviour
# ---------------------------------------------------------------------------
def test_step_records_residual_split_and_default_measure():
    beam = _beam(n_elements=4, length=8.0)
    fluid = LBMFluid2D((40, 32), omega=1.8, rho0=0.2)
    fsi = ImplicitIBMFSI(beam, fluid, dt=1e-3, dx=1.0, ref_length=beam.props.length,
                         sub_tol=1e-8, max_sub_iter=10)
    assert fsi.residual_measure == "dof_scaled"
    rec = fsi.step(1)
    assert set(("residual_translation", "residual_rotation")) <= set(rec)
    assert rec["residual_translation"] >= 0.0
    assert rec["residual_rotation"] >= 0.0


def test_legacy_and_dof_scaled_agree_on_the_coupled_trajectory():
    """Short submerged run: the measure changes the certificate, not the answer."""
    tips = {}
    for measure in ("legacy_isotropic", "dof_scaled"):
        beam = _beam(n_elements=6, length=28.8)
        fluid = LBMFluid2D((40, 32), omega=1.8, rho0=0.2)
        beam.set_initial_deflection(0.5)
        fsi = ImplicitIBMFSI(beam, fluid, dt=1e-3, dx=1.0,
                             ref_length=beam.props.length, sub_tol=1e-6,
                             max_sub_iter=15, residual_measure=measure)
        tips[measure] = [fsi.step(k)["tip"][1] for k in range(1, 6)]
    a, b = tips["legacy_isotropic"], tips["dof_scaled"]
    assert np.allclose(a, b, rtol=0, atol=1e-9)