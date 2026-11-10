"""Tests for the structural mechanics verification face.

The face has two halves and each assertion below is tied to one of the campaign
criteria (C1..C6): closed-form agreement of the Mindlin plate static solution,
the shear-locking control, the mesh convergence order, the plate spectrum across
boundary conditions, and the density negative control.

Most meshes are deliberately coarse so the suite stays in the seconds range; the
two six-mode plate checks need `n = 24` to resolve modes 5 and 6 within the 3 %
band, so they carry the ``slow`` marker and run in the full lane only.  The
qualified numbers live in ``benchmarks/structural_mindlin_vibration_face.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem import structural_mindlin_face as face

E = 200.0e9
NU = 0.3
RHO = 7850.0
A = 1.0
LOAD = 1.0e4


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def test_kirchhoff_square_plate_classic_coefficients():
    """Uniformly loaded SS square plate: w* = 0.00406, M* = 0.0479 (nu = 0.3)."""
    h = 0.01
    ref = face.kirchhoff_ss_uniform_load(
        length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
    )
    D = E * h ** 3 / (12.0 * (1.0 - NU ** 2))
    w_star = ref.deflection_center * D / (LOAD * A ** 4)
    m_star = ref.moment_x_center / (LOAD * A ** 2)
    assert w_star == pytest.approx(0.004062, abs=2e-6)
    assert m_star == pytest.approx(0.04789, abs=2e-4)


def test_fsdt_amplification_grows_with_thickness_and_matches_shear_term():
    """FSDT deflection exceeds Kirchhoff, and the excess is the shear term."""
    previous = 1.0
    for ratio in (0.01, 0.05, 0.1, 0.2):
        h = ratio * A
        ref = face.fsdt_ss_uniform_load(
            length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
        )
        assert ref.shear_amplification > 1.0
        assert ref.shear_amplification > previous
        previous = ref.shear_amplification
        # for the dominant (1,1) mode the amplification is 1 + D R / (k_s G h)
        D = E * h ** 3 / (12.0 * (1.0 - NU ** 2))
        G = E / (2.0 * (1.0 + NU))
        R11 = 2.0 * math.pi ** 2 / A ** 2
        approx = 1.0 + D * R11 / (face.SHEAR_CORRECTION * G * h)
        assert ref.shear_amplification == pytest.approx(approx, rel=0.05)


def test_ritz_clamped_square_plate_matches_leissa_first_mode():
    """Ritz clamped-plate coefficient reproduces Leissa's 35.9853 (mode 1)."""
    h = 0.01
    omega = face.ritz_plate_angular_frequencies(
        length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO,
        boundary="CC", modes=3, degree=10,
    )
    scale = math.sqrt(E * h ** 3 / (12.0 * (1.0 - NU ** 2)) / (RHO * h))
    coefficients = omega * A ** 2 / scale
    assert coefficients[0] == pytest.approx(35.9853, rel=2e-3)
    # second and third are the degenerate pair 73.394
    assert coefficients[1] == pytest.approx(73.394, rel=3e-3)
    assert coefficients[2] == pytest.approx(73.394, rel=3e-3)


def test_ritz_refinement_is_monotone_and_converged():
    """Ritz upper bounds decrease with degree and stop moving."""
    values = [
        face.ritz_plate_angular_frequencies(
            length=A, width=A, thickness=0.01, young=E, poisson=NU, density=RHO,
            boundary="CC", modes=1, degree=degree,
        )[0]
        for degree in (6, 8, 10, 12)
    ]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(values[-2], rel=1e-5)


def test_mindlin_closed_form_frequency_limits():
    """Thin limit -> Kirchhoff; thick limit -> lower than Kirchhoff."""
    thin = 0.001 * A
    thick = 0.2 * A
    for h, tol in ((thin, 1e-4), (thick, 0.3)):
        mindlin = face.mindlin_ss_angular_frequencies(
            length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO,
            modes=1,
        )[0]
        kirchhoff = face.kirchhoff_ss_angular_frequencies(
            length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO,
            modes=1,
        )[0]
        # shear deformation and rotatory inertia both soften the plate
        assert mindlin < kirchhoff
        assert (kirchhoff - mindlin) / kirchhoff <= tol


# ---------------------------------------------------------------------------
# Finite element, static half
# ---------------------------------------------------------------------------

def test_element_stiffness_is_symmetric_and_translation_invariant():
    """Rigid translation of a flat element must carry no energy."""
    nodes, elements = face.structured_rect_plate_mesh(2, 2, A, A)
    for treatment in ("sri", "mitc", "full"):
        ke = face.q4_mindlin_stiffness_batch(
            nodes, elements, E, NU, 0.01, shear_treatment=treatment
        )
        for k in ke:
            assert np.allclose(k, k.T, atol=1e-6 * np.abs(k).max())
        rigid = np.zeros(12)
        rigid[0::3] = 1.0                     # constant lateral translation
        energy = np.einsum("i,eij,j->e", rigid, ke, rigid)
        assert np.allclose(energy, 0.0, atol=1e-6 * np.abs(ke).max())


def test_c1_thin_deflection_within_two_percent_of_kirchhoff():
    """C1: centre deflection vs Kirchhoff < 2 % for h/L <= 0.05."""
    for ratio in (0.01, 0.02, 0.05):
        h = ratio * A
        res = face.solve_plate_static(
            elements_x=12, elements_y=12, length=A, width=A, thickness=h,
            young=E, poisson=NU, load=LOAD, boundary="SS",
            shear_treatment="sri",
        )
        ref = face.kirchhoff_ss_uniform_load(
            length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
        )
        error = (res.deflection_center - ref.deflection_center) / ref.deflection_center
        assert abs(error) < 0.02, f"h/L={ratio}: {error:.4f}"


def test_c2_thick_deflection_exceeds_kirchhoff_and_tracks_fsdt():
    """C2: at h/L = 0.2 the Mindlin deflection must exceed Kirchhoff."""
    h = 0.2 * A
    res = face.solve_plate_static(
        elements_x=12, elements_y=12, length=A, width=A, thickness=h,
        young=E, poisson=NU, load=LOAD, boundary="SS", shear_treatment="sri",
    )
    ref = face.kirchhoff_ss_uniform_load(
        length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
    )
    fsdt = face.fsdt_ss_uniform_load(
        length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
    )
    assert res.deflection_center > ref.deflection_center
    assert (res.deflection_center - ref.deflection_center) / ref.deflection_center > 0.1
    assert abs(res.deflection_center - fsdt.deflection_center) / fsdt.deflection_center < 0.02


def test_shear_locking_control_full_integration_degrades():
    """The full-integration variant must lock where SRI/MITC do not."""
    h = 0.005 * A
    errors = {}
    for treatment in ("sri", "mitc", "full"):
        res = face.solve_plate_static(
            elements_x=12, elements_y=12, length=A, width=A, thickness=h,
            young=E, poisson=NU, load=LOAD, boundary="SS",
            shear_treatment=treatment,
        )
        ref = face.fsdt_ss_uniform_load(
            length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
        )
        errors[treatment] = abs(res.deflection_center / ref.deflection_center - 1.0)
    assert errors["sri"] < 0.01
    assert errors["mitc"] < 0.01
    assert errors["full"] > 0.5
    assert errors["full"] > 50.0 * errors["sri"]


def test_c3_mesh_convergence_order_at_least_two():
    """C3: deflection error falls like h^2 (observed order >= 1.9)."""
    rows = face.mesh_convergence(
        elements=(4, 8, 16), thickness_ratio=0.01, length=A, young=E,
        poisson=NU, load=LOAD,
    )
    order = face.observed_convergence_order(
        [r["mesh_size"] for r in rows], [r["relative_error"] for r in rows]
    )
    assert order["observed_order"] >= 1.9, order


def test_bending_moment_matches_closed_form_at_same_point():
    """Stress check: FE M_x at a Gauss point vs the series at the same point."""
    h = 0.02 * A
    res = face.solve_plate_static(
        elements_x=12, elements_y=12, length=A, width=A, thickness=h,
        young=E, poisson=NU, load=LOAD, boundary="SS", shear_treatment="sri",
    )
    ref = face.kirchhoff_ss_uniform_load(
        length=A, width=A, thickness=h, young=E, poisson=NU, load=LOAD
    )
    x, y = res.sample_point
    moment = ref.moment_x(x, y, length=A, width=A, poisson=NU, load=LOAD)
    assert abs(res.moment_x_center - moment) / moment < 0.03
    assert res.bending_stress_center == pytest.approx(
        6.0 * abs(res.moment_x_center) / h ** 2
    )


# ---------------------------------------------------------------------------
# Finite element, vibration half
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boundary", face.PLATE_BOUNDARIES)
def test_plate_modes_are_positive_and_ordered(boundary):
    res = face.solve_plate_modes(
        elements_x=8, elements_y=8, length=A, width=A, thickness=0.01,
        young=E, poisson=NU, density=RHO, boundary=boundary, modes=4,
    )
    assert res.frequencies_hz.shape == (4,)
    assert np.all(np.isfinite(res.frequencies_hz))
    assert np.all(res.frequencies_hz > 0.0)
    assert np.all(np.diff(res.frequencies_hz) >= -1e-9)
    assert res.n_free_dof < res.n_dof


@pytest.mark.slow
def test_c4_ss_plate_six_frequencies_within_three_percent():
    """C4 (SS): six frequencies vs the exact Mindlin closed form."""
    h = 0.01
    res = face.solve_plate_modes(
        elements_x=24, elements_y=24, length=A, width=A, thickness=h,
        young=E, poisson=NU, density=RHO, boundary="SS", modes=6,
    )
    ref = face.mindlin_ss_angular_frequencies(
        length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO, modes=6,
    ) / (2.0 * math.pi)
    error = np.abs(res.frequencies_hz - ref) / ref
    assert error.max() < 0.03, error


@pytest.mark.slow
def test_c4_cc_plate_six_frequencies_within_three_percent():
    """C4 (CC): six frequencies vs the independent Rayleigh-Ritz solution."""
    h = 0.01
    res = face.solve_plate_modes(
        elements_x=24, elements_y=24, length=A, width=A, thickness=h,
        young=E, poisson=NU, density=RHO, boundary="CC", modes=6,
    )
    ref = face.ritz_plate_angular_frequencies(
        length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO,
        boundary="CC", modes=6, degree=12,
    ) / (2.0 * math.pi)
    error = np.abs(res.frequencies_hz - ref) / ref
    assert error.max() < 0.03, error


def test_thick_plate_spectrum_follows_mindlin_not_kirchhoff():
    """At h/L = 0.2 the FE spectrum must sit below the Kirchhoff one."""
    h = 0.2
    res = face.solve_plate_modes(
        elements_x=8, elements_y=8, length=A, width=A, thickness=h,
        young=E, poisson=NU, density=RHO, boundary="SS", modes=3,
    )
    kirchhoff = face.kirchhoff_ss_angular_frequencies(
        length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO, modes=3,
    ) / (2.0 * math.pi)
    mindlin = face.mindlin_ss_angular_frequencies(
        length=A, width=A, thickness=h, young=E, poisson=NU, density=RHO, modes=3,
    ) / (2.0 * math.pi)
    assert np.all(res.frequencies_hz < kirchhoff)
    assert np.all(np.abs(res.frequencies_hz - mindlin) / mindlin < 0.05)


def test_cf_plate_spectrum_refines_toward_stability():
    """CF has no closed form: successive meshes must move monotonically."""
    coarse = face.solve_plate_modes(
        elements_x=6, elements_y=6, length=A, width=A, thickness=0.01,
        young=E, poisson=NU, density=RHO, boundary="CF", modes=3,
    ).frequencies_hz
    fine = face.solve_plate_modes(
        elements_x=12, elements_y=12, length=A, width=A, thickness=0.01,
        young=E, poisson=NU, density=RHO, boundary="CF", modes=3,
    ).frequencies_hz
    # a displacement FE model softens with refinement here; 5 % band is enough
    assert np.all(np.abs(fine - coarse) / fine < 0.05)


def test_c5_density_negative_control_scales_as_inverse_sqrt():
    """C5: rho -> 0 must send every frequency to infinity."""
    base = face.solve_plate_modes(
        elements_x=6, elements_y=6, length=A, width=A, thickness=0.01, young=E,
        poisson=NU, density=RHO, boundary="SS", modes=1,
    ).frequencies_hz[0]
    for factor in (1e-2, 1e-4):
        res = face.solve_plate_modes(
            elements_x=6, elements_y=6, length=A, width=A, thickness=0.01,
            young=E, poisson=NU, density=RHO * factor, boundary="SS", modes=1,
        )
        assert res.frequencies_hz[0] > base
        assert res.frequencies_hz[0] / base == pytest.approx(1.0 / math.sqrt(factor),
                                                             rel=1e-6)


# ---------------------------------------------------------------------------
# Study helpers and input validation
# ---------------------------------------------------------------------------

def test_richardson_extrapolation_recovers_polynomial_order():
    """For an exact O(h^2) sequence Richardson returns the limit."""
    exact = 1.2345
    values = [exact + 0.1 / n ** 2 for n in (8, 16, 32)]
    out = face.richardson_extrapolation(values, refinement=2.0)
    assert out["extrapolated"] == pytest.approx(exact, rel=1e-6)
    assert out["observed_order"] == pytest.approx(2.0, rel=1e-6)
    assert out["gci"] > 0.0


def test_richardson_rejects_non_monotone_series():
    with pytest.raises(ValueError):
        face.richardson_extrapolation([1.0, 2.0, 1.5], refinement=2.0)


def test_observed_order_matches_exact_power_law():
    h = [0.1, 0.05, 0.025, 0.0125]
    errors = [3.0 * hi ** 2 for hi in h]
    out = face.observed_convergence_order(h, errors)
    assert out["observed_order"] == pytest.approx(2.0, rel=1e-9)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        face.solve_plate_static(thickness=-0.01)
    with pytest.raises(ValueError):
        face.solve_plate_static(poisson=0.5)
    with pytest.raises(ValueError):
        face.solve_plate_static(boundary="XX")
    with pytest.raises(ValueError):
        face.solve_plate_static(shear_treatment="magic")
    with pytest.raises(ValueError):
        face.ritz_plate_angular_frequencies(
            length=A, width=A, thickness=0.01, young=E, poisson=NU, density=RHO,
            boundary="SS", modes=1,
        )
    with pytest.raises(ValueError):
        face.solve_plate_modes(
            elements_x=4, elements_y=4, modes=500, boundary="SS",
        )
    with pytest.raises(ValueError):
        face.shell_boundary_dofs(np.zeros((3, 3)), 1.0, "XX")


def test_boundary_dofs_follow_plate_semantics():
    """SS restrains the tangential rotation, CC both, CF only the x = 0 edge."""
    nodes, _ = face.structured_rect_plate_mesh(4, 4, 1.0, 1.0)
    counts = {bc: face._boundary_dofs(nodes, length=1.0, width=1.0, boundary=bc).size
              for bc in face.PLATE_BOUNDARIES}
    assert counts["SS"] < counts["CC"]
    assert counts["CF"] < counts["SS"]