"""Qualification + fail-closed gates for :mod:`tensorfem.axisym_shell`.

The module adds a 2-node shell-of-revolution ring element with an optional
circular-arc meridian.  What is *qualified* here:

* C1  cylinder under internal pressure, open and closed ends: the exact state is
      pure membrane and a straight generator is represented exactly, so
      ``N_theta``, ``N_s`` and ``w`` must match the closed form to round-off;
* C2  closed sphere under internal pressure with circular-arc meridians: the
      exact membrane field ``N = pR/2`` and ``w = pR^2 (1-nu)/(2Et)`` must be
      recovered to the element's quadrature accuracy, and the error must fall
      with refinement (the case on which the faceted ``Shell4`` path failed at
      ~1006%, see ``docs/benchmark-qualification-status.md``);
* C3  open-ended pressurised cylinder clamped at both ends: bending boundary
      layers against the independent closed form of
      :mod:`tensorfem.cylindrical_pressure_reference` within 1% of the membrane
      deflection;
* C4  geometry guards: circular-arc meridians reproduce their end nodes, and an
      inverted node ordering is rejected instead of silently flipping the
      pressure side.

What is **not** qualified and is locked by a fail-closed test:

* C5  general curved / conical meridians under a non-uniform membrane state
      (truncated cone, quarter torus).  The 2-node assumed-strain formulation
      produces short-wavelength spurious response there (measured: cone
      ``N_theta`` error ~8x, quarter-torus ``N_theta`` error ~39%, both
      mesh-independent).  ``test_C5_*`` asserts the errors stay *large*, so the
      limitation cannot be quietly dropped; the mechanism is recorded in
      ``docs/axisym-shell-of-revolution-verification.md`` (open work).
"""
from __future__ import annotations

import math

import pytest

from tensorfem.axisym_shell import AxisymShell, Meridian
from tensorfem.axisym_shell_benchmarks import (clamped_cylinder, membrane_cone,
                                               membrane_cylinder, sphere_nodes,
                                               sphere_pressure, torus_pressure)

CYL = dict(radius=1.0, length=2.0, thickness=0.01, young=2.0e11, poisson=0.3,
           pressure=1.0e6, facets=8)


def test_C1_cylinder_open_membrane_exact():
    out = membrane_cylinder(closed_ends=False, **CYL)
    assert out["N_theta_max_rel_error"] < 1e-10
    assert out["N_s_max_abs_error"] / out["N_theta_exact"] < 1e-10
    assert out["w_max_rel_error"] < 1e-10
    assert abs(out["axial_reaction"]) < 1e-6 * out["N_theta_exact"]


def test_C1_cylinder_closed_membrane_exact():
    out = membrane_cylinder(closed_ends=True, **CYL)
    assert out["N_theta_max_rel_error"] < 1e-10
    # N_s = pR/2 for a closed vessel: half the hoop force.
    assert abs(out["N_s_max_abs_error"] - 0.0) / out["N_theta_exact"] < 1e-10
    assert out["w_max_rel_error"] < 1e-10
    assert out["pressure_axial_load"] == pytest.approx(math.pi * CYL["radius"] ** 2
                                                       * CYL["pressure"])
    assert abs(out["axial_reaction"]) < 1e-6 * out["pressure_axial_load"]


def test_C2_sphere_membrane_exact_and_convergent():
    kwargs = dict(radius=10.0, thickness=0.04, young=6.825e7, poisson=0.3,
                  pressure=1.0, exact_geometry=True)
    errors = []
    for facets in (4, 8, 16, 32):
        out = sphere_pressure(facets=facets, **kwargs)
        errors.append(out["N_theta_max_rel_error"])
        assert out["w_rel_error"] < 1e-4
    # analytic membrane force pR/2 = 5 N/m, sigma = pR/(2t) = 125 Pa
    fine = sphere_pressure(facets=16, **kwargs)
    assert fine["N_membrane_exact"] == pytest.approx(5.0)
    assert fine["sigma_exact"] == pytest.approx(125.0)
    assert fine["N_theta_max_rel_error"] < 1e-6
    assert fine["N_phi_max_rel_error"] < 1e-6
    assert fine["anisotropy_rel_error"] < 1e-6
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_C3_clamped_cylinder_boundary_layer():
    out = clamped_cylinder(radius=1.0, length=0.5, thickness=0.01, young=2.0e11,
                           poisson=0.3, pressure=1.0e6, facets=200)
    # 1% of the membrane deflection against the independent closed form.
    assert out["w_rel_l2_error"] < 0.01
    assert out["w_max_scaled_error"] < 0.01
    # The boundary layer really is resolved: deflection at mid-length is the
    # membrane value to 0.1%, while the clamped ends sit at zero.
    w = out["w_fe"]
    w_mem = out["w_membrane"]
    assert abs(w[0]) < 1e-9
    assert abs(w[-1]) < 1e-9
    mid = len(w) // 2
    assert abs(w[mid] / out["w_exact"][mid] - 1.0) < 5e-3
    # Shear treatment is not load-bearing at this slenderness: both integrate
    # to the same answer (recorded rather than assumed).
    full = clamped_cylinder(radius=1.0, length=0.5, thickness=0.01, young=2.0e11,
                            poisson=0.3, pressure=1.0e6, facets=200,
                            shear_integration="full")
    assert full["w_rel_l2_error"] < 0.01


def test_C4_arc_geometry_reproduces_nodes():
    mer = Meridian(0.0, 1.0, 0.5, math.cos(math.pi / 6), radius=1.0)
    assert mer.radius_at(0.0) == pytest.approx(0.0, abs=1e-12)
    assert mer.radius_at(mer.length) == pytest.approx(0.5, abs=1e-12)
    assert mer.point(0.0) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert mer.phi(0.0) == pytest.approx(0.0, abs=1e-12)
    assert Meridian(1.0, 2.0, 1.0, 0.0).radius_at(0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        Meridian(0.0, 1.0, 0.5, 0.866, radius=0.1)      # chord exceeds diameter
    with pytest.raises(ValueError):
        Meridian(1.0, 0.0, 1.0, 2.0)                    # inverted ordering
    with pytest.raises(ValueError):
        Meridian(0.0, 0.0, 0.0, 0.0)                    # coincident nodes


def test_C5_documented_limitation_general_meridians():
    """Fail-closed: conical / toroidal meridians are NOT qualified.

    The errors below are asserted to stay large on purpose.  If a future change
    fixes the formulation these assertions fail and force the qualification
    record (docs + this docstring + the benchmark JSON) to be updated instead of
    the limitation silently disappearing.
    """
    cone = membrane_cone(r_small=0.5, r_large=1.5, rise=0.5773502691896257,
                         thickness=0.01, young=2.0e11, poisson=0.3,
                         pressure=1.0e6, facets=12)
    assert cone["N_theta_max_rel_error"] > 1.0
    torus = torus_pressure(ring_radius=1.0, tube_radius=0.25, thickness=0.01,
                           young=2.0e11, poisson=0.3, pressure=1.0e6,
                           facets=16, exact_geometry=True)
    assert torus["N_theta_max_rel_error"] > 0.1
    # Mesh refinement does not repair it (the spurious response is
    # mesh-independent, so this is a formulation issue, not an under-resolution).
    torus_fine = torus_pressure(ring_radius=1.0, tube_radius=0.25, thickness=0.01,
                                young=2.0e11, poisson=0.3, pressure=1.0e6,
                                facets=64, exact_geometry=True)
    assert abs(torus_fine["N_theta_max_rel_error"]
               - torus["N_theta_max_rel_error"]) < 0.05


def test_rigid_axial_translation_is_stress_free():
    """A uniform axial translation of a cylinder must carry no strain energy."""
    model = AxisymShell([(1.0, 2.0), (1.0, 1.0), (1.0, 0.0)], young=2.0e11,
                        poisson=0.3, thickness=0.01)
    u = __import__("torch").zeros(model.ndof, dtype=__import__("torch").float64)
    for node in range(3):
        u[model.dof(node, "u")] = 0.25        # u is the -z direction here
    assert model.strain_energy(u) < 1e-12 * model.strain_energy(
        __import__("torch").ones(model.ndof, dtype=__import__("torch").float64))


def test_sphere_nodes_on_sphere():
    for r, z in sphere_nodes(3.0, 5):
        assert math.hypot(r, z) == pytest.approx(3.0, abs=1e-12)