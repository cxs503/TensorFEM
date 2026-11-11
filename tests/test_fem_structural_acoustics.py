"""Plate-side gates of the structure-acoustic verification face.

Companion to ``tests/test_cylinder_radiation.py``: these tests pin the *plate*
half of ``benchmarks/verified/fem_structural_acoustics`` -- FEM (Q4 Mindlin)
eigenfrequencies against the simply-supported Kirchhoff closed form, and the
radiated power of the FEM mode velocity against the closed-form angular spectrum
and the Maidanik monopole law.
"""
import math
import os

import pytest
import torch

# This machine has 128 cores; the small dense linear algebra of the acoustic
# operators runs several times *slower* when torch spawns one thread per core.
torch.set_num_threads(min(8, os.cpu_count() or 8))

from tensorfem.acoustic_radiation import maidanik_radiation_efficiency
from tensorfem.plate_acoustics import (
    PlateAcousticConfig,
    rayleigh_power,
    sine_mode_field,
    spectral_power_sine_mode,
)
from tensorfem.plate_radiation_chain import fem_modal_analysis, lattice_label

RHO = 1.225
C = 343.0
CONFIG = PlateAcousticConfig(
    length=1.0, width=0.5, thickness=0.01,
    young=210.0e9, poisson=0.3, density_plate=7850.0,
    rho_fluid=RHO, sound_speed=C,
)


@pytest.fixture(scope="module")
def analysis():
    return fem_modal_analysis(CONFIG, elements=20, modes=3)


def test_plate_fem_frequencies_match_kirchhoff(analysis):
    """First three FEM frequencies within 3 % of the Kirchhoff closed form."""
    errors = analysis.relative_frequency_error()
    assert float(errors.max()) < 0.03
    assert analysis.frequencies_hz.shape == (3,)


def test_plate_lattice_labels_are_the_analytic_order(analysis):
    assert lattice_label(analysis, 1) == (1, 1)
    assert lattice_label(analysis, 2) == (2, 1)
    assert lattice_label(analysis, 3) == (3, 1)


def test_plate_fem_velocity_radiation_matches_analytical(analysis):
    """Rayleigh integral of the FEM mode velocity vs the analytic spectrum."""
    ka = 1.0
    frequency = ka * C / (2.0 * math.pi * CONFIG.length)
    surface = analysis.surface_velocity_field(1, points=22)
    power = rayleigh_power(surface, frequency=frequency, rho=RHO, c=C, chunk=32)
    sigma_fem = power / (RHO * C * CONFIG.area * surface.mean_square_velocity)
    lattice = lattice_label(analysis, 1)
    spectral = spectral_power_sine_mode(
        CONFIG, *lattice, frequency=frequency, n_theta=36, n_phi=160
    )
    sigma_sine = spectral / (RHO * C * CONFIG.area * 0.25)
    assert sigma_fem / sigma_sine == pytest.approx(1.0, rel=0.05)


def test_plate_radiation_grows_with_frequency():
    values = []
    for ka in (0.5, 1.0, 2.0):
        frequency = ka * C / (2.0 * math.pi * CONFIG.length)
        power = spectral_power_sine_mode(
            CONFIG, 1, 1, frequency=frequency, n_theta=36, n_phi=160
        )
        values.append(power)
    assert values[0] < values[1] < values[2]


def test_maidanik_monopole_law_matches_exact_spectrum():
    """Maidanik's ``k^2`` law equals the exact spectrum within 3 % at ka = 0.5.

    The factor 2 between the two numbers is the normalisation convention: the
    classical law normalises by the modal RMS velocity, the exact spectrum here by
    the space-averaged mean square velocity.
    """
    ka = 0.5
    frequency = ka * C / (2.0 * math.pi * CONFIG.length)
    spectral = spectral_power_sine_mode(
        CONFIG, 1, 1, frequency=frequency, n_theta=36, n_phi=160
    )
    sigma_exact = spectral / (RHO * C * CONFIG.area * 0.25)
    law = maidanik_radiation_efficiency(
        length=CONFIG.length, width=CONFIG.width, frequency=frequency,
        rho=RHO, c=C, m=1, n=1,
    )
    assert law / (2.0 * sigma_exact) == pytest.approx(1.0, rel=0.03)


def test_plate_negative_control_rigid_baffle():
    """Negative control: a rigid baffled panel (``v_n = 0``) radiates nothing."""
    surface = sine_mode_field(CONFIG, 1, 1, points=16, velocity_amplitude=0.0)
    power = rayleigh_power(surface, frequency=120.0, rho=RHO, c=C)
    assert abs(power) < 1e-30