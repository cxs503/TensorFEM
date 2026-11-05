"""Exact free-vibration spectrum of the simply supported Mindlin plate.

Why this module exists
----------------------
``tensorfem.plate_vibration`` qualifies the four-node SRI Mindlin element at
``a/h = 100`` against the *Kirchhoff* Navier solution, i.e. in the thin-plate
regime, and it can only quote a first-order shear bias
(``omega^2 = omega_K^2 / (1 + D R^2 / (kappa G h))``) as an estimate of how far
that reference is from the element's own theory.  Ship panels are frequently
thick (``a/h = 10 ... 50`` for bottom/side shell in way of brackets), where that
first-order estimate is no longer a reference.  This module supplies the
*exact* first-order shear deformation (FSDT, Mindlin--Reissner) spectrum for the
simply supported rectangular plate, so the element can be checked against its
own theory instead of a thin-plate idealisation.

Formulation
-----------
Time-harmonic FSDT (``w`` transverse displacement, ``psi_x, psi_y`` rotations,
``lam = omega^2``) in the double-sine basis of a hard simply supported plate.
Writing ``R = (m pi/a)^2 + (n pi/b)^2``, ``D = E h^3/(12(1-nu^2))``,
``k_shear = kappa G h`` and the rotary inertia per unit area
``rotary = rho h^3/12``, the wave-number operator factorises (see
``scripts/derive_mindlin_sss_determinant.py``, machine-checked with sympy) into

* a two-branch flexural problem, quadratic in ``lam``::

      rotary lam^2 - ( D R + k_shear + k_shear R h^2/12 ) lam
                   + D k_shear R^2 / (rho h) = 0

  whose lower root is the bending branch and whose upper root is the
  thickness-shear branch;

* a pure-rotation branch::

      lam_rotation = ( k_shear + D (1 - nu) R / 2 ) / rotary

Both limits that the thin-plate document quotes are contained exactly: dropping
the rotary inertia (``rotary = 0``) leaves
``lam = (D R^2/(rho h)) / (1 + D R / (k_shear))``, i.e. the first-order shear
correction used by :func:`tensorfem.plate_vibration.shear_deformation_bias`, and
letting ``k_shear -> infinity`` at fixed ``rotary`` leaves
``lam = D R^2 / (rho h + rotary R)``, the Kirchhoff plate with rotary inertia.

Scope limits (do not over-read this reference)
----------------------------------------------
* Uniform thickness, isotropic, linear elastic, first-order shear deformation
  with a single shear correction factor; no higher-order thickness expansion,
  no thickness stretching, no in-plane prestress, no added mass, no damping.
* The upper (thickness-shear) and pure-rotation branches scale with
  ``1/h^3``; they are asymptotic FSDT branches that a ``C^0`` displacement
  element needs a very fine mesh to reproduce and they are *not* engineering
  panel frequencies.  They are exposed for completeness and for the branch
  ordering checks, not as validation targets.
* The shear correction factor is a modelling choice (``5/6`` for a rectangular
  section), not a fitted constant: changing it moves the thick-plate branch by
  a few per cent at ``a/h = 10``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SHEAR_CORRECTION = 5.0 / 6.0
"""Shear correction factor ``kappa`` for a rectangular cross section."""


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _poisson(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poisson must be a real number")
    value = float(value)
    if not math.isfinite(value) or not (-1.0 < value < 0.5):
        raise ValueError("poisson must satisfy -1 < poisson < 0.5")
    return value


def _shear_correction(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("shear_correction must be a real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("shear_correction must be finite and positive")
    return value


@dataclass(frozen=True)
class MindlinSSSSpectrum:
    """Closed-form spectrum of one ``(m, n)`` double-sine mode.

    All angular frequencies are ``omega`` in rad/s, the ``*_hz`` fields are
    ``f = omega / (2 pi)``.  ``shear_bias`` is ``omega_flexural / omega_K - 1``
    with ``omega_K`` the Kirchhoff (thin-plate) frequency, i.e. the exact
    replacement of the first-order estimate in
    :func:`tensorfem.plate_vibration.shear_deformation_bias`.
    """

    wave_number: float
    flexural_angular: float
    thickness_shear_angular: float
    rotation_angular: float
    kirchhoff_angular: float
    shear_bias: float
    rotary_inertia: bool

    @property
    def flexural_hz(self) -> float:
        return self.flexural_angular / (2.0 * math.pi)

    @property
    def thickness_shear_hz(self) -> float:
        return self.thickness_shear_angular / (2.0 * math.pi)

    @property
    def rotation_hz(self) -> float:
        return self.rotation_angular / (2.0 * math.pi)

    @property
    def kirchhoff_hz(self) -> float:
        return self.kirchhoff_angular / (2.0 * math.pi)


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, float]:
    """Return the two roots of ``a lam^2 + b lam + c = 0`` (``a > 0``, ``c > 0``).

    ``b`` is negative for every admissible FSDT parametrisation, so the
    numerically stable form ``q = -(b + sign(b) sqrt(disc))/2`` is used to avoid
    cancellation in the small root.
    """
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        raise ValueError("FSDT quadratic has no real roots; check the shear correction factor")
    root = math.sqrt(discriminant)
    q = -0.5 * (b + math.copysign(root, b))
    return c / q, q / a


def mindlin_sss_spectrum(
    *,
    length: float,
    width: float | None = None,
    density: float,
    young: float,
    poisson: float,
    thickness: float,
    half_waves_x: int = 1,
    half_waves_y: int = 1,
    shear_correction: float = SHEAR_CORRECTION,
    rotary_inertia: bool = True,
) -> MindlinSSSSpectrum:
    """Exact FSDT spectrum of a hard simply supported rectangular plate.

    Setting ``rotary_inertia=False`` drops the ``rho h^3/12`` terms, which
    reproduces the first-order shear correction quoted by
    :func:`tensorfem.plate_vibration.shear_deformation_bias` exactly.
    """
    length = _positive("length", length)
    width = length if width is None else _positive("width", width)
    density = _positive("density", density)
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    poisson = _poisson(poisson)
    shear_correction = _shear_correction(shear_correction)
    half_waves_x = _count("half_waves_x", half_waves_x)
    half_waves_y = _count("half_waves_y", half_waves_y)

    rigidity = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    shear_modulus = young / (2.0 * (1.0 + poisson))
    shear_stiffness = shear_correction * shear_modulus * thickness
    rotary = density * thickness**3 / 12.0
    wave_number = (half_waves_x * math.pi / length) ** 2 + (half_waves_y * math.pi / width) ** 2
    membrane_mass = density * thickness

    shear_term = shear_stiffness
    if rotary_inertia:
        shear_term += shear_stiffness * wave_number * thickness**2 / 12.0
    coefficient_b = -(rigidity * wave_number + shear_term)
    coefficient_c = rigidity * shear_stiffness * wave_number**2 / membrane_mass

    if rotary_inertia:
        lower, upper = _quadratic_roots(rotary, coefficient_b, coefficient_c)
    else:
        lower, upper = -coefficient_c / coefficient_b, math.inf
    rotation = (shear_stiffness + rigidity * (1.0 - poisson) * wave_number / 2.0) / rotary
    kirchhoff = math.sqrt(rigidity * wave_number**2 / membrane_mass)
    flexural = math.sqrt(lower)
    return MindlinSSSSpectrum(
        wave_number=wave_number,
        flexural_angular=flexural,
        thickness_shear_angular=math.sqrt(upper) if math.isfinite(upper) else math.inf,
        rotation_angular=math.sqrt(rotation),
        kirchhoff_angular=kirchhoff,
        shear_bias=flexural / kirchhoff - 1.0,
        rotary_inertia=bool(rotary_inertia),
    )


def mindlin_sss_modal_frequencies(
    modal_indices: tuple[tuple[int, int], ...],
    **kwargs,
) -> tuple[MindlinSSSSpectrum, ...]:
    """Spectra of several ``(m, n)`` modes, sorted by flexural frequency."""
    if not modal_indices:
        raise ValueError("modal_indices must not be empty")
    spectra = []
    for half_waves_x, half_waves_y in modal_indices:
        spectra.append(
            mindlin_sss_spectrum(half_waves_x=half_waves_x, half_waves_y=half_waves_y, **kwargs)
        )
    return tuple(sorted(spectra, key=lambda item: item.flexural_angular))
