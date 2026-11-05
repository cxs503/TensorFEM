"""Free vibration of smeared-stiffened (orthotropic) ship panels.

Why this module exists
----------------------
The library could already solve *isotropic* panel vibration
(:mod:`tensorfem.plate_vibration`, verified against the Navier sine solution and
Leissa's clamped coefficient) and *orthotropic* stiffened-panel **statics**
(:func:`tensorfem.marine_structures.stiffened_panel_sine_benchmark`).  The
in-between case -- the natural frequencies of a real stiffened deck or side
shell, which is the entry quantity of every ship vibro-acoustic and fatigue
screening study -- had no element, no reference and no test.  The orthotropic
stiffness block now exists on the Q4 Mindlin element
(:class:`tensorfem.plate.PlateRigidity`); this module supplies the engineering
smearing, the independent closed-form reference and the mode-matching machinery
that make the result verifiable instead of merely plausible.

Formulation
-----------
The panel is idealised as an equivalent orthotropic Kirchhoff/Mindlin plate of
uniform thickness ``h`` and density ``rho`` with the four rigidities of
:class:`tensorfem.plate.PlateRigidity`.  The governing equation is

.. code-block:: text

    dx w,xxxx + 2 H w,xxyy + dy w,yyyy - (qx,x + qy,y) = q
    H = dxy + 2 dt

which is Timoshenko's orthotropic plate equation written with the *engineering*
twist curvature ``kappa_xy = theta_x,y + theta_y,x`` used by the element.

Independent reference (closed form, not produced by this code)
--------------------------------------------------------------
For a simply supported (SS1) rectangular panel the Navier modes

.. code-block:: text

    w  = W sin(alpha x) sin(beta y),  theta_x = Phi cos(alpha x) sin(beta y)
    theta_y = Psi sin(alpha x) cos(beta y),  alpha = m pi/a,  beta = n pi/b

are exact eigenfunctions of the *first-order shear deformation* (Mindlin)
orthotropic problem as well, so the eigenvalue is the smallest root of the 3x3
generalised problem

.. code-block:: text

    [ sx a^2 + sy b^2        sx a                 sy b            ] [W  ]        [rho h   W ]
    [ sx a                dx a^2 + dt b^2 + sx   (dxy + dt) a b   ] [Phi] = w^2  [rho h^3/12 Phi]
    [ sy b                (dxy + dt) a b     dy b^2 + dt a^2 + sy ] [Psi]        [rho h^3/12 Psi]

(``a`` = alpha, ``b`` = beta).  This is a stronger target than the classical
Kirchhoff value used by :mod:`tensorfem.plate_vibration`: it removes the
transverse-shear bias from the comparison instead of merely bounding it, which
matters because a stiffened panel with closely spaced longitudinals is a
*thick-ish* plate in the direction transverse to the stiffeners.  The isotropic
reduction of the same closed form is checked against the classical
``omega = D R^2 / sqrt(rho h)`` limit in the tests.

Scope limits (do not over-read the qualification)
-------------------------------------------------
* Linear, small strain, uniform thickness, no in-plane prestress, no added
  mass / fluid coupling, no damping, no stiffener pull-out.
* The panel is **smeared**: stiffener spacing must be small compared with the
  half wavelength of the mode of interest.  :func:`smearing_wavelength_warning`
  reports the mode order at which that assumption stops holding; the benchmark
  document records the numbers and does not extrapolate beyond them.
* Effective breadth of the attached plating is *not* recomputed (the smeared
  ``dx``/``dy`` are declared inputs); the sensitivity of the frequencies to that
  choice is a stated limitation, see ``docs/stiffened-panel-vibration-validation.md``.
* Stiffener **torsion** is excluded from the twist rigidity by default
  (``twist_stiffeners=False``).  The term is available but explicitly *not*
  verified by this module: no independent reference for a smeared orthotropic
  torsional constant at a specific spacing convention is used here.
* Dense algebra through :func:`tensorfem.plate_vibration.solve_plate_vibration`;
  the cost is ``O(ndof^3)`` and meshes beyond about 24x24 elements should use a
  sparse eigensolver (unchanged limitation of the underlying solver).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .plate import PlateRigidity
from .plate_vibration import PlateVibrationResult, solve_plate_vibration

#: Transverse shear correction factor, shared with the element.
SHEAR_CORRECTION = 5.0 / 6.0


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class StiffenerGeometry:
    """A set of identical, evenly spaced, flat-bar stiffeners on one face.

    ``depth`` is the stiffener depth beyond the plating, ``web_thickness`` its
    thickness and ``spacing`` the centre-to-centre spacing.  Only the flat-bar
    (rectangular) cross-section is implemented; bulb and angle profiles would
    need their own ``area``/``second_moment``/``torsion`` data, which is why the
    smearing function takes exactly the three numbers it can defend.
    """

    depth: float
    web_thickness: float
    spacing: float

    def __post_init__(self) -> None:
        for name in ("depth", "web_thickness", "spacing"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))

    @property
    def area(self) -> float:
        return self.depth * self.web_thickness

    @property
    def second_moment(self) -> float:
        """Second moment of the stiffener about its own centroid (bending axis)."""
        return self.web_thickness * self.depth ** 3 / 12.0

    @property
    def torsion_constant(self) -> float:
        """Saint-Venant torsion constant of a thin rectangle, ``J = depth*t^3/3``."""
        return self.depth * self.web_thickness ** 3 / 3.0

    def eccentric_bending(self, plate_thickness: float) -> float:
        """Second moment about the plating mid-surface, per stiffener.

        ``I_s + A_s e^2`` with ``e = (h + depth)/2`` the centroid offset.  The
        attached-plating term is carried by the plate rigidity ``D_p`` itself, so
        it is not added twice here.
        """
        plate_thickness = _positive("plate_thickness", plate_thickness)
        eccentricity = 0.5 * (plate_thickness + self.depth)
        return self.second_moment + self.area * eccentricity ** 2


def smeared_panel_rigidity(
    *, young: float, poisson: float, thickness: float,
    longitudinal: StiffenerGeometry | None = None,
    transverse: StiffenerGeometry | None = None,
    twist_stiffeners: bool = False,
    shear_correction: float = SHEAR_CORRECTION,
) -> PlateRigidity:
    """Equivalent orthotropic rigidity of a stiffened panel.

    ``longitudinal`` stiffeners run in ``x`` (they stiffen ``dx``); ``transverse``
    stiffeners run in ``y`` (they stiffen ``dy``).  Flat-bar smearing follows
    Timoshenko's equivalent-orthotropic plate:

    .. code-block:: text

        D_p   = E h^3 / (12 (1 - nu^2))
        dx    = D_p + E (I_s + A_s e^2) / s_x     (e = (h + depth)/2)
        dy    = D_p + E (I_s + A_s e^2) / s_y
        dxy   = nu sqrt(dx dy)
        dt    = G h^3 / 12  (+ G (J_x/s_x + J_y/s_y) if twist_stiffeners)
        sx=sy = kappa G h

    ``twist_stiffeners`` is off by default because the per-unit-width
    convention for smeared stiffener torsion is *not* qualified by the
    reference used here; see the module docstring.
    """
    young = _positive("young", young)
    thickness = _positive("thickness", thickness)
    if isinstance(poisson, bool) or not isinstance(poisson, (int, float)):
        raise TypeError("poisson must be a real number")
    poisson = float(poisson)
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5")
    for name, stiffener in (("longitudinal", longitudinal), ("transverse", transverse)):
        if stiffener is not None and not isinstance(stiffener, StiffenerGeometry):
            raise TypeError(f"{name} must be a StiffenerGeometry or None")
    if isinstance(twist_stiffeners, bool) is False:
        raise TypeError("twist_stiffeners must be a bool")

    shear_modulus = young / (2.0 * (1.0 + poisson))
    plate_bending = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    dx = plate_bending
    dy = plate_bending
    if longitudinal is not None:
        dx += young * longitudinal.eccentric_bending(thickness) / longitudinal.spacing
    if transverse is not None:
        dy += young * transverse.eccentric_bending(thickness) / transverse.spacing
    dt = shear_modulus * thickness ** 3 / 12.0
    if twist_stiffeners:
        for stiffener in (longitudinal, transverse):
            if stiffener is not None:
                dt += shear_modulus * stiffener.torsion_constant / stiffener.spacing
    shear = float(shear_correction) * shear_modulus * thickness
    return PlateRigidity(
        dx=dx, dy=dy, dxy=poisson * math.sqrt(dx * dy), dt=dt, sx=shear, sy=shear,
    )


def fsdt_navier_angular_frequency(
    *, rigidity: PlateRigidity, density: float, thickness: float,
    length: float, width: float, half_waves_x: int = 1, half_waves_y: int = 1,
) -> float:
    """Exact first-order-shear (Mindlin) Navier frequency of an orthotropic panel.

    Returns the smallest root of the 3x3 generalised eigenproblem documented in
    the module docstring, in rad/s.  The generalised problem is reduced with the
    (diagonal, positive) Cholesky factor of the generalised mass matrix, so the
    result is an ordinary symmetric eigenvalue problem in float64.
    """
    if not isinstance(rigidity, PlateRigidity):
        raise TypeError("rigidity must be a tensorfem.plate.PlateRigidity")
    density = _positive("density", density)
    thickness = _positive("thickness", thickness)
    length = _positive("length", length)
    width = _positive("width", width)
    half_waves_x = _count("half_waves_x", half_waves_x)
    half_waves_y = _count("half_waves_y", half_waves_y)

    alpha = half_waves_x * math.pi / length
    beta = half_waves_y * math.pi / width
    a2, b2, ab = alpha ** 2, beta ** 2, alpha * beta
    stiffness = torch.tensor(
        (
            (rigidity.sx * a2 + rigidity.sy * b2, rigidity.sx * alpha, rigidity.sy * beta),
            (rigidity.sx * alpha,
             rigidity.dx * a2 + rigidity.dt * b2 + rigidity.sx,
             (rigidity.dxy + rigidity.dt) * ab),
            (rigidity.sy * beta,
             (rigidity.dxy + rigidity.dt) * ab,
             rigidity.dy * b2 + rigidity.dt * a2 + rigidity.sy),
        ),
        dtype=torch.float64,
    )
    translatory = density * thickness
    rotary = density * thickness ** 3 / 12.0
    scale = torch.tensor(
        (1.0 / math.sqrt(translatory), 1.0 / math.sqrt(rotary), 1.0 / math.sqrt(rotary)),
        dtype=torch.float64,
    )
    standard = stiffness * scale[:, None] * scale[None, :]
    eigenvalues = torch.linalg.eigvalsh(0.5 * (standard + standard.T))
    smallest = float(eigenvalues.min())
    if not smallest > 0.0:
        raise ValueError("the orthotropic Navier eigenproblem returned a non-positive root")
    return math.sqrt(smallest)


@dataclass(frozen=True)
class NavierMode:
    """One closed-form orthotropic Navier mode."""

    half_waves_x: int
    half_waves_y: int
    angular_frequency: float

    @property
    def frequency_hz(self) -> float:
        return self.angular_frequency / (2.0 * math.pi)


def fsdt_navier_spectrum(
    *, rigidity: PlateRigidity, density: float, thickness: float,
    length: float, width: float, max_half_waves_x: int = 3,
    max_half_waves_y: int = 3,
) -> tuple[NavierMode, ...]:
    """Closed-form spectrum sorted by frequency, with ``(m, n)`` labels kept."""
    max_half_waves_x = _count("max_half_waves_x", max_half_waves_x)
    max_half_waves_y = _count("max_half_waves_y", max_half_waves_y)
    modes = [
        NavierMode(
            half_waves_x=m, half_waves_y=n,
            angular_frequency=fsdt_navier_angular_frequency(
                rigidity=rigidity, density=density, thickness=thickness,
                length=length, width=width, half_waves_x=m, half_waves_y=n,
            ),
        )
        for m in range(1, max_half_waves_x + 1)
        for n in range(1, max_half_waves_y + 1)
    ]
    return tuple(sorted(modes, key=lambda mode: mode.angular_frequency))


@dataclass(frozen=True)
class PanelModeRow:
    """One verified mode: FE frequency against the closed-form reference."""

    half_waves_x: int
    half_waves_y: int
    overlap: float
    frequency_fe_hz: float
    frequency_reference_hz: float

    @property
    def relative_error(self) -> float:
        return abs(self.frequency_fe_hz - self.frequency_reference_hz) / self.frequency_reference_hz


def solve_panel_vibration(
    elements_x: int = 8, elements_y: int = 8, *,
    rigidity: PlateRigidity, length: float, width: float,
    thickness: float, density: float, modes: int = 4,
) -> PlateVibrationResult:
    """Simply supported orthotropic panel eigen-solve (thin wrapper, fail-closed)."""
    if not isinstance(rigidity, PlateRigidity):
        raise TypeError("rigidity must be a tensorfem.plate.PlateRigidity")
    return solve_plate_vibration(
        elements_x, elements_y, length=_positive("length", length),
        width=_positive("width", width), thickness=_positive("thickness", thickness),
        density=_positive("density", density), modes=modes, boundary="simple",
        rigidity=rigidity,
    )


def identify_modes(
    result: PlateVibrationResult, *, length: float, width: float,
    max_half_waves_x: int = 3, max_half_waves_y: int = 3,
) -> tuple[tuple[int, int, float], ...]:
    """Label each computed mode by its overlap with the sine basis.

    For mode ``k`` the transverse displacement ``w`` at the mesh nodes is
    projected onto ``sin(m pi x/a) sin(n pi y/b)``; the largest normalised
    projection identifies ``(m, n)`` and the returned ``overlap`` in ``[0, 1]``
    says how far the mode is from a pure sine mode.  The projection is a
    correlation diagnostic on the nodal values, not a mass-normalised energy
    inner product, so overlaps slightly below 1 are expected on coarse meshes.
    """
    nodes = result.nodes
    length = _positive("length", length)
    width = _positive("width", width)
    max_half_waves_x = _count("max_half_waves_x", max_half_waves_x)
    max_half_waves_y = _count("max_half_waves_y", max_half_waves_y)
    x = nodes[:, 0]
    y = nodes[:, 1]
    basis = torch.zeros(
        (max_half_waves_x * max_half_waves_y, nodes.shape[0]), dtype=nodes.dtype
    )
    labels: list[tuple[int, int]] = []
    index = 0
    for m in range(1, max_half_waves_x + 1):
        for n in range(1, max_half_waves_y + 1):
            basis[index] = torch.sin(m * math.pi * x / length) * torch.sin(
                n * math.pi * y / width
            )
            labels.append((m, n))
            index += 1
    basis = basis / torch.linalg.norm(basis, dim=1, keepdim=True)
    transverse = result.modes[0::3, :]                                   # (n_nodes, modes)
    norms = torch.linalg.norm(transverse, dim=0, keepdim=True)
    if bool((norms <= 0.0).any().item()):
        raise ValueError("a computed mode has zero transverse displacement")
    transverse = transverse / norms
    projections = torch.abs(basis @ transverse)                         # (basis, modes)
    best = torch.argmax(projections, dim=0)
    return tuple(
        (labels[int(best[k])][0], labels[int(best[k])][1],
         float(projections[int(best[k]), k]))
        for k in range(transverse.shape[1])
    )


def panel_mode_table(
    result: PlateVibrationResult, *, rigidity: PlateRigidity, length: float,
    width: float, thickness: float, density: float,
    max_half_waves_x: int = 3, max_half_waves_y: int = 3,
) -> tuple[PanelModeRow, ...]:
    """Match every computed mode to its closed-form FSDT Navier frequency."""
    labels = identify_modes(
        result, length=length, width=width,
        max_half_waves_x=max_half_waves_x, max_half_waves_y=max_half_waves_y,
    )
    rows = []
    for index, (m, n, overlap) in enumerate(labels):
        reference = fsdt_navier_angular_frequency(
            rigidity=rigidity, density=density, thickness=thickness,
            length=length, width=width, half_waves_x=m, half_waves_y=n,
        ) / (2.0 * math.pi)
        frequency = float(result.frequencies_hz[index])
        rows.append(
            PanelModeRow(
                half_waves_x=m, half_waves_y=n, overlap=overlap,
                frequency_fe_hz=frequency, frequency_reference_hz=reference,
            )
        )
    return tuple(rows)


def smearing_wavelength_warning(
    *, rigidity: PlateRigidity, density: float, thickness: float,
    length: float, width: float, stiffener_spacing: float,
    max_half_waves_x: int = 6, max_half_waves_y: int = 6,
) -> tuple[NavierMode, ...]:
    """Closed-form modes whose half wavelength no longer resolves the stiffeners.

    The smeared orthotropy assumes the deformation varies slowly compared with
    the stiffener spacing.  For mode ``(m, n)`` the shortest half wavelength in
    the panel plane is ``min(a/m, b/n)``; every mode for which that is smaller
    than ``2 * stiffener_spacing`` is returned, i.e. the modes this model must
    not be used for.  Returns an empty tuple when all requested modes are safe.
    """
    stiffener_spacing = _positive("stiffener_spacing", stiffener_spacing)
    unsafe = []
    for mode in fsdt_navier_spectrum(
        rigidity=rigidity, density=density, thickness=thickness, length=length,
        width=width, max_half_waves_x=max_half_waves_x, max_half_waves_y=max_half_waves_y,
    ):
        half_wavelength = min(
            length / mode.half_waves_x, width / mode.half_waves_y
        )
        if half_wavelength < 2.0 * stiffener_spacing:
            unsafe.append(mode)
    return tuple(unsafe)