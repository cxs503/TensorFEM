"""Free vibration of shear--flexible (Timoshenko) beams.

The existing :mod:`tensorfem.modal` module covers Euler--Bernoulli beams: its
consistent mass matrix has no rotary inertia and its stiffness has no shear
deformation, so it is only valid for slender members.  For a deep member
(``L/h <~ 20``) the Euler--Bernoulli eigenfrequencies are systematically too
high, which is a documented applicability limit rather than a small
discretisation error.  This module closes that gap.

Formulation
-----------
Two-node element with nodal degrees of freedom ``(w1, theta1, w2, theta2)``.
Transverse displacement uses the interdependent (cubic) interpolation of Reddy
that absorbs the shear parameter

    Phi = 12 EI / (kappa G A L^2)

into the shape functions, while the section rotation is interpolated linearly
(``H1 = 1 - xi``, ``H2 = xi``).  The element matrices are obtained from the
strain-energy / kinetic-energy integrals

    K = int EI (dtheta/dx)^2 dx + int kappa G A (dw/dx - theta)^2 dx
    M = int rho A w^2 dx + int rho I theta^2 dx

evaluated with four-point Gauss--Legendre quadrature.  The integrands are
polynomials of degree <= 6, hence the quadrature is exact and the element is
free of shear locking (the ``Phi``-dependent shape functions remove the
parasitic shear that plagues the plain cubic element).

The two-field element (cubic ``w``, linear ``theta``) is not identical to the
classical ``Phi``-matrix/energy element that assumes the exact static field, so
no textbook stiffness coefficients are transcribed here.  Instead the tests
verify the properties that matter: symmetry, positive semi-definiteness, zero
strain energy for rigid translation and rigid rotation, the exact rigid-body
mass integrals, and O(h^2) convergence to the continuum Timoshenko reference.

Exact reference
---------------
For a simply supported (pinned--pinned) uniform Timoshenko beam, substituting
``w = W sin(kx)`` and ``theta = Theta cos(kx)`` with ``k = n pi / L`` into the
Timoshenko equations and eliminating ``Theta`` yields the Hu--Timoshenko
quadratic in ``omega^2``

    a w^4 - b w^2 + c = 0,
    a = rho I rho A /(kappa G A),
    b = rho A + rho I k^2 + rho A EI k^2 /(kappa G A),
    c = EI k^4.

The lower root is the exact bending frequency of mode ``n``.  The frequently
quoted formula

    omega_n = (n pi / L)^2 sqrt(EI/(rho A))
              / sqrt(1 + (n pi/L)^2 (I/A)(1 + E/(kappa G)))

is the same quadratic with the ``w^4`` term dropped, i.e. an approximation that
under-predicts by 0.13 % (mode 1) up to 5.0 % (mode 4) for ``L/h = 5``; it is
provided as :func:`classical_simply_supported_timoshenko_angular_frequencies`
for comparison only.  The Euler--Bernoulli result is recovered as
``I/A -> 0`` and ``E/(kappa G) -> 0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .modal import ModalResult, solve_modes

__all__ = [
    "TimoshenkoBeamSection",
    "timoshenko_shape_functions",
    "timoshenko_element_matrices",
    "assemble_timoshenko_beam",
    "simply_supported_timoshenko_modes",
    "cantilever_timoshenko_modes",
    "exact_simply_supported_timoshenko_angular_frequencies",
    "timoshenko_equation_eigenvalues",
    "classical_simply_supported_timoshenko_angular_frequencies",
    "euler_bernoulli_simply_supported_angular_frequencies",
    "shear_parameter",
    "rectangular_shear_coefficient",
]

_GAUSS_POINTS = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
_GAUSS_WEIGHTS = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)


@dataclass(frozen=True)
class TimoshenkoBeamSection:
    """Uniform rectangular Timoshenko beam section and material data.

    All quantities are in a consistent unit system (SI by default).  ``width``
    and ``height`` are the section dimensions; ``height`` is measured in the
    plane of bending used for ``I`` and for the slenderness ``L/h``.
    """

    length: float
    height: float
    width: float
    young: float
    poisson: float
    density: float
    shear_coefficient: float = 5.0 / 6.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def inertia(self) -> float:
        """Second moment of area about the bending axis."""
        return self.width * self.height**3 / 12.0

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def rho_area(self) -> float:
        return self.density * self.area

    @property
    def slenderness(self) -> float:
        return self.length / self.height

    def validate(self) -> None:
        for name in (
            "length",
            "height",
            "width",
            "young",
            "density",
            "shear_coefficient",
        ):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.poisson < 0.5:
            raise ValueError("poisson ratio must satisfy 0 <= nu < 0.5")


def rectangular_shear_coefficient(poisson: float) -> float:
    """Effective shear coefficient of a rectangular section, ``10(1+nu)/(12+11nu)``."""
    if not 0.0 <= poisson < 0.5:
        raise ValueError("poisson ratio must satisfy 0 <= nu < 0.5")
    return 10.0 * (1.0 + poisson) / (12.0 + 11.0 * poisson)


def shear_parameter(section: TimoshenkoBeamSection, element_length: float) -> float:
    """Return ``Phi = 12 EI / (kappa G A L^2)`` for one element."""
    section.validate()
    if element_length <= 0.0:
        raise ValueError("element_length must be positive")
    EI = section.young * section.inertia
    kGA = section.shear_coefficient * section.shear_modulus * section.area
    return 12.0 * EI / (kGA * element_length**2)


def timoshenko_shape_functions(
    xi: float, element_length: float, phi: float, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return shape-function data at the local coordinate ``xi`` in ``[0, 1]``.

    Returns ``(N_w, dN_w_dx, N_theta, dN_theta_dx)``.  Each vector has four
    entries ordered as ``(w1, theta1, w2, theta2)``.  ``N_w`` is the
    interdependent cubic interpolation of Reddy; ``N_theta`` is the linear
    Lagrange interpolation of the section rotation.
    """
    L = float(element_length)
    x = float(xi)
    denom = 1.0 + phi
    n_w = torch.tensor(
        [
            (1.0 - 3.0 * x**2 + 2.0 * x**3 + phi * (1.0 - x)) / denom,
            L * (x - 2.0 * x**2 + x**3 + phi * (x - x**2) / 2.0) / denom,
            (3.0 * x**2 - 2.0 * x**3 + phi * x) / denom,
            L * (-x**2 + x**3 + phi * (x**2 - x) / 2.0) / denom,
        ],
        dtype=dtype,
    )
    dn_w_dxi = torch.tensor(
        [
            (-6.0 * x + 6.0 * x**2 - phi) / denom,
            L * (1.0 - 4.0 * x + 3.0 * x**2 + phi * (0.5 - x)) / denom,
            (6.0 * x - 6.0 * x**2 + phi) / denom,
            L * (-2.0 * x + 3.0 * x**2 + phi * (x - 0.5)) / denom,
        ],
        dtype=dtype,
    )
    # Section rotation: linear Lagrange interpolation of the nodal rotations.
    n_theta = torch.tensor([0.0, 1.0 - x, 0.0, x], dtype=dtype)
    dn_theta_dx = torch.tensor([0.0, -1.0 / L, 0.0, 1.0 / L], dtype=dtype)
    return n_w, dn_w_dxi / L, n_theta, dn_theta_dx


def timoshenko_element_matrices(
    section: TimoshenkoBeamSection,
    element_length: float,
    *,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the 4x4 element stiffness and consistent mass matrices."""
    phi = shear_parameter(section, element_length)
    EI = section.young * section.inertia
    kGA = section.shear_coefficient * section.shear_modulus * section.area
    rhoA = section.rho_area
    rhoI = section.density * section.inertia
    jac = 0.5 * float(element_length)  # xi in [0, 1] mapped from [-1, 1]

    K = torch.zeros((4, 4), dtype=dtype)
    M = torch.zeros((4, 4), dtype=dtype)
    for gp, gw in zip(_GAUSS_POINTS, _GAUSS_WEIGHTS):
        xi = 0.5 * (gp + 1.0)
        n_w, dn_w_dx, n_theta, dn_theta = timoshenko_shape_functions(
            xi, element_length, phi, dtype=dtype
        )
        gamma = dn_w_dx - n_theta
        weight = gw * jac
        K += weight * (EI * torch.outer(dn_theta, dn_theta) + kGA * torch.outer(gamma, gamma))
        M += weight * (rhoA * torch.outer(n_w, n_w) + rhoI * torch.outer(n_theta, n_theta))
    return K, M


def assemble_timoshenko_beam(
    section: TimoshenkoBeamSection, elements: int, *, dtype=torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble global stiffness and mass matrices for a uniform beam."""
    section.validate()
    if elements < 1:
        raise ValueError("elements must be >= 1")
    ndof = 2 * (elements + 1)
    K = torch.zeros((ndof, ndof), dtype=dtype)
    M = torch.zeros_like(K)
    le = section.length / elements
    ke, me = timoshenko_element_matrices(section, le, dtype=dtype)
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke
        M[dofs[:, None], dofs] += me
    return K, M


def simply_supported_timoshenko_modes(
    section: TimoshenkoBeamSection, elements: int, modes: int = 4, *, dtype=torch.float64
) -> ModalResult:
    """Eigenfrequencies of a pinned--pinned Timoshenko beam (``w=0`` at both ends)."""
    K, M = assemble_timoshenko_beam(section, elements, dtype=dtype)
    constrained = (0, 2 * elements)  # transverse displacement at both supports
    return solve_modes(K, M, constrained, modes=modes)


def cantilever_timoshenko_modes(
    section: TimoshenkoBeamSection, elements: int, modes: int = 4, *, dtype=torch.float64
) -> ModalResult:
    """Eigenfrequencies of a clamped--free Timoshenko beam (``w=theta=0`` at x=0)."""
    K, M = assemble_timoshenko_beam(section, elements, dtype=dtype)
    constrained = (0, 1)
    return solve_modes(K, M, constrained, modes=modes)


def _timoshenko_modal_coefficients(section: TimoshenkoBeamSection, n: int):
    """Quadratic ``a x^2 - b x + c = 0`` in ``x = omega^2`` for mode ``n``.

    Substituting ``w = W sin(kx)``, ``theta = Theta cos(kx)`` with
    ``k = n pi / L`` into the Timoshenko equations

        rho A w_tt = kappa G A (w_xx - theta_x)
        rho I theta_tt = EI theta_xx + kappa G A (w_x - theta)

    gives, after eliminating ``Theta``, the Hu--Timoshenko quadratic
    ``x^2 (rho I rho A / (kappa G A)) - x (rho A + rho I k^2
    + rho A EI k^2/(kappa G A)) + EI k^4 = 0``.  Its lower root is the
    physically relevant bending frequency of mode ``n``.
    """
    k = n * math.pi / section.length
    kGA = section.shear_coefficient * section.shear_modulus * section.area
    rhoA = section.rho_area
    rhoI = section.density * section.inertia
    EI = section.young * section.inertia
    a = rhoI * rhoA / kGA
    b = rhoA + rhoI * k**2 + rhoA * EI * k**2 / kGA
    c = EI * k**4
    return a, b, c


def exact_simply_supported_timoshenko_angular_frequencies(
    section: TimoshenkoBeamSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """Exact simply supported Timoshenko eigenfrequencies (rad/s).

    Solves the Hu--Timoshenko quadratic in ``omega^2`` in its numerically
    stable form ``x = 2 c / (b + sqrt(b^2 - 4 a c))``.  This is the reference
    that the finite element model must converge to.  :func:`timoshenko_equation_eigenvalues`
    reproduces the same numbers from the 2x2 generalised eigenproblem, giving an
    independent check on this closed form.
    """
    section.validate()
    if modes < 1:
        raise ValueError("modes must be >= 1")
    out = []
    for n in range(1, modes + 1):
        a, b, c = _timoshenko_modal_coefficients(section, n)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            raise ValueError("negative discriminant in Timoshenko dispersion relation")
        x = 2.0 * c / (b + math.sqrt(disc))
        out.append(math.sqrt(x))
    return torch.tensor(out, dtype=dtype)


def timoshenko_equation_eigenvalues(
    section: TimoshenkoBeamSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """Independent reference: 2x2 generalised eigenproblem of the Timoshenko equations.

    For each mode ``n`` the unknowns ``(W, Theta)`` satisfy

        [[kappa G A k^2, -kappa G A k], [-kappa G A k, EI k^2 + kappa G A]]
            [W, Theta]^T = omega^2 diag(rho A, rho I) [W, Theta]^T

    which is the same physics as the closed form but is obtained without
    eliminating ``Theta``.  Used to cross-check :func:`exact_simply_supported_timoshenko_angular_frequencies`.
    """
    section.validate()
    if modes < 1:
        raise ValueError("modes must be >= 1")
    kGA = section.shear_coefficient * section.shear_modulus * section.area
    EI = section.young * section.inertia
    out = []
    for n in range(1, modes + 1):
        k = n * math.pi / section.length
        A = torch.tensor(
            [[kGA * k * k, -kGA * k], [-kGA * k, EI * k * k + kGA]], dtype=dtype
        )
        B = torch.tensor(
            [[section.rho_area, 0.0], [0.0, section.density * section.inertia]], dtype=dtype
        )
        chol = torch.linalg.cholesky(B)
        left = torch.linalg.solve_triangular(chol, A, upper=False)
        S = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        S = 0.5 * (S + S.T)
        values = torch.linalg.eigvalsh(S)
        out.append(torch.sqrt(values.min()))
    return torch.stack(out)


def classical_simply_supported_timoshenko_angular_frequencies(
    section: TimoshenkoBeamSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """The widely quoted first-order Timoshenko formula (approximate).

    ``omega_n = (n pi / L)^2 sqrt(EI/(rho A)) /
    sqrt(1 + (n pi/L)^2 (I/A)(1 + E/(kappa G)))``.

    This is the Hu--Timoshenko quadratic with the ``omega^4`` term dropped; it
    falls below the exact root by an amount that grows with mode number and
    slenderness (0.13 % / 5.0 % for modes 1 / 4 at ``L/h = 5``).  It is kept
    only to document that difference -- use
    :func:`exact_simply_supported_timoshenko_angular_frequencies` as the
    verification reference.
    """
    section.validate()
    if modes < 1:
        raise ValueError("modes must be >= 1")
    out = []
    for n in range(1, modes + 1):
        k = n * math.pi / section.length
        omega_eb = k**2 * math.sqrt(section.young * section.inertia / section.rho_area)
        alpha = 1.0 + section.young / (
            section.shear_coefficient * section.shear_modulus
        )
        out.append(omega_eb / math.sqrt(1.0 + k**2 * (section.inertia / section.area) * alpha))
    return torch.tensor(out, dtype=dtype)


def euler_bernoulli_simply_supported_angular_frequencies(
    section: TimoshenkoBeamSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """Closed-form Euler--Bernoulli reference for the same section."""
    section.validate()
    out = []
    for n in range(1, modes + 1):
        kn = n * math.pi / section.length
        out.append(kn**2 * math.sqrt(section.young * section.inertia / section.rho_area))
    return torch.tensor(out, dtype=dtype)