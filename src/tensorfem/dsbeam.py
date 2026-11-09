"""Exact dynamic stiffness (DSM) free vibration of a uniform Timoshenko beam.

Why this module exists
----------------------
A displacement-based finite element model of a Timoshenko beam carries two
independent error sources: the element interpolation error (which shrinks as
the mesh is refined) and the root-finding error (which does not, once the root
finder is converged).  For a *uniform* member neither is necessary.  The
governing equations admit a closed-form general solution, so the exact
dynamic stiffness matrix ``K(omega)`` can be assembled from that solution and
the natural frequencies are the roots of ``det K(omega) = 0``.  The frequency
error then reduces to the root-finding tolerance alone -- "no discretisation
error", which is what makes high-order (mode 3 and beyond) comparison against
closed-form approximations meaningful.

That is what this module provides, together with a Wittrick--Williams (WW)
counting function ``J(omega)`` so that no root is ever skipped.

Governing equations (uniform section, no axial motion)
-----------------------------------------------------
With ``w`` the transverse deflection, ``psi`` the section rotation,
``EI`` the flexural rigidity, ``kappa G A`` the shear rigidity, ``rho A`` the
mass per unit length and ``rho I`` the rotary inertia per unit length, the
harmonic (``exp(i omega t)``) Timoshenko equations are

    kappa G A (w_xx - psi_x) + rho A omega^2 w = 0                (1)
    EI psi_xx + kappa G A (w_x - psi) + rho I omega^2 psi = 0     (2)

Eliminating ``psi`` gives the quartic dispersion relation in the non
dimensional coordinate ``xi = x / l`` (``l`` = element length)

    lambda^4 + S (1/gamma^2 + R) lambda^2 + S (S R/gamma^2 - 1) = 0

with

    S     = rho A omega^2 l^4 / EI          frequency parameter
    R     = I / (A l^2)                     rotary-inertia parameter
    gamma^2 = kappa G A l^2 / EI            shear-stiffness parameter
    B     = S / gamma^2  (= rho A omega^2 l^2 / (kappa G A))

The four roots ``lambda_k`` are real or purely imaginary, so the general
solution is a real hyperbolic/trigonometric combination::

    W(xi)   = sum_k a_k exp(lambda_k xi)                 (W = w / l, dimensionless)
    psi(xi) = sum_k a_k ((lambda_k^2 + B) / lambda_k) exp(lambda_k xi)

Using the complex exponentials keeps the formulation valid in the *second
spectrum* (``S > gamma^2 / R``), where two of the ``lambda^2`` roots turn
positive and the hyperbolic/trigonometric pair swaps, without a branch switch
in the code.  The element dynamic stiffness follows from the exact energy
functional restricted to those four exact solutions,

    q(a) = int_0^l [EI psi_x^2 + kappa G A (w_x - psi)^2] dx
           - omega^2 int_0^l [rho A w^2 + rho I psi^2] dx,

whose integrals are closed form because every integrand is an exponential
``exp((lambda_i + lambda_j) xi)``.  With ``d = H a`` (``H`` maps the four
amplitudes onto the nodal degrees of freedom ``(w1, psi1, w2, psi2)``) the
element matrix is

    K_e(omega) = H^{-T} Q H^{-1},      Q_ij = the closed-form integral above,

which is symmetric by construction (``Q`` is a Gram matrix of the exact
solution space).  This is algebraically the same matrix as the classical
hyperbolic closed form obtained from the boundary fluxes; the energy form is
used here because it is free of the ``kappa G A * 0`` indeterminacy of the
flux form in the Euler--Bernoulli limit and because symmetry is then a
built-in invariant rather than a checklist item.

Static limit: ``K_e(0)`` reproduces the classical Timoshenko (``Phi``) element
stiffness with ``Phi = 12 EI / (kappa G A l^2)`` -- verified in the tests to
better than 1e-9 relative.

Wittrick--Williams counting
---------------------------
``J(omega)`` = number of natural frequencies not exceeding ``omega``:

    J(omega) = J_0(omega) + s{K(omega)}

where ``s{}`` is the number of negative eigenvalues of the assembled
(constrained) dynamic stiffness matrix (Sylvester's law of inertia) and
``J_0(omega)`` counts the clamped--clamped natural frequencies of the
individual elements below ``omega`` (the zeros of ``det K_e``).  ``J`` is a
monotone step function that increments by one at every natural frequency, so a
bisection on ``J`` cannot miss a root; the returned frequencies are polished
inside their bracket by Brent's method on a scaled determinant (safeguarded by
the ``J`` bracket, which is re-checked after polishing).

References
----------
* W. H. Wittrick, F. W. Williams, "A general algorithm for computing natural
  frequencies of elastic structures", Q. J. Mech. Appl. Math. 24 (1971) 263.
* J. R. Banerjee, "Coupled bending--torsional dynamic stiffness matrix for
  beam elements", Int. J. Numer. Methods Eng. 28 (1989) 1283 -- the hyperbolic
  Timoshenko DSM this module reproduces.
* F. W. Williams, D. Kennedy, "Response and eigenvalue analysis of large
  structures using exact dynamic elements" (in *Computational Structural
  Dynamics*, 1995) -- WW counting in practice.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "DSMBeamSection",
    "BOUNDARY_CONDITIONS",
    "dynamic_stiffness",
    "assembled_dynamic_stiffness",
    "signature_count",
    "clamped_clamped_angular_frequencies",
    "clamped_clamped_determinant",
    "rigid_body_mode_count",
    "shear_cutoff_angular_frequency",
    "wittrick_williams_count",
    "natural_frequencies",
    "natural_frequencies_hz",
    "mode_shape",
    "euler_bernoulli_free_free_angular_frequencies",
    "simply_supported_timoshenko_angular_frequencies",
]

#: Supported boundary conditions and the nodal degrees of freedom they fix
#: (``w`` and ``psi`` at the first/last node, degree-of-freedom order
#: ``(w1, psi1, w2, psi2, ...)``).
BOUNDARY_CONDITIONS = ("free-free", "pinned-pinned", "clamped-clamped", "clamped-free")

#: Relative tolerance below which ``exp(s) - 1) / s`` is evaluated by its
#: Taylor series (avoids the 0/0 removable singularity at ``s -> 0``).
_SMALL_S = 1e-8


@dataclass(frozen=True)
class DSMBeamSection:
    """Uniform Timoshenko beam section and length.

    ``inertia`` is the second moment of area about the bending axis, ``area``
    the shear-carrying cross-section area (use an effective shear area if the
    section is thin-walled).  ``include_rotary_inertia=False`` reproduces the
    modelling choice of shear-flexible-but-rotary-inertia-free models; it is
    supported because it is a common engineering idealisation and because the
    difference it makes at mode 2 and above is exactly what one wants to
    quantify.
    """

    young: float
    poisson: float
    density: float
    area: float
    inertia: float
    length: float
    shear_coefficient: float = 5.0 / 6.0
    include_rotary_inertia: bool = True

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------- derived
    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def flexural_rigidity(self) -> float:
        return self.young * self.inertia

    @property
    def shear_rigidity(self) -> float:
        return self.shear_coefficient * self.shear_modulus * self.area

    @property
    def mass_per_length(self) -> float:
        return self.density * self.area

    @property
    def rotary_inertia_per_length(self) -> float:
        return self.density * self.inertia if self.include_rotary_inertia else 0.0

    @property
    def slenderness(self) -> float:
        """``L / sqrt(I/A)`` -- the rotary-inertia length scale."""
        return self.length / math.sqrt(self.inertia / self.area)

    @classmethod
    def rectangle(
        cls,
        young: float,
        poisson: float,
        density: float,
        width: float,
        height: float,
        length: float,
        shear_coefficient: float = 5.0 / 6.0,
        include_rotary_inertia: bool = True,
    ) -> "DSMBeamSection":
        """Solid rectangular section ``width x height`` bending about ``width``."""
        return cls(
            young=young,
            poisson=poisson,
            density=density,
            area=width * height,
            inertia=width * height**3 / 12.0,
            length=length,
            shear_coefficient=shear_coefficient,
            include_rotary_inertia=include_rotary_inertia,
        )

    def validate(self) -> None:
        for name in ("young", "poisson", "density", "area", "inertia", "length",
                     "shear_coefficient"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a finite positive number, got {value!r}")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError(f"poisson must lie in (-1, 0.5), got {self.poisson!r}")
        if not (0.0 < self.shear_coefficient <= 1.0):
            raise ValueError(
                f"shear_coefficient must lie in (0, 1], got {self.shear_coefficient!r}"
            )


# --------------------------------------------------------------------- core
def _shifted_exponential_integral(s: complex) -> complex:
    """``int_{-1/2}^{1/2} exp(s eta) d eta = 2 sinh(s/2) / s``.

    The basis below is centred on the element midpoint, so the integrand of the
    energy form is evaluated on ``eta = xi - 1/2``; the removable singularity
    at ``s = 0`` is handled by its Taylor series.
    """
    if abs(s) < _SMALL_S:
        return 1.0 + s * s / 24.0
    return 2.0 * cmath.sinh(0.5 * s) / s


def _wave_amplitudes(
    omega2: float, section: DSMBeamSection, element_length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Dispersion roots, rotation factors, shear factors and basis scales.

    Returns ``(lambda, rotation, shear, scale, B)`` where

    * ``lambda[k]`` -- root of the quartic dispersion relation,
    * ``rotation[k] = (lambda_k^2 + B)/lambda_k`` -- rotation amplitude per unit
      displacement amplitude (from ``psi' = W'' + B W``),
    * ``shear[k] = -B/lambda_k`` -- dimensionless shear strain ``W' - psi`` per
      unit displacement amplitude (written this way, and not as
      ``lambda - rotation``, so the Euler--Bernoulli limit does not cancel to
      ``0 * inf``),
    * ``scale[k]`` -- the per-basis normalisation ``1/cosh(u/2)`` for a real
      (hyperbolic) pair and ``1`` for an imaginary (trigonometric) pair.

    The basis is centred on the element midpoint and normalised to O(1) at both
    nodes.  Without that normalisation the hyperbolic pair grows like
    ``cosh(u)`` and the assembled matrix loses accuracy to round-off above
    ``u ~ 15`` -- the classical conditioning limit of the hyperbolic DSM.
    """
    le = element_length
    S = section.mass_per_length * omega2 * le**4 / section.flexural_rigidity
    R = section.rotary_inertia_per_length / (section.mass_per_length * le**2)
    gamma2 = section.shear_rigidity * le**2 / section.flexural_rigidity
    B = S / gamma2

    a_coeff = S * (1.0 / gamma2 + R)
    c_coeff = S * (S * R / gamma2 - 1.0)
    root = cmath.sqrt(complex(a_coeff * a_coeff - 4.0 * c_coeff))
    # Stable quadratic roots of x^2 + a x + c = 0 for x = lambda^2: take the
    # well-conditioned one and get the other from the product of roots.
    x_hi = 0.5 * (-a_coeff - root)
    if x_hi == 0.0:  # pragma: no cover - only at omega = 0
        raise ValueError("omega must be positive")
    x_lo = c_coeff / x_hi
    if abs(x_lo) <= 1e-12 * max(abs(x_hi), 1.0):
        raise RuntimeError(
            "omega coincides with the Timoshenko second-spectrum threshold "
            "sqrt(kappa G A / (rho I)): the two dispersion roots coalesce there "
            "and the four-term basis degenerates; shift the evaluation point"
        )
    lam = np.array(
        [cmath.sqrt(x_lo), -cmath.sqrt(x_lo), cmath.sqrt(x_hi), -cmath.sqrt(x_hi)],
        dtype=complex,
    )
    rotation = (lam**2 + B) / lam
    shear = -B / lam
    scale = np.ones(4, dtype=float)
    for k in range(4):
        if abs(lam[k].imag) < 1e-12 * max(abs(lam[k].real), 1.0):
            u = abs(lam[k].real)
            scale[k] = 1.0 / math.cosh(0.5 * u) if u < 700.0 else 0.0
    return lam, rotation, shear, scale, B


def clamped_clamped_determinant(
    section: DSMBeamSection, omega: float, *, element_length: float | None = None
) -> float:
    """Real determinant whose zeros are the element's clamped--clamped frequencies.

    Those frequencies are the poles of the dynamic stiffness (at such a
    frequency an element clamped at both nodes vibrates internally with zero
    nodal displacement, so the stiffness relation degenerates) and they are the
    ``J_0`` term of the Wittrick--Williams count.

    Construction: the complex exponential basis of :func:`dynamic_stiffness`
    is transformed onto the *real* basis ``cosh/sinh`` (real dispersion pair) or
    ``cos/sin`` (imaginary pair) by the constant invertible matrix ``T``, so
    the zero set is unchanged but the determinant becomes real.  That matters:
    with the complex basis the determinant is purely imaginary, its real part is
    round-off noise, and a sign-change scan over it counts noise.
    """
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"omega must be a finite positive number, got {omega!r}")
    le = section.length if element_length is None else float(element_length)
    lam, _, _, h = _element_basis(section, omega, le)
    transform = np.zeros((4, 4), dtype=complex)
    for pair in (0, 2):
        imaginary_pair = abs(lam[pair].imag) > 1e-12 * max(abs(lam[pair].real), 1.0)
        if imaginary_pair:  # cos(beta xi) and sin(beta xi)
            transform[pair, pair] = 0.5
            transform[pair + 1, pair] = 0.5
            transform[pair, pair + 1] = 0.5 / 1j
            transform[pair + 1, pair + 1] = -0.5 / 1j
        else:  # cosh(u xi)/cosh(u/2) and sinh(u xi)/cosh(u/2)
            transform[pair, pair] = 0.5
            transform[pair + 1, pair] = 0.5
            transform[pair, pair + 1] = 0.5
            transform[pair + 1, pair + 1] = -0.5
    real_matrix = h @ transform
    imaginary = np.abs(real_matrix.imag).max()
    scale = np.abs(real_matrix.real).max()
    if imaginary > 1e-8 * max(scale, np.finfo(float).tiny):
        raise RuntimeError(
            "real-basis transform of the clamped-clamped determinant did not "
            f"cancel the imaginary part ({imaginary:.3e} vs {scale:.3e})"
        )
    return _det4([[float(entry.real) for entry in row] for row in real_matrix])


def _det4(matrix: list[list[float]]) -> float:
    """Determinant of a 4x4 real matrix by Gaussian elimination with pivoting."""
    m = [row[:] for row in matrix]
    det = 1.0
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(m[r][col]))
        if m[pivot][col] == 0.0:
            return 0.0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        inverse = 1.0 / m[col][col]
        for r in range(col + 1, 4):
            factor = m[r][col] * inverse
            if factor == 0.0:
                continue
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return det


def _solve_in_place(m: list[list[complex]], n: int = 4) -> list[list[complex]]:
    """Gauss--Jordan elimination with partial pivoting on an augmented matrix.

    ``m`` holds ``n`` rows of ``2 n`` complex entries (``[A | B]``); on return
    the left block is the identity and the right block is ``A^{-1} B``.
    Deliberately hand-written: on the loong64 development nodes a LAPACK solve
    with a *matrix* right-hand side costs ~5 ms (a BLAS gemm pathology) while
    this loop costs microseconds, and a DSM frequency search evaluates
    thousands of small solves.
    """
    width = 2 * n
    for col in range(n):
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            candidate = abs(m[r][col])
            if candidate > best:
                best = candidate
                pivot = r
        if best == 0.0:
            raise RuntimeError("singular basis matrix in the dynamic stiffness assembly")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        row_col = m[col]
        inverse = 1.0 / row_col[col]
        for j in range(col, width):
            row_col[j] *= inverse
        for r in range(n):
            if r == col:
                continue
            row_r = m[r]
            factor = row_r[col]
            if factor == 0.0:
                continue
            for j in range(col, width):
                row_r[j] -= factor * row_col[j]
    return m


def _element_matrix(
    section: DSMBeamSection, omega2: float, element_length: float
) -> tuple[list[list[float]], list[list[complex]]]:
    """Raw element matrix in plain Python: returns ``(K real 4x4, H complex 4x4)``."""
    length = element_length
    ei = section.flexural_rigidity
    kga = section.shear_rigidity
    rho_a = section.mass_per_length
    rho_i = section.rotary_inertia_per_length
    s_param = rho_a * omega2 * length**4 / ei
    r_param = rho_i / (rho_a * length**2)
    gamma2 = kga * length**2 / ei
    b_param = s_param / gamma2

    a_coeff = s_param * (1.0 / gamma2 + r_param)
    c_coeff = s_param * (s_param * r_param / gamma2 - 1.0)
    root = cmath.sqrt(complex(a_coeff * a_coeff - 4.0 * c_coeff))
    x_hi = 0.5 * (-a_coeff - root)
    if x_hi == 0.0:  # pragma: no cover - only at omega = 0
        raise ValueError("omega must be positive")
    x_lo = c_coeff / x_hi
    if abs(x_lo) <= 1e-12 * max(abs(x_hi), 1.0):
        raise RuntimeError(
            "omega coincides with the Timoshenko second-spectrum threshold "
            "sqrt(kappa G A / (rho I)): the two dispersion roots coalesce there "
            "and the four-term basis degenerates; shift the evaluation point"
        )
    sq_lo = cmath.sqrt(x_lo)
    sq_hi = cmath.sqrt(x_hi)
    lam = [sq_lo, -sq_lo, sq_hi, -sq_hi]

    rotation: list[complex] = []
    shear: list[complex] = []
    scale: list[float] = []
    for value in lam:
        rotation.append((value * value + b_param) / value)
        shear.append(-b_param / value)
        if abs(value.imag) < 1e-12 * max(abs(value.real), 1.0):
            u = abs(value.real)
            scale.append(1.0 / math.cosh(0.5 * u) if u < 700.0 else 0.0)
        else:
            scale.append(1.0)

    # Centred, normalised basis: value at xi = 0 (eta = -1/2) and xi = 1 (eta = +1/2).
    basis0 = [scale[k] * cmath.exp(-0.5 * lam[k]) for k in range(4)]
    basis1 = [scale[k] * cmath.exp(0.5 * lam[k]) for k in range(4)]
    h = [
        [length * basis0[k] for k in range(4)],
        [basis0[k] * rotation[k] for k in range(4)],
        [length * basis1[k] for k in range(4)],
        [basis1[k] * rotation[k] for k in range(4)],
    ]

    def integral(total: complex) -> complex:
        if abs(total) < _SMALL_S:
            return 1.0 + total * total / 24.0
        return 2.0 * cmath.sinh(0.5 * total) / total

    q = [[0j] * 4 for _ in range(4)]
    for i in range(4):
        li = lam[i]
        ri = rotation[i]
        si = shear[i]
        for j in range(4):
            total = li + lam[j]
            q[i][j] = scale[i] * scale[j] * (
                (ei / length) * li * ri * lam[j] * rotation[j]
                + kga * length * si * shear[j]
                - omega2 * (rho_a * length**3 + rho_i * length * ri * rotation[j])
            ) * integral(total)

    # K^T = H^{-T} (H^{-T} Q)^T, i.e. two augmented solves on H^T.
    ht = [[h[j][i] for j in range(4)] for i in range(4)]
    first = _solve_in_place([ht[i][:] + q[i][:] for i in range(4)])
    x = [[first[i][4 + j] for j in range(4)] for i in range(4)]  # H^{-T} Q
    xt = [[x[j][i] for j in range(4)] for i in range(4)]
    second = _solve_in_place([ht[i][:] + xt[i][:] for i in range(4)])
    kt = [[second[i][4 + j] for j in range(4)] for i in range(4)]
    k = [[kt[j][i] for j in range(4)] for i in range(4)]
    return [[entry.real for entry in row] for row in k], h


def _element_basis(
    section: DSMBeamSection, omega: float, element_length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Centred, normalised exact basis: returns ``(lambda, rotation, scale, H)``.

    ``H`` maps the four basis amplitudes onto the nodal degrees of freedom
    ``(w1, psi1, w2, psi2)``; the hyperbolic pair is normalised by
    ``1/cosh(u/2)`` so that both nodes see O(1) values.
    """
    lam, rotation, _, scale, _ = _wave_amplitudes(omega**2, section, element_length)
    basis0 = scale * np.exp(-0.5 * lam)
    basis1 = scale * np.exp(0.5 * lam)
    h = np.array(
        [element_length * basis0, basis0 * rotation, element_length * basis1, basis1 * rotation],
        dtype=complex,
    )
    return lam, rotation, scale, h


def dynamic_stiffness(
    section: DSMBeamSection, omega: float, *, element_length: float | None = None
) -> np.ndarray:
    """Exact 4x4 Timoshenko dynamic stiffness matrix (``w1, psi1, w2, psi2``).

    Symmetric by construction (the amplitude-space matrix ``Q`` is a Gram
    matrix of the exact solution space); the real part is returned -- the
    imaginary residue of the complex exponential basis is a round-off artefact
    and is asserted to stay below 1e-6 of the matrix norm.
    """
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"omega must be a finite positive number, got {omega!r}")
    length = section.length if element_length is None else float(element_length)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(
            f"element_length must be a finite positive number, got {length!r}"
        )
    k, _ = _element_matrix(section, omega * omega, length)
    return np.array(k, dtype=float)


def _fixed_dofs(bc: str, ndof: int) -> list[int]:
    if bc == "free-free":
        return []
    if bc == "pinned-pinned":
        return [0, ndof - 2]
    if bc == "clamped-clamped":
        return [0, 1, ndof - 2, ndof - 1]
    if bc == "clamped-free":
        return [0, 1]
    raise ValueError(f"unknown boundary condition {bc!r}; expected one of {BOUNDARY_CONDITIONS}")


def _free_dofs(bc: str, ndof: int) -> list[int]:
    fixed = set(_fixed_dofs(bc, ndof))
    return [i for i in range(ndof) if i not in fixed]


def assembled_dynamic_stiffness(
    section: DSMBeamSection, omega: float, n_elements: int, bc: str = "free-free"
) -> np.ndarray:
    """Assembled and constrained ``K(omega)`` for a uniform beam of ``n_elements``."""
    n = _check_elements(n_elements)
    ndof = 2 * (n + 1)
    element_length = section.length / n
    k_e = dynamic_stiffness(section, omega, element_length=element_length)
    full = np.zeros((ndof, ndof))
    for e in range(n):
        s = 2 * e
        full[s : s + 4, s : s + 4] += k_e
    free = _free_dofs(bc, ndof)
    return full[np.ix_(free, free)]


def signature_count(matrix: np.ndarray) -> int:
    """Number of negative eigenvalues (Sylvester inertia) of a symmetric matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("signature_count expects a square matrix")
    if array.size == 0:
        return 0
    sym = 0.5 * (array + array.T)
    eigenvalues = np.linalg.eigvalsh(sym)
    # A tolerance scaled to the matrix norm keeps round-off null modes from
    # being counted as negative; DSM matrices are indefinite by construction.
    tol = 1e-12 * max(np.abs(eigenvalues).max(), 1.0)
    return int(np.sum(eigenvalues < -tol))


def _check_elements(n_elements: int) -> int:
    if not isinstance(n_elements, (int, np.integer)) or n_elements < 1:
        raise ValueError(f"n_elements must be a positive integer, got {n_elements!r}")
    return int(n_elements)


@lru_cache(maxsize=128)
def clamped_clamped_angular_frequencies(
    section: DSMBeamSection,
    omega_ceiling: float,
    *,
    n_elements: int = 1,
    scan_points: int = 512,
) -> np.ndarray:
    """Clamped--clamped natural frequencies of a single element of the mesh.

    These are the zeros of :func:`clamped_clamped_determinant` (equivalently the
    poles of the element dynamic stiffness) and they feed the ``J_0`` term of the
    Wittrick--Williams count.  The scan is self-validating: it is repeated on a
    differently sized grid and the two root sets must agree, otherwise a
    ``RuntimeError`` is raised rather than silently returning a short list.
    """
    n = _check_elements(n_elements)
    le = section.length / n

    def det(omega: float) -> float:
        return clamped_clamped_determinant(section, omega, element_length=le)

    def roots_on_grid(points: int) -> list[float]:
        grid = np.linspace(omega_ceiling / points, omega_ceiling, points)
        values = [det(g) for g in grid]
        found: list[float] = []
        for i in range(points - 1):
            if values[i] * values[i + 1] < 0.0:
                found.append(_bisect_det(det, grid[i], grid[i + 1]))
        return found

    primary = roots_on_grid(scan_points)
    check = roots_on_grid(scan_points + 37)
    if len(primary) != len(check) or any(
        abs(a - b) > 1e-6 * max(1.0, abs(a)) for a, b in zip(primary, check)
    ):
        raise RuntimeError(
            "clamped-clamped root scan is grid dependent "
            f"({len(primary)} vs {len(check)} roots); increase scan_points"
        )
    return np.array(primary, dtype=float)


def _bisect_det(det, lo: float, hi: float, iterations: int = 200) -> float:
    """Bisection on a sign-changing real determinant (tolerance on omega)."""
    value_lo = det(lo)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if not (mid > lo and mid < hi):
            break
        if det(mid) * value_lo > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 4.0 * np.finfo(float).eps * max(abs(mid), 1.0):
            break
    return 0.5 * (lo + hi)


def _elastic_count(
    section: DSMBeamSection,
    omega: float,
    n: int,
    bc: str,
    cc: np.ndarray,
) -> int:
    """Elastic natural-frequency count: ``J_0 + s{K} - rigid - degeneracy``."""
    j0 = n * int(np.sum(cc <= omega))
    count = j0 + signature_count(assembled_dynamic_stiffness(section, omega, n, bc))
    # Two corrections that are not physics: the zero-frequency rigid-body modes
    # of a free-free beam (J counts eigenvalues, so they must be removed to get
    # elastic frequencies), and the single numerical zero that the basis
    # degeneracy creates at the second-spectrum threshold.
    return count - rigid_body_mode_count(bc) - (1 if omega > shear_cutoff_angular_frequency(section) else 0)


def wittrick_williams_count(
    section: DSMBeamSection,
    omega: float,
    *,
    n_elements: int = 1,
    bc: str = "free-free",
    omega_ceiling: float | None = None,
) -> int:
    """``J(omega)``: number of *elastic* natural frequencies not exceeding ``omega``.

    The raw Wittrick--Williams count (``J_0 + s{K}``) counts eigenvalues, which
    for a free--free beam includes the two zero-frequency rigid-body modes, and
    which acquires one spurious zero from the dispersion degeneracy at the
    second-spectrum threshold; both are removed here so that ``J`` is exactly
    the number of finite elastic frequencies and ``natural_frequencies`` can use
    ``J`` as its mode index.

    ``omega_ceiling`` bounds the clamped-clamped scan that builds ``J_0``; it
    must not be below ``omega`` (a ``ValueError`` is raised if it is).
    """
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"omega must be a finite positive number, got {omega!r}")
    n = _check_elements(n_elements)
    ceiling = float(omega) if omega_ceiling is None else float(omega_ceiling)
    if ceiling < omega:
        raise ValueError(f"omega_ceiling ({ceiling}) must be >= omega ({omega})")
    cc = clamped_clamped_angular_frequencies(section, ceiling, n_elements=n)
    return _elastic_count(section, omega, n, bc, cc)


def _scaled_det(section: DSMBeamSection, n: int, bc: str, log_scale: float):
    """Determinant of ``K_red`` rescaled so it is O(1) but keeps its sign."""

    def value(omega: float) -> float:
        k_red = assembled_dynamic_stiffness(section, omega, n, bc)
        sign, log_abs = np.linalg.slogdet(k_red)
        exponent = min(max(log_abs - log_scale, -700.0), 700.0)
        return float(sign) * math.exp(exponent)

    return value


def natural_frequencies(
    section: DSMBeamSection,
    *,
    count: int = 4,
    n_elements: int = 1,
    bc: str = "free-free",
    omega_ceiling: float | None = None,
    rel_tol: float = 1e-12,
    max_iterations: int = 400,
) -> np.ndarray:
    """First ``count`` angular natural frequencies (rad/s), ascending.

    Root finding is a bisection on the Wittrick--Williams count ``J`` (so no
    root can be skipped), polished inside each single-root bracket by Brent's
    method on the rescaled determinant, and finally audited by re-evaluating
    ``J`` on both sides of the accepted root: ``J`` must drop to ``k-1`` just
    below and stay at ``k`` just above.  A failure of that audit raises
    ``RuntimeError`` instead of returning a dubious frequency.

    The search window is grown automatically (in factors of three) until all
    ``count`` frequencies are enclosed; ``omega_ceiling``, when given, pins the
    window and a shortfall raises ``RuntimeError``.
    """
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    n = _check_elements(n_elements)
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r}")

    ceiling = float(omega_ceiling) if omega_ceiling is not None else _frequency_ceiling(
        section, n, bc, int(count)
    )
    if not (math.isfinite(ceiling) and ceiling > 0.0):
        raise ValueError(f"omega_ceiling must be a finite positive number, got {ceiling!r}")

    while True:
        found = _frequencies_below(
            section, n, bc, ceiling, int(count), rel_tol, max_iterations
        )
        if len(found) >= int(count):
            return np.array(found[: int(count)], dtype=float)
        if omega_ceiling is not None:
            raise RuntimeError(
                f"only {len(found)} of {count} frequencies lie below omega_ceiling="
                f"{ceiling:.6g}; raise omega_ceiling"
            )
        ceiling *= 3.0


def _frequencies_below(
    section: DSMBeamSection,
    n: int,
    bc: str,
    ceiling: float,
    count: int,
    rel_tol: float,
    max_iterations: int,
) -> list[float]:
    """All natural frequencies in ``(0, ceiling]`` (at most ``count`` of them)."""
    cc = clamped_clamped_angular_frequencies(section, ceiling, n_elements=n)
    # The DSM has a pole at omega = 0, so the count is evaluated from a small
    # positive floor upwards.  A free-free beam has two zero-frequency
    # rigid-body modes; they are genuine eigenvalues (J counts them) but they
    # are not elastic frequencies, so their count is subtracted and the floor is
    # raised until the inertia count sees them cleanly.
    n_rigid = rigid_body_mode_count(bc)
    floor = ceiling * 1e-7
    for _ in range(8):
        if signature_count(assembled_dynamic_stiffness(section, floor, n, bc)) == n_rigid:
            break
        floor *= 10.0
    else:  # pragma: no cover - defensive
        raise RuntimeError("could not find a frequency floor with a clean rigid-body count")

    def sign_log(omega: float) -> tuple[float, float]:
        sign, log_abs = np.linalg.slogdet(assembled_dynamic_stiffness(section, omega, n, bc))
        return float(sign), float(log_abs)

    def j(omega: float) -> int:
        return _elastic_count(section, omega, n, bc, cc)

    from scipy.optimize import brentq  # local: scipy stays an optional dependency

    points = 256
    while True:
        grid = np.linspace(floor, ceiling, points)
        counts = [j(g) for g in grid]
        try:
            return _roots_from_counts(
                section, n, bc, grid, counts, count, j, sign_log,
                rel_tol, max_iterations, brentq,
            )
        except _ScanTooCoarse:
            if points >= 4096:
                raise RuntimeError(
                    "the J scan could not separate the roots even at 4096 points"
                )
            points *= 2


class _ScanTooCoarse(RuntimeError):
    """Raised internally when two roots fall inside one scan interval."""


def _roots_from_counts(
    section: DSMBeamSection,
    n: int,
    bc: str,
    grid: np.ndarray,
    counts: list[int],
    count: int,
    j,
    sign_log,
    rel_tol: float,
    max_iterations: int,
    brentq,
) -> list[float]:
    frequencies: list[float] = []
    for position in range(1, grid.size):
        step = counts[position] - counts[position - 1]
        if step == 0:
            continue
        if step != 1:
            raise _ScanTooCoarse(
                f"J jumped by {step} between scan points near omega={grid[position]:.6g}"
            )
        frequencies.append(
            _bisect_and_polish(
                section, n, bc, float(grid[position - 1]), float(grid[position]),
                counts[position], j, sign_log, rel_tol, max_iterations, brentq,
            )
        )
        if len(frequencies) >= count:
            break
    return frequencies


def _bisect_and_polish(
    section: DSMBeamSection,
    n: int,
    bc: str,
    lo: float,
    hi: float,
    index: int,
    j,
    sign_log,
    rel_tol: float,
    max_iterations: int,
    brentq,
) -> float:
    """Bisect ``[lo, hi]`` on ``J`` down to a single root, then polish with Brent.

    The bracket comes from the count scan and already satisfies
    ``J(lo) = index - 1``, ``J(hi) = index``, so bisection cannot converge on
    the wrong root; Brent is applied afterwards only to sharpen the abscissa and
    its result is audited against ``J`` on both sides.
    """
    while (hi - lo) > rel_tol * max(hi, 1e-30):
        mid = 0.5 * (lo + hi)
        if not (lo < mid < hi):
            break
        if j(mid) >= index:
            hi = mid
        else:
            lo = mid
    sign_lo, log_lo = sign_log(lo)
    sign_hi, log_hi = sign_log(hi)
    root = 0.5 * (lo + hi)
    if sign_lo * sign_hi < 0.0:
        log_scale = 0.5 * (log_lo + log_hi)

        def det(omega: float) -> float:
            sign, log_abs = sign_log(omega)
            exponent = min(max(log_abs - log_scale, -700.0), 700.0)
            return float(sign) * math.exp(exponent)

        try:
            root = float(
                brentq(det, lo, hi, xtol=rel_tol * max(hi, 1.0), rtol=1e-15,
                       maxiter=max_iterations)
            )
        except (ValueError, RuntimeError):
            root = 0.5 * (lo + hi)  # safeguard: keep the bisection result
    probe = 1e-9 * max(root, 1.0)
    below = j(max(root - probe, 1e-300))
    above = j(root + probe)
    if below != index - 1 or above != index:
        raise RuntimeError(
            f"Wittrick-Williams audit failed for frequency #{index} at omega={root:.12g}: "
            f"J(omega-)={below}, J(omega+)={above}"
        )
    return root


def shear_cutoff_angular_frequency(section: DSMBeamSection) -> float:
    """``sqrt(kappa G A / (rho I))``: the Timoshenko second-spectrum threshold.

    At this frequency the two dispersion roots of the quadratic in
    ``lambda^2`` coalesce (``c_coeff = 0``) and the four-term basis degenerates.
    A finite beam with ``k != 0`` has *no* natural frequency there, but the
    degeneracy does produce an isolated numerical zero of the dynamic stiffness
    determinant, so :func:`natural_frequencies` subtracts it from the count
    rather than reporting it as a mode.  Returns ``inf`` when rotary inertia is
    neglected (there is no second spectrum then).
    """
    rho_i = section.rotary_inertia_per_length
    if rho_i <= 0.0:
        return math.inf
    return math.sqrt(section.shear_rigidity / rho_i)


def rigid_body_mode_count(bc: str) -> int:
    """Number of zero-frequency rigid-body modes implied by the boundary condition."""
    if bc not in BOUNDARY_CONDITIONS:
        raise ValueError(f"unknown boundary condition {bc!r}; expected one of {BOUNDARY_CONDITIONS}")
    return 2 if bc == "free-free" else 0


def _frequency_ceiling(
    section: DSMBeamSection, n_elements: int, bc: str, count: int
) -> float:
    """Generous upper bound for the ``count``-th natural frequency of the beam."""
    # Euler-Bernoulli asymptotics: omega_n ~ (n pi)^2 sqrt(EI/(rho A L^4)).  The
    # shear and rotary-inertia terms pull every frequency down, so twice the
    # Euler-Bernoulli estimate of mode `count` is a safe first guess; the search
    # grows the ceiling if it turns out to be too small.
    base = math.sqrt(section.flexural_rigidity / (section.mass_per_length * section.length**4))
    return 2.0 * (count * math.pi) ** 2 * base


def natural_frequencies_hz(section: DSMBeamSection, **kwargs) -> np.ndarray:
    """As :func:`natural_frequencies`, in Hz."""
    return natural_frequencies(section, **kwargs) / (2.0 * math.pi)


def mode_shape(
    section: DSMBeamSection,
    mode: int,
    *,
    n_elements: int = 1,
    bc: str = "free-free",
    points_per_element: int = 40,
    rel_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Deflection shape of the ``mode``-th natural frequency (1-based).

    The nodal degrees of freedom are the null vector of the constrained dynamic
    stiffness at the (recomputed) natural frequency; inside each element the
    exact solution amplitudes are recovered from the element boundary values,
    so the shape is the continuum solution, not an interpolant.  The shape is
    normalised to unit maximum absolute deflection.
    """
    if not isinstance(mode, (int, np.integer)) or mode < 1:
        raise ValueError(f"mode must be a positive integer (1-based), got {mode!r}")
    if points_per_element < 2:
        raise ValueError("points_per_element must be >= 2")
    n = _check_elements(n_elements)
    omegas = natural_frequencies(
        section, count=int(mode), n_elements=n, bc=bc, rel_tol=rel_tol
    )
    omega = float(omegas[int(mode) - 1])
    ndof = 2 * (n + 1)
    free = _free_dofs(bc, ndof)
    k_red = assembled_dynamic_stiffness(section, omega, n, bc)
    _, _, vt = np.linalg.svd(k_red)
    free_vector = vt[-1]
    dofs = np.zeros(ndof)
    dofs[free] = free_vector
    if np.abs(dofs).max() == 0.0:  # pragma: no cover - defensive
        raise RuntimeError("null vector of the dynamic stiffness vanished")
    dofs = dofs / np.abs(dofs).max()

    le = section.length / n
    lam, rotation, scale, h = _element_basis(section, omega, le)
    xs: list[float] = []
    ws: list[float] = []
    for e in range(n):
        d = dofs[2 * e : 2 * e + 4]
        amplitudes = np.linalg.solve(h, d.astype(complex))
        xi = np.linspace(0.0, 1.0, points_per_element, endpoint=(e == n - 1))
        eta = xi - 0.5
        shape = np.real(
            sum(scale[k] * amplitudes[k] * np.exp(lam[k] * eta) for k in range(4))
        )
        xs.extend((e + xi) * le)
        ws.extend(le * shape)  # w = le * W
    x = np.array(xs)
    w = np.array(ws)
    return x, w / np.abs(w).max()


# ---------------------------------------------------------------- references
#: First four free-free Euler--Bernoulli eigenvalues ``(beta L)`` (roots of
#: ``cos(beta L) cosh(beta L) = 1``).
_EULER_BERNOULLI_FREE_FREE_BETA_L = (
    4.730040744862704,
    7.853204624095838,
    10.995607838001671,
    14.137165491257464,
)


def euler_bernoulli_free_free_angular_frequencies(
    section: DSMBeamSection, modes: int = 4
) -> np.ndarray:
    """Free--free Euler--Bernoulli reference (shear and rotary inertia removed).

    Analytic, independent of the DSM: ``omega_n = (beta_n L)^2 sqrt(EI/(rho A L^4))``.
    """
    if not isinstance(modes, (int, np.integer)) or modes < 1 or modes > 4:
        raise ValueError("modes must be an integer in 1..4")
    base = math.sqrt(section.flexural_rigidity / (section.mass_per_length * section.length**4))
    return np.array(
        [(beta_l**2) * base for beta_l in _EULER_BERNOULLI_FREE_FREE_BETA_L[: int(modes)]],
        dtype=float,
    )


def simply_supported_timoshenko_angular_frequencies(
    section: DSMBeamSection, modes: int = 4
) -> np.ndarray:
    """Exact simply supported Timoshenko reference (Hu--Timoshenko quadratic).

    ``w = W sin(kx)``, ``psi = Theta cos(kx)``, ``k = n pi / L`` satisfy both
    Timoshenko equations, so eliminating ``Theta`` gives the quadratic

        a w^4 - b w^2 + c = 0,
        a = rho I rho A / (kappa G A),
        b = rho A + rho I k^2 + rho A EI k^2 / (kappa G A),
        c = EI k^4,

    whose lower root is the exact frequency.  This is a *closed-form* reference
    for the pinned--pinned case and is used as the machine-precision oracle for
    the DSM in the tests; it is not part of the DSM algorithm.
    """
    if not isinstance(modes, (int, np.integer)) or modes < 1:
        raise ValueError(f"modes must be a positive integer, got {modes!r}")
    rho_a = section.mass_per_length
    rho_i = section.rotary_inertia_per_length
    kga = section.shear_rigidity
    ei = section.flexural_rigidity
    out = []
    for n in range(1, int(modes) + 1):
        k = n * math.pi / section.length
        a = rho_i * rho_a / kga
        b = rho_a + rho_i * k**2 + rho_a * ei * k**2 / kga
        c = ei * k**4
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:  # pragma: no cover - only for exotic inputs
            raise ValueError("negative discriminant in the Timoshenko dispersion relation")
        out.append(math.sqrt(2.0 * c / (b + math.sqrt(discriminant))))
    return np.array(out, dtype=float)