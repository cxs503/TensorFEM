"""Modal participation factors and effective modal mass.

Why this module exists
----------------------
Mode-superposition work in this repository (hull-girder vibration, deck
acceleration, whipping/springing screening) truncates the modal basis, but no
module answered the question *how many modes are enough*.  The classical answer
is the **effective modal mass**: for a unit rigid-body displacement in a chosen
direction, the fraction of the total inertia carried by each mode.  Three
identities make it a verification surface rather than a post-processing
convenience:

``completeness``
    If the retained modes span the free subspace exactly, the effective masses
    sum to ``r^T M r`` to machine precision.  Any violation is an assembly,
    constraint or normalisation bug.
``rigid-body signature``
    A free (unconstrained) model puts *all* the participation in a mode of zero
    frequency; every elastic mode must have exactly zero participation.  A mode
    of small but non-zero frequency that carries a large fraction of the mass is
    a polluted rigid-body eigenpair.
``direction symmetry``
    For a symmetric structure the antisymmetric modes have exactly zero
    participation in a symmetric translation.  This catches sign errors that
    frequency-only checks cannot see.

Independent reference
---------------------
The reference values are **not** produced by finite elements.  For a uniform
Euler--Bernoulli beam the eigenfunctions are known in closed form.  With
``xi = x/L`` and ``lam = beta L``

    phi(xi) = cosh(lam xi) - cos(lam xi) - sigma (sinh(lam xi) - sin(lam xi))

where ``sigma = (cosh lam + cos lam) / (sinh lam + sin lam)`` for clamped--free
and ``sigma = (cosh lam - cos lam) / (sinh lam - sin lam)`` for clamped--clamped.
The characteristic roots solve ``cos lam cosh lam = -1`` and ``= +1``
respectively (Blevins, *Formulas for Natural Frequency and Mode Shape*, 1979,
Table 8-1).  Evaluating that expression naively loses all significant digits at
high mode numbers because ``cosh`` reaches ``1e13`` before ``sigma`` cancels it;
this module therefore evaluates the same function in the equivalent
exponentially-scaled form

    phi(xi) = 1/2 [delta e^{lam xi} + (2 - delta) e^{-lam xi}]
              - cos(lam xi) + (1 - delta) sin(lam xi),
    delta = 1 - sigma,

with ``delta`` formed from the stable identity ``cosh lam - sinh lam = e^{-lam}``.
``delta e^{lam xi}`` stays O(1), so the closed form is evaluated to full double
precision at every mode number used here.  The effective-mass ratio of mode n is

    m_eff,n / (mu L) = (int_0^1 phi dxi)^2 / (int_0^1 phi^2 dxi),

which involves no finite-element quantity at all.

Validation of this reference: frequencies reproduce the tabulated
``tensorfem.modal`` roots and the first cantilever ratio is ``0.61307609``,
matching the value 0.6131 quoted for a uniform cantilever in the structural
dynamics literature (Craig & Kurdila, *Fundamentals of Structural Dynamics*;
Chopra, *Dynamics of Structures*).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .modal import ModalResult, solve_modes

#: Boundary-condition identifiers understood by :func:`beam_effective_mass_reference`.
_BEAM_BOUNDARY_SIGN = {"cantilever": 1.0, "clamped_clamped": -1.0}


@dataclass(frozen=True)
class ParticipationSummary:
    """Per-mode participation data of one modal solution.

    Every tensor has one entry per retained mode.  ``total_modal_mass`` is the
    discrete rigid-body inertia ``r^T M r`` of the analysed influence vector, so
    ``effective_mass_fraction`` is a partition of *that* quantity, not of a
    continuum mass: the two agree only in the mesh limit (see the module report).
    """

    angular_frequencies: torch.Tensor
    frequencies_hz: torch.Tensor
    modes: torch.Tensor
    generalized_mass: torch.Tensor
    participation_factor: torch.Tensor
    effective_mass: torch.Tensor
    total_modal_mass: float
    effective_mass_fraction: torch.Tensor
    cumulative_fraction: torch.Tensor
    mass_normalisation_error: float

    def fraction_of(self, reference_mass: float) -> torch.Tensor:
        """Effective masses divided by a *supplied* inertia.

        Passing the continuum rigid-body inertia (``mu * L`` for a beam) makes
        the ratio comparable with closed-form values; ``total_modal_mass`` is the
        discrete counterpart and differs from it at O(h) (see the report).
        """
        if reference_mass <= 0.0:
            raise ValueError("reference_mass must be positive")
        return self.effective_mass / reference_mass

    @property
    def mode_count(self) -> int:
        return int(self.effective_mass.numel())

    def mode_count_for(self, fraction: float = 0.9) -> int:
        """Smallest number of modes whose cumulative fraction reaches ``fraction``.

        Returns ``mode_count`` (plus one) when the retained basis is not rich
        enough, so a caller cannot mistake truncation for sufficiency.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        values = self.cumulative_fraction.tolist()
        for index, value in enumerate(values, start=1):
            if value >= fraction:
                return index
        return self.mode_count + 1

    def rigid_body_candidates(self, frequency_ratio: float = 1e-3) -> tuple[int, ...]:
        """Indices of modes whose frequency is negligible against the highest one.

        A genuine rigid-body mode of a constrained model does not exist, so any
        hit either comes from an unconstrained model (where it should carry the
        whole inertia) or from a polluted eigenpair (where it does not).
        """
        if frequency_ratio <= 0.0:
            raise ValueError("frequency_ratio must be positive")
        if self.mode_count == 0:
            return ()
        reference = float(self.frequencies_hz.abs().max())
        if reference == 0.0:
            return tuple(range(self.mode_count))
        scaled = (self.frequencies_hz.abs() / reference).tolist()
        return tuple(i for i, value in enumerate(scaled) if value < frequency_ratio)


def translation_influence(
    ndof: int, translational_dofs: list[int] | tuple[int, ...], *,
    dtype: torch.dtype = torch.float64, device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the unit-translation influence vector ``r`` of the given DOFs."""
    if ndof <= 0:
        raise ValueError("ndof must be positive")
    r = torch.zeros(ndof, dtype=dtype, device=device)
    indices = torch.as_tensor(list(translational_dofs), dtype=torch.long, device=device)
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= ndof):
        raise ValueError("translational DOF index out of range")
    r[indices] = 1.0
    return r


def modal_participation(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    influence: torch.Tensor,
    constrained_dofs: tuple[int, ...] | list[int] = (),
    modes: int | None = None,
) -> ParticipationSummary:
    """Compute participation factors and effective modal masses.

    Parameters
    ----------
    stiffness, mass : (ndof, ndof) tensors
        Assembled symmetric stiffness and mass matrices.
    influence : (ndof,) tensor
        Influence vector of the rigid-body motion of interest, e.g. a unit
        translation in the excitation direction.  Rotational DOFs of a beam or
        shell model take the corresponding rigid-body rotation, which is zero
        for a pure translation.
    constrained_dofs : sequence of int
        Suppressed DOFs; they are excluded from the eigenproblem and must carry
        no influence (a constrained DOF cannot move).
    modes : int, optional
        Number of modes to retain.  ``None`` means *all* free DOFs, which makes
        the completeness identity exact.

    Returns
    -------
    ParticipationSummary
    """
    if stiffness.shape != mass.shape or stiffness.ndim != 2:
        raise ValueError("stiffness and mass must be square matrices of equal size")
    ndof = int(stiffness.shape[0])
    if influence.shape != (ndof,):
        raise ValueError("influence must have shape (ndof,)")
    constrained = sorted({int(dof) for dof in constrained_dofs})
    if constrained and (constrained[0] < 0 or constrained[-1] >= ndof):
        raise ValueError("constrained DOF index out of range")
    if constrained and bool(influence[torch.as_tensor(constrained)].abs().max() > 0.0):
        raise ValueError("influence must vanish on constrained DOFs")
    n_free = ndof - len(constrained)
    if n_free <= 0:
        raise ValueError("at least one free degree of freedom is required")
    n_modes = n_free if modes is None else int(modes)
    if not 1 <= n_modes <= n_free:
        raise ValueError(f"modes must lie in 1..{n_free}")

    solution: ModalResult = solve_modes(stiffness, mass, constrained, n_modes)
    phi = solution.modes
    if phi.shape[1] == 0:
        raise ValueError("no positive eigenvalue was found")

    m_phi = mass @ phi
    generalized = torch.einsum("ij,ij->j", phi, m_phi)
    if bool(generalized.min() <= 0.0):
        raise ValueError("generalized masses must be positive")
    excitation = m_phi.T @ influence
    gamma = excitation / generalized
    effective = gamma * gamma * generalized
    total = float(influence @ (mass @ influence))
    if total <= 0.0:
        raise ValueError("influence vector has no inertia (r^T M r <= 0)")
    identity = torch.eye(phi.shape[1], dtype=phi.dtype, device=phi.device)
    normalisation_error = float((phi.T @ m_phi - identity).abs().max())
    fractions = effective / total
    return ParticipationSummary(
        angular_frequencies=solution.angular_frequencies,
        frequencies_hz=solution.frequencies_hz,
        modes=phi,
        generalized_mass=generalized,
        participation_factor=gamma,
        effective_mass=effective,
        total_modal_mass=total,
        effective_mass_fraction=fractions,
        cumulative_fraction=torch.cumsum(fractions, dim=0),
        mass_normalisation_error=normalisation_error,
    )


# ---------------------------------------------------------------------------
# Closed-form reference for a uniform beam (no finite element involved)
# ---------------------------------------------------------------------------


def _characteristic_root(boundary: str, mode: int) -> float:
    """Return ``lam = beta L`` of the given uniform-beam boundary condition.

    The characteristic equation is rewritten in the exponentially scaled form
    ``cos(lam) + s * 2 e^{-lam} / (1 + e^{-2 lam}) = 0`` with ``s = +1`` for
    clamped--free and ``s = -1`` for clamped--clamped, which is free of the
    ``cosh`` overflow that the product form suffers at ``lam > ~700``.
    """
    if boundary not in _BEAM_BOUNDARY_SIGN:
        raise ValueError(f"unsupported boundary {boundary!r}")
    if mode < 1:
        raise ValueError("mode must be >= 1")
    sign = _BEAM_BOUNDARY_SIGN[boundary]

    def residual(lam: float) -> float:
        return math.cos(lam) + sign * 2.0 * math.exp(-lam) / (1.0 + math.exp(-2.0 * lam))

    # Clamped--free roots sit just above ``(n - 1/2) pi``; clamped--clamped
    # roots just above ``(n + 1/2) pi`` (Blevins 1979, Table 8-1).
    centre = (mode - 0.5) * math.pi if boundary == "cantilever" else (mode + 0.5) * math.pi
    # Dense scan inside a half-period window: the residual is monotone there but
    # the scan keeps the bracket valid for every mode number without relying on
    # an asymptotic guess.
    samples = 64
    low = centre - 0.5
    previous_x = low
    previous_f = residual(low)
    for index in range(1, samples + 1):
        x = low + (index / samples)
        f = residual(x)
        if previous_f == 0.0:
            return previous_x
        if previous_f * f < 0.0:
            lo, hi, f_lo = previous_x, x, previous_f
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                f_mid = residual(mid)
                if f_mid == 0.0:
                    return mid
                if f_lo * f_mid < 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            return 0.5 * (lo + hi)
        previous_x, previous_f = x, f
    raise ValueError(f"no characteristic root found for {boundary} mode {mode}")


def _sigma(boundary: str, lam: float) -> float:
    """Return ``sigma`` of the closed-form shape function, computed stably."""
    # delta = 1 - sigma is formed from O(1) quantities only; the naive sigma is
    # a quotient of numbers that both grow like e^{lam}/2.
    exponential = math.exp(-lam)
    if boundary == "cantilever":
        denominator = math.cosh(lam) - exponential + math.sin(lam)
        numerator = math.sin(lam) - math.cos(lam) - exponential
    else:
        denominator = math.cosh(lam) - exponential - math.sin(lam)
        numerator = math.cos(lam) - math.sin(lam) - exponential
    if denominator == 0.0:
        raise ValueError("degenerate shape-function denominator")
    return 1.0 - numerator / denominator


def beam_shape_value(boundary: str, lam: float, xi: float) -> float:
    """Evaluate the closed-form beam eigenfunction at ``xi = x/L``."""
    if not 0.0 <= xi <= 1.0:
        raise ValueError("xi must lie in [0, 1]")
    sigma = _sigma(boundary, lam)
    delta = 1.0 - sigma
    return (
        0.5 * (delta * math.exp(lam * xi) + (2.0 - delta) * math.exp(-lam * xi))
        - math.cos(lam * xi)
        + (1.0 - delta) * math.sin(lam * xi)
    )


_GAUSS_NODES = 48


def _gauss_legendre_unit_interval(order: int = _GAUSS_NODES) -> tuple[list[float], list[float]]:
    """Gauss--Legendre nodes and weights on ``[0, 1]`` via Newton iteration."""
    nodes: list[float] = []
    weights: list[float] = []
    for index in range(1, order + 1):
        root = math.cos(math.pi * (index - 0.25) / (order + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, root
            for degree in range(2, order + 1):
                p0, p1 = p1, ((2 * degree - 1) * root * p1 - (degree - 1) * p0) / degree
            derivative = order * (root * p1 - p0) / (root * root - 1.0)
            step = p1 / derivative
            root -= step
            if abs(step) < 1e-16:
                break
        nodes.append(0.5 * (1.0 + root))
        # w_i = 2 / ((1 - x_i^2) [P'_n(x_i)]^2) on [-1, 1].
        weights.append(2.0 / ((1.0 - root * root) * derivative * derivative))
    return nodes, [0.5 * weight for weight in weights]


def beam_effective_mass_reference(boundary: str = "cantilever", modes: int = 8) -> torch.Tensor:
    """Effective-mass fractions of a uniform beam, from the closed-form modes.

    The return value is the fraction of the total beam mass ``mu L`` carried by
    modes 1..``modes``.  The sum approaches 1 as the mode count grows; for
    clamped--clamped the antisymmetric modes carry exactly zero because the
    influence vector (uniform translation) is symmetric.

    Quadrature: 48-point Gauss--Legendre per mode, which is converged to
    ``1e-15`` for the highest mode number used in the verification report.
    """
    if boundary not in _BEAM_BOUNDARY_SIGN:
        raise ValueError(f"unsupported boundary {boundary!r}")
    if modes < 1:
        raise ValueError("modes must be >= 1")
    nodes, weights = _gauss_legendre_unit_interval()
    fractions: list[float] = []
    for mode in range(1, modes + 1):
        lam = _characteristic_root(boundary, mode)
        values = [beam_shape_value(boundary, lam, xi) for xi in nodes]
        moment = sum(w * v for w, v in zip(weights, values))
        norm = sum(w * v * v for w, v in zip(weights, values))
        fractions.append(moment * moment / norm)
    return torch.tensor(fractions, dtype=torch.float64)


def beam_reference_frequencies(boundary: str = "cantilever", modes: int = 8) -> torch.Tensor:
    """Dimensionless closed-form frequencies ``omega_n / sqrt(EI/(mu L^4))``."""
    if boundary not in _BEAM_BOUNDARY_SIGN:
        raise ValueError(f"unsupported boundary {boundary!r}")
    roots = [_characteristic_root(boundary, mode) for mode in range(1, modes + 1)]
    return torch.tensor([root * root for root in roots], dtype=torch.float64)