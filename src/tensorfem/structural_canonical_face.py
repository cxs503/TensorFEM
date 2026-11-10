"""Canonical structural verification face: beam bending, rectangular plates, modal, K_t.

Why this module exists
----------------------
:mod:`tensorfem.structural_verification` established the first structural ladder
in this tree -- cantilever under *uniform* load, **square** Kirchhoff plate,
cantilever modes -- and :mod:`tensorfem.kirsch_hole` established the ``K_t = 3``
stress-concentration reference with **Q4** continuum elements.  Several of the
quantities below are also already anchored under ``benchmarks/verified/``:

* cantilever tip load and tip rotation, ``fem_beam_static`` (rung L1);
* cantilever modes 1-5, ``fem_modal`` (rung L3);
* square plate modes (1,1)..(2,2), ``fem_modal`` / ``fem_plate_modal``;
* square plate uniform-pressure centre deflection, ``fem_plate_static``.

Re-running those arms here is deliberate -- a campaign needs one gate file, not
five -- but their *first* evidence lives in the anchors above, not in this
module.  What this module actually adds is the four statements that had no
evidence anywhere in the tree:

1. a **rectangular** simply supported Kirchhoff plate --
   ``w_centre = alpha q b^4/D`` with the aspect-ratio coefficient ``alpha(a/b)``
   tabulated by Timoshenko -- where every existing plate anchor solves the
   **square** ``alpha = 0.00406`` only, and the rectangular Navier series itself
   (the repository only carries the square series);
2. the ``h``-convergence order of the **deflection field** for **linear** (Q4,
   bilinear) plate elements: the existing rate evidence is for the cubic Hermite
   beam, the bicubic BFS plate element and the Q8 continuum, none of which is
   linear in ``w``;
3. the ``K_t = 3`` limit reached through a **quadratic (Q8)** element family with
   exactly circular hole edges -- an independent element family for the
   stress-concentration gate (``kirsch_hole`` is Q4 only);
4. the tip-load *invariants* (clamp reactions, Clapeyron energy identity) that no
   anchor checks, next to the tip deflection itself.

Everything below drives *existing production paths*; no new element is
introduced.  The machinery written here is (i) the rectangular Navier series,
(ii) a general ``length x width`` assembly wrapper around the existing BFS
element matrices, (iii) the uniform-pressure load of the bilinear plate, and
(iv) the gate bookkeeping.

Closed forms used as references (all independent of this tree)
--------------------------------------------------------------
* Euler--Bernoulli cantilever, tip load ``P``::

      w(x) = P x^2 (3L - x)/(6 EI),    w(L) = P L^3/(3 EI)
      theta(x) = P x (2L - x)/(2 EI),  theta(L) = P L^2/(2 EI)

  ``P`` is the magnitude of the *downward* load; deflections are positive
  downward throughout this module.

* Cantilever first natural frequency (Euler--Bernoulli)::

      f_1 = (beta_1 L)^2/(2 pi L^2) sqrt(EI/(rho A)),   beta_1 L = 1.875104...

  evaluated through :func:`tensorfem.modal.cantilever_exact_angular_frequency`.

* Simply supported rectangular plate under uniform pressure ``q`` (Navier)::

      w(x,y) = 16 q/(pi^6 D) sum_{m,n odd} sin(m pi x/a) sin(n pi y/b)
               / (m n [(m/a)^2 + (n/b)^2]^2),    D = E t^3/(12 (1 - nu^2))

  with ``alpha = w_centre D/(q b^4)`` (``b`` the short side).  Timoshenko &
  Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., table 8 tabulates
  ``alpha`` to three digits; ``alpha -> 5/384`` as ``a/b -> infinity``
  (cylindrical bending).

* Simply supported rectangular plate, first mode::

      omega_11 = pi^2 sqrt(D/(rho t)) [(1/a)^2 + (1/b)^2]

  evaluated through :func:`tensorfem.plate_vibration.kirchhoff_angular_frequency`.

* Kirsch hole: ``sigma_tt(a, pi/2) = 3 T`` (Timoshenko & Goodier, *Theory of
  Elasticity*, 3rd ed., section 39).

Gates (module constants; :func:`run_canonical_face` reports every one)
----------------------------------------------------------------------
``GATE_BEAM_TIP`` (1 %), ``GATE_PLATE_CENTRE`` (2 %), ``GATE_BEAM_FREQUENCY``
(3 %), ``GATE_PLATE_FREQUENCY`` (3 %), ``GATE_STRESS_CONCENTRATION``
(``|K_t - 3| < 0.1``) and ``GATE_LINEAR_ORDER`` (observed ``L_2`` order >= 1.9).

Honesty notes
-------------
* The cantilever with a *nodal* tip load is nodally exact for the Hermite beam
  (the exact field is a cubic and the load is applied at a node), so its tip
  error sits at round-off on every mesh.  The arm therefore states the tip number
  as "exact by construction for this load case" and leans on the rotation,
  reaction and energy identities for information.
* The plate bending references are Kirchhoff; the Q4 arm is a Mindlin element, so
  its residual also contains the ``O((t/b)^2)`` shear-deformation bias of the
  reference.  At ``t/b = 0.01`` that bias is far below the gates and is *not*
  removed.
* ``K_t`` from Q8 nodal recovery approaches ``3.0`` from above (``3.03`` on the
  finest mesh used in the audit); the residual is discretisation, not model error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from .kirsch_hole import KirschHole
from .modal import (
    cantilever_beam_modes,
    cantilever_exact_angular_frequency,
    uniform_beam_matrices,
)
from .plate import q4_mindlin_stiffness
from .plate_buckling_hermite import bfs_plate_element_matrices
from .plate_vibration import (
    BOUNDARY_SIMPLE,
    kirchhoff_angular_frequency,
    solve_plate_vibration,
    structured_rectangular_mesh,
)
from .quadratic_plane import polar_q8_mesh, solve_q8_plane
from .structural_verification import observed_order

__all__ = [
    "GATE_BEAM_TIP",
    "GATE_PLATE_CENTRE",
    "GATE_BEAM_FREQUENCY",
    "GATE_PLATE_FREQUENCY",
    "GATE_STRESS_CONCENTRATION",
    "GATE_LINEAR_ORDER",
    "STRIP_ALPHA",
    "TIMOSHENKO_ALPHA_TABLE",
    "CantileverPointLoadSolution",
    "RectangularPlateSolution",
    "MindlinPlateSolution",
    "PlateConvergenceRow",
    "StressConcentrationResult",
    "cantilever_deflection_field",
    "cantilever_slope_field",
    "cantilever_tip_deflection",
    "cantilever_tip_rotation",
    "solve_cantilever_point_load",
    "rectangular_plate_navier",
    "rectangular_plate_alpha",
    "solve_rectangular_plate_kirchhoff",
    "solve_rectangular_plate_mindlin",
    "plate_deflection_l2_error",
    "linear_element_order_study",
    "convergence_order",
    "pairwise_orders",
    "cantilever_first_frequency",
    "solve_plate_first_frequency",
    "kirsch_q8_hole_edge_hoop_stress",
    "solve_kirsch_q8_concentration",
    "run_canonical_face",
]

# --------------------------------------------------------------------------- #
# gates and literature anchors
# --------------------------------------------------------------------------- #
GATE_BEAM_TIP = 0.01
GATE_PLATE_CENTRE = 0.02
GATE_BEAM_FREQUENCY = 0.03
GATE_PLATE_FREQUENCY = 0.03
GATE_STRESS_CONCENTRATION = 0.10
GATE_LINEAR_ORDER = 1.9

#: Cylindrical-bending limit of the simply supported uniform-pressure plate:
#: ``alpha -> 5/384`` as ``a/b -> infinity``.
STRIP_ALPHA = 5.0 / 384.0

#: ``alpha`` for ``w_centre = alpha q b^4 / D`` (``b`` the short side), quoted to
#: the three digits printed in Timoshenko & Woinowsky-Krieger, *Theory of Plates
#: and Shells*, 2nd ed., table 8.  ``math.inf`` is the strip limit.
TIMOSHENKO_ALPHA_TABLE: dict[float, float] = {
    1.0: 0.00406,
    1.1: 0.00485,
    1.2: 0.00564,
    1.3: 0.00638,
    1.4: 0.00705,
    1.5: 0.00772,
    1.75: 0.00907,
    2.0: 0.01013,
    2.5: 0.01146,
    3.0: 0.01223,
    4.0: 0.01282,
    5.0: 0.01297,
    math.inf: 0.01302,
}


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _relative_error(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / abs(float(reference))


# --------------------------------------------------------------------------- #
# 1. Euler--Bernoulli cantilever under a concentrated tip load
# --------------------------------------------------------------------------- #
def cantilever_tip_deflection(length: float, bending_stiffness: float, load: float) -> float:
    """``w(L) = P L^3/(3 EI)``, positive downward."""
    L = _positive("length", length)
    EI = _positive("bending_stiffness", bending_stiffness)
    P = _positive("load", load)
    return P * L**3 / (3.0 * EI)


def cantilever_tip_rotation(length: float, bending_stiffness: float, load: float) -> float:
    """``theta(L) = P L^2/(2 EI)`` (same sign convention as the deflection)."""
    L = _positive("length", length)
    EI = _positive("bending_stiffness", bending_stiffness)
    P = _positive("load", load)
    return P * L**2 / (2.0 * EI)


def cantilever_deflection_field(
    x: np.ndarray | float, *, length: float, bending_stiffness: float, load: float
) -> np.ndarray:
    """Exact ``w(x) = P x^2 (3L - x)/(6 EI)``, positive downward."""
    L = _positive("length", length)
    EI = _positive("bending_stiffness", bending_stiffness)
    P = _positive("load", load)
    xs = np.asarray(x, dtype=float)
    return P * xs**2 * (3.0 * L - xs) / (6.0 * EI)


def cantilever_slope_field(
    x: np.ndarray | float, *, length: float, bending_stiffness: float, load: float
) -> np.ndarray:
    """Exact ``dw/dx = P x (2L - x)/(2 EI)``."""
    L = _positive("length", length)
    EI = _positive("bending_stiffness", bending_stiffness)
    P = _positive("load", load)
    xs = np.asarray(x, dtype=float)
    return P * xs * (2.0 * L - xs) / (2.0 * EI)


@dataclass(frozen=True)
class CantileverPointLoadSolution:
    """Tip-loaded Hermite cantilever: nodal solution plus its invariants.

    ``transverse``/``rotation`` are nodal dofs with the deflection measured
    **positive downward**, matching :func:`cantilever_deflection_field`.
    """

    length: float
    bending_stiffness: float
    load: float
    elements: int
    transverse: np.ndarray  # (elements + 1,) nodal w, positive downward
    rotation: np.ndarray    # (elements + 1,) nodal dw/dx
    base_reaction: float    # force reaction at the clamp, positive upward
    base_moment: float      # moment reaction at the clamp
    strain_energy: float

    @property
    def exact_tip_deflection(self) -> float:
        return cantilever_tip_deflection(self.length, self.bending_stiffness, self.load)

    @property
    def tip_deflection(self) -> float:
        return float(self.transverse[-1])

    @property
    def tip_deflection_error(self) -> float:
        return _relative_error(self.tip_deflection, self.exact_tip_deflection)

    @property
    def exact_tip_rotation(self) -> float:
        return cantilever_tip_rotation(self.length, self.bending_stiffness, self.load)

    @property
    def tip_rotation(self) -> float:
        return float(self.rotation[-1])

    @property
    def tip_rotation_error(self) -> float:
        return _relative_error(self.tip_rotation, self.exact_tip_rotation)

    @property
    def exact_strain_energy(self) -> float:
        """Clapeyron: ``U = P w(L)/2 = P^2 L^3/(6 EI)``."""
        return 0.5 * self.load * self.exact_tip_deflection

    @property
    def energy_relative_error(self) -> float:
        return _relative_error(self.strain_energy, self.exact_strain_energy)

    def nodes(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.elements + 1)

    def exact_transverse(self) -> np.ndarray:
        return cantilever_deflection_field(
            self.nodes(),
            length=self.length,
            bending_stiffness=self.bending_stiffness,
            load=self.load,
        )


def solve_cantilever_point_load(
    *,
    length: float = 1.0,
    bending_stiffness: float = 2000.0,
    load: float = 10.0,
    elements: int = 8,
) -> CantileverPointLoadSolution:
    """Solve the tip-loaded cantilever with the production Hermite beam.

    The stiffness matrix comes from :func:`tensorfem.modal.uniform_beam_matrices`
    (the assembly used by the modal path), so the element and its formulation are
    the repository's.  Only the concentrated tip load and the invariant
    bookkeeping are added here.
    """
    L = _positive("length", length)
    EI = _positive("bending_stiffness", bending_stiffness)
    P = _positive("load", load)
    n = _count("elements", elements)

    stiffness, _mass = uniform_beam_matrices(L, n, EI, 1.0)
    ndof = 2 * (n + 1)
    load_vector = torch.zeros(ndof, dtype=stiffness.dtype)
    load_vector[2 * n] = -P  # downward nodal force on the tip transverse dof

    fixed = [0, 1]  # clamped at x = 0: w = 0 and w_,x = 0
    free = torch.tensor([d for d in range(ndof) if d not in fixed], dtype=torch.long)
    solution = torch.zeros(ndof, dtype=stiffness.dtype)
    solution[free] = torch.linalg.solve(stiffness[free[:, None], free], load_vector[free])
    reaction = stiffness @ solution - load_vector
    u = solution.numpy()

    return CantileverPointLoadSolution(
        length=L,
        bending_stiffness=EI,
        load=P,
        elements=n,
        transverse=-u[0::2],
        rotation=-u[1::2],
        base_reaction=float(reaction[0]),
        base_moment=float(reaction[1]),
        strain_energy=float(0.5 * solution @ stiffness @ solution),
    )


# --------------------------------------------------------------------------- #
# 2. Simply supported rectangular Kirchhoff plate under uniform pressure
# --------------------------------------------------------------------------- #
def rectangular_plate_navier(
    points: np.ndarray,
    *,
    length: float = 2.0,
    width: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    terms: int = 81,
) -> np.ndarray:
    """Exact Navier deflection of the simply supported rectangular plate.

    ``points`` has shape ``(N, 2)`` (or ``(2,)``).  The double series carries the
    alternating signs of ``sin(m pi x/a) sin(n pi y/b)``, so it converges at every
    interior point; ``terms = 81`` reproduces the centre deflection to better than
    ``1e-6`` relative for aspect ratios up to ``a/b = 5`` and ``terms = 201`` is
    used by the audit script.
    """
    a = _positive("length", length)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    q = _positive("pressure", pressure)
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    terms = _count("terms", terms)
    if terms % 2 == 0:
        raise ValueError("terms must be odd")

    rigidity = young * t**3 / (12.0 * (1.0 - poisson**2))
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    odd = np.arange(1, terms + 1, 2, dtype=float)
    m, n = np.meshgrid(odd, odd, indexing="ij")
    sin_m = np.sin(m[None] * math.pi * pts[:, 0][:, None, None] / a)
    sin_n = np.sin(n[None] * math.pi * pts[:, 1][:, None, None] / b)
    denominator = m * n * ((m / a) ** 2 + (n / b) ** 2) ** 2
    series = (sin_m * sin_n / denominator[None]).sum(axis=(1, 2))
    return 16.0 * q / (math.pi**6 * rigidity) * series


def rectangular_plate_alpha(aspect_ratio: float, *, terms: int = 201) -> float:
    """Series value of ``alpha = w_centre D/(q b^4)`` (``b`` the short side)."""
    ratio = _positive("aspect_ratio", aspect_ratio)
    if ratio < 1.0:
        raise ValueError("aspect_ratio must be >= 1 (long side over short side)")
    centre = rectangular_plate_navier(
        np.array([[0.5 * ratio, 0.5]]),
        length=ratio,
        width=1.0,
        young=12.0,
        poisson=0.3,
        thickness=1.0,
        pressure=1.0,
        terms=terms,
    )[0]
    rigidity = 12.0 / (12.0 * (1.0 - 0.3**2))  # E = t = 1 in the probe above
    return float(centre * rigidity)


@dataclass(frozen=True)
class RectangularPlateSolution:
    """Quantity of interest of the BFS rectangular plate arm."""

    length: float
    width: float
    young: float
    poisson: float
    thickness: float
    pressure: float
    elements_x: int
    elements_y: int
    bending_stiffness: float
    centre_deflection: float
    deflection: np.ndarray  # (elements_y + 1, elements_x + 1) nodal w
    n_dof: int
    n_free: int

    @property
    def short_side(self) -> float:
        return min(self.length, self.width)

    @property
    def aspect_ratio(self) -> float:
        return max(self.length, self.width) / self.short_side

    @property
    def alpha(self) -> float:
        """``w_centre D/(q b^4)`` with ``b`` the short side."""
        return self.centre_deflection * self.bending_stiffness / (
            self.pressure * self.short_side**4
        )

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(0.0, self.length, self.elements_x + 1)
        ys = np.linspace(0.0, self.width, self.elements_y + 1)
        X, Y = np.meshgrid(xs, ys)
        return X, Y, self.deflection

    def navier(self, points: np.ndarray, *, terms: int = 201) -> np.ndarray:
        return rectangular_plate_navier(
            points,
            length=self.length,
            width=self.width,
            young=self.young,
            poisson=self.poisson,
            thickness=self.thickness,
            pressure=self.pressure,
            terms=terms,
        )

    def centre_error(self, *, terms: int = 201) -> float:
        reference = float(
            self.navier(np.array([[0.5 * self.length, 0.5 * self.width]]), terms=terms)[0]
        )
        return _relative_error(self.centre_deflection, reference)


def solve_rectangular_plate_kirchhoff(
    *,
    length: float = 2.0,
    width: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    elements_x: int = 16,
    elements_y: int = 8,
) -> RectangularPlateSolution:
    """Solve the SSSS rectangular plate with the production bicubic BFS element.

    Boundary conditions are the hard simple support used by
    :func:`tensorfem.structural_verification.kirchhoff_ss_plate_uniform_load`:
    ``w = 0`` and the *tangential* slope ``w_,t = 0`` on every edge, with the
    normal bending moment left natural.  Element matrices come from
    :func:`tensorfem.plate_buckling_hermite.bfs_plate_element_matrices`; added here
    are the general ``length x width`` geometry and the consistent
    uniform-pressure load vector.  The helper in ``structural_verification``
    builds square plates only (its element size is ``side/n_x`` by ``side/n_y``
    whatever ``n_y`` says), so a rectangular plate cannot be obtained by scaling
    it; the element matrices themselves are shared.
    """
    a = _positive("length", length)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    q = _positive("pressure", pressure)
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    nx = _count("elements_x", elements_x)
    ny = _count("elements_y", elements_y)
    if nx < 2 or ny < 2 or nx % 2 or ny % 2:
        raise ValueError("elements_x/elements_y must be even and >= 2 so the centre node exists")

    nodes_x, nodes_y = nx + 1, ny + 1
    ndof = 4 * nodes_x * nodes_y
    lx, ly = a / nx, b / ny
    ke, _kg = bfs_plate_element_matrices(lx, ly, young, poisson, thickness)
    # 1-D consistent load integrals of the cubic Hermite basis on [0, L]:
    #   int N = L/2 (value), +L^2/12 (slope at the left node), -L^2/12 (right).
    ix = np.array([0.5 * lx, lx**2 / 12.0, 0.5 * lx, -lx**2 / 12.0])
    iy = np.array([0.5 * ly, ly**2 / 12.0, 0.5 * ly, -ly**2 / 12.0])
    fe = np.zeros(16)
    for jj in (0, 1):
        for ii in (0, 1):
            base = 4 * (2 * jj + ii)
            for k in range(4):
                fe[base + k] = q * ix[k % 2] * iy[k // 2]

    def node_dof(i: int, j: int, k: int) -> int:
        return 4 * (j * nodes_x + i) + k

    connectivity = np.array(
        [
            [node_dof(i, j, k) for j in (jj, jj + 1) for i in (ii, ii + 1) for k in range(4)]
            for jj in range(ny)
            for ii in range(nx)
        ]
    )
    stiffness = np.zeros((ndof, ndof))
    load = np.zeros(ndof)
    rows = connectivity.reshape(-1)
    ne = len(connectivity)
    element_rows = np.repeat(connectivity, 16, axis=1).reshape(-1)
    element_cols = np.tile(connectivity, (1, 16)).reshape(-1)
    np.add.at(stiffness, (element_rows, element_cols), np.tile(ke.reshape(-1), ne))
    np.add.at(load, rows, np.tile(fe, ne))

    fixed: set[int] = set()
    for i in range(nodes_x):  # edges y = 0 and y = b: w and the tangential slope w_,x
        for j in (0, ny):
            fixed.add(node_dof(i, j, 0))
            fixed.add(node_dof(i, j, 1))
    for j in range(nodes_y):  # edges x = 0 and x = a: w and the tangential slope w_,y
        for i in (0, nx):
            fixed.add(node_dof(i, j, 0))
            fixed.add(node_dof(i, j, 2))
    free = np.array([d for d in range(ndof) if d not in fixed])

    solution = np.zeros(ndof)
    solution[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
    rigidity = young * t**3 / (12.0 * (1.0 - poisson**2))
    return RectangularPlateSolution(
        length=a,
        width=b,
        young=young,
        poisson=poisson,
        thickness=t,
        pressure=q,
        elements_x=nx,
        elements_y=ny,
        bending_stiffness=rigidity,
        centre_deflection=float(solution[node_dof(nx // 2, ny // 2, 0)]),
        deflection=solution[0::4].reshape(nodes_y, nodes_x),
        n_dof=ndof,
        n_free=int(free.size),
    )


# --------------------------------------------------------------------------- #
# 3. Bilinear (linear) Mindlin plate: the h-convergence-rate arm
# --------------------------------------------------------------------------- #
def _bilinear_shape(xi: float, eta: float) -> np.ndarray:
    """Bilinear Q4 shape functions in the mesh helper's node order."""
    return 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )


@dataclass(frozen=True)
class MindlinPlateSolution:
    """Bilinear Mindlin plate solution used for the rate-of-convergence evidence."""

    nodes: np.ndarray
    elements: np.ndarray
    displacement: np.ndarray  # (3 * n_nodes,) dof order (w, theta_x, theta_y)
    load: np.ndarray
    element_counts: tuple[int, int]
    length: float
    width: float

    @property
    def deflection(self) -> np.ndarray:
        return self.displacement[0::3]

    @property
    def n_dof(self) -> int:
        return int(self.displacement.size)


def solve_rectangular_plate_mindlin(
    *,
    length: float = 2.0,
    width: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    elements_x: int = 8,
    elements_y: int = 4,
    shear_treatment: str = "sri",
) -> MindlinPlateSolution:
    """Uniformly loaded SSSS rectangular plate on the bilinear Q4 Mindlin element.

    The element is the production :func:`tensorfem.plate.q4_mindlin_stiffness`
    (selective reduced shear integration by default).  On a structured mesh every
    element is a congruent copy of the first one, so the element matrix is formed
    once -- an exact reuse, not an approximation -- which keeps the refinement
    ladder affordable.  Uniform pressure on a constant-Jacobian bilinear element
    has the exact consistent nodal load ``q A/4`` per ``w`` dof.
    """
    a = _positive("length", length)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    q = _positive("pressure", pressure)
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    nx = _count("elements_x", elements_x)
    ny = _count("elements_y", elements_y)

    nodes_t, elements_t = structured_rectangular_mesh(nx, ny, a, b)
    nodes = nodes_t.numpy()
    elements = elements_t.numpy()
    ndof = 3 * nodes.shape[0]
    ke = (
        q4_mindlin_stiffness(
            nodes_t[elements_t[0]], young, poisson, t, shear_treatment=shear_treatment
        )
        .numpy()
    )
    dofs = np.stack((3 * elements, 3 * elements + 1, 3 * elements + 2), axis=2).reshape(-1, 12)
    stiffness = np.zeros((ndof, ndof))
    ne = len(elements)
    element_rows = np.repeat(dofs, 12, axis=1).reshape(-1)
    element_cols = np.tile(dofs, (1, 12)).reshape(-1)
    np.add.at(stiffness, (element_rows, element_cols), np.tile(ke.reshape(-1), ne))
    # Uniform pressure on a constant-Jacobian bilinear element: q A/4 on each of
    # the four transverse dofs (columns 0, 3, 6, 9 of the element dof block).
    load = np.zeros(ndof)
    nodal_pressure = np.zeros((ne, 12))
    nodal_pressure[:, 0:12:3] = q * (a / nx) * (b / ny) / 4.0
    np.add.at(load, dofs.reshape(-1), nodal_pressure.reshape(-1))

    tol = 1e-12 * max(a, b)
    x_edge = (np.abs(nodes[:, 0]) < tol) | (np.abs(nodes[:, 0] - a) < tol)
    y_edge = (np.abs(nodes[:, 1]) < tol) | (np.abs(nodes[:, 1] - b) < tol)
    on_boundary = x_edge | y_edge
    # Hard simple support: w = 0 on the boundary and the tangential slope vanishes
    # on each edge (theta_x on the y-edges, theta_y on the x-edges); the normal
    # rotation stays free.
    fixed = np.unique(
        np.concatenate(
            (
                3 * np.nonzero(on_boundary)[0],
                3 * np.nonzero(y_edge)[0] + 1,
                3 * np.nonzero(x_edge)[0] + 2,
            )
        )
    )
    free = np.setdiff1d(np.arange(ndof), fixed)
    displacement = np.zeros(ndof)
    displacement[free] = np.linalg.solve(stiffness[np.ix_(free, free)], load[free])
    return MindlinPlateSolution(
        nodes=nodes,
        elements=elements,
        displacement=displacement,
        load=load,
        element_counts=(nx, ny),
        length=a,
        width=b,
    )


def plate_deflection_l2_error(
    solution: MindlinPlateSolution,
    *,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    terms: int = 201,
) -> float:
    """Relative ``L_2`` error of the bilinear deflection field against Navier.

    The integral is evaluated with the element's own 2x2 Gauss rule applied to the
    bilinear interpolation of the nodal deflections, i.e. exactly the field the
    element represents.
    """
    nodes = solution.nodes
    elements = solution.elements
    w = solution.deflection
    gp = 1.0 / math.sqrt(3.0)
    points: list[tuple[float, float]] = []
    values: list[float] = []
    weights: list[float] = []
    for conn in elements:
        xy = nodes[conn]
        for xi, eta in product((-gp, gp), (-gp, gp)):
            shape = _bilinear_shape(xi, eta)
            dn_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dn_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            jacobian = np.array(
                [
                    [dn_dxi @ xy[:, 0], dn_dxi @ xy[:, 1]],
                    [dn_deta @ xy[:, 0], dn_deta @ xy[:, 1]],
                ]
            )
            det = abs(float(np.linalg.det(jacobian)))
            point = shape @ xy
            points.append((float(point[0]), float(point[1])))
            values.append(float(shape @ w[conn]))
            weights.append(det)
    exact = rectangular_plate_navier(
        np.array(points),
        length=solution.length,
        width=solution.width,
        young=young,
        poisson=poisson,
        thickness=thickness,
        pressure=pressure,
        terms=terms,
    )
    weights_arr = np.array(weights)
    error = math.sqrt(float(np.sum((np.array(values) - exact) ** 2 * weights_arr)))
    norm = math.sqrt(float(np.sum(exact**2 * weights_arr)))
    return error / norm


@dataclass(frozen=True)
class PlateConvergenceRow:
    """One rung of the bilinear-plate refinement ladder."""

    level: int
    elements_x: int
    elements_y: int
    n_dof: int
    element_size: float
    relative_l2_error: float


def linear_element_order_study(
    levels: tuple[int, ...] = (4, 8, 12, 16, 20),
    *,
    aspect_ratio: float = 2.0,
    short_side: float = 1.0,
    thickness: float = 0.01,
    young: float = 2.1e11,
    poisson: float = 0.3,
    pressure: float = 1.0,
) -> tuple[PlateConvergenceRow, ...]:
    """Refine a bilinear Mindlin plate uniformly and record the deflection L2 error.

    ``level`` is the element count along the **short** side; the long side gets
    ``level * aspect_ratio`` elements so every element stays square.
    """
    ratio = _positive("aspect_ratio", aspect_ratio)
    if abs(ratio - round(ratio)) > 1e-12:
        raise ValueError("aspect_ratio must be an integer for a square-element ladder")
    ratio = int(round(ratio))
    b = _positive("short_side", short_side)
    rows = []
    for level in levels:
        level = _count("level", level)
        solution = solve_rectangular_plate_mindlin(
            length=b * ratio,
            width=b,
            young=young,
            poisson=poisson,
            thickness=thickness,
            pressure=pressure,
            elements_x=level * ratio,
            elements_y=level,
        )
        rows.append(
            PlateConvergenceRow(
                level=level,
                elements_x=level * ratio,
                elements_y=level,
                n_dof=solution.n_dof,
                element_size=b / level,
                relative_l2_error=plate_deflection_l2_error(
                    solution,
                    young=young,
                    poisson=poisson,
                    thickness=thickness,
                    pressure=pressure,
                ),
            )
        )
    return tuple(rows)


def convergence_order(rows: tuple[PlateConvergenceRow, ...]) -> float:
    """Least-squares ``L_2`` order over the whole ladder (no point cherry-picking)."""
    h = np.array([row.element_size for row in rows], dtype=float)
    error = np.array([row.relative_l2_error for row in rows], dtype=float)
    return float(observed_order(h, error))


def pairwise_orders(rows: tuple[PlateConvergenceRow, ...]) -> list[dict[str, float]]:
    """Observed order of every consecutive pair of rungs (transparency aid)."""
    out = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        order = math.log(coarse.relative_l2_error / fine.relative_l2_error) / math.log(
            coarse.element_size / fine.element_size
        )
        out.append(
            {
                "from_level": coarse.level,
                "to_level": fine.level,
                "observed_order": order,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 4. Modal quantities
# --------------------------------------------------------------------------- #
def cantilever_first_frequency(
    *, length: float = 1.0, bending_stiffness: float = 2000.0, mass_per_length: float = 8.0
) -> float:
    """Exact Euler--Bernoulli first frequency of the cantilever, in Hz."""
    omega = cantilever_exact_angular_frequency(
        _positive("length", length),
        _positive("bending_stiffness", bending_stiffness),
        _positive("mass_per_length", mass_per_length),
    )
    return omega / (2.0 * math.pi)


def solve_plate_first_frequency(
    *,
    length: float = 2.0,
    width: float = 1.0,
    young: float = 200.0e9,
    poisson: float = 0.3,
    thickness: float = 0.01,
    density: float = 7850.0,
    elements_x: int = 16,
    elements_y: int = 8,
) -> dict[str, float]:
    """First simply supported natural frequency of the rectangular plate, in Hz.

    Drives the verified production eigen-path
    :func:`tensorfem.plate_vibration.solve_plate_vibration` and compares it with
    ``omega_11 = pi^2 sqrt(D/(rho t)) [(1/a)^2 + (1/b)^2]``.
    """
    a = _positive("length", length)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    nx = _count("elements_x", elements_x)
    ny = _count("elements_y", elements_y)
    result = solve_plate_vibration(
        nx,
        ny,
        length=a,
        width=b,
        young=young,
        poisson=poisson,
        thickness=t,
        density=density,
        boundary=BOUNDARY_SIMPLE,
        modes=1,
    )
    computed = float(result.frequencies_hz[0])
    exact = (
        kirchhoff_angular_frequency(
            length=a, width=b, density=density, young=young, poisson=poisson, thickness=t
        )
        / (2.0 * math.pi)
    )
    return {
        "length": a,
        "width": b,
        "elements_x": nx,
        "elements_y": ny,
        "computed_hz": computed,
        "exact_hz": exact,
        "relative_error": _relative_error(computed, exact),
    }


# --------------------------------------------------------------------------- #
# 5. Kirsch stress concentration through the quadratic (Q8) element family
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StressConcentrationResult:
    """``K_t`` of the circular hole recovered from a Q8 annulus."""

    inner_radius: float
    outer_radius: float
    n_theta: int
    n_radial: int
    far_field: float
    factor: float  # sigma_tt(a, pi/2) / T
    displacement_relative_l2: float
    n_node: int
    cg_relative_residual: float

    @property
    def exact_factor(self) -> float:
        return 3.0

    @property
    def absolute_error(self) -> float:
        return abs(self.factor - self.exact_factor)


def _kirsch_q8_field(
    *,
    n_theta: int,
    n_radial: int,
    inner_radius: float,
    outer_radius: float,
    far_field: float,
    young: float,
    poisson: float,
):
    """Solve the Q8 annulus with the exact Kirsch displacement on the outer rim."""
    nt = _count("n_theta", n_theta)
    if nt % 2:
        raise ValueError("n_theta must be even so that pi/2 is a node line")
    nr = _count("n_radial", n_radial)
    inner = _positive("inner_radius", inner_radius)
    outer = _positive("outer_radius", outer_radius)
    if outer <= inner:
        raise ValueError("outer_radius must exceed inner_radius")
    tension = _positive("far_field", far_field)

    hole = KirschHole(
        radius=inner, far_field=tension, young=young, poisson=poisson, plane="stress"
    )
    nodes, cells = polar_q8_mesh(inner, outer, nt, nr)
    coordinates = nodes.numpy()
    exact_u = np.stack(
        hole.displacement_cartesian(coordinates[:, 0], coordinates[:, 1]), axis=1
    )
    radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    on_rim = radius > outer * (1.0 - 1e-9)
    dirichlet = {}
    for i in np.nonzero(on_rim)[0]:
        dirichlet[2 * int(i)] = float(exact_u[i, 0])
        dirichlet[2 * int(i) + 1] = float(exact_u[i, 1])
    result = solve_q8_plane(
        nodes,
        cells,
        young=young,
        poisson=poisson,
        thickness=1.0,
        plane="stress",
        dirichlet=dirichlet,
    )
    return hole, coordinates, exact_u, result


def kirsch_q8_hole_edge_hoop_stress(
    *,
    n_theta: int = 16,
    n_radial: int = 4,
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
    far_field: float = 1.0,
    young: float = 1.0e5,
    poisson: float = 0.3,
) -> dict[str, np.ndarray]:
    """Hoop stress on the hole edge from Q8 nodal recovery, versus the exact field.

    Returns ``{'theta', 'computed', 'exact'}`` with the hoop stress normalised by
    the far-field tension ``T``.  The exact edge value is ``T (1 - 2 cos 2 theta)``
    (``3T`` at ``theta = pi/2``, ``-T`` at ``theta = 0``).
    """
    hole, coordinates, _exact_u, result = _kirsch_q8_field(
        n_theta=n_theta,
        n_radial=n_radial,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        far_field=far_field,
        young=young,
        poisson=poisson,
    )
    radius = np.hypot(coordinates[:, 0], coordinates[:, 1])
    theta = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    on_hole = radius < inner_radius * (1.0 + 1e-9)
    # keep one representative per distinct angle (the hole rim nodes are unique)
    order = np.argsort(theta[on_hole])
    theta_edge = theta[on_hole][order]
    stress = result.nodal_stress.numpy()[on_hole][order]
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    c, s = np.cos(theta_edge), np.sin(theta_edge)
    computed = sxx * s * s + syy * c * c - 2.0 * sxy * s * c
    exact = hole.far_field * (1.0 - 2.0 * np.cos(2.0 * theta_edge))
    return {
        "theta": theta_edge,
        "computed": computed / hole.far_field,
        "exact": exact / hole.far_field,
    }


def solve_kirsch_q8_concentration(
    *,
    n_theta: int = 16,
    n_radial: int = 4,
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
    far_field: float = 1.0,
    young: float = 1.0e5,
    poisson: float = 0.3,
) -> StressConcentrationResult:
    """Recover ``K_t`` from a Q8 annulus whose outer rim carries the exact field.

    The exact Kirsch displacement is prescribed on ``r = outer_radius``; the hole
    edge is traction free, which the variational statement satisfies *naturally*
    (no equation is imposed there).  The Kirsch field is the exact solution of that
    Dirichlet problem, so the only error is discretisation.  The quadratic
    mid-edge nodes lie exactly on both circles
    (:func:`tensorfem.quadratic_plane.polar_q8_mesh`), so the geometry is exact as
    well.  ``sigma_tt(a, pi/2)`` is the ``sigma_xx`` component at the node
    ``(0, a)`` of the production Gauss-stress extrapolation.
    """
    nt = _count("n_theta", n_theta)
    nr = _count("n_radial", n_radial)
    inner = _positive("inner_radius", inner_radius)
    outer = _positive("outer_radius", outer_radius)
    tension = _positive("far_field", far_field)
    hole, coordinates, exact_u, result = _kirsch_q8_field(
        n_theta=nt,
        n_radial=nr,
        inner_radius=inner,
        outer_radius=outer,
        far_field=tension,
        young=young,
        poisson=poisson,
    )
    nodal_stress = result.nodal_stress.numpy()
    target = np.array([0.0, inner])  # hole edge at theta = pi/2; hoop stress = sigma_xx
    index = int(np.argmin(((coordinates - target) ** 2).sum(axis=1)))
    if float(np.linalg.norm(coordinates[index] - target)) > 1e-9 * outer:
        raise RuntimeError("the hole-edge sample point is not a mesh node")
    factor = float(nodal_stress[index, 0]) / tension

    displacement = result.displacement.numpy().reshape(-1, 2)
    residual = displacement - exact_u
    relative_l2 = math.sqrt(float(np.sum(residual**2)) / float(np.sum(exact_u**2)))
    return StressConcentrationResult(
        inner_radius=inner,
        outer_radius=outer,
        n_theta=nt,
        n_radial=nr,
        far_field=tension,
        factor=factor,
        displacement_relative_l2=relative_l2,
        n_node=int(coordinates.shape[0]),
        cg_relative_residual=float(result.relative_residual),
    )


# --------------------------------------------------------------------------- #
# 6. The campaign face: run every arm and evaluate its gate
# --------------------------------------------------------------------------- #
def run_canonical_face(
    *,
    beam_elements: int = 8,
    plate_elements: tuple[int, int] = (12, 12),
    aspect_ratios: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
    plate_frequency_elements: tuple[int, int] = (16, 8),
    kirsch_n_theta: int = 16,
    kirsch_n_radial: int = 4,
    convergence_levels: tuple[int, ...] = (4, 8, 12, 16, 20),
) -> dict:
    """Run the canonical face and return every gate with its numbers.

    ``plate_elements`` is ``(elements_y, elements_x at a/b = 1)``; the x count is
    scaled by the aspect ratio so every plate element stays square.  The return
    value is JSON-serialisable (numpy containers are converted to lists).
    """
    # --- C1: cantilever under a concentrated tip load -----------------------
    beam = solve_cantilever_point_load(elements=beam_elements)
    c1 = {
        "case": "Euler-Bernoulli cantilever, concentrated tip load P",
        "elements": beam.elements,
        "length_m": beam.length,
        "bending_stiffness_nm2": beam.bending_stiffness,
        "load_n": beam.load,
        "tip_deflection_m": beam.tip_deflection,
        "exact_tip_deflection_m": beam.exact_tip_deflection,
        "relative_error": beam.tip_deflection_error,
        "tip_rotation_rad": beam.tip_rotation,
        "exact_tip_rotation_rad": beam.exact_tip_rotation,
        "tip_rotation_relative_error": beam.tip_rotation_error,
        "base_reaction_n": beam.base_reaction,
        "base_moment_nm": beam.base_moment,
        "strain_energy_j": beam.strain_energy,
        "exact_strain_energy_j": beam.exact_strain_energy,
        "energy_relative_error": beam.energy_relative_error,
        "nodal_field_l2_relative_error": float(
            np.linalg.norm(beam.transverse - beam.exact_transverse())
            / np.linalg.norm(beam.exact_transverse())
        ),
        "nodes": beam.nodes().tolist(),
        "fe_deflection": beam.transverse.tolist(),
        "exact_deflection": beam.exact_transverse().tolist(),
        "gate": GATE_BEAM_TIP,
    }
    c1["pass"] = bool(c1["relative_error"] < GATE_BEAM_TIP)

    # --- C2: rectangular simply supported plate under uniform pressure ------
    elements_y = plate_elements[0]
    rows = []
    for ratio in aspect_ratios:
        ratio = float(ratio)
        elements_x = int(round(elements_y * ratio))
        solution = solve_rectangular_plate_kirchhoff(
            length=ratio, width=1.0, elements_x=elements_x, elements_y=elements_y
        )
        exact_alpha = rectangular_plate_alpha(ratio)
        tabulated = TIMOSHENKO_ALPHA_TABLE.get(ratio)
        rows.append(
            {
                "aspect_ratio": ratio,
                "elements": [elements_x, elements_y],
                "centre_deflection_m": solution.centre_deflection,
                "alpha_fe": solution.alpha,
                "alpha_navier": exact_alpha,
                "relative_error": _relative_error(solution.alpha, exact_alpha),
                "alpha_timoshenko": tabulated,
                "timoshenko_relative_error": (
                    None if tabulated is None else _relative_error(tabulated, exact_alpha)
                ),
                "n_dof": solution.n_dof,
                "n_free": solution.n_free,
                "pass": bool(_relative_error(solution.alpha, exact_alpha) < GATE_PLATE_CENTRE),
            }
        )
    strip = rectangular_plate_alpha(60.0)
    c2 = {
        "case": "SSSS rectangular Kirchhoff plate, uniform pressure, alpha(a/b)",
        "rows": rows,
        "strip_alpha_series": strip,
        "strip_alpha_exact": STRIP_ALPHA,
        "strip_relative_error": _relative_error(strip, STRIP_ALPHA),
        "gate": GATE_PLATE_CENTRE,
    }
    c2["pass"] = all(row["pass"] for row in rows)

    # --- C3: cantilever first frequency ------------------------------------
    beam_modal = cantilever_beam_modes(1.0, 8, 2000.0, 8.0, 1)
    f1_fe = float(beam_modal.frequencies_hz[0])
    f1_exact = cantilever_first_frequency()
    c3 = {
        "case": "Euler-Bernoulli cantilever, first bending mode",
        "elements": 8,
        "computed_hz": f1_fe,
        "exact_hz": f1_exact,
        "relative_error": _relative_error(f1_fe, f1_exact),
        "gate": GATE_BEAM_FREQUENCY,
    }
    c3["pass"] = bool(c3["relative_error"] < GATE_BEAM_FREQUENCY)

    # --- C4: rectangular plate first frequency -----------------------------
    c4 = {
        "case": "SSSS rectangular plate (a/b = 2), first mode",
        **solve_plate_first_frequency(
            elements_x=plate_frequency_elements[0], elements_y=plate_frequency_elements[1]
        ),
        "gate": GATE_PLATE_FREQUENCY,
    }
    c4["pass"] = bool(c4["relative_error"] < GATE_PLATE_FREQUENCY)

    # --- C5: Kirsch stress concentration through the Q8 family -------------
    kirsch = solve_kirsch_q8_concentration(n_theta=kirsch_n_theta, n_radial=kirsch_n_radial)
    c5 = {
        "case": "Kirsch circular hole in an infinite plate, uniaxial tension",
        "n_theta": kirsch.n_theta,
        "n_radial": kirsch.n_radial,
        "n_node": kirsch.n_node,
        "inner_radius": kirsch.inner_radius,
        "outer_radius": kirsch.outer_radius,
        "computed_factor": kirsch.factor,
        "exact_factor": kirsch.exact_factor,
        "absolute_error": kirsch.absolute_error,
        "displacement_relative_l2": kirsch.displacement_relative_l2,
        "cg_relative_residual": kirsch.cg_relative_residual,
        "gate": GATE_STRESS_CONCENTRATION,
    }
    c5["pass"] = bool(c5["absolute_error"] < GATE_STRESS_CONCENTRATION)

    # --- C6: L2 convergence order of the bilinear plate ---------------------
    ladder = linear_element_order_study(convergence_levels)
    order = convergence_order(ladder)
    c6 = {
        "case": "SSSS rectangular plate (a/b = 2), bilinear Q4 deflection L2 error",
        "rows": [
            {
                "level": row.level,
                "elements": [row.elements_x, row.elements_y],
                "n_dof": row.n_dof,
                "element_size": row.element_size,
                "relative_l2_error": row.relative_l2_error,
            }
            for row in ladder
        ],
        "pairwise_orders": pairwise_orders(ladder),
        "observed_order": order,
        "gate": GATE_LINEAR_ORDER,
    }
    c6["pass"] = bool(order >= GATE_LINEAR_ORDER)

    arms = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}
    return {
        "gates": {
            key: {"pass": value["pass"], "gate": value["gate"]} for key, value in arms.items()
        },
        "all_pass": all(value["pass"] for value in arms.values()),
        "arms": arms,
    }
