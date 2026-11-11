"""Three-dimensional cohesive-zone interface element (two 4-node faces).

The element is the standard 8-node (4+4) zero-thickness cohesive interface:
face nodes ``0..3`` belong to the lower body and nodes ``4..7`` to the upper
body, in the same in-plane ordering, so that node ``i+4`` of the upper face
lies opposite node ``i`` of the lower face.  Separation is interpolated
bilinearly, integrated with Gauss quadrature in the mid-surface frame, and
converted to nodal forces/tangents through the work-conjugate relation

    f = int_A B^T R^T t(R B u) dA ,   K = int_A B^T R^T (dt/ddelta) R B dA

with ``R`` the mid-surface frame (rows: normal, in-plane tangent 1, in-plane
tangent 2).

Constitutive law
----------------
The traction--separation law is the three-dimensional extension of the
proportional-path Benzeggagh--Kenane (BK) model already used in
:mod:`tensorfem.mixed_mode_cohesive`.  For a unit mix direction
``c = (c_n, c_s, c_t)`` in the local frame:

* equivalent separation ``lam = |delta|`` (Euclidean, normal component clipped
  at zero) so that ``lamf`` equals the physical mode-I/mode-II failure
  openings ``2 Gc / t0``,
* mixed-mode onset from the quadratic stress criterion,
  ``lam0 = 1/sqrt((kn c_n/tn0)^2 + (ks c_s/ts0)^2 + (ks c_t/ts0)^2)``,
* work-conjugate stiffness ``w = kn c_n^2 + ks (c_s^2 + c_t^2)``,
* critical energy ``Gc = GIc + (GIIc - GIc) (Gs/w)^eta`` with
  ``Gs = ks (c_s^2 + c_t^2)``,
* failure separation ``lamf = 2 Gc / (w lam0)`` from the triangular envelope
  area, i.e. ``int t . ddelta = Gc`` exactly.

Damage is irreversible: unloading/reloading follows the secant to the origin at
the historical maximum equivalent separation.  Normal compression is carried by
the undamaged normal penalty and does not advance tensile history.

Honest boundaries (not claimed by this module)
---------------------------------------------
* Small strains with a *reference-configuration* mid-surface frame.  A rigid
  rotation therefore produces a spurious shear separation of order
  ``omega x h`` (documented in ``docs/cohesive3d-interface-element.md``);
  only rigid *translation* is reproduced exactly.
* Monotonic/proportional paths: non-proportional mixed-mode histories are
  represented only through the scalar history ``lam_max``.
* Prescribed crack path (the interface mesh is given); no crack tracking,
  insertion, or branching.
* Linear (bilinear) interface geometry; no mortar tying and no finite sliding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_SQRT3_INV = 1.0 / math.sqrt(3.0)
#: Regularisation inside ``sqrt`` so that the AD tangent stays finite at a
#: vanishing separation (``d sqrt(x)/dx`` diverges at ``x = 0``).
_TINY = 1e-30

#: Corner coordinates of the reference 4-node face, in the node ordering
#: ``(-1,-1), (+1,-1), (+1,+1), (-1,+1)``.
_FACE_CORNERS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))


def gauss_rule(order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """One-dimensional Gauss--Legendre points and weights on ``[-1, 1]``."""
    if order == 1:
        return (0.0,), (2.0,)
    if order == 2:
        return (-_SQRT3_INV, _SQRT3_INV), (1.0, 1.0)
    if order == 3:
        return (-math.sqrt(0.6), 0.0, math.sqrt(0.6)), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    raise ValueError("supported one-dimensional Gauss orders are 1, 2 and 3")


def face_shape(xi: float, eta: float, *, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear face shape functions ``N`` (4,) and natural derivatives ``dN`` (4,2)."""
    nodes = []
    derivatives = []
    for cx, cy in _FACE_CORNERS:
        nodes.append(0.25 * (1.0 + cx * xi) * (1.0 + cy * eta))
        derivatives.append([
            0.25 * cx * (1.0 + cy * eta),
            0.25 * cy * (1.0 + cx * xi),
        ])
    return (torch.tensor(nodes, dtype=dtype), torch.tensor(derivatives, dtype=dtype))


def interface_frame(
    coordinates: torch.Tensor, xi: float, eta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mid-surface frame ``R`` (3,3) and Jacobian area factor at ``(xi, eta)``.

    Row 0 is the mid-surface normal, rows 1 and 2 the in-plane tangents.  The
    frame is orthonormal, so ``R @ R.T = I`` to machine precision.
    """
    coords = _as_coords(coordinates)
    mid = 0.5 * (coords[0] + coords[1])
    _, dN = face_shape(xi, eta, dtype=coords.dtype)
    a1 = dN[:, 0] @ mid
    a2 = dN[:, 1] @ mid
    cross = torch.linalg.cross(a1, a2)
    area = torch.linalg.vector_norm(cross)
    if float(area) <= 0.0:
        raise ValueError("degenerate interface geometry: zero mid-surface Jacobian")
    normal = cross / area
    tangent_1 = a1 / torch.linalg.vector_norm(a1)
    tangent_2 = torch.linalg.cross(normal, tangent_1)
    return torch.stack((normal, tangent_1, tangent_2)), area


def interface_area(coordinates: torch.Tensor, *, order: int = 2) -> torch.Tensor:
    """Mid-surface area by Gauss integration (exact for planar bilinear faces)."""
    coords = _as_coords(coordinates)
    points, weights = gauss_rule(order)
    total = torch.zeros((), dtype=coords.dtype)
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            _, area = interface_frame(coords, xi, eta)
            total = total + wx * wy * area
    return total


def _as_coords(coordinates: torch.Tensor) -> torch.Tensor:
    coords = torch.as_tensor(coordinates)
    if coords.shape == (2, 4, 3):
        return coords
    if coords.shape == (8, 3) or coords.shape == (24,):
        return coords.reshape(2, 4, 3)
    raise ValueError("coordinates must have shape (8, 3), (2, 4, 3) or (24,)")


def _as_displacement(displacement: torch.Tensor, dtype, device) -> torch.Tensor:
    u = torch.as_tensor(displacement, dtype=dtype, device=device)
    if u.shape == (8, 3) or u.shape == (24,):
        return u.reshape(8, 3)
    raise ValueError("displacement must have shape (8, 3) or (24,)")


@dataclass(frozen=True)
class Cohesive3DLaw:
    """Mixed-mode bilinear cohesive law with the BK propagation criterion.

    Parameters
    ----------
    normal_stiffness, shear_stiffness
        Penalty stiffnesses ``kn`` (normal) and ``ks`` (both shear directions).
    normal_strength, shear_strength
        Peak tractions ``tn0`` and ``ts0`` (the two shear directions share one
        strength).
    mode_i_energy, mode_ii_energy
        ``GIc`` and ``GIIc``; the BK criterion interpolates between them.
    eta
        BK exponent.  The default ``2.284`` is the value fitted by Benzeggagh
        and Kenane (1996) for IM7/8552 carbon--epoxy laminates.
    """

    normal_stiffness: float
    shear_stiffness: float
    normal_strength: float
    shear_strength: float
    mode_i_energy: float
    mode_ii_energy: float
    eta: float = 2.284

    def __post_init__(self) -> None:
        values = (
            self.normal_stiffness,
            self.shear_stiffness,
            self.normal_strength,
            self.shear_strength,
            self.mode_i_energy,
            self.mode_ii_energy,
            self.eta,
        )
        if any(float(v) <= 0.0 for v in values):
            raise ValueError("all cohesive-law parameters must be positive")

    # -- pure-mode reference values -------------------------------------------
    @property
    def onset_opening_mode_i(self) -> float:
        return self.normal_strength / self.normal_stiffness

    @property
    def onset_opening_mode_ii(self) -> float:
        return self.shear_strength / self.shear_stiffness

    @property
    def failure_opening_mode_i(self) -> float:
        return 2.0 * self.mode_i_energy / self.normal_strength

    @property
    def failure_opening_mode_ii(self) -> float:
        return 2.0 * self.mode_ii_energy / self.shear_strength

    # -- mixed-mode measures --------------------------------------------------
    def direction_measures(self, direction: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Onset/failure separations and critical energy for a unit mix direction.

        Returns ``(lam0, lamf, gc, work_stiffness)``.  The direction is
        normalised internally and its normal component must be non-negative.
        """
        c = torch.as_tensor(direction)
        if c.shape != (3,):
            raise ValueError("direction must have shape (3,)")
        norm = torch.linalg.vector_norm(c)
        if float(norm) <= 0.0:
            raise ValueError("direction must be nonzero")
        c = c / norm
        if float(c[0]) < 0.0:
            raise ValueError("normal component of the mix direction must be non-negative")
        kn, ks = self.normal_stiffness, self.shear_stiffness
        tn0, ts0 = self.normal_strength, self.shear_strength
        work_stiffness = kn * c[0] ** 2 + ks * (c[1] ** 2 + c[2] ** 2)
        inverse_lam0_sq = (
            (kn * c[0] / tn0) ** 2 + (ks * c[1] / ts0) ** 2 + (ks * c[2] / ts0) ** 2
        )
        lam0 = 1.0 / torch.sqrt(inverse_lam0_sq)
        shear_fraction = ks * (c[1] ** 2 + c[2] ** 2) / work_stiffness
        gc = self.mode_i_energy + (self.mode_ii_energy - self.mode_i_energy) * (
            shear_fraction ** self.eta
        )
        lamf = 2.0 * gc / (work_stiffness * lam0)
        return lam0, lamf, gc, work_stiffness

    # -- scalar envelope ------------------------------------------------------
    def _envelope_factor(
        self, lam: torch.Tensor, lam0: torch.Tensor, lamf: torch.Tensor
    ) -> torch.Tensor:
        """Scalar ``g`` with ``traction = k * delta * g`` along the envelope."""
        span = lamf - lam0
        if float(span) <= 0.0:
            raise ValueError("fracture energy is too small for the strengths and stiffnesses")
        softening = lam0 * (lamf - lam) / (torch.clamp(lam, min=_TINY) * span)
        return torch.where(
            lam <= lam0,
            torch.ones_like(lam),
            torch.where(lam < lamf, softening, torch.zeros_like(lam)),
        )

    def traction(self, separation: torch.Tensor, maximum_separation: torch.Tensor) -> torch.Tensor:
        """Local traction vector ``(t_n, t_s, t_t)`` -- differentiable in ``separation``."""
        delta = torch.as_tensor(separation)
        if delta.shape != (3,):
            raise ValueError("separation must have shape (3,)")
        kn, ks = self.normal_stiffness, self.shear_stiffness
        dn, ds, dt = delta[0], delta[1], delta[2]
        dn_positive = torch.clamp(dn, min=0.0)
        lam = torch.sqrt(dn_positive**2 + ds**2 + dt**2 + _TINY)
        direction = _unit_direction(torch.stack((dn_positive, ds, dt)))
        lam0, lamf, _, _ = self.direction_measures(direction)
        lam_max = torch.maximum(lam, torch.clamp(maximum_separation, min=0.0))
        factor = self._envelope_factor(lam_max, lam0, lamf)
        tensile_normal = kn * dn_positive * factor
        normal = torch.where(dn < 0.0, kn * dn, tensile_normal)
        return torch.stack((normal, ks * ds * factor, ks * dt * factor))

    def tangent(self, separation: torch.Tensor, maximum_separation: torch.Tensor) -> torch.Tensor:
        """Consistent tangent ``dt/ddelta`` (3,3) by automatic differentiation."""
        delta = torch.as_tensor(separation).clone()
        lam_max = torch.as_tensor(maximum_separation).detach()
        jacobian = torch.autograd.functional.jacobian(
            lambda d: self.traction(d, lam_max), delta, vectorize=True
        )
        return jacobian

    # -- diagnostics ----------------------------------------------------------
    def dissipated_energy_density(
        self, separation: torch.Tensor, maximum_separation: torch.Tensor
    ) -> torch.Tensor:
        """Cohesive energy dissipated per unit area at the historical maximum.

        The work supplied along the envelope up to ``lam_max`` minus the elastic
        energy currently stored in the secant spring; it equals ``Gc(direction)``
        once the interface has fully failed.
        """
        delta = torch.as_tensor(separation)
        dn_positive = torch.clamp(delta[0], min=0.0)
        lam = torch.sqrt(dn_positive**2 + delta[1] ** 2 + delta[2] ** 2 + _TINY)
        direction = _unit_direction(torch.stack((dn_positive, delta[1], delta[2])))
        lam0, lamf, _, work_stiffness = self.direction_measures(direction)
        lam_max = torch.clamp(torch.maximum(lam, maximum_separation), min=0.0)
        span = torch.clamp(lamf - lam0, min=_TINY)
        elastic_work = 0.5 * work_stiffness * torch.clamp(lam_max, max=lam0) ** 2
        bounded = torch.clamp(lam_max, max=lamf)
        softening_work = (
            work_stiffness
            * lam0
            / span
            * (lamf * (bounded - lam0) - 0.5 * (bounded**2 - lam0**2))
        )
        work = torch.where(
            lam_max > lam0,
            elastic_work + softening_work,
            elastic_work,
        )
        factor = self._envelope_factor(lam_max, lam0, lamf)
        stored = 0.5 * work_stiffness * (factor * lam_max) ** 2
        return torch.clamp(work - stored, min=0.0)


@dataclass(frozen=True)
class Cohesive3DResponse:
    """Element-level result of :func:`cohesive_interface_response`."""

    force: torch.Tensor
    stiffness: torch.Tensor
    traction: torch.Tensor
    separation: torch.Tensor
    damage: torch.Tensor
    maximum_separation: torch.Tensor
    dissipated_energy: torch.Tensor
    area: torch.Tensor


def cohesive_interface_response(
    coordinates: torch.Tensor,
    displacement: torch.Tensor,
    law: Cohesive3DLaw,
    *,
    history: torch.Tensor | None = None,
    order: int = 2,
) -> Cohesive3DResponse:
    """Internal force vector and tangent of one 8-node cohesive interface element.

    ``coordinates``/``displacement`` may be given as ``(8, 3)``, ``(2, 4, 3)``
    or flat ``(24,)`` arrays.  ``history`` holds the previous maximum equivalent
    separation per Gauss point (shape ``(n_gp,)``); a fresh element starts from
    zero history.  ``order`` selects the 1-D Gauss order used in each face
    direction (``order=2`` is exact for the bilinear separation fields
    interpolated by the element).
    """
    coords = _as_coords(torch.as_tensor(coordinates))
    dtype, device = coords.dtype, coords.device
    u = _as_displacement(displacement, dtype, device)
    points, weights = gauss_rule(order)
    n_gp = len(points) ** 2

    previous = None
    if history is not None:
        previous = torch.as_tensor(history, dtype=dtype, device=device).reshape(-1)
        if previous.numel() != n_gp:
            raise ValueError(f"history must hold {n_gp} Gauss-point values")

    force = torch.zeros((24,), dtype=dtype, device=device)
    nodal_force = force.reshape(8, 3)
    stiffness = torch.zeros((24, 24), dtype=dtype, device=device)
    traction_out, separation_out, damage_out = [], [], []
    maximum_out, dissipated_out = [], torch.zeros((), dtype=dtype, device=device)
    area_total = torch.zeros((), dtype=dtype, device=device)

    delta_u_nodes = u[4:] - u[:4]  # (4, 3) nodal separation increments

    gp = 0
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            shape, _ = face_shape(xi, eta, dtype=dtype)
            frame, area = interface_frame(coords, xi, eta)
            weight = wx * wy * area
            separation_global = shape @ delta_u_nodes
            separation = frame @ separation_global
            old_max = (
                torch.zeros((), dtype=dtype, device=device)
                if previous is None
                else previous[gp]
            )
            traction = law.traction(separation, old_max)
            tangent = law.tangent(separation, old_max)
            lam = _equivalent_separation(separation, law)
            maximum = torch.maximum(lam, old_max)
            force_global = frame.T @ traction
            tangent_global = frame.T @ tangent @ frame

            for i in range(4):
                contribution = weight * shape[i] * force_global
                nodal_force[i] -= contribution
                nodal_force[i + 4] += contribution
                for j in range(4):
                    block = weight * shape[i] * shape[j] * tangent_global
                    stiffness[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
                    stiffness[3 * (i + 4) : 3 * (i + 4) + 3, 3 * (j + 4) : 3 * (j + 4) + 3] += block
                    stiffness[3 * i : 3 * i + 3, 3 * (j + 4) : 3 * (j + 4) + 3] -= block
                    stiffness[3 * (i + 4) : 3 * (i + 4) + 3, 3 * j : 3 * j + 3] -= block

            traction_out.append(traction)
            separation_out.append(separation)
            damage_out.append(_damage(traction, separation, law))
            maximum_out.append(maximum)
            dissipated_out = dissipated_out + weight * law.dissipated_energy_density(
                separation, old_max
            )
            area_total = area_total + weight
            gp += 1

    return Cohesive3DResponse(
        force=force,
        stiffness=stiffness,
        traction=torch.stack(traction_out),
        separation=torch.stack(separation_out),
        damage=torch.stack(damage_out),
        maximum_separation=torch.stack(maximum_out),
        dissipated_energy=dissipated_out,
        area=area_total,
    )


def _unit_direction(vector: torch.Tensor) -> torch.Tensor:
    """Normalised direction with a safe fallback for a vanishing separation."""
    norm = torch.linalg.vector_norm(vector)
    fallback = torch.tensor([1.0, 0.0, 0.0], dtype=vector.dtype, device=vector.device)
    if float(norm) <= 1e-14:
        return fallback
    return vector / norm


def _equivalent_separation(separation: torch.Tensor, law: Cohesive3DLaw) -> torch.Tensor:
    """Euclidean equivalent separation with the normal component clipped at zero."""
    dn = torch.clamp(separation[0], min=0.0)
    return torch.sqrt(dn**2 + separation[1] ** 2 + separation[2] ** 2 + _TINY)


def _damage(
    traction: torch.Tensor, separation: torch.Tensor, law: Cohesive3DLaw
) -> torch.Tensor:
    """Scalar tensile damage ``1 - |t| / |k delta|`` on the elastic branch."""
    stiffness = torch.tensor(
        [law.normal_stiffness, law.shear_stiffness, law.shear_stiffness],
        dtype=separation.dtype,
        device=separation.device,
    )
    elastic = stiffness * separation
    denominator = torch.linalg.vector_norm(elastic)
    if float(denominator) <= 0.0:
        return torch.zeros((), dtype=separation.dtype, device=separation.device)
    ratio = torch.clamp(torch.linalg.vector_norm(traction) / denominator, max=1.0)
    return 1.0 - ratio

