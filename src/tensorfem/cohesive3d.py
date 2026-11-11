"""Three-dimensional zero-thickness cohesive interface elements.

Scope of this module
--------------------
The 2-D cohesive face of the library (``tensorfem.cohesive``,
``tensorfem.mixed_mode_cohesive``, ``tensorfem.cohesive_bending``) offers a
material-point law, a two-node normal interface and a complete 2-D three-point
bend simulation.  None of them carries a *three-dimensional* interface element,
so a 3-D crack plane (a delamination, a bonded joint, or an ice/structure
adhesive layer) could not be meshed at all.  This module fills exactly that gap
and nothing more:

* :class:`BKCohesive3DLaw` -- an irreversible, work-conjugate bilinear
  traction--separation law with three separation components
  ``(delta_n, delta_s1, delta_s2)`` and the Benzeggagh--Kenane (BK) mixed-mode
  propagation criterion.  It is the 3-D generalisation of the already verified
  2-D :class:`tensorfem.mixed_mode_cohesive.BenzeggaghKenaneLaw`; with
  ``delta_s2 = 0`` the two agree to machine precision (pinned by
  ``tests/test_cohesive3d.py``).
* :func:`cohesive_hex8_response` -- an 8-node, zero-thickness cohesive element
  (Abaqus ``COH3D8`` topology: nodes 0--3 on one face, nodes 4--7 on the
  opposite face, 2x2 Gauss quadrature on the mid-surface) returning the
  24-component internal force vector and the consistent 24x24 tangent.
* :func:`interface_area` / :func:`assemble_cohesive_interface` -- interface
  area and dense global assembly over a pre-meshed interface patch.

Conventions and limits (read before use)
----------------------------------------
* **Small-strain kinematics.**  The internal force is ``f = int B^T t dA`` and
  the tangent is ``K = int B^T D B dA``.  No geometric (initial-stress)
  stiffness term is added, so the element is *not* validated for
  geometrically non-linear host solids.  Rigid-body motion *is* reproduced
  exactly: the local frame is recomputed from the current mid-surface, hence
  rigid translation gives zero traction and rigid rotation rotates the traction
  vector exactly (both pinned to ~1e-14 in the tests).
* **Damage drives on tensile/separation quantities only.**  The effective
  separation is ``a = sqrt(<delta_n>^2 + delta_s1^2 + delta_s2^2)`` with
  ``<.>`` the Macaulay bracket; compression uses the undamaged penalty
  stiffness ``k_n`` and never grows the history variable.  Friction, contact
  and cyclic degradation are out of scope.
* **Quadratic nominal-stress initiation.**  The onset amplitude is chosen so
  that ``(t_n/t_n0)^2 + (t_s/t_s0)^2 = 1`` *exactly* at damage onset, which
  makes the peak traction of a mixed-mode path a closed-form oracle.
* **Energy is exact by construction.**  The failure amplitude
  ``a_f = 2 G_c / (k_w a_0)`` makes the triangular envelope area equal
  ``G_c`` per unit interface area, where ``G_c`` follows BK.  The tests verify
  the dissipated work against ``G_c * A`` analytically integrated through the
  element quadrature.
* **Unloading is secant to the origin** from the historical maximum opening
  (standard irreversible cohesive practice); the tangent returned during
  unloading is the secant stiffness, so a Newton solver sees a positive
  semi-definite operator there.

All quantities are SI (m, Pa, J/m^2, N).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "BKCohesive3DLaw",
    "Cohesive3DResponse",
    "cohesive_hex8_quadrature",
    "cohesive_shape_functions",
    "mid_surface_frame",
    "interface_area",
    "cohesive_hex8_response",
    "assemble_cohesive_interface",
]


# --------------------------------------------------------------------------- #
# Constitutive law
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cohesive3DResponse:
    """Local (interface-frame) response of the 3-D cohesive law.

    ``traction`` has shape ``(..., 3)`` with components ``(t_n, t_s1, t_s2)`` in
    the element normal/tangential frame; ``tangent`` has shape ``(..., 3, 3)``
    and is the derivative ``dt/d(delta)`` (unsymmetric for ``k_n != k_s`` in the
    softening branch, which is the exact consistent tangent);
    ``history`` is the updated maximum effective separation.
    """

    traction: torch.Tensor
    tangent: torch.Tensor
    damage: torch.Tensor
    history: torch.Tensor


class BKCohesive3DLaw:
    """Irreversible bilinear 3-D cohesive law with a BK mixed-mode criterion.

    Parameters
    ----------
    normal_stiffness, shear_stiffness : penalty stiffnesses ``k_n``, ``k_s`` [Pa/m].
    normal_strength, shear_strength : peak tractions ``t_n0``, ``t_s0`` [Pa].
    mode_i_energy, mode_ii_energy : ``G_Ic``, ``G_IIc`` [J/m^2].
    bk_eta : BK exponent ``eta`` in
        ``G_c = G_Ic + (G_IIc - G_Ic) (G_s / (G_n + G_s)) ** eta``.
    """

    def __init__(
        self,
        normal_stiffness: float,
        shear_stiffness: float,
        normal_strength: float,
        shear_strength: float,
        mode_i_energy: float,
        mode_ii_energy: float,
        bk_eta: float = 1.6,
    ):
        values = (
            normal_stiffness,
            shear_stiffness,
            normal_strength,
            shear_strength,
            mode_i_energy,
            mode_ii_energy,
            bk_eta,
        )
        if any(float(v) <= 0.0 for v in values):
            raise ValueError("all cohesive parameters must be positive")
        self.kn = float(normal_stiffness)
        self.ks = float(shear_stiffness)
        self.tn0 = float(normal_strength)
        self.ts0 = float(shear_strength)
        self.gic = float(mode_i_energy)
        self.giic = float(mode_ii_energy)
        self.eta = float(bk_eta)
        # Each pure-mode triangular envelope must be non-degenerate:
        # the failure opening 2 G_c / t_0 has to exceed the onset opening t_0 / k.
        if 2.0 * self.gic / self.tn0 <= self.tn0 / self.kn:
            raise ValueError("G_Ic is too small for the given normal stiffness and strength")
        if 2.0 * self.giic / self.ts0 <= self.ts0 / self.ks:
            raise ValueError("G_IIc is too small for the given shear stiffness and strength")

    # -- closed-form helpers -------------------------------------------------

    @staticmethod
    def _unit_direction(direction: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if bool(torch.any(norm <= 0)):
            raise ValueError("separation direction must be nonzero")
        return direction / norm

    def critical_energy(self, direction: torch.Tensor) -> torch.Tensor:
        """BK critical energy for unit direction ``(c_n, c_s1, c_s2)``."""
        c = self._unit_direction(torch.as_tensor(direction))
        cn = torch.clamp(c[..., 0], min=0.0)
        cs2 = c[..., 1] ** 2 + c[..., 2] ** 2
        wn = self.kn * cn**2
        ws = self.ks * cs2
        total = torch.clamp(wn + ws, min=torch.finfo(c.dtype).tiny)
        fraction = ws / total
        return self.gic + (self.giic - self.gic) * fraction**self.eta

    def onset_amplitude(self, direction: torch.Tensor) -> torch.Tensor:
        """Effective opening at quadratic nominal-stress initiation."""
        c = self._unit_direction(torch.as_tensor(direction))
        cn = torch.clamp(c[..., 0], min=0.0)
        cs = torch.sqrt(torch.clamp(c[..., 1] ** 2 + c[..., 2] ** 2, min=0.0))
        dn0 = self.tn0 / self.kn
        ds0 = self.ts0 / self.ks
        return 1.0 / torch.sqrt((cn / dn0) ** 2 + (cs / ds0) ** 2)

    def failure_amplitude(self, direction: torch.Tensor) -> torch.Tensor:
        """Effective opening at complete failure (``G_c`` area consumed)."""
        c = self._unit_direction(torch.as_tensor(direction))
        cn = torch.clamp(c[..., 0], min=0.0)
        cs2 = c[..., 1] ** 2 + c[..., 2] ** 2
        work_stiffness = self.kn * cn**2 + self.ks * cs2
        peak_work_traction = work_stiffness * self.onset_amplitude(direction)
        return 2.0 * self.critical_energy(direction) / peak_work_traction

    # -- history -------------------------------------------------------------

    def initial_history(self, shape=(), *, dtype=torch.float64, device=None) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=device)

    # -- core update ---------------------------------------------------------

    def effective_separation(self, separation: torch.Tensor) -> torch.Tensor:
        """``a = sqrt(<d_n>^2 + d_s1^2 + d_s2^2)``."""
        dn = torch.clamp(separation[..., 0], min=0.0)
        ds = separation[..., 1] ** 2 + separation[..., 2] ** 2
        return torch.sqrt(dn**2 + ds)

    def update(self, separation: torch.Tensor, history: torch.Tensor | None = None) -> Cohesive3DResponse:
        """Inelastic update at one (or many) integration points.

        ``separation`` has shape ``(..., 3)``; ``history`` holds the maximum
        effective separation reached so far (same leading shape).
        """
        sep = torch.as_tensor(separation)
        if sep.shape[-1] != 3:
            raise ValueError("separation must have trailing dimension 3 (n, s1, s2)")
        dtype, device = sep.dtype, sep.device
        if history is None:
            hist_old = self.initial_history(sep.shape[:-1], dtype=dtype, device=device)
        else:
            hist_old = torch.as_tensor(history, dtype=dtype, device=device).expand(sep.shape[:-1])

        delta_n = sep[..., 0]
        delta_t = sep[..., 1:]
        dn_pos = torch.clamp(delta_n, min=0.0)
        dn_neg = torch.clamp(delta_n, max=0.0)
        a = self.effective_separation(sep)
        tiny = torch.finfo(dtype).tiny
        a_safe = torch.clamp(a, min=tiny)

        # Work-conjugate direction in the (n, s1, s2) frame.  A zero separation
        # (undamaged, unloaded point) has no direction; substitute an arbitrary
        # unit vector -- every quantity below vanishes by delta_eff == 0.
        null_sep = a <= 0
        cx = torch.where(null_sep, torch.ones_like(a), dn_pos / a_safe)
        cy = torch.where(null_sep, torch.zeros_like(a), delta_t[..., 0] / a_safe)
        cz = torch.where(null_sep, torch.zeros_like(a), delta_t[..., 1] / a_safe)
        cs2 = cy**2 + cz**2
        c = torch.stack((cx, cy, cz), dim=-1)
        work_stiffness = self.kn * cx**2 + self.ks * cs2
        gc = self.critical_energy(c)
        onset = self.onset_amplitude(c)
        failure = 2.0 * gc / torch.clamp(work_stiffness * onset, min=tiny)
        if bool(torch.any(failure <= onset)):
            raise ValueError("fracture energy is too small for the requested mixed-mode direction")

        hist_new = torch.maximum(hist_old, a)
        # Monotonic envelope value at a given amplitude.
        def envelope(x: torch.Tensor) -> torch.Tensor:
            elastic = x
            softening = onset * (failure - x) / (failure - onset)
            return torch.where(x <= onset, elastic, torch.where(x < failure, softening, torch.zeros_like(x)))

        g_mono = envelope(torch.clamp(a, min=0.0))
        h_load = torch.where(
            a <= onset,
            torch.ones_like(a),
            torch.where(a < failure, g_mono / torch.clamp(a, min=tiny), torch.zeros_like(a)),
        )
        # d h / d a on the loading branch:  h = a0 (a_f - a) / (a (a_f - a0))
        #                                 -> h' = -a0 a_f / ((a_f - a0) a^2)
        h_prime = torch.where(
            (a > onset) & (a < failure),
            -(onset / (failure - onset)) * failure / a_safe**2,
            torch.zeros_like(a),
        )
        g_hist = envelope(hist_new)
        h_secant = g_hist / torch.clamp(hist_new, min=tiny)

        loading = a >= hist_old
        h = torch.where(loading, h_load, h_secant)
        hp = torch.where(loading, h_prime, torch.zeros_like(a))

        k = torch.stack(
            (
                torch.full_like(a, self.kn),
                torch.full_like(a, self.ks),
                torch.full_like(a, self.ks),
            ),
            dim=-1,
        )
        delta_eff = torch.stack((dn_pos, delta_t[..., 0], delta_t[..., 1]), dim=-1)
        traction = k * delta_eff * h[..., None]
        traction = traction + torch.stack((self.kn * dn_neg, torch.zeros_like(a), torch.zeros_like(a)), dim=-1)

        # Exact consistent tangent:
        #   dt_i/ddelta_m = J_i k_i h delta_im + k_i delta_eff_i dh/ddelta_m + kn (1-J_0) delta_i0
        # with dh/ddelta_m = h' c_m + S . (dc/ddelta_m), where S collects the
        # sensitivity of the envelope to the mode-mix direction c (the onset
        # amplitude, the failure amplitude and BK energy all depend on c, and
        # c depends on delta, so a Newton tangent that freezes the envelope is
        # *not* the derivative of the discrete law -- these terms are).
        tension = (delta_n > 0).to(dtype)
        jac_clamp = torch.stack((tension, torch.ones_like(tension), torch.ones_like(tension)), dim=-1)
        tangent = torch.diag_embed(k * h * jac_clamp)
        coef = (k * delta_eff) * hp
        tangent = tangent + coef[..., :, None] * c[..., None, :]

        # -- envelope sensitivity to the mode mix (softening branch only) ----
        softening = loading & (a > onset) & (a < failure)
        if bool(torch.any(softening)):
            cs = torch.sqrt(torch.clamp(cs2, min=0.0))
            dn0 = self.tn0 / self.kn
            ds0 = self.ts0 / self.ks
            sinv = torch.where(cs > 0, 1.0 / torch.clamp(cs, min=tiny), torch.zeros_like(cs))
            # d onset / d c
            dO = torch.stack(
                (
                    -onset**3 * cx / dn0**2,
                    -onset**3 * cy / ds0**2,
                    -onset**3 * cz / ds0**2,
                ),
                dim=-1,
            )
            # BK energy derivative: f = ks cs^2 / kw, Gc = gic + (giic-gic) f^eta
            frac = torch.clamp(self.ks * cs2 / torch.clamp(work_stiffness, min=tiny), min=0.0, max=1.0)
            dfrac = torch.stack(
                (
                    -2.0 * self.kn * cx * frac / torch.clamp(work_stiffness, min=tiny),
                    2.0 * self.ks * cy * (1.0 - frac) / torch.clamp(work_stiffness, min=tiny),
                    2.0 * self.ks * cz * (1.0 - frac) / torch.clamp(work_stiffness, min=tiny),
                ),
                dim=-1,
            )
            dgc = (self.giic - self.gic) * self.eta * torch.clamp(frac, min=tiny) ** (self.eta - 1.0) * dfrac
            dkw = torch.stack(
                (2.0 * self.kn * cx, 2.0 * self.ks * cy, 2.0 * self.ks * cz), dim=-1
            )
            dF = (
                2.0 * dgc / torch.clamp(work_stiffness * onset, min=tiny)
                - failure[..., None] * dkw / torch.clamp(work_stiffness, min=tiny)[..., None]
                - (failure / onset)[..., None] * dO
            )
            h_O = h / onset + h / (failure - onset)
            h_F = onset * (a - onset) / (a * (failure - onset) ** 2)
            # S = h_O dO/dc + h_F dF/dc  (only meaningful where softening)
            sens = h_O[..., None] * dO + h_F[..., None] * dF
            sens = torch.where(softening[..., None], sens, torch.zeros_like(sens))
            diag_j = torch.diag_embed(jac_clamp)
            dc_ddelta = (diag_j - c[..., :, None] * c[..., None, :]) / a_safe[..., None, None]
            # (dc/ddelta)_mi S_i  -> contract over the c index
            dh_extra = torch.einsum("...mi,...i->...m", dc_ddelta, sens)
            tangent = tangent + (k * delta_eff)[..., :, None] * dh_extra[..., None, :]
        compression = torch.diag_embed(
            torch.stack(
                (
                    self.kn * (1.0 - tension),
                    torch.zeros_like(tension),
                    torch.zeros_like(tension),
                ),
                dim=-1,
            )
        )
        tangent = tangent + compression
        damage = torch.clamp(1.0 - h, min=0.0, max=1.0)
        return Cohesive3DResponse(traction, tangent, damage, hist_new)

    def envelope_traction(self, separation: torch.Tensor) -> torch.Tensor:
        """Monotonic (virgin loading) traction for curve oracles -- no history."""
        sep = torch.as_tensor(separation)
        if sep.shape[-1] != 3:
            raise ValueError("separation must have trailing dimension 3 (n, s1, s2)")
        a = self.effective_separation(sep)
        tiny = torch.finfo(sep.dtype).tiny
        a_safe = torch.clamp(a, min=tiny)
        delta_eff = torch.stack(
            (torch.clamp(sep[..., 0], min=0.0), sep[..., 1], sep[..., 2]), dim=-1
        )
        null_sep = a <= 0
        c = torch.where(
            null_sep[..., None],
            torch.zeros_like(delta_eff) + torch.tensor([1.0, 0.0, 0.0], dtype=sep.dtype, device=sep.device),
            delta_eff / a_safe[..., None],
        )
        onset = self.onset_amplitude(c)
        failure = self.failure_amplitude(c)
        elastic = a
        softening = onset * (failure - a) / (failure - onset)
        g = torch.where(a <= onset, elastic, torch.where(a < failure, softening, torch.zeros_like(a)))
        h = g / a_safe
        k = torch.stack(
            (torch.full_like(a, self.kn), torch.full_like(a, self.ks), torch.full_like(a, self.ks)), dim=-1
        )
        return k * delta_eff * h[..., None]


# --------------------------------------------------------------------------- #
# Element formulation: 8-node zero-thickness cohesive element (COH3D8 topology)
# --------------------------------------------------------------------------- #

_GAUSS_2X2 = (
    (-1.0 / 3.0**0.5, -1.0 / 3.0**0.5),
    (1.0 / 3.0**0.5, -1.0 / 3.0**0.5),
    (1.0 / 3.0**0.5, 1.0 / 3.0**0.5),
    (-1.0 / 3.0**0.5, 1.0 / 3.0**0.5),
)


def cohesive_hex8_quadrature(*, dtype=torch.float64, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """2x2 Gauss points and weights on the mid-surface ``[-1,1]^2``."""
    points = torch.tensor(_GAUSS_2X2, dtype=dtype, device=device)
    weights = torch.ones(4, dtype=dtype, device=device)
    return points, weights


def cohesive_shape_functions(xi: float, eta: float, *, dtype=torch.float64, device=None):
    """Bilinear shape functions and their natural-coordinate derivatives.

    Node order (COH3D8): 0=(−1,−1), 1=(+1,−1), 2=(+1,+1), 3=(−1,+1).
    """
    xi_i = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=dtype, device=device)
    eta_i = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=dtype, device=device)
    n = 0.25 * (1.0 + xi * xi_i) * (1.0 + eta * eta_i)
    dn_dxi = 0.25 * xi_i * (1.0 + eta * eta_i)
    dn_deta = 0.25 * eta_i * (1.0 + xi * xi_i)
    return n, dn_dxi, dn_deta


def mid_surface_frame(coordinates: torch.Tensor, *, dtype=None, device=None):
    """Mid-surface frame at a Gauss point.

    Parameters
    ----------
    coordinates : ``(8, 3)`` element node coordinates, nodes 0--3 on face −,
                 nodes 4--7 on face +.
    Returns
    -------
    normal, e1, e2 : orthonormal interface triad (rows of the rotation ``Q``).
    jacobian : ``|t_xi x t_eta|`` at the Gauss point (area density).
    n : shape functions at the Gauss point.
    mid : mid-surface coordinates of the four corners.
    """
    dtype = dtype or coordinates.dtype
    device = device or coordinates.device
    coords = torch.as_tensor(coordinates, dtype=dtype, device=device)
    if coords.shape != (8, 3):
        raise ValueError("cohesive_hex8_response expects 8 node coordinates of shape (8,3)")
    bottom, top = coords[0:4], coords[4:8]
    mid = 0.5 * (bottom + top)
    points, weights = cohesive_hex8_quadrature(dtype=dtype, device=device)
    frames = []
    for gp in range(points.shape[0]):
        xi, eta = float(points[gp, 0]), float(points[gp, 1])
        n, dn_dxi, dn_deta = cohesive_shape_functions(xi, eta, dtype=dtype, device=device)
        t_xi = dn_dxi @ mid
        t_eta = dn_deta @ mid
        cross = torch.linalg.cross(t_xi, t_eta)
        jac = torch.linalg.vector_norm(cross)
        if float(jac) <= torch.finfo(dtype).eps:
            raise ValueError("degenerate cohesive element: zero mid-surface area")
        normal = cross / jac
        e1 = t_xi / torch.linalg.vector_norm(t_xi)
        e2 = torch.linalg.cross(normal, e1)
        e2 = e2 / torch.linalg.vector_norm(e2)
        frames.append((normal, e1, e2, jac, n))
    return frames, weights, mid


def interface_area(coordinates: torch.Tensor, *, dtype=None, device=None) -> torch.Tensor:
    """Total mid-surface area of one cohesive element."""
    frames, weights, _ = mid_surface_frame(coordinates, dtype=dtype, device=device)
    area = torch.zeros((), dtype=weights.dtype, device=weights.device)
    for (_, _, _, jac, _), w in zip(frames, weights):
        area = area + w * jac
    return area


def cohesive_hex8_response(
    coordinates: torch.Tensor,
    displacements: torch.Tensor,
    law: BKCohesive3DLaw,
    history: torch.Tensor | None = None,
):
    """Internal force and consistent tangent of one 8-node cohesive element.

    Parameters
    ----------
    coordinates : ``(8, 3)`` reference/current node coordinates.
    displacements : ``(8, 3)`` nodal displacements (small strain).
    law : :class:`BKCohesive3DLaw`.
    history : optional ``(4,)`` per-Gauss-point maximum effective separation.

    Returns
    -------
    force : ``(8, 3)`` nodal internal forces.
    tangent : ``(24, 24)`` consistent tangent (ordering ``node*3 + comp``).
    history : ``(4,)`` updated history.
    diagnostics : dict with per-Gauss-point ``separation`` (3,), ``traction``
        (3,), ``area`` (scalar), ``damage`` (scalar), ``normal`` (3,).
    """
    dtype = coordinates.dtype
    device = coordinates.device
    frames, weights, _ = mid_surface_frame(coordinates, dtype=dtype, device=device)
    if history is None:
        history = law.initial_history((len(frames),), dtype=dtype, device=device)
    history = torch.as_tensor(history, dtype=dtype, device=device)
    if history.shape != (len(frames),):
        raise ValueError("history must have one entry per Gauss point")

    force = torch.zeros((8, 3), dtype=dtype, device=device)
    tangent = torch.zeros((24, 24), dtype=dtype, device=device)
    separations = []
    tractions = []
    areas = []
    damages = []
    new_history = []
    normals = []

    for gp, ((normal, e1, e2, jac, n), w) in enumerate(zip(frames, weights)):
        # Rows of the rotation are the local (n, s1, s2) axes; (n, e1, e2) is a
        # cyclic permutation of the right-handed triad (e1, e2, n), hence det=+1.
        rotation = torch.stack((normal, e1, e2))  # local = rotation @ global
        jump = displacements[4:8] - displacements[0:4]  # (4,3)
        # local separation at this Gauss point
        local_jump = jump @ rotation.T  # (4,3)
        separation = torch.einsum("a,ab->b", n, local_jump)
        response = law.update(separation, history[gp])
        area = w * jac
        # B matrix: 3 x 24
        b = torch.zeros((3, 24), dtype=dtype, device=device)
        for a in range(4):
            block = n[a] * rotation  # 3x3
            b[:, 3 * a : 3 * a + 3] = -block
            b[:, 3 * (a + 4) : 3 * (a + 4) + 3] = block
        force = force + (b.T @ response.traction).reshape(8, 3) * area
        tangent = tangent + area * (b.T @ response.tangent @ b)
        separations.append(separation)
        tractions.append(response.traction)
        areas.append(area)
        damages.append(response.damage)
        normals.append(normal)
        new_history.append(response.history)

    diagnostics = {
        "separation": torch.stack(separations),
        "traction": torch.stack(tractions),
        "area": torch.stack(areas),
        "damage": torch.stack([d.reshape(()) for d in damages]),
        "normal": torch.stack(normals),
    }
    return force, tangent, torch.stack([h.reshape(()) for h in new_history]), diagnostics


def assemble_cohesive_interface(
    coordinates: torch.Tensor,
    connectivity: torch.Tensor,
    displacements: torch.Tensor,
    law: BKCohesive3DLaw,
    histories: torch.Tensor | None = None,
    *,
    n_dofs: int | None = None,
):
    """Dense assembly of a pre-meshed interface patch.

    Parameters
    ----------
    coordinates : ``(n_node, 3)`` global node coordinates.
    connectivity : ``(n_elem, 8)`` cohesive element node indices (global).
    displacements : ``(n_node, 3)`` or ``(n_dofs,)`` global displacement vector.
    histories : optional ``(n_elem, 4)``.
    """
    dtype = coordinates.dtype
    device = coordinates.device
    n_node = coordinates.shape[0]
    if displacements.ndim == 1:
        if displacements.numel() != 3 * n_node:
            raise ValueError("displacement vector size must be 3 * n_node")
        u = displacements.reshape(n_node, 3)
    else:
        u = displacements
    n_dofs = n_dofs or 3 * n_node
    force = torch.zeros(n_dofs, dtype=dtype, device=device)
    tangent = torch.zeros((n_dofs, n_dofs), dtype=dtype, device=device)
    new_histories = []
    for e, conn in enumerate(connectivity):
        conn = conn.to(torch.long)
        coords = coordinates[conn]
        elem_u = u[conn]
        hist = None if histories is None else histories[e]
        f, k, hist_new, _ = cohesive_hex8_response(coords, elem_u, law, hist)
        dofs = (3 * conn[:, None] + torch.arange(3, device=device)[None, :]).reshape(-1)
        force = force.index_add(0, dofs, f.reshape(-1))
        tangent = tangent.index_put(
            (dofs[:, None].expand(-1, dofs.numel()), dofs[None, :].expand(dofs.numel(), -1)),
            k,
            accumulate=True,
        )
        new_histories.append(hist_new)
    histories_out = torch.stack(new_histories) if new_histories else None
    return force, tangent, histories_out