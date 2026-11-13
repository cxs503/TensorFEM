"""Contact-pressure patch loads on hull panels (ice-structure contact patch).

Gap closed by this module (``GAP-PP-1-ICE-STRUCT-CONTACT-PATCH``)
-----------------------------------------------------------------
Ice-structure analyses in this repository produced a *contact* solution (a
normal pressure distribution on the ice/structure interface) and a *panel*
solution (a Mindlin plate), but nothing connected the two: the panel was
always driven by a prescribed, uniform pressure.  This module supplies the
missing transfer

    contact pressure field  p(x_i, y_i)   (from a damageable crush-interface
                                           solve on a *non-matching* contact
                                           grid)
        -> consistent panel load array    f = C^T (w * p)
        -> Mindlin plate solve            K u = f
        -> surface stress                  sigma = 6 |M| / t^2
        -> back-coupling                   delta_i = d - g_i - (C u)_i

and the two-way fixed point that closes the loop (the contact pressure depends
on the *deformed* panel surface, the panel load depends on the contact
pressure).

Conventions
-----------
The panel mid-surface is ``[0, span_x] x [0, span_y]``; node ``(i, j)`` of the
structured Q4 mesh lives at index ``j*(nx+1)+i`` and each node carries three
degrees of freedom ``(w, theta_x, theta_y)`` -- the ordering of
:func:`tensorfem.plate.q4_mindlin_stiffness`, which is reused unchanged here.
A positive load acts along ``+z`` and produces positive ``w``; the curvature
convention of that element is ``kappa_x = d theta_x / d x``, so the *sagging*
moment is ``M = -Db kappa``.  Every exported moment already carries that sign
convention (the FE recovery and the series reference use the same branch), and
``sigma_peak`` uses magnitudes.

Two transfer operators, one duality
-----------------------------------
Both directions of the coupling use the *same* bilinear interpolation operator
``C`` (shape functions of the panel element containing each contact point) and
the contact quadrature weights ``w_i`` (contact cell areas):

    w_points = C u                      (panel surface seen by the ice)
    f        = C^T (w * p)              (consistent panel load array)

so the pair is work conjugate, ``p . (C u) = (C^T (w p)) . u``, and the load
array conserves the total contact force exactly,
``sum(f) = sum_i w_i p_i``.  Both identities are asserted in
``tests/test_contact_patch_load.py``.  :func:`patch_from_contact` is the public
entry point; it returns the panel load array together with the pressure field
it was built from and the non-uniformity metrics (peak/mean, coefficient of
variation) that the peak-stress sweep needs.

Verification references
-----------------------
* Navier double-series solution for an *arbitrary* pressure field on a simply
  supported rectangle (:func:`navier_series_reference`) -- the exact solution
  of the same boundary value problem for any integrable load, including the
  patch and ridge loads used in the benchmark.
* Timoshenko & Woinowsky-Krieger (1959) coefficients for the uniformly loaded
  simply supported square: ``w_c = 0.00406235 q a^4 / D`` and
  ``M_max = 0.0479 q a^2`` (nu = 0.3), used as an external literature anchor
  for both the element path and the series implementation.

Honest limitations
------------------
* :class:`CrushInterfaceLaw` is a 1-D damageable compression law
  (elastic - linear softening - residual plateau) applied pointwise on the
  contact grid.  It is *not* a calibrated rate/temperature dependent sea-ice
  constitutive model, and the residual plateau is a homogenised stand-in for
  continuous crushing and extrusion.  Peak traction and softening ratio are
  inputs, exactly like the compressive strength used elsewhere in the
  repository.
* The panel is linear elastic Mindlin (small deflection).  Panel inertia and
  the water-backed added mass are not in the loop, so the coupled solve is
  quasi-static; the load level is scenario dependent and the certified
  quantities are the *transfer* identities, the FE-versus-series errors, and
  the *relative* non-uniformity amplification -- not an absolute ice load.
* Contact points must lie inside the panel bay (a point outside raises
  ``ValueError``), so an ice footprint overhanging the panel is not
  represented.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .ice_panel_fsi import rect_mesh
from .plate import _shape, q4_mindlin_stiffness

D = torch.float64


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _as_points(points) -> torch.Tensor:
    pts = torch.as_tensor(points, dtype=D)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    if pts.shape[0] < 1:
        raise ValueError("points must not be empty")
    return pts


def _inverse_bilinear(xy: torch.Tensor, point: torch.Tensor,
                      iterations: int = 12) -> torch.Tensor:
    """Natural coordinates of ``point`` in the Q4 element ``xy`` (Newton)."""
    xi = torch.zeros(2, dtype=D)
    for _ in range(iterations):
        N, dN_nat = _shape(xi[0], xi[1])
        res = N @ xy - point
        if float(torch.max(torch.abs(res))) < 1.0e-13:
            break
        J = dN_nat.T @ xy
        if float(torch.linalg.det(J)) <= 0.0:
            raise ValueError("plate element has non-positive Jacobian")
        xi = xi - torch.linalg.solve(J, res)
    return xi


def plate_contact_interpolation(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    points,
    point_weights=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation operator from panel nodes to contact points.

    Returns ``(C, weights)`` with ``C`` of shape ``(n_point, n_node)`` and
    ``weights`` the contact quadrature areas.  ``C`` holds the Q4 shape
    function row of the element containing each point, scattered into the
    global node numbering; it is the *same* operator used in both directions
    of the coupling (see the module docstring).  A point that lies outside
    every element raises ``ValueError`` -- silently clipping an ice footprint
    that overhangs the panel would fabricate a load.

    Element search is a vectorised bounding-box rejection followed by a Newton
    inversion of the bilinear map (numpy, so the cost is independent of the
    panel size to first order: the search is a handful of array operations for
    the whole point set).
    """
    nodes = torch.as_tensor(nodes, dtype=D)
    elements = torch.as_tensor(elements, dtype=torch.long)
    pts = _as_points(points)
    nodes_np = nodes.numpy()
    elem_np = elements.numpy()
    pts_np = pts.numpy()
    n_pt = pts_np.shape[0]
    n_node = nodes_np.shape[0]
    xy_all = nodes_np[elem_np]                      # (E, 4, 2)
    lo = xy_all.min(axis=1) - 1e-12                 # (E, 2)
    hi = xy_all.max(axis=1) + 1e-12
    C = np.zeros((n_pt, n_node))
    for k in range(n_pt):
        p = pts_np[k]
        cand = np.nonzero(np.all(p >= lo, axis=1) & np.all(p <= hi, axis=1))[0]
        hit = None
        for e in cand:
            xy = xy_all[e]
            xi = np.zeros(2)
            ok = False
            for _ in range(20):
                N, dN_nat = _shape(torch.as_tensor(xi[0]), torch.as_tensor(xi[1]))
                res = N.numpy() @ xy - p
                if np.max(np.abs(res)) < 1e-14:
                    ok = True
                    break
                J = dN_nat.numpy().T @ xy
                if np.linalg.det(J) <= 0:
                    raise ValueError("plate element has non-positive Jacobian")
                xi = xi - np.linalg.solve(J, res)
            N, _ = _shape(torch.as_tensor(xi[0]), torch.as_tensor(xi[1]))
            if np.max(np.abs(xi)) <= 1.0 + 1e-8 or ok:
                hit = (elem_np[e], N.numpy())
                break
        if hit is None:
            raise ValueError(
                f"contact point {k} at {pts[k].tolist()} is outside the panel")
        conn, N = hit
        C[k, conn] = N
    if point_weights is None:
        area = 0.0
        for e in range(elem_np.shape[0]):
            xs, ys = xy_all[e, :, 0], xy_all[e, :, 1]
            area += abs(float(np.dot(xs, np.roll(ys, -1))
                              - np.dot(ys, np.roll(xs, -1)))) * 0.5
        weights = torch.full((n_pt,), area / n_pt, dtype=D)
    else:
        weights = torch.as_tensor(point_weights, dtype=D)
        if weights.shape != (n_pt,):
            raise ValueError("point_weights must have one entry per point")
        if bool(torch.any(weights <= 0)):
            raise ValueError("point_weights must be positive")
    return torch.as_tensor(C, dtype=D), weights


# --------------------------------------------------------------------------
# the contact -> panel load transfer
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchLoad:
    """Panel load array produced by :func:`patch_from_contact`.

    ``forces_z`` is the nodal transverse force array (``n_node``),
    ``dof_load`` the same array scattered onto the ``(w, theta_x, theta_y)``
    layout of the plate element (rotations receive nothing: a normal pressure
    is a pure force).  ``interpolation``/``weights`` are the transfer operator
    and quadrature weights that produced it (kept so the coupled iteration can
    reuse them and so the work-conjugacy identity can be checked).
    """

    points: torch.Tensor
    pressures: torch.Tensor
    weights: torch.Tensor
    interpolation: torch.Tensor
    forces_z: torch.Tensor
    dof_load: torch.Tensor
    total_force: float
    contact_area: float

    @property
    def n_node(self) -> int:
        return int(self.forces_z.numel())

    @property
    def mean_pressure(self) -> float:
        return float(self.total_force / self.contact_area)

    @property
    def peak_pressure(self) -> float:
        return float(torch.max(self.pressures))

    @property
    def peak_to_mean(self) -> float:
        return self.peak_pressure / self.mean_pressure

    @property
    def coefficient_of_variation(self) -> float:
        w = self.weights / torch.sum(self.weights)
        mean = float(torch.sum(w * self.pressures))
        var = float(torch.sum(w * (self.pressures - mean) ** 2))
        return math.sqrt(max(var, 0.0)) / mean

    def uniform_equivalent(self) -> "PatchLoad":
        """Same total force spread uniformly over the same contact footprint."""
        return patch_from_contact(
            self.points,
            torch.full_like(self.pressures, self.mean_pressure),
            point_weights=self.weights,
            interpolation=self.interpolation,
            n_node=self.n_node,
        )


def patch_from_contact(
    points,
    pressures,
    nodes=None,
    elements=None,
    *,
    point_weights=None,
    interpolation: torch.Tensor | None = None,
    n_node: int | None = None,
) -> PatchLoad:
    """Project a contact normal-pressure field onto the panel load array.

    Parameters
    ----------
    points : ``(n_point, 2)`` contact-point coordinates on the panel surface
        (the grid a contact / crush-interface solve reports its tractions on;
        it does *not* have to match the panel mesh).
    pressures : ``(n_point,)`` normal contact pressure, positive along ``+z``
        (positive pressure pushes the panel along ``+z``; separation is 0).
    nodes, elements : panel mesh, required unless ``interpolation`` is given.
    point_weights : contact quadrature areas (``n_point``); defaults to a
        uniform share of the panel area.
    interpolation : pre-computed operator from
        :func:`plate_contact_interpolation` (reused by the coupled iteration).

    Returns
    -------
    :class:`PatchLoad`

    Notes
    -----
    The load array is ``f = C^T (w * p)``.  With ``C`` the partition of unity
    of the panel shape functions this is the consistent (integrating) nodal
    load of the piecewise-bilinear pressure field, so ``sum(f) = sum(w * p)``
    holds exactly and the load array is precisely what
    :func:`solve_plate_patch_load` assembles element by element.
    """
    pts = _as_points(points)
    p = torch.as_tensor(pressures, dtype=D).reshape(-1)
    if p.shape[0] != pts.shape[0]:
        raise ValueError("pressures must have one entry per point")
    if bool(torch.any(p < 0)):
        raise ValueError(
            "contact pressures must be non-negative (separation is p = 0)")
    if interpolation is None:
        if nodes is None or elements is None:
            raise ValueError("either (nodes, elements) or interpolation must be given")
        C, weights = plate_contact_interpolation(nodes, elements, pts, point_weights)
    else:
        C = torch.as_tensor(interpolation, dtype=D)
        if C.shape[0] != pts.shape[0]:
            raise ValueError("interpolation operator does not match the point set")
        if point_weights is None:
            raise ValueError("point_weights are required together with interpolation")
        weights = torch.as_tensor(point_weights, dtype=D)
        if weights.shape != (pts.shape[0],):
            raise ValueError("point_weights must have one entry per point")
    forces = C.T @ (weights * p)
    nn = n_node if n_node is not None else int(forces.numel())
    if int(forces.numel()) != int(nn):
        raise ValueError("n_node does not match the interpolation operator")
    dof = torch.zeros(3 * int(nn), dtype=D)
    dof[0::3] = forces
    return PatchLoad(
        points=pts, pressures=p, weights=weights, interpolation=C,
        forces_z=forces, dof_load=dof,
        total_force=float(torch.sum(weights * p)),
        contact_area=float(torch.sum(weights)),
    )# --------------------------------------------------------------------------
# Mindlin panel solve with an arbitrary load array
# --------------------------------------------------------------------------
@dataclass
class PlateResponse:
    """Linear Mindlin panel response to an arbitrary nodal load array."""

    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    thickness: float
    boundary: str
    displacement: torch.Tensor          # (3*n_node,)
    reactions_z: torch.Tensor           # (n_node,)
    gauss_xy: torch.Tensor              # (4*n_elem, 2) Gauss-point coordinates
    gauss_moments: torch.Tensor         # (4*n_elem, 3) sagging (Mx, My, Mxy)
    sigma_peak: float                   # 6*max|principal moment| / t^2
    m_peak: float                       # max |principal bending moment|
    strain_energy: float
    external_work: float

    @property
    def w(self) -> torch.Tensor:
        return self.displacement[0::3]

    @property
    def centre_deflection(self) -> float:
        span_x = float(self.nodes[:, 0].max())
        span_y = float(self.nodes[:, 1].max())
        tol = 1.0e-9 * max(span_x, span_y)
        mask = (torch.abs(self.nodes[:, 0] - 0.5 * span_x) < tol) & (
            torch.abs(self.nodes[:, 1] - 0.5 * span_y) < tol)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() != 1:
            raise ValueError("panel mesh must contain exactly one centre node")
        return float(self.displacement[3 * int(idx[0])])


def _boundary_dofs(nodes: torch.Tensor, boundary: str) -> torch.Tensor:
    span_x = float(nodes[:, 0].max())
    span_y = float(nodes[:, 1].max())
    tol = 1.0e-9 * max(span_x, span_y)
    xedge = (torch.abs(nodes[:, 0]) < tol) | (torch.abs(nodes[:, 0] - span_x) < tol)
    yedge = (torch.abs(nodes[:, 1]) < tol) | (torch.abs(nodes[:, 1] - span_y) < tol)
    if boundary == "simply_supported":
        # hard simple support: w = 0 on the whole edge and the tangential
        # rotation = 0 (the convention of plate.solve_simply_supported_sine_square)
        dofs = torch.cat((
            3 * torch.nonzero(xedge | yedge, as_tuple=False).flatten(),
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        ))
    elif boundary == "clamped":
        idx = torch.nonzero(xedge | yedge, as_tuple=False).flatten()
        dofs = torch.cat((3 * idx, 3 * idx + 1, 3 * idx + 2))
    else:
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    return torch.unique(dofs)


class PlateModel:
    """Pre-assembled, pre-factored Q4 Mindlin panel.

    The stiffness matrix, the boundary conditions and the Gauss-point recovery
    operators are built once; every :meth:`solve` call then only assembles the
    load array, does one triangular solve and one vectorised moment recovery.
    The two-way coupling needs tens of panel solves on an unchanged mesh, which
    is what this class exists for (it is what makes a converged, mesh-refined
    coupled sweep affordable).

    The element is :func:`tensorfem.plate.q4_mindlin_stiffness` (selective
    reduced integration) -- no new element formulation is introduced here.
    """

    def __init__(
        self,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        *,
        young: float,
        poisson: float,
        thickness: float,
        boundary: str = "simply_supported",
    ) -> None:
        nodes = torch.as_tensor(nodes, dtype=D)
        elements = torch.as_tensor(elements, dtype=torch.long)
        for name, value in (("young", young), ("thickness", thickness)):
            if not float(value) > 0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 < float(poisson) < 0.5:
            raise ValueError("poisson out of range")
        self.nodes = nodes
        self.elements = elements
        self.young = float(young)
        self.poisson = float(poisson)
        self.thickness = float(thickness)
        self.boundary = boundary
        n_elem = int(elements.shape[0])
        n_node = int(nodes.shape[0])
        ndof = 3 * n_node
        self.ndof = ndof

        K = torch.zeros((ndof, ndof), dtype=D)
        for conn in elements:
            ke = q4_mindlin_stiffness(nodes[conn], self.young, self.poisson,
                                     self.thickness)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2),
                               dim=1).reshape(-1)
            K[dofs[:, None], dofs] += ke
        self.K = K
        fixed = _boundary_dofs(nodes, boundary)
        all_dofs = torch.arange(ndof)
        self.fixed = fixed
        self.free = all_dofs[~torch.isin(all_dofs, fixed)]
        self._factor = torch.linalg.cholesky(
            K[self.free[:, None], self.free])

        # --- recovery operators, built once for all elements and Gauss points
        self._Db = self.young * self.thickness ** 3 / (
            12.0 * (1.0 - self.poisson ** 2)) * torch.tensor(
                [[1.0, self.poisson, 0.0],
                 [self.poisson, 1.0, 0.0],
                 [0.0, 0.0, (1.0 - self.poisson) / 2.0]], dtype=D)
        dof_index = torch.stack(
            (3 * elements, 3 * elements + 1, 3 * elements + 2),
            dim=2).reshape(n_elem, 12)
        self._dof_index = dof_index
        # The shape functions live in reference coordinates, so the four Gauss
        # points contribute element-independent N and dN_nat; only the Jacobian
        # is per element.  Everything below is therefore one batched einsum
        # instead of a Python loop over elements x Gauss points.
        gp = 1.0 / 3.0**0.5
        gp_list = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        N_gp = torch.zeros((4, 4), dtype=D)
        dN_gp = torch.zeros((4, 4, 2), dtype=D)
        for g, (xif, etaf) in enumerate(gp_list):
            N, dN_nat = _shape(nodes.new_tensor(xif), nodes.new_tensor(etaf))
            N_gp[g] = N
            dN_gp[g] = dN_nat
        xy_elem = nodes[elements]                                # (E, 4, 2)
        J = torch.einsum("gik,eic->egkc", dN_gp, xy_elem)        # (E, 4, 2, 2)
        if bool(torch.any(torch.linalg.det(J) <= 0)):
            raise ValueError("plate element has non-positive Jacobian")
        Jinv = torch.linalg.inv(J)
        dN = torch.einsum("gik,egjk->egij", dN_gp, Jinv)         # (E, 4, 4, 2)
        Bb = torch.zeros((n_elem, 4, 3, 12), dtype=D)
        for i in range(4):
            Bb[:, :, 0, 3 * i + 1] = dN[:, :, i, 0]
            Bb[:, :, 1, 3 * i + 2] = dN[:, :, i, 1]
            Bb[:, :, 2, 3 * i + 1] = dN[:, :, i, 1]
            Bb[:, :, 2, 3 * i + 2] = dN[:, :, i, 0]
        self._N_gp = N_gp
        self._Bb = Bb                                            # (E, 4, 3, 12)
        self._gauss_xy = torch.einsum("gk,ekc->egc", N_gp, xy_elem)  # (E, 4, 2)

    def solve(self, *, forces_z=None, dof_load=None) -> PlateResponse:
        """Solve for the given load array and recover the response."""
        f = torch.zeros(self.ndof, dtype=D)
        if dof_load is not None:
            if torch.as_tensor(dof_load).shape != (self.ndof,):
                raise ValueError("dof_load must have 3 entries per node")
            f = f + torch.as_tensor(dof_load, dtype=D)
        elif forces_z is not None:
            if torch.as_tensor(forces_z).shape != (self.ndof // 3,):
                raise ValueError("forces_z must have one entry per node")
            f[0::3] = torch.as_tensor(forces_z, dtype=D)
        else:
            raise ValueError("give forces_z or dof_load")
        u = torch.zeros(self.ndof, dtype=D)
        u[self.free] = torch.cholesky_solve(
            f[self.free][:, None], self._factor).reshape(-1)
        reactions = self.K @ u - f

        u_e = u[self._dof_index]                             # (E, 12)
        kappa = torch.einsum("egij,ej->egi", self._Bb, u_e)   # (E, 4, 3)
        # The element rotation dofs are theta = -w,x (its transverse shear is
        # gamma = w,x - theta), so a sagging curvature is positive here and the
        # sagging moment is +Db kappa -- the branch used by the Navier series.
        moments = kappa @ self._Db.T
        moments = moments.reshape(-1, 3)
        gauss_xy = self._gauss_xy.reshape(-1, 2)
        m_peak, sigma_peak = peak_bending_stress(moments, self.thickness)
        return PlateResponse(
            nodes=self.nodes, elements=self.elements, young=self.young,
            poisson=self.poisson, thickness=self.thickness,
            boundary=self.boundary, displacement=u,
            reactions_z=reactions[0::3], gauss_xy=gauss_xy,
            gauss_moments=moments, sigma_peak=sigma_peak, m_peak=m_peak,
            strain_energy=float(0.5 * u @ (self.K @ u)),
            external_work=float(0.5 * f @ u),
        )


def solve_plate_patch_load(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    *,
    young: float,
    poisson: float,
    thickness: float,
    forces_z=None,
    dof_load=None,
    boundary: str = "simply_supported",
    model: "PlateModel | None" = None,
) -> PlateResponse:
    """Solve ``K u = f`` for the Q4 Mindlin panel with an arbitrary load array.

    ``forces_z`` (nodal transverse forces) or the full ``dof_load``
    (3 entries per node) may be given.  Pass a pre-built :class:`PlateModel` to
    reuse the factorisation (the coupled iteration does); otherwise the model is
    built for this call.
    """
    if model is None:
        model = PlateModel(nodes, elements, young=young, poisson=poisson,
                           thickness=thickness, boundary=boundary)
    return model.solve(forces_z=forces_z, dof_load=dof_load)


def principal_moments(moments: torch.Tensor) -> torch.Tensor:
    """Principal moments ``(M1, M2)`` of a ``(..., 3)`` ``(Mx, My, Mxy)`` array."""
    mx, my, mxy = moments[..., 0], moments[..., 1], moments[..., 2]
    half = 0.5 * (mx + my)
    rad = torch.sqrt(torch.clamp(0.25 * (mx - my) ** 2 + mxy ** 2, min=0.0))
    return torch.stack((half + rad, half - rad), dim=-1)


def peak_bending_stress(moments: torch.Tensor,
                        thickness: float) -> tuple[float, float]:
    """``(m_peak, sigma_peak)``: largest principal-moment magnitude and the
    surface bending stress ``6 |M| / t^2`` it produces."""
    mp = principal_moments(moments)
    m_peak = float(torch.max(torch.abs(mp)))
    return m_peak, 6.0 * m_peak / thickness ** 2


# --------------------------------------------------------------------------
# exact reference: Navier double series for an arbitrary pressure field
# --------------------------------------------------------------------------
def navier_series_reference(
    span_x: float,
    span_y: float,
    *,
    young: float,
    poisson: float,
    thickness: float,
    pressure_fn,
    points=None,
    modes: int = 120,
    model: str = "mindlin",
    shear_correction: float = 5.0 / 6.0,
    quadrature: int = 16,
    edge_margin: float = 0.05,
) -> dict:
    """Exact Navier double-series solution for a simply supported rectangle.

    ``w(x, y) = sum_mn p_mn sin(m pi x/a) sin(n pi y/b) / (D pi^4 lambda_mn^2)``
    with ``p_mn`` the load coefficients (exact in the Gauss quadrature, since
    the integrand is smooth for every load used here) and the sagging bending
    moments from the same series

        ``Mx = sum_mn p_mn [(m/a)^2 + nu (n/b)^2] / (pi^2 lambda^2) sin sin``

    which is the branch used by :func:`solve_plate_patch_load`.
    ``model="mindlin"`` adds the Reissner transverse-shear deflection
    ``p_mn / (kappa G t pi^2 lambda_mn)`` per mode -- the exact FSDT result for
    this boundary condition; the bending moments are unchanged (for a simply
    supported plate FSDT and Kirchhoff share the bending solution), so a moment
    comparison is valid for both models.

    Returns a dict with ``p_mn``, the field values at ``points`` (or the peak
    over a 61x61 grid when ``points`` is None), and the series centre values.
    """
    if not (span_x > 0 and span_y > 0 and young > 0 and thickness > 0):
        raise ValueError("span, young and thickness must be positive")
    if model not in ("kirchhoff", "mindlin"):
        raise ValueError("model must be 'kirchhoff' or 'mindlin'")
    a, b = float(span_x), float(span_y)
    nu = float(poisson)
    t = float(thickness)
    rigidity = young * t ** 3 / (12.0 * (1.0 - nu ** 2))
    shear_modulus = shear_correction * young / (2.0 * (1.0 + nu)) * t

    gx, wx = np.polynomial.legendre.leggauss(quadrature)
    xq = 0.5 * a * (gx + 1.0)
    yq = 0.5 * b * (gx + 1.0)
    wxq = 0.5 * a * wx
    wyq = 0.5 * b * wx
    xx, yy = np.meshgrid(xq, yq, indexing="ij")
    P = np.asarray(pressure_fn(xx, yy), dtype=float)
    if P.shape != xx.shape:
        raise ValueError("pressure_fn must be vectorised over its arguments")
    m_idx = np.arange(1, modes + 1)
    n_idx = np.arange(1, modes + 1)
    Sm = np.sin(np.pi * np.outer(xq, m_idx) / a) * wxq[:, None]     # (qx, m)
    Sn = np.sin(np.pi * np.outer(yq, n_idx) / b) * wyq[:, None]     # (qy, n)
    p_mn = 4.0 / (a * b) * (Sm.T @ P @ Sn)                          # (m, n)

    lam = (m_idx[:, None] / a) ** 2 + (n_idx[None, :] / b) ** 2
    coef_w = p_mn / (rigidity * math.pi ** 4 * lam ** 2)
    if model == "mindlin":
        coef_w = coef_w + p_mn / (shear_modulus * math.pi ** 2 * lam)
    coef_mx = p_mn * ((m_idx[:, None] / a) ** 2 + nu * (n_idx[None, :] / b) ** 2) \
        / (math.pi ** 2 * lam ** 2)
    coef_my = p_mn * ((n_idx[None, :] / b) ** 2 + nu * (m_idx[:, None] / a) ** 2) \
        / (math.pi ** 2 * lam ** 2)
    coef_mxy = (1.0 - nu) * p_mn * (m_idx[:, None] / a) * (n_idx[None, :] / b) \
        / (math.pi ** 2 * lam ** 2)

    ax = np.pi * m_idx / a
    by = np.pi * n_idx / b

    def _eval(xp, yp):
        """Field on the tensor grid ``xp x yp`` (``w[i, j] = w(xp[i], yp[j])``)."""
        Sx = np.sin(np.outer(xp, ax))
        Sy = np.sin(np.outer(yp, by))
        Cx = np.cos(np.outer(xp, ax))
        Cy = np.cos(np.outer(yp, by))
        w = np.einsum("im,mn,jn->ij", Sx, coef_w, Sy, optimize=True)
        mx = np.einsum("im,mn,jn->ij", Sx, coef_mx, Sy, optimize=True)
        my = np.einsum("im,mn,jn->ij", Sx, coef_my, Sy, optimize=True)
        mxy = np.einsum("im,mn,jn->ij", Cx, coef_mxy, Cy, optimize=True)
        return w, mx, my, mxy

    def _eval_points(xp, yp):
        """Field at point pairs (no cross product)."""
        Sx = np.sin(np.outer(xp, ax))
        Sy = np.sin(np.outer(yp, by))
        Cx = np.cos(np.outer(xp, ax))
        Cy = np.cos(np.outer(yp, by))
        w = np.einsum("im,mn,in->i", Sx, coef_w, Sy, optimize=True)
        mx = np.einsum("im,mn,in->i", Sx, coef_mx, Sy, optimize=True)
        my = np.einsum("im,mn,in->i", Sx, coef_my, Sy, optimize=True)
        mxy = np.einsum("im,mn,in->i", Cx, coef_mxy, Cy, optimize=True)
        return w, mx, my, mxy

    if points is None:
        grid = np.linspace(0.0, 1.0, 81)
        xp = a * grid
        yp = b * grid
        w, mx, my, mxy = _eval(xp, yp)
        moments = np.stack((mx, my, mxy), axis=-1)
        mp = principal_moments(torch.as_tensor(moments)).numpy()
        # A truncated series overshoots (Gibbs) inside a thin boundary layer of
        # a supported edge, where the true normal moment jumps to zero; the
        # peak is therefore reported over the interior window only.
        lo = edge_margin
        hi = 1.0 - edge_margin
        inner = (grid >= lo) & (grid <= hi)
        interior = mp[np.ix_(inner, inner)]
        m_peak = float(np.max(np.abs(interior)))
        out_points = None
    else:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must be an (n, 2) array")
        xp = pts[:, 0]
        yp = pts[:, 1]
        if np.any(xp < 0) or np.any(xp > a) or np.any(yp < 0) or np.any(yp > b):
            raise ValueError("points must lie inside the rectangle")
        w, mx, my, mxy = _eval_points(xp, yp)
        moments = np.stack((mx, my, mxy), axis=-1)
        mp = principal_moments(torch.as_tensor(moments)).numpy()
        m_peak = float(np.max(np.abs(mp)))
        out_points = pts
    w_c, mx_c, my_c, _ = _eval_points(np.array([0.5 * a]), np.array([0.5 * b]))
    return {
        "modes": int(modes), "model": model, "p_mn": p_mn,
        "points": out_points, "w": w,
        "Mx": mx, "My": my, "Mxy": mxy,
        "m_peak": m_peak, "sigma_peak": 6.0 * m_peak / t ** 2,
        "w_centre": float(w_c[0]),
        "Mx_centre": float(mx_c[0]), "My_centre": float(my_c[0]),
    }


# --------------------------------------------------------------------------
# crush interface law (compression-side cohesive / damageable law)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrushInterfaceLaw:
    """1-D damageable compression law for the ice/structure interface.

    ``p(delta) = k_n delta``                          for ``delta <= delta_c``
    ``p(delta) = p_c [1 - (1 - r) (delta - delta_c) / (delta_u - delta_c)]``
                                                      for ``delta_c < delta <= delta_u``
    ``p(delta) = r p_c``                              for ``delta > delta_u``

    with ``delta_c = p_c / k_n``, ``delta_u = s delta_c`` and ``r`` the residual
    ratio.  This is the compression-side counterpart of the mode-I bilinear law
    in :mod:`tensorfem.cohesive` (which returns the *undamaged* stiffness in
    compression and therefore cannot represent crushing); the envelope is used
    as the secant law of a monotone indentation path, so no unloading branch is
    needed for the quasi-static coupling.
    """

    stiffness: float
    crush_strength: float
    residual_ratio: float = 0.5
    softening_ratio: float = 4.0

    def validate(self) -> None:
        for name in ("stiffness", "crush_strength", "softening_ratio"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.residual_ratio <= 1.0:
            raise ValueError("residual_ratio must lie in [0, 1]")
        if self.softening_ratio < 1.0:
            raise ValueError("softening_ratio must be at least 1")

    @property
    def delta_c(self) -> float:
        return self.crush_strength / self.stiffness

    @property
    def delta_u(self) -> float:
        return self.softening_ratio * self.delta_c

    @property
    def residual(self) -> float:
        return self.residual_ratio * self.crush_strength

    def envelope(self, penetration) -> torch.Tensor:
        d = torch.clamp(torch.as_tensor(penetration, dtype=D), min=0.0)
        dc, du = self.delta_c, self.delta_u
        elastic = self.stiffness * d
        drop = torch.clamp((d - dc) / (du - dc), min=0.0, max=1.0)
        softening = self.crush_strength * (1.0 - (1.0 - self.residual_ratio) * drop)
        return torch.where(d <= dc, elastic, softening)


# --------------------------------------------------------------------------
# two-way coupled ice/panel solve
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelSpec:
    """Rectangular hull-panel bay (frames/longitudinals as clamped edges)."""

    span_x: float = 0.75
    span_y: float = 2.40
    thickness: float = 0.025
    young: float = 206.0e9
    poisson: float = 0.30
    boundary: str = "clamped"
    nx: int = 12
    ny: int = 24

    def validate(self) -> None:
        for name in ("span_x", "span_y", "thickness", "young"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 < self.poisson < 0.5:
            raise ValueError("poisson out of range")
        if self.boundary not in ("clamped", "simply_supported"):
            raise ValueError("boundary must be 'clamped' or 'simply_supported'")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("need at least a 2x2 mesh")

    def mesh(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate()
        return rect_mesh(self.span_x, self.span_y, self.nx, self.ny)


@dataclass(frozen=True)
class ContactFootprint:
    """Contact grid reported by the interface solve: points plus cell areas."""

    points: torch.Tensor
    weights: torch.Tensor

    def validate(self) -> None:
        pts = _as_points(self.points)
        w = torch.as_tensor(self.weights, dtype=D).reshape(-1)
        if w.shape != (pts.shape[0],):
            raise ValueError("weights must have one entry per contact point")
        if bool(torch.any(w <= 0)):
            raise ValueError("contact weights must be positive")

    @property
    def area(self) -> float:
        return float(torch.sum(self.weights))

    @classmethod
    def grid(cls, x0: float, x1: float, y0: float, y1: float,
             nx: int, ny: int) -> "ContactFootprint":
        """Structured ``nx x ny`` contact cells (cell-area weights)."""
        if not (x1 > x0 and y1 > y0 and nx >= 1 and ny >= 1):
            raise ValueError("invalid contact footprint geometry")
        xs = torch.linspace(x0, x1, nx + 1, dtype=D)
        ys = torch.linspace(y0, y1, ny + 1, dtype=D)
        cx = 0.5 * (xs[:-1] + xs[1:])
        cy = 0.5 * (ys[:-1] + ys[1:])
        xx, yy = torch.meshgrid(cx, cy, indexing="xy")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        cell = (x1 - x0) / nx * (y1 - y0) / ny
        weights = torch.full((points.shape[0],), cell, dtype=D)
        return cls(points=points, weights=weights)


@dataclass
class CouplingResult:
    """Converged two-way ice/panel contact state."""

    law: CrushInterfaceLaw
    advance: float
    load: PatchLoad
    response: PlateResponse
    penetration: torch.Tensor
    gap: torch.Tensor
    iterations: int
    residual_history: list
    converged: bool
    sigma_peak_uniform: float
    one_way: dict

    @property
    def sigma_peak(self) -> float:
        return self.response.sigma_peak

    @property
    def amplification(self) -> float:
        """Peak-stress amplification of the non-uniform patch over the
        statically equivalent uniform patch (same total force, same footprint)."""
        return self.sigma_peak / self.sigma_peak_uniform

    @property
    def total_force(self) -> float:
        return self.load.total_force

    @property
    def peak_to_mean(self) -> float:
        return self.load.peak_to_mean

    @property
    def coefficient_of_variation(self) -> float:
        return self.load.coefficient_of_variation


def solve_contact_plate_coupling(
    panel: PanelSpec,
    contact: ContactFootprint,
    law: CrushInterfaceLaw,
    advance: float,
    *,
    gap=None,
    relaxation: float = 0.6,
    tolerance: float = 1.0e-6,
    max_iterations: int = 100,
    uniform_reference: bool = True,
    model: "PlateModel | None" = None,
    interpolation: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
) -> CouplingResult:
    """Two-way fixed point between the crush-interface pressure and the panel.

    Per iteration:

    1. penetration from the *deformed* panel surface,
       ``delta_i = clamp(advance - g_i - (C u)_i, 0)``;
    2. interface traction ``p_i = law.envelope(delta_i)``;
    3. the load array ``f = C^T (w p)`` (:func:`patch_from_contact`);
    4. Mindlin panel solve (:func:`solve_plate_patch_load`);
    5. under-relaxed update of ``u`` and repeat until
       ``max|w_new - w_old| <= tolerance * max(1, max|w|)``.

    ``gap`` is the initial ice/panel clearance sampled at the contact points
    (defaults to zero: an ideally conforming ice edge).  A non-zero, spatially
    varying gap is the physical driver of an *inhomogeneous* patch: the ice
    bears first where the clearance is smallest.

    The result also carries the residual history and a ``converged`` flag
    (never assumed, always reported), and the rigid-ice (one-way) solution --
    the same interface law evaluated with ``w = 0`` in step 1 -- so the effect
    of the load feedback is measured instead of asserted.
    """
    law.validate()
    panel.validate()
    contact.validate()
    if not advance > 0:
        raise ValueError("advance must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")

    nodes, elements = panel.mesh()
    if model is None:
        model = PlateModel(nodes, elements, young=panel.young,
                           poisson=panel.poisson, thickness=panel.thickness,
                           boundary=panel.boundary)
    points = _as_points(contact.points)
    weights = (torch.as_tensor(contact.weights, dtype=D).reshape(-1)
               if point_weights is None else torch.as_tensor(point_weights, dtype=D))
    if interpolation is None:
        C, _ = plate_contact_interpolation(nodes, elements, points, weights)
    else:
        C = torch.as_tensor(interpolation, dtype=D)
    n_node = int(nodes.shape[0])
    zero_gap = torch.zeros(points.shape[0], dtype=D)
    gap_t = zero_gap if gap is None else torch.as_tensor(gap, dtype=D).reshape(-1)
    if gap_t.shape != (points.shape[0],):
        raise ValueError("gap must have one entry per contact point")

    ndof = 3 * n_node
    u = torch.zeros(ndof, dtype=D)
    w_pts = torch.zeros(points.shape[0], dtype=D)
    history = []
    converged = False
    response = None
    load = None
    penetration = zero_gap
    iterations = 0
    for it in range(1, max_iterations + 1):
        iterations = it
        penetration = torch.clamp(advance - gap_t - w_pts, min=0.0)
        pressure = law.envelope(penetration)
        load = patch_from_contact(
            points, pressure, point_weights=weights,
            interpolation=C, n_node=n_node)
        response = model.solve(dof_load=load.dof_load)
        w_new = C @ response.w
        residual = float(torch.max(torch.abs(w_new - w_pts)))
        history.append({"iteration": it, "residual": residual,
                        "total_force": load.total_force,
                        "sigma_peak": response.sigma_peak})
        scale = max(1.0, float(torch.max(torch.abs(w_new))))
        if residual <= tolerance * scale:
            u = response.displacement
            w_pts = w_new
            converged = True
            # one extra pass so penetration/pressure match the final u
            penetration = torch.clamp(advance - gap_t - w_pts, min=0.0)
            pressure = law.envelope(penetration)
            load = patch_from_contact(
                points, pressure, point_weights=weights,
                interpolation=C, n_node=n_node)
            # one final solve so that the returned response belongs to the
            # returned load (no iteration lag in the reported identities)
            response = model.solve(dof_load=load.dof_load)
            w_pts = C @ response.w
            break
        u = (1.0 - relaxation) * u + relaxation * response.displacement
        w_pts = (1.0 - relaxation) * w_pts + relaxation * w_new

    # statically equivalent uniform patch (same total force and footprint)
    sigma_uniform = float("nan")
    if uniform_reference:
        uniform = load.uniform_equivalent()
        sigma_uniform = model.solve(dof_load=uniform.dof_load).sigma_peak

    # rigid-ice reference: same law, penetration without the panel feedback
    pen_one_way = torch.clamp(advance - gap_t, min=0.0)
    load_one_way = patch_from_contact(
        points, law.envelope(pen_one_way), point_weights=weights,
        interpolation=C, n_node=n_node)
    response_one_way = model.solve(dof_load=load_one_way.dof_load)
    one_way = {
        "total_force": load_one_way.total_force,
        "sigma_peak": response_one_way.sigma_peak,
        "peak_pressure": load_one_way.peak_pressure,
        "peak_to_mean": load_one_way.peak_to_mean,
        "coefficient_of_variation": load_one_way.coefficient_of_variation,
        "penetration_max": float(torch.max(pen_one_way)),
    }

    return CouplingResult(
        law=law, advance=float(advance), load=load, response=response,
        penetration=penetration, gap=gap_t, iterations=iterations,
        residual_history=history, converged=converged,
        sigma_peak_uniform=sigma_uniform, one_way=one_way,
    )