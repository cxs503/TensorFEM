"""CASE-HOLLOW-SPHERE-3D: thick-walled hollow sphere under internal pressure.

Closed-form reference
---------------------
Lamé's solution for a spherical shell ``a <= r <= b`` with internal pressure
``p`` (Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., 1970, Sec. 5.6,
eqs. 5.6.3-5.6.5; originally Lamé 1852)::

    A = p a^3 / (b^3 - a^3)
    B = A b^3
    sigma_r = A - B/r^3
    sigma_t = A + B/(2 r^3)            (hoop = meridional, biaxial state)
    u_r     = [(1 - 2 nu) A r + (1 + nu) B/(2 r^2)] / E

Boundary conditions ``sigma_r(a) = -p``, ``sigma_r(b) = 0`` and the traction
free outer surface are satisfied identically, so the reference is an exact
strong-form solution of the 3-D elasticity problem with self-equilibrated
loading; it needs no FE input.  Strain energy of the exact field is

    U = 2 pi p^2 [(1-2 nu) a^6 + (1+nu) a^3 b^3 / 2] / (E (b^3 - a^3))

(Clapeyron's theorem, ``U = 1/2 * p * u_r(a) * 4 pi a^2``, reproduces it.)

which gives an independent global scalar for energy-norm verification.

Discretisations
---------------
* ``order=1``  -- trilinear HEX8 ``cube-sphere`` mesh (six mapped cube faces
  lofted radially between the two spheres).  Reuses the qualified geometry
  generator of :mod:`tensorfem.sphere_pressure_benchmark`.
* ``order=2``  -- 20-node serendipity bricks obtained by edge-midpoint upgrade,
  with all inner/outer surface midside nodes projected radially onto the exact
  spheres (curved-boundary projection).  This removes the O(h^2) geometric
  faceting error that dominates the HEX8 radial-stress response on the same
  topology, which is the standard remedy for curved-boundary solid meshes
  (Zienkiewicz & Taylor, *The FEM*, 5th ed., Vol. 1, Sec. 8.6).

Both paths share one pressure-load generator: consistent nodal tractions from
Gauss integration of ``t = p * (dx/dxi x dx/deta)`` over the inner surface
(inward normal, matching ``sigma_r(a) = -p``).  The load vector is validated by
its closed-surface resultants (``sum F = 0``, ``sum x x F = 0``).

Case metrics
------------
Probe quantities are compared at *matching locations*: nodal radial
displacement on the exact sphere surfaces, and volume-averaged element stresses
of the innermost/outermost radial layer, where the reference is evaluated at the
same volume-averaged centroid radius.  On the outer surface ``sigma_r -> 0``
identically, so its error is normalised by the applied pressure ``p`` instead of
the local value (a pointwise relative measure there measures round-off, the
classic divide-by-zero trap in pressure-vessel V&V).
"""
from __future__ import annotations

import math
from typing import Mapping

import torch

from .hex20 import hex20_shape_gradients, hex20_stiffness, upgrade_hex8_to_hex20
from .solid3d import elasticity_matrix_3d, hex8_stiffness
from .sparse_core import assemble_coo

CASE = {
    "inner_radius_m": 8.0,
    "outer_radius_m": 10.0,
    "young_modulus_pa": 210e9,
    "poisson_ratio": 0.3,
    "internal_pressure_pa": 1e6,
}

# Quantities scored by the case and the normalisation used for each.
TOLERANCE = 0.05  # case pool <5%
_PRESSURE_NORMALISED = {"radial_stress_outer"}


def case_parameters(case: Mapping[str, float] | None = None) -> dict:
    """Validated case definition (raises on non-physical input)."""
    c = dict(CASE if case is None else case)
    a, b = float(c["inner_radius_m"]), float(c["outer_radius_m"])
    nu = float(c["poisson_ratio"])
    if not (0.0 < a < b):
        raise ValueError("need 0 < inner_radius < outer_radius")
    if not (-1.0 < nu < 0.5):
        raise ValueError("poisson ratio out of range")
    if float(c["young_modulus_pa"]) <= 0 or float(c["internal_pressure_pa"]) < 0:
        raise ValueError("young modulus must be positive and pressure non-negative")
    return c


# ----------------------------------------------------------------------------
# closed-form reference
# ----------------------------------------------------------------------------
def lame_sphere(r, case: Mapping[str, float] | None = None):
    """Exact Lamé fields at radius ``r`` (scalar tensor/sequence)."""
    c = case_parameters(case)
    a, b, E, nu, p = (c["inner_radius_m"], c["outer_radius_m"],
                      c["young_modulus_pa"], c["poisson_ratio"],
                      c["internal_pressure_pa"])
    r = torch.as_tensor(r, dtype=torch.float64)
    if bool((r <= 0).any()):
        raise ValueError("radius must be positive")
    denominator = b ** 3 - a ** 3
    A = p * a ** 3 / denominator
    B = A * b ** 3
    radial = A - B / r ** 3
    hoop = A + B / (2 * r ** 3)
    ur = ((1 - 2 * nu) * A * r + (1 + nu) * B / (2 * r ** 2)) / E
    von_mises = (hoop - radial).abs()
    return {"radial_displacement": ur, "radial_stress": radial,
            "hoop_stress": hoop, "von_mises": von_mises}


def lame_displacement(x, case: Mapping[str, float] | None = None) -> torch.Tensor:
    """Exact displacement vector ``u = u_r(r) * x/|x|`` at points ``x``."""
    x = torch.as_tensor(x, dtype=torch.float64)
    r = torch.linalg.vector_norm(x, dim=1)
    return lame_sphere(r, case)["radial_displacement"][:, None] * (x / r[:, None])


def strain_energy(case: Mapping[str, float] | None = None) -> float:
    """Exact elastic strain energy of the Lamé field (closed form, joules)."""
    c = case_parameters(case)
    a, b, E, nu, p = (c["inner_radius_m"], c["outer_radius_m"],
                      c["young_modulus_pa"], c["poisson_ratio"],
                      c["internal_pressure_pa"])
    return (2 * math.pi * p ** 2 *
            ((1 - 2 * nu) * a ** 6 + (1 + nu) * a ** 3 * b ** 3 / 2) /
            (E * (b ** 3 - a ** 3)))


def shell_volume(case: Mapping[str, float] | None = None) -> float:
    """Exact volume of the shell between the two spheres."""
    c = case_parameters(case)
    a, b = c["inner_radius_m"], c["outer_radius_m"]
    return 4 * math.pi / 3 * (b ** 3 - a ** 3)


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def _mesh(n: int, radial_layers: int, order: int, case):
    from .sphere_pressure_benchmark import sphere_volume_mesh

    a, b = case["inner_radius_m"], case["outer_radius_m"]
    nodes, elements, faces, surface = sphere_volume_mesh(
        n, radial_layers, {"inner_radius_m": a, "outer_radius_m": b})
    if order == 1:
        return nodes, elements, faces
    if order != 2:
        raise ValueError("order must be 1 (HEX8) or 2 (HEX20)")
    nodes, elements = upgrade_hex8_to_hex20(nodes, elements)
    ns = len(faces)
    # Surface faces of the upgrade: element-local ids of the 8-node serendipity
    # quad.  Layer 0 carries the inner surface on local face (0,1,2,3) with
    # midside nodes 8..11; the last layer carries the outer surface on
    # (4,5,6,7) with midside nodes 12..15.
    inner = elements[:ns][:, [0, 1, 2, 3, 8, 9, 10, 11]]
    outer = elements[-ns:][:, [4, 5, 6, 7, 12, 13, 14, 15]]
    for ids, radius in ((torch.unique(inner.flatten()), a),
                        (torch.unique(outer.flatten()), b)):
        r = torch.linalg.vector_norm(nodes[ids], dim=1)
        nodes = nodes.clone()
        nodes[ids] = nodes[ids] * (radius / r)[:, None]
    return nodes, elements, faces


# ----------------------------------------------------------------------------
# 8-node serendipity surface shape functions (for HEX20 pressure loads)
# ----------------------------------------------------------------------------
def _serendipity_quad(xi: torch.Tensor, eta: torch.Tensor):
    N = torch.stack((
        -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
        -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
        -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
        -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
        0.5 * (1 - xi ** 2) * (1 - eta),
        0.5 * (1 + xi) * (1 - eta ** 2),
        0.5 * (1 - xi ** 2) * (1 + eta),
        0.5 * (1 - xi) * (1 - eta ** 2),
    ))
    # corner derivatives: dN/dxi = 0.25 a (1 + b eta)(2 a xi + b eta),
    # dN/deta = 0.25 b (1 + a xi)(a xi + 2 b eta) for corner signs (a, b)
    dN = [torch.stack((0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta),
                       0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)))
          for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    dN += [torch.stack((-xi * (1 - eta), -0.5 * (1 - xi ** 2))),
           torch.stack((0.5 * (1 - eta ** 2), -(1 + xi) * eta)),
           torch.stack((-xi * (1 + eta), 0.5 * (1 - xi ** 2))),
           torch.stack((-0.5 * (1 - eta ** 2), -(1 - xi) * eta))]
    dN = torch.stack(dN)
    return N, dN


def _pressure_loads(nodes, faces, elements, ns, order, pressure, case):
    """Consistent nodal loads of ``pressure`` on the inner surface.

    ``faces`` holds the four corners of every inner-surface quad in the order of
    the HEX8 local face; for HEX20 the midside nodes come from the layer-0
    element connectivity.  Returns ``(force, resultants)``.
    """
    ndof = 3 * len(nodes)
    force = nodes.new_zeros(ndof)
    if order == 1:
        point, weight = 1 / math.sqrt(3), 1.0
    else:
        point, weight = math.sqrt(3 / 5), 1.0
        gauss = (( -point, 5 / 9), (0.0, 8 / 9), (point, 5 / 9))
    total = nodes.new_zeros(3)
    moment = nodes.new_zeros(3)
    if order == 1:
        from .plate import _shape

        gauss = ((-point, weight), (point, weight))
        for xi, wx in gauss:
            for eta, wy in gauss:
                N, dN = _shape(nodes.new_tensor(xi), nodes.new_tensor(eta))
                coords = nodes[faces]
                tangents = torch.einsum("eia,ib->eab", coords, dN)
                area = torch.linalg.cross(tangents[:, :, 0], tangents[:, :, 1], dim=1)
                quadrature = wx * wy
                block = quadrature * pressure * N[None, :, None] * area[:, None, :]
                force.index_add_(0, (3 * faces[:, :, None] + torch.arange(3)).flatten(),
                                 block.flatten())
                total = total + block.sum((0, 1))
                moment = moment + torch.linalg.cross(coords, block, dim=2).sum((0, 1))
    else:
        face_nodes = elements[:ns][:, [0, 1, 2, 3, 8, 9, 10, 11]]
        coords = nodes[face_nodes]
        for xi, wx in gauss:
            for eta, wy in gauss:
                N, dN = _serendipity_quad(nodes.new_tensor(xi), nodes.new_tensor(eta))
                tangents = torch.einsum("eia,ib->eab", coords, dN)
                area = torch.linalg.cross(tangents[:, :, 0], tangents[:, :, 1], dim=1)
                block = wx * wy * pressure * N[None, :, None] * area[:, None, :]
                force.index_add_(0, (3 * face_nodes[:, :, None] + torch.arange(3)).flatten(),
                                 block.flatten())
                total = total + block.sum((0, 1))
                moment = moment + torch.linalg.cross(coords, block, dim=2).sum((0, 1))
    scale = float(torch.linalg.vector_norm(force))
    resultants = {
        "closed_surface_force_resultant": float(torch.linalg.vector_norm(total)) / scale,
        "closed_surface_moment_resultant": float(torch.linalg.vector_norm(moment)) / (scale * case["outer_radius_m"]),
    }
    return force, resultants


def _gauge_dofs(nodes, case):
    """Six tangential constraints: rigid motion removed, radial growth free."""
    fixed = {}
    for axis in range(3):
        point = nodes.new_zeros(3)
        point[axis] = case["outer_radius_m"]
        node = int(torch.linalg.vector_norm(nodes - point, dim=1).argmin())
        if float(torch.linalg.vector_norm(nodes[node] - point)) > 1e-12:
            raise ValueError("axis gauge point absent from mesh nodes")
        for component in range(3):
            if component != axis:
                fixed[3 * node + component] = 0.0
    return fixed


def _element_volumes(x, order):
    if order == 1:
        points = (-1 / math.sqrt(3), 1 / math.sqrt(3))
        weights = (1.0, 1.0)
        rule = [(a, b, c) for a in points for b in points for c in points]
        weights = [1.0] * len(rule)
    else:
        q = math.sqrt(3 / 5)
        points = (-q, 0.0, q)
        w = (5 / 9, 8 / 9, 5 / 9)
        rule = [(a, b, c) for a in points for b in points for c in points]
        weights = [w[i] * w[j] * w[k] for i in range(3) for j in range(3) for k in range(3)]
    volume = x.new_zeros(len(x))
    for (xi, eta, zeta), weight in zip(rule, weights):
        if order == 1:
            from .solid3d import _hex_gradient

            _, det = _hex_gradient(x, xi, eta, zeta)
        else:
            _, dN = hex20_shape_gradients(xi, eta, zeta, like=x)
            jacobian = torch.einsum("eia,ib->eab", x, dN)
            det = torch.linalg.det(jacobian)
            if bool((det <= 0).any()):
                raise ValueError("HEX20 element has non-positive Jacobian")
        volume = volume + weight * det
    return volume


def _voigt_tensor(s):
    out = s.new_zeros((len(s), 3, 3))
    out[:, 0, 0], out[:, 1, 1], out[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = s[:, 3]
    out[:, 1, 2] = out[:, 2, 1] = s[:, 4]
    out[:, 0, 2] = out[:, 2, 0] = s[:, 5]
    return out


# ----------------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------------
def solve_hollow_sphere(n: int, radial_layers: int, *, order: int = 1,
                        case: Mapping[str, float] | None = None,
                        rtol: float = 1e-11, maxiter: int = 20000,
                        keep_arrays: bool = False):
    """Solve the 3-D hollow-sphere case on a ``(n, radial_layers)`` mesh."""
    c = case_parameters(case)
    if radial_layers < 2:
        raise ValueError("radial_layers must be >= 2 so both wall surfaces can be probed")
    nodes, elements, faces = _mesh(n, radial_layers, order, c)
    ns = len(faces)
    ne = len(elements)
    D = elasticity_matrix_3d(nodes.new_full((ne,), c["young_modulus_pa"]),
                             nodes.new_full((ne,), c["poisson_ratio"]))
    if order == 1:
        ke, B = hex8_stiffness(nodes[elements.long()], D)
        per_element = 24
    else:
        ke, B = hex20_stiffness(nodes[elements.long()], D)
        per_element = 60
    edofs = (3 * elements[:, :, None] + torch.arange(3)).reshape(ne, per_element)
    ndof = 3 * len(nodes)
    K = assemble_coo(ke, edofs, ndof)
    force, resultants = _pressure_loads(nodes, faces, elements, ns, order,
                                        c["internal_pressure_pa"], c)
    fixed = _gauge_dofs(nodes, c)
    displacement, iterations, residual = _constrained_pcg(K, force, fixed,
                                                          rtol=rtol, maxiter=maxiter)
    u = displacement.reshape(-1, 3)
    strain = torch.einsum("eij,ej->ei", B, displacement[edofs])
    stress = torch.einsum("eij,ej->ei", D, strain)
    volume = _element_volumes(nodes[elements.long()], order)
    metrics = probe_metrics(nodes, elements, stress, u, volume, ns, c,
                            order=order)
    energy = 0.5 * float((u.flatten() * force).sum())
    residual_check = _equilibrium(nodes, K, force, displacement, fixed)
    if keep_arrays:
        # centroids and nodal vectors are only needed for field/contour output;
        # they stay out of the JSON artifact contract by default.
        result_arrays = {"centres": _element_centres(nodes, elements, order),
                         "von_mises": torch.tensor(metrics["fields"]["von_mises"]),
                         "stress": stress, "volume": volume,
                         "nodes": nodes, "displacement": u}
    else:
        result_arrays = None
    result = {
        "case": c,
        "mesh": {"face_divisions": n, "radial_layers": radial_layers,
                 "order": order, "element": "HEX8" if order == 1 else "HEX20",
                 "nodes": int(len(nodes)), "elements": int(ne),
                 "degrees_of_freedom": int(ndof),
                 "mesh_volume_m3": float(volume.sum()),
                 "shell_volume_m3": shell_volume(c),
                 "volume_error": abs(float(volume.sum()) - shell_volume(c)) / shell_volume(c)},
        "solver": {"iterations": int(iterations), "relative_residual": float(residual)},
        "loads": resultants,
        "equilibrium": residual_check,
        "energy": {"fe_strain_energy_j": energy,
                   "lame_strain_energy_j": strain_energy(c),
                   "relative_error": abs(energy - strain_energy(c)) / strain_energy(c)},
        "metrics": metrics,
        "passed": bool(metrics["passed"] and metrics["worst_relative_error"] < TOLERANCE
                       and residual_check["free_residual_relative"] < 1e-8),
    }
    if keep_arrays:
        result["arrays"] = result_arrays
    return result


def _equilibrium(nodes, K, force, displacement, fixed):
    from .sparse_core import sparse_mv

    residual = sparse_mv(K, displacement) - force
    free = torch.ones_like(force, dtype=torch.bool)
    free[list(fixed)] = False
    scale = float(torch.linalg.vector_norm(force))
    return {"free_residual_relative": float(torch.linalg.vector_norm(residual[free])) / scale,
            "gauge_reaction_relative": float(torch.linalg.vector_norm(residual[list(fixed)])) / scale}



def _constrained_pcg(K, force, fixed, *, rtol, maxiter):
    """Jacobi-preconditioned CG with symmetric elimination of gauge rows.

    ``solve_sparse_static`` runs unpreconditioned CG; the hollow-sphere system is
    a 3-D elasticity operator whose conditioning grows with 1/h^2, so the
    diagonal preconditioner keeps the fine ladders affordable.
    """
    from .sparse_core import conjugate_gradient, sparse_mv

    ndof = force.numel()
    ids = torch.tensor(sorted(fixed), dtype=torch.long)
    diagonal = force.new_zeros(ndof)
    with torch.no_grad():
        index = K.indices()
        ondiag = index[0] == index[1]
        diagonal.index_add_(0, index[0, ondiag], K.values()[ondiag])
        if bool((diagonal <= 0).any()):
            raise ValueError("stiffness diagonal must be positive")
        diagonal[ids] = 1.0
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
            csr = K.to_sparse_csr()

    def operator(q):
        constrained = q.clone()
        constrained[ids] = 0.0
        out = sparse_mv(csr, constrained)
        out[ids] = q[ids]
        return out

    rhs = force.clone()
    rhs[ids] = 0.0
    solved = conjugate_gradient(operator, rhs, diagonal=diagonal, rtol=rtol,
                                maxiter=maxiter)
    if not solved.converged:
        raise RuntimeError(f"constrained PCG did not converge: "
                           f"residual={solved.relative_residual:g}")
    return solved.x, solved.iterations, solved.relative_residual


# ----------------------------------------------------------------------------
# probe metrics
# ----------------------------------------------------------------------------
def probe_metrics(nodes, elements, stress, u, volume, ns, case, *, order=1):
    """Compare FE probes with the Lamé field at matching locations."""
    c = case_parameters(case)
    a, b, p = c["inner_radius_m"], c["outer_radius_m"], c["internal_pressure_pa"]
    centres = _element_centres(nodes, elements, order)
    radius = torch.linalg.vector_norm(centres, dim=1)
    tensor = _voigt_tensor(stress)
    normal = centres / radius[:, None]
    radial = torch.einsum("ei,eij,ej->e", normal, tensor, normal)
    hoop = (tensor.diagonal(dim1=1, dim2=2).sum(1) - radial) / 2
    deviator = tensor - tensor.diagonal(dim1=1, dim2=2).sum(1)[:, None, None] * torch.eye(3, dtype=nodes.dtype) / 3
    von_mises = torch.sqrt(1.5 * (deviator * deviator).sum((1, 2)))

    def layer_average(layer):
        weight = volume[layer]
        r_mean = float((weight * radius[layer]).sum() / weight.sum())
        return {"radius": r_mean,
                "radial_stress": float((weight * radial[layer]).sum() / weight.sum()),
                "hoop_stress": float((weight * hoop[layer]).sum() / weight.sum()),
                "von_mises": float((weight * von_mises[layer]).sum() / weight.sum())}

    inner = layer_average(slice(0, ns))
    outer = layer_average(slice(len(elements) - ns, None))

    def surf_average(target):
        r = torch.linalg.vector_norm(nodes, dim=1)
        selected = (r - target).abs() < 1e-6 * target
        radial_direction = nodes[selected] / r[selected, None]
        ur = (u[selected] * radial_direction).sum(1)
        return float(ur.mean())

    u_inner, u_outer = surf_average(a), surf_average(b)
    ref_inner = {k: float(v) for k, v in lame_sphere(torch.tensor(inner["radius"]), c).items()}
    ref_outer = {k: float(v) for k, v in lame_sphere(torch.tensor(outer["radius"]), c).items()}
    ref_ua = float(lame_sphere(torch.tensor(a), c)["radial_displacement"])
    ref_ub = float(lame_sphere(torch.tensor(b), c)["radial_displacement"])

    def rel(got, ref):
        return abs(got - ref) / abs(ref)

    primary = {
        "radial_displacement_inner": {"ref": ref_ua, "got": u_inner, "unit": "m",
                                      "probe": f"nodal average on r={a:g} m",
                                      "error": rel(u_inner, ref_ua)},
        "radial_displacement_outer": {"ref": ref_ub, "got": u_outer, "unit": "m",
                                      "probe": f"nodal average on r={b:g} m",
                                      "error": rel(u_outer, ref_ub)},
        "radial_stress_inner": {"ref": ref_inner["radial_stress"], "got": inner["radial_stress"],
                                "unit": "Pa", "probe": f"layer volume average at r={inner['radius']:.6g} m",
                                "error": rel(inner["radial_stress"], ref_inner["radial_stress"])},
        "hoop_stress_inner": {"ref": ref_inner["hoop_stress"], "got": inner["hoop_stress"],
                              "unit": "Pa", "probe": f"layer volume average at r={inner['radius']:.6g} m",
                              "error": rel(inner["hoop_stress"], ref_inner["hoop_stress"])},
        "von_mises_inner": {"ref": ref_inner["von_mises"], "got": inner["von_mises"],
                            "unit": "Pa", "probe": f"layer volume average at r={inner['radius']:.6g} m",
                            "error": rel(inner["von_mises"], ref_inner["von_mises"])},
        "hoop_stress_outer": {"ref": ref_outer["hoop_stress"], "got": outer["hoop_stress"],
                              "unit": "Pa", "probe": f"layer volume average at r={outer['radius']:.6g} m",
                              "error": rel(outer["hoop_stress"], ref_outer["hoop_stress"])},
    }
    # sigma_r vanishes on the traction-free outer surface, so the outer-layer
    # radial error is only meaningful as an absolute error normalised by p.
    secondary = {
        "radial_stress_outer": {
            "ref": ref_outer["radial_stress"], "got": outer["radial_stress"], "unit": "Pa",
            "probe": f"layer volume average at r={outer['radius']:.6g} m",
            "error": abs(outer["radial_stress"] - ref_outer["radial_stress"]) / p,
            "normalisation": "applied_pressure",
            "note": "local sigma_r -> 0; relative error is not a meaningful measure here"},
    }
    worst = max(row["error"] for row in primary.values())
    worst_secondary = max(row["error"] for row in secondary.values())
    fields = {"radius": radius.tolist(),
              "radial_stress": radial.tolist(),
              "hoop_stress": hoop.tolist(),
              "von_mises": von_mises.tolist()}
    return {"probes": primary, "secondary_probes": secondary, "fields": fields,
            "worst_relative_error": float(worst),
            "worst_secondary_error": float(worst_secondary),
            "tolerance": TOLERANCE,
            "passed": bool(worst < TOLERANCE and worst_secondary < TOLERANCE),
            "worst_quantity": max(primary, key=lambda k: primary[k]["error"]),
            "probe_radii": {"inner_layer": inner["radius"], "outer_layer": outer["radius"]}}


def _element_centres(nodes, elements, order):
    x = nodes[elements.long()]
    return x.mean(1) if order == 1 else _hex20_centre(x)


def _hex20_centre(x):
    """Physical centroid of a HEX20 (isoparametric centre, xi=eta=zeta=0)."""
    N, _ = hex20_shape_gradients(0.0, 0.0, 0.0, like=x)
    return torch.einsum("i,eia->ea", N, x)


def convergence(cases):
    """Monotone error decay across a refinement ladder (at least two levels)."""
    if len(cases) < 2:
        raise ValueError("convergence needs at least two resolutions")
    ladder = [c["metrics"]["worst_relative_error"] for c in cases]
    order = [c["mesh"]["order"] for c in cases]
    if len(set(order)) != 1:
        raise ValueError("a ladder must hold one element family")
    decreasing = all(b < a for a, b in zip(ladder, ladder[1:]))
    return {"worst_error_ladder": ladder,
            "monotone_decreasing": bool(decreasing),
            # observed rate on the last pair, h ~ 1/n
            "observed_rate_last_pair": (
                math.log(ladder[-2] / ladder[-1]) / math.log(
                    cases[-1]["mesh"]["face_divisions"] / cases[-2]["mesh"]["face_divisions"])
                if decreasing else float("nan"))}