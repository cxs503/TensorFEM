"""First tensile failure in an intact, dynamically loaded plane-stress ice beam."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import factorized, spsolve

from .cohesive_bending import _q4_stiffness, plane_stress_matrix


def first_crack(model, *, dt, speed=0.0025, density=917.0, penalty_ratio=100.0, max_time=1.0):
    """Newmark average acceleration with unilateral punch compliance.

    Coincident interface nodes are welded before assembly. The tensile stress
    is searched at *both* free-surface fibres of every element in the outer
    rows. The criterion is longitudinal tension in flexure. Post-initiation
    propagation and dissipation are outside this elastic calculation.
    """
    if min(dt, speed, density, penalty_ratio, max_time) <= 0:
        raise ValueError("positive integration, loading and material parameters required")
    xy, mapping = np.unique(model.nodes, axis=0, return_inverse=True)
    cells = mapping[model.elements]
    coords = xy[cells]
    dofs = (2 * cells[:, :, None] + np.arange(2)).reshape(-1, 8)
    dmat = plane_stress_matrix(model.young_modulus, model.poisson_ratio)
    ke = _q4_stiffness(coords, dmat, model.thickness)
    ndof = 2 * len(xy)
    k = coo_matrix(
        (ke.ravel(), (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())),
        shape=(ndof, ndof),
    ).tocsr()
    width = coords[:, 1, 0] - coords[:, 0, 0]
    height = coords[:, 3, 1] - coords[:, 0, 1]
    nodal_mass = np.zeros(len(xy))
    np.add.at(
        nodal_mass, cells.ravel(), np.repeat(density * width * height * model.thickness / 4, 4)
    )
    remap = (2 * mapping[:, None] + np.arange(2)).ravel()
    free = np.setdiff1d(np.arange(ndof), remap[list(model.dirichlet)])
    load = np.zeros(ndof)
    np.add.at(load, remap, model.reference_load_vector())
    f = load[free]
    k = k[free][:, free]
    m = np.repeat(nodal_mass, 2)[free]
    penalty = penalty_ratio / (f @ spsolve(k, f))
    direction = csr_matrix(f[:, None])
    a0 = 4 / dt**2
    base = k + diags(a0 * m)
    contact_solve = factorized((base + penalty * (direction @ direction.T)).tocsc())
    open_solve = factorized(base.tocsc())
    # At the middle of each surface edge, recover stress from the rectangular
    # Q4 shape derivatives; both beam faces compete in the same search.
    bottom = np.flatnonzero(np.isclose(coords[:, 0, 1], 0.0))
    top = np.flatnonzero(np.isclose(coords[:, 3, 1], model.depth))
    sel = np.r_[bottom, top]
    surface = np.r_[np.full(len(bottom), -1), np.full(len(top), 1)]
    deriv_x = np.column_stack((-(1 - surface), 1 - surface, 1 + surface, -(1 + surface))) / (
        2 * width[sel, None]
    )
    deriv_y = np.array([-1, -1, 1, 1])[None, :] / (2 * height[sel, None])
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    acc = np.zeros_like(f)
    full = np.zeros(ndof)
    last_force = work = 0.0
    strength = model.law.peak_traction
    for step in range(1, int(np.ceil(max_time / dt)) + 1):
        stroke = step * dt * speed
        predictor = u + dt * v + 0.25 * dt**2 * acc
        rhs = a0 * m * predictor
        new = contact_solve(rhs + penalty * stroke * f)
        force = penalty * (stroke - f @ new)
        if force < 0:
            new = open_solve(rhs)
            force = 0.0
        next_acc = a0 * (new - predictor)
        v += 0.5 * dt * (acc + next_acc)
        u, acc = new, next_acc
        full[free] = u
        dis = full[dofs[sel]].reshape(-1, 4, 2)
        stress = dmat[0, 0] * (deriv_x * dis[:, :, 0]).sum(1) + dmat[0, 1] * (
            deriv_y * dis[:, :, 1]
        ).sum(1)
        index = int(np.argmax(stress))
        work += 0.5 * (last_force + force) * dt * speed
        last_force = force
        if stress[index] >= strength:
            x = float(coords[sel[index], :, 0].mean())
            kinetic = float(0.5 * m @ (v * v))
            elastic = float(0.5 * u @ (k @ u))
            penalty_energy = 0.5 * force**2 / penalty
            return dict(
                time_s=step * dt,
                dt_s=dt,
                punch_depth_m=stroke,
                sample_deflection_m=float(f @ u),
                peak_N=float(force),
                x_m=x,
                y_m=0.0 if surface[index] < 0 else model.depth,
                max_surface_tension_Pa=float(stress[index]),
                x_error_span=abs(x / model.span - 0.5),
                work_J=float(work),
                strain_energy_J=elastic,
                kinetic_energy_J=kinetic,
                penalty_energy_J=float(penalty_energy),
                energy_balance_rel=abs(work - elastic - kinetic - penalty_energy) / work,
            )
    raise RuntimeError("failure criterion not reached in the specified time")
