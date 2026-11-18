"""Displacement-controlled three-point bending with irreversible cohesive damage.

The Q4 bulk is statically condensed. Alternate equilibrium and local damage
updates can follow stable states after sudden load drops at prescribed stroke.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse.linalg import splu

from .cohesive_bending import _assemble_reference, _GP_S, _GP_W


def stroke_control(
    model, *, max_stroke, increments=200, max_iterations=1200, damage_tolerance=1.0e-9
):
    if min(max_stroke, increments, max_iterations, damage_tolerance) <= 0:
        raise ValueError("positive stroke, increment count and solver limits required")
    k, _ = _assemble_reference(model)
    boundary_nodes = np.unique(model.interface)
    keep = np.unique(
        np.r_[2 * boundary_nodes, 2 * boundary_nodes + 1, model.load_dofs, list(model.dirichlet)]
    )
    inside = np.setdiff1d(np.arange(model.n_dofs), keep)
    condensation = splu(k[inside][:, inside].tocsc()).solve(k[inside][:, keep].toarray())
    bulk = k[keep][:, keep].toarray() - k[keep][:, inside] @ condensation
    index = np.full(model.n_dofs, -1, dtype=int)
    index[keep] = np.arange(len(keep))
    punch = np.unique(index[model.load_dofs])
    free = np.setdiff1d(np.arange(len(keep)), np.r_[punch, index[list(model.dirichlet)]])
    law = model.law
    normal, tangent, weights = [], [], []
    for nodes, height in zip(model.interface, model.segment_heights):
        for xi, weight in zip(_GP_S, _GP_W):
            shape = np.array([(1 - xi) / 2, (1 + xi) / 2])
            jump = np.r_[-shape, shape]
            # Straight vertical interface; x is the opening and y the sliding direction.
            bn = np.zeros(len(keep))
            bt = np.zeros(len(keep))
            bn[index[2 * nodes]] = jump
            bt[index[2 * nodes + 1]] = jump
            normal.append(bn)
            tangent.append(bt)
            weights.append(height * model.thickness * weight / 2)
    bn, bt, w = np.array(normal), np.array(tangent), np.array(weights)
    undamaged = bulk + bt.T @ ((w * law.tangential_stiffness)[:, None] * bt)
    u = np.zeros(len(keep))
    kappa = np.zeros(len(w))
    force, stroke, energy, dissipation, crack, iterations = [[0.0] for _ in range(6)]
    converged = True
    reason = "stroke_limit"
    for depth in np.linspace(0, max_stroke, increments + 1)[1:]:
        u[punch] = -depth
        state = kappa.copy()
        current = bn @ u
        for it in range(max_iterations):
            damage = law.damage(state)
            spring = np.where(current < 0, 1.0, 1.0 - damage) * law.normal_stiffness
            system = undamaged + bn.T @ ((w * spring)[:, None] * bn)
            u[free] = np.linalg.solve(
                system[np.ix_(free, free)], -system[np.ix_(free, punch)] @ u[punch]
            )
            opening = bn @ u
            candidate = np.maximum(state, opening)
            change = np.max(np.abs(law.damage(candidate) - damage))
            # A change of unilateral branch also requires a new equilibrium solve.
            switched = (current < 0) != (opening < 0)
            current = opening
            state = candidate
            if change < damage_tolerance and not np.any(switched & (damage > damage_tolerance)):
                break
        else:
            reason = "iteration_limit"
            converged = False
            break
        kappa = state
        damage = law.damage(kappa)
        spring = np.where(opening < 0, 1.0, 1.0 - damage) * law.normal_stiffness
        traction = spring * opening
        residual = undamaged @ u + bn.T @ (w * traction)
        load = -float(residual[punch].sum())
        if np.max(np.abs(residual[free])) > 1.0e-6 * max(load, 1.0):
            reason = "equilibrium_residual"
            converged = False
            break
        # D = total envelope work minus recoverable secant energy at historical maximum.
        dissipated = w @ law.dissipated_energy(kappa)
        stored = 0.5 * (u @ undamaged @ u + w @ (traction * opening))
        force.append(load)
        stroke.append(float(depth))
        energy.append(float(stored))
        dissipation.append(float(dissipated))
        crack.append(
            float(
                model.segment_heights
                @ (damage.reshape(model.n_segments, -1).min(axis=1) >= 1.0 - 1e-8)
            )
        )
        iterations.append(it + 1)
        if max(force) > 1.0 and load < 0.02 * max(force):
            reason = "residual_load_below_2pct"
            break
    peak = int(np.argmax(force))
    work = float(trapezoid(force, stroke))
    return dict(
        force_N=force,
        deflection_m=stroke,
        peak_N=force[peak],
        peak_deflection_m=stroke[peak],
        converged=converged,
        stop_reason=reason,
        iterations=iterations,
        max_stroke_m=max_stroke,
        increments=increments,
        external_work_J=work,
        stored_energy_J=energy[-1],
        fracture_dissipation_J=dissipation[-1],
        final_crack_height_m=crack[-1],
        unresolved_release_J=work - energy[-1] - dissipation[-1],
        irreversibility_violation_J=float(max(0, -min(np.diff(dissipation), default=0))),
    )
