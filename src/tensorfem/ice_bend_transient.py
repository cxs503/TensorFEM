"""Transient mode-I rupture of a Q4 coupon, including fragment kinetic energy."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .cohesive_bending import _GP_S, _GP_W, _assemble_reference


def rupture(
    model, *, step_fraction=0.65, speed=0.0025, density=917.0, punch_ratio=100.0, fixed_dt=None
):
    """Velocity-Verlet with lumped mass, unilateral punch, irreversible cohesion.

    The elastic constant-speed particular solution initializes the specimen
    just before the first softening point. Lost-contact duration and actual
    interface failure are checked before ending the postpeak event.
    """
    if not 0 < step_fraction < 1 or min(speed, density, punch_ratio) <= 0:
        raise ValueError("positive physical parameters and 0 < step_fraction < 1 required")
    bulk, ref = _assemble_reference(model)
    free = np.setdiff1d(np.arange(model.n_dofs), list(model.dirichlet))
    f = ref[free]
    coords = model.nodes[model.elements]
    area = (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 3, 1] - coords[:, 0, 1])
    mass = np.zeros(len(model.nodes))
    np.add.at(mass, model.elements.ravel(), np.repeat(area * density * model.thickness / 4, 4))
    m = np.repeat(mass, 2)[free]
    npts = len(model.interface) * len(_GP_S)
    rows = np.repeat(np.arange(npts), 4)
    cols = np.repeat(model.interface, len(_GP_S), axis=0)
    shape = np.array([[(1 - x) / 2, (1 + x) / 2] for x in _GP_S])
    entries = np.tile(np.column_stack((-shape, shape)), (len(model.interface), 1))

    def jump(axis):
        return coo_matrix(
            (entries.ravel(), (rows, (2 * cols + axis).ravel())), shape=(npts, model.n_dofs)
        ).tocsr()[:, free]

    normal = jump(0)
    tangent = jump(1)
    w = (model.segment_heights[:, None] * np.array(_GP_W) * model.thickness / 2).ravel()
    law = model.law
    linear = bulk[free][:, free] + tangent.T @ diags(w * law.tangential_stiffness) @ tangent
    nt = normal.T.tocsr()
    intact = linear + nt @ diags(w * law.normal_stiffness) @ normal
    unit = spsolve(intact.tocsc(), f)
    penalty = punch_ratio / (f @ unit)
    # Series combination of specimen compliance and punch compliance.
    per_stroke = unit / (f @ unit + 1 / penalty)
    onset = law.damage_opening / np.max(normal @ per_stroke)
    stroke = 0.995 * onset
    u = per_stroke * stroke
    v = per_stroke * speed
    acc = np.zeros_like(u)
    kappa = np.maximum(normal @ u, 0)
    force = penalty * (stroke - f @ u)
    work = 0.5 * force * stroke
    initial_kinetic = 0.5 * (m * v) @ v
    force_peak = force
    peak_displacement = float(f @ u)
    # Row-sum bound on the largest elastic eigenfrequency squared.
    direction = csr_matrix(f[:, None])
    kmax = intact + penalty * (direction @ direction.T)
    norm_mass = diags(1 / np.sqrt(m))
    bound = np.asarray(abs(norm_mass @ kmax @ norm_mass).sum(axis=1)).ravel().max()
    stable_dt = 2 / np.sqrt(bound)
    dt = step_fraction * stable_dt if fixed_dt is None else fixed_dt
    if not 0 < dt < stable_dt:
        raise ValueError("time step must satisfy the elastic stability bound")
    step_fraction = dt / stable_dt
    transit = model.span / np.sqrt(model.young_modulus / density)
    cap = int(np.ceil((0.8 * onset / speed + 4 * transit) / dt))
    dwell = 0
    rows_out = [[stroke, force, float(f @ u)]]
    reason = "stroke_limit"
    # Inline the vector form of the same bilinear law for the explicit update.
    d0 = law.damage_opening
    df = law.failure_opening
    for i in range(1, cap + 1):
        half = v + 0.5 * dt * acc
        u += dt * half
        stroke += dt * speed
        opening = normal @ u
        np.maximum(kappa, opening, out=kappa)
        damage = np.clip(df / (df - d0) * (1 - d0 / np.maximum(kappa, d0)), 0, 1)
        traction = law.normal_stiffness * (
            np.minimum(opening, 0) + (1 - damage) * np.maximum(opening, 0)
        )
        next_force = penalty * max(stroke - f @ u, 0.0)
        acc = (next_force * f - linear @ u - nt @ (w * traction)) / m
        v = half + 0.5 * dt * acc
        work += 0.5 * (force + next_force) * dt * speed
        force = next_force
        if force > force_peak:
            force_peak = force
            peak_displacement = float(f @ u)
        if i % max(1, int(transit / (12 * dt))) == 0:
            rows_out.append([stroke, force, float(f @ u)])
        if force < 0.02 * force_peak and np.mean(damage > 0.999) > 0.70:
            dwell += 1
        else:
            dwell = 0
        if dwell * dt >= transit:
            reason = "sustained_postpeak_unloading"
            break
    rows_out.append([stroke, force, float(f @ u)])
    kinetic = float(0.5 * (m * v) @ v)
    stored = float(0.5 * (u @ (linear @ u) + w @ (traction * opening)))
    diss = float(w @ law.dissipated_energy(kappa))
    contact = 0.5 * force**2 / penalty
    end = stored + kinetic + diss + contact
    return dict(
        dt_s=dt,
        step_fraction=step_fraction,
        steps=i,
        preload_stroke_m=0.995 * onset,
        postpeak_stop=reason,
        peak_N=force_peak,
        peak_deflection_m=peak_displacement,
        end_force_N=force,
        external_work_J=work,
        initial_kinetic_J=initial_kinetic,
        stored_energy_J=stored,
        kinetic_energy_J=kinetic,
        fracture_dissipation_J=diss,
        contact_energy_J=contact,
        energy_balance_rel=abs(work + initial_kinetic - end) / work,
        failed_area_fraction=float(w @ (kappa >= df) / w.sum()),
        punch_ratio=punch_ratio,
        history_columns=["punch_stroke_m", "force_N", "sample_deflection_m"],
        history=rows_out,
    )
