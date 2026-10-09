"""Transactional two-sided surface-to-surface contact qualification path."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .mortar_contact3d import MortarContactAssembly, assemble_mortar_contact


@dataclass(frozen=True)
class SurfacePatchModel:
    slave: torch.Tensor
    slave_faces: torch.Tensor
    slave_weights: torch.Tensor
    master: torch.Tensor
    master_faces: torch.Tensor
    master_weights: torch.Tensor
    slave_foundation: float
    master_foundation: float
    normal_penalty: float
    two_pass: bool = False


@dataclass(frozen=True)
class SurfacePatchState:
    slave_displacement: torch.Tensor
    master_displacement: torch.Tensor


@dataclass(frozen=True)
class SurfacePatchStep:
    state: SurfacePatchState
    pressure: float
    iterations: int
    residual_norm: float
    contact: MortarContactAssembly


def _patch_mesh(cells: int, z: float, *, reverse: bool = False):
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError("cells must be a positive integer")
    d = torch.float64
    vertices = torch.tensor([[i/cells, j/cells, z]
        for j in range(cells+1) for i in range(cells+1)], dtype=d)
    faces = []
    for j in range(cells):
        for i in range(cells):
            a = j*(cells+1)+i
            face = (a, a+1, a+cells+2, a+cells+1)
            faces.append(tuple(reversed(face)) if reverse else face)
    connectivity = torch.tensor(faces, dtype=torch.long)
    weights = torch.zeros(len(vertices), dtype=d)
    for face in connectivity:
        weights[face] += 1.0/(4*cells*cells)
    return vertices, connectivity, weights


def build_surface_patch_model(*, cells: int, master_cells: int | None = None,
                              clearance: float = .01,
                              foundation: float = 2.e4,
                              normal_penalty: float = 1.e6):
    if clearance <= 0 or foundation <= 0 or normal_penalty <= 0:
        raise ValueError("clearance, foundation and penalty must be positive")
    mcells = cells if master_cells is None else master_cells
    master, mf, mw = _patch_mesh(mcells, 0.)
    slave, sf, sw = _patch_mesh(cells, clearance)
    return SurfacePatchModel(slave, sf, sw, master, mf, mw,
                             foundation, foundation, normal_penalty)


def build_parabolic_surface_model(*, cells: int, master_cells: int | None = None,
                                  radius: float = 2.0, clearance: float = .005,
                                  foundation: float = 2.e4,
                                  normal_penalty: float = 1.e6):
    """Build a shallow spherical/parabolic cap above a deformable flat patch."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    model=build_surface_patch_model(cells=cells,master_cells=master_cells,
        clearance=clearance,foundation=foundation,normal_penalty=normal_penalty)
    slave=model.slave.clone()
    radial=(slave[:,0]-.5)**2+(slave[:,1]-.5)**2
    slave[:,2]=clearance+radial/(2*radius)
    return SurfacePatchModel(slave,model.slave_faces,model.slave_weights,
        model.master,model.master_faces,model.master_weights,
        model.slave_foundation,model.master_foundation,model.normal_penalty,True)


def _reflected_role_exchange(model: SurfacePatchModel):
    """Exchange surface discretisations and reverse the new master normal."""
    return SurfacePatchModel(
        model.master, torch.flip(model.master_faces, [1]), model.master_weights,
        model.slave, torch.flip(model.slave_faces, [1]), model.slave_weights,
        model.master_foundation, model.slave_foundation, model.normal_penalty,
        model.two_pass)


def initial_surface_patch_state(model: SurfacePatchModel):
    return SurfacePatchState(torch.zeros_like(model.slave),
                             torch.zeros_like(model.master))


def _assemble(model: SurfacePatchModel, state: SurfacePatchState, pressure: float,
              *, tangent: bool = True):
    contact = assemble_mortar_contact(
        model.slave, model.slave_faces, model.master, model.master_faces,
        state.slave_displacement, state.master_displacement,
        normal_penalty=model.normal_penalty, tangent=tangent)
    if model.two_pass:
        reverse = assemble_mortar_contact(
            model.master, torch.flip(model.master_faces, [1]),
            model.slave, torch.flip(model.slave_faces, [1]),
            state.master_displacement, state.slave_displacement,
            normal_penalty=model.normal_penalty, tangent=tangent)
        ns, nm = model.slave.numel(), model.master.numel()
        order = torch.cat((torch.arange(nm, nm+ns), torch.arange(0, nm)))
        reverse_residual = reverse.residual[order]
        reverse_tangent = reverse.tangent[order][:, order]
        residual = .5*(contact.residual+reverse_residual)
        matrix = .5*(contact.tangent+reverse_tangent)
        slave_forces = .5*(contact.slave_forces+reverse.master_forces)
        master_forces = .5*(contact.master_forces+reverse.slave_forces)
        contact = MortarContactAssembly(
            residual, matrix, slave_forces, master_forces,
            contact.active_quadrature_points+reverse.active_quadrature_points,
            .5*(contact.integrated_area+reverse.integrated_area),
            torch.maximum(contact.maximum_penetration,reverse.maximum_penetration),
            .5*(contact.penalty_energy+reverse.penalty_energy))
    ns = model.slave.numel()
    u = torch.cat((state.slave_displacement.reshape(-1),
                   state.master_displacement.reshape(-1)))
    stiffness = torch.zeros_like(u)
    stiffness[:ns].reshape(-1, 3)[:, 2] = model.slave_foundation*model.slave_weights
    stiffness[ns:].reshape(-1, 3)[:, 2] = model.master_foundation*model.master_weights
    external = torch.zeros_like(u)
    face = model.master_faces[0]
    triangle = model.master[face[:3]]
    normal = torch.linalg.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    normal = normal/torch.linalg.vector_norm(normal)
    external[:ns].reshape(-1, 3)[:] = -pressure*model.slave_weights[:,None]*normal
    external[ns:].reshape(-1, 3)[:] = pressure*model.master_weights[:,None]*normal
    residual = stiffness*u-external+contact.residual
    matrix = contact.tangent + torch.diag(stiffness) if tangent else contact.tangent
    free = torch.cat((torch.arange(2, ns, 3), torch.arange(ns+2, len(u), 3)))
    return contact, residual, matrix, free


def solve_surface_patch_path(model: SurfacePatchModel, pressures: Sequence[float], *,
                             initial_state: SurfacePatchState | None = None,
                             tolerance: float = 1.e-10,
                             max_iterations: int = 20):
    """Solve pressure increments, committing displacement only on convergence."""
    committed = initial_surface_patch_state(model) if initial_state is None else initial_state
    steps = []
    for pressure in pressures:
        if not math.isfinite(pressure) or pressure < 0:
            raise ValueError("pressures must be finite and nonnegative")
        trial = SurfacePatchState(committed.slave_displacement.clone(),
                                  committed.master_displacement.clone())
        converged = False; norm = float("inf")
        for iteration in range(1, max_iterations+1):
            contact, residual, tangent, free = _assemble(model, trial, pressure)
            norm = float(torch.linalg.vector_norm(residual[free]))
            if norm <= tolerance*max(1., pressure):
                converged = True; break
            delta = torch.linalg.solve(tangent[free][:, free], -residual[free])
            flat_s = trial.slave_displacement.clone().reshape(-1)
            flat_m = trial.master_displacement.clone().reshape(-1)
            joined = torch.cat((flat_s, flat_m)); joined[free] += delta
            ns = flat_s.numel()
            trial = SurfacePatchState(joined[:ns].reshape_as(model.slave),
                                      joined[ns:].reshape_as(model.master))
        if not converged:
            raise RuntimeError("surface contact Newton solve failed; committed state unchanged")
        final, _, _, _ = _assemble(model, trial, pressure, tangent=False)
        committed = trial
        steps.append(SurfacePatchStep(committed, pressure, iteration, norm, final))
    return tuple(steps)


def _pressure_oracle(*, applied: float, clearance: float, slave_foundation: float,
                     master_foundation: float, normal_penalty: float):
    compliance = 1/slave_foundation+1/master_foundation
    return normal_penalty*(compliance*applied-clearance)/(1+normal_penalty*compliance)


def run_surface_surface_contact_qualification():
    """Run matching-mesh convergence, balance, objectivity, exchange and rollback."""
    applied, clearance, foundation, penalty = 180., .01, 2.e4, 1.e6
    oracle = _pressure_oracle(applied=applied, clearance=clearance,
        slave_foundation=foundation, master_foundation=foundation,
        normal_penalty=penalty)
    rows = []
    for cells in (1, 2, 3):
        model = build_surface_patch_model(cells=cells, clearance=clearance,
                                          foundation=foundation,
                                          normal_penalty=penalty)
        step = solve_surface_patch_path(model, [applied])[0]
        force = torch.cat((step.contact.slave_forces, step.contact.master_forces))
        points = torch.cat((model.slave+step.state.slave_displacement,
                            model.master+step.state.master_displacement))
        resultant = float(torch.linalg.vector_norm(force.sum(0)))
        moment = float(torch.linalg.vector_norm(torch.linalg.cross(points, force).sum(0)))
        pressure = float(torch.sum(step.contact.slave_forces[:, 2]))
        rows.append({"cells": cells, "quadrature_points": 4*cells*cells,
                     "contact_pressure": pressure,
                     "oracle_pressure": oracle,
                     "relative_error": abs(pressure/oracle-1),
                     "force_imbalance": resultant, "moment_imbalance": moment,
                     "iterations": step.iterations})
    model = build_surface_patch_model(cells=2, clearance=clearance,
                                      foundation=foundation, normal_penalty=penalty)
    step = solve_surface_patch_path(model, [applied])[0]
    angle=.61; c,s=math.cos(angle),math.sin(angle)
    r=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=torch.float64)
    rotated = SurfacePatchModel(model.slave@r.T, model.slave_faces, model.slave_weights,
        model.master@r.T, model.master_faces, model.master_weights,
        foundation, foundation, penalty)
    state_r=SurfacePatchState(step.state.slave_displacement@r.T,
                              step.state.master_displacement@r.T)
    # Compare like with like: the curved qualification model is symmetric
    # two-pass, so its rotated audit must use the same complete assembly rather
    # than a single directional mortar pass.
    contact_r,_,_,_=_assemble(rotated,state_r,applied,tangent=False)
    expected=step.contact.slave_forces@r.T
    objectivity=float(torch.linalg.vector_norm(contact_r.slave_forces-expected)
                      /torch.linalg.vector_norm(expected))
    exchanged=assemble_mortar_contact(model.master, torch.flip(model.master_faces,[1]),
        model.slave, torch.flip(model.slave_faces,[1]), step.state.master_displacement,
        step.state.slave_displacement, normal_penalty=penalty, tangent=False)
    exchange=float(abs(torch.linalg.vector_norm(exchanged.slave_forces)
                       /torch.linalg.vector_norm(step.contact.master_forces)-1))
    initial=initial_surface_patch_state(model); before=initial.slave_displacement.clone()
    failed=False
    try:
        solve_surface_patch_path(model,[applied],initial_state=initial,max_iterations=1,
                                 tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before))
    errors=[row["relative_error"] for row in rows]
    passed=(max(errors)<.03 and all(a>b for a,b in zip(errors,errors[1:]))
            and max(row["force_imbalance"] for row in rows)<1.e-9
            and max(row["moment_imbalance"] for row in rows)<1.e-9
            and objectivity<1.e-10 and exchange<.03 and rollback)
    if not passed: raise AssertionError("surface-to-surface contact qualification failed")
    return {"schema":"tensorfem.surface-surface-contact3d-qualification/1.0",
            "mesh_sequence":rows,"objectivity_relative_error":objectivity,
            "master_slave_exchange_relative_error":exchange,"rollback_exact":rollback,
            "scope":("matching planar QUAD4 surface-to-surface frictionless penalty contact "
                     "with multiple quadrature points and two Winkler-compliant bodies; not "
                     "curved Hertz contact, nonmatching production mortar, friction, "
                     "self-contact, finite strain or impact"),"passed":True}


def run_nonmatching_surface_contact_qualification():
    """Qualify complete Newton paths with unequal slave/master meshes."""
    applied, clearance, foundation, penalty = 180., .01, 2.e4, 1.e6
    oracle = _pressure_oracle(applied=applied, clearance=clearance,
        slave_foundation=foundation, master_foundation=foundation,
        normal_penalty=penalty)
    meshes = ((1, 2), (2, 3), (3, 4))
    rows=[]
    for slave_cells, master_cells in meshes:
        model=build_surface_patch_model(cells=slave_cells,master_cells=master_cells,
            clearance=clearance,foundation=foundation,normal_penalty=penalty)
        step=solve_surface_patch_path(model,[applied])[0]
        force=torch.cat((step.contact.slave_forces,step.contact.master_forces))
        points=torch.cat((model.slave+step.state.slave_displacement,
                          model.master+step.state.master_displacement))
        pressure=float(torch.sum(step.contact.slave_forces[:,2]))
        rows.append({"slave_cells":slave_cells,"master_cells":master_cells,
            "contact_pressure":pressure,"oracle_pressure":oracle,
            "relative_error":abs(pressure/oracle-1),
            "force_imbalance":float(torch.linalg.vector_norm(force.sum(0))),
            "moment_imbalance":float(torch.linalg.vector_norm(
                torch.linalg.cross(points,force).sum(0))),"iterations":step.iterations})

    model=build_surface_patch_model(cells=2,master_cells=3,clearance=clearance,
        foundation=foundation,normal_penalty=penalty)
    step=solve_surface_patch_path(model,[applied])[0]
    exchanged_model=_reflected_role_exchange(model)
    exchanged_step=solve_surface_patch_path(exchanged_model,[applied])[0]
    p=float(torch.sum(step.contact.slave_forces[:,2]))
    pe=float(torch.sum(exchanged_step.contact.slave_forces[:,2]))
    interchange=abs(abs(pe)/abs(p)-1)

    angle=.57;c,s=math.cos(angle),math.sin(angle)
    rotation=torch.tensor([[c,0.,s],[0.,1.,0.],[-s,0.,c]],dtype=torch.float64)
    rotated=SurfacePatchModel(model.slave@rotation.T,model.slave_faces,model.slave_weights,
        model.master@rotation.T,model.master_faces,model.master_weights,
        foundation,foundation,penalty,model.two_pass)
    rotated_state=SurfacePatchState(step.state.slave_displacement@rotation.T,
                                    step.state.master_displacement@rotation.T)
    rotated_contact=assemble_mortar_contact(rotated.slave,rotated.slave_faces,
        rotated.master,rotated.master_faces,rotated_state.slave_displacement,
        rotated_state.master_displacement,normal_penalty=penalty,tangent=False)
    expected=step.contact.slave_forces@rotation.T
    objectivity=float(torch.linalg.vector_norm(rotated_contact.slave_forces-expected)
                      /torch.linalg.vector_norm(expected))

    initial=initial_surface_patch_state(model); before_s=initial.slave_displacement.clone()
    before_m=initial.master_displacement.clone(); failed=False
    try:
        solve_surface_patch_path(model,[applied],initial_state=initial,max_iterations=1,
                                 tolerance=1.e-14)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before_s)
                  and torch.equal(initial.master_displacement,before_m))
    errors=[row["relative_error"] for row in rows]
    passed=(errors[-1]<.03 and all(a>b for a,b in zip(errors,errors[1:]))
        and max(row["force_imbalance"] for row in rows)<1.e-9
        and max(row["moment_imbalance"] for row in rows)<1.e-9
        and objectivity<1.e-10 and interchange<.03 and rollback)
    if not passed: raise AssertionError("nonmatching surface contact qualification failed")
    return {"schema":"tensorfem.nonmatching-surface-contact3d-qualification/1.0",
        "mesh_sequence":rows,"objectivity_relative_error":objectivity,
        "complete_newton_role_exchange_relative_error":interchange,
        "rollback_exact":rollback,"general_surface_to_surface":"blocked",
        "remaining_blockers":["curved-surface public benchmark",
            "production segmentation integration","frictional two-pass contact"],
        "scope":("planar nonmatching QUAD4 frictionless penalty-Mortar with two "
                 "compliant bodies; general curved/frictional surface-to-surface remains blocked"),
        "passed":True}


def run_curved_surface_contact_qualification():
    """Qualify a shallow spherical cap against a compliant plane."""
    applied,gap,radius,foundation,penalty=250.,.005,4.,2.e4,1.e6
    target_indentation=2*applied/foundation-gap
    effective=1/(1/penalty+2/foundation)
    oracle=math.pi*effective*radius*target_indentation**2
    rows=[]
    audit_model=None; audit_step=None
    for slave_cells,master_cells in ((2,3),(3,4),(4,5)):
        model=build_parabolic_surface_model(cells=slave_cells,master_cells=master_cells,
            radius=radius,clearance=gap,foundation=foundation,normal_penalty=penalty)
        step=solve_surface_patch_path(model,[applied])[0]
        if (slave_cells,master_cells)==(3,4):
            audit_model,audit_step=model,step
        forces=torch.cat((step.contact.slave_forces,step.contact.master_forces))
        points=torch.cat((model.slave+step.state.slave_displacement,
                          model.master+step.state.master_displacement))
        load=float(torch.linalg.vector_norm(torch.sum(step.contact.slave_forces,dim=0)))
        rows.append({"slave_cells":slave_cells,"master_cells":master_cells,
            "contact_load":load,"winkler_paraboloid_oracle":oracle,
            "relative_error":abs(load/oracle-1),
            "force_imbalance":float(torch.linalg.vector_norm(forces.sum(0))),
            "moment_imbalance":float(torch.linalg.vector_norm(
                torch.linalg.cross(points,forces).sum(0))),"iterations":step.iterations})
    assert audit_model is not None and audit_step is not None
    model,step=audit_model,audit_step
    angle=.43;c,s=math.cos(angle),math.sin(angle)
    rotation=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=torch.float64)
    rotated=SurfacePatchModel(model.slave@rotation.T,model.slave_faces,model.slave_weights,
        model.master@rotation.T,model.master_faces,model.master_weights,
        foundation,foundation,penalty,model.two_pass)
    state_r=SurfacePatchState(step.state.slave_displacement@rotation.T,
                              step.state.master_displacement@rotation.T)
    # Compare like with like: the curved qualification model is symmetric
    # two-pass, so its rotated audit must use the same complete assembly rather
    # than a single directional mortar pass.
    contact_r,_,_,_=_assemble(rotated,state_r,applied,tangent=False)
    expected=step.contact.slave_forces@rotation.T
    objectivity=float(torch.linalg.vector_norm(contact_r.slave_forces-expected)
                      /torch.linalg.vector_norm(expected))
    exchanged_model=_reflected_role_exchange(model)
    exchanged=solve_surface_patch_path(exchanged_model,[applied])[0]
    load=float(torch.linalg.vector_norm(step.contact.slave_forces.sum(0)))
    exchanged_load=float(torch.linalg.vector_norm(exchanged.contact.slave_forces.sum(0)))
    interchange=abs(exchanged_load/load-1)
    initial=initial_surface_patch_state(model);bs=initial.slave_displacement.clone()
    bm=initial.master_displacement.clone();failed=False
    try: solve_surface_patch_path(model,[applied],initial_state=initial,max_iterations=1)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,bs)
                  and torch.equal(initial.master_displacement,bm))
    errors=[row["relative_error"] for row in rows]
    passed=(errors[-1]<.03 and errors[-1]<errors[0]
        and max(row["force_imbalance"] for row in rows)<1.e-9
        and max(row["moment_imbalance"] for row in rows)<1.e-9
        and objectivity<1.e-10 and interchange<.03 and rollback)
    if not passed:
        raise AssertionError("curved surface contact qualification failed: "
            f"errors={errors}, force={max(row['force_imbalance'] for row in rows)}, "
            f"moment={max(row['moment_imbalance'] for row in rows)}, "
            f"objectivity={objectivity}, interchange={interchange}, rollback={rollback}")
    return {"schema":"tensorfem.curved-surface-contact3d-qualification/1.0",
        "model":"shallow parabolic sphere on two-sided Winkler foundations",
        "mesh_sequence":rows,"target_indentation":target_indentation,
        "objectivity_relative_error":objectivity,
        "complete_newton_role_exchange_relative_error":interchange,
        "rollback_exact":rollback,"general_surface_to_surface":"qualified_curved_frictionless_subset",
        "remaining_blockers":["frictional two-pass contact","self-contact",
            "classical elastic-halfspace Hertz solid discretisation"],
        "scope":("nonmatching faceted parabolic-cap/plane double-deformable complete "
                 "Newton contact with a Winkler sphere-indentation oracle; not classical "
                 "Hertz elastic halfspaces, friction, self-contact, finite strain or impact"),
        "passed":True}


def run_curved_surface_contact_smoke():
    """Fast default gate for the expensive symmetric curved-contact path."""
    model=build_parabolic_surface_model(cells=2,master_cells=3,radius=4.,
        clearance=.005,foundation=2.e4,normal_penalty=1.e6)
    step=solve_surface_patch_path(model,[250.])[0]
    forces=torch.cat((step.contact.slave_forces,step.contact.master_forces))
    imbalance=float(torch.linalg.vector_norm(forces.sum(0)))
    initial=initial_surface_patch_state(model);before=initial.slave_displacement.clone()
    failed=False
    try: solve_surface_patch_path(model,[250.],initial_state=initial,max_iterations=1)
    except RuntimeError: failed=True
    rollback=bool(failed and torch.equal(initial.slave_displacement,before))
    if imbalance>=1.e-9 or not rollback or step.contact.active_quadrature_points<2:
        raise AssertionError("curved surface contact smoke gate failed")
    return {"schema":"tensorfem.curved-surface-contact3d-smoke/1.0",
        "two_pass":model.two_pass,"active_quadrature_points":step.contact.active_quadrature_points,
        "force_imbalance":imbalance,"rollback_exact":rollback,"passed":True}
