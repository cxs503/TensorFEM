"""Qualification closure for two-compliant-body 3-D surface contact."""
from __future__ import annotations

import math
import torch

from .global_contact3d import (
    GlobalSurfaceContactModel, NodeTrianglePair, assemble_global_surface_contact,
    initial_global_surface_state, solve_global_surface_contact_path,
)
from .mortar_contact3d import assemble_mortar_contact


def _model(*, normal_penalty=5.0e4, slave_stiffness=1.0e3,
           master_node_stiffness=1.0e3, clearance=0.05):
    d = torch.float64
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [1/3, 1/3, clearance]], dtype=d)
    k = torch.zeros((12, 12), dtype=d)
    for node in range(3):
        k[3*node:3*node+3, 3*node:3*node+3] = master_node_stiffness*torch.eye(3, dtype=d)
    k[9:12, 9:12] = slave_stiffness*torch.eye(3, dtype=d)
    fixed = tuple(3*node+axis for node in range(4) for axis in (0, 1))
    return GlobalSurfaceContactModel(
        x, k, (NodeTrianglePair(3, ((0, 1, 2),)),), fixed,
        normal_penalty, 1.e3, 0.)


def _balanced_load(model, force):
    load = torch.zeros(12, dtype=model.reference_nodes.dtype)
    load[2:9:3] = force/3
    load[11] = -force
    return load


def _reaction_oracle(*, force, clearance, normal_penalty,
                     slave_stiffness, master_node_stiffness):
    compliance = 1/slave_stiffness + 1/(3*master_node_stiffness)
    return normal_penalty*(compliance*force-clearance)/(1+normal_penalty*compliance)


def run_double_deformable_contact_qualification():
    """Solve and audit a two-compliant-body normal-contact load step."""
    force, clearance, kn, ks, km = 100., .05, 5.e4, 1.e3, 1.e3
    model = _model(normal_penalty=kn, slave_stiffness=ks,
                   master_node_stiffness=km, clearance=clearance)
    load = _balanced_load(model, force)
    initial = initial_global_surface_state(model)
    step = solve_global_surface_contact_path(model, [load], tolerance=1.e-11)[0]
    assembly = assemble_global_surface_contact(model, step.displacement, load,
                                                initial, tangent=False)
    reaction = float(assembly.updates[0].normal_force)
    oracle = _reaction_oracle(force=force, clearance=clearance,
                              normal_penalty=kn, slave_stiffness=ks,
                              master_node_stiffness=km)

    angle = .63; c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]],
                            dtype=torch.float64)
    transform = torch.kron(torch.eye(4, dtype=torch.float64), rotation)
    rotated = GlobalSurfaceContactModel(
        model.reference_nodes@rotation.T, transform@model.stiffness@transform.T,
        model.pairs, model.fixed_dofs, model.normal_penalty,
        model.tangential_penalty, model.friction)
    ur = (step.displacement.reshape(-1, 3)@rotation.T).reshape(-1)
    fr = (load.reshape(-1, 3)@rotation.T).reshape(-1)
    rotated_assembly = assemble_global_surface_contact(
        rotated, ur, fr, initial_global_surface_state(rotated), tangent=False)
    expected = assembly.contact_force.reshape(-1, 3)@rotation.T
    objectivity_error = float(torch.linalg.vector_norm(
        rotated_assembly.contact_force.reshape(-1, 3)-expected)
        / torch.linalg.vector_norm(expected))

    lower = torch.tensor([[0., 0., -1.e-3], [1., 0., -1.e-3],
                          [0., 1., -1.e-3]], dtype=torch.float64)
    upper = lower.clone(); upper[:, 2] = 0.
    face_up = torch.tensor([[0, 1, 2]], dtype=torch.long)
    face_down = torch.tensor([[0, 2, 1]], dtype=torch.long)
    zero = torch.zeros_like(lower)
    forward = assemble_mortar_contact(lower, face_up, upper, face_up,
                                      zero, zero, normal_penalty=2.e5)
    exchanged = assemble_mortar_contact(upper, face_up, lower, face_down,
                                        zero, zero, normal_penalty=2.e5)
    a = torch.sum(forward.slave_forces, dim=0)
    b = torch.sum(exchanged.master_forces, dim=0)
    interchange_error = float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))

    rollback = initial_global_surface_state(model)
    before_u = rollback.displacement.clone(); before = rollback.histories[0]
    failed = False
    try:
        solve_global_surface_contact_path(model, [load], initial_state=rollback,
                                          max_iterations=1, tolerance=1.e-14)
    except RuntimeError:
        failed = True
    rollback_exact = bool(failed and torch.equal(rollback.displacement, before_u)
        and torch.equal(rollback.histories[0].elastic_slip, before.elastic_slip)
        and rollback.histories[0].normal_multiplier == before.normal_multiplier)
    force_balance = float(torch.linalg.vector_norm(assembly.force_imbalance))
    moment_balance = float(torch.linalg.vector_norm(assembly.moment_imbalance))
    relative_error = abs(reaction/oracle-1)
    passed = (relative_error < .03 and force_balance < 1.e-10
              and moment_balance < 1.e-10 and objectivity_error < 1.e-10
              and interchange_error < .03 and rollback_exact)
    if not passed:
        raise AssertionError("double-deformable contact qualification failed")
    return {
        "schema": "tensorfem.double-deformable-contact3d-qualification/1.0",
        "reaction": reaction, "oracle_reaction": oracle,
        "reaction_relative_error": relative_error,
        "force_imbalance": force_balance, "moment_imbalance": moment_balance,
        "objectivity_relative_error": objectivity_error,
        "master_slave_interchange_relative_error": interchange_error,
        "rollback_exact": rollback_exact, "iterations": step.iterations,
        "scope": ("frictionless planar TRI3/node contact between two linearly compliant "
                  "bodies with current-configuration search and penalty enforcement; "
                  "not general double-sided mortar, curved-surface contact, self-contact, "
                  "frictional large sliding, finite-strain solids or impact"),
        "passed": True,
    }
