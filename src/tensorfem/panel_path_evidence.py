"""Energy and failure evidence for accepted nonlinear shell-panel paths.

The routines in this module are observers: they never update or commit a
material state.  In particular, rejected arc-length iterates cannot enter an
energy history.  Stored energy is reconstructed from the accepted displacement
and J2 state; plastic dissipation follows the associative-isotropic-hardening
identity ``D = sigma_y * Delta alpha``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from .finite_rotation_layered_shell4 import _corotated_deformation
from .layered_shell4_plasticity import (
    LayeredShell4Model, LayeredShell4State, _elastic_shear_drilling,
    _generalized_b, _residual_strain, layered_shell4_local_frame,
)


@dataclass(frozen=True)
class ShellStoredEnergy:
    elastic: float
    hardening: float

    @property
    def recoverable(self) -> float:
        return self.elastic + self.hardening


@dataclass(frozen=True)
class PanelPointEvidence:
    step: int
    load_factor: float
    axial_shortening: float
    maximum_transverse_displacement: float
    external_work: float
    recoverable_energy: float
    plastic_dissipation: float
    internal_energy: float
    energy_residual: float
    relative_energy_residual: float
    yielded_fraction: float
    is_peak: bool
    is_post_peak: bool
    failure_mode: str


@dataclass(frozen=True)
class PanelPathEvidence:
    points: tuple[PanelPointEvidence, ...]
    peak_step: int | None
    peak_load: float | None
    post_peak_confirmed: bool
    failure_mode: str


def _element_ids(conn: torch.Tensor) -> torch.Tensor:
    return torch.stack(tuple(6 * conn + i for i in range(6)), 1).reshape(-1).long()


def shell_stored_energy(model: LayeredShell4Model, displacement: torch.Tensor,
                        state: LayeredShell4State) -> ShellStoredEnergy:
    """Reconstruct elastic plus isotropic-hardening energy of an accepted state."""
    model.validate()
    if displacement.shape != (model.n_dofs,):
        raise ValueError("wrong displacement vector length")
    if len(state.points) != len(model.elements):
        raise ValueError("state/model element count mismatch")
    elastic = displacement.new_zeros(())
    hardening = displacement.new_zeros(())
    shear_modulus = model.young / (2 * (1 + model.poisson))
    lame = model.young * model.poisson / ((1 + model.poisson) * (1 - 2*model.poisson))
    gp = 1 / math.sqrt(3)
    gauss = ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp))
    dz = model.thickness / model.layers
    zvalues = tuple(-model.thickness/2 + (i+.5)*dz for i in range(model.layers))
    for e, conn in enumerate(model.elements):
        ids = _element_ids(conn)
        xyz = model.nodes[conn.long()]
        xy, _ = layered_shell4_local_frame(xyz)
        local = _corotated_deformation(xyz, displacement[ids])
        shear_force, _ = _elastic_shear_drilling(xy, local, model)
        elastic = elastic + .5 * torch.dot(local, shear_force)
        for q, (xi, eta) in enumerate(gauss):
            b, determinant = _generalized_b(xy, xi, eta)
            generalized = b @ local
            for layer, z in enumerate(zvalues):
                point = state.points[e][q][layer]
                inplane = generalized[:3] + z*generalized[3:]
                initial = torch.zeros((3, 3), dtype=xyz.dtype, device=xyz.device)
                if model.residual_stress is not None:
                    initial = _residual_strain(
                        model.residual_stress[e, q, layer], model.young, model.poisson
                    )
                # Recover eps_zz from sigma_zz=0 using the already accepted
                # plastic strain.  No return map is rerun and no history moves.
                known_x = inplane[0] + initial[0, 0] - point.plastic_strain[0, 0]
                known_y = inplane[1] + initial[1, 1] - point.plastic_strain[1, 1]
                elastic_zz = -lame * (known_x + known_y) / (lame + 2*shear_modulus)
                full = torch.stack((
                    torch.stack((inplane[0], .5*inplane[2], inplane.new_zeros(()))),
                    torch.stack((.5*inplane[2], inplane[1], inplane.new_zeros(()))),
                    torch.stack((inplane.new_zeros(()), inplane.new_zeros(()),
                                 elastic_zz-initial[2, 2]+point.plastic_strain[2, 2])),
                ))
                ee = full + initial - point.plastic_strain
                density = shear_modulus*torch.sum(ee*ee) + .5*lame*torch.trace(ee)**2
                weight = determinant*dz
                elastic = elastic + density*weight
                hardening = hardening + .5*model.hardening*point.alpha**2*weight
    return ShellStoredEnergy(float(elastic), float(hardening))


def plastic_dissipation_increment(model: LayeredShell4Model,
                                  before: LayeredShell4State,
                                  after: LayeredShell4State) -> float:
    """Integrate ``sigma_y Delta alpha`` and reject non-transactional history."""
    total = model.nodes.new_zeros(())
    gp = 1/math.sqrt(3)
    gauss = ((-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp))
    dz = model.thickness/model.layers
    if len(before.points) != len(model.elements) or len(after.points) != len(model.elements):
        raise ValueError("state/model element count mismatch")
    for e, conn in enumerate(model.elements):
        xy, _ = layered_shell4_local_frame(model.nodes[conn.long()])
        for q, (xi, eta) in enumerate(gauss):
            _, determinant = _generalized_b(xy, xi, eta)
            for layer in range(model.layers):
                delta = after.points[e][q][layer].alpha-before.points[e][q][layer].alpha
                tolerance = 64*torch.finfo(model.nodes.dtype).eps*max(
                    1., abs(float(before.points[e][q][layer].alpha)))
                if float(delta) < -tolerance:
                    raise ValueError("equivalent plastic strain decreased across accepted points")
                total = total + model.yield_stress*torch.clamp(delta, min=0)*determinant*dz
    return float(total)


def yielded_fraction(model: LayeredShell4Model, state: LayeredShell4State,
                     baseline: LayeredShell4State | None = None, *,
                     tolerance: float = 1e-12) -> float:
    """Volume-weighted fraction of points plastified relative to the baseline."""
    baseline = LayeredShell4State.virgin(model) if baseline is None else baseline
    yielded = model.nodes.new_zeros(()); volume = model.nodes.new_zeros(())
    gp = 1/math.sqrt(3); dz = model.thickness/model.layers
    for e, conn in enumerate(model.elements):
        xy, _ = layered_shell4_local_frame(model.nodes[conn.long()])
        for q, (xi, eta) in enumerate(((-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp))):
            _, det = _generalized_b(xy, xi, eta)
            for layer in range(model.layers):
                weight = det*dz; volume += weight
                da = state.points[e][q][layer].alpha-baseline.points[e][q][layer].alpha
                if float(da) > tolerance:
                    yielded += weight
    return float(yielded/volume) if float(volume) else 0.0


def evaluate_panel_path(model: LayeredShell4Model, reference_load: torch.Tensor,
                        accepted_points: Iterable[object], *,
                        initial_displacement: torch.Tensor | None = None,
                        initial_state: LayeredShell4State | None = None,
                        initial_load_factor: float = 0.,
                        initial_external_work: float = 0.,
                        initial_plastic_dissipation: float = 0.,
                        reference_recoverable_energy: float | None = None,
                        post_peak_drop: float = .02) -> PanelPathEvidence:
    """Build auditable evidence from accepted points only.

    A post-peak branch requires both increasing conjugate shortening and a load
    at least ``post_peak_drop`` below the preceding maximum.  A mere negative
    tangent or a single noisy load reversal is therefore not called collapse.
    """
    if not (0 < post_peak_drop < 1):
        raise ValueError("post_peak_drop must be between zero and one")
    load = reference_load.to(model.nodes)
    if load.shape != (model.n_dofs,):
        raise ValueError("wrong shell load length")
    u0 = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    if initial_displacement is not None:
        u0 = initial_displacement.to(model.nodes).clone()
    s0 = LayeredShell4State.virgin(model) if initial_state is None else initial_state
    base_energy = (shell_stored_energy(model, u0, s0).recoverable
                   if reference_recoverable_energy is None
                   else float(reference_recoverable_energy))
    if not all(math.isfinite(value) for value in (
            float(initial_load_factor), float(initial_external_work),
            float(initial_plastic_dissipation), base_energy)):
        raise ValueError("path continuation totals must be finite")
    if initial_plastic_dissipation < 0:
        raise ValueError("initial plastic dissipation cannot be negative")
    previous_u, previous_state = u0, s0
    previous_factor = float(initial_load_factor)
    external_work = float(initial_external_work)
    dissipation = float(initial_plastic_dissipation)
    raw = []
    for ordinal, point in enumerate(accepted_points, 1):
        u = point.displacement.to(model.nodes)
        factor = float(point.load_factor)
        external_work += .5*(previous_factor+factor)*float(torch.dot(load, u-previous_u))
        dissipation += plastic_dissipation_increment(model, previous_state, point.state)
        stored = shell_stored_energy(model, u, point.state).recoverable-base_energy
        internal = stored+dissipation
        residual = external_work-internal
        scale = max(abs(external_work), abs(internal), 1e-30)
        fraction = yielded_fraction(model, point.state, s0)
        transverse = float(torch.max(torch.abs((u-u0)[2::6])))
        shortening = float(torch.dot(load, u-u0))
        raw.append(dict(step=int(getattr(point, "step", ordinal)), load_factor=factor,
                        axial_shortening=shortening,
                        maximum_transverse_displacement=transverse,
                        external_work=external_work, recoverable_energy=stored,
                        plastic_dissipation=dissipation, internal_energy=internal,
                        energy_residual=residual, relative_energy_residual=abs(residual)/scale,
                        yielded_fraction=fraction))
        previous_u, previous_state, previous_factor = u, point.state, factor
    if not raw:
        return PanelPathEvidence((), None, None, False, "undetermined")
    peak_index = max(range(len(raw)), key=lambda i: raw[i]["load_factor"])
    peak = raw[peak_index]["load_factor"]
    confirmed = any(
        i > peak_index and raw[i]["axial_shortening"] > raw[peak_index]["axial_shortening"]
        and raw[i]["load_factor"] <= (1-post_peak_drop)*peak
        for i in range(len(raw))
    )
    mode = "prebuckling"
    final = raw[-1]
    if confirmed:
        if final["yielded_fraction"] < .05:
            mode = "elastic_buckling"
        elif final["maximum_transverse_displacement"] < .25*model.thickness:
            mode = "material_collapse"
        else:
            mode = "interactive_buckling_yielding"
    elif final["yielded_fraction"] > 0:
        mode = "yielding_without_confirmed_peak"
    def point_mode(index: int, values: dict[str, float]) -> str:
        if confirmed and index > peak_index:
            return mode
        if values["yielded_fraction"] > 0:
            return "yielding_without_confirmed_peak"
        return "prebuckling"
    evidence = tuple(PanelPointEvidence(**values, is_peak=i == peak_index,
                                        is_post_peak=confirmed and i > peak_index,
                                        failure_mode=point_mode(i, values))
                     for i, values in enumerate(raw))
    return PanelPathEvidence(evidence, raw[peak_index]["step"], peak, confirmed, mode)
