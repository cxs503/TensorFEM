"""Small-strain J2 plasticity for constant-strain TET4 solids.

The constitutive state passed to a load increment is treated as committed:
Newton iterations always integrate from that immutable state.  State is only
returned after equilibrium converges, which gives transaction-like rollback on
failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .plasticity import J2State, update_j2
from .solid3d import _b, _gradient


@dataclass(frozen=True)
class Tet4J2Model:
    nodes: torch.Tensor
    elements: torch.Tensor
    young: float
    poisson: float
    yield_stress: float
    hardening: float
    fixed_dofs: torch.Tensor

    def __post_init__(self):
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape [n,3]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 4:
            raise ValueError("only TET4 connectivity is supported")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1,0.5)")

    @property
    def n_dofs(self):
        return 3 * self.nodes.shape[0]


@dataclass(frozen=True)
class PlasticSolidState:
    points: tuple[J2State, ...]

    @classmethod
    def virgin(cls, model: Tet4J2Model):
        z = torch.zeros((3, 3), dtype=model.nodes.dtype, device=model.nodes.device)
        a = torch.zeros((), dtype=model.nodes.dtype, device=model.nodes.device)
        return cls(tuple(J2State(z.clone(), a.clone()) for _ in model.elements))


@dataclass(frozen=True)
class PlasticStepResult:
    load_factor: float
    displacement: torch.Tensor
    reaction: torch.Tensor
    stress: torch.Tensor
    equivalent_plastic_strain: torch.Tensor
    state: PlasticSolidState
    iterations: int
    residual_norm: float


def _kinematics(model):
    x = model.nodes[model.elements.long()]
    dn = x.new_tensor([[-1., -1., -1.], [1., 0., 0.],
                       [0., 1., 0.], [0., 0., 1.]])
    gradients, determinants = _gradient(x, dn, "TET4 element")
    b = _b(gradients)
    edofs = torch.stack(tuple(3 * model.elements + i for i in range(3)), 2).reshape(-1, 12).long()
    return b, determinants / 6.0, edofs


def _strain_tensor(v):
    return torch.stack((
        torch.stack((v[0], .5*v[3], .5*v[5])),
        torch.stack((.5*v[3], v[1], .5*v[4])),
        torch.stack((.5*v[5], .5*v[4], v[2])),
    ))


def _stress_vector(s):
    return torch.stack((s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[0, 2]))


def _response(strain, model, committed):
    """Stress, algorithmic tangent and trial state at one integration point."""
    def stress_only(e):
        s, _, _ = update_j2(_strain_tensor(e), model.young, model.poisson,
                            model.yield_stress, model.hardening, committed)
        return _stress_vector(s)
    stress_tensor, trial, _ = update_j2(
        _strain_tensor(strain), model.young, model.poisson,
        model.yield_stress, model.hardening, committed)
    # Exact derivative of the implemented backward-Euler return map.  This is
    # deliberately generated from the material routine to prevent drift
    # between stress integration and the global Newton tangent.
    tangent = torch.autograd.functional.jacobian(stress_only, strain, create_graph=False)
    return _stress_vector(stress_tensor), tangent, trial


def assemble_tet4_j2(model, displacement, committed):
    b, volumes, edofs = _kinematics(model)
    internal = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    tangent = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    stresses, trial_states = [], []
    for e, dofs in enumerate(edofs):
        strain = b[e] @ displacement[dofs]
        stress, material_tangent, trial = _response(strain, model, committed.points[e])
        fe = b[e].T @ stress * volumes[e]
        ke = b[e].T @ material_tangent @ b[e] * volumes[e]
        internal.index_add_(0, dofs, fe)
        tangent.index_put_((dofs[:, None], dofs[None, :]), ke, accumulate=True)
        stresses.append(stress); trial_states.append(trial)
    return internal, tangent, torch.stack(stresses), PlasticSolidState(tuple(trial_states))


def solve_increment(model, external_force, load_factor, committed, displacement,
                    *, tolerance=1e-10, max_iterations=25):
    """Solve one increment; raises without mutating committed state on failure."""
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    force = external_force.to(model.nodes) * load_factor
    u = displacement.clone()
    scale = max(float(torch.linalg.vector_norm(force[free])), 1.0)
    for iteration in range(1, max_iterations + 1):
        internal, tangent, stress, trial = assemble_tet4_j2(model, u, committed)
        residual = force - internal
        norm = float(torch.linalg.vector_norm(residual[free]))
        if norm <= tolerance * scale:
            return PlasticStepResult(float(load_factor), u, internal-force, stress,
                                     torch.stack(tuple(s.alpha for s in trial.points)),
                                     trial, iteration, norm)
        if free.numel() == 0:
            break
        du = torch.linalg.solve(tangent[free][:, free], residual[free])
        u = u.index_add(0, free, du)
    raise RuntimeError(f"plastic Newton failed to converge after {max_iterations} iterations")


def solve_load_steps(model, external_force, load_factors, *, tolerance=1e-10,
                     max_iterations=25, initial_state=None, initial_displacement=None):
    """Sequential load control with state commit after each converged step."""
    state = PlasticSolidState.virgin(model) if initial_state is None else initial_state
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    if initial_displacement is not None:
        u = initial_displacement.to(model.nodes).clone()
    results = []
    for factor in load_factors:
        result = solve_increment(model, external_force, float(factor), state, u,
                                 tolerance=tolerance, max_iterations=max_iterations)
        results.append(result)
        state, u = result.state, result.displacement
    return tuple(results)
