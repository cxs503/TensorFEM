"""Distortion qualification of the 20-node quadratic brick (HEX20).

`docs/element-rank-deficiency-audit.md` section 6 recorded one explicit open item:
``hex20`` had only been audited on the regular unit cube, while *distorted*
20-node bricks and *reduced-integration* variants were unexamined.  This module
closes that debt with four independent, machine-reproducible probes:

1. :func:`nullspace_under_distortion` - rank sufficiency (spurious zero-energy
   mechanisms) of five distortion families plus a randomised sweep.  The measured
   null space is compared with the analytic rigid-body span through principal
   angles (:mod:`tensorfem.element_rank_audit`), so the verdict is an exact
   integer and not a tuned tolerance.
2. :func:`integration_rule_audit` - the 2x2x2 "reduced" Gauss rule is *not* a
   legitimate option for a 20-node brick; it is quantified here as a documented
   negative control so nobody can silently switch the rule for speed.
3. :func:`distorted_patch_test` - field consistency: a strongly distorted,
   geometrically conforming HEX20 block must reproduce an arbitrary linear
   displacement field at the interior nodes to round-off.  Rank sufficiency does
   not imply this, so both are needed.
4. :func:`graded_cantilever` - the practical question.  The production cantilever
   benchmark is re-run on longitudinally graded meshes (element length ratios up
   to 12.7:1) so the tip-deflection error can be compared against the uniform
   mesh and the independent Timoshenko beam reference.

All probes keep the production integration rule (3x3x3 Gauss, exact for the
quadratic shape functions) and never mutate :mod:`tensorfem.hex20` state.
"""
from __future__ import annotations

import math

import torch

from .element_rank_audit import DEFAULT_RTOL, classify_null_space, rigid_body_basis_solid3d
from .hex20 import (
    Hex20Model,
    _b,
    _gradient,
    hex20_shape_gradients,
    hex20_stiffness,
    solve_hex20,
    structured_hex20_mesh,
)
from .solid3d import elasticity_matrix_3d

__all__ = [
    "DISTORTION_FAMILIES",
    "GAUSS_RULES",
    "distortion_strength",
    "distort_hex20",
    "hex20_stiffness_rule",
    "nullspace_under_distortion",
    "randomised_distortion_sweep",
    "integration_rule_audit",
    "distorted_patch_test",
    "graded_cantilever",
    "cantilever_distortion_sweep",
    "run_qualification",
    "YOUNG",
    "POISSON",
]

YOUNG = 200.0e9
POISSON = 0.3

#: Distortion families applied to the reference unit brick.
DISTORTION_FAMILIES = ("taper", "skew", "warp", "trapezoid", "twist")

_Q5 = math.sqrt(3.0 / 5.0)
_Q1 = 1.0 / math.sqrt(3.0)
#: ``order -> (points, weights)``; order 3 is the production rule (exact for quadratic).
GAUSS_RULES = {
    3: ((-_Q5, 0.0, _Q5), (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)),
    2: ((-_Q1, _Q1), (1.0, 1.0)),
}


def _reference_hex20(dtype=torch.float64) -> torch.Tensor:
    """Return the reference 1x1x1 HEX20 brick in ``hex20.py`` node ordering."""
    return structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1, dtype=dtype)[0]


def distort_hex20(nodes: torch.Tensor, kind: str, amplitude: float) -> torch.Tensor:
    """Return ``nodes`` distorted by ``kind`` with ``amplitude`` relative to the edge.

    ``amplitude`` is dimensionless: it is the peak displacement as a fraction of
    the reference edge length (1.0 for :func:`_reference_hex20`), so a value of
    ``0.2`` means the node moves by a fifth of an element edge.  Boundary
    flatness is deliberately *not* preserved - a distorted element is exactly the
    object under test.
    """
    if kind not in DISTORTION_FAMILIES:
        raise ValueError(f"unknown distortion family {kind!r}")
    x = torch.as_tensor(nodes).clone()
    a = float(amplitude)
    for i in range(len(x)):
        px, py, pz = (float(v) for v in x[i])
        if kind == "taper":
            # cross-sections shrink/expand linearly with z
            s = 1.0 + 2.0 * a * pz
            x[i, 0] = px * s
            x[i, 1] = py * s
        elif kind == "skew":
            x[i, 0] = px + a * pz
            x[i, 1] = py + 0.5 * a * pz
        elif kind == "warp":
            # the 20-node brick has no face-centre nodes, so a warp must be non-zero at
            # the edge mid-side nodes: this is a saddle that vanishes only on y = 1/2
            x[i, 2] = pz + a * (4.0 * px * (1.0 - px)) * (2.0 * py - 1.0)
        elif kind == "trapezoid":
            x[i, 0] = px + a * py * pz
            x[i, 1] = py + 0.6 * a * px * pz
        else:  # twist
            ang = a * pz
            x[i, 0] = px * math.cos(ang) - py * math.sin(ang)
            x[i, 1] = px * math.sin(ang) + py * math.cos(ang)
    return x


def distortion_strength(nodes: torch.Tensor, reference: torch.Tensor | None = None) -> float:
    """Return ``max|dx| / edge`` - the realised displacement relative to the edge."""
    ref = _reference_hex20(nodes.dtype) if reference is None else reference
    return float((nodes - ref).abs().max())


def hex20_stiffness_rule(x: torch.Tensor, d: torch.Tensor, order: int = 3):
    """HEX20 stiffness with an explicit Gauss order (``order=2`` is the negative control)."""
    if order not in GAUSS_RULES:
        raise ValueError(f"unsupported Gauss order {order!r}; choose from {sorted(GAUSS_RULES)}")
    points, weights = GAUSS_RULES[order]
    ke = x.new_zeros((len(x), 60, 60))
    for i, r in enumerate(points):
        for j, s in enumerate(points):
            for k, t in enumerate(points):
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                g, det = _gradient(x, dn)
                b = _b(g)
                ke += (b.transpose(1, 2) @ d @ b) * (det * weights[i] * weights[j] * weights[k])[
                    :, None, None
                ]
    _, dn = hex20_shape_gradients(0.0, 0.0, 0.0, like=x)
    return ke, _b(_gradient(x, dn)[0])


def _elasticity(young: float = YOUNG, poisson: float = POISSON, dtype=torch.float64):
    return elasticity_matrix_3d(
        torch.tensor([young], dtype=dtype), torch.tensor([poisson], dtype=dtype)
    )


def _classify(k: torch.Tensor, nodes: torch.Tensor, rtol: float = DEFAULT_RTOL) -> dict:
    return classify_null_space(0.5 * (k + k.transpose(-1, -2)), rigid_body_basis_solid3d(nodes), rtol)


def nullspace_under_distortion(
    amplitudes: tuple[float, ...] = (0.05, 0.15, 0.3),
    families: tuple[str, ...] = DISTORTION_FAMILIES,
    young: float = YOUNG,
    poisson: float = POISSON,
) -> list[dict]:
    """Rank sufficiency of distorted HEX20 elements (spurious mode count per family)."""
    d = _elasticity(young, poisson)
    base = _reference_hex20()
    records = []
    for kind in families:
        for amp in amplitudes:
            nodes = distort_hex20(base, kind, amp)
            k, _ = hex20_stiffness(nodes.unsqueeze(0), d)
            info = _classify(k[0], nodes)
            records.append(
                {
                    "family": kind,
                    "amplitude": amp,
                    "realised": distortion_strength(nodes, base),
                    "null_dim": info["null_dim"],
                    "rigid_dim": info["explained"],
                    "spurious": info["spurious"],
                    "first_positive_ratio": info["first_positive_ratio"],
                }
            )
    return records


def randomised_distortion_sweep(
    trials: int = 200,
    amplitude: float = 0.25,
    seed: int = 7,
    young: float = YOUNG,
    poisson: float = POISSON,
) -> dict:
    """Random node jitter sweep; returns the spurious histogram and rejected-geometry count.

    Random jitter can fold an element; :func:`tensorfem.hex20.hex20_stiffness` raises
    on a non-positive Jacobian, and those trials are counted (not silently dropped)
    in ``rejected_degenerate``.
    """
    d = _elasticity(young, poisson)
    base = _reference_hex20()
    generator = torch.Generator().manual_seed(seed)
    histogram: dict[int, int] = {}
    rejected = 0.0
    worst_strength = 0.0
    for _ in range(trials):
        nodes = base + (torch.rand(base.shape, generator=generator, dtype=base.dtype) - 0.5) * 2 * amplitude
        try:
            k, _ = hex20_stiffness(nodes.unsqueeze(0), d)
        except ValueError:
            rejected += 1
            continue
        info = _classify(k[0], nodes)
        histogram[info["spurious"]] = histogram.get(info["spurious"], 0) + 1
        worst_strength = max(worst_strength, distortion_strength(nodes, base))
    return {
        "trials": trials,
        "amplitude": amplitude,
        "seed": seed,
        "accepted": trials - int(rejected),
        "rejected_degenerate": int(rejected),
        "spurious_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "max_realised": worst_strength,
    }


def integration_rule_audit(young: float = YOUNG, poisson: float = POISSON) -> list[dict]:
    """Compare the production 3x3x3 rule with the 2x2x2 "reduced" rule on the unit brick.

    The reduced rule is expected to be rank deficient: it is reported as a negative
    control, not as an option.
    """
    d = _elasticity(young, poisson)
    base = _reference_hex20()
    records = []
    for order in (3, 2):
        k, _ = hex20_stiffness_rule(base.unsqueeze(0), d, order=order)
        info = _classify(k[0], base)
        records.append(
            {
                "rule": f"{order}x{order}x{order}",
                "order": order,
                "null_dim": info["null_dim"],
                "rigid_dim": info["explained"],
                "spurious": info["spurious"],
                "first_positive_ratio": info["first_positive_ratio"],
            }
        )
    return records


def _linear_field(nodes: torch.Tensor, eps0: torch.Tensor) -> torch.Tensor:
    g = torch.tensor(
        [
            [eps0[0], eps0[3] / 2, eps0[5] / 2],
            [eps0[3] / 2, eps0[1], eps0[4] / 2],
            [eps0[5] / 2, eps0[4] / 2, eps0[2]],
        ],
        dtype=nodes.dtype,
    )
    return nodes @ g.transpose(0, 1)


def distorted_patch_test(
    level: int = 0,
    n: int = 2,
    amplitude: float = 0.1,
    seed: int = 3,
    young: float = YOUNG,
    poisson: float = POISSON,
) -> dict:
    """Conforming distorted HEX20 block reproducing an arbitrary linear field.

    ``level`` selects the number of refinement steps (``n`` is doubled each step).
    Only nodes strictly inside the block are perturbed, so the block remains a
    rectangular domain with a genuinely distorted interior mesh - the classic
    patch-test geometry.
    """
    cells_per_side = n * (2**level)
    base_nodes, cells = structured_hex20_mesh(2.0, 1.0, 1.0, cells_per_side, cells_per_side, cells_per_side)
    generator = torch.Generator().manual_seed(seed)
    edge = 2.0 / cells_per_side
    interior = (
        (base_nodes[:, 0] > 1e-9) & (base_nodes[:, 0] < 2.0 - 1e-9)
        & (base_nodes[:, 1] > 1e-9) & (base_nodes[:, 1] < 1.0 - 1e-9)
        & (base_nodes[:, 2] > 1e-9) & (base_nodes[:, 2] < 1.0 - 1e-9)
    )
    noise = (torch.rand(base_nodes.shape, generator=generator, dtype=base_nodes.dtype) - 0.5) * 2 * amplitude * edge
    nodes = torch.where(interior[:, None], base_nodes + noise, base_nodes)
    used = float(amplitude)
    # a jitter that folds an element is a geometry question, not a physics question:
    # halve it and retry so the probe stays valid instead of reporting "invalid mesh".
    from .hex20 import hex20_stiffness as _ks
    d = _elasticity(young, poisson)
    while used > 1e-6:
        try:
            _ks(nodes[cells], d)
            break
        except ValueError:
            used *= 0.5
            noise = (torch.rand(base_nodes.shape, generator=generator, dtype=base_nodes.dtype) - 0.5) * 2 * used * edge
            nodes = torch.where(interior[:, None], base_nodes + noise, base_nodes)
    eps0 = torch.tensor([1e-3, -2e-4, 5e-4, 3e-4, -1e-4, 2e-4], dtype=nodes.dtype)
    exact = _linear_field(nodes, eps0)
    boundary = ~(
        (nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2.0 - 1e-9)
        & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9)
        & (nodes[:, 2] > 1e-9) & (nodes[:, 2] < 1.0 - 1e-9)
    )
    boundary_ids = torch.where(boundary)[0]
    fixed = torch.stack((3 * boundary_ids, 3 * boundary_ids + 1, 3 * boundary_ids + 2), 1).reshape(-1)
    m = Hex20Model(
        nodes,
        cells,
        nodes.new_tensor(young),
        nodes.new_tensor(poisson),
        nodes.new_zeros(3 * len(nodes)),
        fixed,
        exact.reshape(-1)[fixed],
    )
    result = solve_hex20(m)
    free = torch.ones(3 * len(nodes), dtype=torch.bool)
    free[fixed] = False
    deviation = (result.displacement - exact.reshape(-1)).abs()
    scale = float(exact.abs().max())
    return {
        "level": level,
        "cells_per_side": cells_per_side,
        "nodes": int(len(nodes)),
        "elements": int(len(cells)),
        "amplitude_edges": used,
        "interior_nodes": int((~boundary).sum()),
        "max_interior_error": float(deviation[free].max()),
        "relative_interior_error": float(deviation[free].max()) / scale,
        "field_scale": scale,
    }


def _graded_beam(nx: int, exponent: float, dtype=torch.float64):
    """Longitudinally graded but still straight, prismatic 10x1x1 cantilever mesh.

    ``max(x_i) = L`` and ``min(x_i) = 0`` are unchanged, so the beam geometry,
    the load pattern and the beam-theory reference are identical to the uniform
    mesh; only the element length ratio changes.
    """
    nodes, cells = structured_hex20_mesh(10.0, 1.0, 1.0, nx, 1, 1)
    xi = nodes[:, 0] / 10.0
    out = nodes.clone()
    out[:, 0] = 10.0 * xi**exponent
    return out, cells


def graded_cantilever(nx: int = 8, exponent: float = 1.0) -> dict:
    """Tip deflection of the production HEX20 cantilever on a longitudinally graded mesh."""
    young, poisson, load = 210e9, 0.3, -1e6
    length = 10.0
    nodes, cells = _graded_beam(nx, exponent)
    stations = torch.unique(nodes[:, 0])
    spacing = stations.diff().abs()
    forces = nodes.new_zeros(3 * len(nodes))
    right = torch.where(nodes[:, 0] == length)[0]
    corner = ((nodes[right, 1] == 0) | (nodes[right, 1] == 1)) & ((nodes[right, 2] == 0) | (nodes[right, 2] == 1))
    forces[3 * right[corner] + 2] = load * (-1 / 12)
    forces[3 * right[~corner] + 2] = load * (1 / 3)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3 * left, 3 * left + 1, 3 * left + 2), 1).reshape(-1)
    result = solve_hex20(
        Hex20Model(
            nodes,
            cells,
            nodes.new_tensor(young),
            nodes.new_tensor(poisson),
            forces,
            fixed,
        )
    )
    computed = float(result.displacement[3 * right + 2].mean())
    inertia = 1.0 * 1.0**3 / 12
    shear = young / (2 * (1 + poisson))
    reference = load * length**3 / (3 * young * inertia) + load * length / ((5 / 6) * shear * 1.0 * 1.0)
    return {
        "nx": nx,
        "exponent": exponent,
        "length_ratio": float(spacing.max() / spacing.min()),
        "computed": computed,
        "reference": reference,
        "error_percent": abs(computed - reference) / abs(reference) * 100.0,
    }


def cantilever_distortion_sweep(nx_values: tuple[int, ...] = (4, 8, 16), exponents: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6)) -> list[dict]:
    """Tip-deflection error of the production cantilever over graded meshes."""
    return [graded_cantilever(nx, p) for nx in nx_values for p in exponents]


def run_qualification(
    amplitudes: tuple[float, ...] = (0.05, 0.15, 0.3),
    trials: int = 200,
) -> dict:
    """Run every probe and return a JSON-serialisable evidence bundle."""
    return {
        "material": {"young": YOUNG, "poisson": POISSON},
        "distortion_families": list(DISTORTION_FAMILIES),
        "amplitudes": list(amplitudes),
        "rank_under_distortion": nullspace_under_distortion(amplitudes),
        "randomised_sweep": randomised_distortion_sweep(trials=trials),
        "integration_rules": integration_rule_audit(),
        "patch_test": [distorted_patch_test(level) for level in (0, 1)],
        "cantilever": cantilever_distortion_sweep(),
    }