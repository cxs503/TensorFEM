"""Mortar segment transfer-accuracy study on a prescribed smooth gap field.

Two parallel squares ``[0,1]^2`` are discretised independently (``n_slave``
slave facets against ``n_master`` master facets, deliberately non-matching) and
a smooth analytic penetration field

    g(x, y) = g0 * (1.2 + 0.5 sin(pi x) sin(pi y))

is imposed through the slave nodal displacements.  Because the field is smooth
and strictly positive, every quadrature point is closed and the exact interface
force density is ``t = eps_n g``.  The analytic reference quantities are

    F_exact  = eps_n g0 (1.2 + 2/pi^2)
    ||t||^2  = (eps_n g0)^2 * (1.2^2 * 1 + 0.25 * 1/4 + 2 * 1.2 * 0.5 * 4/pi^2)

which makes the mortar segment rule (three-point Gauss on the clipped polygon
triangles) testable against a closed-form integral under mesh refinement.
"""

from __future__ import annotations
import math
import torch

from .mortar_segment_contact3d import sample_mortar_segment_contact


def _grid(n, z, dtype):
    x = torch.linspace(0.0, 1.0, n + 1, dtype=dtype)
    nodes = torch.tensor(
        [(float(x[i]), float(x[j]), z) for j in range(n + 1) for i in range(n + 1)], dtype=dtype
    )
    faces = []
    for j in range(n):
        for i in range(n):
            k = j * (n + 1) + i
            faces.append((k, k + 1, k + n + 2, k + n + 1))
    return nodes, torch.tensor(faces, dtype=torch.long)


def gap_field(x, y, amplitude):
    return amplitude * (1.2 + 0.5 * torch.sin(math.pi * x) * torch.sin(math.pi * y))


def mortar_transfer_accuracy(
    n_slave: int = 2,
    n_master: int = 3,
    *,
    amplitude: float = 1e-3,
    penalty: float = 1e6,
    dtype=torch.float64,
) -> dict:
    """Relative L2 / total-force error of the mortar segment integration."""
    if n_slave < 1 or n_master < 1:
        raise ValueError("mesh divisions must be positive")
    if amplitude <= 0 or penalty <= 0:
        raise ValueError("amplitude and penalty must be positive")
    slave, slave_faces = _grid(n_slave, 0.0, dtype)
    master, master_faces = _grid(n_master, 0.0, dtype)
    displacement = torch.stack(
        (
            torch.zeros_like(slave[:, 0]),
            torch.zeros_like(slave[:, 0]),
            -gap_field(slave[:, 0], slave[:, 1], amplitude),
        ),
        1,
    )
    samples = sample_mortar_segment_contact(
        slave + displacement,
        slave_faces,
        master,
        master_faces,
        normal_penalty=penalty,
        slave_reference=slave,
        master_reference=master,
    )
    if samples.areas.numel() == 0:
        raise ValueError("no mortar segments were produced")
    pts = samples.points
    exact = penalty * gap_field(pts[:, 0], pts[:, 1], amplitude)
    weights = samples.areas
    exact_total = penalty * amplitude * (1.2 + 2.0 / (math.pi**2))
    exact_energy_measure = (penalty * amplitude) ** 2 * (
        1.2**2 + 0.25 * 0.25 + 2.0 * 1.2 * 0.5 * 4.0 / (math.pi**2)
    )
    total = float(samples.total_normal_force)
    l2 = float(
        ((weights * (samples.normal_traction - exact) ** 2).sum() / (weights * exact**2).sum())
        ** 0.5
    )
    return {
        "n_slave": n_slave,
        "n_master": n_master,
        "quadrature_points": int(weights.numel()),
        "segments": int(samples.slave_faces.numel()),
        "integrated_area": float(weights.sum()),
        "total_force": total,
        "total_force_exact": exact_total,
        "total_force_error": abs(total - exact_total) / exact_total,
        "l2_error": l2,
        "mean_traction": total / float(weights.sum()),
        "exact_mean_traction": exact_total,
        "quadrature_measure": float((weights * exact**2).sum()),
        "exact_measure": exact_energy_measure,
        "maximum_penetration": float(samples.maximum_penetration),
    }


def transfer_convergence(levels=(2, 3, 4, 6, 8), **kwargs) -> list[dict]:
    """Run the transfer study over mesh levels (master mesh = slave + 1)."""
    out = []
    for n in levels:
        row = mortar_transfer_accuracy(n, n + 1, **kwargs)
        out.append(row)
    return out
