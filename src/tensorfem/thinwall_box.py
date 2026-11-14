"""Thin-walled closed box: tip-deflection oracle and conforming HEX20 mesher.

The case-pool anchor ``CASE-CANTILEVER-TIP`` certifies the tip deflection of a
*solid* rectangular cantilever against Euler--Bernoulli and Timoshenko theory on
a slenderness ladder.  A ship box girder (hatch coaming, double-wall stool,
closed stiffener idealisation) is the other canonical cantilever section, and
there the shear compliance is **not** ``kappa * A_gross``: the shear force is
carried by a closed thin-walled ring whose flow ``q(s)`` has a statically
indeterminate constant fixed by the zero-twist (Bredt) condition ``oint q/t ds
= 0``.  Reusing ``kappa = 5/6`` with the gross area therefore silently
mis-states the shear part of the deflection.

This module supplies, for a doubly symmetric closed rectangular box:

``shear_flow_closed_form``
    The shear flow ``q(s)/V`` from ``dq/ds = -V t z / I`` plus
    ``oint q/t ds = 0`` (Timoshenko & Gere; Megson, *Aircraft Structures* for
    the closed-section version), derived with sympy and integrated to the
    *effective shear area* ``1/A_s = oint (q/V)^2 / t ds``, so the shear
    deflection ``delta_s = V L / (G A_s)`` needs no assumed shear coefficient.

``BoxSection``
    Exact area / second moment of a rectangular ring (outer minus inner).

``box_hex20_mesh`` / ``solve_box_cantilever``
    A conforming HEX20 mesh of the *hollow* box (structured grid with the void
    cells removed, then ``upgrade_hex8_to_hex20``) and a fully clamped-root,
    face-loaded cantilever solve with force / moment equilibrium residuals,
    mirroring ``benchmarks/verified/fem_cantilever_shear``.

The oracle and the 3-D continuum share no kinematic assumption, so their
agreement is a genuine cross-model verification rather than a self-consistency
check.  Thin-wall theory neglects wall bending and the through-thickness shear
gradient, so the residual is expected to vanish as ``t / depth -> 0``; the
benchmark measures that trend instead of assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

__all__ = [
    "BoxSection",
    "shear_flow_closed_form",
    "shear_area",
    "shear_coefficient",
    "shear_flow_q0_per_unit_shear",
    "shear_energy_check",
    "delta_bending",
    "delta_shear",
    "delta_thin_wall",
    "delta_naive_timoshenko",
    "box_hex8_mesh",
    "box_hex20_mesh",
    "solve_box_cantilever",
]


# --------------------------------------------------------------------------- #
# Section properties
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoxSection:
    """Doubly symmetric rectangular ring.

    ``width`` is the outer dimension along ``y``, ``depth`` the outer dimension
    along the bending direction ``z`` (a transverse load along ``z`` bends the
    cantilever about ``y``).
    """

    width: float
    depth: float
    thickness: float

    def __post_init__(self) -> None:
        for name in ("width", "depth", "thickness"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not value > 0.0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if self.thickness >= 0.5 * min(self.width, self.depth):
            raise ValueError("thickness must be below half the smaller outer dimension")

    @property
    def inner_width(self) -> float:
        return self.width - 2.0 * self.thickness

    @property
    def inner_depth(self) -> float:
        return self.depth - 2.0 * self.thickness

    @property
    def area(self) -> float:
        return self.width * self.depth - self.inner_width * self.inner_depth

    @property
    def inertia_y(self) -> float:
        """Second moment about the bending axis: ``int z^2 dA``."""
        return (self.width * self.depth**3 - self.inner_width * self.inner_depth**3) / 12.0

    @property
    def centreline_width(self) -> float:
        return self.width - self.thickness

    @property
    def centreline_depth(self) -> float:
        return self.depth - self.thickness

    @property
    def perimeter_centreline(self) -> float:
        return 2.0 * (self.centreline_width + self.centreline_depth)

    def naive_shear_area(self, kappa: float = 5.0 / 6.0) -> float:
        """Shear area a beam check gets by reusing ``kappa`` and the gross area."""
        return kappa * self.area

    def web_only_shear_area(self) -> float:
        """Shear area of the two webs alone (the ``A_s ~ 2 t h`` shortcut)."""
        return 2.0 * self.thickness * self.centreline_depth


# --------------------------------------------------------------------------- #
# Shear flow of the closed ring (symbolic, machine derived)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def shear_flow_closed_form() -> dict:
    """Closed-form shear flow and effective shear area of the closed box.

    Path: start on the right web at mid-depth, go up the web, across the top
    flange, down the left web, back along the bottom flange and close the right
    web.  With ``a = (width - t)/2``, ``c = (depth - t)/2`` and ``I = 1`` the
    flow obeys ``dq/ds = -t z(s)``, so ``q = V Q`` with ``Q = Q_unit / I``.

    Returns the per-segment ``Q`` polynomials, the indeterminate constant
    ``q0/V`` and ``J = oint Q_unit^2 / t ds`` (so ``A_s = I^2 / J``).
    """
    a, c, t, s = sp.symbols("a c t s", positive=True)
    Q0 = sp.symbols("Q0")  # q0 / V, the statically indeterminate constant
    segments = ((c, s), (2 * a, c), (2 * c, c - s), (2 * a, -c), (c, -c + s))
    pieces: list[tuple[sp.Expr, sp.Expr]] = []
    Q = Q0
    for length, z_of_s in segments:
        Q = sp.expand(Q - t * sp.integrate(z_of_s, (s, 0, s)))
        pieces.append((sp.nsimplify(length), Q))
        Q = sp.expand(Q.subs(s, length))
    circulation = sp.expand(sum(sp.integrate(Q, (s, 0, length)) for length, Q in pieces))
    q0 = sp.solve(sp.Eq(circulation, 0), Q0)[0]
    pieces = [(length, sp.expand(Q.subs(Q0, q0))) for length, Q in pieces]
    J = sp.simplify(
        sum(sp.integrate(sp.expand(Q**2 / t), (s, 0, length)) for length, Q in pieces)
    )
    return {
        "a": a,
        "c": c,
        "t": t,
        "s": s,
        "J": J,
        "q0_over_V": sp.simplify(q0),
        "pieces": pieces,
    }


@lru_cache(maxsize=1)
def _closed_form_lambdas() -> dict:
    form = shear_flow_closed_form()
    return {
        "J": sp.lambdify((form["a"], form["c"], form["t"]), form["J"], "math"),
        "q0": sp.lambdify((form["a"], form["c"], form["t"]), form["q0_over_V"], "math"),
    }


def shear_area(section: BoxSection) -> float:
    """Effective shear area ``A_s = I^2 / oint (q/V)^2 t^-1 ds`` [m^2]."""
    a = 0.5 * section.centreline_width
    c = 0.5 * section.centreline_depth
    J = _closed_form_lambdas()["J"](a, c, section.thickness)  # computed with I = 1
    return section.inertia_y**2 / J


def shear_coefficient(section: BoxSection) -> float:
    """``kappa_eff = A_gross / A_s`` -- the coefficient a beam model would need."""
    return section.area / shear_area(section)


def shear_flow_q0_per_unit_shear(section: BoxSection) -> float:
    """The zero-twist (Bredt) constant ``q0 / V`` [1/m]; zero for an open path."""
    a = 0.5 * section.centreline_width
    c = 0.5 * section.centreline_depth
    return _closed_form_lambdas()["q0"](a, c, section.thickness) / section.inertia_y


# --------------------------------------------------------------------------- #
# Deflection oracles
# --------------------------------------------------------------------------- #
def delta_bending(section: BoxSection, length: float, load: float, young: float) -> float:
    return load * length**3 / (3.0 * young * section.inertia_y)


def delta_shear(
    section: BoxSection, length: float, load: float, young: float, nu: float
) -> float:
    shear_modulus = young / (2.0 * (1.0 + nu))
    return load * length / (shear_modulus * shear_area(section))


def delta_thin_wall(
    section: BoxSection, length: float, load: float, young: float, nu: float
) -> float:
    """Euler--Bernoulli bending plus the closed-ring shear-flow shear term."""
    return delta_bending(section, length, load, young) + delta_shear(
        section, length, load, young, nu
    )


def delta_naive_timoshenko(
    section: BoxSection,
    length: float,
    load: float,
    young: float,
    nu: float,
    kappa: float = 5.0 / 6.0,
) -> float:
    """Timoshenko with ``A_s = kappa * A_gross`` -- the shortcut a beam check makes."""
    shear_modulus = young / (2.0 * (1.0 + nu))
    return delta_bending(section, length, load, young) + load * length / (
        kappa * shear_modulus * section.area
    )


def shear_energy_check(
    section: BoxSection, length: float, young: float, nu: float, load: float = 1.0
) -> dict:
    """Independent check of :func:`shear_area` through the strain energy.

    ``U_s(V) = L oint q^2 / (2 G t) ds`` is evaluated by dense trapezoidal
    quadrature of the closed-form ``q(s)/V`` and differentiated numerically.
    That route never touches the analytic integral ``J``, so it validates the
    symbolic integration rather than repeating it.
    """
    form = shear_flow_closed_form()
    s = form["s"]
    subs = {
        form["a"]: 0.5 * section.centreline_width,
        form["c"]: 0.5 * section.centreline_depth,
        form["t"]: section.thickness,
    }
    lengths = [float(sp.nsimplify(length).subs(subs)) for length, _ in form["pieces"]]
    profiles = [sp.lambdify(s, Q.subs(subs), "numpy") for _, Q in form["pieces"]]
    shear_modulus = young / (2.0 * (1.0 + nu))
    inertia = section.inertia_y

    def strain_energy(load_value: float) -> float:
        total = 0.0
        for seg_len, profile in zip(lengths, profiles):
            n = 600
            h = seg_len / n
            acc = 0.0
            for i in range(n + 1):
                weight = 0.5 if i in (0, n) else 1.0
                q = load_value * float(profile(i * h)) / inertia
                acc += weight * q * q
            total += acc * h / section.thickness
        return total * length / (2.0 * shear_modulus)

    step = 1e-4 * max(load, 1.0)
    numeric = (strain_energy(load + step) - strain_energy(load - step)) / (2.0 * step)
    analytic = load * length / (shear_modulus * shear_area(section))
    return {
        "numeric_shear_deflection": numeric,
        "analytic_shear_deflection": analytic,
        "relative_difference": abs(numeric - analytic) / abs(analytic),
        "q0_over_V": shear_flow_q0_per_unit_shear(section),
    }


# --------------------------------------------------------------------------- #
# Conforming hollow-box mesh
# --------------------------------------------------------------------------- #
def _wall_grid(outer: float, thickness: float, wall_cells: int, inner_cells: int) -> list[float]:
    """Coordinate lines: ``wall_cells`` per wall plus ``inner_cells`` in the void."""
    if wall_cells < 1 or inner_cells < 1:
        raise ValueError("wall_cells and inner_cells must be >= 1")
    lines = [-0.5 * outer + thickness * i / wall_cells for i in range(wall_cells + 1)]
    lines += [
        -0.5 * outer + thickness + (outer - 2.0 * thickness) * i / inner_cells
        for i in range(1, inner_cells + 1)
    ]
    lines += [
        0.5 * outer - thickness + thickness * i / wall_cells for i in range(1, wall_cells + 1)
    ]
    return lines


def box_hex8_mesh(
    section: BoxSection,
    length: float,
    *,
    axial_cells: int,
    wall_cells: int = 2,
    inner_cells: int = 8,
) -> dict:
    """Conforming HEX8 mesh of the *hollow* box (the void cells are removed)."""
    if axial_cells < 1:
        raise ValueError("axial_cells must be >= 1")
    x = [length * i / axial_cells for i in range(axial_cells + 1)]
    y = _wall_grid(section.width, section.thickness, wall_cells, inner_cells)
    z = _wall_grid(section.depth, section.thickness, wall_cells, inner_cells)
    ny, nz = len(y) - 1, len(z) - 1

    ids: dict[tuple[int, int, int], int] = {}
    coords: list[tuple[float, float, float]] = []

    def node(ix: int, iy: int, iz: int) -> int:
        key = (ix, iy, iz)
        if key not in ids:
            ids[key] = len(coords)
            coords.append((x[ix], y[iy], z[iz]))
        return ids[key]

    cells: list[list[int]] = []
    removed = 0
    for ix in range(axial_cells):
        for iy in range(ny):
            for iz in range(nz):
                wall = iy < wall_cells or iy >= ny - wall_cells
                wall = wall or iz < wall_cells or iz >= nz - wall_cells
                if not wall:
                    removed += 1
                    continue
                cells.append(
                    [
                        node(ix, iy, iz),
                        node(ix + 1, iy, iz),
                        node(ix + 1, iy + 1, iz),
                        node(ix, iy + 1, iz),
                        node(ix, iy, iz + 1),
                        node(ix + 1, iy, iz + 1),
                        node(ix + 1, iy + 1, iz + 1),
                        node(ix, iy + 1, iz + 1),
                    ]
                )
    return {
        "nodes": coords,
        "cells": cells,
        "removed_cells": removed,
        "wall_cells": wall_cells,
        "inner_cells": inner_cells,
        "axial_cells": axial_cells,
    }


def box_hex20_mesh(
    section: BoxSection,
    length: float,
    *,
    axial_cells: int,
    wall_cells: int = 2,
    inner_cells: int = 8,
):
    """HEX20 mesh of the hollow box plus an *independent* section-measure tally.

    The cells are axis-aligned boxes, so their exact volume and second moment
    follow from the connectivity alone; comparing those totals with
    ``BoxSection.area`` / ``inertia_y`` is the geometry gate (it catches a
    dropped wall, a duplicated wall or a void leak, which a closed-form input
    echo would not).
    """
    import torch

    from tensorfem.hex20 import upgrade_hex8_to_hex20

    mesh = box_hex8_mesh(
        section,
        length,
        axial_cells=axial_cells,
        wall_cells=wall_cells,
        inner_cells=inner_cells,
    )
    nodes = torch.tensor(mesh["nodes"], dtype=torch.float64)
    hex8 = torch.tensor(mesh["cells"], dtype=torch.long)
    hex20_nodes, hex20 = upgrade_hex8_to_hex20(nodes, hex8)
    corners = nodes[hex8]
    # corner 0 and corner 6 are the two ends of the space diagonal of the cell
    diag = (corners[:, 6] - corners[:, 0]).abs()
    dx, dy, dz = diag[:, 0], diag[:, 1], diag[:, 2]
    volume = float((dx * dy * dz).sum())
    z_mid = corners[:, :, 2].mean(dim=1)
    inertia = float((dx * dy * dz * (z_mid**2 + dz**2 / 12.0)).sum()) / length
    return {
        "nodes": hex20_nodes,
        "elements": hex20,
        "hex8": hex8,
        "mesh_area": volume / length,
        "mesh_inertia_y": inertia,
        "removed_cells": mesh["removed_cells"],
        "wall_cells": wall_cells,
        "inner_cells": inner_cells,
        "axial_cells": axial_cells,
        "n_cells": int(hex20.shape[0]),
    }


def solve_box_cantilever(
    section: BoxSection,
    length: float,
    *,
    load: float,
    young: float,
    nu: float,
    axial_cells: int,
    wall_cells: int = 2,
    inner_cells: int = 8,
    sparse: bool = True,
) -> dict:
    """Clamped-root hollow-box cantilever with the tip load spread over the ring."""
    import time

    import torch

    from tensorfem.hex20 import Hex20Model, solve_hex20, solve_hex20_sparse

    mesh = box_hex20_mesh(
        section,
        length,
        axial_cells=axial_cells,
        wall_cells=wall_cells,
        inner_cells=inner_cells,
    )
    nodes, elements = mesh["nodes"], mesh["elements"]
    tol = 1.0e-9 * max(length, 1.0)
    root = torch.where(nodes[:, 0] < tol)[0]
    tip = torch.where(nodes[:, 0] > length - tol)[0]
    fixed = torch.tensor(
        [3 * int(i) + d for i in root for d in range(3)], dtype=torch.long
    )
    forces = torch.zeros(3 * len(nodes), dtype=torch.float64)
    forces[3 * tip + 2] = -load / len(tip)
    model = Hex20Model(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor([young], dtype=torch.float64),
        poisson_ratio=torch.tensor([nu], dtype=torch.float64),
        forces=forces,
        fixed_dofs=fixed,
    )
    start = time.perf_counter()
    result = solve_hex20_sparse(model) if sparse else solve_hex20(model)
    seconds = time.perf_counter() - start
    u = result.displacement
    tip_deflection = float(-u[3 * tip + 2].mean())
    reaction = result.reaction.reshape(-1, 3)[root]
    moment_y = float((nodes[root, 2] * reaction[:, 0]).sum())
    return {
        "tip_deflection_m": tip_deflection,
        "nodes": int(len(nodes)),
        "dofs": int(3 * len(nodes)),
        "cells": mesh["n_cells"],
        "mesh_area": mesh["mesh_area"],
        "mesh_inertia_y": mesh["mesh_inertia_y"],
        "removed_cells": mesh["removed_cells"],
        "force_residual": abs(float(reaction[:, 2].sum()) - load) / load,
        "moment_residual": abs(moment_y + load * length) / (load * length),
        "seconds": round(seconds, 3),
        "displacement": u,
    }