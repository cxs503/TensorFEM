"""Table-driven lattice-Boltzmann lattices with an exact, falsifiable self-check.

Motivation
----------
Every lattice-Boltzmann implementation in this repository so far hard-codes its
discrete velocity set: ``lbm_channel2d`` writes ``CX``/``CY``/``W9``/``OPP`` as
module constants, ``lbm_shan_chen_mcmp`` and ``lbm_wetting`` repeat the same
numbers inside the collision kernels.  Consequences:

* adding a lattice (D3Q19, D3Q27, D1Q3) means touching the solver, not the data;
* a typo in a weight or in the opposite map is *silent*: the code still runs and
  produces plausible-looking fields with a wrong viscosity or a wrong symmetry;
* nothing in the tree proves that a velocity set is a valid LB velocity set.

This module follows the "table-driven lattice + self-check" pattern used by
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow`` (the velocity set
is *data*; a checker validates the data), but goes one step further: all lattice
invariants are evaluated in exact rational arithmetic (``fractions.Fraction``),
so the pass/fail decision carries no floating-point tolerance at all, and the
check is exercised against *corrupted* tables (mutation testing) so that a
"self-check that always passes" cannot masquerade as evidence.

Contents
--------
``LatticeTable``      immutable lattice: velocities, weights, cs2, opposite map.
``LATTICE_SPECS``     the lattices as pure data (D1Q3, D2Q9, D3Q15, D3Q19, D3Q27).
``selfcheck``         exact invariant audit -> ``SelfCheckReport``.
``isotropy_rank``     highest moment rank whose tensor is exactly isotropic.
``equilibrium`` / ``macroscopic`` / ``stream`` / ``collide_bgk``
                      lattice-agnostic kernels: the same NumPy code runs every
                      table, there is no per-lattice branch anywhere.
``run_channel_poiseuille`` / ``run_taylor_green``
                      physical cross-checks: the table's own ``cs2`` and weights
                      predict a viscosity and a no-slip parabola; the solver has
                      to reproduce them against closed-form solutions.

References
----------
[1] Qian, Y.H., D'Humieres, D., Lallemand, P. (1992). Lattice BGK models for
    Navier-Stokes equation. Europhys. Lett. 17, 479 (D2Q9/D3Q15/D3Q19 weights).
[2] Kruger, T. et al. (2017). The Lattice Boltzmann Method: Principles and
    Practice. Springer, ch. 3 (moment isotropy ladder of the DdQq families).
[3] Guo, Z., Zheng, C., Shi, B. (2002). Discrete lattice effects on the forcing
    term in the lattice Boltzmann method. Phys. Fluids 14, 2007.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Sequence

import numpy as np

__all__ = [
    "LatticeTable",
    "LATTICE_SPECS",
    "get_table",
    "table_names",
    "InvariantCheck",
    "SelfCheckReport",
    "selfcheck",
    "isotropy_rank",
    "equilibrium",
    "equilibrium_exact",
    "macroscopic",
    "stream",
    "collide_bgk",
    "run_channel_poiseuille",
    "run_taylor_green",
    "mutate_weight",
    "mutate_speed_class",
    "interior_curvature",
    "fit_wall_planes",
]


# ---------------------------------------------------------------------------
# lattice tables as pure data
# ---------------------------------------------------------------------------
# A spec is: dim + [(speed_class_directions, weight), ...].  Nothing in the code
# below knows what "D3Q19" means; the name is only a dictionary key.
LATTICE_SPECS: dict[str, dict] = {
    "D1Q3": {
        "dim": 1,
        "velocity_sets": [
            (((0,),), Fraction(2, 3)),
            (((1,), (-1,)), Fraction(1, 6)),
        ],
    },
    "D2Q9": {
        "dim": 2,
        "velocity_sets": [
            (((0, 0),), Fraction(4, 9)),
            (((1, 0), (0, 1), (-1, 0), (0, -1)), Fraction(1, 9)),
            (((1, 1), (-1, -1), (1, -1), (-1, 1)), Fraction(1, 36)),
        ],
    },
    "D3Q15": {
        "dim": 3,
        "velocity_sets": [
            (((0, 0, 0),), Fraction(2, 9)),
            (((1, 0, 0), (0, 1, 0), (0, 0, 1),
              (-1, 0, 0), (0, -1, 0), (0, 0, -1)), Fraction(1, 9)),
            (tuple(product((1, -1), repeat=3)), Fraction(1, 72)),
        ],
    },
    "D3Q19": {
        "dim": 3,
        "velocity_sets": [
            (((0, 0, 0),), Fraction(1, 3)),
            (((1, 0, 0), (0, 1, 0), (0, 0, 1),
              (-1, 0, 0), (0, -1, 0), (0, 0, -1)), Fraction(1, 18)),
            (((1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
              (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
              (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)), Fraction(1, 36)),
        ],
    },
    "D3Q27": {
        "dim": 3,
        "velocity_sets": [
            (((0, 0, 0),), Fraction(8, 27)),
            (((1, 0, 0), (0, 1, 0), (0, 0, 1),
              (-1, 0, 0), (0, -1, 0), (0, 0, -1)), Fraction(2, 27)),
            (((1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
              (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
              (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)), Fraction(1, 54)),
            (tuple(product((1, -1), repeat=3)), Fraction(1, 216)),
        ],
    },
}


@dataclass(frozen=True)
class LatticeTable:
    """A discrete velocity set.  Pure data; no method branches on ``name``."""

    name: str
    dim: int
    velocities: tuple[tuple[int, ...], ...]
    weights: tuple[Fraction, ...]
    cs2: Fraction = Fraction(1, 3)

    @property
    def q(self) -> int:
        return len(self.velocities)

    def __post_init__(self) -> None:
        if len(self.velocities) != len(self.weights):
            raise ValueError(
                f"{self.name}: {len(self.velocities)} velocities but "
                f"{len(self.weights)} weights"
            )
        for c in self.velocities:
            if len(c) != self.dim:
                raise ValueError(f"{self.name}: velocity {c} has wrong dimension")
        if self.cs2 <= 0:
            raise ValueError(f"{self.name}: cs2 must be positive")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"{self.name}: negative lattice weight")
        if sum(self.weights) <= 0:
            raise ValueError(f"{self.name}: weights must have positive total")

    @property
    def opposite(self) -> tuple[int, ...]:
        """Index map to the velocity ``-c_i`` (raises if the set is not closed)."""
        index = {c: i for i, c in enumerate(self.velocities)}
        out = []
        for c in self.velocities:
            neg = tuple(-ci for ci in c)
            if neg not in index:
                raise ValueError(f"{self.name}: velocity set not closed under -c ({c})")
            out.append(index[neg])
        return tuple(out)

    def moment(self, axes: Sequence[int]) -> Fraction:
        """Exact ``sum_i w_i * prod_a c_i,axes[a]``."""
        total = Fraction(0)
        for c, w in zip(self.velocities, self.weights):
            prod_c = Fraction(1)
            for a in axes:
                prod_c *= c[a]
            total += w * prod_c
        return total

    def weight_by_speed(self) -> dict[int, list[Fraction]]:
        out: dict[int, list[Fraction]] = {}
        for c, w in zip(self.velocities, self.weights):
            out.setdefault(sum(ci * ci for ci in c), []).append(w)
        return out

    def to_float(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(velocities (q,dim), weights (q,), opposite map (q,)) as NumPy arrays."""
        vel = np.asarray(self.velocities, dtype=np.float64).reshape(self.q, self.dim)
        w = np.asarray([float(x) for x in self.weights], dtype=np.float64)
        return vel, w, np.asarray(self.opposite, dtype=np.int64)


def _table_from_spec(name: str, spec: dict) -> LatticeTable:
    velocities: list[tuple[int, ...]] = []
    weights: list[Fraction] = []
    seen: set[tuple[int, ...]] = set()
    for directions, weight in spec["velocity_sets"]:
        for c in directions:
            c = tuple(int(x) for x in c)
            if c in seen:
                raise ValueError(f"{name}: duplicate velocity {c} in spec")
            seen.add(c)
            velocities.append(c)
            weights.append(Fraction(weight))
    return LatticeTable(
        name=name,
        dim=int(spec["dim"]),
        velocities=tuple(velocities),
        weights=tuple(weights),
        cs2=Fraction(spec.get("cs2", Fraction(1, 3))),
    )


def table_names() -> tuple[str, ...]:
    return tuple(LATTICE_SPECS)


def get_table(name: str) -> LatticeTable:
    try:
        spec = LATTICE_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown lattice {name!r}; known: {sorted(LATTICE_SPECS)}"
        ) from None
    return _table_from_spec(name, spec)


# ---------------------------------------------------------------------------
# self-check: exact invariants, zero tolerance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InvariantCheck:
    key: str
    description: str
    measured: tuple[Fraction, ...]
    expected: tuple[Fraction, ...]
    max_abs_error: Fraction
    passed: bool

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "description": self.description,
            "max_abs_error": str(self.max_abs_error),
            "max_abs_error_float": float(self.max_abs_error),
            "passed": self.passed,
            "measured": [str(m) for m in self.measured],
            "expected": [str(e) for e in self.expected],
        }


@dataclass
class SelfCheckReport:
    table: LatticeTable
    checks: list[InvariantCheck] = field(default_factory=list)
    isotropy_rank: int = 0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        return [c.key for c in self.checks if not c.passed]

    def as_dict(self) -> dict:
        return {
            "lattice": self.table.name,
            "dim": self.table.dim,
            "q": self.table.q,
            "cs2": str(self.table.cs2),
            "isotropy_rank": self.isotropy_rank,
            "passed": self.passed,
            "failures": self.failures,
            "checks": [c.as_dict() for c in self.checks],
        }


def _delta(a: int, b: int) -> int:
    return 1 if a == b else 0


def _pairings(axes: tuple[int, ...]) -> list[list[tuple[int, int]]]:
    """All perfect matchings of the axis indices."""
    if not axes:
        return [[]]
    first, rest = axes[0], axes[1:]
    out = []
    for k, partner in enumerate(rest):
        remaining = rest[:k] + rest[k + 1:]
        for tail in _pairings(remaining):
            out.append([(first, partner)] + tail)
    return out


def _isotropic_even_moment(axes: tuple[int, ...], cs2: Fraction, half: int) -> Fraction:
    """Isotropic tensor value ``cs2**half * sum_matchings prod delta``."""
    total = Fraction(0)
    for matching in _pairings(axes):
        term = cs2 ** half
        for a, b in matching:
            term *= _delta(a, b)
        total += term
    return total


def isotropy_rank(table: LatticeTable, max_rank: int = 6) -> int:
    """Highest rank ``r`` with an exactly isotropic moment tensor.

    Rank ``2n`` must satisfy ``sum_i w_i c c ... c = cs2**n * sum_matchings
    prod delta`` (Einstein summation); odd ranks must vanish.  Odd ranks vanish
    identically for any set closed under ``-c``; the even ladder is what
    separates the DdQq families: D2Q9/D3Q19/D3Q27 are isotropic through rank 4,
    D3Q15 is not (Kruger et al. 2017, ch. 3).
    """
    rank = 0
    for r in range(1, max_rank + 1):
        ok = True
        for axes in product(range(table.dim), repeat=r):
            if r % 2 == 1:
                if table.moment(axes) != 0:
                    ok = False
                    break
            else:
                expected = _isotropic_even_moment(axes, table.cs2, r // 2)
                if table.moment(axes) != expected:
                    ok = False
                    break
        if not ok:
            break
        rank = r
    return rank


def selfcheck(table: LatticeTable, max_rank: int = 6) -> SelfCheckReport:
    """Exact invariant audit of a lattice table.

    Checked invariants
    ------------------
    ``partition_of_unity``   sum_i w_i = 1
    ``first_moment``         sum_i w_i c_ia = 0            (Galilean invariance)
    ``second_moment``        sum_i w_i c_ia c_ib = cs2 d_ab (pressure + viscosity)
    ``fourth_moment``        sum_i w_i c c c c = cs2^2 (dd + dd + dd)
    ``odd_moments``          ranks 3 and 5 of w c...c vanish
    ``opposite_closed``      -c_i is a table entry           (bounce-back)
    ``opposite_symmetry``    w_i = w_opp(i)                  (BGK + bounce-back)
    ``distinct_velocities``  no repeated velocity
    """
    report = SelfCheckReport(table=table)
    dim = table.dim

    mass = table.moment(())
    report.checks.append(InvariantCheck(
        key="partition_of_unity",
        description="sum_i w_i = 1",
        measured=(mass,),
        expected=(Fraction(1),),
        max_abs_error=abs(mass - 1),
        passed=mass == 1,
    ))

    first = [table.moment((a,)) for a in range(dim)]
    report.checks.append(InvariantCheck(
        key="first_moment",
        description="sum_i w_i c_ia = 0",
        measured=tuple(first),
        expected=tuple(Fraction(0) for _ in range(dim)),
        max_abs_error=max(abs(x) for x in first),
        passed=all(x == 0 for x in first),
    ))

    second = [table.moment((a, b)) for a in range(dim) for b in range(dim)]
    expected_second = [table.cs2 * _delta(a, b)
                       for a in range(dim) for b in range(dim)]
    err2 = max(abs(m - e) for m, e in zip(second, expected_second))
    report.checks.append(InvariantCheck(
        key="second_moment",
        description="sum_i w_i c_ia c_ib = cs2 delta_ab",
        measured=tuple(second),
        expected=tuple(Fraction(x) for x in expected_second),
        max_abs_error=err2,
        passed=err2 == 0,
    ))

    fourth_axes = list(product(range(dim), repeat=4))
    fourth = [table.moment(axes) for axes in fourth_axes]
    expected_fourth = [_isotropic_even_moment(axes, table.cs2, 2) for axes in fourth_axes]
    err4 = max(abs(m - e) for m, e in zip(fourth, expected_fourth))
    report.checks.append(InvariantCheck(
        key="fourth_moment",
        description="sum_i w_i c c c c = cs2^2 (dd + dd + dd)",
        measured=tuple(fourth),
        expected=tuple(expected_fourth),
        max_abs_error=err4,
        passed=err4 == 0,
    ))

    odd_err = Fraction(0)
    for r in (3, 5):
        for axes in product(range(dim), repeat=r):
            odd_err = max(odd_err, abs(table.moment(axes)))
    report.checks.append(InvariantCheck(
        key="odd_moments",
        description="odd moment ranks 3 and 5 vanish",
        measured=(odd_err,),
        expected=(Fraction(0),),
        max_abs_error=odd_err,
        passed=odd_err == 0,
    ))

    index = {c: i for i, c in enumerate(table.velocities)}
    missing = [c for c in table.velocities
               if tuple(-ci for ci in c) not in index]
    report.checks.append(InvariantCheck(
        key="opposite_closed",
        description="every velocity has its -c in the table",
        measured=(Fraction(len(missing)),),
        expected=(Fraction(0),),
        max_abs_error=Fraction(len(missing)),
        passed=not missing,
    ))

    sym_err = Fraction(0)
    if not missing:
        opposite = table.opposite
        for i in range(table.q):
            sym_err = max(sym_err, abs(table.weights[i] - table.weights[opposite[i]]))
    else:
        sym_err = Fraction(1)
    report.checks.append(InvariantCheck(
        key="opposite_symmetry",
        description="w_i = w_opp(i)",
        measured=(sym_err,),
        expected=(Fraction(0),),
        max_abs_error=sym_err,
        passed=sym_err == 0,
    ))

    duplicates = table.q - len(set(table.velocities))
    report.checks.append(InvariantCheck(
        key="distinct_velocities",
        description="no repeated velocity vector",
        measured=(Fraction(duplicates),),
        expected=(Fraction(0),),
        max_abs_error=Fraction(duplicates),
        passed=duplicates == 0,
    ))

    if not missing and duplicates == 0:
        report.isotropy_rank = isotropy_rank(table, max_rank=max_rank)
    return report


def mutate_weight(table: LatticeTable, index: int, delta: Fraction) -> LatticeTable:
    """Return a copy with ``weights[index] += delta`` (falsification harness)."""
    weights = list(table.weights)
    weights[index] = weights[index] + Fraction(delta)
    return LatticeTable(
        name=f"{table.name}~mut{index}",
        dim=table.dim,
        velocities=table.velocities,
        weights=tuple(weights),
        cs2=table.cs2,
    )


# ---------------------------------------------------------------------------
# lattice-agnostic kernels (one code path for every table)
# ---------------------------------------------------------------------------
def _as_float_table(table: LatticeTable):
    vel, w, opp = table.to_float()
    return vel, w.reshape((table.q,) + (1,) * table.dim), w, opp


def equilibrium_exact(
    table: LatticeTable,
    rho: Fraction,
    u: Sequence[Fraction],
) -> list[Fraction]:
    """Second-order equilibrium in exact rational arithmetic (verification)."""
    u = tuple(Fraction(x) for x in u)
    usq = sum(ui * ui for ui in u)
    cs2 = table.cs2
    out = []
    for c, w in zip(table.velocities, table.weights):
        cu = sum(Fraction(ci) * ui for ci, ui in zip(c, u))
        out.append(w * rho * (1 + cu / cs2 + cu * cu / (2 * cs2 * cs2) - usq / (2 * cs2)))
    return out


def equilibrium(table: LatticeTable, rho, u: np.ndarray) -> np.ndarray:
    """``f_eq`` for any table.  ``rho`` scalar or grid array, ``u`` (dim, *grid)."""
    vel, w, _, _ = _as_float_table(table)
    cs2 = float(table.cs2)
    shape = (table.q,) + u.shape[1:]
    view = vel.reshape((table.q, table.dim) + (1,) * table.dim)
    cu = np.zeros(shape, dtype=np.float64)
    for a in range(table.dim):
        cu += view[:, a] * u[a][None]
    usq = np.sum(u * u, axis=0)
    return w * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs2 * cs2) - usq / (2.0 * cs2))


def macroscopic(table: LatticeTable, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rho, u)`` with ``u`` of shape ``(dim, *grid)``."""
    vel, _, _, _ = _as_float_table(table)
    rho = f.sum(axis=0)
    view = vel.reshape((table.q, table.dim) + (1,) * table.dim)
    mom = np.zeros((table.dim,) + f.shape[1:], dtype=np.float64)
    for a in range(table.dim):
        mom[a] = (f * view[:, a]).sum(axis=0)
    safe = np.where(rho > 1e-30, rho, 1.0)
    return rho, mom / safe[None]


def stream(table: LatticeTable, f: np.ndarray) -> np.ndarray:
    """Periodic streaming by one cell for every table (no per-lattice branch)."""
    out = np.empty_like(f)
    axes = tuple(range(f.ndim - 1))
    for i, c in enumerate(table.velocities):
        out[i] = np.roll(f[i], shift=tuple(-ci for ci in c), axis=axes)
    return out


def collide_bgk(
    table: LatticeTable,
    f: np.ndarray,
    tau: float,
    force=None,
    force_scheme: str = "guo",
) -> np.ndarray:
    """BGK collision with an optional constant body force (per unit mass).

    ``force_scheme="simple"`` adds ``w_i (c_i.F) / cs2`` to the post-collision
    populations (first-order in the force, the classic textbook shortcut).
    ``force_scheme="guo"`` uses the Guo (2002) splitting with the half-force
    velocity shift, which restores second-order accuracy of the forced steady
    state; see ``run_channel_poiseuille`` for the measured order.
    """
    rho, u = macroscopic(table, f)
    if force is None:
        feq = equilibrium(table, rho, u)
        return f + (feq - f) / tau
    if force_scheme not in ("simple", "guo"):
        raise ValueError(f"unknown force_scheme {force_scheme!r}")
    dim = table.dim
    F = np.asarray([float(x) for x in force], dtype=np.float64).reshape(
        (dim,) + (1,) * dim)
    vel, w, _, _ = _as_float_table(table)
    view = vel.reshape((table.q, dim) + (1,) * dim)
    cs2 = float(table.cs2)
    if force_scheme == "simple":
        feq = equilibrium(table, rho, u)
        cu_F = np.zeros_like(f)
        for a in range(dim):
            cu_F += view[:, a] * F[a]
        return f + (feq - f) / tau + w * cu_F / cs2 * rho[None]

    # Guo: equilibrium evaluated with the half-force shifted velocity
    u_shift = u + F / (2.0 * rho[None])
    feq = equilibrium(table, rho, u_shift)
    cu_F = np.zeros_like(f)
    cu_u = np.zeros_like(f)
    uF = np.zeros(f.shape[1:], dtype=np.float64)
    for a in range(dim):
        cu_F += view[:, a] * F[a]
        cu_u += view[:, a] * u_shift[a]
        uF += u_shift[a] * F[a]
    src = (cu_F - uF[None]) / cs2 + cu_u * cu_F / (cs2 * cs2)
    src = src * w * (1.0 - 0.5 / tau) * rho[None]
    return f + (feq - f) / tau + src


def macroscopic_with_force(
    table: LatticeTable, f: np.ndarray, force, rho: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Guo velocity definition ``u = (sum_i f_i c_i + F/2) / rho``."""
    rho, u = macroscopic(table, f)
    F = np.asarray([float(x) for x in force], dtype=np.float64).reshape(
        (table.dim,) + (1,) * table.dim)
    safe = np.where(rho > 1e-30, rho, 1.0)
    return rho, u + F / (2.0 * safe[None])


def _bounce_back(table: LatticeTable, f: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Swap populations with their opposites on solid nodes (full-way BB).

    Applying this after collision and before streaming puts the no-slip plane
    half a link outside the outermost fluid node, i.e. the standard half-way
    bounce-back wall position.
    """
    opp = np.asarray(table.opposite, dtype=np.int64)
    out = f.copy()
    out[:, solid] = f[opp, :][:, solid]
    return out


def _stream_zero_fill(table: LatticeTable, f: np.ndarray) -> np.ndarray:
    """Stream one cell per direction, zero-filling instead of wrapping."""
    out = np.empty_like(f)
    axes = tuple(range(f.ndim - 1))
    for i, c in enumerate(table.velocities):
        r = np.roll(f[i], shift=tuple(-ci for ci in c), axis=axes)
        for a, ci in enumerate(c):
            shift = -ci
            if shift == 0:
                continue
            sl = [slice(None)] * r.ndim
            if shift > 0:
                sl[a] = slice(0, shift)
            else:
                sl[a] = slice(r.shape[a] + shift, r.shape[a])
            r[tuple(sl)] = 0.0
        out[i] = r
    return out


def stream_half_way(
    table: LatticeTable, f_post: np.ndarray, wall_axis: int = 1
) -> np.ndarray:
    """Stream + half-way bounce-back on the two boundary layers of ``wall_axis``.

    Populations are streamed periodically first, then the populations that
    would have entered through a boundary layer are replaced by the opposite
    populations that just arrived at that layer from the interior.  That is the
    standard half-way bounce-back: the reflection happens half a link outside
    the boundary node, the no-slip planes are at ``-0.5`` and ``n - 0.5``, and
    mass is conserved locally (nothing is dropped, only mirrored).
    """
    opp = table.opposite
    out = stream(table, f_post)
    n = out.shape[1 + wall_axis]
    for i, c in enumerate(table.velocities):
        ci = c[wall_axis]
        if ci == 0:
            continue
        layer = [slice(None)] * (out.ndim - 1)
        layer[wall_axis] = 0 if ci > 0 else n - 1
        out[i][tuple(layer)] = out[opp[i]][tuple(layer)]
    return out


def _streamwise_profile(u_component: np.ndarray, axis: int) -> np.ndarray:
    """Average a component field over every axis except ``axis``."""
    other = tuple(a for a in range(u_component.ndim) if a != axis)
    return u_component.mean(axis=other)


# ---------------------------------------------------------------------------
# physical cross-checks
# ---------------------------------------------------------------------------
def _velocity(table, f, force, force_scheme):
    if force_scheme == "guo":
        return macroscopic_with_force(table, f, (force,) + (0.0,) * (table.dim - 1))
    return macroscopic(table, f)


def run_channel_poiseuille(
    table: LatticeTable,
    ny: int = 19,
    tau: float = 0.8,
    force: float = 4e-5,
    max_steps: int = 200_000,
    tol: float = 1e-14,
    check_every: int = 200,
    force_scheme: str = "guo",
) -> dict:
    """Force-driven channel (Poiseuille) in lattice units.

    Walls: full-way bounce-back on the two solid rows, half-way wall planes at
    ``y = 0.5`` and ``y = ny - 1.5`` (standard half-way convention), so the
    effective channel width is ``W = ny - 2`` and the closed-form solution is

        u(y) = g / (2 nu) * (y - 0.5) * (ny - 1.5 - y),   nu = cs2 (tau - 1/2)

    The same generic kernels are used in ``dim = 1, 2, 3``; transverse
    directions are periodic with two cells.
    """
    if table.dim < 2:
        raise ValueError("channel case needs dim >= 2 (walls are the y direction)")
    nu = float(table.cs2) * (tau - 0.5)
    if nu <= 0:
        raise ValueError(f"{table.name}: tau={tau} gives non-positive viscosity")

    grid = [2] + [ny] + [2] * (table.dim - 2)
    solid = np.zeros(grid, dtype=bool)
    solid[(slice(None), 0)] = True
    solid[(slice(None), ny - 1)] = True

    vel, _, _, _ = _as_float_table(table)
    rho0 = np.ones(grid)
    u0 = np.zeros((table.dim,) + tuple(grid))
    u0[0] = 1e-4  # tiny seed; the force sets the final amplitude
    f = equilibrium(table, rho0, u0)

    profile = None
    steps = 0
    for steps in range(1, max_steps + 1):
        f = collide_bgk(table, f, tau, force=(force,) + (0.0,) * (table.dim - 1),
                        force_scheme=force_scheme)
        f = _bounce_back(table, f, solid)
        f = stream(table, f)
        if steps % check_every == 0:
            _, u = _velocity(table, f, force, force_scheme)
            new_profile = _streamwise_profile(u[0], 1)[1 : ny - 1]
            if profile is not None:
                delta = float(np.max(np.abs(new_profile - profile)))
                if delta < tol:
                    profile = new_profile
                    break
            profile = new_profile

    rho, u = _velocity(table, f, force, force_scheme)
    ux = _streamwise_profile(u[0], 1)[1 : ny - 1]
    y = np.arange(1, ny - 1, dtype=np.float64)

    w_half = float(ny - 2)
    analytic_half = force / (2 * nu) * (y - 0.5) * (ny - 1.5 - y)
    w_center = float(ny - 1)
    analytic_center = force / (2 * nu) * y * (ny - 1.0 - y)

    def rel_err(a, b):
        return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    curvature = interior_curvature(ux)
    yb, yt = fit_wall_planes(ux, curvature, y0=1.0)
    fitted = curvature * (y - yb) * (yt - y)
    # wall-node centres are rows 0 and ny-1; positive offset = wall plane outside
    offset_bot = 0.0 - yb
    offset_top = yt - (ny - 1)

    return {
        "lattice": table.name,
        "ny": ny,
        "tau": tau,
        "nu_table": nu,
        "force": force,
        "force_scheme": force_scheme,
        "steps": steps,
        "converged": steps < max_steps,
        "wall_offset_apriori": 0.5,
        "width_half_way": w_half,
        "u_max_measured": float(ux.max()),
        "u_max_analytic_half_way": float(analytic_half.max()),
        "u_max_analytic_wall_center": float(analytic_center.max()),
        "rel_err_profile_half_way": rel_err(ux, analytic_half),
        "rel_err_profile_wall_center": rel_err(ux, analytic_center),
        "curvature_measured": curvature,
        "curvature_theory": force / (2.0 * nu),
        "curvature_rel_err": abs(curvature - force / (2.0 * nu)) / (force / (2.0 * nu)),
        "wall_y_bot": yb,
        "wall_y_top": yt,
        "wall_offset_bot": offset_bot,
        "wall_offset_top": offset_top,
        "rel_err_profile_fitted": rel_err(ux, fitted),
        "profile_fitted": [float(v) for v in fitted],
        "profile": [float(v) for v in ux],
        "profile_analytic_half_way": [float(v) for v in analytic_half],
        "density_max_dev": float(np.max(np.abs(rho - 1.0))),
    }


def run_taylor_green(
    table: LatticeTable,
    n: int = 64,
    u0: float = 1e-3,
    tau: float = 0.65,
    steps: int = 400,
    probe_every: int = 10,
    periods: int = 1,
) -> dict:
    """2-D Taylor-Green vortex: the table's own viscosity must predict the decay.

    Initial field (exact solution of the incompressible Navier-Stokes equations)

        ux = -u0 cos(kx) sin(ky),  uy = u0 sin(kx) cos(ky),  k = 2 pi / L

    with ``L = n / periods`` cells.  Kinetic energy decays as
    ``E(t) = E(0) exp(-4 nu k^2 t)``, so the *measured* decay rate implies a
    viscosity that is compared with ``nu_table = cs2 (tau - 1/2)``.
    """
    if table.dim != 2:
        raise ValueError("Taylor-Green case is 2-D")
    nu = float(table.cs2) * (tau - 0.5)
    if nu <= 0:
        raise ValueError(f"{table.name}: tau={tau} gives non-positive viscosity")
    periods = int(periods)
    k = 2.0 * np.pi * periods / n
    x = np.arange(n) + 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = np.empty((2, n, n))
    u[0] = -u0 * np.cos(k * X) * np.sin(k * Y)
    u[1] = u0 * np.sin(k * X) * np.cos(k * Y)
    rho0 = np.ones((n, n))
    f = equilibrium(table, rho0, u)

    energy = []
    times = []
    for step in range(steps + 1):
        if step % probe_every == 0:
            _, uu = macroscopic(table, f)
            energy.append(0.5 * float(np.mean(np.sum(uu * uu, axis=0))))
            times.append(float(step))
        if step == steps:
            break
        f = collide_bgk(table, f, tau)
        f = stream(table, f)

    times_a = np.asarray(times)
    energy_a = np.asarray(energy)
    # fit ln E over the second half of the window -> decay rate
    half = len(times_a) // 2
    slope, _ = np.polyfit(times_a[half:], np.log(energy_a[half:]), 1)
    nu_measured = -slope / (4.0 * k * k)

    _, u_final = macroscopic(table, f)
    decay = float(np.exp(-2.0 * nu * k * k * steps))
    ux_analytic = decay * (-u0 * np.cos(k * X) * np.sin(k * Y))
    uy_analytic = decay * (u0 * np.sin(k * X) * np.cos(k * Y))
    amp = float(np.max(np.abs(u_final)))
    field_err = float(
        np.max(np.hypot(u_final[0] - ux_analytic, u_final[1] - uy_analytic)) / amp
    )

    return {
        "lattice": table.name,
        "n": n,
        "periods": periods,
        "k": k,
        "u0": u0,
        "tau": tau,
        "steps": steps,
        "nu_table": nu,
        "nu_measured": float(nu_measured),
        "nu_rel_err": float(abs(nu_measured - nu) / nu),
        "energy_ratio": float(energy_a[-1] / energy_a[0]),
        "energy_ratio_analytic": float(np.exp(-4.0 * nu * k * k * steps)),
        "field_rel_err": field_err,
        "times": [float(t) for t in times_a],
        "energy": [float(e) for e in energy_a],
        "energy_analytic": [float(energy_a[0] * np.exp(-4.0 * nu * k * k * t)) for t in times_a],
    }


def mutate_speed_class(table: LatticeTable, changes: dict[int, Fraction]) -> LatticeTable:
    """Shift every weight of the speed class ``|c|^2 = key`` by ``changes[key]``."""
    weights = []
    for c, w in zip(table.velocities, table.weights):
        speed = sum(ci * ci for ci in c)
        weights.append(w + Fraction(changes.get(speed, 0)))
    return LatticeTable(
        name=f"{table.name}~shift{sorted(changes)}",
        dim=table.dim,
        velocities=table.velocities,
        weights=tuple(weights),
        cs2=table.cs2,
    )


def interior_curvature(profile: np.ndarray) -> float:
    """Wall-independent curvature ``A = -d2u/dy2 / 2`` of a channel profile.

    For the forced channel the closed-form solution is
    ``u(y) = A (y - y_bot)(y_top - y)`` with ``A = g / (2 nu)``, so the *second
    difference of the interior nodes* recovers ``g / (2 nu)`` without knowing
    where the walls are.  Boundary-adjacent nodes are excluded.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size < 5:
        raise ValueError("need at least 5 samples")
    second = profile[:-2] - 2.0 * profile[1:-1] + profile[2:]
    return float(-np.median(second[1:-1]) / 2.0)


def fit_wall_planes(
    profile: np.ndarray, curvature: float, y0: float = 0.0
) -> tuple[float, float]:
    """Least-squares wall planes ``(y_bot, y_top)`` with the curvature held fixed.

    ``u = A (y - yb)(yt - y) = -A y^2 + A (yb + yt) y - A yb yt`` is linear in
    ``(yb + yt)`` and ``yb yt``, so plain linear least squares suffices (no
    scipy dependency).
    """
    y = np.arange(len(profile), dtype=np.float64) + float(y0)
    u = np.asarray(profile, dtype=np.float64)
    rhs = u + curvature * y * y
    design = np.stack([y, np.ones_like(y)], axis=1)
    coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    s = coef[0] / curvature
    prod = -coef[1] / curvature
    disc = max(s * s - 4.0 * prod, 0.0)
    root = np.sqrt(disc)
    return float((s - root) / 2.0), float((s + root) / 2.0)
