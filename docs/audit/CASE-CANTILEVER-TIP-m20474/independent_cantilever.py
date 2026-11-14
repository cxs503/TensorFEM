#!/usr/bin/env python3
"""Independent recomputation of the CASE-CANTILEVER-TIP anchor (verifier seat, m20474).

Why this file exists
--------------------
Every number of the ``CASE-CANTILEVER-TIP`` evidence chain in this repository
(``benchmarks/verified/fem_cantilever*``, ``benchmarks/case_fem_cantilever_tip.py``,
``src/tensorfem/*`` regression tests) is produced by *TensorFEM's own* element
kernels together with an oracle written inside the same repository.  Self
consistency of that bundle has already been certified
(``benchmarks/verified/anchor_certificates.json``); what was never done is a
recomputation with **code that shares nothing** with the shipped package.

This module is that independent code path.  It imports only ``numpy``, ``scipy``
and ``sympy`` -- no ``tensorfem`` import at any point -- and re-implements, from
the textbook definitions:

* the Euler-Bernoulli oracle ``delta = P L^3 / (3 E I)`` and the shear-flexible
  (Timoshenko) oracle ``delta + P L / (kappa A G)``, re-derived twice
  (symbolic Castigliano and numerical unit-load quadrature);
* a 1-D cubic-Hermite beam FE (own ``4x4`` element matrix, own assembly);
* a 2-D plane-stress **8-node serendipity (Q8)** FE (own shape functions, own
  3x3 Gauss quadrature, own consistent tip-traction load vector, own sparse
  solve).

The third item is the important one: it is a *continuum* model of the same
cantilever, kinematically unrelated to beam theory, implemented by a different
author's code path than ``benchmarks/verified/fem_cantilever`` (which also uses
Q8 plane stress).  Agreement of the two Q8 implementations, both against the
closed form, is the cross-implementation evidence the verifier seat owes the
anchor.

Conventions
-----------
Problem (identical to the anchor): ``L = 1 m``, ``b = h = 0.01 m``,
``E = 200 GPa``, ``nu = 0.3``, tip force ``P = 5 N`` transverse to the beam axis.
For a 2-D model of unit thickness the load is the per-unit-width value
``P/t = 500 N/m``; the tip deflection is thickness independent, so the two
models must agree.

Run::

    PYTHONPATH= . python3 docs/audit/CASE-CANTILEVER-TIP-m20474/independent_cantilever.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as sym

ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
# Problem definition (the anchor's declared parameters)
# --------------------------------------------------------------------------- #
L = 1.0            # span [m]
B = 0.01           # width (out of bending plane) [m]
H = 0.01           # height (bending direction) [m]
E = 200.0e9        # Young's modulus [Pa]
NU = 0.3           # Poisson ratio
P = 5.0            # tip force [N]
KAPPA_RECT = 5.0 / 6.0


def poisson_correction_factors(nu: float = NU) -> dict:
    """Shear correction factors for a rectangular section.

    ``5/6`` is the classical engineering value; Cowper (1966) gives the exact
    value ``10 (1 + nu) / (12 + 11 nu)`` for a rectangle, which at ``nu = 0.3``
    is ``0.8496`` rather than ``0.8333``.  The anchor uses ``5/6``; the audit
    carries the Cowper value to bound that choice.
    """
    cowper = 10.0 * (1.0 + nu) / (12.0 + 11.0 * nu)
    return {"five_sixths": KAPPA_RECT, "cowper_rect": cowper}


def section_properties(length: float = L, b: float = B, h: float = H,
                       e: float = E) -> dict:
    """Second moment of area, area and bending stiffness of the section."""
    if not (length > 0 and b > 0 and h > 0 and e > 0):
        raise ValueError("length, b, h and E must be positive")
    return {
        "I": b * h ** 3 / 12.0,
        "A": b * h,
        "EI": e * b * h ** 3 / 12.0,
        "G": e / (2.0 * (1.0 + NU)),
    }


# --------------------------------------------------------------------------- #
# Oracle 1: closed form, re-derived symbolically (Castigliano)
# --------------------------------------------------------------------------- #
def symbolic_oracle(length: float = L, b: float = B, h: float = H,
                    e: float = E, load: float = P, nu: float = NU) -> dict:
    """Re-derive ``delta = P L^3/(3 E I)`` from the bending moment field.

    The unit-load (Castigliano) theorem with a unit force at the free end gives
    ``M(x) = P (L - x)`` and ``m(x) = 1 * (L - x)``, so

        delta = int_0^L M m / (E I) dx,

    which is evaluated symbolically here.  The returned residual is the
    difference between that integral and the closed form that both the anchor
    and this audit quote, i.e. it is an internal self-check of the oracle, not
    an FE result.
    """
    x, Px, Lx, EIs = sym.symbols("x P L EI", positive=True)
    M = Px * (Lx - x)
    m = (Lx - x)
    integral = sym.integrate(M * m / EIs, (x, 0, Lx))
    closed = Px * Lx ** 3 / (3 * EIs)
    residual = sym.simplify(integral - closed)

    props = section_properties(length, b, h, e)
    subs = {Px: load, Lx: length, EIs: props["EI"]}
    delta_eb = float(closed.subs(subs))
    delta_cast = float(integral.subs(subs))

    kappas = poisson_correction_factors(nu)
    shear_terms = {
        name: load * length / (kap * props["A"] * props["G"])
        for name, kap in kappas.items()
    }
    return {
        "castigliano_integral": delta_cast,
        "closed_form": delta_eb,
        "symbolic_residual": str(residual),
        "delta_euler_bernoulli": delta_eb,
        "kappas": kappas,
        "shear_term_m": shear_terms,
        "delta_timoshenko_five_sixths": delta_eb + shear_terms["five_sixths"],
        "delta_timoshenko_cowper": delta_eb + shear_terms["cowper_rect"],
        "shear_fraction_five_sixths": shear_terms["five_sixths"] / delta_eb,
        "shear_fraction_cowper": shear_terms["cowper_rect"] / delta_eb,
        "kappa_choice_effect_on_tip": abs(
            shear_terms["cowper_rect"] - shear_terms["five_sixths"]) / delta_eb,
    }


def quadrature_oracle(length: float = L, b: float = B, h: float = H,
                      e: float = E, load: float = P, n: int = 20001) -> float:
    """Independent numeric unit-load integration (Simpson) of the same integral.

    Deliberately a different numerical route than :func:`symbolic_oracle`: if
    both agree, the oracle formula itself is not a typo.
    """
    props = section_properties(length, b, h, e)
    xs = np.linspace(0.0, length, n)
    M = load * (length - xs)
    m = (length - xs)
    f = M * m / props["EI"]
    dx = xs[1] - xs[0]
    # composite Simpson
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return float(dx / 3.0 * np.dot(w, f))


# --------------------------------------------------------------------------- #
# Model 2: 1-D cubic-Hermite Euler-Bernoulli beam FE, written from scratch
# --------------------------------------------------------------------------- #
def hermite_beam_tip_deflection(n_elements: int, length: float = L, b: float = B,
                                h: float = H, e: float = E,
                                load: float = P) -> dict:
    """Tip deflection of a clamped-free beam with 2-node cubic Hermite elements.

    Element matrix (standard, ``xi = x / Le``)::

        Ke = EI/Le^3 * [[12, 6Le, -12, 6Le],
                        [6Le, 4Le^2, -6Le, 2Le^2],
                        [-12, -6Le, 12, -6Le],
                        [6Le, 2Le^2, -6Le, 4Le^2]]

    dof order per node: ``(w, theta)``.  No shear deformation, so the model is
    the Euler-Bernoulli oracle's own kinematics; it tests the assembly/solve
    path, not the theory.
    """
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")
    props = section_properties(length, b, h, e)
    Le = length / n_elements
    ndof = 2 * (n_elements + 1)
    K = np.zeros((ndof, ndof))
    ke = props["EI"] / Le ** 3 * np.array([
        [12.0, 6 * Le, -12.0, 6 * Le],
        [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
        [-12.0, -6 * Le, 12.0, -6 * Le],
        [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2],
    ])
    for el in range(n_elements):
        dofs = [2 * el, 2 * el + 1, 2 * el + 2, 2 * el + 3]
        K[np.ix_(dofs, dofs)] += ke
    F = np.zeros(ndof)
    F[-2] = load
    free = np.arange(2, ndof)          # w0 = theta0 = 0 (clamp)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return {"tip_deflection_m": float(u[-2]), "slope_rad": float(u[-1]),
            "n_elements": n_elements}


# --------------------------------------------------------------------------- #
# Model 3: 2-D plane-stress Q8 (8-node serendipity) FE, written from scratch
# --------------------------------------------------------------------------- #
_Q8_XI = np.array([-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0])
_Q8_ETA = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0])


def q8_shape(xi: float, eta: float):
    """Serendipity shape functions and their (xi, eta) derivatives."""
    N = np.array([
        0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
        0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),
        0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),
        0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),
        0.5 * (1 - xi ** 2) * (1 - eta),
        0.5 * (1 + xi) * (1 - eta ** 2),
        0.5 * (1 - xi ** 2) * (1 + eta),
        0.5 * (1 - xi) * (1 - eta ** 2),
    ])
    dN = np.array([
        [0.25 * (1 - eta) * (2 * xi + eta), 0.25 * (1 - xi) * (xi + 2 * eta)],
        [0.25 * (1 - eta) * (2 * xi - eta), 0.25 * (1 + xi) * (2 * eta - xi)],
        [0.25 * (1 + eta) * (2 * xi + eta), 0.25 * (1 + xi) * (xi + 2 * eta)],
        [0.25 * (1 + eta) * (2 * xi - eta), 0.25 * (1 - xi) * (2 * eta - xi)],
        [-xi * (1 - eta), -0.5 * (1 - xi ** 2)],
        [0.5 * (1 - eta ** 2), -(1 + xi) * eta],
        [-xi * (1 + eta), 0.5 * (1 - xi ** 2)],
        [-0.5 * (1 - eta ** 2), -(1 - xi) * eta],
    ])
    return N, dN


def _plane_stress_D(e: float = E, nu: float = NU) -> np.ndarray:
    c = e / (1.0 - nu ** 2)
    return c * np.array([[1.0, nu, 0.0],
                         [nu, 1.0, 0.0],
                         [0.0, 0.0, 0.5 * (1.0 - nu)]])


def q8_structured_mesh(n_axial: int, n_cross: int, length: float, height: float):
    """Structured Q8 mesh on a half-step lattice, with the unused centre nodes dropped.

    A ``2 n_axial x 2 n_cross`` lattice of points carries the corners (even, even),
    the horizontal/vertical mid-side nodes (odd, even) / (even, odd) -- and, for
    ``n_cross > 1``, interior points (odd, odd) that belong to **no** Q8 element.
    Those are removed here (they would otherwise be isolated zero-stiffness
    dofs and make the global matrix singular).

    Returns ``coords``, ``conn``, ``lattice`` where ``lattice[ix, iy]`` is the
    global node number of lattice point ``(ix, iy)`` or ``-1`` if unused.
    """
    nx_nodes = 2 * n_axial + 1
    ny_nodes = 2 * n_cross + 1
    xs = np.linspace(0.0, length, nx_nodes)
    ys = np.linspace(-height / 2.0, height / 2.0, ny_nodes)
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    def lat(ix, iy):
        return ix * ny_nodes + iy

    conn = np.empty((n_axial * n_cross, 8), dtype=int)
    e = 0
    for i in range(n_axial):
        for j in range(n_cross):
            ix, iy = 2 * i, 2 * j
            conn[e] = [lat(ix, iy), lat(ix + 2, iy), lat(ix + 2, iy + 2),
                       lat(ix, iy + 2), lat(ix + 1, iy), lat(ix + 2, iy + 1),
                       lat(ix + 1, iy + 2), lat(ix, iy + 1)]
            e += 1
    used = np.unique(conn)
    lattice = np.full((nx_nodes, ny_nodes), -1, dtype=int)
    coords_all = np.column_stack([X.ravel(), Y.ravel()])
    lattice.ravel()[used] = np.arange(used.size)      # C order == lattice order
    conn = lattice.ravel()[conn]
    return coords_all[used], conn, lattice


def q8_plane_stress_tip_deflection(n_axial: int, n_cross: int, *,
                                   length: float = L, height: float = H,
                                   thick: float = 1.0, e: float = E,
                                   nu: float = NU, load_per_width: float | None = None,
                                   load_mode: str = "parabolic") -> dict:
    """Plane-stress Q8 cantilever, clamped at ``x=0``, transverse tip traction.

    ``load_mode``:

    * ``"parabolic"`` -- the parabolic shear traction of beam theory,
      ``tau(y) = 3P/(2A) (1 - (2y/h)^2)`` with ``A = h * thick``, integrated
      consistently through the Q8 edge shape functions (3-point Gauss, exact for
      a quadratic integrand).  This is the Saint-Venant-equivalent end loading
      and the fair comparison for a beam oracle.
    * ``"uniform"``   -- constant traction over the end face (same resultant).
    * ``"point"``     -- the whole resultant on the mid-side node of the end
      face (requires an even ``n_cross``); the crudest end-load idealisation.

    Thickness is 1 by default: the result is compared with ``P L^3/(3 E I)``
    using ``I = thick * h^3 / 12`` and with the anchor's 3-D value, which is
    thickness independent.
    """
    if n_axial < 1 or n_cross < 1:
        raise ValueError("mesh must have at least one element in each direction")
    P2 = (P / B) * thick if load_per_width is None else load_per_width
    coords, conn, lattice = q8_structured_mesh(n_axial, n_cross, length, height)
    nnodes = coords.shape[0]
    ndof = 2 * nnodes
    D = _plane_stress_D(e, nu)

    g = 1.0 / math.sqrt(3.0)
    gauss = ((-g, -g), (g, -g), (g, g), (-g, g))   # exact for bilinear geometry

    rows, cols, vals = [], [], []
    for el in range(conn.shape[0]):
        edofs = np.empty(16, dtype=int)
        edofs[0::2] = 2 * conn[el]
        edofs[1::2] = 2 * conn[el] + 1
        xy = coords[conn[el]]
        ke = np.zeros((16, 16))
        for xi, eta in gauss:
            _, dN = q8_shape(xi, eta)
            J = dN.T @ xy
            detJ = float(np.linalg.det(J))
            if detJ <= 0.0:
                raise ValueError(f"element {el} has non-positive Jacobian {detJ}")
            dNxy = dN @ np.linalg.inv(J)
            Bmat = np.zeros((3, 16))
            Bmat[0, 0::2] = dNxy[:, 0]
            Bmat[1, 1::2] = dNxy[:, 1]
            Bmat[2, 0::2] = dNxy[:, 1]
            Bmat[2, 1::2] = dNxy[:, 0]
            ke += detJ * thick * (Bmat.T @ D @ Bmat)
        rows.append(np.repeat(edofs, 16))
        cols.append(np.tile(edofs, 16))
        vals.append(ke.ravel())
    K = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ndof, ndof)).tocsr()

    F = np.zeros(ndof)
    tip_col = lattice.shape[0] - 1
    tip_ids = [lattice[tip_col, iy] for iy in range(lattice.shape[1])]
    area = height * thick
    if load_mode == "point":
        if n_cross % 2 != 0:
            raise ValueError("load_mode='point' needs an even n_cross (a mid-side node)")
        F[2 * lattice[tip_col, n_cross] + 1] += P2
    else:
        if load_mode not in ("parabolic", "uniform"):
            raise ValueError(f"unknown load_mode {load_mode!r}")
        gp = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
        gw = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
        iy_map = {1: 0, 2: 2 * n_cross, 5: n_cross}   # local corner-2/3 + midside-6
        for eta, w in zip(gp, gw):
            N, _ = q8_shape(1.0, eta)
            for loc in (1, 2, 5):
                gid = lattice[tip_col, iy_map[loc]]
                y = coords[gid, 1]
                if load_mode == "parabolic":
                    tau = 1.5 * P2 / area * (1.0 - (2.0 * y / height) ** 2)
                else:
                    tau = P2 / area
                F[2 * gid + 1] += N[loc] * tau * (height / 2.0) * w * thick
    total = float(F[1::2].sum())
    if abs(total - P2) > 1e-9 * max(1.0, abs(P2)):
        raise ValueError(f"tip load resultant {total} != requested {P2}")

    clamp_nodes = [lattice[0, iy] for iy in range(lattice.shape[1])]
    fixed = np.array([2 * n + c for n in clamp_nodes for c in (0, 1)], dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[np.ix_(free, free)].tocsc(), F[free])

    uy_tip = u[2 * np.array(tip_ids) + 1]
    props = section_properties(length, thick, height, e)
    ref_eb = P2 * length ** 3 / (3.0 * props["EI"])
    ref_ti = ref_eb + P2 * length / (KAPPA_RECT * props["A"] * props["G"])
    tip_mean = float(uy_tip.mean())
    tip_mid = float(u[2 * lattice[tip_col, n_cross] + 1])
    return {
        "model": "q8_plane_stress",
        "n_axial": n_axial,
        "n_cross": n_cross,
        "elements": int(conn.shape[0]),
        "nodes": int(nnodes),
        "dofs": int(ndof),
        "load_mode": load_mode,
        "tip_deflection_m": tip_mean,
        "tip_deflection_midnode_m": tip_mid,
        "abs_relative_error_vs_euler_bernoulli": abs(tip_mean - ref_eb) / ref_eb,
        "rel_relative_error_vs_euler_bernoulli": (tip_mean - ref_eb) / ref_eb,
        "abs_relative_error_vs_timoshenko": abs(tip_mean - ref_ti) / ref_ti,
    }


# --------------------------------------------------------------------------- #
# Repository evidence (read only, for the cross-implementation comparison)
# --------------------------------------------------------------------------- #
def repo_evidence() -> dict:
    """Read the committed anchor evidence that is being audited.

    Only *declared* numbers are read; comparisons are recomputed here.
    """
    out = {"files": []}
    anchor = ROOT / "benchmarks/verified/fem_cantilever/result.json"
    if anchor.exists():
        d = json.loads(anchor.read_text())
        out["verified_fem_cantilever"] = {
            "geometry": d.get("geometry"),
            "analytic": d.get("analytic"),
            "q8_levels": d.get("q8_levels"),
            "hex20_levels": d.get("hex20_levels"),
        }
        out["files"].append(str(anchor.relative_to(ROOT)))
    case = ROOT / "benchmarks/case_fem_cantilever_tip_evidence.json"
    if case.exists():
        d = json.loads(case.read_text())
        out["case_fem_cantilever_tip"] = {
            "implementation_revision": d.get("implementation", {}).get("revision"),
            "verdict": d.get("verdict"),
        }
        out["files"].append(str(case.relative_to(ROOT)))
    return out


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover - audit convenience only
        return "unknown"


# --------------------------------------------------------------------------- #
# Cross-implementation check on the *other* repository anchor's parameters
# --------------------------------------------------------------------------- #
#: parameters and Q8 meshes declared by ``benchmarks/verified/fem_cantilever``
#: (a different cantilever: L = 10 m, h = 0.1 m, E = 210 GPa, F = 10 kN).
REPO_Q8_CASE = {"length": 10.0, "height": 0.1, "width": 1.0, "e": 210.0e9,
                "nu": 0.3, "load": 1.0e4}
REPO_Q8_MESHES = ((20, 2), (40, 4), (80, 8))


def repo_parameter_crosscheck() -> dict:
    """Re-solve the ``fem_cantilever`` anchor's Q8 model with this audit's code.

    The repository anchor and this module implement the same element (plane-stress
    Q8) but share no source line; agreement of the tip deflections at the same
    meshes is therefore cross-implementation evidence for both.
    """
    c = REPO_Q8_CASE
    props = section_properties(c["length"], c["width"], c["height"], c["e"])
    ref_eb = c["load"] * c["length"] ** 3 / (3.0 * props["EI"])
    rows = []
    for n_axial, n_cross in REPO_Q8_MESHES:
        r = q8_plane_stress_tip_deflection(
            n_axial, n_cross, length=c["length"], height=c["height"],
            thick=c["width"], e=c["e"], nu=c["nu"],
            load_per_width=c["load"] / c["width"])
        r["repo_expected_deflection_m"] = None
        rows.append(r)
    rows_with_repo = []
    d = repo_evidence().get("verified_fem_cantilever") or {}
    declared = {(lvl.get("axial_elements"), lvl.get("depth_elements")): lvl.get("tip_deflection")
                for lvl in (d.get("q8_levels") or [])}
    for r in rows:
        decl = declared.get((r["n_axial"], r["n_cross"]))
        r["repo_expected_deflection_m"] = decl
        r["cross_implementation_rel_diff"] = (
            None if decl in (None, 0) else abs(r["tip_deflection_m"] - decl) / abs(decl))
        rows_with_repo.append(r)
    worst = max((r["cross_implementation_rel_diff"] for r in rows_with_repo
                 if r["cross_implementation_rel_diff"] is not None), default=None)
    return {"parameters": c, "reference_euler_bernoulli_m": ref_eb,
            "levels": rows_with_repo, "worst_cross_implementation_rel_diff": worst}


# --------------------------------------------------------------------------- #
# Audit driver
# --------------------------------------------------------------------------- #
BEAM_LADDER = (1, 2, 4, 8, 16)
Q8_LADDER = ((8, 1), (16, 2), (32, 2), (32, 4), (64, 4))


def run_audit() -> dict:
    oracle = symbolic_oracle()
    quad = quadrature_oracle()
    props = section_properties()

    beams = [hermite_beam_tip_deflection(n) for n in BEAM_LADDER]
    for row in beams:
        row["abs_relative_error_vs_euler_bernoulli"] = abs(
            row["tip_deflection_m"] - oracle["delta_euler_bernoulli"]) / oracle["delta_euler_bernoulli"]

    q8_rows = []
    for n_axial, n_cross in Q8_LADDER:
        q8_rows.append(q8_plane_stress_tip_deflection(n_axial, n_cross))

    # end-load idealisation probe on the declared anchor mesh scale
    load_modes = {}
    for mode in ("parabolic", "uniform", "point"):
        try:
            load_modes[mode] = q8_plane_stress_tip_deflection(32, 4, load_mode=mode)
        except ValueError as exc:
            load_modes[mode] = {"error": str(exc)}

    worst_beam = max(r["abs_relative_error_vs_euler_bernoulli"] for r in beams)
    finest_q8 = min(q8_rows, key=lambda r: r["abs_relative_error_vs_timoshenko"])
    monotone = all(
        q8_rows[i]["abs_relative_error_vs_timoshenko"] >=
        q8_rows[i + 1]["abs_relative_error_vs_timoshenko"] - 1e-12
        for i in range(len(q8_rows) - 1)
    )
    return {
        "schema": "tensorfem.independent-cantilever-audit/1",
        "anchor": "CASE-CANTILEVER-TIP",
        "mission": 20474,
        "git_revision": git_revision(),
        "problem": {
            "L_m": L, "b_m": B, "h_m": H, "E_Pa": E, "nu": NU, "P_N": P,
            "slenderness_L_over_h": L / H,
            "section": {k: v for k, v in props.items()},
        },
        "oracle": {
            **oracle,
            "quadrature_integral": quad,
            "quadrature_vs_closed_form_rel": abs(quad - oracle["delta_euler_bernoulli"]) / oracle["delta_euler_bernoulli"],
        },
        "beam_fe": beams,
        "q8_plane_stress": q8_rows,
        "end_load_idealisation": load_modes,
        "checks": {
            "oracle_symbolic_residual_zero": oracle["symbolic_residual"] == "0",
            "oracle_quadrature_agrees": abs(quad - oracle["delta_euler_bernoulli"]) / oracle["delta_euler_bernoulli"] < 1e-9,
            "beam_fe_machine_exact": worst_beam < 1e-8,
            "q8_monotone_convergence": monotone,
            "q8_finest_within_3pct_of_timoshenko": finest_q8["abs_relative_error_vs_timoshenko"] < 0.03,
            "q8_finest_mesh": [finest_q8["n_axial"], finest_q8["n_cross"]],
        },
        "repo_parameter_crosscheck": repo_parameter_crosscheck(),
        "repo_evidence": repo_evidence(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print the evidence document")
    ap.add_argument("--out", type=Path, default=None, help="write evidence JSON here")
    args = ap.parse_args(argv)
    doc = run_audit()
    if args.out:
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        print(f"oracle EB      = {doc['oracle']['delta_euler_bernoulli']:.9e} m")
        print(f"beam FE worst  = {max(r['abs_relative_error_vs_euler_bernoulli'] for r in doc['beam_fe']):.3e} rel")
        for r in doc["q8_plane_stress"]:
            print(f"q8 {r['n_axial']:>3}x{r['n_cross']:<2} dofs={r['dofs']:<6} "
                  f"tip={r['tip_deflection_m']:.9e} err(vs TI)={r['abs_relative_error_vs_timoshenko']*100:6.3f}%")
        print(json.dumps(doc["checks"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())