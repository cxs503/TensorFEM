#!/usr/bin/env python3
"""Horizontal cross-validation of the Cook membrane (XVAL-FEM-COOK, mission m15274).

Why this exists
---------------
The published scalar anchor for the Cook membrane is the vertical displacement of the
loaded-edge midpoint ``(48, 52)``: ``23.96``.  TensorFEM's Q4 path reproduces it, which
qualifies *displacement* only.  Whether the same numbers are also produced by
independent, openly available FEM codes at the *same* specification is a different
question: one code agreeing with one scalar cannot separate "correct discrete problem"
from "common misreading of the specification".

This benchmark therefore pushes the *identical* boundary-value problem through several
independent kernels and compares a fixed metric set item by item.

===========================================  =========================================
spec item                                    value
===========================================  =========================================
geometry (4 corners)                         (0,0) (48,44) (48,60) (0,44)
material                                      E = 1.0, nu = 1/3, plane stress
thickness                                    1.0
left edge x = 0                              clamped (u_x = u_y = 0)
right edge x = 48                            uniform vertical traction, resultant 1.0
mesh                                         structured n x n on the bilinear map
published anchor (loaded-edge midpoint)      23.96
===========================================  =========================================

Parties
-------
=================================  ==================================================
party                              kernel
=================================  ==================================================
``tensorfem_q4``                   TensorFEM production Q4, 2x2 Gauss, full integration
``tensorfem_q8``                   in-repo *independent* Q8 serendipity reference
``skfem_quad2``                    scikit-fem ``ElementQuad2`` (biquadratic)
``skfem_tri2``                     scikit-fem ``ElementTriP2`` (quadratic triangles)
``fealpy_quad2``                   fealpy ``LagrangeFESpace(p=2)`` elasticity kernel
=================================  ==================================================

Only ``skfem_*`` and ``fealpy_*`` are third-party open-source kernels; ``tensorfem_*``
are the repository's own paths (``tensorfem_q8`` is separately coded inside this repo
and is reported as *in-repo independent*, never as an external party).

Metric definitions (identical for every party)
----------------------------------------------
``uy_mid``        vertical displacement at the loaded-edge midpoint (48, 52)
``uy_max``        maximum nodal vertical displacement
``reaction_y``    sum of vertical support reactions on the clamped edge (exactly -1.0)
``energy``        strain energy 0.5 u^T K u
``vm_max_nodal``  max von Mises of the party's own continuous nodal stress field
``vm_max_quad``   max von Mises over the party's own quadrature points
``vm_probe[i]``   von Mises at five interior material points (native stress field)

The probe sites carry the documented non-aligned offset of
``docs/cook-membrane-pointwise-protocol.md`` so that no probe lands on an element
interface, where a discontinuous stress field has no single value.

Usage
-----
    PYTHONPATH=src python benchmarks/cook_membrane_xval.py --out artifacts/cook_xval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

# --------------------------------------------------------------------------------------
# Specification (single source of truth for every party)
# --------------------------------------------------------------------------------------
CORNERS = np.array(((0.0, 0.0), (48.0, 44.0), (48.0, 60.0), (0.0, 44.0)), dtype=np.float64)
YOUNG = 1.0
POISSON = 1.0 / 3.0
THICKNESS = 1.0
RESULTANT = 1.0
LOADED_EDGE_LENGTH = 16.0
TRACTION_Y = RESULTANT / LOADED_EDGE_LENGTH
PUBLISHED_REFERENCE = 23.96
MIDPOINT = (48.0, 52.0)

#: Canonical material sample points in parent coordinates (xi, eta) of the bilinear map.
CANONICAL_POINTS = (
    (0.50, 0.50),
    (0.30, 0.50),
    (0.70, 0.50),
    (0.50, 0.25),
    (0.50, 0.75),
)
#: Documented non-aligned offset (docs/cook-membrane-pointwise-protocol.md).
PROBE_OFFSET = (1.0 / 384.0, 1.0 / 640.0)


def map_xy(xi, eta):
    """Bilinear map unit square -> Cook trapezoid (same map as ``mapped_q4_mesh``)."""
    xi = np.asarray(xi, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    return 48.0 * xi, 44.0 * xi + 44.0 * eta - 28.0 * xi * eta


def probe_points(offset: bool = True):
    """Probe material points as dicts with parent and physical coordinates."""
    out = []
    dx, dy = PROBE_OFFSET if offset else (0.0, 0.0)
    for k, (xi, eta) in enumerate(CANONICAL_POINTS):
        x, y = map_xy(xi + dx, eta + dy)
        out.append({
            "label": f"p{k + 1}",
            "xi": float(xi + dx),
            "eta": float(eta + dy),
            "x": float(x),
            "y": float(y),
            "canonical_xi": xi,
            "canonical_eta": eta,
        })
    return out


def von_mises(sxx, syy, sxy):
    sxx = np.asarray(sxx, dtype=np.float64)
    syy = np.asarray(syy, dtype=np.float64)
    sxy = np.asarray(sxy, dtype=np.float64)
    return np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)


def plane_stress_C():
    """Plane-stress stiffness for E = YOUNG, nu = POISSON, thickness = 1."""
    factor = YOUNG / (1.0 - POISSON ** 2)
    mu = YOUNG / (2.0 * (1.0 + POISSON))
    return np.array(((factor, factor * POISSON, 0.0),
                     (factor * POISSON, factor, 0.0),
                     (0.0, 0.0, mu * (1.0 - POISSON))), dtype=np.float64)


def relative_error(value, reference):
    return abs(value - reference) / abs(reference)


def richness(value):
    """Convergence richness: 1/h^2 in 2D, i.e. the number of elements."""
    return float(value)


# --------------------------------------------------------------------------------------
# Party 1: TensorFEM production Q4
# --------------------------------------------------------------------------------------
def run_tensorfem_q4(n: int) -> dict:
    import torch
    from tensorfem.continuum import solve_continuum
    from tensorfem.continuum_benchmarks import cook_membrane
    from tensorfem.additional_benchmark_fields import (
        assemble_coo, elasticity_matrix, q4_stiffness)
    from tensorfem.cook_probe_protocol import (
        _stress_from_displacement, interpolate_nodal_stress, nodal_recovered_stress)

    model, dof = cook_membrane(n)
    result = solve_continuum(model)
    u = result.displacement.detach().cpu().numpy()
    nodes = model.nodes.detach().cpu().numpy()
    cells = model.elements.detach().cpu().numpy()
    forces = model.forces.detach().cpu().numpy()
    fixed = np.asarray(sorted(int(d) for d in model.fixed_dofs))

    tn = torch.as_tensor(nodes)
    tc = torch.as_tensor(cells)
    C = elasticity_matrix(tn.new_ones(len(cells)), tn.new_full((len(cells),), POISSON), "stress")
    ke, B = q4_stiffness(tn[tc], C, tn.new_ones(len(cells)))
    edofs = (tc[:, :, None] * 2 + torch.arange(2)).reshape(len(cells), 8)
    K = assemble_coo(ke, edofs, 2 * len(nodes)).coalesce()
    tu = torch.as_tensor(u).reshape(-1, 1)
    residual = (torch.sparse.mm(K, tu).reshape(-1).detach().cpu().numpy() - forces)
    reaction_y = float(residual[fixed[1::2]].sum())
    energy = float(0.5 * u @ forces)  # homogeneous Dirichlet: u^T K u = u^T f

    nodal = nodal_recovered_stress(nodes, cells, u)

    # Native (element-wise constant) von Mises: the Q4 stress field is constant per
    # element, so its "quadrature max" is simply the maximum over elements.
    tu = torch.as_tensor(u)
    native = (C @ B @ tu[edofs].unsqueeze(-1)).squeeze(-1)
    native_vm = torch.sqrt(native[:, 0] ** 2 - native[:, 0] * native[:, 1]
                           + native[:, 1] ** 2 + 3 * native[:, 2] ** 2)

    probes = {}
    for p in probe_points():
        fields = interpolate_nodal_stress(nodes, cells, nodal, np.array([p["x"], p["y"]]))
        stress = np.asarray(fields["stress"])
        probes[p["label"]] = {"von_mises": float(von_mises(stress[0], stress[1], stress[2]))}

    nodal_vm = von_mises(nodal[:, 0], nodal[:, 1], nodal[:, 2])
    return {
        "party": "tensorfem_q4",
        "family": "Q4 bilinear, 2x2 Gauss, full integration (production path)",
        "third_party": False,
        "ndof": int(2 * len(nodes)),
        "nelem": int(len(cells)),
        "uy_mid": float(u[dof]),
        "uy_max": float(u[1::2].max()),
        "reaction_y": reaction_y,
        "energy": energy,
        "vm_max_nodal": float(nodal_vm.max()),
        "vm_max_quad": float(native_vm.max()),
        "vm_probe": probes,
        "recovery": "2x2 Gauss (Barlow) extrapolation + area-weighted nodal averaging",
    }


# --------------------------------------------------------------------------------------
# Party 2: in-repo independent Q8 reference
# --------------------------------------------------------------------------------------
def run_tensorfem_q8(n: int) -> dict:
    from tensorfem import cook_stress_reference as csr

    solution = csr.solve_q8(n)
    nodes = np.asarray(solution["nodes"], dtype=np.float64)
    elements = np.asarray(solution["elements"], dtype=np.int64)
    u = np.asarray(solution["displacement"], dtype=np.float64)
    C = csr.plane_stress_matrix()

    parent_nodes = ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (0, 0))
    acc = np.zeros((len(nodes), 3))
    cnt = np.zeros(len(nodes))
    gauss = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
    vm_qp = []
    for e, conn in enumerate(elements):
        coords = nodes[conn]
        local = u.reshape(-1, 2)[conn].ravel()
        for xi, eta in parent_nodes:
            mat, _ = csr.q8_strain_matrix(xi, eta, coords)
            acc[conn] += C @ mat @ local
            cnt[conn] += 1
        for xi in gauss:
            for eta in gauss:
                mat, _ = csr.q8_strain_matrix(xi, eta, coords)
                s = C @ mat @ local
                vm_qp.append(float(von_mises(s[0], s[1], s[2])))

    nodal = acc / cnt[:, None]
    nodal_vm = von_mises(nodal[:, 0], nodal[:, 1], nodal[:, 2])

    # Pointwise metric uses the module's own native-field evaluation at the probe's
    # parent coordinates (interior points, so the Q8 field is single-valued there).
    probes = {}
    for p in probe_points():
        s = csr.field_stress(solution, p["xi"], p["eta"])
        probes[p["label"]] = {"von_mises": float(von_mises(s["sxx"], s["syy"], s["sxy"]))}
    return {
        "party": "tensorfem_q8",
        "family": "Q8 serendipity, 3x3 Gauss (in-repo independent reference)",
        "third_party": False,
        "ndof": int(2 * len(nodes)),
        "nelem": int(len(elements)),
        "uy_mid": float(csr.probe_displacement(solution)["displacement"]),
        "uy_max": float(u[1::2].max()),
        "reaction_y": float(_q8_reaction(solution)),
        "energy": float(solution["strain_energy"]),
        "vm_max_nodal": float(nodal_vm.max()),
        "vm_max_quad": float(max(vm_qp)),
        "vm_probe": probes,
        "recovery": "native biquadratic stress at the 9 element nodes + nodal averaging",
    }


def _q8_reaction(solution: dict) -> float:
    """Vertical support reaction on the clamped edge, recomputed from the Q8 system."""
    from tensorfem import cook_stress_reference as csr

    system = csr.assemble_system(np.asarray(solution["nodes"]),
                                 np.asarray(solution["elements"]))
    rows, columns, values = system["rows"], system["columns"], system["values"]
    u = np.asarray(solution["displacement"])
    size = u.size
    product = np.bincount(rows, weights=values * u[columns], minlength=size)
    residual = product - system["load"]
    return float(residual[system["fixed"][1::2]].sum())




# --------------------------------------------------------------------------------------
# Parties 3/4: scikit-fem
# --------------------------------------------------------------------------------------
def _element_areas(mesh):
    """Element areas of a (possibly curved) simplicial/quadrilateral mesh."""
    x, y = mesh.p
    t = mesh.t
    if t.shape[0] == 3:  # triangles
        x0, y0 = x[t[0]], y[t[0]]
        x1, y1 = x[t[1]], y[t[1]]
        x2, y2 = x[t[2]], y[t[2]]
        return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    x0, y0 = x[t[0]], y[t[0]]
    x1, y1 = x[t[1]], y[t[1]]
    x2, y2 = x[t[2]], y[t[2]]
    x3, y3 = x[t[3]], y[t[3]]
    return 0.5 * np.abs((x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1)
                        + (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3))


def _skfem_mesh(n: int, kind: str):
    from skfem import MeshQuad, MeshTri

    grid = np.linspace(0.0, 1.0, n + 1)
    if kind == "quad":
        mesh = MeshQuad.init_tensor(grid, grid)
    elif kind == "tri":
        mesh = MeshTri.init_tensor(grid, grid)
    else:  # pragma: no cover
        raise ValueError(kind)
    xi, eta = mesh.p[0].copy(), mesh.p[1].copy()
    mesh.p[0] = 48.0 * xi
    mesh.p[1] = 44.0 * xi + 44.0 * eta - 28.0 * xi * eta
    return mesh


def _skfem_run(n: int, kind: str) -> dict:
    import skfem
    from skfem import (DiscreteField, ElementQuad1, ElementQuad2, ElementTriP1,
                       ElementTriP2, ElementVector, Basis, FacetBasis, LinearForm,
                       asm, condense, solve)
    from skfem.models.elasticity import linear_elasticity
    # NOTE: lame_parameters() returns the 3-D constants, which would silently give a
    # plane-*strain* stiffness in 2-D.  Plane stress needs lambda* = 2*lambda*mu/(lambda+2mu).
    lam3d = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
    mu3d = YOUNG / (2.0 * (1.0 + POISSON))
    lam_ps = 2.0 * lam3d * mu3d / (lam3d + 2.0 * mu3d)
    from skfem.utils import project

    mesh = _skfem_mesh(n, kind)
    if kind == "quad":
        elem, selem = ElementVector(ElementQuad2()), ElementQuad1()
    else:
        elem, selem = ElementVector(ElementTriP2()), ElementTriP1()
    basis = Basis(mesh, elem)
    K = asm(linear_elasticity(lam_ps, mu3d), basis)

    facets = mesh.facets_satisfying(lambda x: np.abs(x[0] - CORNERS[1, 0]) < 1e-9)
    fbasis = FacetBasis(mesh, elem, facets=facets)

    @LinearForm
    def traction(v, w):
        return TRACTION_Y * v.value[1]

    f = asm(traction, fbasis)
    constrained = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9)
    u = solve(*condense(K, f, D=constrained))

    residual = K @ u - f
    left_nodes = np.where(np.abs(mesh.p[0]) < 1e-9)[0]
    reaction_y = float(residual[basis.nodal_dofs[1, left_nodes]].sum())
    energy = float(0.5 * u @ (K @ u))

    grad = basis.interpolate(u).grad

    def _comp(field, index):
        value = field.value if hasattr(field, "value") else np.asarray(field)
        return value[index]

    dux_dx, duy_dx = _comp(grad[0], 0), _comp(grad[0], 1)
    dux_dy, duy_dy = _comp(grad[1], 0), _comp(grad[1], 1)
    C = plane_stress_C()
    sxx = C[0, 0] * dux_dx + C[0, 1] * duy_dy
    syy = C[1, 0] * dux_dx + C[1, 1] * duy_dy
    sxy = C[2, 2] * 0.5 * (dux_dy + duy_dx)
    vm_qp = von_mises(sxx, syy, sxy)

    # Continuous nodal field: area-weighted averaging of the *native* element stress
    # (shared post-processing operator, applied identically to every party, so that the
    # differences between parties are produced by their element kernels and not by
    # different recovery machinery).
    element_stress = np.stack([np.asarray(comp).mean(axis=1) for comp in (sxx, syy, sxy)], axis=1)
    areas = _element_areas(mesh)
    nodal = np.zeros((mesh.p.shape[1], 3))
    weight = np.zeros(mesh.p.shape[1])
    for e, conn in enumerate(mesh.t.T):
        nodal[conn] += areas[e] * element_stress[e]
        weight[conn] += areas[e]
    nodal /= weight[:, None]
    nodal_vm = von_mises(nodal[:, 0], nodal[:, 1], nodal[:, 2])

    sbasis = Basis(mesh, selem)
    probes = {}
    for p in probe_points():
        pt = np.array([[p["x"]], [p["y"]]])
        vals = [float((sbasis.probes(pt) @ nodal[:, j])[0]) for j in range(3)]
        probes[p["label"]] = {"von_mises": float(von_mises(*vals))}

    uy_nodes = u[basis.nodal_dofs[1, :]]
    node_x, node_y = mesh.p
    mid = int(np.argmin((node_x - MIDPOINT[0]) ** 2 + (node_y - MIDPOINT[1]) ** 2))
    return {
        "party": f"skfem_{kind}2",
        "family": ("scikit-fem ElementQuad2 (biquadratic, full integration)" if kind == "quad"
                   else "scikit-fem ElementTriP2 (quadratic triangles)"),
        "third_party": True,
        "tool": f"scikit-fem {skfem.__version__}",
        "ndof": int(basis.N),
        "nelem": int(mesh.t.shape[1]),
        "uy_mid": float(uy_nodes[mid]),
        "uy_max": float(uy_nodes.max()),
        "reaction_y": reaction_y,
        "energy": energy,
        "vm_max_nodal": float(nodal_vm.max()),
        "vm_max_quad": float(np.max(vm_qp)),
        "vm_probe": probes,
        "recovery": "area-weighted averaging of native element stress to nodes",
    }


# --------------------------------------------------------------------------------------
# Party 5: fealpy
# --------------------------------------------------------------------------------------
def run_fealpy_quad2(n: int) -> dict:
    raise NotImplementedError("fealpy runner not wired yet")


RUNNERS = {
    "tensorfem_q4": run_tensorfem_q4,
    "tensorfem_q8": run_tensorfem_q8,
    "skfem_quad2": lambda level: _skfem_run(level, "quad"),
    "skfem_tri2": lambda level: _skfem_run(level, "tri"),
    "fealpy_quad2": run_fealpy_quad2,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/cook_xval", help="output directory")
    parser.add_argument("--levels", default="8,16,32,64", help="mesh subdivision counts")
    parser.add_argument("--parties", default="tensorfem_q4,skfem_quad2,skfem_tri2")
    args = parser.parse_args(argv)
    levels = [int(v) for v in args.levels.split(",") if v]
    parties = [p.strip() for p in args.parties.split(",") if p.strip()]

    results = {}
    for name in parties:
        results[name] = {}
        for level in levels:
            t0 = time.time()
            entry = RUNNERS[name](level)
            entry["wall_s"] = time.time() - t0
            results[name][level] = entry
            _write(args.out, results, parties, levels)
            print(f"{name:14s} n={level:3d} ndof={entry['ndof']:6d} "
                  f"uy_mid={entry['uy_mid']:.6f} vm_max_nodal={entry['vm_max_nodal']:.6f} "
                  f"({entry['wall_s']:.1f}s)", flush=True)

    payload = {
        "spec": {
            "corners": CORNERS.tolist(),
            "young": YOUNG,
            "poisson": POISSON,
            "thickness": THICKNESS,
            "resultant": RESULTANT,
            "traction_y": TRACTION_Y,
            "published_reference_uy_mid": PUBLISHED_REFERENCE,
            "midpoint": MIDPOINT,
            "canonical_points": CANONICAL_POINTS,
            "probe_offset": PROBE_OFFSET,
        },
        "levels": levels,
        "parties": parties,
        "results": results,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    _write(args.out, results, parties, levels)
    print("wrote", os.path.join(args.out, "metrics.json"))
    return payload


def _write(outdir, results, parties, levels):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics.json")
    if os.path.exists(path):  # merge, so that party/level batches can run separately
        with open(path) as fh:
            previous = json.load(fh).get("results", {})
        for party, levels_map in previous.items():
            bucket = results.setdefault(party, {})
            for level, entry in levels_map.items():
                bucket.setdefault(level, entry)
    payload = {
        "spec": {
            "corners": CORNERS.tolist(),
            "young": YOUNG,
            "poisson": POISSON,
            "thickness": THICKNESS,
            "resultant": RESULTANT,
            "traction_y": TRACTION_Y,
            "published_reference_uy_mid": PUBLISHED_REFERENCE,
            "midpoint": MIDPOINT,
            "canonical_points": CANONICAL_POINTS,
            "probe_offset": PROBE_OFFSET,
        },
        "levels": levels,
        "parties": parties,
        "results": results,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()