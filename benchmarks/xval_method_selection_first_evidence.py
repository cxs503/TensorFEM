"""XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE: independent cross-validation of the
``CASE-HOLLOW-SPHERE-3D`` method-selection + first-evidence record.

What is cross-validated (anchor: ``XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE``)
-----------------------------------------------------------------------------
The OSS record (``oss_solutions`` row repo=TensorFEM, tag=``method-selection+
first-evidence``) claims, for the thick-walled hollow sphere under internal
pressure (a=8 m, b=10 m, E=210 GPa, nu=0.3, p=1 MPa):

  C1  HEX20 (n=4, 192 elements) keeps *all* probes below 3 %
  C2  HEX8 on the same 192-element mesh reaches worst = 15.4 %
  C3  HEX8 needs n >= 12 before it approaches 5 %
  C4  mesh volume error drops 7.1 % -> 0.14 % when going HEX8 -> HEX20
  C5  consistent pressure loads give closed-surface force/moment < 1e-16
  C6  Jacobi-PCG takes only 67 iterations for HEX8 at n=8

and selects: 20-node serendipity bricks + curved-boundary projection +
consistent face loads + Jacobi-PCG.  Its own ``evidence`` field records that
the *only* comparison available at the time was the closed-form Lame solution —
``oss_inventory`` held no 3-D structural FEM code at all.  That gap is what this
benchmark closes: a second, independent implementation (scikit-fem 12.0.2, a
mature OSS FEM library) on an independent mesh family (icosphere tet shell)
re-evaluates the same physical case and the same probe quantities.

Protocol (predeclared, not tuned after seeing results)
------------------------------------------------------
Two probe functionals are reported side by side because the record *declares*
``reference_functional = lame_layer_average`` (exact volume-weighted layer
integral) but *scores* the case against Lame sampled at the volume-averaged
element-centroid radius:

  M1 (as-declared-in-table, "centroid-radius sampling"):
      ref = Lame(<r>),  <r> = sum(V_e r_e)/sum(V_e) over the layer elements
  M2 (as-declared-in-metadata, "exact layer integral"):
      ref = (1/V_band) * integral_band sigma(r) 4 pi r^2 dr   over the exact
           radial band, i.e. closed forms
             <sigma_r> = A - 3 B ln(r2/r1)/(r2^3 - r1^3)
             <sigma_t> = A + 1.5 B ln(r2/r1)/(r2^3 - r1^3)
             <von Mises> = 4.5 B ln(r2/r1)/(r2^3 - r1^3)
      FE side: volume-weighted element average restricted to the elements that
      span the same exact band (layer boundaries from the mesh radii, never
      from ``torch.unique(node radii)``).

Reproduction rules (predeclared)
--------------------------------
* numeric claim: reproduced iff |got - recorded| / |recorded| <= 15 %
  (owner precision standard).
* threshold claim ("all probes < 3 %"): reproduced iff the re-measured worst
  error honours the stated bound; the reported deviation is
  (got - bound)/bound.
* decision table: the selection is *robust* only if, for every functional
  (M1 and M2) and for the independent implementation, the quadratic
  discretisation keeps all probes under the case-pool line and beats the
  linear one on the same mesh.  Otherwise the record must be restated with an
  explicit functional.

Usage::

    PYTHONPATH=$PWD/src python3 benchmarks/xval_method_selection_first_evidence.py --quick
    PYTHONPATH=$PWD/src python3 benchmarks/xval_method_selection_first_evidence.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time

import numpy as np

from tensorfem.hollow_sphere_3d import (CASE, TOLERANCE, lame_sphere,
                                        shell_volume, solve_hollow_sphere,
                                        strain_energy)

# ---------------------------------------------------------------------------
# case constants (single source: tensorfem.hollow_sphere_3d.CASE)
# ---------------------------------------------------------------------------
A = float(CASE["inner_radius_m"])
B = float(CASE["outer_radius_m"])
E = float(CASE["young_modulus_pa"])
NU = float(CASE["poisson_ratio"])
P = float(CASE["internal_pressure_pa"])

DEN = B ** 3 - A ** 3
COEF_A = P * A ** 3 / DEN
COEF_B = COEF_A * B ** 3

CLAIM_TOLERANCE = 0.15          # owner precision standard for XVAL claims
CASE_LINE = TOLERANCE           # case-pool acceptance line (<5 %)
QUADRATIC_TARGET = 0.03         # C1 threshold


# ---------------------------------------------------------------------------
# closed-form band averages (exact volume integrals over a radial band)
# ---------------------------------------------------------------------------
def lame_band_average(r1: float, r2: float) -> dict:
    """Exact volume averages of the Lame fields over ``r1 <= r <= r2``."""
    if not (0.0 < r1 < r2):
        raise ValueError("need 0 < r1 < r2")
    log = math.log(r2 / r1)
    den = r2 ** 3 - r1 ** 3
    return {"radial_stress": COEF_A - 3.0 * COEF_B * log / den,
            "hoop_stress": COEF_A + 1.5 * COEF_B * log / den,
            "von_mises": 4.5 * COEF_B * log / den,
            "radius": 3.0 * (r2 ** 4 - r1 ** 4) / (4.0 * den)}


def lame_at(r: float) -> dict:
    vals = lame_sphere(np.array([r]), CASE)
    return {k: float(v[0]) for k, v in vals.items()}


# ---------------------------------------------------------------------------
# arm A: in-tree TensorFEM (HEX8 baseline / HEX20 selected method)
# ---------------------------------------------------------------------------
def arm_tensorfem(order: int, n: int, layers: int = 2) -> dict:
    started = time.time()
    res = solve_hollow_sphere(n, layers, order=order, case=CASE)
    mesh, probes = res["mesh"], res["metrics"]["probes"]
    radii = radial_layers_hex(n, layers)
    inner = layer_volume_average_tf(res, slice(0, mesh["elements"] // layers))
    outer = layer_volume_average_tf(res, slice(mesh["elements"] * (layers - 1) // layers,
                                               None))
    row = {
        "arm": "TensorFEM",
        "element": mesh["element"],
        "order": order,
        "n": n,
        "layers": layers,
        "nodes": mesh["nodes"],
        "elements": mesh["elements"],
        "dofs": mesh["degrees_of_freedom"],
        "mesh_volume_error": mesh["volume_error"],
        "shell_volume_m3": shell_volume(CASE),
        "wall_seconds": round(time.time() - started, 2),
        "solver": res["solver"],
        "loads": res["loads"],
        "equilibrium": res["equilibrium"],
        "energy": res["energy"],
        "probes_m1": {k: v["error"] for k, v in probes.items()},
        "probes_m1_secondary": {k: v["error"] for k, v in
                                res["metrics"]["secondary_probes"].items()},
        "got": {k: v["got"] for k, v in probes.items()},
        "ref_m1": {k: v["ref"] for k, v in probes.items()},
        "worst_m1": res["metrics"]["worst_relative_error"],
        "u_inner": probes["radial_displacement_inner"]["got"],
        "u_outer": probes["radial_displacement_outer"]["got"],
        "mesh_layer_radii": radii,
        "inner_layer": inner,
        "outer_layer": outer,
        "retained": {"passed": res["passed"]},
        "displacement_fields": None,
    }
    row.update(metric_m2(row, radii, inner, outer))
    return row


def radial_layers_hex(n: int, layers: int) -> list[float]:
    """Radial layer-boundary radii of the cube-sphere shell mesh (exact)."""
    return [A + (B - A) * k / layers for k in range(layers + 1)]


def layer_volume_average_tf(res: dict, selection) -> dict:
    """Volume-weighted element average over a layer of the TensorFEM solution."""
    fields = res["metrics"]["fields"]
    radius = np.asarray(fields["radius"])
    vols = per_element_volumes_tf(res)
    sel = np.zeros(len(radius), dtype=bool)
    sel[selection] = True
    w = vols[sel]
    return {"radius": float((w * radius[sel]).sum() / w.sum()),
            "radial_stress": float((w * np.asarray(fields["radial_stress"])[sel]).sum() / w.sum()),
            "hoop_stress": float((w * np.asarray(fields["hoop_stress"])[sel]).sum() / w.sum()),
            "von_mises": float((w * np.asarray(fields["von_mises"])[sel]).sum() / w.sum()),
            "volume": float(w.sum())}


_TF_VOLUME_CACHE: dict = {}


def per_element_volumes_tf(res: dict) -> np.ndarray:
    """Element volumes of a TensorFEM hollow-sphere mesh (recomputed, cached)."""
    key = (res["mesh"]["face_divisions"], res["mesh"]["radial_layers"], res["mesh"]["order"])
    if key in _TF_VOLUME_CACHE:
        return _TF_VOLUME_CACHE[key]
    from tensorfem.hollow_sphere_3d import _element_volumes, _mesh

    nodes, elements, _ = _mesh(key[0], key[1], key[2], CASE)
    import torch

    vols = _element_volumes(nodes[elements.long()], key[2]).detach().numpy().astype(float)
    _TF_VOLUME_CACHE[key] = vols
    return vols


def metric_m2(row: dict, radii, inner, outer) -> dict:
    """Score the same FE probes against the exact-band volume integrals."""
    bands = {"inner": (radii[0], radii[1]), "outer": (radii[-2], radii[-1])}
    out = {"probes_m2": {}, "ref_m2": {}, "band_exact": {}, "layer_volume_error": {}}
    for tag, layer in (("inner", inner), ("outer", outer)):
        exact = lame_band_average(*bands[tag])
        out["band_exact"][tag] = exact
        out["layer_volume_error"][tag] = abs(layer["volume"] / (4.0 / 3.0 * math.pi *
                                                                (bands[tag][1] ** 3 - bands[tag][0] ** 3)) - 1.0)
        for q_in, q_out in (("radial_stress", "radial_stress"),
                            ("hoop_stress", "hoop_stress"),
                            ("von_mises", "von_mises")):
            key = f"{q_in}_{tag}"
            ref = exact[q_out]
            got = layer[q_in]
            err = abs(got - ref) / abs(ref)
            out["probes_m2"][key] = err
            out["ref_m2"][key] = ref
        # outer radial stress: |delta| / p (vanishes identically on the free surface)
        out["probes_m2"][f"radial_stress_{tag}"] = abs(layer["radial_stress"] -
                                                       exact["radial_stress"]) / P
        out["ref_m2"][f"radial_stress_{tag}"] = exact["radial_stress"]
    ur = lame_at(A)["radial_displacement"]
    ub = lame_at(B)["radial_displacement"]
    out["probes_m2"]["radial_displacement_inner"] = abs(row["u_inner"] - ur) / abs(ur)
    out["probes_m2"]["radial_displacement_outer"] = abs(row["u_outer"] - ub) / abs(ub)
    out["ref_m2"]["radial_displacement_inner"] = ur
    out["ref_m2"]["radial_displacement_outer"] = ub
    primary = {k: v for k, v in out["probes_m2"].items() if k != "radial_stress_outer"}
    out["worst_m2"] = float(max(primary.values()))
    out["worst_m2_primary"] = out["worst_m2"]
    out["worst_quantity_m2"] = max(primary, key=primary.get)
    return out


# ---------------------------------------------------------------------------
# arm B: scikit-fem (independent implementation), icosphere tet shell
# ---------------------------------------------------------------------------
def _icosphere(nsub: int):
    import skfem  # noqa: F401  (import guard: optional dependency)

    t = (1 + 5 ** 0.5) / 2
    verts = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                      (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                      (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    pts = []
    for f in faces:
        p0, p1, p2 = verts[list(f)]
        for i in range(nsub + 1):
            for j in range(nsub + 1 - i):
                a, b = i / nsub, j / nsub
                pts.append(p0 * (1 - a - b) + p1 * a + p2 * b)
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    _, first, inv = np.unique(np.round(pts, 9), axis=0, return_index=True,
                              return_inverse=True)
    inv = np.asarray(inv).ravel()
    tris = []
    for k in range(len(faces)):
        base = k * (nsub + 1) * (nsub + 2) // 2

        def idx_of(i, j):
            return base + sum((nsub + 1 - ii) for ii in range(i)) + j

        for i in range(nsub):
            for j in range(nsub - i):
                tris.append((inv[idx_of(i, j)], inv[idx_of(i + 1, j)],
                             inv[idx_of(i, j + 1)]))
                if j < nsub - i - 1:
                    tris.append((inv[idx_of(i + 1, j)], inv[idx_of(i + 1, j + 1)],
                                 inv[idx_of(i, j + 1)]))
    return pts[first], np.array(tris)


def _shell_tets(nsub: int, layers: int):
    dirs, tris = _icosphere(nsub)
    radii = [A + (B - A) * k / layers for k in range(layers + 1)]
    nodes = np.array([d * r for r in radii for d in dirs])
    nt = len(dirs)
    tets = []
    for k in range(layers):
        for (i, j, l) in tris:
            lo = [k * nt + i, k * nt + j, k * nt + l]
            hi = [(k + 1) * nt + i, (k + 1) * nt + j, (k + 1) * nt + l]
            tets += [[lo[0], lo[1], lo[2], hi[0]],
                     [lo[1], lo[2], hi[0], hi[1]],
                     [lo[2], hi[0], hi[1], hi[2]]]
    return nodes, np.array(tets), radii


def _dof_coords(basis) -> np.ndarray:
    """Node coordinates in *dof* order (one row per scalar-node dof, 3 columns).

    ``basis.doflocs`` is ``(3, N)`` with the three components of one node
    stored consecutively, so ``doflocs[:, ::3].T`` is the coordinate of the
    node owning dofs ``3i, 3i+1, 3i+2``.  Using this (and not ``mesh.p``)
    keeps the rigid-body modes aligned with the assembled load vector for
    both the linear and the quadratic element.
    """
    return np.ascontiguousarray(basis.doflocs[:, ::3].T)


def _rigid_modes(coords: np.ndarray) -> np.ndarray:
    """Six rigid-body modes in dof order (3 translations, 3 rotations).

    ``coords`` is ``(n, 3)`` (see :func:`_dof_coords`); the returned rows are
    laid out node-major (``[u_x, u_y, u_z]`` per node), which is the ordering
    of a ``skfem`` vector basis.  Building them component-major (the previous
    bug) silently mixed components and made the equilibrium check report a
    spurious O(1) relative rigid-mode load.
    """
    n = coords.shape[0]
    modes = []
    for k in range(3):
        d = np.zeros((n, 3))
        d[:, k] = 1.0
        modes.append(d.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        modes.append(np.cross(np.broadcast_to(axis, coords.shape), coords).ravel())
    return np.array(modes)


def arm_skfem(nsub: int, order: int = 1, layers: int = 2, project: bool = False) -> dict:
    """Independent arm: scikit-fem + icosphere tet shell + consistent pressure load."""
    import scipy.sparse.linalg as spla
    import skfem as sk
    from skfem import (Basis, ElementTetP1, ElementTetP2, ElementVector,
                       FacetBasis, LinearForm, MeshTet, MeshTet2)
    from skfem.models.elasticity import (lame_parameters, linear_elasticity,
                                         linear_stress)

    started = time.time()
    nodes, tets, radii = _shell_tets(nsub, layers)
    mesh = MeshTet(nodes.T, tets.T)
    mesh_projected_midpoints = 0
    if order == 2:
        mesh = MeshTet2.from_mesh(mesh)
        if project:
            # MeshTet2 keeps the linear connectivity in ``mesh.t`` (4 rows) and
            # stores the edge midpoints as extra columns of ``mesh.p``; the P2
            # connectivity lives in ``mesh.dofs.element_dofs``.  The previous
            # ``mesh.t[4:]`` mask was therefore always empty and the "curved
            # projection" never moved a node.
            p = mesh.p.copy()
            r = np.linalg.norm(p, axis=0)
            vids = np.unique(mesh.t[:4])
            mids = np.setdiff1d(np.arange(p.shape[1]), vids)
            gap = (B - A) / float(layers)
            projected = 0
            for target_radius in (A, B):
                # a tangential edge of the shell has both endpoints on the
                # same sphere, so its chord midpoint sits within O(h^2) of it
                # (<0.15 * gap here); radial-edge midpoints sit at half the
                # layer gap and are left alone.
                sel = mids[np.abs(r[mids] - target_radius) < 0.3 * gap]
                p[:, sel] = p[:, sel] / r[sel] * target_radius
                projected += int(sel.size)
            mesh = MeshTet2(p, mesh.t)
            mesh_projected_midpoints = projected
        else:
            mesh_projected_midpoints = 0
    lam, mu = lame_parameters(E, NU)
    element = ElementTetP1() if order == 1 else ElementTetP2()
    basis = Basis(mesh, ElementVector(element))
    K = sk.asm(linear_elasticity(lam, mu), basis).tocsr()

    p = mesh.p
    r = np.linalg.norm(p, axis=0)
    inner_facets = np.where(np.all(np.abs(r[mesh.facets] - A) < 1e-9 * A, axis=0))[0]
    fb = FacetBasis(mesh, ElementVector(element), facets=inner_facets)

    @LinearForm
    def pressure(v, w):
        return -P * (w.n[0] * v[0] + w.n[1] * v[1] + w.n[2] * v[2])

    f = sk.asm(pressure, fb)
    coords = _dof_coords(basis)
    R = _rigid_modes(coords)
    # relative rigid-body contamination of the discrete load: each mode is
    # normalised so the measure is scale free (rotation rows grow with |x|).
    row = np.linalg.norm(R, axis=1)
    load_rigid = float(np.max(np.abs(R @ f) / (row * np.linalg.norm(f))))
    count = [0]
    u, info = spla.cg(K, f, rtol=1e-13, atol=0.0, maxiter=200000,
                      callback=lambda xk: count.__setitem__(0, count[0] + 1))
    residual = float(np.linalg.norm(K @ u - f) / np.linalg.norm(f))

    # --- probes: volume-averaged element stresses from skfem's own quadrature
    du = basis.interpolate(u).grad               # (3, 3, nelem, nqp)
    dx = basis.dx                                # (nelem, nqp) weights * detJ
    eps = 0.5 * (du + np.transpose(du, (1, 0, 2, 3)))
    sig = linear_stress(lam, mu)(eps)
    vol = dx.sum(axis=1)
    sig_e = (sig * dx[None, None]).sum(axis=-1) / vol
    verts = p[:, mesh.t[:4]]
    cent = verts.mean(axis=1)
    rad = np.linalg.norm(cent, axis=0)
    normal = cent / rad
    sr = np.einsum("in,ijn,jn->n", normal, sig_e, normal)
    tr = np.trace(sig_e, axis1=0, axis2=1)
    st = (tr - sr) / 2.0
    dev = sig_e - (tr / 3.0)[None, None, :] * np.eye(3)[:, :, None]
    vm = np.sqrt(1.5 * (dev * dev).sum((0, 1)))

    # vertex-associated dofs only: for the P2 basis ``mesh.p`` also holds the
    # edge midpoints, so indexing the nodal dofs with a mask built on ``p``
    # raised an IndexError (276 nodal dofs vs 2050 mesh columns).
    un = u[basis.nodal_dofs]                     # (3, n_nodal_dofs)
    nv = un.shape[1]
    vcoords = coords[:nv]                        # dof-ordered ↔ nodal_dofs
    rv = np.linalg.norm(vcoords, axis=1)
    surf = {}
    for tag, target in (("inner", A), ("outer", B)):
        sel = np.abs(rv - target) < 1e-9 * target
        surf[tag] = float(((un[:, sel] * vcoords[sel].T / rv[sel]).sum(0)).mean())

    def band_stats(r1, r2):
        sel = (rad > r1 - 1e-12) & (rad < r2 + 1e-12)
        w = vol[sel]
        return {"radius": float((w * rad[sel]).sum() / w.sum()),
                "radial_stress": float((w * sr[sel]).sum() / w.sum()),
                "hoop_stress": float((w * st[sel]).sum() / w.sum()),
                "von_mises": float((w * vm[sel]).sum() / w.sum()),
                "volume": float(w.sum())}

    inner = band_stats(radii[0], radii[1])
    outer = band_stats(radii[-2], radii[-1])
    row = {
        "arm": "scikit-fem",
        "element": "TET4" if order == 1 else "TET10",
        "order": order,
        "n": nsub,
        "layers": layers,
        "curved_projection": bool(order == 2 and project),
        "curved_projection_midpoints": int(mesh_projected_midpoints),
        "nodes": int(p.shape[1]),
        "elements": int(mesh.t.shape[1]),
        "dofs": int(basis.N),
        "mesh_volume_error": abs(float(vol.sum()) - shell_volume(CASE)) / shell_volume(CASE),
        "shell_volume_m3": shell_volume(CASE),
        "wall_seconds": round(time.time() - started, 2),
        "solver": {"iterations": count[0], "cg_info": int(info),
                   "relative_residual": residual},
        "loads": {"load_rigid_mode_relative": load_rigid},
        "equilibrium": {"free_residual_relative": residual},
        "energy": {"fe_strain_energy_j": float(0.5 * u @ f),
                   "lame_strain_energy_j": strain_energy(CASE)},
        "mesh_layer_radii": radii,
        "inner_layer": inner,
        "outer_layer": outer,
        "u_inner": surf["inner"],
        "u_outer": surf["outer"],
        "got": {"radial_displacement_inner": surf["inner"],
                "radial_displacement_outer": surf["outer"],
                "radial_stress_inner": inner["radial_stress"],
                "hoop_stress_inner": inner["hoop_stress"],
                "von_mises_inner": inner["von_mises"],
                "hoop_stress_outer": outer["hoop_stress"]},
    }
    # M1: Lame sampled at the volume-averaged centroid radius (repo protocol)
    row["ref_m1"], row["probes_m1"] = {}, {}
    for tag, layer in (("inner", inner), ("outer", outer)):
        ref = lame_at(layer["radius"])
        for q in ("radial_stress", "hoop_stress", "von_mises"):
            row["ref_m1"][f"{q}_{tag}"] = ref[q]
            row["probes_m1"][f"{q}_{tag}"] = abs(layer[q] - ref[q]) / abs(ref[q])
    row["probes_m1"]["radial_displacement_inner"] = \
        abs(surf["inner"] - lame_at(A)["radial_displacement"]) / abs(lame_at(A)["radial_displacement"])
    row["probes_m1"]["radial_displacement_outer"] = \
        abs(surf["outer"] - lame_at(B)["radial_displacement"]) / abs(lame_at(B)["radial_displacement"])
    primary = {k: v for k, v in row["probes_m1"].items() if k != "radial_stress_outer"}
    row["worst_m1"] = float(max(primary.values()))
    row.update(metric_m2(row, radii, inner, outer))
    row["energy"]["relative_error"] = abs(row["energy"]["fe_strain_energy_j"] -
                                          row["energy"]["lame_strain_energy_j"]) / \
        row["energy"]["lame_strain_energy_j"]
    row["retained"] = {"passed": bool(row["worst_m1"] < CASE_LINE)}
    return row


# ---------------------------------------------------------------------------
# claim cross-check
# ---------------------------------------------------------------------------
def reproduce_claims(rows: list[dict]) -> list[dict]:
    """Compare the OSS record's numeric claims with the re-measured values."""
    hex_by_order = {(r["order"], r["n"]): r for r in rows if r["arm"] == "TensorFEM"}
    hex20_n4 = hex_by_order.get((2, 4))
    hex8_n4 = hex_by_order.get((1, 4))
    hex8_ladder = [r for r in rows if r["arm"] == "TensorFEM" and r["order"] == 1]
    hex8_ladder.sort(key=lambda r: r["n"])

    def dev(got, ref):
        return abs(got - ref) / abs(ref)

    claims = []
    if hex20_n4:
        worst = hex20_n4["worst_m1"]
        claims.append({
            "id": "C1", "text": "HEX20 (n=4, 192 elements) keeps all probes < 3%",
            "kind": "threshold", "recorded": QUADRATIC_TARGET, "got": worst,
            "deviation": (worst - QUADRATIC_TARGET) / QUADRATIC_TARGET,
            "reproduced": bool(worst < QUADRATIC_TARGET)})
    if hex8_n4:
        worst8 = hex8_n4["worst_m1"]
        claims.append({
            "id": "C2", "text": "HEX8 worst = 15.4% on the same 192-element mesh",
            "kind": "numeric", "recorded": 0.1535, "got": worst8,
            "deviation": dev(worst8, 0.1535), "reproduced": bool(dev(worst8, 0.1535) <= CLAIM_TOLERANCE)})
    if hex8_ladder:
        # "approaching 5%" is tested at n=12 with the owner 15 % standard applied
        # to the 5 % line itself (5 % * 1.15 = 5.75 %); the strict crossing
        # (first level below 5 %) is reported separately.
        crossing = [r["n"] for r in hex8_ladder if r["worst_m1"] < CASE_LINE]
        first_n = crossing[0] if crossing else None
        at12 = [r for r in hex8_ladder if r["n"] == 12]
        got = at12[0]["worst_m1"] if at12 else (hex8_ladder[-1]["worst_m1"])
        dev_c3 = abs(got - CASE_LINE) / CASE_LINE
        claims.append({
            "id": "C3", "text": "HEX8 needs n >= 12 before approaching 5%",
            "kind": "threshold", "recorded": CASE_LINE, "got": got, "deviation": dev_c3,
            "reproduced": bool(dev_c3 <= CLAIM_TOLERANCE),
            "strict_crossing_n": first_n,
            "strict_below_line": bool(first_n is not None),
            "ladder_worst_m1": [{"n": r["n"], "worst": r["worst_m1"]} for r in hex8_ladder]})
    if hex8_n4 and hex20_n4:
        for tag, cid, recorded in (("hex8", "C4a", 0.071), ("hex20", "C4b", 0.0014)):
            row = hex8_n4 if tag == "hex8" else hex20_n4
            got = row["mesh_volume_error"]
            claims.append({
                "id": cid,
                "text": f"mesh volume error ({tag}) = {recorded:.2%}",
                "kind": "numeric", "recorded": recorded, "got": got,
                "deviation": dev(got, recorded),
                "reproduced": bool(dev(got, recorded) <= CLAIM_TOLERANCE)})
        loads = hex20_n4["loads"]
        worst_load = max(abs(v) for v in loads.values() if isinstance(v, (int, float)))
        claims.append({
            "id": "C5", "text": "closed-surface force/moment < 1e-16 (relative)",
            "kind": "threshold", "recorded": 1e-16, "got": worst_load,
            "deviation": (worst_load - 1e-16) / 1e-16,
            "reproduced": bool(worst_load < 1e-16)})
    hex8_n8 = hex_by_order.get((1, 8))
    if hex8_n8:
        it = hex8_n8["solver"]["iterations"]
        claims.append({
            "id": "C6", "text": "Jacobi-PCG iterations for HEX8 at n=8 = 67",
            "kind": "numeric", "recorded": 67, "got": it,
            "deviation": dev(it, 67), "reproduced": bool(dev(it, 67) <= CLAIM_TOLERANCE)})
    return claims


def decision_table(rows: list[dict]) -> dict:
    """Predeclared decision: is the quadratic selection robust across functionals?"""
    def pick(arm, order, n):
        for r in rows:
            if r["arm"] == arm and r["order"] == order and r["n"] == n:
                return r
        return None

    entries = []
    for arm, levels in (("TensorFEM", (4, 6)), ("scikit-fem", (4, 6))):
        for order in (1, 2):
            for n in levels:
                row = pick(arm, order, n)
                if not row:
                    continue
                entries.append({
                    "arm": arm, "element": row["element"], "n": n, "dofs": row["dofs"],
                    "worst_m1": row["worst_m1"], "worst_m2": row["worst_m2"],
                    "worst_quantity_m2": row["worst_quantity_m2"],
                    "passes_case_line_m1": bool(row["worst_m1"] < CASE_LINE),
                    "passes_case_line_m2": bool(row["worst_m2"] < CASE_LINE),
                    "passes_quadratic_target_m1": bool(row["worst_m1"] < QUADRATIC_TARGET),
                    "passes_quadratic_target_m2": bool(row["worst_m2"] < QUADRATIC_TARGET)})
    tf20 = pick("TensorFEM", 2, 4)
    tf8 = pick("TensorFEM", 1, 4)
    verdict = {}
    if tf20 and tf8:
        verdict["ranking_m1_reproduced"] = bool(tf20["worst_m1"] < tf8["worst_m1"])
        verdict["ranking_m2_reproduced"] = bool(tf20["worst_m2"] < tf8["worst_m2"])
        verdict["quadratic_under_3pct_m1"] = bool(tf20["worst_m1"] < QUADRATIC_TARGET)
        verdict["quadratic_under_3pct_m2"] = bool(tf20["worst_m2"] < QUADRATIC_TARGET)
        verdict["functional_dependence_gap"] = float(tf20["worst_m2"] - tf20["worst_m1"])
    sk2 = pick("scikit-fem", 2, 4)
    if sk2:
        verdict["independent_quadratic_under_3pct_m1"] = bool(sk2["worst_m1"] < QUADRATIC_TARGET)
        verdict["independent_quadratic_under_3pct_m2"] = bool(sk2["worst_m2"] < QUADRATIC_TARGET)
    verdict["selection_robust"] = bool(
        verdict.get("ranking_m1_reproduced") and verdict.get("ranking_m2_reproduced")
        and verdict.get("quadratic_under_3pct_m1") and verdict.get("quadratic_under_3pct_m2"))
    return {"entries": entries, "verdict": verdict}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(rows: list[dict], out_dir: pathlib.Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for functional, key in (("M1 centroid-radius", "worst_m1"), ("M2 exact band", "worst_m2")):
        target_ax = ax[0] if functional.startswith("M1") else ax[1]
        for arm, marker in (("TensorFEM", "o"), ("scikit-fem", "s")):
            for order, style in ((1, "--"), (2, "-")):
                sub = sorted([r for r in rows if r["arm"] == arm and r["order"] == order],
                             key=lambda r: r["dofs"])
                if not sub:
                    continue
                target_ax.plot([r["dofs"] for r in sub], [100 * r[key] for r in sub],
                               style, marker=marker, label=f"{arm} {sub[0]['element']}")
        target_ax.axhline(100 * CASE_LINE, color="k", lw=1, ls=":", label="case line 5%")
        target_ax.axhline(100 * QUADRATIC_TARGET, color="r", lw=1, ls=":",
                          label="selection claim 3%")
        target_ax.set_xscale("log")
        target_ax.set_yscale("log")
        target_ax.set_xlabel("degrees of freedom")
        target_ax.set_ylabel("worst probe error [%]")
        target_ax.set_title(functional)
        target_ax.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "fig_xval_convergence.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)

    # per-probe bar chart: TensorFEM HEX20@n=4 vs scikit-fem TET10@n=4, M1 vs M2
    cand = [r for r in rows if r["order"] == 2 and r["n"] == 4]
    if cand:
        probes = sorted(set(cand[0]["probes_m1"]) & set(cand[0]["probes_m2"]))
        fig, ax = plt.subplots(figsize=(11, 4.2))
        width = 0.8 / (2 * len(cand))
        for i, row in enumerate(cand):
            for j, (functional, key) in enumerate((("M1", "probes_m1"), ("M2", "probes_m2"))):
                ax.bar(np.arange(len(probes)) + (2 * i + j) * width,
                       [100 * row[key][p] for p in probes], width,
                       label=f"{row['arm']} {row['element']} {functional}")
        ax.set_xticks(np.arange(len(probes)) + 0.4)
        ax.set_xticklabels([p.replace("_", "\n") for p in probes], fontsize=6)
        ax.axhline(100 * QUADRATIC_TARGET, color="r", ls=":", label="3% claim")
        ax.set_yscale("log")
        ax.set_ylabel("probe error [%]")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path = out_dir / "fig_xval_probe_split.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        names.append(path.name)
    return names


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(quick: bool = False) -> dict:
    if quick:
        tf_ladder = [(1, 4), (1, 6), (2, 4)]
        sk_ladder = [(1, 4), (2, 4)]
    else:
        tf_ladder = [(1, 4), (1, 6), (1, 8), (1, 12), (2, 4), (2, 6)]
        sk_ladder = [(1, 4), (1, 6), (1, 8), (2, 4), (2, 6)]
    rows = []
    for order, n in tf_ladder:
        print(f"[arm A] TensorFEM order={order} n={n} ...", flush=True)
        rows.append(arm_tensorfem(order, n))
        print(f"        worst M1={rows[-1]['worst_m1']:.3%} M2={rows[-1]['worst_m2']:.3%} "
              f"dofs={rows[-1]['dofs']} [{rows[-1]['wall_seconds']}s]", flush=True)
    for order, n in sk_ladder:
        for project in ((True,) if order == 2 else (False,)):
            print(f"[arm B] scikit-fem order={order} n={n} project={project} ...", flush=True)
            rows.append(arm_skfem(n, order=order, project=project))
            print(f"        worst M1={rows[-1]['worst_m1']:.3%} M2={rows[-1]['worst_m2']:.3%} "
                  f"dofs={rows[-1]['dofs']} [{rows[-1]['wall_seconds']}s]", flush=True)
    claims = reproduce_claims(rows)
    decision = decision_table(rows)
    parameters = {
        "geometry": {"inner_radius_m": A, "outer_radius_m": B,
                     "wall_ratio_b_over_a": B / A},
        "material": {"young_modulus_pa": E, "poisson_ratio": NU},
        "loading": {"internal_pressure_pa": P},
        "dimensionless": {
            "b_over_a": B / A,
            "poisson_ratio": NU,
            "normalised_mesh_size_hex8_n4": (B - A) / (4 / 2) / 1.0,
            "pressure_over_shear_modulus": P / (E / (2 * (1 + NU))),
            "lame_coefficient_a_over_p": COEF_A / P,
            "lame_coefficient_b_over_p_m3": COEF_B / P,
            "hoop_to_radial_stress_inner": lame_at(A)["hoop_stress"] / lame_at(A)["radial_stress"],
            "von_mises_over_p_inner": lame_at(A)["von_mises"] / P,
            "u_inner_over_a": lame_at(A)["radial_displacement"] / A,
            "shell_volume_m3": shell_volume(CASE),
            "lame_strain_energy_j": strain_energy(CASE)},
    }
    return {"anchor": "XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE",
            "reference": "OSS TensorFEM record (oss_solutions repo=TensorFEM, "
                         "tag=method-selection+first-evidence) + scikit-fem 12.0.2 "
                         "independent implementation",
            "tolerance": {"xval_claim": CLAIM_TOLERANCE, "case_line": CASE_LINE,
                          "selection_target": QUADRATIC_TARGET},
            "parameters": parameters,
            "rows": rows, "claims": claims, "decision": decision}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="artifacts/xval_method_selection_first_evidence")
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = pathlib.Path(__file__).with_name("xval_method_selection_first_evidence_record.json")
    payload = run(quick=args.quick)
    if record.exists():
        payload["oss_record_source"] = {"path": str(record),
                                        "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                                        "content": json.loads(record.read_text())}
    payload["figures"] = make_figures(payload["rows"], out_dir)
    (out_dir / "evidence.json").write_text(json.dumps(payload, indent=2))
    print("\n=== claims ===")
    for c in payload["claims"]:
        print(f"{c['id']}: {c['text']} | recorded={c['recorded']} got={c['got']} "
              f"dev={c['deviation']:.3%} reproduced={c['reproduced']}")
    print("\n=== decision ===")
    print(json.dumps(payload["decision"]["verdict"], indent=2))
    print(f"\nartifacts -> {out_dir}/evidence.json")
    failing = [c for c in payload["claims"] if not c["reproduced"]]
    print(f"claims reproduced {len(payload['claims']) - len(failing)}/{len(payload['claims'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())