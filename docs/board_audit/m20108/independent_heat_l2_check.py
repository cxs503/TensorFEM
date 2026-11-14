#!/usr/bin/env python3
"""Independent cross-check for CASE-FEM-HEAT-L2 (m20054 board audit, m20108).

This script deliberately does NOT import anything from
``tensorfem.heat_l2_convergence`` (the module under audit).  It re-derives the
L2/H1 convergence orders of a bilinear (Q4) steady-conduction discretisation on
the manufactured sine field in two ways:

  A) a from-scratch numpy Q4 FE (own assembly, own 8x8 Gauss element-mean
     source, own 4x4 Gauss error norms) -- full independence from the audited
     code path;
  B) the shipped kernel ``tensorfem.thermal.solve_steady_thermal`` driven by
     *this* script's mesh / source / error norm (shares only the kernel with
     m20054, not the convergence harness).

MMS: T = sin(pi x) sin(pi y), -laplace T = q  =>  q = 2 pi^2 T,
Dirichlet T = T_exact on all four edges.

Run:  PYTHONPATH=src python3 docs/board_audit/m20108/independent_heat_l2_check.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PI = math.pi
LEVELS = (8, 16, 32, 64)


# ---------------------------------------------------------------- exact field
def t_exact(x, y):
    return np.sin(PI * x) * np.sin(PI * y)


def grad_exact(x, y):
    return (PI * np.cos(PI * x) * np.sin(PI * y),
            PI * np.sin(PI * x) * np.cos(PI * y))


def q_exact(x, y):
    return 2.0 * PI * PI * t_exact(x, y)


def gauss_legendre_1d(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def q4_shape(xi, eta):
    return 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                            (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])


def q4_dshape(xi, eta):
    return 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])


# --------------------------------------------------------------------- mesh
def make_mesh(n, lx=1.0, ly=1.0):
    xs = np.linspace(0.0, lx, n + 1)
    ys = np.linspace(0.0, ly, n + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # x fastest, CCW elements
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    elements = []
    for j in range(n):
        for i in range(n):
            a = i + j * (n + 1)
            b = a + 1
            c = a + (n + 1) + 1
            d = a + (n + 1)
            elements.append([a, b, c, d])
    return nodes, np.asarray(elements, dtype=int), (lx / n, ly / n)


# ------------------------------------------------------------- assembly (A)
def assemble(nodes, elements, source_kind):
    ne = len(elements)
    nn = len(nodes)
    K = np.zeros((nn, nn))
    f = np.zeros(nn)
    gx, gw = gauss_legendre_1d(2)
    sx, sw = gauss_legendre_1d(8)
    for e in range(ne):
        conn = elements[e]
        xe = nodes[conn]
        Ke = np.zeros((4, 4))
        for a_, wa in zip(gx, gw):
            for b_, wb in zip(gx, gw):
                dN = q4_dshape(a_, b_)
                J = dN @ xe                       # 2x2, det > 0 for CCW
                detJ = np.linalg.det(J)
                if detJ <= 0.0:
                    raise ValueError("non-positive Jacobian in own assembler")
                dNdx = np.linalg.solve(J, dN)
                Ke += wa * wb * (dNdx.T @ dNdx) * detJ
        K[np.ix_(conn, conn)] += Ke
        area = abs(np.linalg.det(q4_dshape(0.0, 0.0) @ xe)) * 4.0
        if source_kind == "element_mean":
            # consistent load for an element-constant source q_c
            qc = 0.0
            for a_, wa in zip(sx, sw):
                for b_, wb in zip(sx, sw):
                    dN = q4_dshape(a_, b_)
                    detJ = np.linalg.det(dN @ xe)
                    x = q4_shape(a_, b_) @ xe
                    qc += wa * wb * q_exact(x[0], x[1]) * detJ
            qc /= area
            f[conn] += qc * area / 4.0
        elif source_kind == "exact_load":
            for a_, wa in zip(sx, sw):
                for b_, wb in zip(sx, sw):
                    dN = q4_dshape(a_, b_)
                    detJ = np.linalg.det(dN @ xe)
                    N = q4_shape(a_, b_)
                    x = N @ xe
                    f[conn] += wa * wb * N * q_exact(x[0], x[1]) * detJ
        else:
            raise ValueError(source_kind)
    return K, f


# ------------------------------------------------------------------- errors
def error_norms(nodes, elements, T):
    gx, gw = gauss_legendre_1d(4)
    e0 = 0.0
    e1 = 0.0
    n0 = 0.0
    n1 = 0.0
    for conn in elements:
        xe = nodes[conn]
        Te = T[conn]
        for a_, wa in zip(gx, gw):
            for b_, wb in zip(gx, gw):
                N = q4_shape(a_, b_)
                dN = q4_dshape(a_, b_)
                detJ = np.linalg.det(dN @ xe)
                dNdx = np.linalg.solve(dN @ xe, dN)
                x = N @ xe
                th = float(N @ Te)
                gh = Te @ dNdx.T
                gt = np.array(grad_exact(x[0], x[1]))
                tt = t_exact(x[0], x[1])
                e0 += wa * wb * (th - tt) ** 2 * detJ
                n0 += wa * wb * tt ** 2 * detJ
                e1 += wa * wb * float(np.sum((gh - gt) ** 2)) * detJ
                n1 += wa * wb * float(np.sum(gt ** 2)) * detJ
    return math.sqrt(e0 / n0), math.sqrt(e1 / n1)


def boundary_dofs(nodes, h):
    x, y = nodes[:, 0], nodes[:, 1]
    tol = 1e-12 + h * 1e-9
    return np.where((x < tol) | (x > 1 - tol) | (y < tol) | (y > 1 - tol))[0]


def solve_own(n, source_kind):
    nodes, elements, (hx, hy) = make_mesh(n)
    K, f = assemble(nodes, elements, source_kind)
    fixed = boundary_dofs(nodes, max(hx, hy))
    free = np.setdiff1d(np.arange(len(nodes)), fixed)
    T = np.zeros(len(nodes))
    T[fixed] = t_exact(nodes[fixed, 0], nodes[fixed, 1])
    T[free] = np.linalg.solve(K[np.ix_(free, free)], f[free] - K[np.ix_(free, fixed)] @ T[fixed])
    return nodes, elements, T, hx


def solve_kernel(n, torch):
    """Same problem through the *shipped* tensorfem.thermal kernel."""
    from tensorfem.thermal import ThermalModel, solve_steady_thermal

    nodes_np, elements, (hx, hy) = make_mesh(n)
    ne = len(elements)
    sx, sw = gauss_legendre_1d(8)
    qc = np.zeros(ne)
    for e, conn in enumerate(elements):
        xe = nodes_np[conn]
        acc = 0.0
        area = 0.0
        for a_, wa in zip(sx, sw):
            for b_, wb in zip(sx, sw):
                dN = q4_dshape(a_, b_)
                detJ = np.linalg.det(dN @ xe)
                x = q4_shape(a_, b_) @ xe
                acc += wa * wb * q_exact(x[0], x[1]) * detJ
                area += wa * wb * detJ
        qc[e] = acc / area
    fixed = boundary_dofs(nodes_np, max(hx, hy))
    model = ThermalModel(
        nodes=torch.as_tensor(nodes_np, dtype=torch.float64),
        elements=torch.as_tensor(elements, dtype=torch.long),
        conductivity=torch.ones(ne, dtype=torch.float64),
        density=torch.ones(ne, dtype=torch.float64),
        specific_heat=torch.ones(ne, dtype=torch.float64),
        fixed_nodes=torch.as_tensor(fixed, dtype=torch.long),
        fixed_temperatures=torch.as_tensor(t_exact(nodes_np[fixed, 0], nodes_np[fixed, 1]),
                                          dtype=torch.float64),
        heat_source=torch.as_tensor(qc, dtype=torch.float64),
        element_type="q4",
    )
    T = solve_steady_thermal(model).detach().numpy()
    return nodes_np, elements, T, hx


def orders_from(levels, errs, hs):
    return [math.log(errs[i] / errs[i + 1]) / math.log(hs[i] / hs[i + 1])
            for i in range(len(errs) - 1)]


def main() -> int:
    torch = None
    try:
        import torch  # noqa: F811
        torch.set_num_threads(4)
    except Exception as exc:  # pragma: no cover
        print(f"torch unavailable, kernel check skipped: {exc}")

    report = {"schema": "m20108-independent-heat-l2-crosscheck-v1",
              "reference": "MMS T = sin(pi x) sin(pi y), q = 2 pi^2 T",
              "levels": list(LEVELS), "variants": {}}

    for label, runner in (
        ("own_element_mean_source", lambda n: solve_own(n, "element_mean")),
        ("own_exact_load", lambda n: solve_own(n, "exact_load")),
        ("shipped_kernel", (lambda n: solve_kernel(n, torch)) if torch is not None else None),
    ):
        if runner is None:
            continue
        t0 = time.time()
        errs2, errs1, hs = [], [], []
        for n in LEVELS:
            nodes, elements, T, hx = runner(n)
            e0, e1 = error_norms(nodes, elements, T)
            errs2.append(e0)
            errs1.append(e1)
            hs.append(hx)
        o2 = orders_from(LEVELS, errs2, hs)
        o1 = orders_from(LEVELS, errs1, hs)
        report["variants"][label] = {
            "h": hs, "l2_relative": errs2, "h1_relative": errs1,
            "l2_orders": o2, "h1_orders": o1,
            "l2_finest_order": o2[-1], "l2_min_order_coarse_ge_8": min(o2[1:]),
            "runtime_s": round(time.time() - t0, 2),
        }
        print(f"{label:24s} L2 orders: " + " ".join(f"{v:.4f}" for v in o2)
              + f" | finest {o2[-1]:.4f} | min(coarse>=8) {min(o2[1:]):.4f}")
        print(f"{'':24s} H1 orders: " + " ".join(f"{v:.4f}" for v in o1))

    # criterion: L2 order >= 2 (+/- gate band used by the audited case)
    band = (1.95, 2.10)
    checks = []
    for label, v in report["variants"].items():
        checks.append({"name": f"{label}:l2_finest_in_band",
                       "passed": band[0] <= v["l2_finest_order"] <= band[1],
                       "detail": f"finest-pair L2 order {v['l2_finest_order']:.4f} in {band}"})
        checks.append({"name": f"{label}:l2_pair_ge_1.90",
                       "passed": v["l2_min_order_coarse_ge_8"] >= 1.90,
                       "detail": f"min L2 order for coarse nx>=8 is "
                                 f"{v['l2_min_order_coarse_ge_8']:.4f} (gate >=1.90)"})
    report["checks"] = checks
    report["passed"] = all(c["passed"] for c in checks)
    print(f"\nindependent verdict: {'PASS' if report['passed'] else 'FAIL'} "
          f"({sum(c['passed'] for c in checks)}/{len(checks)} checks)")

    out = Path("docs/board_audit/m20108/independent_heat_l2_check.json")
    out.write_text(json.dumps(report, indent=1))
    print(f"report -> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())