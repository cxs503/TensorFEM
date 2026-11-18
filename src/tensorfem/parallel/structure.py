"""Plane-stress Q4 cantilever with serial and non-overlapping Schur solves."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import multiprocessing
import os
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve

from ..cohesive_bending import _q4_stiffness, plane_stress_matrix


def _strip(args):
    start, stop, nx, ny, condense = args
    x = np.arange(start, stop + 1) * (4.0 / nx)
    y = np.arange(ny + 1) / ny
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    column = ny + 1
    node = np.arange(len(xy)).reshape(-1, column)[:-1, :-1].ravel()
    cells = node[:, None] + np.array([0, column, column + 1, 1])
    dofs = (2 * cells[:, :, None] + np.arange(2)).reshape(-1, 8)
    material = plane_stress_matrix(7e9, 0.3)
    ke = _q4_stiffness(xy[cells], material, 0.1)
    k = coo_matrix((ke.ravel(), (np.repeat(dofs, 8, axis=1).ravel(),
                                 np.tile(dofs, (1, 8)).ravel())),
                   shape=(2 * len(xy),) * 2).tocsr()
    if not condense:
        return k
    width = 2 * column
    edge = np.r_[np.arange(width), np.arange(k.shape[0] - width, k.shape[0])]
    inner = np.arange(width, k.shape[0] - width)
    extension = np.zeros((len(inner), 2 * width))
    if len(inner):
        extension = -splu(k[inner][:, inner].tocsc()).solve(k[inner][:, edge].toarray())
    schur = k[edge][:, edge].toarray() + k[edge][:, inner] @ extension
    return schur, extension, os.getpid()


def solve_panel(nx=32, ny=8, method="sparse", workers=2):
    """Clamp the left edge and apply a 1 kN downward resultant on the right.

    Length 4 m, depth 1 m, thickness 0.1 m. All paths use identical quadrature,
    consistent edge tractions, and element-centre Cauchy stresses (Pa).
    """
    if nx < 1 or ny < 1 or method not in {"dense", "sparse", "parallel"}:
        raise ValueError("positive cell counts and dense, sparse or parallel method required")
    if method == "parallel" and not 1 <= workers <= nx:
        raise ValueError("workers must be between one and the number of columns")
    tick = perf_counter()
    width = 2 * (ny + 1)
    f = np.zeros((nx + 1) * width)
    f[-width+1::2] = -1000.0 / ny
    f[[-width+1, -1]] *= 0.5
    u = np.zeros_like(f)
    pids = [os.getpid()]
    if method == "parallel":
        cuts = np.linspace(0, nx, workers + 1, dtype=int)
        jobs = [(a, b, nx, ny, True) for a, b in zip(cuts[:-1], cuts[1:])]
        # One dedicated process per strip makes the partition count observable,
        # including when one small strip finishes before another interpreter starts.
        context = multiprocessing.get_context("spawn")
        with ExitStack() as stack:
            executors = [stack.enter_context(ProcessPoolExecutor(1, mp_context=context))
                         for _ in jobs]
            futures = [executor.submit(_strip, job) for executor, job in zip(executors, jobs)]
            strips = [future.result() for future in futures]
        interface = np.zeros(((workers + 1) * width,) * 2)
        for i, (schur, _, _) in enumerate(strips):
            span = slice(i * width, (i + 2) * width)
            interface[span, span] += schur
        assembled = perf_counter()
        interface_u = np.zeros(len(interface))
        rhs = np.zeros_like(interface_u)
        rhs[-width:] = f[-width:]
        interface_u[width:] = np.linalg.solve(interface[width:, width:], rhs[width:])
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            edge = interface_u[i * width:(i + 2) * width]
            u[a*width:(a+1)*width] = edge[:width]
            u[b*width:(b+1)*width] = edge[width:]
            u[(a+1)*width:b*width] = strips[i][1] @ edge
        reaction = (interface @ interface_u - rhs)[:width]
        pids = [part[2] for part in strips]
    else:
        k = _strip((0, nx, nx, ny, False))
        if method == "dense":
            k = k.toarray()
        assembled = perf_counter()
        solver = np.linalg.solve if method == "dense" else spsolve
        u[width:] = solver(k[width:, width:], f[width:])
        reaction = (k @ u - f)[:width]
    solved = perf_counter()
    grid = u.reshape(nx + 1, ny + 1, 2)
    du_dx = ((grid[1:, 1:] + grid[1:, :-1]) - (grid[:-1, 1:] + grid[:-1, :-1])) * nx / 8
    du_dy = ((grid[1:, 1:] + grid[:-1, 1:]) - (grid[1:, :-1] + grid[:-1, :-1])) * ny / 2
    strain = np.stack((du_dx[:, :, 0], du_dy[:, :, 1], du_dx[:, :, 1]+du_dy[:, :, 0]), axis=-1)
    stress = strain @ plane_stress_matrix(7e9, 0.3).T
    # Compute internal energy from element strains at all four Gauss points
    # via element forces, independently of the work of the external load.
    ex = np.array([[0., 0.], [4/nx, 0.], [4/nx, 1/ny], [0., 1/ny]])
    ke = _q4_stiffness(ex[None], plane_stress_matrix(7e9, 0.3), 0.1)[0]
    local = np.stack((grid[:-1,:-1], grid[1:,:-1], grid[1:,1:], grid[:-1,1:]),axis=2).reshape(-1,8)
    return dict(displacement=u, reaction=reaction, stress=stress, force=f,
                energy=float(0.5 * np.sum((local @ ke) * local)), work=float(0.5 * f @ u),
                assembly_s=assembled-tick, solve_s=solved-assembled, elapsed_s=perf_counter()-tick,
                worker_pids=pids)
