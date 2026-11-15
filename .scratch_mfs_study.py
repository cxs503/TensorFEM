"""Scratch: MFS conditioning study for the doubled rectangle (internal use, fast)."""
import math
import numpy as np

RHO = 1025.0
C, H = 10.0, 6.0


def boundary_rect(c, h, n):
    s = np.cos(np.linspace(0.0, math.pi, n, endpoint=False))
    t = 0.5 * (1.0 + s)
    xr, zr = -c + 2 * c * t, -h + 2 * h * t
    pts, nrm = [], []
    for xs, (nx, nz) in ((xr, (0, 1)), (zr, (1, 0)), (-xr, (0, -1)), (-zr, (-1, 0))):
        if nx == 0:
            pts.append(np.stack([xs, np.full_like(xs, nz * h)], axis=1))
            nrm.append(np.stack([np.zeros_like(xs), np.full_like(xs, float(nz))], axis=1))
        else:
            pts.append(np.stack([np.full_like(xs, nx * c), xs], axis=1))
            nrm.append(np.stack([np.full_like(xs, float(nx)), np.zeros_like(xs)], axis=1))
    return np.concatenate(pts), np.concatenate(nrm)


def solve(b, nb, s, lam):
    seg = np.linalg.norm(np.roll(b, -1, axis=0) - b, axis=1)
    w = 0.5 * (seg + np.roll(seg, 1))
    dx = b[:, None, 0] - s[None, :, 0]
    dz = b[:, None, 1] - s[None, :, 1]
    r2 = dx ** 2 + dz ** 2
    dln = (dx * nb[:, None, 0] + dz * nb[:, None, 1]) / r2
    lnr = 0.5 * np.log(r2)
    ns = s.shape[0]
    A = np.vstack([dln, np.ones((1, ns))])
    bb = np.concatenate([nb[:, 1], [0.0]])
    rw = np.linalg.norm(A, axis=1, keepdims=True)
    As = A / np.maximum(rw, 1e-30)
    bs = bb / np.maximum(rw[:, 0], 1e-30)
    q = np.linalg.solve(As.T @ As + lam * np.eye(ns), As.T @ bs)
    val = RHO * float(np.sum((lnr @ q) * nb[:, 1] * w))
    return val, float(np.max(np.abs(A @ q - bb))), float(np.linalg.cond(As))


def uniq(p):
    _, idx = np.unique(np.round(p, 9), axis=0, return_index=True)
    return p[np.sort(idx)]


for shrink in (0.5, 0.8, 0.95):
    for n in (60, 120, 240):
        b, nb = boundary_rect(C, H, n)
        s = uniq(boundary_rect(shrink * C, shrink * H, n)[0])
        for lam in (0.0, 1e-4):
            v, r, c = solve(b, nb, s, lam)
            print(f"shrink={shrink} n={n:4d} lam={lam:g} A_half={v/2/1e3:9.3f} t/m resid={r:.2e} cond={c:.1e}", flush=True)