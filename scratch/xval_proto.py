"""Prototype 2: independent scikit-fem hollow-sphere (icosphere tet shell)."""
import numpy as np
import scipy.sparse.linalg as spla
import skfem as _sk0
from skfem import (Basis, ElementTetP1, ElementTetP2, ElementVector, FacetBasis,
                   LinearForm, MeshTet)
from skfem.models.elasticity import lame_parameters, linear_elasticity, linear_stress

A_, B_, E_, NU_, P_ = 8.0, 10.0, 210e9, 0.3, 1e6
LAM, MU = lame_parameters(E_, NU_)


def lame(r):
    den = B_ ** 3 - A_ ** 3
    Aa = P_ * A_ ** 3 / den
    Bb = Aa * B_ ** 3
    sr = Aa - Bb / r ** 3
    st = Aa + Bb / (2 * r ** 3)
    ur = ((1 - 2 * NU_) * Aa * r + (1 + NU_) * Bb / (2 * r ** 2)) / E_
    return Aa, Bb, ur, sr, st


AA, BB, _, _, _ = lame(np.array([A_]))


def band_average(r1, r2):
    """Exact volume averages of sigma_r, sigma_theta, von Mises over r1..r2."""
    v = (r2 ** 3 - r1 ** 3) / 3.0
    sr = AA - 3 * BB * (r2 - r1) / (r2 ** 3 - r1 ** 3)
    st = AA + 1.5 * BB * (r2 - r1) / (r2 ** 3 - r1 ** 3)
    vm = 4.5 * BB * np.log(r2 / r1) / (r2 ** 3 - r1 ** 3)
    return sr, st, vm


def icosphere(nsub):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
             (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = np.array(verts, float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts = []
    for f in faces:
        p0, p1, p2 = v[list(f)]
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


def shell_mesh(nsub, layers):
    dirs, tris = icosphere(nsub)
    radii = np.linspace(A_, B_, layers + 1)
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


def rigid_modes(p):
    n = p.shape[1]
    R = []
    for k in range(3):
        d = np.zeros((3, n))
        d[k] = 1.0
        R.append(d.ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        R.append(np.cross(np.broadcast_to(axis[:, None], p.shape).T, p.T).T.ravel())
    return np.array(R)


def solve(nsub, layers=2, order=1, project=True):
    nodes, tets, radii = shell_mesh(nsub, layers)
    mesh = MeshTet(nodes.T, tets.T)
    element = ElementTetP1() if order == 1 else ElementTetP2()
    if order == 2:
        p = mesh.p.copy()
        r = np.linalg.norm(p, axis=0)
        mid = np.zeros(p.shape[1], dtype=bool)
        mid[mesh.t[4:]] = True
        tgt = np.where(np.abs(r - A_) < 1e-9 * A_, A_,
                       np.where(np.abs(r - B_) < 1e-9 * B_, B_, 0.0))
        sel = mid & (tgt > 0) if project else np.zeros_like(mid)
        p[:, sel] = p[:, sel] / r[sel] * tgt[sel]
        mesh = MeshTet(p, tets.T)
    el = ElementVector(element)
    basis = Basis(mesh, el)
    import skfem as _sk; K = _sk.asm(linear_elasticity(LAM, MU), basis).tocsr()
    p = mesh.p
    r = np.linalg.norm(p, axis=0)
    inner = np.where(np.all(np.abs(r[mesh.facets] - A_) < 1e-9 * A_, axis=0))[0]
    fb = FacetBasis(mesh, el, facets=inner)

    @LinearForm
    def press(v, w):
        return -P_ * (w.n[0] * v[0] + w.n[1] * v[1] + w.n[2] * v[2])

    f = _sk.asm(press, fb)
    R = rigid_modes(p)
    stats = {"load_rigid_relative": float(np.linalg.norm(R @ f) / np.linalg.norm(f))}
    count = [0]
    u, info = spla.cg(K, f, rtol=1e-13, atol=0.0, maxiter=200000,
                      callback=lambda xk: count.__setitem__(0, count[0] + 1))
    stats.update(cg_iterations=count[0], cg_info=int(info),
                 equilibrium_relative=float(np.linalg.norm(K @ u - f) / np.linalg.norm(f)),
                 rigid_leak=float(np.linalg.norm(R @ u)))
    return mesh, basis, u, radii, stats


def probes(mesh, basis, u, order):
    from skfem.models.elasticity import sym_grad
    du = basis.interpolate(u).grad          # (3,3,nelem,nqp)
    dx = basis.dx                            # (nqp, nelem) weights * detJ
    eps = 0.5 * (du + np.transpose(du, (1, 0, 2, 3)))
    sig = linear_stress(LAM, MU)(eps)        # (3,3,nelem,nqp)
    vol = dx.sum(axis=1)
    sig_e = (sig * dx[None, None]).sum(axis=-1) / vol    # volume average per element
    verts = mesh.p[:, mesh.t[:4]]
    cent = verts.mean(axis=1)
    rad = np.linalg.norm(cent, axis=0)
    n = cent / rad
    sr = np.einsum("in,ijn,jn->n", n, sig_e, n)
    tr = np.trace(sig_e, axis1=0, axis2=1)
    st = (tr - sr) / 2
    dev = sig_e - np.trace(sig_e, axis1=0, axis2=1)[None, None, :] / 3 * np.eye(3)[:, :, None]
    vm = np.sqrt(1.5 * (dev * dev).sum((0, 1)))
    # nodal radial displacement on exact surfaces
    p = mesh.p
    rr = np.linalg.norm(p, axis=0)
    un = u[basis.nodal_dofs]
    out = {}
    for tag, target in (("inner", A_), ("outer", B_)):
        sel = np.abs(rr - target) < 1e-9 * target
        ur = (un[:, sel] * p[:, sel] / rr[sel]).sum(0)
        out[f"u_{tag}"] = float(ur.mean())
    return {"sr": sr, "st": st, "vm": vm, "rad": rad, "vol": vol, "cent": cent, "mesh_vol": float(vol.sum())}


def layer_stats(pr, r1, r2):
    sel = (pr["rad"] > r1 - 1e-12) & (pr["rad"] < r2 + 1e-12)
    w = pr["vol"][sel]
    return dict(radius=float((w * pr["rad"][sel]).sum() / w.sum()),
                radial_stress=float((w * pr["sr"][sel]).sum() / w.sum()),
                hoop_stress=float((w * pr["st"][sel]).sum() / w.sum()),
                von_mises=float((w * pr["vm"][sel]).sum() / w.sum()),
                band_volume=float(w.sum()))


if __name__ == "__main__":
    import time
    for order in (1, 2):
        for nsub in (4, 6):
            t0 = time.time()
            mesh, basis, u, radii, stats = solve(nsub, 2, order, project=True)
            pr = probes(mesh, basis, u, order)
            inner = layer_stats(pr, radii[0], radii[1])
            outer = layer_stats(pr, radii[1], radii[2])
            ex_in = band_average(radii[0], radii[1])
            ex_out = band_average(radii[1], radii[2])
            ur_a = lame(np.array([A_]))[2][0]
            ur_b = lame(np.array([B_]))[2][0]
            print(f"order={order} n={nsub} ndof={basis.N} elems={mesh.t.shape[1]} "
                  f"vol_err={abs(pr['mesh_vol']-2044.130)/2044.130:.4%} "
                  f"sr_in={inner['radial_stress']:.0f} (exact {ex_in[0]:.0f}) "
                  f"sr_out={outer['radial_stress']:.0f} (exact {ex_out[0]:.0f}) "
                  f"u_a={pr and pr['u_inner'] if False else 0:.3e}")
            print(f"   vol_err={abs(pr['mesh_vol']-2044.130)/2044.130:.4%} "
                  f"sr_in_err={abs(inner['radial_stress']-ex_in[0])/abs(ex_in[0]):.3%} "
                  f"st_in_err={abs(inner['hoop_stress']-ex_in[1])/ex_in[1]:.3%} "
                  f"vm_in_err={abs(inner['von_mises']-ex_in[2])/ex_in[2]:.3%} "
                  f"st_out_err={abs(outer['hoop_stress']-ex_out[1])/ex_out[1]:.3%} "
                  f"sr_out/p={abs(outer['radial_stress']-ex_out[0])/P_:.3%} "
                  f"stats={stats} [{time.time()-t0:.1f}s]")