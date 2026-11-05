import math, torch
torch.set_num_threads(1)
from tensorfem.hex20 import (structured_hex20_mesh, hex20_stiffness, hex20_shape_gradients,
                             _gradient, _b, Hex20Model, solve_hex20)
from tensorfem.solid3d import elasticity_matrix_3d
from tensorfem.element_rank_audit import rigid_body_basis_solid3d

dt = torch.float64
D = elasticity_matrix_3d(torch.tensor([200e9], dtype=dt), torch.tensor([0.3], dtype=dt))
q = math.sqrt(3/5)
base, _ = structured_hex20_mesh(1., 1., 1., 1, 1, 1)

def ke_rule(x, pts, wts):
    ke = x.new_zeros((len(x), 60, 60))
    for i, r in enumerate(pts):
        for j, s in enumerate(pts):
            for k, t in enumerate(pts):
                _, dn = hex20_shape_gradients(r, s, t, like=x)
                g, det = _gradient(x, dn); b = _b(g)
                ke += (b.transpose(1, 2) @ D @ b) * (det * wts[i]*wts[j]*wts[k])[:, None, None]
    return ke

k2 = ke_rule(base[None], (-1/math.sqrt(3), 1/math.sqrt(3)), (1., 1.))[0]
k3 = hex20_stiffness(base[None], D)[0][0]
for name, k in (("2x2x2", k2), ("3x3x3", k3)):
    w = torch.linalg.eigvalsh(k).abs()
    print(name, "smallest 16 eigenvalues:", [f"{v:.3e}" for v in w[:16]])
    for tol in (1e-14, 1e-10, 1e-8, 1e-6):
        print("   tol", tol, int((w < tol*w.max()).sum()))

# randomized distortion rank sweep
g = torch.Generator().manual_seed(7)
bad = 0; nuls = []
for trial in range(200):
    x = base + (torch.rand(base.shape, generator=g, dtype=dt) - .5) * 0.5
    try:
        k = hex20_stiffness(x[None], D)[0][0]
    except ValueError:
        bad += 1; continue
    w, v = torch.linalg.eigh(k)
    nul = v[:, w.abs() < 1e-8 * w.abs().max()]
    sv = torch.linalg.svdvals(rigid_body_basis_solid3d(x).T @ nul)
    nuls.append(int(nul.shape[1]))
print("random sweep: trials=200 invalid_jacobian=%d  null dims set=%s" % (bad, sorted(set(nuls))))

# cantilever with production load pattern, distortion normalised to element size
def cant(nx, ny, rel, seed=0):
    gg = torch.Generator().manual_seed(seed)
    L = 10.; nodes, cells = structured_hex20_mesh(L, 1., 1., nx, ny, ny)
    h = L/nx
    if rel > 0:
        interior = ((nodes[:,0] > 1e-9) & (nodes[:,0] < L-1e-9) &
                    (nodes[:,1] > 1e-9) & (nodes[:,1] < 1-1e-9) &
                    (nodes[:,2] > 1e-9) & (nodes[:,2] < 1-1e-9))
        noise = (torch.rand(nodes.shape, generator=gg, dtype=dt) - .5) * 2 * (rel*h)
        nodes = torch.where(interior[:, None], nodes + noise, nodes)
    forces = nodes.new_zeros(3*len(nodes)); right = torch.where(nodes[:,0] == L)[0]
    corner = (((nodes[right,1] == 0) | (nodes[right,1] == 1)) & ((nodes[right,2] == 0) | (nodes[right,2] == 1)))
    load = -1e6
    forces[3*right[corner]+2] = load*(-1/12)
    forces[3*right[~corner]+2] = load*(1/3)
    left = torch.where(nodes[:,0] == 0)[0]
    fixed = torch.stack((3*left, 3*left+1, 3*left+2), 1).reshape(-1)
    r = solve_hex20(Hex20Model(nodes, cells, nodes.new_tensor(210e9), nodes.new_tensor(.3), forces, fixed))
    tip = float(r.displacement[3*right+2].mean())
    E, nu_ = 210e9, .3; I = 1./12; G = E/(2*(1+nu_))
    ref = 1e6*L**3/(3*E*I) + 1e6*L/((5/6)*G*1.)
    return abs(tip), ref, abs(abs(tip)-ref)/ref*100

for nx in (4, 8, 16):
    for rel in (0.0, 0.05, 0.10, 0.20):
        try:
            t, ref, e = cant(nx, 2, rel)
            print(f"nx={nx} h={10/nx:.3f} rel={rel}: |tip|={t:.6e} err={e:.4f}%")
        except ValueError as ex:
            print(f"nx={nx} rel={rel}: invalid ({ex})")