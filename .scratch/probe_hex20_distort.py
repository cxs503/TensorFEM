import math, torch
torch.set_num_threads(1)
from tensorfem.hex20 import (structured_hex20_mesh, hex20_stiffness, hex20_shape_gradients,
                             _gradient, _b, _CORNERS, _EDGES)
from tensorfem.solid3d import elasticity_matrix_3d, _b as _b3
from tensorfem.element_rank_audit import rigid_body_basis_solid3d

dtype = torch.float64
D = elasticity_matrix_3d(torch.tensor([200e9], dtype=dtype), torch.tensor([0.3], dtype=dtype))

base, _ = structured_hex20_mesh(1., 1., 1., 1, 1, 1)

def nullspace_dim(k, rel=1e-8):
    w = torch.linalg.eigvalsh(k)
    return int((w.abs() < rel * w.abs().max()).sum()), w.abs().min().item(), w

def principal_angles(k, rigid, tol=1e-8):
    w, v = torch.linalg.eigh(k)
    scale = w.abs().max()
    nul = v[:, (w.abs() < tol * scale)]
    if nul.shape[1] == 0:
        return 0, 99.
    sv = torch.linalg.svdvals(rigid.T @ nul)
    dim = int((sv > 1 - 1e-6).sum())
    return dim, sv.min().item()

def distort(kind, amp):
    x = base.clone()
    for i in range(len(x)):
        r, s, t = x[i] - 0.5
        if kind == "taper":
            x[i, 1] *= 1 + amp * x[i, 2] * 2
            x[i, 0] *= 1 + amp * x[i, 2] * 2
            continue
        if kind == "skew":
            x[i, 0] += amp * x[i, 2]
            x[i, 1] += amp * 0.5 * x[i, 2]
            continue
        if kind == "warp":
            x[i, 2] += amp * math.sin(math.pi * x[i, 0]) * math.sin(math.pi * x[i, 1])
            continue
        if kind == "trapezoid":
            x[i, 0] += amp * x[i, 1] * x[i, 2]
            x[i, 1] += amp * 0.6 * x[i, 0] * x[i, 2]
            continue
        if kind == "twist":
            ang = amp * x[i, 2]
            x[i, 0], x[i, 1] = x[i, 0] * math.cos(ang) - x[i, 1] * math.sin(ang), x[i, 0] * math.sin(ang) + x[i, 1] * math.cos(ang)
            continue
    return x

for kind in ("taper", "skew", "warp", "trapezoid", "twist"):
    for amp in (0.05, 0.15, 0.3):
        x = distort(kind, amp)
        try:
            k, _ = hex20_stiffness(x[None], D)
        except Exception as e:
            print(f'{kind} amp={amp}: raised {e}')
            continue
        nd, mn, spec = nullspace_dim(k[0])
        dim, sv = principal_angles(k[0], rigid_body_basis_solid3d(x))
        print(f"{kind:10s} amp={amp}: null={nd} rigid_match={dim} min_sv={sv:.3e} min|lam|={mn:.3e}")