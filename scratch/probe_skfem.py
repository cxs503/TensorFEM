import numpy as np
from skfem import (MeshQuad, ElementQuad2, ElementVector, Basis, FacetBasis,
                   LinearForm, asm, condense, solve)
from skfem.models.elasticity import linear_elasticity, lame_parameters

E, nu = 1.0, 1.0/3.0
n = 8
m = MeshQuad.init_tensor(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
s, t = m.p[0].copy(), m.p[1].copy()
m.p[0] = 48.0 * s
m.p[1] = 44.0 * s + 44.0 * t - 28.0 * s * t
print("max x,y", m.p[0].max(), m.p[1].max(), "min", m.p.min(axis=1))

e = ElementVector(ElementQuad2())
basis = Basis(m, e)
K = asm(linear_elasticity(*lame_parameters(E, nu)), basis)
facets = m.facets_satisfying(lambda x: np.abs(x[0] - 48.0) < 1e-9)
fb = FacetBasis(m, e, facets=facets)

@LinearForm
def load(v, w):
    return (1.0 / 16.0) * v.value[1]
f = asm(load, fb)
print("sum f", f.sum(), "ndofs", basis.N)

left = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9)
u = solve(*condense(K, f, D=left))

P = np.array([[48.0], [52.0]])
try:
    vals = basis.probes(P) @ u
    print("probes value", vals.ravel())
except Exception as exc:
    print("probes failed:", exc)
    d = (m.p[0] - 48.0) ** 2 + (m.p[1] - 52.0) ** 2
    i = int(np.argmin(d))
    print("nearest node", m.p[:, i], np.sqrt(d[i]), "u slices", u[2*i+1], u[i], u[i + m.p.shape[1]])