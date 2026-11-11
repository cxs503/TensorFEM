"""Adversarial probe: is skfem reaction_y=-0.42 a loading-consistency artefact or a metric bug?

Claim under test (docs/cook-xval-report.md §4 note): "scikit-fem reaction forces are not
exact because the traction is applied as nodal point loads rather than consistent edge loads".
Statics requires sum(R_y) = -(sum of applied f_y) regardless of how the load is distributed.
So we measure both: total applied load, and the constrained-dof residual sum.
"""
import numpy as np
from skfem import (Basis, FacetBasis, ElementQuad1, ElementQuad2, ElementTriP2,
                   ElementVector, LinearForm, MeshQuad, MeshTri, asm, condense, solve)
from skfem.models.elasticity import linear_elasticity

E, NU, T, R = 1.0, 1.0/3.0, 1.0, 1.0/16.0
CORNERS = np.array([[0.,0.],[48.,44.],[48.,60.],[0.,44.]])

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
from cook_membrane_xval import _skfem_mesh          # same mesh generator as the audited harness


def mesh_for(n, kind):
    return _skfem_mesh(n, kind)

def run(n, kind):
    lam3d = E*NU/((1+NU)*(1-2*NU)); mu = E/(2*(1+NU))
    lam_ps = 2*lam3d*mu/(lam3d+2*mu)
    m = mesh_for(n, kind)
    elem = ElementVector(ElementQuad2() if kind=="quad" else ElementTriP2())
    basis = Basis(m, elem)
    K = asm(linear_elasticity(lam_ps, mu), basis)
    facets = m.facets_satisfying(lambda x: np.abs(x[0]-48.0) < 1e-9)
    fb = FacetBasis(m, elem, facets=facets)
    @LinearForm
    def trac(v, w):
        return R * v.value[1]
    f = asm(trac, fb)
    D = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9)
    u = solve(*condense(K, f, D=D))
    res = K @ u - f
    right_nodes = np.where(np.abs(m.p[0]-48.0) < 1e-9)[0]
    left_nodes = np.where(np.abs(m.p[0]) < 1e-9)[0]
    return dict(
        n=n, kind=kind,
        f_total=float(f.sum()),
        f_y_right=float(f[basis.nodal_dofs[1, right_nodes]].sum()),
        res_y_left=float(res[basis.nodal_dofs[1, left_nodes]].sum()),
        res_all_y=float(res[basis.nodal_dofs[1, :]].sum()),
        res_all=float(res[D.all()].sum()),
        uy_mid=float(u[basis.nodal_dofs[1, np.argmin((m.p[0]-48)**2+(m.p[1]-52)**2)]]),
    )

for kind in ("quad", "tri"):
    for n in (8, 16):
        print(run(n, kind))
