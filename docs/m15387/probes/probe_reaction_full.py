"""Second probe: include ElementQuad2 *edge* dofs on the clamped edge (the audited harness
only sums basis.nodal_dofs[1, left_nodes], i.e. vertex y-dofs)."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
from cook_membrane_xval import _skfem_mesh
from skfem import Basis, FacetBasis, ElementQuad2, ElementTriP2, ElementVector, LinearForm, asm, condense, solve
from skfem.models.elasticity import linear_elasticity

E, NU, R = 1.0, 1.0/3.0, 1.0/16.0
lam3d = E*NU/((1+NU)*(1-2*NU)); mu = E/(2*(1+NU)); lam_ps = 2*lam3d*mu/(lam3d+2*mu)

for kind in ("quad", "tri"):
    for n in (8, 16):
        m = _skfem_mesh(n, kind)
        elem = ElementVector(ElementQuad2() if kind == "quad" else ElementTriP2())
        basis = Basis(m, elem)
        K = asm(linear_elasticity(lam_ps, mu), basis)
        feats = m.facets_satisfying(lambda x: np.abs(x[0]-48.0) < 1e-9)
        fb = FacetBasis(m, elem, facets=feats)
        @LinearForm
        def trac(v, w):
            return R * v.value[1]
        f = asm(trac, fb)
        D = basis.get_dofs(lambda x: np.abs(x[0]) < 1e-9)
        u = solve(*condense(K, f, D=D))
        res = K @ u - f
        left_nodes = np.where(np.abs(m.p[0]) < 1e-9)[0]
        # full dof set of the clamped edge (all components, vertex + edge dofs)
        allC = D.all()
        vertex_y = basis.nodal_dofs[1, left_nodes]
        # identify y-component dofs among the constrained set via doflocs bookkeeping:
        # for a vector basis, dof d belongs to component c if d in basis.nodal_dofs[c] /
        # basis.edge_dofs[c].  Safest: complement of x-component dofs.
        # component dof sets: vertex (nodal) + mid-side (facet) + cell interior dofs
        comp = []
        for c in (0, 1):
            comp.append(set(int(v) for v in np.concatenate([
                np.ravel(basis.nodal_dofs[c]),
                np.ravel(basis.facet_dofs[c]),
                np.ravel(basis.interior_dofs[c])])))
        ydofs = [int(d) for d in allC if int(d) in comp[1]]
        xdofs = [int(d) for d in allC if int(d) in comp[0]]
        print(dict(kind=kind, n=n,
                   sum_all_constrained=float(res[allC].sum()),
                   sum_y_full=float(res[ydofs].sum()),
                   sum_x_full=float(res[[int(d) for d in allC if int(d) in xdofs]].sum()),
                   sum_y_vertex_only=float(res[vertex_y].sum()),
                   n_y_dofs_full=len(ydofs), n_y_vertex=len(vertex_y)))
