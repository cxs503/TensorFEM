import torch

from tensorfem.spherical_shell import hemisphere_with_hole


def test_hemisphere_with_hole_converges_and_fine_mesh_is_below_three_percent():
    results=[hemisphere_with_hole(n) for n in (6,8,12)]
    displacements=[r.displacement for r in results]
    assert displacements[0] < displacements[1] < displacements[2]
    assert results[-1].relative_error < .03


def test_hemisphere_qualification_is_not_drilling_factor_calibration():
    # A two-decade bracket around the inherited fixed factor remains below 3%.
    results=[hemisphere_with_hole(12,drilling_factor=d) for d in (1e-7,1e-6,1e-5)]
    assert max(r.relative_error for r in results) < .03


def test_hemisphere_free_residual_and_vertical_gauge_reaction():
    r=hemisphere_with_hole(8); n=8; fixed=[]
    for i in range(n+1):
        a=i*(n+1); fixed += [6*a+1,6*a+3,6*a+5]
        b=a+n; fixed += [6*b+0,6*b+4,6*b+5]
    fixed.append(2); mask=torch.ones_like(r.reaction,dtype=torch.bool);mask[torch.tensor(sorted(set(fixed)))]=False
    assert float(torch.linalg.vector_norm(r.reaction[mask])) < 1e-8
    assert abs(float(r.reaction[2])) < 1e-9


def test_spherical_mesh_is_genuinely_doubly_curved():
    r=hemisphere_with_hole(4)
    normals=[]
    for eid in (0,1,4):
        xyz=r.nodes[r.elements[eid]]; n=torch.linalg.cross(xyz[1]-xyz[0],xyz[3]-xyz[0]); normals.append(n/torch.linalg.vector_norm(n))
    assert abs(float(normals[0]@normals[1])) < .999
    assert abs(float(normals[0]@normals[2])) < .999
