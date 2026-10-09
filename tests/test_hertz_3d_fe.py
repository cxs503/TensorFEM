from tensorfem.hertz_3d_fe import solve_hertz_cap_block
import torch
from tensorfem.hertz_quarter_patch import build_hertz_quarter_volume

def test_hertz_quarter_patch_builds_positive_tet_volume_and_mortar_faces():
    mesh=build_hertz_quarter_volume(contact_radius=.2, contact_cells=6,
        domain_radii=6., outer_cells=3, depth_cells=8)
    assert mesh.nodes.shape[1]==3 and mesh.elements.shape[1]==4
    assert mesh.surface_faces.shape[1]==4 and mesh.surface_faces.shape[0]>0
    x=mesh.nodes[mesh.elements]
    det=torch.linalg.det(x[:,1:]-x[:,:1])
    assert bool(torch.all(det>0))

def test_true_tet4_two_solid_hertz_level_one_closure():
    result=solve_hertz_cap_block(cells=2)
    assert result.force>0 and result.reference_force>0
    assert result.residual_norm<1e-7
    assert result.minimum_jacobian>0
    assert result.force_imbalance<1e-9
