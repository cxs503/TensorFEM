import torch

from tensorfem.advanced_contact import (NodeSegmentPair,
    assemble_node_segment_constraints, independent_active_reactions)
from tensorfem.contact import solve_rigid_plane_contact

D = torch.float64


def test_two_node_segment_contacts_match_independent_schur_solution():
    # Two slave nodes above a deformable three-node master surface.
    xy = torch.tensor([[.5,.10],[1.5,.20],[0.,0.],[1.,0.],[2.,0.]], dtype=D)
    pairs = [NodeSegmentPair(0,2,3), NodeSegmentPair(1,3,4)]
    constraints = assemble_node_segment_constraints(xy, pairs)
    kdiag = torch.tensor([200.,800.,300.,1200.,500.,1000.,500.,900.,500.,1100.], dtype=D)
    K = torch.diag(kdiag)
    f = torch.zeros(10, dtype=D); f[1] = -180.; f[3] = -330.
    expected = independent_active_reactions(K, f, constraints.matrix, constraints.initial_gap)
    result = solve_rigid_plane_contact(K, f, constraints.matrix, constraints.initial_gap)
    assert torch.all(expected > 0)
    assert torch.max(torch.abs((result.contact_force-expected)/expected)) < .03
    assert torch.max(torch.abs(result.gap)) < 1e-12
    assert torch.max(torch.abs(K@result.displacement-constraints.matrix.T@result.contact_force-f)) < 1e-11
    assert torch.max(torch.abs(result.gap*result.contact_force)) < 1e-12


def test_projection_weights_and_rigid_translation_invariance():
    xy = torch.tensor([[.75,.2],[0.,0.],[2.,0.]], dtype=D)
    c = assemble_node_segment_constraints(xy,[NodeSegmentPair(0,1,2)])
    assert abs(float(c.projection[0])-.375) < 1e-14
    rigid = torch.tensor([.4,-.7]*3,dtype=D)
    assert abs(float(c.matrix@rigid)) < 1e-14


def test_outside_projection_is_rejected():
    xy=torch.tensor([[2.,.1],[0.,0.],[1.,0.]],dtype=D)
    try: assemble_node_segment_constraints(xy,[NodeSegmentPair(0,1,2)])
    except ValueError: pass
    else: raise AssertionError("outside projection must be rejected")
