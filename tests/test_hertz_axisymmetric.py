import torch

from tensorfem.hertz_axisymmetric import (
    assemble_axisymmetric_stiffness, axisymmetric_mesh, hertz_reference,
    axisymmetric_hertz_errors, solve_axisymmetric_hertz,
)

D=torch.float64


def errors(n):
    result=solve_axisymmetric_hertz(nr=n,nz=n)
    error=axisymmetric_hertz_errors(result)
    return result,tuple(error[key] for key in
                        ("load","contact_radius","peak_pressure","pressure_l2"))


def test_axisymmetric_elastic_operator_is_symmetric_and_positive():
    nodes,elements=axisymmetric_mesh(2,2,1.,1.,grading=1.,dtype=D)
    K=assemble_axisymmetric_stiffness(nodes,elements,1e5,.3)
    assert torch.allclose(K,K.T,rtol=1e-13,atol=1e-9)
    assert torch.linalg.eigvalsh(K).min() > -1e-8*torch.linalg.eigvalsh(K).max()


def test_deformable_hertz_mesh_convergence_and_three_percent_gate():
    # The chosen sequence resolves both the contact patch and the far field.
    coarse,ec=errors(16); medium,em=errors(24); qualified,eq=errors(48)
    assert coarse.converged and medium.converged and qualified.converged
    assert all(m<c for c,m in zip(ec,em))
    assert all(q<m for m,q in zip(em,eq))
    assert max(eq) < .03
    # Penalty overlap remains below 3% of prescribed indentation.
    penalty=28.*1e5/.5
    penetration=float(qualified.raw_peak_pressure)/penalty
    assert penetration/.025 < .03
    assert torch.all(qualified.pressure >= 0)
    audit=axisymmetric_hertz_errors(qualified)
    assert audit["indentation"] < 1e-12
    assert audit["force_balance"] < 1e-10


def test_halfspace_domain_truncation_study_at_comparable_contact_resolution():
    # Radial/depth counts scale with domain size so the contact-zone spacing
    # remains comparable; the 24a and 32a domains agree within 3%.
    medium=solve_axisymmetric_hertz(nr=40,nz=40,width=12.,depth=12.)
    large=solve_axisymmetric_hertz(nr=53,nz=53,width=16.,depth=16.)
    for name in ("load","contact_radius","peak_pressure"):
        m=getattr(medium,name); l=getattr(large,name)
        assert abs(float(m/l-1)) < .03


def test_hertz_reference_and_input_validation():
    load,a,p0=hertz_reference(.025,10.,1e5,.3)
    assert abs(a-.5) < 1e-15
    assert abs(load-4/3*(1e5/(1-.3**2))*.5**3/10) < 1e-12
    assert abs(p0-3*load/(2*torch.pi*a*a)) < 1e-12
    try: axisymmetric_mesh(1,4,1.,1.)
    except ValueError: pass
    else: raise AssertionError("under-resolved mesh accepted")
