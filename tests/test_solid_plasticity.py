import pytest
import torch

from tensorfem.solid_plasticity import (
    PlasticSolidState, Tet4J2Model, assemble_tet4_j2, solve_increment,
    solve_load_steps,
)

D = torch.float64


def model():
    # One constant-strain tetrahedron.  Three free normal DOFs permit the
    # lateral Poisson contractions of a homogeneous uniaxial-stress state.
    return Tet4J2Model(
        torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=D),
        torch.tensor([[0,1,2,3]]), 200000., .3, 250., 10000.,
        torch.tensor([0,1,2,4,5,6,8,9,10]),
    )


def test_force_controlled_uniaxial_path_matches_closed_form():
    m = model(); volume = 1./6.
    final_stress = 400.
    # For the affine mode u_x=epsilon*x, virtual work gives Q=sigma*volume.
    force = torch.zeros(m.n_dofs, dtype=D); force[3] = final_stress * volume
    results = solve_load_steps(m, force, torch.linspace(.05, 1., 20, dtype=D))
    r = results[-1]
    # For linear isotropic hardening: eps = sigma/E + (sigma-sy)/H.
    exact_strain = final_stress/m.young + (final_stress-m.yield_stress)/m.hardening
    assert abs(float(r.displacement[3])-exact_strain)/exact_strain < 1e-10
    assert abs(float(r.stress[0,0])-final_stress)/final_stress < 1e-10
    assert abs(float(r.stress[0,1])) < 1e-8
    assert max(x.iterations for x in results) <= 3


def test_algorithmic_tangent_matches_internal_force_difference():
    m=model(); state=PlasticSolidState.virgin(m)
    u=torch.zeros(m.n_dofs,dtype=D); u[3]=.01; u[7]=u[11]=-.003
    f,k,_,_=assemble_tet4_j2(m,u,state)
    h=1e-7
    for j in (3,7,11):
        up=u.clone(); um=u.clone(); up[j]+=h; um[j]-=h
        fp=assemble_tet4_j2(m,up,state)[0]; fm=assemble_tet4_j2(m,um,state)[0]
        rel=torch.linalg.vector_norm((fp-fm)/(2*h)-k[:,j])/torch.linalg.vector_norm(k[:,j])
        assert float(rel)<1e-7


def test_failed_increment_rolls_back_committed_state():
    m=model(); committed=PlasticSolidState.virgin(m)
    force=torch.zeros(m.n_dofs,dtype=D); force[3]=300.
    before=committed.points[0].plastic_strain.clone()
    with pytest.raises(RuntimeError):
        solve_increment(m,force,1.,committed,torch.zeros(m.n_dofs,dtype=D),max_iterations=1)
    assert torch.equal(committed.points[0].plastic_strain,before)
    assert float(committed.points[0].alpha)==0.


def test_elastic_plastic_unload_energy_balance():
    m=model(); volume=1./6.
    force=torch.zeros(m.n_dofs,dtype=D); force[3]=400.*volume
    # A factor of .625 (sigma=250) is included so trapezoidal work integrates
    # both piecewise-linear branches exactly.
    up=solve_load_steps(m,force,torch.linspace(.025,1.,40,dtype=D))
    down=solve_load_steps(m,force,torch.linspace(.975,0.,40,dtype=D),
                          initial_state=up[-1].state,initial_displacement=up[-1].displacement)
    residual_strain=float(down[-1].displacement[3])
    exact=(400.-250.)/m.hardening
    assert abs(residual_strain-exact)/exact<1e-10
    assert abs(float(down[-1].stress[0,0]))<1e-8
    # Closed load/unload work equals plastic dissipation plus stored isotropic
    # hardening energy: V*(sigma_max^2-sigma_y^2)/(2H).
    displacements=torch.tensor([0.]+[float(x.displacement[3]) for x in up+down],dtype=D)
    generalized=torch.tensor([0.]+[float(x.load_factor*force[3]) for x in up+down],dtype=D)
    work=torch.sum(.5*(generalized[:-1]+generalized[1:])*(displacements[1:]-displacements[:-1]))
    exact_work=volume*(400.**2-m.yield_stress**2)/(2*m.hardening)
    assert abs(float(work)-exact_work)/exact_work < 1e-10
