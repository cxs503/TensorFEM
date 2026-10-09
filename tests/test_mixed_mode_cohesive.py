import torch

from tensorfem.mixed_mode_cohesive import BenzeggaghKenaneLaw

D=torch.float64


def model():
    return BenzeggaghKenaneLaw(1000.,800.,10.,8.,.5,1.2,1.6)


def integrated_work(law,direction):
    probe=law.proportional_response(torch.tensor([0.],dtype=D),direction)
    a=torch.linspace(0.,float(probe.failure_amplitude),20001,dtype=D)
    r=law.proportional_response(a,direction)
    work_traction=r.traction@r.direction
    return torch.trapezoid(work_traction,a),r


def test_pure_mode_fracture_energies():
    law=model()
    for direction,expected in ((torch.tensor([1.,0.],dtype=D),law.gic),
                               (torch.tensor([0.,1.],dtype=D),law.giic)):
        energy,_=integrated_work(law,direction)
        assert abs(float(energy)-expected)/expected < .03


def test_mixed_mode_path_matches_bk_energy_and_peak_initiation():
    law=model(); direction=torch.tensor([3.,4.],dtype=D)
    energy,r=integrated_work(law,direction)
    assert abs(float(energy-r.critical_energy)/float(r.critical_energy)) < .03
    at_peak=law.proportional_response(r.onset_amplitude.reshape(1),direction)
    normalized=(at_peak.traction[0,0]/law.tn0)**2+(at_peak.traction[0,1]/law.ts0)**2
    assert abs(float(normalized)-1.) < 1e-12
    failed=law.proportional_response(r.failure_amplitude.reshape(1),direction)
    assert torch.max(torch.abs(failed.traction)) < 1e-12


def test_bk_closed_form_for_equal_modal_work_partition():
    law=model()
    # Choose direction so kn*cn^2 == ks*cs^2, hence shear fraction=1/2.
    d=torch.tensor([law.ks**.5,law.kn**.5],dtype=D)
    got=law.critical_energy(d)
    ref=law.gic+(law.giic-law.gic)*(.5**law.eta)
    assert abs(float(got)-ref)/ref < 1e-14
