"""Atomic macroscopic energy-budget gate for the local rigid-fluid fixture.

A diagnostic safety gate, not a remedy for moving-mask pressure artifacts.
No force clipping, velocity rescaling, or automatic physical qualification.
"""
from __future__ import annotations
import copy
import math
import torch
from .fluid_rigid_feedback import RigidFluidFeedback, fluid_kinetic_J


def macroscopic_energy(sim):
    """Kinetic plus isothermal density free energy, SI units.

    rho_ref is the initial lattice density (1); mask-volume changes remain a
    known discretization defect. This functional does not certify full LBM
    entropy stability or a flexible-shell energy law.
    """
    f=sim.fluid.solver.f
    rho=f.sum(0)[~sim.fluid.solver.solid]
    if not torch.isfinite(rho).all() or bool((rho<=0).any()):
        raise ValueError('positive fluid density required')
    c=sim.fluid.config
    scale=c['density_kg_m3']*c['thickness_m']*c['dx_m']**4/(3*c['dt_s']**2)
    # x*log(x)-x+1 is nonnegative, with minimum at uniform initial density.
    compression=float((rho*torch.log(rho)-rho+1).sum())*scale
    kinetic=fluid_kinetic_J(sim.fluid)+float(.5*sim.mass*sim.velocity.square().sum())
    return {'kinetic_J':kinetic,'density_free_energy_J':compression,'total_J':kinetic+compression}


class EnergyBudgetExceeded(RuntimeError):
    def __init__(self, report):
        self.report=report
        super().__init__('unforced macroscopic energy budget exceeded; candidate rolled back')


class PassiveRigidFeedback:
    """Stop before accepting an unforced energy-growing local fluid step.

    The cumulative budget is fixed at the initial state. Failed candidates do
    not change populations, pose, velocity, clocks, or either exchange history.
    The rejection report is returned on the exception; it is not solver state.
    """
    def __init__(self, *, relative_tolerance=1e-3, absolute_tolerance_J=1e-12, **config):
        for value in (relative_tolerance,absolute_tolerance_J):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError('finite nonnegative energy tolerance required')
        if config.get('backend','local')!='local':
            raise ValueError('energy gate requires a local boundary without an external reservoir')
        self.sim=RigidFluidFeedback(**config)
        self.relative_tolerance=float(relative_tolerance)
        self.absolute_tolerance_J=float(absolute_tolerance_J)
        self.initial_energy_J=macroscopic_energy(self.sim)['total_J']
        self.accepted=[]

    def step(self):
        checkpoint=copy.deepcopy(self.sim.snapshot())
        try:
            entry=self.sim.step()
            energy=macroscopic_energy(self.sim)
            if self.sim.reservoir_impulse.abs().max().item()!=0:
                raise ValueError('external reservoir work is not accounted for')
            limit=self.initial_energy_J*(1+self.relative_tolerance)+self.absolute_tolerance_J
            audit={'candidate_time_s':self.sim.fluid.time_s,**energy,'initial_energy_J':self.initial_energy_J,
                   'allowed_energy_J':limit,'excess_J':energy['total_J']-limit,
                   'accepted':energy['total_J']<=limit,'physical_accuracy_qualified':False}
            if not audit['accepted']:
                raise EnergyBudgetExceeded(audit)
        except BaseException:
            self.sim=RigidFluidFeedback.restore(checkpoint)
            raise
        self.accepted.append(audit)
        return {'exchange':entry,'energy_budget':audit}

    def snapshot(self):
        return copy.deepcopy({'schema':'tensorfem.passive-rigid-feedback/1','relative_tolerance':self.relative_tolerance,
                'absolute_tolerance_J':self.absolute_tolerance_J,'initial_energy_J':self.initial_energy_J,
                'accepted':self.accepted,'solver':self.sim.snapshot()})

    @classmethod
    def restore(cls,data):
        fields={'schema','relative_tolerance','absolute_tolerance_J','initial_energy_J','accepted','solver'}
        if not isinstance(data,dict) or set(data)!=fields or data['schema']!='tensorfem.passive-rigid-feedback/1':
            raise ValueError('invalid energy-gate restart schema')
        data=copy.deepcopy(data)
        raw=data['solver']
        obj=cls(relative_tolerance=data['relative_tolerance'],absolute_tolerance_J=data['absolute_tolerance_J'],
                backend=raw['backend'],body_mass_kg=raw['body_mass_kg'],**raw['fluid']['config'])
        obj.sim=RigidFluidFeedback.restore(raw)
        if not math.isfinite(data['initial_energy_J']) or data['initial_energy_J']<0:
            raise ValueError('invalid initial energy budget')
        if abs(data['initial_energy_J']-obj.sim.initial_kinetic_J)>1e-12:
            raise ValueError('initial energy budget does not match solver baseline')
        if len(data['accepted'])!=len(obj.sim.history):
            raise ValueError('energy and solver history clock mismatch')
        expected_limit=data['initial_energy_J']*(1+obj.relative_tolerance)+obj.absolute_tolerance_J
        for i,row in enumerate(data['accepted']):
            numeric=('candidate_time_s','kinetic_J','density_free_energy_J','total_J','initial_energy_J','allowed_energy_J','excess_J')
            if any(isinstance(row.get(k),bool) or not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in numeric):
                raise ValueError('nonfinite or missing energy history field')
            if abs(row['total_J']-row['kinetic_J']-row['density_free_energy_J'])>1e-12 or row['initial_energy_J']!=data['initial_energy_J'] or abs(row['allowed_energy_J']-expected_limit)>1e-12 or abs(row['excess_J']-row['total_J']+expected_limit)>1e-12:
                raise ValueError('inconsistent energy history budget')
            if row['accepted'] is not True or row['physical_accuracy_qualified'] is not False or row['total_J']>expected_limit:
                raise ValueError('invalid accepted energy record')
            if abs(row['candidate_time_s']-(i+1)*obj.sim.fluid.config['dt_s'])>1e-12:
                raise ValueError('stale energy history clock')
        if macroscopic_energy(obj.sim)['total_J']>expected_limit:
            raise ValueError('restart exceeds energy budget')
        if data['accepted'] and abs(data['accepted'][-1]['total_J']-macroscopic_energy(obj.sim)['total_J'])>1e-12:
            raise ValueError('restart energy field mismatch')
        obj.initial_energy_J=data['initial_energy_J'];obj.accepted=copy.deepcopy(data['accepted'])
        return obj
