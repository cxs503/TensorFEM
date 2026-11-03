"""Diagnostic: quantify plane-stress condensation tangent accuracy + determinism."""
import os, sys, math, json
sys.path.insert(0, "src")
import torch
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model, LayeredShell4State, plane_stress_j2_update)
from tensorfem.plasticity import update_j2

D = torch.float64


def make_model():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=D)
    return LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3,
                              .1, 250., 1200.)


def richardson(model, state, strain, j, base=1e-4):
    """High-accuracy column j of dsigma/dstrain by Richardson extrapolation."""
    e = torch.zeros(3, dtype=D); e[j] = 1.
    def col(h):
        return (plane_stress_j2_update(strain + h*e, model, state, tangent=False)[0]
                - plane_stress_j2_update(strain - h*e, model, state, tangent=False)[0]) / (2*h)
    c1 = col(base); c2 = col(base/2); c3 = col(base/4)
    return (4*c2 - c1)/3, (16*c3 - c2*4 + c1*0 - (4*c2-c1))/15


def condensation_residual(model, state, strain):
    """Residual sigma_zz left by the internal local Newton (recon) after the call."""
    stress = plane_stress_j2_update(strain, model, state, tangent=False)[0]
    # reconstruct the converged eps_zz: sigma_zz=0 => trace relation; solve by bisection-free Newton
    zz = -model.poisson/(1-model.poisson)*float(strain[0]+strain[1])
    for _ in range(200):
        full = torch.stack((
            torch.stack((strain[0], .5*strain[2], torch.zeros((), dtype=D))),
            torch.stack((.5*strain[2], strain[1], torch.zeros((), dtype=D))),
            torch.stack((torch.zeros((), dtype=D), torch.zeros((), dtype=D),
                         torch.tensor(zz, dtype=D)))))
        s = update_j2(full, model.young, model.poisson, model.yield_stress,
                      model.hardening, state)[0]
        h = 1e-7
        sp = update_j2(_shift(full, zz+h), model.young, model.poisson, model.yield_stress,
                       model.hardening, state)[0][2, 2]
        sm = update_j2(_shift(full, zz-h), model.young, model.poisson, model.yield_stress,
                       model.hardening, state)[0][2, 2]
        d = (sp-sm)/(2*h)
        step = float(s[2, 2])/float(d)
        zz = zz - step
        if abs(step) < 1e-16:
            break
    full = _shift(full, zz)
    return abs(float(update_j2(full, model.young, model.poisson, model.yield_stress,
                               model.hardening, state)[0][2, 2]))


def _shift(T, zz):
    out = T.clone(); out[2, 2] = zz; return out


def main():
    model = make_model(); state = LayeredShell4State.virgin(model).points[0][0][0]
    # uniaxial plastic state at 310 MPa
    target = 310.; alpha = (target-250.)/1200.
    strain = torch.tensor([target/model.young+alpha,
                           -model.poisson*target/model.young-alpha/2, 0.], dtype=D)
    tangent = plane_stress_j2_update(strain, model, state)[1]
    ref = torch.stack([richardson(model, state, strain, j)[0] for j in range(3)], 1)
    err = float(torch.linalg.matrix_norm(tangent-ref)/torch.linalg.matrix_norm(ref))
    print("tangent rel Frobenius error vs Richardson:", f"{err:.3e}")
    print("  sigma_zz residual left by local Newton:", f"{condensation_residual(model, state, strain):.3e}")
    # determinism over thread count
    scales = {}
    for nthread in (1, 4, 16):
        torch.set_num_threads(nthread)
        t = plane_stress_j2_update(strain, model, state)[1]
        scales[nthread] = t
    base = scales[1]
    for k, v in scales.items():
        print(f"  threads={k:2d} bitwise-equal-to-1:", torch.equal(base, v),
              " rel-diff:", f"{float(torch.linalg.matrix_norm(v-base)/torch.linalg.matrix_norm(base)):.3e}")
    # an element-level stiffness assembled at two thread counts
    from tensorfem.layered_shell4_plasticity import assemble_layered_shell4
    out = {}
    for nthread in (1, 16):
        torch.set_num_threads(nthread)
        u = torch.zeros(model.n_dofs, dtype=D); u[6] = u[12] = .0025
        _, K, _, _ = assemble_layered_shell4(model, u, LayeredShell4State.virgin(model))
        out[nthread] = K
    print("element tangent bitwise equal 1 vs 16 threads:", torch.equal(out[1], out[16]),
          " rel:", f"{float(torch.linalg.matrix_norm(out[16]-out[1])/torch.linalg.matrix_norm(out[1])):.3e}")


if __name__ == "__main__":
    main()