"""Patch: analytic plane-stress J2 tangential operator (no nested finite differences)."""
import pathlib, re

# ---------------------------------------------------------------- plasticity.py
p = pathlib.Path("src/tensorfem/plasticity.py")
s = p.read_text()
addition = '''

_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def j2_consistent_tangent(strain: torch.Tensor, young: float, poisson: float,
                          hardening: float, state: J2State, dgamma: torch.Tensor):
    """Exact algorithmic tangent of the radial return returned by ``update_j2``.

    ``dgamma`` is the third value of ``update_j2`` for the same strain and
    committed state, so the operator always belongs to the branch that was
    actually integrated (elastic when ``dgamma == 0``).  The matrix is Voigt
    ordered ``[xx, yy, zz, xy, yz, zx]`` and its shear columns act on
    engineering shear strains ``gamma = 2*eps``.

    ``C = K 1x1 + 2G a I_dev + c s_trial(x)s_trial`` with
    ``a = 1 - 3G dgamma/seq`` and ``c = 3G d(a)/d(seq)/seq``.
    """
    if strain.shape != (3, 3):
        raise ValueError("strain must be 3x3")
    dtype, device = strain.dtype, strain.device
    G = young / (2.0 * (1.0 + poisson))
    K = young / (3.0 * (1.0 - 2.0 * poisson))
    eye3 = torch.eye(3, dtype=dtype, device=device)
    ones3 = torch.ones((3, 3), dtype=dtype, device=device)
    elastic_strain = strain - state.plastic_pstrain if False else strain - state.plastic_strain
    s_trial = 2.0 * G * (elastic_strain - torch.trace(elastic_strain) / 3.0 * eye3)
    seq = torch.sqrt(1.5 * torch.sum(s_trial * s_trial))
    plastic = bool(float(dgamma) > 0.)
    scale = torch.ones((6, 6), dtype=dtype, device=device) * K
    tangent = torch.zeros((6, 6), dtype=dtype, device=device)
    if not plastic:
        tangent[:3, :3] = scale[:3, :3] + 2.0 * G * (eye3 - ones3 / 3.0)
        tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G
        return tangent
    if not bool(seq > 0.):
        raise RuntimeError("plastic radial return reported zero trial stress")
    a = 1.0 - 3.0 * G * float(dgamma) / float(seq)
    # d(a)/d(seq) = -3G [1/((3G+H) seq) - dgamma/seq**2]
    da = -3.0 * G * (1.0 / ((3.0 * G + hardening) * float(seq))
                     - float(dgamma) / (float(seq) ** 2))
    c = 2.0 * G * (1.5 / float(seq)) * da
    tangent[:3, :3] = scale[:3, :3] + 2.0 * G * a * (eye3 - ones3 / 3.0)
    tangent[3, 3] = tangent[4, 4] = tangent[5, 5] = G * a
    vector = torch.stack([s_trial[i, j] for i, j in _VOIGT_PAIRS])
    return tangent + c * torch.outer(vector, vector)


def condense_plane_stress(tangent: torch.Tensor) -> torch.Tensor:
    """Schur-condense a Voigt 6x6 tangent onto ``[xx, yy, xy]`` with ``sigma_zz=0``."""
    in_plane = torch.tensor([0, 1, 3], dtype=torch.long, device=tangent.device)
    zz = 2
    normal = tangent[zz, zz]
    if abs(float(normal)) <= torch.finfo(tangent.dtype).eps * max(
            1.0, float(tangent.abs().max())):
        raise RuntimeError("plane-stress condensation has singular sigma_zz tangent")
    return (tangent[in_plane][:, in_plane]
            - torch.outer(tangent[in_plane, zz], tangent[zz, in_plane]) / normal)
'''
s = s.replace("elastic_strain = strain - state.plastic_pstrain if False else strain - state.plastic_strain",
              "elastic_strain = strain - state.plastic_strain")
p.write_text(s + addition)

# ------------------------------------------- layered_shell4_plasticity.py body
q = pathlib.Path("src/tensorfem/layered_shell4_plasticity.py")
t = q.read_text()
t = t.replace("from .plasticity import J2State, update_j2",
              "from .plasticity import (J2State, condense_plane_stress,\n"
              "                         j2_consistent_tangent, update_j2)")

start = t.index("def plane_stress_j2_update(")
end = t.index("def _shape_derivatives(")
body = '''def plane_stress_j2_update(strain: torch.Tensor, model: LayeredShell4Model,
                           committed: J2State, residual_stress: torch.Tensor | None = None,
                           *, tangent: bool = True):
    """Return ``[sx,sy,txy]``, exact condensed consistent tangent and trial state.

    The out-of-plane strain is locally condensed by enforcing ``sigma_zz=0``.
    Both the local condensation and the returned tangent use the analytical
    algorithmic tangent of the radial return, never a finite difference of the
    return map.  A central difference straddles the yield-surface kink: just
    below yield it mixes the elastic and plastic branches, which previously
    biased the tangent by several tens of percent exactly where Newton needs it.
    """
    if strain.shape != (3,):
        raise ValueError("plane-stress strain must be [ex,ey,gxy]")
    initial = torch.zeros((3, 3), dtype=strain.dtype, device=strain.device)
    if residual_stress is not None:
        initial = _residual_strain(residual_stress, model.young, model.poisson)

    def response(value: torch.Tensor):
        """Stress, exact condensed tangent and trial state at in-plane ``value``."""
        zz = -model.poisson/(1-model.poisson) * (value[0]+value[1])
            for _ in range(40):
            full = _tensor_from_inplane(value, zz) + initial
            sigma, trial, dgamma = update_j2(
                full, model.young, model.poisson, model.yield_stress,
                model.hardening, committed)
            tangent_full = j2_consistent_tangent(
                full, model.young, model.poisson, model.hardening, committed, dgamma)
            residual = float(sigma[2, 2])
            if abs(residual) <= 1e-15 * max(model.young, model.yield_stress):
                break
            step = residual / float(tangent_full[2, 2])
            zz -= step
            if abs(step) <= 1e-17 * max(1.0, abs(zz)):
                full = _tensor_from_inplane(value, zz) + initial
                sigma, trial, dgamma = update_j2(
                    full, model.young, model.poisson, model.yield_stress,
                    model.hardening, committed)
                tangent_full = j2_consistent_tangent(
                    full, model.young, model.poisson, model.hardening, committed, dgamma)
                break
        else:
            raise RuntimeError("plane-stress local condensation did not converge")
        stress = torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1]))
        return stress, condense_plane_stress(tangent_full), trial

    stress, material, trial = response(strain)
    if not tangent:
        return stress, None, trial
    return stress, material, trial


'''
t = t[:start] + body + t[end:]
q.write_text(t)
print("patched")