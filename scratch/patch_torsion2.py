"""Second patch: expose the per-plane rotation profile so window studies reuse one solve."""
from __future__ import annotations

import pathlib

path = pathlib.Path("src/tensorfem/torsion_rectangular.py")
text = path.read_text()

old = '''    n_short: int
    n_long: int
    nz: int
'''
new = '''    n_short: int
    n_long: int
    nz: int
    planes: torch.Tensor
    thetas: torch.Tensor
    length: float

    def twist_rate_in(self, window: tuple[float, float]) -> float:
        """Least-squares section rotation rate over a slice of the bar.

        Exposed so a window-sensitivity study costs one solve instead of one per window.
        """
        inside = (self.planes >= window[0] * self.length) & (self.planes <= window[1] * self.length)
        if int(inside.sum()) < 2:
            raise ValueError("window contains fewer than two node planes")
        design = torch.stack((self.planes[inside],
                              torch.ones(int(inside.sum()), dtype=self.planes.dtype)), 1)
        return float(torch.linalg.lstsq(design, self.thetas[inside]).solution[0])
'''
assert old in text
text = text.replace(old, new, 1)

old_tail = '''    inside = (planes >= window[0] * length) & (planes <= window[1] * length)
    design = torch.stack((planes[inside], torch.ones(int(inside.sum()), dtype=nodes.dtype)), 1)
    twist_rate = float(torch.linalg.lstsq(design, thetas[inside]).solution[0])

    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    centroid_stress = torch.einsum("eij,ej->ei", modulus, strain)
    return _PrismSolve(nodes=nodes, cells=cells, displacement=displacement,
                       centroid_stress=centroid_stress, twist_rate=twist_rate,
                       applied_torque=applied, end_traction_residual=residual,
                       n_short=n_short, n_long=n_long, nz=nz)
'''
new_tail = '''    strain = torch.einsum("eij,ej->ei", b_matrix, displacement[element_dofs])
    centroid_stress = torch.einsum("eij,ej->ei", modulus, strain)
    solve = _PrismSolve(nodes=nodes, cells=cells, displacement=displacement,
                        centroid_stress=centroid_stress, twist_rate=float("nan"),
                        applied_torque=applied, end_traction_residual=residual,
                        n_short=n_short, n_long=n_long, nz=nz, planes=planes, thetas=thetas,
                        length=length)
    twist_rate = solve.twist_rate_in(window)
    return _PrismSolve(nodes=nodes, cells=cells, displacement=displacement,
                       centroid_stress=centroid_stress, twist_rate=twist_rate,
                       applied_torque=applied, end_traction_residual=residual,
                       n_short=n_short, n_long=n_long, nz=nz, planes=planes, thetas=thetas,
                       length=length)
'''
assert old_tail in text
text = text.replace(old_tail, new_tail, 1)
path.write_text(text)
print("patched", len(text.splitlines()), "lines")