import math
import numpy as np
import sys
sys.path.insert(0, "src")
from tensorfem.phase_field_fracture import (
    PhaseFieldProblem, structured_quad_mesh, nodes_on_segment,
    crack_line_elements, sharp_crack_energy_release,
)

E, nu = 1000.0, 0.3
Ep = E / (1 - nu**2)
W, H, a, sigma = 1.0, 1.0, 0.3, 1.0
neg = 72
h = W / neg


def build(a_crack):
    nodes, elems = structured_quad_mesh(0.0, W, -H, H, neg, 2 * neg)
    vid = crack_line_elements(nodes, elems, (0.0, 0.0), (a_crack, 0.0), tol=h * 0.6)
    return PhaseFieldProblem(nodes, elems, E, nu, Gc=1e-3, l=0.05,
                             thickness=1.0, void_elements=vid), nodes


probs = [build(a - h)[0], build(a)[0], build(a + h)[0]]
nodes = build(a)[1]
f = np.zeros(2 * probs[1].n_nodes)
for ytarget, sign in ((H, +1.0), (-H, -1.0)):
    ys = np.where(np.abs(nodes[:, 1] - ytarget) < 1e-12)[0]
    order = np.argsort(nodes[ys, 0]); ys = ys[order]; xs = nodes[ys, 0]
    for k in range(len(ys) - 1):
        L = xs[k + 1] - xs[k]
        f[2 * ys[k] + 1] += sign * sigma * L * 0.5
        f[2 * ys[k + 1] + 1] += sign * sigma * L * 0.5
bl = np.argmin(nodes[:, 0]**2 + (nodes[:, 1] + H)**2)
br = np.argmin((nodes[:, 0] - W)**2 + (nodes[:, 1] + H)**2)
fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br: 0.0}

out = sharp_crack_energy_release(probs, f, fixed, h)
G = out["G"]
K_fem = math.sqrt(abs(G) * Ep)
al = a / W
F = 1.12 - 0.231 * al + 10.55 * al**2 - 21.72 * al**3 + 30.39 * al**4
K_ana = sigma * math.sqrt(math.pi * a) * F
print(out["details"])
print("G", G, "K_fem", K_fem, "K_ana", K_ana, "err%", 100 * (K_fem - K_ana) / K_ana)
print([d["potential"] for d in out["details"]])