"""Three-way resolution helper for src/tensorfem/stress_recovery.py (add/add conflict).

Union strategy
--------------
main  : Torch production kernels  (q4_gauss_stresses / recover_nodal_stress /
        interpolate_nodal_field / locate_point / probe_errors / normalized_error ...)
m3402 : NumPy audit kernels       (gauss_point_stresses / nodal_recovered_stress /
        interpolate_nodal_stress / containing_elements / recovery_error ...)

Both are kept.  Three names exist on both sides and are unified by explicit
dispatch/precedence rules documented in the code:
  * EXTRAPOLATION_1D : np.ndarray, readable by both index styles
  * q4_shape         : numpy by default, Torch when dtype/device are given
  * element_areas    : type driven (torch input -> torch, array input -> numpy)
"""

import re
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m3873")
main_txt = (ROOT / ".scratch/main_sr.py").read_text()
theirs_txt = (ROOT / ".scratch/m3402_sr.py").read_text()

# --- split the m3402 blob into top-level function blocks -------------------
blocks = {}
for chunk in re.split(r"\n(?=def )", theirs_txt):
    match = re.match(r"def (\w+)\(", chunk)
    if match:
        blocks[match.group(1)] = chunk.rstrip() + "\n"

for name in (
    "_element_stress",
    "extrapolation_matrix",
    "local_coordinates",
    "containing_elements",
    "gauss_point_stresses",
    "nodal_recovered_stress",
    "interpolate_nodal_stress",
    "recovery_error",
):
    assert name in blocks, name

# The NumPy kernels must reach the cook_stress_reference implementations, not the
# Torch kernels of the same name that this module publishes.
stress_helper = blocks["_element_stress"]
assert "plane_stress_matrix()" in stress_helper and "q4_strain_matrix(xi, eta" in stress_helper
stress_helper = stress_helper.replace(
    "plane_stress_matrix()", "_reference_plane_stress_matrix()"
).replace("q4_strain_matrix(xi, eta", "_reference_q4_strain_matrix(xi, eta")
blocks["_element_stress"] = stress_helper


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:60])
    return text.replace(old, new)


def replace_function(text, name, new):
    pattern = re.compile(r"def %s\(.*?(?=\n\ndef |\Z)" % name, re.S)
    matches = pattern.findall(text)
    assert len(matches) == 1, (name, len(matches))
    return pattern.sub(lambda _m: new.rstrip() + "\n", text)


# --- 1. header: docstring union + imports ---------------------------------
union_docstring_note = """Union API (PR #79 merge)
------------------------
This module carries two numerically equivalent implementations that were developed in
parallel by two agents and are now published together, because both are in production
use:

* the **Torch** kernels (``q4_gauss_stresses``, ``recover_nodal_stress``,
  ``interpolate_nodal_field``, ``locate_point``, ``probe_errors``, ``normalized_error``)
  consumed by ``additional_benchmark_fields`` and
  ``scripts/cook_recovery_production_evidence.py``;
* the **NumPy** kernels (``gauss_point_stresses``, ``nodal_recovered_stress``,
  ``interpolate_nodal_stress``, ``containing_elements``, ``recovery_error``) consumed by
  ``cook_probe_protocol`` and ``cook_recovery_qualification``.

``q4_shape`` and ``element_areas`` exist in both families and are unified by dispatch:
``q4_shape`` returns a NumPy array unless a ``dtype``/``device`` is supplied (Torch), and
``element_areas`` follows the type of its ``nodes`` argument.  ``EXTRAPOLATION_1D`` is a
2x2 ``numpy.ndarray``, readable by both index styles.  The two families are cross-checked
against each other by ``scripts/cook_recovery_production_evidence.py`` and by
``tests/test_stress_recovery.py``.
"""
main_txt = replace_once(
    main_txt,
    "* It is an independent **Torch** implementation.  ``cook_probe_protocol.nodal_recovered_stress``\n"
    "  is a separate **NumPy** implementation written for the earlier audit; the two are\n"
    "  cross-checked against each other by ``scripts/cook_recovery_production_evidence.py``\n"
    "  and by ``tests/test_stress_recovery.py``.\n",
    union_docstring_note,
)

main_txt = replace_once(
    main_txt,
    "from __future__ import annotations\n\nimport math\n\nimport torch\n",
    "from __future__ import annotations\n\nimport math\n\n"
    "import numpy as np\nimport torch\n\n"
    "# NumPy reference kernels (the audit path).  Imported under private aliases so the\n"
    "# public Torch kernels of the same name below keep their signatures.\n"
    "from .cook_stress_reference import (\n"
    "    plane_stress_matrix as _reference_plane_stress_matrix,\n"
    "    q4_strain_matrix as _reference_q4_strain_matrix,\n"
    ")\n\n"
    "__all__ = [\n"
    '    "EXTRAPOLATION_1D",\n'
    '    "GAUSS",\n'
    '    "GAUSS_ORDER",\n'
    '    "NODE_ORDER",\n'
    '    "containing_elements",\n'
    '    "element_areas",\n'
    '    "extrapolation_matrix",\n'
    '    "gauss_point_stresses",\n'
    '    "interpolate_nodal_field",\n'
    '    "interpolate_nodal_stress",\n'
    '    "local_coordinates",\n'
    '    "locate_point",\n'
    '    "nodal_recovered_stress",\n'
    '    "normalized_error",\n'
    '    "probe_errors",\n'
    '    "q4_extrapolation_matrix",\n'
    '    "q4_gauss_stresses",\n'
    '    "q4_shape",\n'
    '    "q4_shape_derivative",\n'
    '    "q4_strain_matrix",\n'
    '    "recover_nodal_stress",\n'
    '    "recovery_error",\n'
    '    "von_mises",\n'
    "]\n",
)

# --- 2. EXTRAPOLATION_1D as a 2x2 ndarray ---------------------------------
main_txt = replace_once(
    main_txt,
    "_A = 0.5 + 0.5 / GAUSS\n_B = 0.5 - 0.5 / GAUSS\nEXTRAPOLATION_1D = ((_A, _B), (_B, _A))\n",
    "_A = 0.5 + 0.5 / GAUSS\n_B = 0.5 - 0.5 / GAUSS\n"
    "# 2x2 ndarray: the Torch path indexes a row and then an entry, the NumPy path\n"
    "# indexes ``[row, column]``; both read the same numbers.\n"
    "EXTRAPOLATION_1D = np.array(((_A, _B), (_B, _A)), dtype=np.float64)\n",
)

# --- 3. q4_shape: numpy by default, Torch on dtype/device -----------------
q4_shape_union = '''def q4_shape(xi: float, eta: float, dtype=None, device=None):
    """Bilinear shape functions in the Q4 node order.

    Union of two call conventions: with ``dtype``/``device`` (or a Torch ``dtype`` passed
    positionally) a ``torch.Tensor`` is returned -- the production Torch path; with no
    dtype/device a ``numpy.ndarray`` is returned -- the audit path used by
    ``cook_probe_protocol`` and ``cook_recovery_qualification``.
    """
    values = (
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    )
    if dtype is None and device is None:
        return np.array(values, dtype=np.float64)
    return torch.tensor(
        values, dtype=torch.float64 if dtype is None else dtype, device=device
    )
'''
main_txt = replace_function(main_txt, "q4_shape", q4_shape_union)

# --- 4. element_areas: type-driven dispatch ------------------------------
element_areas_union = '''def element_areas(nodes, elements):
    """Unsigned area of each Q4 element.

    Type driven: Torch inputs use the two-triangle cross-product sum ``(0,1,2)+(0,2,3)``,
    NumPy inputs use the shoelace rule on the node loop.  Both are exact for a convex
    bilinear quad (the only case either implementation is used with) and each preserves
    the numbers of the implementation it comes from.
    """
    if not isinstance(nodes, torch.Tensor):
        corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
        x, y = corners[:, :, 0], corners[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        )
    coordinates = nodes[elements]
    total = torch.zeros(len(elements), dtype=nodes.dtype, device=nodes.device)
    for a, b, c in ((0, 1, 2), (0, 2, 3)):
        ab = coordinates[:, b] - coordinates[:, a]
        ac = coordinates[:, c] - coordinates[:, a]
        total = total + 0.5 * torch.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    return total
'''
main_txt = replace_function(main_txt, "element_areas", element_areas_union)

# --- 5. append the NumPy audit section -----------------------------------
numpy_section = [
    "# ---------------------------------------------------------------------------",
    "# NumPy audit API (kept verbatim from the m3402 side of the PR #79 merge).",
    "# ---------------------------------------------------------------------------",
    "",
]
for name in (
    "_element_stress",
    "extrapolation_matrix",
    "local_coordinates",
    "containing_elements",
    "gauss_point_stresses",
    "nodal_recovered_stress",
    "interpolate_nodal_stress",
    "recovery_error",
):
    numpy_section.append(blocks[name])
    numpy_section.append("")

union = main_txt.rstrip() + "\n\n\n" + "\n".join(numpy_section).rstrip() + "\n"
(ROOT / "src/tensorfem/stress_recovery.py").write_text(union)
print("written lines:", union.count("\n"))
print("conflict markers:", sum(union.count(m) for m in ("<<<<<<<", "=======", ">>>>>>>")))