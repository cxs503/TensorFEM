"""Hemisphere response evidence and centre shell stresses recovered from DOFs."""

import torch
from .spherical_shell import hemisphere_with_hole
from .hemisphere_postprocess import build_hemisphere_post, _plain
from .plate import _shape


def surface_stress(nodes, elements, translation, rotation, *, radial_directors=False):
    """Outward/inward plane stresses at facet centres, in local surface bases.

    Each offset follows u(z) = u + z (rotation x director). The legacy flat
    projection uses the facet normal at all four corners. The optional smooth
    director follows the exact nodal radius of this spherical benchmark.
    """
    xyz = nodes[elements]
    outward = torch.linalg.cross(xyz[:, 2] - xyz[:, 0], xyz[:, 3] - xyz[:, 1], dim=-1)
    outward = torch.nn.functional.normalize(outward, dim=-1)
    if torch.any((outward * xyz.mean(1)).sum(1) <= 0):
        raise ValueError("hemisphere facets must be outward oriented")
    directors = (
        torch.nn.functional.normalize(xyz, dim=-1)
        if radial_directors
        else outward[:, None, :].expand_as(xyz)
    )
    du = torch.linalg.cross(rotation[elements], directors, dim=-1)
    _, dn = _shape(xyz.new_tensor(0.0), xyz.new_tensor(0.0))
    C = xyz.new_tensor([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 0.35]]) * (6.825e7 / (1 - 0.3**2))
    stress = []
    for z in (0.02, -0.02):
        face = xyz + z * directors
        tangent = face.transpose(1, 2) @ dn
        normal = torch.nn.functional.normalize(
            torch.linalg.cross(tangent[:, :, 0], tangent[:, :, 1], dim=-1), dim=-1
        )
        edge = face[:, 1] - face[:, 0]
        e1 = torch.nn.functional.normalize(
            edge - (edge * normal).sum(-1, keepdim=True) * normal, dim=-1
        )
        basis = torch.stack((e1, torch.linalg.cross(normal, e1, dim=-1)), 1)
        grad = dn @ torch.linalg.inv(basis @ tangent)
        deriv = basis @ (translation[elements] + z * du).transpose(1, 2) @ grad
        strain = torch.stack((deriv[:, 0, 0], deriv[:, 1, 1], deriv[:, 0, 1] + deriv[:, 1, 0]), 1)
        stress.append(strain @ C.T)
    return torch.stack(stress, 1)


def hemisphere_fields(n, drilling_factor=1e-6, *, radial_directors=False):
    result = hemisphere_with_hole(n, drilling_factor=drilling_factor)
    post = build_hemisphere_post(result)
    record = _plain(post)
    record["mesh"] = n
    record["drilling_factor"] = drilling_factor
    stress = surface_stress(
        result.nodes,
        result.elements,
        post.displacement,
        post.rotation,
        radial_directors=radial_directors,
    )
    vm = torch.sqrt(
        stress[:, :, 0] ** 2
        - stress[:, :, 0] * stress[:, :, 1]
        + stress[:, :, 1] ** 2
        + 3 * stress[:, :, 2] ** 2
    )
    record["stress_top_bottom"] = stress.tolist()
    record["von_mises_top_bottom"] = vm.tolist()
    return record


def hemisphere_field_report():
    cases = [hemisphere_fields(n) for n in (8, 12, 16, 20, 24, 28, 32, 40)]
    sensitivity = [
        (cases[-1] if f == 1e-6 else hemisphere_fields(40, f))["validation"]
        | {"drilling_factor": f}
        for f in (1e-7, 1e-6, 1e-5)
    ]
    last_change = abs(cases[-1]["probe"]["value"] / cases[-2]["probe"]["value"] - 1)
    passed = (
        cases[-1]["validation"]["passed"]
        and last_change < 0.03
        and all(s["passed"] for s in sensitivity)
    )
    return {
        "physical_case_id": "hemisphere-hole",
        "cases": cases,
        "sensitivity": sensitivity,
        "coarse_sensitivity": [
            (next(c for c in cases if c["mesh"] == 24) if f == 1e-6 else hemisphere_fields(24, f))[
                "validation"
            ]
            | {"drilling_factor": f}
            for f in (1e-7, 1e-6, 1e-5)
        ],
        "last_response_change": last_change,
        "stress_accuracy_qualified": False,
        "scope": "loaded-equator displacement qualified; face stresses require a separate same-side HEX20 comparison",
        "status": "response-qualified" if passed else "blocked",
    }
