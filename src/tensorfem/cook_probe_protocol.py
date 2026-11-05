"""Probe-protocol audit for the Cook membrane Q4 stress field.

The Cook membrane stress qualification (``scripts/audit_cook_membrane_stress.py``)
compares the production Q4 stress at five material points against an independent Q8
Richardson-extrapolated reference and reports a residual von Mises error above the
3 % gate (5.02 % at n=64).  This module asks whether that residual is a property of
the discretisation or of the measurement protocol.

Two protocol defects are quantified here:

1. **Ill-posed probe sites.**  Three of the five canonical sample points sit on the
   parametric mid-lines ``xi=0.5`` / ``eta=0.25, 0.5, 0.75``.  On the structured
   ``n x n`` Q4 grid these points land exactly on element corner nodes whenever ``n``
   is a multiple of four, where the Q4 strain field is discontinuous: the sampled
   value depends on which of the (up to four) incident elements the floor/ceil
   tie-break selects.  ``one_sided_values`` returns every admissible one-sided value
   so that the spread can be measured instead of being hidden behind one tie-break.

2. **Recovery underestimation.**  A constant-per-element stress is O(h) accurate;
   the standard remedy is Gauss-point (Barlow) extrapolation to nodes followed by
   nodal averaging and re-interpolation.  ``nodal_recovered_stress`` implements that
   recovery without touching the production assembly or solver.

Nothing here mutates production paths; the module is measurement-only and every
returned number is derived from the production displacement field passed in.
"""

from __future__ import annotations

import math

import numpy as np

from .cook_stress_reference import (
    inverse_map,
    map_point,
    plane_stress_matrix,
    q4_strain_matrix,
)

from .stress_recovery import (  # noqa: F401  (re-exported for the audit API)
    EXTRAPOLATION_1D,
    GAUSS,
    containing_elements,
    element_areas,
    gauss_point_stresses,
    interpolate_nodal_stress,
    nodal_recovered_stress,
    q4_shape,
    von_mises,
)


def _stress_from_displacement(connectivity, coordinates, displacement, xi, eta) -> np.ndarray:
    matrix, _ = q4_strain_matrix(xi, eta, coordinates[connectivity])
    return plane_stress_matrix() @ matrix @ np.asarray(displacement).reshape(-1, 2)[
        connectivity
    ].ravel()


def one_sided_values(nodes, elements, displacement, point) -> dict:
    """Every one-sided Q4 stress value admissible at ``point``.

    ``spread_relative`` is the max pairwise von Mises difference normalised by the
    mean: it is the *irreducible* ambiguity of a pointwise probe at a discontinuous
    site and must be small before any error figure can be interpreted as accuracy.
    """
    values = []
    for hit in containing_elements(nodes, elements, point):
        stress = _stress_from_displacement(
            hit["nodes"],
            np.asarray(nodes, dtype=np.float64),
            displacement,
            hit["local"][0],
            hit["local"][1],
        )
        values.append(
            {
                "element": hit["element"],
                "local": hit["local"],
                "stress": [float(s) for s in stress],
                "von_mises": von_mises(stress),
            }
        )
    if not values:
        raise ValueError(f"point {point} is not contained in any element")
    magnitudes = [entry["von_mises"] for entry in values]
    mean = float(np.mean(magnitudes))
    spread = (max(magnitudes) - min(magnitudes)) / abs(mean) if len(magnitudes) > 1 else 0.0
    return {
        "point": [float(p) for p in np.asarray(point, dtype=np.float64)],
        "candidates": values,
        "candidate_count": len(values),
        "von_mises_min": min(magnitudes),
        "von_mises_max": max(magnitudes),
        "spread_relative": float(spread),
        "well_posed": bool(len(magnitudes) == 1 or spread < 1e-9),
    }


def normalized_error(value: float, reference: float, scale: float) -> float:
    """Same floor convention as ``cook_stress_reference.normalized_error`` (5 % of scale)."""
    return abs(value - reference) / max(abs(reference), 0.05 * scale)


def canonical_points(reference: dict) -> list[dict]:
    """The five reference sample points with their parametric and physical coordinates."""
    return [
        {
            "xi": point["xi"],
            "eta": point["eta"],
            "point": point["point"],
            "von_mises_reference": point["von_mises"]["extrapolated"],
            "component_reference": {
                key: point[key]["extrapolated"] for key in ("sxx", "syy", "sxy")
            },
        }
        for point in reference["points"]
    ]


def probe_grid_contact(divisions: int, xi: float, eta: float, tolerance: float = 1e-9) -> dict:
    """Whether a parametric probe lands exactly on a structured ``divisions`` grid node.

    A Q4 grid node is hit when ``xi*divisions`` and ``eta*divisions`` are both
    integers.  ``xi*divisions`` alone puts the probe on an element edge.
    """
    scaled = (xi * divisions, eta * divisions)
    on_edge = tuple(
        bool(abs(value - round(value)) < tolerance) for value in scaled
    )
    return {
        "divisions": divisions,
        "scaled": [float(scaled[0]), float(scaled[1])],
        "on_element_edge": on_edge,
        "on_grid_node": bool(on_edge[0] and on_edge[1]),
    }


def admissible_error_band(candidates, references, scale: float) -> dict:
    """Worst-site normalised error spanned by every admissible one-sided tie-break.

    ``candidates[i]`` lists the von Mises values the pointwise protocol may return at
    probe site ``i`` (:func:`one_sided_values`), ``references[i]`` is the reference value
    at that site and ``scale`` the global von Mises scale of the 5 % error floor.

    At a site that sits on an element edge the merged protocol's ``floor`` element choice
    is decided by the last bit of the parent-coordinate inversion, which is not
    reproducible across BLAS builds: the same source tree yields 0.0802 or 0.0943 at
    ``n=32`` depending on the machine.  A single residual number is therefore an artefact
    of the machine that produced it, while the band spanned by the admissible selections
    is reproducible — and an accuracy verdict has to hold over the whole band.
    """
    if len(candidates) != len(references):
        raise ValueError("one candidate list per reference value required")
    if not candidates:
        raise ValueError("at least one probe site required")
    per_site = []
    for values, reference in zip(candidates, references):
        if not len(values):
            raise ValueError("every probe site needs at least one admissible value")
        errors = [
            normalized_error(float(value), float(reference), float(scale)) for value in values
        ]
        per_site.append(
            {
                "candidate_count": len(errors),
                "min_error": float(min(errors)),
                "max_error": float(max(errors)),
            }
        )
    return {
        "scale": float(scale),
        "per_site": per_site,
        # Worst site under the most / least favourable admissible tie-break.
        "worst_site_min_error": float(max(site["min_error"] for site in per_site)),
        "worst_site_max_error": float(max(site["max_error"] for site in per_site)),
        "ambiguous_sites": int(sum(site["candidate_count"] > 1 for site in per_site)),
    }
