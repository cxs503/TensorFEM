"""API handler for stiffened-plate validation endpoint.

Provides ``POST /api/v1/calculations/structural/stiffened_plate_validation``
as a self-contained function that can be mounted on any ASGI/WSGI framework.
The ship-performance-platform imports this handler and registers the route.
"""
from __future__ import annotations

import json

from .stiffened_plate_validation import (
    DEFAULT_PARAMS,
    DEFAULT_T_PROFILE,
    TStiffenerProfile,
    run_stiffened_plate_validation,
)


def handle_stiffened_plate_validation(body: dict) -> dict:
    """Handle POST /api/v1/calculations/structural/stiffened_plate_validation.

    Request body (all optional, defaults to the benchmark specification):
    {
        "length": 1.0,
        "width": 0.5,
        "thickness": 0.01,
        "young": 2.0e11,
        "poisson": 0.3,
        "density": 7850.0,
        "t_profile": {
            "web_height": 0.04,
            "web_thickness": 0.005,
            "flange_width": 0.03,
            "flange_thickness": 0.005,
            "spacing": 0.5
        },
        "mesh_fine": 20,
        "mesh_medium": 12,
        "mesh_coarse": 8,
        "error_tolerance": 0.03,
        "mac_threshold": 0.95
    }

    Returns a JSON-serialisable dict with all anchor results, GCI, and scaling.
    """
    params = dict(DEFAULT_PARAMS)
    for key in ("length", "width", "thickness", "young", "poisson", "density"):
        if key in body:
            params[key] = float(body[key])

    t_profile = DEFAULT_T_PROFILE
    if "t_profile" in body:
        tp = body["t_profile"]
        t_profile = TStiffenerProfile(
            web_height=float(tp["web_height"]),
            web_thickness=float(tp["web_thickness"]),
            flange_width=float(tp["flange_width"]),
            flange_thickness=float(tp["flange_thickness"]),
            spacing=float(tp["spacing"]),
        )

    kwargs = {
        "t_profile": t_profile,
        "error_tolerance": float(body.get("error_tolerance", 0.03)),
        "mac_threshold": float(body.get("mac_threshold", 0.95)),
    }
    for key in ("mesh_fine", "mesh_medium", "mesh_coarse"):
        if key in body:
            kwargs[key] = int(body[key])

    result = run_stiffened_plate_validation(**params, **kwargs)

    def _anchor_to_dict(a):
        return {
            "name": a.name,
            "reference_type": a.reference_type,
            "mode_labels": [list(ml) for ml in a.mode_labels],
            "fe_frequencies_hz": list(a.fe_frequencies_hz),
            "reference_frequencies_hz": list(a.reference_frequencies_hz),
            "relative_errors": list(a.relative_errors),
            "max_error": a.max_error,
            "passed": a.passed,
            "mac_diagonal": list(a.mac_diagonal),
        }

    gci = result.gci_result
    scaling = result.scaling_result

    return {
        "status": "ok",
        "all_passed": result.all_passed,
        "anchors": {
            "simply_supported_plate": _anchor_to_dict(result.anchor_plate),
            "single_stiffened_plate": _anchor_to_dict(result.anchor_single_stiffened),
            "double_stiffened_plate": _anchor_to_dict(result.anchor_double_stiffened),
        },
        "gci": {
            "refinement_ratio": gci.refinement_ratio,
            "observed_order": gci.observed_order,
            "gci_fine": gci.gci_fine,
            "gci_coarse": gci.gci_coarse,
            "frequencies": list(gci.frequencies),
        },
        "dimensional_scaling": {
            "base_frequency_hz": scaling.base_frequency,
            "scaled_frequency_hz": scaling.scaled_frequency,
            "expected_ratio": scaling.expected_ratio,
            "actual_ratio": scaling.actual_ratio,
            "relative_error": scaling.relative_error,
        },
        "parameters": {
            "length": params["length"],
            "width": params["width"],
            "thickness": params["thickness"],
            "young": params["young"],
            "poisson": params["poisson"],
            "density": params["density"],
        },
    }


def handle_stiffened_plate_validation_json(raw_body: str | bytes) -> str:
    """Convenience wrapper: parse JSON body, return JSON string."""
    body = json.loads(raw_body) if raw_body else {}
    return json.dumps(handle_stiffened_plate_validation(body))
