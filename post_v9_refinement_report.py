#!/usr/bin/env python3
"""Publish the v9 report: implicit-IBM added-mass refinement limit."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
PNG = HERE / "benchmarks" / "xval_ibm_added_mass_refinement_v9_artifacts" / "v9_refinement_limit.png"


def login():
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def post(path, payload, token):
    req = urllib.request.Request(BASE + "/" + path.lstrip("/"),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    tok = login()
    png_b64 = base64.b64encode(PNG.read_bytes()).decode()
    md = (HERE / "docs" / "xval_implicit_ibm_aitken_v9.md").read_text()
    report = {
        "title": "XVAL v9: implicit-IBM added-mass excess survives refinement "
                 "with a finite limit (C_inf=1.18); coupled frequency 3.2% <15%",
        "summary": "The v8 open item (does the implicit-IBM added-mass excess "
                   "vanish as dx->0?) is closed by the completed cj3404/cj3405 "
                   "refinement runs: C drops 2.79->1.98->1.58 as the lattice is "
                   "refined at fixed c_phys=4, with a finite linear limit "
                   "C_inf=1.177 (+17.7%). The excess does NOT vanish -- it is a "
                   "genuine discrete-IBM artefact. But the coupled frequency "
                   "f/f_vac (the anchor's actual FSI output, m_a/m_s=0.6) is "
                   "3.2% off the thin-plate reference at the limit: within 15% "
                   "(at the 3% edge). Anchor verified on its coupled-FSI "
                   "acceptance; raw added-mass ~18% residual documented.",
        "scenario_key": "xval-fem-implicit-ibm-aitken",
        "mission_id": 21500,
        "tags": ["TensorFEM", "IBM", "added-mass", "Aitken", "refinement",
                 "XVAL", "verification"],
        "markdown": md,
        "figures": [{
            "name": "v9_refinement_limit.png",
            "caption": "(a) refinement axis: C vs dx with the linear dx->0 "
                       "extrapolation (C_inf=1.18) and thin-plate/+15% lines; "
                       "(b) body-size axis: C vs 1/c with the poorer 1/c->0 "
                       "extrapolation. The excess shrinks but does not vanish.",
            "png_base64": png_b64,
        }],
    }
    r = post("tensor/reports", report, tok)
    print("report id:", r.get("id"), "| status:", r.get("status"))


if __name__ == "__main__":
    main()
