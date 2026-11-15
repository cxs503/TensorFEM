#!/usr/bin/env python3
"""Publish the bound-admissibility audit report (anchor GAP-PP-3).

The markdown is the case document; the figure is the PNG written by
``scripts/gen_gap_pp3_bound_admissibility_figure.py``.  Nothing is transcribed
by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DOC = HERE / "docs" / "CASE-GAP-PP-3-BOUND-ADMISSIBILITY-AUDIT.md"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    markdown = DOC.read_text() + (
        "\n\n---\n\n*图 `artifacts/gap_pp3_bound_admissibility.png`：左为斑载荷板的四种界"
        "（红色为不可采的旧 Ilyushin 双正弦值），右为双正弦机制弯曲/扭率之比；"
        "由 `scripts/gen_gap_pp3_bound_admissibility_figure.py` 程序化生成。*\n")
    payload = {
        "title": "GAP-PP-3 板格斑载荷塑性极限：Ilyushin 上界基准不可采 —— 反例、修正与回归锚点",
        "summary": ("用全板简支方板精确解 24 m_p/a^2 作反例，证明验收判据的分母 "
                    "ilyushin_sine_upper_bound (19.74 m_p/a^2) 低于已知精确解、因而并非上界；"
                    "新增保留扭率的可采上界 johansen_twist_upper_bound（全板 31.0、长条 9.97 "
                    "m_p/a^2 均满足上界性质），斑载荷 0.5x0.5 的可采上界由 1.5152 MPa（不可采）"
                    "修正为 2.3800 MPa，配合金字塔界给出可采区间 [0.8188, 1.8422] MPa；"
                    "21+14 项 pytest 全绿；mesh 阶梯作业 cj3497 仍在排队，not_run 如实记录。"),
        "scenario_key": "gap_pp3_ice_struct_plastic_set",
        "mission_id": 21530,
        "tags": ["GAP-PP-3", "TensorFEM", "limit-analysis", "upper-bound-admissibility",
                 "Johansen", "patch-load", "ice-structure"],
        "markdown": markdown,
        "figures": [{"name": "gap_pp3_bound_admissibility.png",
                     "caption": ("左：斑载荷板四种界，红色为被判定不可采的旧 Ilyushin 双正弦值；"
                                 "右：双正弦机制的弯曲/扭率比（量级相当），说明漏掉扭率项不可接受。"),
                     "png_base64": base64.b64encode(
                         (ART / "gap_pp3_bound_admissibility.png").read_bytes()).decode()}],
    }
    token = login()
    req = urllib.request.Request(
        BASE + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        answer = json.loads(r.read())
    print(json.dumps(answer)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
