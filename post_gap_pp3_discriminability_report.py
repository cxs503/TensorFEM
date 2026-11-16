#!/usr/bin/env python3
"""Publish the GAP-PP-3 acceptance-discriminability note (markdown + figure)."""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
DOC = HERE / "docs" / "CASE-GAP-PP-3-ACCEPTANCE-DISCRIMINABILITY.md"
FIG = HERE / "artifacts" / "gap_pp3_acceptance_discriminability.png"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def main() -> int:
    token = login()
    png = base64.b64encode(FIG.read_bytes()).decode()
    payload = {
        "title": "GAP-PP-3 验收判据的可判定性：<10 % vs 上界对任何正确解都不可满足",
        "summary": ("把 GAP-PP-3 总账判据「塑性极限载荷 vs Ilyushin/上限分析 < 10 %」从『待重述』"
                    "变成可判定：锚点几何可采纳区间宽 3.600，任何正确解对上限的最小相对误差 "
                    "72.2 %，判据需把区间收紧 3.24 倍才有意义；新增离散 Johansen 机制 LP，"
                    "在满布正方形板上复现精确解 24 m_p/a^2（误差 0.00 %）作工具校验，"
                    "并诚实记录它在锚点几何上比闭合式金字塔更松（不收紧区间）。"),
        "scenario_key": "gap_pp3_ice_struct_plastic_set",
        "mission_id": 23068,
        "tags": ["TensorFEM", "GAP-PP-3", "plastic-limit", "upper-bound",
                 "verification-debt", "yield-line", "acceptance-criterion"],
        "markdown": DOC.read_text(),
        "figures": [{"name": "gap_pp3_acceptance_discriminability.png",
                     "caption": "左：锚点板格的塑性界（下界条带、金字塔/Johansen 上界、"
                                "机制 LP 16x8 与 24x12）；右：机制 LP 在满布正方形板上"
                                "复现精确 Johansen 极限 24 m_p/a^2。",
                     "png_base64": png}],
    }
    req = urllib.request.Request(BASE + "/tensor/reports",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print("report id:", out.get("id") or out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
