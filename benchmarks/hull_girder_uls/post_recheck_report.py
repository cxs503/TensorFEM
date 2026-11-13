"""Publish the W26 recheck as a platform report (six elements + figures).

The markdown body is the generated ``README.md`` of the artifact directory, so
the prose cannot disagree with the JSON.  Figures are embedded as base64 PNG.

Usage::

    python benchmarks/hull_girder_uls/post_recheck_report.py          # dry run
    python benchmarks/hull_girder_uls/post_recheck_report.py --post   # publish

Credentials come from ``TENSOR_API_BASE`` / ``TENSOR_AGENT_USER`` /
``TENSOR_AGENT_PASS``; nothing is written to disk except the payload in /tmp.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "papers" / "artifacts" / "w26" / "hull-girder-ultimate-strength"
VERIFY = Path("/tmp/report.json")

FIGURES = (
    ("fig_recheck_quote_matrix.png",
     "图 1 锚件 19 条条目的核销粒度地图（verbatim / normalised / restated / WITHHELD）"
     "与所引页-实定位页对照"),
    ("fig_recheck_wave_coefficients.png",
     "图 2 Cw 分支接缝（L = 300/350 m）与 CSR 中垂波浪弯矩对 UR S11 独立表述的逐点对比"),
    ("fig_recheck_utilisation.png",
     "图 3 按系数加权的 CSR 利用率（含 f_beta 替代值）与 Smith 逐步崩溃容量沿站位分布"),
)


def payload() -> dict:
    artifact = json.loads((ARTIFACT_DIR / "csr_uls_recheck.json").read_text())
    markdown = (ARTIFACT_DIR / "README.md").read_text()
    figures = []
    for name, caption in FIGURES:
        blob = (ARTIFACT_DIR / name).read_bytes()
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(blob).decode()})
    summary = artifact["quote_recheck"]["summary"]
    return {
        "title": ("W26 CSR ULS 判据与 Smith 法基准的锚件二次核销："
                  "6/6 判据可核销、3/3 文档 sha256 复算一致"),
        "summary": (f"{summary['transcribed']} 条转录条目全部在重抓的 IACS 原件中"
                    f"重新定位（{summary['pass']} 逐字 + {summary['pass_restated']} "
                    f"重述型、0 缺失），六条 ULS 判据 100 % 可核销，Table 2 的 "
                    f"γS/γW 由坐标级重读独立复原，锚件 3 份文档 sha256 与 Smith "
                    f"M_u 基准均从零复算一致；发现 f_beta 转录颗粒度过粗（替代值 "
                    f"1.0 会翻转 2 例裁决）。"),
        "scenario_key": "paper-exp-hull-girder-ultimate-strength-1",
        "mission_id": 17566,
        "tags": ["TensorFEM", "CSR", "ULS", "hull-girder", "anchor-recheck",
                 "Smith-collapse", "IACS", "W26"],
        "markdown": markdown,
        "figures": figures,
    }


def login(base: str) -> str:
    data = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    request = urllib.request.Request(f"{base}/auth/login", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read())
    return body.get("access_token") or body["token"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--post", action="store_true", help="publish to the platform")
    parser.add_argument("--out", default=str(VERIFY))
    args = parser.parse_args(argv)

    body = payload()
    Path(args.out).write_text(json.dumps(body))
    print(f"payload: {args.out} ({Path(args.out).stat().st_size} bytes, "
          f"{len(body['figures'])} figures)")
    if not args.post:
        return 0

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    token = login(base)
    request = urllib.request.Request(
        f"{base}/tensor/reports", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        print(response.status, response.read().decode("utf-8", "replace")[:400])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
