"""Publish the element rank-deficiency audit as a platform progress report.

Reads the audit report (``docs/element-rank-deficiency-audit.md``) and the three
generated figures, downscales the figures, and POSTs a single report payload to
the platform API.  Credentials come from the environment
(``TENSOR_AGENT_USER`` / ``TENSOR_AGENT_PASS``) and the base URL from
``TENSOR_API_BASE`` (default ``http://127.0.0.1:8010/api/v1``).

Usage::

    PYTHONPATH=src python3 scripts/publish_element_rank_report.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from pathlib import Path

from PIL import Image

MISSION_ID = 2880
TITLE = "单元零能（秩亏）审计：Q4 Mindlin 板/壳的伪零能机构与假定剪应变修复"
SUMMARY = (
    "对 TensorFEM 全部可调用单元路径做零能（秩亏）审计：三维固体/膜 8 条路径秩充分（0 伪零能），"
    "Q4 Mindlin 板 SRI 核恒有 2 个伪零能机构，shell4 与 shell4_mitc 各继承 2 个；给出非经验修复"
    "（Dvorkin–Bathe 假定自然剪应变，opt-in），自由-自由模态零频个数由 4 恢复为 3，且已验收精度不变"
    "（简支 ω11 误差 +0.731% 逐位一致）。"
)
FIGURES = [
    (
        "element_rank_free_free_modes.png",
        "自由-自由方板频谱：SRI 剪应变场恒有 4 个零频模态，假定剪应变场（MITC）恢复为 3；"
        "右图为简支板 ω11 误差随 h 的收敛，两种剪应变场逐位一致。",
    ),
    (
        "element_rank_element_nullspace.png",
        "单元零能审计：解析期望零空间维数（蓝）与伪零能机构个数（红）。三维固体与膜单元全为 0，"
        "Q4 Mindlin 板 SRI 与其两个壳面元各有 2 个。",
    ),
    (
        "element_rank_mesh_scaling.png",
        "自由网格扫描：最软弹性特征值比随 h 下降（左），装配网格中伪零能模态个数（右）。",
    ),
]


def _encode(path: Path, max_width: int = 1000) -> str:
    image = Image.open(path).convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, int(image.height * max_width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    if buffer.tell() > 70_000:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode()


def build_payload(root: Path) -> dict:
    body = (root / "docs" / "element-rank-deficiency-audit.md").read_text()
    figures = [
        {
            "name": name,
            "caption": caption,
            "png_base64": _encode(root / "docs" / "assets" / name),
        }
        for name, caption in FIGURES
    ]
    return {
        "title": TITLE,
        "summary": SUMMARY,
        "scenario_key": "tensorfem-element-rank-audit",
        "mission_id": MISSION_ID,
        "tags": ["TensorFEM", "verification", "rank-deficiency", "hourglass", "Mindlin", "shell", "MITC4"],
        "markdown": "## 摘要\n\n" + SUMMARY + "\n\n" + body,
        "figures": figures,
    }


def _post(url: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    payload = build_payload(root)
    size_kb = len(json.dumps(payload)) // 1024
    login = _post(
        f"{base}/auth/login",
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )
    token = login.get("access_token") or login.get("token")
    result = _post(f"{base}/tensor/reports", payload, token)
    print(f"payload={size_kb} KB; report response keys: {sorted(result)}")
    print(json.dumps(result, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()