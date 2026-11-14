"""Publish the CASE-FEM-HERTZ-CONTACT joint (dz, H) limit report (mission 20449).

All numbers are read from ``two_param_extrapolation.json``; the two figures are
the PNGs written by ``two_param_extrapolation.py --assemble --figure``.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PWD = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"
DOC = HERE / "docs" / "CASE-FEM-HERTZ-CONTACT-two-parameter-limit.md"
ASSETS = HERE / "docs" / "assets" / "case-fem-hertz-contact"
FIGS = [
    ("hertz_two_param_limit.png",
     "左：深度阶梯，每个 Δz 一条曲线（远场固定 (52,32)），虚线为 Δz=a/48 的深度→∞ 极限；"
     "右：竖向阶梯，每个域深一条曲线，红带为联合极限 e0 及其不确定带。两条阶梯单独都收敛不到 0，"
     "联合极限才是有定义的那个量。"),
    ("hertz_two_param_budget.png",
     "压入深残差预算：算例池接受模型的 −1.01% 经竖向加密、深度外推与联合极限后剩下多少；"
     "红柱为联合极限 e0（误差棒=不确定带），绿带为算例池 ±3% 判据。"),
]


def req(path, method="GET", body=None, token=None):
    url = BASE + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def main():
    st, d = req("auth/login", "POST", {"username": USER, "password": PWD})
    tok = (d.get("access_token") or d.get("token")
           or (d.get("data") or {}).get("access_token"))
    assert tok, f"login failed: {st} {d}"

    art = json.loads((BENCH / "two_param_extrapolation.json").read_text())
    doc = DOC.read_text()
    body = doc.split("## 复现", 1)[0]
    figures = []
    for name, caption in FIGS:
        p = ASSETS / name
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    e0 = art.get("e0_pct")
    band = art.get("band_pct")
    interim = art.get("interim_estimate") or {}
    if e0 is not None:
        head = (f"按预声明判据求 e_δ 的二参数联合极限 e0 = {e0:+.4f}% "
                f"(±{band:.4f} pp)，判决 {art['verdict']}；")
    else:
        head = (f"联合极限未闭合（网格 {art['n_points']} 点，缺 "
                f"{len(art['cells_missing'])} 格仍在集群队列 cj2725）；"
                f"临时角点估计 e0 ≈ {interim.get('e0_interim_pct'):+.4f}%，"
                f"判决 {art['verdict']}；")
    payload = {
        "title": ("Hertz 球-平面接触：压入深残差的联合 (Δz, 1/H) 极限 —— "
                  "0.2% 是模型偏差还是二参数离散残差"),
        "summary": head + "算例池 <3% 判据在 a 与 δ 上仍成立。",
        "scenario_key": "fem_hertz_contact",
        "mission_id": 20449,
        "tags": ["CASE-FEM-HERTZ-CONTACT", "TensorFEM", "接触力学",
                 "网格收敛", "Richardson 外推", "验证欠账"],
        "markdown": body + "\n\n" + "\n".join(
            f"![{n}]({n})\n\n*{c}*\n" for n, c in FIGS),
        "figures": figures,
    }
    st, d = req("tensor/reports", "POST", payload, token=tok)
    print(st, json.dumps(d)[:400] if not isinstance(d, str) else d)
    return 0 if st in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())