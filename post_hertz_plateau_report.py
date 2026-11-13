"""Publish the CASE-FEM-HERTZ-CONTACT plateau-mechanism report (mission 18800)."""
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
DOC = HERE / "docs" / "CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md"
FIGS = [
    ("hertz_plateau_mechanism.png",
     "左：径向 h/a 阶梯（前序判据，Δz=a/6 固定）几乎不动，竖向 dz 阶梯（本单，Δr 逐位不变）"
     "把压入深误差从 −0.480% 拉到 −0.299%；右：远场层数 16→64 加密对三个量都无影响（H_far 被否定）。"),
    ("hertz_plateau_arms.png",
     "各机制臂的三量误差柱状图（±3% 判据带虚线）：竖向加密使 p0 误差从 +0.589% 降到 −0.039%"
     "（15×），同时接触半径估计量变号、放大到 +1.36%——精度瓶颈已从求解器转到边缘估计量。"),
    ("hertz_plateau_depth.png",
     "域深阶梯：Δz=a/24（本单）对比前序 Δz=a/6 的 Richardson 极限 −0.428%；"
     "竖向加密后外推极限明显下移，且近场量 a 对域深不敏感（近场量不该被截断污染）。"),
]


def req(path, method="GET", body=None, token=None):
    url = BASE + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=180) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def main():
    st, d = req("auth/login", "POST", {"username": USER, "password": PWD})
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
    assert tok, f"login failed: {st} {d}"

    far = json.loads((HERE / "benchmarks/verified/fem_hertz_contact_loadcontrol/"
                            "farfield_discretisation.json").read_text())
    zres = json.loads((HERE / "benchmarks/verified/fem_hertz_contact_loadcontrol/"
                             "nearfield_resolution.json").read_text())
    conv = json.loads((HERE / "benchmarks/verified/fem_hertz_contact_loadcontrol/"
                             "converged_model.json").read_text())
    gates = json.loads((HERE / "benchmarks/verified/fem_hertz_contact_loadcontrol/"
                               "gates_plateau.json").read_text())
    doc = DOC.read_text()
    # report = generated report body + the figure section (both programmatic)
    body = doc.split("## 误差判定", 1)[0]
    fig_md = "\n## ④ 结果图（matplotlib 生成，内嵌）\n\n"
    figures = []
    for name, caption in FIGS:
        png = HERE / "docs" / "assets" / "case-fem-hertz-contact" / name
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(png.read_bytes()).decode(),
        })
        fig_md += f"**{name}** — {caption}\n\n![{name}](fig://{name})\n\n"

    err_lines = []
    for g in gates["gates"]:
        err_lines.append(f"| `{g['id']}` | {g['value']:.6g} {g['operator']} {g['threshold']:g} "
                         f"| {'PASS' if g['passed'] else 'FAIL'} |")
    tail = (
        "\n## ⑥ 误差判定（是否 <3%）\n\n"
        "| 闸门 | 值 | 判定 |\n|---|---|---|\n" + "\n".join(err_lines) +
        "\n\n算例池 `<3%` 判据：前序证书 `a`、`δ` 四个闸门（a/R=0.05/0.10 最细网格）全部 PASS"
        "（−0.445%/−1.010%/−1.750%/−1.010%），本单**未改动**接受模型，只新增机制归因闸门。\n\n"
        f"**是否全部 <3%？** `a`、`δ` 均 <3% ✅。本单预声明的更严阈值 `|e_δ| ≤ 0.15%`（残余偏差被完全解释）"
        f"**未达成 ❌**（{{E_INF}}%）：竖向加密解释掉 "
        f"{conv['share_of_prior_bias_explained_pct']:.1f}% 的前序 −0.428%，剩余部分机制未定位，如实记为开放项。\n"
    )
    e_inf = conv["extrapolation"]["e_inf_pct"]
    tail = tail.replace("{E_INF}", f"{e_inf:+.3f}")

    summary = (
        "算例池 FEM-HERTZ-CONTACT（a、δ vs Hertz，<3%）的压入深平台（−1.01%，外推 −0.428%）机制归因："
        "前序『网格无关模型偏差』的证据链不成立——判据阶梯只加密径向、竖向细带钉死在 Δz=a/6。"
        f"只加密竖向（径向逐位不变）即可把 e_δ 从 −0.480% 移到 −0.299%、p0 从 +0.589% 移到 −0.039%；"
        f"远场拉伸臂被否定（移动 {far['arm_verdict']['shift_pp']:.4f} pp）；"
        f"竖向加密 + 深度外推后残余 {e_inf:+.3f}%（预声明 0.15% 未达成，如实记为开放项）。"
    )

    payload = {
        "title": "CASE-FEM-HERTZ-CONTACT：压入深平台机制归因（只加密 r、从未加密 z）",
        "summary": summary,
        "scenario_key": "TensorFEM",
        "mission_id": 18800,
        "tags": ["TensorFEM", "Hertz", "contact", "CASE-FEM-HERTZ-CONTACT",
                 "mesh-anisotropy", "verification"],
        "markdown": body + fig_md + tail,
        "figures": figures,
    }
    (HERE / "report_hertz_plateau.json").write_text(json.dumps(payload, ensure_ascii=False))
    st, res = req("tensor/reports", "POST", payload, token=tok)
    print("status", st, str(res)[:400])
    return 0 if st and 200 <= st < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())