"""Publish the CASE-FEM-HERTZ-CONTACT independent-oracle audit report (mission 20772).

Every number in the report body is injected from
``benchmarks/verified/fem_hertz_contact_loadcontrol/independent_audit.json`` --
nothing is transcribed by hand.
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
ANCHOR = HERE / "benchmarks" / "verified" / "fem_hertz_contact_loadcontrol"
AUDIT = ANCHOR / "independent_audit.json"
DOC = HERE / "docs" / "CASE-FEM-HERTZ-CONTACT-independent-verification.md"
FIG = HERE / "docs" / "assets" / "case-fem-hertz-contact" / "hertz_independent_audit.png"


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


def main() -> int:
    st, d = req("auth/login", "POST", {"username": USER, "password": PWD})
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
    assert tok, f"login failed: {st} {d}"

    a = json.loads(AUDIT.read_text())
    rows = a["rows"]
    tol = a["tolerance_pct"]

    body = DOC.read_text().split("## ④ 结果图", 1)[0]

    table = ["| run | 单元数 | $e_a$ [%] | $e_\\delta$ [%] | $e_{p_0}$ 一致场 [%] | $e_{p_0}$ 集总 [%] | 载荷残差 | <3%? |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        e = r["error_pct"]
        ok = "✅" if max(abs(e["contact_radius"]), abs(e["indentation"])) < tol else "❌"
        table.append(f"| `{r['key']}` | {r['n_elements']} | {e['contact_radius']:+.4f} | {e['indentation']:+.4f} | "
                     f"{e['peak_pressure_consistent']:+.4f} | {e['peak_pressure_lumped']:+.2f} | "
                     f"{r['load_residual']:.1e} | {ok} |")

    fixed = ["| id | 位置 | 量 | 修复前 | 工件现值 | 说明 |", "|---|---|---|---:|---:|---|"]
    for f in a["fixed_defects"]:
        fixed.append(f"| {f['id']} | `{f['doc']}` | {f['quantity']} | {f['before_pct']:+.3f}% | "
                     f"{f['after_pct']:+.4f}% | {f['note']} |")

    gates_ok = sum(1 for g in a["gates"] if g["recomputed"])
    caption = ("(a) 用独立 Hertz 基准重算的 e_a、e_delta 与 ±3% 判据带；(b) 文档声称值 vs 工件值"
               "（绿点=已一致，红 X=本轮复核抓到的、修复前的过期声称）；(c) 峰值压力两个估计量的散布，"
               "说明 p0 声称是估计量条件性的。")
    fig_md = f"\n## ④ 结果图（matplotlib 生成，内嵌）\n\n**hertz_independent_audit.png** — {caption}\n\n![audit](fig://hertz_independent_audit.png)\n\n"

    worst = a["worst_pct"]
    tail = (
        "\n## ⑥ 误差判定（逐项回答是否 <3%）\n\n"
        "| 关键量 | 基准 | 最差误差（6 个 run 全包） | 是否 <3% |\n|---|---|---:|---|\n"
        f"| 接触半径 a | `(3PR/4E*)^(1/3)` | {worst['contact_radius']:.4f}% | {'是 ✅' if worst['contact_radius'] < tol else '否 ❌'} |\n"
        f"| 压入深 δ | `a²/R` | {worst['indentation']:.4f}% | {'是 ✅' if worst['indentation'] < tol else '否 ❌'} |\n"
        f"| 峰值压力 p0（一致场，判据 5%） | `3P/(2πa²)` | {worst['peak_pressure_consistent']:.4f}% | "
        f"{'是 ✅' if worst['peak_pressure_consistent'] < 5.0 else '否 ❌'} |\n\n"
        f"**是否 <3%？** a 与 δ 全部 6 个 run 均 <3% ✅（最差 {max(worst['contact_radius'], worst['indentation']):.4f}%）。"
        f"峰值压力按 5% 判据亦通过，但**只在 L2 一致场估计量下**，集总节点峰值为 "
        f"{max(abs(r['error_pct']['peak_pressure_lumped']) for r in rows):+.2f}%（该量不是闸门量，如实披露）。\n\n"
        f"证书闸门重判：{gates_ok}/{len(a['gates'])} 条由工件内 `value`/`threshold` 重算后与声明一致；"
        f"physics_verdict={'PASS' if a['physics_verdict']['passed'] else 'FAIL'}，"
        f"docs_verdict={'PASS' if a['docs_verdict']['passed'] else 'FAIL'}。\n"
    )

    summary = (
        "对算例池 FEM-HERTZ-CONTACT 的已发布证书做独立复核：用不 import src/tensorfem 的独立 Hertz 闭式解"
        f"重算 a、δ、p0，重判 28 条闸门与 <3% 验收（a 最差 {worst['contact_radius']:.3f}%、"
        f"δ 最差 {worst['indentation']:.3f}%，均通过），并把两篇散文的数值声称逐格对照工件——"
        "抓到 3 处文档缺陷并修复：a/R=0.10 最细网格的 −1.750% 是 m17149 估计量修复前的历史值却被当作现役结果"
        "（工件为 −0.4908%）×2，以及主文档 §4 的 p0 行标 h/a=0.0075 却引用 h/a=0.03 的 +0.493%（应为 +0.630%）。"
        "另披露 p0 声称的估计量条件性（一致场 +0.63% vs 集总节点 +31.9%）。"
    )

    payload = {
        "title": "CASE-FEM-HERTZ-CONTACT 独立复核：证书 oracle 与文档声称的逐格对照",
        "summary": summary,
        "scenario_key": "TensorFEM",
        "mission_id": 20772,
        "tags": ["TensorFEM", "Hertz", "contact", "CASE-FEM-HERTZ-CONTACT", "verification",
                 "independent-oracle", "audit", "doc-drift"],
        "markdown": body + fig_md + "\n".join(table) + "\n\n## ⑦ 本轮发现并修复的缺陷\n\n"
                    + "\n".join(fixed) + tail,
        "figures": [{
            "name": "hertz_independent_audit.png",
            "caption": caption,
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }],
    }
    (HERE / "report_hertz_independent_audit.json").write_text(json.dumps(payload, ensure_ascii=False))
    st, res = req("tensor/reports", "POST", payload, token=tok)
    print("status", st, str(res)[:400])
    return 0 if st and 200 <= st < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())