"""Publish the CASE-CANTILEVER-TIP elastica mesh-independence report (mission 20390)."""
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
DOC = HERE / "docs" / "CASE-CANTILEVER-TIP-elastica-mesh-independence.md"
ASSETS = HERE / "docs" / "assets" / "elastica-cantilever-mesh"
RESULTS = HERE / "artifacts" / "elastica_cantilever" / "results.json"
FLOOR = HERE / "artifacts" / "elastica_cantilever" / "floor_study.json"

FIGS = [
    ("load_deflection.png",
     "载荷-挠度：共旋梁 FE（n=4/8/16，全部收敛）与 elastica 椭圆积分基准、线性理论 α/3 的对比；"
     "大挠度段明显低于线性外推，FE 三条曲线在图上不可分。"),
    ("mesh_convergence.png",
     "端挠度相对误差 vs 单元数（双对数，四个载荷水平）：修复后全部单调，斜率与 O(n^-2) 参考线一致；"
     "修复前 n=8/16 的点在旧报告中是「未收敛」而无法落点。"),
    ("residual_floor.png",
     "根因：收敛后实际达到的 ‖r‖ 随网格增长（1.5e-10@n=2 → 1.2e-9@n=16），旧判据的绝对容差 1e-10 "
     "落在该地板之下，而已收敛状态被误判；灰点线为修复后采纳的舍入地板（仍要求步长≤1e-12）。"),
    ("prefix_replay.png",
     "判决性证据：同一能量、同一差分切线，仅把收敛判据换回旧版（重放），α=1 时能达到的载荷因子从 "
     "1.000 跌到 0.004/0.000，端挠度误差 99.6%/100%；修复后恢复 1.000。"),
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


def main() -> int:
    results = json.loads(RESULTS.read_text())
    floor = json.loads(FLOOR.read_text())
    doc = DOC.read_text()

    st, d = req("auth/login", "POST", {"username": USER, "password": PWD})
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
    if not tok:
        raise SystemExit(f"login failed: {st} {d}")

    bench = results["benchmark"]
    tol = bench["tolerance"]
    table = {}
    for c in bench["cases"]:
        table.setdefault(c["alpha"], {})[c["elements"]] = c
    worst_fine = max(c["deflection_error"] for a in table for n, c in table[a].items() if n >= 8)

    err_lines = ["| α | n | 端挠度误差 | <3%? | 轴向回缩短 | <3%? |", "|---:|---:|---:|:---:|---:|:---:|"]
    for a in sorted(table):
        for n in sorted(table[a]):
            c = table[a][n]
            err_lines.append(
                f"| {a:g} | {n} | {c['deflection_error']:.4%} | "
                f"{'是' if c['deflection_error'] < tol else '**否**'} | "
                f"{c['shortening_error']:.4%} | "
                f"{'是' if c['shortening_error'] < tol else '**否**'} |")

    replay = "\n".join(
        f"| {r['alpha']:g} | {r['elements']} | {r['load_factor']:.3f} | "
        f"{r['deflection_error']:.2%} | "
        f"{next(x['load_factor'] for x in floor['fixed_convergence_test'] if x['alpha'] == r['alpha'] and x['elements'] == r['elements']):.3f} | "
        f"{next(x['deflection_error'] for x in floor['fixed_convergence_test'] if x['alpha'] == r['alpha'] and x['elements'] == r['elements']):.4%} |"
        for r in floor["prefix_replay"])

    figures = []
    fig_md = "\n## ④ 结果图（matplotlib 生成，内嵌）\n\n"
    for name, caption in FIGS:
        png = ASSETS / name
        figures.append({"name": name, "caption": caption,
                        "png_base64": base64.b64encode(png.read_bytes()).decode()})
        fig_md += f"**{name}** — {caption}\n\n![{name}](fig://{name})\n\n"

    body = doc.split("## 4. Error vs benchmark", 1)[0]
    tail = (
        "\n## ⑤ 注意点\n\n"
        "- 修复只动收敛判据（两分支：相对力残差，或「残差落在舍入地板 且 Newton 步长 ≤1e-12(1+‖q‖)」），"
        "**能量、差分切线、载荷步进一字未改**；旧判据已能通过的 n=2/4 位移逐位不变（0.7030%/0.1643%/"
        "4.5844%/1.0147% 与旧报告一致），所以本单不移动任何已发表数字。\n"
        "- 舍入分支的适用范围有界：地板随装配规模增长，实测到 n=32（残差 1.39e-9，误差 0.0219%）；"
        "更细网格应重设 RFLOOR 而非加大 max_iterations。\n"
        "- 基准是**不可伸长** Euler-Bernoulli elastica（无横向剪切、无轴向可伸长），不到 α→∞ 的贴卷极限；"
        "FE 侧是专用 2 节点共旋梁，**不得外推**到 Shell4 载荷控制。\n"
        "- 顺带更正旧报告 §1 参数表两处数字：EI = Ebt³/12 = **100**（旧写 10），故 P = **1.0·α**（旧写 0.1·α）；"
        "无量纲 α 与全部结论不受影响，脚本已改为从几何派生。\n\n"
        "## ⑥ 误差判定（是否 <3%）\n\n" + "\n".join(err_lines) +
        f"\n\n**是否 <3%？是**：`n≥8` 的全部 rungs 端挠度误差最大 **{worst_fine:.4%}**（α=1,n=8 为 0.0407%），"
        f"最细 `n=16` 最大 0.0636%；门禁 `qualified={bench['qualified']}`、"
        f"`all_converged={bench['all_converged']}`、`monotone_refinement={bench['monotone_refinement']}`。\n"
        "粗网格 n=2 的 α=5/10（3.48%/4.58%）与 α=1 轴向回缩（3.88%）**超 3%，如实保留未剔除**，"
        "它们不是最细网格，不进入 qualified 判定，但已逐项列出。\n\n"
        "### 判决性重放（旧判据 vs 修复后，同一能量/切线）\n\n"
        "| α | n | 旧载荷因子 | 旧端挠度误差 | 新载荷因子 | 新端挠度误差 |\n"
        "|---:|---:|---:|---:|---:|---:|\n" + replay + "\n\n"
        "旧判据在 n=8/16 上把载荷因子压到 0.004–0.013（甚至 0.000），端挠度误差 95–100%；"
        "这正是已合并报告里「细网格不收敛」的原始现象，**归因是收敛判据的舍入地板，不是切线条件数**。\n"
    )

    payload = {
        "title": "CASE-CANTILEVER-TIP｜elastica 档网格无关性修复：收敛判据舍入地板根因（旧报告的「细网格不收敛」被证伪归因）",
        "summary": "把悬臂梁端载大挠度档从「n≥8 不收敛、门禁 qualified=False」修到 n=2…32 全收敛、"
                   "端挠度误差单调 4.58%→0.0219%、门禁 True；根因是绝对残差容差 1e-10 低于双精度舍入地板"
                   "（随网格增长），非切线条件数；重放实验复现旧现象（载荷因子 0.004→0.000），"
                   "已收敛状态数位逐位不变。顺带更正 EI/P 两处参数误写。",
        "scenario_key": "CASE-CANTILEVER-TIP",
        "mission_id": 20390,
        "tags": ["TensorFEM", "CASE-CANTILEVER-TIP", "elastica", "convergence-gate",
                 "adversarial-review", "defect-fix", "mesh-independence"],
        "markdown": body + fig_md + tail,
        "figures": figures,
    }
    st, d = req("tensor/reports", "POST", payload, token=tok)
    print(st, json.dumps(d)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())