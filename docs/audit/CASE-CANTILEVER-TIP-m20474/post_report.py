#!/usr/bin/env python3
"""Post the m20474 CASE-CANTILEVER-TIP independent-recompute report to the platform."""
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = Path("docs/audit/CASE-CANTILEVER-TIP-m20474")
DOC = Path("docs/CASE-CANTILEVER-TIP-independent-recompute.md")


def post(path, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(path, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(BASE + path, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
ev = json.loads((HERE / "evidence.json").read_text())
o = ev["oracle"]
q8 = ev["q8_plane_stress"]
cross = ev["repo_parameter_crosscheck"]
checks = ev["checks"]
case = ev["repo_evidence"]["case_fem_cantilever_tip"]["verdict"]

figs = []
for name, caption in (
    ("fig_q8_convergence.png",
     "独立 Q8 平面应力收敛曲线（蓝）与仓库 HEX20 声明点（红）、仓库 fem_cantilever 参数下的独立 Q8（绿），灰虚线为 3% 锚点容差。"),
    ("fig_endload_idealisation.png",
     "端部加载理想化探针（抛物/均布/结点集中）在 Q8 32×4 上的端挠度，黑实线 EB、橙虚线 Timoshenko 基准。"),
):
    figs.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode((HERE / name).read_bytes()).decode(),
    })

markdown = DOC.read_text()
summary = (
    "verifier 席独立复算 CASE-CANTILEVER-TIP：用不 import tensorfem 的自写代码（sympy Castigliano 符号重导 + "
    "自写 4×4 Hermite 梁单元 + 自写 Q8 平面应力 8 结点单元与一致端部抛物线剪流加载）复算悬臂梁端挠度。"
    f"符号重导残差 {o['symbolic_residual']}，Simpson 积分与闭式差 {o['quadrature_vs_closed_form_rel']:.1e}；"
    f"独立梁单元对三次解相对误差 ≤{max(r['abs_relative_error_vs_euler_bernoulli'] for r in ev['beam_fe']):.1e}；"
    f"独立 Q8 连续体单调收敛，最细 64×4 对 Timoshenko 误差 {q8[-1]['abs_relative_error_vs_timoshenko']*100:.3f}% < 3%，"
    f"最粗 8×1 也仅 {q8[0]['abs_relative_error_vs_timoshenko']*100:.2f}%。"
    f"跨实现核对：用本审计代码复解仓库另一锚点 fem_cantilever 的 Q8 模型，与仓库声明值最差差 "
    f"{cross['worst_cross_implementation_rel_diff']*100:.4f}%（40×4）/8e-5（80×8），两实现收敛到同一极限。"
    f"仓库声明 verdict：参考网格误差 {case['abs_relative_error']*100:.3f}%（容差 {case['tolerance_rel']*100:.0f}%，passed={case['passed']}）被独立复算佐证。"
    "结论 disposition=approve with notes：三条非阻断注记（L/h≤10 时 κ=5/6 vs Cowper 0.8496 的约定、粗网格端部加载理想化、Q8 孤立结点陷阱）。"
)

report = {
    "title": "CASE-CANTILEVER-TIP 独立复算审计：零共享代码的三路验证（符号 Castigliano + 自写 Hermite 梁 + 自写 Q8 连续体）",
    "summary": summary,
    "scenario_key": "CASE-CANTILEVER-TIP",
    "mission_id": 20474,
    "tags": ["TensorFEM", "CASE-CANTILEVER-TIP", "independent-recompute", "verification-debt",
             "q8-plane-stress", "adversarial-review", "anchor-audit"],
    "markdown": markdown,
    "figures": figs,
}
res = post("/tensor/reports", report, token)
print(json.dumps(res)[:500])