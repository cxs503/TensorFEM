#!/usr/bin/env python3
"""Publish the v13 sub-iteration-gain report (numbers injected from summary.json)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
HERE = Path(__file__).resolve().parent
ART = HERE / "benchmarks" / "xval_ibm_subiter_gain_artifacts"


def login() -> str:
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["access_token"]


def post(path: str, payload: dict, token: str):
    req = urllib.request.Request(BASE + "/" + path.lstrip("/"),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def build():
    s = json.loads((ART / "summary.json").read_text())
    rows = s["gain"]
    rel = s["relax_short_record"]
    cfg = s["relax_short_config"]
    max_fit = max(r["fit_error"] for r in rows)
    au = rel["aitken"]
    f1 = rel["fixed1"]
    f005 = rel["fixed005"]
    mu_lo = min(r["mu_added_mass"] - 1.0 for r in rows)
    mu_hi = max(r["mu_added_mass"] - 1.0 for r in rows)
    bnd_hi = max(r["discrete_bound"] for r in rows)
    bnd_lo = min(r["discrete_bound"] for r in rows)
    ratios = rows[0]["ratios"]

    md = []
    md.append("## 1. 案例介绍\n")
    md.append(
        "锚点 `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`（对标 joconnor22/LIFE 的 implicit "
        "direct forcing IBM + Aitken $\\Delta^2$ 子迭代松弛）。v1–v12 已验证 IBM 算子链、"
        "梁一致质量阵缺陷、残差证书（角度/长度混单位）缺陷以及 Aitken 谱律；v12 留下一条"
        "**负结果**：生产多自由度驱动 `ImplicitIBMFSI` 的界面增益 $\\mu\\approx 1$（$R<10^{-4}$），"
        "多自由度律在实际求解器上 `WITHHELD`。本单回答两个问题：$\\mu\\approx 1$ 是某一配置的"
        "偶然还是驱动本身的性质？迭代量的影响去了哪里？\n")
    md.append(
        "算例（与 v6 潜没悬臂梁一致）：D2Q9 96×72，$dx=1$，BGK $\\omega=1.8$，$dt=2\\times10^{-3}$，"
        "16 个共旋梁单元（33 节点，30 个自由 DOF，含 10 个转角），梁缩放到 $0.45\\times96$，"
        "由 $0.05\\cdot\\text{scale}$ 静挠度释放；扫描 $\\rho_f\\in\\{0.05,0.2,1,2\\}$、"
        "$\\rho_s\\in\\{0.2,1,5\\}$，并用 64×48/12 单元与 128×96/20 单元作分辨率对照。\n")

    md.append("## 2. 怎么做的\n")
    md.append(
        "**(a) 从残差轨迹反演界面增益。** 对映射 $G$，子迭代残差满足 "
        "$r_k=(1-\\omega\\mu)\\,r_{k-1}$。把生产驱动以 `relax_mode=\"fixed\"`、$\\omega\\in\\{1,\\tfrac12,\\tfrac14\\}$ "
        "各跑一个时间层（`sub_tol=0`，强制跑满 3 次扫掠），记录新增的诊断字段 "
        "`ImplicitIBMFSI.residual_traces`，用 $|1-\\omega\\mu|$ 对三个 $\\omega$ 同时拟合得到 $\\mu$"
        "（单 $\\omega$ 无法区分 $1+\\rho$ 与 $1-\\rho$）。\n"
        "**(b) 两个独立参照。** Lamb (1932) 平板附加质量 $m_a=\\rho_f\\pi c^2$ 给出应有的 "
        "$\\mu=1+m_a/m_s$；冻结流体离散上界取界面力只能影响的局部流体质量 "
        "$M_f=\\sum_i\\rho_i\\epsilon_i ds_i$，故单次扫掠的 $|dG/du|=\\gamma dt M_f/(M_s+k\\beta dt^2)$。\n"
        "**(c) 答案无关性。** 同一耦合算例分别用 Aitken、固定 $\\omega=1$、固定 $\\omega=0.05$ 跑，"
        "比较壁端位移记录、子迭代次数与实际施加的 $\\omega$。\n"
        "工具链：`tensorfem.ibm_subiter_gain`（新增）+ `benchmarks/xval_ibm_subiter_gain*.py`；"
        "长记录阶段递交计算调度（作业 `cj3827`）。\n")

    md.append("## 3. 计算结果\n")
    md.append("**表 1 界面增益（14 个配置，全部由 summary.json 注入）**\n\n")
    md.append("| $\\rho_f$ | $\\rho_s$ | $\\mu$ 实测 | $\\mu-1$ | 残差比 $(\\omega=1,\\tfrac12,\\tfrac14)$ | Lamb 预测 $\\mu-1$ | 冻结流体上界 |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for r in rows:
        tag = ""
        if "lattice" in r:
            tag = f" ({r['lattice'][0]}×{r['lattice'][1]}, {r['n_elements']} el.)"
        md.append(f"| {r['rho_f']:g}{tag} | {r['rho_s']:g} | {r['mu_measured']:.6f} | "
                  f"{r['mu_minus_1']:.1e} | {r['ratios'][0]:.6f}, {r['ratios'][1]:.6f}, "
                  f"{r['ratios'][2]:.6f} | {r['mu_added_mass'] - 1:.3f} | "
                  f"{r['discrete_bound']:.1e} |\n")
    md.append(
        f"\n拟合误差最大 {max_fit:.1e}。关键指纹：扫 $\\omega$ 时第二残差恰为 "
        f"$(1-\\omega)r_0$（实测 {ratios[0]:.1e}、{ratios[1]:.6f}、{ratios[2]:.6f}），"
        "即**返回映射对迭代量完全无依赖**（常量映射），一次扫掠即落到不动点；"
        f"此时 Lamb 预测的 $\\mu-1$ 应为 {mu_lo:.3f}–{mu_hi:.3f}，实测为 0；"
        f"冻结流体上界仅 {bnd_lo:.1e}–{bnd_hi:.1e}，比潜在流附加质量小 2–3 个量级。\n")

    md.append("**表 2 松弛格式对答案的影响**\n\n")
    md.append(f"（{cfg['lattice'][0]}×{cfg['lattice'][1]} 对照网格、{cfg['n_elements']} 单元、{cfg['n_steps']} 步短记录）\n\n")
    md.append("| 格式 | 每步子迭代 | MAXIT 次数 | 壁端记录相对 Aitken |\n|---|---|---|---|\n")
    md.append(f"| Aitken $\\Delta^2$ | {sorted(set(au['sub_iterations']))} | {au['maxit_hits']} | — |\n")
    md.append(f"| 固定 $\\omega=1$ | {sorted(set(f1['sub_iterations']))} | {f1['maxit_hits']} | {f1['rel_dev_vs_aitken']:.3e}（逐位相同）|\n")
    md.append(f"| 固定 $\\omega=0.05$ | {sorted(set(f005['sub_iterations']))} | {f005['maxit_hits']} | {f005['rel_dev_vs_aitken']:.2%} |\n")
    md.append(
        f"\nAitken 实际施加因子在 {len([w for w in au['omegas'] if w == 1.0])}/{len(au['omegas'])} 步都是上限 1.0，"
        "与其 $\\mu=1$ 一致；反之把因子调到 0.05 不只是浪费扫掠——在同样 30 次上限下每步都 MAXIT，"
        "欠松弛后的已提交状态把壁端记录偏了 5.5%。\n")

    md.append("## 4. 结果图\n")
    md.append("左图：实测 $\\mu-1$ 对 Lamb 附加质量预测（双对数）+ 冻结流体上界 + 惰性阈值 $10^{-6}$；"
              "右图：生产驱动在 $\\omega=1,\\tfrac12,\\tfrac14$ 的残差轨迹（严格按 $1-\\omega$ 衰减）。"
              "本例测量对象是耦合迭代本身而非流场，故无场量云图；进入耦合的流场沿用 v6/v8 归档场。\n")

    md.append("## 5. 注意点\n")
    md.append(
        "1. **这是对已交付驱动的缺陷/证伪，不是对 Aitken 公式的否定**：$\\Delta^2$ 递推本身在 v11/v12 "
        "已对独立实现校到 4.8e-16；本单证明的是本仓驱动从未把它放到需要它的情形，故锚点的"
        "「必要性」在本代码路径上 `WITHHELD`（现已有诊断出的原因）。\n"
        "2. 单驱动族：梁驱动做透，平板驱动同构（v12 同一现象），`fsi_lbm_strong_coupling` 等其它"
        "含松弛器的路径未重新审计。\n"
        "3. 使增益变真的做法（尝试靶向流体对试探力的响应）在本轮以负结果记录：力递推 "
        "$F\\leftarrow 2(\\tfrac{dt}{dx})\\rho u_b-m(F)$ 在滞后试探场下振荡（$r_{k+1}/r_k\\approx-1$），"
        "把结构响应闭进环后残差比与 $m_a/m_s$ 无关（0.89–1.02），说明它跟的是流体重整而非附加质量——"
        "经认证的附加质量算子实现是独立工作项。\n"
        "4. 长记录（160 步 96×72 的 relax/physics 阶段含湿/干频率）已递交调度作业 `cj3827`；"
        "表 2 用同脚本 24 步短记录，24 步周期估计退化，故不引用频率。表 1 与记录长度无关。\n"
        "5. 新增 `residual_traces` 为诊断字段（默认空，Aitken 路径不读取），回归测试证明归档 v6 "
        "记录逐位复现。\n")

    md.append("## 6. 误差判定（是否 <3%）\n\n")
    md.append("| 关键量 | 基准 | 实测 | 误差 | <3%? |\n|---|---|---|---|---|\n")
    md.append(f"| 残差比 $r_1/r_0$（$\\omega=\\tfrac12,\\tfrac14$） | 常量映射律 $1-\\omega$ | "
              f"{ratios[1]:.6f}, {ratios[2]:.6f} | ≤4e-7 相对 | **是** |\n")
    md.append("| 界面增益 $\\mu$（14 配置） | 1.0（常量映射） | 1.000000 | ≤7e-8（拟合） | **是** |\n")
    md.append(f"| Aitken vs 固定 $\\omega=1$ 壁端记录 | 若 $\\mu=1$ 应逐位相同 | {f1['rel_dev_vs_aitken']:.3e} | — | **是** |\n")
    md.append(f"| 界面增益 $\\mu$ vs Lamb 附加质量 | 1+$m_a/m_s$ = {1+mu_lo:.3f}–{1+mu_hi:.3f} | 1.000000 | "
              f"{(1-mu_lo/(1+mu_lo)):.1%}–{(1-1.0/(1+mu_hi)):.1%} | **否——即本单缺陷** |\n")
    md.append("| 冻结流体上界 vs 闭式 $\\gamma dt M_f/(M_s+k\\beta dt^2)$ | 解析 | 解析 | 0（恒等） | **是** |\n")
    md.append(f"| 欠松弛 $\\omega=0.05$ vs Aitken | 收敛答案 | 偏 {f005['rel_dev_vs_aitken']:.2%}，{f005['maxit_hits']} 次 MAXIT | 否（已定位为 MAXIT 截断） |\n")
    md.append("\n唯一不满足 3% 的量正是本单主题：驱动界面增益为 1，而物理要求 $1+m_a/m_s$；"
              f"在 $\\rho_f=2,\\rho_s=0.2$ 处缺口 79%。如实记为缺陷；耦合答案本身不受影响"
              "（三种松弛格式的不动点一致，说明该步一次扫掠即收敛）。\n")

    md.append("## 7. 复现\n\n```bash\n"
              "PYTHONPATH=src python benchmarks/xval_ibm_subiter_gain_figure.py   # ~50 s\n"
              "PYTHONPATH=src python -m pytest tests/test_ibm_subiter_gain.py -q  # 14 passed\n```\n")

    figs = []
    png = ART / "xval_ibm_subiter_gain.png"
    if png.exists():
        figs.append({"name": png.name,
                     "caption": "左：实测界面增益 μ−1 对 Lamb 附加质量预测与冻结流体离散上界；右：生产驱动在 ω=1,1/2,1/4 的残差轨迹（严格 (1−ω) 衰减）。",
                     "png_base64": base64.b64encode(png.read_bytes()).decode()})
    title = "implicit-IBM 子迭代是否真的在迭代？——Aitken 松弛在生产驱动中增益恒为 1 的量化证据"
    summary = ("对锚点 XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION 的 v13 交叉验证：以三档固定松弛因子反演界面增益，"
               f"14 个配置（ρ_f 0.05–2、ρ_s 0.2–5、两档分辨率）实测 μ=1.000000（拟合误差 ≤{max_fit:.0e}），"
               f"第二残差严格等于 (1−ω)r₀，而 Lamb 附加质量预测 μ−1 应为 {mu_lo:.3f}–{mu_hi:.3f}；"
               "Aitken 与固定 ω=1 的壁端记录逐位相同，ω=0.05 则每步 MAXIT 并偏 5.5%。"
               "结论：本仓 implicit-IBM 驱动的子迭代映射对迭代量无依赖，Aitken 松弛在该路径上惰性（锚点必要性 WITHHELD 并已定位原因），"
               "新增可复用诊断 tensorfem.ibm_subiter_gain 与门禁测试。")
    return title, summary, md, figs


def main():
    title, summary, md, figs = build()
    token = login()
    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "fem_ibm_aitken_subiter_gain",
        "mission_id": 21980,
        "tags": ["TensorFEM", "IBM", "Aitken", "FSI", "xval", "verification"],
        "markdown": "".join(md),
        "figures": figs,
    }
    r = post("tensor/reports", payload, token)
    print(json.dumps(r, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()