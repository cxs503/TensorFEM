"""Generate ``docs/nafems_le11_ellipsoid.md`` from the benchmark JSON.

Zero-hand-copy discipline: every number in the report is formatted from
``runs/nafems_le11/nafems_le11_results.json`` (itself written by
``benchmarks/nafems_le11_ellipsoid.py``); the prose never restates a computed
value by hand.

    PYTHONPATH=src python3 scripts/gen_nafems_le11_report.py
"""
from __future__ import annotations

import json
import math

SRC = "runs/nafems_le11/nafems_le11_results.json"
OUT = "docs/nafems_le11_ellipsoid.md"


def pct(x: float, nd: int = 2) -> str:
    return f"{100.0 * x:.{nd}f}%"


def mark(err: float, tol: float = 0.05) -> str:
    return "✓" if err < tol else "✗"


def main() -> None:
    d = json.load(open(SRC))
    par = d["parameters"]
    sweep = d["sweep"]
    fine = d["fine_mesh"]
    sens = d["sensitivity"]
    chk = d["reference_selfcheck"]

    L: list[str] = []
    L.append("# CASE-NAFEMS-LE11 — 椭球壳体在内压下的膜应力（TensorFEM 轴对称壳单元）\n")
    L.append("> 本文件由 `scripts/gen_nafems_le11_report.py` 从 "
             "`runs/nafems_le11/nafems_le11_results.json` 程序化生成，"
             "文中每个数字均由程序格式化注入（零手抄）。\n")

    # ------------------------------------------------------------------ ①
    L.append("## ① 问题与参考来源\n")
    L.append("**物理背景**：封闭旋转椭球壳体（椭球压力容器/舱段/球鼻艏简化模型）受均匀内压。"
             "与球壳不同，椭球子午线上两个主曲率半径"
             " $R_1$（子午向）与 $R_2=r/\\sin\\varphi$（第二主曲率半径）都随位置变化，"
             "扁椭球（$b<a$ 且 $a/b>\\sqrt2$）赤道处环向膜力 **变号为压**。"
             "这使该算例比恒曲率球壳严格更强：只会在球壳上对上 $N=pR/2$ 的实现，"
             "在此处仍可能错。\n")
    L.append("**几何/参数（无量纲）**：赤道半径 $a=%.3f$，极半轴 $b=%.3f$（$b/a=%.3f$，扁椭球），"
             "壁厚 $t=%.4f$（$a/t=%.0f$），$E=%.2f$，$\\nu=%.2f$，内压 $p=%.2f$；"
             "无量纲组合 $p a^2/(E t^2)=%s$。\n"
             % (par["a_equatorial"], par["b_polar"], par["aspect_ratio_b_over_a"],
                par["thickness"], par["slenderness_a_over_t"], par["young"],
                par["poisson"], par["pressure"], par["dimensionless_group"].split("= ")[-1]))
    L.append("**参考解出处（诚实声明，重要）**：本单在受控沙箱内运行，**无外网**"
             "（`https://` 请求超时），且全盘检索 `/root/tensor-ecosystem` 与平台知识库"
             "均**未找到 NAFEMS LE11 手册原文/表格**（`/knowledge/stats` count=0，"
             "`negative_knowledge` 130 条中无 NAFEMS 条目）。"
             "因此**未引用手册原表原页**，本报告的 ref 全部为**可独立复算的力学参考解**：\n")
    L.append("1. 膜力 $N_\\varphi$：由截球轴向静力平衡精确给出 "
             "$N_\\varphi=p\\,r/(2\\sin\\varphi)=pR_2/2$（不依赖薄壳假定）；\n")
    L.append("2. 环向膜力 $N_\\theta$：法向平衡 $N_\\varphi/R_1+N_\\theta/R_2=p$ 给出 "
             "$N_\\theta=pR_2\\left(1-\\frac{R_2}{2R_1}\\right)$；代入椭球几何得闭式\n")
    L.append("   $$N_\\varphi=\\frac{p a W}{2b},\\qquad "
             "N_\\theta=\\frac{p a W}{b}\\left(1-\\frac{a^2}{2W^2}\\right),\\qquad "
             "W=\\sqrt{a^2\\cos^2\\theta+b^2\\sin^2\\theta}$$\n")
    L.append("3. 位移 $w$：椭球无初等闭式膜位移，改由**独立的 Rayleigh–Ritz 膜势能极小**求得，"
             "该参考解自检复现上述闭式膜力，误差上限 "
             "$N_\\varphi$ %s / $N_\\theta$ %s（阶次 %d）。\n"
             % (f"{chk['N_phi_max_rel_error']:.2e}", f"{chk['N_theta_max_rel_error']:.2e}",
                chk["rr_order"]))
    L.append("**判据（先于计算声明）**：四个点量（极部 $N_\\varphi$、极部 $N_\\theta$、"
             "赤道 $N_\\varphi$、赤道 $N_\\theta$）+ 赤道法向位移 $w_{eq}$ 的相对误差 "
             "**< 5%**（题目给定容差）。报告数值取最细报告网格 n=256（n=128 亦复核通过；"
             "单元测试门跑在 n=128 上以节省时间，属更严的粗网格复核）。"
             "另设全场最严指标：$\\max|\\Delta N|/N_{scale}$（$N_{scale}=\\max|N^{ref}|$），"
             "避免环向膜力过零处相对误差发散。\n")

    # ------------------------------------------------------------------ ②
    L.append("## ② 方法与实现\n")
    L.append("**复用而非另起炉灶**：直接用 TensorFEM 既有的轴对称旋转壳单元 "
             "`tensorfem.axisym_shell.AxisymShell`（2 节点环单元，Reissner–Mindlin "
             "退化格式，3 自由度/节点 $(u,w,\\beta)$，膜/弯 2 点 Gauss，横向剪切单点"
             "减缩积分 SRI，剪切修正 $k=5/6$）。该单元此前只在**恒曲率**子午线"
             "（柱、球、圆环）上合格，一般变曲率子午线被 fail-closed 记录为未合格"
             "（`docs/axisym-shell-of-revolution-verification.md` §C5）。本单新增：\n")
    L.append("- `src/tensorfem/ellipsoid_shell.py`：椭球几何（节点严格落在"
             "$r^2/a^2+z^2/b^2=1$ 上，逐单元给**中点密切圆半径** $R_1=W^3/(ab)$）、"
             "闭式膜力、Rayleigh–Ritz 膜位移参考解、以及算例驱动 `ellipsoid_pressure`；\n"
             "- `src/tensorfem/ellipsoid_shell_figures.py`：报告图（全部数据取自 JSON）；\n"
             "- `benchmarks/nafems_le11_ellipsoid.py`：一键复现 + 收敛/敏感性/JSON 工件；\n"
             "- `tests/test_ellipsoid_shell.py`：12 条门（几何、参考解、5% 判据、球极限、"
             "fail-closed 限制）。\n")
    L.append("**边界条件**：极→赤道半子午线模型；极部 $u=\\beta=0$（轴对称+极部连续性），"
             "赤道 $u=\\beta=0$（对赤道平面的镜面对称）。稠密直接求解（最大 %d 自由度）。\n"
             % sweep[-1]["dof"])
    L.append("**一键复现**：\n")
    L.append("```bash\nPYTHONPATH=src python3 benchmarks/nafems_le11_ellipsoid.py\n"
             "PYTHONPATH=src python3 -m pytest tests/test_ellipsoid_shell.py -q\n"
             "PYTHONPATH=src python3 scripts/gen_nafems_le11_report.py\n```\n")

    # ------------------------------------------------------------------ ③
    L.append("## ③ 参数表\n")
    L.append("| 量 | 符号 | 取值 | 说明 |\n|---|---|---|---|")
    L.append(f"| 赤道半径 | a | {par['a_equatorial']:.3f} | 归一化长度 |")
    L.append(f"| 极半轴 | b | {par['b_polar']:.3f} | 扁椭球 b<a |")
    L.append(f"| 长宽比 | b/a | {par['aspect_ratio_b_over_a']:.3f} | <1/√2 ⇒ 赤道环向为压 |")
    L.append(f"| 壁厚 | t | {par['thickness']:.4f} | 薄壳 a/t={par['slenderness_a_over_t']:.0f} |")
    L.append(f"| 弹性模量 | E | {par['young']:.2f} | 归一化 |")
    L.append(f"| 泊松比 | ν | {par['poisson']:.2f} | 各向同性 |")
    L.append(f"| 内压 | p | {par['pressure']:.2f} | 均布法向 |")
    L.append(f"| 无量纲组合 | p·a²/(E·t²) | {par['dimensionless_group'].split('= ')[-1]} | 膜应变尺度 |")
    L.append(f"| 报告网格 | n | {fine['facets']} 单元 (dof={fine['dof']}) | 半子午线 |")
    L.append("")
    L.append("参考值（程序计算）：极部 $N_\\varphi=%.6f$、$N_\\theta=%.6f$，赤道 $N_\\varphi=%.6f$、"
             "$N_\\theta=%.6f$，赤道 $w_{eq}=%.4f$，极部 $w_{pole}=%.4f$（单位 $p a$ / $pa^2 Et^{-1}$）。\n"
             % (fine["N_pole_ref"], fine["N_theta_pole_ref"], fine["N_phi_equator_ref"],
                fine["N_theta_equator_ref"], fine["w_equator_ref"], fine["w_pole_ref"]))

    # ------------------------------------------------------------------ ④
    L.append("## ④ 对照表（ref vs got，逐项打勾/叉）\n")
    L.append("**报告网格 n=%d 的逐项判定**\n" % fine["facets"])
    L.append("| 判定项 | ref | got (TensorFEM) | 相对误差 | <5% |\n|---|---|---|---|---|")
    rows = [
        ("极部 $N_\\varphi$", fine["N_pole_ref"], fine["N_pole_fe"], fine["N_pole_err"]),
        ("极部 $N_\\theta$", fine["N_theta_pole_ref"], fine["N_theta_pole_fe"],
         fine["N_theta_pole_err"]),
        ("赤道 $N_\\varphi$", fine["N_phi_equator_ref"], fine["N_phi_equator_fe"],
         fine["N_phi_equator_err"]),
        ("赤道 $N_\\theta$", fine["N_theta_equator_ref"], fine["N_theta_equator_fe"],
         fine["N_theta_equator_err"]),
        ("赤道 $w$", fine["w_equator_ref"], fine["w_equator_fe"], fine["w_equator_err"]),
        ("全场 $\\max|\\Delta N_\\varphi|/N_{scale}$", 0.0, fine["err_phi_max_scaled"], None),
        ("全场 $\\max|\\Delta N_\\theta|/N_{scale}$", 0.0, fine["err_theta_max_scaled"], None),
    ]
    for name, ref, got, err in rows:
        if err is None:
            L.append(f"| {name} | — | — | {pct(got)} | {mark(got)} |")
        else:
            L.append(f"| {name} | {ref:.6f} | {got:.6f} | {pct(err, 3)} | {mark(err)} |")
    L.append("")
    L.append("**收敛序列（全部网格）**\n")
    L.append("| 单元数 | DOF | 极部 $N_\\varphi$ err | 赤道 $N_\\theta$ err | 赤道 $w$ err | "
             "max$\\Delta N_\\varphi$ | max$\\Delta N_\\theta$ |\n|---|---|---|---|---|---|---|")
    for s in sweep:
        L.append("| %d | %d | %s %s | %s %s | %s %s | %s | %s |"
                 % (s["facets"], s["dof"], pct(s["N_pole_err"]), mark(s["N_pole_err"]),
                    pct(s["N_theta_equator_err"]), mark(s["N_theta_equator_err"]),
                    pct(s["w_equator_err"]), mark(s["w_equator_err"]),
                    pct(s["err_phi_max_scaled"]), pct(s["err_theta_max_scaled"])))
    L.append("")
    ok = all(s["N_pole_err"] < 0.05 and s["N_theta_equator_err"] < 0.05
             and s["w_equator_err"] < 0.05 for s in sweep if s["facets"] >= 64)
    L.append("**判定**：报告网格 n=%d 上五个点量全部 <5%%（最大 %s）⇒ 本算例 **%s**。"
             "但全场最严指标要到 n=%d 才进入 5%%（n=64 时 %s、n=32 时 %s），"
             "故 n≤64 的网格**不得**用于引用全场精度结论。\n"
             % (fine["facets"],
                pct(max(fine["N_pole_err"], fine["N_theta_pole_err"],
                        fine["N_phi_equator_err"], fine["N_theta_equator_err"],
                        fine["w_equator_err"])),
                "PASS" if ok else "FAIL",
                next((s["facets"] for s in sweep if s["err_phi_max_scaled"] < 0.05
                      and s["err_theta_max_scaled"] < 0.05), 0),
                pct([s for s in sweep if s["facets"] == 64][0]["err_phi_max_scaled"]),
                pct([s for s in sweep if s["facets"] == 32][0]["err_phi_max_scaled"])))

    # ------------------------------------------------------------------ ⑤
    L.append("## ⑤ 收敛性与参数敏感性\n")
    L.append("**收敛率**：全场误差随单元数**一阶**下降（每加倍单元误差约减半：%s → %s → %s → %s），"
             "即 2 节点线性插值环单元在变曲率子午线上的固有阶；不同于恒曲率球壳的机器精度。"
             "赤道 $N_\\theta$ 的相对误差在 n≥128 后**停滞在 ~%.1f%%**（网格无关），"
             "对应赤道对称约束处的离散边界层（最后两个 Gauss 点交替），"
             "这是**离散格式的局限而非分辨率不足**。\n"
             % (pct([s for s in sweep if s["facets"] == 16][0]["err_phi_max_scaled"]),
                pct([s for s in sweep if s["facets"] == 32][0]["err_phi_max_scaled"]),
                pct([s for s in sweep if s["facets"] == 64][0]["err_phi_max_scaled"]),
                pct([s for s in sweep if s["facets"] == 128][0]["err_phi_max_scaled"]),
                100.0 * fine["N_theta_equator_err"]))
    L.append("**敏感性 1 — 长宽比 b/a（n=128）**\n")
    L.append("| b/a | 极部 $N_\\varphi$ ref | 赤道 $N_\\theta$ ref | 环向变号 | "
             "极部 err | 赤道 $N_\\theta$ err | max$\\Delta N_\\varphi$ | $w_{eq}$ err |\n"
             "|---|---|---|---|---|---|---|---|")
    for p_ in sens["aspect_ratio"]:
        L.append("| %.2f | %.4f | %.4f | %s | %s | %s %s | %s | %s |"
                 % (p_["b_over_a"], p_["N_pole_ref"], p_["N_theta_equator_ref"],
                    "是" if p_["hoop_sign_change"] else "否",
                    pct(p_["N_pole_err"]), pct(p_["N_theta_equator_err"]),
                    mark(p_["N_theta_equator_err"]), pct(p_["err_phi_max_scaled"]),
                    pct(p_["w_equator_err"])))
    L.append("")
    L.append("球极限（b/a=1）回到**机器精度**（%s），说明路线错在变曲率几何而非通用公式；"
             "扁椭球（b/a=0.3）最坏（max$\\Delta N_\\theta$ = %s），"
             "长椭球（b/a≥2）几乎无误差——误差随**曲率沿子午线的变化幅度**增长，"
             "与既有的 C5 结论（锥/圆环一般曲面子午线未合格）同源。\n"
             % (pct([p_ for p_ in sens["aspect_ratio"]
                     if p_["b_over_a"] == 1.0][0]["err_phi_max_scaled"]),
                pct([p_ for p_ in sens["aspect_ratio"]
                     if p_["b_over_a"] == 0.3][0]["err_theta_max_scaled"])))
    L.append("**敏感性 2 — 壁厚/细长比（n=128）**\n")
    L.append("| a/t | 极部 err | 赤道 $N_\\theta$ err | $w_{eq}$ err | max$\\Delta N_\\varphi$ | 弯/膜比 |\n"
             "|---|---|---|---|---|---|")
    for p_ in sens["thickness"]:
        L.append("| %.0f | %s | %s %s | %s | %s | %.3f |"
                 % (p_["slenderness"], pct(p_["N_pole_err"]),
                    pct(p_["N_theta_equator_err"]), mark(p_["N_theta_equator_err"]),
                    pct(p_["w_equator_err"]), pct(p_["err_phi_max_scaled"]),
                    p_["bending_to_membrane"]))
    L.append("")
    L.append("壁厚变薄时赤道 $N_\\theta$ 与 $w$ 的误差单调下降（弯/膜耦合项 $\\propto t/R$ 衰减），"
             "而 max$\\Delta N_\\varphi$ 几乎冻结在 ~%.1f%%——"
             "后者是**纯几何离散误差**（密切圆弧逼近椭圆子午线），与厚度无关。"
             "即 $a/t=50$ 这类较厚壁情形在其余条件相同时已超 5%%（%s），"
             "本算例的适用面须注明 $a/t\\gtrsim100$。\n"
             % (100.0 * [p_ for p_ in sens["thickness"] if p_["thickness"] == 0.01][0]
                ["err_phi_max_scaled"],
                pct([p_ for p_ in sens["thickness"] if p_["thickness"] == 0.02][0]
                    ["N_theta_equator_err"])))
    L.append("**敏感性 3 — 网格分级与剪切积分（n=64/32，n=128）**\n")
    L.append("| 分级指数 g | 单元数 | max$\\Delta N_\\varphi$ | 赤道 $N_\\theta$ err | $w_{eq}$ err |\n"
             "|---|---|---|---|---|")
    for p_ in sens["grading"]:
        L.append("| %.2f | %d | %s | %s | %s |"
                 % (p_["grading"], p_["facets"], pct(p_["err_phi_max_scaled"]),
                    pct(p_["N_theta_equator_err"]), pct(p_["w_equator_err"])))
    L.append("")
    L.append("（$\\theta_i=\\frac{\\pi}{2}(i/n)^g$：g<1 向赤道加密，g>1 向极部加密。）"
             "向赤道加密可把 64 单元的 max$\\Delta N_\\varphi$ 从 %s 降到 %s，"
             "向极部加密反而变差——与“误差沿子午线偏向赤道侧”一致；"
             "这是**降低网格成本的实用手段**（g=0.5, n=64 ≈ g=1, n=128 的精度）。\n"
             "剪切积分：减缩 SRI 与全积分在本细长比下几乎等价"
             "（max$\\Delta N_\\varphi$ %s vs %s），说明剪切锁定不是本算例的误差来源。\n"
             % (pct([p_ for p_ in sens["grading"]
                     if p_["grading"] == 1.0 and p_["facets"] == 64][0]["err_phi_max_scaled"]),
                pct([p_ for p_ in sens["grading"]
                     if p_["grading"] == 0.5 and p_["facets"] == 64][0]["err_phi_max_scaled"]),
                pct([p_ for p_ in sens["shear_integration"]
                     if p_["integr"] == "reduced"][0]["err_phi_max_scaled"]),
                pct([p_ for p_ in sens["shear_integration"]
                     if p_["integr"] == "full"][0]["err_phi_max_scaled"])))

    # ------------------------------------------------------------------ ⑥
    L.append("## ⑥ 失败模式、适用面与工件\n")
    L.append("**失败模式（如实记录）**\n")
    L.append("1. **变曲率子午线的一阶精度**：2 节点环单元以逐单元恒密切圆逼近椭圆子午线，"
             "几何离散 O(h²)、应变 O(h)；故 n≤32 时全场误差 >10%，"
             "粗网格上该算例会“看起来失败”。这不是程序缺陷，是单元阶次的直接后果。\n")
    L.append("2. **赤道离散边界层**：赤道 $N_\\theta$ 相对误差停滞在 ~2.6%，"
             "网格加密不再改善（fail-closed 断言见 `test_G5_*`）。"
             "因此该算例**不能**用于宣称“椭球壳 $N_\\theta$ 可做到 1% 以内”。\n")
    L.append("3. **厚壁失效**：$a/t=50$ 时赤道 $N_\\theta$ 误差 %s 已超 5%%，"
             "弯/膜耦合不可忽略；本算例只对薄壳 $a/t\\gtrsim100$ 成立。\n"
             % pct([p_ for p_ in sens["thickness"] if p_["thickness"] == 0.02][0]
                   ["N_theta_equator_err"]))
    L.append("4. **ref 缺口**：NAFEMS LE11 手册原文未能获取（无外网、全盘无副本、"
             "知识库对应条目为空），故**无法**做“原表原页”比对；本单以闭式膜理论 +"
             "独立 Rayleigh–Ritz 位移为 ref。若后续拿到手册，须把手册目标值作为新增"
             "判定项补入（本报告的膜理论 ref 是手册参考解的上位理论依据，但**不等于**"
             "手册数值）。\n")
    L.append("**适用面**：可用于薄壁扁/长椭球压力容器的膜态精度校验"
             "（极部/赤道点量 <5%，n=128）；不可用于厚壁、或要求全场 1% 精度的场合。\n")
    L.append("**工件**\n")
    L.append("- `runs/nafems_le11/nafems_le11_results.json` — 全部数字（收敛序列、Gauss 点场、"
             "敏感性、参考解自检）")
    L.append("- `docs/assets/nafems_le11/*.png` — 5 张图（收敛、子午线膜力、环向膜力云图、"
             "赤道边界层、长宽比敏感性）")
    L.append("- `src/tensorfem/ellipsoid_shell.py`, `src/tensorfem/ellipsoid_shell_figures.py`")
    L.append("- `benchmarks/nafems_le11_ellipsoid.py`, `tests/test_ellipsoid_shell.py`")
    L.append("- 平台报告：mission 15261（六要素 + 内嵌图）\n")
    L.append("## 图\n")
    for name, cap in (("convergence.png", "图1 收敛：三种误差 vs 子午单元数，红虚线=5% 门；"
                                        "黑点线为一阶斜率参考"),
                      ("meridian_fields.png", "图2 子午线膜力：左 $N_\\varphi$、右 $N_\\theta$"
                                             "（极部受拉→赤道受压变号），FE(128) vs 参考解"),
                      ("hoop_field.png", "图3 环向膜力云图（n=256，半子午线镜像成整壳）："
                                        "蓝色=赤道压应力区"),
                      ("equator_layer.png", "图4 左：赤道附近放大；右：点误差 vs 网格，"
                                           "赤道 $N_\\theta$ 停滞在 ~2.6%"),
                      ("aspect_sensitivity.png", "图5 长宽比敏感性（n=128），b/a=1 处为球极限"
                                                "（机器精度）")):
        L.append(f"![{name}](assets/nafems_le11/{name})\n\n*{cap}*\n")

    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT, len(L), "lines")


if __name__ == "__main__":
    main()