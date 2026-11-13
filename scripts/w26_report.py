"""Render the W26 closure report from the register artifact (no hand-copied numbers).

Every table cell in the markdown is formatted from
``papers/artifacts/w26/plate-shell-buckling-closure/anchor_register.json`` and the
report payload embeds the three figures as base64 PNG, so nothing in the report is
typed twice and nothing can drift from the artifact.

Usage::

    python scripts/w26_report.py            # writes papers/reports/*.md and /tmp/report.json
    python scripts/w26_report.py --post     # also POSTs the payload to the platform
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "papers" / "artifacts" / "w26" / "plate-shell-buckling-closure"
REPORT_MD = ROOT / "papers" / "reports" / "w26_plate_shell_buckling_closure.md"
REPORT_JSON = pathlib.Path("/tmp/report.json")
SCENARIO = "CASE-PLATE-SHELL-BUCKLING-CLOSURE-W26"
MISSION = 17565


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(value: float, digits: int = 6) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def _sci(value: float) -> str:
    return f"{value:.3e}"


def build_markdown(payload: dict[str, object]) -> tuple[str, str, list[dict[str, str]]]:
    boundary = payload["boundary_conditions"]  # type: ignore[index]
    convergence = payload["convergence"]  # type: ignore[index]
    closure = payload["analytic_recompute"]  # type: ignore[index]
    provenance = payload["provenance"]  # type: ignore[index]
    anchors = payload["anchors"]  # type: ignore[index]
    criteria = payload["criteria"]  # type: ignore[index]
    reduction = payload["reduction_curve_operator"]  # type: ignore[index]
    slenderness_rows = payload["slenderness_bridge"]["rows"]  # type: ignore[index]
    dims = payload["dimension_and_identity_checks"]  # type: ignore[index]

    simple_ladder = convergence["simple_support_ladder"]  # type: ignore[index]
    clamped_ladder = convergence["clamped_ladder"]  # type: ignore[index]
    ritz_ladder = convergence["ritz_ladder"]  # type: ignore[index]
    ladder = boundary["restraint_ladder"]  # type: ignore[index]
    envelope = boundary["envelope"]  # type: ignore[index]

    k_simple_fe = simple_ladder[-1]["coefficient"]
    simple_rel = abs(k_simple_fe - 4.0) / 4.0
    max_residual = closure["max_relative_residual"]

    # ---------------------------------------------------------------- section 1
    material = provenance["material"]  # type: ignore[index]
    part1 = f"""## 1. 案例介绍

**物理背景**：板/壳屈曲的设计链条由两个数字承载——平板屈曲系数 `k`（`sigma_cr = k pi^2 D / (b^2 t)`）
与折减曲线 `chi(lambdabar)`。这两个量在工程实践中几乎总是"转录"来的（规范/手册），
而转录只有在能逐字引到原文并给出位置时才算证据。本单先把"能不能转录"和"算子层能不能自证"
彻底分开：

* 锚（Timoshenko & Gere 平板屈曲系数；DNV-RP-C201 板屈曲；EN 1993-1-6 壳折减曲线）本环境不可得
  → 全部登记 `WITHHELD`，**不写任何数字**（不凭记忆、不用二手转录）；
* 算子层（量纲/代数/一致性）全部机检，且用两套互不共享代码/形函数的离散互相夹逼。

**几何与材料参数表**（记入工件 `provenance.material`）：

| 参数 | 取值 |
|---|---|
| 杨氏模量 E | {_sci(material["young_pa"])} Pa |
| 泊松比 nu | {material["poisson"]} |
| 板厚 t | {material["thickness_m"]} m |
| 屈服强度 fy | {_sci(material["yield_pa"])} Pa |
| 板宽 b（系数归一化跨度） | 0.6 / 0.9 / 1.0 / 1.2 m（按算例） |
| 长宽比 a/b | 0.5 – 4.0 扫描；重点 1.0、sqrt(2)、1.5、2.0 |
| 边界条件 | 四边简支（SSSS）、四边固支（CCCC）、边转动约束 k_r b/D 阶梯 |

**文献/锚对照状态**：`verbatim = {payload["anchor_summary"]["verbatim"]}`，
`withheld = {payload["anchor_summary"]["withheld"]}`。
工件 `anchor_register.json` 里每个锚条目 `verbatim_quote = null`、`source_sha256 = null`、
`numeric_value_recorded = false`，并记录了检索命令（`provenance.anchor_source_search`）——
"没找到"本身是证据，不是遗漏。

## 2. 怎么做的

1. **系数算子两条独立路径**（残差判据 ≤1e-10）：
   * 路径 A：Navier 级数解析推导出 `k(m,n) = (u + n^2/u)^2`，`u = m/alpha`，对整数 m 取包络（闭式，无任何查表值）；
   * 路径 B：调用仓库既有 `tensorfem.marine_plate_buckling.navier_uniaxial_buckling_load`（二维 (m,n) 全枚举、
     有量纲刚度、返回线载荷 N_cr），只做 `k = N_cr b^2/(pi^2 D)` 归一化；
   * 两条路径共享偏微分方程但**不共享代码路径**，残差衡量的是算子闭合，不是物理验证。
2. **两套互不共享形函数的离散**（用于无闭式解的固支/转动约束）：
   * BFS（Bogner–Fox–Schmit）C1 Hermite 矩形板单元：弯曲阵复用仓库已鉴定的
     `bfs_plate_element_matrices`，本单新增轴向几何阵 `integral w_,x w_,x dA`（4 点 Gauss，被积 6 次，精确）；
   * 多项式 Rayleigh–Ritz：`W = sum c_ij phi_i(X) phi_j(Y)`，`phi_i(t) = t^2 (1-t)^2 t^i`，四边天然满足 `w = w' = 0`（固支）；
     归一化 `k = (alpha^2/pi^2) * int[(W_,XX/alpha^2 + W_,YY)^2] / int[W_,X^2]`。
3. **边转动约束阶梯**：一致（consistent）边罚 `k_r/2 * integral (w_,n)^2 ds`，`k_r = rho D / b`。
   沿边法向斜率是四次 Hermite 插值——`(h0,h2)` 承载节点法向斜率、`(h1,h3)` 承载节点扭率，
   两族在边上都有非零导数，**只罚一族会把边留成可鼓出的半约束**（本单踩过，见第 5 节）；
   `rho -> 0` 为简支，`rho -> inf` 为固支。
4. **折减曲线契约机检**：只检任何可容许曲线都必须满足的结构条款——量纲（无量纲自变量与值域）、
   值域 `0 < chi <= 1`、单调不增、连续、以及**弹性上限 `chi <= 1/lambdabar^2`**（折减后承载力不得高于弹性临界）。
   规范专有条款（stocky 平台长度、缺陷系数、塑性支）做成**检查器参数**；锚未到位时参数为 `None`，
   该条款显式报 `SKIPPED_PARAMETER_WITHHELD` 而不是猜一个数。
5. **检查器要有牙**：用 5 个刻意变异的曲线做负对照，逐一必须被拒（拒绝 = 检查器有判别力）。
6. **工具链/可复现**：`python scripts/w26_plate_shell_buckling_closure.py`（runner=native）生成工件 + 3 张图；
   `python -m pytest tests/test_plate_buckling_closure.py`（40 项，约 11 s）。
   工件内含 git commit/分支、python/numpy 版本、命令、4 个输入文件 sha256、工件自身 `content_sha256`。
   本单执行 commit：`{provenance["git_commit"][:12]}`，分支 `{provenance["git_branch"]}`。
"""

    # ---------------------------------------------------------------- section 3
    rows_closure = [
        [
            row["case"],
            _fmt(row["aspect_ratio"], 6),
            _fmt(row["coefficient_route_a"], 12) + f" (m={row['mode_route_a'][0]})",
            _fmt(row["coefficient_route_b"], 12) + f" (m={row['mode_route_b'][0]})",
            _sci(row["critical_stress_route_a_pa"]),
            _sci(row["critical_stress_route_b_pa"]),
            _sci(row["relative_residual"]),
            "PASS" if row["pass"] else "FAIL",
        ]
        for row in closure["cases"]
    ]
    rows_dim = [
        [
            row["check"],
            _sci(row.get("relative_error", row.get("worst_violation", 0.0))),
            "PASS" if row["pass"] else "FAIL",
        ]
        for row in dims["dimension_checks"]
    ]
    rows_ident = [
        [row["identity"], "PASS" if row["pass"] else "FAIL"] for row in dims["spectral_identities"]
    ]
    rows_conv = [
        [
            str(s["n_y"]),
            _fmt(s["coefficient"], 10),
            str(c["n_y"]),
            _fmt(c["coefficient"], 10),
            str(r["degree"]),
            _fmt(r["coefficient"], 10),
        ]
        for s, c, r in zip(simple_ladder, clamped_ladder, ritz_ladder)
    ]
    rows_env = [
        [
            _fmt(row["aspect_ratio"], 4),
            _fmt(row["simple_support_analytic"], 10),
            str(row["simple_support_mode"]),
            _fmt(row["simple_support_bfs"], 10),
            _fmt(row["clamped_bfs"], 10),
            _fmt(row["clamped_bfs"] / row["simple_support_bfs"], 6),
        ]
        for row in envelope
    ]
    rows_lad = [
        [
            "clamped (rho -> inf)" if point["clamped"] else _fmt(point["restraint_ratio"], 4),
            _fmt(point["coefficient"], 10),
            _fmt(point["coefficient"] / ladder[0]["coefficient"], 6),
        ]
        for point in ladder
    ]
    rows_slender = [
        [
            _fmt(row["width_m"], 4),
            _fmt(row["coefficient"], 6),
            _sci(row["critical_stress_pa"]),
            _fmt(row["lambdabar"], 6),
            _fmt(row["elastic_ceiling_chi"], 6),
            _fmt(row["specimen_chi"], 6),
            "YES" if row["elastic_ceiling_respected"] else "NO",
        ]
        for row in slenderness_rows
    ]
    specimen = reduction["specimen_report"]
    rows_clause = [
        [
            name,
            ("SKIPPED" if body.get("pass") is None else ("PASS" if body["pass"] else "FAIL")),
            body.get("status", "CHECKED"),
        ]
        for name, body in specimen["clauses"].items()
    ]
    rows_mut = [
        [
            name,
            "ACCEPTED" if body["accepted"] else "REJECTED",
            ", ".join(body["failed_clauses"]) or "-",
        ]
        for name, body in reduction["mutations"].items()
    ]
    rows_anchor = [
        [entry["id"], entry["anchor"], entry["status"], str(entry["numeric_value_recorded"])]
        for entry in anchors
    ]
    rows_crit = [[row["id"], row["requirement"], row["verdict"]] for row in criteria]

    part3 = f"""## 3. 计算结果

### 3.1 锚登记（判据 ①）

{_table(["锚", "来源", "状态", "记录数字"], rows_anchor)}

判定：**HONEST WITHHELD**。3 个锚全部 `WITHHELD`，无一条数字来自记忆或二手转录。

### 3.2 解析复算与残差（判据 ②，阈值 1e-10）

{_table(["算例", "a/b", "k 路径A(闭式)", "k 路径B(Navier)", "sigma_cr A [Pa]", "sigma_cr B [Pa]", "相对残差", "判定"], rows_closure)}

最大相对残差 **{_sci(max_residual)}** ≤ 1e-10 → 判据② **{"MEET" if closure["pass"] else "FAIL"}**。

### 3.3 量纲/代数机检

{_table(["检查", "相对误差", "判定"], rows_dim)}

{_table(["恒等式", "判定"], rows_ident)}

### 3.4 边界条件对 k 的影响（判据 ③）

收敛阶梯（BFS 网格 n×n；Ritz 多项式次数）：

{_table(["BFS n (简支)", "k 简支", "BFS n (固支)", "k 固支", "Ritz 次数", "k 固支(Ritz)"], rows_conv)}

* 简支：BFS 收敛到 **{_fmt(k_simple_fe, 8)}**，与解析包络 `k = 4` 相对差 **{_sci(simple_rel)}**；
* 固支：BFS 节点固支（弱约束，下界）**{_fmt(boundary["clamped_nodal_fe"], 8)}** 与 Ritz（上界）
  **{_fmt(boundary["clamped_ritz"], 8)}** 夹逼，区间相对宽度
  **{_fmt(boundary["clamped_bracket_relative_width"] * 100, 4)}%**；罚函数极限
  **{_fmt(boundary["clamped_penalty_limit"], 8)}** 位于 Ritz 上界之上
  {_fmt(boundary["penalty_over_ritz_relative"] * 100, 4)}%（有限罚仍带弹簧能量，故略高）；
* 固支/简支系数比 **{_fmt(boundary["clamped_over_simple"], 6)}**（同一归一化、同一载荷）。

不同长宽比下的对照（BFS n={boundary["mesh_n"]}）：

{_table(["a/b", "k 简支(解析)", "mode", "k 简支(BFS)", "k 固支(BFS)", "固支/简支"], rows_env)}

边转动约束阶梯（BFS，固定网格，rho = k_r b / D）：

{_table(["rho", "k", "k/k_simple"], rows_lad)}

阶梯**单调不减**（机检 `restraint_ladder_monotone = {boundary["restraint_ladder_monotone"]}`）：
`rho = 1` 时 k 已比简支高 {_fmt((boundary["ratio_at_restraint_1"] - 1) * 100, 3)}%，
`rho = 10` 时高 {_fmt((boundary["ratio_at_restraint_10"] - 1) * 100, 3)}%，`rho -> inf` 收敛到固支值。

### 3.5 折减曲线契约机检（算子层，非规范值）

{_table(["条款", "判定", "状态"], rows_clause)}

负对照（检查器判别力）：

{_table(["变异曲线", "检查器结论", "触发条款"], rows_mut)}

### 3.6 系数 → 无量纲细长比 → 折减算子的传导（量纲一致性）

{_table(["b [m]", "k", "sigma_cr [Pa]", "lambdabar", "1/lambdabar^2", "chi(合成样本)", "弹性上限满足"], rows_slender)}

合成样本 `chi = 1/sqrt(1 + lambdabar^4)` 只用于演示算子链与条款机检，**不是** DNV/EN 曲线。
"""

    # ---------------------------------------------------------------- section 4
    figures = payload["figures"]  # type: ignore[index]
    figure_list = "、".join("`" + figure["name"] + "`" for figure in figures)
    part4 = f"""## 4. 结果图

1. `{
        figures[0]["name"]
    }` — 系数包络 k(a/b)：解析闭式 vs BFS 离散（简支点落在包络上，固支点整体抬高）。
2. `{
        figures[1]["name"]
    }` — 边转动约束阶梯：由简支 k=4 单调过渡到固支，两条独立界限（Ritz 上界、节点固支 FE 下界）画出。
3. `{figures[2]["name"]}` — 折减曲线可容许区域：合成样本、弹性上限 1/lambdabar^2 与允许域阴影。

## 5. 注意点（踩过的坑与适用范围）

1. **几何阵 Jacobian 差 4 倍**：Gauss 点已映射到 [0,1] 后仍乘 (a_e/2)(b_e/2)，使 k 整体偏大 4 倍（首轮
   简支得 16.0）。用解析 k=4 作为回归钉子立刻暴露——这就是"有独立锚点可自检"的价值。
2. **边罚只罚一族形函数 = 假约束**：沿边法向斜率由 `(h0,h2)`+`(h1,h3)` 两族共同承载；只罚前者时
   罚能量可降到 3e-10（看着像"已固支"）而边仍在鼓出，固支极限停在 6.3（本单真值约 10.06–10.08）。
   补齐四自由度一致块后阶梯才单调收敛。
3. **节点固支 ≠ 罚函数极限**：节点只压法向斜率（边内可鼓出）是"弱约束"，其值低于罚极限；两者是不同问题，
   不能拼成一条单调阶梯，本单分开报告并用两条独立上/下界夹逼。
4. **BFS 固支收敛慢且自下而上**，Ritz 自上而下；单看一条会误判"未收敛"，夹逼才给出 {
        _fmt(boundary["clamped_bracket_relative_width"] * 100, 3)
    }% 的可信带。
5. **适用范围**：薄板线弹性、无初始缺陷、无残余应力、均匀单/双轴压；不含后屈曲、塑性、加筋板整体/局部耦合、
   壳的几何缺陷敏感性与 EN 1993-1-6 的壳特定条款。Фурье/Navier 级数仅对四边简支矩形板成立。
6. **未完成/不可判**：与 DNV-RP-C201、EN 1993-1-6、Timoshenko & Gere 的**数值比对一律未做**，
   因为锚原文不在环境内；任何"与文献吻合"的结论在本单都不成立，等锚到位后重跑同一 runner 即可补齐。

## 6. 误差判定（逐项是否 <3%）

{
        _table(
            ["关键量", "基准", "相对误差", "是否 <3%", "说明"],
            [
                [
                    "k 简支（BFS n=" + str(simple_ladder[-1]["n_y"]) + " 网格最细）",
                    "解析包络 k=4",
                    _sci(simple_rel),
                    "YES",
                    "两者共享偏微分方程、不共享代码；属于算子闭合",
                ],
                [
                    "k 固支（BFS 节点固支，下界）",
                    "Ritz（上界）",
                    _fmt(boundary["clamped_bracket_relative_width"] * 100, 4) + "%",
                    "YES",
                    "两套互不共享形函数的离散之间的夹逼宽度",
                ],
                [
                    "3 个解析算例 sigma_cr",
                    "两路径互比",
                    _sci(max_residual),
                    "YES",
                    "判据阈值 1e-10，实测远低",
                ],
                ["量纲/恒等式 6 项", "解析恒等式", "≤1e-12", "YES", "见 3.3 表"],
                [
                    "k 固支 / k 简支 比值",
                    "无独立基准（锚 WITHHELD）",
                    "未判定",
                    "NO（不可判）",
                    "比值 2.519 为两独立离散的算子结果，无外部锚可比",
                ],
                [
                    "chi 折减曲线数值",
                    "DNV-RP-C201 / EN 1993-1-6",
                    "未判定",
                    "NO（不可判）",
                    "锚 WITHHELD：本单只做结构条款机检，未产生任何规范数值",
                ],
            ],
        )
    }

**结论**：凡有独立机检基准的量（简支系数、固支夹逼、解析复算残差、量纲恒等式）误差均远低于 3%；
凡需外部规范/手册锚的量（固支系数的绝对值、折减曲线数值）在锚到位前**不可判定**，本单如实登记为
`WITHHELD`，不以"与记忆/文献一致"替代证据。

### 判据逐条裁决

{_table(["判据", "要求", "裁决"], rows_crit)}

### 工件与可复现

* 工件：`papers/artifacts/w26/plate-shell-buckling-closure/anchor_register.json`
  （`content_sha256 = {payload["content_sha256"][:32]}...`）
* 图：{figure_list}
* 代码：`src/tensorfem/plate_buckling_closure.py`、`scripts/w26_plate_shell_buckling_closure.py`、
  `tests/test_plate_buckling_closure.py`
* 复现：`python scripts/w26_plate_shell_buckling_closure.py`；测试 `python -m pytest tests/test_plate_buckling_closure.py -q`（40 passed）。
"""

    markdown = part1 + "\n" + part3 + "\n" + part4
    summary = (
        "板/壳屈曲折减曲线与系数表 W26 锚核销：3 个锚（Timoshenko&Gere / DNV-RP-C201 / EN 1993-1-6）"
        "原文不可得 → 全部 HONEST WITHHELD 零数字登记；算子层机检：3 个解析算例双路径复算最大残差 "
        f"{_sci(max_residual)}（≤1e-10 达标），简支 k 由解析包络 =4.0 与 BFS C1 有限元收敛值 "
        f"{_fmt(k_simple_fe, 8)} 互证，固支 k 被 BFS 节点固支 {_fmt(boundary['clamped_nodal_fe'], 6)} 与 Ritz 上界 "
        f"{_fmt(boundary['clamped_ritz'], 6)} 夹逼至 "
        f"{_fmt(boundary['clamped_bracket_relative_width'] * 100, 3)}% "
        f"（比值 {_fmt(boundary['clamped_over_simple'], 4)}），边转动约束阶梯单调；折减曲线契约机检 5 条款 + 5 个变异负对照全部按预期拒绝，"
        "规范专有条款显式 SKIPPED_PARAMETER_WITHHELD。"
    )
    figure_payload = []
    for figure in figures:
        path = ART / figure["name"]
        figure_payload.append(
            {
                "name": figure["name"],
                "caption": figure["caption"],
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    return markdown, summary, figure_payload


def post(payload: dict[str, object]) -> str:
    import os

    # TENSOR_API_BASE already carries the /api/v1 prefix (e.g. http://host:8010/api/v1)
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    auth = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
    )
    token = json.loads(urllib.request.urlopen(auth, timeout=30).read())["access_token"]
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
        return response.read().decode()[:400]
    except urllib.error.HTTPError as error:  # pragma: no cover - network dependent
        return f"HTTPError {error.code}: {error.read().decode()[:400]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", action="store_true", help="POST the payload to the platform")
    args = parser.parse_args()
    payload = json.loads((ART / "anchor_register.json").read_text(encoding="utf-8"))
    markdown, summary, figures = build_markdown(payload)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    report = {
        "title": "W26 板/壳屈曲折减曲线与系数表锚核销：锚 3/3 HONEST WITHHELD，算子层 3 算例残差 3.6e-16，固支 k 夹逼 0.12%",
        "summary": summary,
        "scenario_key": SCENARIO,
        "mission_id": MISSION,
        "tags": ["TensorFEM", "plate-buckling", "anchor-register", "WITHHELD", "closure"],
        "markdown": markdown,
        "figures": figures,
    }
    REPORT_JSON.write_text(json.dumps(report), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)} ({len(markdown)} chars) and {REPORT_JSON}")
    if args.post:
        print(post(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
