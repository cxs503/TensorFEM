#!/usr/bin/env python3
"""Render docs/cylinder-radiation-verified-anchor.md from the anchor evidence.

Every number in the generated page is read from
``benchmarks/verified/fem_cylinder_radiation/{result.json,gates_verified.json}``
(零手抄), so the page can be regenerated after any re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_cylinder_radiation"
DOC = ROOT / "docs" / "cylinder-radiation-verified-anchor.md"


def main() -> None:
    result = json.loads((ANCHOR / "result.json").read_text(encoding="utf-8"))
    gates = json.loads((ANCHOR / "gates_verified.json").read_text(encoding="utf-8"))
    passed = sum(1 for g in gates["gates"] if g["passed"])
    lines = [
        "# 锚点 `fem_cylinder_radiation`：圆柱模态辐射效率的独立验证证据（m16372）",
        "",
        "本页由 `scripts/gen_cylinder_radiation_doc.py` 从锚点证据文件程序化生成（零手抄）。",
        "",
        f"- 生成源：`benchmarks/verified/fem_cylinder_radiation/result.json`、`gates_verified.json`",
        f"- 闸门总数：{len(gates['gates'])}，PASS {passed}，锚点裁定 `passed = {str(gates['passed']).lower()}`",
        f"- 环境：{gates.get('environment', {}).get('machine', 'n/a')} / "
        f"python {gates.get('environment', {}).get('python', 'n/a')} / "
        f"torch {gates.get('environment', {}).get('torch', 'n/a')}",
        "",
        "## 闸门明细（引用式：每条闸门的基准公式与阈值见 manifest）",
        "",
        "| gate | 参考式 | 值 | 阈值 | 判定 |",
        "|---|---|---|---:|---|",
    ]
    for g in gates["gates"]:
        ref = g["reference"].replace("|", "\\|")
        val = g["value"]
        val_s = f"{val:.6g}" if isinstance(val, (int, float)) else str(val)
        thr = g["threshold"]
        thr_s = f"{thr:g}" if isinstance(thr, (int, float)) else str(thr)
        lines.append(
            f"| `{g['id']}` | {ref} | {val_s} | {thr_s} | **{'PASS' if g['passed'] else 'FAIL'}** |"
        )
    lines += [
        "",
        "## 关键量的收敛/一致性数字（result.json 原值）",
        "",
        "| 量 | 值 |",
        "|---|---:|",
    ]
    for key in ("window_closure_ratio", "window_coarse_over_fine", "velocity_scaling_ratio",
                "mode_share"):
        if key in result:
            lines.append(f"| `{key}` | {result[key]:.12g} |")
    if "dominant_pair" in result:
        lines.append(f"| `dominant_pair` | {tuple(result['dominant_pair'])} |")
    if "piston_law_errors" in result:
        errs = ", ".join(f"{e:.3e}" for e in result["piston_law_errors"])
        lines.append(f"| `piston_law_errors` (ka = 1e-4, 1e-3, 1e-2) | {errs} |")
    if "esm_sphere_ratios" in result:
        ratios = ", ".join(f"{r:.9f}" for r in result["esm_sphere_ratios"])
        kas = ", ".join(f"{k:g}" for k in result["esm_sphere_ka"])
        lines.append(f"| `esm_sphere_ratios` (ka = {kas}) | {ratios} |")
    lines += [
        "",
        "## 图",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `figs/efficiency_vs_ka.png` | 各周向阶 `n` 的效率 `sigma(ka)` vs 活塞律/平台渐近 |",
        "| `figs/window_closure_convergence.png` | 窗口闭合的谱采样收敛（sigma/sigma(8001)） |",
        "| `figs/esm_pulsating_sphere.png` | ESM 等效源 vs 脉动球解析功率 |",
        "| `figs/evanescent_decay.png` | `k_x > k` 倏逝段电导塌陷（含 `k_x = k` 截断点） |",
        "",
        "## 本轮修复（m16372）",
        "",
        "- `infinite_cylinder_conductance` 在**精确轴向截断** `k_x = k` 处原先返回 `nan`"
        "（`H_n^(1)(0)` 与 `kappa = 0` 的 0/0），并使 `evanescent_decay` 图静默带一个 `nan` 点；"
        "现按物理极限（掠射、无辐射功率）返回 `0.0`，与 `k_x > k` 侧连续。"
        "回归测试 `tests/test_verified_cylinder_radiation_anchor.py::test_axial_cutoff_returns_zero_not_nan`。",
        "- 锚点自此具备完整证据链（`result.json` + `gates_verified.json` + 图 + README + 独立复算测试），"
        "不再是「只有 driver、无证据」的半成锚点。",
        "",
        "## 注意点与适用范围",
        "",
        "- 空气介质（`rho = 1.225`、`c = 343`）：未含流体加载/附加质量，结构-声耦合的湿模态另见 `fsi_wet_modal`。",
        "- 验证对象是**给定表面速度场**的辐射算子，不是壳体特征值问题本身。",
        "- 窗口闭合是轴向谱一维求积，继承无限长柱闭合；有限长带端盖柱的端部效应对低 `ka` 或低阶 `m` "
        "仍是开口问题（`docs/reports/fem-structural-acoustics.md` 的 C3/C4 诚实 FAIL 即此）。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
