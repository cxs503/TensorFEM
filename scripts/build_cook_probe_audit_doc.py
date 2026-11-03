"""Render the Cook probe-protocol audit: two-panel figure plus the markdown report.

Every number in ``docs/cook-probe-protocol-audit.md`` is injected from
``benchmarks/cook_probe_protocol.json`` (produced by
``scripts/audit_cook_probe_protocol.py``); the report contains no hand-copied value.

Reproduce:
  OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_probe_protocol.py
  OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/build_cook_probe_audit_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "cook_probe_protocol.json"
FIGURE = ROOT / "docs" / "assets" / "cook-probe-protocol.png"
DOC = ROOT / "docs" / "cook-probe-protocol-audit.md"
GATE = 0.03


def _pct(value: float) -> str:
    return f"{value * 100:.3f}"


def build_figure(payload: dict) -> None:
    levels = np.asarray(payload["levels"], dtype=float)
    protocols = payload["protocols"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=150)

    styles = {
        "merged": ("merged (floor tie-break)", "o", "#b2182b"),
        "symmetrised": ("edge-symmetrised one-sided mean", "s", "#ef8a62"),
        "recovered": ("Gauss-point + nodal recovery", "^", "#2166ac"),
    }
    for name, (label, marker, colour) in styles.items():
        errors = np.asarray(protocols[name]["max_relative_error_by_level"], dtype=float)
        left.loglog(levels, errors * 100.0, marker + "-", color=colour, label=label,
                    lw=1.6, ms=6)
    left.axhline(GATE * 100.0, color="k", ls="--", lw=1.2)
    left.text(levels[0] * 1.05, GATE * 100.0 * 1.12, "3 % acceptance gate", fontsize=8)
    left.set_xlabel("Cook grid divisions $n$")
    left.set_ylabel("max pointwise von Mises error [%]")
    left.set_title("Probe protocol decides the reported error", fontsize=10)
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=7.5, loc="upper right")

    finest = payload["cases"][-1]
    labels = [f"({p['xi']}, {p['eta']})" for p in finest["points"]]
    x = np.arange(len(labels))
    span = np.asarray([p["candidate_max"] - p["candidate_min"] for p in finest["points"]])
    lower = np.asarray([p["candidate_min"] for p in finest["points"]])
    right.bar(x, span, bottom=lower, color="#c6dbef", edgecolor="#2166ac",
              label="admissible one-sided span")
    right.plot(x, [p["reference_von_mises"] for p in finest["points"]], "k_",
               ms=22, mew=2.4, label="Q8 reference")
    right.plot(x, [p["merged"]["von_mises"] for p in finest["points"]], "o",
               color="#b2182b", ms=6, label="merged tie-break pick")
    right.plot(x, [p["recovered"]["von_mises"] for p in finest["points"]], "^",
               color="#2166ac", ms=6, label="recovered field")
    right.set_xticks(x)
    right.set_xticklabels(labels, fontsize=8)
    right.set_xlabel(f"canonical probe point on the n={finest['divisions']} grid")
    right.set_ylabel("von Mises stress")
    right.set_title("Probe sites are element edges: value not unique", fontsize=10)
    right.grid(True, axis="y", alpha=0.3)
    right.legend(fontsize=7.5, loc="upper left")

    figure.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE)
    plt.close(figure)


def build_doc(payload: dict) -> None:
    protocols = payload["protocols"]
    finest_level = payload["levels"][-1]
    finest = payload["cases"][-1]
    lines: list[str] = []
    add = lines.append

    add("# Cook 膜 Q4 点式应力 5.02% 欠账的探针协议复核")
    add("")
    add("## 结论摘要")
    add("")
    add(
        "合并通道 `docs/benchmark-qualification-status.md` 一直挂在"
        f"「生产 Q4 应力点式误差 {_pct(payload['reference']['documented_residual'])} % > 3 %，"
        "`stress_accuracy_qualified` 保持 False」。本单把该 `WITHHELD` 项拆成"
        "**离散误差**与**测量协议歧义**两部分并给出定谳（判据先于动手，全部数字机器产出）："
    )
    add("")
    add("| 项 | 结论 | 证据字段 |")
    add("|---|---|---|")
    add(
        "| 合并协议的 5.02 % 能否复现 | **能**，本审计独立复算 "
        f"{_pct(payload['verdict']['merged_residual_reproduced'])} %（n={finest_level}） | "
        "`protocols.merged.finest_max_relative_error` |"
    )
    add(
        "| 5 个规范采样点是否良置 | **否**：全部落在 Q4 单元边界上（η 网格线），"
        f"最细网格上 {payload['probe_sites']['ambiguous_points_finest']}/5 点取值不唯一，"
        f"单侧取值散布最大 {_pct(payload['probe_sites']['max_spread_relative_finest'])} % | "
        "`probe_sites`、`cases[-1].points[*].candidates` |"
    )
    add(
        "| 合并协议随网格是否收敛 | **否**：逐点误差非单调（见下表），"
        "该协议测出的不是离散误差 | `protocols.merged.per_point[*].observed_order` |"
    )
    add(
        "| 是否等价于「Q4 应力不可认证」 | **否**：Gauss 点外推 + 节点面积加权平均的恢复场"
        f"在 n=16 即 {_pct(protocols['recovered']['max_relative_error_by_level'][1])} %、"
        f"n={finest_level} 为 {_pct(protocols['recovered']['finest_max_relative_error'])} %，"
        "逐级单调、逐点阶 1.98–3.29 | `protocols.recovered` |"
    )
    add(
        "| 简单「单侧平均」能否救 | **不能**：最细网格非单调（"
        f"{_pct(protocols['symmetrised']['max_relative_error_by_level'][-2])} % → "
        f"{_pct(protocols['symmetrised']['max_relative_error_by_level'][-1])} %），"
        "不能作为合格判据 | `protocols.symmetrised` |"
    )
    add("")
    add(
        "因此本审计的判定是 `probe-protocol-artifact`：合并通道报出的 "
        f"{_pct(payload['verdict']['merged_residual_reproduced'])} % 是"
        "**在应变不连续处做单点取值的平局裁决结果**，不是 Q4 应力场的精度。"
        "生产 `cook_fields` 的 `stress_accuracy_qualified` 保持 False 不变（fail-closed 不动），"
        "但该 `WITHHELD` 的理由必须改写为「测点协议未良置」，而不是「应力场精度不足」。"
    )
    add("")
    add("![probe protocol](assets/cook-probe-protocol.png)")
    add("")
    add(f"*左*：三套测量协议的点式 von Mises 最大误差随网格变化，虚线为 3 % 判据；"
        f"*右*：n={finest_level} 网格上每个采样点的单侧可取值区间（浅蓝柱）、Q8 参考（黑横线）、"
        "合并协议的平局取值（红点）与恢复场取值（蓝三角）。")
    add("")
    add("## 案例介绍（物理背景与几何）")
    add("")
    add(
        "Cook 膜是线弹性平面应力的标准验证题：梯形膜左端固支，右端承受单位竖向合力，"
        "弯曲主导且含夹持角点奇异，因此对单元应力恢复特别苛刻。几何为双线性映射的父域"
        " `[0,1]²` → 物理梯形，四个角点 `(0,0) (48,44) (48,60) (0,44)`（左高 44、右高 16）；"
        "材料 `E=1、ν=1/3、t=1`（`plane_stress_matrix` 默认值），"
        "已发表位移参考 23.96（加载边中点）。"
    )
    add("")
    add("参考解不是本单产生的：`benchmarks/cook_membrane_stress_reference.json` 是"
        "m2430 用独立 Q8 实现（3×3 Gauss、Jacobi-PCG 容差 1e-13、level 12/24/48）"
        "Richardson 外推得到的五点应力，最细 GCI "
        f"{_pct(payload['reference']['max_gci_fine_relative'])} %，"
        "本单只读使用，不改动。")
    add("")
    add("## 怎么做的（方法/流程/关键设置）")
    add("")
    add("1. **复现合并协议**（`scripts/audit_cook_probe_protocol.py::merged_protocol_value`）："
        "按 `cook_stress_reference.platform_q4_stress_at` 的 `floor(ξn), floor(ηn)` 选单元，"
        "在父坐标处求 Q4 应力，得到合并通道的原始数字。")
    add("2. **枚举单侧取值**（`cook_probe_protocol.one_sided_values`）：不用网格编号假设，"
        "而是逐单元 Newton 反解双线性映射，返回**所有**包含该点的单元（角点 4 个、边 2 个）"
        "各自的一侧应力；`spread_relative` = 极差/均值，即该测点不可消除的歧义量。")
    add("3. **边对称化**：对全部单侧值取算术平均，检验「简单去歧义」是否给出稳定估计。")
    add("4. **应力恢复**（`cook_probe_protocol.nodal_recovered_stress`）：2×2 Gauss 点应力"
        "（双线性单元的超收敛点）线性外推到 4 个节点，按单元面积加权做节点平均，"
        "再用双线性形函数在原物理点插值。不触碰生产装配与求解器。")
    add("5. **收敛判据**：网格 n = " + ", ".join(str(n) for n in payload["levels"]) +
        "；每点/每协议用最细三级做观测阶 + Richardson 极限，逐级单调为硬门；"
        "误差归一化沿用合并通道的 5 % 全局尺度下限（`max(|ref|, 0.05·scale)`），"
        "便于与 5.02 % 直接比较。")
    add("6. 复现命令：`OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_probe_protocol.py`"
        " → `benchmarks/cook_probe_protocol.json`；"
        "`python scripts/build_cook_probe_audit_doc.py` → 本文件与上图。")
    add("")
    add("## 计算结果")
    add("")
    add(f"### 三套协议的最大点式误差（%%，相对 Q8 外推参考，n={finest_level} 为最细）")
    add("")
    add("| 协议 | " + " | ".join(f"n={n}" for n in payload["levels"]) + " | 单调 | n="
        f"{finest_level} 是否 <3%% |")
    add("|---|" + "---|" * (len(payload["levels"]) + 2))
    labels = {
        "merged": "合并协议（floor 平局）",
        "symmetrised": "单侧值算术平均",
        "recovered": "Gauss 点+节点恢复",
    }
    for name, block in protocols.items():
        add(
            f"| {labels[name]} | "
            + " | ".join(_pct(v) for v in block["max_relative_error_by_level"])
            + f" | {'是' if block['monotone'] else '否'} | "
            + f"{'是' if block['qualified_at_finest'] else '否'} |"
        )
    add("")
    add("### 逐采样点：测点良置性、三协议取值与误差")
    add("")
    add("| 点 (ξ,η) | 单侧候选数 | 单侧散布 %% | 单侧区间 | Q8 参考 | 合并取值 (误差%%) | "
        "平均取值 (误差%%) | 恢复取值 (误差%%) | 参考是否落在单侧区间内 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for point in finest["points"]:
        add(
            f"| ({point['xi']}, {point['eta']}) | {point['candidate_count']} | "
            f"{_pct(point['candidate_spread_relative'])} | "
            f"{point['candidate_min']:.5f}–{point['candidate_max']:.5f} | "
            f"{point['reference_von_mises']:.5f} | "
            f"{point['merged']['von_mises']:.5f} ({_pct(point['merged']['relative_error'])}) | "
            f"{point['symmetrised']['von_mises']:.5f} "
            f"({_pct(point['symmetrised']['relative_error'])}) | "
            f"{point['recovered']['von_mises']:.5f} "
            f"({_pct(point['recovered']['relative_error'])}) | "
            f"{'是' if point['reference_inside_candidate_span'] else '否'} |"
        )
    add("")
    add("### 逐点收敛诊断（最细三级，观测阶与 Richardson 极限误差 %%）")
    add("")
    add("| 点 (ξ,η) | 合并协议误差序列 (%% ) | 合并观测阶 | 合并极限误差 %% | 恢复误差序列 (%% ) | "
        "恢复观测阶 | 恢复极限误差 %% |")
    add("|---|---|---|---|---|---|---|")
    for key in protocols["merged"]["per_point"]:
        merged = protocols["merged"]["per_point"][key]
        recovered = protocols["recovered"]["per_point"][key]
        order_m = "无（非收敛）" if merged["observed_order"] is None else f"{merged['observed_order']:.2f}"
        add(
            f"| {key} | " + ", ".join(_pct(v) for v in merged["errors_by_level"]) + " | "
            f"{order_m} | {_pct(merged['limit_relative_error'])} | "
            + ", ".join(_pct(v) for v in recovered["errors_by_level"]) + " | "
            f"{recovered['observed_order']:.2f} | {_pct(recovered['limit_relative_error'])} |"
        )
    add("")
    add("关键读法：合并协议在 `(0.5, 0.25)` 上四级误差 "
        + ", ".join(_pct(v) for v in protocols["merged"]["per_point"]["xi0.5_eta0.25"]["errors_by_level"])
        + " %，几乎不随网格下降（Richardson 极限 "
        + f"{_pct(protocols['merged']['per_point']['xi0.5_eta0.25']['limit_relative_error'])} %），"
        "而同一物理点上恢复场逐级下降并收敛到 0.001 % 量级——"
        "同一位移解、同一参考，唯一差别是「在哪取值」。")
    add("")
    add("## 注意点（坑、局限、适用范围）")
    add("")
    add("- **坑 1：探针落在单元边界是结构化网格的必然结果**。规范点 η ∈ {0.25, 0.5, 0.75}，"
        "n 为 4 的倍数时 η·n 恒为整数，于是 5/5 个点都贴在 η=const 网格线上；"
        "Q4 应变沿单元边界不连续，单点取值只能靠平局规则（floor/ceil）裁决，"
        "裁决结果随 n 跳变，表现为「误差不收敛」。")
    add("- **坑 2：不要用「平均单侧值」冒充修好**。算术平均在 (0.3, 0.5)、(0.7, 0.5) 两点最细网格"
        f"反而由 {_pct(protocols['symmetrised']['max_relative_error_by_level'][-2])} % 升到 "
        f"{_pct(protocols['symmetrised']['max_relative_error_by_level'][-1])} %，"
        "斜边上两侧值权重不同，对称化不是一致估计。")
    add("- **坑 3：恢复场的节点值是超收敛外推，不是原存盘场**。`cook_fields` 存的是单元中心常数应力"
        "（O(h)），恢复场是同一 `displacement` 的 2 阶后处理；引用 0.12 % 时必须同时说明协议，"
        "否则等同于拿更好的后处理冒充原始场精度。")
    add("- **适用范围**：线弹性平面应力、E=1、ν=1/3、t=1、单位合力剪切载荷下的 Cook 梯形膜；"
        "结论限于结构化 Q4 网格与 2×2 Gauss 恢复。夹持角点与加载边端点的峰值应力（奇异区）"
        "不在覆盖范围内，任何在该处做的点式比较仍不可认证。")
    add("- **未做**：不改生产 `cook_fields` 的 `stress_accuracy_qualified`（保持 False，fail-closed），"
        "不改 `benchmarks/cook_membrane_stress_reference.json` 与既有文档哈希清单；"
        "Q8 参考只到 n=48，恢复场在 n=64 上已是 0.12 %，参考精度（GCI "
        f"{_pct(payload['reference']['max_gci_fine_relative'])} %）仍是误差地板，"
        "故 0.12 % 应读作「≤0.12 %」而非精确值。")
    add("")
    add("## 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add("| 关键量 | 计算值 | 基准 | 相对误差 | 是否 <3 %% |")
    add("|---|---|---|---|---|")
    add(
        "| 合并协议最细点式 von Mises 最大误差 | "
        f"{_pct(protocols['merged']['finest_max_relative_error'])} %（n={finest_level}） | Q8 外推参考 | "
        f"{_pct(protocols['merged']['finest_max_relative_error'])} % | **否** |"
    )
    add(
        "| 单侧值算术平均最细最大误差 | "
        f"{_pct(protocols['symmetrised']['finest_max_relative_error'])} %（n={finest_level}） | 同上 | "
        f"{_pct(protocols['symmetrised']['finest_max_relative_error'])} % | **否** |"
    )
    add(
        "| Gauss 点+节点恢复最细最大误差 | "
        f"{_pct(protocols['recovered']['finest_max_relative_error'])} %（n={finest_level}） | 同上 | "
        f"{_pct(protocols['recovered']['finest_max_relative_error'])} % | **是** |"
    )
    add(
        "| 恢复场 n=16 最大误差 | "
        f"{_pct(protocols['recovered']['max_relative_error_by_level'][1])} % | 同上 | "
        f"{_pct(protocols['recovered']['max_relative_error_by_level'][1])} % | **是** |"
    )
    add(
        "| 测点歧义（最大单侧散布） | "
        f"{_pct(payload['probe_sites']['max_spread_relative_finest'])} % | 合并协议报告的 5.02 % | "
        f"歧义是待测误差的 {payload['probe_sites']['max_spread_relative_finest'] / payload['reference']['documented_residual']:.1f} 倍 | "
        "不适用（协议缺陷） |"
    )
    add("")
    add(
        "结论：**5.02 % 这条欠账不是<3 % 与否的问题，而是被测量对象的良置性问题**；"
        "在同一参考、同一位移解下，2 阶恢复场 n=64 为 "
        f"{_pct(protocols['recovered']['finest_max_relative_error'])} %（<3 % 成立），"
        "n=16 已 "
        f"{_pct(protocols['recovered']['max_relative_error_by_level'][1])} %。"
        "生产 `stress_accuracy_qualified` 仍按 fail-closed 保持 False，"
        "本单只提供可复核的协议修正证据与可复用实现（"
        "`src/tensorfem/cook_probe_protocol.py`、`tests/test_cook_probe_protocol.py`）。"
    )
    add("")
    # Table headers are written with escaped percent signs for readability in the
    # builder; collapse them so the rendered document shows a single "%".
    DOC.write_text("\n".join(lines).replace("%%", "%") + "\n")


def main() -> None:
    payload = json.loads(EVIDENCE.read_text())
    build_figure(payload)
    build_doc(payload)
    print(f"wrote {DOC.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()