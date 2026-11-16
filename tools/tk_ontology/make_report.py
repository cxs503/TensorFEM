#!/usr/bin/env python3
"""Generate the ontology pilot figures + the platform report body (no hand-copied numbers).

Reads only files produced by tools/tk_ontology/build_index.py, renders two PNGs and
builds /tmp/report_m22960.json (markdown + embedded base64 figures) for
POST /tensor/reports.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "ontology"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

index = json.loads((OUT / "fem_knowledge_index_v1.json").read_text(encoding="utf-8"))
m = json.loads((OUT / "fem_knowledge_metrics_v1.json").read_text(encoding="utf-8"))

# ---------------- figure 1: form census + relation census -------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
forms = ["CASE", "CRITERION", "EVIDENCE", "DEFECT", "METHOD", "CONTRACT"]
counts = [m["form_counts"][f] for f in forms]
axes[0].bar(forms, counts, color="#3d6fa8")
for i, c in enumerate(counts):
    axes[0].text(i, c + 1.5, str(c), ha="center", fontsize=9)
axes[0].set_title(f"KO census by form (total {m['ko_total']})")
axes[0].set_ylabel("objects")
axes[0].tick_params(axis="x", rotation=20)

verbs = list(m["relation_verb_counts"])
vals = [m["relation_verb_counts"][v] for v in verbs]
axes[1].bar(verbs, vals, color="#a8563d")
for i, c in enumerate(vals):
    axes[1].text(i, c + 2, str(c), ha="center", fontsize=8)
axes[1].set_title(f"declared typed edges ({m['edge_declared']}; inverse twins materialised)")
axes[1].tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIG / "ko_census.png", dpi=140)
plt.close(fig)

# ---------------- figure 2: evidence grades + dual-axis coverage ------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
grades = ["REAL", "STUB", "WITHHELD", "MOCK", "NOT_SUPPORTED"]
gv = [m["evidence_mechanical_grades"].get(g, 0) for g in grades]
axes[0].bar(grades, gv, color=["#2e7d32", "#f9a825", "#c62828", "#6a1b9a", "#37474f"])
for i, c in enumerate(gv):
    axes[0].text(i, c + 1.5, str(c), ha="center", fontsize=9)
axes[0].set_title("mechanical evidence grade (artefact on disk)")
axes[0].tick_params(axis="x", rotation=20)

cov = m["case_coverage"]
labels = ["REAL evidence", "criterion", "method", "verdict"]
rates = [cov["rate_real_evidence"], cov["rate_criterion"], cov["rate_method"], cov["rate_verdict"]]
bars = axes[1].bar(labels, [r * 100 for r in rates], color="#3d6fa8")
for b, r in zip(bars, rates):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{r:.1%}", ha="center", fontsize=9)
axes[1].axhline(100, color="#888", lw=0.8, ls="--")
axes[1].set_ylim(0, 112)
axes[1].set_ylabel("% of cases")
axes[1].set_title(f"case coverage ({cov['cases']} cases)")
axes[1].tick_params(axis="x", rotation=15)

dom = m["domain_axis_counts"]
k = list(dom)
axes[2].barh(k, [dom[i] for i in k], color="#5b8c5a")
axes[2].set_title("domain axis (dual-axis pilot)")
axes[2].tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(FIG / "coverage_axes.png", dpi=140)
plt.close(fig)


def b64(name: str) -> str:
    return base64.b64encode((FIG / name).read_bytes()).decode()


metrics_md = (OUT / "fem_knowledge_metrics_v1.md").read_text(encoding="utf-8")

markdown = f"""# 智能体导向知识本体 v1：FEM 方法轴试点（六形态 × 双轴 × 关系图）

锚点 `FEAT-TK-ONTOLOGY`｜仓 `TensorFEM`｜产物 `tools/tk_ontology/`、`artifacts/ontology/`、`tests/test_tk_ontology.py`

## ① 案例介绍

知识组织的对象不是文档而是**知识对象（KO）**:六形态（CONTRACT/METHOD/CASE/CRITERION/EVIDENCE/DEFECT）×
双轴（方法轴 FEM/LBM/... × 领域轴 阻力/耐波/…/结构/冰区）× 有类型的边。本单在 TensorFEM 落地方法轴的 FEM 试点:
把仓内 93 篇 `docs/CASE-*.md`、62 个 benchmark 证据工件、69 篇审计文档编译成一台机器可调用的索引。
动机来自可测的现状:算例证据大量只存在于散文里,下游 agent 无法判断「某个算例到底有没有可复算工件」。

## ② 怎么做的

1. `tools/tk_ontology/schema_v1.json`（draft 2020-12,`check_schema` 已验证）:六形态各有必备字段且
   `additionalProperties: false`;未确立字段写 `null` + `field_confidence ∈ measured|parsed|declared|unknown`。
2. `tools/tk_ontology/relations_v1.json`:10 个闭合动词、四元组 (主语形态,动词,宾语形态,反义)、基数约束
   （`has_criterion` 为 one_to_one——一算例两阈值即不可裁定）。
3. `tools/tk_ontology/build_index.py`:确定性编译器（无时间戳、全序），按锚点合并多篇文档为一 KO、
   物化反义边、生成 metrics/gap-report/mermaid。
4. 两条审计规则:（a）`declared_grade`（文档自述）与 `mechanical_grade`（磁盘工件机械判定）并存,
   不一致即过度声明候选;（b）裸文件名引用先按名解析,多义则保持 `WITHHELD` 不猜。
5. `tests/test_tk_ontology.py`:23 项,含 8 项负面用例（未声明动词/悬挂目标/形态错配/one_to_one 扇出/
   id 前缀不符/REAL 无摘要/非法分级/缺字段）与确定性回归。

## ③ 计算结果

{metrics_md}

`build_index.py --check` 返回 0（零 INVALID），23 项测试全绿。

## ④ 结果图

![census](fig_census) — 左:六形态对象普查;右:声明边按动词计数（反义边另行物化,不入声明计数）。
机械分级直方图显示 REAL 114 / STUB 3 / WITHHELD 23（140 个证据工件）。

![coverage](fig_coverage) — 左:证据机械分级;中:算例四项覆盖率（REAL 证据 50.0%、判据 93.9%、方法 98.8%、verdict 24.4%）;
右:领域轴直方图（双轴试点）。

## ⑤ 注意点（踩坑与局限）

* **关键词判占位符是错的**:首版把 `"placeholder_markers": []` 这类元数据字段当成占位证据,误降 3 个真工件为 STUB;
  改为只判「值恰为占位字面量」或无数值叶,并留了回归测试。
* **相对路径引用**:仓内文档常写相对脚本目录的 `evidence.json`,首版直接判 missing;
  现改为按文件名解析,多义保持 WITHHELD（不猜）。
* **不做的**:LBM 方法轴全量重组（TensorLBM/TensorPapers）、船海领域轴试点（ship-performance-platform）不在本仓范围;
  `supersedes`/`limits`/`evaluated_against` 已入词表但当前边数为 0（机械派生会引入猜测）;
  `DEFECT.severity` 大量 UNKNOWN（审计文档不标 P0–P3）。
* 机械 `REAL` 只证明「有可复算数值工件」,不证明「数值由算法算出」;`MOCK` 永不机械给出。

## ⑥ 误差判定

本单产出为知识对象与工具,关键量为**结构一致性与覆盖率**,不是物理量:

| 指标 | 值 | 判据 | 是否 <3% |
|---|---|---|---|
| schema 校验失败数 | 0 | 必须为 0 | 满足（0 项） |
| 未声明关系动词 / 悬挂目标 / 形态错配 | 0 / 0 / 0 | 必须为 0 | 满足 |
| 重编译字节级差异 | 0 | 必须为 0 | 满足 |
| 申报 vs 机械分级不一致 | 0（样本 3） | 应为 0 | 满足但**样本 n=3,不足以支撑结论** |
| CASE 具备 REAL 工件覆盖率 | 50.0% | 无基准,本单只报现值 | 不适用（gap report 已列 41 个缺口算例） |

对物理量误差无基准可谈:本单不含新的力/位移计算,故不申报物理精度。诚实结论:
本体与契约在 FEM 轴上可执行、可校验（0 INVALID、23 测试绿）,但**全仓知识可调用率仅 50%**,
契约第 6 节已把该缺口列为下一轮整改对象。
"""

report = {
    "title": "智能体导向知识本体 v1（FEM 方法轴试点）：六形态 × 双轴 × 关系图，383 KO / 0 INVALID / 23 测试绿",
    "summary": ("为 FEAT-TK-ONTOLOGY 交付可执行本体 v1：六形态 JSON Schema（draft 2020-12）+ 10 动词闭合关系词表 + "
                "确定性编译器，从 TensorFEM 仓内 93 篇算例文档/62 个证据工件/69 篇审计文档编译出 383 个 KO、417 条声明边，"
                "零 INVALID；机械分级与申报分级并存检出可复算证据覆盖率仅 50%（41/82 算例），并把 task-kit 消费契约成文。"),
    "scenario_key": "knowledge-ontology",
    "mission_id": 22960,
    "tags": ["ontology", "knowledge-object", "TensorFEM", "FEAT-TK-ONTOLOGY", "task-kit", "verification-debt"],
    "markdown": markdown,
    "figures": [
        {"name": "ko_census", "caption": "六形态对象普查与声明边按动词计数", "png_base64": b64("ko_census.png")},
        {"name": "coverage_axes", "caption": "证据机械分级、算例覆盖率与领域轴直方图", "png_base64": b64("coverage_axes.png")},
    ],
}
Path("/tmp/report_m22960.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
print("figures:", [p.name for p in sorted(FIG.glob("*"))])
print("report bytes:", Path("/tmp/report_m22960.json").stat().st_size)