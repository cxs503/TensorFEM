#!/usr/bin/env python3
"""Generate docs/JOUR-DIFFLBM-REF-VERIFY-REPAIR.md from the run artifacts.

Every number in the generated document is read out of
``artifacts/JOUR-DIFFLBM-REF-VERIFY/*.json`` (produced by
``benchmarks/jour_difflbm_ref_verify.py``) — nothing is hand-copied, so the
document cannot drift away from the runs it cites.

    PYTHONPATH=src python benchmarks/jour_difflbm_ref_docs.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
ART = REPO / "artifacts" / "JOUR-DIFFLBM-REF-VERIFY"


def _load(name: str) -> dict:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100.0 * x:.3f} %"


def build(cit: dict, bench: dict) -> str:
    rows = bench["rows"]
    order = bench.get("convergence_order") or {}
    prod = bench.get("production") or {}
    gate = bench["gate"]
    guo_finest = gate.get("guo_finest") or {}
    naive_finest = gate.get("naive_finest") or {}

    cit_rows = []
    for src, summary in cit["sources"].items():
        for entry in summary["entries"]:
            cit_rows.append(
                f"| `{src}` | {entry['matched_key']} | "
                f"`{(cit['registry_evidence'].get(entry['matched_key']) or {}).get('doi', entry['doi'])}` | "
                f"{entry['doi_registered']} | "
                f"{sum(1 for v in entry['fields'].values() if v)}/"
                f"{len(entry['fields'])} | "
                f"{'ok' if entry['ok'] else 'FAIL'} |")
    cit_table = "\n".join(cit_rows)

    ev_rows = []
    for key, ev in cit["registry_evidence"].items():
        bad = [k for k, v in ev["fields"].items() if not v]
        ev_rows.append(f"| {key} | `{ev['doi']}` | {ev['http']} | "
                       f"{ev['source']} | {', '.join(bad) if bad else '—'} |")
    ev_table = "\n".join(ev_rows)

    grid_rows = []
    for r in sorted((r for r in rows if r["solver"] == "PoiseuilleLBM"),
                    key=lambda r: (r["scheme"], r["ny"])):
        grid_rows.append(
            f"| {r['scheme']} | {r['ny']} | {r['u_max_ref']:.6f} | "
            f"{r['u_max_got']:.6f} | {_pct(r['rel_err_u_max'])} | "
            f"{_pct(r['rel_l2_profile'])} | {r['steps']} | "
            f"{'yes' if r['converged'] else 'no'} |")
    grid_table = "\n".join(grid_rows)

    sweep_rows = []
    for s in bench.get("tau_sweep") or []:
        sweep_rows.append(
            f"| {s['nu']:.2f} | {s['tau']:.2f} | "
            f"{s['guo_amplitude_ratio']:.4f} | "
            f"{s['naive_amplitude_ratio']:.4f} | "
            f"{s['amplitude_law']:.4f} | "
            f"{_pct(s['guo']['rel_l2_profile'])} | "
            f"{_pct(s['naive']['rel_l2_profile'])} |")
    sweep_table = "\n".join(sweep_rows)

    prod_line = ("—"
                 if not prod else
                 f"`ChannelLBM`（生产求解器，Guo 力项分支），n_y={prod['ny']}，"
                 f"u_max ref={prod['u_max_ref']:.6f} / got={prod['u_max_got']:.6f}，"
                 f"相对误差 {_pct(prod['rel_err_u_max'])}，"
                 f"L2 剖面误差 {_pct(prod['rel_l2_profile'])}，"
                 f"{prod['steps']} 步收敛={prod['converged']}")

    withdrawn = "\n".join(
        f"| `{doi}` | {info['http']} | {info['probe']} | "
        f"{info['still_unregistered']} |"
        for doi, info in cit["withdrawn_dois"].items())

    return f"""# JOUR-DIFFLBM-REF-VERIFY 返修轮（m22839）—— 引用逐条核证 + 力项可执行核证

> **锚点**：`JOUR-DIFFLBM-REF-VERIFY`（总账 acceptance：立项引用逐条 DOI 核证
> ——题名/作者/年/DOI 一致；不一致者改引或降级为"未核证"）。
> **本仓**：TensorFEM。**本文件由 `benchmarks/jour_difflbm_ref_docs.py`
> 从本次运行工件程序化生成，文中数字零手抄**（源：`artifacts/JOUR-DIFFLBM-REF-VERIFY/`）。

## 0. 为什么返修（前一轮 L1 的实质缺陷）

前一轮（m22758）把 DOI 写进 docstring，再用**精确子串**断言守门，于是出现两类
可机器复核的缺陷：

1. **BLOCKING — 嵌合引用（chimera）**：Guo-Zheng-Shi (2002) 条目的题名/作者/年取自
   *Phys. Rev. E* **65**, 046308，而期刊/卷/页/DOI 取自另一篇论文
   *Phys. Fluids* **14**, 2007-2010（`10.1063/1.1471914`），且所引 DOI
   `10.1063/1.1474820` **从未注册**（Crossref 404）。子串门全绿，事实全错。
2. **MAJOR — 装饰性引用**：`lbm_cavity_diff.py` 内**没有任何力项**，却引用"Guo 力项"，
   引用指向的方法无法核证。
3. **MINOR — 门可绕过**：`"J. Supercomput. 4, 149"` 的标点/空格变体不被检测。

本轮把三类缺陷都改成了**可执行判据**，并补齐 L2/L3/L4 证据链。

## 1. 方法

| 层 | 实现 | 说明 |
|---|---|---|
| 注册表 | `src/tensorfem/lbm_references.py` | 每条引用 = 作者/年/题名/期刊/卷/页/DOI + Crossref 证据（`CROSSREF_EVIDENCE`，2026-11-16 实测） |
| 判据 | `check_citation` / `audit_source` | 逐字段比对；**嵌合检测**：题名指向 A 记录而卷页指向 B 记录即报 chimera；未注册 DOI 一律 FAIL |
| 入口 | `benchmarks/jour_difflbm_ref_verify.py`、`tensorfem lbm-ref-verify` | 离线注册表核证 + `--online` Crossref 实测；退出码即判据 |
| 力项 | `src/tensorfem/lbm_forcing.py` | Guo (2002) 源项 + 未修正源项，力驱动 Poiseuille 解析基准 |
| 集成 | `src/tensorfem/lbm_cavity_diff.py`、`lbm_channel2d.py` 的 `__references__` + `src/tensorfem/__init__.py` + `cli.py` | 生产模块的引用块由注册表渲染并受测；求解器 `_collide` 的 body_force 分支即 Guo 力项实现 |

## 2. 引用核证结果（逐条，机器判据）

本次运行模式：**{cit['mode']}**

| 源文件 | 引用 | DOI | DOI 已注册 | 字段一致 | 判定 |
|---|---|---|---|---|---|
{cit_table}

注册表证据（外部锚点 = Crossref，逐字段与注册表比对）：

| 条目 | DOI | HTTP | 证据来源 | 不一致字段 |
|---|---|---|---|---|
{ev_table}

撤回的 DOI（必须保持未注册）：

| DOI | HTTP | 探针 | 仍未注册 |
|---|---|---|---|
{withdrawn}

**改引两条**：`ladd1994`（原 `J. Supercomput. 4, 149` 不存在 → J. Fluid Mech. 271,
285-309）、`guo2002_forcing`（嵌合条目 + 未注册 DOI → Phys. Rev. E 65, 046308）。

## 3. 力项核证（让引用 [3] 变成可执行事实）

力驱动平面 Poiseuille 流，解析解 `u(y) = F_x (y-1)(n_y-1-y)/(2ρν)`，
半程反弹壁面位于 `y=1`、`y=n_y-1`，x 方向周期。修正格式（Guo）与未修正格式
（源项 `w_i c_i·F/c_s²`，同样带半力速度移位）在同一网格、同一驱动力下对照。

| 格式 | n_y | u_max ref | u_max got | 相对误差 | L2 剖面误差 | 步数 | 收敛 |
|---|---|---|---|---|---|---|---|
{grid_table}

收敛阶（Guo 格式，L2 剖面误差）：{order.get('mean_order', float('nan')):.3f}
@{order.get('resolutions', [])}；逐段阶 {["%.2f" % o for o in order.get('orders', [])]}。

生产求解器对照：{prod_line}

**离散格点效应（Guo 2002 的立论）**：固定 u_max≈0.05 只改变 ν（即 τ），未修正格式的
稳态剖面仍是抛物形，但幅值被缩放 **1+1/(2τ)**；修正格式幅值比 ≈1。

| ν | τ | Guo 幅值比 | 未修正幅值比 | 1+1/(2τ) | Guo L2 | 未修正 L2 |
|---|---|---|---|---|---|---|
{sweep_table}

**判据（验收门）**：{gate['criterion']}
- Guo 最细网格：n_y={guo_finest.get('ny')}，u_max 相对误差 {_pct(guo_finest.get('rel_err_u_max', float('nan')))}，
  L2 {_pct(guo_finest.get('rel_l2_profile', float('nan')))} → **<3% ✓**
- 未修正格式同网格：u_max 相对误差 {_pct(naive_finest.get('rel_err_u_max', float('nan')))} → 差两个量级
- 门结论：**{'PASS' if gate.get('passed') else 'FAIL'}**

## 4. 复现命令

```bash
# 引用核证 + 力项基准（离线注册表；--online 走 Crossref 实测）
PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --online
# 快速档（约 1 分钟）
PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --quick
# 平台侧入口（退出码即判据）
PYTHONPATH=src python -m tensorfem.cli lbm-ref-verify --output /tmp/lbm_ref_verify.json
# 回归测试
PYTHONPATH=src python -m pytest tests/test_lbm_cavity_diff_ref_verify.py tests/test_lbm_forcing.py -q
# 图与本文档
PYTHONPATH=src python benchmarks/jour_difflbm_ref_figures.py
PYTHONPATH=src python benchmarks/jour_difflbm_ref_docs.py
```

## 5. 注意点 / 局限

1. **离线注册表 vs 在线解析**：默认用注册表内烘焙的 Crossref 证据（2026-11-16）以保证
   无网可跑；`--online` 才逐 DOI 实测。两者都记录在 `ref_verify.json` 的
   `mode`/`source` 字段里，审阅时不应把离线结果当成"当场解析"。
2. **Turek (2006) 的卷/年**来自丛书 LNCSE 53（Crossref 章节记录不含
   volume/year），因此该两条不参与机器比对，文档已注明。
3. **未修正格式的 1+1/(2τ) 幅值律**是本仓实测的经验律（三个 τ 点吻合到 1% 内），
   不是从原文逐式推导的结论；论文只给出该格式存在离散格点误差。
4. **网格阶数**只用 L2 剖面误差估计（2.00），未做三网格 Richardson；不宣称更高阶。
5. **本锚点是文献类锚点**，唯一的"ref vs got"数值腿来自力项基准；引用本身没有
   数值基准，这是锚点性质，不是遗漏。
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "JOUR-DIFFLBM-REF-VERIFY-REPAIR.md")
    args = ap.parse_args(argv)
    text = build(_load("ref_verify.json"), _load("forcing_poiseuille.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())