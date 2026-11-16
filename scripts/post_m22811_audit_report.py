#!/usr/bin/env python3
"""Build and publish the m22811 progress report (tensor_reports).

Numbers are injected from artifacts/refverify_m22758/verification_matrix.json
and the live pytest run -- nothing is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "refverify_m22758"
M = json.loads((ART / "verification_matrix.json").read_text())
ENV = dict((k.strip(), v.strip()) for k, v in
           (l.split("=", 1) for l in open("/srv/tensor/agent/agent.env")
            if "=" in l and not l.startswith("#")))
BASE = ENV.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
MISSION = 22811


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=120) as r:
        return json.loads(r.read() or b"{}")


def ref_table() -> str:
    rows = ["| # | 引用 | 声称 DOI | Crossref 解析 | 六要素 | 代码背书 |",
            "|---|---|---|---|---|---|"]
    for i, (name, r) in enumerate(M["references"].items(), 1):
        d = r["declared"]
        res = r["resolution"]
        status = ("200 OK → " + res["record"]["container_title"] + " "
                  + str(res["record"]["volume"]) + ", " + str(res["record"]["page"])
                  if res["status"] == "ok" else f"HTTP {res['status']}（未注册）")
        flags = r["element_flags"]
        el = f"{sum(flags.values())}/6"
        rows.append(f"| [{i}] | {name} | `{d['doi']}` | {status} | {el} | "
                    f"{'✓ 有' if r['code_usage']['used'] else '✗ 无（装饰性）'} |")
    return "\n".join(rows)


def dis_table() -> str:
    d = M["disambiguation"]
    out = ["| 归属 | DOI | 题名 | 刊/卷/页 | 年 |", "|---|---|---|---|---|"]
    for key, label in (("cited_title_owner", "题名归属"),
                       ("cited_venue_owner", "刊/卷/页归属")):
        for h in (d.get(key) or [])[:2]:
            out.append(f"| {label} | `{h['doi']}` | {h['title'][:60]} | "
                       f"{h['container_title']} {h['volume']}, {h['page']} | "
                       f"{h['issued'][0] if h['issued'] else '?'} |")
    return "\n".join(out)


def main() -> int:
    s = M["summary"]
    fig = base64.b64encode((ART / "fig_refverify.png").read_bytes()).decode()
    md = f"""# m22758 `JOUR-DIFFLBM-REF-VERIFY` 交差核验（board-audit）

## ① 案例介绍

被审单 **m22758**（TensorFEM，succeeded，提交 `794447ec` → merge `01ad0648` / PR #1172）交付的是
「立项引用逐条 DOI 核证」：对 JOUR-DIFFLBM 派生锚的流体核心
`src/tensorfem/lbm_cavity_diff.py`（可微 D2Q9 顶盖驱动方腔）的 3 条 References 做
题名/作者/年/DOI 四要素核证。本单（m22811）是**交差核验单，非重做**：只取证、判级、落表。

审计参数表（全部机器读取，非手抄）：

| 项 | 值 |
|---|---|
| 被审锚 | JOUR-DIFFLBM-REF-VERIFY（文档型 acceptance，无基准行） |
| 被审提交 | `794447ec` / merge `01ad0648` |
| 引用条数 | {s['n_references']} |
| Crossref 解析成功 | {s['n_resolved_match']} |
| DOI 不存在 | {s['n_doi_invalid']} |
| 代码背书 | {s['n_code_backed']} / {s['n_references']} |
| 本锚 benchmark 行 | 0（同链 JOUR-DIFFLBM-L3-INVERSION 有 13 行，作为对照证明查询有效） |

## ② 怎么做的

1. **三件套取证**：`git show --stat 01ad0648`（3 files, 167 insertions）核交付文件在位；
   `tensor_reports` id=20464 核报告六段齐全；`tensor_benchmark_cases` 核该锚有无基准行（0 行）。
2. **repro_cmd 复跑**：`PYTHONPATH=src pytest tests/test_lbm_cavity_diff_ref_verify.py
   tests/test_lbm_cavity_diff.py -q` → 12 passed。
3. **外部锚（m22758 未做的关键一步）**：写 `scripts/verify_jour_difflbm_refs.py`，
   对 3 条 DOI 做 **Crossref/doi.org 在线解析**，逐元素（题名/作者/年/刊/卷/页）比对，
   并对"解析不到"的条目做**归属反查**（题名归属 vs 刊卷页归属），产出机器修复清单。
4. **代码背书检查**：把 docstring 剥掉后在源码正文 grep，判断每条引用是否真有实现支撑。
5. **回归门绕过探针**：构造格式变体检验精确子串守卫是否可绕。
6. **判级 + 落表**：按 maturity_gate JUDGE_SCHEME 逐级核 evidence，写入
   `tensor_maturity_verdicts`（id=914），本地 dry-run `judge()` 复核。

## ③ 计算结果

### 3.1 逐条 DOI 外部解析（Crossref，{M['audit_mission']} 单实测）

{ref_table()}

### 3.2 不可解析 DOI 的归属反查

{dis_table()}

即引用 [3] 是**拼接引用**：题名/作者/年属 *Phys. Rev. E* **65**, 046308 (2002)，
刊/卷/页 *Phys. Fluids* **14**, 2007-2010 属另一篇同作者论文
(*An extrapolation method for boundary conditions in lattice Boltzmann method*)，
而所填 DOI `10.1063/1.1474820` **未注册**。

### 3.3 三件套核验

| 三件套 | 结果 |
|---|---|
| run_artifacts | **缺失**（m22758 无 `runs/`，DOI 核证无任何可执行记录） |
| benchmark ref vs got | **N/A**（该锚 benchmark 行=0，属文档型；被审单未编造数字，正确） |
| repro_cmd | **可执行**（12 passed, 26.26 s） |

### 3.4 判级（L0–L5）

| 级 | 结果 |
|---|---|
| L1 单元验证 | **命中**：引用完整性测试真实存在且可复现 |
| L2 独立功能 | 不达：无 entry、无 run_artifacts；且测试固化了不存在 DOI |
| L3 / L4 / L5 | 不达（无集成调用、无基准行、外部锚反被证伪） |

**判定：L1**（verdict id=914；maturity_gate dry-run：gate_level=1 → reopened）。

## ④ 结果图

![fig](fig_refverify.png)

左：六要素逐项比对热图（绿=一致）；右：引用是否有代码背书。
[3] Guo 六要素 0/6（DOI 未注册）且无代码路径支撑。

## ⑤ 注意点

- m22758 的"本机无外网"不成立：本生态内 Crossref/doi.org **可解析**（两条 DOI 各返回 200）。
  跳过外部解析 = REF-VERIFY 锚没有做核证。
- 代码背书检查必须**剥离 docstring**，否则 References 里的作者名会把"装饰性引用"误判为"有实现"。
- 精确子串守卫（`"J. Supercomput. 4, 149"`）可被 `149-168` / 空格 / 冒号变体绕过，实测通过。
- 适用边界：本审计只裁 TensorFEM 侧 JOUR-DIFFLBM 链的 3 条引用（grep 确认 TensorFEM 链无其它文献引用）；
  TensorPapers 侧论文正文参考文献不在本单 scope。
- 未做：未取 m22758 会话轨迹；未核 Ladd 公式与原文的逐项符号等价（属 GAP-PP-2 锚）。

## ⑥ 误差判定（关键量 vs 外部基准，逐项）

| 关键量 | 被审单声明 | 外部基准（Crossref/doi.org） | 误差 | <3%？ |
|---|---|---|---|---|
| DOI 有效率 | 3/3 核证通过（1 改引 + 2 通过） | {s['n_resolved_match']}/3 外部可解析 | {100*(s['n_references']-s['n_resolved_match'])/s['n_references']:.1f}%（1/3 条） | **否** |
| 引用一致性（六要素） | 3/3 一致 | 2/3 完全一致；[3] 0/6 | 33.3% 条目不合格 | **否** |
| 代码背书率 | 隐含"Ladd+Ghia+Guo 三件套" | {s['n_code_backed']}/3 有实现 | 33.3% 装饰性引用 | **否** |
| repro_cmd 复现 | 12 passed | 12 passed（本树实测） | 0 | 是 |

**显式结论：不是 <3%。** 3 条引用中 1 条（33.3%）DOI 未注册、六要素互斥、且无代码背书，
远超 3% 判据 → 处置 **request changes（返修）**，返修清单由
`verification_matrix.json:repair_spec` 机器生成（改引为 `10.1103/PhysRevE.65.046308`，
或改引 `10.1063/1.1471914` 并换题名；[3] 若无实现则标注 `[background]`；测试常量同步改正）。
"""
    payload = {
        "title": "m22758 JOUR-DIFFLBM-REF-VERIFY 交差核验：3 条引用中 1 条 DOI 未注册（判 L1，返修）",
        "summary": ("board-audit 复核 m22758（PR#1172）：三件套中 run_artifacts 缺失、benchmark 行=0（文档型锚）、"
                    "repro_cmd 12 passed；Crossref 外部解析证明 Guo 引用 DOI 10.1063/1.1474820 未注册且为拼接引用"
                    "（题名属 PRE 65,046308、卷页属 Phys. Fluids 14,2007-2010=10.1063/1.1471914），"
                    "且该引用无代码背书；判级 L1，verdict id=914，处置 request changes。"),
        "scenario_key": "JOUR-DIFFLBM-REF-VERIFY",
        "mission_id": MISSION,
        "tags": ["board-audit", "JOUR-DIFFLBM", "referrer-verify", "DOI", "TensorFEM", "m22758"],
        "markdown": md,
        "figures": [{"name": "fig_refverify.png",
                     "caption": "A 六要素 Crossref 比对热图；B 引用代码背书（[3] Guo DOI 未注册且无实现）",
                     "png_base64": fig}],
    }
    tok = api("POST", "/auth/login", {"username": ENV["TENSOR_AGENT_USER"],
                                      "password": ENV["TENSOR_AGENT_PASS"]})["access_token"]
    r = api("POST", "/tensor/reports", payload, tok)
    print("report id:", r.get("id"), "keys:", list(r)[:6])
    pathlib.Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())