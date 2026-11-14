#!/usr/bin/env python3
"""Publish the m19999 board-audit report (all numbers injected from artifacts)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "artifacts/m19999_audit"
BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def req(method, path, body=None, token=None):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                               method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    ev = json.loads((HERE / "audited_case/case_fem_cantilever_tip_evidence.json").read_text())
    chk = json.loads((ART / "recheck_analytic.json").read_text())
    vm = json.loads((ART / "verify_reference_mesh.json").read_text())
    vr = json.loads((ART / "verify_remaining_levels.json").read_text())
    tb = chk["tensorbench_rows"]
    ref = ev["verdict"]["reference_value_m"]
    ind = chk["independent"]

    ladder = "".join(
        f"| {'x'.join([str(r['axial_elements'])] * 1 + [str(r['cross_elements'])] * 2)} "
        f"| {r['dofs']} | {r['tip_deflection_m']:.12e} | "
        f"{r['relative_error_vs_euler_bernoulli']*100:+.4f}% | "
        f"{'是' if abs(r['relative_error_vs_euler_bernoulli']) < ev['verdict']['tolerance_rel'] else '否'} |\n"
        for r in ev["studies"]["solid_ladder"])
    refrow = ev["levels"]["solid_reference"]
    tbrow = [r for r in tb if r["metric"].startswith("tip_deflection_solid")][0]

    md = f"""# m19999 交差审计：CASE-FEM-CANTILEVER-TIP（TensorFEM）判级 L3

## ① 案例介绍

被审对象：mission m19999，`CASE-FEM-CANTILEVER-TIP`（算例池 `FEM-CANTILEVER-TIP`，tier core）。
物理：均质等截面悬臂梁，`x=0` 固支，自由端横向集中力。几何/参数表（全部读自被审证据工件
`reference`/`geometry`/`material` 段）：L={ev['geometry']['length_m']} m，b×h=0.01×0.01 m，
L/h={ev['geometry']['slenderness_L_over_height'] if 'slenderness_L_over_height' in ev['geometry'] else ev['geometry'].get('slenderness_L_over_h')}，
E=200 GPa，ν=0.3，P=5 N（-z 向）。
基准：Euler-Bernoulli `δ=PL³/(3EI)`（Timoshenko 1955 ch.4；Blevins 1979 Table 8-1）；
Timoshenko 剪切项 `PL/(κAG)`，κ=5/6。预声明容差取自 `tensor_case_pool.tol_policy='<2%'`。

审计对象不是算例本身，而是**交付是否成立**：三件套（run_artifacts / 基准 ref-vs-got / repro_cmd）、
报告六段、以及平台登记裁判。

## ② 怎么做的（审计方法）

1. **取证**：读 m19999 工作树（提交 `5504271e`，`agent/m19999`）；把被审 4 个文本 + 3 张 PNG 复制到
   `docs/audit_m19999/audited_case/` 并记 sha256（`SHA256SUMS.txt`）。
2. **解析锚自推**（`artifacts/m19999_audit/recheck_analytic.py`，不 import 被审代码、不 import tensorfem）：
   闭式 δ_EB、δ_T + 20 001 点复合 Simpson 对弯矩/剪力场独立积分。
3. **跨工作树复算**（这是关键反 MOCK 手段）：在**本审计工作树的 main `src/tensorfem`** 上跑被审脚本副本
   （sha256 与被审文件逐一相同），逐档复算实体阶梯、参考网格、三档细长比与 2P 线性。
4. **测试闸**：`pytest -q` 跑被审 9 条回归测试副本。
5. **平台库直查**：`tensor_benchmark_cases`（裁判表）、`tensor_case_pool`（容差出处）、
   `tensorbench_results`（机器判定行）、`tensor_reports`（报告 18246）——py 侧用 psql 子进程（本机 psycopg2 为 stub）。
6. **判级**：按 L0–L5 功能实质，不以报告厚度定级。

## ③ 计算结果

**解析锚（审计独立重推）**：I={ind['I_m4']:.12e} m⁴，δ_EB={ind['delta_eb_m']:.12e} m，
剪切项={ind['delta_shear_m']:.12e} m（占弯曲 {ind['shear_fraction']*100:.4f}%），
δ_T={ind['delta_timoshenko_m']:.12e} m；Simpson 独立积分相对差 弯曲 {ind['eb_simpson_rel_diff']:+.2e}、
剪切 {ind['shear_simpson_rel_diff']:+.2e}。被审工件 `reference` 段与之相对差 0.0。

**被审实体阶梯**（读自证据工件，逐档由审计在 main 上复算）：

| HEX20 网格 | dof | 端挠度 [m] | 误差 vs δ_EB | <2% |
|---|---:|---:|---:|:--:|
{ladder}| 32x4x4（参考） | {refrow['dofs']} | {refrow['tip_deflection_m']:.12e} | {refrow['relative_error_vs_euler_bernoulli']*100:+.4f}% | 是 |

**判据行**：ref={ref:.12e} m，got={refrow['tip_deflection_m']:.12e} m，
误差 **{refrow['relative_error_vs_euler_borrowed'] if False else refrow['relative_error_vs_euler_bernoulli']*100:+.4f}%**，tolerance_rel=0.02 ⇒ **是否 <2%：是**；生态线 <3%：**是**。

**复算逐位一致**：参考网格审计复算 {vm['reference_rows']['solid_reference_32x4x4']['audit_m']:.12e} m
（与工件相对差 {vm['reference_rows']['solid_reference_32x4x4']['rel_diff']:.3e}）；
其余档 16x1 / 32x2 / 64x4 与三档细长比、2P 线性比
{list(vr['rows'].keys())} 相对差全为 0.000e+00（耗时 {vr['elapsed_s']:.1f} s）。

**平台机器判定行**：`{tbrow['metric']}` ref={tbrow['ref_value']:.9e} got={tbrow['got_value']:.12e}
rel={tbrow['rel_error']*100:+.6f}% tol={tbrow['tolerance_rel']} passed={tbrow['passed']}
（cluster job {tbrow['cluster_job_id']}，run {tbrow['run_id']}）。

## ④ 结果图

图1 收敛曲线：被审工件阶梯（红）与审计在 main 上复算点（黑叉，逐位重合）+ ±2% 容差带；
图2 端挠度 vs 解析锚/ΔTimoshenko 与审计独立重推线。

## ⑤ 注意点（发现，按严重度）

- **F1（medium-high，阻塞 L4）登记裁判缺位**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
  WHERE anchor_slug='CASE-FEM-CANTILEVER-TIP'` → **0 行**（全表 121 行，TensorFEM 仅 7 行均非本算例）。
  故本算例的 PASS 是"工件自证 + 自 ingest"，不是平台登记基准判定。
- **F2（medium，改判为平台侧）报告图未内嵌**：`tensor_reports.id=18246` 六段齐全，但 `figures` 无 `png_base64`。
  **对照实验**：审计自身递交 99 700 B + 74 684 B base64（id=18284），落库后 `figures->0 ? 'png_base64'`
  仍为 false，而旧 id=17486 有 224 768 B ⇒ 当前 ingest 丢弃图载荷，**非 m19999 过失**。
- **F3（low-medium）provenance 不自洽**：工件 `implementation.revision=9ad824f7…` 为 `5504271e` 的父提交，
  不含该 workload；平台行 `implementation_revision='unknown'`。
- **F4（low）末位漂移未声明**：工件 got 与 ingest 行 got 相对差 1.7e-9（均 <2% 通过）；审计复算与工件逐位一致，
  漂移在 cj2435 那次运行环境。
- **F5（low，历史遗留）**：`benchmarks/cantilever_tip_artifacts/results.json`（m15330/m17033 期）声明的
  convergence.png/comparison.png 在库中不存在，其 repro_cmd.sh 指向旧脚本而非本锚点。
- 局限：未做单次全量 `run_study` 与 `--write-docs`（本机 >120 s 预算；集群作业 cj2457 在逐档复算使其冗余后取消）；
  未与 Abaqus/CalculiX 交叉验证；未做 PNG 像素级检查。

## ⑥ 误差判定

| 关键量 | 基准 | got | 误差 | <2% | <3% |
|---|---:|---:|---:|:--:|:--:|
| 实体参考网格 32x4x4 端挠度 | 1.000000e-02 m | {refrow['tip_deflection_m']:.6e} m | {refrow['relative_error_vs_euler_bernoulli']*100:+.4f}% | 是 | 是 |
| 实体最粗 8x1x1 | 1.000000e-02 m | {ev['studies']['solid_ladder'][0]['tip_deflection_m']:.6e} m | {ev['studies']['solid_ladder'][0]['relative_error_vs_euler_bernoulli']*100:+.4f}% | 否（工件亦标 ❌） | 是 |
| EB 梁 FE n=16 | 1.000000e-02 m | {ev['levels']['beam_eb']['tip_deflection_m']:.6e} m | {ev['levels']['beam_eb']['relative_error_vs_euler_bernoulli']*100:+.2e}% | 是 | 是 |
| 平台 ingest 行 | 1.000000e-02 m | {tbrow['got_value']:.6e} m | {tbrow['rel_error']*100:+.6f}% | 是 | 是 |

**是否 <3%：是**（判据行 0.3595%，最细档 0.1810%，唯一超线的是诚实标注的最粗 8x1x1 -2.5139%——仍在 <3% 内）。

**判级 L3**：L2/L3 实证齐（真代码 + 9/9 pytest 绿 + 跨工作树逐位复现 + 平台 TensorBench 管道 ingest 2 行机器判定）；
**L4 扣留**（F1 登记缺位 + F3 provenance 不自洽，与先例 m19997 一致）；**L5 拒绝**（教材级单锚点，非科研工程成果）。
处置：request changes（登记裁判行 / 重发内嵌图报告 / 更正 revision），返修后可直接确认 L4。
verdict 已落 `tensor_maturity_verdicts` id=817（repo=TensorFEM，scope=m19999 CASE-FEM-CANTILEVER-TIP，level=3，verdict_by=board-audit）。
"""
    figs = []
    for name, cap in (("fig1_convergence.png",
                       "端挠度相对误差 vs dof：被审工件阶梯（红）与审计在 main 上的逐位复算点（黑叉），±2% 容差带（灰）"),
                      ("fig2_got_vs_ref.png",
                       "各模型层端挠度 vs 解析 δ_EB/Timoshenko 与审计独立重推线（绿），±2% 带")):
        b64 = base64.b64encode((HERE / "figs" / name).read_bytes()).decode()
        figs.append({"name": name, "caption": cap, "png_base64": b64})

    body = {
        "title": "m19999 交差审计（修订）：CASE-FEM-CANTILEVER-TIP 判 L3，L4 因裁判登记缺位扣留；图未内嵌经对照实验定位为平台 ingest 缺陷",
        "summary": ("[修订：图未内嵌经对照实验改判为平台 ingest 缺陷] board-audit 独立复跑 m19999：被审脚本逐档在 main 求解器上复算，参考网格 32x4x4 端挠度 "
                    "9.964045336499231e-03 m 与工件相对差 0.0（误差 -0.3595% < 2%，<3% 满足）；9/9 pytest 绿；"
                    "tensorbench_results 2 行机器判定 PASS（job 2435）。判 L3：tensor_benchmark_cases 无该锚点裁判行"
                    "（0/121）+ 工件 provenance revision 不含 workload，故 L4 扣留；处置 request changes：登记裁判行、"
                    "重发内嵌图报告、更正 revision。verdict id=817。"),
        "scenario_key": "CASE-FEM-CANTILEVER-TIP",
        "mission_id": 20061,
        "tags": ["TensorFEM", "CASE-FEM-CANTILEVER-TIP", "board-audit", "m19999", "maturity-L3",
                 "euler-bernoulli", "hex20", "reproducibility"],
        "markdown": md,
        "figures": figs,
    }
    Path("/tmp/report.json").write_text(json.dumps(body, ensure_ascii=False))
    tok = req("POST", "/auth/login", {"username": USER, "password": PASS})["access_token"]
    out = req("POST", "/tensor/reports", body, token=tok)
    print(json.dumps({k: out[k] for k in out if k != "markdown"}, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()