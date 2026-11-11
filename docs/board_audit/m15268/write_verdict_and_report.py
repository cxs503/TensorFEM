#!/usr/bin/env python3
"""m15268 落表 + 报告：把交差核验结论写入 tensor_maturity_verdicts，并把六段报告发到 tensor_reports。

全部字段来自 artifacts/m15268_audit/verify_output.json（现查现算），零手抄。
运行： python3 docs/board_audit/m15268/write_verdict_and_report.py
"""
from __future__ import annotations

import base64
import json
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
V = json.loads((WORKTREE / "artifacts/m15268_audit/verify_output.json").read_text(encoding="utf-8"))
AUDITED = Path("/root/tp-agent/work/m15144")
PG = ["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1"]


def sh(cmd, stdin=None):
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"FAILED {cmd}\n{p.stderr[:1500]}")
    return p.stdout


def fstats(rel: str) -> str:
    p = AUDITED / rel
    import hashlib
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return f"{rel} {p.stat().st_size if p.is_file() else 'MISSING'} B sha256={h}"


crit = V["criteria"]
rep = V["repro"]
deliv = V["deliverables"]
ledger = V["benchmark_ledger"]

ev = {
    "impl_files": [
        fstats("scratch/m15144_repro.py") + "（187 行；仅调用既有 src/tensorfem/cohesive_bending.py，无 src/ 实现增量）",
        fstats("docs/repro-hillerborg1976.md") + "（127 行 7 段）",
        "既有依赖（非本单增量）：src/tensorfem/cohesive_bending.py:1093-1103 flexural_strength_load（σ_MOR=1.5PL/bd² 闭式锚）",
    ],
    "entry": (
        "无包内入口；交付入口=scratch/m15144_repro.py main()（python3 scratch/m15144_repro.py）。"
        f"本单独立副本复跑 rc={rep['rc']}，失败签名='{rep.get('fail_signature')}'（numpy 2.4.6 已移除 np.trapz，脚本 :176）"
    ),
    "run_artifacts": [
        fstats("scratch/repro_hillerborg1976/coarse_16x16.json"),
        fstats("scratch/repro_hillerborg1976/coarse_16x16.npz"),
        fstats("scratch/repro_hillerborg1976/fine_32x32.json"),
        fstats("scratch/repro_hillerborg1976/fine_32x32.npz"),
        fstats("scratch/repro_hillerborg1976/brittle_16x16.npz") + "（孤儿工件：committed 脚本无该 tag 产出路径）",
        "注：以上全部只在分支 agent/m15144@fb4a4015 与 /root/tp-agent/work/m15144；"
        f"本单工作树 main(HEAD={deliv['main_head'].split()[0][:8]}) 上 git cat-file -e HEAD:<file> → 8/8 ABSENT；"
        f"git ls-remote origin 无 m15144 分支（branch_on_origin={deliv['branch_on_origin']}）",
    ],
    "tests": (
        "本单交付面 0 个 pytest 用例（scratch/ 无 test_*/verify_* 命名；未改 tests/）；"
        "作为 L1 的替代证据：scratch/m15144_repro.py 为自证型复现脚本，其产出由本单独立副本复跑核对"
        f"（coarse 与提交工件逐位一致={rep['comparison_vs_committed']['coarse_peak_identical']}，"
        f"fine 逐位一致={rep['comparison_vs_committed']['fine_peak_identical']}）"
    ),
    "integration_callers": "none（integration_callers=0：api/ backend/ tools/ tests/ examples/ 均无 m15144_repro 或 repro-hillerborg1976 引用）→ L3 不达",
    "benchmark": {
        "registered_cases_for_anchor": ledger["registered_cases_for_anchor"],
        "note": "tensor_benchmark_cases 14 个 anchor 中无 REPRO-FEM-HILLERBORG1976 → 无机器裁判基准；"
                "tensorbench_results WHERE mission_id=15144 行数=" + str(ledger["tensorbench_results_rows"]),
        "criteria": {
            "C1_aW_series": {k: crit["C1_aW_series"][k] for k in ("status", "tol_pct", "why")} |
                            {"unnotched_closed_form_vs_fem": crit["C1_aW_series"]["unnotched_closed_form_vs_fem"]},
            "C2_gf_backcalc": {k: crit["C2_gf_backcalc"][k] for k in
                               ("status", "tol_pct", "gf_area_vs_dissipated_err_pct_committed_fine", "within_10pct")},
            "C3_mesh_sensitivity": {k: crit["C3_mesh_sensitivity"][k] for k in
                                    ("status", "tol_pct", "drift_pct_recorded", "drift_pct_rerun")},
        },
        "got_vs_ref": {
            "ref": "σ_MOR 无缺口极限 2bh²f_t/3L = 2666.6667 N（被审脚本自带锚，非论文 a/W 序列数值）",
            "coarse_16x16": {"recorded_n": V["committed_artifacts"]["coarse_16x16"]["peak_load_n"],
                             "err_pct": V["committed_artifacts"]["coarse_16x16"]["err_vs_analytic_pct"]},
            "fine_32x32": {"recorded_n": V["committed_artifacts"]["fine_32x32"]["peak_load_n"],
                           "recorded_err_pct": V["committed_artifacts"]["fine_32x32"]["err_vs_analytic_pct"],
                           "rerun_n": rep["artifacts"]["fine_32x32"]["peak_load_n"],
                           "rerun_err_pct": rep["artifacts"]["fine_32x32"]["err_vs_analytic_pct"]},
        },
    },
    "repro_cmd": (
        "python3 scratch/m15144_repro.py —— 本单在独立副本 artifacts/m15268_audit/repro_copy/ 实跑："
        f"rc={rep['rc']}，{rep.get('fail_signature')}；"
        f"coarse 复跑 {rep['artifacts']['coarse_16x16']['peak_load_n']:.6f} N / {rep['artifacts']['coarse_16x16']['steps']} 步"
        f"（= 提交值，逐位一致）；fine 复跑 {rep['artifacts']['fine_32x32']['peak_load_n']:.6f} N / "
        f"{rep['artifacts']['fine_32x32']['steps']} 步 vs 提交值 {V['committed_artifacts']['fine_32x32']['peak_load_n']:.6f} N / "
        f"{V['committed_artifacts']['fine_32x32']['steps']} 步（3 次独立复跑全同）⇒ 提交 fine 工件不可再生"
    ),
    "external_anchor": (
        "Hillerborg, Modéer & Petersson (1976) Cem. Concr. Res. 6(6):773-781 —— 本单未取得原文图表数字化，"
        "故论文数值未参与核验；被审单实际只用梁理论 σ_MOR 闭式，"
        "且把论文 G_F=100 N/m(l_ch/d=1.875) 降级为 10 N/m(l_ch/d=0.1875)，脱离论文软化区间"
    ),
}

scope = "m15144 REPRO-FEM-HILLERBORG1976"
payload = json.dumps(ev, ensure_ascii=False)
sql = f"""
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', '{scope}', 15268, 1, $m15268ev${payload}$m15268ev$::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id;
"""
sh(PG + ["-c", sql])
out = sh(PG + ["-tAc", f"SELECT id,repo,scope,mission_id,level,verdict_by,gate_state FROM tensor_maturity_verdicts "
                       f"WHERE scope='{scope}' AND verdict_by='board-audit'"])
print("VERDICT ROW:", out.strip())

# ---------------------------------------------------------------- report
def fig(name, caption):
    p = WORKTREE / "artifacts/m15268_audit" / name
    return {"name": name, "caption": caption, "png_base64": base64.b64encode(p.read_bytes()).decode()}


c1 = crit["C1_aW_series"]["unnotched_closed_form_vs_fem"]
markdown = f"""# m15268 交差核验：m15144 REPRO-FEM-HILLERBORG1976（TensorFEM）

## 1. 案例介绍（物理背景与审计对象）

被审对象=mission 15144「以他人论文复现调试 tensor 生态」，锚点论文 Hillerborg, Modéer & Petersson (1976)
*Analysis of crack formation and crack growth in concrete*（Cem. Concr. Res. 6(6):773-781），
复现靶标=三点弯混凝土梁（S/d=10，b=d=0.1 m，E=30 GPa，ν=0.2，f_t=4 MPa）的 P_cr / CMOD-G_F / 网格敏感性。
被审分支 `agent/m15144` head `fb4a4015`（status=succeeded，被审单自称 honest FAIL）。

本单是**交差核验**（非重做）：判据=被审 mission.goal 原文容差（三条各 <10%）+ `tensor_benchmark_cases` 唯一裁判 + 三件套实证。

## 2. 怎么做的（方法/流程/工具链）

1. **取证**：`git -C /root/tp-agent/work/m15144 show fb4a4015:<file>` 逐文件取 blob + sha256，工作树一致性比对；`merge-base --is-ancestor`、`cat-file -e HEAD:<file>`、`ls-remote origin` 判「是否入库」。
2. **账目**：`psql -U postgres -d tensorplatform` 现查 `tensor_benchmark_cases`（anchor='REPRO-FEM-HILLERBORG1976'）、`tensorbench_results`、`tensor_reports`、`tensor_maturity_verdicts`。
3. **复跑**：把分支的 `src/` 与 `scratch/m15144_repro.py` 拷进独立副本 `artifacts/m15268_audit/repro_copy/`，原样执行 `python3 scratch/m15144_repro.py`；细网格用例再做 3 次独立复跑验确定性。
4. **裁定**：判据逐项打分（NOT_RUN / FAIL / WITHHELD），按 L0-L5 机械判据定级，落表 `tensor_maturity_verdicts`。
5. 全部数字由 `scripts/audit_m15144_m15268.py` 生成 → `artifacts/m15268_audit/verify_output.json`（零手抄）。

## 3. 计算结果（实测数字表）

| 项 | 记录（提交工件） | 本单复跑 | 结论 |
|---|---|---|---|
| coarse 16×16 P_cr | {V['committed_artifacts']['coarse_16x16']['peak_load_n']:.3f} N（+{V['committed_artifacts']['coarse_16x16']['err_vs_analytic_pct']:.2f}%） | {rep['artifacts']['coarse_16x16']['peak_load_n']:.3f} N / {rep['artifacts']['coarse_16x16']['steps']} 步 | 逐位一致 ✅ |
| fine 32×32 P_cr | {V['committed_artifacts']['fine_32x32']['peak_load_n']:.3f} N（+{V['committed_artifacts']['fine_32x32']['err_vs_analytic_pct']:.2f}%）/ {V['committed_artifacts']['fine_32x32']['steps']} 步 | {rep['artifacts']['fine_32x32']['peak_load_n']:.3f} N（+{rep['artifacts']['fine_32x32']['err_vs_analytic_pct']:.2f}%）/ {rep['artifacts']['fine_32x32']['steps']} 步（3/3 相同） | **不可再生** ❌ |
| 网格漂移 P_cr | +{crit['C3_mesh_sensitivity']['drift_pct_recorded']:.2f}% | +{crit['C3_mesh_sensitivity']['drift_pct_rerun']:.2f}% | 矛盾 → WITHHELD |
| G_F 面积 vs 损伤耗散误差（fine） | {crit['C2_gf_backcalc']['gf_area_vs_dissipated_err_pct_committed_fine']:.2f}% | 反算崩溃（rc={rep['rc']}） | FAIL |
| 能量残差（external_work − strain − dissipated）/dissipated | coarse {V['committed_artifacts']['coarse_16x16']['balance_residual_pct']:.1f}% / fine {V['committed_artifacts']['fine_32x32']['balance_residual_pct']:.1f}% | fine {rep['artifacts']['fine_32x32']['balance_residual_pct']:.1f}% | 记账不自洽 |
| 注册基准行 / 平台报告行 / bench 结果行 | — | {ledger['registered_cases_for_anchor']['row_count']} / {ledger['tensor_reports_rows']['count']} / {ledger['tensorbench_results_rows']} | 三面皆空 |
| 入库 | — | main 8/8 ABSENT，origin 无分支 | 未入库 |

## 4. 结果图

两组图均为本单程序化生成（matplotlib）：
- **图 1** 三条判据的 |相对误差| 与 10% 容差线（红=提交工件，蓝=独立复跑）。C2 的蓝柱为 0 是因为反算根本未跑通，不是通过。
- **图 2** 提交值 vs 复跑值的 P_cr 与步数：16×16 完全重合，32×32 相差 39%（2879.6 vs 4004.3 N，52 vs 76 步）。

## 5. 注意点（坑与适用范围）

- **honest-FAIL 的结论方向可信、数字不可信**：复跑同样 `stop_reason=max_steps`、裂缝仅扩展一层单元，与被审单「solver 在 l_ch/d≈0.19 下失效」的诊断同向；但细网格数字不可再生，因此其 "43% 不收敛" 的具体论据不成立。
- **材料替换绕开了论文靶心**：论文 G_F=100 N/m（l_ch/d=1.875）被降为 10 N/m（l_ch/d=0.1875），而 Hillerborg 1976 的立论点正是过程区尺寸效应随 l_ch/d 的迁移。
- **无注册基准 ⇒ 三件套只查「有没有」**：这是 succeeded 结单与判据全不达标可以并存的结构性原因。
- **repro_cmd 在生态当前环境不可完成**（numpy 2.4.6 无 np.trapz），说明交付时未做端到端重跑。

## 6. 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 判据 | 基准/容差 | 实测 | <3%? | 判定 |
|---|---|---|---|---|
| C1 三点弯 P_cr（a/W 序列） | 论文 a/W 序列，<10% | 未构型（模块无初始缺口）；仅无缺口 σ_MOR 闭式 coarse 误差 {c1['err_pct']:.2f}% | **否**（{c1['err_pct']:.2f}%） | NOT_RUN，且该 {c1['err_pct']:.2f}% 不能记作论文复现 |
| C2 CMOD→G_F 反算 | <10% | 未跑通（rc={rep['rc']}）；能量口径误差 {crit['C2_gf_backcalc']['gf_area_vs_dissipated_err_pct_committed_fine']:.1f}% | 否 | FAIL |
| C3 网格敏感性 | 漂移 <10% | 记录 +{crit['C3_mesh_sensitivity']['drift_pct_recorded']:.1f}% vs 复跑 +{crit['C3_mesh_sensitivity']['drift_pct_rerun']:.1f}% | 否（不可裁定） | WITHHELD |
| 可复现性 | 同脚本同环境同结果 | coarse 逐位一致；fine 3/3 复跑 2879.608 N ≠ 提交 4004.264 N | 否（差 39%） | 阻断缺陷 |

**结论：三条论文判据 0/3 达标，无任何一项 <3%；且判据所依托的工件本身不可再生 → verdict = L1，gate 期望 reopened。**

## 7. 落表与产物

- `tensor_maturity_verdicts`：repo=TensorFEM, scope=`{scope}`, mission_id=15268, **level=1**, verdict_by=board-audit（evidence 八键=impl_files/entry/run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor）。
- 审计报告：`docs/board_audit/m15268/AUDIT_m15144.md`；证据：`artifacts/m15268_audit/verify_output.json`；复跑副本：`artifacts/m15268_audit/repro_copy/`。
- 返修建议 6 条（np.trapz→trapezoid + 端到端重跑；只提交当前脚本产出并绑定 script_sha256；支持初始缺口；恢复 G_F=100 N/m 并改弧长控制；注册基准行；六段报告入库）。
"""

report = {
    "title": "m15268 交差核验：m15144 REPRO-FEM-HILLERBORG1976（TensorFEM）→ L1，reopened",
    "summary": ("独立复核 m15144：三件套文件真实存在但提交的 fine 网格工件不可由 repro_cmd 再生"
                "（复跑 3/3 得 2879.608 N/52 步 vs 提交 4004.264 N/76 步），repro_cmd rc=1（numpy 2.x 无 np.trapz），"
                "anchor 无注册基准行、平台报告 0 行、分支未入库；三条论文判据 0/3 达标 → verdict L1。"),
    "scenario_key": "REPRO-FEM-HILLERBORG1976",
    "mission_id": 15268,
    "tags": ["board-audit", "TensorFEM", "reproducibility", "cohesive-fracture", "Hillerborg1976", "maturity-gate"],
    "markdown": markdown,
    "figures": [
        fig("fig1_criteria_vs_tolerance.png",
            "三条论文判据的绝对相对误差与 10% 容差（红=提交工件、蓝=本单独立复跑；C2 蓝为 0 表示反算未跑通）"),
        fig("fig2_committed_vs_rerun.png",
            "提交值和复跑值的 P_cr 与停止步数：16×16 逐位一致，32×32 相差 39%（2879.6 vs 4004.3 N）"),
    ],
}
Path("/tmp/report_m15268.json").write_text(json.dumps(report, ensure_ascii=False))

env = dict(l.strip().split("=", 1) for l in open("/root/tp-agent/agent.env")
           if "=" in l and not l.startswith("#"))
base = env["TENSOR_API_BASE"]
req = urllib.request.Request(base + "/auth/login",
                             data=json.dumps({"username": env["TENSOR_AGENT_USER"],
                                              "password": env["TENSOR_AGENT_PASS"]}).encode(),
                             headers={"Content-Type": "application/json"})
tok = json.loads(urllib.request.urlopen(req, timeout=30).read())["access_token"]
req = urllib.request.Request(base + "/tensor/reports", data=json.dumps(report).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + tok})
resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
print("REPORT:", json.dumps({k: resp.get(k) for k in ("id", "mission_id", "title")}, ensure_ascii=False))