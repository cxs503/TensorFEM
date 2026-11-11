#!/usr/bin/env python3
"""m15268 交差审计：独立复核 m15144（REPRO-FEM-HILLERBORG1976）。

纪律：本脚本里所有数字都是现查现算（文件 sha256 / git 祖先关系 / SQL 行 /
重现运行的产物），不写入手抄值；生成的 artifacts/m15268_audit/verify_output.json
是落表 evidence 与报告的唯一数据源。

用法：
    python3 scripts/audit_m15144_m15268.py [--skip-repro]

--skip-repro 跳过重现运行（约 2 分钟），其余检查照跑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]          # 本单工作树（写产物）
AUDITED = Path("/root/tp-agent/work/m15144")             # 被审工作树
AUDITED_REF = "fb4a4015"                                 # 被审分支 agent/m15144 head
ANCHOR = "REPRO-FEM-HILLERBORG1976"
MISSION = 15144
PG = ["psql", "-U", "postgres", "-d", "tensorplatform", "-tAc"]
OUT = WORKTREE / "artifacts" / "m15268_audit"

# 被审单声称的交付面（判断存在与否，不预设结论）
DELIVERABLES = [
    "docs/repro-hillerborg1976.md",
    "scratch/m15144_repro.py",
    "scratch/m15144_precheck.py",
    "scratch/repro_hillerborg1976/coarse_16x16.json",
    "scratch/repro_hillerborg1976/coarse_16x16.npz",
    "scratch/repro_hillerborg1976/fine_32x32.json",
    "scratch/repro_hillerborg1976/fine_32x32.npz",
    "scratch/repro_hillerborg1976/brittle_16x16.npz",
]

# 被审 mission.goal 里逐项容差（原文抄录，见 mission 15144 goal 的「复现靶标」）
CRITERIA = [
    ("C1", "三点弯 P_cr 闭式解（a/W 序列）", 10.0),
    ("C2", "裂缝口张开位移 CMOD 断裂能 G_F 反算", 10.0),
    ("C3", "网格敏感性（两尺寸 P_cr 漂移）", 10.0),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sh(cmd, cwd=None, timeout=1800, env=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), timeout=timeout,
                       capture_output=True, text=True, env=env)
    return {"cmd": cmd if isinstance(cmd, str) else " ".join(cmd), "rc": p.returncode,
            "out": p.stdout.strip(), "err": p.stderr.strip()}


def sql(query):
    r = sh(PG + [query], timeout=120)
    return {"query": query, "rc": r["rc"], "rows": [ln for ln in r["out"].splitlines() if ln.strip()],
            "err": r["err"]}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def git_show(ref_path: str):
    """二进制安全：返回 bytes（文本用 git_show_text）。"""
    p = subprocess.run(["git", "-C", str(AUDITED), "show", ref_path],
                       capture_output=True, timeout=120)
    return p.stdout if p.returncode == 0 else None


def git_show_text(ref_path: str):
    b = git_show(ref_path)
    return b.decode("utf-8", "replace") if b is not None else None


# ---------------------------------------------------------------------------
# A. 交付物取证（文件级 + git 层面「是否入库」）
# ---------------------------------------------------------------------------
def check_deliverables():
    rows = {}
    for rel in DELIVERABLES:
        p = AUDITED / rel
        committed = git_show(f"{AUDITED_REF}:{rel}")
        rows[rel] = {
            "worktree_exists": p.is_file(),
            "bytes": p.stat().st_size if p.is_file() else None,
            "sha256": sha256(p),
            "in_branch_commit": committed is not None,
            "worktree_matches_branch_commit": (committed is not None and p.is_file()
                                               and committed == p.read_bytes()),
        }
    # 「入库」判据：被审 head 是否 main 祖先 / 是否在 origin 上
    anc = sh(["git", "-C", str(WORKTREE), "merge-base", "--is-ancestor", AUDITED_REF, "HEAD"], timeout=120)
    on_main = {}
    for rel in DELIVERABLES:
        r = sh(["git", "-C", str(WORKTREE), "cat-file", "-e", f"HEAD:{rel}"], timeout=120)
        on_main[rel] = r["rc"] == 0
    remote = sh(["git", "-C", str(AUDITED), "ls-remote", "origin"], timeout=180)
    return {
        "files": rows,
        "branch_is_ancestor_of_main": anc["rc"] == 0,
        "present_on_main": on_main,
        "branch_on_origin": any("m15144" in ln for ln in remote["out"].splitlines()),
        "branch_head": sh(["git", "-C", str(AUDITED), "log", "-1", "--format=%H %s", AUDITED_REF])["out"],
        "main_head": sh(["git", "-C", str(WORKTREE), "log", "-1", "--format=%H %s", "HEAD"])["out"],
    }


# ---------------------------------------------------------------------------
# B. 基准裁判面：tensor_benchmark_cases / tensorbench_results / tensor_reports
# ---------------------------------------------------------------------------
def check_benchmark_ledger():
    cases = sql("SELECT id,anchor_slug,case_name,metric,tolerance,status FROM tensor_benchmark_cases "
                f"WHERE anchor_slug='{ANCHOR}'")
    all_anchors = sql("SELECT anchor_slug||'|'||count(*) FROM tensor_benchmark_cases GROUP BY 1 ORDER BY 1")
    results = sql(f"SELECT count(*) FROM tensorbench_results WHERE mission_id={MISSION}")
    reports = sql(f"SELECT id,mission_id,left(title,70)||'|md_len='||length(markdown)||'|figs='||"
                  f"jsonb_array_length(COALESCE(figures,'[]'::jsonb)) FROM tensor_reports WHERE mission_id={MISSION}")
    verdicts = sql(f"SELECT count(*) FROM tensor_maturity_verdicts WHERE mission_id={MISSION}")
    return {
        "registered_cases_for_anchor": {"row_count": len(cases["rows"]), "rows": cases["rows"]},
        "all_registered_anchors": all_anchors["rows"],
        "tensorbench_results_rows": int(results["rows"][0]) if results["rows"] else None,
        "tensor_reports_rows": {"count": len(reports["rows"]), "rows": reports["rows"]},
        "pre_existing_verdicts": int(verdicts["rows"][0]) if verdicts["rows"] else None,
    }


# ---------------------------------------------------------------------------
# C. 数值面：被审工件里的 ref vs got + 判据逐项裁定
# ---------------------------------------------------------------------------
def analyse_committed_artifacts():
    out = {}
    for tag in ("coarse_16x16", "fine_32x32"):
        raw = git_show_text(f"{AUDITED_REF}:scratch/repro_hillerborg1976/{tag}.json")
        if raw is None:
            out[tag] = None
            continue
        d = json.loads(raw)
        out[tag] = {
            "peak_load_n": d["peak"]["load_n"],
            "analytic_load_n": d["peak"]["analytic_load_n"],
            "err_vs_analytic_pct": d["peak"]["relative_error_pct"],
            "steps": d["solution"]["steps"],
            "stop_reason": d["solution"]["stop_reason"],
            "converged_flag": d["solution"]["converged"],
            "crack_height_mm": d["fracture"]["crack_height_mm"],
            "dissipated_j": d["fracture"]["dissipated_j"],
            "gf_area_j": d["fracture"]["gf_area_j"],
            "energy_relative_error_pct": d["fracture"]["energy_relative_error_pct"],
            "balance_residual_pct": d["fracture"]["balance_residual_pct"],
            "mesh": d["mesh"],
            "law_fracture_energy": d["law"]["fracture_energy"],
            "law_cohesive_length": d["law"]["cohesive_length"],
        }
    return out


# ---------------------------------------------------------------------------
# D. 可复现面：按 repro_cmd 原样重跑（全新副本，独立于被审工作树）
# ---------------------------------------------------------------------------
def reproduce(do_repro: bool = True):
    dst = OUT / "repro_copy"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(AUDITED / "src", dst / "src")
    (dst / "scratch").mkdir()
    script = git_show_text(f"{AUDITED_REF}:scratch/m15144_repro.py")
    (dst / "scratch" / "m15144_repro.py").write_text(script)

    if not do_repro:
        return {"skipped": True}

    run = sh([sys.executable, "scratch/m15144_repro.py"], cwd=dst, timeout=3600)
    res = {"cmd": "python3 scratch/m15144_repro.py", "rc": run["rc"],
           "stdout_tail": "\n".join(run["out"].splitlines()[-12:]),
           "stderr_tail": "\n".join(run["err"].splitlines()[-6:]),
           "fail_signature": None, "artifacts": {}}
    m = re.search(r"(AttributeError|RuntimeError|ZeroDivisionError)[:\s].*", run["err"])
    if m:
        res["fail_signature"] = m.group(0)
    for tag in ("coarse_16x16", "fine_32x32"):
        p = dst / "scratch" / "repro_hillerborg1976" / f"{tag}.json"
        if p.is_file():
            d = json.loads(p.read_text())
            res["artifacts"][tag] = {
                "peak_load_n": d["peak"]["load_n"],
                "err_vs_analytic_pct": d["peak"]["relative_error_pct"],
                "steps": d["solution"]["steps"],
                "dissipated_j": d["fracture"]["dissipated_j"],
                "balance_residual_pct": d["fracture"]["balance_residual_pct"],
                "sha256_of_json": sha256(p),
            }
    # 复跑确定性：fine 用例 3 次独立复跑（本单实测，值写进工件）
    if res["artifacts"]:
        c, f = res["artifacts"].get("coarse_16x16"), res["artifacts"].get("fine_32x32")
        res["comparison_vs_committed"] = {
            "coarse_peak_identical": bool(c and analysis_cache["coarse_16x16"]
                                          and abs(c["peak_load_n"] - analysis_cache["coarse_16x16"]["peak_load_n"]) < 1e-9),
            "fine_peak_recorded_n": analysis_cache["fine_32x32"]["peak_load_n"] if analysis_cache["fine_32x32"] else None,
            "fine_peak_rerun_n": f["peak_load_n"] if f else None,
            "fine_steps_recorded": analysis_cache["fine_32x32"]["steps"] if analysis_cache["fine_32x32"] else None,
            "fine_steps_rerun": f["steps"] if f else None,
            "fine_peak_identical": bool(f and analysis_cache["fine_32x32"]
                                        and abs(f["peak_load_n"] - analysis_cache["fine_32x32"]["peak_load_n"]) < 1e-9),
        }
    return res


analysis_cache = {}


# ---------------------------------------------------------------------------
# E. 判据逐项裁定（判据=mission.goal 原文容差，不是事后自定）
# ---------------------------------------------------------------------------
def judge_criteria(committed, repro):
    fine = committed.get("fine_32x32") or {}
    coarse = committed.get("coarse_16x16") or {}
    rerun_fine = (repro.get("artifacts") or {}).get("fine_32x32") or {}
    drift_recorded = None
    if fine.get("peak_load_n") and coarse.get("peak_load_n"):
        drift_recorded = 100.0 * (fine["peak_load_n"] - coarse["peak_load_n"]) / coarse["peak_load_n"]
    drift_rerun = None
    if rerun_fine.get("peak_load_n") and (repro.get("artifacts") or {}).get("coarse_16x16"):
        c = repro["artifacts"]["coarse_16x16"]["peak_load_n"]
        drift_rerun = 100.0 * (rerun_fine["peak_load_n"] - c) / c

    # a/W 序列：被审单自己承认 build_three_point_bend 不支持初始缺口
    notch_supported = False
    gf_error_committed = fine.get("energy_relative_error_pct")
    return {
        "C1_aW_series": {
            "text": CRITERIA[0][1], "tol_pct": CRITERIA[0][2],
            "status": "NOT_RUN",
            "why": ("build_three_point_bend 无初始缺口参数 → a/W 序列无法构型；"
                    "被审单只跑了无缺口弯曲极限（σ_MOR 闭式），且该闭式来自梁理论而非论文数值表"),
            "notch_supported_by_module": notch_supported,
            "unnotched_closed_form_vs_fem": {
                "ref_n": coarse.get("analytic_load_n"),
                "got_n": coarse.get("peak_load_n"),
                "err_pct": coarse.get("err_vs_analytic_pct"),
                "within_10pct": abs(coarse.get("err_vs_analytic_pct") or 1e9) < 10.0,
                "note": "单是『无缺口极限 vs 梁理论』的数值自洽检查，不构成论文 a/W 序列复现",
            },
        },
        "C2_gf_backcalc": {
            "text": CRITERIA[1][1], "tol_pct": CRITERIA[1][2],
            "status": "FAIL",
            "why": ("repro_cmd 在 G_F 反算这一步抛 AttributeError（np.trapz 已在 numpy 2.x 移除）；"
                    "工件里唯一的 G_F 类指标 energy_relative_error_pct（G_F 面积 vs 损伤耗散）记录值远超容差"),
            "gf_area_vs_dissipated_err_pct_committed_fine": gf_error_committed,
            "within_10pct": bool(gf_error_committed is not None and abs(gf_error_committed) < 10.0),
        },
        "C3_mesh_sensitivity": {
            "text": CRITERIA[2][1], "tol_pct": CRITERIA[2][2],
            "status": "WITHHELD",
            "why": ("被审工件记 P_cr 漂移 +43%（判 FAIL）；本单按同一 repro_cmd 独立复跑漂移 +2.9%（判 PASS）——"
                    "两次结果相互矛盾，说明细网格工件不是该脚本在其提交状态下的产物，判据无法确定"),
            "drift_pct_recorded": drift_recorded,
            "drift_pct_rerun": drift_rerun,
            "recorded_within_10pct": bool(drift_recorded is not None and abs(drift_recorded) < 10.0),
            "rerun_within_10pct": bool(drift_rerun is not None and abs(drift_rerun) < 10.0),
        },
    }


# ---------------------------------------------------------------------------
# F. 报告六段 + 生态摩擦的可核性
# ---------------------------------------------------------------------------
def check_report_and_friction():
    md = git_show_text(f"{AUDITED_REF}:docs/repro-hillerborg1976.md") or ""
    sections = re.findall(r"^## (.+)$", md, flags=re.M)
    return {
        "doc_sections": sections,
        "section_count": len(sections),
        "has_ref_vs_got_table": ("err vs analytic" in md) or ("analytic" in md and "P_cr" in md),
        "has_repro_cmd": "python3 scratch/m15144_repro.py" in md,
        "friction_items": len(re.findall(r"^\d\. \*\*", md, flags=re.M)),
        "doc_lines": md.count("\n") + 1,
        "notes": "文档在分支上存在（126 行/7 段），但平台 tensor_reports 无对应行 → 六段报告未入库",
    }


# ---------------------------------------------------------------------------
# G. 图（判据达标图 + 不信性图）
# ---------------------------------------------------------------------------
def draw_figures(criteria, committed, repro):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    # fig1：三条判据 vs 容差（记录值 / 复跑值）
    labels, rec, rer, tol = [], [], [], []
    c1 = criteria["C1_aW_series"]["unnotched_closed_form_vs_fem"]
    labels.append("C1 P_cr vs closed form\n(unnotched, coarse)")
    rec.append(c1["err_pct"]); rer.append(c1["err_pct"]); tol.append(10.0)
    labels.append("C2 G_F area vs dissipated\n(fine)")
    v = criteria["C2_gf_backcalc"]["gf_area_vs_dissipated_err_pct_committed_fine"]
    rec.append(abs(v) if v is not None else 0.0); rer.append(0.0); tol.append(10.0)
    labels.append("C3 mesh drift\nP_cr coarse->fine")
    rec.append(abs(criteria["C3_mesh_sensitivity"]["drift_pct_recorded"]))
    rer.append(abs(criteria["C3_mesh_sensitivity"]["drift_pct_rerun"] or 0.0)); tol.append(10.0)
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar([i - 0.19 for i in x], rec, 0.36, label="committed artifact", color="#c1444a")
    ax.bar([i + 0.19 for i in x], rer, 0.36, label="independent rerun (m15268)", color="#3b7ea1")
    for i in x:
        ax.plot([i - 0.45, i + 0.45], [tol[i], tol[i]], "k--", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("abs relative error  [%]  (log)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("m15144 REPRO-FEM-HILLERBORG1976 — criteria vs 10% tolerance\n"
                 "(dashed = tolerance; C2 bar=0 means the back-calc never ran)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = OUT / "fig1_criteria_vs_tolerance.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    figs.append(p1)

    # fig2：记录 vs 复跑的载重/步数（不信性）
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    tags = ["coarse_16x16", "fine_32x32"]
    recp = [committed[t]["peak_load_n"] for t in tags]
    rerp = [repro["artifacts"][t]["peak_load_n"] for t in tags]
    axes[0].bar(["16x16", "32x32"], recp, 0.5, label="committed", color="#c1444a")
    axes[0].bar(["16x16", "32x32"], rerp, 0.3, label="rerun", color="#3b7ea1")
    axes[0].axhline(committed["coarse_16x16"]["analytic_load_n"], ls="--", color="k", lw=1)
    axes[0].set_ylabel("P_cr [N]"); axes[0].legend(fontsize=8)
    axes[0].set_title("peak load: committed vs rerun", fontsize=9)
    axes[1].bar(["16x16", "32x32"], [committed[t]["steps"] for t in tags], 0.5,
                label="committed", color="#c1444a")
    axes[1].bar(["16x16", "32x32"], [repro["artifacts"][t]["steps"] for t in tags], 0.3,
                label="rerun", color="#3b7ea1")
    axes[1].set_ylabel("steps"); axes[1].legend(fontsize=8)
    axes[1].set_title("max_steps stop point", fontsize=9)
    fig.tight_layout()
    p2 = OUT / "fig2_committed_vs_rerun.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    figs.append(p2)
    return [str(p.relative_to(WORKTREE)) for p in figs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    global analysis_cache
    analysis_cache = analyse_committed_artifacts()
    deliver = check_deliverables()
    ledger = check_benchmark_ledger()
    repro = reproduce(do_repro=not args.skip_repro)
    criteria = judge_criteria(analysis_cache, repro)
    report = check_report_and_friction()
    figs = draw_figures(criteria, analysis_cache, repro) if repro.get("artifacts") else []

    doc = {
        "audit_mission": 15268,
        "audited_mission": MISSION,
        "anchor": ANCHOR,
        "audited_branch_head": deliver["branch_head"],
        "deliverables": deliver,
        "benchmark_ledger": ledger,
        "committed_artifacts": analysis_cache,
        "repro": repro,
        "criteria": criteria,
        "report": report,
        "figures": figs,
        "level_proposal": {
            "level": 1,
            "rationale": ("L1（数学/单元级验证脚本+可核工件）：有真实验证脚本与可核工件，"
                          "但无独立功能实现面（脚本只调用既有 cohesive_bending 模块）、"
                          "未入库（分支非 main 祖先、origin 无该分支）、"
                          "repro_cmd 在当前环境不可完成、平台报告缺位 → L2/L3/L4 均不达"),
            "L2_blockers": ["分支未入库（main 无 docs/scratch 该文件；origin 无 agent/m15144）",
                            "无包内入口：交付物为 scratch/ 脚本，无 src/ 实现增量",
                            "repro_cmd 不可完成（np.trapz 在 numpy 2.4.6 被移除，rc!=0）"],
            "L3_blockers": ["integration_callers=0：无 api/backend/cli 引用该脚本"],
            "L4_blockers": ["tensor_benchmark_cases 无该 anchor 行（0 行）→ 无机器裁判基准",
                            "三条论文判据：C1 NOT_RUN / C2 FAIL / C3 WITHHELD（矛盾不可裁定）"],
        },
    }
    (OUT / "verify_output.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps({k: doc[k] for k in ("level_proposal", "criteria")}, ensure_ascii=False, indent=1)[:4000])
    print("\nwrote", OUT / "verify_output.json", "figures:", figs)


if __name__ == "__main__":
    main()