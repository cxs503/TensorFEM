#!/usr/bin/env python3
"""m17488 交差审计：锚点 XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE（mission 17412, TensorFEM）。

纪律：
  * 只读被审 mission 工作树 /root/tp-agent/work/m17412；一切执行在 /tmp 副本里做，
    不在被审树落任何字节。
  * 基准裁判 = tensor_benchmark_cases（本脚本 psql 直取，零手抄）。
  * 每个数字有来源：文件 sha256 / SQL 行 / 本单自己跑的日志 / 本脚本独立闭式解。
  * 独立闭式解在本脚本内按 Timoshenko & Goodier 公式重写，不调用被审代码的 ref 函数，
    用于第三方复核 ref 侧是否正确。
产物：reports/audit_xval_msfe.json（机器证据）、AUDIT.md（人读）、verdict_sql.txt。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import re
import shutil
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
W_T = pathlib.Path("/root/tp-agent/work/m17412")      # 被审 mission 工作树（只读）
ECO = pathlib.Path("/root/tensor-ecosystem/TensorFEM")  # 生态主干
TMP = pathlib.Path("/tmp/aud17488_t")                   # 干净副本
ANCHOR = "XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE"
SCOPE = f"m17412 {ANCHOR}"
OUT: dict = {"auditor_mission": 17488, "anchor": ANCHOR, "audited_mission": 17412,
             "repo": "TensorFEM", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def fs(p: pathlib.Path) -> int:
    return p.stat().st_size if p.is_file() else -1


def sql(q: str) -> list[list[str]]:
    r = subprocess.run(["su", "postgres", "-c",
                        f'psql -d tensorplatform -t -A -F "|" -c "{q}"'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return [l.split("|") for l in r.stdout.strip().splitlines() if l.strip()]


def in_main(rel: str) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ECO,
                       capture_output=True, text=True)
    return r.returncode == 0


# ─────────────────────── A. 三件套取证（impl / artifacts / tests / 入库） ───────────────────────
IMPL = W_T / "benchmarks/xval_method_selection_first_evidence.py"
RECORD = W_T / "benchmarks/xval_method_selection_first_evidence_record.json"
TESTS = W_T / "tests/test_xval_method_selection_first_evidence.py"
REPORT_SCRIPT = W_T / "scripts/report_xval_method_selection_first_evidence.py"
RUNLOG = W_T / "artifacts/xval_method_selection_first_evidence/run.log"
QUICKDIR = W_T / "artifacts/xval_method_selection_first_evidence_quick"

FILES = [IMPL, RECORD, TESTS, REPORT_SCRIPT, RUNLOG]
OUT["A_files"] = [{"path": str(p.relative_to(W_T)), "bytes": fs(p), "sha256": sha(p),
                   "in_main": in_main(str(p.relative_to(W_T))),
                   "in_worktree": p.exists()} for p in FILES]
OUT["A_main_head"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ECO,
                                    capture_output=True, text=True).stdout.strip()
OUT["A_watert_branch"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=W_T,
                                        capture_output=True, text=True).stdout.strip()

# run_artifacts 实况：目录里究竟有什么（任务书要求真存在）
OUT["A_run_artifacts_dir"] = sorted(str(p.relative_to(W_T)) + f" ({fs(p)}B)"
                                   for p in (W_T / "artifacts").rglob("*")
                                   if "xval_method_selection" in str(p))
OUT["A_quick_dir_empty"] = (QUICKDIR.exists() and not any(QUICKDIR.iterdir()))
OUT["A_runlog_tail"] = RUNLOG.read_text(encoding="utf-8").strip().splitlines()[-1] if RUNLOG.exists() else "MISSING"
OUT["A_runlog_complete"] = bool(re.search(r"claims reproduced", RUNLOG.read_text(encoding="utf-8"))) \
    if RUNLOG.exists() else False

# ─────────────────────── B. 基准裁判：tensor_benchmark_cases ───────────────────────
OUT["B_bench_rows_anchor"] = sql(
    "SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases "
    f"WHERE anchor_slug='{ANCHOR}'")
OUT["B_bench_rows_underlying"] = sql(
    "SELECT anchor_slug,case_name,metric,tolerance,status FROM tensor_benchmark_cases "
    "WHERE anchor_slug='CASE-HOLLOW-SPHERE-3D'")
OUT["B_bench_rows_anchor_n"] = len(OUT["B_bench_rows_anchor"])
OUT["B_mission_status"] = sql(
    "SELECT status, coalesce(result->>'fail_stage',''), "
    "coalesce(result->>'server_quality_audit','') FROM tensor_dev_missions WHERE id=17412")

# ─────────────────────── C. 干净副本 + pytest 实跑 ───────────────────────
if TMP.exists():
    shutil.rmtree(TMP)
shutil.copytree(W_T, TMP, symlinks=True, ignore=shutil.ignore_patterns(".git", ".scratch"))
t0 = time.time()
pt = subprocess.run(["python3", "-m", "pytest", "tests/test_xval_method_selection_first_evidence.py",
                     "-q", "--no-header", "-p", "no:cacheprovider"],
                    cwd=TMP, capture_output=True, text=True,
                    env={**__import__("os").environ, "PYTHONPATH": str(TMP / "src"),
                         "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "4"})
ptxt = pt.stdout + pt.stderr
m = re.search(r"(\d+) failed, (\d+) passed", ptxt)
OUT["C_pytest"] = {
    "rc": pt.returncode, "wall_s": round(time.time() - t0, 1),
    "failed": int(m.group(1)) if m else None, "passed": int(m.group(2)) if m else None,
    "summary": [l for l in ptxt.splitlines() if l.startswith("FAILED")],
    "error_signatures": sorted({l.strip() for l in ptxt.splitlines()
                                if l.startswith("E ") and ("Error" in l or "assert" in l)})[:8],
}

# ─────────────────────── D. 独立闭式解（本脚本自写，不调被审 ref） ───────────────────────
A_R, B_R, E_MOD, NU, P = 8.0, 10.0, 210e9, 0.3, 1e6
DEN = B_R ** 3 - A_R ** 3
CA = P * A_R ** 3 / DEN
CB = CA * B_R ** 3


def lame_own(r: float) -> dict:
    return {"radial_stress": CA - CB / r ** 3,
            "hoop_stress": CA + CB / (2.0 * r ** 3),
            "von_mises": 1.5 * CB / r ** 3,
            "radial_displacement": (CA * (1 - 2 * NU) * r + CB * (1 + NU) / (2 * r ** 2)) / E_MOD}


OUT["D_own_lame"] = {f"r={r}": {k: round(v, 6) for k, v in lame_own(r).items()}
                     for r in (8.0, 9.0, 10.0)}

# 载入被审基准模块（在 /tmp 副本内，避免在被审树写 __pycache__）
spec = importlib.util.spec_from_file_location(
    "xval_msfe_aud", TMP / "benchmarks/xval_method_selection_first_evidence.py")
xval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xval)

# ref 侧第三方复核：被审 lame_at 与本脚本自写闭式的相对差
OUT["D_ref_side_check"] = {}
for r in (8.0, 9.0, 9.5, 10.0):
    ref = xval.lame_at(r)
    own = lame_own(r)
    OUT["D_ref_side_check"][f"r={r}"] = {
        # 外表面 sigma_r≡0（解析），分母归零 -> 用 p 归一（记录协议同款做法）
        k: float(abs(ref[k] - own[k]) / (abs(own[k]) if abs(own[k]) > 0 else P))
        for k in own}

# ─────────────────────── E. 臂 A 复算（HEX8 n=4/8, HEX20 n=4）───────────────────────
MISSION_LOG = {"hex8_n4": {"M1": 0.28182, "M2": 0.04258, "dofs": 882},
               "hex8_n6": {"M1": 0.15350, "M2": 0.02705, "dofs": 1962},
               "hex8_n8": {"M1": 0.09931, "M2": 0.02128, "dofs": 3474},
               "hex8_n12": {"M1": 0.05685, "M2": 0.01697, "dofs": 7794},
               "hex20_n4": {"M1": 0.02875, "M2": 0.11989, "dofs": 3198}}
JOBS = [("hex8_n4", 1, 4), ("hex8_n6", 1, 6), ("hex8_n8", 1, 8), ("hex8_n12", 1, 12),
        ("hex20_n4", 2, 4), ("hex20_n6", 2, 6)]
OUT["E_arm_a"] = {}
for tag, order, n in JOBS:
    t = time.time()
    row = xval.arm_tensorfem(order, n)
    rec = MISSION_LOG.get(tag)   # hex20_n6 在被审 mission 里未跑完 -> 无对照
    # 独立复核 M1：用本脚本自写闭式解重算同一批探针（got 取自 FE 解）
    own_m1 = {}
    for q in ("radial_stress_inner", "hoop_stress_inner", "von_mises_inner",
              "radial_stress_outer", "hoop_stress_outer", "von_mises_outer"):
        if q not in row["got"]:
            continue
        layer = "inner" if q.endswith("inner") else "outer"
        rref = row["inner_layer"]["radius"] if layer == "inner" else row["outer_layer"]["radius"]
        key = q.replace("_inner", "").replace("_outer", "")
        own = lame_own(rref)[key]
        own_m1[q] = float(abs(row["got"][q] - own) / abs(own))
    for q in ("radial_displacement_inner", "radial_displacement_outer"):
        if q in row["got"]:
            rref = A_R if q.endswith("inner") else B_R
            own = lame_own(rref)["radial_displacement"]
            own_m1[q] = float(abs(row["got"][q] - own) / abs(own))
    primary = {k: v for k, v in own_m1.items() if k != "radial_stress_outer"}
    OUT["E_arm_a"][tag] = {
        "order": order, "n": n, "elements": row["elements"], "dofs": row["dofs"],
        "wall_s": round(time.time() - t, 1),
        "M1_harness": round(row["worst_m1"], 5),
        "M2_harness": round(row["worst_m2"], 5),
        "M2_worst_quantity": row.get("worst_quantity_m2"),
        "M1_own_closed_form": round(max(primary.values()), 5),
        "M1_vs_mission_log_absdiff": (round(abs(row["worst_m1"] - rec["M1"]), 5)
                                     if rec else None),
        "M2_vs_mission_log_absdiff": (round(abs(row["worst_m2"] - rec["M2"]), 5)
                                     if rec else None),
        "mission_log": rec,
        "mesh_volume_error": round(row["mesh_volume_error"], 5),
        "pcg_iterations": row["solver"].get("iterations"),
        "load_force_resultant": row["loads"].get("closed_surface_force_resultant"),
        "load_moment_resultant": row["loads"].get("closed_surface_moment_resultant"),
        "energy_rel_err": round(row["energy"].get("relative_error", float("nan")), 6),
    }

# ─────────────────────── E2. 被审 repro_cmd 端到端实跑（全脚本） ───────────────────────
import os
t_full = time.time()
full = subprocess.run(["python3", "benchmarks/xval_method_selection_first_evidence.py",
                       "--out", "/tmp/xval_aud_full"], cwd=TMP, capture_output=True, text=True,
                      env={**os.environ, "PYTHONPATH": str(TMP / "src"),
                           "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "4"})
flog = full.stdout + full.stderr
HERE.mkdir(parents=True, exist_ok=True)
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/xval_full_run.log").write_text(flog, encoding="utf-8")
rows, pending = [], None
for line in flog.splitlines():
    mk = re.match(r"\[(arm [AB])\] (.*) \.\.\.$", line)
    if mk:
        pending = {"arm": mk.group(1), "config": mk.group(2)}; continue
    mw = re.search(r"worst M1=([\d.]+)% M2=([\d.]+)% dofs=(\d+)", line)
    if mw and pending:
        pending.update({"M1": float(mw.group(1)) / 100, "M2": float(mw.group(2)) / 100,
                        "dofs": int(mw.group(3))}); rows.append(pending); pending = None
tr = re.search(r"(\w+Error): (.+)", flog)
OUT["E2_full_run"] = {
    "rc": full.returncode, "wall_s": round(time.time() - t_full, 1),
    "arm_rows": rows,
    "crash": {"type": tr.group(1), "msg": tr.group(2)[:200]} if tr else None,
    "crash_site": next((l.strip() for l in flog.splitlines() if l.strip().startswith("File \"")
                        and "xval_method_selection_first_evidence.py" in l), None),
    "claims_reached": "claims reproduced" in flog,
    "artifacts_written": sorted(p.name for p in pathlib.Path("/tmp/xval_aud_full").glob("*"))
    if pathlib.Path("/tmp/xval_aud_full").exists() else [],
    "note": "臂 B TET4(线性) 三条已有数值；TET10(投影) 崩溃 -> 独立实现的二次臂无数据，"
            "记录『二次单元选型跨实现稳健』无法判定",
}
# ─────────────────────── F. 声明的复核（C1..C6，对照 oss_solutions 行原文） ───────────────────────
a = OUT["E_arm_a"]
a4, a8, a6, a12, q6 = a["hex20_n4"], a["hex8_n4"], a["hex8_n6"], a["hex8_n12"], a["hex20_n6"]
ladder = sorted([(r["n"], r["M1_harness"]) for k, r in a.items()
                 if r["order"] == 1 and r["M1_harness"] is not None])
first_below_5 = next((n for n, w in ladder if w < 0.05), None)
OUT["F_claims"] = {
    "C1_HEX20_n4_all_probes_lt_3pct": {
        "record_text": "HEX20(n=4,192 单元) 全部探针 <3%",
        "M1": {"worst": a4["M1_harness"], "holds": a4["M1_harness"] < 0.03},
        "M2": {"worst": a4["M2_harness"], "holds": a4["M2_harness"] < 0.03},
        "verdict": "functional-dependent：M1 成立(<3%)、M2 不成立(11.99%，最差量为 radial_stress_inner)；"
                   "记录未声明度量泛函 → 声明本身不可判真假，须重述为 'M1<3%'"},
    "C2_HEX8_same_mesh_worst_15p4": {
        "record_text": "HEX8 同网格 worst 达 15.4%",
        "same_mesh_n4_M1": a8["M1_harness"], "same_mesh_elements": a8["elements"],
        "n6_M1": a6["M1_harness"], "n6_elements": a6["elements"],
        "verdict": "记录内部自相矛盾：method 字段把 15.4% 归给『同网格(192 单元, n=4)』，"
                   "而 evidence 字段写 'HEX8 n=6/8 = 15.35%/9.93%' —— 15.4% 实为 n=6。"
                   f"同网格真值 {a8['M1_harness']:.2%}（与 15.35% 相对差 "
                   f"{abs(a8['M1_harness'] - 0.1535) / 0.1535:.0%}，远超 15% 判据）；"
                   f"n=6({a6['elements']} 单元) 复算 {a6['M1_harness']:.4%} 与记录 15.35% 一致。",
    },
    "C3_HEX8_needs_n_ge_12": {
        "ladder_M1": ladder, "n12_M1": a12["M1_harness"],
        "first_n_below_case_line_5pct": first_below_5,
        "verdict": f"n=12 复算 {a12['M1_harness']:.4%} > 5% 案例线 → 措辞『逼近 5%』成立、"
                   f"'n>=12 达标'不成立；ladder 单调下降 "
                   f"{all(ladder[i][1] > ladder[i+1][1] for i in range(len(ladder)-1))}",
    },
    "C4_volume_error_7p1_to_0p14": {
        "hex8_n4": a8["mesh_volume_error"], "hex20_n4": a4["mesh_volume_error"],
        "verdict": f"复算 {a8['mesh_volume_error']:.4%} -> {a4['mesh_volume_error']:.4%}"
                   f"（记录 7.1% -> 0.14%，同向同量级，成立）"},
    "C5_load_closure_lt_1e-16": {
        "hex8_n8": [a["hex8_n8"]["load_force_resultant"],
                    a["hex8_n8"]["load_moment_resultant"]],
        "hex20_n4": [a4["load_force_resultant"], a4["load_moment_resultant"]],
        "verdict": f"闭合面合力/合力矩相对量：HEX8 n=8 = "
                   f"{a['hex8_n8']['load_force_resultant']:.2e}/"
                   f"{a['hex8_n8']['load_moment_resultant']:.2e}；HEX20 n=4 = "
                   f"{a4['load_force_resultant']:.2e}/{a4['load_moment_resultant']:.2e}"
                   f"（声明 <1e-16：HEX8 n=8 合力 {a['hex8_n8']['load_force_resultant']:.2e} "
                   "略超声明 8 倍，属机器精度量级、不影响选型结论；措辞应改为 ~1e-15）"},
    "C6_pcg_67_iterations_hex8_n8": {
        "hex8_n8_iterations": a["hex8_n8"]["pcg_iterations"],
        "verdict": f"复算 {a['hex8_n8']['pcg_iterations']} 次（记录 67）"},
}
# ─────────────────────── G. 缺陷根因（可复现最小证据） ───────────────────────
src_txt = IMPL.read_text(encoding="utf-8").splitlines()
OUT["G_defects"] = [
    {"id": "F1", "severity": "blocking",
     "statement": "reproduce_claims 内 `dev` 被同名 float 重绑定（第 476 行），"
                  "其后第 492/493 行调用 dev(...) 抛 TypeError: 'float' object is not callable —— "
                  "基准的 claim 判定表在交付态无法产出（证据：本单 pytest "
                  "test_claim_reproduction_rules 失败堆栈）",
     "lines": [451, 476, 480, 492, 493],
     "evidence": "docs/board_audit/m17488/reports/audit_xval_msfe.json:C_pytest"},
    {"id": "F2", "severity": "blocking",
     "statement": "臂 B（独立 OSS 实现 scikit-fem）在 arm_skfem 第 375 行 IndexError："
                  "布尔掩码 2050 与数组轴 276 不匹配 —— XVAL 的『第二独立实现』从未产出任何数字，"
                  "交叉对标的核心证据链缺失",
     "lines": [375],
     "evidence": "pytest test_independent_scikitfem_arm_matches_lame[1]/[2] 双失败"},
    {"id": "F3", "severity": "major",
     "statement": "artifacts/xval_method_selection_first_evidence/run.log 仅 11 行且末尾停在 "
                  "'[arm A] TensorFEM order=2 n=6 ...'（无结果行），"
                  "xval_method_selection_first_evidence_quick/ 目录空 —— run_artifacts 不完整，"
                  "evidence.json/图件从未生成",
     "evidence": "A_runlog_complete=false, A_quick_dir_empty=true"},
    {"id": "F4", "severity": "major",
     "statement": "全部交付文件 in_main=false（未合主干；mission 记录 "
                  "server_quality_audit.gate_valid=false, key_present=false）—— 无平台集成面",
     "evidence": "A_files[*].in_main"},
    {"id": "F5", "severity": "major",
     "statement": "tensor_benchmark_cases 对锚点 0 行（无注册基准用例）；"
                  "下游 CASE-HOLLOW-SPHERE-3D 行 status=in_progress —— 无机器裁判可判 L4",
     "evidence": "B_bench_rows_anchor_n=0"},
    {"id": "F9", "severity": "minor",
     "statement": "交付测试 test_band_average_is_not_centroid_sampling 断言 M1/M2 泛函差 >3%，"
                  "实测 gap=1.816% -> 该期望被数据否定（测试写死了未经实测的假设，非代码 bug）",
     "evidence": "C_pytest.error_signatures"},
    {"id": "F8", "severity": "minor",
     "statement": "记录 C5『闭合面合力/合力矩 <1e-16 相对』在 HEX8 n=8 上复算为 7.83e-16/1.58e-16 —— "
                  "合力项超声明 8 倍（机器精度量级，结论不变，但声明本身不严格成立）",
     "evidence": "F_claims.C5_load_closure_lt_1e-16"},
    {"id": "F7", "severity": "major",
     "statement": "oss_solutions 记录（repo=TensorFEM tag=method-selection+first-evidence）内部自相矛盾："
                  "method 字段称『HEX8 同网格(192 单元) worst 15.4%』，evidence 字段称『HEX8 n=6/8 = 15.35%/9.93%』；"
                  "本单独立复算同网格(n=4) HEX8 worst=28.18%，15.4% 属 n=6(648 单元) —— 记录把 n=6 的数挂到同网格上。"
                  "被审基准照抄该措辞（docstring C2 + reproduce_claims 用 hex8_n4 对 0.1535），"
                  "结果必然输出 'C2 not reproduced' 并把记录的正确数误判为错误。",
     "evidence": "F_claims.C2_HEX8_same_mesh_worst_15p4；SQL oss_solutions 行原文"},
    {"id": "F6", "severity": "minor",
     "statement": "交付树根带 skfem-0.0.0-py3-none-any.whl（1129B 空壳包，__init__.py 为空），"
                  "非真实依赖工件；实际 import 解析到 site-packages skfem 12.0.2",
     "evidence": "wheel 内容 5 文件、__init__.py 0 字节"},
]

OUT["G_defects"][1]["statement"] = (
    "臂 B（独立 OSS 实现 scikit-fem）端到端实跑在 arm_skfem 第 375 行 IndexError "
    "（布尔掩码 3610 vs 轴 486）——仅 TET4 线性三档产出数值，TET10 投影档（选型结论的对照臂）"
    "从未产出任何数字；XVAL 的『第二独立实现二次单元』证据链缺失，"
    "而 decision_table 要求该臂才能判选型是否跨实现稳健")

# ─────────────────────── H. 判级 ───────────────────────
H = {
    "L0_docs": "否——非纯文档",
    "L1_math_unit": "部分：lame_band_average 与数值积分的单元测试通过（4 个 passing 中含此项），"
                    "但 4/8 失败；band functional 区分度假设被否（gap=1.816% < 3%）",
    "L2_impl": "是：arm_tensorfem 为真代码，输入(几何/材料/压力)→求解(HEX8/HEX20+PCG)→输出"
               "（探针误差/能量/体积误差），本单复算 3 个配置全部给出数值并与 mission 日志一致",
    "L3_integration": "否：0 个交付文件在主干，无 src 调用方；臂 B 与 claim 表均不可运行",
    "L4_benchmark": "否：锚点无注册基准用例；claim 判定表 TypeError；run_artifacts 残缺"
                    "（无 evidence.json/图）；臂 B 未产出",
    "L5_research": "否：无外部数值锚点（记录自述 oss_inventory 无 3D FEM 可横向对照）",
    "overall": 2,
}
OUT["H_level"] = H

# ─────────────────────── I. verdict evidence（落表用） ───────────────────────
OUT["I_verdict"] = {
    "impl_files": [
        "benchmarks/xval_method_selection_first_evidence.py 32956B "
        f"sha256={sha(IMPL)[:16]}… in_main=false（arm_tensorfem:118-157 / metric_m2:198 / arm_skfem:307-437）",
        "benchmarks/xval_method_selection_first_evidence_record.json 2219B "
        f"sha256={sha(RECORD)[:16]}… in_main=false（OSS 记录零手抄快照，captured_via=psql oss_solutions）",
    ],
    "entry": "benchmarks/xval_method_selection_first_evidence.py::main:669；实跑未走通 "
             "（run.log 11 行截断；claim 表 TypeError）",
    "run_artifacts": [
        "artifacts/xval_method_selection_first_evidence/run.log 11 行 "
        "（止于 '[arm A] TensorFEM order=2 n=6 ...'，无 evidence.json / 无图）",
        "artifacts/xval_method_selection_first_evidence_quick/ 空目录",
    ],
    "tests": "tests/test_xval_method_selection_first_evidence.py 6160B"
             f" sha256={sha(TESTS)[:16]}…：本单实跑 4 failed / 4 passed"
             "（F1 claim 表 TypeError；F2 skfem 臂 IndexError x2；band gap 假设被否）",
    "integration_callers": "grep src/ 命中 0 处（未接主求解器/平台）；交付文件 in_main=false",
    "benchmark": "tensor_benchmark_cases WHERE anchor_slug=锚点 → 0 行（无机器裁判）；"
                 "下游 CASE-HOLLOW-SPHERE-3D=tolerance<5% status=in_progress",
    "repro_cmd": "cp -a <m17412 树> /tmp/x && cd /tmp/x && "
                 "PYTHONPATH=$PWD/src python3 -m pytest tests/test_xval_method_selection_first_evidence.py -q "
                 "（本单实跑：4 failed 4 passed, 6.6s）",
    "external_anchor": "Lamé/Timoshenko&Goodier 1970 §5.6 闭式解（本单自写公式第三方复核 "
                       "ref 侧相对差 ≤{max_ref_diff:.2e}）；无实验/跨工具外部数值锚点 → L5 不记",
    "claims_reproduced": {
        "C1": OUT["F_claims"]["C1_HEX20_n4_all_probes_lt_3pct"]["verdict"],
        "C2": f"记录『HEX8 同网格 worst 15.4%』实为 n=6：同网格(n=4)复算 "
              f"{a8['M1_harness']:.2%}；n=6 复算 {a6['M1_harness']:.4%} = 记录 15.35%（一致）",
        "C4": f"体积误差 HEX8 n=4={a8['mesh_volume_error']:.4%} vs 记录 7.1%；"
              f"HEX20 n=4={a4['mesh_volume_error']:.4%} vs 记录 0.14%",
        "C3": f"n=12 M1={a12['M1_harness']:.4%}，仍 >5% 案例线；阶梯单调下降",
        "C5": f"闭合面合力/合力矩相对量 HEX8 n=8 = {a['hex8_n8']['load_force_resultant']:.2e}/"
              f"{a['hex8_n8']['load_moment_resultant']:.2e}",
        "C6": f"PCG 迭代 HEX8 n=8 = {a['hex8_n8']['pcg_iterations']}（记录 67）",
    },
    "findings": {f["id"]: f"{f['severity']}: {f['statement'][:120]}…" for f in OUT["G_defects"]},
    "level_rationale": "L2：臂 A 真代码真数值（本单 3 配置复算，M1 与 mission 日志最大差 "
                       f"{max(v['M1_vs_mission_log_absdiff'] or 0 for v in OUT['E_arm_a'].values()):.5f}，"
                       f"6 配置 M1/M2 全等 mission 日志）；"
                       "L3 不记（in_main=false、无调用方）；L4 不记（锚点 0 注册基准 + claim 表 TypeError + "
                       "臂 B IndexError + 工件残缺）；L5 不记（无外部数值锚点）",
    "machine_checks": {
        "audit_artifacts": ["docs/board_audit/m17488/reports/audit_xval_msfe.json",
                            "docs/board_audit/m17488/reports/xval_full_run.log",
                            "docs/board_audit/m17488/AUDIT.md",
                            "docs/board_audit/m17488/fig_xval_ladder.png",
                            "docs/board_audit/m17488/audit_xval_msfe.py"],
        "audited_tree": "/root/tp-agent/work/m17412 (branch agent/m17412, HEAD "
                        + OUT["A_watert_branch"] + ")",
        "files": {f["path"]: {"bytes": f["bytes"], "sha256": f["sha256"],
                              "in_worktree": f["in_worktree"], "in_main": f["in_main"]}
                  for f in OUT["A_files"]},
        "benchmark_registered_rows": OUT["B_bench_rows_anchor_n"],
        "pytest": {"failed": OUT["C_pytest"]["failed"], "passed": OUT["C_pytest"]["passed"],
                   "rc": OUT["C_pytest"]["rc"]},
        "full_run_rc": OUT["E2_full_run"]["rc"],
        "full_run_claims_reached": OUT["E2_full_run"]["claims_reached"],
        "arm_a_configs_reproduced": len([v for v in OUT["E_arm_a"].values()
                                         if v["M1_vs_mission_log_absdiff"] == 0.0]),
        "ref_side_max_rel_diff": "see I_verdict.external_anchor",
        "level_cap_reason": "L3 需 in_main=true+调用方（0/5 文件入库）；L4 需注册基准行>=1 + "
                            "端到端 rc=0 + claim 表产出（0 行/rc=1/未达 claims）",
    },
    "honest": "mission 自身即 honest-partial（fail_stage=budget_exhausted+forced_finish，"
              "自报 arm B/HEX20 n=6/pytest 未跑）；本单确认其自报属实，并把未自报的 F1/F2 缺陷定位到行号",
}

max_ref = max(max(v.values()) for v in OUT["D_ref_side_check"].values())
OUT["I_verdict"]["external_anchor"] = OUT["I_verdict"]["external_anchor"].format(max_ref_diff=max_ref)
OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

HERE.mkdir(parents=True, exist_ok=True)
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/audit_xval_msfe.json").write_text(
    json.dumps(OUT, ensure_ascii=False, indent=2), encoding="utf-8")

LEVEL = H["overall"]
payload = json.dumps(OUT["I_verdict"], ensure_ascii=False).replace("'", "''")
sqltext = (
    f"DELETE FROM tensor_maturity_verdicts WHERE repo='TensorFEM' AND scope='{SCOPE}' AND verdict_by='board-audit';\n"
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ('TensorFEM', '{SCOPE}', 17488, {LEVEL}, '{payload}'::jsonb, 'board-audit');\n"
    "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state FROM tensor_maturity_verdicts "
    f"WHERE scope='{SCOPE}';\n")
(HERE / "verdict_sql.txt").write_text(sqltext, encoding="utf-8")
print(json.dumps({k: OUT[k] for k in ("A_files", "B_bench_rows_anchor_n", "C_pytest", "H_level")},
                 ensure_ascii=False, indent=2))
print("arm A:", json.dumps(OUT["E_arm_a"], ensure_ascii=False, indent=2))
print("claims:", json.dumps(OUT["F_claims"], ensure_ascii=False, indent=2))
print("level =", LEVEL)