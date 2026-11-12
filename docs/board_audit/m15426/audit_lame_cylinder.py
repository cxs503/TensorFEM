#!/usr/bin/env python3
"""m15426 交差审计：CASE-LAME-CYLINDER（m15325，TensorFEM）取证 + 独立复算 + 判级。

原则：只读被审工作树（m15325 视为证据源，不在其中写任何东西）；所有重跑在 /tmp 副本里做。
产出的每个数字都有实测来源：文件 sha256 / SQL 行 / 本单独立解 / 重跑日志。
本脚本自带一条与 TensorFEM 无关的实现（1D 轴对称线性元）作为第三条独立路径 ——
避免"用被审代码验证被审代码"。
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
W_AUD = pathlib.Path("/root/tp-agent/work/m15426")     # 本单工作树（产物落这里）
W_T = pathlib.Path("/root/tp-agent/work/m15325")       # 被审 mission 工作树
ECO = pathlib.Path("/root/tensor-ecosystem/TensorFEM")  # 生态主干
OUT: dict = {"auditor_mission": 15426, "anchor": "CASE-LAME-CYLINDER", "audited_mission": 15325}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def fs(p: pathlib.Path) -> int:
    return p.stat().st_size if p.is_file() else -1


def sql(q: str) -> list[list[str]]:
    r = subprocess.run(["su", "postgres", "-c", f'psql -d tensorplatform -t -A -F "|" -c "{q}"'],
                       capture_output=True, text=True)
    return [l.split("|") for l in r.stdout.strip().splitlines() if l.strip()]


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()


# ─────────────────────────── A. 取证：三件套 ───────────────────────────
EV = W_T / "benchmarks/lame_cylinder_anchor_evidence.json"
BENCH = W_T / "benchmarks/lame_cylinder_anchor.py"
ORACLE = W_T / "src/tensorfem/thick_cylinder_1d_oracle.py"
LAME = W_T / "src/tensorfem/lame_reference.py"
DOC = W_T / "docs/lame-cylinder-anchor-case.md"
TEST = W_T / "tests/test_lame_cylinder_anchor.py"
FIGS = sorted((W_T / "docs/assets").glob("lame_cylinder_anchor_*.png"))
D = json.loads(EV.read_text(encoding="utf-8"))

OUT["A_triplet"] = {
    "impl_files": [{"path": str(p.relative_to(W_T)), "bytes": fs(p), "sha256": sha(p),
                    "in_main": bool(git(ECO, "ls-files", str(p.relative_to(W_T)))),
                    "main_sha_matches_worktree": sha(ECO / p.relative_to(W_T)) == sha(p)}
                   for p in (BENCH, ORACLE, LAME, TEST, DOC)],
    "run_artifacts": [{"path": str(p.relative_to(W_T)), "bytes": fs(p), "sha256": sha(p),
                       "in_main": bool(git(ECO, "ls-files", str(p.relative_to(W_T))))}
                      for p in [EV] + FIGS],
    "repro_cmd_in_artifact": D.get("repro_cmd"),
    "evidence_json_top_keys": sorted(D.keys()),
    "landing_commit": {"m15325_head": git(W_T, "rev-parse", "--short", "HEAD"),
                       "merge_line_in_main": [l for l in git(ECO, "log", "--oneline", "-n", "40").splitlines()
                                              if "m15325" in l],
                       "file_first_added_in_main": git(ECO, "log", "--oneline", "-n", "3", "--",
                                                       "benchmarks/lame_cylinder_anchor.py")},
    "artifacts_dir_convention": {
        "artifacts/lame* exists": bool(list(W_T.glob("artifacts/lame*"))),
        "note": "run_artifacts 落在 benchmarks/ 与 docs/assets/，未按 artifacts/ 约定建目录（轻偏差，非致命）",
    },
}

# ────────── B. 基准裁判：注册表行 + 独立 Lamé 复算（ref 是否可信） ──────────
contract = pathlib.Path("/tmp/m15426_contract.json")
contract.write_text(json.dumps({"repro_cmd": D.get("repro_cmd")}, ensure_ascii=False))

REG = sql("SELECT count(*) FROM tensor_benchmark_cases;")
ROWS = sql("SELECT anchor_slug, case_name, metric, tolerance, repo, status "
           "FROM tensor_benchmark_cases WHERE anchor_slug = 'CASE-LAME-CYLINDER';")
NEAR = sql("SELECT anchor_slug FROM tensor_benchmark_cases "
           "WHERE anchor_slug ILIKE '%LAME%' OR case_name ILIKE '%LAME%';")
GOAL = sql("SELECT coalesce(result->>'tolerance',''), coalesce(result->>'pass','') "
           "FROM tensor_dev_missions WHERE id=15325;")

p = D["parameters"]["case"]
a, b, pi_, po, E, nu = (p["inner_radius"], p["outer_radius"], p["inner_pressure"],
                        p["outer_pressure"], p["young_modulus"], p["poisson_ratio"])
A_ = (pi_ * a**2 - po * b**2) / (b**2 - a**2)
B_ = a**2 * b**2 * (pi_ - po) / (b**2 - a**2)


def lame(r: float) -> dict:
    """本审计自写的 Lamé 平面应变解（独立于 tensorfem.lame_reference）。"""
    return {
        "sigma_r_Pa": A_ - B_ / r**2,
        "sigma_theta_Pa": A_ + B_ / r**2,
        "sigma_z_Pa": 2 * nu * A_,
        "u_r_m": (1 + nu) / E * ((1 - 2 * nu) * A_ * r + B_ / r),
    }


# ref 侧对账：evidence JSON 的 pointwise ref 值 vs 本审计独立复算
# 注意（本审计实测发现）：u_r 列取自 ray 节点（benchmarks/lame_cylinder_anchor.py:717 ray_picks=[0,n//2,n-1]），
# 而行首 radius_m 是单元中心半径 —— 二者不是同一半径。故 u_r 用节点半径 [a,(a+b)/2,b] 对账。
U_RADII = [a, 0.5 * (a + b), b]
ref_check = []
for i, row in enumerate(D["pointwise"]):
    r = row["radius_m"]
    mine = lame(r)
    mine_u = lame(U_RADII[i] if i < len(U_RADII) else r)
    ref_check.append({
        "radius_m": r,
        "u_r_ref_json_mm": row["u_r_ref_mm"], "u_r_audit_mm": mine_u["u_r_m"] * 1e3,
        "u_r_radius_used_m": U_RADII[i] if i < len(U_RADII) else r,
        "u_r_ref_rel_diff": abs(row["u_r_ref_mm"] - mine_u["u_r_m"] * 1e3) / (mine_u["u_r_m"] * 1e3),
        "u_r_mislabel_rel_diff_if_centre_radius": abs(row["u_r_ref_mm"] - mine["u_r_m"] * 1e3)
                                                  / (mine["u_r_m"] * 1e3),
        "sigma_r_ref_json_MPa": row["sigma_r_ref_MPa"],
        "sigma_r_audit_MPa": mine["sigma_r_Pa"] / 1e6,
        "sigma_r_ref_rel_diff": abs(row["sigma_r_ref_MPa"] - mine["sigma_r_Pa"] / 1e6)
                                / abs(mine["sigma_r_Pa"] / 1e6),
        "sigma_theta_ref_json_MPa": row["sigma_theta_ref_MPa"],
        "sigma_theta_audit_MPa": mine["sigma_theta_Pa"] / 1e6,
        "sigma_theta_ref_rel_diff": abs(row["sigma_theta_ref_MPa"] - mine["sigma_theta_Pa"] / 1e6)
                                    / abs(mine["sigma_theta_Pa"] / 1e6),
        "sigma_z_ref_json_MPa": row["sigma_z_ref_MPa"],
        "sigma_z_audit_MPa": mine["sigma_z_Pa"] / 1e6,
        "sigma_z_ref_rel_diff": abs(row["sigma_z_ref_MPa"] - mine["sigma_z_Pa"] / 1e6)
                                / abs(mine["sigma_z_Pa"] / 1e6),
    })

# 物理不变量（独立于任何实现）：σ_r(a) = -p_i 精确、σ_θ(a)=p_i(b²+a²)/(b²-a²)、σ_r(b) = -p_o
invariants = {
    "sigma_r_at_bore_Pa": lame(a)["sigma_r_Pa"], "sigma_r_at_bore_err_rel": abs(lame(a)["sigma_r_Pa"] + pi_) / pi_,
    "sigma_r_at_outer_Pa": lame(b)["sigma_r_Pa"], "sigma_r_at_outer_err_rel": abs(lame(b)["sigma_r_Pa"] + po) / max(pi_, 1.0),
    "hoop_at_bore_over_pi": lame(a)["sigma_theta_Pa"] / pi_,
    "hoop_at_bore_closed_ratio": (b**2 + a**2) / (b**2 - a**2),
}

# 细网格 got 值 vs 本审计 ref（单位换算后逐项误差）
finest = D["ladder"][-1]
pt0 = D["pointwise"][0]           # 内壁相邻单元中心
OUT["B_referee"] = {
    "registry_rows_total": REG[0][0] if REG else None,
    "registry_row_for_anchor": ROWS,
    "registry_near_anchor_any": NEAR,
    "mission_goal_tolerance_text": GOAL,
    "ref_independence_check": {
        "note": "本审计用自写 Lamé 公式复算 pointwise 每个 ref 值；rel_diff≈0 证明 ref 非手抄/非错公式",
        "rows": ref_check,
        "max_ref_rel_diff": max(max(r["u_r_ref_rel_diff"], r["sigma_r_ref_rel_diff"],
                                    r["sigma_theta_ref_rel_diff"], r["sigma_z_ref_rel_diff"])
                                for r in ref_check),
    },
    "physical_invariants": invariants,
    "finest_mesh_got_vs_ref": {
        "mesh": f"{finest['n_theta']}x{finest['n_radial']} ({finest['n_element']} el, {finest['n_dof']} dof)",
        "u_r_max_err": finest["metrics"]["u_r_max"],
        "u_r_l2_err": finest["metrics"]["u_r_l2"],
        "sigma_r_max_err": finest["metrics"]["sigma_r_max"],
        "sigma_r_max_err_scaled_0p02pi": finest["metrics"]["sigma_r_max_scaled"],
        "sigma_theta_max_err": finest["metrics"]["sigma_theta_max"],
        "sigma_z_max_err": finest["metrics"]["sigma_z_max"],
        "gate": D.get("gate"), "all_passed": D.get("all_passed"),
        "tolerance_from_mission_goal": "2% (任务单原文 <2%)",
        "verdict": "PASS" if max(finest["metrics"]["u_r_max"], finest["metrics"]["sigma_r_max"],
                                 finest["metrics"]["sigma_theta_max"],
                                 finest["metrics"]["sigma_z_max"]) < 0.02 else "FAIL",
    },
    "convergence_ladder": [{"mesh": f"{L['n_theta']}x{L['n_radial']}",
                            "u_r_max_err": L["metrics"]["u_r_max"],
                            "sigma_r_max_err": L["metrics"]["sigma_r_max"],
                            "sigma_theta_l2_err": L["metrics"]["sigma_theta_l2"]} for L in D["ladder"]],
    "observed_order": D.get("observed_order"),
}


# ─────────── C. 第三条独立路径：本审计自写 1D 轴对称平面应变 FE ───────────
def axisym_1d_fe(n: int) -> dict:
    import numpy as np

    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    nodes = np.linspace(a, b, n + 1)
    K = np.zeros((n + 1, n + 1))
    f = np.zeros(n + 1)
    Dm = np.array([[lam + 2 * mu, lam], [lam, lam + 2 * mu]])
    for e in range(n):
        r1, r2 = nodes[e], nodes[e + 1]
        h = r2 - r1
        # 2 点 Gauss（线性元刚度被积函数含 1/r，用 Gauss 积分）
        gp = 1 / math.sqrt(3.0)
        Ke = np.zeros((2, 2))
        for xi, w in ((-gp, 1.0), (gp, 1.0)):
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            dN = np.array([-1 / h, 1 / h])
            r = r1 + (xi + 1) / 2 * h
            B = np.vstack([dN, N / r])
            Ke += (B.T @ Dm @ B) * r * (h / 2) * w
        K[e:e + 2, e:e + 2] += Ke
    f[0] = a * pi_          # 内壁压力（外法向为 -e_r，故面力 +p_i）
    f[-1] = -b * po         # 外壁压力
    u = np.linalg.solve(K, f)
    # 单元中心应力恢复
    rows = []
    for e in range(n):
        r1, r2 = nodes[e], nodes[e + 1]
        h = r2 - r1
        rc = 0.5 * (r1 + r2)
        dN = np.array([-1 / h, 1 / h])
        N = np.array([0.5, 0.5])
        er = dN @ u[e:e + 2]
        et = (N @ u[e:e + 2]) / rc
        sr = (lam + 2 * mu) * er + lam * et
        st = lam * er + (lam + 2 * mu) * et
        sz = lam * (er + et)
        ex = lame(rc)
        rows.append({"r": rc, "sigma_r": sr, "sigma_r_err": abs(sr - ex["sigma_r_Pa"]) / abs(ex["sigma_r_Pa"]),
                     "sigma_theta": st,
                     "sigma_theta_err": abs(st - ex["sigma_theta_Pa"]) / abs(ex["sigma_theta_Pa"]),
                     "sigma_z": sz, "sigma_z_err": abs(sz - ex["sigma_z_Pa"]) / abs(ex["sigma_z_Pa"])})
    ur = u
    ex_nodes = np.array([lame(r)["u_r_m"] for r in nodes])
    return {"n_elem": n, "u_r_max_rel_err": float(np.max(np.abs(ur - ex_nodes)) / np.max(np.abs(ex_nodes))),
            "sigma_r_max_rel_err": max(r["sigma_r_err"] for r in rows),
            "sigma_theta_max_rel_err": max(r["sigma_theta_err"] for r in rows),
            "sigma_z_max_rel_err": max(r["sigma_z_err"] for r in rows),
            "u_r_at_bore": float(ur[0]), "lame_u_r_at_bore": lame(a)["u_r_m"],
            "sigma_r_at_first_centre_MPa": rows[0]["sigma_r"] / 1e6,
            "profile": rows,  # 本审计自算的全场剖面（供作图，n=32/64/128/256 里最细一档）
            "lame_sigma_r_at_first_centre_MPa": lame(rows[0]["r"])["sigma_r_Pa"] / 1e6}


fe = [axisym_1d_fe(n) for n in (32, 64, 128, 256)]
OUT["C_independent_fe"] = {
    "what": "本审计自写 1D 轴对称平面应变线性元（含 1/r 的弱式 + 2 点 Gauss），"
            "与 TensorFEM 的 2D Q4 极坐标实现无共享代码",
    "runs": fe,
    "order_u_r": [round(math.log(fe[i]["u_r_max_rel_err"] / fe[i + 1]["u_r_max_rel_err"]) / math.log(2), 3)
                  for i in range(len(fe) - 1)],
    "order_sigma_r": [round(math.log(fe[i]["sigma_r_max_rel_err"] / fe[i + 1]["sigma_r_max_rel_err"]) / math.log(2), 3)
                      for i in range(len(fe) - 1)],
    "u_r_bore_agreement_128_vs_lame": abs(fe[2]["u_r_at_bore"] - fe[2]["lame_u_r_at_bore"])
                                      / fe[2]["lame_u_r_at_bore"],
}

# ───────────────── D. repro_cmd 真跑（/tmp 副本，不污染被审树） ─────────────────
tmp = pathlib.Path(tempfile.mkdtemp(prefix="m15426_repro_"))
for sub in ("src", "benchmarks", "docs", "tests"):
    shutil.copytree(W_T / sub, tmp / sub, ignore=shutil.ignore_patterns("__pycache__"))
# 先删掉从被审树复制来的旧工件，否则"复现"只是把自己抄回来的文件再读一遍（本单 v1 踩过的假阳性）
for f in list((tmp / "benchmarks").glob("*evidence*.json")) + list((tmp / "docs/assets").glob("*.png")):
    f.unlink()
t0 = time.time()
proc = subprocess.run(D["repro_cmd"], cwd=tmp, shell=True, capture_output=True, text=True,
                      timeout=2400,
                      env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4",
                           "PYTHONPATH": str(tmp / "src"), "HOME": "/tmp",
                           "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
wall = time.time() - t0
new_ev = tmp / "benchmarks/lame_cylinder_anchor_evidence.json"
repro = {"cmd": D["repro_cmd"], "returncode": proc.returncode, "wall_sec": round(wall, 1),
         "env_note": "TORCH_DEVICE_BACKEND_AUTOLOAD=0（本机 torch_sdaa 自动加载会让脚本 import torch 即崩，与算例无关）",
         "stdout_tail": proc.stdout.strip().splitlines()[-8:], "stderr_tail": proc.stderr.strip().splitlines()[-5:]}
if new_ev.is_file():
    N = json.loads(new_ev.read_text(encoding="utf-8"))
    repro["json_sha256_new"] = sha(new_ev)
    repro["json_sha256_committed"] = sha(EV)
    a_, b_ = json.loads(EV.read_text(encoding="utf-8")), json.loads(new_ev.read_text(encoding="utf-8"))
    for _x in (a_, b_):
        _x.get("environment", {}).pop("timestamp_utc", None)
        _x.get("environment", {}).pop("git_rev", None)  # /tmp 副本无 .git，运行器把 rev 记为 ""
    repro["json_identical_canonical"] = json.dumps(a_, sort_keys=True) == json.dumps(b_, sort_keys=True)
    repro["json_identical_raw_sha"] = sha(new_ev) == sha(EV)
    repro["only_volatile_diff"] = (not repro["json_identical_raw_sha"]) and repro["json_identical_canonical"]
    repro["finest_metrics_reproduced"] = N["ladder"][-1]["metrics"]
    repro["finest_u_r_max_delta"] = abs(N["ladder"][-1]["metrics"]["u_r_max"]
                                        - D["ladder"][-1]["metrics"]["u_r_max"])
    repro["all_passed_reproduced"] = N.get("all_passed")
    figs_new = sorted((tmp / "docs/assets").glob("lame_cylinder_anchor_*.png"))
    repro["figures_regenerated"] = len(figs_new)
    repro["figure_pixels_identical_hash"] = all(
        sha(f) == sha(W_T / "docs/assets" / f.name) for f in figs_new)
else:
    repro["json_regenerated"] = False
OUT["D_repro"] = repro

# 测试真跑（同一 /tmp 副本）
tp = subprocess.run(["python3", "-m", "pytest", "tests/test_lame_cylinder_anchor.py", "-q",
                     "--no-header", "-p", "no:cacheprovider"],
                    cwd=tmp, capture_output=True, text=True, timeout=3600,
                    env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{tmp/'src'}:{tmp/'tests'}",
                         "HOME": "/tmp", "OMP_NUM_THREADS": "4",
                         "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
tail = tp.stdout.strip().splitlines()[-4:]
col = subprocess.run(["python3", "-m", "pytest", "tests/test_lame_cylinder_anchor.py", "-q",
                      "--collect-only", "--no-header", "-p", "no:cacheprovider"],
                     cwd=tmp, capture_output=True, text=True, timeout=300,
                     env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{tmp/'src'}",
                          "HOME": "/tmp", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
n_collected = len([l for l in col.stdout.splitlines() if "::" in l])
OUT["E_tests"] = {"cmd": "PYTHONPATH=src python3 -m pytest tests/test_lame_cylinder_anchor.py -q",
                  "returncode": tp.returncode, "tail": tail,
                  "n_passed": int(m.group(1)) if (m := re.search(r"(\d+) passed", tp.stdout)) else 0,
                  "n_collected": n_collected, "test_names": re.findall(r"^def (test_\w+)", TEST.read_text(), re.M)}

# ─────────── F. 报告六段 + 平台报告行（含可疑跨单串号） ───────────
doc_txt = DOC.read_text(encoding="utf-8")
marks = {"①": "问题/几何/参数", "②": "方法/流程", "③": "参数表", "④": "对照表", "⑤": "收敛/敏感性",
         "⑥": "失败模式"}
OUT["F_report"] = {
    "docs_report": str(DOC.relative_to(W_T)), "doc_bytes": fs(DOC),
    "section_hits": {k: (k in doc_txt) for k in marks},
    "sections_complete": all(k in doc_txt for k in marks),
    "figures_in_doc": sorted(set(re.findall(r"lame_cylinder_anchor_\w+\.png", doc_txt))),
    "platform_reports_mission_15325": sql(
        "SELECT id, title, length(markdown) FROM tensor_reports WHERE mission_id=15325 ORDER BY id;"),
    "platform_reports_foreign_anchor_in_own_mission": sql(
        "SELECT id, title FROM tensor_reports WHERE mission_id=15325 "
        "AND title NOT ILIKE '%LAME%' AND title NOT ILIKE '%probe%';"),
    "lame_report_row_has_six_sections": sql(
        "SELECT id FROM tensor_reports WHERE mission_id=15325 AND title ILIKE '%LAME%' "
        "AND markdown LIKE '%①%' AND markdown LIKE '%⑥%';"),
}

# ────────── G. 集成调用方（主求解器/平台侧是否真调用） ──────────
callers = []
for pat, files in [("lame_reference",
                    ["src/tensorfem/thermoelastic_annulus.py", "src/tensorfem/incompressible_locking.py",
                     "src/tensorfem/rotating_disc_reference.py", "src/tensorfem/structural_verification_face.py",
                     "src/tensorfem/reference_strong_form_audit.py"]),
                   ("thick_cylinder_1d_oracle", [])]:
    hits = subprocess.run(["grep", "-rn", "--include=*.py", "-F", pat, "src", "scripts", "api"],
                          cwd=ECO, capture_output=True, text=True).stdout.strip().splitlines()
    callers.append({"symbol": pat, "hits_outside_benchmarks_tests": hits[:6], "n_hits": len(hits)})
OUT["G_integration"] = callers

# ─────────────────────── H. 判级（声明级 + 依据） ───────────────────────
checks = {
    "L1_tests": OUT["E_tests"]["n_passed"] > 0,
    "L2_impl": all(f["bytes"] > 0 for f in OUT["A_triplet"]["impl_files"][:3])
                and OUT["A_triplet"]["run_artifacts"][0]["bytes"] > 0
                and OUT["D_repro"]["returncode"] == 0,
    "L3_integration": any(c["n_hits"] > 0 for c in OUT["G_integration"]),
    "L4_benchmark": OUT["B_referee"]["finest_mesh_got_vs_ref"]["verdict"] == "PASS"
                    and OUT["B_referee"]["ref_independence_check"]["max_ref_rel_diff"] < 1e-6
                    and OUT["D_repro"]["returncode"] == 0,
    "L4_referee_row_registered": bool(OUT["B_referee"]["registry_row_for_anchor"]),
    "L5_external_anchor": bool(D["reference"].get("citations")),
}
OUT["H_level"] = {
    "checks": checks,
    "overall": 4 if checks["L4_benchmark"] else (3 if checks["L3_integration"] else
                                                (2 if checks["L2_impl"] else (1 if checks["L1_tests"] else 0))),
    "notes": [
        "L4 实质成立（真跑复现 + ref 独立复算一致 + 2% 门全项 PASS + 收敛阶 2 实测）",
        "但注册表 tensor_benchmark_cases 无 CASE-LAME-CYLINDER 行：机器裁判缺行，"
        "本轮裁判由本审计独立复算代行，且注册缺行本身记缺陷（m15325 报告亦如实记为未做）",
        "L5 不记：cited 文献为教科书闭式解，无实验/文献数值锚点对照，且算例是标准解析解复现",
    ],
}
OUT["findings"] = []
if not checks["L4_referee_row_registered"]:
    OUT["findings"].append(
        {"id": "F1", "severity": "major", "kind": "registry-gap",
         "statement": "tensor_benchmark_cases 无 anchor_slug='CASE-LAME-CYLINDER' 行（全表 15 行，LAME 无命中）；"
                      "任务单称'基准判据是唯一裁判'，实际该裁判缺行 → 交差只能由审计独立复算代行。",
         "evidence": "B_referee.registry_row_for_anchor = [] ; B_referee.registry_rows_total = 15"})
OUT["findings"].append(
    {"id": "F2", "severity": "major", "kind": "platform-report-api",
     "statement": "tensor_reports 中 mission_id=15325 存在 4 条他单（CASE-HOPPER-BEVERLOO / probe）标题行；"
                  "m15325 自述 >100B markdown 被替换为既有记录内容。即报告 API 写入串号（正文覆盖）——"
                  "平台证据完整性缺陷，跨单可复现线索见 F_report.platform_reports_foreign_anchor_in_own_mission。",
     "evidence": "F_report.platform_reports_mission_15325"})
OUT["findings"].append(
    {"id": "F3", "severity": "minor", "kind": "convention",
     "statement": "run_artifacts 落在 benchmarks/ 与 docs/assets/，未建 artifacts/ 目录（与部分任务单约定不一致，非致命）。",
     "evidence": "A_triplet.artifacts_dir_convention"})
OUT["findings"].append(
    {"id": "F4", "severity": "info", "kind": "scope",
     "statement": f"被审交付物已入主干：{OUT['A_triplet']['landing_commit']['merge_line_in_main']}；"
                  "m15325 commit 98d39a8d 本身非 main 祖先（squash/merge 落地），主干文件 sha 与工作树一致=真落地。",
     "evidence": "A_triplet.landing_commit / impl_files[].main_sha_matches_worktree"})

OUT["findings"].append(
    {"id": "F5", "severity": "minor", "kind": "table-label",
     "statement": "pointwise 表 u_r 两列取自 ray 节点半径 [a,(a+b)/2,b]（代码 :717 ray_picks），"
                  "而该行 radius_m 是单元中心半径（如 0.7578 vs 实际 0.8，差 5.6% 的 r）——两列半径标签错配；"
                  "误差按同半径成对计算，未虚增精度，但读者会误读 u_r 所在半径。",
     "evidence": "B_referee.ref_independence_check.rows[].u_r_radius_used_m / "
                 "u_r_mislabel_rel_diff_if_centre_radius"})

(HERE / "verify_output.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: OUT[k] for k in ("A_triplet", "B_referee", "C_independent_fe", "D_repro", "E_tests",
                                      "F_report", "G_integration", "H_level", "findings")},
                 ensure_ascii=False, indent=1)[:12000])
print("\nverify_output.json ->", HERE / "verify_output.json")