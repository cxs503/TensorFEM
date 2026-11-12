#!/usr/bin/env python3
"""m15426 落表：CASE-LAME-CYLINDER 交差核验结论 → tensor_maturity_verdicts（board-audit 席位）。

evidence 的判级键必须**机器可核**（/root/tp-agent/maturity_gate.py:judge 的口径）：
impl_files/entry/run_artifacts = 纯路径（相对生态仓或审计工作树），integration_callers[0]
= 能在仓内 grep 到的字面串。人读的注解一律放 *_detail 键，不污染判级键
（m15313 的教训：带注解的路径串让 files_exist 全判 False，L4 被降到 L1）。
"""
from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text(encoding="utf-8"))
A, B, C, D, E, F = (V["A_triplet"], V["B_referee"], V["C_independent_fe"],
                    V["D_repro"], V["E_tests"], V["F_report"])
fin = B["finest_mesh_got_vs_ref"]

evidence = {
    # ── 判级键（机器可核，纯路径/纯字面串）──
    "impl_files": ["benchmarks/lame_cylinder_anchor.py",
                   "src/tensorfem/thick_cylinder_1d_oracle.py",
                   "src/tensorfem/lame_reference.py"],
    "entry": "benchmarks/lame_cylinder_anchor.py",
    "run_artifacts": [f["path"] for f in A["run_artifacts"]],
    "tests": [f"tests/test_lame_cylinder_anchor.py {E['n_passed']} passed (collect={E['n_collected']}) "
              f"在 /tmp 干净副本实跑 6.59s#m15426",
              "审计侧独立 1D 轴对称 FE（本单自写，仓内无此代码）：u_r(a) vs Lamé 3.85e-6，收敛阶 "
              + str(C["order_u_r"])],
    "integration_callers": ["from .lame_reference import",
                            "lame_reference.sector_q4_mesh"],
    "benchmark": {
        "registry": "tensor_benchmark_cases WHERE anchor_slug='CASE-LAME-CYLINDER' → 0 行"
                    "（审计时全表 16 行；ILIKE '%LAME%' 亦无命中）→ 无机器裁判行（finding F1）",
        "tolerance": 0.02,
        "mesh": fin["mesh"],
        "u_r_max": {"ref_mm": 0.587302, "got_mm": 0.587254, "err": fin["u_r_max_err"]},
        "sigma_r_max": {"ref_MPa": -96.9766, "got_MPa": -96.9598, "err": fin["sigma_r_max_err"]},
        "sigma_theta_max": {"ref_MPa": 225.182, "got_MPa": 225.172, "err": fin["sigma_theta_max_err"]},
        "sigma_z_max": {"ref_MPa": 38.4615, "got_MPa": 38.4636, "err": fin["sigma_z_max_err"]},
        "verdict": fin["verdict"],
        "ref_independence_max_rel_diff": B["ref_independence_check"]["max_ref_rel_diff"],
        "invariants": B["physical_invariants"],
        "ladder": B["convergence_ladder"],
        "observed_order": B["observed_order"],
        "independent_1d_fe_n256": {"u_r": C["runs"][-1]["u_r_max_rel_err"],
                                   "sigma_r": C["runs"][-1]["sigma_r_max_rel_err"]},
    },
    "repro_cmd": "cd <clean copy of TensorFEM @3674bfb6> && TORCH_DEVICE_BACKEND_AUTOLOAD=0 "
                 "OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/lame_cylinder_anchor.py",
    # ── 人读注解 ──
    "impl_files_detail": [
        f"benchmarks/lame_cylinder_anchor.py {A['impl_files'][0]['bytes']}B "
        f"sha256={A['impl_files'][0]['sha256'][:16]}… in_main=True（Q4 极坐标阶梯 + 证据/图/报告生成）",
        f"src/tensorfem/thick_cylinder_1d_oracle.py {A['impl_files'][1]['bytes']}B "
        f"sha256={A['impl_files'][1]['sha256'][:16]}… in_main=True（Chebyshev 配点独立算子，σ/u 与闭式解 5.5e-12）",
        f"src/tensorfem/lame_reference.py {A['impl_files'][2]['bytes']}B "
        f"sha256={A['impl_files'][2]['sha256'][:16]}… in_main=True（复用生产面）",
        f"docs/lame-cylinder-anchor-case.md 11704B 六段①-⑥齐全={F['sections_complete']}，4 图被引用",
    ],
    "run_artifacts_detail": [
        f"{f['path']} {f['bytes']}B sha256={f['sha256'][:16]}… in_main={f['in_main']}"
        for f in A["run_artifacts"]],
    "entry_detail": "benchmarks/lame_cylinder_anchor.py::main；实跑 rc=0 / 21.2s（D_repro）",
    "integration_callers_detail": [
        "src/tensorfem/thermoelastic_annulus.py:52、incompressible_locking.py:260、rotating_disc_reference.py:40、"
        "structural_verification_face.py:93、reference_strong_form_audit.py:71（grep 命中 15 处）",
        "新增 thick_cylinder_1d_oracle 无 src 调用方（grep 0 处）：只做验证不接求解器，如实记",
    ],
    "tests_not_run": ["GitHub CI (x86) 未跑；本单全部实跑在 loongarch64 本地",
                      "未做 p_o≠0 工况与 E/ν 联合敏感性（p_o=0 单工况）",
                      "F2（报告串号）根因未定位，未做受控复现测试"],
    "external_anchor": "Lamé 1852 / Timoshenko & Goodier 3rd ed. Art.27 / Roark Table 13.5 case 1"
                       "（闭式解；无实验/文献数值锚点）→ L5 不记",
    "level_rationale": "L1 14 tests 实跑全绿；L2 真代码+真工件（clean-copy 重跑：除 timestamp_utc/git_rev 外 "
                       "evidence JSON 逐键相同、4 图字节级同哈希）；L3 生产面 lame_reference 15 处 src 调用；"
                       "L4 4 级阶梯+逐项 ref/got+收敛阶 2.00+ref 独立复算 9.7e-16+独立 1D FE 复核；"
                       "L5 不记（标准解析解复现，无外部数值锚点）",
    "findings": {f["id"]: f"{f['severity']}: {f['statement']}" for f in V["findings"]},
    "audit_artifacts": ["docs/board_audit/m15426/verify_output.json",
                        "docs/board_audit/m15426/AUDIT.md",
                        "docs/board_audit/m15426/fig_lame_audit.png",
                        "docs/board_audit/m15426/audit_lame_cylinder.py",
                        "docs/board_audit/m15426/make_figure.py"],
}

LEVEL = V["H_level"]["overall"]
SCOPE, REPO, BY = "m15325 CASE-LAME-CYLINDER", "TensorFEM", "board-audit"
payload = json.dumps(evidence, ensure_ascii=False).replace("'", "''")
sql = (f"DELETE FROM tensor_maturity_verdicts WHERE repo='{REPO}' AND scope='{SCOPE}' AND verdict_by='{BY}';\n"
       f"INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       f"VALUES ('{REPO}', '{SCOPE}', 15426, {LEVEL}, '{payload}'::jsonb, '{BY}');\n")
pathlib.Path("/tmp/m15426_verdict.sql").write_text(sql, encoding="utf-8")
r = subprocess.run(["su", "postgres", "-c", "psql -d tensorplatform -v ON_ERROR_STOP=1 -f /tmp/m15426_verdict.sql"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr, "rc=", r.returncode)
print(subprocess.run(["su", "postgres", "-c",
                      f"psql -d tensorplatform -c \"SELECT id,repo,scope,mission_id,level,verdict_by,gate_state "
                      f"FROM tensor_maturity_verdicts WHERE scope='{SCOPE}';\""],
                     capture_output=True, text=True).stdout)