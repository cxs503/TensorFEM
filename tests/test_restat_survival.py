"""AUDIT-RESTAT v3 落地存活层：口径纯函数单测 + 产物不变量断言。

两组：
  A. 纯函数（脱库/脱 git）：清单解析、路径归一、类别、落地档、膨胀、漏斗、仓名映射、
     以及渲染器在合成数据上不崩。
  B. 产物不变量（若 artifacts/audit_restat/restat_v3_*.json 存在）：核账恒等式
     （宣称 = 存活 + 仅同名 + 丢失）、漏斗单调、档位计数守恒、快照字段非空。
     这组不重跑统计，只断言「报告内部自洽」，防止渲染/聚合环节静默改口径。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "audit_restat"))

import restat_v3_survival as r3  # noqa: E402


# ------------------------------------------------------------------ A. 纯函数

def test_parse_manifest_merges_sources_and_dedups():
    cf = json.dumps(["src/a.py", "docs/x.md"])
    odc = json.dumps(["docs/x.md", "tests/test_a.py"])
    assert r3.parse_manifest(cf, odc) == ["src/a.py", "docs/x.md", "tests/test_a.py"]


@pytest.mark.parametrize("bad", [None, "", "not json", "{}", '{"a": 1}', "[]"])
def test_parse_manifest_tolerates_garbage(bad):
    assert r3.parse_manifest(bad) == []


def test_parse_manifest_skips_non_strings_and_strips_quotes():
    assert r3.parse_manifest('[1, null, " src/a.py ", ""]') == ["src/a.py"]


def test_parse_manifest_handles_two_sources_of_same_file():
    # 同一文件既在 changed_files（相对）又在 on_disk_changes（绝对）：两个形态都保留，
    # 由 path_candidates 归一 —— 解析层不做去重猜测。
    cf = json.dumps(["src/a.py"])
    odc = json.dumps(["/srv/tensor/agent/work/m1/repo/src/a.py"])
    got = r3.parse_manifest(cf, odc)
    assert len(got) == 2
    # 第一个候选（剥 checkout 层）必须一致，否则同一文件会被算成两条命
    assert r3.path_candidates(got[0])[0] == r3.path_candidates(got[1])[0] == "src/a.py"


def test_path_candidates_relative():
    assert r3.path_candidates("src/a.py") == ["src/a.py"]
    assert r3.path_candidates("./src/a.py") == ["src/a.py"]


def test_path_candidates_absolute_gives_two_ordered_options():
    cands = r3.path_candidates("/srv/tensor/agent/work/m22858/src/tensorlbm/lbm_config.py")
    # 剥 checkout 层优先，其次不剥
    assert cands == ["tensorlbm/lbm_config.py", "src/tensorlbm/lbm_config.py"]


def test_path_candidates_absolute_ambiguous_layer_keeps_both():
    # 无法从字符串判断有没有 checkout 层：两种候选都保留，严格比对时各自成立即可
    assert r3.path_candidates("/srv/tensor/agent/work/m1/src/a.py") == ["a.py", "src/a.py"]
    assert r3.path_candidates("/srv/tensor/agent/work/m1/only.py") == ["only.py"]


def test_path_candidates_rejects_outside_repo_and_escaping():
    assert r3.path_candidates("/etc/passwd") == []
    assert r3.path_candidates("../../etc/passwd") == []


def test_normalize_path_provenance_levels():
    assert r3.normalize_path("src/a.py") == ("src/a.py", "repo_relative")
    rel, prov = r3.normalize_path("/srv/tensor/agent/work/m1/repo/src/a.py")
    assert rel == "src/a.py" and prov == "worktree_absolute"
    assert r3.normalize_path("/etc/passwd") == (None, "outside_repo")
    assert r3.normalize_path("") == (None, "empty")


@pytest.mark.parametrize("path,kind", [
    ("src/tensorfem/plate.py", "src"),
    ("tensorlbm/config.py", "src"),
    ("tests/test_x.py", "tests"),
    ("test_root.py", "tests"),
    ("docs/CASE.md", "docs"),
    ("papers/p1/manuscript.md", "docs"),
    ("benchmarks/xval/run.py", "bench"),
    ("tools/gen.py", "tools"),
    ("scripts/post_report.py", "tools"),
    ("artifacts/run/evidence.json", "artifacts"),
    ("figures/fig1.png", "artifacts"),
    ("_scratch/probe.py", "scratch"),
    ("README.md", "root_doc"),
    ("post_report_x.py", "root_script"),
    ("report.json", "artifacts"),
    ("data/table.csv", "artifacts"),
    ("weird.xyz", "root_other"),
    (None, "unparsed"),
])
def test_classify_path(path, kind):
    assert r3.classify_path(path) == kind


def test_classify_root_script_is_not_impl_core():
    # m22264 判据：根目录一次性脚本不算「实现」
    assert r3.classify_path("post_report_m123.py") == "root_script"
    assert r3.classify_path("src/pkg/mod.py") == "src"


def test_resolve_repo_exact_alias_case_and_missing():
    avail = ["TensorFEM", "ship-performance-platform"]
    assert r3.resolve_repo("TensorFEM", avail) == "TensorFEM"
    assert r3.resolve_repo("ship-platform", avail) == "ship-performance-platform"
    assert r3.resolve_repo("tensorfem", avail) == "TensorFEM"
    assert r3.resolve_repo("NoSuchRepo", avail) is None
    assert r3.resolve_repo(None, avail) is None


def test_survival_strict_basename_absent():
    files = ["src/a.py", "docs/b.md", "c/gone.py"]
    rec = r3.survival(files, {"src/a.py"}, {"a.py", "b.md"})
    assert rec["n_files"] == 3
    assert rec["n_strict"] == 1
    assert rec["n_basename_only"] == 1      # docs/b.md 仅同名存在
    assert rec["n_absent"] == 1             # c/gone.py
    assert rec["surv_kinds"] == Counter({"src": 1})
    assert rec["absent_samples"] == ["c/gone.py"]


def test_survival_absolute_path_hits_via_second_candidate():
    files = ["/srv/tensor/agent/work/m1/repo/src/a.py"]
    rec = r3.survival(files, {"src/a.py"}, {"a.py"})
    assert rec["n_strict"] == 1
    assert rec["n_abs_uncommitted"] == 1


def test_survival_basename_fallback_can_be_disabled():
    rec = r3.survival(["docs/b.md"], set(), None)
    assert rec["n_basename_only"] == 0 and rec["n_absent"] == 1


def test_survival_absent_samples_capped():
    files = ["gone%d.py" % i for i in range(20)]
    rec = r3.survival(files, set(), None)
    assert len(rec["absent_samples"]) == 5 and rec["n_absent"] == 20


@pytest.mark.parametrize("rec,expected", [
    ({"n_files": 0}, "P_nomanifest"),
    ({"n_files": 3, "n_strict": 0}, "P0_gone"),
    ({"n_files": 2, "n_strict": 1, "surv_kinds": Counter({"docs": 1})}, "P1_docs_only"),
    ({"n_files": 2, "n_strict": 1, "surv_kinds": Counter({"bench": 1})}, "P1_impl_peripheral"),
    ({"n_files": 2, "n_strict": 1, "surv_kinds": Counter({"src": 1})}, "P2_core"),
    ({"n_files": 2, "n_strict": 2, "surv_kinds": Counter({"src": 1, "tests": 1})},
     "P2_core_and_tests"),
])
def test_persistence_class(rec, expected):
    assert r3.persistence_class(rec, has_manifest=rec.get("n_files", 0) > 0) == expected


def test_root_script_alone_is_not_impl():
    rec = {"n_files": 1, "n_strict": 1, "surv_kinds": Counter({"root_script": 1})}
    assert r3.persistence_class(rec) == "P1_docs_only"


def test_ratio_and_inflation():
    assert r3.ratio_pct(1, 4) == 25.0
    assert r3.ratio_pct(1, 0) is None
    assert r3.inflation(100, 50) == 2.0
    assert r3.inflation(100, 0) is None


def test_funnel_retention_chain():
    f = r3.funnel([("a", 100), ("b", 50), ("c", 25)])
    assert [s["retention_pct"] for s in f] == [None, 50.0, 50.0]
    assert [s["n"] for s in f] == [100, 50, 25]


def test_render_md_on_synthetic_report():
    rep = {
        "generated_at": "2026-11-16T00:00:00+00:00", "snapshot": "2026-11-16 00:00:00+00",
        "coverage": {"missions_total": 10, "missions_claiming_manifest": 8,
                     "manifest_coverage_pct": 80.0, "missions_unmapped_repo": 0,
                     "missions_without_head": 0, "files_claimed": 20, "files_alive_strict": 10,
                     "files_alive_basename_only": 2, "files_absent": 8,
                     "file_survival_pct": 50.0, "file_survival_upper_pct": 60.0,
                     "abs_path_entries": 4, "abs_path_share_pct": 20.0,
                     "contested_share_of_surviving_pct": 30.0},
        "inflation": {"missions_claiming": 8, "missions_persisted_any": 6,
                      "missions_persisted_impl": 4, "missions_zero_persisted": 2,
                      "zero_persisted_pct": 25.0, "claim_over_persisted_factor": 1.333,
                      "claim_over_impl_factor": 2.0, "files_claim_over_alive_factor": 2.0},
        "mission_status_counts": {"succeeded": 10, "failed": 3},
        "ledger_status_counts": {"open": 5, "verified": 2}, "ledger_total": 7,
        "ledger_unfinished": [{"status": "open", "domain": "FEM", "slug": "X",
                               "acceptance": "a|b"}],
        "persistence_classes": {"P0_gone": 2, "P1_docs_only": 1, "P1_impl_peripheral": 1,
                               "P2_core": 1, "P2_core_and_tests": 3, "P_nomanifest": 2},
        "per_kind": {"src": {"claimed": 5, "alive_strict": 4, "survival_pct": 80.0},
                     "artifacts": {"claimed": 5, "alive_strict": 1, "survival_pct": 20.0}},
        "per_repo": {"TensorFEM": {"missions": 10, "with_manifest": 8, "files": 20,
                                   "files_alive": 10, "file_survival_pct": 50.0,
                                   "classes": {"P0_gone": 2}}},
        "per_pr_state": {"merged": {"missions": 6, "with_manifest": 6, "files": 15,
                                    "files_alive": 9, "file_survival_pct": 60.0,
                                    "classes": {"P0_gone": 1}}},
        "gate_x_persistence": {"L2": {"P2_core": 3}}, "gate_l2plus_zero_persistence": [],
        "gate_l2plus_zero_persistence_count": 0,
        "funnel_missions": r3.funnel([("succeeded", 10), ("claim", 8), (">=1 alive", 6)]),
    }
    md = r3.render_md(rep)
    for token in ("## 0. 结论摘要", "## 1. 六类成果统计表", "## 6. 判定层",
                  "## 7. 未达验收待完成名单", "## 8. 注意点与局限"):
        assert token in md
    assert "TensorFEM" in md and "X" in md


# ------------------------------------------------------------------ B. 产物不变量

ARTIFACTS = sorted((ROOT / "artifacts" / "audit_restat").glob("restat_v3_*.json"))


@pytest.mark.skipif(not ARTIFACTS, reason="no restat_v3 artifact in tree")
def test_artifact_accounting_identities_hold():
    rep = json.loads(ARTIFACTS[-1].read_text())
    cov = rep["coverage"]
    assert cov["files_claimed"] == (cov["files_alive_strict"]
                                    + cov["files_alive_basename_only"]
                                    + cov["files_absent"])
    # 存活档之和 = succeeded 总数（不重不漏）
    assert sum(rep["persistence_classes"].values()) == cov["missions_total"]
    # 漏斗单调不增
    ns = [s["n"] for s in rep["funnel_missions"]]
    assert ns == sorted(ns, reverse=True)
    assert rep["snapshot"] and rep["generated_at"]
    # 每个被核对的仓都有 HEAD sha 记录
    for name, head in rep["repo_heads"].items():
        assert "head_sha" in head and len(head["head_sha"]) >= 7
    # 分仓计数不大于总数
    assert sum(v["missions"] for v in rep["per_repo"].values()) == cov["missions_total"]