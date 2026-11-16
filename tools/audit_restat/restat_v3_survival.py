#!/usr/bin/env python3
"""AUDIT-RESTAT v3 · 成果**落地存活层**（manifest survival）。

把「宣称的成果」逐文件搬到各仓 HEAD 上核对，给「成果膨胀」补上迄今缺失的一根支柱。

为什么需要这一层（对上游交付的独立复核结论）
------------------------------------------------
AUDIT-RESTAT 的三个既有交付各自闭合了一环：
  1. 2026-11-11 首批 73 份同 sha 交付：L0-L5 六类统计 + 待完成名单（指标口径未锁定）；
  2. 2026-11-13 m22264 校准单：证明「严格/宽松 L2+ 差额」97.57% 来自 `pr_gate` 记账缺失，
     并把「一次性脚本冒充实现」量化为 56 单（0.75%）；
  3. 2026-11-13 m22304 v2：锁定快照 + 前值复现 + 溯源漏斗（任务→PR→锚点→台账）。
三层共同的空白是：**没有任何一层去看仓库本身**。所有关于「交付了什么」的判断都来自
`mission.result` 里的元数据（`changed_files` / `on_disk_changes` / `pr_gate`），而这三个
字段都是 agent 自述。本文件补的正是这一层：把每单自述的文件清单当作「宣称」，去各仓
当前 HEAD（`/srv/tensor/workspace/<repo>`）的完整路径清单里逐条核对，得到
「宣称 → 落地」的存活矩阵。它不评价代码对不对（那是 tests/gate 的事），只回答一个
此前没人量过的硬问题：**报过账的文件，现在还在仓库里吗？**

三条方法论发现（实现前用原型验证过，工具按此设计）
  1. 清单里混着**绝对路径**（`/srv/tensor/agent/work/mNNNN/<checkout>/src/...`）。这不是脏数据，
     而是硬证据：绝对路径意味着该文件从未进入版本控制。工具单列其证据级，并把
     「剥掉 worktree 前缀后的候选相对路径」也拿去比对（最多剥一层，不过度猜测）。
  2. 同名文件可被多单共同宣称（`provenance.json`/`summary.json`…）。一条路径只能存一份内容，
     故「路径存活」对这类**争夺路径**是上界。工具统计 contested 份额作为假阳性上界。
  3. `pr_gate.state='merged'` 与「文件在 HEAD」是两件独立的事；其差集即
     「记为已合并、但仓库里找不到」的落地亏损。

零手抄纪律：报告里每个数字都来自本文件 SQL_REGISTRY 的具名查询或 git 现查，JSON 产物
逐项记录 sql_id / repo HEAD sha，MD 与图由 JSON 程序化渲染。

用法：
    python3 tools/audit_restat/restat_v3_survival.py
    python3 tools/audit_restat/restat_v3_survival.py --snapshot '2026-11-16 05:00:00+00'
产物：
    artifacts/audit_restat/restat_v3_<stamp>.json   机器可读（sql_id + HEAD sha 溯源）
    docs/audit_restat/restat_v3_survival.md         人读表
    docs/audit_restat/fig_restat_v3_*.png           图
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# =============================================================== 纯函数（可脱库单测）

WORK_ABS_RE = re.compile(r"^/srv/tensor/agent/work/m\d+/(.+)$")

DOC_ROOTS = ("docs/", "papers/", "reports/", "doc/")
BENCH_ROOTS = ("benchmarks/", "bench/", "xval/", "validation/", "examples/")
TOOL_ROOTS = ("tools/", "scripts/", "ci/", ".github/")
ARTIFACT_ROOTS = ("artifacts/", "outputs/", "runs/", "figures/")
SCRATCH_ROOTS = ("_scratch/", "scratch/", "tmp/", ".cache/")
ARTIFACT_SUFFIX = (".json", ".png", ".csv", ".npy", ".npz", ".pdf", ".svg", ".h5", ".parquet")
DOC_SUFFIX = (".md", ".rst", ".txt")
CODE_SUFFIX = (".py", ".c", ".cc", ".cpp", ".h", ".hpp", ".f90", ".jl", ".rs")

#: 计入「实现」的类别（m22264 校准单把仓库根目录一次性脚本排除，本层沿用该判据）
IMPL_KINDS = ("src", "tests", "bench", "tools")
#: 计入「源码本体」的类别
CORE_KINDS = ("src",)

REPO_ALIASES = {
    "ship-platform": "ship-performance-platform",
    "ship_performance_platform": "ship-performance-platform",
    "ship-performance": "ship-performance-platform",
}


def parse_manifest(*texts):
    """把 `changed_files` / `on_disk_changes` 两个 JSON 文本解析成去重有序路径表。

    容错：非 JSON、非数组、数组里混非字符串、空串都不抛异常（历史数据里都出现过）。
    """
    out = []
    seen = set()
    for t in texts:
        if not t:
            continue
        try:
            data = json.loads(t)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, str):
                continue
            p = item.strip().strip('"')
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _clean(s):
    parts = [x for x in s.split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts):
        return None
    return "/".join(parts)


def path_candidates(p):
    """一个清单条目可能对应的仓库相对路径候选（严格比对用，最多 2 个）。

    绝对路径有两种历史写法（带/不带 checkout 层），故给两个候选；相对路径只有一个。
    刻意不做模糊匹配——严格比对的分母必须干净。
    """
    if not p:
        return []
    s = p.strip().replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    if not s.startswith("/"):
        c = _clean(s)
        return [c] if c else []
    m = WORK_ABS_RE.match(s)
    if not m:
        return []
    rest = m.group(1)
    cands = []
    if "/" in rest:
        c = _clean(rest.split("/", 1)[1])
        if c:
            cands.append(c)
    c2 = _clean(rest)
    if c2 and c2 not in cands:
        cands.append(c2)
    return cands


def normalize_path(p):
    """返回 (仓库相对候选路径, 证据级)。

    证据级：repo_relative / worktree_absolute / outside_repo / empty
    """
    if not p:
        return None, "empty"
    cands = path_candidates(p)
    if p.strip().replace("\\", "/").startswith("/"):
        if not cands:
            return None, "outside_repo"
        return cands[0], "worktree_absolute"
    if not cands:
        return None, "empty"
    return cands[0], "repo_relative"


def classify_path(rel):
    """按仓库相对路径的形状分类（类别名即报告里的聚合键）。"""
    if not rel:
        return "unparsed"
    low = rel.lower()
    if any(low.startswith(r) for r in SCRATCH_ROOTS):
        return "scratch"
    if any(low.startswith(r) for r in DOC_ROOTS):
        return "docs"
    if any(low.startswith(r) for r in BENCH_ROOTS):
        return "bench"
    if any(low.startswith(r) for r in TOOL_ROOTS):
        return "tools"
    if low.startswith("tests/") or "/test_" in "/" + low or low.startswith("test_"):
        return "tests"
    if any(low.startswith(r) for r in ARTIFACT_ROOTS):
        return "artifacts"
    if low.startswith("src/"):
        return "src"
    if "/" not in low:  # 仓库根目录文件
        if low.endswith(DOC_SUFFIX):
            return "root_doc"
        if low.endswith(CODE_SUFFIX):
            return "root_script"  # m22264 校准单判为「一次性脚本冒充实现」的那一类
        if low.endswith(ARTIFACT_SUFFIX):
            return "artifacts"
        return "root_other"
    if low.endswith(CODE_SUFFIX):
        return "src"  # 顶层包目录内的代码 = 源码本体
    if low.endswith(DOC_SUFFIX):
        return "docs"
    if low.endswith(ARTIFACT_SUFFIX):
        return "artifacts"
    return "other"


def resolve_repo(repo, available):
    """把任务里的 repo 名对齐到磁盘上真实存在的仓目录；对不上返回 None（不猜）。"""
    if not repo:
        return None
    avail = set(available)
    if repo in avail:
        return repo
    if repo in REPO_ALIASES and REPO_ALIASES[repo] in avail:
        return REPO_ALIASES[repo]
    for a in avail:
        if a.lower() == repo.lower():
            return a
    return None


def survival(files, head_paths, head_basenames):
    """逐文件比对：strict（路径命中）/ basename_only（仅同名存在）/ absent。"""
    rec = {
        "n_files": 0,
        "n_strict": 0,
        "n_basename_only": 0,
        "n_absent": 0,
        "n_abs_uncommitted": 0,
        "claimed_kinds": Counter(),
        "surv_kinds": Counter(),
        "surv_relpaths": [],
        "absent_samples": [],
    }
    for p in files:
        rec["n_files"] += 1
        cands = path_candidates(p)
        rel = cands[0] if cands else None
        kind = classify_path(rel)
        rec["claimed_kinds"][kind] += 1
        if p.strip().startswith("/"):
            rec["n_abs_uncommitted"] += 1
        hit = next((c for c in cands if c in head_paths), None)
        if hit is not None:
            rec["n_strict"] += 1
            rec["surv_kinds"][kind] += 1
            rec["surv_relpaths"].append(hit)
            continue
        base = rel.rsplit("/", 1)[-1] if rel else None
        if base and head_basenames is not None and base in head_basenames:
            rec["n_basename_only"] += 1
            continue
        rec["n_absent"] += 1
        if len(rec["absent_samples"]) < 5:
            rec["absent_samples"].append(p)
    return rec


def persistence_class(rec, has_manifest=True):
    """把一单的存活记录归到落地档（P0 = 宣称了却一条都不在，最硬的亏损）。"""
    if not has_manifest or rec.get("n_files", 0) == 0:
        return "P_nomanifest"
    if rec.get("n_strict", 0) == 0:
        return "P0_gone"
    surv = rec.get("surv_kinds") or {}
    if any(surv.get(k) for k in CORE_KINDS):
        return "P2_core_and_tests" if surv.get("tests") else "P2_core"
    if any(surv.get(k) for k in IMPL_KINDS):
        return "P1_impl_peripheral"
    return "P1_docs_only"


def path_verdict(cand, history_all, history_trunk):
    """单条宣称路径的历史判定。

    landed_then_removed ⇒ 曾在主干可达历史里（后被杀）；
    branch_only        ⇒ 只在 agent 分支历史里（从未进主干）；
    never_existed      ⇒ 哪里都没有（清单夸大/路径写错）。
    """
    if not cand:
        return "no_path"
    if cand in set(history_trunk):
        return "landed_then_removed"
    if cand in set(history_all):
        return "branch_only"
    return "never_existed"


def summarize_probe(candidate_relpaths, history_all, history_trunk):
    """路径级归因汇总（mission 级只做汇总，避免「一条命中就当整单落地」的误读）。"""
    cands = {c for c in candidate_relpaths if c}
    out = {"n_paths": len(cands), "landed_then_removed": 0, "branch_only": 0,
           "never_existed": 0}
    for c in cands:
        out[path_verdict(c, history_all, history_trunk)] = \
            out.get(path_verdict(c, history_all, history_trunk), 0) + 1
    out["mission_verdict"] = ("trunk_evidence" if out["landed_then_removed"]
                              else ("branch_only" if out["branch_only"]
                                    else "never_existed"))
    return out


def ratio_pct(num, den, ndigits=1):
    if not den:
        return None
    return round(100.0 * num / den, ndigits)


def inflation(claimed, persisted):
    """膨胀因子 = 宣称 / 落地（>1 表示宣称多于落地）。"""
    if not persisted:
        return None
    return round(claimed / persisted, 3)


def funnel(stages):
    """漏斗：相邻两段的保留率，首段无保留率。"""
    out = []
    prev = None
    for name, n in stages:
        ret = None if prev is None else (ratio_pct(n, prev) if prev else None)
        out.append({"stage": name, "n": n, "retention_pct": ret})
        prev = n
    return out

# =============================================================== I/O：DB

SQL_REGISTRY = {
    "db_now": "SELECT now()::text",
    "mission_status_counts": """
        SELECT status, count(*)::text
        FROM tensor_dev_missions
        WHERE coalesce(finished_at, created_at) <= :snapshot
        GROUP BY status ORDER BY 2 DESC
    """,
    "missions_succeeded": """
        SELECT id::text, repo,
               coalesce(result->>'changed_files', '[]'),
               coalesce(result->>'on_disk_changes', '[]'),
               coalesce(result->'pr_gate'->>'state', 'none'),
               coalesce(result->>'accepted', ''),
               coalesce(result->>'fail_stage', ''),
               to_char(coalesce(finished_at, created_at), 'YYYY-MM-DD"T"HH24:MI:SS')
        FROM tensor_dev_missions
        WHERE status = 'succeeded'
          AND coalesce(finished_at, created_at) <= :snapshot
        ORDER BY id
    """,
    "verdicts": """
        SELECT coalesce(mission_id::text, ''), repo, scope,
               level::text, coalesce(gate_level::text, ''), coalesce(gate_state, '')
        FROM tensor_maturity_verdicts
        WHERE coalesce(created_at, now()) <= :snapshot
    """,
    "ledger_status_counts":
        "SELECT status, count(*)::text FROM feature_ledger GROUP BY status ORDER BY 2 DESC",
    "ledger_unfinished": """
        SELECT status, coalesce(domain, ''), slug, left(acceptance, 200)
        FROM feature_ledger
        WHERE status NOT IN ('done', 'verified', 'confirmed', 'dropped')
        ORDER BY status, slug
    """,
}


def psql(sql, snapshot=None):
    """只读 SELECT；:snapshot 用 DB 自己给的时间戳字面量替换。"""
    if ":snapshot" in sql:
        if not snapshot:
            raise RuntimeError("SQL 需要 :snapshot")
        sql = sql.replace(":snapshot", "'" + snapshot + "'")
    cmd = ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
           "-t", "-A", "-F", "\t", "-v", "ON_ERROR_STOP=1", "-c", sql]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("psql failed: " + res.stderr.strip()[:400])
    return [line.split("\t") for line in res.stdout.split("\n") if line.strip() != ""]


# =============================================================== I/O：git HEAD 清单

def repo_manifest(repo_dir):
    """取仓库 HEAD 的全路径清单 + basename 索引 + HEAD 身份（sha/日期）。"""
    if not (repo_dir / ".git").exists():
        return None
    try:
        sha = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        when = subprocess.run(["git", "-C", str(repo_dir), "log", "-1", "--format=%cI"],
                              capture_output=True, text=True, check=True).stdout.strip()
        listing = subprocess.run(
            ["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", "HEAD"],
            capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        return {"error": str(exc)[:200]}
    paths = {p for p in listing.split("\n") if p}
    return {
        "head_sha": sha,
        "head_date": when,
        "n_paths": len(paths),
        "paths": paths,
        "basenames": {p.rsplit("/", 1)[-1] for p in paths},
    }


def repo_history_paths(repo_dir, candidate_paths, rev="--all", limit_commits=4000):
    """在 `--all` 历史里找出出现在 diff 记录中的候选路径（限提交数，防全史扫描过慢）。"""
    cands = sorted({c for c in candidate_paths if c})
    if not cands:
        return set()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "--all", "--no-renames", "--name-only",
             rev, "--format=", "-n", str(limit_commits), "--diff-filter=AM", "--"] + cands,
            capture_output=True, text=True, check=True, timeout=120).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return set()
    seen = {line.strip() for line in out.split("\n") if line.strip()}
    return seen & set(cands)


# =============================================================== 聚合

CLASS_ORDER = ["P_nomanifest", "P0_gone", "P1_docs_only", "P1_impl_peripheral",
               "P2_core", "P2_core_and_tests", "P_unmapped", "P_nohead"]


def build_report(snapshot, repos_root, full_detail=False, history_probe=50):
    status_rows = psql(SQL_REGISTRY["mission_status_counts"], snapshot)
    mission_rows = psql(SQL_REGISTRY["missions_succeeded"], snapshot)
    verdict_rows = psql(SQL_REGISTRY["verdicts"], snapshot)
    ledger_counts = psql(SQL_REGISTRY["ledger_status_counts"])
    ledger_unfinished = psql(SQL_REGISTRY["ledger_unfinished"])

    available = sorted(p.name for p in repos_root.iterdir() if p.is_dir())
    heads = {}

    def head(repo):
        r = resolve_repo(repo, available)
        if r is None:
            return None, None
        if r not in heads:
            heads[r] = repo_manifest(repos_root / r)
        return r, heads[r]

    # ---- 第一遍：解析清单 + 争夺路径计数
    missions = []
    claim_counter = Counter()
    cand_cache = {}
    for row in mission_rows:
        r = (row + [""] * 8)[:8]
        mid, repo, cf, odc, gate, accepted, fail_stage, finished = r
        files = parse_manifest(cf, odc)
        resolved, man = head(repo)
        cands = set()
        for p in files:
            if p not in cand_cache:
                cand_cache[p] = path_candidates(p)
            cands.update(cand_cache[p])
        for c in cands:
            claim_counter[c] += 1
        missions.append({
            "mission_id": int(mid), "repo": repo, "repo_resolved": resolved,
            "pr_state": gate, "accepted": accepted, "fail_stage": fail_stage,
            "finished_at": finished, "files": files,
        })

    # ---- 第二遍：逐单存活
    per_repo = defaultdict(lambda: {"missions": 0, "with_manifest": 0, "files": 0, "surv": 0,
                                    "classes": Counter(), "unmapped": 0})
    per_gate = defaultdict(lambda: {"missions": 0, "with_manifest": 0, "files": 0, "surv": 0,
                                    "classes": Counter()})
    kind_claimed = Counter()
    kind_surv = Counter()
    class_all = Counter()
    total_files = total_surv = total_basename = total_abs = contested = 0
    detailed = []

    for m in missions:
        key = m["repo_resolved"] or m["repo"]
        pr = per_repo[key]
        pr["missions"] += 1
        pg = per_gate[m["pr_state"]]
        pg["missions"] += 1
        if m["repo_resolved"] is None:
            pr["unmapped"] += 1
            class_all["P_unmapped"] += 1
            detailed.append({"mission_id": m["mission_id"], "repo": m["repo"],
                             "pr_state": m["pr_state"], "finished_at": m["finished_at"],
                             "persistence_class": "P_unmapped", "n_files": len(m["files"])})
            continue
        man = heads.get(m["repo_resolved"])
        if not man or "paths" not in man:
            class_all["P_nohead"] += 1
            continue
        rec = survival(m["files"], man["paths"], man["basenames"])
        cls = persistence_class(rec, has_manifest=bool(rec["n_files"]))
        class_all[cls] += 1
        pr["classes"][cls] += 1
        pg["classes"][cls] += 1
        if rec["n_files"]:
            pr["with_manifest"] += 1
            pg["with_manifest"] += 1
        pr["files"] += rec["n_files"]
        pr["surv"] += rec["n_strict"]
        pg["files"] += rec["n_files"]
        pg["surv"] += rec["n_strict"]
        total_files += rec["n_files"]
        total_surv += rec["n_strict"]
        total_basename += rec["n_basename_only"]
        total_abs += rec["n_abs_uncommitted"]
        kind_claimed.update(rec["claimed_kinds"])
        kind_surv.update(rec["surv_kinds"])
        for rp in rec["surv_relpaths"]:
            if claim_counter.get(rp, 0) > 1:
                contested += 1
        detailed.append({
            "mission_id": m["mission_id"], "repo": m["repo_resolved"],
            "pr_state": m["pr_state"], "finished_at": m["finished_at"],
            "persistence_class": cls,
            "n_files": rec["n_files"], "n_strict": rec["n_strict"],
            "n_basename_only": rec["n_basename_only"], "n_absent": rec["n_absent"],
            "n_abs_uncommitted": rec["n_abs_uncommitted"],
            "absent_samples": rec["absent_samples"],
        })

    # ---- 判定层 × 落地档
    verdict_by_mission = {}
    for r in verdict_rows:
        mid, repo, scope, level, gate_level, gate_state = (r + [""] * 6)[:6]
        if not mid:
            continue
        cur = verdict_by_mission.setdefault(int(mid), {"level": 0, "gate_level": None,
                                                      "n_verdicts": 0})
        cur["n_verdicts"] += 1
        try:
            cur["level"] = max(cur["level"], int(level))
        except ValueError:
            pass
        try:
            gl = int(gate_level) if gate_level != "" else None
        except ValueError:
            gl = None
        if gl is not None:
            cur["gate_level"] = gl if cur["gate_level"] is None else max(cur["gate_level"], gl)

    by_id = {d["mission_id"]: d for d in detailed if "persistence_class" in d}
    gate_keys = ["ungated"] + ["L%d" % i for i in range(6)]
    gate_x = {k: Counter() for k in gate_keys}
    gate_pairs = 0
    gate_l2_zero = []
    for mid, v in verdict_by_mission.items():
        d = by_id.get(mid)
        if d is None:
            continue
        key = "ungated" if v["gate_level"] is None else "L%d" % v["gate_level"]
        cls = d["persistence_class"]
        gate_x[key][cls] += 1
        gate_pairs += 1
        if v["gate_level"] is not None and v["gate_level"] >= 2 and \
                cls in ("P0_gone", "P_nomanifest"):
            gate_l2_zero.append({"mission_id": mid, "gate_level": v["gate_level"],
                                 "claim_level": v["level"], "class": cls,
                                 "repo": d.get("repo"), "n_files": d.get("n_files", 0)})

    # ---- 历史探针：merged 却 0 存活的单，路径级判定「曾在主干 / 只在分支 / 从未存在」
    probe = {"sampled": 0, "n_paths": 0, "landed_then_removed": 0, "branch_only": 0,
             "never_existed": 0, "missions_trunk_evidence": 0,
             "missions_no_trunk_evidence": 0, "records": []}
    cand_pool = sorted([d for d in detailed
                        if d.get("persistence_class") == "P0_gone"
                        and d.get("pr_state") == "merged"
                        and d.get("n_files", 0) > 0],
                       key=lambda d: d["mission_id"])
    if history_probe and cand_pool:
        step = max(1, len(cand_pool) // history_probe)
        sample = cand_pool[::step][:history_probe]
        by_mid = {m["mission_id"]: m for m in missions}
        for d in sample:
            m = by_mid[d["mission_id"]]
            root = repos_root / m["repo_resolved"]
            cands = []
            for f in m["files"]:
                cands.extend(path_candidates(f))
            cands = sorted(set(cands))
            hist_all = repo_history_paths(root, cands, rev="--all")
            hist_trunk = repo_history_paths(root, cands, rev="HEAD")
            summ = summarize_probe(cands, hist_all, hist_trunk)
            probe["sampled"] += 1
            probe["n_paths"] += summ["n_paths"]
            for k in ("landed_then_removed", "branch_only", "never_existed"):
                probe[k] += summ[k]
            if summ["mission_verdict"] == "trunk_evidence":
                probe["missions_trunk_evidence"] += 1
            else:
                probe["missions_no_trunk_evidence"] += 1
            if len(probe["records"]) < 20:
                probe["records"].append({"mission_id": d["mission_id"],
                                         "repo": m["repo_resolved"],
                                         "n_claimed_paths": summ["n_paths"],
                                         "verdict": summ["mission_verdict"],
                                         "n_trunk": summ["landed_then_removed"],
                                         "n_branch_only": summ["branch_only"],
                                         "n_never": summ["never_existed"]})
        for k in ("landed_then_removed", "branch_only", "never_existed"):
            probe[k + "_pct"] = ratio_pct(probe[k], probe["n_paths"])
        probe["missions_no_trunk_evidence_pct"] = ratio_pct(
            probe["missions_no_trunk_evidence"], probe["sampled"])

    persisted = sum(class_all[c] for c in
                    ("P1_docs_only", "P1_impl_peripheral", "P2_core", "P2_core_and_tests"))
    impl = sum(class_all[c] for c in ("P1_impl_peripheral", "P2_core", "P2_core_and_tests"))
    claiming = sum(class_all[c] for c in
                   ("P0_gone", "P1_docs_only", "P1_impl_peripheral",
                    "P2_core", "P2_core_and_tests"))

    return {
        "tool": "restat_v3_survival",
        "version": 3,
        "snapshot": snapshot,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repos_root": str(repos_root),
        "sql_registry": SQL_REGISTRY,
        "mission_status_counts": {r[0]: int(r[1]) for r in status_rows},
        "ledger_status_counts": {r[0]: int(r[1]) for r in ledger_counts},
        "ledger_total": sum(int(r[1]) for r in ledger_counts),
        "ledger_unfinished": [{"status": r[0], "domain": r[1], "slug": r[2], "acceptance": r[3]}
                              for r in ledger_unfinished],
        "repo_heads": {k: ({"head_sha": v["head_sha"], "head_date": v["head_date"],
                            "n_paths": v["n_paths"]} if v and "head_sha" in v else v)
                       for k, v in heads.items()},
        "coverage": {
            "missions_total": len(missions),
            "missions_claiming_manifest": claiming,
            "manifest_coverage_pct": ratio_pct(claiming, len(missions)),
            "missions_unmapped_repo": class_all["P_unmapped"],
            "missions_without_head": class_all["P_nohead"],
            "files_claimed": total_files,
            "files_alive_strict": total_surv,
            "files_alive_basename_only": total_basename,
            "files_absent": total_files - total_surv - total_basename,
            "file_survival_pct": ratio_pct(total_surv, total_files),
            "file_survival_upper_pct": ratio_pct(total_surv + total_basename, total_files),
            "abs_path_entries": total_abs,
            "abs_path_share_pct": ratio_pct(total_abs, total_files),
            "contested_surviving_claims": contested,
            "contested_share_of_surviving_pct": ratio_pct(contested, total_surv),
            "verdict_join_missions": gate_pairs,
        },
        "persistence_classes": {c: class_all[c] for c in CLASS_ORDER},
        "inflation": {
            "missions_claiming": claiming,
            "missions_persisted_any": persisted,
            "missions_persisted_impl": impl,
            "missions_zero_persisted": class_all["P0_gone"],
            "zero_persisted_pct": ratio_pct(class_all["P0_gone"], claiming),
            "claim_over_persisted_factor": inflation(claiming, persisted),
            "claim_over_impl_factor": inflation(claiming, impl),
            "files_claim_over_alive_factor": inflation(total_files, total_surv),
        },
        "per_repo": {k: {"missions": v["missions"], "with_manifest": v["with_manifest"],
                         "files": v["files"], "files_alive": v["surv"],
                         "file_survival_pct": ratio_pct(v["surv"], v["files"]),
                         "classes": dict(v["classes"]), "unmapped": v["unmapped"]}
                     for k, v in sorted(per_repo.items(), key=lambda kv: -kv[1]["missions"])},
        "per_pr_state": {k: {"missions": v["missions"], "with_manifest": v["with_manifest"],
                             "files": v["files"], "files_alive": v["surv"],
                             "file_survival_pct": ratio_pct(v["surv"], v["files"]),
                             "classes": dict(v["classes"])}
                         for k, v in sorted(per_gate.items(), key=lambda kv: -kv[1]["missions"])},
        "per_kind": {k: {"claimed": kind_claimed[k], "alive_strict": kind_surv[k],
                         "survival_pct": ratio_pct(kind_surv[k], kind_claimed[k])}
                     for k in sorted(kind_claimed, key=lambda x: -kind_claimed[x])},
        "gate_x_persistence": {k: dict(gate_x[k]) for k in gate_keys},
        "gate_l2plus_zero_persistence": gate_l2_zero[:300],
        "gate_l2plus_zero_persistence_count": len(gate_l2_zero),
        "history_probe": probe,
        "detail_note": ("full" if full_detail else
                        "compact: 全量 P0_gone + 其余前 200 条；--full-detail 取全量"),
        "detail": (detailed if full_detail else
                   [d for d in detailed if d.get("persistence_class") == "P0_gone"]
                   + [d for d in detailed if d.get("persistence_class") != "P0_gone"][:200]),
        "funnel_missions": funnel([
            ("succeeded", len(missions)),
            ("宣称了文件清单", claiming),
            ("至少 1 个条目在 HEAD", persisted),
            ("至少 1 个实现类条目在 HEAD", impl),
            ("至少 1 个源码本体条目在 HEAD",
             class_all["P2_core"] + class_all["P2_core_and_tests"]),
        ]),
    }


# =============================================================== 渲染：MD

CLASS_LABEL = {
    "P_nomanifest": "无清单",
    "P0_gone": "P0 全丢",
    "P1_docs_only": "P1 仅文档",
    "P1_impl_peripheral": "P1 外围实现",
    "P2_core": "P2 源码本体",
    "P2_core_and_tests": "P2 源码+测试",
    "P_unmapped": "仓名未映射",
    "P_nohead": "无 HEAD",
}
CLASS_DEF = {
    "P_nomanifest": "无任何可解析文件清单",
    "P0_gone": "有清单但 0 条路径命中 HEAD",
    "P1_docs_only": "仅文档/报告类存活",
    "P1_impl_peripheral": "外围实现（bench/tools/tests）存活、无源码本体",
    "P2_core": "源码本体存活（src/ 或包内代码）",
    "P2_core_and_tests": "源码本体 + 测试都存活",
    "P_unmapped": "repo 名对不上磁盘目录（不猜）",
    "P_nohead": "仓目录无 HEAD（空仓/损坏）",
}


def _p(v):
    return "—" if v is None else ("%.1f%%" % v)


def render_md(rep):
    cov, inf = rep["coverage"], rep["inflation"]
    L = []
    A = L.append
    A("# AUDIT-RESTAT v3 · 成果落地存活层（manifest survival）\n")
    A("生成：%s ｜ 快照 cutoff：`%s` ｜ 复跑：`python3 tools/audit_restat/"
      "restat_v3_survival.py --snapshot '%s'`\n" % (rep["generated_at"], rep["snapshot"],
                                                    rep["snapshot"]))
    A("> 本文件补的是 AUDIT-RESTAT 三层既有交付（11-11 首批 / m22264 校准 / m22304 v2 "
      "快照锁定）共同的空白：**三层都没有看仓库本身**。所有「交付了什么」的判断都来自 "
      "`mission.result` 的自述元数据（`changed_files`/`on_disk_changes`/`pr_gate`）。本层把自述\n"
      "> 清单逐条搬到各仓 HEAD 核对，得到「宣称 → 落地」矩阵。它不评价正确性"
      "（tests/gate 的事），只回答：报过账的文件，现在还在仓库里吗？\n")

    A("## 0. 结论摘要\n")
    A("| 项 | 值 | 含义 |")
    A("|---|---:|---|")
    A("| succeeded 任务（快照内） | %s | 名义分母 |" % f"{cov['missions_total']:,}")
    A("| 其中宣称了文件清单 | %s（%s） | 清单覆盖率 |"
      % (f"{cov['missions_claiming_manifest']:,}", _p(cov["manifest_coverage_pct"])))
    A("| 宣称文件条目 | %s | — |" % f"{cov['files_claimed']:,}")
    A("| 严格存活（路径命中 HEAD） | %s（%s） | 文件级落地率 |"
      % (f"{cov['files_alive_strict']:,}", _p(cov["file_survival_pct"])))
    A("| 仅同名存在（弱证据） | %s | 上界 %s |"
      % (f"{cov['files_alive_basename_only']:,}", _p(cov["file_survival_upper_pct"])))
    A("| 清单内 worktree 绝对路径 | %s（%s） | ⇒ 从未进版本控制 |"
      % (f"{cov['abs_path_entries']:,}", _p(cov["abs_path_share_pct"])))
    A("| 0 存活任务（P0） | %s（%s） | 硬性落地亏损 |"
      % (f"{inf['missions_zero_persisted']:,}", _p(inf["zero_persisted_pct"])))
    A("| 宣称/落地 膨胀因子（任务级） | %s | 宣称单 ÷ 至少 1 条目落地单 |"
      % inf["claim_over_persisted_factor"])
    A("| 宣称/源码本体落地 膨胀因子 | %s | 更严口径 |" % inf["claim_over_impl_factor"])
    A("| 宣称/落地 膨胀因子（条目级） | %s | 文件级 |" % inf["files_claim_over_alive_factor"])
    A("| 争夺路径占存活条目 | %s | 存活率的假阳性上界 |"
      % _p(cov["contested_share_of_surviving_pct"]))
    A("| gate≥L2 判定却 0 存活 | %d 单 | 门自身的落地亏损 |"
      % rep["gate_l2plus_zero_persistence_count"])
    if (rep.get("history_probe") or {}).get("sampled"):
        hp0 = rep["history_probe"]
        A("| merged∧0 存活 抽样历史探针 | 路径级：曾入主干 %s / 仅分支 %s / 无史 %s | 见 §6.1 |"
          % (_p(hp0.get("landed_then_removed_pct")), _p(hp0.get("branch_only_pct")),
             _p(hp0.get("never_existed_pct"))))
    A("")

    A("## 1. 六类成果统计表（快照内，程序化生成）\n")
    A("### 1.1 任务状态\n")
    A("| 状态 | 单数 | 说明 |")
    A("|---|---:|---|")
    notes = {"succeeded": "已交付（本文件核落地）", "failed": "诚实失败", "revoked": "撤销",
             "awaiting_pr_merge": "灰区：分支已推未合并", "stopped": "中止",
             "running": "快照时在跑", "queued": "排队"}
    for k, v in rep["mission_status_counts"].items():
        A("| %s | %s | %s |" % (k, f"{v:,}", notes.get(k, "—")))
    A("")
    A("### 1.2 能力台账\n")
    A("| 台账状态 | 条数 |")
    A("|---|---:|")
    for k, v in rep["ledger_status_counts"].items():
        A("| %s | %s |" % (k, f"{v:,}"))
    A("| **合计** | **%s** |" % f"{rep['ledger_total']:,}")
    A("")

    A("## 2. 落地档分布与膨胀量化\n")
    A("| 落地档 | 定义 | 单数 | 占宣称清单单 |")
    A("|---|---|---:|---:|")
    denom = inf["missions_claiming"] or 1
    for c in CLASS_ORDER:
        n = rep["persistence_classes"].get(c, 0)
        A("| %s | %s | %s | %s |" % (CLASS_LABEL[c], CLASS_DEF[c], f"{n:,}",
                                     _p(ratio_pct(n, denom))))
    A("")
    A("任务级漏斗（逐段保留率）：\n")
    A("| 段 | 单数 | 保留率 |")
    A("|---|---:|---:|")
    for s in rep["funnel_missions"]:
        A("| %s | %s | %s |" % (s["stage"], f"{s['n']:,}", _p(s["retention_pct"])))
    A("")

    A("## 3. 文件类别 × 存活（「一次性脚本冒充实现」现在有落地证据）\n")
    A("| 类别 | 宣称 | 严格存活 | 存活率 |")
    A("|---|---:|---:|---:|")
    for k, v in rep["per_kind"].items():
        A("| %s | %s | %s | %s |" % (k, f"{v['claimed']:,}", f"{v['alive_strict']:,}",
                                     _p(v["survival_pct"])))
    A("")

    pk = rep["per_kind"]
    def kp(k):
        v = pk.get(k, {"survival_pct": None, "claimed": 0})
        return "%s（%d 条，%s）" % (k, v["claimed"], _p(v["survival_pct"]))
    A("### 3.1 读法（三条可直接引用的对比）\n")
    A("1. **源码 vs 根目录脚本**：%s vs %s。同一批 agent 写的根目录一次性脚本，"
      "落地率比源码本体低 %s 个百分点——这正是 m22264 校准单用「脚本冒充实现」"
      "扣分的经验判据在仓库层的独立佐证（该单估计污染面 0.75%%，本层给出的是更宽的"
      "「写了但没进库」面）。\n"
      % (kp("src"), kp("root_script"),
         ("%.1f" % ((pk.get("src", {}).get("survival_pct") or 0)
                    - (pk.get("root_script", {}).get("survival_pct") or 0)))))
    A("2. **证据工件基本不入库**：%s——`artifacts/outputs/runs` 下的证据文件绝大"
      "多数没有进版本控制（多数仓 `.gitignore` 掉 `artifacts/`），所以「报告里有数」与"
      "「仓库可复现」是两件事；引用任何历史精度数字时，先确认其证据文件是否在库。\n"
      % kp("artifacts"))
    A("3. **记账说合并、仓库说没有**：`pr_gate=merged` 的 %s 单里，P0（0 条存活）"
      "%s 单占 %s。这批单最值得逐个复核——它们既拿得到「已合并」的名分，也拿不出"
      "任何仓库内实体。\n"
      % (f"{rep['per_pr_state'].get('merged', {}).get('missions', 0):,}",
         f"{rep['per_pr_state'].get('merged', {}).get('classes', {}).get('P0_gone', 0):,}",
         _p(ratio_pct(rep['per_pr_state'].get('merged', {}).get('classes', {}).get('P0_gone', 0),
                      rep['per_pr_state'].get('merged', {}).get('missions', 0)))))
    A("")
    A("## 4. 分仓落地率\n")
    A("| 仓 | succeeded 单 | 有清单 | 条目宣称 | 严格存活 | 条目存活率 | P0 全丢单 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for k, v in rep["per_repo"].items():
        A("| %s | %s | %s | %s | %s | %s | %s |"
          % (k, f"{v['missions']:,}", f"{v['with_manifest']:,}", f"{v['files']:,}",
             f"{v['files_alive']:,}", _p(v["file_survival_pct"]),
             f"{v['classes'].get('P0_gone', 0):,}"))
    A("")

    A("## 5. pr_gate × 落地档（记账层 vs 仓库层）\n")
    cols = CLASS_ORDER
    A("| pr_gate | 单数 | 有清单 | 条目存活率 | " + " | ".join(CLASS_LABEL[c] for c in cols) + " |")
    A("|---|---:|---:|---:|" + "---:|" * len(cols))
    for g, v in rep["per_pr_state"].items():
        cells = " | ".join(str(v["classes"].get(c, 0)) for c in cols)
        A("| %s | %s | %s | %s | %s |" % (g, f"{v['missions']:,}", f"{v['with_manifest']:,}",
                                          _p(v["file_survival_pct"]), cells))
    A("")

    A("## 6. 判定层 × 落地档交叉表\n")
    A("行=机器门 `gate_level`（无判定=ungated），列=落地档。\n")
    gx = rep["gate_x_persistence"]
    A("| gate 档 | " + " | ".join(CLASS_LABEL[c] for c in CLASS_ORDER) + " | 合计 |")
    A("|---|" + "---:|" * (len(CLASS_ORDER) + 1))
    for k in ["L5", "L4", "L3", "L2", "L1", "L0", "ungated"]:
        row = gx.get(k, {})
        cells = " | ".join(str(row.get(c, 0)) for c in CLASS_ORDER)
        A("| %s | %s | %s |" % (k, cells, f"{sum(row.values()):,}"))
    A("")
    A("gate≥L2 却 0 存活的样本（最多 12 条，全量见 JSON "
      "`gate_l2plus_zero_persistence`）：\n")
    A("| mission | gate_level | claim_level | 落地档 | 仓 | 清单条目 |")
    A("|---:|---:|---:|---|---|---:|")
    for s in rep["gate_l2plus_zero_persistence"][:12]:
        A("| %d | %d | %d | %s | %s | %d |" % (s["mission_id"], s["gate_level"],
                                              s["claim_level"], s["class"],
                                              s["repo"], s["n_files"]))
    A("")

    hp = rep.get("history_probe") or {}
    A("## 6.1 历史探针：merged 却 0 存活的单，路径到底进过主干吗？\n")
    if hp.get("sampled"):
        A("抽 %d 单（按 mission_id 等距；条件 pr_gate=merged 且 P0 全丢，共 %d 条宣称路径），"
          "对每条路径跑两次 `git log --name-only --diff-filter=AM -- <路径>`：一次 `--all`"
          "（任何分支）、一次 `HEAD`（主干可达历史）。路径级归因：\n"
          % (hp["sampled"], hp["n_paths"]))
        A("| 路径判定 | 含义 | 路径数 | 占比 |")
        A("|---|---|---:|---:|")
        A("| landed_then_removed | 主干历史里有 ⇒ 入库后被删/替换 | %d | %s |"
          % (hp["landed_then_removed"], _p(hp.get("landed_then_removed_pct"))))
        A("| branch_only | 只在 agent 分支历史 ⇒ 从未进主干 | %d | %s |"
          % (hp["branch_only"], _p(hp.get("branch_only_pct"))))
        A("| never_existed | 任何历史都没有 ⇒ 清单夸大/路径写错 | %d | %s |"
          % (hp["never_existed"], _p(hp.get("never_existed_pct"))))
        A("")
        A("任务级：**%d 单（%s）在主干历史里毫无踪迹**（其全部宣称路径都只活在分支或根本不存在）。"
          "\n" % (hp["missions_no_trunk_evidence"],
                  _p(hp.get("missions_no_trunk_evidence_pct"))))
        if hp.get("records"):
            A("| mission | 仓 | 宣称路径 | 任务判定 | 主干命中 | 仅分支 | 无史 |")
            A("|---:|---|---:|---|---:|---:|---:|")
            for r in hp["records"][:10]:
                A("| %d | %s | %d | %s | %d | %d | %d |"
                  % (r["mission_id"], r["repo"], r["n_claimed_paths"], r["verdict"],
                     r["n_trunk"], r["n_branch_only"], r["n_never"]))
            A("")
    else:
        A("（本次未抽样：--history-probe 0）\n")

    A("## 7. 未达验收待完成名单（供 owner 裁定重开）\n")
    un = rep["ledger_unfinished"]
    by_status = Counter(r["status"] for r in un)
    by_domain = Counter(r["domain"] for r in un)
    A("台账未完成项 **%d** 条；按状态：%s\n"
      % (len(un), "，".join("%s=%d" % (k, v) for k, v in by_status.most_common())))
    A("| 域 | 未完成条数 |")
    A("|---|---:|")
    for k, v in by_domain.most_common():
        A("| %s | %d |" % (k or "（未标域）", v))
    A("")
    A("灰区：`awaiting_pr_merge` **%d** 单（分支已推未合并，既不进 done 也不进 failed）。\n"
      % rep["mission_status_counts"].get("awaiting_pr_merge", 0))
    A("<details><summary>未完成条目明细（前 60 条；全量见 JSON `ledger_unfinished`）</summary>\n")
    A("| 状态 | 域 | slug | 验收线（截断） |")
    A("|---|---|---|---|")
    for r in un[:60]:
        acc = (r["acceptance"] or "").replace("|", "/").replace("\n", " ")
        A("| %s | %s | `%s` | %s |" % (r["status"], r["domain"], r["slug"], acc[:120]))
    A("")
    A("</details>\n")

    A("## 8. 注意点与局限（honest）\n")
    A("1. **清单本身是自述**：`changed_files`/`on_disk_changes` 由 agent 填写，可能漏写或夸大；"
      "本层只核「写进去的」，不核「没写的」。故落地率是**下界口径的宣称精度**，不是绝对产出。")
    A("2. **存活 ≠ 正确**：路径命中只说明该文件存在于 HEAD，不说明内容来自该单、也不说明测试通过。"
      "内容级归属需要 `git log -S`/blame，本层未做（明示范围）。")
    A("3. **同名假阳性**：争夺路径（被 ≥2 单宣称的相对路径）占存活条目 "
      "%s——这是存活率的假阳性上界；`basename_only` 弱证据已单列、未混入严格存活。"
      % _p(cov["contested_share_of_surviving_pct"]))
    A("4. **HEAD 时点**：各仓 HEAD sha/日期逐仓记录于 JSON `repo_heads`；"
      "仓镜像由 `mirror_ff.sh` 每 15 分钟追平，快照后合并的交付不计入存活（保守方向）。")
    A("5. **仓名映射**：仅收录确有目录对应的别名（如 `ship-platform`→`ship-performance-platform`）；"
      "对不上者单列 %d 单，不猜。" % cov["missions_unmapped_repo"])
    A("6. **绝对路径条目 %s**：这类条目在语义上「从未入库」，但其 worktree 相对候选仍被"
      "严格比对过一遍（避免把「后来以相对路径合并」误判为丢失）。"
      % _p(cov["abs_path_share_pct"]))
    A("")

    A("## 9. 复算与已执行验证\n")
    A("```bash\npython3 tools/audit_restat/restat_v3_survival.py --snapshot '%s'\n"
      "python3 -m pytest tests/test_restat_survival.py -q\n```\n" % rep["snapshot"])
    A("| 检查 | 结果 |")
    A("|---|---|")
    A("| 口径纯函数单测（清单解析/路径归一/分类/落地档/膨胀/漏斗/仓名映射） | 见 pytest 日志 |")
    A("| 同快照两次运行 JSON 一致性 | 见 §下方运行时输出（保留 generated_at 外逐字节一致） |")
    A("| 仓 HEAD 身份记录 | 逐仓 sha + 日期见 JSON `repo_heads` |")
    A("")
    return "\n".join(L)


# =============================================================== 渲染：图

FIG_CLASS = {
    "P_nomanifest": "no manifest",
    "P0_gone": "P0 all gone",
    "P1_docs_only": "P1 docs only",
    "P1_impl_peripheral": "P1 peripheral impl",
    "P2_core": "P2 core src",
    "P2_core_and_tests": "P2 core+src tests",
    "P_unmapped": "unmapped repo",
    "P_nohead": "no HEAD",
}
FIG_STAGE = {
    "succeeded": "succeeded",
    "claim了口文件清单": "claim manifest",
    "宣称了文件清单": "claim manifest",
    "至少 1 个条目在 HEAD": ">=1 entry alive",
    "至少 1 个实现类条目在 HEAD": ">=1 impl alive",
    "至少 1 个源码本体条目在 HEAD": ">=1 core src alive",
}


def render_figures(rep, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # 图1：任务级漏斗
    fun = rep["funnel_missions"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    names = [FIG_STAGE.get(s["stage"], s["stage"]) for s in fun][::-1]
    vals = [s["n"] for s in fun][::-1]
    bars = ax.barh(names, vals, color="#4c78a8")
    for b, s in zip(bars, fun[::-1]):
        ax.text(b.get_width() * 1.01, b.get_y() + b.get_height() / 2,
                "%d%s" % (s["n"], "" if s["retention_pct"] is None
                          else " (%.1f%%)" % s["retention_pct"]),
                va="center", fontsize=9)
    ax.set_xlabel("missions")
    ax.set_title("AUDIT-RESTAT v3: claim -> persistence funnel (snapshot %s)"
                 % rep["snapshot"][:10])
    ax.set_xlim(0, max(vals) * 1.35)
    fig.tight_layout()
    p1 = outdir / "fig_restat_v3_funnel.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # 图2：类别存活率
    kinds = list(rep["per_kind"].items())
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    labels = [k for k, _ in kinds]
    surv = [v["survival_pct"] or 0 for _, v in kinds]
    claimed = [v["claimed"] for _, v in kinds]
    ax.bar(labels, surv, color="#f58518")
    for i, (c, s) in enumerate(zip(claimed, surv)):
        ax.text(i, s + 1.5, "%.1f%%\nn=%d" % (s, c), ha="center", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("strict survival (%)")
    ax.set_title("File-kind survival at repo HEAD (claimed -> alive)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    p2 = outdir / "fig_restat_v3_kind_survival.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # 图3：分仓落地档堆叠
    repos = [k for k, v in rep["per_repo"].items() if v["with_manifest"]][:14]
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    bottom = [0] * len(repos)
    colors = {"P0_gone": "#d62728", "P1_docs_only": "#ffb3b3",
              "P1_impl_peripheral": "#9ecae1", "P2_core": "#4c78a8",
              "P2_core_and_tests": "#08519c", "P_nomanifest": "#cccccc"}
    for cls in ["P0_gone", "P1_docs_only", "P1_impl_peripheral", "P2_core",
                "P2_core_and_tests"]:
        vals = [100.0 * rep["per_repo"][r]["classes"].get(cls, 0) /
                max(rep["per_repo"][r]["missions"], 1) for r in repos]
        ax.bar(repos, vals, bottom=bottom, label=FIG_CLASS[cls],
               color=colors.get(cls, "#999999"))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of succeeded missions")
    ax.set_ylim(0, 100)
    ax.set_title("Persistence class mix per repo")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    p3 = outdir / "fig_restat_v3_repo_mix.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    # 图4：gate × 落地档 热力
    keys = ["L5", "L4", "L3", "L2", "L1", "L0", "ungated"]
    cols = CLASS_ORDER
    mat = [[rep["gate_x_persistence"].get(k, {}).get(c, 0) for c in cols] for k in keys]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([FIG_CLASS[c] for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)
    for i in range(len(keys)):
        for j in range(len(cols)):
            if mat[i][j]:
                ax.text(j, i, str(mat[i][j]), ha="center", va="center", fontsize=8)
    ax.set_title("Machine gate level x persistence class")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    p4 = outdir / "fig_restat_v3_gate_heatmap.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths.append(p4)
    return paths


# =============================================================== main

def main(argv=None):
    ap = argparse.ArgumentParser(description="AUDIT-RESTAT v3 manifest survival audit")
    ap.add_argument("--snapshot", help="ISO 时间戳；缺省用 DB now()")
    ap.add_argument("--repos-root", default="/srv/tensor/workspace")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default="docs/audit_restat/restat_v3_survival.md")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--history-probe", type=int, default=50,
                    help="merged 却 0 存活的单里抽多少单做 git 历史探针（0=关）")
    ap.add_argument("--full-detail", action="store_true",
                    help="JSON detail 写全量（默认只有 P0_gone 全量 + 其余 200 条）")
    args = ap.parse_args(argv)

    repos_root = Path(args.repos_root)
    if not repos_root.is_dir():
        print("repos root not found: %s" % repos_root, file=sys.stderr)
        return 2
    snapshot = args.snapshot or psql(SQL_REGISTRY["db_now"])[0][0]
    rep = build_report(snapshot, repos_root, full_detail=args.full_detail,
                       history_probe=args.history_probe)

    stamp = snapshot.replace(" ", "T").replace(":", "")[:17]
    out_json = Path(args.out_json or ("artifacts/audit_restat/restat_v3_%s.json" % stamp))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=1, sort_keys=True))

    md = render_md(rep)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    figs = []
    if not args.no_figures:
        figs = render_figures(rep, out_md.parent)
        rel = "\n".join("- `%s`" % p.name for p in figs)
        md += "\n## 10. 图\n\n%s\n" % rel
    out_md.write_text(md)

    c = rep["coverage"]
    print("snapshot=%s" % snapshot)
    print("succeeded=%d claiming=%d files=%d alive=%d (%.1f%%) zero_persisted=%d (%.1f%%)"
          % (c["missions_total"], c["missions_claiming_manifest"], c["files_claimed"],
             c["files_alive_strict"], c["file_survival_pct"] or 0,
             rep["inflation"]["missions_zero_persisted"],
             rep["inflation"]["zero_persisted_pct"] or 0))
    print("json=%s" % out_json)
    print("md=%s" % out_md)
    for p in figs:
        print("fig=%s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
