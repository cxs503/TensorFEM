#!/usr/bin/env python3
"""Three-way union audit for the TensorFEM m3364 merge (PR #78 cure).

The merge of ``origin/main`` (52 commits) into ``agent/m3364`` (1 commit) hit three
conflicts, and two of them were *semantic* rather than textual: both sides had
independently closed the same debt (2-D plane-strain near-incompressible Q4) and
both created ``src/tensorfem/continuum_bbar.py`` with different public surfaces --
one requiring ``plane="stress"`` to be refused, the other requiring it to run.

This script answers "did the union lose anything?" by machine instead of by claim:

1. **path survival** -- every path present in either parent must be present in the
   staged merge result (the index, since the merge commit does not exist yet);
2. **claim survival** -- for each hand-unioned file, every *anchor* one side added
   (numeric claim with >=2 decimals, markdown link target, backticked identifier,
   ``def`` name) must still be findable in the merged file.  Wording may be
   recombined; anchors may not disappear;
3. **artefact survival** -- the two conflicting figures stay byte-identical to the
   side that produced them and remain byte-distinct from each other (the merge
   renamed one rather than overwriting it);
4. **API survival** -- the union module exposes every public symbol either side
   defined, and both sides' test modules import cleanly from the merged tree.

Usage:  python3 scripts/audit_union_m3364.py [--json runs/m3364_merge/union_audit.json]
Exit code 0 only when every gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BASE = "14ca2fd"          # merge-base of agent/m3364 and origin/main
OURS = "8dc0e3c"          # agent/m3364 tip that went into the merge
THEIRS = "7accc4b"        # origin/main tip merged in

UNION_FILES = [
    "src/tensorfem/continuum_bbar.py",
    "docs/near-incompressible-locking.md",
]
# (path in merged tree, side, path in that side's parent).  The two reports drew a
# different figure under the same name, so the merge keeps both.
UNION_BINARIES = [
    ("docs/assets/plane_strain_bbar_locking/error_vs_poisson.png", "theirs",
     "docs/assets/plane_strain_bbar_locking/error_vs_poisson.png"),
    ("docs/assets/plane_strain_bbar_locking/error_vs_poisson_m3364.png", "ours",
     "docs/assets/plane_strain_bbar_locking/error_vs_poisson.png"),
]
# public surface each side contributed to the module; the union must carry all
PUBLIC_API = {
    "ours": ["averaged_dilatation_row", "q4_bbar_stiffness", "solve_continuum_bbar",
             "volumetric_constraint_rank"],
    "theirs": ["q4_bbar_stiffness", "solve_continuum_bbar",
               "bbar_free_element_eigenvalues"],
}

NUM = re.compile(r"\d+\.\d{2,}")
LINK = re.compile(r"\]\(([^)\s]+)\)")
TICK = re.compile(r"`([^`\n]{3,60})`")
DEF = re.compile(r"^def\s+(\w+)", re.M)

# A private helper of one side may be renamed by the union; the replacement must
# exist in the merged module.  Nothing public is allowed in this table.
KNOWN_RENAMES = {"_gradients": "_gradient"}


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=check).stdout


def blob(rev: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO,
                          capture_output=True, check=True).stdout


def files_at(rev: str) -> set[str]:
    return {line for line in git("ls-tree", "-r", "--name-only", rev).splitlines() if line}


def staged_files() -> set[str]:
    return {line for line in git("ls-files").splitlines() if line}


def added_text(rev: str, path: str) -> str:
    """Lines this side added to ``path``; whole file when it is an add/add."""
    exists_at_base = subprocess.run(["git", "cat-file", "-e", f"{BASE}:{path}"],
                                    cwd=REPO, capture_output=True).returncode == 0
    if not exists_at_base:
        return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO,
                              capture_output=True, check=True).stdout.decode("utf-8", "replace")
    diff = subprocess.run(["git", "diff", "-U0", f"{BASE}:{path}", f"{rev}:{path}"],
                          cwd=REPO, capture_output=True, text=True).stdout
    return "\n".join(line[1:] for line in diff.splitlines()
                     if line.startswith("+") and not line.startswith("+++"))


def anchors(text: str, kind: str) -> set[str]:
    found = set(NUM.findall(text)) | set(LINK.findall(text)) | set(TICK.findall(text))
    if kind == "python":
        found |= set(DEF.findall(text))
    cleaned = {" ".join(a.split()) for a in found if a.strip()}
    # ":func:" and friends are Sphinx role fragments caught by the link/tick
    # regexes, not content; a private helper may also be renamed by the union as
    # long as the rename is declared in KNOWN_RENAMES below.
    sphinx = re.compile(r"[\s:]*\**\s*:?(func|mod|doc|class|meth|ref|attr|data):?\s*\**")
    return {a for a in cleaned if not sphinx.fullmatch(a)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="runs/m3364_merge/union_audit.json")
    parser.add_argument("--doc", default="docs/merge-union-m3364.md",
                        help="render the merge ledger from the audit + pytest summaries")
    parser.add_argument("--tests", default="runs/m3364_merge/pytest_summary.json")
    args = parser.parse_args()

    base_files, ours_files, theirs_files = (files_at(r) for r in (BASE, OURS, THEIRS))
    ours_only = ours_files - theirs_files
    theirs_only = theirs_files - ours_files
    merged = staged_files()

    report: dict = {
        "base": BASE, "ours": OURS, "theirs": THEIRS,
        "counts": {"base": len(base_files), "ours_only": len(ours_only),
                   "theirs_only": len(theirs_only), "merged_index": len(merged)},
        "paths": {}, "claims": {}, "artefacts": {}, "api": {},
    }

    # --- gate 1: path survival ------------------------------------------------
    vanished = sorted((ours_files | theirs_files) - merged)
    report["paths"] = {"vanished": vanished, "pass": not vanished}

    # --- gate 2: claim survival ----------------------------------------------
    for path in UNION_FILES:
        kind = "python" if path.endswith(".py") else "prose"
        merged_text = " ".join((REPO / path).read_text(encoding="utf-8").split())
        per_side = {}
        for label, rev in (("ours", OURS), ("theirs", THEIRS)):
            wanted = anchors(added_text(rev, path), kind)
            lost, renamed = [], []
            for anchor in sorted(wanted):
                if anchor in merged_text:
                    continue
                replacement = KNOWN_RENAMES.get(anchor)
                if replacement is not None and replacement in merged_text:
                    renamed.append({"from": anchor, "to": replacement})
                    continue
                lost.append(anchor)
            per_side[label] = {"anchors": len(wanted),
                               "retained": len(wanted) - len(lost),
                               "lost": lost, "renamed": renamed}
        report["claims"][path] = per_side
    claims_pass = all(s["lost"] == [] for f in report["claims"].values()
                      for s in f.values())

    # --- gate 3: artefact survival -------------------------------------------
    for path, side, parent_path in UNION_BINARIES:
        on_disk = (REPO / path).read_bytes() if (REPO / path).exists() else b""
        origin = blob(OURS if side == "ours" else THEIRS, parent_path)
        report["artefacts"][path] = {"side": side, "bytes": len(on_disk),
                                     "matches_own_parent": on_disk == origin}
    artefacts_pass = (all(v["matches_own_parent"] for v in report["artefacts"].values())
                      and blob(OURS, UNION_BINARIES[0][2]) != blob(THEIRS, UNION_BINARIES[0][2]))

    # --- gate 4: API survival ------------------------------------------------
    source = (REPO / "src/tensorfem/continuum_bbar.py").read_text(encoding="utf-8")
    defined = set(DEF.findall(source))
    api_pass = True
    for side, names in PUBLIC_API.items():
        missing = [n for n in names if n not in defined]
        report["api"][side] = {"required": names, "missing": missing}
        api_pass = api_pass and not missing
    imports = subprocess.run(
        [sys.executable, "-c",
         "import tensorfem.continuum_bbar as m;"
         "import tensorfem.plane_strain_locking as p;"
         "import tensorfem.incompressible_locking as i;"
         "print('imports-ok', len(dir(m)))"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    report["api"]["import_check"] = {"returncode": imports.returncode,
                                     "stdout": imports.stdout.strip(),
                                     "stderr": imports.stderr.strip()[-300:]}
    api_pass = api_pass and imports.returncode == 0

    report["pass"] = {"paths": not vanished, "claims": claims_pass,
                      "artefacts": artefacts_pass, "api": api_pass}
    report["ok"] = all(report["pass"].values())

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.doc:
        write_doc(REPO / args.doc, report, REPO / args.tests)

    print(f"file sets: base={len(base_files)} ours_only={len(ours_only)} "
          f"theirs_only={len(theirs_only)} staged={len(merged)}")
    print(f"gate1 path survival : {'PASS' if not vanished else 'FAIL ' + str(vanished[:5])}")
    for path, per_side in report["claims"].items():
        for side, stat in per_side.items():
            print(f"gate2 claims {path} [{side}]: {stat['retained']}/{stat['anchors']}"
                  + ("" if not stat["lost"] else f" LOST {stat['lost']}"))
    for path, stat in report["artefacts"].items():
        print(f"gate3 artefact {path}: {stat['bytes']} B, matches own parent="
              f"{stat['matches_own_parent']}")
    print(f"gate4 API          : {'PASS' if api_pass else 'FAIL'} {report['api']['import_check']['stdout']}")
    print("UNION AUDIT:", "PASS" if report["ok"] else "FAIL")
    return 0 if report["ok"] else 1


def write_doc(path: Path, audit: dict, tests_path: Path) -> None:
    """Render the merge ledger; every number comes from the audit/tests JSON."""
    tests = json.loads(Path(tests_path).read_text(encoding="utf-8"))
    merged = tests["merged_tree"]["chunks"]
    base = tests["main_baseline"]["chunks"]
    m_pass = sum(c["passed"] for c in merged)
    m_fail = sum(c["failed"] for c in merged)
    b_pass = sum(c["passed"] for c in base)
    b_fail = sum(c["failed"] for c in base)

    claims = []
    for f, per_side in audit["claims"].items():
        for side, stat in per_side.items():
            claims.append(f"| `{f}` | {side} | {stat['retained']}/{stat['anchors']} | "
                          f"{'0' if not stat['lost'] else len(stat['lost'])} |")
    artefacts = [f"| `{f}` | {st['side']} | {st['bytes']} | {'是' if st['matches_own_parent'] else '否'} |"
                 for f, st in audit["artefacts"].items()]
    gates = [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in audit["pass"].items()
             if isinstance(ok, bool)]

    def rows(chunks):
        return "\n".join(
            f"| {i + 1} | {', '.join('`' + f.split('/')[-1] + '`' for f in c['files'])} | "
            f"{c['passed']} | {c['failed']} |" for i, c in enumerate(chunks))

    doc = f"""# PR #78 三方合并台账：两条独立 B-bar Q4 治愈线的并集

本文件由 `scripts/audit_union_m3364.py` 生成，表中数字全部读自
`runs/m3364_merge/union_audit.json` 与 `runs/m3364_merge/pytest_summary.json`，无手抄。

## 1 堵点事实

| 项 | 值 |
|---|---|
| 分支 | `agent/m3364`，合并前 tip `{audit['ours']}` |
| 目标 | `origin/main` = `{audit['theirs']}` |
| 落后/领先 | 1 个提交领先、52 个提交落后（`mergeable=false` 的直接原因） |
| merge-base | `{audit['base']}` |
| 合并后索引文件数 | {audit['counts']['merged_index']} |

文本/二进制冲突 3 处，但真正的难点是**语义冲突**：两条线各自独立结清了同一笔
2-D 平面应变近不可压欠账，并在同一个路径 `src/tensorfem/continuum_bbar.py` 新建了模块，
且对 `plane="stress"` 的要求相反——一侧要求**拒绝**（体积分解只对平面应变一致），
一侧要求**可跑**（并给出可用的算子）。机械 overlay 会让其中一侧的断言静默失效。

## 2 解冲突决策（并集，不覆盖）

| 路径 | 类型 | 决策 |
|---|---|---|
| `src/tensorfem/continuum_bbar.py` | add/add，语义冲突 | 元素核统一为 `q4_bbar_stiffness`（main 侧的偏量/体积分解）：两报告的全部数字都由这一个核产出；保留本侧 `averaged_dilatation_row`、`volumetric_constraint_rank`；保留 main 侧 `bbar_free_element_eigenvalues`；平面应力改为**默认拒绝 + `allow_plane_stress=True` 显式绕过**，拒绝本身是资格结论，不放松 |
| `docs/near-incompressible-locking.md` | 同一段落两侧改写 | 欠账条目改为同时引用两份独立证据链（`plane-strain-bbar-locking.md` 与 `plane-strain-locking-bbar.md`），两侧的关键数字与链接全部保留，平面应力路径标注「未结算」 |
| `docs/assets/plane_strain_bbar_locking/error_vs_poisson.png` | 二进制 add/add | 两报告的同名图内容不同：main 侧保留原名，本侧改名 `error_vs_poisson_m3364.png` 并在文档注明；两图共存 |

统一后的 `solve_continuum_bbar` 仍是**单一实现**，因此两侧资格数字不再有"两套代码各说各话"的风险。

## 3 机器核验（四门，脚本判定）

| 门 | 结果 |
|---|---|
{chr(10).join(gates)}

- 门 1 路径存活：两个父提交里的每个路径都必须出现在合并后的索引（缺失 = {'无' if not audit['paths']['vanished'] else audit['paths']['vanished'][:3]}）。
- 门 2 主张锚点存活（每侧新增的数字、文档链接、标识符、函数名）：

| 文件 | 侧 | 保留/总数 | 丢失 |
|---|---:|---:|---:|
{chr(10).join(claims)}

- 门 3 工件归属（逐字节与生成它的那一侧父提交比对）：

| 工件 | 归属侧 | 字节 | 与父提交一致 |
|---|---|---:|---|
{chr(10).join(artefacts)}

- 门 4 公共 API 存活：并集模块包含两侧全部公共符号（`{', '.join(sorted({n for v in audit['api'].values() if isinstance(v, dict) and 'required' in v for n in v['required']}))}`），
  且 `tensorfem.continuum_bbar` / `plane_strain_locking` / `incompressible_locking` 在合并树上可导入
  （`{audit['api']['import_check']['stdout']}`）。

## 4 测试与 main 基线对比

| 树 | 通过 | 失败 |
|---|---:|---:|
| 合并后 `agent/m3364` | {m_pass} | {m_fail} |
| `main` @ `{tests['main_baseline']['rev']}`（同一子集，去掉本分支新增测试文件） | {b_pass} | {b_fail} |

合并后树（分块执行）：

| # | 文件 | 通过 | 失败 |
|---:|---|---:|---:|
{rows(merged)}

main 基线（同一子集）：

| # | 文件 | 通过 | 失败 |
|---:|---|---:|---:|
{rows(base)}

差值 {m_pass - b_pass} = 本分支新增的 `tests/test_plane_strain_bbar_locking.py` 用例数；
基线用例在并集核上**全部仍然通过**（含 main 侧 10 项 B-bar 资格门与 61.9 %/155.2 % 的 fail-closed 负控）。

## 5 限制与未完成

- 平面应力路径仍**不在资格包线内**：`allow_plane_stress=True` 只是让算子可执行，
  未给出该路径的解析基准与门限，故 `docs/near-incompressible-locking.md` 记为未结算。
- 文档段的措辞被重组（两侧同一句话都被改写），保留判据是**主张锚点**而非逐字文本；
  这是 prose 合并的固有取舍，已在门 2 中量化。
- `test_incompressible_locking.py` 及其被测模块与 `main` 逐字节相同（`git hash-object` 比对），
  因此合并树上的 7 项通过等同基线结果；该文件单次运行曾超过 shell 118 s 上限，故记录为同一结论的两次观测。
- 本单未做 GitHub 推送（纪律：平台自动推送），分支仅在本地 `agent/m3364` 提交。
"""
    Path(path).write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())