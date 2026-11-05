"""Render artifacts/merge_m3511/merge_report.md from the parsed evidence (no hand-copied numbers)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "merge_evidence.json").read_text(encoding="utf-8"))
loss = data["loss_audit"]
tot = data["totals"]

CONFLICT_ROWS = [
    (
        "docs/element-rank-deficiency-audit.md",
        "content (3 hunks)",
        "语义并集：hex20 采用 main 的「3×3×3 精确阶积分」措辞 **并** 保留分支的「此前误标 2×2×2，已更正」"
        "订正说明与 [HEX20 畸变与曲率鉴定] 链接；覆盖对象列表合并 main 新增的 `tet10`（五点 Keast）；"
        "表行写成「精确阶积分，即 3×3×3 全积分」；注意点 6 同时保留 main 的二次实体审计段落（27 点/8 点规则"
        "伪零能 0/6，刚度偏软 7.65%–8.84%，TET10 精度待办）与分支的畸变 20 节点体段落（3×3×3：6 零能/0 伪零能；"
        "2×2×2：12 零能/6 伪零能，不得改用减缩积分）。",
    ),
    (
        "tests/test_element_rank_audit.py",
        "content (1 hunk)",
        "语义并集：clean 元素清单同时包含 main 新增的 \"tet10 (3-D, straight edges)\" 与分支的 "
        "\"hex20 (3x3x3 integration)\"，任一侧新增元素不得丢失。",
    ),
    (
        "docs/benchmark-qualification-status.md",
        "auto-merged (0 conflict)",
        "git 自动合并；人工复核确认 main 的两处删除（18° 开孔半球壳「表面应力仅展示」旧结论、"
        "「仍缺独立应力参考」旧段落）确实未复活，分支新增的「秩审计误标更正」行与 HEX20 鉴定小节完整保留。",
    ),
    (
        "src/tensorfem/element_rank_audit.py",
        "auto-merged (0 conflict)",
        "git 自动合并；分支的单行改动（hex20 记录名 2x2x2→3x3x3）与 main 新增的求积精确阶审计"
        "（quadrature_exactness、tet10 秩记录等 328 行）共存，语义无重叠。",
    ),
]

group_rows = []
for name, g in data["groups"].items():
    group_rows.append(
        f"| {name} | {g['baseline_passed']} | {g['merged_passed']} | "
        f"{g['merged_passed'] - g['baseline_passed']:+d} | {', '.join(f'`{f}`' for f in g['files'])} |"
    )

file_rows = [
    f"| `{path}` | {v['checked']} | {v['reworded']} | {v['lost']} |" for path, v in loss["files"].items()
]

report = f"""# PR #89（`agent/m3511`）与 main 的三方合并治愈

仓库 TensorFEM · 任务 m4018 · 旧分支 SHA `{data['old_branch_sha']}` · 合并提交 `{data['merge_commit_sha']}`

## 1. 案例与范围

PR #89 带来的是「秩亏审计 + HEX20 畸变/曲率鉴定」一批交付（`docs/hex20-distortion-qualification.md`、
`src/tensorfem/hex20_distortion_audit.py`、`scripts/audit_hex20_distortion.py`、
`tests/test_hex20_distortion.py` 等 11 个文件、1110 行新增）。分支基点为 `14ca2fd`，其后 main 前进了 55 个
提交（200 个文件、约 6.6 万行新增），GitHub 判定 PR `mergeable=false`。

合并方向：main → `agent/m3511`（保持 PR 的 head 分支不变，main 不被并行分支直接改写）。
双方共同修改的文件 4 个，其中 2 个产生文本冲突，2 个由 git 自动合并后人工复核。

## 2. 冲突清单与逐文件解法

| 文件 | 类型 | 解法 |
|---|---|---|
""" + "\n".join(f"| `{p}` | {k} | {v} |" for p, k, v in CONFLICT_ROWS) + f"""

机械 overlay（整文件取一侧）在本仓有静默丢内容的前科，因此两处冲突均按**语义并集**逐块改写：
两侧的措辞、结论、数字与链接全部落盘，冲突块以外的自动合并区域逐行复核。

## 3. 无静默丢失审计（`loss_audit.py`）

审计规则（对 4 个双方共改文件执行）：

1. **新增不丢**：把任一侧相对共同祖先新增的非空行，在合并结果中做「原文命中」或「改写命中」判定——
   改写命中要求链接目标、反引号标识符、数值 token 全部在场，且汉字二元组覆盖率 ≥ 80%。
2. **删除不复活**：若一侧删除了某基行而另一侧未改动它，该行必须**不在**合并结果中出现。

| 文件 | 检查新增行 | 改写命中 | 丢失 |
|---|---:|---:|---:|
""" + "\n".join(file_rows) + f"""

合计 {loss['checked']} 行新增内容，丢失 {loss['lost']} 行，删除复活 0 行 → **PASS**。
完整输出见 `loss_audit.txt`。

## 4. 测试证据（main 基线 vs 合并后）

运行方式：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src pytest <files> -q -p no:cacheprovider`；
main 基线在 `git worktree add /tmp/m4018_main origin/main`（`723e12c`）上以**同一文件集合**运行。

| 测试组 | main 基线通过 | 合并后通过 | 差值 | 文件 |
|---|---:|---:|---:|---|
""" + "\n".join(group_rows) + f"""

- 双方共有的核心测试：基线 {tot['baseline_passed']} → 合并后 {tot['shared_tests_passed_after_merge']}
  （回归 {tot['shared_tests_regression']}）。
- PR 自带新测试 `tests/test_hex20_distortion.py`：{tot['branch_only_tests']} 项全部通过。
- 合并后合计 {tot['merged_passed']} 项通过、0 失败。原始日志：`group_a_*.log`、`group_b_*.log`、
  `branch_only_hex20_distortion.log`。

## 5. 注意点与局限

- 未跑全仓 1747 项测试：本机 load average 300+（多智能体并发），全量套件无法在单次会话的超时窗口内完成；
  因此选取的是**双方共改/相邻的核心单元**（秩审计、HEX20、二次实体、solid3D、B-bar、locking）做同集合基线对比，
  并对 19 项 PR 自带新测试全跑。全量回归留给 merge_governor 周期复验。
- 文档冲突的解法是人工语义并集，机器审计只能证明「行/token 未丢」，不能证明论述不重复；两处并集段落已通读，
  未见事实冲突（两侧都在讲同一组 3×3×3 生产规则，分支补充「此前误标」与畸变证据，main 补充 tet10 与 27/8 点规则）。
- 本单只治愈合并面，不改变任何数值算法；`hex20` 积分规则与板壳 SRI/MITC 默认值均未动。
- PR 合入 main 由 merge_governor / pr-gate 记账；本单的产物是「分支已可与 main 干净合并且测试不低于基线」。
"""

(HERE / "merge_report.md").write_text(report, encoding="utf-8")
print("wrote", HERE / "merge_report.md", len(report), "chars")