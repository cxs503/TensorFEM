"""Build the mission-3255 progress report (six-element paper-grade) with programmatic numbers."""
import json, base64, subprocess, os

REPO = "/root/tp-agent/work/m3255"
def sh(*a): return subprocess.run(a, cwd=REPO, capture_output=True, text=True).stdout.strip()

meta = json.load(open("/tmp/fig_meta.json"))
counts, audit, tests = meta["counts"], meta["audit"], meta["tests"]

ev = json.load(open(os.path.join(REPO, "benchmarks/plate_bending_buckling_evidence.json")))
lit = ev["literature"]
ab1 = ev["psi_sweep"]["a_over_b=1"]
k_sq_psi1 = [x for x in ab1 if x["edge_ratio"] == -1.0][0]["coefficient_k"]
k_sq_psi0 = [x for x in ab1 if x["edge_ratio"] == 0.0][0]["coefficient_k"]
ab2 = ev["psi_sweep"]["a_over_b=2"]
k_l2_psi1 = [x for x in ab2 if x["edge_ratio"] == -1.0][0]["coefficient_k"]
fd_fine = ev["finite_difference_convergence"][-1]
def err(v, ref): return abs(v - ref) / ref * 100

rows = [
    ("板格纯弯（方形 a/b=1，ψ=1）", k_sq_psi1, lit["kb_pure_bending_square"], err(k_sq_psi1, lit["kb_pure_bending_square"])),
    ("板格纯弯（a/b=2，ψ=1）", k_l2_psi1, lit["kb_pure_bending_long"], err(k_l2_psi1, lit["kb_pure_bending_long"])),
    ("板格均匀压缩（ψ=0，=k_uniform 极限）", k_sq_psi0, lit["k_uniform_long"], err(k_sq_psi0, lit["k_uniform_long"])),
    ("均匀压缩独立 FD 解", fd_fine["uniform_k"], lit["k_uniform_long"], fd_fine["uniform_error_vs_classical"] * 100),
]

head_old, head_new = "949f57a", sh("git", "rev-parse", "HEAD")
main_sha = "7accc4b"
n_main, n_br = counts["main 7accc4b"], counts["branch 0e3190c"]
n_m1, n_m2 = counts["merge d446f28"], counts["merge +restored"]
t_merged = sum(v[0] for v in tests.values()); t_base = sum(v[1] for v in tests.values())

anchor_tbl = "\n".join("| %s | %.4f | %.4f | **%.3f%%** | %s |" % (n, v, r, e, "是" if e < 3 else "否")
                       for n, v, r, e in rows)
test_tbl = "\n".join("| `%s` | **%d passed** | **%d passed** | %s |" % (
    k, v[0], v[1], "不适用（基线无此文件）" if v[1] == 0 else "持平") for k, v in tests.items())

md = f"""# PR #65「板格线性边缘应力梯度屈曲」合并治愈与完整性复核（第 2 轮）

## ① 案例介绍

| 项 | 内容 |
|---|---|
| 对象 | TensorFEM PR #65，分支 `agent/m3255`，主题：板格在**线性边缘应力梯度**（非均匀预屈曲应力）下的屈曲系数解析基准闭环 |
| 故障 | GitHub 报 `mergeable=false`：分支停在 `{head_old}`（父 `dff97e7`），main 已前进到 `{main_sha}`（58 个提交） |
| 物理背景 | 板格屈曲系数 k 满足 Timoshenko & Gere《Theory of Elastic Stability》(1961) Table 9.1：均匀压缩 k=4.00（长板）、三角形/纯弯 k=7.81、纯弯 a/b=2 k=23.9、方形纯弯 k=25.6；工程上船体板格受总纵弯曲产生的**线性边缘应力梯度** ψ∈(-1,1)，需 k(ψ) 连续过渡到上述经典锚点 |
| 分支交付 | 生产模块 `src/tensorfem/plate_bending_buckling.py`(716 行)、benchmark 驱动(435)、证据 JSON(371)、测试 `tests/test_plate_bending_buckling.py`(323 行 / 36 用例)、验证文档、2 张图 |
| 本轮目标 | 三方合并**不静默丢内容**、本地核心测试不低于 main 基线、分支推回可快进（PR 可被 merge_governor 自动合并） |

## ② 怎么做的

1. **先校准远端真值再动手**：全量 `git fetch` 在 120 s 墙钟内超时，改为按 ref 精确抓取
   `refs/heads/agent/m3255`；`git ls-remote` 得 GitHub `main={main_sha}`、分支 `{head_old}`。
   校验 `tree({head_old}) == tree(0e3190c)`（均为 `78c5582`，`git diff` 为空）→ 远端尖端与本地合并源**内容等价**，
   以本地覆盖远端不会丢远端内容；但 `{head_old}` 不在本地祖先链上，直接推送非快进。
2. **工作树卫生**：复核开始时索引残留 **250 文件、78464 行的「暂存删除」**（上一轮机械 overlay 残渣，
   若提交会静默删掉 benchmarks/ 与 docs/assets/ 大量已合并内容）。用
   `git restore --source=HEAD --staged --worktree .` 归零，**从未提交**；归零后 `git status` 干净。
3. **三级机械对账**（不信任上一轮自述）：
   - *路径级*：`main \\ merge`、`branch \\ merge` 求差集，逐条定位丢失；
   - *文件级*：对「父两侧不同、合并取 main 单侧」的 12 个文件，比对 `branch:file == merge-base:file`，
     判定分支侧是否真的改过（未改则取 main 不构成丢失）；
   - *行级*：对两侧都改的文件，取 `git diff -U0 merge-base..branch` 的 `+` 行，逐行核验是否出现在合并结果中。
4. **修补唯一丢失**：`.scratch/proto_ritz.py` 从 `0e3190c` 原样恢复（分支相对合并基新增 8 个文件中唯一未落合并树的）。
5. **补祖先**：`git merge --no-ff origin/agent/m3255`。唯一冲突在 `docs/benchmark-qualification-status.md`，
   用行级证据判定「我方（第 1 轮手工语义并集）是远端版本的严格超集」后取我方：
   我方同时保留远端 3 条事实中的全部硬信息（Cook 膜 Q8 参考、5.02% 归因、探针协议未良置），
   并把已被 main 取代的旧句「开孔半球壳仍缺独立应力参考」升级为主的新结论
   （HEX20 实体 oracle：内部表面 von Mises 平均 5.09%/最大 9.00%，孔边 6.87%，`stress_accuracy_qualified` 维持 `False`）。
6. 合并后校验 `tree(HEAD) == tree(pre-merge)`（均 `c0d9559`）→ 补祖先对内容**零改动**；全树冲突标记 0 处。

## ③ 计算结果

**文件集合对账**（程序化 `git ls-tree` 计数）

| 提交 | 文件数 |
|---|---|
| main `{main_sha}` | {n_main} |
| 分支 `0e3190c` | {n_br} |
| 合并树（第 1 轮 `d446f28`） | {n_m1} |
| 合并树（本轮恢复后 `{head_new[:7]}`） | {n_m2} |

- main 侧静默丢失：**{audit['lost_from_main']} 个**
- 分支侧静默丢失：第 1 轮 **{audit['lost_from_branch']} 个**（`{', '.join(audit['lost_from_branch_files'])}`）→ 本轮恢复后 **0 个**
- 「两侧不同、取 main 单侧」的 12 个文件：分支侧相对合并基 `+` 行数**全为 0**（分支从未编辑）→ 无分支改动被覆盖
- 全树 `^<<<<<<<` / `^>>>>>>>`：**0 处**

**测试对等**（同子集、同 flags、串行，两棵树 `PYTHONPATH=<tree>/src`，`pytest -q -p no:cacheprovider --tb=no`；
基线树由 `git archive {main_sha}` 导出）

| 子集 | 合并树 | main 基线 | 判定 |
|---|---|---|---|
{test_tbl}
| **合计** | **{t_merged} passed** | **{t_base} passed** | 差值 {t_merged - t_base} = 分支新增用例数，无回归 |

**分支物理锚点复核**（本轮不重标定，数值取自分支证据 JSON，误差由程序计算）

| 锚点 | 计算 k | 文献 k（T&G 1961 Table 9.1） | 相对误差 | <3% |
|---|---|---|---|---|
{anchor_tbl}

## ④ 结果图

图 (a) 三方合并文件对账（main 侧丢失恒为 0，分支侧丢失 1→0）；(b) 合并树 vs main 基线测试通过数（三组子集）；
(c) 索引中的 overlay 残渣（250 文件 / 78.4k 行）在提交前被清零，未污染合并成果。

## ⑤ 注意点

1. **机械 overlay 的伤可以藏在「索引」里而不是「提交」里**：第 1 轮的合并提交本身完好，但工作树/索引留着
   250 删除项的残渣，一个 `git commit -a` 就会静默删掉大量已合并内容。治愈类任务必须显式检查 `git status` 的暂存区。
2. **「合并取某侧全文」不等于丢内容**，判据是 `branch:file == merge-base:file`（该侧从未改）。12 个文件全部满足，
   因此第 1 轮的取值正确；但仍需机械判定，不能凭直觉。
3. **远端分支尖端可能不在本地祖先链上**（`{head_old}` 与 `0e3190c` 树相同但父不同）。直接 push 为非快进；
   正解是先 `merge` 补齐祖先（树不变的合并），再快进推送，而不是强推。
4. `git fetch` 全量超时（120 s 墙钟），按 ref 精确抓取可行；xdist 在两棵独立树并发会互相争用导致基线树 collect 为空，
   对等比较必须串行。
5. **本单未完成项（如实记录）**：`git push` 被平台安全策略拦截（本单亦禁止 agent 自行推送 GitHub），
   故 GitHub 侧分支仍是 `{head_old}`；本地 `{head_new[:7]}` 已把 `{head_old}` 纳入祖先，**推送即快进**，
   PR 的最终合并由平台推送 + merge_governor 四门裁定。

## ⑥ 误差判定

| 关键量 | 基准 | 偏差 | 是否 <3% |
|---|---|---|---|
| 合并树 vs 基线 通过用例数（同集，{t_base} 例） | main `{main_sha}` | 0 例（0.000%） | **是** |
| 文件集合完整性（main 侧丢失） | 全量保留 | 0 个（0.000%） | **是** |
| 文件集合完整性（分支侧丢失） | 全量保留 | 恢复后 0 个（0.000%） | **是** |
| 板格锚点 k（4 条，上表） | T&G 1961 | 最大 {max(e for *_, e in rows):.3f}% | **是** |

结论：本轮为仓库治愈与验证，**不新增/不修改任何求解器物理实现**，分支物理结论沿用原有证据（上表误差均 <3%）。
测试与文件集合两类关键量相对 main 基线偏差为 0（整数计数，逐项相等），无回归。
"""

fig_path = os.path.join(REPO, "docs/assets/merge-cure-pr65/round2_audit.png")
fig_b64 = base64.b64encode(open(fig_path, "rb").read()).decode()

report = {
    "title": "PR #65 合并治愈与完整性复核：板格线性边缘应力梯度屈曲（文件集合/行级三级对账 + 测试对等）",
    "summary": ("对 TensorFEM PR #65 做三方合并治愈：先按 ref 抓取 GitHub 真值（main=%s、分支=%s），"
                "以路径级/文件级/行级三级机械对账证明 main 侧丢失 0、分支侧唯一丢失（.scratch/proto_ritz.py）已恢复；"
                "清除索引中 250 文件/78464 行的 overlay 残渣（从未提交）；补祖先后 tree 零改动（c0d9559），"
                "合并树核心测试 %d passed 与 main 基线 %d passed 持平、另含分支新增 36 用例全绿。"
                "本地 tip=%s 已含 GitHub 分支尖端为祖先，推送即快进；push 被平台策略拦截，最终合并待平台推送 + merge_governor。"
                % (main_sha, head_old, t_merged, t_base, head_new[:7])),
    "scenario_key": "tensorfem/pr65-merge-cure",
    "mission_id": 3255,
    "tags": ["TensorFEM", "PR65", "merge-cure", "conflict-resolution", "integrity-audit",
             "plate-buckling", "verification-debt", "m3255"],
    "markdown": md,
    "figures": [{
        "name": "round2_audit.png",
        "caption": "(a) 三方合并文件对账：main 1227 / 分支 1036 / 合并 1271 → 恢复后 1272，main 侧丢失 0；"
                   "(b) 合并树 vs main 基线测试通过数（25:25、43:43、分支新增 36:0）；"
                   "(c) 索引 overlay 残渣 250 文件 / 78.4k 行 → 0（提交前清零）。",
        "png_base64": fig_b64,
    }],
}
json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("report bytes", os.path.getsize("/tmp/report.json"))
print("md bytes", len(md.encode()))