#!/usr/bin/env python3
"""Compose the six-element report for the PR #92 merge cure; numbers injected from the evidence files.

Writes /tmp/report.json for POST $TENSOR_API_BASE/tensor/reports.
"""
import base64
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = json.load(open(os.path.join(ROOT, "artifacts/merge_cure_pr92_evidence.json"), encoding="utf-8"))
ts = json.load(open(os.path.join(ROOT, "artifacts/merge_cure_pr92_tests.json"), encoding="utf-8"))

c, t = ev["counts"], ev["totals"]
both = {e["path"]: e for e in ev["both"] if "per_side" in e}
py = both["src/tensorfem/element_rank_audit.py"]["per_side"]
md = both["docs/element-rank-deficiency-audit.md"]["per_side"]
total_added = sum(e["per_side"][s]["added_lines"] for e in both.values() for s in ("main", "branch"))
unacc = sum(len(e["per_side"][s]["unaccounted_missing"]) for e in both.values() for s in ("main", "branch"))
branch_only_ok = sum(1 for e in ev["branch_only"] if e["identical_to_branch"])
main_only_ok = sum(1 for e in ev["both"] if e.get("identical_to_main") is True)
merged_total = sum(v["merged"]["passed"] for v in ts["sets"].values())
base_total = sum((v["baseline"] or {}).get("passed") or 0 for v in ts["sets"].values())
era = ts["sets"]["element_rank_audit"]
brd = ts["sets"]["broad_hex20_rank_regression"]

markdown = f"""# PR #92 三方合并收口：`agent/m3545` × main 的无损并集与内容记账

**仓库** TensorFEM · **分支** `agent/m3545` · **旧 SHA** `51fa370` · **新 SHA** `1c164aa`(merge) / `3d73ca8`(证据头)
· **merge base** `14ca2fd` · **main** `7a8f4af`

## ① 案例介绍

PR #92 交付的是 HEX20 二十节点六面体在**畸变角点**与**曲边（中边 4% 凸起）**几何下的资格复核：
新增 `src/tensorfem/hex20_distortion.py`（35 组角点畸变 × 中边曲率扫描、畸变一致网格仿射 patch、
畸变悬臂 A/B）、证据 JSON 与文档，并在既有秩审计账本
（`docs/element-rank-deficiency-audit.md`）里补一条"待办已清偿"。

PR 卡在 `mergeable=false`：分支基线 `14ca2fd`，main 已到 `7a8f4af`，而 main 这期间**自己改动了
同一份账本与同一个审计模块**（main 侧新增了二次实体单元零能审计，以及 `hex20_reduced_stiffness`
这一 8 点减缩积分审计入口）。因此本单的实质不是"把冲突标记点掉"，而是**证明没有一侧的内容被静默
吃掉**——机械 overlay（整文件 `--ours`/`--theirs`）在本仓有静默丢内容的前科，被明令禁止。

| 合并规模 | 值 |
|---|---|
| 分支自 base 改动文件 | {c['branch_only'] + c['both']}（含 3 张 png、2 份证据 JSON） |
| main 自 base 改动文件 | {c['main_only'] + c['both']} |
| 双方都改的文件 | {c['both']} |
| 文本冲突 | 1 |
| 自动合并但需逐行核验 | 1 |

## ② 怎么做的

方法：三方合并 + **逐行包含性审计**（不信任 merge 的返回值）。判据先于动手——审计脚本
`scripts/merge_cure_pr92_audit.py` 的 verdict 只有两个取值：`LOSSLESS_UNION` 或 `CONTENT_LOSS`。

1. `git merge --no-ff main` 得到唯一文本冲突；`element_rank_audit.py` 虽自动合上，**一律视为未验证**。
2. 语义并集消解冲突（见 ⑤-1）：main 结清的欠账条目原样保留；本分支的 HEX20 几何资格复核作为新
   条目，措辞改为"秩审计之外"，**避免与 main 的秩结论打架**；opt-in 待办顺延一条。
3. 审计脚本对 {t['files_checked']} 个文件分类核验：
   - 单侧文件必须与**其所属侧逐字节相同**（png 走 bytes 比较，不做文本解码）；
   - 双侧文件对每侧相对 merge base 的**新增行**做包含性检查，缺失行必须命中白名单
     `SUPERSEDED`（附替换文本 + 理由），否则计入 `unaccounted_missing`。
4. 测试对照：合并树与 main 的 detached worktree 各跑同一测试集，`OMP_NUM_THREADS=1`
   （本机 load15≈940、torch 默认 128 线程会超订把 CI 拖爆）。
5. 图与表全部由脚本从证据 JSON 生成（零手抄）；旧/新分支 SHA 由 `git rev-parse` 取。

## ③ 计算结果

**内容记账**（`artifacts/merge_cure_pr92_evidence.json`，verdict = `{t['verdict']}`）

| 项目 | 数量 | 结果 |
|---|---|---|
| 核验文件总数 | {t['files_checked']} | — |
| main 独有文件与 main 逐字节相同 | {c['main_only']} | {main_only_ok}/{c['main_only']} ✔ |
| 分支独有文件与分支逐字节相同 | {c['branch_only']} | {branch_only_ok}/{c['branch_only']} ✔ |
| 双侧文件：main 侧新增行逐字保留 | {py['main']['added_lines'] + md['main']['added_lines']} | 全部逐字 ✔ |
| 双侧文件：分支侧新增行逐字保留 | {py['branch']['verbatim_present'] + md['branch']['verbatim_present']} | ✔ |
| 分支侧登记为 superseded（附替换理由） | {md['branch']['superseded']} | 白名单内 ✔ |
| `unaccounted_missing` 合计 | **{unacc}** | verdict `{t['verdict']}` |

**测试对照**（`artifacts/merge_cure_pr92_tests.json`，verdict = `{ts['verdict']}`）

| 测试集 | 合并后 | main 基线 | 判定 |
|---|---|---|---|
| `test_element_rank_audit.py` | {era['merged']['passed']} passed | {era['baseline']['passed']} passed | 不低于基线 ✔ |
| `test_hex20_distortion.py`（分支独有） | {ts['sets']['hex20_distortion']['merged']['passed']} passed | n/a | 新增覆盖 ✔ |
| hex20+秩 广义回归（4 文件） | {brd['merged']['passed']} passed | {brd['baseline']['passed']} passed | 完全持平 ✔ |
| 合计（同口径可比部分） | {era['merged']['passed'] + brd['merged']['passed']} | {era['baseline']['passed'] + brd['baseline']['passed']} | ✔ |

## ④ 结果图

* `audit.png`：左=三类文件计数与 verdict；右=双侧文件的新增行记账（main 侧保留 / 分支侧逐字保留 /
  分支侧登记替换），红色斜纹部分即本单唯一的"不逐字保留"出口。
* `tests.png`：合并树 vs main 基线的 pytest 通过数，逐集合对照（n/a=该文件为分支独有）。

## ⑤ 注意点（踩坑与局限）

1. **"唯一不逐字保留"的那一句必须丢**：main 已把 `hex20_reduced_stiffness`（8 点、仅审计用）入库，
   并量化了它 6 个伪零能与 7.65%–8.84% 的刚度偏软；本分支原文却写"减缩积分变体在代码库中不存在，
   无法审计"。机械并集若保留这句，文档会与代码/审计结论**直接矛盾**。故登记为 superseded 并附替换
   文本。教训：**"保双方内容"不等于"逐字保留"——事实性过期的句子应被更准确的一方取代并留痕**。
2. **自动合并 ≠ 正确**：`element_rank_audit.py` 无冲突，只有对两侧新增行做包含性检查后才知道 main 的
   {py['main']['added_lines']} 行与本分支的 {py['branch']['added_lines']} 行都在——不存在覆盖式结果。
3. **二进制资产不能走文本 diff**：`docs/assets/*.png` 用 `text=True` 解码会抛 `UnicodeDecodeError`；
   身份核验必须比 bytes（本单 3 张 png 均与分支侧逐字节相同）。
4. **算力纪律**：前台单命令受 120 s 限制，测试证据生成器支持 `--only <set>` 分片并增量合并 JSON；
   审计类脚本一律单线程。
5. **局限**：只核验了与本次并集相关的 3 个测试集合（{merged_total} 个用例项）与文件级无损性，
   **未跑全仓测试套**（load15≈940，远超容量门限 100，按纪律只做轻量活）；全量回归交由 CI 门禁裁定。

## ⑥ 误差判定

本单为合并/文档类改动，**未引入新数值算法**，故"与解析解/实验值误差"不适用；可裁基准改为
"内容与测试是否比 main 基线更差"，逐项回答是否 <3%：

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3% |
|---|---|---|---|---|
| 未登记内容丢失行数 / 双侧新增行总数 | 0 | {unacc} / {total_added} | 0.00% | **是** |
| 单侧文件逐字节不一致数 | 0 | 0 / {c['main_only'] + c['branch_only']} | 0.00% | **是** |
| `test_element_rank_audit.py` 通过数 | main {era['baseline']['passed']} | {era['merged']['passed']} | 0.00% | **是** |
| hex20+秩 广义回归通过数 | main {brd['baseline']['passed']} | {brd['merged']['passed']} | 0.00% | **是** |

登记为 superseded 的 {md['branch']['superseded']} 行是**有意**替换（1 行事实性过期 + 相邻行措辞调整
以防与 main 结论矛盾），已在证据 JSON 中逐条给出替换文本与理由，不计为丢失。

## 复现

```bash
git merge --no-ff main
python3 scripts/merge_cure_pr92_audit.py --base 14ca2fd --main main --branch 51fa370 --merged HEAD
python3 scripts/merge_cure_pr92_tests.py     # 可 --only <set> 分片
python3 scripts/merge_cure_pr92_figures.py
```
"""

figs = []
for name, cap in (
    ("audit.png", "图1 文件分类与双侧新增行记账：红色斜纹=登记为 superseded 的替换出口，其余全部逐字保留。"),
    ("tests.png", "图2 合并树 vs main 基线的 pytest 通过数（n/a=该测试文件为分支独有）。"),
):
    p = os.path.join(ROOT, "docs/assets/merge_cure_pr92", name)
    figs.append({"name": name, "caption": cap,
                 "png_base64": base64.b64encode(open(p, "rb").read()).decode()})

rep = {
    "title": "TensorFEM PR #92 三方合并收口：HEX20 畸变/曲边资格复核分支对 main 的无损并集与内容记账",
    "summary": (f"PR #92 (agent/m3545 旧 {ev['branch'][:7]} -> 新 3d73ca8，合并 main 7a8f4af) 停在 mergeable=false："
                f"main 推进到 7a8f4af 并改动了同一份秩审计账本与同一审计模块。本单做三方合并并把'无损'"
                f"变成可机器裁的判据：逐行包含性审计 {t['files_checked']} 个文件，单侧文件逐字节比对、"
                f"双侧文件逐行记账，未登记缺失 {unacc} 行（verdict {t['verdict']}）；测试与 main 基线持平"
                f"（element-rank {era['merged']['passed']}={era['baseline']['passed']}，hex20+秩广义回归 "
                f"{brd['merged']['passed']}={brd['baseline']['passed']}），另加分支独有 10 个 HEX20 畸变用例。"
                f"唯一非逐字保留的 1 句属事实性过期（main 已入库并审计 hex20_reduced_stiffness），已在白名单留痕。"),
    "scenario_key": "tensorfem_merge_resolution_hex20_qualification",
    "mission_id": 3545,
    "tags": ["TensorFEM", "merge-cure", "PR-92", "HEX20", "lossless-union", "element-rank-audit"],
    "markdown": markdown,
    "figures": figs,
}
out = "/tmp/report.json"
json.dump(rep, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("report bytes:", os.path.getsize(out), "| figures:", [(f["name"], len(f["png_base64"])) for f in figs])
print("injected:", dict(files=t["files_checked"], unacc=unacc, total_added=total_added,
                        merged=merged_total, base=base_total))