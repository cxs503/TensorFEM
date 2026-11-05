# PR #92 merge cure (round 1)：`agent/m3545` 与 main 的无损并集

**分支**：`agent/m3545` · **merge base**：`14ca2fd` · **main**：`7a8f4af` ·
**旧分支 SHA**：`51fa370` · **新分支 SHA**：`1c164aa`（三方合并提交，父提交 = `51fa370` + `7a8f4af`）
· **平台报告**：`tensor_reports` id `4205`（`scenario_key = tensorfem_merge_resolution_hex20_qualification`）

## 1. 案例介绍

PR #92 的内容是 HEX20 六面体在**畸变角点**与**中边曲率**（曲边）几何下的资格复核
（`src/tensorfem/hex20_distortion.py`、`docs/hex20-distortion-robustness.md`、
`docs/benchmarks/tutorials/hex20-distortion-audit.json`，以及既有
`element_rank_audit.py` / `element-rank-deficiency-audit.md` 的补账）。

PR 停在 `mergeable=false`：分支基于 `14ca2fd`，而 main 已推进到 `7a8f4af`，期间 main 自身
改动了同一份账本（`docs/element-rank-deficiency-audit.md`）与同一个审计模块
（`src/tensorfem/element_rank_audit.py`）。本单即三方解冲突，判据是**内容无损**而非"能合上"。

| 量 | 值 |
|---|---|
| 分支自 base 改动文件 | 12（含 3 张 png、2 份证据 JSON） |
| main 自 base 改动文件 | 211 |
| 双方都改的文件 | 2 |
| 文本冲突 | 1（`docs/element-rank-deficiency-audit.md`） |
| 自动合并成功但需逐行核验 | 1（`src/tensorfem/element_rank_audit.py`） |

## 2. 怎么做的

工具链：纯 git 三方合并（`git merge --no-ff main`）+ 自写逐行审计脚本
（`scripts/merge_cure_pr92_audit.py`）+ pytest 基线对照（`scripts/merge_cure_pr92_tests.py`）
+ 作图脚本（`scripts/merge_cure_pr92_figures.py`）。**禁止**机械 overlay（`git checkout --ours/--theirs`
按文件整取），因为它会静默丢弃一侧内容。

流程：

1. `git merge --no-ff main`，得到唯一冲突文件；`element_rank_audit.py` 标为 auto-merged，
   但**视为未验证**，进入第 3 步逐行核验。
2. 人工语义并集消解冲突（见 §5 第 1 条）：main 的交集条目原样保留为第 6 条，本分支的
   HEX20 几何资格复核作为第 7 条（措辞改为"秩审计之外"，以**避免与 main 的秩结论矛盾**），
   opt-in 待办顺延为第 8 条。
3. `scripts/merge_cure_pr92_audit.py` 对 221 个文件逐类核验：
   - 单侧文件（10 个分支侧 + 209 个 main 侧）必须与**其所属侧逐字节相同**（二进制 png 走
     bytes 比较，不做文本解码）；
   - 双侧文件（2 个）对每一侧相对 merge base 的**新增行**做包含性检查，缺失行必须命中
     `SUPERSEDED` 白名单，且白名单条目要给出替换文本与理由，否则计为 `unaccounted_missing`。
4. 把上面的判断固化为可复跑的脚本与 JSON 证据（不手抄数字），再跑测试对照基线。

关键设计：白名单（`SUPERSEDED`）是唯一允许"不逐字保留"的出口，且每条都附替换文本——
这样"静默丢内容"在结构上不可能通过：任何未登记的新增行缺失都会让 verdict 变成 `CONTENT_LOSS`。

## 3. 计算结果

审计结论（`artifacts/merge_cure_pr92_evidence.json`）：

| 项目 | 数量 | 结果 |
|---|---|---|
| 核验文件总数 | 221 | — |
| main 独有文件与 main 逐字节相同 | 209 | 209/209 ✔ |
| 分支独有文件与分支逐字节相同 | 10 | 10/10 ✔ |
| 双侧文件中 main 侧新增行逐字保留 | 327 | 327/327 ✔ |
| 双侧文件中分支侧新增行保留 | 12（`element_rank_audit.py`） | 12/12 逐字 ✔ |
| 分支侧新增行（`element-rank-deficiency-audit.md`） | 7 | 4 逐字 + 3 登记为 superseded，0 未登记缺失 ✔ |
| `unaccounted_missing` 总数 | **0** | verdict = `LOSSLESS_UNION` |

测试对照（`artifacts/merge_cure_pr92_tests.json`，`OMP_NUM_THREADS=1`）：

| 测试集 | 合并后 | main 基线 | 判定 |
|---|---|---|---|
| `tests/test_element_rank_audit.py` | 11 passed | 11 passed | 不低于基线 ✔ |
| `tests/test_hex20_distortion.py` | 10 passed | 分支独有（main 无此文件） | N/A，新增覆盖 ✔ |
| 广义回归：`test_hex20.py` + `test_quadratic_solid_rank_audit.py` + `test_hemisphere_hex20_oracle.py` + `test_solid3d_mass.py` | 37 passed | 37 passed | 完全持平 ✔ |

图：`docs/assets/merge_cure_pr92/audit.png`（文件分类与双侧新增行记账）、
`docs/assets/merge_cure_pr92/tests.png`（合并后 vs 基线通过数），均由脚本从证据 JSON 程序化生成。

## 4. 注意点

1. **main 侧的一句话**把"HEX20 减缩积分变体待补"结清了：`hex20_reduced_stiffness`
   （order 2 / 8 点，仅审计用）已入库并审计过。分支原文却写着"减缩积分变体在代码库中不存在，
   无法审计"——**并集时必须丢这句**，否则文档自相矛盾且与代码不符。这是本单唯一
   "不逐字保留"的地方，已登记在白名单里并附替换文本。
2. **自动合并 ≠ 正确**。`element_rank_audit.py` 无冲突却可能是覆盖式结果，必须对两侧新增行
   逐条做包含性检查才能下结论。
3. **二进制资产不能走文本 diff**。`docs/assets/*.png` 若用 `text=True` 解码会抛
   `UnicodeDecodeError`；核验身份必须比 bytes。本单 3 张 png 均与分支侧逐字节相同。
4. **算力纪律**：本机 nproc=128 但 load15≈940，所有 pytest 均设 `OMP_NUM_THREADS=1`，
   否则 Torch 默认线程数会把 CI 拖到超时。单次前台调用受 120 s 限制，测试证据生成器支持
   `--only <set>` 分片执行并增量合并 JSON。
5. **范围局限**：只验证了与本次并集有关的 3 个测试集合（共 58 个用例项）与文件级无损性，
   **未**跑全仓测试套（load15 远超 100，按容量纪律只做轻量活）；全量回归留给 CI 门禁。

## 5. 误差判定

本单是**合并/文档类改动，未引入任何新数值算法**，因此"与解析解误差"不适用；可裁的量化基准
是"内容与测试是否可比基线更差"：

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 <3% |
|---|---|---|---|---|
| 未登记内容丢失行数 / 双侧新增行总数 | 0 | 0 / 353 | 0.00% | **是** |
| `test_element_rank_audit.py` 通过数 | main 11 | 11 | 0.00% | **是** |
| 广义回归通过数 | main 37 | 37 | 0.00% | **是** |
| 单侧文件逐字节不一致数 | 0 | 0 / 219 | 0.00% | **是** |

注册为 superseded 的 3 行是**有意**替换（1 行事实性过期、2 行为避免与 main 秩结论矛盾的
措辞调整），已在证据 JSON 中逐条给出替换文本与理由，不计为丢失。

## 6. 复现

```bash
git merge --no-ff main                                   # 得到唯一冲突并人工并集
python3 scripts/merge_cure_pr92_audit.py \
    --base 14ca2fd --main main --branch 51fa370 --merged HEAD
python3 scripts/merge_cure_pr92_tests.py                 # 可加 --only <set> 分片
python3 scripts/merge_cure_pr92_figures.py
```

证据：`artifacts/merge_cure_pr92_evidence.json`、`artifacts/merge_cure_pr92_tests.json`。