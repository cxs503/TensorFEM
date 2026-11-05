# PR #61 第二轮治愈：`agent/m3202` 追上 `main` 7a8f4af（冲突清单 · 逐文件解法 · 内容并集审计 · 功能复核）

- 仓库：`cxs503/TensorFEM`（`src/tensorfem/`，有限元求解器库）
- 被治愈的 PR：#61，源分支 `agent/m3202`
- 本工作树：`/root/tp-agent/work/m3918`（分支 `agent/m3918`，起点 = PR 旧头 `75a648ab9`）
- 机器证据：`artifacts/pr61_r2_union_evidence.json`（并集审计）、
  `artifacts/pr61_r2_negative_control_*.json`（负控）、
  `artifacts/pr61_r2_physics_check.json`（合并树功能复核）、
  `artifacts/evidence_logs/r2_*.log`（pytest 原始日志）、
  `docs/assets/pr61_merge_cure_r2/{audit,physics}.png`（出图器 `artifacts/pr61_r2_finalize.py`）

## ① 案例介绍：这一轮的"脏"是什么，业务载荷是什么

第一轮治愈（`docs/pr61-merge-reconciliation.md`）把分支对齐到了 `main` 的 `7accc4b`。此后
`main` 又前进了 **PR #72 的治愈提交**（`7a8f4af`，`agent/m3327` 的三方并集），而分支没有跟上，
于是 PR #61 再次被判 `mergeable=false`。**冲突不在业务代码，而在两份互相独立的文档增量**：

| 侧 | 业务载荷 | 触及文件 |
|---|---|---|
| 分支 `agent/m3202` | 船舶**大倾角稳性**：棱柱形驳按精确浸水剖面求 `GZ`/`KN`（`tensorfem.marine_large_angle_gz`，554 行） | 新增 `src/`、`tests/`、`benchmarks/large_angle_gz_reference.json`、`docs/marine-large-angle-stability.md`、`scripts/plot_large_angle_gz.py` |
| `main`（7a8f4af） | 船舶**板面内线性变化压缩屈曲**：非均匀预屈曲应力资格认证（`tensorfem.plate_bending_buckling`） | 新增 `src/`、`benchmarks/plate_bending_buckling*.py`、`docs/marine-plate-linear-stress-buckling-validation.md`、PR #72 治理证据 |

两侧都不改对方新增的文件，唯一同路径改动是 `CHANGELOG.md` 顶部（双方各追加一节），
这就是全部文本冲突的来源。

| 角色 | SHA（旧 → 新） |
|---|---|
| 分叉点 / 本轮 merge-base | `7accc4b9cfb45369958c432a8aa2f0ebbb846a5b`（= GitHub `main`） |
| 合并目标 `main` | `7a8f4afeb6dd79c2ad3be480a6117ed2550c2347`（本机 `tensorfem_local/main`，PR #72 治愈后） |
| 分支旧头（本地 `agent/m3202`） | `75a648ab900631e1bc5a4685cd379ce064416e25` |
| 分支旧头（**GitHub 上** Java 侧所见） | `6b4c4fa48efea38e63303f1d5ae921392d468fe6`（落后 `main` 一个 PR #72） |
| 本轮三方合并提交 | `dab43f46e6a80f801a6e3ee70478e3b3f3d03099`（父 = `75a648ab9` + `7a8f4afeb`） |
| 交付后分支头 | 本文件所在提交（平台推送后 GitHub 重算 `mergeable`） |

## ② 怎么做的

1. **建工作树、对齐三方**：`git worktree`（本工作树）+ 只读基线树 `/tmp/m3918_baseline`
   （`7a8f4af`），先 `git merge-tree` 预演，确认只有 `CHANGELOG.md` 一处文本冲突（1 个文件）。
2. **逐文件解冲，语义并集**：`CHANGELOG.md` 的两节是**独立发布说明**，任一侧胜出都会丢掉
   一条 PR 记录，故按「保留双方原文、按日期新者在先」合并，解冲器 `artifacts/pr61_r2_resolve_changelog.py`
   机器可复算（解冲后自检：两节标题都在、无冲突标记、上方空行未丢）。其余 42 个单侧文件
   直接取该侧 blob，**不做文本重写**。
3. **并集义务审计**（第一轮审计器复用，`--base` 钉住历史分叉点）：
   - 散文（`CHANGELOG.md`）→ 原子（反引号标识符/链接/路径/数字）+ 内容词义务；
   - 代码/数据单侧文件 → **逐字节等于该侧 blob**；
   - 删除台账与冲突标记扫描。
4. **负控（审计器灵敏度）**：把合并结果换成"只取 main"或"只取分支"的机械 overlay，审计必须判死。
5. **功能复核**（文本干净≠算子还对）：在合并树上重跑两侧载荷的解析锚——
   大倾角 `GZ` 对墙侧闭式、板屈曲 Ritz 对独立差分条路线与经典系数。
6. **回归测试**：同一 13 模块选择在合并树与 `7a8f4af` 基线树上各跑一次，比较通过数。

复现：

```bash
python3 artifacts/pr61_r2_resolve_changelog.py            # 冲突态下解冲（已应用）
python3 artifacts/pr61_merge_union_check.py --base 7accc4b9c --main 7a8f4afeb \
    --branch 75a648ab9 --merge dab43f46e --check
python3 artifacts/pr61_r2_physics_check.py                # 合并树功能复核
python3 artifacts/pr61_r2_finalize.py                     # 出图 + 生成本文件
```

## ③ 计算结果（全部由证据文件注入）

**并集审计**（`artifacts/pr61_r2_union_evidence.json`，verdict = `UNION_OK`）：

| 量 | 值 |
|---|---|
| 相对分叉点变更文件总数 | 43 |
| main 单侧 / 分支单侧 / 双侧 | 20 / 22 / 1 |
| 单侧文件逐字节一致性检查 | 42 / 42 通过 |
| 删除台账 / 残留冲突标记 | 0 / 0 |
| `CHANGELOG.md` main 侧原子丢失 / 内容词丢失 | 0 / 0 |
| `CHANGELOG.md` 分支侧原子丢失 / 内容词丢失 | 0 / 0 |
| 合并后 `CHANGELOG.md` 行数 | 400 |

**负控**（机械 overlay 必须被判死）：

| 被审计对象 | main 内容丢失 | 分支内容丢失 | 判定 |
|---|---|---|---|
| `--merge 7a8f4afeb`（只取 main） | 0 | 27 | UNION_FAIL |
| `--merge dcc632543`（只取分支） | 24 | 0 | UNION_FAIL |

**合并树功能复核**（`artifacts/pr61_r2_physics_check.json`）：

| 载荷 | 基准 | 结果 | 相对误差 |
|---|---|---|---|
| 大倾角 `GZ`（branch） | 墙侧闭式 `sinφ(GM + BM tan²φ/2)`，0…18.43° | 带内最大残差 1.332e-15 m | 8.26e-16 |
| 板屈曲 `k`（main） | 经典系数 4.0 / 7.81 / 23.9 / 25.6 | 见下表 | 最大 0.280% |

| 算例 | a/b | k(Ritz 16×16) | k(差分条 200 节点) | 经典 k | Ritz 误差 | 双路线差 |
|---|---|---|---|---|---|---|
| uniform_compression | 1.0 | 4.0000 | 3.9998 | 4.000 | 0.000% | 5.14e-03% |
| triangular_compression | 2.0 | 7.8120 | 7.8115 | 7.810 | 0.025% | 5.57e-03% |
| pure_bending_long_panel | 6.0 | 23.8818 | 23.8782 | 23.900 | 0.076% | 1.50e-02% |
| pure_bending_square | 1.0 | 25.5283 | 25.5244 | 25.600 | 0.280% | 1.55e-02% |

**回归测试**（原始日志 `artifacts/evidence_logs/r2_merged_core.log` 与 `r2_baseline_core.log`）：

| 树 | 选择 | 通过 | 失败 |
|---|---|---|---|
| `main` 基线 `7a8f4afeb` | 12 模块（两树共有） | 132 | 0 |
| 合并树 | 同一选择 + 分支新增的 `test_marine_large_angle_gz.py` | 171 | 0 |

差值 39 = 分支新增测试模块的用例数；**共有子集
132 项在两树全绿，回归 0 项**。

结构层面还有一条更强的判据：合并树相对 `main` 的 `src/`+`tests/` 差异**只有新增**
（`src/tensorfem/marine_large_angle_gz.py`、`tests/test_marine_large_angle_gz.py`，共
22 个分支单侧文件全部为新增），共享源码**一行未改**；
唯一的既有文件改动是两份散文文档（`CHANGELOG.md`、`docs/marine-hydrostatics.md`）。
因此回归风险被结构性压缩到"新增模块的导入副作用"，这也是测试选择覆盖的全部区域。

## ④ 结果图

- `docs/assets/pr61_merge_cure_r2/audit.png`：左＝合并面文件分布与删除/标记计数；中＝
  并集义务 vs 两个机械 overlay 负控（shipped 全 0，overlay 各丢 27
  /24 个原子）；右＝合并树与基线树的通过数（共有子集并列）。
- `docs/assets/pr61_merge_cure_r2/physics.png`：左＝大倾角 `GZ` 曲线与墙侧闭式叠加；中＝
  逐点残差（对数轴，全部收敛到机器精度）；右＝四类板屈曲 `k` 的三路对照（Ritz／差分条／经典）。

## ⑤ 注意点与局限（不粉饰）

1. **GitHub 侧分支头仍可能是旧 SHA**：本工作树内 `git push` 被运行环境安全策略拦截
   （`REJECTED: 命令被安全策略拦截`），故本轮只能把治愈后的分支头**落在工作树/本地目标仓**，
   GitHub 上的 `agent/m3202` 需平台推送器更新；`mergeable` 的最终重算由 `merge_governor`
   下一周期完成。本单能证明的是：治愈后的头**同时包含** GitHub `main`（`7accc4b9c`）
   与本机 `main`（`7a8f4afeb`），对两者都是 fast-forward 可合。
   本地目标仓 `tensorfem_local` 的 `agent/m3202` 仍指向 `75a648ab9`：该分支正被
   另一工作树（`/root/tp-agent/work/m3202`）检出，`git fetch` 拒绝更新已检出的 ref
   （`refusing to fetch into branch ... checked out at ...`）。因此**分支指针只能由平台推送器 /
   治理器做一次 fast-forward**：`dab43f46e6a80f801a6e3ee70478e3b3f3d03099` → 本轮证据提交（`git log -1`）。
2. **全量套件未在本工作树完成**：1761 个用例的全量跑在宿主 load15 ≈ 1300（128 核）下多次被
   宿主回收，日志 `artifacts/evidence_logs/r2_merged_full.log` 保留为空文件（事实记录，不隐藏）。
   本轮回归证据是 13 模块核心选择（171 vs 132）+ 上面的
   结构性判据（共享源码零改动）。
3. **散文并集只保证"断言不丢"**，不保证语序与断行与任一侧逐字相同；代码/数据仍是逐字节严格。
4. **`GZ` 的闭式锚只在墙侧带内成立**（本例 18.43°）；带外必须用几何
   求解器，这一点由分支载荷自带测试钉住，不在本单重复。
5. **板屈曲的经典系数是文献值**（4.0 / 7.81 / 23.9 / 25.6），`pure_bending_square` 的 0.280%
   是四例中最差的一项，仍远低于 3% 门槛；无文献锚点的正交各向异性算例本单不作精度宣称。

## ⑥ 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 实测 | 判定（阈值 3%） |
|---|---|---|---|
| 并集义务：合并结果丢失的内容单元 | 0（义务） | 0 | **是** |
| 负控灵敏度：机械 overlay 丢失的原子 | 必须 >0 | 27 / 24 | **是（审计器会咬）** |
| 单侧文件字节一致性 | 42/42 | 42/42 | **是** |
| 大倾角 `GZ` vs 墙侧闭式 | 闭式解 | 8.26e-16 | **是（机器精度）** |
| 板屈曲 `k` vs 经典系数（最差项） | 25.6 | 0.280% | **是** |
| 板屈曲双路线互校（最差项） | Ritz vs 差分条 | 1.55e-02% | **是** |
| 回归：共有子集通过数 | 132 | 132 | **是（0 回归）** |
| 全量套件（1761 用例） | 基线未知（宿主 Killing） | 未测得 | **无基准，声明原因**（见 ⑤-2） |
