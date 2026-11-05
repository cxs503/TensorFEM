# PR #70 治愈记录：`agent/m3297` 三方合并入 main

**范围**：TensorFEM PR #70（分支 `agent/m3297`）此前 `mergeable=false`，本记录给出合并、逐文件解冲、语义并集判据、测试证据与误差判定。
**结果**：`agent/m3297`: `9bf7378` → `ea42337`（merge commit，第一父 `7accc4b` = main）。

---

## ① 案例介绍

PR #70 的内容是**非均匀预屈曲应力下板格屈曲**的闭环验证：

* 物理背景。船体舷侧板与甲板板在总纵弯曲下，面内压应力沿板宽**线性变化** `σ(y) = σ_max (1 - 2y/b)`，中性轴处应力过零。既有 Galerkin 屈曲引擎只会把**常数**应力合量乘进几何刚度（单轴 `k = 4`、剪切 `k_s = 9.34/5.34`），纯弯曲工况根本无法表达；若用宽度平均值代替 `σ(y)`，结果会**低 6 倍**（不安全）而现有锚点毫无察觉。
* 分支交付。`tensorfem.plate_bending_buckling`（任意剖面几何刚度、自适应 Gauss–Legendre 求积、线性纯弯曲闭式矩阵、验证过的非正定铅笔求解器）+ 一条**不共享代码路径**的独立参考（无限长板、`W(y)sin(kx)`、y 向三次 Hermite C1 单元 + 半波长搜索），并对 `plate_shear_buckling` 的共享铅笔求解器修了一个潜在缺陷。
* 几何/参数（无量纲，与模块测试一致）：`b = 1`、`D = 1`、各向同性 `coupling_rigidity = D`；文献锚点 Timoshenko & Gere（Sec. 9.2）/ Bulson 1970：`k_b = 23.9`；验收门限 3%。
* 合并侧参数：main = `7accc4b`（owner 2026-11-05 手动排空后），分支 = `9bf7378`，共同祖先 = `dff97e7`。

## ② 怎么做的

**流程**

1. `git worktree add` 建隔离工作树；`main` 与 `agent/m3297` 全部走本地 ref（远端 fetch 在本环境超时，见 ⑤）。
2. 先试合到当时可见的 main `14ca2fd`（干净），但随后 main 被 owner 手动排空推进到 `7accc4b`，**该次尝试整体作废**，改在 `7accc4b` 上重做三方合并。
3. `git merge --no-commit --no-ff agent/m3297` → 1 个冲突文件、1 个冲突块。
4. 逐文件解冲（下节），**并用程序校验双方内容零丢失**，而不是靠人眼看。
5. 提交 merge commit；在合并树上跑测试；把 `agent/m3297` 指到 merge commit。

**冲突文件清单与逐文件解法**

| 文件 | 状态 | 解法 |
|---|---|---|
| `docs/benchmark-qualification-status.md` | **CONFLICT（1 块）** | 语义并集：保 main 的「开孔半球壳参照已补并给出否定结论（均值 5.09% / 最大 9.00%，`stress_accuracy_qualified` 维持 False）」；把分支的「非均匀预屈曲应力已闭环（`k_b = 23.8818` vs 23.9）」**逐字搬运**到 `仍缺：` 之前；再把已闭环项从 `仍缺` 清单删掉（否则同一句自相矛盾：既说闭环又列未闭环）。分支侧残留的「开孔半球壳仍缺独立应力参考」是 main 已被推翻的旧结论，**不得**保留。 |
| `CHANGELOG.md` | 自动合并 | 通过：两侧行集完整（main 265/265 行、分支 219/219 行），分支 4 条 Unreleased 条目与 main 的新条目并存，无重复段。 |
| `docs/assets/*.png`×3, `benchmarks/plate_bending_buckling*`, `docs/plate-bending-buckling-validation.md`, `src/tensorfem/plate_bending_buckling.py`, `tests/test_plate_bending_buckling.py` | 新增（无冲突） | 原样入库。 |
| `src/tensorfem/plate_shear_buckling.py` | 单侧修改 | 分支的同余变换修复（`L^-1 G L^-T`，原为 `L^-T G L^-T`）原样保留；main 未触及该文件。 |

**反静默丢失校验（不只是 git 的自动合并，机械 overlay 有前科）**

```
merged vs origin/main : 11 files, +1489/-3   ← 恰好等于分支自身的改动集
merged vs agent/m3297 : main 的 192 files 全集，无 deletions 丢失（diff --filter=D 为空）
CHANGELOG.md 行集      : main 265/265、分支 219/219 全部保留
冲突标记残留           : 0
```

解法脚本可复算：`artifacts/pr70/resolve_status_doc.py`（内含断链断言：`5.09%`、`23.8818`、两个报告链接、`Hertz 接触` 必须同时存在；`开孔半球壳仍缺独立应力参考` 必须不存在）。

## ③ 计算结果

**测试证据（合并树 vs main 基线 `7accc4b`）**

| 项 | 合并树 | main 基线 | 判定 |
|---|---|---|---|
| 测试收集数 | 1712 | 1692 | +20（新模块），收集无错误 → 全树无导入破损 |
| `tests/test_plate_bending_buckling.py` | **20 passed**（66.6 s） | 文件不存在 | 新增能力全绿 |
| `tests/test_plate_shear_buckling.py` | **15 passed**（61.7 s） | 15 passed | 持平（修复未破坏既有剪切锚点） |
| `tests/test_marine_plate_combined_buckling.py` | **32 passed**（13.6 s） | 32 passed | 持平（main 新模块不受铅笔修复影响） |
| 共享两文件合计 | **47 passed** | **47 passed** | 不低于基线 |

**误差表（门限 3%，数据源 `benchmarks/plate_bending_buckling_evidence.json`，并由合并树上 20 个测试重新断言）**

| 关键量 | 参考值 | 计算值 | 相对误差 | <3%？ |
|---|---|---|---|---|
| `k_b`，Ritz a/b=2 basis 32 | 23.9 | 23.881813 | **0.0761%** | 是 |
| `k_b`，Hermite C1（32 单元，独立路径） | 23.9 | 23.880627 | **0.0811%** | 是 |
| Ritz ↔ Hermite 交叉 | — | — | **4.97e-5** | 是 |
| 均匀压缩极限 | 4.0 | 3.999999999999999 | ~1e-15 | 是 |
| `k_s`，a/b=1 | 9.34 | 9.324673 | **0.164%** | 是 |
| `k_s`，a/b=6（长板） | 5.34 | 5.479724 | **2.62%** | 是（临界带） |

**图**

* `artifacts/pr70/figs/kb_anchor_and_convergence.png`：`k_b(a/b)` 扫掠 vs 23.9 锚点、vs 均匀压缩 4.0、vs 经典最小点 a/b=2/3；右图基函数收敛 + Hermite 独立参考。
* `artifacts/pr70/figs/field_and_mode.png`：预屈曲应力场 `σ(y)=1-2y/b` 与预应力加权屈曲模态云图（半波长 0.672 b）。
* `artifacts/pr70/figs/error_budget.png`：全部锚点 vs 3% 门限（对数轴）。

## ④ 注意点 / 局限

1. **基线会动**：首次合并基于 `14ca2fd` 是干净通过的，之后 main 被手动排空推进 6→ 数十个 PR，旧合并整体作废。合并必须**在提交前重新核对 `origin/main`**，否则复验必然再次 dirty。
2. **自动合并 ≠ 语义正确**：`docs/benchmark-qualification-status.md` 的两侧文本都在描述「尚未闭环的工作」，机械取一侧会分别丢失「半球壳否定结论」或「弯曲项闭环」。此类状态文档必须逐句合并并删掉矛盾项。
3. **同余变换的可观测性**：`_smallest_positive_from_symmetric_indefinite` 的 `L^-T G L^-T` 只在弯曲矩阵非对角（非各向同性/正交各向异性）时才与正确的 `L^-1 G L^-T` 不同——这正是所有已发布剪切锚点都没发现它的原因；回归测试只能用**解析 3×3 铅笔**造出来。
4. **本机无法跑全量套件（honest FAIL）**：全量 1418+ 测试在 load average 5676–6215（128 核严重超订）下两次尝试均被环境收割——单会话跑到 96% 时 worker 丢失（pluggy teardown `OSError: cannot send`），分块运行在 ~90% 收到 SIGTERM；且后台任务会在启动调用结束后约 120 s 被整组收割。因此**没有拿到全量套件的通过汇总**；单次 8×8 Ritz 求解实测墙钟 85 s（user 13 s），全量代价不可承受。全量门禁以 `.github/workflows/ci.yml`（push/PR 触发，Python 3.11+3.12）为准。
5. 合并树上的重算脚本（`post_merge_recheck.py`）与精简版同样被上述负载打断，因此表 3 的数字取自分支入库证据 JSON——但这些数字**已被合并树上通过的 20 个测试逐条重新断言**（含 `k_b` 3% 门限、Hermite 双向收敛、均匀极限 1e-9、解析铅笔回归），故结论不依赖未复算的中间量。
6. 远端推送：本环境 `git ls-remote` 超时，`agent/m3297` 的远端更新需由平台自动推送完成；本地 ref 已指到 `ea42337`。

## ⑤ 误差判定

| 判据 | 阈值 | 实测 | 是否达标 |
|---|---|---|---|
| `k_b`（两条独立路径） | 3% | 0.076% / 0.081% | **是** |
| `k_s`（方板 / 长板） | 3% | 0.164% / 2.62% | **是**（长板余量仅 0.38 pt，属已知临界带） |
| 共享测试不退化 | ≥ 基线 | 47 vs 47 | **是** |
| 全量套件 | 无失败 | 未取得汇总 | **未判定（环境所致，如实记录）** |

## 复现

```bash
git worktree add /tmp/pr70 base=7accc4b            # main
git merge --no-commit --no-ff agent/m3297          # 1 conflict
python artifacts/pr70/resolve_status_doc.py        # 语义并集 + 断言
python -m pytest -q tests/test_plate_bending_buckling.py \
        tests/test_plate_shear_buckling.py tests/test_marine_plate_combined_buckling.py
python artifacts/pr70/make_figures.py              # 图与误差表
```