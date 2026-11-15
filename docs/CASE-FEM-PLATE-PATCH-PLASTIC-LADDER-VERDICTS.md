# CASE-FEM-PLATE-PATCH-PLASTIC-LADDER-VERDICTS — 验收判定层（锚点 GAP-PP-3-ICE-STRUCT-PLASTIC-SET）

**为什么有这个案例**：本锚点的三项验收判据里，两项目前由
`docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md` 承载，而该文档的「§6 误差判定」表
是**手写条件串**：每行的 `met / not met` 取自一个布尔量、数值取自另一个字段，两者都
不是从 artifacts 现算的。这一层一旦和 artifacts 脱钩，文档就能在数据不支持的情况下
声称通过——这正是本锚点历次交付里已经发生过的问题（同一张表里 `-34.88 %` 配 `<= 0 %`
被判成 `not met`，而按上界语义 `-34.88 % < 0` 是**允许**的）。

本案例交付的是**判定层本身**：把「验收判据 → artifacts」的映射写成可测函数，
并修掉一个只在三段网格梯子**跑完的那一刻**才会触发的崩溃。

---

## 1. 修掉的实质缺陷

### 缺陷 D1（阻塞级，`KeyError`，恰好在对数据可用时触发）

`scripts/patch_plate_plastic_ladder_figures.py` 的网格收敛判据：

```python
monotone = all(b[1] < a[1] for a, b in zip(
    [x for x in summary["records"] if x["geometry"] == "free"],
    [y for y in summary["records"] if y["geometry"] == "free"][1:])) \
    if len(meshes) > 1 else None
```

`zip(...)` 产生的是 `(record, record)` 对，`a[1]` 是**用整数下标索引 dict** →
`KeyError: 1`。单网格时被 `else None` 掩盖，所以此前只跑过 `nx = 8` 时看不出来；
**三段梯子一落地、`len(meshes) > 1` 的第一秒**，文档生成器就崩，且崩在 `analyse()`
已经花掉算力之后。

修法：判据改成对**极限载荷序列**做单调性判断（`ladder_monotone(values)`），
并显式区分「单调递减/单调递增/非单调」，方向不同但都单调时仍然算收敛。

### 缺陷 D2（语义级，判据表可被构造性满足）

`docs/…-MESH-LADDER.md §6` 的 `within_10pct_of_ilyushin` 用**逐网格**误差判定，
而锚点验收对象是**网格无关值**。粗网格误差恰好落在 10 % 内、收敛值落在外面（或反之）
时，结论会随网格数任意翻转。且 `ilyushin_sine` 这一支在模块里已被注明
「neglects the twist term … can fall *below* the exact collapse load」，
即它**不是**可靠的上界，拿它做 `< 10 %` 的通过判据在力学上不能成立。

修法：判据表改为对着**收敛值**（`converged_limit()`：有三网格则 Richardson，
否则**明写** "finest mesh nx = N (not extrapolated)"），并把刚性塑性判定改成
**区间准入**：`strip_lower <= p <= yield_line`，越界任一方向都判 `not met` 并写明
是越了上界还是下界（越上界证伪上界或 FE 模型，越下界证伪 FE 模型）。

### 缺陷 D3（可用性，`n/a` 与 `not_run` 混淆）

- 单网格时原文写「3-level mesh convergence … pending meshes」却和 `met/not met`
  同列，读起来像「做了没通过」；现在单网格/有缺值一律 `not_run`。
- 某个 `nx` 上没有任何极限载荷判据触发（ramp 太短/未软化）时，现在按记录里的
  `limit_tight is None` 自动落到 `n/a` 行，绝不格式化成数字。
- `membrane`（围栅板）记录**不进入**验收表：它的极限载荷按构造高于纯弯上界，
  和弯弯曲曲判据不可比。

---

## 2. 判定层的函数与不变量（`scripts/patch_plate_plastic_ladder_figures.py`）

| 函数 | 不变量 |
|---|---|
| `free_ladder(summary)` | 只取 `geometry == "free"`，按 `nx` 升序（= `h` 递减） |
| `ladder_monotone(values)` | 单值、空集、**含 `None`** 一律 `None`（缺值不是收敛证据） |
| `convergence_trend(values)` | 单调时给出方向；非单调返回 `not monotone` |
| `converged_limit(summary)` | 优先 Richardson `p_inf`；否则最细网格值并**标注未外推** |
| `bracket_verdict(limit, bounds)` | 双向越界都判 violated；缺上界时 `withheld` |
| `acceptance_table(summary, records, bounds)` | 每一格由 artifacts 现算；无法回答即 `not_run` |
| `ladder_bounds(summary, records, bounds)` | summary 无 `references` 时回落到原始 per-mesh 记录 |

参考值（`plate_patch_plastic_limit` 现算，DH36 @ −40 ℃，σ_y = 383.8 MPa，
m_p = 38.38 kN·m/m，斑载荷 0.5 × 0.5 m）：

| 量 | 值 [MPa] | 性质 |
|---|---|---|
| 解析首次屈服 `p_y` | 0.8692 | Navier 场 |
| 条带下限 `strip_lower` | 0.8188 | 静力允许**下**界 |
| Ilyushin 双正弦 | 1.5152 | 原文当上界，**含缺失扭转耗散项，§1/D2 已说明不作准入闸门** |
| 屈服线上界 `yield_line` | 1.8422 | 运动允许**上**界（斑载荷机构） |
| Johansen（保留扭转） | 2.3800 | 更松的上界 |

准入区间宽度 `yield_line / strip_lower = 2.25`——区间本身很宽，这是本锚点
「与上界比 10 %」判据不适用的**定量理由**，而不是借口：区间宽 → 用区间准入 + 收敛性
代替「与某一支参考值比 10 %」。

---

## 3. 验证（本次运行）

```
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_ladder_verdicts.py \
    tests/test_plate_patch_plastic_mesh_ladder.py -q
# 34 passed in 63.86 s   (20 new + 14 pre-existing)
```

新增 20 项测试（`tests/test_plate_patch_plastic_ladder_verdicts.py`），全部是纯 python
合成 summary，不跑 FE、不用队列；它们断言的是**不变量**而非数字：

- 单调性：递减/递增都判 `True` 且方向标签不同；非单调判 `False`；单值/空/含 `None` → `None`；
- 区间准入：区间内 `admissible`（并报出在区间内的相对位置）；高于上界/低于下界都 `violated`
  且标明方向；缺上界 `withheld`；
- 收敛值：单网格值必须被标注 `not extrapolated`；有 Richardson 时用 `p_inf` 而不是最细网格值；
- 判据表：**用收敛值判 Ilyushin 10 %**（构造 1.50 MPa 收敛值 + 1.40 MPa 粗网格值，
  只有前者在 10 % 内 → 必须 `met`，若错误地用粗网格值则测试失败）；
- 判据表：收敛值越上界 → `not met (violated (above upper bound))`；
- 单网格 → `not_run`；非单调 → `not met` 且带 `not monotone`；
- 缺 first-yield 证据 → `not_run`；`membrane` 记录不出现在表里。

端到端冒烟（`scratch/smoke_ladder_doc.py`，合成三段梯子 1.1997/1.16/1.14 MPa）：
`analyse()` → 文档 + 3 张图全流程跑通，Richardson 给出 `q = 0.98`、
`p_inf = 1.0785 MPa`，判定表 7 行全部现算。**修 D1 之前此路径在第一行就 `KeyError`。**

---

## 4. 仍未闭合的验收项（honest，不粉饰）

| 锚点验收判据 | 状态 | 依据 |
|---|---|---|
| 弹性→塑性过渡点 vs 解析屈服载荷 < 5 % | **已闭合**（历史实测 nx = 8：+0.79 %） | `docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md §4` |
| 塑性极限载荷 vs Ilyushin/上限分析 < 10 % | **not_run 本轮；且判据本身待重述** | 网格无关值需三段梯子；`ilyushin_sine` 非可靠上界（§1/D2）。本案例提供的是**区间准入 + 收敛值**的判定机制 |
| 3 档网格收敛 | **not_run 本轮** | 三段梯子已递交计算队列：`run_compute` 作业 **cj3810**（3 shard：`--plan-index 0/1/2`，stage_dir 为本工作树），递交后 58 min 仍在排队（`queued`），本单预算内未启动。作业保留在队列，落地后只需 `--analyse` + 生成器即可出数 |
| pytest ≥ 10 绿 | **met** | 34 passed（本案例新增 20 项） |

**不许含糊的地方**：本案例没有给出任何新的 FE 极限载荷数字。上面出现的 1.1997 MPa
是 `…-MESH-LADDER.md` 里 nx = 8 的历史实测值（其 artifact 不在本分支），用于说明判定层
的计算路径；`p_inf = 1.0785 MPa` 是**合成数据**上的方法自检，不是物理结果。

## 5. 复现

```bash
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 0   # nx = 8
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 1   # nx = 12
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 2   # nx = 16
PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --analyse        # 汇总 + Richardson
PYTHONPATH=src python scripts/patch_plate_plastic_ladder_figures.py               # 文档 + 图（判定现算）
PYTHONPATH=src python scratch/smoke_ladder_doc.py                                 # 生成器冒烟（合成数据）
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_ladder_verdicts.py -q
```

## 6. 注意点 / 局限

* 判定层校验的是「文档结论是否由 artifacts 支撑」，**不校验 FE 结果本身**；
  物理精度必须由真实梯子数给出。
* `convergence_trend` 接受「单调递增」也算收敛：位移型单元的极限载荷通常从**上方**
  逼近（粗网格更刚），但围栅板的膜效应会改变方向，故不把方向写死成必须递减。
* 判定表把 `strip_lower`/`yield_line` 当刚性塑性区间；若将来加入更紧的静力允许场，
  只需改 `ladder_bounds()` 的来源，判定语义不变。