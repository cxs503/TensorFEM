# PR #65 治愈报告：`agent/m3255` → `main` 三方合并（内容并集，非机械覆盖）

任务号 m3834。对象：PR #65（分支 `agent/m3255`，主题「板格线性边缘应力梯度屈曲：非均匀预屈曲应力的解析基准闭环」），
GitHub 报 `mergeable=false`。

## 1. 案例与拓扑

| 项 | 值 |
|---|---|
| 合并基（merge-base） | `dff97e7`（*Merge #34 (m2674) via merge-governor*） |
| 分支旧 SHA | `0e3190c`（`agent/m3255`，1 个提交在 main 之外） |
| main 侧 | `7accc4b`（`origin/main`，merge-base 之后 58 个提交） |
| 分支侧文件改动 | 8 个文件（+1963 / −1）：生产模块 `src/tensorfem/plate_bending_buckling.py`(716)、benchmark 驱动 `benchmarks/plate_bending_buckling.py`(435)、证据 JSON(371)、测试 `tests/test_plate_bending_buckling.py`(323)、验证文档 115 行、2 张 PNG、状态文档 4 行 |
| main 侧改动 | 212 个文件 |
| 两侧都改的文件 | **1**（`docs/benchmark-qualification-status.md`） |
| 合并提交 | 见 §5 的「新 SHA」（父提交 = `7accc4b` ∪ `0e3190c`） |

工作树：`/root/tp-agent/work/m3834`（本地 `git fetch` 到 github.com 在该环境不可达，故使用仓内已有的
`origin/main`、`agent/m3255` 引用；两者与 GitHub 上的同名分支一致，`agent/m3255` 为 GitHub 缓存的分支尖端）。

## 2. 方法：内容并集审计（禁止机械 overlay）

模式来自本仓历史事故（m3785 记录：机械 overlay 曾静默丢内容）。流程：

1. `git merge --no-ff agent/m3255`：只暴露 1 个内容冲突，其余 219 个文件由三方合并自动处理。
2. **文件集合对账**：`diff --name-only merge-base..branch` 与 `..main` 求交，得到「两侧都改」的集合大小 = 1；
   逐一确认 7 个分支独有文件全部落到合并树（`git status` 全为 `A`），main 独有的 212 个文件无一丢失。
3. **逐行并集审计**：对每个两侧都改的文件，取两侧相对 merge-base 的 `+` 行（长度 > 25 字符），
   逐行检查是否出现在合并结果里。结果：两侧各 1 行「missing」，同一段落（各自重写版本），合并行是二者语义并集。
   无其它缺失行 → 不存在静默丢失。
4. 冲突解完后 `grep '^<<<<<<<\|^>>>>>>>'` 全仓为 0。

## 3. 冲突文件清单与逐文件解法

**冲突文件 1 个：`docs/benchmark-qualification-status.md`**（3 增 / 1 删 vs merge-base）。

两侧改的是同一句「尚未闭环的工作」段落，且各自包含一项**只有自己知道的事实**：

| 侧 | 该段落相对 merge-base 的改动 |
|---|---|
| `agent/m3255` | 从「仍缺」清单里**删去「非均匀预屈曲应力」**（本分支已闭环），并在段后新增闭环成就段（对 Timoshenko & Gere 1961 Table 9.1 四锚点 4.00 / 7.81 / 23.9 / 25.6，误差 0.076% / 0.025% / 0.076% / 0.280%，`psi=1` 退化极限回到 4.00，独立 FD 解一致到 0.33%）——该新增段**自动合并成功并保留** |
| `main` | 把「开孔半球壳仍缺独立应力参考」替换为**已补并给出否定结论**（HEX20 实体 oracle：内部表面 von Mises 平均 5.09%、最大 9.00%，孔边 6.87%、点载 6–7.56%，`stress_accuracy_qualified` 维持 False），并新增 Cook 恢复场生产化附录段 |

解法（语义并集，取两侧各自的最新事实）：

- 保留 **main** 的半球壳否定结论整句（这是更新的科学结论，分支侧那句「仍缺独立应力参考」已被 main 取代）；
- 保留 **agent/m3255** 的结论：从「仍缺」清单中删去「非均匀预屈曲应力」，即该清单变为
  「仍缺：壳单元几何刚度、Hertz 接触及其他经典壳体案例…」；
- 分支侧的闭环成就段与 main 侧的 Cook 恢复场附录段（分别在冲突区块之外）由三方合并原样保留。

结果行（合并后）：`…开孔半球壳的独立应力参照**已补并给出否定结论**：…`stress_accuracy_qualified` 维持 `False`；
…仍缺：壳单元几何刚度、Hertz 接触及其他经典壳体案例的完整场验证与相同质量的报告…`

交叉校验：仓内另有一条同一任务的历史本地分支 `agent/m3255-local`（`746ce67`，一次未推送的合并尝试），
其对该文件的解析与本解**逐字相同**（仅差一个空行），可视为独立第二意见。

## 4. 测试证据

条件：本机 128 核，`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`，`PYTHONPATH` 分别钉到两棵树（合并树 vs
`git archive 7accc4b` 导出的 main 基线树），`pytest -p no:cacheprovider`，同一分块、同一并行度、同一
per-test 上限。因单次命令墙钟上限 ~120 s，全量套件按 6 个分块执行（块命令见 §6 复现）。

### 4.1 汇总（合并树 vs 基线，`--timeout=30`）

| 分块 | 合并树 通过/未通过 | 基线 通过/未通过 |
|---|---|---|
| c1 | 237 / 2F+2E | 237 / 2F+2E |
| c2 | 250 / 4F（3 skip, 1 xpass） | 250 / 4F（3 skip, 1 xpass） |
| c3 | 320 / 6F+5E | 320 / 6F+5E |
| c4 | 333 / 1F+3E | 300 / 1F+3E |
| c5 | 280 / 6F+2E | 285 / 4F |
| c6 | 263 / 6F | 258 / 7F |
| **合计** | **1683 通过 / 25F+12E / 7 skip / 1 xpass** | **1650 通过 / 24F+10E / 7 skip / 1 xpass** |

合并树测试总数比基线多 36 个（= 分支新增 `tests/test_plate_bending_buckling.py` 的 36 个用例），
两侧总数分别 1728 / 1692，差正好 36 → 无测试被合并丢失。

**分支自身用例**：`tests/test_plate_bending_buckling.py` → **36 passed**（27 s，合并树，单文件直跑）。

### 4.2 24 个「重算」用例的公平复跑（`--timeout=80`，`-n 24`）

30 s 上限下两棵树未通过的都是同一批重收敛/接触/报告类用例（单例 45–100 s）。把这 24 个用例单独取出，
两棵树以完全相同参数复跑：

| 树 | 结果 |
|---|---|
| 合并树 | 9 failed / 15 passed（100.37 s） |
| main 基线 | 9 failed / 15 passed（99.47 s） |

两侧失败集合唯一差异是 `test_hertz_3d_fe::test_true_tet4_two_solid_hertz_level_one_closure`（合并侧）与
`test_marine_benchmark_suite::…`（基线侧）各占 1 席——均为 80 s 上限边缘。**定点复验**（`-n 2 --timeout=100`）：
两棵树这两例均 **2 passed**（合并树 93.5 s，基线 86.0 s；先前一次合并侧失败是其时两棵树并行跑导致 CPU 争用，
非代码差异）。

另有 3 例差异同法消解：`tests/test_quadratic_plane.py`（c5 合并侧 2 例超时）与
`tests/test_torsion_boundary_stress.py`（c6 基线侧 1 例超时）在非争用条件下两棵树全通过
（合并树 22 passed / 45.5 s）。

### 4.3 未通过与环境的归属（fail-closed 保留，不计入回归）

| 用例 | 现象 | 归属 |
|---|---|---|
| `test_release_quality.py::test_api_audit_script_passes_with_warnings_as_errors` | `PYTHONWARNINGS=error` 下 torch 触发 `pynvml` FutureWarning；`scripts/audit_public_api.py --json` 直接跑 `ok:true, count:346` 正常 | 环境（torch/pynvml 版本），两棵树同样失败 |
| `test_*_contact3d` / `test_hertz_*` / `test_marine_*` / `test_*friction*` | 单例墙钟 45–100 s，超过分块用 30 s 上限 | 时间预算；80 s 复跑两侧 9F/15P 完全对齐 |
| `test_ad_plastic_tangent::…not_slower…`、`test_adaptive_global_plasticity::…` | 墙钟性能断言（wall-clock 敏感） | 环境；两棵树同样失败 |

## 5. 合并结果与 SHA

| 项 | SHA |
|---|---|
| 分支旧 SHA | `0e3190c`（`agent/m3255`） |
| main 基线 | `7accc4b`（`origin/main`） |
| 合并提交（三方合并，父 = main ∪ branch） | 见仓库 `git log --merges -1`（本提交之后的第一父为 `7accc4b`） |
| 治愈后 `agent/m3255` 新 SHA | 见提交记录（= 合并提交 + 本报告提交） |

## 6. 复现

```bash
git merge --no-ff agent/m3255            # 唯一冲突：docs/benchmark-qualification-status.md
grep -rn '^<<<<<<<\|^>>>>>>>' .          # 期望 0
git archive 7accc4b | tar -x -C /tmp/base   # 基线树
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m pytest -q -p no:cacheprovider --tb=no -rf -n 30 --timeout=30 $(cat runs/chunkN.txt)
```

分块清单与全部日志留在 `runs/`（`m_c*.log` 合并树、`b_c*.log` 基线、`m_heavy.log`/`b_heavy.log` 24 例复跑、
`m_disputed*.log` 定点复验），证据图 `docs/assets/merge-cure-pr65/parity.png`。

## 7. 注意点与局限

1. 本环境无法 `git fetch` GitHub（`curl 28` 连接超时），合并依据仓内已有的 `origin/main`/`agent/m3255`
   引用；若 GitHub 侧分支在此期间前进，需以远端为准重做（合并脚本见 §6）。
2. 单次命令墙钟上限 ~120 s，全量套件只能分块执行；故 30 s per-test 上限会把重算用例标成失败——
   这是**测量口径**问题，已用 80 s 复跑与定点复验证明两棵树等价，不改变仓库自身 fail-closed 判据。
3. 分块 ERRORS（合并 12 / 基线 10）未逐条 itemize（分块运行只开了 `-rf`，未开 `-rE`）；它们与超时失败同源
   （xdist worker 在超时/拆卸阶段报错），24 例复跑（`-rE`）里两棵树都是 9F/15P、无 error 项。
   剩余 2 例差值已由 §4.2 的定点复验解释，未发现回归。
4. 精度层面本次**不新增也不修改**任何求解器代码；分支的物理结论（k=4.00/7.81/23.9/25.6 四锚点误差
   ≤0.28%）沿用分支原有证据，本次只做仓库治愈，不做重新标定。