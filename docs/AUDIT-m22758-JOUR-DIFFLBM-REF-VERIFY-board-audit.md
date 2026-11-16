# AUDIT — m22758 `JOUR-DIFFLBM-REF-VERIFY`（board-audit 交差核验）

**被审对象**：m22758（TensorFEM，status=succeeded，task_key=`selfstart-tp-ds31-1-1116-0210`）
**被审提交**：`794447ec`（分支）→ merge `01ad0648` "Merge #1172 (m22758) via merge-master"（PR #1172, merged）
**审计单**：m22811（本文件所在工作树 `/root/tp-agent/work/m22811`，分支 `agent/m22811`）
**审计类型**：交差核验（非重做）。**判据唯一裁判** = `tensor_benchmark_cases` + 本锚 acceptance，
报告文字不构成证据。

---

## ① 取证：三件套是否真存在

| 三件套 | 声称 | 机器核验结果 | 证据 |
|---|---|---|---|
| run_artifacts（运行工件） | 未声称 | **缺失**：m22758 工作树 `runs/` 不存在；`artifacts/` 下唯一新增为无关历史目录 `merge_m3511/`；DOI 核证本身**无任何可执行记录/日志** | `/root/tp-agent/work/m22758/`（无 `runs/`） |
| benchmark ref vs got | 未声称（无基准行） | **N/A**：`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='JOUR-DIFFLBM-REF-VERIFY'` → **0 行**（同链 `JOUR-DIFFLBM-L3-INVERSION` 有 13 行）。本锚为文档型 acceptance，无比值可比 → 该锚"以基准表比对 got"一项**在本锚上不存在**，m22758 亦未编造数字（正确） | `psql -U postgres -d tensorplatform`（见 §⑤ 复现） |
| repro_cmd 可执行 | `PYTHONPATH=src pytest tests/test_lbm_cavity_diff_ref_verify.py tests/test_lbm_cavity_diff.py -q → 12 passed` | **可执行、结果复现**：本工作树（含 merge 01ad0648）重跑 → `12 passed, 1 warning in 26.26s` | `docs/REF-VERIFY-JOUR-DIFFLBM-DOI-AUDIT.md`、`tests/test_lbm_cavity_diff_ref_verify.py` 均存在于 merge 提交内（3 files changed, 167 insertions） |
| 报告六段（tensor_reports id=20464） | 六段齐全 | **齐全**：①案例介绍②方法③计算结果表④无图理由⑤注意点⑥误差判定（显式声明无基准） | `tensor_reports` id=20464, mission_id=22758, scenario_key=`JOUR-DIFFLBM-REF-VERIFY`, status=published |

**结论**：交付物在形式上齐（文档 + 源码注释 + 回归测试 + 报告），但**核心验收内容（DOI 逐条核证）不成立**——见 §② F1。

---

## ② 缺陷清单（按严重度；全部有复现）

### F1 — BLOCKING / HIGH：引用 [3] Guo 的 DOI 不存在，且是"拼接引用"；审计却判"核证通过"

被审源码 `src/tensorfem/lbm_cavity_diff.py:57-59`（merged 01ad0648）声称：

> Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice effects on the forcing term in the
> lattice Boltzmann method.* **Phys. Fluids 14, 2007-2010.** DOI: **10.1063/1.1474820**

Crossref / doi.org 独立解析结果（`artifacts/refverify_m22758/verification_matrix.json`）：

| 解析对象 | 结果 |
|---|---|
| `10.1063/1.1474820`（引用里的 DOI） | **404 — 该 DOI 未注册**（Crossref `works/10.1063/1.1474820` → 404；`doi.org/10.1063/1.1474820` → 404） |
| 题名真正归属 | **10.1103/PhysRevE.65.046308**，*Physical Review E* **65**, 046308 (2002)，作者 Z. Guo, C. Zheng, B. Shi |
| 卷页真正归属 | **10.1063/1.1471914**，*Physics of Fluids* **14**, 2007-2010 (2002)，题名 *An extrapolation method for boundary conditions in lattice Boltzmann method*（同为 Guo/Zheng/Shi） |

即：该条引用把 **PRE 65,046308 的题名/作者/年** 与 **Phys. Fluids 14,2007-2010 的刊/卷/页**
缝在一起，并配了一个**不存在的 DOI**——正是本锚（"题名/作者/年/DOI 一致；不一致者改引"）
设计要抓的缺陷类，而 m22758 判为 **"核证通过"**（`docs/REF-VERIFY-JOUR-DIFFLBM-DOI-AUDIT.md:54-63`；
报告 ③[3]）。**这是审计结论与外部事实直接矛盾**，属虚标，非表述瑕疵。

**加重情节**：`tests/test_lbm_cavity_diff_ref_verify.py:25,62-64` 把**不存在的 DOI**
硬编码为 `canonical` 并断言其存在——返修时若改对，测试反而**报红**；回归门在保护错误事实。

### F2 — MAJOR：审计方法无外部锚，且其"外网不可用"的理由在本生态不成立

报告 ⑤/文档均声明"本机无外网，未做 DOI 在线解析"，全靠"领域常识 + 唯一权威出处"断言。
本审计在**同一生态**内实测：`api.crossref.org` 与 `doi.org` **均可解析**
（Ladd/Ghia 两条 DOI 各返回 200 与完整著录，见 `crossref_cache.json`）。
→ "无外网"不是可成立的方法学豁免；REF-VERIFY 锚的**唯一**可机器核验动作就是外部解析，
跳过它等于没做核证。

### F3 — MAJOR：引用 [3] 在代码里根本没有对应实现（"装饰性引用"）

`src/tensorfem/lbm_cavity_diff.py` **正文（docstring 之外）无任何力项/forcing/Guo 耦合**：
`grep -i 'guo|forcing|force_term'` 在剥离 docstring 后 **0 命中**（`artifacts/refverify_m22758/test_guard_loophole.txt`）。
而审计文档 §① 与报告 ① 均称该模块为"Ladd 移动壁半程反弹 + Ghia 基准 + **Guo 力项**"——
"Guo 力项"这一用途声明**无实现背书**。
（[1] Ladd、[2] Ghia 均有真实代码/数据背书：`_stream()` 的 `2.0*W9[i]/CS2*rho*c_i·u_lid` 壁面修正；
`examples/lbm_cavity_diff_validation.py` 的 `GHIA` 参考表。）

### F4 — MINOR：回归门只做精确子串匹配，可被格式变体绕过

`test_fabricated_ladd_journal_absent` 断言 refs 块不含精确串 `"J. Supercomput. 4, 149"`；
把错引改写成 `"J. Supercomput. 4, 149-168"` / `"J. Supercomput 4, 149"` / `"J. Supercomput. 4:149"`
**均照样通过**（实测见 `test_guard_loophole.txt`）。测试断言的是"字符串存在"，不是"DOI 有效"。

### F5 — INFO（不扣分，记录边界）：本锚无基准行

`tensor_benchmark_cases` 中该锚 **0 行**；m22758 没有、也无法提供 ref-vs-got 数值。
"百页报告不改变级别"在此同样适用：级别由功能实质决定（见 §③），不由文书量决定。

### F6 — INFO：无运行工件

DOI 核证全过程无可执行产物（无脚本、无缓存、无日志），证据链=散文+文本测试 → 不可机检。

---

## ③ 判级（L0–L5，按功能实质）

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 仅文档 | 已超过（有源码注释改引 + 回归测试） |
| **L1 数学/单元验证** | 单元级验证机制真实存在 | **命中**：`tests/test_lbm_cavity_diff_ref_verify.py`（6 用例）真实运行、可复现（12 passed），确实守住一类回退 |
| L2 独立功能 | impl_files 非占位 + entry + run_artifacts | **不达**：无 entry、无任何运行工件（§①）；且测试内容固化了错误 DOI |
| L3 系统集成 | +integration_callers | 不达 |
| L4 基准验证 | +benchmark ref-vs-got +repro_cmd | 不达（本锚无基准行） |
| L5 科研工程 | +external_anchor | 不达——**恰恰相反**：该单唯一的外部锚（DOI 解析）未做，且一条引用与之矛盾 |

**判级 = L1**。理由：交付的机制（引用完整性单元测试 + 源码改引）是真实的 L1 级验证骨架，
但**其内容含一条虚构 DOI**，故仅凭 L1 计，且 verifier 结论为 **request changes（返修）**。

---

## ④ 处置（request changes / 返修清单，机器生成）

由 `scripts/verify_jour_difflbm_refs.py` 生成（`verification_matrix.json:repair_spec`）：

1. **引用 [3] 改引**：DOI 换成 `10.1103/PhysRevE.65.046308`，刊/卷/页改成 *Phys. Rev. E* **65**, 046308 (2002)；
   若确实要引 *Phys. Fluids* **14**, 2007-2010，则题名必须是 *An extrapolation method for boundary
   conditions in lattice Boltzmann method* 且 DOI `10.1063/1.1471914`——**两者不可拼接**。
2. **用途声明二选一**：模块实现 Guo 力项（`F_i = (1-1/(2τ)) w_i (c_i-u)/cs^2 · F` 之类）并加数值自检；
   或把 [3] 降级标注为 `[background]` 并从"用途=离散力项"的表述中删除。
3. **测试改造**：[3] 的 `canonical DOI` 常量改为 `10.1103/PhysRevE.65.046308`；把精确串守卫改为
   正则（如 `r"J\.\s*Supercomput"`）；新增"每条引用必须有代码锚点或被标 `[background]`"的用例。
4. **方法学**：REF-VERIFY 类任务必须留**外部解析工件**（Crossref/doi.org 原始响应 + 缓存），
   并把解析脚本纳入 `scripts/`（本单已提供 `scripts/verify_jour_difflbm_refs.py` 供返修直接复用）。

---

## ⑤ 复现（本审计全部命令与结果）

```
# 0) 被审文件在位（merge 01ad0648 内含）
git log --oneline -1 01ad0648            # Merge #1172 (m22758) via merge-master
git show --stat 01ad0648                 # docs/…DOI-AUDIT.md | src/…lbm_cavity_diff.py | tests/…ref_verify.py

# 1) repro_cmd 可执行（被审单声称 12 passed）
PYTHONPATH=src python3 -m pytest tests/test_lbm_cavity_diff_ref_verify.py tests/test_lbm_cavity_diff.py -q
# -> 12 passed, 1 warning in 26.26s

# 2) 外部锚：DOI 逐条解析（Crossref）
python3 scripts/verify_jour_difflbm_refs.py
# ref    grade            code-backed  notes
# Ladd   RESOLVED_MATCH   True
# Ghia   RESOLVED_MATCH   True
# Guo    DOI_INVALID      False        DOI does not resolve (registrar status 404)
# summary: n_resolved_match=2 n_doi_invalid=1 n_code_backed=2

# 3) 回归门绕过演示
python3 - <<'PY'   # 见 artifacts/refverify_m22758/test_guard_loophole.txt
...
PY

# 4) 基准裁判：本锚无基准行
runuser -u postgres -- psql -d tensorplatform -c \
  "select case_name,metric,tolerance from tensor_benchmark_cases where anchor_slug='JOUR-DIFFLBM-REF-VERIFY';"
# -> (0 rows)

# 5) 独立复核该锚同链基准行存在（对照组，证明查询有效）
# JOUR-DIFFLBM-L3-INVERSION -> 13 rows
```

**产物**：`artifacts/refverify_m22758/{verification_matrix.json, crossref_cache.json,
disambiguation.json, test_guard_loophole.txt, fig_refverify.png}`，
脚本 `scripts/verify_jour_difflbm_refs.py`、`scripts/gen_jour_difflbm_refverify_figure.py`。

---

## ⑥ 未做的检查（不得视为通过）

- 未取得 m22758 的 agent 会话轨迹（`tensor_dev_missions.result` 只有 summary/checks），
  无法判断其"领域常识核证"是否另有未公开来源；本审计只裁**交付物可核验性**。
- 未对 TensorPapers 侧 JOUR-DIFFLBM 论文正文的参考文献做全链核证（跨仓，超出本单 scope）；
  TensorFEM 侧 JOUR-DIFFLBM 链（CASE/AUDIT/benchmark 文档）经 grep **仅**存在这 3 条文献引用。
- 未运行 m22758 未声称的其它测试（如全仓 `pytest`），故不对其回归面背书。
- 未验证 [1] Ladd 在 `_stream()` 中的**符号/系数**与 Ladd 1994 原式的逐项等价（仅核到
  与模块 docstring 公式一致 + 既有 Couette XVAL 测试通过）；该数值正确性属 `GAP-PP-2` 锚，不在本单。

---

## ⑦ 声明夸大部分（逐条对照）

| 声明（m22758） | 事实 | 判定 |
|---|---|---|
| "3 条引用中 1 条错引、2 条核证通过" | 1 条错引 + **1 条 DOI 不存在**；真正"核证通过"只有 2 条中的 1 条（Ladd） | **夸大**（Guo 亦不合格） |
| "无'未核证'残留" | [3] 的 DOI 未注册 → 应为"未核证/待改引" | **夸大** |
| "本机无外网，未做 DOI 在线解析" | 生态内 Crossref/doi.org 均可解析 | **理由不成立** |
| 报告 ⑥ "核证判据四要素一致性…全部达成（3/3）" | 3/3 中 1 条四要素互斥（题名属 PRE、卷页属 Phys. Fluids） | **结论错误** |

**处置**：**request changes**（返修，见 §④）；verdict 落 `tensor_maturity_verdicts`
（repo=TensorFEM, scope=`m22758 JOUR-DIFFLBM-REF-VERIFY`, level=1, verdict_by=board-audit），
由 `maturity_gate` 机器裁定（gate_level≤1 → reopened + 自动开返修单）。
---

## ⑧ 交付记录（本审计单）

- **verdict**：`tensor_maturity_verdicts` id=**914**，repo=TensorFEM，scope=`m22758 JOUR-DIFFLBM-REF-VERIFY`，
  mission_id=22811，level=**1**，verdict_by=`board-audit`，gate_state=`pending`。
  本地 dry-run `maturity_gate.judge(1, evidence, 'TensorFEM', 22811)` → **gate_level=1, reasons=[]**，
  预测动作 **reopened**（≤L1 → 自动开返修单）。
- **report**：`tensor_reports` id=**20484**（mission_id=22811，scenario_key=JOUR-DIFFLBM-REF-VERIFY，
  六段齐全，1 张图内嵌）。
- **可离线复现**：`python3 scripts/verify_jour_difflbm_refs.py`（在线解析并刷新缓存）与
  `--offline`（仅用 `crossref_cache.json`，含已缓存的 404）**判定一致**：
  `n_resolved_match=2, n_doi_invalid=1, n_code_backed=2`。
