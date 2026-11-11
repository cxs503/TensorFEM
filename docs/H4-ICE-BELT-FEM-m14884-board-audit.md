# 交差审计：m14884 H4-ICE-BELT-FEM（冰带局部结构 FEM）→ 判定 L2

审计单 m15020 · 审计员 board-audit · 对象 m14884（TensorFEM，succeeded，PR #653 merged）
审计日期 2026-11-11 · 被审工作树 `/root/tp-agent/work/m14884` · 审计工作树 `/root/tp-agent/work/m15020`

结论先行：**功能实质 = L2（独立功能实现）**。交付是真的：真代码、真调用仓内两个独立求解器、
对 Kirchhoff 解析解 1.08 % / 1.91 %（< 3 %）、重跑**逐字节等价**、28/28 测试复跑通过。
判不到 L3 的唯一硬理由：**没有任何主求解器/平台侧调用方**（`src/tensorfem/__init__.py` 导出 0 行，
`grep -rl --include=*.py ice_belt_pressure_area src benchmarks examples` 只命中模块自身）。
判不到 L4/L5 的第二硬理由：**核心物理量（设计冰压力）没有外部锚**——IACS UR I2 规则表离线不可得，
模块如实把 `criterion_1.rule_table_comparison` 标为 `WITHHELD`，`p_ref = 2.0 MPa` 是
`DEMONSTRATION anchor ... NOT a rule value`。因此"设计压力数值准不准"在本轮**未被验证**，
本轮被验证的是"压面律实现对不对 + 板格响应随离散是否收敛 + 区域划分是否自洽"。

## 1. 取证：被审单声称的三件套逐项核（不是读它的散文，是重跑）

| 声称 | 审计动作 | 结果 |
|---|---|---|
| run_artifacts 文件存在 | `artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json`(35 067 B) + `fig_h4_pressure_area.png`(280 604 B) + `README.md` + `runs/H4-ICE-BELT-FEM-round1.json` | 存在；sha256 pin 于 §5 |
| benchmark ref vs got | 从 artifact 逐字段重取，按**预登记**阈值（物理 3 %、声明互证 10 %、实现一致性 1e-6）复判 17 条断言 | 17/17 PASS（`artifacts/audit_m14884/repro_audit.json`） |
| repro_cmd 可执行 | ① `pytest tests/test_ice_belt_pressure_area.py -q`；② `python -m tensorfem.ice_belt_pressure_area --out …`；③ `python scripts/fig_h4_ice_belt_pressure_area.py` | ① 28 passed / 58.6 s（在**审计工作树 m15020 = main 合并后代码**复跑）；② 44 s，输出与出厂 artifact **sha256 全等**（`8e95bd25…`）；③ 重生成的 png 与前版**逐字节相同**（`git status` 干净） |

独立复现工具（可复用，非一次性脚本）：`scripts/audit_h4_ice_belt_m14884.py`
→ 重跑模块到临时文件 + 递归逐字段比对 + 17 条断言复判 + 缺口扫描，
产出 `artifacts/audit_m14884/repro_audit.json`，退出码 0/1 机器可读。

复现结果（程序化，未手抄）：

```
reproduce: ran=True  field_diff_count=0  byte_identical=True
  shipped_sha256     = 8e95bd251ac728f993f7ecf820351062d2c805855bc73e3f25383ef994f5748f
  regenerated_sha256 = 8e95bd251ac728f993f7ecf820351062d2c805855bc73e3f25383ef994f5748f
```

## 2. 关键数字（全部来自 artifact，逐项对照基准）

| 量 | got | ref / 基准 | 误差 | 3 % 线 | 判定 |
|---|---|---|---|---|---|
| 满板峰值挠度 MITC4 | 1.978 479 63e-4 m | 2.000 000e-4 m（Kirchhoff Navier 双级数） | 1.08 % | ✔ | PASS |
| 满板峰值挠度 Q4-SRI | 1.997 876 84e-4 m | 同上 | 0.106 % | ✔ | PASS |
| 满板 MITC4↔Q4-SRI | — | 两独立离散 | 0.97 % | 10 % 线 ✔ | PASS |
| 冰斑（0.64 m²，四分之一中心斑）MITC4 | 2.038 00e-4 m | Navier 斑级数 | 1.91 % | ✔ | PASS |
| 冰斑 MITC4↔Q4-SRI | — | 两独立离散 | 2.19 % | 10 % 线 ✔ | PASS |
| 网格加密 7→11→15→21 | 5.23 %→2.08 %→1.08 %→0.49 % | Navier | 单调收敛 | ✔ | PASS |
| 压面律四路径最大分歧（marginal 路） | 2.22e-11 | 模块自报 ~1e-11 | — | 1e-6 | PASS |
| 指数拟合 | −0.300 000 000 000 000 2 | −0.3 | 2.2e-16 | 1e-6 | PASS |
| 缩放不变量 p·A^α 散布 | 2.2e-16 | 恒定 | — | 1e-6 | PASS |

**显式回答"是否 < 3 %"**：能对标解析解的 6 个量（3 组 × 2 载荷工况）**全部 < 3 %**
（最大 2.19 %，为 MITC4↔Q4-SRI 互证，非对解析解）。**不能对标的部分**：设计压力
绝对值（p_ref=2.0 MPa DEMO）与冰级压力因子（`DEMO_CLASS_FACTORS`）**无外部基准**，
本轮不作任何精度声明。

## 3. 判级（L0–L5，按功能实质）

| 级 | 判据 | 本单 |
|---|---|---|
| L0 文档 | 报告/文档 | 有（216 行 docs + 报告 #13989），但不作为级别 |
| L1 数学/单元验证 | 单元测试/不变量 | ✔ 28 测试 + 四路径互证 + 不变量 + 求积 |
| **L2 独立功能实现** | 真代码 输入→计算→输出 | ✔ **1353 行真实现**，非 placeholder；`main()` CLI 端到端出 artifact |
| L3 系统集成 | 接入主求解器/平台 | ✘ **无调用方**：非 `__init__` 导出、无 solver/API/CLI 调用；模块反向调用 `ice_belt_collapse.PlateModel` 与 `plate_mindlin` 属"消费"不是"接入" |
| L4 基准验证 | 完整算例+误差分析+可复现 | 数字面已达 L4 质量（解析解 + 误差 + 逐字节复现），但 L3 前置键为空，按方案不越级 |
| L5 科研工程 | L4 + 外部锚（文献/实验） | ✘ 规则表 WITHHELD；仅有解析解锚，无实验/规范值锚 |

**落表**：`tensor_maturity_verdicts(repo='TensorFEM', scope='m14884 H4-ICE-BELT-FEM',
mission_id=15020, level=2, verdict_by='board-audit')`，evidence 键 = impl_files/entry/
run_artifacts/tests/integration_callers/benchmark/repro_cmd/external_anchor，值均为真实指针
（见 §5）。审计后 gate 状态由 `maturity_gate.py` 周期自动裁定（它只裁 succeeded 审计单，
本单 finish 后下一周期生效）。

## 4. 发现（按严重度；均不改变 L2 判定，但决定第 2 轮做什么）

**F1（阻断 L3）无集成调用方。** `src/tensorfem/__init__.py` 对
`ice_belt_pressure_area` 导出 **0 行**；`grep -rl` 在 `src/ benchmarks/ examples/` 仅命中模块自身。
后果：能力不可从任一求解器/平台路径到达，一次性验证脚本与长生能力的差别就在这里。
修法（第 2 轮）：让 `ice_belt_collapse` 的资格门调用本模块的 `panel_cross_validation` +
`check_design_pressure_zones`（一条 import + 一个准入判据），或在 `__init__` 导出并在
ship platform 侧暴露一个只读校验端点。

**F2（阻断 L4/L5）设计压力无外部锚，如实 WITHHELD。** artifact
`criterion_1.rule_table_comparison.status = "WITHHELD"`, `fail_stage = "rule-table ingestion"`，
`law.provenance = "DEMONSTRATION anchor p_ref=2.0 MPa: mechanism check only, NOT a rule value"`，
`withheld_items = ["criterion_1.rule_table_comparison"]`。这是**诚实申报，不是造假**——
审计确认代码里有 `check_pressure_provenance(require_design_data=True)` 会拒绝把演示数据用于设计
（fail-closed）。第 2 轮输入已由被审单自己写清：把 UR I2 表按"版本+条款"作为数据文件注入
（比较器 `compare_with_rule_table` 已就绪，零改码），或由 ship platform 出规则数据端点。

**F3（非阻断，信息量低）Z9 冰级排序 PASS 接近同义反复。** 55/55 对通过，但被比较的
`class_pressures` 由本模块 `DEMO_CLASS_FACTORS`（PC1 1.15 → IC 0.22，人工单调）构造，
外部输入只有 **顺序**（ship platform `/api/v1/polaris/ice-classes`，本次实测可达、12 级 rank 12→1）。
即：该 PASS 证明的是"注入顺序与因子单调性不矛盾"，不是任何压力精度。artifact 的
`class_check.ranks_source` 只写"injected ice-class registry"，**未标注因子为 DEMO**——建议补一个
`factors_source: "DEMONSTRATION"` 字段，避免下游误读。

**F4（非阻断）报告数字与 artifact 一致，无夸大。** 报告称 0.97 %/2.19 % 与 1.08 %/1.91 %，
对应 artifact `0.009708908…`/`0.02185894…`/`0.01076018…`/`0.01912884…`，逐项吻合；
"28 tests" 复跑吻合；`overall_status=PARTIAL_WITHHELD` 与 `withheld_items` 一致。
未发现把 WITHHELD 写成 PASS 的情形。

**F5（非阻断，可追溯性）artifact 未嵌入来源指纹。** `pressure_area_zoned_panel.json` 里没有
生成它的 git sha / 环境版本，只有 `mission`/`round`。本次审计用外部 sha256 pin 补上了
（§5），建议后续 artifact 增加 `generated_by: {git_sha, python, numpy, torch}` 字段。

**F6（说明，非缺陷）出厂件不含 artifacts/runs 不是丢件。** 合并提交 `47944e23` 只带
`src/`+`tests/`+`docs/`+`scripts/`，是 `tp_pusher.py` 推送黑名单
（`_EXCLUDE = ^(artifacts?/|runs/|results-…|REPORT_GATE)`）的既定行为——纯记录区不上 GitHub。
后果是"新克隆无法直接看到 ref-vs-got"，只能重跑；本次审计已验证重跑**逐字节等价**，
因此审计分支把 m14884 的 artifact/run record 原样复制为审计证据副本（sha256 与原单一致），
不新增事实，只恢复可核性。

## 5. 证据指针（verdict.evidence 的值，逐条可查）

- impl_files：`src/tensorfem/ice_belt_pressure_area.py`（1353 行，sha256 `7c473f1c…f2e8`；
  与 m14884 工作树同名文件同哈希）
- entry：`src/tensorfem/ice_belt_pressure_area.py` 的 `main()`（`python -m
  tensorfem.ice_belt_pressure_area --out …` → 打印 `PARTIAL_WITHHELD`）
- tests：`tests/test_ice_belt_pressure_area.py`（360 行，sha256 `562dd332…1030`；m15020 复跑 28/28 PASS / 58.6 s）
- run_artifacts：`artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json`
  （sha256 `8e95bd25…4f5748f`）、`artifacts/audit_m14884/repro_audit.json`
  （sha256 `d8badde7…10227`）、`artifacts/audit_m14884/fig_audit.png`、`runs/H4-ICE-BELT-FEM-round1.json`
- integration_callers：**空**（F1，gate 会因该键为空把 L3+ 降级——这正是本轮要记录的缺口）
- benchmark：见 §2 表（ref/got/err 三元组齐全，`repro_audit.json.claims.rows` 17 条）
- repro_cmd：`PYTHONPATH=src python -m tensorfem.ice_belt_pressure_area --out
  artifacts/H4-ICE-BELT-FEM/pressure_area_zoned_panel.json && PYTHONPATH=src python
  scripts/audit_h4_ice_belt_m14884.py`（已执行，exit 0）
- external_anchor：`WITHHELD`（IACS UR I2 表离线不可得）；可用的替代锚 =
  Timoshenko & Woinowsky-Krieger (1959) §30 Kirchhoff 双级数解析解（本次实际对标）、
  Dvorkin & Bathe (1984) MITC4、Hughes/Cohen/Haroun (1978) Q4-SRI、ISO 19906:2010 压面律形式
- 报告：tensor_reports #13989（含图 `fig_h4_pressure_area.png`）；PR **#653** merged（commit `47944e23`）

## 6. 独立性限制 / 未跑的检查（不冒充通过）

- **未做**：与 IACS UR I2 规范值逐点对照（无规则数据；这是被审单自己的 `fail_stage`，不是审计跳过）。
  因此**不声称**设计压力误差任何数值。
- **未做**：实验/实船标定；无 Cb/L/B 几何（本例是 1.6 m × 1.6 m 板格，非船体）。
- **未做**：>2 载荷工况之外的参数扫描、非线性/塑性段（`elastic_perfectly_plastic` 仅在线性段内用）。
- 复现环境：本机 x86_64，python 3.11，numpy/torch 同工作树；未上 loong64 集群（单算例 44 s，无必要）。
- 独立性：审计者复用了被审代码作为被测对象，但基准（Navier 解析解）与被测 FE 路线代码不相交；
  且审计的比对脚本、阈值、"逐字节等价"判据均为本次独立编写、先钉阈值后看数。

## 7. 处置（disposition）

- **approve at L2**：m14884 的交付真实、可复现、诚实申报 WITHHELD；不构成虚标。
- **request changes（第 2 轮输入，按成本排序）**：
  1. F1 集成：`ice_belt_collapse` 资格门调用本模块（最小一条 import + 一个准入判据），即可升 L3；
  2. F2 规则数据注入：UR I2 表按"版本+条款"入数据文件，`compare_with_rule_table` 零改码接上，即可冲 L5 的物理锚；
  3. F3/F5：artifact 增 `factors_source`、`generated_by` 字段（一行）。
- 触碰账目：本轮不改 m14884 终态、不改其分支；只在审计分支恢复其 artifact 副本以保可核性。