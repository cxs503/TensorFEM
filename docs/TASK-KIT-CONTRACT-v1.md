# Task-kit 消费契约 v1（KO 消费协议）

机器可读版本：KO `CONTRACT-TASK-KIT-V1` 与 `CONTRACT-EVIDENCE-GRADE-V1`
（`tools/tk_ontology/seeds/contracts_v1.json`，编译后进 `artifacts/ontology/fem_knowledge_index_v1.json`）。
本文件是同一契约的散文版；两者冲突时以 seed 为准（seed 被 schema 与校验器约束）。

## 1. 适用范围

任何以本仓知识为输入或被派单到本仓开发项的任务（task-kit 装配、验证欠账复核、benchmark 补行、
报告发布）。契约的对象是**产出**而不是流程：agent 内部怎么想不受约束，交付物必须满足下列条款。

## 2. 强制条款（每条带机器可校验判据）

| # | 条款 | 可校验判据 |
|---|---|---|
| C1 | 锚点必须解析为 KO；若不存在，本单创建它而不是写自由文本 | `grep '"id": "<锚点>"' artifacts/ontology/fem_knowledge_index_v1.json` |
| C2 | 结果必须由某个 `CRITERION` KO 裁定（引用其 id），不得临时拍阈值 | KO 的 `has_criterion` 边存在且 `threshold_pct` 非空 |
| C3 | 每个关键量至少挂一个 `EVIDENCE` KO：可重跑命令 + 工件路径 | `fields.run_command` 与 `fields.artifact_path` 非空；`mechanical_grade == REAL` |
| C4 | 必须申报证据分级，且机械分级对审阅者可见；二者不一致要显式说明 | `declared_grade` 与 `mechanical_grade` 同时存在；不一致项出现在 gap report 第 2 节 |
| C5 | 发现的缺陷必须落 `DEFECT` KO 并连到暴露它的 CASE | KO 存在 + `found_in` 边存在 |
| C6 | 全部产出通过本体校验，零 `INVALID` 行 | `python tools/tk_ontology/build_index.py --check` 返回 0 |
| C7 | 索引确定性：同 commit 重编译逐字节相同 | `python -m pytest tests/test_tk_ontology.py -q -k deterministic` |

## 3. 容差口径（tolerance policy）

* `CRITERION-PLATFORM-15PCT`：相对误差 > 15% **不可接受**，不得报 PASS。
* `CRITERION-PLATFORM-3PCT`：报告目标，关键量应 ≤ 3%。
* 落在两者之间（3% < e ≤ 15%）：必须写成「honest partial PASS」并给出原因（基准自身不确定度、
  模型偏差、离散误差分解），**不得写成 PASS**；无基准时必须声明「无基准」及理由，不得含糊。

## 4. 消费协议（agent 侧读取顺序）

1. `CASE` → 读 `benchmark_reference` / `tolerance_text` / `run_command`：确定复现入口与判据出处。
2. 沿 `has_criterion` 取 `CRITERION`：拿到 `threshold_pct` 与 `source`，不要另立阈值。
3. 沿 `has_evidence` 取 `EVIDENCE`：先看 `mechanical_grade`。`WITHHELD`/`STUB` 的工件**不可作为已完成证据引用**，
   只能作为「待补工件」的线索。
4. 沿 `uses_method` 取 `METHOD`：先读 `validity_limits`，再决定该配置是否在适用域内；域外结果必须降级呈报。
5. 反向查 `has_defect`（经 `DEFECT.found_in` 的逆边）：该算例的已知失效模式，避免重复踩坑。
6. 需要判据/义务本身时读 `CONTRACT` → `requires` → `CRITERION`。

## 5. 反例（v1 明确禁止）

* 在报告里写「误差很小」而无可复算工件路径（违反 C3）。
* 用新阈值替换已有算例判据以让结果转绿（违反 C2；`has_criterion` 是 one_to_one，改判据要改 CRITERION 对象本身并留 provenance）。
* 把 `WITHHELD`（工件缺失）叙述成 `REAL`（违反 C4）。
* 只写散文不落 KO，导致下一单必须重读全部文档（违反 C1/C5，也是 50% REAL 覆盖率的成因）。

## 6. 当前符合度（自评，可复算）

* C6/C7：满足——`build_index.py --check` 与 `pytest tests/test_tk_ontology.py`（23 项）在本单内为绿。
* C1–C5：本单自身的产出满足（KG 文档、本契约、编译器、23 项测试均带 provenance 与 sha256 可追溯工件）；
  但**全仓层面不满足**：CASE 具备 REAL 工件的比例仅 50.0%、显式 verdict 仅 24.4%（见
  `artifacts/ontology/fem_knowledge_metrics_v1.md`）。这是本契约下一轮的整改对象，不粉饰。