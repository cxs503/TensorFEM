# 智能体导向知识本体 v1 —— FEM 方法轴试点（schema + 关系词表 + 编译器）

锚点：`FEAT-TK-ONTOLOGY`（智能体导向知识体系组织学：双轴×六形态×关系图）
仓：`TensorFEM`（方法轴 = FEM 试点位）；产物：`tools/tk_ontology/`、`artifacts/ontology/`、`tests/test_tk_ontology.py`
本文与 `docs/TASK-KIT-CONTRACT-v1.md` 一起，构成该锚「本体 schema v1 + task-kit 契约成文」的可执行部分。

## 0. 一句话

本仓的知识条目不再以「文档」为单位、而以 **知识对象（KO）** 为单位存在：
一个 KO = 六形态之一（是什么） × 双轴标签（关于什么） × 有类型的边（能推出什么），
由 `tools/tk_ontology/build_index.py` 从仓内既有证据**机械派生**，可被 agent 直接检索与校验，无需阅读散文。

## 1. 为什么是「六形态」而不是「文档分类」

文档分类回答「这份文件属于哪个目录」，那是给人读的。agent 调用知识时问的是**决策问题**，
每个形态恰好对应一个决策问题；形态一旦缺位，agent 就只能靠猜：

| 形态 | 回答的决策问题 | 必备字段（schema v1） |
|---|---|---|
| `CONTRACT` 契约 | 我必须满足什么、按什么口径报？ | `scope` `obligations[]` `tolerance_policy` `consumer` |
| `METHOD` 方法 | 这个方案适用于什么、什么时候失效？ | `name` `family` `applicability` `validity_limits[]` `implementation_paths[]` `references[]` |
| `CASE` 算例 | 用什么物理配置/参数复现？ | `code` `tier` `benchmark_reference` `tolerance_text` `run_command` `artifact_paths[]` `verdict` |
| `CRITERION` 判据 | 数值到多少才算过？ | `metric` `comparator` `threshold_pct` `unit` `source` |
| `EVIDENCE` 证据 | 有没有可离线复算的数字工件？ | `case_id` `artifact_path` `declared_grade` `mechanical_grade` `grade_basis` `sha256` `n_numeric_leaf` |
| `DEFECT` 缺陷 | 已知在哪里翻车、被推翻的说法是什么？ | `symptom` `severity` `scope` `refuted_claim` `evidence_path` |

两条设计纪律写进了 schema（`tools/tk_ontology/schema_v1.json`，draft 2020-12，已用 `check_schema` 验证）：

1. **未确立的字段写 `null`，不写「省略」**，并在 `field_confidence` 里标 `measured|parsed|declared|unknown`。
   消费者永远不需要在「不存在」与「没查」之间猜。
2. **六形态的 `fields` 各自 `additionalProperties: false`**。给 CASE 塞一个 `grade` 字段会被 schema 直接拒绝——
   形态之间的边界靠 schema 守，不靠文档纪律守。

## 2. 双轴：方法轴 × 领域轴

* 方法轴（`method_axis`）：`FEM / LBM / FVM / BEM / SPH / DEM / DG / CAD / ANALYTIC / EXPERIMENT`。
  本仓对象恒带 `FEM`；跨方法引用（例如 DG 算例池条目）保留第二标签而不是另开一套命名。
* 领域轴（`domain_axis`）：船海学科优先（`resistance / propulsion / seakeeping / manoeuvring / stability /
  structure / vibration_noise / green / ice`），其余落 `contact / heat_transfer / general_mechanics`。
  领域轴由图内关键词表决定（`DOMAIN_KEYWORDS`，最specific优先），因此**零硬编码领域词条进代码语义**：
  关键词表就是配置，换域只改表。

双轴的意义：同一个 `CASE` 既能在方法轴上被「同类方法的第 N 级」比较，也能在领域轴上被
「同一条船/同一物理问题」调用；单轴组织必然要求其中一种检索退化成全文搜索。

## 3. 关系词表 v1（闭合，`tools/tk_ontology/relations_v1.json`）

只有表内的 `(主语形态, 动词, 宾语形态)` 组合合法；边只写在主语上，反义词由编译器确定性物化。
`cardinality` 是硬约束而非注释。

| 动词 | 主语 → 宾语 | 反义 | 基数 | 语义 |
|---|---|---|---|---|
| `has_criterion` | CASE → CRITERION | `criterion_of` | one_to_one | 该算例由这一条数值判据裁定 |
| `has_evidence` | CASE → EVIDENCE | `evidence_of` | one_to_many | 该算例产出的工件 |
| `uses_method` | CASE → METHOD | `method_used_in` | one_to_many | 算例检验的方法 |
| `evaluated_against` | EVIDENCE → CRITERION | `evaluated_by` | one_to_one | 工件按哪条判据比对 |
| `found_in` | DEFECT → CASE | `has_defect` | one_to_many | 缺陷在哪暴露 |
| `violates` | DEFECT → CRITERION | `violated_by` | one_to_many | 缺陷是判据被破的具体实例 |
| `limits` | DEFECT → METHOD | `limited_by` | one_to_many | 缺陷圈定方法失效域 |
| `requires` | CONTRACT → CRITERION | `required_by` | one_to_many | 契约强制按该判据申报 |
| `supersedes` | EVIDENCE → EVIDENCE | `superseded_by` | one_to_many | 工件版本链 |
| `refines` | METHOD → METHOD | `refined_by` | one_to_many | 方法细化关系 |

`has_criterion` 定为 `one_to_one` 是刻意的：一个算例挂两条阈值后，「到底哪条算过」就不可判定了，
必须靠 `CONTRACT` 指定主判据、其余降级为附加指标。校验器对该基数做了 fan-out 检测（见测试）。

## 4. 编译管道（确定性，无时间戳，全部集合有序）

```
docs/CASE-*.md ──解析(锚点/判据/工件/verdict/证据分级)──┐
benchmarks/**/*evidence*.json ──未被引用者──┐           ├─► 按锚点合并 (一个算例多篇文档 → 一个 KO)
docs/*AUDIT|VERDICT|DEFECT|round2/3*.md ──┴───────────┤
tools/tk_ontology/seeds/{fem_methods,contracts}_v1.json ─┘
                                    │
                    关系装配(声明边 → 物化反义边)
                                    ▼
        artifacts/ontology/fem_knowledge_index_v1.json / _metrics_v1.{json,md} / _gap_report_v1.md / _graph_v1.mmd
```

复现与校验：

```bash
python tools/tk_ontology/build_index.py            # 编译 + 校验 + 落盘
python tools/tk_ontology/build_index.py --check    # 只编译 + 校验（CI 用）
python -m pytest tests/test_tk_ontology.py -q      # 23 项：schema/词表/确定性/负面用例/评分规则
```

三条关键规则：

1. **证据分级的「机械口径」与「申报口径」并存**。`declared_grade` 抄文档自述，`mechanical_grade`
   由磁盘上工件判定：文件存在且 JSON 含数值叶 → `REAL`；含占位字面量（值恰为 `TODO/placeholder/...`）或无任何数值 → `STUB`；
   路径解析不到/不可解析 → `WITHHELD`。二者不一致即「可能的过度声明」，进 gap report 第 2 节。
   `MOCK` 机械口径永不给出——硬编码只能靠读生产者证明，这是人工判断的保留地。
2. **引用解析规则显式化**。仓内文档常写相对脚本目录的文件名（`evidence.json`），因此裸名先按文件名在仓内检索：
   唯一命中 → `resolved_by_name:<path>`；同名多个 → `ambiguous_relative_reference:<n>` 保持 `WITHHELD`。
   **不猜**：多义引用按证据不足处理，这正是 v1 想要的诚实。
3. **合并不改名**。同一锚点的多篇文档（正例 + round2 + 审计）合并成一个 KO，`provenance.path` 是主文档、
   `provenance.supporting_paths` 保留其余；id 后缀逐字保留仓库已有的锚点拼写（含小写），本体不重命名人已引用的对象。

## 5. 现状数字（全部由 `artifacts/ontology/fem_knowledge_metrics_v1.md` 程序化生成，勿手抄）

见 `artifacts/ontology/fem_knowledge_metrics_v1.md`（对象/边计数、分级直方图、双轴直方图）与
`artifacts/ontology/fem_knowledge_gap_report_v1.md`（缺口明细）。当前快照的关键读数：

* 383 个 KO：CASE 82 / CRITERION 79 / EVIDENCE 140 / DEFECT 71 / METHOD 9 / CONTRACT 2；声明边 417 条（含反义物化边共 834 条）。
* **CASE 具备可复算 REAL 工件：41/82 = 50.0%**；有机器可读判据 93.9%；方法归属 98.8%；显式 PASS/FAIL 仅 24.4%。
* 申报分级与机械分级一致率 3/3（样本量极小：只有 3 篇文档申报了分级 → 这一项**不足以支撑任何结论**）。

这些数字是本体的**审计输出**，不是本体的成绩单：50% 的 REAL 覆盖率与 24% 的 verdict 覆盖率说明
本仓近半算例的证据只存在于散文里，机器无法调用——这正是 v1 存在的理由。

## 6. 局限（未做的部分，如实列出）

1. **LBM 方法轴全量重组**不在本仓范围（`TensorLBM`/`TensorPapers`）。本试点只交付 schema + 一套可直接复用的编译器与词表。
2. **船海领域轴试点**同理（`ship-performance-platform`）；本仓的 `ice/structure/vibration_noise` 标签是该轴的子集验证。
3. `supersedes` / `limits` / `evaluated_against` 三个动词**已入词表但当前边数为 0**：版本链需工件内显式 `round` 元数据，
   失效域需人工判断，机械派生会引入猜测，故留空并在此声明。
4. `DEFECT.severity` 大量为 `UNKNOWN`（审计文档普遍不标 P0–P3）。这是文档侧的缺口，不是解析失败。
5. 机械分级的上界性质：`REAL` 只证明「有可复算数值工件」，不证明「数值由算法算出」。证明后者仍是评审的活。

## 7. 下游

* 消费协议与强制项：`docs/TASK-KIT-CONTRACT-v1.md`（`CONTRACT-TASK-KIT-V1` / `CONTRACT-EVIDENCE-GRADE-V1` 两个 KO 的散文版）。
* 建议后续派单：`TensorLBM` 侧「LBM 方法轴全量重组（复用本仓 compiler + 词表）」、
  `ship-performance-platform` 侧「船海领域轴试点（阻力/耐波首两域）」。