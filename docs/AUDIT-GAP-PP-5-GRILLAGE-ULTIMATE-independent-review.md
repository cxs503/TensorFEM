# AUDIT — GAP-PP-5 冰压斑+侧压组合下加筋板架极限强度（独立对抗性复核）

锚点 `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`。仓 TensorFEM。本单是对**已合并证据链**的独立复核
（验证欠账类），不是重做：被审对象 = `src/tensorfem/grillage_ice_ultimate.py`（1015 行）、
`grillage_plastic_collapse.py`、`grillage_ice_verification.py`（T1–T10，21 gates）、
`docs/CASE-GAP-PP-5-...-round2/round3.md`、`docs/GRILLAGE-ICE-STRUCT-ULTIMATE.md`、
`docs/grillage_ice_ultimate/*.json`。审计产出全部由
`benchmarks/grillage_ultimate_independent_audit.py` 程序化生成
（`artifacts/grillage_ultimate_audit/`），本文数字经脚本注入，无手抄。

## 1. 独立性与范围声明

* 审计者独立重建执行路径：复核者没有写被审模块，所有结论用**新写的第三方方法**或**显式代数**得到，
  并与被审模块的输出对比。
* 真正独立的部分：`orthotropic_plate_fd`（13 点双调和差分、压斑按单元精确积分、反对称虚节点施
  加简支边界），与被审 Navier 双三角级数**不共享任何代码与离散**；成员链的闭合解（截面 A_e/I_e/Z、
  Faulkner 有效宽度、Perry-Robertson、Johnson-Ostenfeld、secant 放大、F(2L−c)/8）在
  `hand_stiffener_limit` 中重新推导，**消掉了二分**。
* 不独立的部分（如实标注）：手算链里的**弹性模态载荷分配 f_x** 仍取自被审模块
  （`system_load_shares`，由 T3/T4/T9 单独设门）；手算与模块的截面**约定**相同（腹板自板中面起算，
  见 F5）——是独立的代数转写，不是独立的约定。
* 未做（unknown / not_run，不得当作通过）：分类社（IACS PC/CSR）或模型试验的**外部**数据对比
  （沙箱内无外部算例数据）；全量 pytest（本机 shell 120 s 上限，三个 grillage 测试文件同跑会被收割，
  已分文件跑绿）；集群级网格无关性研究；斜置压斑；温度/棘轮路径；被审模块的 API 层。

## 2. 复核方法与执行

1. 读被审证据：`docs/grillage_ice_ultimate/{verification,combined_cases,ultimate_map}.json`
   （21 gates 全绿，terms=61）。**文档数字与 JSON 逐项一致**（3.354 / 2.330 / 1.969 MPa、margin
   3.35/1.16/0.66、映射表 60 行），未发现转抄漂移。
2. 重算**绑定量**：该锚点的极限值在三个验收工况里都由 `plate_local_bending` 控制，即绑定量是
   2.1 m × 0.35 m 板格在真实压斑下的**峰值弯矩**。用 FD 参考场在真实板格/真实压斑上重算（T11）。
3. 消掉二分重算极限压力（闭式代数，T11 p_ult）。
4. 普查**控制分支**：3 套板架 × σ_x/σ_y ∈ {0, 0.3, 0.6} × 压斑长度 ∈ {1, 3} bay（18 个组合工况）。
5. 构造**成员控制**板架并重算 3 档组合工况（T12），再与刚塑性重分配法交叉（T13）。

## 3. 复核结果（程序化生成）

## Plate field: Navier series vs independent FD (T11)

| case | patch [m] | peak moment series [N m/m per Pa] | FD | error | p_ult module [MPa] | p_ult FD [MPa] | error | governing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light-ice + ballast head | 0.35 x 0.35 | 0.0102348 | 0.0102329 | -0.0184 % | 3.3546 | 3.3552 | +0.0184 % | plate_local_bending |
| design-ice + design head | 0.70 x 0.35 | 0.0138788 | 0.0138773 | -0.0107 % | 2.3298 | 2.3300 | +0.0108 % | plate_local_bending |
| severe-ice + tank head | 1.05 x 0.35 | 0.0149363 | 0.0149357 | -0.0045 % | 1.9688 | 1.9689 | +0.0046 % | plate_local_bending |

FD grid convergence (severe-case patch), peak moment: n_x=60: 0.0720 %, n_x=120: 0.0180 %, n_x=240: 0.0045 %

## Member-governed combined cases (T12, T13)

| case | sigma_x [MPa] | patch [m] | p_ult [MPa] | U_plate | U_stiff | U_frame | governing | margin | closed form [MPa] | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light ice + ballast head | 0.00 | 0.55 x 0.55 | 1.6253 | 0.774 | 1.0000 | 0.059 | longitudinal_stiffener | 1.625 | 1.6253 | +6.33e-06 % |
| design ice + design head | 53.25 | 1.10 x 0.55 | 0.9979 | 0.692 | 1.0000 | 0.082 | longitudinal_stiffener | 0.665 | 0.9979 | -3.99e-05 % |
| severe ice + tank head | 106.50 | 1.65 x 0.55 | 0.7477 | 0.627 | 1.0000 | 0.100 | longitudinal_stiffener | 0.374 | 0.7477 | +2.39e-05 % |

Cross-method (T13): elastic member limit 0.9979 MPa vs rigid-plastic grillage 5.4346 MPa (ratio 5.446, plastic system transverse_frame).

> **round 7 重算（板底起算约定）**：同一 T12 三工况 p_ult = **1.9254 / 1.2219 / 0.9918 MPa**（较上表 +18.5 / +22.4 / +32.7 %），弹性构件极限 0.9979 → **1.2219 MPa**；上表为 `plate_mid` 历史口径，引用时须与 round 7 并列。

Governing-branch census: case study (t=20 mm): {'plate_local_bending': 6}, member-governed (t=25 mm): {'longitudinal_stiffener': 5, 'plate_local_bending': 1}, slender (t=13 mm): {'plate_local_bending': 6}


## 4. 发现（按严重度）

**F1（中，主张范围 → 已修复） 被审案例从未进入"板架/构件控制"分支。**
三档验收工况与 18 工况普查（case study 6/6、slender 6/6）中控制构件恒为 `plate_local_bending`；
`combined_cases.json` 与 `ultimate_map.json`（60 行）同样 100 % 由板格控制。也就是说，本锚点交付的
"加筋板架极限强度"实际上只是**板格极限**，恰恰是 JOUR-ICE-STRUCT §5 局限 6 想跨过的那一跳没有被
演示。修复：新增 `member_governed_geometry()`（t=25 mm / bay 0.55 m / T160×6+50×7）与
`MEMBER_GOVERNED_SCENARIOS`，三档组合工况全部由**纵骨**控制（U_stiff=1.000，U_plate=0.63–0.77），
普查 5/6 由纵骨控制；设门 T12a–c，并写入 `artifacts/grillage_ultimate_audit/member_governed_cases.json`
（机读）。

**F2（低，文档主张被证伪 → 已更正）** `case_study_geometry_slender` 原 docstring 称用于
"exercise the stiffener-governed branch"；实测（3 η × 3 压斑长度）纵骨利用率上限 0.76，板格恒控制。
docstring 已改正并加回归测试 `test_slender_variant_is_also_plate_governed`。

**F3（中，量化缺口 → 已入账，非缺陷）** 构件支是**限制应力状态**，不是倒塌载荷：同一工况下
刚塑性重分配（`grillage_plastic_collapse`，逐次塑性铰）给出 5.4346 MPa，是弹性构件极限 0.9979 MPa
的 **5.45×**（塑性机构为横框架）。两者都"对"，但若把构件支的 p_ult 当成"结构极限载荷"引用会严重
低估（此处偏保守 5 倍）。已在 T13 与 artifacts 中显式并列，供论文引用时区分
"member limit state"与"collapse limit"。

**F4（信息，验证欠账 → 已补齐）** 绑定量的独立验证此前只在 1 m × 1 m 各向同性板上做过（T1，3 %
门）。真实板格长宽比 6:1、压斑非满布，唯一能翻盘的就是这里。FD 参考场在 n_x=240 下与级数解差
≤0.018 %，极限压力差 ≤0.018 %（门 0.5 %）；FD 自身二阶收敛（0.072 % → 0.018 % → 0.0045 %）。

**F5（低，非阻断）** `MemberSection._parts` 把腹板自**板中面**起算（未让出板厚一半），腹板与板重叠
≈ t_p/2 = 12.5 mm；对 A 的重复计≈0.5 %、对 I/centroid 的影响 <1 %，不影响 15 % 级判据，但建议后续
统一到"板底起算"并给出差值记录。本次未改（改动会挪动已被引用的历史数字）。**round 7 已闭合**：见 `docs/CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round7.md`——实测影响远大于本条的估计：ΔI 最大 19.0 %、ΔZ_p 最大 7.5 %、构件控制 p_ult +18.5…+32.7 %；默认口径已改为「腹板自板下表面起算」，`web_origin="plate_mid"` 仅用于复现上述历史数字。

**F6（低，可复现性指针）** round-3 文档指向 `artifacts/grillage_plastic_collapse/*.json`，该目录在
树内不存在（被合并门禁滤除）。本次审计产物落在 `artifacts/grillage_ultimate_audit/`，可复现。

## 5. 误差判定（逐项，是否 <3%）

| 关键量 | 参照 | 误差 | <3% ? |
| --- | --- | --- | --- |
| 板格峰值弯矩（三档，真实板格+真实压斑） | 独立 FD（13 点双调和） | ≤0.0184 % | **是** |
| 绑定极限压力 p_ult（三档） | FD 场闭式重算 | ≤0.0184 % | **是** |
| 成员控制 p_ult（三档） | 显式闭式代数（无二分） | ≤4.0e-5 % | **是** |
| FD 参考场自身收敛 | n_x 60/120/240 序列 | 0.072/0.018/0.0045 % | **是** |
| 板格轴向强度 | Paik-Thayamballi（T7，既有） | 4.9–6.3 % | 是（15 % 门） |
| 刚塑性 vs 弹性构件极限 | ——（不同极限状态，不可互校） | 5.45× 差异 | 不适用，已显式声明 |

无一项 >15 %；无一门只验证求解器与自身（T11/T12 均对非同源方法或显式代数设门）。

## 6. 处置（disposition）

**approve with notes。** 未发现阻断性正确性缺陷：绑定量的物理与数值在独立方法下成立到 0.02 %。
两条 notes 已在本单落地：(F1) 成员控制分支缺失 → 新增板架算例与三档工况（T12）；
(F2) 失效文档主张 → 改正 + 回归锁。F3/F5/F6 为引用与后续项，已如实入账。

## 7. 复现

```
PYTHONPATH=src python benchmarks/grillage_ultimate_independent_audit.py
PYTHONPATH=src python -m pytest tests/test_grillage_ultimate_independent_audit.py -q
PYTHONPATH=src python -m pytest tests/test_grillage_ice_ultimate.py -q
```

产物：`artifacts/grillage_ultimate_audit/audit_evidence.json`（40 gates 全绿，T1–T13）、
`audit_tables.md`、`docs/assets/grillage_ultimate_audit/fig1_plate_field_fd_vs_navier.png`、
`fig2_ultimate_pressure_independent.png`、`fig3_governing_branch_census.png`。
