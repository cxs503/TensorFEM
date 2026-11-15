# GAP-PP-5 round 5 — 成员需求载荷路径包络：均化份额的两个位置盲假设被推翻并替换

锚点：`GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`（加筋板架在冰压斑 + 侧压组合下的极限强度）
前序：round 1 半解析板架层 → round 2 板格应力→极限映射 → round 3 偏心压斑/偶数模态 →
round 4 独立离散梁格（证明「斑压在框架线上时框架需求被低估 2.08×」，并在 §3.2 末尾建议新条目
`GAP-PP-5-ICE-STRUCT-GRILLAGE-FRAME-DEMAND`）。
本轮**直接实现那条建议**（不改 slug，属本锚点的实质增量）：把成员需求从「均化份额 + 两个位置盲
启发式」换成**位置感知的载荷路径分支**，并把 round 4 的离散梁格当作独立下界校核。

## 1. 案例与参数

与 round 2–4 同一标定板架（`case_study_geometry()`）：板架 8.40 m × 2.80 m；板 t = 20 mm、
EH36 σ_y = 355 MPa、E = 206 GPa、ν = 0.3；纵骨 Tee 250×10+80×12 @ b = 0.35 m；
横框架 Tee 550×10+120×14 @ a = 2.10 m（带板 0.84 m，M_p = 0.905 MN·m）；
冰压斑 1.05 m × 0.35 m（w 按 0.35 / 0.70 / 1.05 扫描），p_ice 扫描值 6 MPa（放大需求差）。

**新增拓扑维度**：`n_spans` 决定压斑与框架线的相对位置 —— `n_spans = 4` 时斑中心正好压在框架
线上（round 2–4 的算例），`n_spans = 3` 时斑中心落在**跨中**。这是本轮找到新缺陷的关键自由度。

## 2. 方法

新增 `src/tensorfem/grillage_member_envelope.py`：

| 分支 | 载荷路径 | 实现 |
| --- | --- | --- |
| B1 纵骨路径 | 板条→纵骨（tributary 重叠加权）→纵骨简支跨框架→端反力 F/2 传给两侧框架 | `tributary_weights` + `patch_moment_centre` + `ss_beam_moment` |
| B2 框架路径 | 板→框架（x 向 tributary 重叠加权）→框架简支跨纵桁 | 同上 |
| C 系统路径 | 连续离散梁格（round 4 的 `grillage_beam_grid`，已对经典梁闭合解校核到 0.000 %） | `_unit_grid`（单位压力求解 + 线性放大 + LRU 缓存） |

每个分支都**独自承载整个压斑**（不分裂、不重复计数）：B1/B2 是对真实两向板作用的下/上界理想化，
包络取逐成员 `max(B1, B2, C)`；C 用于证明包络不比显式结构模型更松。
成员利用率定义与父模块一致（纵骨：`(σ_axial + β_m M/Z_e)/σ_lim`，Johnson–Ostenfeld/Paik 限值；
框架：`M/M_p`，即与父模块 `frame_plastic_collapse_force` 同源的首塑性铰判据）。

接入：`grillage_utilisations(..., member_model="smeared"|"panel"|"system"|"envelope")`，
`ultimate_ice_pressure` / `panel_stress_ultimate_map` 同步透传；**默认仍是 `smeared`**（旧数值不变，
向后兼容），新分支只增不改。

## 3. 结果（全部数字由 `benchmarks/grillage_member_envelope_round5.py` 程序化注入，见
`docs/grillage_member_envelope_tables.md`）

### 3.1 两条位置盲假设都被推翻

1. `n_overlap = round(l_y/b)+1`：判给 2 根纵骨，但宽 0.35 m 的斑正好落在一条纵骨线上，
   几何上只属于该纵骨的 tributary（权重 1.0）。
2. `per_frame = 0.5·F·f_y`：「恰好两条框架分摊」只对斑落在跨中成立；斑压在框架线上时
   该框架单独吃掉全部斑载（round 4：2.08×）。
3. **本轮新增、且更严重**：均化层对**均布侧压**同样除以 2，而静水压把相邻两个板格同时加载，
   框架的公称 tributary 就是整个 a = 2.10 m。闭式 `p·a·L²/8`（0.1 MPa → 205.8 kN·m）与离散梁格
   （216.5 kN·m，+5.2 %）一致，而均化层给出的框架利用率只有系统分支的 **2.9 %**（即低估约 35×）；
   再加上它用**固端**塑性倒塌力（`F_u = 8M_p/(L(1−c/2L))`，c = L 时 16M_p/L）去比一个**单板格**
   的力，需求与能力口径不一致。
4. 冰压斑拓扑（6 个）：均化/系统 的框架利用率比 = 0.35–0.48（跨中斑）、0.24（框架线斑）；
   纵骨侧均化在跨中斑时也低 2.5×。

### 3.2 组合工况（锚点验收的三档）在三种成员模型下的差别

见 `docs/grillage_member_envelope_tables.md` §「Acceptance combined cases」：
三档的 `p_ult` 在 smeared / system 下**完全相同**（均由板格局部弯曲控制，2.42 / 2.06 MPa），
但框架利用率从 0.094→0.615（设计档，6.5×）、0.128→0.955（严酷档，7.4×）——
**headline 极限压力不变，报告的框架安全裕度被低估 6.5–7.4 倍**，且严酷档的框架已到 95.5 %
（「板格控制」这一结论在此压力下只剩 4.5 % 余量，属脆弱结论）。
`panel`（单板格保守理想化）给出 0.77 / 0.30 MPa，由纵骨控制。

### 3.3 冰区带载（belt）储备与经典对照

* 冰区带载时纵骨：面板简支 192.9 kN·m vs 系统 119.5 kN·m → 1.61×（单板格理想化的真实保守度；
  只加载单个板格时该储备可达 ~7×，这是「包络 vs 系统」差距的来源，已在文档写明适用条件）。
* 板格极限 vs 经典闭式：模型 2.3768 MPa vs 简支板条首铰 `8M_p/b²` = 2.3184 MPa → **+2.52 %**；
  并落在 [首铰 2.318, 固支板条塑性倒塌 `16M_p/b²` = 4.637] MPa 的经典区间内。
  注：**分类社规范文本在本环境不可获取**（`benchmarks/anchors/iacs-csr-w22.json`,
  `acquisition_status = NOT_OBTAINED`，含实际尝试记录），因此这里的「文献/规范对照」用的是
  教科书可推导的经典闭式，不是规范条款，不得当作 CSR/IACS 合规声明。

## 4. 图

* `docs/assets/grillage_member_envelope/fig_frame_utilisation.png`：6 个拓扑下 smeared/system/envelope
  的框架利用率柱状图（含 U = 1 线）。
* `docs/assets/grillage_member_envelope/fig_uniform_defect.png`：均布侧压框架弯矩 经典/系统/均化（对数轴）。
* `docs/assets/grillage_member_envelope/fig_belt_and_classical.png`：冰区带载储备 + 板格经典区间与误差。

## 5. 注意点

1. **包络不是「更多精度」，而是把模型形式不确定性摆出来**：smeared 与 system 在板格控制工况下
   给出同一个 headline，只改成员裕度；在成员控制工况下才改 `p_ult`。两者必须一起报。
2. `load_split`（斑载在纵骨线/框架线上的显式分配）仍是假设；本轮 6 个拓扑的 system 分支在
   sub = 6/8/12 下框架利用率变化 < 5 %（门 G8），但纵骨在 `load_split → 1` 时会翻倍（round 4 已记录）。
3. 框架端部条件仍是模型选择：离散梁格用简支（首铰 `p a L²/8`，U = 0.239），父模块能力用固端
   （首铰 `p a L²/12`，U = 0.151）。本轮报的是**较保守的简支值**，两者都在文档给出。
4. `_unit_grid` 的 LRU 缓存依赖「成员弯矩对压力线性」，已用 G-线性测试锁住（1 vs 3 MPa 差 1e-9）。

## 6. 误差判定（逐项回答「是否 < 3 %」）

| 关键量 | 基准 | 误差 | < 3 % ? |
| --- | --- | --- | --- |
| 简支梁集中载荷弯矩 | `FL/4` | 1e-12 相对 | 是 |
| 简支梁中心斑弯矩 | `F(2L−c)/8` | 1e-12 相对 | 是 |
| 均布侧压框架弯矩（系统） | `p a L²/8` | **+5.2 %** | 否（<15 % 台账带宽，且为简支/固端口径差异） |
| 板格极限（模型 vs 经典简支板条） | `8M_p/b²` | **+2.52 %** | **是** |
| 系统分支网格收敛（sub 8 vs 12） | 自身 | < 5 %（门 G8 实测见 summary.json） | 否（<15 %） |
| 均化框架需求（均布侧压） | 系统分支 | **−97 %**（低估 35×） | **否 — 缺陷，本轮如实入账** |
| 均化框架需求（框架线斑） | 系统分支 | −76 %（低估 4.2×） | **否 — 缺陷（round 4 已记）** |
| 包络 ≥ 系统（6 拓扑） | 系统分支 | 恒成立（门 G3/G4） | 是（不等式） |

诚实判定：求解器级与经典闭式级全部 ≤5.2 %；**均化成员需求两条是真实不保守缺陷，本轮不粉饰**，
门 G5/G7 在 `summary.json` 中就是 `passed = false`（12 门 10 通过、2 个缺陷门失败）。

## 7. 复现

```
PYTHONPATH=src python benchmarks/grillage_member_envelope_round5.py sweep
PYTHONPATH=src python benchmarks/grillage_member_envelope_round5.py combined
PYTHONPATH=src python benchmarks/grillage_member_envelope_round5.py maps
PYTHONPATH=src python benchmarks/grillage_member_envelope_round5.py build
PYTHONPATH=src python gen_grillage_member_envelope_figures.py
PYTHONPATH=src python -m pytest tests/test_grillage_member_envelope.py -q      # 17 passed
```

产物：`artifacts/grillage_member_envelope/{sweep,core,combined_cases,map_system,map_panel,summary}.json`、
三张图、`docs/grillage_member_envelope_tables.md`。