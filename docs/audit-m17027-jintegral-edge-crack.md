# 交差审计：m17027 · CASE-FEM-JINTEGRAL-EDGE-CRACK（单边裂纹板 J 积分与应力强度因子）

- 审计单：m17122（board-audit，owner 2026-11-11 任务榜令）
- 被审对象：m17027（status=succeeded，finished_at 2026-11-13 00:41:31Z，auto-approve dev_acceptance）
- 审计人：board-audit（m17122）
- 审计日期：2026-11-13

## 1. 案例介绍（被审交付物）

m17027 在 TensorFEM 闭合总账锚点 CASE-FEM-JINTEGRAL-EDGE-CRACK 的**平面应变 a/W=0.3 验收点**：
单边裂纹拉伸板（SENT），W=0.1 m，H/W=2.0，σ=100 MPa，E=210 GPa，ν=0.3，
a/W=0.20/0.30/0.50 三级网格阶梯（coarse/medium/fine，fine=2842 六节点三角形），
沿裂纹尖端扇形网格环路的 Rice J 积分 → K_I=√(J·E′)，E′=E/(1−ν²)，
对标 Tada/Paris/Irwin《应力强度因子手册》3rd ed. 有限宽修正 F(a/W)。

## 2. 审计方法（怎么做的）

1. **工件取证**：核对 m17027 工作树 `benchmarks/verified/fem_sent_j_integral/` 下
   result.json / gates_verified.json / _cache/plane_strain.json / 6 张 PNG 全部在盘；
   git 历史含 commit `3e6577d2`（12 files, +2586/−28）。
2. **基准裁判**：
   - `SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-JINTEGRAL-EDGE-CRACK'`
     → **0 行**（全表 32 行，ILIKE 变体检索均无）→ 基准台账缺登记（见 §5 缺口 A）。
   - 改以 feature_ledger 锚点验收判据为裁判：
     「归一化 K_I/(σ√(πa)) vs 手册有限宽修正（a/W=0.3，平面应变），容差 <5%」。
   - **独立复算参考式**：F(α)=1.12−0.231α+10.55α²−21.72α³+30.39α⁴
     （src/tensorfem/edge_crack_sent.py:119，适用 α≤0.6、H/W≥1，标称 ±0.5%），
     手算复现 F(0.2)=1.370664、F(0.3)=1.659919、F(0.5)=2.826375，与工件一致。
3. **实现真实性**：`j_integral_subring`（edge_crack_sent.py:584）为真 Rice 轮廓积分
   （3 点 Gauss 边求积、能量通量−做功通量、外法向定向校验、K=√(J·E′)），非 MOCK/STUB。
4. **可复现性**：重跑 `pytest tests/test_sent_edge_crack_j_integral_plane_strain.py -q`
   → **9 passed, 48.89 s**（含 medium 网格 a/W=0.3 平面应变 <3% 验收测试）。

## 3. 计算结果（got vs ref，程序化摘自 result.json，零手抄）

平面应变（m17027 新增验收面）：

| a/W | F_Tada (ref) | F_FEM (got) | 相对误差 | 容差 <5% | 平台门 <3% |
|-----|-------------|-------------|---------|---------|-----------|
| 0.20 | 1.370664 | 1.356323 | −1.046% | PASS | PASS |
| **0.30** | **1.659919** | **1.647677** | **−0.7375%** | **PASS** | **PASS** |
| 0.50 | 2.826375 | 2.802708 | −0.837% | PASS | PASS |

K_I(a/W=0.3)：got 50.583 MPa√m vs ref 50.959 MPa√m（−0.7375%）。
路径独立性：8 条轮廓 path_spread 0.16%–0.43%；反应力比 ≈1.0（平衡闭合）。
GCI 三级阶梯单调收敛（ladder_errors 0.30：−1.98%→−1.35%→−0.74%）。
平面应力基线（m16291，同工件内）：a/W=0.2/0.35/0.5 误差 −0.98%/−0.45%/−0.86%，同判 PASS。

## 4. 结果图

- `sent_plane_strain_shape_factor.png`：平面应变 F(a/W) 恢复值 vs Tada 拟合（含平面应力对照）。
- `sent_plane_strain_convergence.png`：三级网格收敛。
- `sent_plane_strain_path_independence.png`：J 积分路径独立性。
（图件为 m17027 工件原图，路径见 §6 evidence。）

## 5. 注意点（缺口与局限）

- **缺口 A（登记缺陷，非数值缺陷）**：`tensor_benchmark_cases` 无本锚点行——
  基准台账未登记，机器裁判面缺一条。数值本身对 feature_ledger 判据 PASS。
  建议补行：case_name=SENT 单边裂纹板 J 积分 K_I，metric=K_I/(σ√(πa)) vs Tada F(a/W)，
  tolerance=<5%（ledger）/<3%（平台门），ref_source=Tada 2000 DOI:10.1115/1.3601206。
- **缺口 B（报告欠账）**：tensor_reports 无 mission_id=17027 报告（report_owed=t），
  报告六段无法核验；本单报告为审计视角补发，m17027 原单报告仍欠。
- height/field 两阶段在 result.json 中如实记 not_run（H/W=3.0 CG 不收敛，确定性），
  不粉饰；Tada 拟合适用域 α≤0.6、H/W≥1，阶梯已限制在适用域内。
- 环境：loongarch64，torch 2.7.1，python 3.11.6。

## 6. 误差判定与判级

- 关键量 K_I/(σ√(πa)) @ a/W=0.3 平面应变：误差 **−0.7375% < 3%**（显式回答：是）。
- 判级 **L4（基准验证）**：完整算例（几何/载荷/网格/求解/后处理全链）+
  误差分析（GCI/路径独立/收敛阶梯/非奇异对照）+ 可复现（9/9 测试重跑绿）。
  附百页报告不改变级别；级别=功能实质。
- 降级考量：缺口 A/B 属登记/报告欠账，不构成数值或功能虚标，故不降 L3；
  但 verdict evidence 中如实标注，gate 复核可见。

## 7. Verdict 落表

INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm17027 CASE-FEM-JINTEGRAL-EDGE-CRACK', 17122, 4, <evidence JSON>, 'board-audit')
