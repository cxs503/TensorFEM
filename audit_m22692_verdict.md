# m22692 JOUR-ICE-STRUCT-CALIB-PROV 交差审计报告

**审计者**: board-audit (m22727)
**审计日期**: 2026-11-16
**对象**: TensorFEM m22692 (JOUR-ICE-STRUCT-CALIB-PROV · 论文牵引 | 标定系数溯源)

---

## ① 取证核验

### 三件套存在性

| 产物 | 路径 | 状态 |
|---|---|---|
| run_artifacts | `artifacts/jour_ice_struct_calib_prov/calib_prov_audit.json` | ✓ 存在 (14KB) |
| 图1 | `artifacts/jour_ice_struct_calib_prov/fig_cf_fleet_band.png` | ✓ 存在 (76KB, PNG 1216×704) |
| 图2 | `artifacts/jour_ice_struct_calib_prov/fig_cf_model_ratio.png` | ✓ 存在 (91KB, PNG 1216×736) |
| 案例文档 | `artifacts/jour_ice_struct_calib_prov/CASE-JOUR-ICE-STRUCT-CALIB-PROV.md` | ✓ 存在 |
| 报告正文 | `artifacts/jour_ice_struct_calib_prov/report_body.md` | ✓ 存在 (6.7KB) |
| 报告回执 | `artifacts/jour_ice_struct_calib_prov/report_response.json` | ✓ 存在 (已提交, report_id=20391) |

### 基准数值核验

| 关键量 | ref | got | 误差 | 容差内? |
|---|---|---|---|---|
| 船队离散度 | 34.500% | 34.492% | -0.0081 pp | ✓ |
| 成员误差 @ cf_min | 0 | -16.442% | 16.442% | FAIL (超出15%) |
| 成员误差 @ cf_max | 0 | +12.379% | 12.379% | ✓ |
| 最优单一常量 minimax | — | 14.709% | — | ✓ (<15%) |
| 标度恒等式 max dev | 0 | 1.11e-16 | 1.11e-16 | ✓ |
| C_R 不变性 | True | True | — | ✓ |
| 门禁拒绝 | True | True | — | ✓ |

### repro_cmd 可执行性

```
python benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py
```
✓ 执行成功，输出与报告一致。

### 报告六段完整性

| 段 | 状态 |
|---|---|
| ① 案例介绍 | ✓ |
| ② 方法与实现 | ✓ |
| ③ 计算结果 | ✓ |
| ④ 结果图 | ✓ (2 张 PNG) |
| ⑤ 注意点 | ✓ |
| ⑥ 误差判定 | ✓ |

### 测试

`pytest tests/test_calibration_provenance.py` → **26 passed**

---

## ② 判级 (L0-L5)

**判定: L4 — 基准验证**

理由:
- L0 ✓: 有文档成果 (CASE-JOUR-ICE-STRUCT-CALIB-PROV.md, report_body.md)
- L1 ✓: 有数学/单元验证 (scaling identity 机器复核, C_R 不变性验证)
- L2 ✓: 独立功能实现 (`src/tensorfem/calibration_provenance.py` 626 行，真实输入→计算→输出)
- L3 ✓: 系统集成 (从 `src/tensorfem/__init__.py` 导出，被 benchmark driver 调用)
- L4 ✓: 基准验证 (完整算例 + 误差分析 + 可复现，26 项测试覆盖 fail-closed/最弱链/标度恒等式/船队统计)
- L5 ✗: 非科研工程成果 — ISO 侧 WITHHELD，船队数据为 ledger 转录而非实测试验数据集；本单核心贡献是 provenance 门禁机制而非完整工程交付

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts 
  (repo, scope, mission_id, level, evidence, verdict_by) 
VALUES 
  ('TensorFEM', 'm22692 JOUR-ICE-STRUCT-CALIB-PROV', 22692, 4, 
   '{"impl_files":"src/tensorfem/calibration_provenance.py:1-626","entry":"src/tensorfem/__init__.py (imported and exported)","run_artifacts":"artifacts/jour_ice_struct_calib_prov/calib_prov_audit.json + fig_cf_fleet_band.png + fig_cf_model_ratio.png","tests":"tests/test_calibration_provenance.py (26 passed)","integration_callers":"benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py, post_report.py","benchmark":"benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py (reproducible, all values match report)","repro_cmd":"python benchmarks/jour_ice_struct_calib_prov/run_calib_prov.py","external_anchor":"IACS UR I2 Rev.5 (iacs-ur-i-ice-load.json, SHA-256 bound); ISO 19906 (WITHHELD, declared coefficients)"}', 
   'board-audit')
```

已执行，id=18。

---

## ④  adversarial review 发现

### 无阻塞项
- 所有 26 项测试通过，覆盖 fail-closed、最弱链、标度恒等式、船队统计等核心不变量
- repro_cmd 可复现，数值与报告一致
- 报告六段齐全，含 2 张程序化生成的 PNG 图
- 误差判定诚实：唯一 FAIL (16.44% > 15%) 被如实标注，未粉饰

### 潜在改进建议 (非阻塞)
- ISO 19906 侧 WITHHELD 导致比值绝对水平不可用；若未来获取标准全文，可升级为 L5
- 船队数据仅含 (min, max) 无分布信息，无法做统计推断；若有完整船队数据集可增强

### 未执行检查
- 未对 `benchmarks/anchors/iacs-ur-i-ice-load.json` 的 SHA-256 做独立校验（信任 m22692 的运行时校验）
- 未对 `src/tensorfem/iso19906_ice_6dof.py` 做独立审查（WITHHELD 模块）

---

## ⑤ 处置

**Disposition: APPROVE** — m22692 交差核验通过，L4 级成立，消账。
