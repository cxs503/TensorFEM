# 锚点证书债务收账（m21948）：fem_thermal / fem_sent_j_integral 图件闭环

锚点 `GAP-PP-1-ICE-STRUCT-CONTACT-PATCH` 经核对已由他单完成（报告 #19692 / m21791；
`benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify/verification.json` all_passed=true，
`tests/test_patch_from_contact*.py` 26 项绿）。按 owner 2026-11-09 不重复令不重做，
改做本仓验证欠账：`benchmarks/verified/anchor_certificate_debt.json` 中两条 contradicted
锚点的 `E_ARTIFACT_MISSING` 缺陷（证书在、图件丢）。

## 修复
1. `fem_sent_j_integral`：`run.py --stage assemble` 从已提交缓存零求解重建 6 张 PNG。
2. `fem_thermal`：`run.py` 全量重跑（34.9 s）重建 result.json + 4 张 PNG + gates_verified.json。
3. `git add` 全部 10 张 PNG（审计器检查 git 跟踪）。
4. 债务账本 shrink-only 7→5；审计器 certified 37→39、contradicted 3→1；
   `tests/test_verified_anchor_certificates.py` 16 绿。

## honest FAIL 保留
`fem_thermal.case_c_convergence_order` 实测阶 0.0（判据 ≥1.0）仍 FAIL：2D 瞬态 Q4 后向
Euler 在 dt∝h² 同步加密下误差平台。真实数值欠账，不粉饰，留待后续单。

## 建议新条目（总账）
- slug: `GAP-FEM-CERT-DEBT-REMAINING`
- 标题: 收账剩余 5 个锚点证书债务（4×E_NO_GATES + 1×E_SCHEMA）
- 量化验收: 债务账本 5→0；`tools/verified_anchor_audit.py` contradicted=0 ∧ uncertified=0；
  `tests/test_verified_anchor_certificates.py` 绿；fem_structural_acoustics 重发规范信封
  （schema/anchor/passed/gates，闸门带数值判据与有限值）
- 所属域: TensorFEM
