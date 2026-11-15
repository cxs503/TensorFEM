# GAP-PP-1-ICE-STRUCT-CONTACT-PATCH — HEAD 对抗性复核工件落盘（m22149）

锚点：`GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`（冰压斑载荷由 CZM 接触求解生成并喂给板格求解器，双向耦合）。

## 本单做了什么

`tools/verify_gap_pp1_adversarial.py`（verifier 席新增的对抗性复核工具，A–E 五节全部从
HEAD 重算、只调公开 API）此前**从未产出过落盘工件**——`benchmarks/artifacts/` 下没有
`GAP-PP-1-adversarial-verify/`，即「审计工具在、证据不在」的验证欠账。本单补上：

```
python tools/verify_gap_pp1_adversarial.py --quick \
    --out benchmarks/artifacts/GAP-PP-1-adversarial-verify
```

产出（已提交）：

* `benchmarks/artifacts/GAP-PP-1-adversarial-verify/audit.json` — 机读全量结果
* `benchmarks/artifacts/GAP-PP-1-adversarial-verify/fig_convergence.png`
* `benchmarks/artifacts/GAP-PP-1-adversarial-verify/fig_validity.png`

## 关键数字（HEAD 重算，quick 网格阶梯 nx=16/24）

| 节 | 结果 | 判定 |
|----|------|------|
| A 收敛 | 均匀斑 FEM vs Navier：nx16 −1.558 % → nx24 **+0.496 %**（单调）；高斯斑：nx16 −4.187 % → nx24 **+0.735 %**（单调） | 最细网格 < 1 %，全部 < 3 %：PASS |
| B 链独立性 | 两条 FEM 链共用 `foundation_plate_bending` 装配，同网格差 1.7e-12 | **非独立**交叉验证（如实记录） |
| C 归档复现 | m20624 归档行在 HEAD 漂移 **10.4 %**（均匀斑 368.34→330.03 MPa） | **honest FAIL**：归档数字不随代码更新 |
| D 小挠度 | 原 demo 载荷 w/t = 14.0 > 0.5；m22084 的 valid_case（w/t≈0.32）在适用域内 | 原算例绝对应力不可用（已修） |
| E 面积语义 | 本链所用斑的足迹=承压面积（~1e-12）；投影误差 1.8e-16 | PASS |

## 与验收判据的对应

* 「数值与既有 H3-ICE-PLATE-FEM 锚（0.339 % 级）同量级一致性检查」：
  nx=24 时 |0.496 %| ≤ 3×0.339 % = 1.017 % → **同量级：是**（机读：`audit.json`
  → `A_crosscheck_convergence.rows`）。
* 「非均匀斑相对均匀斑峰值应力差异机读 JSON」：本工件 + 既有
  `benchmarks/artifacts/GAP-PP-1-end-to-end-demo/end_to_end_demo.json`（valid_case 节）。
* 「pytest ≥ 5 绿」：本席本地已绿 5 项（`test_ice_panel_patch_contact.py` 快速子集，
  23.7 s）；全量 3 文件（36 测试）已递交集群 cj3931，全分辨率扫描（nx=40/ny=35/grid=97）
  已递交 cj3932，结果待取回后补报。

## 未运行 / 局限

* quick 模式只到 nx=24；完整阶梯 16/24/32/48 与全分辨率扫描在集群排队（cj3931/cj3932），
  本单预算内未取回。
* 双向耦合（`coupled_patch_solve`）收敛性、塑性/屈曲/大挠度区间不在本复核口径内。
* C 节 FAIL 是**预期内的诚实结果**（m21350/m21619 合并统一了两条链，归档 json 未随之更新），
  不覆盖、不粉饰；后续单应把归档 verification.json 按 HEAD 重算刷新。
