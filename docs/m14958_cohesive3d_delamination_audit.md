# 交差审计：m14686 · GAP-FEM-3D-COHESIVE（TensorFEM 3-D cohesive 单元）

审计单 m14958（board-audit）· 对象 m14686（succeeded）· 被审交付 commit `d91f42ad` → 合入 `99f9afb1 Merge #635 (m14686) via merge-master`
审计时点：2026-11-11 · 审计工作树 `/root/tp-agent/work/m14958`（分支 `agent/m14958`，main = `2b9e6644`）

## 1. 结论（先给判定）

| 项 | 判定 |
|---|---|
| 判级 | **L1**（数学/单元级验证成立；m14686 自交付的试样级验证面**未通过其自身判据**） |
| 三件套 | 工件 ✗（未入库）· 数值 ✗（自身判据全数失败）· repro_cmd ✓（可执行，但结果=FAIL，exit 1） |
| 建议 gate 走向 | **reopened**（返修试样级基准与驱动，不动单元本体） |

> 级别=功能实质：3-D cohesive 单元本体（`src/tensorfem/cohesive3d.py`）是真实实现且有闭式单元级验证（本审计重跑 44 passed），这是 L1 的实质；
> 但 m14686 的唯一交付物 `benchmarks/cohesive3d_delamination.py`（733 行）作为"DCB/ENF/MMB 试样级验证面"，在**其自己声明的网格**上得不到通过：
> 线性柔度/VCCT 与闭式解偏差 ~100%，DCB 黏聚区推进求解不收敛 → L2「真代码 输入→计算→输出」名义存在、实质不成立（输出是错的），L3/L4 无从谈起。

## 2. 取证：m14686 到底交付了什么

```
$ git show --stat d91f42ad
d91f42ad m14686: matured mission deliverables
 benchmarks/cohesive3d_delamination.py | 733 ++++++++++++++++++++++++++++++++++
 1 file changed, 733 insertions(+)
```

- 交付面只有 1 个文件；`git log --oneline -3 -- benchmarks/cohesive3d_delamination.py` → `99f9afb1 Merge #635 (m14686)`，且 `diff <(git show 2b9e6644:…py) <m14686 工作树/…py>` = **IDENTICAL**（合入无损）。
- **run_artifacts 不存在**：脚本自述写 `benchmarks/cohesive3d_delamination.json` 与 `docs/assets/cohesive3d-delamination/response.png`，两个路径在主树均 `No such file`。对照同目录同类基线 `benchmarks/cohesive3d_verification.json`、`cohesive3d_verification_tables.json` 均已入库 —— 即"同类交付有工件、本单没有"，不是平台惯例差异。
- **tensor_reports 无本单报告**：`SELECT id,mission_id,title FROM tensor_reports WHERE mission_id=14686;` → 0 行（库内 9352 份报告，14686 空）。故 mission 自述的 ref/got 数值在平台侧不可核，只能用其代码复算。
- **零测试接入**：`grep -rn cohesive3d_delamination tests/ .github/ scripts/` = 0 命中；本单未加任何 pytest，也没有 CI 钩子。
- **无集成调用方**：`grep -rn "cohesive3d" src/` 仅 `src/tensorfem/__init__.py:254` 的再导出；除 benchmark 外无任何库内功能调用方（`assemble_cohesive_interface` 亦零调用方）→ L3 无据。

## 3. 复算（本审计独立执行，不改被审代码）

复算脚本：`artifacts/m14958_audit/repro_checks.py`（import 被审脚本原文、调用其自身 check 函数，不手抄公式）
原始输出：`artifacts/m14958_audit/repro_checks.json`

命令与耗时：`timeout 115 python artifacts/m14958_audit/repro_checks.py --zone`（< 2 min；注意：官方全量 `python benchmarks/cohesive3d_delamination.py` 在本机运行 >15 min 未出结果，见 §5 坑 3）

| 判据（被审脚本 `run()` 内） | ref | got | 相对误差 | 判据 | 通过? |
|---|---|---|---|---|---|
| `dcb_linear_compliance`（声明网格 nx100/nz3） | C=1.60714e-5 m/N | 1.11929e-9 m/N | **99.993 %** | <5 % | ✗ |
| `dcb_linear_vcct`（声明网格） | G_I=4.46429e-2 | 3.66426e-30 | **100 %** | <5 % | ✗ |
| `enf_linear_compliance`（声明网格） | C=2.37909e-5 | 4.85523e-5 | **104.08 %** | <5 % | ✗ |
| `enf_linear_vcct`（声明网格） | G_II=2.27865e-2 | 3.51529e-5 | **99.846 %** | <5 % | ✗ |
| `dcb_mode_i_debond`（黏聚区推进，20 步） | G_Ic=300 J/m² | 0（峰值载荷 0 N） | **100 %**，`converged_all=False` | <5 % | ✗ |

龙格二阶一致性检查（换网格不改变结论）：`nx60/nz2` 上 DCB 柔度误差 99.995 %、G_I 误差 100 %；ENF 柔度 74.26 %、G_II 99.94 %。
→ 不是网格不足，是**试样模型/驱动本身错**（`--quick` 亦同样的 100 % 量级偏差）。

**是否 <3%？** 否。全部关键量误差在 74 %–100 % 量级，比平台 15 % 硬线高一个数量级，比该单自设的 5 % 判据高两个数量级。

## 4. 缺陷定性（adversarial review）

**阻塞级（B1）试样装配/边界条件错，使"裂纹"根本不自由。**
`check_dcb_linear` 得 C_fem≈1.1e-9 m/N，比闭式 1.6e-5 小 4 个量级——等价于"上下臂全长粘死"。同时 `model.vcct()` 在裂尖取到的力/张开量亦近 0（G_I~1e-30），说明**裂尖处无相对位移**。指向 `build_specimen()`（`benchmarks/cohesive3d_delamination.py:152-186`）的上下臂节点编号/coincident-plane 约定，或 `bc_clamp_left/ dcb_faces`（:268-283）的加载面选择——两者之一是根因，需返修方定位（本审计未深入到根因行，属未完成项，见 §6）。

**阻塞级（B2）黏聚区推进不可用。** 第 1 个加载步 Newton 即失败（`converged_all=False`，峰值载荷 0 N），`run()` 的 `all_passed` 必为 False、`main()` 必返回 1。即该 benchmark **在任何环境下都不可能"通过"**——与它 docstring 的承诺（"Acceptance: |G_sim-G_c|/G_c < 5%"）直接冲突。

**中等级（M1）判据含自证成分（即便 B1/B2 修好也需改）。** `mmb_mixed_mode_debond` 的 `got=g1+g2` 与 `ref=bk_toughness(g1,g2)` 共用同一组线性 VCCT 的 g1/g2，mode mix 又由同一线性解决定；该判据实际退化为"FE 极限载荷² vs 线性 SERR 起裂载荷²"，无外部锚，模式比误差在两式同比例抵消。`dcb_mode_i_debond` 的 "got" 亦是把闭式 `dcb_g()` 施于 FE 峰值载荷（等价于比峰值载荷），未用 FE 自身 VCCT 在损伤前沿求 G。

**轻微（m1）工件未入库、零 pytest、零 CI 钩子**：即使数值正确，同类交付（`cohesive3d_verification.json`）已入库的先例下，本单的"可复现"只停留在"能跑"，没有 frozen 工件可供后人比对。

## 5. 注意点（踩过的坑）

1. `maturity_gate.py` 的 `files_exist()` 把相对路径解析到 `/root/tensor-ecosystem/<repo>` 或 `/root/tp-agent/work/m<audit_mid>`，所以审计证据必须落在本工作树内（本单：`artifacts/m14958_audit/`）。
2. `tensor_reports` 表需 `psql -U postgres` 才可读（默认角色 permission denied），API `GET /tensor/reports?mission_id=` 对老任务常返回空（分页上限 100），别据此断言"没报告"——要直查表。
3. **全量 repro 很慢**：`python benchmarks/cohesive3d_delamination.py` 本机 >15 min 无 stdout（输出只在末尾），harness 单命令 120 s 超时会被整组收割 → 审计要用后台 setsid + 分段跑 check 函数，否则会误判"命令跑不起来"。
4. `--quick` 不是"快"，是"退化"：nx60/nz2 同样 100 % 误差，不能拿 quick 当通过证据。
5. 审计脚本勿写 `/tmp` 通用文件名：同容器多任务并行，`/tmp/geo.py` 被别的 agent 覆盖过（导入 tensorlbm 报错），要用工作树内唯一路径。

## 6. 未完成 / 局限

- **根因行未定位**：B1 的根因（节点编号 vs 边界条件）未逐行确认，仅由"裂尖张开量≈0 + 全长刚性"反推。返修方需先用 2 单元最小模型二分。
- **未跑官方全量 `main()`**：>15 min 且必然 FAIL（由 B1/B2 可推），故用 `run()` 的 check 函数直调复算；`all_passed=False` 的判据来自 `run()` 源码逻辑，非本审计伪造。
- **external_anchor 不足**：全部参考值是梁理论闭式（ASTM D5528/D7905/D6671 归约），无实验/文献数据点；即使修好 B1/B2 也只到 L4（闭式锚），不是 L5。
- 未审 `cohesive3d.py` 单元本体的数值细节（本单范围=试样级验证面）；其单元级 44 tests 本审计重跑通过（`44 passed in 86.66s`），仅作为"本体 L1 成立"的依据。

## 7. 返修建议（给下一单）

1. 用最小 2 单元/4 节点模型验证 `build_specimen` 的 coincident 面对（`up(i,j,0)` 与 `lo(i,j,nz)` 同坐标）与裂纹区自由性；先让 `check_dcb_linear` 的柔度误差进 5 %。
2. `dcb_mode_i_debond` / `mmb_mixed_mode_debond` 改用 FE 解上的裂尖 VCCT（含损伤前沿位置）而非把闭式公式套在 FE 载荷上，去掉自证成分。
3. 把 `cohesive3d_delamination.json` + `response.png` 入库（对齐 `cohesive3d_verification.json` 先例），并加一条 <30 s 的 pytest 冒烟（小网格、断言柔度误差判据），接 CI。