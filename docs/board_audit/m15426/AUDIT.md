# 交差审计：m15325 CASE-LAME-CYLINDER（TensorFEM）→ 判级 **L4**

审计单 m15426（board-audit 席位）· 锚点 = CASE-LAME-CYLINDER · 被审单 m15325（status=succeeded）
证据文件：`docs/board_audit/m15426/verify_output.json`（脚本 `audit_lame_cylinder.py` 一次跑出，全部数字实测）
图：`docs/board_audit/m15426/fig_lame_audit.png`（`make_figure.py` 从 verify_output.json 生成）

## 0. 一句话结论

被审算例**不是**手抄/贴牌：所有 ref 数值可由本审计自写的 Lamé 公式复算到 **9.7e-16** 相对差；所有 got 数值可由
repro_cmd 在干净副本上**重跑复现**（除 `timestamp_utc`/`git_rev` 外 JSON 逐键相同，4 张 PNG 字节级同哈希）；
96x32 四项指标（u_r/σ_r/σ_θ/σ_z）误差 **0.008 % / 0.309 % / 0.007 % / 0.005 %**，全部 < 2 % 门；14 项测试实跑全绿。
故判 **L4（基准验证）**。**但**平台注册表 `tensor_benchmark_cases` 无该锚点行（F1），本轮"裁判"由本审计独立复算代行——
L4 的依据是证据本身，不是一条注册行。

## 1. 三件套核验（B1 取证）

| 件 | 实测指针 | 判 |
|---|---|---|
| impl_files | `benchmarks/lame_cylinder_anchor.py` 48191 B sha256=`84faac94…`；`src/tensorfem/thick_cylinder_1d_oracle.py` 6156 B sha256=`e5b5ac6c…`；复用 `src/tensorfem/lame_reference.py` 9978 B | 真代码，非占位（main 内且 sha 与工作树一致） |
| entry | `python benchmarks/lame_cylinder_anchor.py`（main 里 `benchmarks/lame_cylinder_anchor.py:main`） | 可执行，实跑 rc=0 |
| run_artifacts | `benchmarks/lame_cylinder_anchor_evidence.json` 18324 B sha256=`aa6e7dc2…` + `docs/assets/lame_cylinder_anchor_{profiles,convergence,field,sensitivity}.png` 4 张（75424/43587/109656/91290 B） | 真实运行工件（重跑后逐字节复现） |
| repro_cmd | `OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/lame_cylinder_anchor.py`（evidence JSON 自带，W 内可执行） | 干净副本重跑 rc=0，21.2 s，JSON canonical 相同、4 图同哈希 |
| 落主干 | 主干 `3674bfb6 Merge #695 (m15325) via merge-master`；m15325 的 commit 98d39a8d 非 main 祖先（squash 落地），但主干文件 sha = 工作树 sha | 真落地 |

> 复现环境注记：本机 `torch_sdaa` 自动加载会让 `import torch` 直接崩（与算例无关），必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`；
> 首次审计运行未加该变量且未先删除副本里的旧工件 → 得到"rc=1 但 JSON 相同"的**假阳性**，已修正后重跑（本条留痕，防后人复踩）。

## 2. 基准裁判（B2）

- 注册表：`SELECT … FROM tensor_benchmark_cases WHERE anchor_slug='CASE-LAME-CYLINDER'` → **0 行**（审计时全表 16 行，ILIKE '%LAME%' 亦无命中）。任务单要求的"唯一裁判"缺行 → **F1（major）**。
- 容差判据来源：任务单原文 `<2%`（mission.result 无 tolerance 字段，`goal` 文本内）。m15325 报告自述"benchmark_registry 全量 >120 s 未能本地验证故本单未注册 id（如实记为未做）"—— honest，不是隐瞒。
- **替代裁判（本审计独立复算）**：
  - ref 侧：自写 Lamé 平面应变解 `A=(p_i a²−p_o b²)/(b²−a²)`、`B=a²b²(p_i−p_o)/(b²−a²)`、`σ_r=A−B/r²`、`σ_θ=A+B/r²`、`σ_z=2νA`、`u_r=(1+ν)/E[(1−2ν)Ar+B/r]`，对 evidence 里每个 ref 值复算 → 最大相对差 **9.68e-16**。
  - 物理不变量：`σ_r(a)=−p_i` 相对误差 **1.5e-16**；`σ_θ(a)/p_i = 2.282051` vs 闭式比 `(b²+a²)/(b²−a²) = 2.282051`。
  - got 侧（96x32，3072 单元 / 6402 自由度）：u_r_max 8.10e-5、σ_r_max 3.09e-3（缩放口径 1.68e-4）、σ_θ_max 6.98e-5、σ_z_max 5.37e-5 → 四项 **PASS <2%**；力平衡残差 3.12e-14（70.7107 MN = p_i·a），轴对称性 1.16e-13。
- **第三条独立路径**（拒绝"用被审代码验证被审代码"）：本审计自写 **1D 轴对称平面应变线性元**（含 1/r 弱式 + 2 点 Gauss，TensorFEM 无共享代码）：

| n_elem | u_r 最大相对误差 | σ_r 最大相对误差 | σ_θ | σ_z |
|---|---|---|---|---|
| 32 | 6.165e-05 | 5.174e-04 | 2.488e-05 | 5.368e-05 |
| 64 | 1.541e-05 | 2.515e-04 | 6.258e-06 | 1.385e-05 |
| 128 | 3.854e-06 | 1.239e-04 | 1.569e-06 | 3.516e-06 |
| 256 | 9.635e-07 | 6.150e-05 | 3.928e-07 | 8.858e-07 |

（σ_r 为逐点级最大相对误差，含内壁陡梯度单元；明细见 `verify_output.json:C_independent_fe`）u_r 收敛阶实测 **[2.0, 2.0, 2.0]**，与 Q4 的 2.00 一致；
`u_r(a)` 与 Lamé 差 **3.85e-6**（128 元）。→ 被审算例的物理与数量级**可被独立实现复现**，不是自证。

## 3. 报告六段（B3）

- 仓内 `docs/lame-cylinder-anchor-case.md`（11704 B）：①–⑥ 标记齐全，4 张图被引用 → 齐全。
- 平台 `tensor_reports` id=**14429**（mission_id=15325，1690 B）：①–⑥ 齐全，标题/LAME 匹配。

## 4. 判级（L0–L5）

| 级 | 判据 | 实测 | 判 |
|---|---|---|---|
| L1 数学/单元 | tests 非空 | `tests/test_lame_cylinder_anchor.py` 实跑 **14 passed in 6.59 s**（collect=14） | ✅ |
| L2 独立功能 | 真代码 + 运行工件 | 输入→Q4 极坐标装配→稠密解→单元中心应力恢复→evidence JSON/图 | ✅ |
| L3 系统集成 | 主求解器/平台调用 | 复用生产面 `lame_reference`（15 处调用：`src/tensorfem/thermoelastic_annulus.py:52`、`incompressible_locking.py:260`、`rotating_disc_reference.py:40`、`structural_verification_face.py:93`、`reference_strong_form_audit.py:71`）；**新增 oracle 模块只被 benchmark+test 调用（0 处 src 引用）**，如实记 | ✅（含注） |
| L4 基准验证 | 完整算例+误差分析+可复现 | 4 级阶梯 12x4→96x32、逐项 ref vs got、收敛阶 2.00、两份敏感性（壁厚比 1.6/2.0/3.0；ν=0.30/0.45/0.49）+ 平面应力/应变判别（9.896 % vs 解析 9.890 %）、repro_cmd 实跑复现、本审计独立复算 | ✅ |
| L5 科研工程 | + 外部锚点 | ref 为教科书闭式解（Lamé 1852 / Timoshenko Art.27 / Roark T13.5），无实验或文献数值锚点对照；标准解析解复现本身不构成科研增量 | ❌ 不记 |

**判级 = L4。** 扣分点不在数值，在**credential（注册行缺失）**与**表标签语义**（F5）；若板上标准要求"L4 必须有注册行"，则本单应读作 "L4 证据成立 / 认证待补"——本审计按功能实质记 4，并把缺行作为返修向的独立缺陷留档。

## 5. 发现（severity 排序）

| id | sev | 内容 |
|---|---|---|
| **F1** | major | `tensor_benchmark_cases` 无 `CASE-LAME-CYLINDER` 行（全表 16 行无 LAME 命中）→ 交差链上的机器裁判缺失；m15325 报告已如实记为未做，属池化治理缺口而非造假 |
| **F2** | major | `tensor_reports` 中 mission_id=15325 下存在 **3 条他单标题**（CASE-HOPPER-BEVERLOO / m15259）与 1 条 probe 行（14425/14426/14427/14428）；m15325 自述"markdown >~100 B 被替换为既有记录内容"。证据面污染，跨单可复现；根因未定位（平台 API 的 INSERT 逻辑本身无 upsert，怀疑发布侧垫片），**不作断言**，只登记现象+指针。受控对照：本单 4383 B markdown 经 `POST /tensor/reports` 落库（id=14498），`md5(markdown)` 与服务端完全相同 → **平台写入路径本身不替换正文**，串号发生在发布侧（非本审计可定位范围） |
| **F5** | minor | `pointwise` 表 u_r 两列取自 ray **节点**半径 [a,(a+b)/2,b]（`benchmarks/lame_cylinder_anchor.py:717` `ray_picks=[0,n//2,n-1]`），行首 `radius_m` 是**单元中心**半径（0.7578 vs 实际 0.8，r 差 5.6 %）；误差按同半径成对算，未虚增精度，但读者会误读 u_r 所在半径 |
| **F3** | minor | run_artifacts 落 `benchmarks/` 与 `docs/assets/`，未建 `artifacts/` 目录（约定偏差，非致命） |
| **F4** | info | 交付物经 PR #695 squash 入主干（`3674bfb6`），主干文件 sha = 工作树 sha → 真落地（commit 98d39a8d 本身非 main 祖先） |

## 6. 未做的检查（不得当已过）

- 未在 GitHub CI（x86）跑；本单全部实跑在 loongarch64 本地（架构差异未覆盖）。
- 未做网格无关性外的第三份敏感性（如 E/ν 同时变化、p_o≠0 工况）；p_o=0 单工况。
- 未定位 F2 根因（未读发布侧垫片代码，未做受控复现写测试）；仅登记 DB 现象。
- 未注册 benchmark 行（F1 的修复动作不在本审计单授权范围：改平台数据需对应仓的机制单）。

## 7. 复现命令

```bash
cd /root/tp-agent/work/m15426
python3 docs/board_audit/m15426/audit_lame_cylinder.py   # → docs/board_audit/m15426/verify_output.json
python3 docs/board_audit/m15426/make_figure.py           # → fig_lame_audit.png
python3 docs/board_audit/m15426/insert_verdict.py        # → tensor_maturity_verdicts 落表
```