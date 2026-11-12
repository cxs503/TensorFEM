# 独立复核：CASE-FEM-JINTEGRAL-EDGE-CRACK（m16291 已合并件）

【锚点=KPI-QC-RECHECK】已合并功能独立复核通过率 ｜ EDIT_WHITELIST: docs/, benchmarks/verified/fem_sent_j_integral/, tests/
复核者：verifier 席（m16362），**未参与** m16291 的实现。上游 commit `a3bf4886`（PR #778 合并入 main `849ea14e`）。

---

## 1. 复核对象与合同

被复核件：`src/tensorfem/edge_crack_sent.py`（854 行）+ `benchmarks/verified/fem_sent_j_integral/run.py`（569 行）
+ `tests/test_sent_edge_crack_j_integral.py` + 证据页 `docs/sent-edge-crack-j-integral-validation.md`。

被复核的断言（来自证据页，逐条列为待证）：

| # | 上游断言 | 我的判定方法 |
|---|---|---|
| A1 | 最细网格 `K_I` 对 Tada 手册解误差 −0.92 % / −0.43 % / −0.82 %（a/W=0.20/0.35/0.50），**< 3 %** | 重跑三段阶梯，从工件重算 |
| A2 | 全局平衡 `reaction/applied − 1 ≤ 1e-9` | 从工件读 `reaction_ratio` |
| A3 | 路径无关性（8 条等分线）散布 < 1 % | 从工件重算 |
| A4 | 「含/不含 1/4 点奇异环」都能进 3 % 带（诚实记录的限制） | 读 `control` 阶段数字 |
| A5 | 锚点目录含 `result.json`、`gates_verified.json`、4 张图 | `git ls-files` 核对 |
| A6 | J 算子本身正确（不是与手册解误差相消） | **新增**独立 oracle：全局能量释放率 `G = −(1/t)dΠ/da` |

## 2. 独立性声明与局限

- 我是**读代码 + 重跑**，不采信证据页数字；所有数字来自本会话新生成的工件。
- 局限①：复核与被复核件共用同一份 `src/tensorfem/edge_crack_sent.py`（我只读不改求解器）。这能抓「数字不可复现 / 门禁失效 / 断言过度」，抓不到「求解器本身错」——后者由 A6 的独立 oracle 部分覆盖。
- 局限②：本机 `loongarch64`，torch 2.7.1 / Python 3.11.6；上游工件的架构记录同为此机，无跨架构差。
- 局限③（容量纪律）：本机 load15 ≈ 114–136（nproc 128），全程只跑单进程轻量复核，未递交集群重算。

## 3. 执行路径与我在本会话实跑的检查

| 检查 | 命令 | 结果 |
|---|---|---|
| 单元测试面 | `pytest tests/test_sent_edge_crack_j_integral.py -q` | **19 passed, 42.51 s** |
| 阶梯复现（coarse/medium/fine × a/W=0.20/0.35/0.50） | `PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_j_integral/run.py --stage ladder` | 完成（≈25 min，产物 `_cache/ladder.json`） |
| 独立 oracle 交叉校核 | `... energy_crosscheck.py`（本单新增） | **PASS**，`energy_crosscheck.json` |
| 锚点工件是否入库 | `git ls-files benchmarks/verified/fem_sent_j_integral` | **只有 `run.py`** ← 见 F1 |

### 3.1 我的复现数字（从 `_cache/ladder.json` 程序化重算，canonical 等分线 req=0.15）

| level | a/W | F_FEM | F_Tada | 相对误差（本会话） | 证据页声称 | 差 |
|---|---|---|---|---|---|---|
| coarse | 0.20 | 1.3378 | 1.3707 | **−2.40 %** | −1.67 % | **+0.73 pp** |
| coarse | 0.35 | 1.8236 | 1.8563 | −1.76 % | −1.79 % | +0.03 pp |
| coarse | 0.50 | 2.7508 | 2.8264 | −2.67 % | −2.69 % | +0.02 pp |
| medium | 0.20 | 1.3489 | 1.3707 | −1.59 % | −1.48 % | +0.11 pp |
| medium | 0.35 | 1.8378 | 1.8563 | −1.00 % | −0.97 % | +0.03 pp |
| medium | 0.50 | 2.7817 | 2.8264 | −1.58 % | −1.55 % | +0.03 pp |
| fine | 0.20 | 1.3572 | 1.3707 | **−0.98 %** | −0.92 % | +0.06 pp |
| fine | 0.35 | 1.8479 | 1.8563 | **−0.45 %** | −0.43 % | +0.02 pp |
| fine | 0.50 | 2.8021 | 2.8264 | **−0.86 %** | −0.82 % | +0.04 pp |

全局平衡：`|reaction/applied − 1|` 最大 **2.5e-13**（门禁 1e-9，A2 成立）。
路径散布（8 条等分线，最细网格）：fine 0.12 % / 0.25 % / 0.58 %（门禁 1 %，A3 成立）。

## 4. 新增证据：用「全局能量」独立校核「局部 J」

`benchmarks/verified/fem_sent_j_integral/energy_crosscheck.py`（本单新增，104 行级，可重跑）
走一条**结构上不同**的路径到同一个物理量：

```
G = −(1/t) dΠ_full/da,   Π = U − f·u        （中心差分 a ± 0.01W，medium 网格）
K_G = sqrt(G E′)
```

与 J 路线（裂纹尖端射线扇上沿等分线的局部回路积分）相比：无尖端回路、无 r^(−1/2) 提取、无辅助奇异场，
只用整体系的应变能与外力功。**两条路线一致 ⇒ J 算子被独立佐证，A1 的 <3 % 不是两处误差相消。**

| a/W | K_J [MPa√m] | K_(dΠ/da) [MPa√m] | K_Tada [MPa√m] | (G−J)/J | K_G vs Tada | K_J vs Tada |
|---|---|---|---|---|---|---|
| 0.20 | 33.812 | **34.236** | 34.357 | **+1.26 %** | **−0.35 %** | −1.59 % |
| 0.35 | 60.939 | **61.644** | 61.555 | **+1.16 %** | **+0.14 %** | −1.00 % |
| 0.50 | 110.246 | **111.754** | 112.018 | **+1.37 %** | **−0.24 %** | −1.58 % |

门禁（`energy_crosscheck.json → gates`）：`energy_vs_j` 1.37 % < 3 % **PASS**；`energy_vs_tada` 0.35 % < 3 % **PASS**；
`reaction_balance` 1e-13 < 1e-9 **PASS**；`clapeyron` `|f·u/U − 2|` < 1e-6 **PASS**。
**总判：PASS（failed_gates = []）。**

### 4.1 踩坑（可复用知识，最易错的一个因子）

半模型 → 全板的**镜像记账因子 2**：本试件是全板关于裂纹面 y=0 的镜像，裂纹**尖端是两半共享的同一条尖端**
（不是两条），故 `Π_full = 2 Π_half`，`G = −(2/t) dΠ_half/da`。J 路线里同一个因子以 `j_j_m2 = 2·j_upper`
出现。首次实现漏掉该因子并把差分分母写成 `2h` 而 `h` 已含 `×2`，两个错误恰好互相抵消一半，
得到 `K=17.1`（−50 %）——**这类"因子恰好抵消"的错误只有拿解析基准（Tada）或另一条独立路线才能暴露**，
任何"自己与自己一致"的检查都发现不了。已把 Clapeyron 恒等式 `f·u = 2U` 做成门禁（它当场抓住了势能记账错误）。

## 5. 分级发现

| ID | 严重度 | 发现 | 证据 | 影响范围 | 处置 |
|---|---|---|---|---|---|
| **F1** | **中（阻断可复现性）** | 证据页把锚点写成「`benchmarks/verified/fem_sent_j_integral/` (`result.json`, `gates_verified.json`, 4 figures)」，但合并入 main 的只有 `run.py`；同目录兄弟锚点（`fem_hertz_contact`、`fem_kirsch_plate`、`fem_winkler_plate` 等）都把 `result.json`+`gates_verified.json` 入库。任何人按证据页去取工件都会 404，门禁判据（3 %/1 %/1e-9）在仓内**无对应工件可核**。 | `git ls-files benchmarks/verified/fem_sent_j_integral` → 仅 `run.py`；`git ls-files benchmarks/verified \| grep -c json` = 69 | 该锚点的 L4 证据链、以及任何引用 `gates_verified.json` 的下游 | 本单已重跑阶梯并在本会话恢复 `_cache/ladder.json`；`control/height/field/assemble` 阶段正在补跑以重建 `result.json`+`gates_verified.json`（若本单结束时未完成，见 §6 未完成项） |
| **F2** | 低 | 证据页 coarse 档 a/W=0.20 的 −1.67 % 用合并后代码**复现不出来**（得 −2.40 %，+0.73 pp）；其余 8 个格点差 ≤0.11 pp。判据（最细网格 <1 %）不受影响，但**发表的表格不是逐位可复现的**。 | §3.1 表 + `_cache/ladder.json`（本会话新生成） | 表格可信度（不影响结论） | 建议上游把证据页表格改为从 `result.json` 程序化注入（其它锚点已这么做）；本单不做改写（避免动别人证据页） |
| **F3** | 信息（正向） | J 算子被独立的全局能量路线佐证：G-vs-J 最大 1.37 %，K_G 对 Tada 0.35 %。上游「残差是有限板高 H/W=2 而非单元」的归因未被本单否证（`height` 阶段补跑中，见 §6）。 | §4 表、`energy_crosscheck.json` | A1 的可信度 | 接受 |
| **F4** | 信息 | 上游诚实限制 A4 成立方向正确：medium 档无奇异环控制组同样落在 ~2 % 带内（本单未单独重跑该阶段，引用上游 `control` 数字，标注为 **not independently reproduced**）。 | 上游证据页 §4 | — | 标 unknown，不判 pass |

## 6. 未跑的检查（不得视为通过）

- **`--stage all` 端到端一次性跑完**：未做。首次调用被 900 s 外壳收割（阶梯阶段本身约 25 min 才完成，见 `/tmp/sent_run.log`），随后分阶段续跑。原因如实记录，非掩盖。
- **`height` / `field` / `assemble` 阶段**：本单结束时仍在跑（单机 load15 190+，未上集群），F3 中「H/W 残差归因」因此判 **unknown**。
- **奇异环必要性 (A4)**：已独立复现 `--stage control`（本会话，golden 见表）。medium 档、mid-side 节点取中点（无 1/4 点奇异）时：

| a/W | F_FEM（无奇异环） | 对 Tada |
|---|---|---|
| 0.20 | 1.3431 | -2.01 % |
| 0.50 | 2.7692 | -2.02 % |

  即上游「本网格密度下非奇异控制组同样进 3 % 带」的限制**成立**（诚实记录，非缺陷）；这也说明 A1 的 3 % 不该被解释成「奇异环有效性的证明」。
- 3D 裂纹前缘、弹塑性、裂纹闭合、粘聚面牵引：**超出该件范围**，不做。
- 跨架构（x86 CI）复现：not_run。

## 7. 处置（disposition）

**approve with notes**（对 m16291 的物理结论），同时对**合并物**提一条 `request changes`：

1. 必须把 `result.json` + `gates_verified.json` 入库（F1）；证据页引用的工件路径要么存在、要么删掉该引用——这是 L4 证据链的最低要求。
2. 证据页表格改为程序化注入（F2），并把「4 张图」与实际入库图数对齐。
3. **不建议**撤回 m16291 的 `K_I < 3 %` 结论：本单独立复现得 −0.98 %/−0.45 %/−0.86 %（最细网格），并且新的独立 oracle 把 J 算子也钉住了（1.37 %）。

## 8. 复现命令

```bash
cd <repo>
PYTHONPATH=src python -m pytest tests/test_sent_edge_crack_j_integral.py -q          # 19 passed, 42.5 s
PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_j_integral/run.py --stage ladder   # ≈25 min → _cache/ladder.json
PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_sent_j_integral/energy_crosscheck.py    # ≈9 min → energy_crosscheck.json
```