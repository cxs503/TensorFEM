# 独立复核（adversarial review）：DCB 内聚单元「能量释放率 vs 解析」判据

锚点：【锚点=KPI-QC-RECHECK】（已合并功能独立复核通过率，in_progress）
EDIT_WHITELIST：`docs/`, `tests/`, `outputs/`

复核者：verifier 席（m16800），**未参与** #809 / #810 的实现。被复核对象：

| 项 | 身份 |
|---|---|
| 测算判据 | `benchmarks/cohesive3d_dcb_energy_release.json` + `tests/test_cohesive3d_dcb_energy_release.py`（合并 #810 / m16698，commit `5cc7ca27`） |
| 载荷/能量约定修复 | `src/tensorfem/cohesive3d_dcb.py`（合并 #809 / m16661，commit `e6ae5b70`） |
| 被复核的共同断言 | 「能量释放率 vs 解析基准 < 10 % → PASS」+「约定修复把 `G` 系数 `6→3` 纠正为正确值」 |

全部数字由本会话从已提交工件程序化重算，**零手抄**：
`PYTHONPATH=src python outputs/verify_m16800/audit_g_criterion.py` → `outputs/verify_m16800/audit_g_criterion.json`
（图：`outputs/verify_m16800/cohesive3d_g_criterion_audit.png`）

---

## 1. 结论（disposition）：**approve with notes**

- **可复现性：无缺陷。** 三个发布值从原始曲线重算，相对偏差 **0.0e+00**（local 311.1134 / fit 302.5380 /
  global 310.2163 J/m²，逐位一致）。
- **`6→3` 约定修复：另有独立证据支持（本单新增）。** 见发现 F3——用 m16698 采集的独立量
  （界面功口径的 G）与梁论在实测载荷/裂纹前沿处的 G 对比，修复后的系数把偏差从 37 % 压到 15 %。
- **主断言被高估（M1，非阻塞但须改措辞）：** 该「解析基准」不是独立基准，而是**同一条内聚律的
  牵引-分离曲线下面积**，与观测量同源（面积恒等式残差 2.2e-16）。因此该判据是**自洽性/记账检查**，
  不是对 `G_Ic` 或物理的验证：即使律的韧性值错 2×，判据仍会 PASS。
- **文档陈述过期（M2）：** m16698 证据页 §⑤ 仍写「能量平衡残差 ~50 % 为既有记账现象」，
  但 #809 已把同一探针的该残差修到 **2.73e-05**，并以 `0.25·(P_prev+P_cur)·Δδ` 外功分解为证。
  两页并存在同一 main 上会误导后续引用者。
- **未跑项（诚实标注）：** 未做 `G_Ic` 扰动的决定性反证跑（把 `G_Ic` 改成 150 J/m² 重跑
  `benchmarks/cohesive3d_dcb_energy_release.py` 并确认测得 G 亦随之减半）——该跑需驱动裂纹
  扩展 17.5 mm，成本远超本单轻量额度；建议归下次集群单。本单用**律面积恒等式**（机器精度）
  代替，结论等价且不依赖重算。

## 2. 独立性与范围

- 独立性：我读代码 + 重算工件，不采信证据页数字；判据用到的两个锚（律面积、梁论 G）
  **都不含**被考核的 `G_Ic` 参考值（除 F1 的恒等式本身，那正是待证的循环性）。
- 局限①：与作者共用同一份 `src/tensorfem/cohesive3d_dcb.py`（只读不改求解器），能抓
  「数字不可复现 / 断言过度 / 约定自相矛盾」，抓不到「求解器物理错」。
- 局限②：本机 x86_64 / Python 3.11；工件 `host` 同族，无跨架构差。
- 局限③（容量纪律）：本机 load15 ≈ 60（nproc 128），全程只跑**纯后处理**（无有限元求解），
  测试 7.1 s。

## 3. 执行过的检查

| # | 检查 | 命令 | 结果 |
|---|---|---|---|
| A1 | 三值重算 vs 发布值 | `outputs/verify_m16800/audit_g_criterion.py` | 逐位一致（dev 0.0e+00） |
| A2 | 「解析基准」独立性：律面积 vs `G_Ic` | 同上（`law_area_identity`） | 300.00000000000006 vs 300 → **同源** |
| A3 | 独立物理锚：梁论 G（实测 P、a） | 同上（`beam_crosscheck`） | 269.84 J/m²（a=47.5 mm, P=24.92 N） |
| A4 | 修复系数 vs 旧系数的判别力 | `tests/test_m16800_cohesive3d_g_criterion_audit.py` | 修复 15.3 % vs 旧 37.1 % → 修复更近 |
| A5 | 复核判据固化成可回归测试 | `pytest tests/test_m16800_cohesive3d_g_criterion_audit.py -q` | **3 passed, 7.12 s** |
| A6 | 能量平衡残差现状 | 读 `docs/GAP-FEM-3D-COHESIVE-energy-convention-evidence.json` | 2.73e-05（与 m16698 文档的 ~50 % 冲突） |

## 4. 发现（按严重度）

### M1 —— 「解析基准」与观测量同源，判据为自洽性检查（高影响 / 证据确凿 / 非代码缺陷）
内聚律 Mode-I 牵引-分离曲线下面积 = 3.000000000000e+02 J/m²，与参考值 `G_Ic` 相对偏差 **2.2e-16**。
而观测量 `G = dW_interface/dA_cracked` 中的 `W_interface` 就是该律所做功的累计。
故在完全损伤区上 `G ≡ G_Ic`；实测 +3.7 %（局部）/ +3.4 %（全局）正是**过程区未完全失效点**
（部分损伤）的渗漏量，不是物理偏差。
复现：`audit_g_criterion.py` 的 `law_area_identity`（见 JSON）。
影响：证据页「④ 结果」与头段的「PASS」措辞会被读作「FE 独立复现了 G_Ic」。
**建议改措辞**：标为「界面功/开裂面积记账自洽性 ≤ 3.7 %」，并把它与 A3 的梁论对照并列作为
真正的外部锚（后者偏差 +15.3 %，方向为「内聚测量高于梁论」，与过程区/根转角模型一致）。

### M2 —— 同一 main 上两页证据陈述互相矛盾（中影响 / 文档缺陷）
`docs/GAP-FEM-3D-COHESIVE-energy-release-verification.md` §⑤ 写「能量平衡残差 ~50 %（既有现象）」；
`docs/GAP-FEM-3D-COHESIVE-energy-convention-repair.md` 与
`docs/GAP-FEM-3D-COHESIVE-energy-convention-evidence.json`（`energy_balance_rel.after_specimen_probe`）
给出 **2.73e-05**。前者是把 #809 之前的 quick JSON 当作现状引用。
**建议**：#810 的局限段加一行「#809 后残差 2.7e-05（见 …-repair.md），本条仅适用于 #809 之前的工件」。
（本单不修改他人证据页 —— 改他人证据页等同于改结论，留给出账人。）

### F3（非缺陷，属「复核支持上游修复」的正结果）
在稳态窗末端（a = 47.5 mm, P = 24.92 N）用梁论算 G：修复系数 → **269.84 J/m²**，
旧系数 `6P²a²/(Eb²h³)` → 同样的载荷/裂纹长给出 **约 2×**；实测界面功口径 G = 311.11 J/m²，
与两者相对偏差分别 **15.3 % / 37.1 %**。这条对照**不含 `G_Ic`**，独立于 M1 的循环性，
因此构成对 #809「`6→3` 修复」的第三方支持，也说明该修复没有把 `G` 从「偏大」推成「偏小」。
已固化为 `test_measured_g_is_closer_to_beam_theory_with_repaired_convention`。

## 5. 未跑 / unknown（不得视为通过）

- `G_Ic` 扰动的决定性反证跑（见 §1 未跑项）——**not_run**。
- 宽试件（`b/h ≥ 6`，平面应变）与厚臂（`a/h < 10`）的根转角/韧带柔度——#809 已自认欠账，本单未覆盖。
- 混合模式 / 动力 / 全贯穿——未覆盖，与本判据适用范围一致。
- 峰值载荷 vs LEFM 的 honest-FAIL：**保留**（#810 明确不重做、#809 保留 `peak_load_vs_lefm.passed=False`）；
  我未复核该 FAIL 的归因（强度控制区）是否成立，**unknown**，不做判定。

## 6. 误差判定表（本单关键量 vs 基准，显式回答 < 3 %）

| 量 | 本单值 | 基准 | 误差 | <3 % ? |
|---|---:|---|---:|---|
| 发布三值 vs 原始曲线重算 | 311.1134 / 302.5380 / 310.2163 | 同 | **0.0e+00** | **是** |
| 律面积 vs 参考 `G_Ic` | 300.000000000 | 300 | 2.2e-16 | 是（且证明循环性） |
| 实测 G vs 梁论（修复系数） | 311.11 | 269.84 | **+15.3 %** | **否**（梁论为近似锚，须如实保留） |
| 实测 G vs 梁论（旧系数） | 311.11 | 539.7（约 2×） | −42 %（即 dev 37.1 %） | 否 |

第三行是本复核最诚实的一条：**能量释放率判据 PASS 不等于 FE 的 G 已与经典 DCB 梁论对齐到 3 %**；
15.3 % 的残差应作为已知适用范围写入能力台账，而不是被「< 10 % PASS」掩盖。