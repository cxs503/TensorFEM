# CASE-FEM-SHELL-MODAL — 机制归因：`cylindrical_shell4` 的谱为何漂 21–51 %（含积分规则对照）

【锚点=CASE-FEM-SHELL-MODAL-MECHANISM】｜EDIT_WHITELIST: `benchmarks/verified/fem_shell_modal/`, `src/tensorfem/`, `tests/`, `docs/`, `scratch/`
（仓缺口账本对应条目：`benchmarks/verified/anchor_certificate_debt.json::fem_shell_modal`，
锚点判据见 `benchmarks/verified/fem_shell_modal/gates_verified.json` 的 G3/G4 两道 FAIL 门）

## 0. 一句话结论

`fem_shell_modal` 的两道 FAIL 门（G3 最低物理模态低 50 %、G4 前十阶含纯旋转模态）
**根因在单元本身，不在对照解**：生产单元 24 DOF 的零能模态数是 **11**（应为 6），
每单元 **5 个伪机构**；把它换成满积分膜+剪切能消掉机构，却因膜锁死把谱推高 223–447 %。
本次**未交付单元修复**（honest FAIL 保持），交付的是可复算的归因证据 + 一条量化的修复路线。

## 1. 合同（判据先于动手）

| 项 | 值 |
|---|---|
| 现有锚点判据 G3 | 最低物理模态（径向参与 > 1e-6）vs Donnell(DMV) 最低值，`<= 15 %` |
| 现有锚点判据 G4 | 前十阶内纯旋转（径向参与 ≤ 1e-6）模态数 `== 0` |
| 独立对照解 | DMV 薄壳理论，`(m,n)=(1,5)` → **202.14 Hz**（R=0.5 m, L=1 m, t=5 mm, E=210 GPa, ρ=7800, ν=0.3） |
| 必要条件（本次新增，机器判） | 4 节点壳单元零能模态数 = 6（3 平动 + 3 转动），多出的都是伪机构 |

必要条件先于精度判据：单元有伪机构时，组装谱可以**恰好**对上一个基准值，但网格收敛行为是任意的
（这正是 pinched-cylinder 锚点此前的教训，见 `docs/audit-cylindrical-shell4-rank-deficiency.md`）。

## 2. 做法

1. **单元级零能普查**（秒级，纯线性代数）：
   `python benchmarks/verified/fem_shell_modal/rank_census.py --write`
   → `rank_census.json`。同一套几何/本构（曲率项、钻转罚项、坐标变换一字不改），
   只换膜应变与横向剪应变的积分规则。

2. **模态阶梯对照**：把 `tensorfem.cylindrical_shell_modal` 使用的
   `cylindrical_shell4_stiffness` 猴子补丁成同一运动学的各规则变体，原样重跑锚点阶梯
   （4x8 / 8x8 / 8x16，`n_modes=20`，SS 隔板边界），按径向参与 > 1e-6 分类物理/机构模态。
   原始日志入库：`benchmarks/verified/fem_shell_modal/integration_variants.log`；
   程序化注入 → `integration_variants.json` → 图 `integration_variants.png`
   （`integration_variants_figure.py`，数字从日志解析，无手抄）。

## 3. 结果

### 3.1 单元零能模态普查（`rank_census.json`）

| 变体（膜 / 剪切积分） | 零能模态数 | 伪机构 | 判定 |
|---|---|---|---|
| **1pt / 1pt（生产默认）** | **11** | **+5** | 秩亏 |
| 1pt / 2x2 | 8 | +2 | 秩亏 |
| 1pt / MITC4-ANS | 9 | +3 | 秩亏 |
| 2x2 / 1pt | 6 | 0 | 恰好 6 |
| 2x2 / 2x2 | 5 | −1 | 少于 6 |
| 2x2 / MITC4-ANS | 5 | −1 | 少于 6 |

秩亏的全部来自**一点积分的膜应变**（剪切规则单独换成 2x2 或 ANS 只把 11 减到 8–9）；
把膜改满积分即可回到 6。但 "5 < 6" 不是"更好"：说明有一个刚体模态被罚出了零能空间
（直边几何 + 钻转罚项对轴向刚体转动的残余能量），这是**另一类**不可认证缺陷——
所以不能拿 "null=5" 当通过证据。

### 3.2 模态阶梯（vs 202.14 Hz，锚点门带 ±15 %）

| 变体 | 4x8 | 8x8 | 8x16 | 前十阶机构数（4x8/8x8/8x16） |
|---|---|---|---|---|
| 1pt / 1pt（生产） | 100.35 Hz (+50.4 %) | 98.87 Hz (+51.1 %) | 159.79 Hz (+21.0 %) | 0 / 0 / 1 |
| 1pt / MITC4-ANS | 100.35 (+50.4 %) | 98.87 (+51.1 %) | 159.79 (+21.0 %) | 0 / 0 / 1 |
| 2x2 / 1pt | 845.96 (+318.5 %) | 839.19 (+315.2 %) | 653.50 (+223.3 %) | 2 / 2 / 1 |
| 2x2 / MITC4-ANS | 845.96 (+318.5 %) | 839.19 (+315.2 %) | 653.50 (+223.3 %) | 2 / 2 / 1 |
| 2x2 / 2x2 | 1105.51 (+446.9 %) | 1059.30 (+424.1 %) | 775.24 (+283.5 %) | 2 / 2 / 3 |

三条读得出来的结论：

1. **剪切积分规则对薄壳谱无影响**（1pt/1pt 与 1pt/MITC 逐位相同，2x2/1pt 与 2x2/MITC 逐位相同）：
   t/R = 0.01 时横剪能比膜/弯能小三个量级，"剪切锁死"不是这里的主病。**主病在膜应变的积分**。
2. **SRI 膜太软、满积分膜锁死**：同一网格下两族相差 8.4 倍（100 Hz vs 846 Hz），
   真值 202 Hz 落在中间——典型的"选择性减缩 vs 满积分"两难，其中任一都不是可认证方案。
3. **最细网格仍偏 21 %**（8x16，生产规则），且**不单调**：4x8→8x8 反而从 100.35 降到 98.87。
   所以锚点原有的 4x8/8x8/8x16 阶梯即便只看最细一档也不足以支撑 G3；
   同时 (m,n)=(1,5) 在 8x8 周向只有 1.6 个单元/半波，阶梯本身分辨率不足（见 §5.2 局限）。

### 3.3 顺带发现：旋转惯性块的质量矩阵积分不一致（线索，未修）

单元质量矩阵的平动块总量精确（`Σ M_trans = ρ t A`，比值 0.9999999999999998），
但旋转块总量只有解析值的 **90.24 %**（`Σ M_rot / (ρ t³/12 · A) = 0.902368927`）。
对 t/R=0.01 的薄壳影响可忽略，但它说明旋转惯性块的积分点数与形函数不匹配，
修单元时应一并修掉。复算：

```python
PYTHONPATH=. python -c "
import math, torch
from src.tensorfem.cylindrical_shell_modal import cylindrical_shell4_mass_matrix
R,L,t,rho=0.5,1.0,0.005,7800.0; dth=2*math.pi/8
x=torch.tensor([[0.,0.],[L/8,0.],[L/8,dth],[0.,dth]],dtype=torch.float64)
M=cylindrical_shell4_mass_matrix(x,R,rho,t); A=R*L/8*dth
print(sum(float(M[6*a+4,6*b+4]) for a in range(4) for b in range(4))/(rho*t**3/12*A))"
```

## 4. 修复路线（未执行，留给后续单）

按收益/风险排序：

1. **假定应变膜（assumed-strain membrane, MITC4 型）+ 满积分弯曲**：把膜应变在边中点采样后插值，
   可同时避免膜锁死与膜机构，是文献里的标准解；判据 = `rank_census.py` 报 null=6
   **且** 阶梯 8x16/16x48 进 ±15 % 带。
2. **SRI 膜 + 显式小时glass 稳定**：保持太软的优点（精度）同时补秩；稳定化参数需按谱能标定，
   风险是引入不可控人工刚度。
3. 无论走哪条，锚点阶梯必须加密到周向 ≥ 4 单元/半波（(1,5) 模态 → 周向 ≥ 40 单元）并给出收敛阶，
   否则 G3 的 15 % 门仍然不是在测单元精度。

在 (1)(2) 落地之前，`fem_shell_modal` **保持 honest FAIL**，本文件不覆盖任何既有 FAIL 记录。

## 5. 注意点（踩坑与局限）

1. **先必要条件后精度**：只看"最低频率对不上"会误判成对照解写错。本次顺序是
   零能普查（单元级、秒级）→ 才去比对频率，一步就把问题定位到膜应变积分。
2. **锚点阶梯分辨率不足**：8x8 周向对 (1,5) 只有 1.6 单元/半波。这是锚点现有设计缺陷，
   本单未改锚点判据（改判据属评审动作），只把它记为下一步的必做项。
3. **零能判定的容差敏感性**：null 计数用相对容差 1e-9；"2x2/1pt" 是 6、"2x2/2x2" 是 5，
   差的那一个来自刚体模态的**小但非零**能量（直边几何 + 钻转罚项），不是四舍五入。
   因此本文件把 "null < 6" 也判为不可认证，而不是当成通过。
4. **容量纪律**：本机 load15 ≈ 152–272（nproc 128），重网格收敛阶梯（16x64 以上）未在本会话跑完
   （后台作业被 120 s 工具上限收割），因此 §3.2 最细只到 8x16。这正是修复路线第 3 条待补的部分。
5. **口径**："物理模态"= 径向参与 > 1e-6，与锚点 G4 口径一致；未改口径。

## 6. 误差判定（逐项，显式回答是否 < 3 %）

| 量 | 本单数值 | 基准 | 误差 | < 3 % ? |
|---|---|---|---|---|
| 生产单元零能模态数 | 11 | 6（理论刚性模态数） | +5 个伪机构 | **否** |
| 最低物理模态（4x8 / 8x8 / 8x16） | 100.35 / 98.87 / 159.79 Hz | 202.14 Hz（DMV） | +50.4 / +51.1 / +21.0 % | **否** |
| 最低物理模态（2x2 膜族，三套网格） | 846–654 Hz | 202.14 Hz | +223…+447 % | **否** |
| 单元平动质量总量 | Σ M = ρtA | 解析 ρtA | 1.2e-16（相对） | **是** |
| 单元旋转惯性总量 | 0.9024 × 解析 | ρt³/12·A | **−9.76 %** | **否**（线索） |

即：**本单的结论是一个被量化的缺陷归因，不是一个通过**。没有任何一项关键力学量进入 3 %，
锚点 G3/G4 保持 FAIL。

## 7. 本单同时修掉的一个平台级缺陷

`tests/test_verified_anchor_certificates.py::test_no_unregistered_certificate_defect` 在 main 上是**红**的：
实况审计发现 `fem_shell_modal` 带缺陷（`E_ARTIFACT_MISSING`）却不在只增不减的欠账登记表里。
跑 `python tools/verified_anchor_audit.py --write` 刷新登记（`anchor_certificate_debt.json`、
`anchor_certificates.json`、`docs/verified-anchor-certificate-table.md`），比对为空 diff：
`debt-diff unregistered=[] stale=[] drift=[]`，测试回到 **16 passed**。
即"欠账不能被静默新增"这条不变量恢复了机器可执行；`fem_shell_modal` 的物理欠账则如实留在表里。

## 8. 复算清单

```bash
cd <repo>
PYTHONPATH=. python benchmarks/verified/fem_shell_modal/rank_census.py --write   # ~1 s
PYTHONPATH=. python benchmarks/verified/fem_shell_modal/integration_variants_figure.py  # 由图+JSON
python -m pytest tests/test_verified_anchor_certificates.py -q                    # 16 passed
python tools/verified_anchor_audit.py --check                                     # 0 drift
```