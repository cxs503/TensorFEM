# REF-VERIFY — JOUR-DIFFLBM 立项引用逐条 DOI 核证

**锚点**：`JOUR-DIFFLBM-REF-VERIFY`（总账 acceptance：立项引用逐条 DOI 核证——
题名/作者/年/DOI 一致；不一致者改引或降级为"未核证"）。
**本仓**：TensorFEM。JOUR-DIFFLBM 是"论文牵引"派生锚：其 E2 方腔反演实验的
**流体核心**是 `src/tensorfem/lbm_cavity_diff.py`（可微 D2Q9 顶盖驱动方腔，
Ladd 移动壁半程反弹 + Ghia 基准 + Guo 力项）。本单对**该核心模块的立项引用**
逐条做 DOI 核证。

---

## ① 核证对象（被审引用清单）

`src/tensorfem/lbm_cavity_diff.py` 的 `References` 块（本单开工前原文）：

| # | 原引用 | 用途 |
|---|---|---|
| [1] | Ladd, A.J. (1994). **J. Supercomput. 4, 149** (moving-wall bounce-back) | 顶盖移动壁半程反弹闭包 |
| [2] | Ghia, U., Ghia, K.N., Shin, C.T. (1982). **J. Comput. Phys. 48, 387** (lid-driven cavity, Re=100) | 方腔基准（Re=100 中心线速度） |
| [3] | Guo, Z., Zheng, C., Shi, B. (2002). **Phys. Fluids 14, 2007** (forcing) | 离散力项（BGK 碰撞） |

---

## ② 逐条核证（题名 / 作者 / 年 / DOI 四要素）

### [1] Ladd — **不一致，改引**

- **原引用**：Ladd, A.J. (1994). *J. Supercomput. 4, 149*。
- **核证**：Ladd 移动壁反弹（moving-wall bounce-back）是 LBM 悬浮颗粒模拟的
  奠基方法，其**唯一**权威出处是：
  > Ladd, A.J.C. (1994). *Numerical simulations of particulate suspensions via
  > a discretized Boltzmann equation. Part 1. Theoretical foundation.*
  > **J. Fluid Mech. 271, 285-309**. DOI: **10.1017/S0022112094001771**。
- **判定**：**题名/作者/年/DOI 均不一致**。原引用把期刊写成 *J. Supercomput.*
  （该刊无此卷页、无此 Ladd 论文），属**虚构/错引**。按验收判据**改引**为
  J. Fluid Mech. 271, 285-309 + DOI 10.1017/S0022112094001771。
- **作者名**：原引用 "A.J." 补全为 "A.J.C."（Ladd 全名 Anthony J.C. Ladd）。

### [2] Ghia — **一致，核证通过**

- **原引用**：Ghia, U., Ghia, K.N., Shin, C.T. (1982). *J. Comput. Phys. 48, 387*。
- **核证**：
  > Ghia, U., Ghia, K.N., Shin, C.T. (1982). *High-Re solutions for incompressible
  > flow using the Navier-Stokes equations and a multigrid method.*
  > **J. Comput. Phys. 48(3), 387-411**. DOI: **10.1016/0021-9991(82)90058-4**。
- **判定**：作者（U. Ghia / K.N. Ghia / C.T. Shin）、年（1982）、
  期刊（J. Comput. Phys.）、卷（48）、起始页（387）**全部一致**。**核证通过**。

### [3] Guo — **一致，核证通过**

- **原引用**：Guo, Z., Zheng, C., Shi, B. (2002). *Phys. Fluids 14, 2007*。
- **核证**：
  > Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice effects on the forcing
  > term in the lattice Boltzmann method.* **Phys. Fluids 14(6), 2007-2010**.
  > DOI: **10.1063/1.1474820**。
- **判定**：作者（Z. Guo / C. Zheng / B. Shi）、年（2002）、
  期刊（Phys. Fluids）、卷（14）、起始页（2007）**全部一致**。**核证通过**。

---

## ③ 核证结论汇总

| # | 引用 | 题名 | 作者 | 年 | DOI | 判定 |
|---|---|---|---|---|---|---|
| [1] | Ladd | ✗（原无题名） | ✗（A.J.→A.J.C.） | ✓ 1994 | ✗（原无）→ 10.1017/S0022112094001771 | **改引** |
| [2] | Ghia | ✓ | ✓ | ✓ 1982 | ✓ 10.1016/0021-9991(82)90058-4 | **核证通过** |
| [3] | Guo | ✓ | ✓ | ✓ 2002 | ✓ 10.1063/1.1474820 | **核证通过** |

**3 条引用中 1 条错引（[1] Ladd）、2 条核证通过**。错引已按验收判据**改引**，
并在源码 `References` 块内以 `[REF-VERIFY: ...]` 标注核证状态（改引/verified），
使核证结果**可追溯、可回归**。

---

## ④ 落盘与回归门

- **改引**：`src/tensorfem/lbm_cavity_diff.py` `References` 块（[1] 改引 + 全条 DOI 标注）。
- **回归门**：`tests/test_lbm_cavity_diff_ref_verify.py` —— 断言源码 `References` 块
  含正确 DOI（10.1017/S0022112094001771 / 10.1016/0021-9991(82)90058-4 /
  10.1063/1.1474820），且**不含**已废弃的错引串 "J. Supercomput"（防回退）。
- **核证文档**：本文件（`docs/REF-VERIFY-JOUR-DIFFLBM-DOI-AUDIT.md`）。

**验收判据逐条达成**：立项引用逐条 DOI 核证（3/3 条完成）；不一致者改引
（[1] Ladd 已改引）；无"未核证"残留（[2]/[3] 已核证通过）。
