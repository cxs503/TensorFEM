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

### [3] Guo — **不一致，改引（返修轮 m22839 推翻原判）**

- **本轮原文**：Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice effects
  on the forcing term in the lattice Boltzmann method.* **Phys. Fluids 14,
  2007-2010**. DOI: **10.1063/1.1474820**。
- **核证（Crossref 实测，2026-11-16）**：
  - `10.1063/1.1474820` → **HTTP 404，未注册**（doi.org 同样 404）；
  - 该**题名/作者/年**属于 **10.1103/PhysRevE.65.046308**：
    *Phys. Rev. E* **65**, 046308 (2002)；
  - 该**期刊/卷/页**属于 **10.1063/1.1471914**：
    *An extrapolation method for boundary conditions in lattice Boltzmann
    method*, *Phys. Fluids* **14**, 2007-2010 (2002)，作者同三人但**是另一篇论文**。
- **判定**：**题名/作者/年/DOI 不一致**——原引用是把两篇论文拼起来的"嵌合引用"
  （chimera）：题名取自 PRE 65, 046308，卷页与 DOI 取自 Phys. Fluids 14, 2007-2010。
  原判"核证通过"**错误**，本轮按验收判据**改引**为：

  > Guo, Z., Zheng, C., Shi, B. (2002). *Discrete lattice effects on the
  > forcing term in the lattice Boltzmann method.* **Phys. Rev. E 65, 046308**.
  > DOI: **10.1103/PhysRevE.65.046308**。

  同时该引用原为"装饰性"（`lbm_cavity_diff.py` 内无任何力项），本轮在
  :mod:`tensorfem.lbm_forcing` 中实现 Guo 力项并以解析 Poiseuille 剖面定量核证
  （见 `docs/JOUR-DIFFLBM-REF-VERIFY-REPAIR.md`）。

---

## ③ 核证结论汇总（返修后）

| # | 引用 | 题名 | 作者 | 年 | DOI | 判定 |
|---|---|---|---|---|---|---|
| [1] | Ladd | ✗（原无题名） | ✗（A.J.→A.J.C.） | ✓ 1994 | ✗（原无）→ 10.1017/S0022112094001771 | **改引** |
| [2] | Ghia | ✓ | ✓ | ✓ 1982 | ✓ 10.1016/0021-9991(82)90058-4 | **核证通过** |
| [3] | Guo（力项） | ✓ | ✓ | ✓ 2002 | ✗ 10.1063/1.1474820（404）→ 10.1103/PhysRevE.65.046308 | **改引** |

**3 条引用中 2 条错引（[1] Ladd 虚构期刊、[3] Guo 嵌合引用+未注册 DOI）、
1 条核证通过（[2] Ghia）**。两条错引均已按验收判据**改引**，并由机器判据
（`src/tensorfem/lbm_references.py` + `benchmarks/jour_difflbm_ref_verify.py`）
逐字段复核，不再依赖人工断言。

> **返修说明（m22839）**：本文件初版把 [3] 判为"核证通过"，是**错误的**
> ——初版只核对了"作者/年/期刊缩写/卷/起始页"这些字符串是否自相一致，
> 没有把 DOI 解析到 Crossref 记录上做**跨字段一致性**检查，因此没有发现题名与
> 卷页来自两篇不同论文。返修轮以 Crossref 实测证据改判，并把该缺陷类别写成
> 可执行判据（嵌合引用检测），见 `docs/JOUR-DIFFLBM-REF-VERIFY-REPAIR.md`。

## ④ 落盘与回归门

- **改引**：`src/tensorfem/lbm_cavity_diff.py` `References` 块（[1] 改引 + 全条 DOI 标注）。
- **回归门（返修后）**：`tests/test_lbm_cavity_diff_ref_verify.py` —— 不再用
  精确子串断言（旧门可被 "J. Supercomput 4:149" 之类的标点变体绕过），改为
  `lbm_references.audit_source()` **逐字段**核验（题名/作者/年/期刊/卷/页/DOI），
  外加嵌合引用（chimera）检测、未注册 DOI 检测与标点变体回归。
- **核证文档**：本文件（`docs/REF-VERIFY-JOUR-DIFFLBM-DOI-AUDIT.md`）。

**验收判据逐条达成（返修后）**：立项引用逐条 DOI 核证（3/3 条，均解析到
Crossref 记录并逐字段比对）；不一致者改引（[1] Ladd、[3] Guo 已改引）；
无"未核证"残留（[2] 核证通过，[1]/[3] 改引后核证通过）。

**返修轮证据**：`artifacts/JOUR-DIFFLBM-REF-VERIFY/ref_verify.json`、
`artifacts/JOUR-DIFFLBM-REF-VERIFY/forcing_poiseuille.json`、
`artifacts/JOUR-DIFFLBM-REF-VERIFY/fig_ref_verify.png`；
入口 `benchmarks/jour_difflbm_ref_verify.py`、CLI `tensorfem lbm-ref-verify`。
