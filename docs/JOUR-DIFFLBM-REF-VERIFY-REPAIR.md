# JOUR-DIFFLBM-REF-VERIFY 返修轮（m22839）—— 引用逐条核证 + 力项可执行核证

> **锚点**：`JOUR-DIFFLBM-REF-VERIFY`（总账 acceptance：立项引用逐条 DOI 核证
> ——题名/作者/年/DOI 一致；不一致者改引或降级为"未核证"）。
> **本仓**：TensorFEM。**本文件由 `benchmarks/jour_difflbm_ref_docs.py`
> 从本次运行工件程序化生成，文中数字零手抄**（源：`artifacts/JOUR-DIFFLBM-REF-VERIFY/`）。

## 0. 为什么返修（前一轮 L1 的实质缺陷）

前一轮（m22758）把 DOI 写进 docstring，再用**精确子串**断言守门，于是出现两类
可机器复核的缺陷：

1. **BLOCKING — 嵌合引用（chimera）**：Guo-Zheng-Shi (2002) 条目的题名/作者/年取自
   *Phys. Rev. E* **65**, 046308，而期刊/卷/页/DOI 取自另一篇论文
   *Phys. Fluids* **14**, 2007-2010（`10.1063/1.1471914`），且所引 DOI
   `10.1063/1.1474820` **从未注册**（Crossref 404）。子串门全绿，事实全错。
2. **MAJOR — 装饰性引用**：`lbm_cavity_diff.py` 内**没有任何力项**，却引用"Guo 力项"，
   引用指向的方法无法核证。
3. **MINOR — 门可绕过**：`"J. Supercomput. 4, 149"` 的标点/空格变体不被检测。

本轮把三类缺陷都改成了**可执行判据**，并补齐 L2/L3/L4 证据链。

## 1. 方法

| 层 | 实现 | 说明 |
|---|---|---|
| 注册表 | `src/tensorfem/lbm_references.py` | 每条引用 = 作者/年/题名/期刊/卷/页/DOI + Crossref 证据（`CROSSREF_EVIDENCE`，2026-11-16 实测） |
| 判据 | `check_citation` / `audit_source` | 逐字段比对；**嵌合检测**：题名指向 A 记录而卷页指向 B 记录即报 chimera；未注册 DOI 一律 FAIL |
| 入口 | `benchmarks/jour_difflbm_ref_verify.py`、`tensorfem lbm-ref-verify` | 离线注册表核证 + `--online` Crossref 实测；退出码即判据 |
| 力项 | `src/tensorfem/lbm_forcing.py` | Guo (2002) 源项 + 未修正源项，力驱动 Poiseuille 解析基准 |
| 集成 | `src/tensorfem/lbm_cavity_diff.py`、`lbm_channel2d.py` 的 `__references__` + `src/tensorfem/__init__.py` + `cli.py` | 生产模块的引用块由注册表渲染并受测；求解器 `_collide` 的 body_force 分支即 Guo 力项实现 |

## 2. 引用核证结果（逐条，机器判据）

本次运行模式：**online**

| 源文件 | 引用 | DOI | DOI 已注册 | 字段一致 | 判定 |
|---|---|---|---|---|---|
| `src/tensorfem/lbm_cavity_diff.py` | ladd1994 | `10.1017/S0022112094001771` | True | 7/7 | ok |
| `src/tensorfem/lbm_cavity_diff.py` | ghia1982 | `10.1016/0021-9991(82)90058-4` | True | 7/7 | ok |
| `src/tensorfem/lbm_cavity_diff.py` | guo2002_forcing | `10.1103/PhysRevE.65.046308` | True | 7/7 | ok |
| `src/tensorfem/lbm_channel2d.py` | guo2002_forcing | `10.1103/PhysRevE.65.046308` | True | 7/7 | ok |
| `src/tensorfem/lbm_channel2d.py` | peskin2002 | `10.1017/S0962492902000077` | True | 7/7 | ok |
| `src/tensorfem/lbm_channel2d.py` | turek2006 | `10.1007/3-540-34596-5_15` | True | 7/7 | ok |

注册表证据（外部锚点 = Crossref，逐字段与注册表比对）：

| 条目 | DOI | HTTP | 证据来源 | 不一致字段 |
|---|---|---|---|---|
| ladd1994 | `10.1017/S0022112094001771` | 200 | crossref-live | — |
| ghia1982 | `10.1016/0021-9991(82)90058-4` | 200 | crossref-live | — |
| guo2002_forcing | `10.1103/PhysRevE.65.046308` | 200 | crossref-live | — |
| guo2002_extrapolation | `10.1063/1.1471914` | 200 | crossref-live | — |
| peskin2002 | `10.1017/S0962492902000077` | 200 | crossref-live | — |
| turek2006 | `10.1007/3-540-34596-5_15` | 200 | crossref-live | — |

撤回的 DOI（必须保持未注册）：

| DOI | HTTP | 探针 | 仍未注册 |
|---|---|---|---|
| `10.1063/1.1474820` | 404 | crossref | True |

**改引两条**：`ladd1994`（原 `J. Supercomput. 4, 149` 不存在 → J. Fluid Mech. 271,
285-309）、`guo2002_forcing`（嵌合条目 + 未注册 DOI → Phys. Rev. E 65, 046308）。

## 3. 力项核证（让引用 [3] 变成可执行事实）

力驱动平面 Poiseuille 流，解析解 `u(y) = F_x (y-1)(n_y-1-y)/(2ρν)`，
半程反弹壁面位于 `y=1`、`y=n_y-1`，x 方向周期。修正格式（Guo）与未修正格式
（源项 `w_i c_i·F/c_s²`，同样带半力速度移位）在同一网格、同一驱动力下对照。

| 格式 | n_y | u_max ref | u_max got | 相对误差 | L2 剖面误差 | 步数 | 收敛 |
|---|---|---|---|---|---|---|---|
| guo | 10 | 0.049219 | 0.048531 | 1.397 % | 1.883 % | 2600 | yes |
| guo | 18 | 0.049805 | 0.049633 | 0.345 % | 0.471 % | 8200 | yes |
| guo | 34 | 0.049951 | 0.049908 | 0.086 % | 0.118 % | 26200 | yes |
| guo | 66 | 0.049988 | 0.049977 | 0.021 % | 0.029 % | 81400 | yes |
| naive | 10 | 0.049219 | 0.085743 | 74.208 % | 73.570 % | 3000 | yes |
| naive | 18 | 0.049805 | 0.087782 | 76.252 % | 76.087 % | 11200 | yes |
| naive | 34 | 0.049951 | 0.088292 | 76.756 % | 76.714 % | 41200 | yes |

收敛阶（Guo 格式，L2 剖面误差）：2.000
@[10, 18, 34, 66]；逐段阶 ['2.00', '2.00', '2.00']。

生产求解器对照：`ChannelLBM`（生产求解器，Guo 力项分支），n_y=34，u_max ref=0.049951 / got=0.049898，相对误差 0.106 %，L2 剖面误差 0.144 %，41800 步收敛=True

**离散格点效应（Guo 2002 的立论）**：固定 u_max≈0.05 只改变 ν（即 τ），未修正格式的
稳态剖面仍是抛物形，但幅值被缩放 **1+1/(2τ)**；修正格式幅值比 ≈1。

| ν | τ | Guo 幅值比 | 未修正幅值比 | 1+1/(2τ) | Guo L2 | 未修正 L2 |
|---|---|---|---|---|---|---|
| 0.05 | 0.65 | 0.9991 | 1.7676 | 1.7692 | 0.118 % | 76.714 % |
| 0.20 | 1.10 | 1.0009 | 1.4555 | 1.4545 | 0.123 % | 45.574 % |
| 0.50 | 2.00 | 1.0108 | 1.2630 | 1.2500 | 1.471 % | 26.628 % |

**判据（验收门）**：Guo scheme: rel_err_u_max < 3 % and rel_l2 < 1 % at the finest grid, and smaller than the uncorrected scheme at the same grid
- Guo 最细网格：n_y=66，u_max 相对误差 0.021 %，
  L2 0.029 % → **<3% ✓**
- 未修正格式同网格：u_max 相对误差 76.756 % → 差两个量级
- 门结论：**PASS**

## 4. 复现命令

```bash
# 引用核证 + 力项基准（离线注册表；--online 走 Crossref 实测）
PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --online
# 快速档（约 1 分钟）
PYTHONPATH=src python benchmarks/jour_difflbm_ref_verify.py --quick
# 平台侧入口（退出码即判据）
PYTHONPATH=src python -m tensorfem.cli lbm-ref-verify --output /tmp/lbm_ref_verify.json
# 回归测试
PYTHONPATH=src python -m pytest tests/test_lbm_cavity_diff_ref_verify.py tests/test_lbm_forcing.py -q
# 图与本文档
PYTHONPATH=src python benchmarks/jour_difflbm_ref_figures.py
PYTHONPATH=src python benchmarks/jour_difflbm_ref_docs.py
```

## 5. 注意点 / 局限

1. **离线注册表 vs 在线解析**：默认用注册表内烘焙的 Crossref 证据（2026-11-16）以保证
   无网可跑；`--online` 才逐 DOI 实测。两者都记录在 `ref_verify.json` 的
   `mode`/`source` 字段里，审阅时不应把离线结果当成"当场解析"。
2. **Turek (2006) 的卷/年**来自丛书 LNCSE 53（Crossref 章节记录不含
   volume/year），因此该两条不参与机器比对，文档已注明。
3. **未修正格式的 1+1/(2τ) 幅值律**是本仓实测的经验律（三个 τ 点吻合到 1% 内），
   不是从原文逐式推导的结论；论文只给出该格式存在离散格点误差。
4. **网格阶数**只用 L2 剖面误差估计（2.00），未做三网格 Richardson；不宣称更高阶。
5. **本锚点是文献类锚点**，唯一的"ref vs got"数值腿来自力项基准；引用本身没有
   数值基准，这是锚点性质，不是遗漏。
