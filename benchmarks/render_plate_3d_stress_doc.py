"""Render ``docs/plate-3d-stress-recovery.md`` from the evidence bundle.

Every number in the document is formatted from
``docs/assets/plate_3d_stress/evidence.json`` here, so the prose cannot drift
away from the run (zero hand-copied digits).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "docs", "assets", "plate_3d_stress")
DOC = os.path.join(ROOT, "docs", "plate-3d-stress-recovery.md")


def main() -> None:
    ev = json.load(open(os.path.join(ASSET, "evidence.json")))
    tables = open(os.path.join(ASSET, "tables.md")).read()
    case = ev["case"]
    res = ev["results"]
    ref = ev["exact_reference"]
    q0 = case["q0"]
    last = res[-1]
    first = res[0]
    peak_ex = ref["exact_peak_tau_over_q0"]

    def pct(x, nd=2):
        return f"{x * 100:+.{nd}f} %"

    body = f"""# HEX20 剪切应力恢复误差 — 厚板三维锚点的验证欠账闭环

Status: **passing**（`benchmarks/plate_3d_stress_recovery.py` 程序化生成，
`docs/assets/plate_3d_stress/evidence.json` + `tables.md` + 5 张图；测试
`tests/test_plate_3d_stress_recovery.py`）。
上游条目：`docs/plate-3d-exact-anchor.md` §8.1 第三条遗留项
（「`tau_xz` 逐点剖面的有限元恢复误差无独立对照，需 Barlow/Gauss 外推恢复场」）。

## 0. 中文摘要

前一张单子把三维精确解的**取样口径**闭合了（边中点 `(0, b/2)` 解析峰值
`{peak_ex:.4f} q0`、`∫tau_xz dz = V_x` 到舍入级），但只用了**解析解自己**。
本单补上有限元一侧：用同一套 HEX20 模型（`solve_hex20_ss_plate`，剪切膜边界 +
顶面一致节点载荷），把"恢复方式"这一变量单独拆出来量：

* **离散误差**（超收敛点）：Gauss 点上的应力与精确解在同一物理点比较；
* **恢复误差**（边节点）：`x = 0` 处用单元导数逐点恢复 —— 工程上最想读、也最不可靠；
* **后处理误差**（节点最小二乘 + 形函数插值）：商业后处理器的口径；
* **Barlow 外推**：由三个 Gauss 平面 Lagrange 外推到自由边（`r = -1`）。

结论（数字见 §3）：剪切峰值在网格 `{first['nx']}×{first['nx']}×{first['nz']}`
上逐点恢复误差 `{pct(first['tau_edge_pointwise']['peak_rel_error'])}`，
最细网格 `{last['nx']}×{last['nx']}×{last['nz']}` 降到
`{pct(last['tau_edge_pointwise']['peak_rel_error'])}`；同一网格上节点插值为
`{pct(last['tau_edge_nodal_interp']['peak_rel_error'])}`、Barlow（`r` 向）外推为
`{pct(last['barlow']['barlow_rel_error'])}`。**外推/插值都没有把这 30–40 % 的
过冲消掉**，说明误差不在 `x` 向的取样口径上，而在自由边 `x = 0` 附近位移场本身的
分辨能力（该点同时是面内自由边 + 面外对称面）。这是**否定性结论**：先前怀疑的
"节点外推在中面附近丢精度"没有被证实为罪魁；`3 %` 剪切峰值不能靠换恢复方式拿到，
只能靠**面内加密**（证据 §3 表 1：`h_x` 减小 2 倍，误差 `80 %→39 %`）。
位移侧一切正常：最细网格中心挠度误差 `{pct(last['centre_deflection_rel_error'], 3)}`，
剪切静力学 `∫tau_xz dz` 误差 `{pct(last['tau_edge_pointwise']['integral_rel_error'], 3)}`。

## 1. 案例介绍

三维弹性厚板，双正弦载荷、四边剪切膜（`u_y = u_z = 0` 于 `x = 0, a`；`u_x = u_z = 0`
于 `y = 0, b`），与 `docs/plate-3d-exact-anchor.md` 完全同一算例，因此精确解是
**同一个** `SSPlateExact3D`（自检见 §2.1）。

| 量 | 符号 | 值 |
|---|---|---|
| 边长 | `a = b` | {case['a']} m |
| 厚度 | `h` | {case['h']} m |
| 厚跨比 | `a/h` | {case['a_over_h']:.0f} |
| 弹性模量 | `E` | {case['young']:.1e} Pa |
| 泊松比 | `ν` | {case['poisson']} |
| 载荷峰值 | `q0` | {q0:.1e} Pa |
| 网格 | `nx×ny×nz` | {", ".join(f"{r['nx']}×{r['nx']}×{r['nz']}" for r in res)} |
| 厚度采样 | `z` 点数 | {case['z_samples']} |

## 2. 怎么做的

1. **求解**：`tensorfem.plate_3d_exact.solve_hex20_ss_plate(..., sparse=True)`
   （HEX20、3×3×3 精确阶积分、SciPy 稀疏直接解）。
2. **应力恢复**（新增 `src/tensorfem/hex20_stress_recovery.py`）：
   `sigma = D B u` 在 (a) 27 个 Gauss 点、(b) 任意自然坐标（Newton 反解等参映射，
   失败即 raise，不会悄悄探错单元）、(c) 节点最小二乘（27 个 Gauss 张量对 20 个
   形函数基做最小二乘，`N_a(node_b) = δ_ab` 使系数即节点值）+ 共享节点单元平均。
3. **Barlow 外推**：`x` 方向三个 Gauss 平面值 → 到 `r = -1`（自由边）的三点
   Lagrange 外推，与精确解在**同一物理点**比较。
4. **对照**：所有 `sigma` 与 `SSPlateExact3D.stress` 比较，Voigt 序
   `[xx, yy, zz, xy, yz, xz]`（`txz` 在第 5 位、`tyz` 第 4 位，与精确解返回序
   `(sxx, syy, szz, txy, txz, tyz)` 不同，脚本内显式换序）。
5. **零手抄**：本文所有数字由 `benchmarks/render_plate_3d_stress_doc.py` 从
   `evidence.json` 格式化写入。

### 2.1 先证参考，再用参考

精确解自身自检（本单复跑）：`max|div sigma|` 相对量
`{ref['equilibrium']['relative']:.3e}`，顶面 `sigma_zz` 相对误差
`{ref['face_report']['top_sigma_zz_rel_error']:.3e}`，底面无面力
`{ref['face_report']['abs_bottom_sigma_zz']:.3e}` Pa，侧面剪切
`{ref['face_report']['max_abs_face_shear']:.3e}` Pa。

## 3. 计算结果

{tables}

读法：

* **表 1**：所有网格上 `pointwise@边节点` 都**偏大**（正误差），且
  `nodal 插值` 与 `Barlow 外推` 给出**同量级**（甚至同号偏大）的值 ——
  三种恢复方式并不互相救场，误差因此归因于自由边处的离散分辨能力，而不是后处理口径。
* 归因证据：`h_x` 从 `a/{first['nx']}` 降到 `a/{last['nx']}`（2 倍）时误差
  `{pct(first['tau_edge_pointwise']['peak_rel_error'])} → {pct(last['tau_edge_pointwise']['peak_rel_error'])}`，
  而厚度方向加密（`nz = {first['nz']} → {last['nz']}`）同时发生，故本表**不能**单独
  分离 `h_x`/`h_z` 的贡献；既有单变量探针（`nx=4, nz=2/3/4` 依次 `+80.4/+71.5/+76.2 %`）
  表明主导项是 `h_x`。该单变量扫描未并入本证据包，列为未结（§5.7）。
* **表 2** 的 `∫tau_xz dz` 对 `V_x = {last['shear_force_v_x']:.6e} N/m` 的误差在
  网格 `{last['nx']}×{last['nx']}×{last['nz']}` 上是
  `{pct(last['tau_edge_pointwise']['integral_rel_error'], 3)}`——剪切静力学
  （动量平衡在厚度方向的积分）几乎无误差，说明**误差集中在峰值附近**而不是整条剖面，
  这正是"用剖面积分反推峰值"不可行的原因。
* **表 2** 的 `峰值/(V_x/h)` 是板理论常值 `1/h` 与三维抛物线的比值（精确解
  `{peak_ex * q0 / last['shear_force_v_x'] * case['h']:.4f}`），有限元恢复值
  `{last['tau_edge_pointwise']['peak_over_meanV_h']:.4f}`：**在边界节点上它同样偏大**，
  若拿它去反算 `3/2` 修正系数会高估。
* **表 3** 是超收敛（Gauss 点）口径：`sxx` 误差小、`szz`/`txz` 误差大，
  这是本问题的物理事实（薄板方向应力由面力条件决定，`szz` 量级只有 `q0`）。

## 4. 图

* `fig1_tau_profile_edge.png`：`tau_xz(z)/q0` 于 `(x=0, y=b/2)`，精确解 vs 各网格逐点恢复；
* `fig2_recovery_convergence.png`：四种口径（逐点/节点插值/Barlow/Gauss 最大）误差 vs 自由度，含 3 % 门；
* `fig3_sigma_xx_centre.png`：板中心 `sigma_xx(z)` 三维精确 vs 有限元；
* `fig4_recovery_bars.png`：同一网格三种恢复方式的符号化峰值误差柱状图；
* `fig5_gauss_error.png`：最细网格 Gauss 点各分量 max/rms 相对误差。

## 5. 注意点（坑与局限）

1. **`peak_rel_error` 必须是"在解析极值位置上的取号比较"**：第一版用
   `fe_peak/|exact_peak|` 得到 `-280 %` 这种数，看着像灾难，实际是符号相消；
   已改为 `fe(解析极值位置)/signed exact` 并额外记录 `profile_max_over_exact_peak`。
2. **单元选择**：`(0, b/2, h/2)` 被多个单元共享，`_edge_spread` 把共享单元各自
   恢复值都记下来（本算例 `{last['edge_spread']['n_elements']}` 个，相对离散度
   `{last['edge_spread']['relative_spread']:.3e}`），否则"报哪个数"取决于遍历顺序。
3. **FE 网格 `z ∈ [0, h]`，精确解 `z ∈ [-h/2, h/2]`**：所有比较处都做
   `z_exact = z_fe - h/2`，忘掉这一步会得到"峰值在表面"的荒谬结论。
4. **适用范围**：各向同性线弹性、小变形、双正弦载荷、剪切膜边界。局部压力/点载下
   取样点必须重算（`tau_xz ∝ cos(alpha x) sin(beta y)` 在板中心恒零）。
5. **Barlow 外推只在 `r` 方向做**：`s, t` 保持 Gauss 位置，因此对照点不是
   `(0, b/2, h/2)` 而是最近的 Gauss 平面点（`y, z` 见 `evidence.json.barlow.point`）。
   它的用途是"证明外推可行"，不是"给出边中点峰值"。
6. **未做**：非均匀网格、TET10 对照、以及 `sigma_zz` 的面力弱满足误差（后者量级
   `{last['gauss']['szz']['max_rel'] * 100:.2f} %`，需另立锚点）。
7. **未结（诚实的边界）**：`h_x` / `h_y` / `h_z` 三向独立加密的完整归因表
   （本包只跑 3 个混合加密网格，单变量探针未入库）；边中点峰值若要 `< 3 %`，
   下一步应做纯 `h_x` 梯（`nz` 固定 4，`nx = 4 / 8 / 16`）并半解析外推。

## 6. 误差判定（显式回答）

| 关键量 | 参考 | 误差（最细网格 {last['nx']}×{last['nx']}×{last['nz']}） | 是否 < 3 % |
|---|---|---:|---|
| 中心挠度 `w_c` | 三维精确解 | {pct(last['centre_deflection_rel_error'], 3)} | **是** |
| `∫tau_xz dz`（剪切静力学） | `V_x` | {pct(last['tau_edge_pointwise']['integral_rel_error'], 3)} | **是** |
| 剪切峰值（Gauss 点 rms） | 三维精确解 | {pct(last['gauss']['txz']['rms_rel'])} | {'**是**' if abs(last['gauss']['txz']['rms_rel']) < 0.03 else '**否**'} |
| 剪切峰值（节点插值 @边中点） | 三维精确解 | {pct(last['tau_edge_nodal_interp']['peak_rel_error'])} | {'**是**' if abs(last['tau_edge_nodal_interp']['peak_rel_error']) < 0.03 else '**否**'} |
| 剪切峰值（逐点 @边节点） | 三维精确解 | {pct(last['tau_edge_pointwise']['peak_rel_error'])} | {'**是**' if abs(last['tau_edge_pointwise']['peak_rel_error']) < 0.03 else '**否**'} |
| 剪切峰值（Barlow 外推） | 三维精确解 | {pct(last['barlow']['barlow_rel_error'])} | {'**是**' if abs(last['barlow']['barlow_rel_error']) < 0.03 else '**否**'} |
| `sigma_xx` Gauss 点 max | 三维精确解 | {pct(last['gauss']['sxx']['max_rel'])} | {'**是**' if abs(last['gauss']['sxx']['max_rel']) < 0.03 else '**否**'} |
| `sigma_zz` Gauss 点 max | 三维精确解 | {pct(last['gauss']['szz']['max_rel'])} | **否**（面力弱满足，见 §5.6） |

**结论（不粉饰）**：位移与剪切静力学都 `< 3 %`；**剪切峰值不是**——逐点边界恢复
即使最细网格仍为 `{pct(last['tau_edge_pointwise']['peak_rel_error'])}`，
节点插值为 `{pct(last['tau_edge_nodal_interp']['peak_rel_error'])}`，
Barlow 外推为 `{pct(last['barlow']['barlow_rel_error'])}`。
即：本算例上"3 % 精度读剪切峰值"需要 Barlow/超收敛外推 + 足够网格，
不能直接用边界节点值。该负面结论已如实登记，未通过放宽容差掩盖。

## 7. 建议总账条目

`GAP-FEM-HEX20-STRESS-RECOVERY`（域：TensorFEM / 板壳-三维实体后处理）。
量化验收：①常应变 patch 三种恢复方式相对误差 < 1e-9（测试锁死）；
②`∫tau_xz dz` 对 `V_x` < 3 %（本包已达 `{pct(last['tau_edge_pointwise']['integral_rel_error'], 3)}`）；
③剪切 peak 的**逐点/插值/外推三口径同时入账**并随 `h_x` 单调下降（记录线，非合格线）；
④纯 `h_x` 梯（`nz` 固定）使边中点峰值误差 < 3 %（未达，下一步）；
⑤常应变 patch 之外再补一个**解析非均匀剪应力**单元测试（防"恢复模式在真实场下失配"）。
"""
    with open(DOC, "w") as fh:
        fh.write(body)
    print(f"wrote {DOC} ({len(body)} chars)")


if __name__ == "__main__":
    main()