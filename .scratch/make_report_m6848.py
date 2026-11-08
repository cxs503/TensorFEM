import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
RES = json.load(open("benchmarks/mindlin_contact/result.json"))
GATES = json.load(open("benchmarks/mindlin_contact/gates.json"))
tok = open("/tmp/tok_m6848.txt").read().strip()


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + tok})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


h3 = RES["hertz_3d"]
h2 = RES["hertz_2d_cylinder"]
cm = RES["cattaneo_mindlin"]
uni = RES["unilateral"]
fe = RES["finite_element_cross_check"]


def tbl(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


ladder3 = tbl([{"rings": r["rings"],
                "a[mm]": "%.6f" % (r["contact_radius_m"] * 1e3),
                "err": "%+.3f%%" % (100 * r["contact_radius_rel_error"]),
                "a_raw[mm]": "%.6f" % (r["contact_radius_raw_m"] * 1e3),
                "p0[GPa]": "%.5f" % (r["peak_pressure_Pa"] / 1e9),
                "err2": "%+.4f%%" % (100 * r["peak_pressure_rel_error"]),
                "resid": "%.2e" % r["complementarity_residual"]} for r in h3["mesh_ladder"]],
              ["rings", "a[mm]", "err", "a_raw[mm]", "p0[GPa]", "err2", "resid"])

pa = tbl([{"P[N]": "%.4e" % r["load_N"], "delta[m]": "%.5e" % r["approach_m"],
           "P_Hertz[N]": "%.5e" % r["load_reference_N"],
           "dev": "%.2e" % r["rel_error"]} for r in h3["load_approach"]],
         ["P[N]", "delta[m]", "P_Hertz[N]", "dev"])

ladder2 = tbl([{"segments": r["segments"],
                "b[mm]": "%.6f" % (r["half_width_m"] * 1e3),
                "err": "%+.3f%%" % (100 * r["half_width_rel_error"]),
                "p0[GPa]": "%.5f" % (r["peak_pressure_Pa"] / 1e9),
                "err2": "%+.4f%%" % (100 * r["peak_pressure_rel_error"])}
               for r in h2["mesh_ladder"]],
              ["segments", "b[mm]", "err", "p0[GPa]", "err2"])

cmt = tbl([{"seg": r["segments"], "Q/muP": "%.2f" % r["tangential_fraction"],
            "Q[N/m]": "%.4e" % r["tangential_load_N_per_m"],
            "c[mm]": "%.5f" % (r["stick_half_width_m"] * 1e3),
            "c_ref[mm]": "%.5f" % (r["stick_half_width_reference_m"] * 1e3),
            "err": "%+.3f%%" % (100 * r["stick_boundary_rel_error"]),
            "stick/contact": "%d/%d" % (r["stick_segments"], r["contact_segments"])}
           for r in cm["rows"]], ["seg", "Q/muP", "Q[N/m]", "c[mm]", "c_ref[mm]", "err", "stick/contact"])

fet = tbl([{"mesh": r["mesh"], "load[N/m]": "%.5e" % r["load_N_per_m"],
            "extent[mm]": "%.5f" % (r["contact_extent_m"] * 1e3),
            "err": "%+.2f%%" % (100 * r["contact_extent_rel_error"]),
            "p0[GPa]": "%.5f" % (r["peak_nodal_pressure_Pa"] / 1e9),
            "err2": "%+.3f%%" % (100 * r["peak_pressure_rel_error"]),
            "resid": "%.1e" % r["complementarity_residual"]}
           for r in fe["rows"]], ["mesh", "load[N/m]", "extent[mm]", "err", "p0[GPa]", "err2", "resid"])

gt = tbl([{"gate": g["gate"], "value": "%.4e" % g["value"], "limit": "%.4e" % g["limit"],
           "verdict": "PASS" if g["passed"] else "FAIL", "note": g["note"]}
          for g in GATES["gates"]], ["gate", "value", "limit", "verdict", "note"])

markdown = f"""## 1 案例介绍

刚球/刚柱压入线弹性半空间的两类经典接触问题 + 单边（Signorini）约束：

* **Hertz**：接触半径 `a`、峰值压力 `p0`、球体载荷-位移 `P(delta)`；
* **Cattaneo-Mindlin 部分滑移**：法向预载 `P` 后施加切向载荷 `Q < mu P`，界面分成中心粘着区（半宽 `c`，切向力在库仑锥内）与两侧滑移带（`|q| = mu p` 恰在锥面）；
* **单边约束**：`gap>=0`、`force>=0`、`gap*force=0`，用主动集 Lagrange 乘子**精确**满足（非罚函数），互补残差是机器舍入量级而非罚因子调参产物。

参数（程序化注入，见 result.json）：E=210 GPa, nu=0.3, R=0.05 m；圆柱线载 P=3.625e6 N/m（使 b=1.000 mm）；球体载荷 1.0e5~1.05e6 N 共 6 档（比率 1.6）；mu=0.3，Q/(mu P) in {{0.15,0.35,0.60,0.85}}；三维环网格 60/150/300 三档，二维分段 400/800/1600 三档。平面应变模量 E*=E/(1-nu^2)=230.77 GPa。

## 2 怎么做的

**算子**：接触算子取半空间**精确**弹性柔度——二维平面应变 Flamant 核（按段解析积分 `∫ln|x-s|ds = [(t-x)(ln|t-x|-1)]`）与轴对称 Boussinesq/Love 圆盘闭式解（叠成环，无奇异积分）。因此唯一误差是牵引插值误差，单一网格即可收敛，且牵引场光滑，压力/粘着边界无需后处理平滑。

**离散与求解**：分段常数牵引 + 分段/分环中心配点，得对称正定柔度矩阵 S（`w = S p`）。单边问题为凸二次规划 `min 1/2 p'Sp + (d-delta)'p, p>=0`；固定主动集时压力对压入深度**仿射**（`p = delta A^-1 1 - A^-1 d`），可用闭式求出恰好给出目标载荷的深度，再用释放/加入扫描收敛主动集——**无罚函数、无位移增量步、无二分**。切向同构：固定粘着集时 `q = u_t A^-1 1 - A^-1(S q_slip)`，闭式求 `u_t` 命中目标 `Q`，锥外段释放为滑移 `q = mu p`。

**工具链**：Python 3.11 / NumPy / SciPy（ellipk, ellipe, splu）/ matplotlib(Agg)。复现：`python benchmarks/mindlin_contact/run.py`（115 s 单核）与 `python -m pytest tests/test_hertz_mindlin.py`（17 项，32 s）。

**有限元旁证**：`block_mesh`+`plane_strain_stiffness`+`ElasticSolver`+`solve_indentation_fe` 用 Q4 + 节点对节点接触 + 精确 Lagrange 乘子解同一问题，作为体网格独立校核（含单轴应变 patch test，精确到 6e-15）。

## 3 计算结果

### 3.1 三维 Hertz 接触半径与峰值压力（E1/E2）
a_ref=4.331196e-3 m，p0_ref=1.27261e10 Pa。

{ladder3}

### 3.2 三维 Hertz 载荷-位移（E3）
R² = {h3['load_approach_r2']:.10f}，拟合指数 n = {h3['power_law_exponent']:.7f}（Hertz 3/2 律）。

{pa}

### 3.3 二维圆柱 Hertz（同算子，平面应变）
b_ref=9.9905e-4 m，p0_ref=2.30772e9 Pa。

{ladder2}

### 3.4 Cattaneo-Mindlin 粘着/滑移边界（E4）

{cmt}

切向载荷控制精确（相对偏差 <1e-8），粘着区内 `|q| < mu p`，滑移段 `|q| = mu p`（机器精度），粘着区随 Q 单调收缩。

### 3.5 单边约束（E5）与无接触负控（E6）

| 量 | 数值 |
|---|---|
| max|gap*force|，二维加载态（1600 段） | {uni['contact_complementarity_residual_2d']:.2e} |
| max|gap*force|，三维加载态（300 环） | {uni['contact_complementarity_residual_3d']:.2e} |
| 接触区最小 gap | {uni['contact_min_gap_2d']:.1e} m |
| 接触区最小压力 | {uni['contact_min_pressure_2d']:.1f}（非激活段严格 0） |
| 缩回压头（处处脱离），二维 | {uni['open_contact_force_2d']:.1f} N/m 严格零，最小 gap {uni['open_gap_min_2d']:.2e} m |
| 缩回压头，三维 | {uni['open_contact_force_3d']:.1f} N 严格零 |

### 3.6 有限元体网格旁证

{fet}

（FE 的接触半宽取最外激活节点，被节点间距量化，故为下界；细化后 -2.80%。）

## 4 结果图

* 图1 二维 Hertz 压力分布 vs 解析式 + 加载/脱离两态的间隙（单边性可视化）。
* 图2 三维 Hertz 压力剖面 + 载荷-位移对数图与解析 3/2 幂律。
* 图3 Cattaneo-Mindlin 切向牵引（4 个 Q/(mu P)，含库仑锥）+ c/b 与 sqrt(1-Q/(mu P)) 对比。

## 5 注意点（踩过的坑 / 局限）

**踩坑（负知识，已写入文档）**：
1. `block_mesh` 的 j 沿 y 递减，首版单元连接在物理面内是顺时针 → detJ<0、单元刚度负定；症状是 patch test 位移符号反且带 ±13% 波动。改为 (i,j+1),(i+1,j+1),(i+1,j),(i,j) 后 patch test 精确 6e-15。
2. `idx[:,0]` 是**表面**行而非左端面；首版把 u_x 钳在接触面上。
3. 单边约束符号：半空间解用“压入沉降为正”，FE 用 u_y（向上为正），同一表达式在一侧正确、另一侧反号；反号版看似收敛但中心乘子为负、主动集被清空。
4. 切向载荷控制的闭式 u_t 必须减去滑移带已承担的 `Q_slip`，否则实际切向载荷是目标的 1.5~2 倍、粘着区过度释放（Q/(mu P)=0.35 时 c/b=0.67 而非 0.81）。
5. 用“减去参考点列”构造柔度会使算子非对称，粘着区块条件数恶化（牵引量级失真 1e4 倍）；保留对称 Flamant 原核、让压头深度吸收基准即可。

**局限（适用边界）**：切向问题为二维平面应变，故部分滑移律是**平方根**律 `c = b sqrt(1-Q/(mu P))`，**不**复现三维 2/3 次幂律；法向/切向核未耦合（二维交叉项为 O(1-2nu) 的常数/阶跃）；线弹性、小滑移、单个凸刚体压头、单调法向后切向加载、无自接触/塑性/粘着/粗糙度、准静态；二维趋近量对数非局部，故二维只做载荷控制闸门（b、p0 vs P），不对 delta 设门；峰值压力闸门用最小二乘 Hertz 剖面恢复，同时列出无模型的 `a_raw`/`p0_raw`（≤0.05%），说明结论不依赖该假设。

## 6 误差判定（逐项 vs 基准，是否 <3%）

基准为闭式 Hertz/Cattaneo（Johnson, Contact Mechanics, 1985; Cattaneo 1938; Mindlin 1949）。

| 量 | 基准 | 计算 | 相对误差 | <3%? |
|---|---|---|---|---|
| 三维接触半径 a（300 环） | 4.331196e-3 m | 4.331196e-3 m | 0.000% | 是 |
| 三维峰值压力 p0 | 1.27261e10 Pa | 1.272585e10 Pa | 0.002% | 是 |
| 三维接触半径（最粗 60 环） | 4.331196e-3 m | 4.320368e-3 m | 0.250% | 是 |
| 载荷-位移（6 档） | P_Hertz(delta) | P_FE | 1.4e-5 | 是 |
| 二维半宽 b（1600 段） | 9.9905e-4 m | 9.998178e-4 m | 0.019% | 是 |
| 二维峰值压力 p0 | 2.30772e9 Pa | 2.307715e9 Pa | 0.000% | 是 |
| 粘着半宽 c（最差档，1600 段） | Cattaneo | 影响系数解 | 0.521% | 是 |
| 粘着半宽 c（最粗 400 段） | Cattaneo | 影响系数解 | 2.73% | 是 |
| 互补残差 | 0 | 3.45e-9 | 绝对闸 1e-8 | 是 |
| 无接触力 | 0 | 0.0 | 严格 | 是 |

全部闸门判据达成（gates.json 全 PASS，无 honest-FAIL）。

{gt}
"""

figs = []
for name, caption in (
        ("fig1_hertz_2d.png", "图1 二维 Hertz 压力分布（实线=影响系数解，虚线=解析 p0 sqrt(1-(x/b)^2)）与单边间隙：加载态接触区内 gap=0，脱离态处处 gap>0。"),
        ("fig2_hertz_3d.png", "图2 三维 Hertz 压力剖面（vs 解析）/ 载荷-位移对数图（点=计算，线=4/3 E* sqrt(R) delta^1.5），R^2=%.10f。" % h3["load_approach_r2"]),
        ("fig3_cattaneo_mindlin.png", "图3 切向牵引 q/(mu p0)（4 个 Q/(mu P)，虚线为局部库仑锥）+ 计算的 c/b 与 Cattaneo sqrt(1-Q/(mu P)) 对比。")):
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(
                     open("benchmarks/mindlin_contact/" + name, "rb").read()).decode()})

body = {
    "title": "P22-N02 Hertz接触 + Cattaneo-Mindlin摩擦滑移 + 单边约束互补残差（TensorFEM）",
    "summary": ("新增 tensorfem.hertz_mindlin：半空间影响系数接触求解器（精确 Flamant 2D 核 + 轴对称 Love 圆盘叠核）+ 主动集 Lagrange 乘子"
                "精确满足单边互补条件（非罚函数），载荷控制利用压力/牵引对压入量与切向平移的仿射性闭式求解。"
                "六项机器判据全 PASS：接触半径 0.000%(<=2%)、峰值压力 0.0021%(<=5%)、载荷-位移 R²=1.0000(>=0.99)、"
                "粘着/滑移边界 0.521%(<=5%)、互补残差 3.4e-9(<=1e-8)、无接触力严格 0。附 17 项 pytest 回归与论文级报告。"),
    "scenario_key": "contact_mechanics_hertz_cattaneo_mindlin",
    "mission_id": 6848,
    "tags": ["TensorFEM", "contact-mechanics", "Hertz", "Cattaneo-Mindlin", "complementarity",
             "P22-N02", "verification", "partial-slip"],
    "markdown": markdown,
    "figures": figs,
}
out = post("/tensor/reports", body)
print("report posted:", out.get("id"), out.get("status"))