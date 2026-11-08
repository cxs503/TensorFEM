import base64
import json
import pathlib

out = pathlib.Path("/root/tp-agent/work/m6706/artifacts/contact_hertz_cattaneo")
figs = []
caps = {
    "hertz_pressure_profiles.png": "图1 三构型（球-球/球-平面/圆柱-平面）最细网格的 p/p0 vs r/a 与 Hertz 剖面 sqrt(1-(r/a)^2) 对比。",
    "hertz_convergence.png": "图2 接触半径、峰值压力、趋近量的相对误差 vs 接触区单元尺寸 h/a（F=1000 N），标出实测收敛阶。",
    "complementarity.png": "图3 左：罚函数互补残差与穿透量 vs k_n（含 1/k_n 参考线与 active-set 条件数）；右：ALM 残差 vs 迭代数（1e-8 门槛）。",
    "energy_and_traction.png": "图4 左：用于能量判据的压入曲线 vs Hertz；右：Q=0.9μN 时切向牵引 q(r) 与 Coulomb 极限 μp(r) 的粘着/滑移分区。",
    "cattaneo_mindlin.png": "图5 Cattaneo-Mindlin 部分滑移：Q/(mu N) vs u_t/a 与 c/a vs Q/(mu N)，FE 与闭式解对比（记录 honest FAIL）。",
}
for name, cap in caps.items():
    png = out / name
    figs.append({"name": name.replace(".png", ""), "caption": cap,
                 "png_base64": base64.b64encode(png.read_bytes()).decode()})

md = """# P45-N02 TensorFEM 接触力学验证面：Hertz + Cattaneo-Mindlin + 单边互补残差门

## 1 案例介绍
钢制刚性压头压入弹性体（E=210 GPa, nu=0.3），三构型：球-球（R_eff=R/2=25 mm）、球-平面（R_eff=R=50 mm）、圆柱-平面（平面应变，R=50 mm，线载 P'=F/10 mm）；
载荷 F in {10,100,1000} N；网格 h/a in {1/6, 1/8.5, 1/12}（接触区 2a 均匀加密，几何增长 1.3 到 100a 远场）。
Cattaneo-Mindlin：球-平面 N=100 N、mu=0.3，切向位移 0 到 1.25*u_t(0.9 mu N)。互补条件：罚函数 k_n 8 档 + 增广 Lagrange(ALM) 6 档容差。

## 2 方法与流程
- 新模块 `tensorfem.cattaneo_hertz`：轴对称（点接触）/ 平面应变（线接触）Q4 网格，**静力凝聚**到接触面自由度（u_C = H f_C，精确线代，单次 LU 服务所有载荷与构型）。Hertz 半径只用于布网格与无量纲标定，不进平衡、接触检测与恢复；a、p0 由 p^2 = p0^2 (1-(r/a)^2) 最小二乘反演（仅用计算压力场）。
- 两个单边求解器：**罚函数**（精确 active set，逐次记录 cond(I+k_n H_AA)）与**增广 Lagrange**（子问题等价于偏移 offset-lam/rho 的罚问题 + 乘子更新 lam <- max(0, lam - rho g)）。
- 切向：刚性粘着集 (u_r=s) + Coulomb 滑移环 (q=mu p)，单调加载 active set。
- 独立 oracle：Hertz 点/线闭式；线接触趋近量用 **Flamant 核数值积分**（不拟合对数常数）；点接触趋近量用 **Boussinesq 核（AGM 椭圆积分）** 做「参考点相对」度量，使有限域误差在比较两侧同时出现而抵消。

## 3 计算结果
| 构型（3 载荷） | a/a_H-1 | p0/p0_H-1 | delta/delta_ref-1 |
|---|---|---|---|
| 球-球 | **+0.325%** | **-0.526%** | **-0.106%** |
| 球-平面 | **+0.325%** | **-0.526%** | **-0.106%** |
| 圆柱-平面 | **+0.422%** | **+0.554%** | **-0.248%** |

三档载荷给出**逐位相同**的相对误差：无量纲问题严格自相似（a ~ F^(1/3)），故载荷阶梯验证的是标定律而非新物理（如实说明，不把 9 点当 9 个独立试验）。

互补条件（A6 PASS）：ALM 残差 **3.3e-13**（10 次迭代，min g = -1.4e-11 为舍入）；罚函数残差 4.30e-6 -> 2.16e-6 -> 4.34e-7 -> ... -> 2.17e-9，**k_n 每加倍恰好减半**（比值 0.5024 / 0.5003 / 0.5000 / 0.5000），cond(I+k_n H_AA) 36 -> 42 饱和（受 active-set 尺度支配），未出现条件数爆炸。

负控（A8 PASS）：间隙全正 -> 两法接触力严格 0.0；切向位移 0 -> 切向力严格 0.0。
能量：接触应变能 0.5 f'H f vs (2/5)F delta = **-1.08%**（独立用积分 F d(delta) 交叉验证 -1.13%）。

## 4 判据对照（A1-A9）
A1 接触半径 <=2% -> **PASS 0.42%**；A2 峰值压力 <=3% -> **PASS 0.55%**；A3 趋近量 <=3% -> **PASS 0.25%**；
A4 Cattaneo Q(u_t) <=5% -> **FAIL（+249%）**；A5 c/a <=5% -> **FAIL**；A6 互补残差 -> **PASS（3.3e-13，罚函数成半衰减）**；
A7 收敛阶 >=1.5 -> **FAIL（p0 阶 4.19/2.10 良好；a 反演噪声地板约 0.5% 使三网格阶不可估；delta 已低于截断地板）**；
A8 负控 -> **PASS**；A9 能量 <=1% -> **FAIL（-1.08%，差 0.08 个百分点）**。

## 5 注意点（踩坑 / 局限）
1. **有限域偏差**：远场固支使 delta 与能量系统性偏低 O(a/W)：W=30a 时 delta 偏差 -2.29%，W=100a 时 -0.86%。故 A3 采用「参考点相对」度量（接触中心 vs 4a 处表面点，两侧同核），把偏差移出比较；A9 未做此处理，恰好卡在 1% 门外。
2. **Cattaneo 的根本原因（重要）**：轴对称 **n=0 扇区**施加的是接触斑的**径向膨胀**（hoop 作用），而 Cattaneo 切向问题是**单向刚体错动**（同几何的 n=1 谐波）。证据：平冲头校验 Q/s = 5.57 / 5.53 / 5.51（网格 24/32/40，接触内 11/15/19 节点），解析 8(G/E)/(2-nu) = 1.81，偏差 **+205% 且不随网格收敛** => 表述性错误而非离散误差。平面应变替身同样不可用：半模型在对称面固支 u_x=0（压入的正确对称），禁掉了切向刚体位移。修复方向：n=1 谐波 2.5D 扇区（或切向半模型改 u_z=0、u_x 自由）。
3. 罚函数 / ALM 的残差归一化 ||p.g||_1/(N delta) 对点/线接触量纲不同（面积测度 vs 长度测度），仅作同族比较。
4. 粘着半径 c/a 是节点分辨率估计（接触内 11-19 节点 => 约正负 1 个节点间距）。
5. 不主张：有限滑移、可变形-可变形接触、塑性接触、真 3D 非轴对称切向接触。

## 6 误差判定（是否 <3%）
- 接触半径（三构型 x 三载荷）：<=0.42% -> **是**
- 峰值压力：<=0.55% -> **是**
- 趋近量：<=0.25% -> **是**
- 互补残差（ALM）：3.3e-13 <= 1e-8 -> **是**
- 能量：-1.08% -> 相对 3% 门槛 **是**；相对 A9 的 1% 子门槛 **否（差 0.08pp）**
- Cattaneo u_t / c/a：最高 +249% / 100% -> **否（honest FAIL，已定位 n=0 扇区表述错误）**
- GCI 收敛阶：a/delta 不可估 -> **否（honest FAIL，反演噪声 / 截断地板）**

工件：artifacts/contact_hertz_cattaneo 下 result.json、gates.json、5 张 png；测试 tests/test_contact_hertz_cattaneo.py（16 passed, 68 s）；模块 src/tensorfem/cattaneo_hertz.py；文档 docs/contact-hertz-cattaneo-complementarity.md。
"""

report = {
    "title": "P45-N02 TensorFEM 接触力学验证面：Hertz(a/p0/delta <=0.55%) + 单边互补残差门(ALM 3.3e-13) + Cattaneo honest-FAIL(根因 n=0 扇区)",
    "summary": ("三构型 x 三载荷 Hertz 达 0.42% / 0.55% / 0.25%；ALM 互补残差 3.3e-13、罚函数残差每加倍 k_n 恰好减半；"
                "Cattaneo-Mindlin 切向面被证伪为轴对称 n=0 扇区表述错误（平冲头校验 +205% 且网格无关），诚实 FAIL 并给出修复路径；"
                "A1/A2/A3/A6/A8 PASS，A4/A5/A7/A9 FAIL。"),
    "scenario_key": "fem_contact_hertz_cattaneo",
    "mission_id": 6706,
    "tags": ["TensorFEM", "contact_mechanics", "Hertz", "Cattaneo-Mindlin",
             "complementarity", "augmented-Lagrangian", "verification",
             "benchmark", "honest-FAIL"],
    "markdown": md,
    "figures": figs,
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
print("report bytes", pathlib.Path("/tmp/report.json").stat().st_size, "figs", len(figs))