import base64
import json
import pathlib

art = pathlib.Path("artifacts/broken_ice_floe_field")
study = json.loads((art / "study.json").read_text())


def fig(name, caption):
    return {
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode((art / name).read_bytes()).decode(),
    }


rows = "\n".join(
    f"| {r['density_ratio']:.2f} | {r['freeboard_analytic']:.4f} | "
    f"{r['freeboard_measured']:.4f} | {r['rel_error']:.1e} |"
    for r in study["floatation"]
)
hv = "\n".join(
    f"| {r['thickness']:.1f} | {r['period_analytic']:.4f} | "
    f"{r['period_measured']:.4f} | {r['rel_error']:.1e} |"
    for r in study["heave"]
)
wd = "\n".join(
    f"| {r['half_angle_deg']:.0f} | {r['vx_analytic']:.4f} | "
    f"{r['vx_measured']:.4f} | {r['relative_error']:.1e} |"
    for r in study["wedge"]
)
sp = "\n".join(
    f"| {r['speed']:.1f} | {r['n_corridor_initial']} | {r['clearing_rate_centroid']:.3f} | "
    f"{r['clearing_rate_coverage']:.3f} | {r['dragged_forward_fraction']:.3f} |"
    for r in study["clearing_speed"]
)
ang = "\n".join(
    f"| {r['half_angle_deg']:.0f} | {r['clearing_rate_centroid']:.3f} | "
    f"{r['clearing_rate_coverage']:.3f} | {r['n_corridor_initial']} |"
    for r in study["clearing_bow_angle"]
)
fr = "\n".join(
    f"| {r['friction']:.1f} | {r['clearing_rate_centroid']:.3f} | "
    f"{r['clearing_rate_coverage']:.3f} | {r['dragged_forward_fraction']:.3f} |"
    for r in study["clearing_friction"]
)

md = f"""# H1-BROKEN-ICE-DEM（TensorFEM 侧）：碎冰场浮态、接触与航道清空验证

## ① 案例介绍
碎冰场（brash/broken ice）中浮冰块浮于水面、相互接触、被船首侧向排开形成航道。
工程关心的两个量：浮冰浮态（吃水/干舷，密度比 0.9 → 干舷 10% 冰厚）与航道清空率。
模型：刚性圆盘浮冰块（r=1.5–3.0 m，h=0.4–1.0 m，ρ_i=900，ρ_w=1025 kg/m³），
竖向单自由度非线性浮力 `F_b = ρ_w g A clip(h/2−z, 0, h)`，水平脉冲法接触
（restitution 0.2，默认 μ=0），结构为半顶角 α 的船首楔形 + 平行中体，定速 V 扫掠
50×60 m 场域（80 块，seed=20261113，dt=0.018 s）。

## ② 方法
解析基准先行（阿基米德浮态、垂荡频率、楔面稳态速度 `v=(V·n)n` ⇒ `v_x = V sinα cosα`）；
浮态由动力学从“块底触水”沉降出来（不从解析平衡态起步）；垂荡周期用关阻尼轨迹的上穿零点
独立量；接触子模型以动量守恒/回弹系数/无能量注入与楔面稳态速度对表；清空统计把走廊内
冰盘分为侧向清空 / 推土（顶出分析窗口）/ 残留三类，另给走廊面积覆盖率清空率。
工具链：torch float64、pytest、matplotlib；积分循环固定单线程（消除归约顺序非确定性）。

## ③ 计算结果
干舷 vs 阿基米德（h = 0.6 m）：

| ρ_i/ρ_w | 解析干舷 (m) | 动力学干舷 (m) | 相对误差 |
|---|---|---|---|
{rows}

垂荡周期 vs 2π/ω：

| h (m) | 解析 (s) | 轨迹量 (s) | 相对误差 |
|---|---|---|---|
{hv}

船首斜面稳态侧向速度（V = 1.0 m/s）：

| α (°) | 解析 v_x (m/s) | 实测 v_x (m/s) | 相对误差 |
|---|---|---|---|
{wd}

航道清空 vs 航速（α = 30°，μ = 0）：

| V (m/s) | 初始走廊块数 | 侧向清空率 | 覆盖率清空率 | 推土比 |
|---|---|---|---|---|
{sp}

船首半顶角敏感性（V = 1.5 m/s）：

| α (°) | 侧向清空率 | 覆盖率清空率 | 初始走廊块数 |
|---|---|---|---|
{ang}

摩擦敏感性（V = 1.5 m/s）：

| μ | 侧向清空率 | 覆盖率清空率 | 推土比 |
|---|---|---|---|
{fr}

## ④ 结果图
三张图：干舷/垂荡对表、碎冰场扫掠前后平面图、清空率与敏感性，均由脚本从 study.json 生成。

## ⑤ 注意点
1) 浮态必须从非平衡初值沉降，否则验证退化为恒等式；2) 阻尼 `c = 2ζωm` 少乘 √m 会小 123 倍
（曾导致干舷误差 83%）；3) 法向冲量符号错会让冰-冰接触完全不发生，而“动量守恒/无能量注入”
测试仍然全绿——是回弹系数断言抓出来的；4) 楔尖端点接触的法向不是斜面法向（15° 工况偏 91%），
冰盘须置于斜面中段外法向；5) 多线程归约顺序造成同 seed 非确定性（穿插量 0.063 vs 0.018 m）；
6) `v=(V·n)n` 只对 e→0 持续接触成立，e>0 时被 `(1+e)(V·n)` 界住但不等（我的原假设已被测试否定）。

## ⑥ 误差判定（逐项回答是否 < 3%）
- 干舷（0.9 工况）：1.1e-14 → **< 3%（且满足锚点 5% 判据）**；0.85–0.95 扫描 ≤ 1.7e-14 → **是**
- 垂荡周期（h = 0.3–1.2 m）：≤ 1.6e-4 → **是**
- 楔面稳态侧向速度（α = 15–50°）：≤ 1.9e-15 → **是**
- 冰-冰接触：动量残差 < 1e-9、回弹复现 1e-9、动能不增 → **是**
- 航道清空率：**无仓内实测数据集，误差不适用（WITHHELD）**，只作为模型统计量如实报出
  （0.769–0.923，依半顶角与摩擦），不声称与实测一致。理想化模型下清空率对航速不敏感
  （0.5–3.0 m/s 恒为 12/13），作为适用边界如实登记。
"""

report = {
    "title": "H1-BROKEN-ICE-DEM：碎冰场浮态/接触/航道清空验证（TensorFEM 侧）",
    "summary": (
        "碎冰场刚体浮冰块动力学参考实现：干舷对阿基米德误差 1e-14（判据 5%）、"
        "垂荡周期对 2π/ω ≤1.6e-4、船首斜面稳态侧向速度对 V sinα cosα ≤1.9e-15；"
        "航道清空率如实报出（0.769–0.923）并声明无实测数据集可对标。"
    ),
    "scenario_key": "tensorfem.broken_ice_floe_field",
    "mission_id": 17081,
    "tags": [
        "H1-BROKEN-ICE-DEM",
        "TensorFEM",
        "broken-ice",
        "floatation",
        "channel-clearing",
        "verification",
    ],
    "markdown": md,
    "figures": [
        fig("fig_floatation_heave.png", "干舷 vs 阿基米德（含 ±5% 判据带）与垂荡周期 vs 2π/ω"),
        fig("fig_floe_field_map.png", "碎冰场初始（灰）与船首扫掠后（蓝）平面布居 + 航道走廊"),
        fig("fig_clearing.png", "清空率/推土/残留 vs 航速；船首半顶角与摩擦敏感性"),
    ],
}
pathlib.Path("/tmp/report.json").write_text(json.dumps(report))
print("report bytes", len(json.dumps(report)))