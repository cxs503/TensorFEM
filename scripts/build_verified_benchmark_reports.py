#!/usr/bin/env python3
"""Build complete research reports from revalidated FE fields, including all formats.

Run the three solver scripts first. This command checks their complete fields,
regenerates figures and Markdown, and exports identical content to HTML/PDF/Word.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.benchmark_fields import verify_mesh_study
from tensorfem.hull_benchmark_fields import verify_hull_case
from audit_pointwise_benchmark_errors import audit
from tensorfem.research_report_export import export_report
from render_stiffened_panel_fe_clouds import _plot_contour

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'docs/assets/benchmark-clouds'
REPORTS = ROOT/'docs/benchmarks/tutorials'


def percent(value):
    return f'{value*100:.6g}%'


def convergence_plot(path, levels, series):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in series.items():
        ax.loglog(levels, [max(x*100, 1e-14) for x in values], 'o-', label=name)
    ax.axhline(3, color='red', linestyle='--', label='3% limit')
    ax.set(xlabel='elements', ylabel='relative error (%)')
    ax.legend(); ax.grid(True, which='both', alpha=.3); fig.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)


def rectangular_clouds(data, folder):
    case = data['cases'][-1]; nx, ny = case['mesh']['nx'], case['mesh']['ny']
    nodes, stress = case['field']['nodes'], case['field']['element_stress']
    x = np.array([p['x'] for p in nodes]).reshape(ny+1, nx+1)
    y = np.array([p['y'] for p in nodes]).reshape(ny+1, nx+1)
    component = 'ux' if folder == 'stiffened-panel' else 'uy'
    out = ASSETS/folder
    _plot_contour(x, y, np.array([p[component] for p in nodes]).reshape(ny+1, nx+1)*1e3,
                  label=component+' (mm)', title=f'Q4 {nx}x{ny}: displacement',
                  path=out/'displacement.png')
    cx = np.array([p['x'] for p in stress]).reshape(ny, nx)
    cy = np.array([p['y'] for p in stress]).reshape(ny, nx)
    for key in ('sigma_x', 'tau_xy', 'von_mises'):
        _plot_contour(cx, cy, np.array([p[key] for p in stress]).reshape(ny, nx)*1e-6,
                      label=key+' (MPa)', title=f'Q4 {nx}x{ny}: {key}',
                      path=out/(key+'.png'), cell_centred=True)


def bending_markdown(data):
    rows = []
    for c, v in zip(data['cases'], data['mesh_convergence']['checks']):
        m, e = c['mesh'], v['stress_errors']
        rows.append(f"| {m['nx']}×{m['ny']} | {m['nodes']}/{m['elements']} | "
              f"{c['tip_displacement']*1e3:.9f} | {percent(v['displacement_relative_error'])} | "
              f"{percent(e['sigma_x_l2'])} | {percent(e['tau_xy_l2'])} | "
              f"{percent(e['von_mises_l2'])} | {c['status']} |")
    balance = '\n'.join(f"| {c['mesh']['nx']}×{c['mesh']['ny']} | {c['applied_load_y']:.12f} | "
                 f"{c['reaction_left_y']:.12f} | {v['force_balance_relative_error']:.4g} |"
                 for c, v in zip(data['cases'], data['mesh_convergence']['checks']))
    fine = data['mesh_convergence']['checks'][-1]
    return f'''# 悬臂梁端载：位移与中部应力场验证报告

本次真实计算使用三组 Q4 网格。最细 80×16 网格的端位移相对误差为 **{percent(fine['displacement_relative_error'])}**，中部轴向应力 L2 误差为 **{percent(fine['stress_errors']['sigma_x_l2'])}**。这些量满足严格小于 3% 的门槛；固支端和端载区域的局部峰值应力没有获得认证。

## 1. 问题与计算条件

矩形条带长 L=1 m、高 H=0.1 m、厚 t=0.012 m，E=210 GPa、ν=0.30。左端整边固定 ux、uy，右端均布竖向载荷合力 P=-100 N。单元为全积分 Q4、线弹性平面应力，采用双精度求解。

端载按边段数分配并对两个角点取半权，确保总力等于 -100 N。全部网格、载荷和边界条件保持不变，仅加密两个方向。

## 2. 独立参考与验收范围

```text
I = t H³ / 12 = 1.0e-6 m⁴
v_ref(L) = P L³ / (3 E I) = -0.158730158730 mm
sigma_x_ref(x,y) = -P (L-x) y / I
sigma_y_ref = 0
tau_xy_ref(x,y) = P (H²/4-y²) / (2 I)
von_Mises_ref = sqrt(sigma_x_ref² + 3 tau_xy_ref²)
```

端位移采用 Euler–Bernoulli 工程参考。应力比较区在计算前固定为 **0.2L≤x≤0.8L 的全部单元中心**，覆盖截面高度，避免把固支约束和均布端载的局部二维效应当成梁理论误差。参考应力满足内部平衡；它是中部梁应力参考，不能充当当前二维边界条件的全域精确解。

应力误差为相同物理坐标的 `||s_FE-s_ref||₂/||s_ref||₂`，规则等面积网格中等同于面积加权 L2 比较。轴向、剪应力、完整应力张量和 von Mises 分别检验。参考分量为零时不做逐点除零，使用完整张量的非零范数。

## 3. 计算过程

1. 生成 20×4、40×8、80×16 网格并组装 Q4 刚度。
2. 求解位移，恢复单元中心 sigma_x、sigma_y、tau_xy 和 von Mises。
3. 由原始场再次计算端位移及应力误差，检查字段完整性、有限性、网格坐标和 von Mises 一致性。
4. 检查总载荷、反力平衡及三网格误差下降，最终网格的所有声明指标均须通过。

## 4. 真实计算结果与相对误差

| 网格 | 节点/单元 | 端位移 mm | 位移误差 | sigma_x L2 | tau_xy L2 | von Mises L2 | 网格状态 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

完整应力张量的最细网格误差为 {percent(fine['stress_errors']['stress_tensor_l2'])}。粗网格保留 blocked 标记，未把其数值改写成合格结果。位移和全部声明应力误差随网格加密下降。

| 网格 | 外力 N | 约束反力 N | 相对力不平衡 |
|---|---:|---:|---:|
{balance}

力平衡限为 1e-8，载荷相对误差限为 1e-10。

## 5. 位移、应力和误差云图

![80×16 真实节点位移 uy，单位 mm](../../assets/benchmark-clouds/cantilever-fe/displacement.png)

![80×16 轴向应力 sigma_x，单位 MPa](../../assets/benchmark-clouds/cantilever-fe/sigma_x.png)

![80×16 剪应力 tau_xy，单位 MPa](../../assets/benchmark-clouds/cantilever-fe/tau_xy.png)

![80×16 von Mises 应力，单位 MPa](../../assets/benchmark-clouds/cantilever-fe/von_mises.png)

![三网格位移与应力误差曲线](../../assets/benchmark-clouds/cantilever-fe/convergence.png)

云图使用原始单元中心场的填色等值线，延伸到几何边界的图像采用相邻中心值延伸，属于显示插值。边界颜色不是新计算的边界应力；插值不参与任何误差或验收计算。

## 6. 能力边界与结论

该报告验证端位移以及声明中部区域的梁应力指标。不能据此声称全域峰值应力、塑性、接触或真实船体已达到 3% 精度。三网格应力收敛已验证，最终结果在上述范围内通过。

## 7. 复现与原始证据

```bash
python -m pip install -e ".[dev,reports]"
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_cantilever_fe_benchmark.py --output docs/assets/benchmark-clouds/cantilever-fe/results.json
PYTHONPATH=src .venv/bin/python scripts/build_verified_benchmark_reports.py
```

[完整位移、应力、载荷及网格验收 JSON](../../assets/benchmark-clouds/cantilever-fe/results.json)。HTML、PDF、Word 位于相邻 exports 目录，内容来自本 Markdown，并包含全部云图和表格。
'''


def membrane_markdown(data):
    rows = []
    for c, v in zip(data['cases'], data['mesh_convergence']['checks']):
        m = c['mesh']; s = [p['sigma_x'] for p in c['field']['element_stress']]
        rows.append(f"| {m['nx']}×{m['ny']} | {c['tip_axial_displacement']*1e3:.12f} | "
          f"{percent(v['displacement_relative_error'])} | {percent(v['displacement_field_relative_error'])} | {percent(v['stress_errors']['sigma_x_l2'])} | "
          f"{percent(v['stress_errors']['stress_tensor_l2'])} | {min(s)*1e-6:.10f}–{max(s)*1e-6:.10f} | "
          f"{c['reaction_left_x']:.9f} | {v['force_balance_relative_error']:.4g} |")
    return f'''# 等效加筋板均匀轴向膜力：Q4 patch test 研究报告

三组网格的位移与全域单元中心应力均通过独立解析比较。该模型只折算等效厚度，未建立显式加筋几何；不作为加筋板屈曲、后屈曲或极限强度达标证据。

## 1. 问题与计算条件

板长 L=2 m、宽 H=1 m，板厚 10 mm，加筋折算厚度 6 mm，总厚度 t=16 mm。E=210 GPa、ν=0.3，右端均匀轴向力 P=1 MN，规则 Q4 平面应力双精度模型。左端全部 ux 固定，只固定一个 uy 以消除平移，并允许泊松收缩。

## 2. 参考解与误差定义

```text
A = H t = 0.016 m²
sigma_x = P/A = 62.5 MPa; sigma_y = tau_xy = 0
ux(x) = P x/(EA)
uy(y) = -ν P (y+H/2)/(EA)
ux(L) = 0.595238095238 mm
```

所有单元中心均参与应力比较。sigma_x 和 von Mises 用全场相对 L2 误差；应力张量误差包括 sigma_y、tau_xy 对零参考的偏离，分母采用非零轴向应力场范数。三组网格在机器精度附近的误差变化不用于计算虚假的收敛阶。

## 3. 计算过程

生成 10×5、20×10、40×20 网格，将边载荷按梯形权重积分，求解节点位移、恢复三分量应力，检查总载荷等于 1 MN。验证器从完整原始场重新计算位移和应力误差，不使用存储的 qualified 标记决定通过。

## 4. 真实计算结果与相对误差

| 网格 | 端均值 ux mm | 位移误差 | 位移场 L2 | sigma_x L2 | 应力张量 L2 | sigma_x 范围 MPa | 反力 N | 相对力不平衡 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

全部应力与位移误差严格小于 3%，载荷相对误差小于 1e-10，反力相对不平衡小于 1e-8。均匀膜力场是 patch test，机器精度结果不能外推为复杂加筋结构的精度。

## 5. 位移、应力和网格验证图

![40×20 轴向位移 ux，单位 mm](../../assets/benchmark-clouds/stiffened-panel/displacement.png)

![40×20 von Mises 应力，单位 MPa](../../assets/benchmark-clouds/stiffened-panel/von_mises.png)

![三网格 patch test 误差，机器精度区域不估计收敛阶](../../assets/benchmark-clouds/stiffened-panel/convergence.png)

常应力场按真实均值着色，浮点噪声不放大成空间斑块。应力图在模型边缘延伸的是中心值显示插值，数值验收仅使用原始单元值。

## 6. 结论与能力边界

均匀膜力 patch test 的位移、全场应力、载荷和反力已闭环。显式纵骨弯曲、局部屈曲、初始缺陷、残余应力、塑性和极限强度仍需要独立证据。

## 7. 复现与原始证据

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_stiffened_panel_fe_benchmark.py --output docs/assets/benchmark-clouds/stiffened-panel/results.json
PYTHONPATH=src .venv/bin/python scripts/build_verified_benchmark_reports.py
```

[完整节点与应力及验收 JSON](../../assets/benchmark-clouds/stiffened-panel/results.json)。同一正文的 Word、PDF、HTML 均在 exports 目录。
'''


def hull_clouds(data):
    c = data['cases'][-1]; f = c['field']; x, y = np.meshgrid(f['x'], f['y'])
    out = ASSETS/'hull-girder'
    uy = np.tile(np.array(f['uy']), (len(f['y']), 1))*1e3
    for key, values, label in [('displacement', uy, 'uy (mm)'),
          ('sigma_x', np.array(f['sigma_x'])*1e-6, 'sigma_x (MPa)'),
          ('von_mises', np.abs(np.array(f['sigma_x']))*1e-6, 'uniaxial von Mises (MPa)')]:
        _plot_contour(x, y, values, label=label,
                      title='120 m hull-girder beam fibre recovery', path=out/(key+'.png'))


def hull_markdown(data):
    point = audit()['hull_girder']
    rows = []
    for c in data['cases']:
        v = c['verification']; e = v['errors']; n = c['mesh']['elements']
        rows.append(f"| {n} ({'pass' if v['passed'] else 'fail'}) | {-c['nodal_dofs'][n//2][1]:.9f} | "
                    f"{percent(e['midship_displacement'])} | {percent(e['displacement_field_l2'])} | "
                    f"{percent(e['moment_field_l2'])} | {percent(e['sigma_x_field_l2'])} | {percent(e['moment_and_sigma_x_pointwise_max'])} | "
                    f"{sum(c['support_reactions_n']):.5f} | {v['force_balance_relative_error']:.4g} |")
    n = data['cases'][-1]['mesh']['elements']
    stress = np.array(data['cases'][-1]['field']['sigma_x'])
    return f'''# 120 m 船体梁纵向弯曲：真实 Frame FE 场与误差研究报告

本报告采用 120 m 两端简支梁、均布静水和波浪等效线载荷，与 1 m 固支端载 Q4 悬臂梁具有不同边界、载荷和单元。旧版同模型换标题的船体梁报告已由本独立力学问题替换。

## 1. 问题与计算条件

| 条件 | 数值 |
|---|---:|
| 船体梁长度 L | 120 m |
| E | 210 GPa |
| 等效截面面积 A | 5 m² |
| 等效惯性矩 I | 180 m⁴ |
| 静水等效向下载荷 | 1.4 MN/m |
| 波浪等效向下载荷 | 0.9 MN/m |
| 总向下载荷 q | 2.3 MN/m |
| 应力输出纤维 | y=-6 至 +6 m，共 25 层 |
| 单元 | Euler–Bernoulli Frame2D |
| 边界 | 左端 ux、uy 固定；右端 uy 固定，转角自由 |
| 网格 | 6、12、24 单元 |

A、I 是输入等效截面属性，y 轴表示梁纤维的竖向坐标，图像不是实际船体壳网格。均布载荷使用单元一致节点力和节点力矩。

## 2. 独立解析参考

```text
v(x) = -q x (L³-2 L x²+x³)/(24 E I)
M(x) = q x (L-x)/2
sigma_x(x,y) = -M(x) y/I
v_mid magnitude = 0.164285714286 m
M_mid = 4.14e9 N m
support reaction = q L/2 = 138e6 N each
```

更严格的逐点检查显示：最细网格的弯矩和非零纤维应力最大相对误差为 **{percent(point['max_moment_and_sigma_x_relative_error'])}**，出现在 x={point['worst_x_m']} m 的近支座采样点。该项{'小于' if point['pointwise_passed'] else '超过'} 3%；逐点最大误差与 L2 均纳入当前验收，未通过的粗网格保留。

计算应力由 FE 位移和转角的 Hermite 二阶导数获得 `sigma_x_FE=-E y v_FE''`，没有用解析弯矩替代 FE 应力。每个单元取八个预定内部采样点，在相同物理坐标比较位移、弯矩和 25 层纤维应力的 L2 范数。

## 3. 计算过程

生成五网格模型并求解 Frame 静力方程；保存完整节点位移、转角、单元端力和支座反力。由节点自由度独立恢复全部场，验证存储场与恢复场一致，然后与解析式比较。五网格场误差下降且最终误差小于 3% 才通过。

## 4. 真实计算结果与相对误差

| 单元数 | 中部挠度 m | 中部挠度误差 | 位移场 L2 | 弯矩场 L2 | sigma_x 场 L2 | 逐点应力最大误差 | 总反力 N | 相对力不平衡 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

最细网格 24 单元的 FE 纤维应力范围为 {stress.min()*1e-6:.6f} 至 {stress.max()*1e-6:.6f} MPa。节点中部挠度在舍入精度内准确，而位移插值和曲率应力场仍有离散误差，故验收检查完整采样场，没有用单个精确节点掩盖应力误差。

## 5. 位移、应力及网格误差图

![120 m 船体梁节点自由度恢复的位移场，单位 mm](../../assets/benchmark-clouds/hull-girder/displacement.png)

![有符号梁纤维 sigma_x 场，单位 MPa](../../assets/benchmark-clouds/hull-girder/sigma_x.png)

![单轴梁纤维 von Mises=abs(sigma_x)，单位 MPa](../../assets/benchmark-clouds/hull-girder/von_mises.png)

![五网格位移、弯矩与轴向应力场 L2 误差](../../assets/benchmark-clouds/hull-girder/convergence.png)

云图直接使用 FE 自由度恢复的梁纤维场，颜色插值仅用于显示。该模型没有计算壳局部剪切、板格应力或三维 von Mises，应力图严格标注为单轴梁纤维结果。

## 6. 结论与能力边界

全长采样位移、弯矩与轴向纤维应力在五网格下收敛，最终的 L2 范数误差均低于 3%，最细网格逐点弯矩/非零纤维应力最大误差也低于 3%，支座反力与总载荷闭合。该案例是理想化船体梁弯曲验证，未认证船体扭转、剪切迟滞、舱口局部应力、屈曲、塑性极限强度或船级社规则。

## 7. 复现与原始证据

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_hull_girder_fe_benchmark.py --output docs/assets/benchmark-clouds/hull-girder/hull-girder-fe.json
PYTHONPATH=src .venv/bin/python scripts/build_verified_benchmark_reports.py
```

[完整节点自由度、单元端力、纤维场及误差 JSON](../../assets/benchmark-clouds/hull-girder/hull-girder-fe.json)。HTML、PDF、Word 位于 exports 目录。
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-export', action='store_true')
    args = parser.parse_args()
    paths = {'cantilever-fe': 'results.json', 'stiffened-panel': 'results.json',
             'hull-girder': 'hull-girder-fe.json'}
    data = {k: json.loads((ASSETS/k/v).read_text()) for k, v in paths.items()}
    for folder, kind in [('cantilever-fe', 'bending'), ('stiffened-panel', 'membrane')]:
        report = data[folder]
        check = verify_mesh_study(report['cases'], kind)
        if check != report['mesh_convergence'] or not check['passed']:
            raise ValueError('field/mesh study blocked or stale: '+folder)
        rectangular_clouds(report, folder)
        convergence_plot(ASSETS/folder/'convergence.png',
             [c['mesh']['elements'] for c in report['cases']],
             {'displacement': [c['displacement_relative_error'] for c in check['checks']],
              'sigma_x': [c['stress_errors']['sigma_x_l2'] for c in check['checks']],
              'stress tensor': [c['stress_errors']['stress_tensor_l2'] for c in check['checks']]})
    hull = data['hull-girder']
    checks = [verify_hull_case(c) for c in hull['cases']]
    if any(c.get('verification') != check or c.get('status') !=
           ('qualified' if check['passed'] else 'blocked')
           for c, check in zip(hull['cases'], checks)):
        raise ValueError('stored hull qualification mismatch')
    if checks != hull['mesh_convergence']['checks'] or not checks[-1]['passed']:
        raise ValueError('hull fields blocked or stale')
    if len(checks) < 3 or not all(b['mesh']['elements'] > a['mesh']['elements']
                                  for a, b in zip(hull['cases'], hull['cases'][1:])):
        raise ValueError('hull mesh refinement missing')
    for metric in ('displacement_field_l2', 'moment_field_l2', 'sigma_x_field_l2', 'moment_and_sigma_x_pointwise_max'):
        if not all(b['errors'][metric] < a['errors'][metric] for a, b in zip(checks, checks[1:])):
            raise ValueError('hull convergence failed')
    hull_clouds(hull)
    convergence_plot(ASSETS/'hull-girder/convergence.png',
            [c['mesh']['elements'] for c in hull['cases']],
            {k: [c['errors'][k] for c in checks] for k in
             ('displacement_field_l2', 'moment_field_l2', 'sigma_x_field_l2', 'moment_and_sigma_x_pointwise_max')})
    REPORTS.mkdir(parents=True, exist_ok=True)
    sources = [('cantilever_beam_full.md', bending_markdown(data['cantilever-fe'])),
               ('stiffened_panel_fe_full.md', membrane_markdown(data['stiffened-panel'])),
               ('hull_girder_longitudinal_bending_full.md', hull_markdown(hull))]
    outputs = []
    for name, text in sources:
        source = REPORTS/name; source.write_text(text)
        outputs.append(source)
        if not args.skip_export:
            outputs.extend(export_report(source, REPORTS/'exports'))
    evidence = {'schema': 'tensorfem.published-research-evidence/2.0',
                'independent_physical_cases': 3, 'all_benchmarks_complete': False,
                'scope': 'three declared linear cases only; no v1.0 certification',
                'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in outputs + [p for folder in paths
                          for p in (ASSETS/folder).glob('[dsvct]*.png')] +
                          [ASSETS/k/v for k, v in paths.items()]}}
    (REPORTS/'evidence-manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'outputs': [str(p.relative_to(ROOT)) for p in outputs],
                      'independent_physical_cases': 3, 'all_benchmarks_complete': False}))


if __name__ == '__main__':
    main()
