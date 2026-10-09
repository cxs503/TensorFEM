#!/usr/bin/env python
"""Render audited initial upward-impact evidence in four report formats."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from tensorfem.research_report_export import export_report

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets/suboff-ice'
DOC=ROOT/'docs/benchmarks/tutorials/suboff_upward_ice_collision.md'


def figure_surface(path,r):
    x=np.array(r['nodes']);u=np.array(r['final_dofs'])[:,:3]
    hull=np.array(r['hull_elements']);ice=np.array(r['ice_elements'])
    vm=np.array(r['peak_contact_von_mises_top_bottom_Pa'])[:len(hull)].max(axis=(1,2))/1e6
    fig=plt.figure(figsize=(13,6))
    ax=fig.add_subplot(121,projection='3d')
    norm=plt.Normalize(0,float(vm.max()))
    ax.add_collection3d(Poly3DCollection(x[hull],facecolors=plt.cm.viridis(norm(vm)),linewidths=.1,edgecolors='#444'))
    ax.set(xlim=(x[:,0].min(),x[:,0].max()),ylim=(-.6,.6),zlim=(-.4,.4),
           title='Hull von Mises at peak contact (MPa)\nMaximum over four Gauss points and both skins')
    ax.set_box_aspect((5,1.2,.8));fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,shrink=.5,pad=.02,fraction=.035)
    ax=fig.add_subplot(122,projection='3d')
    deform=x+20*u
    ax.add_collection3d(Poly3DCollection(deform[ice],facecolor='#b8e1f0',edgecolor='#555',linewidths=.4,alpha=.7))
    pairs=np.array(r['cohesive_node_pairs']);damage=np.array(r['cohesive_damage'])
    for seam in sorted(set(r['cohesive_seam_ids'])):
        ids=np.where(np.array(r['cohesive_seam_ids'])==seam)[0]
        endpoints=pairs[ids[[0,2]],0]
        p=deform[endpoints];d=float(damage[ids].max())
        ax.plot(p[:,0],p[:,1],p[:,2],color=plt.cm.inferno(d),linewidth=1+2*d)
    ax.set(xlim=(x[:,0].min(),x[:,0].max()),ylim=(-.6,.6),
           zlim=(x[:,2].max()-.01,x[:,2].max()+.09),title='Final ice deformation x20\nSeam colour: maximum cohesive damage (0-1)')
    ax.set_box_aspect((5,1.2,.5));fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='inferno'),ax=ax,shrink=.5,pad=.02,fraction=.035)
    for axes in fig.axes:
        axes.tick_params(labelsize=7)
        axes.title.set_fontsize(10)
        if hasattr(axes,'set_zticks'):
            axes.set_yticks([-.6,0,.6])
            lo,hi=axes.get_zlim();axes.set_zticks([lo,(lo+hi)/2,hi])
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def write_vtk(path,r,frame):
    x=np.array(r['nodes']);u=np.array(frame['dofs'])[:,:3];e=r['hull_elements']+r['ice_elements']
    out=['# vtk DataFile Version 3.0','SUBOFF ice impact; physical displacement scale 1',
         'ASCII','DATASET UNSTRUCTURED_GRID',f'POINTS {len(x)} double']
    out+=[' '.join(map(str,p)) for p in x+u]
    out+=[f'CELLS {len(e)} {5*len(e)}']+['4 '+' '.join(map(str,c)) for c in e]
    out+=[f'CELL_TYPES {len(e)}']+['9']*len(e)
    out+=[f'POINT_DATA {len(x)}','VECTORS displacement_m double']+[' '.join(map(str,p)) for p in u]
    if frame['time_s']==r['history'][-1]['time_s']:
        vm=np.array(r['final_von_mises_top_bottom_Pa']).max(axis=(1,2))
        out+=[f'CELL_DATA {len(e)}','SCALARS von_mises_gauss_max_Pa double 1','LOOKUP_TABLE default']+list(map(str,vm))
    path.write_text('\n'.join(out)+'\n')


def fracture_topology(r):
    ids=np.array(r['cohesive_seam_ids']);damage=np.array(r['cohesive_damage'])
    pairs=np.array(r['cohesive_node_pairs']);nh=len(r['outer_hull_nodes'])
    count=len(r['ice_elements']);adj=[set() for _ in range(count)];severed=0
    for sid in sorted(set(ids)):
        at=np.where(ids==sid)[0]
        a,b=(pairs[at[0]]-nh)//4
        if np.all(damage[at]>=.999):
            severed+=1
        else:
            adj[a].add(b);adj[b].add(a)
    unseen=set(range(count));components=[]
    while unseen:
        first=unseen.pop();todo=[first];component=[first]
        while todo:
            a=todo.pop()
            for b in adj[a]:
                if b in unseen:
                    unseen.remove(b);todo.append(b);component.append(b)
        components.append(component)
    return {'fully_severed_seams':severed,'ice_connected_components':len(components),
            'component_tile_counts':sorted(map(len,components),reverse=True)}


def build():
    study=json.loads((ASSETS/'study.json').read_text())
    r=json.loads((ASSETS/'hull-32.json').read_text());h=r['history'];m=r['metrics'];c=r['config']
    t=np.array([x['time_s'] for x in h])*1000
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    axs[0,0].plot(t,[x['contact_force_N']/1000 for x in h]);axs[0,0].set(ylabel='Contact force (kN)')
    for key in ('kinetic_J','shell_strain_J','cohesive_stored_J','contact_stored_J','fracture_dissipation_J'):
        axs[0,1].plot(t,[x[key] for x in h],label=key.removesuffix('_J'))
    axs[0,1].legend(fontsize=7);axs[0,1].set(ylabel='Energy (J)')
    axs[1,0].plot(t,[x['maximum_damage'] for x in h],label='Maximum damage')
    axs[1,0].plot(t,[x['failed_points'] for x in h],label='Failed cohesive points');axs[1,0].legend(fontsize=8)
    axs[1,1].plot(t,[x['hull_vertical_velocity_m_s'] for x in h]);axs[1,1].set(ylabel='Hull mean upward speed (m/s)')
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ASSETS/'history.png',dpi=160);plt.close(fig)
    figure_surface(ASSETS/'fields.png',r)
    fig,ax=plt.subplots(figsize=(8,4))
    for entry in study['cases'][:3]:
        raw=json.loads((ROOT/entry['path']).read_text());hist=raw['history']
        ax.plot([x['time_s']*1000 for x in hist],[x['contact_force_N']/1000 for x in hist],label=entry['name'])
    ax.set(xlabel='Time (ms)',ylabel='Contact force (kN)',title='Hull mesh sensitivity: retained numerical evidence')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(ASSETS/'mesh-force.png',dpi=160);plt.close(fig)
    for i,frame in enumerate(r['snapshots']):write_vtk(ASSETS/f'frame-{i:03d}.vtk',r,frame)
    vtk_manifest=['<?xml version="1.0"?>','<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>']
    vtk_manifest += [f'<DataSet timestep="{frame["time_s"]}" file="frame-{i:03d}.vtk"/>' for i,frame in enumerate(r['snapshots'])]
    (ASSETS/'impact.pvd').write_text('\n'.join(vtk_manifest+['</Collection></VTKFile>'])+'\n')
    topology={entry['name']:fracture_topology(json.loads((ROOT/entry['path']).read_text())) for entry in study['cases']}
    (ASSETS/'fracture-topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    parameter_rows=[('艇长 / 最大直径',f"4.356 m / {2*max(np.linalg.norm(np.array(r['outer_hull_nodes'])[:,1:],axis=1)):.6f} m"),
        ('钢壳厚度 / E / ν / 密度','3 mm / 210 GPa / 0.3 / 7850 kg/m³'),
        ('总质量 / 初始速度','700 kg / 0.3 m/s，自由初始速度'),
        ('钢壳 / 设备和压载质量',f"{m['steel_shell_mass_kg']:.6f} kg / {m['attached_equipment_and_ballast_mass_kg']:.6f} kg"),
        ('冰板尺寸 / 厚度','4.956 × 1.2 m / 10 mm'),
        ('冰 E / ν / 密度','5 GPa / 0.3 / 900 kg/m³'),
        ('接缝峰值强度 / 断裂能','0.5 MPa / 5 J/m²，人为设定'),
        ('初始艇—冰间隙 / 时长','0.05 mm / 6 ms'),
        ('边界 / 外力','冰板外周固支；艇体自由；无外部驱动力'),
        ('初始能量 / 动量',f"{m['initial_kinetic_energy_J']:.6f} J / {m['initial_vertical_momentum_kg_m_s']:.6f} kg·m/s")]
    rows='\n'.join(f'| {a} | {b} |' for a,b in parameter_rows)
    table='\n'.join(f"| {e['name']} | {e['metrics']['peak_contact_force_N']/1000:.6f} | {e['metrics']['maximum_hull_gauss_von_mises_Pa']/1e6:.6f} | {e['metrics']['fracture_dissipation_J']:.6f} | {e['metrics']['maximum_energy_relative_error']*100:.6g}% | {topology[e['name']]['fully_severed_seams']} |" for e in study['cases'])
    comparisons='\n'.join(f"| {x['from']} → {x['to']} | "+' | '.join(f"{v*100:.5f}%" for v in x['relative_changes'].values())+f" | {'通过' if x['all_below_3_percent'] else '未通过'} |" for x in study['comparisons'])
    text=f'''# SUBOFF 人工壳体向上撞冰：初始冲击与黏聚裂损研究案例

## 状态与用途

这是可复现的模型尺度初始上浮撞冰案例。几何来自 TensorLBM，钢壳、设备/压载、冰材料和接缝均为人为定义。当前物理精度未认证；没有破冰实验或独立软件参考解。完整出水、真实海冰承载力和极限冰厚不能由本报告确定。

六组计算的数值敏感性全指标 3% 门槛：**{'通过' if study['all_numerical_sensitivity_gates_passed'] else '未通过，仍需优化'}**。失败比较保留在下表中。

## 参数与质量自洽

| 项目 | 数值 |
|---|---|
{rows}

最细裸艇体外表面封闭体积 {r['geometry_audit']['enclosed_volume_m3']:.6f} m³，对应假设水密度 1025 kg/m³ 时约 {r['geometry_audit']['enclosed_volume_m3']*1025:.3f} kg 排水质量；700 kg 总质量处于相近量级。设备和压载按壳节点面积分布，附着在壳体上。计算保留物理壳厚转动惯量，没有为放大时间步进行质量缩放。这组参数定义一个假想模型，不对应真实潜艇或特定海冰试验。

## 几何和离散方法

复用 TensorLBM `src/tensorlbm/suboff_cad.py` 的 `suboff_radius_profile`，原始源码 SHA256：`{r['geometry_source']['file_sha256']}`；来源提交：`{r['geometry_source']['commit']}`。只导出数值节点和连接关系，TensorFEM 求解无需安装 TensorLBM。艇首、艇尾采用方形映射到圆盘的全四边形封帽，所有节点均核对原几何半径函数。

外几何为 CAD 表面；沿面积加权法向向内偏移半个钢壳厚度得到壳参考面，接触使用外几何高度及壳平移。闭合性、边方向、正面积、连通性及 Euler 特征数 2 均检查。32 级网格为 {len(r['outer_hull_nodes'])} 个艇体节点、{len(r['hull_elements'])} 个壳单元。

采用投影 Q4 膜—Mindlin 壳及稀疏刚度。冰板每片有独立节点；相邻片通过两个端点、两个厚度 Gauss 点的黏聚接缝相连，裂损后节点能够分离。没有删单元、删质量或依据绘图生成裂纹。初始等效开口刚度为 `cohesive_factor × E_ice / min(dx,dy)`。接缝使用相同强度的拉伸/剪切等效开口，受压部分保持弹性；该假设不包含海冰的压碎、应变率或各向异性。

艇节点与冰片通过竖向无摩擦罚接触耦合，形函数将反力分配给四个冰节点，作用力与反作用力严格相反。罚刚度为 `contact_factor × E_ice / h_ice × 节点面积`。接触对象始终为可变形冰片。水平接触归属固定，因此适用范围是短时、主要竖向的冲击。

## 时间积分、断裂能和验收

使用速度 Verlet 积分；初始上浮速度随后由质量、壳刚度、接触和裂损共同决定。时间步取无损壳、全部接缝、全部接触刚度的质量归一化绝对行和上界；还包含黏聚软化斜率界。最细网格名义步长 {r['solver']['time_step_s']:.9g} s，共 {r['solver']['steps']} 步。

不可逆最大开口控制卸载刚度。失效开口 `δf = 2 Gc / σc`；接缝耗散由双线性包络积分减去可恢复能量计算。每一步检查总动能、壳应变能、黏聚储能、接触储能及断裂耗散之和。竖向动量与冰板固支边界反力的梯形积分核对。

应力由位移和转角恢复，每步检查所有壳单元四个平面 Gauss 点的上、下表面平面应力。表中 von Mises 不包含横向剪切。355 MPa 只是声明的弹性适用性参考，未接入钢壳塑性。

![接触力、能量、接缝损伤和上浮速度](../../assets/suboff-ice/history.png)

![壳体应力与冰板裂损](../../assets/suboff-ice/fields.png)

右图形变放大 20 倍，损伤色标 0–1；VTK 使用真实尺度。接缝局部达到损伤 1 与整条接缝完全断开分别计数。名义最细网格完全断开的接缝 {topology['hull-32']['fully_severed_seams']} 条，冰片连通分量 {topology['hull-32']['ice_connected_components']} 个。该短时结果表示裂损起始和发展，尚未证明艇体穿透冰层并完成出水。

## 六组计算

| 计算 | 峰值接触力 kN | 全时程钢壳 Gauss 最大 VM MPa | 断裂耗散 J | 最大能量收支误差 | 完全断开接缝 |
|---|---|---|---|---|---|
{table}

艇壳网格 16/24/32；名义冰板网格 12×4；时间步减半、接触罚刚度加倍、冰板 24×8 单独复核。冰板加密同时改变预设裂纹路径网格，不能解释为仅改变单元数的连续体收敛试验。

| 比较 | 峰值接触力变化 | 断裂耗散变化 | 冰挠度变化 | 钢壳 VM 变化 | 全部 <3% |
|---|---|---|---|---|---|
{comparisons}

![三组艇体网格接触力](../../assets/suboff-ice/mesh-force.png)

节点罚接触、壳面离散和预设裂纹网络的网格敏感性均参与结果。能量守恒证明了当前离散模型的收支一致性，不能替代碰撞力与断裂预测的物理精度验证。

## 可复现命令与原始字段

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/suboff_upward_ice_collision.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/run_suboff_ice_study.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/validate_suboff_ice_evidence.py
PYTHONPATH=src .venv/bin/python scripts/build_suboff_ice_report.py
```

通过 `--config` 读取 SI 单位参数 JSON，也可用 `--speed`、`--duration`、`--time-safety`。变更参数会重新检查稳定步长和适用性。

- [研究矩阵与敏感性 JSON](../../assets/suboff-ice/study.json)
- [默认参数 JSON](../../assets/suboff-ice/default-config.json)
- [最细网格原始字段](../../assets/suboff-ice/hull-32.json)
- [闭合外几何与溯源](../../assets/suboff-ice/geometry-32.json)
- [裂损连通性 JSON](../../assets/suboff-ice/fracture-topology.json)
- [ParaView 时间序列](../../assets/suboff-ice/impact.pvd)

下一阶段需要控制接触几何的网格敏感性，并明确海水附加质量、浮力/推进与冰板支撑。真实破冰预测还需要海冰材料标定、独立参考计算、大变形接触、压碎和碎冰再接触。
'''
    DOC.write_text(text)
    export_report(DOC,DOC.parent/'exports')
    paths=[DOC,*ASSETS.glob('*.png'),*ASSETS.glob('*.vtk'),ASSETS/'impact.pvd',ASSETS/'study.json',ASSETS/'fracture-topology.json',ASSETS/'default-config.json']
    paths+=list((DOC.parent/'exports').glob(DOC.stem+'.*'))
    manifest={'schema':'tensorfem.suboff-ice-report/1','physical_accuracy_qualified':False,
              'artifacts':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ASSETS/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'report':str(DOC),'numerical_sensitivity_passed':study['all_numerical_sensitivity_gates_passed'],'fracture_topology':topology},indent=2))


if __name__=='__main__':build()
