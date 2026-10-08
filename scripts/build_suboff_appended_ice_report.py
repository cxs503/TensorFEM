#!/usr/bin/env python
"""Build reproducible appended SUBOFF/ice figures, fields and report exports."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from tensorfem.research_report_export import export_report
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'docs/assets/suboff-ice-v2'
DOC=ROOT/'docs/benchmarks/tutorials/suboff_appended_ice_simulation.md'
NAMES=('baseline','mesh-24','time-half','elastic','open-water','floating-precracked')

def vtk(path,r,frame):
    x=np.array(r['nodes'])+np.array(frame['dofs'])[:,:3]
    cells=r['vehicle_elements']+r['ice_elements'];labels=r['element_component_ids']+[-1]*len(r['ice_elements'])
    lines=['# vtk DataFile Version 3.0','SUBOFF appended initial ice impact; displacement scale 1','ASCII','DATASET UNSTRUCTURED_GRID',f'POINTS {len(x)} double']
    lines+=[' '.join(map(str,p)) for p in x];lines +=[f'CELLS {len(cells)} {5*len(cells)}']
    lines+=['4 '+' '.join(map(str,c)) for c in cells];lines +=[f'CELL_TYPES {len(cells)}']+['9']*len(cells)
    lines +=[f'CELL_DATA {len(cells)}','SCALARS component int 1','LOOKUP_TABLE default']+list(map(str,labels))
    if frame['time_s']==r['history'][-1]['time_s']:
        vm=np.array(r['final_von_mises_top_bottom_Pa']).max(axis=(1,2))
        lines +=['SCALARS final_gauss_von_mises_Pa double 1','LOOKUP_TABLE default']+list(map(str,vm))
    lines +=[f'POINT_DATA {len(x)}','VECTORS displacement double']+[' '.join(map(str,p[:3])) for p in frame['dofs']]
    path.write_text('\n'.join(lines)+'\n')

def build():
    cases={n:json.loads((A/(n+'.json')).read_text()) for n in NAMES};r=cases['baseline']
    x=np.array(r['outer_vehicle_nodes']);el=np.array(r['vehicle_elements']);lab=np.array(r['element_component_ids'])
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(111,projection='3d')
    for cid,c in enumerate(r['components']):
        ax.add_collection3d(Poly3DCollection(x[el[lab==cid]],facecolor=plt.cm.tab10(cid),edgecolor='#333',linewidth=.15,alpha=.85,label=c['name']))
    ax.set(xlim=(0,4.356),ylim=(-.55,.55),zlim=(-.55,.6),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='SUBOFF equivalent shell midsurfaces: welded sail and four fins')
    ax.set_yticks([-.5,0,.5]);ax.tick_params(labelsize=8)
    ax.set_box_aspect((4.356,1.1,1.15));ax.view_init(20,-65);ax.legend(fontsize=7,loc='upper left');fig.tight_layout();fig.savefig(A/'geometry.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(111,projection='3d')
    vm=np.array(r['peak_contact_von_mises_top_bottom_Pa'])[:len(el)].max(axis=(1,2))/1e6
    norm=plt.Normalize(0,float(vm.max()))
    ax.add_collection3d(Poly3DCollection(x[el],facecolors=plt.cm.viridis(norm(vm)),edgecolor='#444',linewidth=.1))
    ax.set(xlim=(0,4.356),ylim=(-.55,.55),zlim=(-.55,.6),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Vehicle stress at peak contact: four Gauss points and both skins')
    ax.set_yticks([-.5,0,.5]);ax.tick_params(labelsize=8);ax.set_box_aspect((4.356,1.1,1.15));ax.view_init(20,-65)
    ax.set_zlabel('z (m)',labelpad=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,pad=.13,shrink=.6,label='von Mises (MPa)')
    fig.tight_layout();fig.savefig(A/'stress.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for name in NAMES:
        h=cases[name]['history'];t=np.array([p['time_s'] for p in h])*1000
        axs[0,0].plot(t,[p['contact_force_N']/1000 for p in h],label=name)
    axs[0,0].set(ylabel='Contact force (kN)');axs[0,0].legend(fontsize=7)
    h=r['history'];t=np.array([p['time_s'] for p in h])*1000
    for k in ('kinetic_J','shell_strain_J','cohesive_stored_J','new_fracture_dissipation_J'):
        axs[0,1].plot(t,[p[k] for p in h],label=k[:-2])
    axs[0,1].set(ylabel='Energy (J)');axs[0,1].legend(fontsize=7)
    for name in ('baseline','floating-precracked'):
        h=cases[name]['history'];t=np.array([p['time_s'] for p in h])*1000
        axs[1,0].plot(t,[p['fully_failed_points'] for p in h],label=name)
        axs[1,1].plot(t,[p['ice_component_count'] for p in h],label=name)
    axs[1,0].set(ylabel='Fully failed cohesive points');axs[1,1].set(ylabel='Connected ice components')
    for ax in axs.flat:ax.set_xlabel('Time (ms)');ax.grid(alpha=.25)
    axs[1,0].legend(fontsize=7);axs[1,1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(A/'history.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axs,('open-water','floating-precracked')):
        q=cases[name];nodes=np.array(q['nodes']);cells=np.array(q['ice_elements']);h=np.array(q['ice_cell_thickness_m'])*1000
        from matplotlib.collections import PolyCollection
        polygons=nodes[cells][:,:,:2];pc=PolyCollection(polygons,array=h,cmap='Blues',edgecolors='#777',linewidths=.5);ax.add_collection(pc)
        ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x (m)',ylabel='y (m)',title=name+'; initial thickness and gaps')
        if name=='floating-precracked':ax.axhline(0,color='red',label='Prescribed initial crack');ax.legend(fontsize=8)
        fig.colorbar(pc,ax=ax,label='Ice thickness (mm)',shrink=.6)
    fig.tight_layout();fig.savefig(A/'ice-layout.png',dpi=160,bbox_inches='tight');plt.close(fig)
    for i,frame in enumerate(r['snapshots']):vtk(A/f'frame-{i:03d}.vtk',r,frame)
    (A/'impact.pvd').write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>\n'+''.join(f'<DataSet timestep="{f["time_s"]}" file="frame-{i:03d}.vtk"/>\n' for i,f in enumerate(r['snapshots']))+'</Collection></VTKFile>\n')
    m=r['metrics'];mesh=cases['mesh-24']['metrics'];half=cases['time-half']['metrics']
    delta=lambda a,b:abs(a/b-1)*100
    sensitivity={'mesh_peak_force_change_percent':delta(mesh['peak_contact_force_N'],m['peak_contact_force_N']), 'mesh_peak_stress_change_percent':delta(mesh['maximum_vehicle_gauss_von_mises_Pa'],m['maximum_vehicle_gauss_von_mises_Pa']), 'time_half_peak_force_change_percent':delta(half['peak_contact_force_N'],m['peak_contact_force_N']), 'time_half_peak_stress_change_percent':delta(half['maximum_vehicle_gauss_von_mises_Pa'],m['maximum_vehicle_gauss_von_mises_Pa']), 'acceptance_percent':3.,'physical_accuracy_qualified':False,'mesh_converged':False}
    rows=[]
    for n,q in cases.items():
        z=q['metrics'];rows.append(f"| {n} | {len(q['vehicle_elements'])} | {z['peak_contact_force_N']/1000:.4f} | {z['maximum_vehicle_gauss_von_mises_Pa']/1e6:.4f} | {z['initial_failed_points']}→{z['final_failed_points']} | {q['history'][0]['ice_component_count']}→{z['final_ice_component_count']} | {z['maximum_energy_relative_error']:.2e} |")
    text=f'''# 带附体 SUBOFF 与可配置冰层：初始上浮撞击

## 状态与适用范围

这是实际执行并保存全场数据的实验功能。六个算例的几何、能量、竖直动量、接触预期、质量保留和小变形检查均通过；物理精度未认证，尚未证明网格和罚接触参数收敛，也不是完整上浮破冰模拟。此前裸艇案例和其未收敛记录保留。

## 几何、结构与参数

数值几何取自本地 TensorLBM 的 SUBOFF CAD，来源提交和源文件 SHA256 见 appended-16.json / appended-24.json。艇长 4.356 m、直径约 0.5083 m；新增围壳和四片十字尾舵，根部共享艇壳全部六个自由度。艇壳为闭合压力壳；附体是人为构建的等效中面薄片，保留 CAD 围壳顶部轮廓和尾舵后掠平面形状，不能视为完整 NACA 厚翼外皮或真实 SUBOFF 内部结构。围壳顶部接触面积按 CAD 宽度积分。

钢壳和附体厚 3 mm，E=210 GPa，ν=0.3，ρ=7850 kg/m³；钢材质量 {m['steel_mass_kg']:.6f} kg，加声明的设备/压载使总质量为 700 kg。上浮速度 0.3 m/s，初始顶部间隙 0.05 mm，积分时长 1.5 ms。无质量缩放，旋转惯性按物理厚度计算。基准冰厚 10 mm，E=5 GPa、ν=0.3、ρ=900 kg/m³，缝抗拉/剪强度 0.5 MPa，断裂能 5 J/m²。它们是自洽的人工研究参数，未标定海冰试验。

![附体等效壳模型](../../assets/suboff-ice-v2/geometry.png)

## 新增冰模拟功能

- 独立 Q4 冰片和双线性不可逆黏聚缝；受压缝仍提供弹性抗力。
- 每格冰厚可变；厚度为零或 open_water_cells 指定开水区。厚度阶跃缝在同一实际高度积分，保持刚体旋转相容。
- precracked_pairs 指定原始网格相邻编号之间的初始裂缝；初始断裂能单独记录，不冒充新产生的断裂耗散。
- clamped、simply_supported（仅约束竖直位移）、free 三种外围边界；cohesive 或 elastic 两种冰模式。
- 非均匀冰厚按同一水面计算各自吃水与干舷；基于浮动平衡的重力/静水恢复力，浸没高度截断到 [0,h] 并配套连续势能；默认水密度 1025 kg/m³。
- 可声明每面积附加水质量和线性阻力；阻力使用精确指数半步并记录耗散与外部冲量。这不是求解流体运动的 FSI。
- 只有一道缝所有厚度积分点均失效才断开连通性；输出各碎片网格编号、质量、中心、平均速度和动能。断裂不删除质量。

![开水区与预裂缝非均匀冰厚](../../assets/suboff-ice-v2/ice-layout.png)

floating-precracked 的冰厚为 8/12 mm，整条中线预裂缝使其初始就有两块冰，附加水质量系数 10 kg/m²、阻力系数 20 kg/(m²·s)。该算例的两块碎片是预设初始状态，不能当作计算中新形成的贯穿裂缝。

## 保存结果

| 算例 | 结构单元 | 峰值接触 kN | 全时程最大钢材 VM MPa | 失效积分点 初→终 | 冰连通块 初→终 | 最大能量相对误差 |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

基准峰值力 {m['peak_contact_force_N']/1000:.4f} kN，接触发生在围壳顶部。艇壳和尾舵有非零应力，说明附体真实参与传力。开水区对照无接触，弹性冰对照无断裂耗散。所有算例原始结果均有 SHA256 来源记录，独立重组验证器核对最终应力、能量、动量、损伤、质量和碎片。

![峰值接触时钢材应力](../../assets/suboff-ice-v2/stress.png)

![接触、能量、失效与连通性时程](../../assets/suboff-ice-v2/history.png)

## 收敛检查与边界

时间步减半后峰值力变化 {sensitivity['time_half_peak_force_change_percent']:.6f}%，最大应力变化 {sensitivity['time_half_peak_stress_change_percent']:.6f}%。结构网格加密后峰值力变化 {sensitivity['mesh_peak_force_change_percent']:.4f}%，最大应力变化 {sensitivity['mesh_peak_stress_change_percent']:.4f}%。对照门槛为 3%；只有两级结构网格，且接触节点采样随加密变化，仍不足以认证网格收敛。罚接触独立敏感性和外部试验/独立求解器对照尚未完成。

当前采用小应变、固定水平位置的竖直接触注册，没有冰压碎、应变率效应、一般碎片再接触/自接触和有限转动破冰。基准冰最终仍连成一块，不能宣称艇体已穿透冰层。若延长时间或改变参数，应检查结果中的 blocked 状态，不能忽略小变形门限。

## 重现与文件

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/suboff_appended_ice_simulation.py --config docs/assets/suboff-ice-v2/configs/baseline.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/validate_suboff_ice_simulation.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/build_suboff_appended_ice_report.py
```

configs/ 下保存全部参数；appended-16.json / appended-24.json 保存结构几何。结果 JSON 包含所有最终自由度、速度、应力、损伤、时程及五帧位移；impact.pvd 和 frame-*.vtk 可在 ParaView 打开，位移比例为 1。PNG 几何图显示等效中面，未放大位移。

[HTML](exports/suboff_appended_ice_simulation.html) · [PDF](exports/suboff_appended_ice_simulation.pdf) · [Word](exports/suboff_appended_ice_simulation.docx)
'''
    DOC.write_text(text);export_report(DOC,DOC.parent/'exports')
    study={'schema':'tensorfem.suboff-ice-study/2','physical_accuracy_qualified':False,'sensitivity':sensitivity,'cases':[{'name':n,'path':str((A/(n+'.json')).relative_to(ROOT)),'status':q['status']} for n,q in cases.items()]}
    (A/'study.json').write_text(json.dumps(study,indent=2)+'\n')
    files=list(A.glob('*.json'))+list(A.glob('*.png'))+list(A.glob('*.vtk'))+[A/'impact.pvd',DOC]+list((DOC.parent/'exports').glob(DOC.stem+'.*'))+list((A/'configs').glob('*.json'))
    manifest={'schema':'tensorfem.suboff-ice-report/2','physical_accuracy_qualified':False,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='report-manifest.json'}}
    (A/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(study,indent=2))

if __name__=='__main__':build()
