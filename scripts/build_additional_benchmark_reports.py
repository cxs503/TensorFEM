#!/usr/bin/env python
"""Render four additional real FE cases and export the same narrative in 4 formats."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tensorfem.research_report_export import export_report
from audit_pointwise_benchmark_errors import audit

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets/benchmark-clouds'
DOCS=ROOT/'docs/benchmarks/tutorials'
TITLES={'sphere-pressure':'三维厚球壳内压：HEX8 与 Lamé 精确解',
        'mindlin-navier':'Mindlin 简支板：正弦荷载与 Navier 精确解',
        'cook-membrane':'Cook 膜：加载边中点位移与真实应力场',
        'hemisphere-hole':'18° 开孔半球壳：修正后的位移收敛与应力场'}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def cloud2(p,nodes,values,title):
    nodes=np.asarray(nodes);values=np.asarray(values)
    n=int(round(len(nodes)**.5))
    fig,ax=plt.subplots(figsize=(7,5))
    c=ax.contourf(nodes[:,0].reshape(n,n),nodes[:,1].reshape(n,n),values.reshape(n,n),24,cmap='viridis')
    fig.colorbar(c,ax=ax);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x');ax.set_ylabel('y')
    fig.tight_layout();fig.savefig(p,dpi=150);plt.close(fig)

def cloud3(p,nodes,cells,values,title,nodal=False):
    x=np.asarray(nodes);e=np.asarray(cells);v=np.asarray(values)
    faces=x[e]; colors=v[e].mean(1) if nodal else v
    # Expose the inner volume through a geometric cutaway for the full sphere.
    if 'sphere-pressure' in str(p):
        keep=faces.mean(1)[:,0]<=.5
        faces,colors=faces[keep],colors[keep]
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(projection='3d')
    norm=plt.Normalize(float(v.min()),float(v.max()))
    poly=Poly3DCollection(faces,facecolors=plt.cm.viridis(norm(colors)),linewidths=0)
    ax.add_collection3d(poly)
    lim=np.max(np.abs(x));ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim),title=title)
    ax.set_box_aspect((1,1,1));fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,shrink=.7)
    fig.tight_layout();fig.savefig(p,dpi=150);plt.close(fig)


def build():
    manifest={'schema':'tensorfem.additional-reports/1.0','all_benchmarks_complete':False,'reports':[]}
    for name,title in TITLES.items():
        folder=ASSETS/name;raw=folder/'results.json';r=json.loads(raw.read_text());c=r['cases'][-1]
        figures=[];n=np.asarray(c['nodes']);u=np.asarray(c['displacement'])
        def add(filename,caption):
            figures.append((filename,caption));return folder/(filename+'.png')
        if name=='sphere-pressure':
            from tensorfem.sphere_pressure_benchmark import cube_sphere_surface
            div=c['mesh']['face_divisions'];nr=c['mesh']['radial_layers'];surface,faces=cube_sphere_surface(div)
            faces=faces.numpy()+(len(surface)*nr)
            cloud3(add('displacement','外表面位移幅值（m）；切去一侧以显示球面），原始 FE 节点值取面均值。'),n,faces,np.linalg.norm(u,axis=1),'Outer displacement / m',True)
            cloud3(add('von_mises','外层单元中心 von Mises（Pa），颜色对应最外径向层单元。'),n,faces,c['von_mises'][-len(faces):],'Outer-layer von Mises / Pa')
            physical='内外半径 8/10 m，E=210 GPa，ν=0.3，内压 1 MPa，外压为零。完整球体使用 HEX8；三个坐标轴处的六个切向约束只消除刚体运动，径向膨胀自由。内压在离散内表面一致积分。'
            reference='独立精确解：A=p a³/(b³−a³)，B=p a³b³/(b³−a³)；σr=A−B/r³，σt=A+B/(2r³)，ur=[(1−2ν)Ar+(1+ν)B/(2r²)]/E。由球对称平衡 dσr/dr+2(σr−σt)/r=0 及内外表面 σr(a)=−p、σr(b)=0 得到；结合三维 Hooke 定律确定位移。参考解不参与装配与载荷。'
            scope='位移向量全节点 L2/最大相对误差、中心应力张量/径向/环向/von Mises 的体积加权 L2，径向最大绝对误差除以内压。最后一项避免外表面零应力造成相对误差奇异。全部声明误差须低于 3%。原有范数指标在全部网格级别下降；所有指标（含逐点最大误差）须在预定 20/32/40/64 细化段下降。8/12 粗网格的环向逐点最大误差非单调，记录保留；粗网格失败保留。应力取单元中心，没有边界峰值外推。'
            point=audit()['sphere_pressure']
            scope+=f" 逐点检查：径向应力最大相对误差 {100*point['max_radial_relative_error']:.6f}%；{'通过' if point['pointwise_radial_passed'] else '未通过'}每个径向应力采样点小于 3% 的门槛。当前验收同时检查全节点位移，以及全部单元中心应力张量、径向、环向和 von Mises 的逐点最大相对误差；粗网格超标记录保留。"
        elif name=='mindlin-navier':
            cloud2(add('displacement','FE 节点挠度 w（m）。'),n,u[:,0],'FE deflection / m')
            cloud2(add('von_mises','单元中心上表面弯曲 von Mises（Pa）。'),c['centers'],c['von_mises'],'Top bending von Mises / Pa')
            cloud2(add('shear','单元中心横向剪力 Qx（N/m）。'),c['centers'],np.asarray(c['shear'])[:,0],'Shear resultant Qx / N/m')
            physical='边长 1 m，E=10 MPa，ν=0.3，厚度 0.01 m，荷载 q=sin(πx)sin(πy) Pa。硬简支边界：w=0，切向转角=0；Q4 Mindlin 选择性积分，剪切修正系数 5/6。'
            reference='令 k=π，λ=2k²，D=Et³/[12(1−ν²)]，Ds=(5/6)Gt，Wb=q0/(Dλ²)，Ws=q0/(Dsλ)。精确 w=(Wb+Ws)sin(kx)sin(ky)，θx=−kWb cos(kx)sin(ky)，θy=−kWb sin(kx)cos(ky)。κ=(k²Wb sin sin,k²Wb sin sin,−2k²Wb cos cos)，M=Dbκ；Q=(kq0/λ)(cos sin,sin cos)，表面弯曲应力 σ=6M/t²。'
            scope='节点挠度和转角 L2，单元中心弯矩、横向剪力、上表面弯曲应力及 von Mises L2，外加总荷载积分误差。参考与 FE 使用相同坐标但独立公式。通过只覆盖 Mindlin 模型的弯曲应力与剪力，不涵盖三维厚度方向应力。'
        elif name=='cook-membrane':
            cloud2(add('displacement','FE 节点位移幅值（benchmark length）。'),n,np.linalg.norm(u,axis=1),'FE displacement magnitude')
            cloud2(add('von_mises','单元中心真实 FE von Mises；尚无独立应力误差基准。'),c['centers'],c['von_mises'],'FE von Mises (unqualified accuracy)')
            physical='四角 (0,0),(48,44),(48,60),(0,44)，E=1，ν=1/3，厚度 1，平面应力。左边全固定，右边均布向上荷载的合力为 1。完全积分 Q4，稀疏 CG。数值采用 benchmark 单位。'
            reference='项目沿用参考位移 23.96，采样位置为加载边中点 (48,52)。旧代码误用了右上角；该位置错误会在细网格上暴露。现在明确区分中点与角点，参考值未调整。当前报告未重新取得原始出版物中的数值表，外部来源溯源仍需补齐。'
            scope='仅验证加载边中点位移与细化趋势。应力来自实际 FE 位移，通过中心 B 矩阵恢复；应力精度没有独立解析或高精度参考，不能宣布全场误差达标。'
        else:
            cloud3(add('displacement','半球四分之一模型真实节点位移幅值，benchmark length。'),n,c['elements'],np.linalg.norm(u,axis=1),'Hemisphere displacement',True)
            vm=np.asarray(c['von_mises_top_bottom'])[:,0]
            cloud3(add('von_mises','局部外表面中心弯膜组合 von Mises，benchmark force/length²；应力精度未认证。'),n,c['elements'],vm,'Top shell von Mises (accuracy pending)')
            physical='R=10，t=0.04，E=6.825×10⁷，ν=0.3，18° 开孔四分之一半球，赤道两角交替径向单位力。对称面约束及一个垂直平移规范约束。基准长度/力单位保持原定义。'
            reference='MacNeal–Harder (1985)，DOI 10.1016/0168-874X(85)90003-4，沿用加载赤道点参考径向位移 0.0924。局部 Mindlin 坐标导数转换已修正为逆转置，消除了斜网格刚体转动的虚假刚度。旧 12×12 的 0.69% 误差记录作废。'
            scope='默认 drilling_factor=10⁻⁶ 的点位移收敛与平衡单独验证。表面应力由局部膜应变±tκ/2恢复；没有独立应力参考。两数量级 drilling 参数检查独立列出，失败会阻止整体稳健性资格，禁止选参数贴合参考。'
        lines=[f'# {title}','', '## 模型与求解','',physical,'','## 独立参考与验收范围','',reference,'',scope,'','## 网格收敛结果','', '| 网格 | 指标 | 值 |','|---|---|---:|']
        plotrows=[]
        for case in r['cases']:
            mesh=case['mesh']
            if isinstance(mesh,dict):label=f"{mesh['face_divisions']}×{mesh['face_divisions']}×{mesh['radial_layers']}";x=mesh['face_divisions']
            else:label=f'{mesh}×{mesh}';x=mesh
            v=case.get('verification',case.get('validation'))
            errors=v.get('errors',{'response_relative_error':v.get('response_relative_error',v.get('relative_error'))})
            plotrows.append((x,errors))
            for key,value in errors.items():lines.append(f'| {label} | {key} | {100*value:.6f}% |')
            lines.append(f"| {label} | 本网格验收 | {'通过' if v['passed'] else '未通过'} |")
        fig,ax=plt.subplots(figsize=(7,4))
        for key in plotrows[0][1]:ax.loglog([a for a,b in plotrows],[max(b[key]*100,1e-9) for a,b in plotrows],'o-',label=key)
        ax.axhline(3,color='red',ls='--',label='3% limit');ax.set(xlabel='Surface/edge divisions',ylabel='Error / %');ax.legend(fontsize=7);ax.grid(True,which='both',alpha=.3)
        fig.tight_layout();fig.savefig(add('convergence','全部网格级别的误差曲线；红线为 3% 门槛。'),dpi=150);plt.close(fig)
        v=c.get('verification',c.get('validation'))
        lines+=['','## 求解与平衡复核','',f"最细模型：{len(c['nodes'])} 个节点，{len(c['elements'])} 个单元；双精度实际 FE 位移恢复上述应力。"]
        if name=='sphere-pressure':
            lines+=['',f"{c['solver']['algorithm']}：{c['solver']['iterations']} 次迭代，自由残差 {v['free_residual_relative']:.3e}，规范约束反力 {v['gauge_reaction_relative']:.3e}；载荷合力/合矩误差 {v['applied_force_balance_relative']:.3e}/{v['applied_moment_balance_relative']:.3e}。平衡门槛 10⁻⁸。"]
        elif name=='mindlin-navier':
            lines+=['',f"弯曲采用 2×2 Gauss 积分，剪切采用中心积分；全自由 DOF 相对残差 {v['free_residual']:.3e}，反力与荷载合力误差 {v['force_balance']:.3e}；均须低于 10⁻⁸。"]
        elif name=='cook-membrane':
            lines+=['',f"Q4 使用 2×2 Gauss 完全积分。CG 容差 10⁻¹¹，实际相对残差 {v['relative_residual']:.3e}；位移探针值 {v['response']:.10f}。"]
        else:
            lines+=['',f"线性投影 Q4 壳每节点六自由度，膜与板采用选择性积分。自由残差范数 {v['free_residual_norm']:.3e}，合力相对误差 {v['normalized_force_balance_error']:.3e}，合矩相对误差 {v['normalized_moment_balance_error']:.3e}；门槛 10⁻⁷。"]
        if name=='hemisphere-hole':
            lines+=['','## 参数稳健性','', '| drilling factor（40×40） | 位移误差 | 全部验收 |','|---|---:|---|']
            for s in r['sensitivity']:lines.append(f"| {s['drilling_factor']:g} | {s['relative_error']*100:.6f}% | {'通过' if s['passed'] else '未通过'} |")
            lines+=['',f"末两级默认参数点位移变化：{r['last_response_change']*100:.6f}%。",'', '保留的 24×24 参数检查：','', '| drilling factor | 位移误差 | 验收 |','|---|---:|---|']
            for s in r['coarse_sensitivity']:
                lines.append(f"| {s['drilling_factor']:g} | {s['relative_error']*100:.6f}% | {'通过' if s['passed'] else '未通过'} |")
            lines+=['', '40×40 的全部参数通过不消除上述粗网格失败记录。']
        lines+=['','## 场结果','']
        for filename,caption in figures:lines += [f'![{caption}](../../assets/benchmark-clouds/{name}/{filename}.png)','']
        lines+=['## 结论与可复现证据','',f"当前状态：**{r['status']}**。验收严格遵循上文范围。",'','原始 JSON 包含全部节点、连接、位移及恢复应力，可重新计算误差；平滑绘图仅用于显示，不用于改变验收值。',f'[原始 FE 数据](../../assets/benchmark-clouds/{name}/results.json)','', '运行：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/rebuild_additional_benchmarks.py`。']
        source=DOCS/(name.replace('-','_')+'_full.md');source.write_text('\n'.join(lines)+'\n')
        outputs=export_report(source,DOCS/'exports')
        paths=[raw,source,*outputs,*[folder/(f+'.png') for f,_ in figures]]
        manifest['reports'].append({'case_id':r['physical_case_id'],'status':r['status'],'scope':r.get('scope',r.get('qualification_scope')),'files':{str(p.relative_to(ROOT)):digest(p) for p in paths}})
    (DOCS/'additional-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({r['case_id']:r['status'] for r in manifest['reports']})

if __name__=='__main__':build()
