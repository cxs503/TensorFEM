"""Welded equivalent sail/fin shell midsurfaces from numerical CAD samples.

The sail roof silhouette and swept-fin planforms come from TensorLBM. These
are artificial sheet structures with separately declared thicknesses, not an
exact reproduction of the original thick airfoil skins or pressure structure.
Shared root nodes enforce all six DOFs without penalty ties or floating pieces.
"""
from __future__ import annotations

import math
import torch
from .suboff_geometry import audit_closed_quads, closed_revolved_mesh


def appended_shell_mesh(profile,length,radius,axial,circumferential,sail_samples,
                        fin_samples,*,sail_layers=3,fin_layers=3):
    if min(sail_layers,fin_layers)<1:
        raise ValueError('appendage span layers must be positive')
    stations=[.015+(.99-.015)*i/axial for i in range(axial+1)]
    stations=sorted(set(stations+[p[0]/length for p in sail_samples]+[p[0]/length for p in fin_samples]))
    body,body_cells,body_audit=closed_revolved_mesh(profile,length,radius,axial,
                                                  circumferential,axial_coordinates=stations)
    nodes=body.tolist();cells=body_cells.tolist();labels=[0]*len(cells)
    components=[{'id':0,'name':'bare_hull','root_nodes':[],'representation':'closed pressure shell'}]
    top_nodes=[];contact_area={}
    def root(x,theta):
        station=min(range(len(stations)),key=lambda i:abs(stations[i]*length-x))
        if abs(stations[station]*length-x)>1e-10:
            raise ValueError('appendage root is missing from body axial mesh')
        j=round(((theta+math.pi/4)%(2*math.pi))*circumferential/(2*math.pi))%circumferential
        return station*circumferential+j
    # Sail is a vertical median sheet, from the body crown to the CAD roof.
    grid=[];roots=[]
    for x,z_top,half_width in sail_samples:
        a=root(x,math.pi/2);roots.append(a);column=[a]
        z0=nodes[a][2]
        for j in range(1,sail_layers+1):
            column.append(len(nodes));nodes.append((x,0.,z0+(z_top-z0)*j/sail_layers))
        grid.append(column);top_nodes.append(column[-1])
    for i in range(len(grid)-1):
        for j in range(sail_layers):
            cells.append([grid[i][j],grid[i+1][j],grid[i+1][j+1],grid[i][j+1]]);labels.append(1)
        dx=sail_samples[i+1][0]-sail_samples[i][0]
        for k in (i,i+1):
            # Projected roof area: CAD width times half adjacent axial interval.
            contact_area[top_nodes[k]]=contact_area.get(top_nodes[k],0.)+dx*sail_samples[k][2]
    components.append({'id':1,'name':'sail','root_nodes':roots,'top_nodes':top_nodes,
                       'representation':'vertical equivalent shell midsurface; CAD roof silhouette'})
    for name,theta in (('fin_starboard',0.),('fin_port',math.pi),
                       ('fin_upper',math.pi/2),('fin_lower',3*math.pi/2)):
        cid=len(components);grid=[];roots=[]
        for x_root,x_tip,r_tip in fin_samples:
            a=root(x_root,theta);roots.append(a);column=[a]
            start=torch.tensor(nodes[a],dtype=torch.float64)
            end=start.new_tensor([x_tip,r_tip*math.cos(theta),r_tip*math.sin(theta)])
            for j in range(1,fin_layers+1):
                column.append(len(nodes));nodes.append((start+(end-start)*j/fin_layers).tolist())
            grid.append(column)
        for i in range(len(grid)-1):
            for j in range(fin_layers):
                cells.append([grid[i][j],grid[i+1][j],grid[i+1][j+1],grid[i][j+1]]);labels.append(cid)
        components.append({'id':cid,'name':name,'root_nodes':roots,
                           'representation':'swept equivalent shell midsurface; thick NACA skin omitted'})
    x=torch.tensor(nodes,dtype=torch.float64);e=torch.tensor(cells,dtype=torch.long)
    audit=audit_appended_mesh(x,e,len(body),len(body_cells),components)
    if not audit['passed']:
        raise ValueError(f'invalid appended shell topology: {audit}')
    return {'nodes':x.tolist(),'elements':e.tolist(),'body_node_count':len(body),
            'body_element_count':len(body_cells),'element_component_ids':labels,
            'components':components,'audit':audit,'body_audit':body_audit,
            'sail_roof_contact_area_m2':{str(k):v for k,v in contact_area.items()},
            'source_sail_samples':sail_samples,'source_fin_samples':fin_samples}


def audit_appended_mesh(nodes,elements,body_node_count,body_element_count,components):
    from .suboff_ice_collision import _area
    if nodes.ndim!=2 or nodes.shape[1]!=3 or elements.ndim!=2 or elements.shape[1]!=4:
        raise ValueError('appended mesh requires nodes[n,3] and cells[e,4]')
    if not bool(torch.isfinite(nodes).all()) or int(elements.max())>=len(nodes) or int(elements.min())<0:
        raise ValueError('invalid appended node/connectivity field')
    body=audit_closed_quads(nodes[:body_node_count],elements[:body_element_count])
    adjacency=[set() for _ in nodes]
    areas=[]
    for el in elements.tolist():
        if len(set(el))!=4:
            raise ValueError('repeated quadrilateral corner')
        areas.append(_area(nodes[el]))
        for a,b in zip(el,el[1:]+el[:1]):
            adjacency[a].add(b);adjacency[b].add(a)
    connected={0};todo=[0]
    while todo:
        for b in adjacency[todo.pop()]:
            if b not in connected:
                connected.add(b);todo.append(b)
    root_valid=all(p['root_nodes'] and all(0<=i<body_node_count for i in p['root_nodes']) for p in components[1:])
    return {'passed':body['passed'] and root_valid and len(connected)==len(nodes),
            'body_closed':body['passed'],'shared_roots_valid':root_valid,
            'all_nodes_connected':len(connected)==len(nodes),'nodes':len(nodes),
            'elements':len(elements),'appendage_count':len(components)-1,
            'minimum_projected_area_m2':min(areas),
            'body_enclosed_volume_m3':body['enclosed_volume_m3']}
