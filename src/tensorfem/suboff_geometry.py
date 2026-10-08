"""Closed quadrilateral geometry generated from a caller supplied CAD profile.

TensorLBM is used by the export script, never imported by the solver package.
The exported numerical geometry carries its source commit and file hash.
"""
from __future__ import annotations

from collections import Counter
import math
import torch


def closed_revolved_mesh(profile, length: float, radius: float,
                         axial: int, circumferential: int):
    """Mesh a revolved outer surface with square-to-disc all-quad pole caps."""
    if axial < 8 or circumferential < 8 or circumferential % 8:
        raise ValueError("axial >= 8 and circumferential must be a multiple of 8")
    if not math.isfinite(length + radius) or min(length, radius) <= 0:
        raise ValueError("positive finite geometry dimensions required")
    m = circumferential // 4
    nodes, cells = [], []
    # Avoid coincident ring nodes at either pole. Caps lie on the same profile.
    ends = (.015, .99)
    angles = [-math.pi/4 + 2*math.pi*j/circumferential
              for j in range(circumferential)]
    for i in range(axial + 1):
        xi = ends[0] + (ends[1] - ends[0])*i/axial
        r = radius*float(profile(xi))
        nodes.extend((length*xi, r*math.cos(a), r*math.sin(a)) for a in angles)
    for i in range(axial):
        for j in range(circumferential):
            k = (j+1) % circumferential
            cells.append((i*circumferential+j, i*circumferential+k,
                          (i+1)*circumferential+k, (i+1)*circumferential+j))
    for end, xi_end in enumerate(ends):
        ids = {}
        cap_radius = float(profile(xi_end))*radius
        for j in range(m+1):
            for i in range(m+1):
                a, b = -1+2*i/m, -1+2*j/m
                s = max(abs(a), abs(b))
                if s == 0:
                    angle = 0.
                elif abs(a) >= abs(b):
                    angle = math.pi/4*(b/a) + (math.pi if a < 0 else 0)
                else:
                    angle = math.pi/2 - math.pi/4*(a/b) + (math.pi if b < 0 else 0)
                if s == 1:
                    k = round(((angle+math.pi/4) % (2*math.pi)) /
                              (2*math.pi)*circumferential) % circumferential
                    ids[j, i] = (0 if end == 0 else axial)*circumferential+k
                else:
                    r = s*cap_radius
                    lo, hi = (0., xi_end) if end == 0 else (xi_end, 1.)
                    if s == 0:
                        xi = float(end)
                    else:
                        for _ in range(55):
                            mid = (lo+hi)/2
                            if (float(profile(mid))*radius < r) == (end == 0):
                                lo = mid
                            else:
                                hi = mid
                        xi = (lo+hi)/2
                    ids[j, i] = len(nodes)
                    nodes.append((xi*length, r*math.cos(angle), r*math.sin(angle)))
        for j in range(m):
            for i in range(m):
                c = (ids[j,i],ids[j,i+1],ids[j+1,i+1],ids[j+1,i])
                cells.append(c[::-1] if end == 0 else c)
    x = torch.tensor(nodes, dtype=torch.float64)
    e = torch.tensor(cells, dtype=torch.long)
    audit = audit_closed_quads(x, e)
    if not audit['passed']:
        raise ValueError(f"invalid revolved mesh: {audit}")
    return x, e, audit


def audit_closed_quads(nodes, elements):
    edges = Counter()
    orientation = Counter()
    for cell in elements.tolist():
        if len(set(cell)) != 4:
            return {'passed': False, 'reason': 'repeated cell node'}
        for a,b in zip(cell,cell[1:]+cell[:1]):
            edges[min(a,b),max(a,b)] += 1
            orientation[min(a,b),max(a,b)] += 1 if a < b else -1
    xyz = nodes[elements]
    area_vector = .5*torch.linalg.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,1],dim=1)
    area = torch.linalg.vector_norm(area_vector,dim=1)
    # Divergence theorem applied to the two triangles of each quadrilateral.
    volume = nodes.new_zeros(())
    for a,b,c in ((0,1,2),(0,2,3)):
        volume += torch.sum(xyz[:,a]*torch.linalg.cross(xyz[:,b],xyz[:,c],dim=1))/6
    connected = {0}; adjacency = [set() for _ in nodes]
    for a,b in edges:
        adjacency[a].add(b); adjacency[b].add(a)
    frontier = [0]
    while frontier:
        for b in adjacency[frontier.pop()]:
            if b not in connected:
                connected.add(b); frontier.append(b)
    defect = len(nodes)-len(edges)+len(elements)-2
    passed = (bool(torch.isfinite(nodes).all()) and float(area.min()) > 0
              and all(v == 2 for v in edges.values())
              and all(v == 0 for v in orientation.values())
              and len(connected) == len(nodes) and defect == 0 and float(volume) > 0)
    return {'passed': passed, 'nodes':len(nodes), 'elements':len(elements),
            'edges':len(edges), 'euler_characteristic':defect+2,
            'minimum_area_m2':float(area.min()), 'enclosed_volume_m3':float(volume),
            'boundary_or_nonmanifold_edges':sum(v != 2 for v in edges.values()),
            'orientation_defects':sum(v != 0 for v in orientation.values())}


def inward_midsurface(outer, elements, thickness):
    """Offset CAD nodes inward by t/2 along area-weighted nodal normals."""
    xyz = outer[elements]
    normal = torch.linalg.cross(xyz[:,2]-xyz[:,0],xyz[:,3]-xyz[:,1],dim=1)
    nodal = torch.zeros_like(outer)
    for k in range(4):
        nodal.index_add_(0,elements[:,k],normal)
    nodal /= torch.linalg.vector_norm(nodal,dim=1)[:,None]
    mid = outer - thickness/2*nodal
    if not audit_closed_quads(mid,elements)['passed']:
        raise ValueError("thickness offset invalidated hull mesh")
    return mid, nodal
