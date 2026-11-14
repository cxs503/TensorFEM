"""Two-way elastic-restraint verification + patch-load integration audit.

Anchor: GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT (JOUR-ICE-STRUCT §4.2).

Produces docs/data/elastic_restraint_two_way_evidence.json.

Sections
--------
1. ``load_conservation``   -- applied patch load vs ``q * patch_area`` on several
   meshes.  Guards the load-integration defect found here (a 2x2 Gauss rule on
   the whole element sampled a sub-element patch only at its Gauss points, so
   the total load oscillated with the mesh).
2. ``fe_one_way``          -- FE spring edge moment vs the **exact** Kirchhoff
   Levy solution with ``beta_y = 0`` (reference exists for one-way restraint).
3. ``fe_two_way``          -- FE two-way restrained plate vs the independent
   double-sine Rayleigh-Ritz reference (four springs).
4. ``panel_peak_band``     -- manuscript panel: all four candidate definitions
   of "peak bending stress" across the beta band, plus convergence.

Uniform-load cases are unaffected by the load-integration fix, so section 3 is
directly comparable with the pre-fix runs quoted in the case document.
Sign conventions differ between the module and the series references, so all
moment comparisons are on magnitudes.

Run:  PYTHONPATH=src python scratch/probe_two_way_peak.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.plate_elastic_restraint import (
    _patch_consistent_load_vector,
    _rect_edges,
    beta_to_zeta,
    edge_moment_von_mises_factor,
    edge_spring_stress,
    moments_at_nodes,
    peak_bending_stress,
    rect_patch_pressure,
    solve_restrained_panel,
    structured_rect_mesh,
    von_mises_bending_at_nodes,
)
from tensorfem.plate_levy_restraint import (
    flexural_rigidity,
    levy_restrained_plate,
    ritz_restrained_plate,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "data" / \
    "elastic_restraint_two_way_evidence.json"

PANEL = dict(l=0.8, s=0.7, t=0.022, E=206e9, nu=0.3, p_ref=1.0e6, c_over_l=0.1)
VER = dict(a=1.0, b=1.0, t=0.01, E=200e9, nu=0.3, q=1.0e3)


def rel(x, y):
    return (x - y) / y


def _centre_w(res):
    n = res.nodes
    k = int(torch.argmin((n[:, 0] - res.side_x / 2) ** 2 +
                         (n[:, 1] - res.side_y / 2) ** 2))
    return float(res.w()[k])


def _spring_mid(res, key, target):
    d = edge_spring_stress(res)[key]
    along = 1 if key in ('x0', 'xa') else 0
    k = int(torch.argmin((res.nodes[d['idx'], along] - target).abs()))
    return abs(float(d['M_n'][k])), float(d['von_mises'][k])


def _panel_press(patch):
    c = PANEL['c_over_l'] * PANEL['l']
    if not patch:
        return PANEL['p_ref']
    return rect_patch_pressure(PANEL['p_ref'], 0.5 * (PANEL['l'] - c),
                               0.5 * (PANEL['l'] + c), 0.0, PANEL['s'])


def _panel(nx, beta, patch):
    ny = max(1, int(round(nx * PANEL['s'] / PANEL['l'])))
    return solve_restrained_panel(nx, ny, PANEL['l'], PANEL['s'], PANEL['E'],
                                  PANEL['nu'], PANEL['t'], _panel_press(patch),
                                  beta_x=beta, beta_y=beta)


def _peak(res):
    p = peak_bending_stress(res)
    vm = von_mises_bending_at_nodes(res)
    rect = _rect_edges(res.nodes, res.side_x, res.side_y)
    spring = edge_spring_stress(res)
    ext, spr = {}, {}
    for key, idx in rect.items():
        c = res.nodes[idx]
        corner = (((c[:, 0] <= 1e-12) | (c[:, 0] >= res.side_x - 1e-12)) &
                  ((c[:, 1] <= 1e-12) | (c[:, 1] >= res.side_y - 1e-12)))
        keep = idx[~corner]
        ext[key] = float(vm[keep].abs().max()) if int(keep.numel()) else 0.0
        spr[key] = float(spring[key]['von_mises'].max()) if key in spring else 0.0
    return dict(centre_von_mises=p['centre_von_mises'],
                extrapolated_global_max=p['max_von_mises'],
                extrapolated_corner_max=float(vm.max()),
                extrapolated_edge_max=max(ext.values()),
                extrapolated_edge_max_per_edge=ext,
                spring_edge_max=max(spr.values()),
                spring_edge_max_per_edge=spr,
                peak_consistent=p['peak_von_mises_consistent'],
                peak_source=p['peak_source'],
                w_centre=p['w_centre'])


def section_load_conservation():
    rows = []
    c = PANEL['c_over_l'] * PANEL['l']
    press = rect_patch_pressure(PANEL['p_ref'], 0.5 * (PANEL['l'] - c),
                                0.5 * (PANEL['l'] + c), 0.0, PANEL['s'])
    exact = PANEL['p_ref'] * c * PANEL['s']
    for nx in (8, 12, 16, 24, 32, 48):
        ny = max(1, int(round(nx * PANEL['s'] / PANEL['l'])))
        nodes, els = structured_rect_mesh(nx, ny, PANEL['l'], PANEL['s'])
        fe = _patch_consistent_load_vector(nodes, els, press.patch_bounds,
                                          press.intensity)
        rows.append(dict(nx=nx, ny=ny, applied=float(fe.sum()), exact=exact,
                         rel_err=rel(float(fe.sum()), exact)))
    return dict(case=PANEL, rows=rows,
                note='exact quadrature on the element/patch intersection')


def section_fe_one_way():
    D = flexural_rigidity(VER['E'], VER['nu'], VER['t'])
    out = []
    for nx in (8, 16, 24):
        row = dict(nx=nx)
        for beta in (0.2, 0.5, 0.8):
            res = solve_restrained_panel(nx, nx, VER['a'], VER['b'], VER['E'],
                                         VER['nu'], VER['t'], VER['q'],
                                         beta_x=beta, beta_y=0.0)
            K = beta_to_zeta(beta) * D / VER['a']
            lev = levy_restrained_plate(VER['a'], VER['b'], D, VER['nu'], K,
                                        VER['q'], n_modes=401)
            M_ref = abs(lev.moments(0.0, VER['b'] / 2.0)[0])
            M_fe, vm_fe = _spring_mid(res, 'x0', VER['b'] / 2.0)
            M = moments_at_nodes(res)
            edge = torch.nonzero(res.nodes[:, 0] == 0.0).flatten()
            k = int(torch.argmin((res.nodes[edge, 1] - VER['b'] / 2.0).abs()))
            M_ext = abs(float(M[int(edge[k]), 0]))
            row[f'beta{beta}'] = dict(
                Mn_ref=M_ref, Mn_spring=M_fe, spring_rel_err=rel(M_fe, M_ref),
                Mn_extrapolated=M_ext, extrapolated_rel_err=rel(M_ext, M_ref),
                spring_von_mises=vm_fe,
                w_fe=_centre_w(res), w_ref=lev.w_centre(),
                w_rel_err=rel(_centre_w(res), lev.w_centre()))
        out.append(row)
    return dict(case=VER, rows=out,
                note='exact Levy reference, two sprung edges / two simple edges')


def section_fe_two_way():
    D = flexural_rigidity(VER['E'], VER['nu'], VER['t'])
    rows = []
    for nx in (24,):
        for beta in (0.2, 0.5, 0.8):
            res = solve_restrained_panel(nx, nx, VER['a'], VER['b'], VER['E'],
                                         VER['nu'], VER['t'], VER['q'],
                                         beta_x=beta, beta_y=beta)
            K = beta_to_zeta(beta) * D / VER['a']
            rit = ritz_restrained_plate(VER['a'], VER['b'], D, VER['nu'], K, K,
                                        VER['q'], 40, 40)
            h = 1e-6
            Mn_rit = abs(K * (rit.w(h, VER['b'] / 2.0)
                              - rit.w(0.0, VER['b'] / 2.0)) / h)
            Mn_fe, _ = _spring_mid(res, 'x0', VER['b'] / 2.0)
            rows.append(dict(nx=nx, beta=beta, w_fe=_centre_w(res),
                             w_ritz=rit.w_centre(),
                             w_rel_err=rel(_centre_w(res), rit.w_centre()),
                             Mn_fe=Mn_fe, Mn_ritz=Mn_rit,
                             Mn_rel_err=rel(Mn_fe, Mn_rit)))
    return dict(case=VER, rows=rows,
                note='four-edge springs, double-sine Rayleigh-Ritz (no FEM); '
                     'reference error budget from section reference_calibration: '
                     'w 3.4e-3, Mn 1.2e-2 at beta = 0.8, 40x40')


def section_panel_peak_band():
    rows = []
    for beta in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        rows.append(dict(beta=beta,
                         uniform=_peak(_panel(16, beta, False)),
                         patch=_peak(_panel(16, beta, True))))
    conv = []
    for nx in (8, 12, 16, 24, 32, 48):
        p = _peak(_panel(nx, 0.8, True))
        conv.append(dict(nx=nx, spring_edge_max=p['spring_edge_max'],
                         extrapolated_edge_max=p['extrapolated_edge_max'],
                         peak_consistent=p['peak_consistent'],
                         peak_source=p['peak_source'], w_centre=p['w_centre']))
    rich, order = math.nan, math.nan
    v1, v2, v3 = (r['spring_edge_max'] for r in conv[-3:])
    if v2 != v3:
        r = (v1 - v2) / (v2 - v3)
        if r > 0 and r != 1.0:
            order = math.log2(r)
            rich = v3 + (v3 - v2) / (r - 1.0)
    return dict(case=PANEL, rows=rows, convergence=conv,
                spring_edge_richardson=rich, observed_order=order)


def main():
    ev = dict(
        anchor='GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT',
        purpose='two-way (beta_x, beta_y) band verification + patch-load '
                'integration and peak-stress口径 audit',
        edge_von_mises_factor=dict(nu=PANEL['nu'],
                                   factor=edge_moment_von_mises_factor(PANEL['nu'])),
        reference_calibration=dict(
            note='Ritz (four springs) vs exact Levy at Ky=0, measured in this '
                 'worktree: w rel err 2e-5 / 1e-4 / 3.4e-3 and Mn rel err '
                 '1.2e-3 / 4.3e-3 / 1.16e-2 for beta = 0.2 / 0.5 / 0.8 at 40x40'),
        load_conservation=section_load_conservation(),
        fe_one_way=section_fe_one_way(),
        fe_two_way=section_fe_two_way(),
        panel_peak_band=section_panel_peak_band(),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ev, indent=2))

    print('== load conservation (patch, exact = %.0f N) =='
          % ev['load_conservation']['rows'][0]['exact'])
    for r in ev['load_conservation']['rows']:
        print(f"  nx={r['nx']:2d} ny={r['ny']:2d} applied={r['applied']:12.4f}"
              f" rel={100*r['rel_err']:+.2e} %")
    print('== FE one-way vs exact Levy ==')
    for row in ev['fe_one_way']['rows']:
        for k, v in row.items():
            if k == 'nx':
                continue
            print(f"  nx={row['nx']:2d} {k}: spring Mn {100*v['spring_rel_err']:+.2f}%"
                  f"  extrapolated Mn {100*v['extrapolated_rel_err']:+.1f}%"
                  f"  w {100*v['w_rel_err']:+.3f}%")
    print('== FE two-way vs Ritz ==')
    for r in ev['fe_two_way']['rows']:
        print(f"  nx={r['nx']:2d} beta={r['beta']:.1f}  w {100*r['w_rel_err']:+.3f}%"
              f"  Mn {100*r['Mn_rel_err']:+.3f}%")
    print('== manuscript panel peak stress by口径 (patch load, nx=16) ==')
    for r in ev['panel_peak_band']['rows']:
        p = r['patch']
        print(f"  beta={r['beta']:.1f} extrap_global={p['extrapolated_global_max']/1e6:7.2f}"
              f" extrap_edge={p['extrapolated_edge_max']/1e6:7.2f}"
              f" spring_edge={p['spring_edge_max']/1e6:7.2f}"
              f" consistent={p['peak_consistent']/1e6:7.2f} MPa"
              f" src={p['peak_source']}")
    print('== beta = 0.8 patch panel convergence ==')
    for c in ev['panel_peak_band']['convergence']:
        print(f"  nx={c['nx']:2d} spring_edge={c['spring_edge_max']/1e6:7.3f}"
              f" extrap_edge={c['extrapolated_edge_max']/1e6:7.3f}"
              f" consistent={c['peak_consistent']/1e6:7.3f}"
              f" wc={c['w_centre']*1e3:7.4f} mm src={c['peak_source']}")
    print(f"  Richardson spring edge = "
          f"{ev['panel_peak_band']['spring_edge_richardson']/1e6:.3f} MPa"
          f" (observed order {ev['panel_peak_band']['observed_order']:.2f})")
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()