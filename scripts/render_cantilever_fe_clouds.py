"""Render verified FE displacement and stress contours from the latest mesh."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
from render_stiffened_panel_fe_clouds import _plot_contour
from tensorfem.benchmark_fields import verify_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    case = json.loads(a.input.read_text())['cases'][-1]
    if not verify_case(case, 'bending')['passed']:
        raise ValueError('cantilever field verification blocked')
    nx, ny = case['mesh']['nx'], case['mesh']['ny']
    nodes, stress = case['field']['nodes'], case['field']['element_stress']
    a.output_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([p['x'] for p in nodes]).reshape(ny+1, nx+1)
    y = np.array([p['y'] for p in nodes]).reshape(ny+1, nx+1)
    _plot_contour(x, y, np.array([p['uy'] for p in nodes]).reshape(ny+1, nx+1)*1e3,
                  label='uy (mm)', title=f'Q4 {nx}x{ny}: displacement',
                  path=a.output_dir/'displacement.png')
    x = np.array([p['x'] for p in stress]).reshape(ny, nx)
    y = np.array([p['y'] for p in stress]).reshape(ny, nx)
    for key in ('sigma_x', 'tau_xy', 'von_mises'):
        _plot_contour(x, y, np.array([p[key] for p in stress]).reshape(ny, nx)*1e-6,
                      label=key+' (MPa)', title=f'Q4 {nx}x{ny}: {key}',
                      path=a.output_dir/(key+'.png'), cell_centred=True)


if __name__ == '__main__':
    main()
