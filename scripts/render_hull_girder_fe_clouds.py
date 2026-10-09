"""Render full-span beam-fibre fields from the 120 m hull-girder FE report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
from render_stiffened_panel_fe_clouds import _plot_contour
from tensorfem.hull_benchmark_fields import verify_hull_case


def main():
    p = argparse.ArgumentParser()
    p.add_argument('report', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    case = json.loads(a.report.read_text())['cases'][-1]
    if not verify_hull_case(case)['passed']:
        raise ValueError('hull field verification blocked')
    field = case['field']; x, y = np.meshgrid(field['x'], field['y'])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values, label in [
            ('displacement', np.tile(field['uy'], (len(field['y']), 1))*1e3, 'uy (mm)'),
            ('sigma_x', np.array(field['sigma_x'])*1e-6, 'sigma_x (MPa)'),
            ('von_mises', np.abs(np.array(field['sigma_x']))*1e-6, 'uniaxial von Mises (MPa)')]:
        _plot_contour(x, y, values, label=label, title='120 m hull-girder beam fibres',
                      path=a.output_dir/(name+'.png'))


if __name__ == '__main__':
    main()
