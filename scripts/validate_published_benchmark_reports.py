#!/usr/bin/env python3
"""Revalidate fields, convergence and hashed research-report exports."""
import hashlib
import json
from pathlib import Path

from tensorfem.benchmark_fields import verify_mesh_study
from tensorfem.hull_benchmark_fields import verify_hull_case

ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT):
    root = Path(root)
    reports = root/'docs/benchmarks/tutorials'
    manifest = json.loads((reports/'evidence-manifest.json').read_text())
    if manifest.get('schema') != 'tensorfem.published-research-evidence/2.0':
        raise ValueError('unsupported report manifest')
    required = set()
    for name in ('cantilever_beam_full', 'hull_girder_longitudinal_bending_full',
                 'stiffened_panel_fe_full'):
        required.add(f'docs/benchmarks/tutorials/{name}.md')
        for suffix in ('html', 'pdf', 'docx'):
            required.add(f'docs/benchmarks/tutorials/exports/{name}.{suffix}')
    files = manifest.get('files', {})
    for folder, name in [('cantilever-fe', 'results.json'),
                         ('stiffened-panel', 'results.json'),
                         ('hull-girder', 'hull-girder-fe.json')]:
        required.add(f'docs/assets/benchmark-clouds/{folder}/{name}')
        for fig in ('displacement.png', 'sigma_x.png', 'von_mises.png', 'convergence.png'):
            required.add(f'docs/assets/benchmark-clouds/{folder}/{fig}')
    if not required.issubset(files):
        raise ValueError('missing exported report or evidence asset')
    for name, digest in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('report manifest path escapes repository')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('report artifact hash mismatch: '+name)
    ids = set()
    for folder, name, kind in [('cantilever-fe', 'results.json', 'bending'),
                             ('stiffened-panel', 'results.json', 'membrane'),
                             ('hull-girder', 'hull-girder-fe.json', 'hull')]:
        data = json.loads((root/'docs/assets/benchmark-clouds'/folder/name).read_text())
        if kind != 'hull':
            check = verify_mesh_study(data['cases'], kind)
            if check != data['mesh_convergence'] or not check['passed']:
                raise ValueError('field study blocked or stale: '+folder)
        else:
            checks = [verify_hull_case(c) for c in data['cases']]
            if any(c.get('verification') != check or c.get('status') !=
                   ('qualified' if check['passed'] else 'blocked')
                   for c, check in zip(data['cases'], checks)):
                raise ValueError('stored hull qualification mismatch')
            if len(checks) < 3 or checks != data['mesh_convergence']['checks']:
                raise ValueError('hull field study stale or incomplete')
            if not checks[-1]['passed']:
                raise ValueError('hull final mesh blocked')
            for a, b in zip(data['cases'], data['cases'][1:]):
                if a['problem'] != b['problem'] or a['mesh']['elements'] >= b['mesh']['elements']:
                    raise ValueError('hull meshes not consistently refined')
                for k in ('displacement_field_l2', 'moment_field_l2', 'sigma_x_field_l2', 'moment_and_sigma_x_pointwise_max'):
                    if not b['verification']['errors'][k] < a['verification']['errors'][k]:
                        raise ValueError('hull field does not converge')
        ids.add(data['physical_case_id'])
    if len(ids) != 3 or manifest['independent_physical_cases'] != len(ids):
        raise ValueError('duplicate physical benchmark counted as independent')
    if manifest.get('all_benchmarks_complete') is not False:
        raise ValueError('three reports cannot certify all benchmarks')
    return {'published_reports': 3, 'independent_physical_cases': 3,
            'field_checks_passed': True, 'exports_verified': True,
            'all_benchmarks_complete': False}


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
