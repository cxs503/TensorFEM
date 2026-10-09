"""Recompute the 120 m hull-girder frame benchmark and full-span stress fields."""
import argparse
import json
from pathlib import Path
from tensorfem.hull_benchmark_fields import hull_report, solve_hull_case as solve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = hull_report()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(a.output), 'status': report['status'],
                      'checks': report['mesh_convergence']['checks']}))
    if report['status'] != 'qualified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
