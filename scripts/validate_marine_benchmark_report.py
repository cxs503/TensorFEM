#!/usr/bin/env python3
"""Validate marine scalar evidence without promoting it to field certification."""
import hashlib
import json
import math
import sys
from pathlib import Path


def validate(path):
    report = json.loads(Path(path).read_text())
    if report.get('schema') != 'tensorfem.marine-benchmark-report/1.0':
        raise ValueError('unsupported marine report schema')
    clean = {k: v for k, v in report.items() if k != 'evidence_sha256'}
    digest = hashlib.sha256(json.dumps(clean, sort_keys=True,
                            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != report.get('evidence_sha256'):
        raise ValueError('marine report hash mismatch')
    cases = report.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('nonempty marine scalar evidence required')
    for case in cases:
        result = case['results']
        computed, reference = float(result['computed']), float(result['reference'])
        if not all(math.isfinite(x) for x in (computed, reference)) or reference == 0:
            raise ValueError('invalid scalar/reference')
        error = abs(computed-reference)/abs(reference)
        if not math.isclose(error, result['relative_error'], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError('stored marine error does not match values')
        expected = 'qualified' if error < .03 else 'blocked'
        if result.get('scalar_status') != expected:
            raise ValueError('marine scalar status mismatch')
        if result.get('status') != ('reference-only' if error < .03 else 'blocked'):
            raise ValueError('scalar evidence cannot certify a field report')
        if not all(case.get(k) for k in ('problem', 'source', 'unit')):
            raise ValueError('marine scalar specification missing')
        if case.get('field_results', {}).get('stress_cloud') is not None or \
           case.get('field_results', {}).get('displacement_cloud') is not None:
            raise ValueError('unexpected certified field in scalar-only report')
    passed = sum(c['results']['scalar_status'] == 'qualified' for c in cases)
    expected_summary = {'total': len(cases), 'qualified': 0, 'scalar_qualified': passed,
                        'reference_only': passed, 'blocked': len(cases)-passed}
    if report.get('summary') != expected_summary:
        raise ValueError('marine scalar/field counts mismatch')
    return report


if __name__ == '__main__':
    report = validate(sys.argv[1])
    print(json.dumps({'valid': True, 'summary': report['summary'],
                      'qualification_scope': 'scalar checks only'}))
