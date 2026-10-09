"""Small scalar snapshots must never masquerade as full field reports."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_standard_benchmark_report import validate
from tensorfem.standard_benchmarks import cantilever_beam

ROOT = Path(__file__).resolve().parents[1]


def test_standard_builder_keeps_small_scalar_snapshot_reference_only(tmp_path):
    payload = tmp_path/'fe.json'
    payload.write_text(json.dumps({'cantilever_beam_tip_load':
                                  {'value': cantilever_beam()['tip_displacement']}}))
    output = tmp_path/'report.json'
    subprocess.run([sys.executable, str(ROOT/'scripts/run_standard_benchmark_report.py'),
                    '--output', str(output), '--fe-results', str(payload)], check=True)
    report = validate(output)
    case = report['cases']['cantilever_beam_tip_load']
    assert case['scalar_status'] == 'qualified'
    assert case['status'] == 'reference-only'
    case['status'] = 'qualified'
    clean = {k: v for k, v in report.items() if k != 'evidence_sha256'}
    report['evidence_sha256'] = hashlib.sha256(json.dumps(clean, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    output.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='scalar snapshot'):
        validate(output)
