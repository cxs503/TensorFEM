"""Field gates must reject plausible-looking but incomplete/incorrect evidence."""
import math
from pathlib import Path

import pytest

from tensorfem.benchmark_fields import verify_case, verify_mesh_study
from tensorfem.hull_benchmark_fields import solve_hull_case, verify_hull_case
from tensorfem.validation import Reference, ValidationResult, require_all, validate
from scripts.run_stiffened_panel_fe_benchmark import solve_case
from scripts.validate_published_benchmark_reports import validate as validate_publication

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def membrane():
    return solve_case(4, 2)


def test_membrane_has_verified_displacement_and_stress_fields(membrane):
    check = verify_case(membrane, 'membrane')
    assert check['passed']
    assert check['displacement_field_relative_error'] < 1e-12
    assert max(check['stress_errors'].values()) < 1e-12


def test_wrong_stress_cannot_pass_by_retaining_displacement_or_status(membrane):
    for row in membrane['field']['element_stress']:
        row['sigma_x'] *= 2
        row['von_mises'] = math.sqrt(row['sigma_x']**2-row['sigma_x']*row['sigma_y']
                          + row['sigma_y']**2+3*row['tau_xy']**2)
    assert membrane['status'] == 'qualified'
    assert not verify_case(membrane, 'membrane')['passed']


def test_wrong_interior_displacement_cannot_pass_by_retaining_tip(membrane):
    membrane['field']['nodes'][6]['uy'] = 1.
    assert not verify_case(membrane, 'membrane')['passed']


@pytest.mark.parametrize('mutation', ['missing', 'nonfinite', 'duplicate'])
def test_incomplete_or_invalid_field_rejected(membrane, mutation):
    if mutation == 'missing':
        membrane['field']['element_stress'].pop()
    elif mutation == 'nonfinite':
        membrane['field']['element_stress'][0]['sigma_x'] = float('nan')
    else:
        membrane['field']['element_stress'][0] = membrane['field']['element_stress'][1]
    with pytest.raises(ValueError):
        verify_case(membrane, 'membrane')


def test_unbalanced_reactions_fail(membrane):
    membrane['reaction_left_x'] = 0.
    assert not verify_case(membrane, 'membrane')['passed']


def test_mesh_study_rejects_missing_levels_and_repeated_meshes(membrane):
    with pytest.raises(ValueError, match='three mesh'):
        verify_mesh_study([membrane], 'membrane')
    with pytest.raises(ValueError, match='refine'):
        verify_mesh_study([membrane]*3, 'membrane')


def test_hull_stress_is_recovered_from_fe_curvature():
    case = solve_hull_case(6)
    assert not verify_hull_case(case)['passed']
    assert verify_hull_case(solve_hull_case(96))['passed']
    # Midpoint displacement is essentially exact, but the field has FE error.
    assert case['verification']['errors']['sigma_x_field_l2'] > .005
    case['field']['sigma_x'][0][0] *= 2
    with pytest.raises(ValueError, match='match FE DOFs'):
        verify_hull_case(case)


def test_empty_nonfinite_and_forged_scalar_evidence_rejected():
    with pytest.raises(AssertionError, match='nonempty'):
        require_all([])
    with pytest.raises(ValueError, match='finite'):
        Reference('case', 'u', float('nan'), 'm', 'analytical')
    ref = Reference('case', 'u', 1., 'm', 'analytical')
    with pytest.raises(ValueError, match='finite'):
        validate(ref, float('nan'))
    forged = ValidationResult(ref, 2., 0., True)
    with pytest.raises(AssertionError, match='accuracy gate'):
        require_all([forged])


def test_published_field_studies_have_three_distinct_physical_cases():
    result = validate_publication(ROOT)
    assert result['independent_physical_cases'] == 3
    assert result['exports_verified']
    assert not result['all_benchmarks_complete']
