import pytest

from tensorfem.sparse_shell4_plastic_qualification import (
    PlasticPathQualification, compare_plastic_paths, yielding_shell4_case,
)


def _report(backend, scale=1.):
    return PlasticPathQualification(
        backend, True, 80, 7.7*scale, .1*scale, .4*scale, .03*scale,
        .37*scale, 1., .02*scale, .2*scale, .1*scale, True,
        backend, 1., 1000, 10 if backend == "superlu" else 0,
        500 if backend == "superlu" else None,
        800 if backend == "superlu" else None,
        1e-14 if backend == "superlu" else None,
    )


def test_yielding_case_contract_and_physical_comparison_gate():
    model, load, fixed = yielding_shell4_case()
    assert model.layers == 3 and load.shape == (model.n_dofs,) and len(fixed) == 20
    comparison = compare_plastic_paths(_report("dense"), _report("superlu", 1.0001))
    assert comparison["passed"]
    assert not comparison["byte_identical_state"]
    failed = compare_plastic_paths(_report("dense"), _report("superlu", 1.02))
    assert not failed["passed"]


def test_comparison_rejects_wrong_backend_order():
    with pytest.raises(ValueError, match="dense and superlu"):
        compare_plastic_paths(_report("superlu"), _report("dense"))
