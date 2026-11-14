"""Integration test: CASE-FEM-HEAT-L2 convergence study via public API."""
import math

import pytest

from tensorfem import run_heat_conduction_l2, HeatConductionL2Result


def test_heat_conduction_l2_returns_result():
    """run_heat_conduction_l2 returns a HeatConductionL2Result."""
    result = run_heat_conduction_l2()
    assert isinstance(result, HeatConductionL2Result)


def test_heat_conduction_l2_four_levels():
    """Default run has 4 mesh levels."""
    result = run_heat_conduction_l2()
    assert len(result.levels) == 4


def test_heat_conduction_l2_convergence_order():
    """Mean observed order >= 1.8 (target 2.0)."""
    result = run_heat_conduction_l2()
    assert result.mean_order >= 1.8, f"Mean order {result.mean_order:.4f} < 1.8"


def test_heat_conduction_l2_finest_error():
    """Finest mesh L2 relative error < 3%."""
    result = run_heat_conduction_l2()
    assert result.levels[-1].l2_relative < 0.03


def test_heat_conduction_l2_heat_balance():
    """Heat balance residual at free nodes is machine precision."""
    result = run_heat_conduction_l2()
    for lv in result.levels:
        assert lv.heat_balance_residual < 1e-10, (
            f"Level {lv.nx}x{lv.ny}: heat_balance={lv.heat_balance_residual:.2e}"
        )


def test_heat_conduction_l2_all_gates_pass():
    """All acceptance gates pass."""
    result = run_heat_conduction_l2()
    assert result.all_gates_pass


def test_heat_conduction_l2_orders_approach_two():
    """Observed orders monotonically approach 2.0."""
    result = run_heat_conduction_l2()
    orders = result.observed_orders
    assert len(orders) == 3
    # Each refinement should get closer to 2.0
    for o in orders:
        assert o >= 1.5, f"Order {o:.4f} too low"
    assert orders[-1] > orders[0], "Orders should increase with refinement"


def test_heat_conduction_l2_custom_mesh():
    """Custom mesh levels work."""
    result = run_heat_conduction_l2(mesh_levels=[(2, 2), (4, 4), (8, 8)])
    assert len(result.levels) == 3
    assert result.levels[0].n_elements == 4
    assert result.levels[-1].n_elements == 64
