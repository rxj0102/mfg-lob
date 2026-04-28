"""Tests for LOB model components."""
import numpy as np
import pytest

from mfg_lob.models.costs import QuadraticCost, ExponentialCost, LOBCostTerminal
from mfg_lob.models.hamiltonians import MarketMakerHamiltonian, TrendFollowerHamiltonian
from mfg_lob.models.lob_model import LOBModel


q = np.linspace(-5, 5, 51)
m = np.ones(51) / (51 * 0.2)  # uniform density


class TestQuadraticCost:
    def test_zero_at_zero(self):
        cost = QuadraticCost(phi=1.0, alpha=0.0)
        assert cost(np.array([0.0]), np.array([0.0]), 0.0)[0] == pytest.approx(0.0)

    def test_symmetric(self):
        cost = QuadraticCost(phi=1.0, alpha=0.0)
        assert cost(np.array([-2.0]), np.array([0.0]), 0.0)[0] == pytest.approx(
            cost(np.array([2.0]), np.array([0.0]), 0.0)[0]
        )

    def test_shape(self):
        cost = QuadraticCost()
        result = cost(q, m, 0.0)
        assert result.shape == q.shape


class TestTerminalCost:
    def test_zero_at_zero(self):
        g = LOBCostTerminal(A=1.0, B=0.5)
        assert g(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_nonnegative(self):
        g = LOBCostTerminal(A=1.0, B=0.5)
        assert np.all(g(q) >= 0)


class TestMarketMakerHamiltonian:
    def test_nonnegative(self):
        H = MarketMakerHamiltonian(Lambda0=1.0, kappa=1.0)
        p = np.zeros_like(q)
        assert np.all(H(q, p, m, 0.0) >= 0)

    def test_optimal_spread_positive(self):
        H = MarketMakerHamiltonian(kappa=1.0)
        p = np.linspace(-0.5, 0.5, 11)
        da, db = H.optimal_spread(p)
        assert np.all(da >= 0)
        assert np.all(db >= 0)


class TestTrendFollowerHamiltonian:
    def test_quadratic_in_p(self):
        H = TrendFollowerHamiltonian(epsilon=2.0)
        p = np.array([1.0, 2.0])
        vals = H(q[:2], p, m[:2], 0.0)
        assert vals[1] == pytest.approx(4.0 * vals[0])

    def test_optimal_drift_sign(self):
        H = TrendFollowerHamiltonian(epsilon=1.0)
        # positive p → negative drift (sell)
        b = H.optimal_drift(np.array([0.0]), np.array([1.0]), np.array([0.0]), 0.0)
        assert b[0] < 0


class TestLOBModel:
    def test_market_maker_construction(self):
        model = LOBModel(sigma=0.5, phi=0.1, alpha=1.0, agent_type="market_maker")
        assert model.sigma == 0.5

    def test_trend_follower_construction(self):
        model = LOBModel(epsilon=2.0, agent_type="trend_follower")
        assert model.epsilon == 2.0

    def test_invalid_agent_type(self):
        with pytest.raises(ValueError):
            LOBModel(agent_type="invalid")

    def test_initial_density_normalised(self):
        model = LOBModel()
        m0 = model.initial_density(q)
        assert np.trapezoid(m0, q) == pytest.approx(1.0, rel=1e-3)

    def test_terminal_cost_nonnegative(self):
        model = LOBModel()
        assert np.all(model.terminal_cost(q) >= 0)

    def test_hamiltonian_callable(self):
        model = LOBModel()
        p = np.zeros_like(q)
        H = model.hamiltonian(q, p, m, 0.0)
        assert H.shape == q.shape
