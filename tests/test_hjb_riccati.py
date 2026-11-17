"""The HJB solver against an exact linear quadratic solution.

With eta = lam = sigma = 0 the execution model is a single agent problem:
minimise the integral of phi q^2 + psi nu^2 plus a terminal penalty A q^2,
with dq = -nu dt. The value function is u(t, q) = h(t) q^2 where
h solves the Riccati equation dh/dtau = phi - h^2 / psi, h(tau = 0) = A,
tau = T - t. The optimal rate is nu = h q / psi.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mfglob.grids import Grid1D, TimeGrid
from mfglob.hjb import HJBSolver
from mfglob.mfg_solver import MFGSolver
from mfglob.models.optimal_execution import OptimalExecutionMFG

PHI, PSI, T, A = 0.5, 1.0, 1.0, 1.0


def riccati(tau):
    sol = solve_ivp(lambda s, h: PHI - h**2 / PSI, [0.0, T], [A],
                    dense_output=True, rtol=1e-12, atol=1e-12)
    return sol.sol(tau)[0]


def hjb_value(n_x, n_t):
    model = OptimalExecutionMFG(phi=PHI, psi=PSI, eta=0.0, lam=0.0, sigma=0.0)
    grid = Grid1D(0.0, 1.5, n_x)
    tgrid = TimeGrid(T, n_t)
    m = np.ones((tgrid.n_t if hasattr(tgrid, "n_t") else n_t + 1, n_x)) / 1.5
    try:
        u = HJBSolver(grid, tgrid, model).solve(m)["value_function"]
    except (ValueError, IndexError):
        m = np.ones((n_t, n_x)) / 1.5
        u = HJBSolver(grid, tgrid, model).solve(m)["value_function"]
    return grid.points, u


def test_value_matches_riccati():
    x, u = hjb_value(200, 400)
    h0 = riccati(T)
    inside = (x > 0.2) & (x < 1.2)
    err = np.abs(u[0, inside] - h0 * x[inside] ** 2).max()
    assert err < 0.01


def test_value_is_below_terminal_penalty():
    # Trading can only lower the cost of holding q, so u(0, q) < A q^2.
    x, u = hjb_value(100, 200)
    inside = (x > 0.2) & (x < 1.2)
    assert np.all(u[0, inside] < A * x[inside] ** 2)


def test_error_shrinks_with_refinement():
    h0 = riccati(T)
    errs = []
    for n_x, n_t in ((100, 100), (200, 400), (400, 1600)):
        x, u = hjb_value(n_x, n_t)
        i = np.argmin(np.abs(x - 1.0))
        errs.append(abs(u[0, i] - h0 * x[i] ** 2))
    assert errs[2] < errs[0]
    assert errs[2] < 0.003


def test_mean_inventory_follows_riccati_trajectory():
    model = OptimalExecutionMFG(phi=PHI, psi=PSI, eta=0.0, lam=0.0, Q0=1.0, sigma=0.02)
    grid = Grid1D(0.0, 1.5, 200)
    tgrid = TimeGrid(T, 400)
    sol = MFGSolver(model, grid, tgrid, damping=0.5, tol=1e-6).solve()
    m, x = sol["density"], grid.points
    t = np.linspace(0.0, T, m.shape[0])
    mean_q = (m * x).sum(axis=1) / m.sum(axis=1)

    fine = np.linspace(0.0, T, 4001)
    h = riccati(T - fine)
    log_q = -np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(fine))]) / PSI
    exact = np.interp(t, fine, np.exp(log_q)) * mean_q[0]
    assert np.abs(mean_q - exact).max() == pytest.approx(0.0, abs=0.02)
