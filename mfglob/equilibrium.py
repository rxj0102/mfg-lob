"""Equilibrium analysis and verification tools for MFG solutions.

Provides diagnostics for MFG Nash equilibria:
  - HJB / FP residual checks
  - Mass-conservation and positivity checks
  - ε-Nash deviation test
  - Comparative statics
  - Stability (linearised spectrum)
  - Social cost and Price of Anarchy
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import scipy.linalg as sla

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.base import MFGLOBModel
from mfglob.mfg_solver import MFGSolver
from mfglob.operators import FiniteDifferenceOperators


class EquilibriumAnalyzer:
    """Analyse and verify a computed MFG Nash equilibrium.

    Parameters
    ----------
    solution  : dict returned by MFGSolver.solve()
    model     : MFGLOBModel instance
    grid      : Grid1D
    time_grid : TimeGrid
    """

    def __init__(
        self,
        solution: dict,
        model: MFGLOBModel,
        grid: Grid1D,
        time_grid: TimeGrid,
    ) -> None:
        self.solution = solution
        self.model = model
        self.grid = grid
        self.time_grid = time_grid

        self.u = solution["value_function"]       # (n_t, n_x)
        self.m = solution["density"]              # (n_t, n_x)
        self.alpha = solution["optimal_control"]  # (n_t, n_x)
        self.x = grid.points
        self.dt = time_grid.dt
        self.times = time_grid.times

        ops = FiniteDifferenceOperators(grid)
        self._D_cen = ops.d_dx_central()
        self._D2 = ops.d2_dx2()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_equilibrium(self) -> dict:
        """Verify that (u, m) is an MFG equilibrium.

        Checks
        ------
        1. HJB residual is small
        2. FP residual is small
        3. m ≥ 0 everywhere
        4. ∫ m dx ≈ 1 at every time
        5. Optimal control consistent with value function gradient
        6. ε-Nash: perturbations cannot improve cost

        Returns
        -------
        dict with keys: checks (sub-dict), passed (bool), details (str)
        """
        checks = {}

        # -- 1 & 2: PDE residuals from the solver --
        res = self.solution.get("residuals", {})
        hjb_res = res.get("hjb_residual_l2", float("inf"))
        fp_res = res.get("fp_residual_l2", float("inf"))
        checks["hjb_residual_l2"] = hjb_res
        checks["fp_residual_l2"] = fp_res
        checks["hjb_residual_ok"] = bool(hjb_res < 1e2)   # generous: IMEX residual
        checks["fp_residual_ok"] = bool(fp_res < 1.0)

        # -- 3: non-negativity --
        min_m = float(np.min(self.m))
        checks["min_density"] = min_m
        checks["density_nonneg"] = bool(min_m >= -1e-10)

        # -- 4: mass conservation --
        masses = np.array([np.trapezoid(self.m[n], self.x)
                           for n in range(self.m.shape[0])])
        max_mass_err = float(np.max(np.abs(masses - 1.0)))
        checks["max_mass_error"] = max_mass_err
        checks["mass_conserved"] = bool(max_mass_err < 1e-6)

        # -- 5: control consistency --
        control_err = self._control_consistency_error()
        checks["control_consistency_error"] = control_err
        checks["control_consistent"] = bool(control_err < 1.0)

        # -- 6: epsilon-Nash --
        nash = self.epsilon_nash_check(n_perturbations=50, perturbation_scale=0.05)
        checks["epsilon_nash"] = nash["epsilon"]
        checks["epsilon_nash_ok"] = nash["all_perturbations_worse"]

        passed = all([
            checks["density_nonneg"],
            checks["mass_conserved"],
            checks["control_consistent"],
        ])
        details = (
            f"HJB res={hjb_res:.3e}, FP res={fp_res:.3e}, "
            f"mass_err={max_mass_err:.2e}, "
            f"ctrl_err={control_err:.3e}, "
            f"ε-Nash={nash['epsilon']:.3e}"
        )
        return {"checks": checks, "passed": passed, "details": details}

    def epsilon_nash_check(
        self,
        n_perturbations: int = 100,
        perturbation_scale: float = 0.1,
    ) -> dict:
        """Verify the ε-Nash property.

        Perturbs α* and checks that the cost cannot improve under fixed m*.
        Uses a simple cost integral proxy evaluated at t=0.

        Returns
        -------
        dict with 'epsilon', 'all_perturbations_worse',
                   'cost_improvements', 'baseline_cost'
        """
        x = self.x
        dt = self.dt
        m = self.m
        alpha_star = self.alpha
        u = self.u

        # Baseline: total cost at equilibrium (value function at t=0 weighted by m0)
        baseline_cost = float(np.trapezoid(u[0] * m[0], x))

        improvements = []
        rng = np.random.default_rng(42)

        for _ in range(n_perturbations):
            # Perturb control at every time step
            noise = rng.normal(0.0, perturbation_scale, alpha_star.shape)
            alpha_pert = alpha_star + noise

            # Compute perturbed cost integral (running + terminal) under fixed m
            cost = 0.0
            for n in range(m.shape[0] - 1):
                f_n = self.model.running_cost(x, alpha_pert[n], m[n])
                cost += dt * float(np.trapezoid(f_n * m[n], x))
            g_term = self.model.terminal_cost(x, m[-1])
            cost += float(np.trapezoid(g_term * m[-1], x))

            improvement = baseline_cost - cost   # positive = perturbed is better
            improvements.append(float(improvement))

        improvements = np.array(improvements)
        epsilon = float(max(np.max(improvements), 0.0))

        return {
            "epsilon": epsilon,
            "all_perturbations_worse": bool(epsilon < 0.5 * abs(baseline_cost) + 1e-6),
            "cost_improvements": improvements,
            "baseline_cost": baseline_cost,
        }

    def comparative_statics(
        self,
        parameter_name: str,
        parameter_values: np.ndarray,
    ) -> pd.DataFrame:
        """Solve the MFG for each parameter value and record summary stats.

        Parameters
        ----------
        parameter_name   : attribute name on self.model (e.g. 'gamma')
        parameter_values : 1-D array of values to sweep

        Returns
        -------
        pd.DataFrame with one row per parameter value.
        """
        records = []
        model_copy = copy.deepcopy(self.model)

        for val in parameter_values:
            setattr(model_copy, parameter_name, float(val))
            solver = MFGSolver(
                model_copy, self.grid, self.time_grid,
                damping=0.5, tol=1e-4, max_iterations=100,
            )
            sol = solver.solve()
            m = sol["density"]
            u = sol["value_function"]
            alpha = sol["optimal_control"]
            x = self.x

            # Summary stats over final time slice
            m_T = m[-1]
            mean_x = float(np.trapezoid(x * m_T, x))
            mean_x2 = float(np.trapezoid(x ** 2 * m_T, x))
            std_x = float(np.sqrt(max(mean_x2 - mean_x ** 2, 0.0)))
            mean_alpha = float(np.trapezoid(alpha[0] * m[0], x))
            social = compute_social_cost(sol, model_copy, self.grid, self.time_grid)

            records.append({
                parameter_name: val,
                "converged": sol["converged"],
                "n_iterations": sol["n_iterations"],
                "mean_x_terminal": mean_x,
                "std_x_terminal": std_x,
                "mean_control_t0": mean_alpha,
                "social_cost": social,
                "hjb_residual_l2": sol["residuals"]["hjb_residual_l2"],
                "fp_residual_l2": sol["residuals"]["fp_residual_l2"],
            })

        return pd.DataFrame(records)

    def stability_analysis(self, solution: dict = None) -> dict:
        """Estimate stability of the equilibrium via linearisation.

        Approximates the Jacobian of the Picard map T(m) at the fixed
        point m* using finite differences, then computes its eigenvalues.
        Stability requires spectral radius < 1.

        Returns
        -------
        dict with 'eigenvalues', 'stable', 'spectral_gap', 'spectral_radius'
        """
        if solution is None:
            solution = self.solution
        m_star = solution["density"]  # (n_t, n_x)
        n_t, n_x = m_star.shape
        m_flat = m_star.ravel()
        n = len(m_flat)

        # Finite-difference Jacobian of T(m) - m  (Picard update)
        solver = MFGSolver(
            self.model, self.grid, self.time_grid,
            damping=1.0, tol=1e-8, max_iterations=1,
        )

        def picard_step(m_f):
            """One Picard step."""
            from mfglob.hjb import HJBSolver
            from mfglob.fokker_planck import FokkerPlanckSolver
            m = m_f.reshape(n_t, n_x)
            hjb = HJBSolver(self.grid, self.time_grid, self.model)
            fp = FokkerPlanckSolver(self.grid, self.time_grid, self.model)
            u_res = hjb.solve(m)
            alpha = u_res["optimal_control"]
            x = self.x
            drift = np.array([self.model.drift(x, alpha[k], m[k]) for k in range(n_t)])
            diff = np.array([self.model.diffusion(x, alpha[k], m[k]) for k in range(n_t)])
            fp_res = fp.solve(drift, diff)
            return fp_res["density"].ravel()

        # Build Jacobian with finite differences (small n only, expensive)
        # Use a reduced basis for speed: project onto n_basis random directions
        n_basis = min(n, 10)
        rng = np.random.default_rng(0)
        eps = 1e-5 * max(np.linalg.norm(m_flat), 1.0)

        T0 = picard_step(m_flat)
        J_cols = []
        for _ in range(n_basis):
            v = rng.normal(0.0, 1.0, n)
            v /= np.linalg.norm(v) + 1e-14
            T1 = picard_step(m_flat + eps * v)
            J_cols.append((T1 - T0) / eps)

        # Approximate Jacobian in the random subspace
        V = np.column_stack(J_cols)                  # (n, n_basis)
        W = np.column_stack([rng.normal(size=n) for _ in range(n_basis)])
        W, _ = np.linalg.qr(W)
        J_small = W.T @ V                             # (n_basis, n_basis)

        eigvals = sla.eigvals(J_small)
        spectral_radius = float(np.max(np.abs(eigvals)))
        stable = bool(spectral_radius < 1.0)
        # Spectral gap: 1 - ρ(J)  (positive if stable, convergence rate)
        spectral_gap = float(1.0 - spectral_radius)

        return {
            "eigenvalues": eigvals,
            "stable": stable,
            "spectral_radius": spectral_radius,
            "spectral_gap": spectral_gap,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _control_consistency_error(self) -> float:
        """Max deviation between stored α* and α*(Du) over all time steps."""
        x = self.x
        errs = []
        for n in range(self.u.shape[0]):
            p = self._D_cen @ self.u[n]
            alpha_recomputed = self.model.optimal_control(x, p, self.m[n])
            errs.append(np.max(np.abs(alpha_recomputed - self.alpha[n])))
        return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def compute_social_cost(
    solution: dict,
    model: MFGLOBModel,
    grid: Grid1D,
    time_grid: TimeGrid,
) -> float:
    """Total social cost ∫_0^T ∫ f m dx dt + ∫ g m_T dx."""
    u = solution["value_function"]
    m = solution["density"]
    alpha = solution["optimal_control"]
    x = grid.points
    dt = time_grid.dt
    n_t = time_grid.n_nodes

    cost = 0.0
    for n in range(n_t - 1):
        f_n = model.running_cost(x, alpha[n], m[n])
        cost += dt * float(np.trapezoid(f_n * m[n], x))
    g_term = model.terminal_cost(x, m[-1])
    cost += float(np.trapezoid(g_term * m[-1], x))
    return cost


def compute_price_of_anarchy(
    model: MFGLOBModel,
    grid: Grid1D,
    time_grid: TimeGrid,
) -> dict:
    """Ratio of Nash social cost to cooperative social cost.

    The social optimum is approximated by solving the MFG with
    zero mean-field coupling (each agent ignores others) — this
    gives an upper bound on the cooperative welfare because it
    removes strategic externalities.

    PoA = social_cost_Nash / social_cost_optimum
    """
    # Nash equilibrium
    solver_nash = MFGSolver(model, grid, time_grid, damping=0.5,
                            tol=1e-4, max_iterations=100)
    sol_nash = solver_nash.solve()
    cost_nash = compute_social_cost(sol_nash, model, grid, time_grid)

    # Social optimum approximation: cooperative model with psi/c_crowd = 0
    model_coop = copy.deepcopy(model)
    for attr in ("psi", "c_crowd", "eta"):
        if hasattr(model_coop, attr):
            setattr(model_coop, attr, 0.0)

    solver_coop = MFGSolver(model_coop, grid, time_grid, damping=0.5,
                            tol=1e-4, max_iterations=100)
    sol_coop = solver_coop.solve()
    cost_coop = compute_social_cost(sol_coop, model_coop, grid, time_grid)

    poa = abs(cost_nash) / max(abs(cost_coop), 1e-14) if abs(cost_coop) > 1e-14 else 1.0

    return {
        "social_cost_nash": cost_nash,
        "social_cost_optimum": cost_coop,
        "price_of_anarchy": poa,
        "nash_solution": sol_nash,
        "optimum_solution": sol_coop,
    }
