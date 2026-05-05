"""Price impact extraction from MFG equilibrium solutions.

For optimal execution models:
  - Temporary impact: η · M(t)  where M(t) = ∫ ν*(t,q) m(t,q) dq
  - Permanent impact: η · ∫_0^T M(t) dt  (total price displacement)
  - Impact curve: ΔS(Q) as function of meta-order size Q
  - Square-root law comparison and Kyle's λ
"""

from __future__ import annotations

import copy
import numpy as np

from mfglob.grids import Grid1D, TimeGrid
from mfglob.models.base import MFGLOBModel
from mfglob.mfg_solver import MFGSolver


class PriceImpactAnalyzer:
    """Extract price impact curves from an MFG equilibrium.

    Designed for OptimalExecutionMFG, but works with any model that has
    an `eta` attribute (permanent impact coefficient) and whose optimal
    control represents an execution rate.

    Parameters
    ----------
    solution  : dict returned by MFGSolver.solve()
    model     : MFGLOBModel with .eta attribute
    grid      : Grid1D  (state = remaining inventory)
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

        self.x = grid.points
        self.times = time_grid.times
        self.dt = time_grid.dt

        alpha = solution["optimal_control"]  # (n_t, n_x)
        m = solution["density"]              # (n_t, n_x)
        # Aggregate execution rate M(t) = ∫ α(t,q) m(t,q) dq
        # For execution models, drift = -ν, so ν = -drift  (non-negative)
        self._nu = np.maximum(alpha, 0.0)   # treat control as execution rate
        self._m = m
        self._M = np.array([
            float(np.trapezoid(self._nu[n] * m[n], self.x))
            for n in range(time_grid.n_nodes)
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def temporary_impact(self) -> np.ndarray:
        """η · M(t) — temporary price impact at each time node.

        Returns shape (n_times,).
        """
        eta = getattr(self.model, "eta", 1.0)
        return eta * self._M

    def permanent_impact(self, meta_order_size: float = None) -> float:
        """η · ∫_0^T M(t) dt — total permanent price displacement.

        Parameters
        ----------
        meta_order_size : if provided, rescales by Q / model.Q0 to
                          approximate the impact for a different order size
                          without re-solving the full PDE.
        """
        eta = getattr(self.model, "eta", 1.0)
        integral = float(np.trapezoid(self._M, self.times))
        impact = eta * integral

        if meta_order_size is not None:
            Q0 = getattr(self.model, "Q0", 1.0)
            if Q0 > 1e-14:
                impact *= meta_order_size / Q0

        return impact

    def impact_curve(self, order_sizes: np.ndarray = None) -> dict:
        """Compute ΔS(Q) by re-solving the MFG for each order size Q.

        Parameters
        ----------
        order_sizes : 1-D array of Q values.  Defaults to
                      np.linspace(0.1, 2.0, 10) * model.Q0

        Returns
        -------
        dict with 'order_sizes', 'permanent_impact', 'temporary_impact_peak'
        """
        Q0 = getattr(self.model, "Q0", 1.0)
        if order_sizes is None:
            order_sizes = np.linspace(0.1, 2.0, 10) * Q0

        perm_impacts = []
        temp_peak = []

        for Q in order_sizes:
            impact, t_peak = self._solve_for_Q(float(Q))
            perm_impacts.append(impact)
            temp_peak.append(t_peak)

        return {
            "order_sizes": np.asarray(order_sizes),
            "permanent_impact": np.asarray(perm_impacts),
            "temporary_impact_peak": np.asarray(temp_peak),
        }

    def compare_to_square_root_law(self, order_sizes: np.ndarray = None) -> dict:
        """Fit a power law ΔS = a Q^b and compare to the empirical √Q law.

        The empirical square-root law: ΔS ∝ √Q (exponent b ≈ 0.5).
        MFG models typically give b ∈ [0.4, 0.6] for realistic parameters.

        Returns
        -------
        dict with keys 'order_sizes', 'mfg_impact', 'sqrt_law',
                        'mfg_exponent', 'mfg_coefficient'
        """
        Q0 = getattr(self.model, "Q0", 1.0)
        if order_sizes is None:
            order_sizes = np.linspace(0.2, 2.0, 8) * Q0

        curve = self.impact_curve(order_sizes)
        Qs = curve["order_sizes"]
        impacts = curve["permanent_impact"]

        # Fit log(ΔS) = log(a) + b·log(Q) via linear regression on positive values
        mask = (Qs > 1e-10) & (impacts > 1e-10)
        if mask.sum() >= 2:
            log_Q = np.log(Qs[mask])
            log_I = np.log(impacts[mask])
            b, log_a = np.polyfit(log_Q, log_I, 1)
            a = np.exp(log_a)
        else:
            b, a = 0.5, float(np.mean(impacts[mask])) if mask.any() else 1.0

        # Square-root benchmark anchored at the largest order
        ref_Q = Qs[-1] if len(Qs) else Q0
        ref_I = impacts[-1] if len(impacts) else 1.0
        sqrt_coeff = ref_I / max(np.sqrt(ref_Q), 1e-14)
        sqrt_law = sqrt_coeff * np.sqrt(Qs)

        return {
            "order_sizes": Qs,
            "mfg_impact": impacts,
            "sqrt_law": sqrt_law,
            "mfg_exponent": float(b),
            "mfg_coefficient": float(a),
        }

    def kyle_lambda(self) -> float:
        """Kyle's lambda: slope of ΔS(Q) at Q → 0.

        Estimated as ΔS / Q for the smallest available Q.
        """
        Q0 = getattr(self.model, "Q0", 1.0)
        small_Qs = np.array([0.05, 0.1, 0.15]) * Q0
        impacts = []
        for Q in small_Qs:
            imp, _ = self._solve_for_Q(float(Q))
            impacts.append(imp)
        impacts = np.asarray(impacts)

        # Linear fit ΔS ≈ λ Q
        if np.all(small_Qs > 1e-14):
            lam = float(np.mean(impacts / small_Qs))
        else:
            lam = 0.0
        return max(lam, 0.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _solve_for_Q(self, Q: float) -> tuple[float, float]:
        """Re-solve the MFG with Q0=Q, return (permanent_impact, peak_temp_impact).

        Uses a grid that covers [0, 1.6·Q] so that the initial distribution
        is always within the domain regardless of Q.
        """
        model_q = copy.deepcopy(self.model)
        if hasattr(model_q, "Q0"):
            model_q.Q0 = Q

        # Build a domain large enough to contain the initial distribution at Q
        x_max = max(Q * 1.6, 0.5)
        from mfglob.grids import Grid1D as _G
        grid_q = _G(0.0, x_max, self.grid.n)

        solver = MFGSolver(
            model_q, grid_q, self.time_grid,
            damping=0.5, tol=1e-4, max_iterations=50,
        )
        sol = try_solve(solver)
        x_q = grid_q.points

        alpha = np.maximum(sol["optimal_control"], 0.0)
        m = sol["density"]
        M = np.array([
            float(np.trapezoid(alpha[n] * m[n], x_q))
            for n in range(self.time_grid.n_nodes)
        ])
        eta = getattr(model_q, "eta", 1.0)
        perm = eta * float(np.trapezoid(M, self.times))
        temp_peak = eta * float(np.max(M))
        return perm, temp_peak


def try_solve(solver: MFGSolver) -> dict:
    """Run solver.solve() with a fallback to 10 Picard iterations."""
    try:
        return solver.solve()
    except Exception:
        solver.max_iterations = 10
        solver.tol = 1e-2
        return solver.solve()
