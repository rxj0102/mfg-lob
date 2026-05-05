"""Calibration of MFG-LOB model parameters to empirical LOB data.

Minimises a composite loss over spread, depth profile, and price impact.
Optimisation: differential_evolution, Nelder-Mead, or two-stage.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as so

from mfglob.grids import Grid1D, TimeGrid
from mfglob.mfg_solver import MFGSolver
from mfglob.price_impact import PriceImpactAnalyzer


class MFGCalibrator:
    """Calibrate MFG-LOB model parameters to target empirical data.

    Parameters
    ----------
    model_class : class (e.g. AvellanedaStoikovMFG)
    target_data : dict with optional keys:
        'book_shape'   — dict with 'levels' (array), 'density' (array)
        'impact_curve' — dict with 'order_sizes', 'permanent_impact'
        'spread'       — float
    param_bounds : {param_name: (lower, upper)}
    grid_params  : dict with n_x, n_t, T, x_min, x_max
    """

    def __init__(
        self,
        model_class,
        target_data: dict,
        param_bounds: dict = None,
        grid_params: dict = None,
    ) -> None:
        self.model_class = model_class
        self.target_data = target_data
        self.param_bounds = param_bounds or {}
        self._param_names = list(self.param_bounds.keys())

        # Grid defaults
        gp = grid_params or {}
        self._n_x = int(gp.get("n_x", 20))
        self._n_t = int(gp.get("n_t", 15))
        self._T = float(gp.get("T", 0.1))
        self._x_min = gp.get("x_min", None)
        self._x_max = gp.get("x_max", None)

        # Weights
        self._w_book = 1.0
        self._w_impact = 1.0
        self._w_spread = 1.0

        self._n_evals = 0

    # ------------------------------------------------------------------
    # Build model from param vector
    # ------------------------------------------------------------------

    def _build_model(self, params: np.ndarray):
        """Instantiate model_class and set each parameter by name."""
        model = self.model_class()
        for name, val in zip(self._param_names, params):
            setattr(model, name, float(val))
        return model

    def _build_grids(self, model):
        """Build Grid1D and TimeGrid (use model domain if x_min/x_max not set)."""
        x_min = self._x_min
        x_max = self._x_max
        if x_min is None:
            x_min = getattr(model, "x_min", getattr(model, "Q_max", 1.0) * -1.0)
        if x_max is None:
            x_max = getattr(model, "x_max", getattr(model, "Q_max", 1.0))
        # Ensure valid domain
        if x_min >= x_max:
            x_min, x_max = -1.0, 1.0
        grid = Grid1D(float(x_min), float(x_max), self._n_x)
        tgrid = TimeGrid(self._T, self._n_t)
        return grid, tgrid

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self, params: np.ndarray) -> float:
        """Compute weighted MSE loss between MFG solution and target_data."""
        self._n_evals += 1
        try:
            model = self._build_model(params)
            grid, tgrid = self._build_grids(model)

            solver = MFGSolver(
                model, grid, tgrid,
                damping=0.5, tol=1e-3, max_iterations=30,
            )
            solution = solver.solve(verbose=False)

            m = solution["density"]  # shape (n_t+1, n_x)
            x = grid.points

            loss = 0.0

            # --- Book shape term ---
            if "book_shape" in self.target_data:
                bs = self.target_data["book_shape"]
                target_levels = np.asarray(bs["levels"], dtype=float)
                target_density = np.asarray(bs["density"], dtype=float)

                # Use terminal density m[-1]
                terminal_m = m[-1]
                # Normalise for comparison
                mass = np.trapezoid(terminal_m, x)
                if mass > 1e-14:
                    terminal_m = terminal_m / mass

                # Interpolate onto target levels
                model_density = np.interp(target_levels, x, terminal_m)

                # Normalise target
                td = target_density.copy()
                td_mass = np.trapezoid(td, target_levels) if len(target_levels) > 1 else td.sum()
                if td_mass > 1e-14:
                    td = td / td_mass

                diff = model_density - td
                loss += self._w_book * float(np.mean(diff ** 2))

            # --- Impact curve term ---
            if (
                "impact_curve" in self.target_data
                and hasattr(model, "eta")
                and getattr(model, "eta", 0.0) > 0
            ):
                ic = self.target_data["impact_curve"]
                target_sizes = np.asarray(ic["order_sizes"], dtype=float)
                target_impact = np.asarray(ic["permanent_impact"], dtype=float)

                try:
                    analyzer = PriceImpactAnalyzer(solution, model, grid, tgrid)
                    model_curve = analyzer.impact_curve(target_sizes)
                    model_impact = model_curve["permanent_impact"]
                    diff = model_impact - target_impact
                    loss += self._w_impact * float(np.mean(diff ** 2))
                except Exception:
                    loss += self._w_impact * 1.0

            # --- Spread term ---
            if "spread" in self.target_data:
                target_spread = float(self.target_data["spread"])
                # Use 1/k as equilibrium spread proxy if model has k attr
                if hasattr(model, "k") and getattr(model, "k", 0.0) > 1e-14:
                    model_spread = 1.0 / model.k
                else:
                    model_spread = target_spread  # no penalty
                loss += self._w_spread * (model_spread - target_spread) ** 2

            return float(loss)

        except Exception:
            return 1e10

    # ------------------------------------------------------------------
    # Calibrate
    # ------------------------------------------------------------------

    def calibrate(
        self,
        method: str = "differential_evolution",
        n_workers: int = 1,
        verbose: bool = True,
    ) -> dict:
        """Run optimisation to find best-fit parameters.

        Parameters
        ----------
        method   : 'differential_evolution', 'nelder_mead', 'two_stage'
        n_workers: number of parallel workers (for DE)
        verbose  : print progress

        Returns
        -------
        dict with keys: 'params', 'objective', 'success', 'n_evals', 'message'
        """
        self._n_evals = 0
        bounds_list = [self.param_bounds[name] for name in self._param_names]
        lowers = np.array([b[0] for b in bounds_list])
        uppers = np.array([b[1] for b in bounds_list])

        if method == "differential_evolution":
            result = self._run_de(bounds_list, verbose)
            x_opt = result.x
            success = result.success
            message = result.message

        elif method == "nelder_mead":
            result = self._run_nelder_mead(lowers, uppers, verbose)
            x_opt = result.x
            success = result.success
            message = result.message

        elif method == "two_stage":
            # Stage 1: DE
            de_result = self._run_de(bounds_list, verbose)
            x0_nm = np.clip(de_result.x, lowers, uppers)
            # Stage 2: Nelder-Mead refine
            nm_result = self._run_nelder_mead(lowers, uppers, verbose, x0=x0_nm)
            x_opt = nm_result.x
            success = nm_result.success
            message = f"two_stage: DE then NM. {nm_result.message}"

        else:
            raise ValueError(f"Unknown method: {method!r}. "
                             "Choose 'differential_evolution', 'nelder_mead', or 'two_stage'.")

        # Build param dict
        params_dict = {name: float(x_opt[i]) for i, name in enumerate(self._param_names)}
        obj_val = self.objective(x_opt)

        return {
            "params": params_dict,
            "objective": float(obj_val),
            "success": bool(success),
            "n_evals": self._n_evals,
            "message": str(message),
        }

    # ------------------------------------------------------------------
    # Private optimisation helpers
    # ------------------------------------------------------------------

    def _run_de(self, bounds_list, verbose):
        """Run scipy differential_evolution."""
        return so.differential_evolution(
            self.objective,
            bounds=bounds_list,
            maxiter=50,
            popsize=5,
            seed=42,
            disp=verbose,
            tol=1e-4,
        )

    def _run_nelder_mead(self, lowers, uppers, verbose, x0=None):
        """Run Nelder-Mead in normalised [0,1]^d space."""
        ranges = uppers - lowers

        def to_unit(x):
            return (x - lowers) / np.where(ranges > 1e-14, ranges, 1.0)

        def from_unit(u):
            return lowers + u * ranges

        def obj_unit(u):
            x = from_unit(u)
            x = np.clip(x, lowers, uppers)
            return self.objective(x)

        if x0 is None:
            # Start from centre of parameter space
            x0 = 0.5 * (lowers + uppers)
        u0 = to_unit(x0)

        result_nm = so.minimize(
            obj_unit,
            u0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4, "disp": verbose},
        )

        # Transform back
        x_opt_unit = result_nm.x
        x_opt = from_unit(x_opt_unit)
        x_opt = np.clip(x_opt, lowers, uppers)

        # Patch result to use actual params
        result_nm.x = x_opt
        return result_nm

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def diagnostic_plots(self, solution: dict) -> None:
        """Plot book shape, impact curve, and spread diagnostics."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            # matplotlib not available — skip plotting silently
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        m = solution.get("density")
        if m is not None:
            axes[0].plot(m[-1], label="Terminal density m[-1]")
            axes[0].set_title("Book Shape (terminal density)")
            axes[0].set_xlabel("Grid index")
            axes[0].set_ylabel("Density")
            axes[0].legend()

        alpha = solution.get("optimal_control")
        if alpha is not None:
            axes[1].plot(alpha[-1], label="Optimal control (terminal)")
            axes[1].set_title("Impact Curve (control proxy)")
            axes[1].set_xlabel("Grid index")
            axes[1].set_ylabel("Control")
            axes[1].legend()

        ch = solution.get("convergence_history")
        if ch is not None and len(ch) > 0:
            axes[2].semilogy(ch, label="Convergence history")
            axes[2].set_title("Spread / Convergence")
            axes[2].set_xlabel("Iteration")
            axes[2].set_ylabel("Error")
            axes[2].legend()

        plt.tight_layout()
        plt.close(fig)
