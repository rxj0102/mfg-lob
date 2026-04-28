"""MFG-LOB calibration to empirical data.

Finds model parameters (sigma, phi, alpha, kappa) that minimise the distance
between the MFG equilibrium statistics and their empirical counterparts:

Objective (composite loss):
    L(θ) = w_spread * (spread_model - spread_emp)²
          + w_depth * ||depth_model - depth_emp||²
          + w_impact * ||impact_model - impact_emp||²

Optimisation: Nelder-Mead (gradient-free) over the log-parameter space.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize

from mfg_lob.models.lob_model import LOBModel
from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.mfg_solver import MFGSolver
from mfg_lob.analysis.price_impact import PriceImpactAnalyzer
from mfg_lob.analysis.lob_shape import LOBShapeAnalyzer
from mfg_lob.calibration.empirical import EmpiricalLOB


@dataclass
class CalibrationResult:
    """Result of MFG model calibration.

    Attributes
    ----------
    params : dict
        Calibrated parameter values.
    loss : float
        Final objective value.
    n_evaluations : int
        Total number of solver calls.
    success : bool
    message : str
    loss_history : list of float
    param_history : list of dict
    """

    params: dict
    loss: float
    n_evaluations: int
    success: bool
    message: str
    loss_history: list = field(default_factory=list)
    param_history: list = field(default_factory=list)


class MFGCalibrator:
    """Calibrate MFG-LOB model parameters to empirical LOB data.

    Parameters
    ----------
    empirical : EmpiricalLOB
        Empirical data to calibrate against.
    grid : Grid
        Computational grid (fixed during calibration).
    param_names : list of str
        Names of LOBModel parameters to calibrate.
    param_bounds : list of (lo, hi) tuples
        Bounds on each parameter (in original, not log, space).
    base_params : dict
        Fixed (non-calibrated) parameter values.
    weights : dict, optional
        Relative weights for spread, depth, and impact in the loss.
    solver_kwargs : dict, optional
        Keyword arguments for MFGSolver.
    tick_size : float
        Price tick used for depth profile matching.
    n_depth_levels : int
        Number of price levels in the depth profile objective.
    n_impact_points : int
        Number of volume points in the price impact objective.
    """

    def __init__(
        self,
        empirical: EmpiricalLOB,
        grid: Grid,
        param_names: list[str] | None = None,
        param_bounds: list[tuple[float, float]] | None = None,
        base_params: dict | None = None,
        weights: dict | None = None,
        solver_kwargs: dict | None = None,
        tick_size: float = 0.01,
        n_depth_levels: int = 10,
        n_impact_points: int = 20,
    ) -> None:
        self.empirical = empirical
        self.grid = grid
        self.param_names = param_names or ["sigma", "phi", "alpha"]
        self.param_bounds = param_bounds or [(0.01, 5.0)] * len(self.param_names)
        self.base_params = base_params or {}
        self.weights = weights or {"spread": 1.0, "depth": 1.0, "impact": 0.5}
        self.solver_kwargs = solver_kwargs or {"max_iter": 50, "tol": 1e-5}
        self.tick_size = tick_size
        self.n_depth_levels = n_depth_levels
        self.n_impact_points = n_impact_points

        # Pre-compute empirical targets
        self._emp_spread = empirical.mean_spread
        emp_prof = empirical.mean_depth_profile(n_depth_levels, tick_size)
        self._emp_depth = (emp_prof["bid_cum"] + emp_prof["ask_cum"]) / 2.0
        self._emp_levels = emp_prof["levels"]
        impact_vols = np.linspace(0, self._emp_depth[-1] * 0.5, n_impact_points)
        self._emp_impact = empirical.empirical_impact(
            impact_vols, tick_size=tick_size, n_levels=n_depth_levels
        )
        self._impact_vols = impact_vols

        self._n_evals = 0
        self._loss_history: list[float] = []
        self._param_history: list[dict] = []

    def _params_from_vec(self, x: np.ndarray) -> dict:
        """Convert log-parameter vector → parameter dict."""
        params = dict(self.base_params)
        for name, log_val in zip(self.param_names, x):
            params[name] = float(np.exp(log_val))
        return params

    def _loss(self, x: np.ndarray) -> float:
        params = self._params_from_vec(x)
        try:
            model = LOBModel(**params)
            solver = MFGSolver(model, self.grid, **self.solver_kwargs)
            eq = solver.solve()
        except Exception:
            return 1e6

        pia = PriceImpactAnalyzer(eq, price_tick=self.tick_size)
        lsa = LOBShapeAnalyzer(eq)

        # Spread objective
        snap = lsa.snapshot(t=0.0)
        spread_model = snap.spread_proxy
        loss_spread = (spread_model - self._emp_spread) ** 2 / (self._emp_spread ** 2 + 1e-14)

        # Depth profile objective — normalise each profile to unit mean
        # to remove the scale mismatch between model density and empirical volume counts.
        prof = lsa.depth_profile(t=0.0, n_levels=self.n_depth_levels)
        depth_model = (prof["bid_volume"] + prof["ask_volume"]) / 2.0
        depth_model_norm = depth_model / (np.mean(depth_model) + 1e-14)
        emp_depth_norm = self._emp_depth / (np.mean(self._emp_depth) + 1e-14)
        loss_depth = float(np.mean((depth_model_norm - emp_depth_norm) ** 2))

        # Price impact objective — same normalisation
        curve = pia.compute(t=0.0, n_volumes=self.n_impact_points)
        imp_model = np.interp(self._impact_vols, curve.volumes, curve.impact)
        imp_model_norm = imp_model / (np.mean(imp_model) + 1e-14)
        emp_impact_norm = self._emp_impact / (np.mean(self._emp_impact) + 1e-14)
        loss_impact = float(np.mean((imp_model_norm - emp_impact_norm) ** 2))

        w = self.weights
        total = (w["spread"] * loss_spread
                 + w["depth"] * loss_depth
                 + w["impact"] * loss_impact)

        self._n_evals += 1
        self._loss_history.append(float(total))
        self._param_history.append(params)
        return float(total)

    def calibrate(
        self,
        x0: np.ndarray | None = None,
        method: str = "Nelder-Mead",
        max_iter: int = 200,
        verbose: bool = False,
    ) -> CalibrationResult:
        """Run the calibration optimisation.

        Parameters
        ----------
        x0 : ndarray, optional
            Initial parameter guess in log-space. Defaults to log of midpoint
            of each parameter's bounds.
        method : str
            scipy.optimize.minimize method.
        max_iter : int
            Maximum optimizer iterations.
        verbose : bool

        Returns
        -------
        CalibrationResult
        """
        self._n_evals = 0
        self._loss_history = []
        self._param_history = []

        lo = np.array([b[0] for b in self.param_bounds])
        hi = np.array([b[1] for b in self.param_bounds])

        if x0 is None:
            x0 = np.log(np.sqrt(lo * hi))  # geometric midpoint
        else:
            x0 = np.log(np.asarray(x0, dtype=float))

        if verbose:
            print(f"Calibrating {self.param_names} via {method}...")

        opt = minimize(
            self._loss,
            x0,
            method=method,
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6,
                     "disp": verbose},
        )

        best_params = self._params_from_vec(opt.x)

        return CalibrationResult(
            params=best_params,
            loss=float(opt.fun),
            n_evaluations=self._n_evals,
            success=bool(opt.success),
            message=str(opt.message),
            loss_history=list(self._loss_history),
            param_history=list(self._param_history),
        )
