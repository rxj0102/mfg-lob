"""Comparative statics: how MFG equilibrium changes with model parameters.

For each parameter sweep we:
  1. Solve the MFG for a sequence of parameter values.
  2. Extract equilibrium statistics (spread, depth, price impact exponent, etc.).
  3. Return a structured result for plotting and analysis.

This module implements the publishable contribution of numerically tracing
comparative statics of the MFG-LOB equilibrium.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

from mfg_lob.models.lob_model import LOBModel
from mfg_lob.solvers.grid import Grid
from mfg_lob.solvers.mfg_solver import MFGSolver, MFGEquilibrium
from mfg_lob.analysis.price_impact import PriceImpactAnalyzer
from mfg_lob.analysis.lob_shape import LOBShapeAnalyzer


@dataclass
class StaticResult:
    """Result of a single parameter point in the comparative statics sweep.

    Attributes
    ----------
    param_value : Any
        The parameter value used.
    equilibrium : MFGEquilibrium
    spread_proxy : float
        Equilibrium half-spread proxy at t=0.
    impact_beta : float
        Power-law exponent of the price impact curve.
    impact_Y : float
        Scale coefficient of the price impact curve.
    total_depth : float
        Total liquidity (integral of density at t=0).
    converged : bool
    n_iterations : int
    """

    param_value: Any
    equilibrium: MFGEquilibrium
    spread_proxy: float
    impact_beta: float
    impact_Y: float
    total_depth: float
    converged: bool
    n_iterations: int


@dataclass
class StaticsResult:
    """Collection of results from a full comparative statics sweep."""

    param_name: str
    param_values: np.ndarray
    results: list[StaticResult] = field(default_factory=list)

    @property
    def spread_proxies(self) -> np.ndarray:
        return np.array([r.spread_proxy for r in self.results])

    @property
    def impact_betas(self) -> np.ndarray:
        return np.array([r.impact_beta for r in self.results])

    @property
    def impact_Ys(self) -> np.ndarray:
        return np.array([r.impact_Y for r in self.results])

    @property
    def total_depths(self) -> np.ndarray:
        return np.array([r.total_depth for r in self.results])

    @property
    def converged(self) -> np.ndarray:
        return np.array([r.converged for r in self.results])

    def plot(self, ax=None, fig=None):
        """Plot comparative statics: spread, depth, and impact exponent."""
        import matplotlib.pyplot as plt
        if fig is None or ax is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            axes = ax

        pv = self.param_values
        axes[0].plot(pv, self.spread_proxies, "o-")
        axes[0].set_xlabel(self.param_name)
        axes[0].set_ylabel("Spread proxy")
        axes[0].set_title("Equilibrium Spread")

        axes[1].plot(pv, self.total_depths, "s-", color="steelblue")
        axes[1].set_xlabel(self.param_name)
        axes[1].set_ylabel("Total depth")
        axes[1].set_title("Liquidity Depth")

        axes[2].plot(pv, self.impact_betas, "^-", color="tomato")
        axes[2].set_xlabel(self.param_name)
        axes[2].set_ylabel("Impact exponent β")
        axes[2].set_title("Price Impact Power Law")
        axes[2].axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="β=0.5 (empirical)")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        return fig, axes


class ComparativeStatics:
    """Run comparative statics sweeps over model parameters.

    Parameters
    ----------
    base_model : LOBModel
        Base parameter configuration.
    grid : Grid
    solver_kwargs : dict
        Keyword arguments passed to MFGSolver (e.g. tol, max_iter, damping).
    """

    def __init__(
        self,
        base_model: LOBModel,
        grid: Grid,
        solver_kwargs: dict | None = None,
    ) -> None:
        self.base_model = base_model
        self.grid = grid
        self.solver_kwargs = solver_kwargs or {}

    def _make_model(self, param_name: str, param_value: float) -> LOBModel:
        """Create a modified model with one parameter changed."""
        kwargs = {
            "sigma": self.base_model.sigma,
            "phi": self.base_model.phi,
            "alpha": self.base_model.alpha,
            "A_term": self.base_model.A_term,
            "B_term": self.base_model.B_term,
            "Lambda0": self.base_model.Lambda0,
            "kappa": self.base_model.kappa,
            "epsilon": self.base_model.epsilon,
            "mu0": self.base_model.mu0,
            "sigma0": self.base_model.sigma0,
            "agent_type": self.base_model.agent_type,
        }
        kwargs[param_name] = param_value
        return LOBModel(**kwargs)

    def sweep(
        self,
        param_name: str,
        param_values: np.ndarray,
        verbose: bool = False,
    ) -> StaticsResult:
        """Sweep a single model parameter and compute equilibrium statistics.

        Parameters
        ----------
        param_name : str
            Name of the LOBModel parameter to vary.
        param_values : array-like
            Values to sweep over.
        verbose : bool

        Returns
        -------
        StaticsResult
        """
        param_values = np.asarray(param_values, dtype=float)
        result = StaticsResult(param_name=param_name, param_values=param_values)

        for i, val in enumerate(param_values):
            if verbose:
                print(f"  [{i+1}/{len(param_values)}] {param_name} = {val:.4g}")

            model = self._make_model(param_name, val)
            solver = MFGSolver(model, self.grid, **self.solver_kwargs)
            eq = solver.solve()

            pia = PriceImpactAnalyzer(eq)
            lsa = LOBShapeAnalyzer(eq)

            snap = lsa.snapshot(t=0.0)
            impact_curve = pia.compute(t=0.0)

            total_depth = float(np.trapezoid(snap.density, snap.q))
            beta = impact_curve.fit_params.get("beta", float("nan"))
            Y = impact_curve.fit_params.get("Y", float("nan"))

            result.results.append(StaticResult(
                param_value=val,
                equilibrium=eq,
                spread_proxy=snap.spread_proxy,
                impact_beta=float(beta) if not np.isnan(float(str(beta))) else float("nan"),
                impact_Y=float(Y) if not np.isnan(float(str(Y))) else float("nan"),
                total_depth=total_depth,
                converged=eq.converged,
                n_iterations=eq.n_iterations,
            ))

        return result
