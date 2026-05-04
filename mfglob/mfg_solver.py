"""Coupled MFG system solver.

Two solvers:
  MFGSolver       — Picard fixed-point iteration (Lasry-Lions)
  NewtonMFGSolver — Newton-Krylov refinement (quadratic convergence)

Picard iteration:
  1. Initialise m^(0)
  2. Solve HJB backward given m^(k)  →  u^(k+1), α*^(k+1)
  3. Build drift/diffusion fields from α*^(k+1) and m^(k)
  4. Solve FP forward with those fields  →  m̃^(k+1)
  5. Damp:  m^(k+1) = λ·m̃^(k+1) + (1−λ)·m^(k)  then renormalise
  6. Check ‖m^(k+1) − m^(k)‖ < tol
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mfglob.grids import Grid1D, TimeGrid
from mfglob.hjb import HJBSolver
from mfglob.fokker_planck import FokkerPlanckSolver
from mfglob.operators import FiniteDifferenceOperators


# ---------------------------------------------------------------------------
# MFGSolver
# ---------------------------------------------------------------------------

class MFGSolver:
    """
    Solve the coupled MFG system via Picard fixed-point iteration.

    HJB:  −∂u/∂t + H(x, Du, D²u, m) = 0,   u(T) = g(x, m(T))
    FP:   ∂m/∂t = (σ²/2)∂²m/∂x² − ∂(b*m)/∂x,  m(0) = m₀

    Parameters
    ----------
    model               : MFGLOBModel
    grid                : Grid1D
    time_grid           : TimeGrid
    damping             : λ ∈ (0, 1] — Picard mixing weight
    max_iterations      : maximum Picard iterations
    tol                 : convergence tolerance on ‖Δm‖
    convergence_metric  : 'l2', 'linf', or 'w2'
    """

    def __init__(
        self,
        model,
        grid: Grid1D,
        time_grid: TimeGrid,
        damping: float = 0.5,
        max_iterations: int = 200,
        tol: float = 1e-6,
        convergence_metric: str = "l2",
    ) -> None:
        self.model = model
        self.grid = grid
        self.time_grid = time_grid
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.tol = float(tol)
        self.convergence_metric = convergence_metric

        self.x = grid.points
        self.n_x = grid.n
        self.n_times = time_grid.n_nodes
        self.dt = time_grid.dt

        self._hjb = HJBSolver(grid=grid, time_grid=time_grid, model=model)
        self._fp = FokkerPlanckSolver(grid=grid, time_grid=time_grid, model=model)

        ops = FiniteDifferenceOperators(grid)
        self._D_cen = ops.d_dx_central()
        self._D2 = ops.d2_dx2()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, verbose: bool = False) -> dict:
        """
        Solve the coupled MFG system.

        Returns dict with 'value_function', 'density', 'optimal_control',
        'optimal_drift', 'convergence_history', 'n_iterations', 'converged',
        'residuals'.
        """
        m_current = self._init_density()
        convergence_history: list[float] = []

        u = np.zeros((self.n_times, self.n_x))
        alpha = np.zeros((self.n_times, self.n_x))
        b_star_field = np.zeros((self.n_times, self.n_x))
        converged = False
        k = 0

        for k in range(self.max_iterations):
            m_new, u, alpha, b_star_field = self._picard_step(m_current)

            error = self._compute_metric(m_new, m_current)
            convergence_history.append(error)

            if verbose:
                print(f"  iter {k + 1:3d}  error = {error:.3e}")

            # Check convergence BEFORE damping so we store undamped error
            if error < self.tol:
                m_current = m_new
                converged = True
                break

            m_current = self._apply_damping(m_new, m_current)

        residuals = self.compute_residuals(u, m_current, alpha)

        return {
            "value_function": u,
            "density": m_current,
            "optimal_control": alpha,
            "optimal_drift": b_star_field,
            "convergence_history": convergence_history,
            "n_iterations": k + 1,
            "converged": converged,
            "residuals": residuals,
        }

    # ------------------------------------------------------------------
    # Picard step
    # ------------------------------------------------------------------

    def _picard_step(self, m_current: np.ndarray) -> tuple:
        """One Picard iteration: HJB backward → FP forward."""
        x = self.x

        # --- Solve HJB backward given m_current ---
        hjb_result = self._hjb.solve(m_current)
        u = hjb_result["value_function"]
        alpha = hjb_result["optimal_control"]

        # --- Build drift and diffusion space-time fields for FP ---
        drift_field = np.zeros((self.n_times, self.n_x))
        diff_field = np.zeros((self.n_times, self.n_x))
        for n in range(self.n_times):
            drift_field[n] = self.model.drift(x, alpha[n], m_current[n])
            diff_field[n] = self.model.diffusion(x, alpha[n], m_current[n])

        # --- Solve FP forward ---
        fp_result = self._fp.solve(
            drift_field,
            diff_field,
            initial_distribution=self.model.initial_distribution(x),
        )
        m_new = fp_result["density"]

        return m_new, u, alpha, drift_field

    # ------------------------------------------------------------------
    # Damping and normalisation
    # ------------------------------------------------------------------

    def _apply_damping(self, m_new: np.ndarray, m_old: np.ndarray) -> np.ndarray:
        """Damped Picard update with per-step renormalisation."""
        lam = self.damping
        m_mixed = lam * m_new + (1.0 - lam) * m_old
        for n in range(m_mixed.shape[0]):
            mass = np.trapezoid(m_mixed[n], self.x)
            if mass > 1e-14:
                m_mixed[n] /= mass
        return m_mixed

    # ------------------------------------------------------------------
    # Convergence metric
    # ------------------------------------------------------------------

    def _compute_metric(self, m_new: np.ndarray, m_old: np.ndarray) -> float:
        diff = m_new - m_old
        if self.convergence_metric == "l2":
            # Max over time of the L²(Ω) norm
            return float(np.max([
                np.sqrt(np.trapezoid(diff[n] ** 2, self.x))
                for n in range(diff.shape[0])
            ]))
        elif self.convergence_metric == "linf":
            return float(np.max(np.abs(diff)))
        elif self.convergence_metric == "w2":
            return self._wasserstein2_max(m_new, m_old)
        else:
            raise ValueError(f"Unknown convergence_metric: {self.convergence_metric!r}")

    def _wasserstein2_max(self, m_new: np.ndarray, m_old: np.ndarray) -> float:
        """Max Wasserstein-2 distance over all time slices (1-D formula)."""
        x = self.x
        w2_max = 0.0
        for n in range(m_new.shape[0]):
            # Normalise slices
            mn = m_new[n]
            mo = m_old[n]
            mn = mn / max(np.trapezoid(mn, x), 1e-14)
            mo = mo / max(np.trapezoid(mo, x), 1e-14)
            # CDFs via cumulative trapezoid
            cdf_n = np.concatenate([[0.0], np.cumsum(0.5 * (mn[:-1] + mn[1:]) * np.diff(x))])
            cdf_o = np.concatenate([[0.0], np.cumsum(0.5 * (mo[:-1] + mo[1:]) * np.diff(x))])
            # W₂² ≈ ∫ |F_n(x) − F_o(x)|² dx  (McCann 1995, 1-D approximation)
            w2_sq = np.trapezoid((cdf_n - cdf_o) ** 2, x)
            w2_max = max(w2_max, np.sqrt(max(w2_sq, 0.0)))
        return float(w2_max)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_density(self) -> np.ndarray:
        """Tile the initial distribution across all time steps."""
        m0 = self.model.initial_distribution(self.x)
        return np.tile(m0, (self.n_times, 1))

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    def compute_residuals(
        self,
        u: np.ndarray,
        m: np.ndarray,
        alpha: np.ndarray,
    ) -> dict:
        """
        Evaluate the continuous PDE residuals at the discrete solution.

        HJB residual at (n, i): R_HJB = −(u[n+1]−u[n])/dt + H(x, Du[n], D²u[n], m[n])
        FP  residual at (n, i): R_FP  = (m[n+1]−m[n])/dt − σ²/2·D²m[n] + ∂(b*m)[n]/∂x

        Both should be O(dt + dx²).
        """
        x = self.x
        dt = self.dt
        D_cen = self._D_cen
        D2 = self._D2

        hjb_res_all = []
        fp_res_all = []

        for n in range(self.n_times - 1):
            a_n = alpha[n]
            m_n = m[n]
            m_np1 = m[n + 1]

            # Spatial derivatives at n
            Du_n = D_cen @ u[n]
            D2u_n = D2 @ u[n]

            # Hamiltonian at (n)
            H_n = self.model.hamiltonian(x, Du_n, D2u_n, m_n)

            # HJB residual: −(u[n+1]−u[n])/dt + H[n]
            r_hjb = -(u[n + 1] - u[n]) / dt + H_n
            hjb_res_all.append(r_hjb)

            # FP residual: (m[n+1]−m[n])/dt − σ²/2·D²m[n] + ∂(b*m)[n]/∂x
            sigma_n = self.model.diffusion(x, a_n, m_n)
            b_n = self.model.drift(x, a_n, m_n)
            D2m_n = D2 @ m_n

            # Conservative divergence ∂(bm)/∂x via upwind (reuse FP helper)
            div_bm = self._fp._conservative_advection(m_n, b_n)

            r_fp = (m_np1 - m_n) / dt - 0.5 * sigma_n ** 2 * D2m_n + div_bm
            fp_res_all.append(r_fp)

        hjb_arr = np.array(hjb_res_all)
        fp_arr = np.array(fp_res_all)

        def l2(arr):
            return float(np.sqrt(np.mean(arr ** 2)))

        def linf(arr):
            return float(np.max(np.abs(arr)))

        return {
            "hjb_residual_l2": l2(hjb_arr),
            "hjb_residual_linf": linf(hjb_arr),
            "fp_residual_l2": l2(fp_arr),
            "fp_residual_linf": linf(fp_arr),
        }

    # ------------------------------------------------------------------
    # Continuation
    # ------------------------------------------------------------------

    def continuation_solve(
        self,
        parameter_name: str,
        parameter_values: np.ndarray,
        initial_guess: dict = None,
    ) -> list:
        """
        Solve the MFG along a parameter path, using each solution as the
        initial guess for the next value (warm-starting).

        Args:
            parameter_name: attribute name on self.model to vary
            parameter_values: sequence of parameter values (easiest → hardest)
            initial_guess: optional dict from a previous solve()

        Returns list of solution dicts, one per parameter value.
        """
        results = []
        m_warm = None

        # Apply initial warm-start if provided
        if initial_guess is not None:
            m_warm = initial_guess.get("density")

        for val in parameter_values:
            # Update model parameter
            setattr(self.model, parameter_name, val)

            if m_warm is not None:
                # Use previous density as starting point
                old_init = self.model.initial_distribution
                _m_warm = m_warm  # capture

                def _patched_init(x):
                    return _m_warm[0]

                # Temporarily patch the initialisation inside _picard_step
                # by running one extra Picard step from m_warm
                m_current = _m_warm.copy()
            else:
                m_current = self._init_density()

            # Re-run Picard from warm start
            convergence_history = []
            u = np.zeros((self.n_times, self.n_x))
            alpha = np.zeros((self.n_times, self.n_x))
            b_field = np.zeros((self.n_times, self.n_x))
            converged = False

            for k in range(self.max_iterations):
                m_new, u, alpha, b_field = self._picard_step(m_current)
                error = self._compute_metric(m_new, m_current)
                convergence_history.append(error)
                if error < self.tol:
                    m_current = m_new
                    converged = True
                    break
                m_current = self._apply_damping(m_new, m_current)

            residuals = self.compute_residuals(u, m_current, alpha)
            result = {
                "value_function": u,
                "density": m_current,
                "optimal_control": alpha,
                "optimal_drift": b_field,
                "convergence_history": convergence_history,
                "n_iterations": k + 1,
                "converged": converged,
                "residuals": residuals,
                "parameter_value": val,
            }
            results.append(result)
            m_warm = m_current

        return results


# Backward-compatibility alias
MFGSystemSolver = MFGSolver


# ---------------------------------------------------------------------------
# NewtonMFGSolver
# ---------------------------------------------------------------------------

class NewtonMFGSolver:
    """
    Newton-Krylov refinement for the coupled MFG system.

    Reformulates F(u, m) = 0 (stacked HJB + FP residuals) and applies
    inexact Newton steps with GMRES for the linear sub-problems.

    Typically used AFTER a Picard warm-start:
        picard = MFGSolver(model, grid, tgrid, tol=1e-4)
        result = picard.solve()
        newton = NewtonMFGSolver(model, grid, tgrid, tol=1e-9)
        refined = newton.solve(result['value_function'], result['density'])

    Parameters
    ----------
    model          : MFGLOBModel
    grid           : Grid1D
    time_grid      : TimeGrid
    max_iterations : maximum Newton iterations
    tol            : Newton convergence tolerance on ‖F(z)‖₂
    """

    def __init__(
        self,
        model,
        grid: Grid1D,
        time_grid: TimeGrid,
        max_iterations: int = 50,
        tol: float = 1e-8,
    ) -> None:
        self.model = model
        self.grid = grid
        self.time_grid = time_grid
        self.max_iterations = int(max_iterations)
        self.tol = float(tol)

        self.x = grid.points
        self.n_x = grid.n
        self.n_times = time_grid.n_nodes
        self.dt = time_grid.dt

        self._hjb = HJBSolver(grid=grid, time_grid=time_grid, model=model)
        self._fp = FokkerPlanckSolver(grid=grid, time_grid=time_grid, model=model)
        self._picard = MFGSolver(model, grid, time_grid, tol=1e-4, max_iterations=50)

        ops = FiniteDifferenceOperators(grid)
        self._D_cen = ops.d_dx_central()
        self._D2 = ops.d2_dx2()

    def solve(
        self,
        initial_u: np.ndarray = None,
        initial_m: np.ndarray = None,
    ) -> dict:
        """
        Solve via Newton-Krylov on the Picard-map residual F(m) = T(m) − m.

        T(m) applies one HJB+FP sweep, so F = 0 exactly at the MFG fixed point.
        Starting from a Picard warm-start the initial residual is ≤ Picard tol,
        giving rapid Newton convergence.
        """
        if initial_u is None or initial_m is None:
            picard_result = self._picard.solve(verbose=False)
            initial_m = picard_result["density"]

        x = self.x
        shape_m = initial_m.shape

        def picard_map(m_flat: np.ndarray) -> np.ndarray:
            """One HJB+FP sweep; returns new density (flat)."""
            m = m_flat.reshape(shape_m)
            hjb_res = self._hjb.solve(m)
            alpha = hjb_res["optimal_control"]
            N = self.n_times
            drift_arr = np.zeros_like(m)
            diff_arr = np.zeros_like(m)
            for n in range(N):
                drift_arr[n] = self.model.drift(x, alpha[n], m[n])
                diff_arr[n] = self.model.diffusion(x, alpha[n], m[n])
            fp_res = self._fp.solve(drift_arr, diff_arr)
            m_new = fp_res["density"]
            return m_new.ravel()

        def residual(m_flat: np.ndarray) -> np.ndarray:
            return picard_map(m_flat) - m_flat

        m0 = initial_m.ravel()
        convergence_history = [np.linalg.norm(residual(m0))]
        converged = False
        m_opt = m0.copy()

        def _callback(x_cb, f):
            convergence_history.append(np.linalg.norm(f))

        try:
            m_opt = so.newton_krylov(
                residual,
                m0,
                method="lgmres",
                f_tol=self.tol,
                maxiter=self.max_iterations,
                callback=_callback,
                verbose=False,
            )
            converged = True
        except so.NoConvergence as exc:
            m_opt = np.asarray(exc.args[0])
        except Exception:
            pass

        m_final = m_opt.reshape(shape_m)
        m_final = np.maximum(m_final, 0.0)
        for n in range(m_final.shape[0]):
            mass = np.trapezoid(m_final[n], x)
            if mass > 1e-14:
                m_final[n] /= mass

        # Final HJB pass to get consistent u and alpha
        hjb_final = self._hjb.solve(m_final)
        u_final = hjb_final["value_function"]
        alpha_final = hjb_final["optimal_control"]

        return {
            "value_function": u_final,
            "density": m_final,
            "optimal_control": alpha_final,
            "convergence_history": convergence_history,
            "n_iterations": len(convergence_history),
            "converged": converged,
        }
