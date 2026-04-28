"""Spatial and temporal grid construction for MFG-LOB PDE solvers.

Three grid classes cover all discretisation needs:

  Grid1D   — one-dimensional spatial grid (uniform or tanh-stretched)
  TimeGrid — temporal grid with both forward and backward views
  Grid2D   — tensor-product grid for two-dimensional state spaces
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Grid1D
# ---------------------------------------------------------------------------

class Grid1D:
    """One-dimensional spatial grid for finite-difference discretisation.

    In LOB models the spatial variable can represent:
    - Inventory  q ∈ [q_min, q_max]   (agent's signed position)
    - Quote depth δ ∈ [0, δ_max]      (half-spread from mid)
    - LOB depth  x ∈ [0, x_max]      (distance from best price)

    Parameters
    ----------
    x_min, x_max : float
        Domain boundaries (inclusive).
    n_points : int
        Number of grid points.
    grid_type : {'uniform', 'tanh'}
        'uniform'  — constant spacing Δx = (x_max - x_min) / (n_points - 1).
        'tanh'     — hyperbolic-tangent stretching: finer near `refinement_point`,
                     coarser elsewhere.
    refinement_point : float, optional
        The coordinate at which points are clustered for 'tanh' grids.
        Defaults to the midpoint (x_min + x_max) / 2.
    concentration : float
        Stretching strength for 'tanh' grids (higher → stronger clustering).
    """

    _VALID_TYPES = {"uniform", "tanh"}

    def __init__(
        self,
        x_min: float,
        x_max: float,
        n_points: int,
        grid_type: str = "uniform",
        refinement_point: float | None = None,
        concentration: float = 3.0,
    ) -> None:
        if grid_type not in self._VALID_TYPES:
            raise ValueError(f"grid_type must be one of {self._VALID_TYPES}, got {grid_type!r}")
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        if x_min >= x_max:
            raise ValueError("x_min must be strictly less than x_max")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self._n_points = int(n_points)
        self.grid_type = grid_type
        self.concentration = float(concentration)

        if grid_type == "uniform":
            self._points = np.linspace(x_min, x_max, n_points)
            self._dx: float | np.ndarray = float(self._points[1] - self._points[0])
        else:
            if refinement_point is None:
                refinement_point = 0.5 * (x_min + x_max)
            self.refinement_point = float(refinement_point)
            self._points = self._build_tanh_grid(x_min, x_max, n_points,
                                                  self.refinement_point, concentration)
            self._dx = np.diff(self._points)

        self.refinement_point = (
            float(refinement_point) if refinement_point is not None
            else 0.5 * (x_min + x_max)
        )

    # ------------------------------------------------------------------
    # Grid construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tanh_grid(
        x_min: float,
        x_max: float,
        n: int,
        x_ref: float,
        beta: float,
    ) -> np.ndarray:
        """Build a tanh-stretched grid clustering points near x_ref.

        Strategy: split at x_ref, apply independent tanh stretching on each
        half so that the densest spacing is at x_ref on both sides.
        """
        # Proportion of points allocated to each half
        left_frac = (x_ref - x_min) / (x_max - x_min)
        n_left = max(2, round(left_frac * (n - 1)) + 1)
        n_right = n - n_left + 1  # the shared point at x_ref counts once

        # Guard: both halves need at least 2 points
        n_left = max(n_left, 2)
        n_right = max(n_right, 2)
        # Adjust for exact count
        if n_left + n_right - 1 != n:
            n_right = n - n_left + 1

        # Left half [x_min → x_ref]: cluster near s=1 (→ x_ref)
        if n_left >= 2:
            s = np.linspace(0.0, 1.0, n_left)
            t = np.tanh(beta * s) / np.tanh(beta)          # dense near s=1
            x_left = x_min + t * (x_ref - x_min)
            x_left[-1] = x_ref
        else:
            x_left = np.array([x_ref])

        # Right half [x_ref → x_max]: cluster near s=0 (→ x_ref)
        if n_right >= 2:
            s = np.linspace(0.0, 1.0, n_right)
            t = 1.0 - np.tanh(beta * (1.0 - s)) / np.tanh(beta)  # dense near s=0
            x_right = x_ref + t * (x_max - x_ref)
            x_right[0] = x_ref
        else:
            x_right = np.array([x_ref])

        # Concatenate, removing the duplicate x_ref
        x = np.concatenate([x_left[:-1], x_right])

        # Fallback: interpolate to exactly n points if rounding drifted
        if len(x) != n:
            old_t = np.linspace(0.0, 1.0, len(x))
            new_t = np.linspace(0.0, 1.0, n)
            x = np.interp(new_t, old_t, x)

        x[0] = x_min
        x[-1] = x_max
        return x

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def dx(self) -> float | np.ndarray:
        """Grid spacing: scalar for uniform, 1-D array of length n-1 for non-uniform."""
        return self._dx

    @property
    def points(self) -> np.ndarray:
        """Grid point coordinates, shape (n,)."""
        return self._points

    @property
    def n(self) -> int:
        """Number of grid points."""
        return self._n_points

    def interior_mask(self) -> np.ndarray:
        """Boolean mask selecting interior points (excluding the two boundary nodes)."""
        mask = np.zeros(self._n_points, dtype=bool)
        mask[1:-1] = True
        return mask

    def __len__(self) -> int:
        return self._n_points

    def __repr__(self) -> str:
        return (
            f"Grid1D(x_min={self.x_min}, x_max={self.x_max}, "
            f"n={self._n_points}, type={self.grid_type!r})"
        )


# ---------------------------------------------------------------------------
# TimeGrid
# ---------------------------------------------------------------------------

class TimeGrid:
    """Temporal grid for time-stepping in coupled HJB-FP systems.

    In MFG problems:
    - The HJB equation is solved **backward** in time: T → 0
    - The Fokker-Planck equation is solved **forward** in time: 0 → T

    Both views are provided without copying the underlying array.

    Parameters
    ----------
    T : float
        Terminal time (horizon).
    n_steps : int
        Number of time steps. The grid has n_steps + 1 nodes.
    """

    def __init__(self, T: float, n_steps: int) -> None:
        if T <= 0:
            raise ValueError("T must be positive")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        self.T = float(T)
        self.n_steps = int(n_steps)
        self._times = np.linspace(0.0, T, n_steps + 1)

    @property
    def dt(self) -> float:
        """Uniform time step Δt = T / n_steps."""
        return self.T / self.n_steps

    @property
    def times(self) -> np.ndarray:
        """Time nodes [0, Δt, 2Δt, …, T], shape (n_steps + 1,)."""
        return self._times

    @property
    def times_reversed(self) -> np.ndarray:
        """Time nodes [T, T-Δt, …, 0] for the backward HJB sweep."""
        return self._times[::-1]

    @property
    def n_nodes(self) -> int:
        """Total number of time nodes (= n_steps + 1)."""
        return self.n_steps + 1

    def __len__(self) -> int:
        return self.n_nodes

    def __repr__(self) -> str:
        return f"TimeGrid(T={self.T}, n_steps={self.n_steps}, dt={self.dt:.4g})"


# ---------------------------------------------------------------------------
# Grid2D
# ---------------------------------------------------------------------------

class Grid2D:
    """Two-dimensional spatial grid as a tensor product of two Grid1D objects.

    Used for MFG models with two state variables, e.g.:
    - (inventory q, private signal s) in informed-trading models
    - (price, volatility) in stochastic-volatility LOB models

    Parameters
    ----------
    grid_x : Grid1D
        Grid along the first axis.
    grid_y : Grid1D
        Grid along the second axis.
    """

    def __init__(self, grid_x: Grid1D, grid_y: Grid1D) -> None:
        self.grid_x = grid_x
        self.grid_y = grid_y

    @property
    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) meshgrid arrays, each of shape (n_x, n_y).

        Uses 'ij' indexing so that X[i, j] = grid_x.points[i]
        and Y[i, j] = grid_y.points[j].
        """
        X, Y = np.meshgrid(self.grid_x.points, self.grid_y.points, indexing="ij")
        return X, Y

    @property
    def n_total(self) -> int:
        """Total number of grid points (n_x × n_y)."""
        return self.grid_x.n * self.grid_y.n

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the 2-D grid array (n_x, n_y)."""
        return (self.grid_x.n, self.grid_y.n)

    def __repr__(self) -> str:
        return (
            f"Grid2D(x={self.grid_x!r}, y={self.grid_y!r}, "
            f"n_total={self.n_total})"
        )
