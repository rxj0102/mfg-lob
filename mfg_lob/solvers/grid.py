"""Computational grid for the MFG-LOB PDE system."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Grid:
    """Uniform finite-difference grid over (inventory q, time t) space.

    Inventory axis represents each agent's signed inventory position q ∈ [q_min, q_max].
    Time runs forward from t=0 to t=T (terminal condition at t=T).

    Parameters
    ----------
    q :  1-D array of inventory nodes, shape (nq,)
    t :  1-D array of time nodes,      shape (nt,)
    """

    q: np.ndarray
    t: np.ndarray
    dq: float = field(init=False)
    dt: float = field(init=False)
    nq: int = field(init=False)
    nt: int = field(init=False)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        if self.q.ndim != 1 or self.t.ndim != 1:
            raise ValueError("q and t must be 1-D arrays.")
        if len(self.q) < 3 or len(self.t) < 2:
            raise ValueError("Grid requires at least 3 inventory nodes and 2 time nodes.")
        self.nq = len(self.q)
        self.nt = len(self.t)
        self.dq = float(self.q[1] - self.q[0])
        self.dt = float(self.t[1] - self.t[0])

    @classmethod
    def uniform(
        cls,
        q_min: float = -10.0,
        q_max: float = 10.0,
        nq: int = 101,
        t_min: float = 0.0,
        t_max: float = 1.0,
        nt: int = 201,
    ) -> "Grid":
        """Construct a uniform grid."""
        return cls(
            q=np.linspace(q_min, q_max, nq),
            t=np.linspace(t_min, t_max, nt),
        )

    @property
    def q_min(self) -> float:
        return float(self.q[0])

    @property
    def q_max(self) -> float:
        return float(self.q[-1])

    @property
    def t_min(self) -> float:
        return float(self.t[0])

    @property
    def t_max(self) -> float:
        return float(self.t[-1])

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (Q, T) meshgrids with shape (nq, nt)."""
        Q, T = np.meshgrid(self.q, self.t, indexing="ij")
        return Q, T

    def courant_number(self, max_drift: float) -> float:
        """CFL number |a| * dt/dq for stability diagnostics."""
        return abs(max_drift) * self.dt / self.dq
