"""Abstract base class for MFG-LOB models.

Every concrete model must subclass MFGLOBModel and implement the six
abstract methods below.  The coupled PDE system that each model instance
defines is:

BACKWARD Hamilton-Jacobi-Bellman equation (solved T → 0):

    -∂u/∂t + H(x, Du, D²u, m) = 0       in (0,T) × Ω
    u(T, x) = g(x, m(T))                   terminal condition

where the Hamiltonian is defined by the Legendre-Fenchel transform:

    H(x, p, M, m) = sup_{α ∈ A} { -f(x,α,m) - b(x,α,m)·p
                                   - (1/2)σ²(x,α,m)·M }

and the optimal control satisfies the first-order condition:

    ∂/∂α [-f - b·p - (1/2)σ²·M] = 0   ⟹   α*(x, p, m)

FORWARD Kolmogorov / Fokker-Planck equation (solved 0 → T):

    ∂m/∂t - (1/2) ∂²(σ²·m)/∂x² + ∂(b*·m)/∂x = 0   in (0,T) × Ω
    m(0, x) = m₀(x)                                    initial distribution

where  b*(x, m) = b(x, α*(x, Du, m), m)  is the drift under optimal control.

NASH EQUILIBRIUM: a pair (u, m) satisfying both PDEs simultaneously.
The HJB depends on m (through costs/drift), and the FP depends on u
(through the optimal control).  The Lasry-Lions fixed-point iteration
alternates between the two equations until self-consistency is achieved.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class MFGLOBModel(ABC):
    """Abstract base class for mean-field game models of limit order books.

    Subclass this and implement all abstract methods to define a complete
    MFG-LOB model ready for use with HJBSolver, FPSolver, and MFGSolver.

    The state variable x can represent any scalar market quantity:
    inventory, quote depth, deviation from fair value, etc.
    """

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def hamiltonian(
        self,
        x: np.ndarray,
        p: np.ndarray,
        M: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Evaluate H(x, p, M, m) at every grid point.

        Parameters
        ----------
        x : (n,) array — spatial grid points
        p : (n,) array — Du, gradient of value function
        M : (n,) array — D²u, second derivative of value function
        m : (n,) array — population density at current time

        Returns
        -------
        (n,) array of Hamiltonian values.
        """

    @abstractmethod
    def optimal_control(
        self,
        x: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Compute α*(x, p, m) from the Pontryagin maximum principle.

        Parameters
        ----------
        x : (n,) array — spatial grid points
        p : (n,) array — costate variable Du
        m : (n,) array — population density

        Returns
        -------
        (n,) array of optimal control values.
        """

    @abstractmethod
    def drift(
        self,
        x: np.ndarray,
        alpha: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """State-process drift b(x, α, m).

        Parameters
        ----------
        x     : (n,) array
        alpha : (n,) array — control
        m     : (n,) array — population density

        Returns
        -------
        (n,) array.
        """

    @abstractmethod
    def diffusion(
        self,
        x: np.ndarray,
        alpha: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """State-process diffusion coefficient σ(x, α, m).

        Parameters
        ----------
        x     : (n,) array
        alpha : (n,) array — control
        m     : (n,) array — population density

        Returns
        -------
        (n,) array of σ values (NOT σ²).
        """

    @abstractmethod
    def running_cost(
        self,
        x: np.ndarray,
        alpha: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Running cost f(x, α, m).

        Parameters
        ----------
        x     : (n,) array
        alpha : (n,) array — control
        m     : (n,) array — population density

        Returns
        -------
        (n,) array.
        """

    @abstractmethod
    def terminal_cost(
        self,
        x: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Terminal cost g(x, m(T)).

        Parameters
        ----------
        x : (n,) array
        m : (n,) array — terminal population density

        Returns
        -------
        (n,) array.
        """

    @abstractmethod
    def initial_distribution(self, x: np.ndarray) -> np.ndarray:
        """Initial population distribution m₀(x).

        Must be non-negative and integrate to 1 over the domain.

        Parameters
        ----------
        x : (n,) array — spatial grid points

        Returns
        -------
        (n,) array, normalised (∫ m₀ dx = 1).
        """

    @property
    @abstractmethod
    def state_dimension(self) -> int:
        """Dimension d of the state space (1 for scalar, 2 for bivariate, …)."""

    @property
    @abstractmethod
    def param_dict(self) -> dict:
        """Model parameters as a plain dictionary (for logging / serialisation)."""

    # ------------------------------------------------------------------
    # Convenience helpers (non-abstract, can be overridden)
    # ------------------------------------------------------------------

    def equilibrium_drift(
        self,
        x: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Drift under the optimal control: b*(x, p, m) = b(x, α*(x,p,m), m).

        This is the drift that enters the Fokker-Planck equation.
        """
        alpha_star = self.optimal_control(x, p, m)
        return self.drift(x, alpha_star, m)

    def equilibrium_diffusion(
        self,
        x: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
    ) -> np.ndarray:
        """Diffusion under the optimal control: σ*(x, p, m)."""
        alpha_star = self.optimal_control(x, p, m)
        return self.diffusion(x, alpha_star, m)

    def verify_initial_distribution(self, x: np.ndarray) -> bool:
        """Check that initial_distribution integrates (approximately) to 1."""
        m0 = self.initial_distribution(x)
        mass = np.trapezoid(m0, x)
        return bool(np.isclose(mass, 1.0, rtol=1e-3))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.param_dict})"
