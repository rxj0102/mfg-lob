"""Finite-difference operators on 1-D grids.

All operators are returned as scipy sparse matrices (CSR format) so they
compose efficiently with standard sparse linear-algebra routines.

Theoretical background
----------------------
Let u : Ω → R with Ω = [x_min, x_max] discretised by Grid1D at n points
{x_0, …, x_{n-1}}.  Define:

  Δx_i = x_{i+1} - x_i          (forward spacing, i = 0, …, n-2)
  Δx̄_i = 0.5*(Δx_{i-1}+Δx_i)  (average half-spacing, interior nodes)

Forward difference  (D⁺):   (D⁺u)_i = (u_{i+1} - u_i) / Δx_i
Backward difference (D⁻):   (D⁻u)_i = (u_i - u_{i-1}) / Δx_{i-1}
Central difference  (D⁰):   (D⁰u)_i = (u_{i+1} - u_{i-1}) / (Δx_{i-1}+Δx_i)
Second derivative   (D²):   (D²u)_i = 2*(u_{i+1}/Δx_i - u_i/Δx̄_i + u_{i-1}/Δx_{i-1}) / (Δx_{i-1}+Δx_i)

Upwind advection (Godunov):
  (v·Du)_i = max(v_i,0)·(D⁻u)_i + min(v_i,0)·(D⁺u)_i

This is the monotone upwind scheme.  When v > 0 everywhere it reduces to the
backward-difference operator scaled by v; when v < 0 to the forward-difference
operator scaled by v.  Monotonicity guarantees convergence to the viscosity
solution (Barles-Souganidis theorem, 1991).

Boundary handling
-----------------
At boundary nodes the standard interior stencil is one-sided:
  i = 0    : forward stencil  (D⁻ uses forward, D⁺ uses forward)
  i = n-1  : backward stencil (D⁺ uses backward, D⁻ uses backward)
This keeps all operators square and applicable on the full grid vector.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp

from mfglob.grids import Grid1D


class FiniteDifferenceOperators:
    """Finite-difference operators on a 1-D spatial grid.

    All methods return scipy.sparse.csr_matrix of shape (n, n).

    Parameters
    ----------
    grid : Grid1D
    """

    def __init__(self, grid: Grid1D) -> None:
        self.grid = grid
        self._n = grid.n
        # Pre-compute spacings
        if grid.grid_type == "uniform":
            dx_scalar = float(grid.dx)
            self._dxf = np.full(grid.n - 1, dx_scalar)  # forward spacings
        else:
            self._dxf = np.asarray(grid.dx, dtype=float)  # shape (n-1,)

        self._dxb = self._dxf  # backward spacing at i = spacing at i-1

    # ------------------------------------------------------------------
    # Core difference operators
    # ------------------------------------------------------------------

    def d_dx_forward(self) -> sp.csr_matrix:
        """Forward difference operator D⁺: (D⁺u)_i = (u_{i+1} - u_i) / Δx_i.

        First-order accurate.  At the right boundary (i = n-1) a backward
        one-sided difference is used so the operator is square.
        """
        n = self._n
        dxf = self._dxf  # shape (n-1,)

        rows, cols, vals = [], [], []

        for i in range(n - 1):
            # Standard forward diff
            rows += [i, i]
            cols += [i, i + 1]
            vals += [-1.0 / dxf[i], 1.0 / dxf[i]]

        # Right boundary: one-sided backward diff  (u_{n-1} - u_{n-2}) / dx
        rows += [n - 1, n - 1]
        cols += [n - 2, n - 1]
        vals += [-1.0 / dxf[-1], 1.0 / dxf[-1]]

        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def d_dx_backward(self) -> sp.csr_matrix:
        """Backward difference operator D⁻: (D⁻u)_i = (u_i - u_{i-1}) / Δx_{i-1}.

        First-order accurate.  At the left boundary (i = 0) a one-sided forward
        difference is used so the operator is square.
        """
        n = self._n
        dxf = self._dxf  # shape (n-1,)

        rows, cols, vals = [], [], []

        # Left boundary: forward diff (u_1 - u_0) / dx
        rows += [0, 0]
        cols += [0, 1]
        vals += [-1.0 / dxf[0], 1.0 / dxf[0]]

        for i in range(1, n):
            # Standard backward diff
            rows += [i, i]
            cols += [i - 1, i]
            vals += [-1.0 / dxf[i - 1], 1.0 / dxf[i - 1]]

        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def d_dx_central(self) -> sp.csr_matrix:
        """Central difference operator D⁰: (D⁰u)_i = (u_{i+1}-u_{i-1})/(2Δx).

        Second-order accurate at interior nodes.  Boundary nodes use the same
        one-sided differences as D⁺ / D⁻ (first-order).
        """
        n = self._n
        dxf = self._dxf

        rows, cols, vals = [], [], []

        # Left boundary: forward one-sided
        rows += [0, 0]
        cols += [0, 1]
        vals += [-1.0 / dxf[0], 1.0 / dxf[0]]

        # Interior nodes: symmetric central difference
        for i in range(1, n - 1):
            width = dxf[i - 1] + dxf[i]   # Δx_{i-1} + Δx_i
            rows += [i, i]
            cols += [i - 1, i + 1]
            vals += [-1.0 / width, 1.0 / width]

        # Right boundary: backward one-sided
        rows += [n - 1, n - 1]
        cols += [n - 2, n - 1]
        vals += [-1.0 / dxf[-1], 1.0 / dxf[-1]]

        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def d2_dx2(self) -> sp.csr_matrix:
        """Second-derivative operator D²: standard 3-point stencil.

        Interior:  (D²u)_i = 2*(u_{i+1}/dxr - u_i*(1/dxl+1/dxr) + u_{i-1}/dxl)
                              / (dxl + dxr)
        Boundary:  one-sided 3-point stencils at i=0 (forward) and i=n-1 (backward),
                   exact for polynomials of degree ≤ 2 (e.g., u=x² gives 2 everywhere).
        """
        n = self._n
        dxf = self._dxf

        rows, cols, vals = [], [], []

        # Left boundary: one-sided 3-point forward stencil (exact for degree ≤ 2)
        dx0, dx1 = dxf[0], dxf[1]
        w0 = dx0 + dx1
        rows += [0, 0, 0]
        cols += [0, 1, 2]
        vals += [2.0 / (dx0 * w0), -2.0 / (dx0 * dx1), 2.0 / (dx1 * w0)]

        # Interior nodes
        for i in range(1, n - 1):
            dxl = dxf[i - 1]
            dxr = dxf[i]
            width = dxl + dxr
            rows += [i, i, i]
            cols += [i - 1, i, i + 1]
            vals += [
                2.0 / (dxl * width),
                -2.0 / (dxl * dxr),
                2.0 / (dxr * width),
            ]

        # Right boundary: one-sided 3-point backward stencil (exact for degree ≤ 2)
        dxNm2, dxNm1 = dxf[-2], dxf[-1]
        wN = dxNm2 + dxNm1
        rows += [n - 1, n - 1, n - 1]
        cols += [n - 3, n - 2, n - 1]
        vals += [2.0 / (dxNm2 * wN), -2.0 / (dxNm2 * dxNm1), 2.0 / (dxNm1 * wN)]

        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def upwind_advection(self, velocity: np.ndarray) -> sp.csr_matrix:
        """Upwind (Godunov) discretisation of the first-order advection v·∂u/∂x.

        At each node i the stencil depends on the sign of v_i:

            (v·Du)_i = max(v_i, 0)·(D⁻u)_i + min(v_i, 0)·(D⁺u)_i

        This is the monotone upwind scheme:
        - v_i > 0 → uses backward difference (upwind in +x direction)
        - v_i < 0 → uses forward  difference (upwind in -x direction)

        The matrix is computed as:
            A = diag(max(v, 0)) @ D⁻ + diag(min(v, 0)) @ D⁺

        Parameters
        ----------
        velocity : array-like, shape (n,)
            Advection velocity at each grid point.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n, n)
        """
        v = np.asarray(velocity, dtype=float)
        if v.shape != (self._n,):
            raise ValueError(f"velocity must have shape ({self._n},), got {v.shape}")

        v_pos = np.maximum(v, 0.0)   # max(v, 0)
        v_neg = np.minimum(v, 0.0)   # min(v, 0)

        D_back = self.d_dx_backward()
        D_fwd  = self.d_dx_forward()

        A = sp.diags(v_pos, 0, format="csr") @ D_back \
          + sp.diags(v_neg, 0, format="csr") @ D_fwd

        return A.tocsr()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def apply_boundary_conditions(
        self,
        operator: sp.csr_matrix,
        bc_type: str = "dirichlet",
        bc_left: float = 0.0,
        bc_right: float = 0.0,
    ) -> sp.csr_matrix:
        """Return a modified operator with boundary rows replaced to enforce BCs.

        The returned matrix A is such that (A @ u)_0 and (A @ u)_{n-1}
        encode the boundary constraint (the RHS values bc_left / bc_right
        are handled separately when solving the linear system).

        Boundary condition types
        ------------------------
        'dirichlet' / 'absorbing':
            Row 0   → e_0^T   (identity row: u_0 = bc_left)
            Row n-1 → e_{n-1}^T

        'neumann' / 'reflecting':
            Row 0   → [-1/dx, +1/dx, 0, …]    (∂u/∂x|_left  = bc_left)
            Row n-1 → [0, …, -1/dx, +1/dx]    (∂u/∂x|_right = bc_right)

        Parameters
        ----------
        operator : csr_matrix, shape (n, n)
            The PDE operator whose boundary rows are to be overwritten.
        bc_type : str
            'dirichlet', 'absorbing', 'neumann', or 'reflecting'.
        bc_left, bc_right : float
            Boundary values (used for the RHS, not stored in the matrix).

        Returns
        -------
        csr_matrix, shape (n, n)
            Modified operator with boundary rows set according to bc_type.
        """
        DIRICHLET = {"dirichlet", "absorbing"}
        NEUMANN   = {"neumann", "reflecting"}
        if bc_type not in DIRICHLET | NEUMANN:
            raise ValueError(
                f"bc_type must be one of {DIRICHLET | NEUMANN}, got {bc_type!r}"
            )

        n = self._n
        A = operator.tolil()   # lil_matrix allows efficient row editing

        if bc_type in DIRICHLET:
            # Row 0: [1, 0, 0, …]
            A[0, :] = 0.0
            A[0, 0] = 1.0
            # Row n-1: […, 0, 1]
            A[n - 1, :] = 0.0
            A[n - 1, n - 1] = 1.0

        else:  # Neumann / reflecting
            dx0 = float(self._dxf[0])
            dxN = float(self._dxf[-1])
            # Row 0: du/dx|_left = (u_1 - u_0) / dx_0  →  [-1/dx, +1/dx, 0, …]
            A[0, :] = 0.0
            A[0, 0] = -1.0 / dx0
            A[0, 1] = 1.0 / dx0
            # Row n-1: du/dx|_right = (u_{n-1} - u_{n-2}) / dx_{n-1}
            A[n - 1, :] = 0.0
            A[n - 1, n - 2] = -1.0 / dxN
            A[n - 1, n - 1] = 1.0 / dxN

        return A.tocsr()
