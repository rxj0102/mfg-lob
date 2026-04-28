"""Tests for mfglob.operators: FiniteDifferenceOperators."""

import numpy as np
import pytest
import scipy.sparse as sp

from mfglob.grids import Grid1D
from mfglob.operators import FiniteDifferenceOperators


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N = 21   # number of grid points

@pytest.fixture
def grid():
    return Grid1D(0.0, 1.0, N)

@pytest.fixture
def ops(grid):
    return FiniteDifferenceOperators(grid)

@pytest.fixture
def x(grid):
    return grid.points   # 0, 0.05, 0.10, …, 1.0


# ---------------------------------------------------------------------------
# Forward difference D⁺
# ---------------------------------------------------------------------------

class TestForwardDifference:
    def test_linear_u_gives_one(self, ops, x):
        """D⁺ of u(x) = x must be 1 at every grid point."""
        D = ops.d_dx_forward()
        du = D @ x
        assert np.allclose(du, 1.0, atol=1e-12), f"max error: {np.abs(du - 1).max()}"

    def test_constant_gives_zero(self, ops, x):
        """D⁺ of constant function must be 0 everywhere."""
        D = ops.d_dx_forward()
        u = np.ones_like(x)
        assert np.allclose(D @ u, 0.0, atol=1e-12)

    def test_is_sparse(self, ops):
        assert sp.issparse(ops.d_dx_forward())

    def test_shape(self, ops):
        assert ops.d_dx_forward().shape == (N, N)

    def test_interior_stencil(self, ops):
        """For interior rows i ∈ [1, n-2], non-zeros must be at cols i and i+1."""
        D = ops.d_dx_forward().toarray()
        for i in range(1, N - 2):
            row = D[i, :]
            nonzero_cols = np.where(np.abs(row) > 1e-14)[0].tolist()
            assert nonzero_cols == [i, i + 1], (
                f"Row {i}: expected non-zeros at [{i}, {i+1}], got {nonzero_cols}"
            )


# ---------------------------------------------------------------------------
# Backward difference D⁻
# ---------------------------------------------------------------------------

class TestBackwardDifference:
    def test_linear_u_gives_one(self, ops, x):
        """D⁻ of u(x) = x must be 1 at every grid point."""
        D = ops.d_dx_backward()
        du = D @ x
        assert np.allclose(du, 1.0, atol=1e-12)

    def test_constant_gives_zero(self, ops, x):
        D = ops.d_dx_backward()
        assert np.allclose(D @ np.ones_like(x), 0.0, atol=1e-12)

    def test_is_sparse(self, ops):
        assert sp.issparse(ops.d_dx_backward())

    def test_shape(self, ops):
        assert ops.d_dx_backward().shape == (N, N)

    def test_interior_stencil(self, ops):
        """For interior rows i ∈ [1, n-2], non-zeros must be at cols i-1 and i."""
        D = ops.d_dx_backward().toarray()
        for i in range(1, N - 1):
            row = D[i, :]
            nonzero_cols = np.where(np.abs(row) > 1e-14)[0].tolist()
            assert nonzero_cols == [i - 1, i], (
                f"Row {i}: expected non-zeros at [{i-1}, {i}], got {nonzero_cols}"
            )


# ---------------------------------------------------------------------------
# Central difference D⁰
# ---------------------------------------------------------------------------

class TestCentralDifference:
    def test_linear_u_gives_one_interior(self, ops, x):
        """D⁰ of u(x) = x must be 1 at interior points (second-order exact)."""
        D = ops.d_dx_central()
        du = D @ x
        assert np.allclose(du[1:-1], 1.0, atol=1e-12)

    def test_linear_u_gives_one_everywhere(self, ops, x):
        """With one-sided BCs, the full vector should also give 1."""
        D = ops.d_dx_central()
        du = D @ x
        assert np.allclose(du, 1.0, atol=1e-12)

    def test_quadratic_interior_exact(self, ops, x):
        """D⁰ of u(x) = x² = 2x at interior points (exactly by symmetry)."""
        D = ops.d_dx_central()
        u = x**2
        du = D @ u
        expected = 2.0 * x
        assert np.allclose(du[1:-1], expected[1:-1], atol=1e-12)

    def test_constant_gives_zero(self, ops, x):
        D = ops.d_dx_central()
        assert np.allclose(D @ np.ones_like(x), 0.0, atol=1e-12)

    def test_is_sparse(self, ops):
        assert sp.issparse(ops.d_dx_central())

    def test_shape(self, ops):
        assert ops.d_dx_central().shape == (N, N)

    def test_interior_stencil_anti_symmetric(self, ops):
        """Central diff rows at interior nodes must have entries at i-1 and i+1 only."""
        D = ops.d_dx_central().toarray()
        for i in range(1, N - 1):
            row = D[i, :]
            nonzero_cols = sorted(np.where(np.abs(row) > 1e-14)[0].tolist())
            assert nonzero_cols == [i - 1, i + 1], (
                f"Row {i}: expected non-zeros at [{i-1}, {i+1}], got {nonzero_cols}"
            )
            assert row[i - 1] == pytest.approx(-row[i + 1], rel=1e-12), (
                "Central diff must be anti-symmetric."
            )


# ---------------------------------------------------------------------------
# Second derivative D²
# ---------------------------------------------------------------------------

class TestSecondDerivative:
    def test_quadratic_interior_exact(self, ops, x):
        """D² of u(x) = x² must give 2 at all interior points (exactly)."""
        D2 = ops.d2_dx2()
        d2u = D2 @ (x**2)
        assert np.allclose(d2u[1:-1], 2.0, atol=1e-10), (
            f"max error: {np.abs(d2u[1:-1] - 2.0).max()}"
        )

    def test_quadratic_all_nodes_exact(self, ops, x):
        """D² of u(x) = x² must give 2 at ALL nodes (including boundaries)
        when the reflecting ghost-point stencil is used."""
        D2 = ops.d2_dx2()
        d2u = D2 @ (x**2)
        assert np.allclose(d2u, 2.0, atol=1e-10)

    def test_linear_gives_zero(self, ops, x):
        """D² of u(x) = x must be 0 at interior points."""
        D2 = ops.d2_dx2()
        d2u = D2 @ x
        assert np.allclose(d2u[1:-1], 0.0, atol=1e-12)

    def test_constant_gives_zero(self, ops, x):
        D2 = ops.d2_dx2()
        d2u = D2 @ np.ones_like(x)
        assert np.allclose(d2u[1:-1], 0.0, atol=1e-12)

    def test_is_sparse(self, ops):
        assert sp.issparse(ops.d2_dx2())

    def test_shape(self, ops):
        assert ops.d2_dx2().shape == (N, N)

    def test_interior_off_diagonal_symmetric(self, ops):
        """Off-diagonal entries of D² must be symmetric between interior rows."""
        D2 = ops.d2_dx2().toarray()
        for i in range(2, N - 1):
            assert D2[i, i - 1] == pytest.approx(D2[i - 1, i], rel=1e-12), (
                f"Interior off-diagonal symmetry failed at row {i}"
            )

    def test_interior_three_point_stencil(self, ops):
        """Interior rows must have non-zeros at cols i-1, i, i+1 only."""
        D2 = ops.d2_dx2().toarray()
        for i in range(1, N - 1):
            row = D2[i, :]
            nonzero_cols = sorted(np.where(np.abs(row) > 1e-14)[0].tolist())
            assert nonzero_cols == [i - 1, i, i + 1], (
                f"Row {i}: expected 3-point stencil, got non-zeros at {nonzero_cols}"
            )

    def test_diagonal_negative_interior(self, ops):
        """Interior diagonal entries of D² must be negative (it's a Laplacian)."""
        D2 = ops.d2_dx2()
        diag = D2.diagonal()
        assert np.all(diag[1:-1] < 0), "Interior diagonal of D² should be negative."


# ---------------------------------------------------------------------------
# Upwind advection
# ---------------------------------------------------------------------------

class TestUpwindAdvection:
    def test_positive_velocity_uses_backward_stencil(self, ops):
        """With v > 0 everywhere, the upwind stencil at each interior row
        must only involve col i-1 and col i (backward difference)."""
        v = np.ones(N)
        D_up = ops.upwind_advection(v).toarray()
        for i in range(1, N - 1):
            row = D_up[i, :]
            # col i+1 must be zero (backward stencil does not use i+1)
            assert D_up[i, i + 1] == pytest.approx(0.0, abs=1e-14), (
                f"Row {i}: positive velocity should NOT use col i+1 (forward), "
                f"but D_up[{i},{i+1}] = {D_up[i, i+1]}"
            )
            # col i-1 must be non-zero (backward stencil uses i-1)
            assert abs(D_up[i, i - 1]) > 1e-14, (
                f"Row {i}: positive velocity SHOULD use col i-1 (backward)."
            )

    def test_negative_velocity_uses_forward_stencil(self, ops):
        """With v < 0 everywhere, the upwind stencil at each interior row
        must only involve col i and col i+1 (forward difference)."""
        v = -np.ones(N)
        D_up = ops.upwind_advection(v).toarray()
        for i in range(1, N - 1):
            # col i-1 must be zero (forward stencil does not use i-1)
            assert D_up[i, i - 1] == pytest.approx(0.0, abs=1e-14), (
                f"Row {i}: negative velocity should NOT use col i-1 (backward), "
                f"but D_up[{i},{i-1}] = {D_up[i, i-1]}"
            )
            # col i+1 must be non-zero (forward stencil uses i+1)
            assert abs(D_up[i, i + 1]) > 1e-14, (
                f"Row {i}: negative velocity SHOULD use col i+1 (forward)."
            )

    def test_positive_velocity_applied_to_linear(self, ops, x):
        """v * D_upwind @ x = v * 1 for v > 0 (backward diff of u=x is 1)."""
        v = 2.0 * np.ones(N)
        D_up = ops.upwind_advection(v)
        result = D_up @ x
        assert np.allclose(result[1:], 2.0, atol=1e-12)

    def test_negative_velocity_applied_to_linear(self, ops, x):
        """v * D_upwind @ x = v * 1 for v < 0 (forward diff of u=x is 1)."""
        v = -3.0 * np.ones(N)
        D_up = ops.upwind_advection(v)
        result = D_up @ x
        assert np.allclose(result[:-1], -3.0, atol=1e-12)

    def test_zero_velocity_gives_zero(self, ops, x):
        v = np.zeros(N)
        D_up = ops.upwind_advection(v)
        assert np.allclose(D_up @ x, 0.0, atol=1e-14)

    def test_is_sparse(self, ops):
        v = np.ones(N)
        assert sp.issparse(ops.upwind_advection(v))

    def test_shape(self, ops):
        v = np.ones(N)
        assert ops.upwind_advection(v).shape == (N, N)

    def test_wrong_velocity_length_raises(self, ops):
        with pytest.raises(ValueError):
            ops.upwind_advection(np.ones(N + 5))


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    def test_dirichlet_first_row_identity(self, ops):
        """Dirichlet BC: row 0 must be the identity row [1, 0, 0, …]."""
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, bc_type="dirichlet")
        row0 = A.toarray()[0, :]
        assert row0[0] == pytest.approx(1.0)
        assert np.allclose(row0[1:], 0.0, atol=1e-14)

    def test_dirichlet_last_row_identity(self, ops):
        """Dirichlet BC: row n-1 must be [0, …, 0, 1]."""
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, bc_type="dirichlet")
        row_last = A.toarray()[-1, :]
        assert row_last[-1] == pytest.approx(1.0)
        assert np.allclose(row_last[:-1], 0.0, atol=1e-14)

    def test_absorbing_same_as_dirichlet(self, ops):
        """'absorbing' is an alias for 'dirichlet'."""
        D = ops.d2_dx2()
        A_dir = ops.apply_boundary_conditions(D, "dirichlet").toarray()
        A_abs = ops.apply_boundary_conditions(D, "absorbing").toarray()
        assert np.allclose(A_dir, A_abs)

    def test_neumann_first_row(self, ops, grid):
        """Neumann BC: row 0 must encode the forward finite difference du/dx."""
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, bc_type="neumann")
        dx = float(grid.dx)
        row0 = A.toarray()[0, :]
        assert row0[0] == pytest.approx(-1.0 / dx, rel=1e-12)
        assert row0[1] == pytest.approx(1.0 / dx, rel=1e-12)
        assert np.allclose(row0[2:], 0.0, atol=1e-14)

    def test_neumann_last_row(self, ops, grid):
        """Neumann BC: row n-1 must encode the backward finite difference du/dx."""
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, bc_type="neumann")
        dx = float(grid.dx)
        row_last = A.toarray()[-1, :]
        assert row_last[-1] == pytest.approx(1.0 / dx, rel=1e-12)
        assert row_last[-2] == pytest.approx(-1.0 / dx, rel=1e-12)
        assert np.allclose(row_last[:-2], 0.0, atol=1e-14)

    def test_reflecting_same_as_neumann(self, ops):
        """'reflecting' is an alias for 'neumann'."""
        D = ops.d2_dx2()
        A_neu = ops.apply_boundary_conditions(D, "neumann").toarray()
        A_ref = ops.apply_boundary_conditions(D, "reflecting").toarray()
        assert np.allclose(A_neu, A_ref)

    def test_invalid_bc_type_raises(self, ops):
        D = ops.d2_dx2()
        with pytest.raises(ValueError):
            ops.apply_boundary_conditions(D, bc_type="periodic")

    def test_returns_sparse(self, ops):
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, "dirichlet")
        assert sp.issparse(A)

    def test_interior_rows_unchanged_dirichlet(self, ops):
        """Dirichlet BC must leave interior rows of the operator unchanged."""
        D = ops.d2_dx2()
        A = ops.apply_boundary_conditions(D, "dirichlet")
        D_arr = D.toarray()
        A_arr = A.toarray()
        assert np.allclose(D_arr[1:-1, :], A_arr[1:-1, :], atol=1e-14)
