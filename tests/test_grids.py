"""Tests for mfglob.grids: Grid1D, TimeGrid, Grid2D."""

import numpy as np
import pytest

from mfglob.grids import Grid1D, TimeGrid, Grid2D


# ---------------------------------------------------------------------------
# Grid1D — uniform
# ---------------------------------------------------------------------------

class TestGrid1DUniform:
    def test_constant_spacing(self):
        g = Grid1D(0.0, 1.0, 11)
        diffs = np.diff(g.points)
        assert np.allclose(diffs, diffs[0], atol=1e-14), "Uniform grid must have constant spacing."

    def test_includes_left_endpoint(self):
        g = Grid1D(-3.0, 7.0, 51)
        assert g.points[0] == pytest.approx(-3.0)

    def test_includes_right_endpoint(self):
        g = Grid1D(-3.0, 7.0, 51)
        assert g.points[-1] == pytest.approx(7.0)

    def test_correct_n_points(self):
        for n in (2, 5, 21, 101):
            g = Grid1D(0.0, 1.0, n)
            assert g.n == n
            assert len(g.points) == n

    def test_dx_is_scalar(self):
        g = Grid1D(0.0, 2.0, 21)
        dx = g.dx
        assert np.isscalar(dx) or (isinstance(dx, np.ndarray) and dx.ndim == 0)

    def test_dx_value(self):
        g = Grid1D(0.0, 1.0, 11)
        assert g.dx == pytest.approx(0.1, rel=1e-12)

    def test_interior_mask_shape(self):
        g = Grid1D(0.0, 1.0, 21)
        mask = g.interior_mask()
        assert mask.shape == (21,)
        assert mask.dtype == bool

    def test_interior_mask_excludes_boundaries(self):
        g = Grid1D(0.0, 1.0, 21)
        mask = g.interior_mask()
        assert not mask[0]
        assert not mask[-1]

    def test_interior_mask_includes_interior(self):
        g = Grid1D(0.0, 1.0, 21)
        mask = g.interior_mask()
        assert np.all(mask[1:-1])
        assert mask.sum() == 19  # 21 - 2 boundary nodes

    def test_points_are_sorted(self):
        g = Grid1D(-5.0, 5.0, 101)
        assert np.all(np.diff(g.points) > 0)

    def test_invalid_xmin_ge_xmax_raises(self):
        with pytest.raises(ValueError):
            Grid1D(1.0, 0.0, 10)

    def test_invalid_n_points_raises(self):
        with pytest.raises(ValueError):
            Grid1D(0.0, 1.0, 1)

    def test_invalid_grid_type_raises(self):
        with pytest.raises(ValueError):
            Grid1D(0.0, 1.0, 10, grid_type="sinusoidal")


# ---------------------------------------------------------------------------
# Grid1D — non-uniform (tanh)
# ---------------------------------------------------------------------------

class TestGrid1DTanh:
    def test_finer_near_refinement_point(self):
        ref = 0.3
        g = Grid1D(0.0, 1.0, 51, grid_type="tanh", refinement_point=ref)
        diffs = np.diff(g.points)
        # Index closest to refinement_point
        idx = int(np.searchsorted(g.points, ref))
        idx = np.clip(idx, 1, len(diffs) - 2)
        dx_near = diffs[max(0, idx - 2):idx + 2].min()
        avg_dx = (1.0 - 0.0) / (51 - 1)
        assert dx_near < avg_dx, "Tanh grid must be finer near the refinement point."

    def test_endpoints_preserved(self):
        g = Grid1D(0.0, 1.0, 41, grid_type="tanh", refinement_point=0.5)
        assert g.points[0] == pytest.approx(0.0, abs=1e-14)
        assert g.points[-1] == pytest.approx(1.0, abs=1e-14)

    def test_correct_n_points(self):
        g = Grid1D(-5.0, 5.0, 61, grid_type="tanh", refinement_point=0.0)
        assert g.n == 61
        assert len(g.points) == 61

    def test_dx_is_array(self):
        g = Grid1D(0.0, 1.0, 21, grid_type="tanh", refinement_point=0.5)
        dx = g.dx
        assert isinstance(dx, np.ndarray), "Non-uniform grid dx should be an array."
        assert len(dx) == 20  # n-1 spacings

    def test_sorted(self):
        g = Grid1D(-2.0, 2.0, 51, grid_type="tanh", refinement_point=0.0)
        assert np.all(np.diff(g.points) > 0)

    def test_default_refinement_at_midpoint(self):
        g = Grid1D(0.0, 1.0, 51, grid_type="tanh")
        diffs = np.diff(g.points)
        # Smallest spacing should be near the midpoint (0.5)
        min_idx = np.argmin(diffs)
        mid_idx = len(diffs) // 2
        assert abs(min_idx - mid_idx) <= 5, "Minimum spacing should be near the midpoint."

    def test_tanh_denser_than_uniform_near_ref(self):
        """Tanh grid minimum spacing < uniform grid spacing."""
        n = 51
        ref = 0.3
        g_tanh = Grid1D(0.0, 1.0, n, grid_type="tanh", refinement_point=ref)
        g_unif = Grid1D(0.0, 1.0, n)
        assert np.min(np.diff(g_tanh.points)) < float(g_unif.dx)


# ---------------------------------------------------------------------------
# TimeGrid
# ---------------------------------------------------------------------------

class TestTimeGrid:
    def test_dt_value(self):
        tg = TimeGrid(T=1.0, n_steps=100)
        assert tg.dt == pytest.approx(0.01, rel=1e-12)

    def test_times_length(self):
        tg = TimeGrid(T=2.0, n_steps=50)
        assert len(tg.times) == 51  # n_steps + 1

    def test_times_start_at_zero(self):
        tg = TimeGrid(T=1.0, n_steps=10)
        assert tg.times[0] == pytest.approx(0.0)

    def test_times_end_at_T(self):
        tg = TimeGrid(T=3.5, n_steps=100)
        assert tg.times[-1] == pytest.approx(3.5, rel=1e-12)

    def test_times_uniformly_spaced(self):
        tg = TimeGrid(T=1.0, n_steps=20)
        diffs = np.diff(tg.times)
        assert np.allclose(diffs, tg.dt, atol=1e-14)

    def test_times_reversed_starts_at_T(self):
        tg = TimeGrid(T=2.0, n_steps=40)
        assert tg.times_reversed[0] == pytest.approx(2.0)

    def test_times_reversed_ends_at_zero(self):
        tg = TimeGrid(T=2.0, n_steps=40)
        assert tg.times_reversed[-1] == pytest.approx(0.0)

    def test_times_reversed_is_reversed(self):
        tg = TimeGrid(T=1.0, n_steps=10)
        assert np.allclose(tg.times_reversed, tg.times[::-1])

    def test_n_nodes(self):
        tg = TimeGrid(T=1.0, n_steps=99)
        assert tg.n_nodes == 100

    def test_invalid_T_raises(self):
        with pytest.raises(ValueError):
            TimeGrid(T=-1.0, n_steps=10)

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError):
            TimeGrid(T=1.0, n_steps=0)


# ---------------------------------------------------------------------------
# Grid2D
# ---------------------------------------------------------------------------

class TestGrid2D:
    def setup_method(self):
        self.gx = Grid1D(0.0, 1.0, 11)
        self.gy = Grid1D(-1.0, 1.0, 21)
        self.g2 = Grid2D(self.gx, self.gy)

    def test_n_total(self):
        assert self.g2.n_total == 11 * 21

    def test_shape(self):
        assert self.g2.shape == (11, 21)

    def test_meshgrid_shapes(self):
        X, Y = self.g2.meshgrid
        assert X.shape == (11, 21)
        assert Y.shape == (11, 21)

    def test_meshgrid_x_values(self):
        X, Y = self.g2.meshgrid
        # Each column of X should equal gx.points
        for j in range(21):
            assert np.allclose(X[:, j], self.gx.points)

    def test_meshgrid_y_values(self):
        X, Y = self.g2.meshgrid
        # Each row of Y should equal gy.points
        for i in range(11):
            assert np.allclose(Y[i, :], self.gy.points)

    def test_stores_grids(self):
        assert self.g2.grid_x is self.gx
        assert self.g2.grid_y is self.gy
