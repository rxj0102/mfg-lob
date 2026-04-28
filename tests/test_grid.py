"""Tests for the computational grid."""
import numpy as np
import pytest

from mfg_lob.solvers.grid import Grid


def test_uniform_grid_shape():
    g = Grid.uniform(q_min=-5, q_max=5, nq=51, t_min=0, t_max=1, nt=101)
    assert g.nq == 51
    assert g.nt == 101
    assert g.q[0] == pytest.approx(-5.0)
    assert g.q[-1] == pytest.approx(5.0)


def test_uniform_grid_spacing():
    g = Grid.uniform(nq=11, nt=21)
    assert g.dq == pytest.approx(g.q[1] - g.q[0])
    assert g.dt == pytest.approx(g.t[1] - g.t[0])


def test_meshgrid_shape():
    g = Grid.uniform(nq=11, nt=21)
    Q, T = g.meshgrid()
    assert Q.shape == (11, 21)
    assert T.shape == (11, 21)


def test_courant_number():
    g = Grid.uniform(nq=101, nt=201)
    cfl = g.courant_number(1.0)
    assert cfl >= 0.0


def test_invalid_grid_raises():
    with pytest.raises(ValueError):
        Grid(q=np.array([0, 1]), t=np.array([0, 0.5, 1.0]))
