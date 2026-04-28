"""Numerical utilities shared across the mfglob package."""

from __future__ import annotations
import numpy as np


def thomas_algorithm(
    lower: np.ndarray,
    main: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system Ax = b via the Thomas algorithm in O(n).

    Parameters
    ----------
    lower : (n-1,) array — sub-diagonal a_i (i = 1, …, n-1)
    main  : (n,)   array — main diagonal  b_i (i = 0, …, n-1)
    upper : (n-1,) array — super-diagonal c_i (i = 0, …, n-2)
    rhs   : (n,)   array — right-hand side d_i

    Returns
    -------
    x : (n,) array — solution vector
    """
    n = len(main)
    c = upper.copy().astype(float)
    d = rhs.copy().astype(float)
    b = main.copy().astype(float)
    a = lower.copy().astype(float)

    # Forward sweep
    for i in range(1, n):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    # Back substitution
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def l2_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a - b||_2 / ||b||_2."""
    denom = np.linalg.norm(b)
    if denom < 1e-14:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def linf_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L∞ error max|a - b| / max|b|."""
    denom = np.max(np.abs(b))
    if denom < 1e-14:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b)) / denom)


def normalise_density(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return m normalised so that ∫ m dx = 1 (trapezoidal rule)."""
    mass = np.trapezoid(m, x)
    if mass < 1e-14:
        raise ValueError("Density has zero or negative mass — cannot normalise.")
    return m / mass
