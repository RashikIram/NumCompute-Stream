import numpy as np
from typing import Callable


def grad(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    h: float = 1e-5,
    method: str = 'central'
) -> np.ndarray:
    """
    Estimate the gradient of a scalar function using finite differences.

    Parameters
    ----------
    function : Callable[[np.ndarray], float]
        Function that maps a 1D input array of shape (n,) to a scalar.
    point : np.ndarray of shape (n,)
        Point at which the gradient is evaluated.
    h : float, default=1e-5
        Step size for finite differences.
    method : {'central', 'forward'}, default='central'
        Finite difference method:
        - 'forward': forward difference approximation
        - 'central': central difference approximation (more accurate)

    Returns
    -------
    np.ndarray of shape (n,)
        Estimated gradient vector.

    Raises
    ------
    ValueError
        If method is invalid, input is not 1D, or function does not return a scalar.

    Notes
    -----
    Central difference provides O(h^2) accuracy, while forward difference provides O(h).
    Choice of step size `h` affects numerical accuracy:
    - Too large → truncation error
    - Too small → floating-point rounding error
    Typical values for `h` are in the range 1e-5 to 1e-8.

    Complexity
    ----------
    Time: O(n * T_f)
        where T_f is the cost of evaluating `function`.
    Space: O(n)
    """
    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Supported methods are: 'central', 'forward'.")

    if h <= 0:
        raise ValueError("Step size h must be positive")

    point = np.asarray(point, dtype=float)

    if point.ndim != 1:
        raise ValueError(f"grad expects a 1D array, got shape {point.shape}")

    base_value = function(point)

    if np.ndim(base_value) != 0:
        raise ValueError("grad expects function(point) to return a scalar")

    gradient = np.zeros_like(point, dtype=float)

    for index in range(len(point)):
        forward_point = point.copy()
        forward_point[index] += h

        if method == 'forward':
            gradient[index] = (function(forward_point) - base_value) / h

        elif method == 'central':
            backward_point = point.copy()
            backward_point[index] -= h
            gradient[index] = (
                function(forward_point) - function(backward_point)
            ) / (2 * h)

    return gradient


def jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    h: float = 1e-5,
    method: str = 'central'
) -> np.ndarray:
    """
    Estimate the Jacobian matrix of a vector-valued function.

    Parameters
    ----------
    function : Callable[[np.ndarray], np.ndarray]
        Function that maps a 1D input array of shape (n,) to a 1D output array of shape (m,).
    point : np.ndarray of shape (n,)
        Point at which the Jacobian is evaluated.
    h : float, default=1e-5
        Step size for finite differences.
    method : {'central', 'forward'}, default='central'
        Finite difference method.

    Returns
    -------
    np.ndarray of shape (m, n)
        Jacobian matrix where element (i, j) = dF_i / dx_j.

    Raises
    ------
    ValueError
        If method is invalid, input is not 1D, or function output is not 1D.

    Notes
    -----
    Central difference is more accurate but requires twice as many function evaluations.
    Choice of step size `h` affects numerical accuracy:
    - Too large → truncation error
    - Too small → floating-point rounding error
    Typical values for `h` are in the range 1e-5 to 1e-8.

    Complexity
    ----------
    Time: O(n * T_f)
    Space: O(m * n)
    """
    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Supported methods are: 'central', 'forward'.")

    if h <= 0:
        raise ValueError("Step size h must be positive")

    point = np.asarray(point, dtype=float)

    if point.ndim != 1:
        raise ValueError(f"jacobian expects a 1D array, got shape {point.shape}")

    base_output = np.asarray(function(point), dtype=float)

    if base_output.ndim != 1:
        raise ValueError("jacobian expects function(point) to return a 1D array")

    output_size = len(base_output)
    input_size = len(point)

    jacobian_matrix = np.zeros((output_size, input_size), dtype=float)

    for index in range(input_size):
        forward_point = point.copy()
        forward_point[index] += h

        if method == 'forward':
            forward_output = np.asarray(function(forward_point), dtype=float)
            jacobian_matrix[:, index] = (forward_output - base_output) / h

        elif method == 'central':
            backward_point = point.copy()
            backward_point[index] -= h

            forward_output = np.asarray(function(forward_point), dtype=float)
            backward_output = np.asarray(function(backward_point), dtype=float)

            jacobian_matrix[:, index] = (
                forward_output - backward_output
            ) / (2 * h)

    return jacobian_matrix