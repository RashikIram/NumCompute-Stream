import numpy as np
from typing import Tuple


def sort(arr: np.ndarray, axis: int = -1, descending: bool = False) -> np.ndarray:
    """
    Perform a stable sort along a specified axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    axis : int, default=-1
        Axis along which to sort.
    descending : bool, default=False
        If True, sort in descending order.

    Returns
    -------
    np.ndarray
        Sorted array.

    Notes
    -----
    Uses NumPy's stable sorting algorithm to preserve order of equal elements.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """
    sorted_arr = np.sort(arr, axis=axis, kind='stable')

    if descending:
        # Flip along the specified axis (correct for multi-dimensional arrays)
        sorted_arr = np.flip(sorted_arr, axis=axis)

    return sorted_arr


def multisort(arr: np.ndarray, keys, axis: int = -1) -> np.ndarray:
    """
    Perform multi-key sorting using lexicographic order.

    Parameters
    ----------
    arr : np.ndarray of shape (n_samples, n_features)
        Input 2D array.
    keys : sequence of int
        Column indices to sort by (last key is primary).
    axis : int, default=-1
        Axis along which sorting is performed (not used explicitly).

    Returns
    -------
    np.ndarray
        Sorted array.

    Raises
    ------
    ValueError
        If input array is not 2D or keys are invalid.

    Notes
    -----
    Uses np.lexsort, which sorts using the last key as primary.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError("Input array must be 2D.")

    if not keys:
        raise ValueError("keys cannot be empty")

    # Validate keys
    for key in keys:
        if not isinstance(key, int) or key < 0 or key >= arr.shape[1]:
            raise ValueError("Invalid key index")

    # Extract columns for lexsort (reverse order because last key is primary)
    sort_keys = tuple(arr[:, key] for key in reversed(keys))

    order = np.lexsort(sort_keys)

    return arr[order]


def topk(
    values: np.ndarray,
    k: int,
    largest: bool = True,
    return_indices: bool = False
) -> np.ndarray:
    """
    Select top-k elements using partial sorting.

    Parameters
    ----------
    values : np.ndarray of shape (n,)
        Input array.
    k : int
        Number of elements to select.
    largest : bool, default=True
        If True, select largest k elements; otherwise smallest.
    return_indices : bool, default=False
        If True, return indices instead of values.

    Returns
    -------
    np.ndarray of shape (k,)
        Top-k values or their indices.

    Raises
    ------
    ValueError
        If k is invalid.

    Notes
    -----
    Uses np.argpartition for O(n) average time complexity.
    Result is not fully sorted.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"topk expects 1D array, got shape {values.shape}")

    if k <= 0:
        raise ValueError("k must be a positive integer and greater than 0.")
    if k > values.size:
        raise ValueError("k cannot be greater than the number of elements.")

    if largest:
        indices = np.argpartition(values, -k)[-k:]
    else:
        indices = np.argpartition(values, k)[:k]

    return indices if return_indices else values[indices]


def quickselect(arr: np.ndarray, k: int) -> float:
    """
    Select the k-th smallest element using Quickselect algorithm.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
        Input array.
    k : int
        Index (0-based) of the desired element.

    Returns
    -------
    float
        The k-th smallest element.

    Raises
    ------
    ValueError
        If k is out of bounds.

    Notes
    -----
    Average-case O(n), worst-case O(n^2).
    Operates in-place on a copy of the array.

    Complexity
    ----------
    Time: O(n) average
    Space: O(1)
    """
    arr = np.asarray(arr).copy()

    if arr.ndim != 1:
        raise ValueError(f"quickselect expects 1D array, got shape {arr.shape}")

    if k < 0 or k >= len(arr):
        raise ValueError("k out of bounds")

    left, right = 0, len(arr) - 1

    while True:
        pivot_index = _partition(arr, left, right)

        if pivot_index == k:
            return float(arr[pivot_index])
        elif pivot_index < k:
            left = pivot_index + 1
        else:
            right = pivot_index - 1


def _partition(arr: np.ndarray, l: int, r: int) -> int:
    """
    Partition helper for quickselect.

    Parameters
    ----------
    arr : np.ndarray
    l : int
        Left index.
    r : int
        Right index.

    Returns
    -------
    int
        Final pivot index.
    """
    pivot = arr[r]
    i = l

    for j in range(l, r):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1

    arr[i], arr[r] = arr[r], arr[i]
    return i


def binary_search(arr: np.ndarray, target: float) -> Tuple[int, bool]:
    """
    Perform binary search using NumPy.

    Parameters
    ----------
    arr : np.ndarray of shape (n,)
        Sorted input array.
    target : float
        Value to search for.

    Returns
    -------
    tuple (int, bool)
        Insertion index and whether the target exists.

    Notes
    -----
    Uses np.searchsorted for O(log n) search.

    Complexity
    ----------
    Time: O(log n)
    Space: O(1)
    """
    arr = np.asarray(arr)

    idx = np.searchsorted(arr, target)

    found = (
        idx < len(arr)
        and arr[idx] == target
    )

    return int(idx), bool(found)