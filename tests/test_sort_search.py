import numpy as np
import pytest

from numcompute.sort_search import (
    sort,
    multisort,
    topk,
    quickselect,
    binary_search,
)

# =========================
# SORT TESTS
# =========================

def test_stable_sort_basic():
    arr = np.array([3, 1, 2])
    sorted_arr = sort(arr)
    assert np.array_equal(sorted_arr, np.array([1, 2, 3]))


def test_stable_sort_duplicates():
    arr = np.array([2, 1, 2, 1])
    sorted_arr = sort(arr)
    assert np.array_equal(sorted_arr, np.array([1, 1, 2, 2]))


# =========================
# MULTISORT TESTS
# =========================

def test_multisort_basic():
    arr = np.array([
        [1, 3],
        [1, 2],
        [2, 1]
    ])

    sorted_arr = multisort(arr, keys=[0, 1])

    expected = np.array([
        [1, 2],
        [1, 3],
        [2, 1]
    ])

    assert np.array_equal(sorted_arr, expected)



# =========================
# TOP-K TESTS
# =========================

def test_topk_largest():
    arr = np.array([5, 1, 3, 7, 2])
    vals = topk(arr, 2)
    print(vals)

    assert set(vals) == {7, 5}


def test_topk_smallest():
    arr = np.array([5, 1, 3, 7, 2])
    vals= topk(arr, 2, largest=False)
    print(vals)

    assert set(vals) == {1, 2}


def test_topk_k_equals_n():
    arr = np.array([3, 1, 2])
    vals= topk(arr, 3)

    assert len(vals) == 3


def test_topk_invalid_k():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        topk(arr, 0)


# =========================
# QUICKSELECT TESTS
# =========================

def test_quickselect_basic():
    arr = np.array([7, 2, 1, 6, 8])
    val = quickselect(arr, 2)  # 3rd smallest

    assert val == np.sort(arr)[2]


def test_quickselect_first():
    arr = np.array([5, 3, 1])
    val = quickselect(arr, 0)

    assert val == 1


def test_quickselect_last():
    arr = np.array([5, 3, 1])
    val = quickselect(arr, 2)

    assert val == 5


def test_quickselect_invalid_k():
    arr = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        quickselect(arr, 5)


# =========================
# BINARY SEARCH TESTS
# =========================

def test_binary_search_found():
    arr = np.array([1, 2, 3, 4, 5])
    idx, found = binary_search(arr, 3)

    assert found is True
    assert arr[idx] == 3


def test_binary_search_not_found():
    arr = np.array([1, 2, 4, 5])
    idx, found = binary_search(arr, 3)

    assert found is False


def test_binary_search_insert_position():
    arr = np.array([1, 2, 4, 5])
    idx, _ = binary_search(arr, 3)

    assert idx == 2


