import numpy as np
import pytest

from numcompute_stream.utils import (
    train_test_split,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    pairwise_distances,
    sigmoid,
    softmax,
    logsumexp,
    batch_iterator,
)

def test_train_test_split_shapes():
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert len(X_test) == 10
    assert len(X_train) == 40
    assert len(y_test) == 10
    assert len(y_train) == 40

def test_euclidean_distance():
    assert np.isclose(euclidean_distance([0, 0], [3, 4]), 5.0)


def test_manhattan_distance():
    assert np.isclose(manhattan_distance([1, 2, 3], [3, 2, 1]), 4.0)


def test_cosine_distance():
    assert np.isclose(cosine_distance([1, 0], [0, 1]), 1.0)


def test_pairwise_distances_shape():
    X = np.array([[0, 0], [3, 4]])
    D = pairwise_distances(X, metric="euclidean")
    assert D.shape == (2, 2)
    assert np.isclose(D[0, 1], 5.0)


def test_softmax_sums_to_one():
    x = np.array([[1, 2, 3], [2, 4, 6]])
    result = softmax(x, axis=1)
    assert np.allclose(np.sum(result, axis=1), 1.0)


def test_logsumexp_stability():
    x = np.array([1000, 1000])
    result = logsumexp(x)
    assert np.isfinite(result)


def test_batch_iterator():
    X = np.arange(10)
    batches = list(batch_iterator(X, batch_size=3))
    assert len(batches) == 4
    assert np.array_equal(batches[0], np.array([0, 1, 2]))


def test_distance_shape_mismatch():
    with pytest.raises(ValueError):
        euclidean_distance([1, 2], [1, 2, 3])


def test_distance_non_numeric():
    with pytest.raises(ValueError):
        manhattan_distance(["a", "b"], [1, 2])


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_distance([0, 0], [1, 2])


def test_pairwise_invalid_metric():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        pairwise_distances(X, metric="invalid")


def test_pairwise_feature_mismatch():
    X = np.array([[1, 2]])
    Y = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        pairwise_distances(X, Y)


def test_sigmoid_extreme_values():
    x = np.array([-1000, 0, 1000])
    result = sigmoid(x)
    assert np.all(np.isfinite(result))
    assert np.isclose(result[1], 0.5)


def test_softmax_extreme_values():
    x = np.array([1000, 1001, 1002])
    result = softmax(x)
    assert np.all(np.isfinite(result))
    assert np.isclose(np.sum(result), 1.0)


def test_logsumexp_empty_array():
    with pytest.raises(ValueError):
        logsumexp(np.array([]))


def test_batch_iterator_invalid_batch_size():
    with pytest.raises(ValueError):
        list(batch_iterator(np.arange(10), batch_size=0))


def test_batch_iterator_y_length_mismatch():
    with pytest.raises(ValueError):
        list(batch_iterator(np.arange(10), batch_size=3, y=np.arange(5)))


def test_pairwise_manhattan_distance():
    X = np.array([[0, 0], [1, 1]])
    D = pairwise_distances(X, metric="manhattan")
    assert np.isclose(D[0, 1], 2.0)


def test_pairwise_cosine_distance():
    X = np.array([[1, 0], [0, 1]])
    D = pairwise_distances(X, metric="cosine")
    assert np.isclose(D[0, 1], 1.0)


def test_batch_iterator_with_y():
    X = np.arange(10)
    y = np.arange(10) * 2
    batches = list(batch_iterator(X, batch_size=4, y=y))
    X_batch, y_batch = batches[0]
    assert np.array_equal(X_batch, np.array([0, 1, 2, 3]))
    assert np.array_equal(y_batch, np.array([0, 2, 4, 6]))


# if __name__ == "__main__":
#     test_batch_iterator_invalid_batch_size()
#     test_batch_iterator_y_length_mismatch()
#     test_distance_non_numeric()
#     test_distance_shape_mismatch()
#     test_cosine_zero_vector()
#     test_pairwise_invalid_metric()
#     test_pairwise_feature_mismatch()
#     test_pairwise_manhattan_distance()
#     test_pairwise_cosine_distance()
#     test_pairwise_distances_shape()
#     test_sigmoid_extreme_values()
#     test_softmax_extreme_values()
#     test_logsumexp_stability()
#     test_logsumexp_empty_array()
#     test_batch_iterator()
#     test_batch_iterator_with_y()
#     test_cosine_distance()
#     test_euclidean_distance()
#     test_manhattan_distance()
#     test_softmax_sums_to_one()



#     print("All optim tests have passed.")