import numpy as np
import pytest

from numcompute_stream.benchmarking import (
    make_chunks,
    train_test_split_simple,
    benchmark_streaming_model,
    benchmark_streaming_dataset,
)
from numcompute_stream.tree import DecisionTreeClassifier


class DummyStreamingModel:
    def __init__(self):
        self.classes_ = None
        self.majority_class = 0
        self.fitted = False

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")

        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[0], self.majority_class)


def test_make_chunks_basic():
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 0, 1, 0])

    chunks = list(make_chunks(X, y, chunk_size=2))

    assert len(chunks) == 3
    assert chunks[0][0].shape == (2, 2)
    assert chunks[-1][0].shape == (1, 2)


def test_make_chunks_invalid_chunk_size():
    X = np.arange(6).reshape(3, 2)
    y = np.array([0, 1, 0])

    with pytest.raises(ValueError):
        list(make_chunks(X, y, chunk_size=0))


def test_train_test_split_simple_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split_simple(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_train_test_split_invalid_test_size():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    with pytest.raises(ValueError):
        train_test_split_simple(X, y, test_size=1.5)


def test_benchmark_streaming_model_returns_expected_keys():
    X = np.array([
        [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]
    ])
    y = np.array([0, 0, 1, 1, 1, 0])

    model = DummyStreamingModel()

    result = benchmark_streaming_model(
        model,
        X[:4],
        y[:4],
        X[4:],
        y[4:],
        classes=np.array([0, 1]),
        chunk_size=2,
    )

    expected_keys = {
        "fit_time",
        "predict_time",
        "total_time",
        "peak_memory_mb",
        "mean_chunk_accuracy",
        "cumulative_accuracy",
        "test_accuracy",
        "n_chunks",
        "n_train_samples",
        "n_test_samples",
    }

    assert set(result.keys()) == expected_keys
    assert result["n_chunks"] == 2
    assert result["n_train_samples"] == 4
    assert result["n_test_samples"] == 2
    assert 0 <= result["test_accuracy"] <= 1


def test_benchmark_streaming_model_with_real_tree():
    X = np.array([
        [0.0], [0.1], [0.2], [1.0], [1.1], [1.2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = DecisionTreeClassifier(max_depth=2, random_state=42)

    result = benchmark_streaming_model(
        model,
        X[:4],
        y[:4],
        X[4:],
        y[4:],
        classes=np.array([0, 1]),
        chunk_size=2,
    )

    assert result["n_chunks"] == 2
    assert result["fit_time"] >= 0
    assert result["peak_memory_mb"] >= 0
    assert 0 <= result["cumulative_accuracy"] <= 1


def test_benchmark_streaming_dataset_with_temporary_csv(tmp_path):
    file = tmp_path / "toy.csv"
    file.write_text(
        "f1,f2,target\n"
        "0.0,0.0,A\n"
        "0.1,0.0,A\n"
        "1.0,1.0,B\n"
        "1.1,1.0,B\n"
        "0.2,0.1,A\n"
        "1.2,1.1,B\n"
    )

    model_factories = {
        "Tree": lambda seed: DecisionTreeClassifier(max_depth=2, random_state=seed)
    }

    results = benchmark_streaming_dataset(
        "Toy",
        file,
        chunk_size=2,
        random_state=42,
        model_factories=model_factories,
    )

    assert len(results) == 1
    assert results[0]["dataset"] == "Toy"
    assert results[0]["model"] == "Tree"
    assert 0 <= results[0]["test_accuracy"] <= 1
