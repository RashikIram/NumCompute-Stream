import numpy as np
import pytest

from numcompute.pipeline import Pipeline


class DummyTransformer:
    def __init__(self):
        self.fitted = False
        self.mean = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("DummyTransformer has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return X - self.mean


class DummyStreamingTransformer:
    def __init__(self):
        self.fitted = False
        self.n_updates = 0
        self.mean = None

    def partial_fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        self.fitted = True
        self.n_updates += 1
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("DummyStreamingTransformer has not been fitted yet.")
        X = np.asarray(X, dtype=float)
        return X - self.mean


class DummyEstimator:
    def __init__(self):
        self.fitted = False
        self.majority_class = None

    def fit(self, X, y):
        self.fitted = True
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("DummyEstimator has not been fitted yet.")
        return np.full(X.shape[0], self.majority_class)


class DummyStreamingEstimator:
    def __init__(self):
        self.fitted = False
        self.n_updates = 0
        self.y_seen = []
        self.majority_class = None

    def partial_fit(self, X, y):
        self.fitted = True
        self.n_updates += 1
        self.y_seen.extend(np.asarray(y).tolist())

        values, counts = np.unique(self.y_seen, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("DummyStreamingEstimator has not been fitted yet.")
        return np.full(X.shape[0], self.majority_class)


class BadTransformerNoTransform:
    def fit(self, X):
        return self


class BadEstimatorNoPredict:
    def fit(self, X, y=None):
        return self


class BadStepNoFit:
    def transform(self, X):
        return X


class BadFinalNoFitOrPartialFit:
    def predict(self, X):
        return np.zeros(X.shape[0])


# ---------------- EXISTING BATCH PIPELINE TESTS ---------------- #

def test_pipeline_fit_predict_basic():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (3,)
    assert np.array_equal(preds, np.array([1, 1, 1]))


def test_pipeline_transform_after_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    pipe = Pipeline([
        ("transform", DummyTransformer())
    ])

    pipe.fit(X)
    X_out = pipe.transform(X)

    assert X_out.shape == X.shape
    assert np.allclose(np.mean(X_out, axis=0), [0, 0])


def test_pipeline_fit_transform():
    X = np.array([[1, 2], [3, 4], [5, 6]])

    pipe = Pipeline([
        ("transform", DummyTransformer())
    ])

    X_out = pipe.fit_transform(X)

    assert X_out.shape == X.shape
    assert np.allclose(np.mean(X_out, axis=0), [0, 0])


def test_pipeline_empty_steps():
    with pytest.raises(ValueError):
        Pipeline([])


def test_pipeline_bad_transformer_missing_transform():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("bad", BadTransformerNoTransform()),
        ("model", DummyEstimator())
    ])

    with pytest.raises(ValueError):
        pipe.fit(X, y)


def test_pipeline_bad_step_missing_fit():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("bad", BadStepNoFit())
    ])

    with pytest.raises(ValueError):
        pipe.fit(X)


def test_pipeline_predict_final_step_missing_predict():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("bad_model", BadEstimatorNoPredict())
    ])

    pipe.fit(X, y)

    with pytest.raises(ValueError):
        pipe.predict(X)


def test_pipeline_predict_before_fit_raises_from_transformer():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    with pytest.raises(ValueError):
        pipe.predict(X)


def test_pipeline_transform_only_applies_transformers_before_model():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    X_transformed = pipe.transform(X)

    assert X_transformed.shape == X.shape
    assert np.allclose(np.mean(X_transformed, axis=0), [0, 0])


def test_pipeline_single_estimator_predict():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([2, 2, 1])

    pipe = Pipeline([
        ("model", DummyEstimator())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert np.array_equal(preds, np.array([2, 2, 2]))


# ---------------- STREAMING PIPELINE TESTS ---------------- #

def test_pipeline_partial_fit_with_streaming_transformer_and_estimator():
    X1 = np.array([[1, 2], [3, 4]])
    y1 = np.array([0, 1])

    X2 = np.array([[5, 6], [7, 8], [9, 10]])
    y2 = np.array([1, 1, 0])

    transformer = DummyStreamingTransformer()
    estimator = DummyStreamingEstimator()

    pipe = Pipeline([
        ("stream_transform", transformer),
        ("stream_model", estimator)
    ])

    pipe.partial_fit(X1, y1)
    pipe.partial_fit(X2, y2)

    preds = pipe.predict(X2)

    assert transformer.n_updates == 2
    assert estimator.n_updates == 2
    assert preds.shape == (3,)
    assert np.all(preds == 1)


def test_pipeline_partial_fit_falls_back_to_fit_for_non_streaming_transformer():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 1])

    pipe = Pipeline([
        ("transform", DummyTransformer()),
        ("model", DummyStreamingEstimator())
    ])

    pipe.partial_fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (3,)
    assert np.all(preds == 1)


def test_pipeline_partial_fit_single_streaming_estimator():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 1])

    pipe = Pipeline([
        ("model", DummyStreamingEstimator())
    ])

    pipe.partial_fit(X, y)
    preds = pipe.predict(X)

    assert np.array_equal(preds, np.array([1, 1]))


def test_pipeline_partial_fit_transformer_only_pipeline():
    X = np.array([[1, 2], [3, 4]])

    pipe = Pipeline([
        ("transform", DummyStreamingTransformer())
    ])

    pipe.partial_fit(X)
    X_out = pipe.transform(X)

    assert X_out.shape == X.shape
    assert np.allclose(np.mean(X_out, axis=0), [0, 0])


def test_pipeline_partial_fit_bad_transformer_missing_transform():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("bad", BadTransformerNoTransform()),
        ("model", DummyStreamingEstimator())
    ])

    with pytest.raises(ValueError):
        pipe.partial_fit(X, y)


def test_pipeline_partial_fit_bad_final_step_missing_fit_and_partial_fit():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    pipe = Pipeline([
        ("model", BadFinalNoFitOrPartialFit())
    ])

    with pytest.raises(ValueError):
        pipe.partial_fit(X, y)
