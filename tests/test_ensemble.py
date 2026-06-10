import numpy as np
import pytest

from numcompute_stream.ensemble import (
    EnsembleClassifier,
    OnlineBaggingClassifier,
    RandomForestClassifier,
    RandomSubspaceClassifier,
    ExtraTreesClassifier,
    AdaBoostSAMMEClassifier,
)


def make_binary_data():
    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.2, 0.1, 1.0, 0.9],
        [1.0, 1.0, 0.0, 0.0],
        [1.2, 0.9, 0.1, 0.0],
        [0.1, 0.2, 1.1, 1.0],
        [1.1, 1.2, 0.0, 0.2],
    ])
    y = np.array([0, 0, 1, 1, 0, 1])
    return X, y


# ---------------- VALIDATION TESTS ---------------- #

def test_base_ensemble_invalid_n_estimators_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)


def test_base_ensemble_invalid_voting_raises():
    with pytest.raises(ValueError):
        EnsembleClassifier(voting="invalid")


def test_predict_before_fit_raises():
    X, _ = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=3, random_state=1)

    with pytest.raises(ValueError):
        model.predict(X)


def test_fit_with_shape_mismatch_raises():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=3, random_state=1)

    with pytest.raises(ValueError):
        model.fit(X, y[:-1])


def test_partial_fit_feature_mismatch_raises():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=3, random_state=1)
    model.partial_fit(X[:, :2], y)

    with pytest.raises(ValueError):
        model.partial_fit(X, y)


# ---------------- ONLINE BAGGING TESTS ---------------- #

def test_online_bagging_fit_predict_shape():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=5, max_depth=3, random_state=1)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert len(model.estimators_) == 5
    assert model.n_samples_seen_ == len(y)


def test_online_bagging_predict_proba_shape_and_rows_sum_to_one():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=5, max_depth=3, random_state=1)

    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(y), 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_online_bagging_partial_fit_accumulates_chunks():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=4, max_depth=3, random_state=1)

    model.partial_fit(X[:3], y[:3])
    model.partial_fit(X[3:], y[3:])

    assert model.n_samples_seen_ == len(y)
    assert model.predict(X).shape == y.shape


def test_online_bagging_hard_voting_works():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(
        n_estimators=5,
        max_depth=3,
        voting="hard",
        random_state=1
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape


def test_online_bagging_score_is_valid():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=5, max_depth=3, random_state=1)

    model.fit(X, y)
    score = model.score(X, y)

    assert 0 <= score <= 1


def test_online_bagging_reset_clears_state():
    X, y = make_binary_data()
    model = OnlineBaggingClassifier(n_estimators=3, random_state=1)
    model.fit(X, y)
    model.reset()

    assert model.estimators_ == []
    assert model.classes_ is None
    assert model.n_samples_seen_ == 0


# ---------------- RANDOM FOREST TESTS ---------------- #

def test_random_forest_fit_predict():
    X, y = make_binary_data()
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=2)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert model.max_features == "sqrt"


def test_random_forest_partial_fit():
    X, y = make_binary_data()
    model = RandomForestClassifier(n_estimators=4, max_depth=3, random_state=2)

    model.partial_fit(X[:2], y[:2])
    model.partial_fit(X[2:], y[2:])

    assert model.n_samples_seen_ == len(y)
    assert model.predict_proba(X).shape == (len(y), 2)


# ---------------- RANDOM SUBSPACE TESTS ---------------- #

def test_random_subspace_feature_indices_are_created():
    X, y = make_binary_data()
    model = RandomSubspaceClassifier(
        n_estimators=4,
        max_depth=3,
        max_features=2,
        random_state=3
    )

    model.fit(X, y)

    assert len(model.feature_indices_) == 4
    assert all(len(features) == 2 for features in model.feature_indices_)


def test_random_subspace_predict_shape():
    X, y = make_binary_data()
    model = RandomSubspaceClassifier(
        n_estimators=4,
        max_depth=3,
        max_features="sqrt",
        random_state=3
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape


def test_random_subspace_reset_clears_feature_indices():
    X, y = make_binary_data()
    model = RandomSubspaceClassifier(n_estimators=3, random_state=3)
    model.fit(X, y)
    model.reset()

    assert model.feature_indices_ == []


# ---------------- EXTRA TREES TESTS ---------------- #

def test_extra_trees_fit_predict():
    X, y = make_binary_data()
    model = ExtraTreesClassifier(n_estimators=5, max_depth=3, random_state=4)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert len(model.estimators_) == 5


def test_extra_trees_partial_fit():
    X, y = make_binary_data()
    model = ExtraTreesClassifier(n_estimators=5, max_depth=3, random_state=4)

    model.partial_fit(X[:3], y[:3])
    model.partial_fit(X[3:], y[3:])

    assert model.n_samples_seen_ == len(y)
    assert model.predict(X).shape == y.shape


# ---------------- ADABOOST SAMME TESTS ---------------- #

def test_adaboost_fit_predict_shape():
    X, y = make_binary_data()
    model = AdaBoostSAMMEClassifier(n_estimators=5, max_depth=1, random_state=5)

    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert len(model.estimators_) >= 1
    assert len(model.estimator_weights_) == len(model.estimators_)


def test_adaboost_predict_proba_shape_and_rows_sum_to_one():
    X, y = make_binary_data()
    model = AdaBoostSAMMEClassifier(n_estimators=5, max_depth=1, random_state=5)

    model.fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(y), 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_adaboost_partial_fit_accumulates_chunks():
    X, y = make_binary_data()
    model = AdaBoostSAMMEClassifier(n_estimators=5, max_depth=1, random_state=5)

    model.partial_fit(X[:3], y[:3])
    model.partial_fit(X[3:], y[3:])

    assert model.n_samples_seen_ == len(y)
    assert model.predict(X).shape == y.shape


def test_adaboost_invalid_learning_rate_raises():
    with pytest.raises(ValueError):
        AdaBoostSAMMEClassifier(learning_rate=0)


def test_adaboost_single_class_chunk_works():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    model = AdaBoostSAMMEClassifier(n_estimators=3, random_state=5)
    model.fit(X, y)

    assert np.array_equal(model.predict(X), y)


# ---------------- MULTICLASS / CLASS EXPANSION TESTS ---------------- #

def test_online_bagging_multiclass_predict_proba():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.1, 0.0],
        [1.1, 1.0],
        [2.1, 2.0],
    ])
    y = np.array([0, 1, 2, 0, 1, 2])

    model = OnlineBaggingClassifier(n_estimators=5, max_depth=3, random_state=6)
    model.fit(X, y)

    proba = model.predict_proba(X)

    assert proba.shape == (len(y), 3)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_streaming_new_class_later_chunk_expands_classes():
    X1 = np.array([[0.0], [0.1]])
    y1 = np.array([0, 0])

    X2 = np.array([[1.0], [1.1]])
    y2 = np.array([1, 1])

    model = OnlineBaggingClassifier(n_estimators=3, max_depth=2, random_state=7)
    model.partial_fit(X1, y1)
    model.partial_fit(X2, y2)

    assert np.array_equal(model.classes_, np.array([0, 1]))
    assert model.predict_proba(np.vstack([X1, X2])).shape == (4, 2)
