import numpy as np
import pytest

from numcompute.tree import DecisionTreeClassifier


# ---------------- BASIC FIT / PREDICT TESTS ---------------- #

def test_decision_tree_fit_predict_perfect_simple_split():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


def test_decision_tree_score():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    assert np.isclose(tree.score(X, y), 1.0)


def test_predict_proba_shape_and_rows_sum_to_one():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_entropy_criterion_works():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, criterion="entropy")
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, y)


# ---------------- STOPPING / EDGE STRUCTURE TESTS ---------------- #

def test_max_depth_zero_creates_majority_leaf():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([1, 1, 1, 1]))


def test_min_samples_split_prevents_split():
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([1, 1, 1]))


def test_all_same_class_predicts_same_class():
    X = np.array([[0], [1], [2]], dtype=float)
    y = np.array([2, 2, 2])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert np.array_equal(tree.predict(X), np.array([2, 2, 2]))


def test_constant_feature_falls_back_to_majority_class():
    X = np.array([[1], [1], [1], [1]], dtype=float)
    y = np.array([0, 1, 1, 1])

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X, y)

    assert np.array_equal(tree.predict(X), np.array([1, 1, 1, 1]))


def test_tie_breaking_uses_smallest_sorted_class():
    X = np.array([[1], [1]], dtype=float)
    y = np.array([1, 0])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([0, 0]))


# ---------------- PARTIAL FIT STREAMING TESTS ---------------- #

def test_partial_fit_single_chunk_predicts():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X, y)

    assert np.array_equal(tree.predict(X), y)
    assert tree.n_samples_seen_ == 4


def test_partial_fit_multiple_chunks_accumulates_samples():
    X1 = np.array([[0], [1]], dtype=float)
    y1 = np.array([0, 0])

    X2 = np.array([[2], [3]], dtype=float)
    y2 = np.array([1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X1, y1)
    tree.partial_fit(X2, y2)

    X_all = np.vstack([X1, X2])
    y_all = np.concatenate([y1, y2])

    assert tree.n_samples_seen_ == 4
    assert np.array_equal(tree.predict(X_all), y_all)


def test_partial_fit_accepts_classes_argument():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 0])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.partial_fit(X, y, classes=np.array([0, 1]))

    assert np.array_equal(tree.classes_, np.array([0, 1]))
    assert tree.predict_proba(X).shape == (2, 2)


def test_partial_fit_feature_mismatch_raises():
    tree = DecisionTreeClassifier()

    tree.partial_fit(np.array([[1, 2]], dtype=float), np.array([0]))

    with pytest.raises(ValueError):
        tree.partial_fit(np.array([[1, 2, 3]], dtype=float), np.array([0]))


# ---------------- MAX FEATURES TESTS ---------------- #

def test_max_features_sqrt_works():
    X = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [2, 1, 0, 0],
        [3, 1, 0, 0]
    ], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, max_features="sqrt", random_state=1)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == y.shape


def test_max_features_log2_works():
    X = np.array([
        [0, 0, 1, 1],
        [1, 0, 1, 1],
        [2, 1, 0, 0],
        [3, 1, 0, 0]
    ], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2, max_features="log2", random_state=1)
    tree.fit(X, y)

    assert tree.predict(X).shape == y.shape


def test_max_features_int_and_float_work():
    X = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [2, 1, 0],
        [3, 1, 0]
    ], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree_int = DecisionTreeClassifier(max_depth=2, max_features=2, random_state=1)
    tree_float = DecisionTreeClassifier(max_depth=2, max_features=0.5, random_state=1)

    tree_int.fit(X, y)
    tree_float.fit(X, y)

    assert tree_int.predict(X).shape == y.shape
    assert tree_float.predict(X).shape == y.shape


def test_invalid_max_features_raises_during_fit():
    X = np.array([[0, 1], [1, 0]], dtype=float)
    y = np.array([0, 1])

    tree = DecisionTreeClassifier(max_features="bad")

    with pytest.raises(ValueError):
        tree.fit(X, y)


# ---------------- SAMPLE WEIGHT TESTS ---------------- #

def test_sample_weight_changes_majority_leaf():
    X = np.array([[1], [1], [1]], dtype=float)
    y = np.array([0, 1, 1])
    weights = np.array([10.0, 1.0, 1.0])

    tree = DecisionTreeClassifier(max_depth=0)
    tree.fit(X, y, sample_weight=weights)

    preds = tree.predict(X)

    assert np.array_equal(preds, np.array([0, 0, 0]))


def test_negative_sample_weight_raises():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, y, sample_weight=np.array([1.0, -1.0]))


def test_sample_weight_length_mismatch_raises():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, y, sample_weight=np.array([1.0]))


# ---------------- NAN AND VALIDATION TESTS ---------------- #

def test_nan_values_do_not_crash_prediction():
    X = np.array([[0], [1], [np.nan], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == y.shape


def test_predict_before_fit_raises():
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.predict(np.array([[1]], dtype=float))


def test_predict_proba_before_fit_raises():
    tree = DecisionTreeClassifier()

    with pytest.raises(ValueError):
        tree.predict_proba(np.array([[1]], dtype=float))


def test_fit_empty_X_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.empty((0, 2)), np.array([]))


def test_fit_shape_mismatch_raises():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0])

    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, y)


def test_fit_non_2d_X_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.array([1, 2, 3]), np.array([0, 1, 1]))


def test_fit_non_1d_y_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.array([[1], [2]]), np.array([[0], [1]]))


def test_fit_non_numeric_y_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(np.array([[1], [2]], dtype=float), np.array(["a", "b"]))


def test_predict_feature_mismatch_raises():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)

    with pytest.raises(ValueError):
        tree.predict(np.array([[0, 1]], dtype=float))


# ---------------- PARAMETER VALIDATION TESTS ---------------- #

def test_invalid_max_depth_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(max_depth=-1)


def test_invalid_min_samples_split_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_samples_split=1)


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bad")


# ---------------- RESET / FEATURE IMPORTANCE TESTS ---------------- #

def test_reset_clears_state():
    X = np.array([[0], [1]], dtype=float)
    y = np.array([0, 1])

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X, y)
    tree.reset()

    assert tree.root_ is None
    assert tree.classes_ is None
    assert tree.n_samples_seen_ == 0


def test_feature_importances_shape():
    X = np.array([[0, 1], [1, 1], [2, 0], [3, 0]], dtype=float)
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(max_depth=2)
    tree.fit(X, y)

    assert tree.feature_importances_.shape == (2,)
    assert np.all(tree.feature_importances_ >= 0)
