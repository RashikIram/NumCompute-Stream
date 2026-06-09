import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    confusion_matrix,
    precision,
    recall,
    specificity,
    f1,
    balanced_accuracy,
    macro_precision,
    macro_recall,
    macro_f1,
    micro_precision,
    micro_recall,
    micro_f1,
    weighted_f1,
    matthews_corrcoef,
    cohen_kappa,
    top_k_accuracy,
    log_loss,
    brier_score,
    mse,
    rmse,
    mad,
    mape,
    roc_curve,
    auc,
    StreamingConfusionMatrix,
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingBalancedAccuracy,
    StreamingMacroF1,
    StreamingWeightedF1,
    StreamingMCC,
    StreamingCohenKappa,
    RollingAccuracy,
    RollingF1,
)


# ---------------- BATCH CLASSIFICATION TESTS ---------------- #

def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(accuracy(y_true, y_pred), 0.8)


def test_confusion_matrix_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    cm = confusion_matrix(y_true, y_pred)

    expected = np.array([
        [1, 1],
        [1, 1]
    ])

    assert np.array_equal(cm, expected)


def test_confusion_matrix_with_fixed_classes():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1, 0])

    cm = confusion_matrix(y_true, y_pred, classes=[0, 1, 2])

    assert cm.shape == (3, 3)
    assert np.array_equal(cm[2], np.array([0, 0, 0]))


def test_binary_precision_recall_f1_specificity():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(precision(y_true, y_pred), 1.0)
    assert np.isclose(recall(y_true, y_pred), 2 / 3)
    assert np.isclose(specificity(y_true, y_pred), 1.0)
    assert 0 <= f1(y_true, y_pred) <= 1


def test_balanced_accuracy_binary():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    assert np.isclose(balanced_accuracy(y_true, y_pred), (2 / 3 + 1.0) / 2)


def test_precision_no_positive_predictions_raises():
    with pytest.raises(ValueError):
        precision([1, 1, 1], [0, 0, 0])


def test_recall_no_actual_positives_raises():
    with pytest.raises(ValueError):
        recall([0, 0, 0], [0, 0, 0])


def test_specificity_no_actual_negatives_raises():
    with pytest.raises(ValueError):
        specificity([1, 1, 1], [1, 1, 1])


def test_shape_mismatch_accuracy_raises():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


def test_non_numeric_metric_input_raises():
    with pytest.raises(ValueError):
        accuracy(["a", "b"], ["a", "b"])


# ---------------- MULTICLASS METRIC TESTS ---------------- #

def test_macro_micro_weighted_scores():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 2, 1, 0])

    assert 0 <= macro_precision(y_true, y_pred) <= 1
    assert 0 <= macro_recall(y_true, y_pred) <= 1
    assert 0 <= macro_f1(y_true, y_pred) <= 1
    assert 0 <= weighted_f1(y_true, y_pred) <= 1

    assert np.isclose(micro_precision(y_true, y_pred), accuracy(y_true, y_pred))
    assert np.isclose(micro_recall(y_true, y_pred), accuracy(y_true, y_pred))
    assert np.isclose(micro_f1(y_true, y_pred), accuracy(y_true, y_pred))


def test_mcc_perfect_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    assert np.isclose(matthews_corrcoef(y_true, y_pred), 1.0)


def test_mcc_multiclass_range():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])

    score = matthews_corrcoef(y_true, y_pred)

    assert -1 <= score <= 1


def test_cohen_kappa_perfect_prediction():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])

    assert np.isclose(cohen_kappa(y_true, y_pred), 1.0)


def test_cohen_kappa_range():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])

    score = cohen_kappa(y_true, y_pred)

    assert -1 <= score <= 1


def test_top_k_accuracy_basic():
    y_true = np.array([0, 2, 1])

    y_scores = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.3, 0.5],
        [0.1, 0.7, 0.2]
    ])

    assert np.isclose(top_k_accuracy(y_true, y_scores, k=1), 1.0)


def test_top_k_accuracy_invalid_k_raises():
    y_true = np.array([0, 1])
    y_scores = np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ])

    with pytest.raises(ValueError):
        top_k_accuracy(y_true, y_scores, k=0)


def test_log_loss_and_brier_score():
    y_true = np.array([0, 1])

    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])

    assert log_loss(y_true, y_proba) >= 0
    assert brier_score(y_true, y_proba) >= 0


def test_log_loss_invalid_probability_shape_raises():
    with pytest.raises(ValueError):
        log_loss([0, 1], [0.9, 0.1])


def test_brier_score_invalid_class_index_raises():
    y_true = np.array([0, 3])
    y_proba = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7]
    ])

    with pytest.raises(ValueError):
        brier_score(y_true, y_proba)


# ---------------- REGRESSION TESTS ---------------- #

def test_regression_metrics_non_negative():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    assert mse(y_true, y_pred) >= 0
    assert rmse(y_true, y_pred) >= 0
    assert mad(y_true, y_pred) >= 0
    assert mape(y_true, y_pred) >= 0


def test_mape_zero_case_raises():
    with pytest.raises(ValueError):
        mape([0, 0, 0], [1, 2, 3])


# ---------------- ROC / AUC TESTS ---------------- #

def test_roc_curve_valid():
    y_true = np.array([0, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

    fpr, tpr = roc_curve(y_true, y_scores)

    assert len(fpr) == len(tpr)
    assert np.all(fpr >= 0)
    assert np.all(tpr >= 0)


def test_roc_invalid_labels_raises():
    with pytest.raises(ValueError):
        roc_curve([0, 1, 2], [0.1, 0.5, 0.9])


def test_auc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        auc([0, 0.5], [0.1])


# ---------------- STREAMING CORE TESTS ---------------- #

def test_streaming_accuracy_matches_batch_accuracy():
    y_true_1 = np.array([1, 0, 1])
    y_pred_1 = np.array([1, 0, 0])

    y_true_2 = np.array([0, 1])
    y_pred_2 = np.array([0, 1])

    metric = StreamingAccuracy()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert np.isclose(metric.result(), accuracy(y_true_all, y_pred_all))


def test_streaming_confusion_matrix_accumulates():
    metric = StreamingConfusionMatrix(classes=[0, 1])

    metric.update([0, 1], [0, 1])
    metric.update([1, 0], [0, 0])

    expected = confusion_matrix(
        np.array([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        classes=np.array([0, 1])
    )

    assert np.array_equal(metric.result(), expected)


def test_streaming_confusion_matrix_expands_classes():
    metric = StreamingConfusionMatrix()

    metric.update([0, 1], [0, 1])
    metric.update([2], [2])

    assert metric.result().shape == (3, 3)


def test_streaming_precision_recall_f1():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    p = StreamingPrecision().update(y_true, y_pred).result()
    r = StreamingRecall().update(y_true, y_pred).result()
    f = StreamingF1().update(y_true, y_pred).result()

    assert np.isclose(p, precision(y_true, y_pred))
    assert np.isclose(r, recall(y_true, y_pred))
    assert np.isclose(f, f1(y_true, y_pred))


def test_streaming_balanced_accuracy():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    metric = StreamingBalancedAccuracy()
    metric.update(y_true, y_pred)

    assert np.isclose(metric.result(), balanced_accuracy(y_true, y_pred))


def test_streaming_reset():
    metric = StreamingAccuracy()
    metric.update([1, 0], [1, 0])

    assert metric.result() == 1.0

    metric.reset()

    assert metric.result() == 0.0


def test_streaming_shape_mismatch_raises():
    metric = StreamingAccuracy()

    with pytest.raises(ValueError):
        metric.update([1, 0], [1])


# ---------------- STREAMING ADVANCED TESTS ---------------- #

def test_streaming_macro_f1_matches_batch_macro_f1():
    y_true_1 = np.array([0, 1, 2])
    y_pred_1 = np.array([0, 2, 2])
    y_true_2 = np.array([2, 1, 0])
    y_pred_2 = np.array([2, 1, 0])

    metric = StreamingMacroF1()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert np.isclose(metric.result(), macro_f1(y_true_all, y_pred_all))


def test_streaming_weighted_f1_matches_batch_weighted_f1():
    y_true_1 = np.array([0, 1, 2])
    y_pred_1 = np.array([0, 2, 2])
    y_true_2 = np.array([2, 1, 0])
    y_pred_2 = np.array([2, 1, 0])

    metric = StreamingWeightedF1()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert np.isclose(metric.result(), weighted_f1(y_true_all, y_pred_all))


def test_streaming_mcc_matches_batch_mcc():
    y_true_1 = np.array([0, 1, 1])
    y_pred_1 = np.array([0, 1, 0])
    y_true_2 = np.array([0, 1, 0])
    y_pred_2 = np.array([0, 1, 1])

    metric = StreamingMCC(classes=[0, 1])
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert np.isclose(metric.result(), matthews_corrcoef(y_true_all, y_pred_all))


def test_streaming_kappa_matches_batch_kappa():
    y_true_1 = np.array([0, 1, 2])
    y_pred_1 = np.array([0, 2, 2])
    y_true_2 = np.array([0, 1, 2])
    y_pred_2 = np.array([0, 1, 1])

    metric = StreamingCohenKappa()
    metric.update(y_true_1, y_pred_1)
    metric.update(y_true_2, y_pred_2)

    y_true_all = np.concatenate([y_true_1, y_true_2])
    y_pred_all = np.concatenate([y_pred_1, y_pred_2])

    assert np.isclose(metric.result(), cohen_kappa(y_true_all, y_pred_all))


def test_empty_streaming_advanced_metrics_return_zero():
    assert StreamingMacroF1().result() == 0.0
    assert StreamingWeightedF1().result() == 0.0
    assert StreamingMCC().result() == 0.0
    assert StreamingCohenKappa().result() == 0.0


def test_streaming_advanced_metrics_reset():
    metric = StreamingMCC(classes=[0, 1])
    metric.update([0, 1], [0, 1])

    assert metric.result() == 1.0

    metric.reset()

    assert metric.result() == 0.0


# ---------------- ROLLING METRIC TESTS ---------------- #

def test_rolling_accuracy_uses_recent_window():
    metric = RollingAccuracy(window_size=3)

    metric.update([1, 1, 1], [1, 1, 1])
    metric.update([0, 0], [1, 0])

    # Last 3 samples: true [1, 0, 0], pred [1, 1, 0] => 2/3
    assert np.isclose(metric.result(), 2 / 3)


def test_rolling_f1_basic():
    metric = RollingF1(window_size=4)

    metric.update([1, 0, 1, 1], [1, 0, 0, 1])

    assert 0 <= metric.result() <= 1


def test_rolling_invalid_window_raises():
    with pytest.raises(ValueError):
        RollingAccuracy(window_size=0)

    with pytest.raises(ValueError):
        RollingF1(window_size=0)

def test_streaming_accuracy_empty_chunk_keeps_result_zero():
    metric = StreamingAccuracy()

    metric.update(np.array([]), np.array([]))

    assert metric.result() == 0.0


def test_streaming_confusion_matrix_late_unseen_class():
    metric = StreamingConfusionMatrix()

    metric.update([0, 1], [0, 1])
    metric.update([2, 2], [2, 1])

    result = metric.result()

    assert result.shape == (3, 3)
    assert np.sum(result) == 4


def test_log_loss_class_index_out_of_bounds_raises():
    y_true = np.array([0, 2])
    y_proba = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])

    with pytest.raises(ValueError):
        log_loss(y_true, y_proba)


def test_brier_score_class_index_out_of_bounds_raises():
    y_true = np.array([0, 2])
    y_proba = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])

    with pytest.raises(ValueError):
        brier_score(y_true, y_proba)


def test_all_wrong_prediction_metrics_are_valid():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])

    assert np.isclose(accuracy(y_true, y_pred), 0.0)
    assert np.isclose(matthews_corrcoef(y_true, y_pred), -1.0)
    assert 0 <= macro_f1(y_true, y_pred) <= 1
    assert 0 <= weighted_f1(y_true, y_pred) <= 1
