import os
import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_model_comparison,
    plot_predictions_vs_ground_truth,
    plot_class_distribution,
    plot_class_distribution_over_time,
    plot_confusion_matrix,
    plot_missing_values,
    plot_feature_histogram,
    plot_roc_curve,
    plot_multiclass_roc_curves,
)


# ---------------- REQUIRED PLOT TESTS ---------------- #

def test_plot_metric_over_time_returns_figure():
    fig = plot_metric_over_time([0.5, 0.6, 0.7], show=False)

    assert isinstance(fig, Figure)


def test_plot_metric_over_time_empty_raises():
    with pytest.raises(ValueError):
        plot_metric_over_time([], show=False)


def test_compare_models_returns_figure():
    fig = compare_models([0.5, 0.6], [0.4, 0.7], labels=("A", "B"), show=False)

    assert isinstance(fig, Figure)


def test_compare_models_length_mismatch_raises():
    with pytest.raises(ValueError):
        compare_models([0.5, 0.6], [0.4], show=False)


def test_compare_models_invalid_labels_raises():
    with pytest.raises(ValueError):
        compare_models([0.5], [0.4], labels=("A", "B", "C"), show=False)


def test_predictions_vs_ground_truth_returns_figure():
    fig = plot_predictions_vs_ground_truth([0, 1, 1], [0, 0, 1], show=False)

    assert isinstance(fig, Figure)


def test_predictions_vs_ground_truth_shape_mismatch_raises():
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1], [0], show=False)


# ---------------- MODEL COMPARISON TESTS ---------------- #

def test_plot_model_comparison_returns_figure():
    fig = plot_model_comparison({
        "Tree": [0.5, 0.6, 0.7],
        "Forest": [0.6, 0.7, 0.8]
    }, show=False)

    assert isinstance(fig, Figure)


def test_plot_model_comparison_empty_dict_raises():
    with pytest.raises(ValueError):
        plot_model_comparison({}, show=False)


def test_plot_model_comparison_length_mismatch_raises():
    with pytest.raises(ValueError):
        plot_model_comparison({
            "A": [0.1, 0.2],
            "B": [0.1]
        }, show=False)


# ---------------- CLASS DISTRIBUTION TESTS ---------------- #

def test_plot_class_distribution_returns_figure():
    fig = plot_class_distribution([0, 1, 1, 2], show=False)

    assert isinstance(fig, Figure)


def test_plot_class_distribution_empty_raises():
    with pytest.raises(ValueError):
        plot_class_distribution([], show=False)


def test_plot_class_distribution_over_time_returns_figure():
    counts = np.array([
        [3, 2],
        [2, 4]
    ])

    fig = plot_class_distribution_over_time(counts, classes=[0, 1], show=False)

    assert isinstance(fig, Figure)


def test_plot_class_distribution_over_time_class_mismatch_raises():
    counts = np.array([
        [3, 2, 1],
        [2, 4, 1]
    ])

    with pytest.raises(ValueError):
        plot_class_distribution_over_time(counts, classes=[0, 1], show=False)


# ---------------- CONFUSION MATRIX TESTS ---------------- #

def test_plot_confusion_matrix_returns_figure():
    cm = np.array([
        [2, 1],
        [0, 3]
    ])

    fig = plot_confusion_matrix(cm, classes=[0, 1], show=False)

    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_normalized_returns_figure():
    cm = np.array([
        [2, 1],
        [0, 3]
    ])

    fig = plot_confusion_matrix(cm, classes=[0, 1], normalize=True, show=False)

    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_non_square_raises():
    with pytest.raises(ValueError):
        plot_confusion_matrix(np.array([[1, 2, 3], [4, 5, 6]]), show=False)


def test_plot_confusion_matrix_class_mismatch_raises():
    cm = np.array([
        [1, 0],
        [0, 1]
    ])

    with pytest.raises(ValueError):
        plot_confusion_matrix(cm, classes=[0, 1, 2], show=False)


# ---------------- MISSING VALUES / HISTOGRAM TESTS ---------------- #

def test_plot_missing_values_returns_figure():
    X = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
        [np.nan, 4.0]
    ], dtype=object)

    fig = plot_missing_values(X, feature_names=["A", "B"], show=False)

    assert isinstance(fig, Figure)


def test_plot_missing_values_feature_name_mismatch_raises():
    X = np.array([
        [1.0, np.nan],
        [2.0, 3.0]
    ], dtype=object)

    with pytest.raises(ValueError):
        plot_missing_values(X, feature_names=["A"], show=False)


def test_plot_feature_histogram_returns_figure():
    X = np.array([
        [1.0, 10.0],
        [2.0, 11.0],
        [3.0, 12.0]
    ])

    fig = plot_feature_histogram(X, feature_index=0, bins=2, show=False)

    assert isinstance(fig, Figure)


def test_plot_feature_histogram_invalid_feature_index_raises():
    X = np.array([
        [1.0, 10.0],
        [2.0, 11.0]
    ])

    with pytest.raises(ValueError):
        plot_feature_histogram(X, feature_index=5, show=False)


def test_plot_feature_histogram_invalid_bins_raises():
    X = np.array([
        [1.0],
        [2.0]
    ])

    with pytest.raises(ValueError):
        plot_feature_histogram(X, feature_index=0, bins=0, show=False)


def test_plot_feature_histogram_all_nan_raises():
    X = np.array([
        [np.nan],
        [np.nan]
    ])

    with pytest.raises(ValueError):
        plot_feature_histogram(X, feature_index=0, show=False)


# ---------------- ROC TESTS ---------------- #

def test_plot_roc_curve_returns_figure():
    fig = plot_roc_curve([0.0, 0.5, 1.0], [0.0, 0.8, 1.0], show=False)

    assert isinstance(fig, Figure)


def test_plot_roc_curve_shape_mismatch_raises():
    with pytest.raises(ValueError):
        plot_roc_curve([0.0, 1.0], [0.0], show=False)


def test_plot_roc_curve_out_of_range_raises():
    with pytest.raises(ValueError):
        plot_roc_curve([0.0, 1.5], [0.0, 1.0], show=False)

# ---------------- MULTICLASS ROC VISUALISATION TESTS ---------------- #

def test_plot_multiclass_roc_curves_returns_figure():
    y_true = np.array([0, 1, 2, 0, 1, 2])

    y_scores = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85],
    ])

    fig = plot_multiclass_roc_curves(
        y_true,
        y_scores,
        classes=[0, 1, 2],
        show=False
    )

    assert isinstance(fig, Figure)


def test_plot_multiclass_roc_curves_class_mismatch_raises():
    y_true = np.array([0, 1, 2])

    y_scores = np.array([
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ])

    with pytest.raises(ValueError):
        plot_multiclass_roc_curves(
            y_true,
            y_scores,
            classes=[0, 1],
            show=False
        )


def test_plot_multiclass_roc_curves_no_valid_curve_raises():
    y_true = np.array([0, 0, 0])

    y_scores = np.array([
        [0.90, 0.10],
        [0.85, 0.15],
        [0.80, 0.20],
    ])

    with pytest.raises(ValueError):
        plot_multiclass_roc_curves(
            y_true,
            y_scores,
            classes=[0, 1],
            show=False
        )


def test_plot_multiclass_roc_curves_save_file(tmp_path):
    y_true = np.array([0, 1, 2, 0, 1, 2])

    y_scores = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.85, 0.10, 0.05],
        [0.10, 0.80, 0.10],
        [0.05, 0.10, 0.85],
    ])

    output_path = tmp_path / "multiclass_roc.png"

    fig = plot_multiclass_roc_curves(
        y_true,
        y_scores,
        classes=[0, 1, 2],
        save_path=str(output_path),
        show=False
    )

    assert isinstance(fig, Figure)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# ---------------- SAVE TEST ---------------- #

def test_plot_saves_to_file(tmp_path):
    save_path = tmp_path / "metric_plot.png"

    fig = plot_metric_over_time([0.5, 0.6], save_path=save_path, show=False)

    assert isinstance(fig, Figure)
    assert os.path.exists(save_path)


def test_show_false_closes_figure():
    fig = plot_metric_over_time([0.5, 0.6], show=False)

    assert not plt.fignum_exists(fig.number)
