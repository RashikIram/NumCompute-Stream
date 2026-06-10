import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------- VALIDATION HELPERS ---------------- #

def _validate_1d(values, name="values", allow_empty=False):
    """
    Validate and convert input into a 1D NumPy array.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        Input values.
    name : str, default="values"
        Name used in error messages.
    allow_empty : bool, default=False
        Whether empty arrays are allowed.

    Returns
    -------
    arr : np.ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If input is None, not 1D, or empty when allow_empty=False.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    if values is None:
        raise ValueError(f"{name} cannot be None")

    arr = np.asarray(values)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return arr


def _validate_2d(values, name="values", allow_empty=False):
    """
    Validate and convert input into a 2D NumPy array.

    Parameters
    ----------
    values : array-like of shape (n_samples, n_features)
        Input matrix.
    name : str, default="values"
        Name used in error messages.
    allow_empty : bool, default=False
        Whether empty arrays are allowed.

    Returns
    -------
    arr : np.ndarray

    Raises
    ------
    ValueError
        If input is None, not 2D, or empty when allow_empty=False.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    if values is None:
        raise ValueError(f"{name} cannot be None")

    arr = np.asarray(values)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return arr


def _auto_figsize(n_items=1, base_width=7.0, base_height=4.5, horizontal=False):
    """
    Choose a readable matplotlib figure size based on label count.

    Parameters
    ----------
    n_items : int, default=1
        Number of plotted categories or x-axis ticks.
    base_width : float, default=7.0
        Minimum figure width.
    base_height : float, default=4.5
        Minimum figure height.
    horizontal : bool, default=False
        If True, increase height instead of width.

    Returns
    -------
    figsize : tuple[float, float]

    Complexity
    ----------
    Time: O(1)
    Space: O(1)
    """
    n_items = max(1, int(n_items))

    if horizontal:
        return (base_width, max(base_height, 0.38 * n_items + 1.8))

    return (max(base_width, 0.62 * n_items + 2.0), base_height)


def _style_axis(ax, title, xlabel, ylabel):
    """
    Apply consistent HD-style axis title and label formatting.
    """
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")


def _rotate_xticks(ax, rotation=45):
    """
    Rotate x-axis tick labels and align them for readability.
    """
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")


def _place_legend(ax, n_labels):
    """
    Place legends outside the axes when there are many plotted series.
    """
    if n_labels > 4:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        ax.legend(loc="best")


def _metric_ylim_if_probability(ax, arrays):
    """
    Use a probability-like y-axis when all plotted values are in [0, 1].
    """
    values = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays])
    values = values[np.isfinite(values)]

    if values.size > 0 and np.min(values) >= 0 and np.max(values) <= 1:
        lower = max(0.0, np.min(values) - 0.05)
        upper = min(1.05, max(1.0, np.max(values) + 0.05))
        ax.set_ylim(lower, upper)


def _finalize_plot(fig, save_path=None, show=True):
    """
    Save and/or show a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    save_path : str, optional
        File path where figure should be saved.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Complexity
    ----------
    Time: O(1), excluding rendering/saving cost
    Space: O(1)
    """
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ---------------- REQUIRED PLOTS ---------------- #

def plot_metric_over_time(metric_values, title="Metric over Time", ylabel="Metric",
                          xlabel="Chunk", save_path=None, show=True):
    """
    Plot a metric value across streaming chunks.

    Parameters
    ----------
    metric_values : array-like of shape (n_chunks,)
        Metric values logged over time.
    title : str, default="Metric over Time"
        Plot title.
    ylabel : str, default="Metric"
        Y-axis label.
    xlabel : str, default="Chunk"
        X-axis label.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If metric_values is not a non-empty 1D array.

    Complexity
    ----------
    Time: O(n_chunks)
    Space: O(n_chunks)
    """
    values = _validate_1d(metric_values, name="metric_values")
    chunks = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=_auto_figsize(len(values), base_width=7.0))
    ax.plot(chunks, values, marker="o", linewidth=2.0, markersize=5)
    _style_axis(ax, title, xlabel, ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _metric_ylim_if_probability(ax, [values])

    return _finalize_plot(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model Comparison", ylabel="Metric",
                   xlabel="Chunk", save_path=None, show=True):
    """
    Compare two model metric histories across streaming chunks.

    This required convenience function delegates to plot_model_comparison() so
    two-model and multi-model comparisons share the same plotting behaviour.
    """
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two names")

    return plot_model_comparison(
        {labels[0]: metric1, labels[1]: metric2},
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
        save_path=save_path,
        show=show
    )


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth",
                                     save_path=None, show=True):
    """
    Plot predicted labels against ground-truth labels on the latest chunk.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    title : str, default="Predictions vs Ground Truth"
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If inputs are invalid or shape-mismatched.

    Complexity
    ----------
    Time: O(n_samples)
    Space: O(n_samples)
    """
    y_true = _validate_1d(y_true, name="y_true")
    y_pred = _validate_1d(y_pred, name="y_pred")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    x_axis = np.arange(len(y_true))
    classes = np.unique(np.concatenate([y_true, y_pred]))
    mismatches = y_true != y_pred

    fig, ax = plt.subplots(figsize=_auto_figsize(len(y_true), base_width=8.5))

    ax.scatter(x_axis, y_true, marker="o", s=42, alpha=0.8, label="Ground Truth")
    ax.scatter(x_axis, y_pred, marker="x", s=52, alpha=0.9, label="Prediction")

    if np.any(mismatches):
        ax.scatter(
            x_axis[mismatches],
            y_pred[mismatches],
            marker="s",
            facecolors="none",
            s=90,
            linewidths=1.5,
            label="Mismatch"
        )

    _style_axis(ax, title, "Sample", "Class label")
    ax.set_yticks(classes)
    ax.set_ylim(np.min(classes) - 0.25, np.max(classes) + 0.25)
    ax.grid(True, alpha=0.3)
    _place_legend(ax, 3 if np.any(mismatches) else 2)

    return _finalize_plot(fig, save_path, show)


# ---------------- EXTRA STREAMING / MODEL PLOTS ---------------- #

def plot_model_comparison(metrics_by_model, title="Model Comparison",
                          ylabel="Metric", xlabel="Chunk",
                          save_path=None, show=True):
    """
    Compare multiple models over streaming chunks.

    Parameters
    ----------
    metrics_by_model : dict[str, array-like]
        Dictionary mapping model name to metric history.
    title : str, default="Model Comparison"
        Plot title.
    ylabel : str, default="Metric"
        Y-axis label.
    xlabel : str, default="Chunk"
        X-axis label.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If input dictionary is empty or metric lengths differ.

    Complexity
    ----------
    Time: O(n_models * n_chunks)
    Space: O(n_models * n_chunks)
    """
    if not isinstance(metrics_by_model, dict) or len(metrics_by_model) == 0:
        raise ValueError("metrics_by_model must be a non-empty dictionary")

    lengths = []
    validated = {}

    for name, values in metrics_by_model.items():
        arr = _validate_1d(values, name=f"metrics for {name}")
        validated[name] = arr
        lengths.append(len(arr))

    if len(set(lengths)) != 1:
        raise ValueError("all model metric histories must have the same length")

    chunks = np.arange(1, lengths[0] + 1)
    fig_width = max(8.0, 0.8 * lengths[0] + 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    for name, values in validated.items():
        ax.plot(chunks, values, marker="o", linewidth=2.0, markersize=5, label=name)

    _style_axis(ax, title, xlabel, ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(chunks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _metric_ylim_if_probability(ax, list(validated.values()))
    _place_legend(ax, len(validated))

    return _finalize_plot(fig, save_path, show)


def plot_class_distribution(y, classes=None, title="Class Distribution",
                            save_path=None, show=True):
    """
    Plot class counts for a label vector.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels.
    classes : array-like, optional
        Class order to display.
    title : str, default="Class Distribution"
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Complexity
    ----------
    Time: O(n + c)
    Space: O(c)
    """
    y = _validate_1d(y, name="y")

    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(classes)

    labels = [str(cls) for cls in classes]
    counts = np.array([np.sum(y == cls) for cls in classes])

    fig, ax = plt.subplots(figsize=_auto_figsize(len(labels), base_width=6.5))
    ax.bar(labels, counts, edgecolor="black", linewidth=0.8)
    _style_axis(ax, title, "Class", "Count")

    if max(len(label) for label in labels) > 8 or len(labels) > 5:
        _rotate_xticks(ax, rotation=35)

    return _finalize_plot(fig, save_path, show)


def plot_class_distribution_over_time(counts_by_chunk, classes,
                                      title="Class Distribution over Stream",
                                      xlabel="Chunk", ylabel="Count",
                                      save_path=None, show=True):
    """
    Plot class counts across streaming chunks.

    Parameters
    ----------
    counts_by_chunk : array-like of shape (n_chunks, n_classes)
        Class counts per chunk.
    classes : array-like of shape (n_classes,)
        Class labels.
    title : str, default="Class Distribution over Stream"
        Plot title.
    xlabel : str, default="Chunk"
        X-axis label.
    ylabel : str, default="Count"
        Y-axis label.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If shapes are inconsistent.

    Complexity
    ----------
    Time: O(n_chunks * n_classes)
    Space: O(n_chunks * n_classes)
    """
    counts = _validate_2d(counts_by_chunk, name="counts_by_chunk")
    classes = _validate_1d(classes, name="classes")

    if counts.shape[1] != len(classes):
        raise ValueError("counts_by_chunk column count must match number of classes")

    chunks = np.arange(1, counts.shape[0] + 1)

    fig, ax = plt.subplots(figsize=_auto_figsize(counts.shape[0], base_width=7.5))

    for idx, cls in enumerate(classes):
        ax.plot(chunks, counts[:, idx], marker="o", linewidth=2.0, markersize=5, label=str(cls))

    _style_axis(ax, title, xlabel, ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(chunks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _place_legend(ax, len(classes))

    return _finalize_plot(fig, save_path, show)


def plot_confusion_matrix(cm, classes=None, normalize=False,
                          title="Confusion Matrix", save_path=None, show=True):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        Confusion matrix.
    classes : array-like, optional
        Class labels.
    normalize : bool, default=False
        Whether to normalize rows.
    title : str, default="Confusion Matrix"
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If cm is not square or class labels mismatch.

    Complexity
    ----------
    Time: O(c^2)
    Space: O(c^2)
    """
    cm = _validate_2d(cm, name="cm")

    if cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square matrix")

    if classes is None:
        classes = np.arange(cm.shape[0])
    else:
        classes = np.asarray(classes)

    if len(classes) != cm.shape[0]:
        raise ValueError("number of classes must match confusion matrix size")

    cm_display = cm.astype(float)

    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm_display / row_sums

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(cm_display, aspect="auto", cmap="Blues")
    fig.colorbar(image, ax=ax)

    _style_axis(ax, title, "Predicted", "True")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels([str(cls) for cls in classes])
    ax.set_yticklabels([str(cls) for cls in classes])

    if max(len(str(cls)) for cls in classes) > 6:
        _rotate_xticks(ax, rotation=35)

    max_value = np.max(cm_display) if cm_display.size > 0 else 0.0
    threshold = max_value / 2.0

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            text_color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold"
            )

    return _finalize_plot(fig, save_path, show)


def plot_missing_values(X, feature_names=None, title="Missing Value Rate",
                        missing_tokens=("", "nan", "na", "n/a", "?"),
                        save_path=None, show=True):
    """
    Plot missing-value rate per feature.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    feature_names : array-like, optional
        Names of features.
    title : str, default="Missing Value Rate"
        Plot title.
    missing_tokens : tuple of str, default=("", "nan", "na", "n/a", "?")
        String tokens treated as missing. "None" is intentionally not included
        by default because it can be a valid class/category label.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If feature_names length does not match n_features.

    Complexity
    ----------
    Time: O(n_samples * n_features)
    Space: O(n_samples * n_features)
    """
    X = _validate_2d(X, name="X", allow_empty=False)
    X_obj = np.asarray(X, dtype=object)

    # Detect missing tokens from string representation. This correctly handles
    # numeric NaN values because str(np.nan).lower() is "nan", while ordinary
    # categorical values such as "Male" or "None" are not treated as missing.
    X_str = np.char.lower(np.char.strip(X_obj.astype(str)))
    tokens = np.asarray([str(token).lower().strip() for token in missing_tokens])
    missing_mask = np.isin(X_str, tokens)
    rates = np.mean(missing_mask, axis=0)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_obj.shape[1])]
    else:
        feature_names = list(feature_names)

    if len(feature_names) != X_obj.shape[1]:
        raise ValueError("feature_names length must match number of features")

    labels = [str(name) for name in feature_names]
    use_horizontal = len(labels) > 6 or max(len(label) for label in labels) > 12

    fig, ax = plt.subplots(figsize=_auto_figsize(len(labels), base_width=8.0, horizontal=use_horizontal))

    if use_horizontal:
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, rates, edgecolor="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        _style_axis(ax, title, "Missing Rate", "Feature")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", alpha=0.25)
    else:
        ax.bar(labels, rates, edgecolor="black", linewidth=0.8)
        _style_axis(ax, title, "Feature", "Missing Rate")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)

    return _finalize_plot(fig, save_path, show)


def plot_feature_histogram(X, feature_index, bins=20, title=None,
                           save_path=None, show=True):
    """
    Plot histogram for one numeric feature.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    feature_index : int
        Column index to plot.
    bins : int, default=20
        Number of histogram bins.
    title : str, optional
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If feature index or bins are invalid.

    Complexity
    ----------
    Time: O(n_samples)
    Space: O(n_samples)
    """
    X = _validate_2d(X, name="X")

    if not isinstance(feature_index, int) or feature_index < 0 or feature_index >= X.shape[1]:
        raise ValueError("feature_index is out of bounds")

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")

    feature = np.asarray(X[:, feature_index], dtype=float)
    feature = feature[~np.isnan(feature)]

    if feature.size == 0:
        raise ValueError("selected feature contains no valid numeric values")

    if title is None:
        title = f"Feature {feature_index} Histogram"

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(feature, bins=bins, edgecolor="black", linewidth=0.8)
    _style_axis(ax, title, f"Feature {feature_index}", "Frequency")
    ax.grid(True, axis="y", alpha=0.25)

    return _finalize_plot(fig, save_path, show)


def plot_roc_curve(fpr, tpr, title="ROC Curve", save_path=None, show=True):
    """
    Plot a single ROC curve from false-positive and true-positive rates.

    This can be used for binary ROC-AUC or for one class in a one-vs-rest
    multiclass ROC-AUC setup. For plotting all multiclass one-vs-rest curves,
    use plot_multiclass_roc_curves().

    Parameters
    ----------
    fpr : array-like of shape (n_points,)
        False-positive rates.
    tpr : array-like of shape (n_points,)
        True-positive rates.
    title : str, default="ROC Curve"
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If inputs are shape-mismatched or outside [0, 1].

    Complexity
    ----------
    Time: O(n_points)
    Space: O(n_points)
    """
    fpr = _validate_1d(fpr, name="fpr")
    tpr = _validate_1d(tpr, name="tpr")

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if np.any((fpr < 0) | (fpr > 1)) or np.any((tpr < 0) | (tpr > 1)):
        raise ValueError("fpr and tpr values must be between 0 and 1")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(fpr, tpr, marker="o", linewidth=2.0, markersize=5, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    _style_axis(ax, title, "False Positive Rate", "True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    return _finalize_plot(fig, save_path, show)

def _binary_roc_curve_for_plot(y_true_binary, y_scores):
    """
    Compute a binary ROC curve for plotting.
    """
    y_true_binary = _validate_1d(y_true_binary, name="y_true_binary")
    y_scores = _validate_1d(y_scores, name="y_scores")

    if y_true_binary.shape != y_scores.shape:
        raise ValueError("y_true_binary and y_scores must have the same shape")

    unique = np.unique(y_true_binary)

    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("y_true_binary must contain only 0 and 1")

    if len(unique) < 2:
        raise ValueError("ROC curve requires both positive and negative samples")

    order = np.argsort(-y_scores)
    y_true_sorted = y_true_binary[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    tp_total = tp[-1]
    fp_total = fp[-1]

    if tp_total == 0 or fp_total == 0:
        raise ValueError("ROC curve requires both positive and negative samples")

    tpr = np.concatenate([[0.0], tp / tp_total, [1.0]])
    fpr = np.concatenate([[0.0], fp / fp_total, [1.0]])

    return fpr, tpr


def _auc_for_plot(fpr, tpr):
    """
    Compute AUC using the trapezoidal rule.
    """
    return float(np.trapezoid(tpr, fpr))


def plot_multiclass_roc_curves(y_true, y_scores, classes=None,
                               title="Multiclass ROC Curve",
                               save_path=None, show=True):
    """
    Plot one-vs-rest ROC curves for multiclass classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels. Labels must correspond to the class order used by
        the columns of y_scores.
    y_scores : array-like of shape (n_samples, n_classes)
        Predicted class probabilities or confidence scores.
    classes : array-like, optional
        Class labels corresponding to y_scores columns. If None, class labels
        are assumed to be 0, 1, ..., n_classes - 1.
    title : str, default="Multiclass ROC Curve"
        Plot title.
    save_path : str, optional
        File path for saving the plot.
    show : bool, default=True
        Whether to display the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If inputs have invalid shapes or class labels are inconsistent.

    Complexity
    ----------
    Time: O(n_classes * n_samples log n_samples)
    Space: O(n_samples)
    """
    y_true = _validate_1d(y_true, name="y_true")
    y_scores = _validate_2d(y_scores, name="y_scores")

    if y_scores.shape[0] != len(y_true):
        raise ValueError("y_true and y_scores must contain the same number of samples")

    n_classes = y_scores.shape[1]

    if classes is None:
        classes = np.arange(n_classes)
    else:
        classes = _validate_1d(classes, name="classes")

    if len(classes) != n_classes:
        raise ValueError("number of classes must match y_scores columns")

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    plotted = 0

    for idx, cls in enumerate(classes):
        y_true_binary = (y_true == cls).astype(int)

        # Skip classes that cannot form a valid one-vs-rest ROC curve.
        if np.sum(y_true_binary) == 0 or np.sum(y_true_binary) == len(y_true_binary):
            continue

        fpr, tpr = _binary_roc_curve_for_plot(y_true_binary, y_scores[:, idx])
        class_auc = _auc_for_plot(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            label=f"Class {cls} (AUC={class_auc:.3f})"
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No valid one-vs-rest ROC curves could be computed")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    _style_axis(ax, title, "False Positive Rate", "True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    _place_legend(ax, plotted + 1)

    return _finalize_plot(fig, save_path, show)