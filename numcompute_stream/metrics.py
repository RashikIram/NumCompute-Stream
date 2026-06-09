import numpy as np


# ---------------- VALIDATION HELPERS ---------------- #

def _validate_inputs(y_true, y_pred=None):
    """
    Validate metric inputs.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels or target values.
    y_pred : array-like of shape (n_samples,), optional
        Predicted labels or target values.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray or None

    Raises
    ------
    ValueError
        If inputs are None, non-numeric, or shape-mismatched.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    if y_true is None:
        raise ValueError("y_true cannot be None")

    y_true = np.asarray(y_true)

    if not np.issubdtype(y_true.dtype, np.number):
        raise ValueError("y_true must be numeric")

    if y_pred is not None:
        y_pred = np.asarray(y_pred)

        if not np.issubdtype(y_pred.dtype, np.number):
            raise ValueError("y_pred must be numeric")

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

    return y_true, y_pred


def _safe_divide(numerator, denominator, zero_value=0.0):
    """
    Safely divide two values.

    Parameters
    ----------
    numerator : float
        Numerator value.
    denominator : float
        Denominator value.
    zero_value : float, default=0.0
        Value returned when denominator is zero.

    Returns
    -------
    float

    Complexity
    ----------
    Time: O(1)
    Space: O(1)
    """
    if denominator == 0:
        return zero_value

    return numerator / denominator


def _get_classes(y_true, y_pred=None, classes=None):
    """
    Return sorted class labels.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,), optional
        Predicted labels.
    classes : array-like, optional
        Fixed class order.

    Returns
    -------
    classes : np.ndarray
        Sorted class labels.

    Complexity
    ----------
    Time: O(n log n)
    Space: O(n)
    """
    if classes is not None:
        return np.asarray(classes)

    if y_pred is None:
        return np.unique(y_true)

    return np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))


def _per_class_from_confusion_matrix(cm):
    """
    Compute per-class precision, recall, F1, and support from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix. Rows are true classes and columns are predicted classes.

    Returns
    -------
    precision_scores : np.ndarray of shape (n_classes,)
    recall_scores : np.ndarray of shape (n_classes,)
    f1_scores : np.ndarray of shape (n_classes,)
    support : np.ndarray of shape (n_classes,)

    Complexity
    ----------
    Time: O(c)
    Space: O(c)
    """
    cm = np.asarray(cm, dtype=float)

    tp = np.diag(cm)
    predicted = np.sum(cm, axis=0)
    actual = np.sum(cm, axis=1)

    precision_scores = np.divide(
        tp,
        predicted,
        out=np.zeros_like(tp, dtype=float),
        where=predicted != 0
    )

    recall_scores = np.divide(
        tp,
        actual,
        out=np.zeros_like(tp, dtype=float),
        where=actual != 0
    )

    denom = precision_scores + recall_scores
    f1_scores = np.divide(
        2 * precision_scores * recall_scores,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom != 0
    )

    return precision_scores, recall_scores, f1_scores, actual


def _mcc_from_confusion_matrix(cm):
    """
    Compute multiclass Matthews Correlation Coefficient from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    Returns
    -------
    float
        MCC value. Returns 0.0 when denominator is zero.

    Notes
    -----
    Uses the multiclass MCC formulation based on confusion matrix sums.

    Complexity
    ----------
    Time: O(c^2)
    Space: O(c)
    """
    cm = np.asarray(cm, dtype=float)

    t_sum = np.sum(cm, axis=1)
    p_sum = np.sum(cm, axis=0)
    n_correct = np.trace(cm)
    n_samples = np.sum(cm)

    numerator = (n_correct * n_samples) - np.dot(t_sum, p_sum)
    denominator = np.sqrt(
        (n_samples ** 2 - np.sum(p_sum ** 2)) *
        (n_samples ** 2 - np.sum(t_sum ** 2))
    )

    return _safe_divide(numerator, denominator)


def _kappa_from_confusion_matrix(cm):
    """
    Compute Cohen's Kappa from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    Returns
    -------
    float
        Cohen's Kappa value. Returns 0.0 when expected agreement is 1.

    Complexity
    ----------
    Time: O(c)
    Space: O(c)
    """
    cm = np.asarray(cm, dtype=float)
    total = np.sum(cm)

    if total == 0:
        return 0.0

    observed = np.trace(cm) / total
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    expected = np.sum(row_sum * col_sum) / (total ** 2)

    return _safe_divide(observed - expected, 1 - expected)


# ---------------- CLASSIFICATION METRICS ---------------- #

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Fraction of correctly classified samples.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, classes=None):
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    classes : array-like, optional
        Class label order.

    Returns
    -------
    cm : np.ndarray of shape (n_classes, n_classes)
        Rows represent true labels, columns represent predicted labels.

    Complexity
    ----------
    Time: O(n * c)
    Space: O(c^2)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    classes = _get_classes(y_true, y_pred, classes)

    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))

    return cm


def precision(y_true, y_pred):
    """
    Compute binary precision for positive class 1.

    Raises
    ------
    ValueError
        If there are no positive predictions.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        raise ValueError("No positive predictions; precision undefined")

    return tp / (tp + fp)


def recall(y_true, y_pred):
    """
    Compute binary recall/sensitivity for positive class 1.

    Raises
    ------
    ValueError
        If there are no actual positives.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        raise ValueError("No actual positives; recall undefined")

    return tp / (tp + fn)


def specificity(y_true, y_pred):
    """
    Compute binary specificity for negative class 0.

    Raises
    ------
    ValueError
        If there are no actual negatives.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tn + fp == 0:
        raise ValueError("No actual negatives; specificity undefined")

    return tn / (tn + fp)


def f1(y_true, y_pred):
    """
    Compute binary F1 score.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        raise ValueError("Precision and Recall are zero; F1 undefined")

    return 2 * p * r / (p + r)


def balanced_accuracy(y_true, y_pred):
    """
    Compute binary balanced accuracy.

    Balanced accuracy = (sensitivity + specificity) / 2.
    """
    return (recall(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def precision_recall_f1_per_class(y_true, y_pred, classes=None):
    """
    Compute precision, recall, F1, and support for each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    classes : array-like, optional
        Class label order.

    Returns
    -------
    precision_scores : np.ndarray of shape (n_classes,)
    recall_scores : np.ndarray of shape (n_classes,)
    f1_scores : np.ndarray of shape (n_classes,)
    supports : np.ndarray of shape (n_classes,)

    Complexity
    ----------
    Time: O(n * c)
    Space: O(c)
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, classes=classes)

    return _per_class_from_confusion_matrix(cm)


def macro_precision(y_true, y_pred):
    p, _, _, _ = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean(p)


def macro_recall(y_true, y_pred):
    _, r, _, _ = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean(r)


def macro_f1(y_true, y_pred):
    _, _, f, _ = precision_recall_f1_per_class(y_true, y_pred)
    return np.mean(f)


def weighted_f1(y_true, y_pred):
    """
    Compute weighted F1 using class support as weights.
    """
    _, _, f, support = precision_recall_f1_per_class(y_true, y_pred)

    if np.sum(support) == 0:
        return 0.0

    return np.sum(f * support) / np.sum(support)


def micro_precision(y_true, y_pred):
    """
    Compute micro precision.

    For single-label classification, micro precision equals accuracy.
    """
    return accuracy(y_true, y_pred)


def micro_recall(y_true, y_pred):
    """
    Compute micro recall.

    For single-label classification, micro recall equals accuracy.
    """
    return accuracy(y_true, y_pred)


def micro_f1(y_true, y_pred):
    """
    Compute micro F1.

    For single-label classification, micro F1 equals accuracy.
    """
    return accuracy(y_true, y_pred)


def matthews_corrcoef(y_true, y_pred):
    """
    Compute Matthews Correlation Coefficient.

    Supports binary and multiclass labels through the confusion matrix.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return _mcc_from_confusion_matrix(cm)


def cohen_kappa(y_true, y_pred):
    """
    Compute Cohen's Kappa agreement score.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return _kappa_from_confusion_matrix(cm)


def top_k_accuracy(y_true, y_scores, k=1):
    """
    Compute top-k accuracy from class score/probability matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels as integer class indices.
    y_scores : array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities.
    k : int, default=1
        Number of top classes considered correct.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs have invalid shapes or k is out of range.

    Complexity
    ----------
    Time: O(n * c log c)
    Space: O(n * k)
    """
    y_true, _ = _validate_inputs(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    if y_scores.ndim != 2:
        raise ValueError("y_scores must be a 2D array")

    if y_scores.shape[0] != len(y_true):
        raise ValueError("y_true and y_scores must have the same number of samples")

    if k <= 0 or k > y_scores.shape[1]:
        raise ValueError("k must be between 1 and number of classes")

    top_k = np.argsort(y_scores, axis=1)[:, -k:]
    correct = np.any(top_k == y_true[:, None], axis=1)

    return np.mean(correct)


def log_loss(y_true, y_proba, eps=1e-15):
    """
    Compute multiclass log loss.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class indices.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities.
    eps : float, default=1e-15
        Clipping value for numerical stability.

    Returns
    -------
    float

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    y_true, _ = _validate_inputs(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array")

    if y_proba.shape[0] != len(y_true):
        raise ValueError("y_true and y_proba must have the same number of samples")

    if np.any(y_true < 0) or np.any(y_true >= y_proba.shape[1]):
        raise ValueError("y_true contains class index outside probability columns")

    y_proba = np.clip(y_proba, eps, 1 - eps)
    row_sums = np.sum(y_proba, axis=1, keepdims=True)
    y_proba = y_proba / row_sums

    return -np.mean(np.log(y_proba[np.arange(len(y_true)), y_true]))


def brier_score(y_true, y_proba):
    """
    Compute multiclass Brier score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class indices.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    float

    Complexity
    ----------
    Time: O(n * c)
    Space: O(n * c)
    """
    y_true, _ = _validate_inputs(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2:
        raise ValueError("y_proba must be a 2D array")

    if y_proba.shape[0] != len(y_true):
        raise ValueError("y_true and y_proba must have the same number of samples")

    if np.any(y_true < 0) or np.any(y_true >= y_proba.shape[1]):
        raise ValueError("y_true contains class index outside probability columns")

    one_hot = np.zeros_like(y_proba, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0

    return np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))


# ---------------- REGRESSION METRICS ---------------- #

def mse(y_true, y_pred):
    """
    Compute mean squared error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Compute root mean squared error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mad(y_true, y_pred):
    """
    Compute mean absolute deviation/error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Compute mean absolute percentage error.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    non_zero = y_true != 0

    if not np.any(non_zero):
        raise ValueError("All y_true values are zero; MAPE undefined")

    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


# ---------------- ROC / AUC ---------------- #

def roc_curve(y_true, y_scores):
    """
    Compute binary ROC curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels 0/1.
    y_scores : array-like of shape (n_samples,)
        Probability scores, not labels.

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    """
    if y_true is None or y_scores is None:
        raise ValueError("Inputs cannot be None")

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")

    unique = np.unique(y_true)

    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("y_true must contain only binary labels (0 and 1)")

    desc = np.argsort(-y_scores)
    y_true = y_true[desc]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    tp_total = tp[-1]
    fp_total = fp[-1]

    if tp_total == 0 or fp_total == 0:
        raise ValueError("ROC undefined: need both positive and negative samples")

    tpr = np.concatenate([[0.0], tp / tp_total, [1.0]])
    fpr = np.concatenate([[0.0], fp / fp_total, [1.0]])

    return fpr, tpr


def auc(fpr, tpr):
    """
    Compute area under curve using trapezoidal rule.
    """
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if len(fpr) < 2:
        raise ValueError("At least two points required to compute AUC")

    return np.trapz(tpr, fpr)




class StreamingROCAUC:
    """
    Streaming binary ROC-AUC accumulator.

    Parameters
    ----------
    max_buffer : int or None, default=None
        Maximum number of recent score/label pairs to store. If None, all
        observed pairs are kept and cumulative ROC-AUC is returned.
    positive_label : int or float, default=1
        Label treated as the positive class.

    Methods
    -------
    update(y_true_chunk, y_score_chunk)
    result()
    reset()

    Notes
    -----
    ROC-AUC is ranking-based, so it cannot be represented by only TP/FP counts.
    This class buffers score/label pairs and recomputes the curve at result()
    time. With max_buffer set, it becomes a rolling ROC-AUC metric.

    Complexity
    ----------
    update: O(n)
    result: O(w log w), where w is the buffer size
    Space: O(w)
    """

    def __init__(self, max_buffer=None, positive_label=1):
        if max_buffer is not None and max_buffer <= 0:
            raise ValueError("max_buffer must be positive or None")

        self.max_buffer = max_buffer
        self.positive_label = positive_label
        self.reset()

    def reset(self):
        self.y_true_buffer = []
        self.y_score_buffer = []
        return self

    def update(self, y_true_chunk, y_score_chunk):
        if y_true_chunk is None or y_score_chunk is None:
            raise ValueError("Inputs cannot be None")

        y_true_chunk = np.asarray(y_true_chunk)
        y_score_chunk = np.asarray(y_score_chunk, dtype=float)

        if y_true_chunk.ndim != 1 or y_score_chunk.ndim != 1:
            raise ValueError("y_true_chunk and y_score_chunk must be 1D arrays")
        if y_true_chunk.shape != y_score_chunk.shape:
            raise ValueError("y_true_chunk and y_score_chunk must have the same shape")

        binary_labels = (y_true_chunk == self.positive_label).astype(int)

        self.y_true_buffer.extend(binary_labels.tolist())
        self.y_score_buffer.extend(y_score_chunk.tolist())

        if self.max_buffer is not None and len(self.y_true_buffer) > self.max_buffer:
            self.y_true_buffer = self.y_true_buffer[-self.max_buffer:]
            self.y_score_buffer = self.y_score_buffer[-self.max_buffer:]

        return self

    def result(self):
        if len(self.y_true_buffer) == 0:
            return 0.0

        y_true = np.asarray(self.y_true_buffer)
        y_scores = np.asarray(self.y_score_buffer, dtype=float)

        if len(np.unique(y_true)) < 2:
            return 0.0

        fpr, tpr = roc_curve(y_true, y_scores)
        return float(auc(fpr, tpr))


class RollingROCAUC(StreamingROCAUC):
    """
    Rolling-window binary ROC-AUC accumulator.

    Parameters
    ----------
    window_size : int, default=100
        Maximum number of recent score/label pairs stored.
    positive_label : int or float, default=1
        Label treated as the positive class.
    """

    def __init__(self, window_size=100, positive_label=1):
        super().__init__(max_buffer=window_size, positive_label=positive_label)


# ---------------- STREAMING METRICS ---------------- #

class StreamingConfusionMatrix:
    """
    Streaming confusion matrix accumulator.

    Parameters
    ----------
    classes : array-like, optional
        Fixed class labels. If None, classes are inferred and expanded over time.

    Methods
    -------
    update(y_true_chunk, y_pred_chunk)
    result()
    reset()

    Complexity
    ----------
    update: O(n * c)
    result: O(1)
    """

    def __init__(self, classes=None):
        self.initial_classes = None if classes is None else np.asarray(classes)
        self.classes = None
        self.matrix = None
        self.reset()

    def reset(self):
        if self.initial_classes is None:
            self.classes = np.array([], dtype=int)
            self.matrix = np.zeros((0, 0), dtype=int)
        else:
            self.classes = self.initial_classes.copy()
            self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)

        return self

    def _ensure_classes(self, y_true, y_pred):
        new_classes = np.unique(np.concatenate([y_true, y_pred]))
        all_classes = np.unique(np.concatenate([self.classes, new_classes]))

        if len(all_classes) == len(self.classes) and np.array_equal(all_classes, self.classes):
            return

        new_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)

        for i, old_true in enumerate(self.classes):
            new_i = np.where(all_classes == old_true)[0][0]
            for j, old_pred in enumerate(self.classes):
                new_j = np.where(all_classes == old_pred)[0][0]
                new_matrix[new_i, new_j] = self.matrix[i, j]

        self.classes = all_classes
        self.matrix = new_matrix

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)
        self._ensure_classes(y_true_chunk, y_pred_chunk)

        chunk_cm = confusion_matrix(y_true_chunk, y_pred_chunk, classes=self.classes)
        self.matrix += chunk_cm

        return self

    def result(self):
        return self.matrix.copy()


class StreamingAccuracy:
    """
    Streaming accuracy accumulator.

    Methods
    -------
    update(y_true_chunk, y_pred_chunk)
    result()
    reset()

    Complexity
    ----------
    update: O(n)
    result: O(1)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.correct += int(np.sum(y_true_chunk == y_pred_chunk))
        self.total += len(y_true_chunk)

        return self

    def result(self):
        return _safe_divide(self.correct, self.total)


class StreamingPrecision:
    """
    Streaming binary precision accumulator for positive class 1.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.tp += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 1)))
        self.fp += int(np.sum((y_true_chunk == 0) & (y_pred_chunk == 1)))

        return self

    def result(self):
        return _safe_divide(self.tp, self.tp + self.fp)


class StreamingRecall:
    """
    Streaming binary recall accumulator for positive class 1.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fn = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.tp += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 1)))
        self.fn += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 0)))

        return self

    def result(self):
        return _safe_divide(self.tp, self.tp + self.fn)


class StreamingF1:
    """
    Streaming binary F1 accumulator.
    """

    def __init__(self):
        self.precision_metric = StreamingPrecision()
        self.recall_metric = StreamingRecall()

    def reset(self):
        self.precision_metric.reset()
        self.recall_metric.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.precision_metric.update(y_true_chunk, y_pred_chunk)
        self.recall_metric.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        p = self.precision_metric.result()
        r = self.recall_metric.result()
        return _safe_divide(2 * p * r, p + r)


class StreamingBalancedAccuracy:
    """
    Streaming binary balanced accuracy accumulator.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.tp += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 1)))
        self.tn += int(np.sum((y_true_chunk == 0) & (y_pred_chunk == 0)))
        self.fp += int(np.sum((y_true_chunk == 0) & (y_pred_chunk == 1)))
        self.fn += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 0)))

        return self

    def result(self):
        sensitivity = _safe_divide(self.tp, self.tp + self.fn)
        specificity_score = _safe_divide(self.tn, self.tn + self.fp)

        return (sensitivity + specificity_score) / 2


class StreamingMacroF1:
    """
    Streaming macro F1 accumulator using cumulative confusion matrix.

    Parameters
    ----------
    classes : array-like, optional
        Fixed class labels. If None, classes are inferred over time.

    Complexity
    ----------
    update: O(n * c)
    result: O(c)
    """

    def __init__(self, classes=None):
        self.cm = StreamingConfusionMatrix(classes=classes)

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        matrix = self.cm.result()

        if matrix.size == 0:
            return 0.0

        _, _, f1_scores, _ = _per_class_from_confusion_matrix(matrix)

        return float(np.mean(f1_scores))


class StreamingWeightedF1:
    """
    Streaming weighted F1 accumulator using cumulative confusion matrix.

    Parameters
    ----------
    classes : array-like, optional
        Fixed class labels. If None, classes are inferred over time.

    Complexity
    ----------
    update: O(n * c)
    result: O(c)
    """

    def __init__(self, classes=None):
        self.cm = StreamingConfusionMatrix(classes=classes)

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        matrix = self.cm.result()

        if matrix.size == 0:
            return 0.0

        _, _, f1_scores, support = _per_class_from_confusion_matrix(matrix)

        if np.sum(support) == 0:
            return 0.0

        return float(np.sum(f1_scores * support) / np.sum(support))


class StreamingMCC:
    """
    Streaming Matthews Correlation Coefficient accumulator.

    Parameters
    ----------
    classes : array-like, optional
        Fixed class labels. If None, classes are inferred over time.

    Notes
    -----
    MCC is computed from the cumulative confusion matrix, so predictions do not
    need to be stored.

    Complexity
    ----------
    update: O(n * c)
    result: O(c^2)
    """

    def __init__(self, classes=None):
        self.cm = StreamingConfusionMatrix(classes=classes)

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        matrix = self.cm.result()

        if matrix.size == 0:
            return 0.0

        return _mcc_from_confusion_matrix(matrix)


class StreamingCohenKappa:
    """
    Streaming Cohen's Kappa accumulator.

    Parameters
    ----------
    classes : array-like, optional
        Fixed class labels. If None, classes are inferred over time.

    Notes
    -----
    Kappa is computed from the cumulative confusion matrix, so predictions do not
    need to be stored.

    Complexity
    ----------
    update: O(n * c)
    result: O(c)
    """

    def __init__(self, classes=None):
        self.cm = StreamingConfusionMatrix(classes=classes)

    def reset(self):
        self.cm.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.cm.update(y_true_chunk, y_pred_chunk)
        return self

    def result(self):
        matrix = self.cm.result()

        if matrix.size == 0:
            return 0.0

        return _kappa_from_confusion_matrix(matrix)


class RollingAccuracy:
    """
    Rolling-window accuracy.

    Parameters
    ----------
    window_size : int, default=100
        Maximum number of recent samples stored.

    Complexity
    ----------
    update: O(n)
    result: O(w)
    """

    def __init__(self, window_size=100):
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.window_size = window_size
        self.reset()

    def reset(self):
        self.y_true_window = []
        self.y_pred_window = []
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.y_true_window.extend(y_true_chunk.tolist())
        self.y_pred_window.extend(y_pred_chunk.tolist())

        if len(self.y_true_window) > self.window_size:
            self.y_true_window = self.y_true_window[-self.window_size:]
            self.y_pred_window = self.y_pred_window[-self.window_size:]

        return self

    def result(self):
        if len(self.y_true_window) == 0:
            return 0.0

        return accuracy(np.asarray(self.y_true_window), np.asarray(self.y_pred_window))


class RollingF1:
    """
    Rolling-window binary F1 score.

    Parameters
    ----------
    window_size : int, default=100
        Maximum number of recent samples stored.

    Complexity
    ----------
    update: O(n)
    result: O(w)
    """

    def __init__(self, window_size=100):
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.window_size = window_size
        self.reset()

    def reset(self):
        self.y_true_window = []
        self.y_pred_window = []
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)

        self.y_true_window.extend(y_true_chunk.tolist())
        self.y_pred_window.extend(y_pred_chunk.tolist())

        if len(self.y_true_window) > self.window_size:
            self.y_true_window = self.y_true_window[-self.window_size:]
            self.y_pred_window = self.y_pred_window[-self.window_size:]

        return self

    def result(self):
        if len(self.y_true_window) == 0:
            return 0.0

        y_true_arr = np.asarray(self.y_true_window)
        y_pred_arr = np.asarray(self.y_pred_window)

        p_metric = StreamingPrecision().update(y_true_arr, y_pred_arr)
        r_metric = StreamingRecall().update(y_true_arr, y_pred_arr)

        p_val = p_metric.result()
        r_val = r_metric.result()

        return _safe_divide(2 * p_val * r_val, p_val + r_val)
