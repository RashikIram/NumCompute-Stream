import time
import tracemalloc
import numpy as np

from numcompute_stream.metrics import accuracy, confusion_matrix


class StreamTrainer:
    """
    Manage chunk-wise streaming training, scoring, and logging.

    StreamTrainer connects a streaming-compatible model or pipeline with
    per-chunk metric tracking. The wrapped object must implement
    ``partial_fit(X_chunk, y_chunk)`` for training and ``predict(X_chunk)``
    for scoring.

    Parameters
    ----------
    model : object
        Streaming-compatible estimator or pipeline. Must implement
        partial_fit() and predict().
    classes : array-like, optional
        Global class labels. Passed to partial_fit() when supported. This is
        useful when early chunks do not contain every class.
    track_memory : bool, default=True
        If True, uses tracemalloc to estimate Python-level peak memory during
        each fit_chunk() call.

    Attributes
    ----------
    logs : list of dict
        Per-chunk training and scoring records.
    n_samples_seen : int
        Total number of samples processed through fit_chunk().
    total_correct : int
        Cumulative number of correct predictions from scored chunks.
    total_scored : int
        Cumulative number of scored samples.

    Complexity
    ----------
    fit_chunk: depends on wrapped model; logging overhead is O(n_samples)
    score_chunk: O(n_samples + prediction cost)
    Space: O(n_logs)
    """

    def __init__(self, model, classes=None, track_memory=True):
        if model is None:
            raise ValueError("model cannot be None")

        if not hasattr(model, "partial_fit"):
            raise ValueError("model must implement partial_fit")

        if not hasattr(model, "predict"):
            raise ValueError("model must implement predict")

        self.model = model
        self.classes = None if classes is None else np.asarray(classes)
        self.track_memory = track_memory

        self.logs = []
        self.n_samples_seen = 0
        self.total_correct = 0
        self.total_scored = 0
        self.chunk_index = 0
        self._has_fitted = self._infer_model_is_fitted()

    def _infer_model_is_fitted(self):
        """
        Best-effort check for models that were already fitted before being passed
        into StreamTrainer.
        """
        for attr in ("n_samples_seen_", "n_samples_seen"):
            value = getattr(self.model, attr, 0)
            try:
                if np.asarray(value).sum() > 0:
                    return True
            except Exception:
                pass

        for attr in ("classes_", "root_", "estimators_"):
            value = getattr(self.model, attr, None)

            if value is None:
                continue

            if isinstance(value, (list, tuple)) and len(value) == 0:
                continue

            return True

        return False

    def _validate_chunk(self, X, y=None):
        """
        Validate streaming chunk inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature chunk.
        y : array-like of shape (n_samples,), optional
            Target chunk.

        Returns
        -------
        X : ndarray
        y : ndarray or None

        Raises
        ------
        ValueError
            If X is not 2D, y is not 1D, or sample counts mismatch.

        Complexity
        ----------
        Time: O(n)
        Space: O(n)
        """
        if X is None:
            raise ValueError("X cannot be None")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if y is None:
            return X, None

        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if X.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples")

        return X, y

    def _call_partial_fit(self, X, y):
        """
        Call model.partial_fit(), passing classes when supported.
        """
        if self.classes is None:
            return self.model.partial_fit(X, y)

        try:
            return self.model.partial_fit(X, y, classes=self.classes)
        except TypeError:
            # Some custom pipelines/models may not expose a classes argument.
            return self.model.partial_fit(X, y)

    def fit_chunk(self, X_chunk, y_chunk):
        """
        Train the wrapped model on one incoming chunk.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Feature chunk.
        y_chunk : array-like of shape (n_samples,)
            Target chunk.

        Returns
        -------
        self : StreamTrainer

        Raises
        ------
        ValueError
            If chunk shapes are invalid.

        Complexity
        ----------
        Time: depends on wrapped model
        Space: depends on wrapped model
        """
        X_chunk, y_chunk = self._validate_chunk(X_chunk, y_chunk)

        start = time.perf_counter()

        if self.track_memory:
            tracemalloc.start()
            self._call_partial_fit(X_chunk, y_chunk)
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            self._call_partial_fit(X_chunk, y_chunk)
            current_memory = 0
            peak_memory = 0

        self._has_fitted = True

        fit_time = time.perf_counter() - start

        self.n_samples_seen += len(y_chunk)
        self.chunk_index += 1

        self.logs.append({
            "chunk": self.chunk_index,
            "event": "fit",
            "n_samples": len(y_chunk),
            "n_samples_seen": self.n_samples_seen,
            "fit_time": fit_time,
            "current_memory_mb": current_memory / (1024 ** 2),
            "peak_memory_mb": peak_memory / (1024 ** 2),
        })

        return self

    def score_chunk(self, X_chunk, y_chunk, chunk_id=None):
        """
        Score the wrapped model on one incoming chunk.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Feature chunk.
        y_chunk : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        metrics : dict
            Dictionary containing chunk accuracy, cumulative accuracy,
            confusion matrix, prediction time, and sample counts.

        Raises
        ------
        ValueError
            If chunk shapes are invalid.

        Complexity
        ----------
        Time: O(n_samples + prediction cost)
        Space: O(n_classes^2)
        """
        X_chunk, y_chunk = self._validate_chunk(X_chunk, y_chunk)

        if not self._has_fitted:
            raise ValueError(
                "Cannot score before the model has been fitted. "
                "Call fit_chunk() first, or use fit_score_chunk(score_before_fit=True), "
                "which skips the first prequential score automatically."
            )

        start = time.perf_counter()
        y_pred = self.model.predict(X_chunk)
        score_time = time.perf_counter() - start

        y_pred = np.asarray(y_pred)

        if y_pred.shape != y_chunk.shape:
            raise ValueError("Predictions must have the same shape as y_chunk")

        chunk_accuracy = accuracy(y_chunk, y_pred)
        correct = int(np.sum(y_chunk == y_pred))

        self.total_correct += correct
        self.total_scored += len(y_chunk)

        cumulative_accuracy = self.total_correct / self.total_scored if self.total_scored > 0 else 0.0

        if self.classes is None:
            cm = confusion_matrix(y_chunk, y_pred)
        else:
            cm = confusion_matrix(y_chunk, y_pred, classes=self.classes)

        metrics = {
            "chunk": self.chunk_index if chunk_id is None else chunk_id,
            "event": "score",
            "n_samples": len(y_chunk),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "score_time": score_time,
            "confusion_matrix": cm,
        }

        self.logs.append(metrics)

        return metrics

    def fit_score_chunk(self, X_chunk, y_chunk, score_before_fit=False):
        """
        Score and train on a chunk in one call.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Feature chunk.
        y_chunk : array-like of shape (n_samples,)
            Target chunk.
        score_before_fit : bool, default=False
            If True, use prequential test-then-train evaluation. The first chunk is
            trained without scoring when the model is not fitted yet. If False,
            train first and then score the same chunk.

        Returns
        -------
        metrics : dict
            Score metrics for the chunk, or a score-skipped log record for the first
            prequential chunk when no fitted model exists yet.
        """
        X_chunk, y_chunk = self._validate_chunk(X_chunk, y_chunk)

        if score_before_fit:
            next_chunk_id = self.chunk_index + 1

            if not self._has_fitted:
                skipped = {
                    "chunk": next_chunk_id,
                    "event": "score_skipped",
                    "n_samples": len(y_chunk),
                    "skipped": True,
                    "reason": "First prequential chunk skipped because model is not fitted yet.",
                    "score_time": 0.0,
                }

                self.logs.append(skipped)
                self.fit_chunk(X_chunk, y_chunk)
                return skipped

            metrics = self.score_chunk(X_chunk, y_chunk, chunk_id=next_chunk_id)
            self.fit_chunk(X_chunk, y_chunk)
            return metrics

        self.fit_chunk(X_chunk, y_chunk)
        return self.score_chunk(X_chunk, y_chunk)

    def get_logs(self):
        """
        Return stored training and scoring logs.

        Returns
        -------
        logs : list of dict
            Copy of log records.
        """
        return list(self.logs)

    def get_metric_history(self, metric_name="chunk_accuracy"):
        """
        Extract one metric from score logs.

        Parameters
        ----------
        metric_name : str, default="chunk_accuracy"
            Metric key to extract.

        Returns
        -------
        values : list
            Metric values across chunks where available.
        """
        return [log[metric_name] for log in self.logs if metric_name in log]

    def reset(self):
        """
        Reset trainer logs and cumulative scoring state.

        Returns
        -------
        self : StreamTrainer
        """
        self.logs = []
        self.n_samples_seen = 0
        self.total_correct = 0
        self.total_scored = 0
        self.chunk_index = 0
        self._has_fitted = self._infer_model_is_fitted()
        return self