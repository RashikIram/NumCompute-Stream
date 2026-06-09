import numpy as np


class ZeroRClassifier:
    """
    Streaming baseline classifier that predicts the most frequent class.

    This model is useful as a simple classification baseline for benchmarking.
    It supports both fit() and partial_fit() by maintaining cumulative class
    counts over incoming chunks.
    """

    def __init__(self):
        self.most_common = None
        self.classes_ = None
        self.class_counts_ = None
        self.n_samples_seen_ = 0

    def reset(self):
        """
        Reset learned class counts.

        Returns
        -------
        self : ZeroRClassifier
        """
        self.most_common = None
        self.classes_ = None
        self.class_counts_ = None
        self.n_samples_seen_ = 0
        return self

    def fit(self, X, y):
        """
        Fit from a full batch by resetting and then applying partial_fit().
        """
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y, classes=None):
        """
        Incrementally update class counts from one chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Only the number of rows is used.
        y : array-like of shape (n_samples,)
            Class labels.
        classes : array-like, optional
            Fixed class order for streaming use.

        Returns
        -------
        self : ZeroRClassifier
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if X.shape[0] != len(y):
            raise ValueError("X and y must contain the same number of samples")

        new_classes = np.asarray(classes) if classes is not None else np.unique(y)

        if self.classes_ is None:
            self.classes_ = np.unique(new_classes)
            self.class_counts_ = np.zeros(len(self.classes_), dtype=float)
        else:
            all_classes = np.unique(np.concatenate([self.classes_, new_classes]))
            if len(all_classes) != len(self.classes_) or not np.array_equal(all_classes, self.classes_):
                new_counts = np.zeros(len(all_classes), dtype=float)
                for i, cls in enumerate(self.classes_):
                    new_counts[np.where(all_classes == cls)[0][0]] = self.class_counts_[i]
                self.classes_ = all_classes
                self.class_counts_ = new_counts

        for cls in np.unique(y):
            idx = np.where(self.classes_ == cls)[0]
            if len(idx) > 0:
                self.class_counts_[idx[0]] += np.sum(y == cls)

        self.n_samples_seen_ += len(y)
        self.most_common = self.classes_[np.argmax(self.class_counts_)]

        return self

    def predict(self, X):
        """
        Predict the most frequent class seen so far.
        """
        if self.most_common is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)

        return np.full(X.shape[0], self.most_common)

    def predict_proba(self, X):
        """
        Predict empirical class probabilities for every sample.
        """
        if self.most_common is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)
        total = np.sum(self.class_counts_)

        if total == 0:
            proba = np.zeros(len(self.classes_), dtype=float)
        else:
            proba = self.class_counts_ / total

        return np.tile(proba, (X.shape[0], 1))

    def score(self, X, y):
        """
        Compute accuracy score.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)

        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(y) != len(y_pred):
            raise ValueError("X and y must contain the same number of samples")

        return np.mean(y == y_pred)


class ZeroRRegressor:
    """
    Predicts the mean value.

    This class is retained as a simple legacy baseline, but the current
    assignment focuses on classification-only models.
    """

    def __init__(self):
        self.mean_value = None
        self.n_samples_seen_ = 0
        self._sum = 0.0

    def reset(self):
        self.mean_value = None
        self.n_samples_seen_ = 0
        self._sum = 0.0
        return self

    def fit(self, X, y):
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y):
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        self._sum += np.sum(y)
        self.n_samples_seen_ += len(y)
        self.mean_value = self._sum / self.n_samples_seen_

        return self

    def predict(self, X):
        if self.mean_value is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)

        return np.full(X.shape[0], self.mean_value)
