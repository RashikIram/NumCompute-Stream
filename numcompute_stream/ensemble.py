import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


# ---------------- VALIDATION HELPERS ---------------- #

def _validate_X_y(X, y):
    """
    Validate feature matrix and target labels for classifiers.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Numeric class labels.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If X or y has invalid shape, non-numeric values, or mismatched rows.

    Complexity
    ----------
    Time: O(n_samples * n_features)
    Space: O(n_samples * n_features)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    if X.shape[0] != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("y must be numeric")

    return X, y


def _validate_X(X, n_features=None):
    """
    Validate feature matrix for prediction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    n_features : int, optional
        Expected number of features.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)

    Raises
    ------
    ValueError
        If X is invalid or feature count differs.

    Complexity
    ----------
    Time: O(n_samples * n_features)
    Space: O(n_samples * n_features)
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[0] == 0:
        raise ValueError("X cannot be empty")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X has different number of features than fitted data")

    return X


def _resolve_max_features(max_features, n_features):
    """
    Convert max_features configuration into an integer feature count.

    Parameters
    ----------
    max_features : int, float, {'sqrt', 'log2'}, or None
    n_features : int

    Returns
    -------
    int
        Number of features to use.

    Raises
    ------
    ValueError
        If max_features is invalid.

    Complexity
    ----------
    Time: O(1)
    Space: O(1)
    """
    if max_features is None:
        return n_features

    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")

    if isinstance(max_features, int):
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        return min(max_features, n_features)

    if isinstance(max_features, float):
        if max_features <= 0 or max_features > 1:
            raise ValueError("float max_features must be in (0, 1]")
        return max(1, int(max_features * n_features))

    raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")


class EnsembleClassifier:
    """
    Base class for tree-based classifier ensembles.

    This class stores shared configuration and provides common prediction logic.
    Specific ensemble behaviour is implemented by subclasses such as
    OnlineBaggingClassifier, RandomForestClassifier, RandomSubspaceClassifier,
    ExtraTreesClassifier, and AdaBoostSAMMEClassifier.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of decision trees in the ensemble.
    max_depth : int, default=5
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split tree nodes.
    criterion : {'gini', 'entropy'}, default='gini'
        Split criterion used by each tree.
    max_features : int, float, {'sqrt', 'log2'}, or None, default=None
        Number of features considered by each tree.
    voting : {'hard', 'soft'}, default='soft'
        Voting method used for final predictions.
    random_state : int or None, default=None
        Seed for reproducible randomness.

    Attributes
    ----------
    estimators_ : list
        Fitted decision trees.
    classes_ : ndarray of shape (n_classes,)
        Observed class labels.
    n_features_ : int or None
        Number of features seen during fitting.
    n_samples_seen_ : int
        Number of original stream samples processed.

    Complexity
    ----------
    predict: O(n_estimators * n_samples * tree_depth)
    Space: O(n_estimators * tree_size)
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        voting="soft",
        random_state=None
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if voting not in ["hard", "soft"]:
            raise ValueError("voting must be either 'hard' or 'soft'")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.voting = voting
        self.random_state = random_state

        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._rng = np.random.default_rng(random_state)

    def reset(self):
        """
        Reset ensemble state.

        Returns
        -------
        self : EnsembleClassifier

        Complexity
        ----------
        Time: O(1)
        Space: O(1)
        """
        self.estimators_ = []
        self.classes_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._rng = np.random.default_rng(self.random_state)
        return self

    def _make_tree(self, random_state=None, max_features=None):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features if max_features is None else max_features,
            random_state=random_state
        )

    def _ensure_fitted(self):
        if len(self.estimators_) == 0:
            raise ValueError("EnsembleClassifier has not been fitted yet")

    def _check_chunk(self, X, y):
        X, y = _validate_X_y(X, y)

        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError("X has different number of features than previous chunks")

        if self.n_features_ is None:
            self.n_features_ = X.shape[1]

        if self.classes_ is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, np.unique(y)]))

        return X, y

    def _align_proba(self, tree, proba):
        aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=float)

        for i, cls in enumerate(tree.classes_):
            matches = np.where(self.classes_ == cls)[0]
            if len(matches) > 0:
                aligned[:, matches[0]] = proba[:, i]

        return aligned

    def predict_proba(self, X):
        """
        Predict class probabilities by averaging tree probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)

        Raises
        ------
        ValueError
            If ensemble has not been fitted.
        """
        self._ensure_fitted()
        X = _validate_X(X, self.n_features_)

        probas = []
        for tree in self.estimators_:
            probas.append(self._align_proba(tree, tree.predict_proba(X)))

        proba = np.mean(probas, axis=0)
        row_sums = np.sum(proba, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)

        return proba / row_sums

    def predict(self, X):
        """
        Predict class labels using hard or soft voting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        self._ensure_fitted()
        X = _validate_X(X, self.n_features_)

        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        
        votes = np.vstack([tree.predict(X) for tree in self.estimators_]).T

        sort_order = np.argsort(self.classes_)
        sorted_classes = self.classes_[sort_order]

        vote_idx_sorted = np.searchsorted(sorted_classes, votes)
        counts_sorted = np.zeros((votes.shape[0], len(self.classes_)), dtype=int)

        rows = np.repeat(np.arange(votes.shape[0]), votes.shape[1])
        cols = vote_idx_sorted.ravel()

        np.add.at(counts_sorted, (rows, cols), 1)

        best_sorted = np.argmax(counts_sorted, axis=1)
        return sorted_classes[best_sorted]

    def score(self, X, y):
        """
        Compute accuracy score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float
        """
        y = np.asarray(y)
        y_pred = self.predict(X)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(y) != len(y_pred):
            raise ValueError("X and y must contain the same number of samples")

        return np.mean(y == y_pred)


class OnlineBaggingClassifier(EnsembleClassifier):
    """
    Online bagging ensemble using Poisson bootstrap sampling.

    Each incoming sample receives a Poisson(1) weight per tree. This
    approximates bootstrap sampling in a streaming setting without materialising
    repeated rows.

    Methods
    -------
    fit(X, y)
    partial_fit(X_chunk, y_chunk)
    predict(X)
    predict_proba(X)
    reset()

    Complexity
    ----------
    partial_fit: O(n_estimators * tree_fit_cost)
    predict: O(n_estimators * n_samples * tree_depth)
    """

    def fit(self, X, y):
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._check_chunk(X_chunk, y_chunk)

        if len(self.estimators_) == 0:
            self.estimators_ = [
                self._make_tree(random_state=self._rng.integers(0, 1_000_000))
                for _ in range(self.n_estimators)
            ]

        for tree in self.estimators_:
            counts = self._rng.poisson(1.0, size=len(y_chunk))

            # Ensure every tree receives at least one sample from very small chunks.
            if np.sum(counts) == 0:
                counts[self._rng.integers(0, len(counts))] = 1

            tree.partial_fit(
                X_chunk,
                y_chunk,
                classes=self.classes_,
                sample_weight=counts.astype(float)
            )

        self.n_samples_seen_ += len(y_chunk)
        return self


class RandomForestClassifier(OnlineBaggingClassifier):
    """
    Streaming random forest classifier.

    RandomForestClassifier extends online bagging by using random feature
    selection inside each decision tree. By default, max_features='sqrt'.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        voting="soft",
        random_state=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            voting=voting,
            random_state=random_state
        )


class RandomSubspaceClassifier(EnsembleClassifier):
    """
    Random subspace ensemble classifier.

    Each tree is assigned a fixed random subset of features. The same feature
    subset is reused during prediction, which increases diversity and reduces
    per-tree feature dimensionality.

    Attributes
    ----------
    feature_indices_ : list of ndarray
        Feature indices used by each tree.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        voting="soft",
        random_state=None
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            voting=voting,
            random_state=random_state
        )
        self.feature_indices_ = []

    def reset(self):
        super().reset()
        self.feature_indices_ = []
        return self

    def fit(self, X, y):
        self.reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._check_chunk(X_chunk, y_chunk)

        if len(self.estimators_) == 0:
            k = _resolve_max_features(self.max_features, self.n_features_)

            self.feature_indices_ = [
                np.sort(self._rng.choice(self.n_features_, size=k, replace=False))
                for _ in range(self.n_estimators)
            ]

            self.estimators_ = [
                self._make_tree(random_state=self._rng.integers(0, 1_000_000), max_features=None)
                for _ in range(self.n_estimators)
            ]

        for tree, features in zip(self.estimators_, self.feature_indices_):
            tree.partial_fit(X_chunk[:, features], y_chunk, classes=self.classes_)

        self.n_samples_seen_ += len(y_chunk)
        return self

    def predict_proba(self, X):
        self._ensure_fitted()
        X = _validate_X(X, self.n_features_)

        probas = []
        for tree, features in zip(self.estimators_, self.feature_indices_):
            probas.append(self._align_proba(tree, tree.predict_proba(X[:, features])))

        proba = np.mean(probas, axis=0)
        row_sums = np.sum(proba, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)

        return proba / row_sums

    def predict(self, X):
        self._ensure_fitted()
        X = _validate_X(X, self.n_features_)

        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        votes = []
        for tree, features in zip(self.estimators_, self.feature_indices_):
            votes.append(tree.predict(X[:, features]))

        votes = np.vstack(votes).T  # shape: (n_samples, n_estimators)

        sort_order = np.argsort(self.classes_)
        sorted_classes = self.classes_[sort_order]

        flat_votes = votes.ravel()
        flat_pos_sorted = np.searchsorted(sorted_classes, flat_votes)

        valid = (
            (flat_pos_sorted >= 0) &
            (flat_pos_sorted < len(sorted_classes)) &
            (sorted_classes[flat_pos_sorted] == flat_votes)
        )

        if not np.all(valid):
            raise ValueError("Estimator predicted a class not present in self.classes_")

        flat_pos_original = sort_order[flat_pos_sorted]

        vote_counts = np.zeros((votes.shape[0], len(self.classes_)), dtype=int)

        rows = np.repeat(np.arange(votes.shape[0]), votes.shape[1])
        np.add.at(vote_counts, (rows, flat_pos_original), 1)

        return self.classes_[np.argmax(vote_counts, axis=1)]


class ExtraTreesClassifier(RandomSubspaceClassifier):
    """
    Extra Trees-style streaming ensemble classifier.

    This implementation combines fixed random feature subspaces with random
    threshold selection inside each decision tree. The random thresholds make
    this model genuinely different from RandomSubspaceClassifier while keeping
    the same streaming-compatible partial_fit() interface.

    Parameters
    ----------
    n_random_thresholds : int, default=1
        Number of random candidate thresholds sampled per selected feature in
        each tree split. Larger values reduce randomness but may improve split
        quality.
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        voting="soft",
        random_state=None,
        n_random_thresholds=1
    ):
        if n_random_thresholds <= 0:
            raise ValueError("n_random_thresholds must be positive")

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            voting=voting,
            random_state=random_state
        )
        self.n_random_thresholds = n_random_thresholds

    def _make_tree(self, random_state=None, max_features=None):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            max_features=self.max_features if max_features is None else max_features,
            random_state=random_state,
            splitter="random",
            n_random_thresholds=self.n_random_thresholds
        )


class AdaBoostSAMMEClassifier(EnsembleClassifier):
    """
    SAMME AdaBoost classifier using shallow decision trees.

    The classifier supports streaming-style partial_fit() by accumulating chunks
    and refitting the boosted ensemble on all samples seen so far. This keeps
    boosting weights consistent across the accumulated stream.

    Parameters
    ----------
    n_estimators : int, default=20
        Number of boosting rounds.
    learning_rate : float, default=1.0
        Shrinks estimator weights.
    max_depth : int, default=1
        Depth of weak decision trees.

    Complexity
    ----------
    fit/partial_fit: O(n_estimators * tree_fit_cost)
    predict: O(n_estimators * n_samples * tree_depth)
    """

    def __init__(
        self,
        n_estimators=20,
        learning_rate=1.0,
        max_depth=1,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        voting="soft",
        random_state=None
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            max_features=max_features,
            voting=voting,
            random_state=random_state
        )
        self.learning_rate = learning_rate
        self.estimator_weights_ = []
        self._X_memory = None
        self._y_memory = None

    def reset(self):
        super().reset()
        self.estimator_weights_ = []
        self._X_memory = None
        self._y_memory = None
        return self

    def fit(self, X, y):
        self.reset()
        X, y = self._check_chunk(X, y)
        self._X_memory = X.copy()
        self._y_memory = y.copy()
        self.n_samples_seen_ = len(y)
        self._fit_boosted()
        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk, y_chunk = self._check_chunk(X_chunk, y_chunk)

        if self._X_memory is None:
            self._X_memory = X_chunk.copy()
            self._y_memory = y_chunk.copy()
        else:
            self._X_memory = np.vstack([self._X_memory, X_chunk])
            self._y_memory = np.concatenate([self._y_memory, y_chunk])

        self.n_samples_seen_ += len(y_chunk)
        self._fit_boosted()
        return self

    def _fit_boosted(self):
        X = self._X_memory
        y = self._y_memory
        n_samples = len(y)
        n_classes = len(self.classes_)

        self.estimators_ = []
        self.estimator_weights_ = []

        if n_classes < 2:
            tree = self._make_tree(random_state=self._rng.integers(0, 1_000_000))
            tree.fit(X, y)
            self.estimators_.append(tree)
            self.estimator_weights_.append(1.0)
            return

        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=float)

        for _ in range(self.n_estimators):
            tree = self._make_tree(random_state=self._rng.integers(0, 1_000_000))
            tree.fit(X, y, sample_weight=sample_weight)

            pred = tree.predict(X)
            incorrect = pred != y
            error = np.sum(sample_weight[incorrect]) / np.sum(sample_weight)

            if error <= 0:
                alpha = 1.0
                self.estimators_.append(tree)
                self.estimator_weights_.append(alpha)
                break

            # SAMME requires error < 1 - 1 / n_classes.
            if error >= (1.0 - (1.0 / n_classes)):
                break

            alpha = self.learning_rate * (
                np.log((1.0 - error) / error) + np.log(n_classes - 1)
            )

            self.estimators_.append(tree)
            self.estimator_weights_.append(alpha)

            sample_weight = sample_weight * np.exp(alpha * incorrect)
            sample_weight = sample_weight / np.sum(sample_weight)

        if len(self.estimators_) == 0:
            tree = self._make_tree(random_state=self._rng.integers(0, 1_000_000))
            tree.fit(X, y)
            self.estimators_.append(tree)
            self.estimator_weights_.append(1.0)

    def predict_proba(self, X):
        self._ensure_fitted()
        X = _validate_X(X, self.n_features_)

        scores = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        sort_order = np.argsort(self.classes_)
        sorted_classes = self.classes_[sort_order]

        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = tree.predict(X)
            pred_idx_sorted = np.searchsorted(sorted_classes, pred)
            pred_idx_original = sort_order[pred_idx_sorted]

            scores[np.arange(X.shape[0]), pred_idx_original] += alpha

        row_sums = np.sum(scores, axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)

        return scores / row_sums
