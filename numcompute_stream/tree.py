import numpy as np


class DecisionTreeClassifier:
    """
    Depth-limited decision tree classifier using Gini impurity or entropy.

    The classifier is implemented using only NumPy and supports both batch
    training through fit() and streaming-style updates through partial_fit().
    For streaming use, chunks are accumulated and the tree is rebuilt after
    each update. This keeps the implementation simple, deterministic, and
    suitable for small-to-medium educational streaming simulations.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the decision tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : {'gini', 'entropy'}, default='gini'
        Impurity criterion used to evaluate candidate splits.
    max_features : int, float, {'sqrt', 'log2'}, or None, default=None
        Number of features considered at each split.
        - None: use all features
        - int: use exactly max_features features
        - float: use int(max_features * n_features)
        - 'sqrt': use sqrt(n_features)
        - 'log2': use log2(n_features)
    random_state : int or None, default=None
        Seed for reproducible feature selection.
    splitter : {'best', 'random'}, default='best'
        Strategy used to generate candidate thresholds. 'best' evaluates all
        midpoint thresholds, while 'random' samples random thresholds for an
        Extra Trees-style split.
    n_random_thresholds : int, default=1
        Number of random candidate thresholds sampled per selected feature when
        splitter='random'.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels observed during fitting.
    root_ : dict or None
        Root node of the learned tree.
    n_features_ : int or None
        Number of features seen during fitting.
    n_samples_seen_ : int
        Total number of samples accumulated through fit/partial_fit.

    Complexity
    ----------
    fit: O(depth * n_samples * n_features * n_thresholds)
    predict: O(n_samples * depth)
    Space: O(n_samples + n_nodes)
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None,
        splitter="best",
        n_random_thresholds=1
    ):
        if max_depth is None or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if criterion not in ["gini", "entropy"]:
            raise ValueError("criterion must be either 'gini' or 'entropy'")
        if splitter not in ["best", "random"]:
            raise ValueError("splitter must be either 'best' or 'random'")
        if n_random_thresholds <= 0:
            raise ValueError("n_random_thresholds must be positive")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.splitter = splitter
        self.n_random_thresholds = n_random_thresholds

        self.root_ = None
        self.classes_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self.feature_importances_ = None

        self._X_memory = None
        self._y_memory = None
        self._sample_weight_memory = None
        self._rng = np.random.default_rng(random_state)

    def reset(self):
        """
        Reset the fitted tree and stored streaming memory.

        Returns
        -------
        self : DecisionTreeClassifier

        Complexity
        ----------
        Time: O(1)
        Space: O(1)
        """
        self.root_ = None
        self.classes_ = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self.feature_importances_ = None
        self._X_memory = None
        self._y_memory = None
        self._sample_weight_memory = None
        self._rng = np.random.default_rng(self.random_state)
        return self

    def fit(self, X, y, sample_weight=None, classes=None):
        """
        Fit a decision tree from a full dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative sample weights.
        classes : array-like, optional
            Global list of possible classes. Useful when not all classes appear
            in the current data.

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        ValueError
            If shapes are invalid, inputs are non-numeric, or data is empty.

        Complexity
        ----------
        Time: O(depth * n_samples * n_features * n_thresholds)
        Space: O(n_samples + n_nodes)
        """
        X, y, sample_weight = self._validate_fit_inputs(X, y, sample_weight)

        self._X_memory = X.copy()
        self._y_memory = y.copy()
        self._sample_weight_memory = sample_weight.copy()
        self.n_samples_seen_ = len(y)
        self.n_features_ = X.shape[1]

        if classes is not None:
            self.classes_ = np.asarray(classes)
        else:
            self.classes_ = np.unique(y)

        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self._rng = np.random.default_rng(self.random_state)

        self.root_ = self._build_tree(X, y, sample_weight, depth=0)

        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def partial_fit(self, X_chunk, y_chunk, classes=None, sample_weight=None):
        """
        Update the tree using a new data chunk.

        Chunks are accumulated and the tree is rebuilt after each update. This
        simulates streaming adaptation while preserving deterministic tree logic.

        Parameters
        ----------
        X_chunk : array-like of shape (n_samples, n_features)
            Incoming feature chunk.
        y_chunk : array-like of shape (n_samples,)
            Incoming class labels.
        classes : array-like, optional
            Global list of possible classes. Useful when not all classes appear
            in the first chunk.
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative sample weights for the current chunk.

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        ValueError
            If chunk shapes are invalid or inconsistent with previous chunks.
        """
        X_chunk, y_chunk, sample_weight = self._validate_fit_inputs(
            X_chunk,
            y_chunk,
            sample_weight
        )

        if self.n_features_ is not None and X_chunk.shape[1] != self.n_features_:
            raise ValueError("X_chunk has different number of features than previous chunks")

        if self._X_memory is None:
            self._X_memory = X_chunk.copy()
            self._y_memory = y_chunk.copy()
            self._sample_weight_memory = sample_weight.copy()
        else:
            self._X_memory = np.vstack([self._X_memory, X_chunk])
            self._y_memory = np.concatenate([self._y_memory, y_chunk])
            self._sample_weight_memory = np.concatenate([
                self._sample_weight_memory,
                sample_weight
            ])

        if classes is not None:
            self.classes_ = np.asarray(classes)
        else:
            self.classes_ = np.unique(self._y_memory)

        # Rebuild using accumulated data and accumulated per-sample weights.
        # This preserves online bagging / boosting weights across chunks.
        self.fit(
            self._X_memory,
            self._y_memory,
            sample_weight=self._sample_weight_memory,
            classes=self.classes_
        )

        return self

    def predict(self, X):
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)

        Raises
        ------
        ValueError
            If the tree has not been fitted or feature count differs.
        """
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier has not been fitted yet")

        X = self._validate_X(X)

        return np.asarray([self._predict_one(row, self.root_) for row in X])

    def predict_proba(self, X):
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier has not been fitted yet")

        X = self._validate_X(X)

        probabilities = [self._predict_proba_one(row, self.root_) for row in X]
        return np.vstack(probabilities)

    def score(self, X, y):
        """
        Compute classification accuracy.

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

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError("X has different number of features than fitted data")

        return X

    def _validate_fit_inputs(self, X, y, sample_weight=None):
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

        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be a 1D array")
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight must match number of samples")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight cannot contain negative values")

        return X, y, sample_weight

    def _build_tree(self, X, y, sample_weight, depth):
        prediction = self._majority_class(y, sample_weight)
        probabilities = self._class_probabilities(y, sample_weight)

        node = {
            "is_leaf": True,
            "prediction": prediction,
            "proba": probabilities,
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None,
            "depth": depth
        }

        if self._should_stop(X, y, depth):
            return node

        split = self._best_split(X, y, sample_weight)

        if split is None:
            return node

        feature, threshold, gain, left_mask, right_mask = split

        if gain <= 0:
            return node

        node["is_leaf"] = False
        node["feature"] = feature
        node["threshold"] = threshold

        self.feature_importances_[feature] += gain

        node["left"] = self._build_tree(
            X[left_mask],
            y[left_mask],
            sample_weight[left_mask],
            depth + 1
        )
        node["right"] = self._build_tree(
            X[right_mask],
            y[right_mask],
            sample_weight[right_mask],
            depth + 1
        )

        return node

    def _should_stop(self, X, y, depth):
        if depth >= self.max_depth:
            return True
        if len(y) < self.min_samples_split:
            return True
        if len(np.unique(y)) == 1:
            return True
        return False
    
    def _best_split(self, X, y, sample_weight):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        best_left_mask = None
        best_right_mask = None

        parent_impurity = self._impurity(y, sample_weight)
        features = self._select_features(X.shape[1])

        y_class_idx, n_classes = self._class_indices_sorted(y)
        total_counts = self._weighted_counts_from_indices(
            y_class_idx,
            sample_weight,
            n_classes
        )
        total_weight = np.sum(sample_weight)

        for feature in features:
            values = X[:, feature]
            valid_mask = ~np.isnan(values)

            if np.sum(valid_mask) < self.min_samples_split:
                continue

            valid_values = values[valid_mask]

            if np.unique(valid_values).size <= 1:
                continue

            # Random splitter can stay simple because it usually has very few thresholds.
            if self.splitter == "random":
                unique_values = np.unique(valid_values)
                low, high = unique_values[0], unique_values[-1]

                if low == high:
                    continue

                thresholds = self._rng.uniform(
                    low,
                    high,
                    size=self.n_random_thresholds
                )

                for threshold in thresholds:
                    left_mask = values <= threshold
                    right_mask = ~left_mask

                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue

                    gain = self._information_gain(
                        y,
                        sample_weight,
                        left_mask,
                        right_mask,
                        parent_impurity
                    )

                    if gain > best_gain:
                        best_feature = feature
                        best_threshold = threshold
                        best_gain = gain
                        best_left_mask = left_mask
                        best_right_mask = right_mask

                continue

            # Vectorized best-threshold scoring.
            valid_indices = np.where(valid_mask)[0]
            order = np.argsort(valid_values)

            sorted_indices = valid_indices[order]
            sorted_values = values[sorted_indices]
            sorted_class_idx = y_class_idx[sorted_indices]
            sorted_weights = sample_weight[sorted_indices]

            one_hot_weighted = np.eye(n_classes)[sorted_class_idx] * sorted_weights[:, None]

            left_counts = np.cumsum(one_hot_weighted, axis=0)[:-1]
            right_counts = total_counts[None, :] - left_counts

            left_weight = np.sum(left_counts, axis=1)
            right_weight = total_weight - left_weight

            valid_splits = (
                (sorted_values[:-1] != sorted_values[1:]) &
                (left_weight > 0) &
                (right_weight > 0)
            )

            if not np.any(valid_splits):
                continue

            left_probs = left_counts / np.maximum(left_weight[:, None], 1e-12)
            right_probs = right_counts / np.maximum(right_weight[:, None], 1e-12)

            if self.criterion == "gini":
                left_impurity = 1.0 - np.sum(left_probs ** 2, axis=1)
                right_impurity = 1.0 - np.sum(right_probs ** 2, axis=1)
            else:
                left_log = np.zeros_like(left_probs)
                right_log = np.zeros_like(right_probs)

                np.log2(left_probs, out=left_log, where=left_probs > 0)
                np.log2(right_probs, out=right_log, where=right_probs > 0)

                left_impurity = -np.sum(left_probs * left_log, axis=1)
                right_impurity = -np.sum(right_probs * right_log, axis=1)

                
            child_impurity = (
                left_weight * left_impurity +
                right_weight * right_impurity
            ) / np.maximum(total_weight, 1e-12)

            gains = parent_impurity - child_impurity
            gains[~valid_splits] = -np.inf

            local_best_idx = int(np.argmax(gains))
            local_best_gain = gains[local_best_idx]

            if local_best_gain > best_gain:
                threshold = (
                    sorted_values[local_best_idx] +
                    sorted_values[local_best_idx + 1]
                ) / 2.0

                left_mask = values <= threshold
                right_mask = ~left_mask

                best_feature = feature
                best_threshold = threshold
                best_gain = local_best_gain
                best_left_mask = left_mask
                best_right_mask = right_mask

        if best_feature is None:
            return None

        return best_feature, best_threshold, best_gain, best_left_mask, best_right_mask

    def _select_features(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                k = max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")
        elif isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("max_features must be positive")
            k = min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            if self.max_features <= 0 or self.max_features > 1:
                raise ValueError("float max_features must be in (0, 1]")
            k = max(1, int(self.max_features * n_features))
        else:
            raise ValueError("max_features must be None, int, float, 'sqrt', or 'log2'")

        return self._rng.choice(n_features, size=k, replace=False)
    
    def _class_indices_sorted(self, y):
        """
        Map labels to integer class indices using sorted class order.

        This is used internally for vectorized class-count operations.
        """
        sorted_classes = np.sort(self.classes_)
        idx = np.searchsorted(sorted_classes, y)

        valid = (
            (idx >= 0) &
            (idx < len(sorted_classes)) &
            (sorted_classes[idx] == y)
        )

        if not np.all(valid):
            raise ValueError("y contains labels not present in classes_")

        return idx, len(sorted_classes)


    def _weighted_counts_from_indices(self, class_idx, sample_weight, n_classes):
        """
        Compute weighted class counts using np.bincount.
        """
        return np.bincount(
            class_idx,
            weights=sample_weight,
            minlength=n_classes
        ).astype(float)


    def _impurity_from_counts(self, counts):
        """
        Compute Gini or entropy from class-count vector.
        """
        total = np.sum(counts)

        if total <= 0:
            return 0.0

        probs = counts / total
        probs = probs[probs > 0]

        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)

        return -np.sum(probs * np.log2(probs))

    def _information_gain(self, y, sample_weight, left_mask, right_mask, parent_impurity):
        left_weight = np.sum(sample_weight[left_mask])
        right_weight = np.sum(sample_weight[right_mask])
        total_weight = left_weight + right_weight

        if total_weight == 0:
            return 0.0

        left_impurity = self._impurity(y[left_mask], sample_weight[left_mask])
        right_impurity = self._impurity(y[right_mask], sample_weight[right_mask])

        child_impurity = (
            (left_weight / total_weight) * left_impurity +
            (right_weight / total_weight) * right_impurity
        )

        return parent_impurity - child_impurity
    
    def _impurity(self, y, sample_weight):
        if len(y) == 0:
            return 0.0

        class_idx, n_classes = self._class_indices_sorted(y)
        counts = self._weighted_counts_from_indices(class_idx, sample_weight, n_classes)

        return self._impurity_from_counts(counts)

    def _majority_class(self, y, sample_weight):
        classes = np.unique(y)
        sort_order = np.argsort(classes)
        sorted_classes = classes[sort_order]

        idx = np.searchsorted(sorted_classes, y)
        counts = np.bincount(idx, weights=sample_weight, minlength=len(sorted_classes))

        return sorted_classes[np.argmax(counts)]

    def _class_probabilities(self, y, sample_weight):
        probabilities = np.zeros(len(self.classes_), dtype=float)
        total_weight = np.sum(sample_weight)

        if total_weight <= 0:
            return probabilities

        sort_order = np.argsort(self.classes_)
        sorted_classes = self.classes_[sort_order]

        idx = np.searchsorted(sorted_classes, y)
        counts_sorted = np.bincount(idx, weights=sample_weight, minlength=len(sorted_classes))

        counts_original_order = np.zeros_like(counts_sorted, dtype=float)
        counts_original_order[sort_order] = counts_sorted

        return counts_original_order / total_weight

    def _predict_one(self, row, node):
        while not node["is_leaf"]:
            feature = node["feature"]
            threshold = node["threshold"]

            if np.isnan(row[feature]):
                node = node["right"]
            elif row[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node["prediction"]

    def _predict_proba_one(self, row, node):
        while not node["is_leaf"]:
            feature = node["feature"]
            threshold = node["threshold"]

            if np.isnan(row[feature]):
                node = node["right"]
            elif row[feature] <= threshold:
                node = node["left"]
            else:
                node = node["right"]

        return node["proba"]
