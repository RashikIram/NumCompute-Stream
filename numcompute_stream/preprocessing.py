
import numpy as np
import warnings


class Imputer:
    """
    Replace missing numeric values using mean, median, or constant strategy.

    Parameters
    ----------
    strategy : str, default="mean"
        Imputation strategy. Supported values are "mean", "median", and "constant".
    fill_value : float, default=0
        Value used when strategy="constant" or when a column contains only NaN values.

    Attributes
    ----------
    statistics : ndarray of shape (n_features,)
        Learned replacement value for each column.
    n_samples_seen : ndarray of shape (n_features,)
        Number of non-missing values seen per column during streaming updates.
    _values_seen : list of list
        Stored non-missing values for median streaming updates.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(n_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self, strategy="mean", fill_value=0):
        if strategy not in ["mean", "median", "constant"]:
            raise ValueError("Invalid strategy. Supported strategies are: 'mean', 'median', 'constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics = None
        self.n_samples_seen = None
        self._sums = None
        self._values_seen = None

    def fit(self, X):
        """
        Compute replacement statistics from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : Imputer

        Raises
        ------
        ValueError
            If X is not a 2D array.
        """
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update imputation statistics using a new data chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric data chunk.

        Returns
        -------
        self : Imputer

        Raises
        ------
        ValueError
            If X is not 2D or if the number of columns changes between chunks.

        Notes
        -----
        Mean strategy is updated using running sums and counts.
        Median strategy stores observed non-missing values to recompute median.
        Constant strategy keeps a fixed fill value.

        Time Complexity
        ---------------
        O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Imputer expects a 2D array")

        n_features = X.shape[1]

        if self.statistics is not None and n_features != len(self.statistics):
            raise ValueError("X has different number of columns than fitted data")

        if self.strategy == "constant":
            self.statistics = np.full(n_features, self.fill_value)
            self.n_samples_seen = np.zeros(n_features, dtype=int)
            return self

        valid_mask = ~np.isnan(X)

        if self.statistics is None:
            self.statistics = np.full(n_features, self.fill_value, dtype=float)
            self.n_samples_seen = np.zeros(n_features, dtype=int)

            if self.strategy == "mean":
                self._sums = np.zeros(n_features, dtype=float)

            elif self.strategy == "median":
                self._values_seen = [[] for _ in range(n_features)]

        if self.strategy == "mean":
            chunk_sums = np.nansum(X, axis=0)
            chunk_counts = np.sum(valid_mask, axis=0)

            self._sums += chunk_sums
            self.n_samples_seen += chunk_counts

            self.statistics = np.where(
                self.n_samples_seen > 0,
                self._sums / np.maximum(self.n_samples_seen, 1),
                self.fill_value
            )

        elif self.strategy == "median":
            for j in range(n_features):
                values = X[valid_mask[:, j], j]
                if len(values) > 0:
                    self._values_seen[j].extend(values.tolist())

                if len(self._values_seen[j]) > 0:
                    self.statistics[j] = np.median(np.asarray(self._values_seen[j], dtype=float))
                    self.n_samples_seen[j] = len(self._values_seen[j])
                else:
                    self.statistics[j] = self.fill_value

        return self

    def transform(self, X):
        """
        Replace NaN values using learned statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)

        Raises
        ------
        ValueError
            If the imputer has not been fitted, if X is not 2D, or if column count differs.
        """
        if self.statistics is None:
            raise ValueError("Imputer has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("Imputer expects a 2D array")

        if X.shape[1] != len(self.statistics):
            raise ValueError("X has different number of columns than fitted data")

        X_out = X.copy()

        nan_rows, nan_cols = np.where(np.isnan(X_out))
        X_out[nan_rows, nan_cols] = self.statistics[nan_cols]

        return X_out

    def fit_transform(self, X):
        """
        Fit the imputer and transform X in one step.
        """
        return self.fit(X).transform(X)

    def reset(self):
        """
        Reset learned imputation statistics.

        Returns
        -------
        self : Imputer
        """
        self.statistics = None
        self.n_samples_seen = None
        self._sums = None
        self._values_seen = None
        return self


class StandardScaler:
    """
    Standardize numeric features using z-score scaling.

    Formula
    -------
    X_scaled = (X - mean) / std

    Attributes
    ----------
    mean : ndarray of shape (n_features,)
    std : ndarray of shape (n_features,)
    var : ndarray of shape (n_features,)
    n_samples_seen : ndarray of shape (n_features,)

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(n_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self):
        self.mean = None
        self.var = None
        self.std = None
        self.n_samples_seen = None
        self._M2 = None

    def fit(self, X):
        """
        Compute column-wise mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler
        """
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update running mean and variance from a new data chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        ValueError
            If X is not 2D or if the number of columns changes between chunks.

        Notes
        -----
        Uses a vectorised Welford-style merge of existing statistics and chunk statistics.
        NaN values are ignored.

        Time Complexity
        ---------------
        O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array")

        n_features = X.shape[1]

        valid_counts = np.sum(~np.isnan(X), axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_mean = np.nanmean(X, axis=0)
            chunk_var = np.nanvar(X, axis=0)

        chunk_mean = np.where(np.isnan(chunk_mean), 0.0, chunk_mean)
        chunk_var = np.where(np.isnan(chunk_var), 0.0, chunk_var)
        chunk_M2 = chunk_var * valid_counts

        if self.mean is None:
            self.mean = np.zeros(n_features, dtype=float)
            self.var = np.zeros(n_features, dtype=float)
            self.std = np.ones(n_features, dtype=float)
            self.n_samples_seen = np.zeros(n_features, dtype=int)
            self._M2 = np.zeros(n_features, dtype=float)

        elif n_features != len(self.mean):
            raise ValueError("X has different number of columns than fitted data")

        total_count = self.n_samples_seen + valid_counts
        nonzero = total_count > 0

        delta = chunk_mean - self.mean

        new_mean = self.mean.copy()
        new_M2 = self._M2.copy()

        new_mean[nonzero] = self.mean[nonzero] + (
            delta[nonzero] * valid_counts[nonzero] / total_count[nonzero]
        )

        new_M2[nonzero] = (
            self._M2[nonzero]
            + chunk_M2[nonzero]
            + (delta[nonzero] ** 2)
            * self.n_samples_seen[nonzero]
            * valid_counts[nonzero]
            / total_count[nonzero]
        )

        self.mean = new_mean
        self._M2 = new_M2
        self.n_samples_seen = total_count

        self.var = np.where(
            self.n_samples_seen > 0,
            self._M2 / np.maximum(self.n_samples_seen, 1),
            0.0
        )

        self.std = np.sqrt(self.var)
        self.std = np.where(self.std == 0, 1, self.std)

        return self

    def transform(self, X):
        """
        Apply z-score scaling using learned mean and standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("StandardScaler expects a 2D array")

        if X.shape[1] != len(self.mean):
            raise ValueError("X has different number of columns than fitted data")

        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """
        Fit the scaler and transform X in one step.
        """
        return self.fit(X).transform(X)

    def reset(self):
        """
        Reset learned scaling statistics.

        Returns
        -------
        self : StandardScaler
        """
        self.mean = None
        self.var = None
        self.std = None
        self.n_samples_seen = None
        self._M2 = None
        return self


class MinMaxScaler:
    """
    Scale numeric features to a fixed range.

    Formula
    -------
    X_scaled = ((X - min) / (max - min)) * (high - low) + low

    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired output range.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(n_samples * n_features)
    transform: O(n_samples * n_features)
    """

    def __init__(self, feature_range=(0, 1)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must contain two values")

        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range minimum must be less than maximum")

        self.feature_range = feature_range
        self.min = None
        self.max = None
        self.range = None

    def fit(self, X):
        """
        Compute column-wise minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler
        """
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update running minimum and maximum values from a new data chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : MinMaxScaler

        Raises
        ------
        ValueError
            If X is not 2D or if column count changes.

        Time Complexity
        ---------------
        O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array")

        n_features = X.shape[1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chunk_min = np.nanmin(X, axis=0)
            chunk_max = np.nanmax(X, axis=0)

        chunk_min = np.where(np.isnan(chunk_min), np.inf, chunk_min)
        chunk_max = np.where(np.isnan(chunk_max), -np.inf, chunk_max)

        if self.min is None:
            self.min = np.full(n_features, np.inf)
            self.max = np.full(n_features, -np.inf)

        elif n_features != len(self.min):
            raise ValueError("X has different number of columns than fitted data")

        # Keep infinities internally until a valid value is observed. Replacing
        # them too early can pollute later chunks when an all-NaN column becomes
        # valid. Safe finite values are used during transform().
        self.min = np.minimum(self.min, chunk_min)
        self.max = np.maximum(self.max, chunk_max)

        safe_min = np.where(np.isinf(self.min), 0.0, self.min)
        safe_max = np.where(np.isinf(self.max), 0.0, self.max)

        self.range = safe_max - safe_min
        self.range = np.where(self.range == 0, 1, self.range)

        return self

    def transform(self, X):
        """
        Apply min-max scaling using learned minimum and maximum values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
        """
        if self.min is None or self.max is None or self.range is None:
            raise ValueError("MinMaxScaler has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("MinMaxScaler expects a 2D array")

        if X.shape[1] != len(self.min):
            raise ValueError("X has different number of columns than fitted data")

        low, high = self.feature_range
        safe_min = np.where(np.isinf(self.min), 0.0, self.min)
        X_scaled = (X - safe_min) / self.range

        return X_scaled * (high - low) + low

    def fit_transform(self, X):
        """
        Fit the scaler and transform X in one step.
        """
        return self.fit(X).transform(X)

    def reset(self):
        """
        Reset learned min-max statistics.

        Returns
        -------
        self : MinMaxScaler
        """
        self.min = None
        self.max = None
        self.range = None
        return self


class OneHotEncoder:
    """
    Encode categorical columns as one-hot numeric arrays.

    Parameters
    ----------
    handle_unknown : {'ignore', 'error'}, default='ignore'
        Behaviour for categories not seen during fitting. 'ignore' encodes them
        as all zeros unless add_unknown=True, while 'error' raises ValueError.
    add_unknown : bool, default=False
        If True, add one extra column per categorical feature to capture unseen
        categories after categories are frozen. The default is False so the
        encoder output width equals the number of learned categories.
    freeze_after_first_fit : bool, default=False
        If True, categories learned from the first partial_fit() are frozen.
        This keeps output dimensionality stable inside streaming pipelines.

    Attributes
    ----------
    categories : list of ndarray
        Unique categories learned for each column.
    frozen : bool
        Whether new categories are prevented from expanding output dimensions.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features log n_samples)
    partial_fit: O(n_samples * n_features log n_samples)
    transform: O(n_samples * total_categories)
    """

    def __init__(self, handle_unknown="ignore", add_unknown=False, freeze_after_first_fit=False):
        if handle_unknown not in ["ignore", "error"]:
            raise ValueError("handle_unknown must be either 'ignore' or 'error'")

        self.handle_unknown = handle_unknown
        self.add_unknown = add_unknown
        self.freeze_after_first_fit = freeze_after_first_fit
        self.categories = None
        self.frozen = False

    def fit(self, X):
        """
        Learn unique categories for each categorical column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : OneHotEncoder
        """
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Update known categories using a new data chunk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        ValueError
            If X is not 2D or column count changes between chunks.

        Notes
        -----
        Categories expand incrementally until freeze() is called or until
        freeze_after_first_fit=True locks the first observed category set.
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 2D array")

        if self.categories is None:
            self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]

            if self.freeze_after_first_fit:
                self.freeze()

        else:
            if X.shape[1] != len(self.categories):
                raise ValueError("X has different number of columns than fitted data")

            if not self.frozen:
                self.categories = [
                    np.unique(np.concatenate([self.categories[i], np.unique(X[:, i])]))
                    for i in range(X.shape[1])
                ]

        return self

    def freeze(self):
        """
        Freeze known categories to keep future transform dimensions stable.

        Returns
        -------
        self : OneHotEncoder
        """
        if self.categories is None:
            raise ValueError("OneHotEncoder cannot be frozen before fitting")

        self.frozen = True
        return self

    def transform(self, X):
        """
        Transform categorical values into one-hot encoded columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_encoded : ndarray of shape (n_samples, total_encoded_features)
        """
        if self.categories is None:
            raise ValueError("OneHotEncoder has not been fitted yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("OneHotEncoder expects a 2D array")

        if X.shape[1] != len(self.categories):
            raise ValueError("X has different number of columns than fitted data")

        encoded_columns = []

        for i in range(X.shape[1]):
            col = X[:, i]
            categories = self.categories[i]
            known_mask = np.isin(col, categories)

            if self.handle_unknown == "error" and np.any(~known_mask):
                raise ValueError("X contains categories that were not seen during fitting")

            one_hot = (col[:, None] == categories).astype(int)

            if self.add_unknown:
                unknown_col = (~known_mask).astype(int).reshape(-1, 1)
                one_hot = np.hstack([one_hot, unknown_col])

            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)

    def fit_transform(self, X):
        """
        Fit the encoder and transform X in one step.
        """
        return self.fit(X).transform(X)

    def reset(self):
        """
        Reset learned categories.

        Returns
        -------
        self : OneHotEncoder
        """
        self.categories = None
        self.frozen = False
        return self


class ColumnTransformer:
    """
    Apply separate preprocessing to numeric and categorical columns.

    Numeric columns:
        Imputer -> StandardScaler

    Categorical columns:
        OneHotEncoder

    If num_cols and cat_cols are not provided, column types are automatically detected:
        - Columns convertible to float are treated as numeric
        - Remaining columns are treated as categorical

    Parameters
    ----------
    num_cols : list of int, optional
        Indices of numeric columns.
    cat_cols : list of int, optional
        Indices of categorical columns.
    freeze_categories : bool, default=False
        If True, categorical output dimensions remain fixed after the first
        partial_fit(). Unseen categories are sent to an unknown category column.

    Time Complexity
    ---------------
    fit: O(n_samples * n_features)
    partial_fit: O(n_samples * n_features)
    transform: O(n_samples * transformed_features)
    """

    def __init__(self, num_cols=None, cat_cols=None, freeze_categories=False):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.freeze_categories = freeze_categories

        self.imputer = Imputer(strategy="mean")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(add_unknown=freeze_categories, freeze_after_first_fit=freeze_categories)

    def _detect_columns(self, X):
        """
        Automatically detect numeric and categorical columns.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        num_cols : list of int
        cat_cols : list of int
        """
        num_cols = []
        cat_cols = []

        for i in range(X.shape[1]):
            col = X[:, i]

            try:
                col.astype(float)
                num_cols.append(i)
            except (ValueError, TypeError):
                cat_cols.append(i)

        return num_cols, cat_cols

    def fit(self, X):
        """
        Fit numeric and categorical preprocessing components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : ColumnTransformer
        """
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """
        Incrementally fit numeric and categorical preprocessing components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self : ColumnTransformer

        Raises
        ------
        ValueError
            If X is not 2D or if no columns are selected.

        Time Complexity
        ---------------
        O(n_samples * n_features)
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("ColumnTransformer expects a 2D array")

        if self.num_cols is None and self.cat_cols is None:
            self.num_cols, self.cat_cols = self._detect_columns(X)

        self.num_cols = [] if self.num_cols is None else self.num_cols
        self.cat_cols = [] if self.cat_cols is None else self.cat_cols

        if len(self.num_cols) == 0 and len(self.cat_cols) == 0:
            raise ValueError("No columns were selected for transformation")

        if len(self.num_cols) > 0:
            X_num = X[:, self.num_cols].astype(float)
            self.imputer.partial_fit(X_num)
            X_num = self.imputer.transform(X_num)
            self.scaler.partial_fit(X_num)

        if len(self.cat_cols) > 0:
            X_cat = X[:, self.cat_cols]
            self.encoder.partial_fit(X_cat)

        return self

    def transform(self, X):
        """
        Transform selected numeric and categorical columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of shape (n_samples, transformed_features)
        """
        X = np.asarray(X, dtype=object)

        if X.ndim != 2:
            raise ValueError("ColumnTransformer expects a 2D array")

        parts = []

        if len(self.num_cols) > 0:
            if self.scaler.mean is None:
                raise ValueError("ColumnTransformer numeric part has not been fitted yet.")

            X_num = X[:, self.num_cols].astype(float)
            X_num = self.imputer.transform(X_num)
            X_num = self.scaler.transform(X_num)
            parts.append(X_num)

        if len(self.cat_cols) > 0:
            if self.encoder.categories is None:
                raise ValueError("ColumnTransformer categorical part has not been fitted yet.")

            X_cat = X[:, self.cat_cols]
            X_cat = self.encoder.transform(X_cat)
            parts.append(X_cat)

        if not parts:
            raise ValueError("No columns were selected for transformation")

        return np.hstack(parts)

    def fit_transform(self, X):
        """
        Fit all preprocessing components and transform X in one step.
        """
        return self.fit(X).transform(X)

    def reset(self):
        """
        Reset all preprocessing components.

        Returns
        -------
        self : ColumnTransformer
        """
        self.imputer.reset()
        self.scaler.reset()
        self.encoder.reset()
        return self
