import numpy as np
import warnings


# ---------------- VALIDATION HELPERS ---------------- #

def _validate_numeric(values, name="values", allow_empty=False):
    """
    Validate and convert numeric input to a NumPy array.

    Parameters
    ----------
    values : array-like
        Numeric values.
    name : str, default="values"
        Name used in error messages.
    allow_empty : bool, default=False
        Whether empty arrays are allowed.

    Returns
    -------
    arr : ndarray

    Raises
    ------
    ValueError
        If values are None, empty when not allowed, or non-numeric.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    if values is None:
        raise ValueError(f"{name} cannot be None")

    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must contain numeric values")

    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} cannot be empty")

    return arr


def _validate_axis(axis, ndim):
    """
    Validate an axis argument against an array dimensionality.
    """
    if axis is None:
        return axis

    if not isinstance(axis, int):
        raise ValueError("axis must be None or an integer")

    if axis < -ndim or axis >= ndim:
        raise ValueError("axis is out of bounds for input array")

    return axis


# ---------------- BATCH STATISTICAL FUNCTIONS ---------------- #

def mean(values, axis=None, ignore_nan=True):
    """
    Compute the arithmetic mean.

    Parameters
    ----------
    values : array-like
        Numeric values.
    axis : int or None, default=None
        Axis over which to compute the mean.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or ndarray

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmean(arr, axis=axis) if ignore_nan else np.mean(arr, axis=axis)

    return result


def variance(values, axis=None, ddof=0, ignore_nan=True):
    """
    Compute variance.

    Parameters
    ----------
    values : array-like
        Numeric values.
    axis : int or None, default=None
        Axis over which to compute variance.
    ddof : int, default=0
        Delta degrees of freedom.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or ndarray
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanvar(arr, axis=axis, ddof=ddof) if ignore_nan else np.var(arr, axis=axis, ddof=ddof)

    return result


def standard_deviation(values, axis=None, ddof=0, ignore_nan=True):
    """
    Compute standard deviation.
    """
    return np.sqrt(variance(values, axis=axis, ddof=ddof, ignore_nan=ignore_nan))


def std(values, axis=None, ddof=0, ignore_nan=True):
    """
    Alias for standard_deviation().
    """
    return standard_deviation(values, axis=axis, ddof=ddof, ignore_nan=ignore_nan)


def median(values, axis=None, ignore_nan=True):
    """
    Compute the median.
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(arr, axis=axis) if ignore_nan else np.median(arr, axis=axis)

    return result


def quantile(values, q, axis=None, ignore_nan=True):
    """
    Compute one or more quantiles.

    Parameters
    ----------
    values : array-like
        Numeric values.
    q : float or array-like
        Quantile or sequence of quantiles in [0, 1].
    axis : int or None, default=None
        Axis over which to compute quantiles.
    ignore_nan : bool, default=True
        If True, NaN values are ignored.

    Returns
    -------
    float or ndarray
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)
    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanquantile(arr, q_arr, axis=axis) if ignore_nan else np.quantile(arr, q_arr, axis=axis)

    return result


def percentile(values, q, axis=None, ignore_nan=True):
    """
    Compute one or more percentiles.

    q must be in [0, 100].
    """
    q_arr = np.asarray(q, dtype=float)

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(values, q_arr / 100.0, axis=axis, ignore_nan=ignore_nan)



def quantiles(values, q, axis=None, ignore_nan=True):
    """
    Backwards-compatible percentile helper.

    Unlike quantile(), q is expressed in the 0..100 percentile scale.
    """
    return percentile(values, q, axis=axis, ignore_nan=ignore_nan)


def describe(values, axis=None, ignore_nan=True):
    """
    Return common descriptive statistics for numeric data.
    """
    return {
        "mean": mean(values, axis=axis, ignore_nan=ignore_nan),
        "median": median(values, axis=axis, ignore_nan=ignore_nan),
        "variance": variance(values, axis=axis, ignore_nan=ignore_nan),
        "std": std(values, axis=axis, ignore_nan=ignore_nan),
        "min": min_value(values, axis=axis, ignore_nan=ignore_nan),
        "max": max_value(values, axis=axis, ignore_nan=ignore_nan),
    }

def min_value(values, axis=None, ignore_nan=True):
    """
    Compute the minimum value.
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmin(arr, axis=axis) if ignore_nan else np.min(arr, axis=axis)

    return result


def max_value(values, axis=None, ignore_nan=True):
    """
    Compute the maximum value.
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmax(arr, axis=axis) if ignore_nan else np.max(arr, axis=axis)

    return result


def data_range(values, axis=None, ignore_nan=True):
    """
    Compute max - min.
    """
    return max_value(values, axis=axis, ignore_nan=ignore_nan) - min_value(values, axis=axis, ignore_nan=ignore_nan)


def histogram(values, bins=10, value_range=None):
    """
    Compute a histogram using NumPy.

    Parameters
    ----------
    values : array-like
        Numeric values. NaNs are ignored.
    bins : int or array-like, default=10
        Histogram bins.
    value_range : tuple, optional
        Lower and upper range for bins.

    Returns
    -------
    counts : ndarray
    bin_edges : ndarray
    """
    arr = _validate_numeric(values, allow_empty=True).ravel()
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        if isinstance(bins, int):
            if bins <= 0:
                raise ValueError("bins must be positive")
            return np.zeros(bins, dtype=int), np.linspace(0.0, 1.0, bins + 1)
        edges = np.asarray(bins, dtype=float)
        return np.zeros(max(len(edges) - 1, 0), dtype=int), edges

    return np.histogram(arr, bins=bins, range=value_range)


def covariance(x, y=None, ddof=1):
    """
    Compute covariance using np.cov-compatible semantics.
    """
    x = _validate_numeric(x, name="x")

    if y is not None:
        y = _validate_numeric(y, name="y")

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        return np.cov(x.ravel(), y.ravel(), ddof=ddof)[0, 1]

    return np.cov(x, rowvar=False, ddof=ddof)


def correlation(x, y=None):
    """
    Compute Pearson correlation.
    """
    x = _validate_numeric(x, name="x")

    if y is not None:
        y = _validate_numeric(y, name="y")

        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")

        if x.size < 2:
            return 0.0

        matrix = np.corrcoef(x.ravel(), y.ravel())
        value = matrix[0, 1]
        return 0.0 if np.isnan(value) else value

    matrix = np.corrcoef(x, rowvar=False)
    return np.nan_to_num(matrix, nan=0.0)


def normalize(values, feature_range=(0, 1), axis=0):
    """
    Min-max normalize values to a target range.

    Parameters
    ----------
    values : array-like
        Numeric values.
    feature_range : tuple, default=(0, 1)
        Desired output range.
    axis : int, default=0
        Axis used to compute minimum and maximum.

    Returns
    -------
    ndarray
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    if len(feature_range) != 2 or feature_range[0] >= feature_range[1]:
        raise ValueError("feature_range must contain increasing (min, max) values")

    low, high = feature_range
    arr_min = np.nanmin(arr, axis=axis, keepdims=True)
    arr_max = np.nanmax(arr, axis=axis, keepdims=True)
    arr_range = arr_max - arr_min
    arr_range = np.where(arr_range == 0, 1.0, arr_range)

    scaled = (arr - arr_min) / arr_range
    return scaled * (high - low) + low


def z_score(values, axis=0):
    """
    Standardize values to zero mean and unit variance.
    """
    arr = _validate_numeric(values)
    axis = _validate_axis(axis, arr.ndim)

    arr_mean = np.nanmean(arr, axis=axis, keepdims=True)
    arr_std = np.nanstd(arr, axis=axis, keepdims=True)
    arr_std = np.where(arr_std == 0, 1.0, arr_std)

    return (arr - arr_mean) / arr_std


def mode(values):
    """
    Compute the most frequent value with deterministic tie-breaking.

    np.unique sorts values, so ties are resolved by the smallest sorted value.
    """
    if values is None:
        raise ValueError("values cannot be None")

    arr = np.asarray(values)

    if arr.size == 0:
        raise ValueError("values cannot be empty")

    values_unique, counts = np.unique(arr, return_counts=True)
    return values_unique[np.argmax(counts)]


# ---------------- STREAMING STATISTICS ---------------- #

class StreamingStats:
    """
    Maintain streaming statistics for numeric chunks.

    This implementation keeps the newer column-wise behaviour for 2D input, but
    also supports the older 1D API expected by the tests: bins=..., histogram(),
    quantiles(), scalar count/mean/variance for 1D streams, and NaN results for
    an empty/all-NaN stream.
    """

    def __init__(self, store_values=True, bins=10, value_range=None):
        if isinstance(bins, int) and bins <= 0:
            raise ValueError("bins must be positive")
        self.store_values = store_values
        self.bins = bins
        self.value_range = value_range
        self.reset()

    def reset(self):
        self._data = [] if self.store_values else None
        self._is_1d = None
        self.mean = None
        self.var = None
        self.std = None
        self.n_samples_seen = None
        self.min = None
        self.max = None
        return self

    def update_stats(self, X_chunk):
        X_chunk = _validate_numeric(X_chunk, name="X_chunk", allow_empty=True)

        if X_chunk.ndim == 0:
            X_chunk = X_chunk.reshape(1)

        if X_chunk.ndim == 1:
            current_is_1d = True
            X_2d = X_chunk.reshape(-1, 1)
        elif X_chunk.ndim == 2:
            current_is_1d = False
            X_2d = X_chunk
        else:
            raise ValueError("X_chunk must be a 1D or 2D array")

        if self._is_1d is None:
            self._is_1d = current_is_1d
        elif self._is_1d != current_is_1d:
            raise ValueError("X_chunk dimensionality changed between updates")

        if self.store_values:
            valid_values = X_2d[~np.isnan(X_2d)] if current_is_1d else None
            if current_is_1d:
                self._data.extend(valid_values.tolist())
            else:
                if self._data == []:
                    self._data = [[] for _ in range(X_2d.shape[1])]
                if len(self._data) != X_2d.shape[1]:
                    raise ValueError("X_chunk has different number of columns than previous chunks")
                for j in range(X_2d.shape[1]):
                    vals = X_2d[~np.isnan(X_2d[:, j]), j]
                    self._data[j].extend(vals.tolist())
        else:
            # Keep minimal running raw arrays unnecessary; this compatibility
            # path is intentionally exact only when store_values=True.
            if not hasattr(self, "_running_chunks"):
                self._running_chunks = []
            self._running_chunks.append(X_2d.copy())

        self._recompute()
        return self

    def _observed_array(self):
        if self.store_values:
            if self._is_1d is False:
                max_len = max((len(v) for v in self._data), default=0)
                # Stats are recomputed column-wise directly for 2D; this method
                # is mainly for 1D histogram/quantiles.
                return self._data
            return np.asarray(self._data, dtype=float)
        chunks = getattr(self, "_running_chunks", [])
        if not chunks:
            return np.array([], dtype=float)
        return np.vstack(chunks) if self._is_1d is False else np.concatenate([c.ravel() for c in chunks])

    def _recompute(self):
        if self._is_1d is False:
            data_cols = self._data if self.store_values else [np.asarray(np.vstack(getattr(self, "_running_chunks", []))[:, j]) for j in range(getattr(self, "_running_chunks", [np.empty((0,0))])[-1].shape[1])]
            n_features = len(data_cols)
            counts = np.array([len(v) for v in data_cols], dtype=int)
            means = np.full(n_features, np.nan)
            vars_ = np.full(n_features, np.nan)
            stds = np.full(n_features, np.nan)
            mins = np.full(n_features, np.nan)
            maxs = np.full(n_features, np.nan)
            for j, vals in enumerate(data_cols):
                vals = np.asarray(vals, dtype=float)
                if vals.size:
                    means[j] = np.mean(vals)
                    vars_[j] = np.var(vals)
                    stds[j] = np.std(vals)
                    mins[j] = np.min(vals)
                    maxs[j] = np.max(vals)
            self.mean, self.var, self.std = means, vars_, stds
            self.min, self.max, self.n_samples_seen = mins, maxs, counts
            return

        values = self._observed_array()
        count = int(values.size)
        self.n_samples_seen = count
        if count == 0:
            self.mean = np.nan
            self.var = np.nan
            self.std = np.nan
            self.min = np.nan
            self.max = np.nan
        else:
            self.mean = float(np.mean(values))
            self.var = float(np.var(values))
            self.std = float(np.std(values))
            self.min = float(np.min(values))
            self.max = float(np.max(values))

    def result(self):
        if self._is_1d is None:
            return {
                "mean": np.nan,
                "variance": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "count": 0,
            }
        return {
            "mean": np.array(self.mean).copy() if isinstance(self.mean, np.ndarray) else self.mean,
            "variance": np.array(self.var).copy() if isinstance(self.var, np.ndarray) else self.var,
            "std": np.array(self.std).copy() if isinstance(self.std, np.ndarray) else self.std,
            "min": np.array(self.min).copy() if isinstance(self.min, np.ndarray) else self.min,
            "max": np.array(self.max).copy() if isinstance(self.max, np.ndarray) else self.max,
            "count": np.array(self.n_samples_seen).copy() if isinstance(self.n_samples_seen, np.ndarray) else self.n_samples_seen,
        }

    def variance(self):
        return self.result()["variance"]

    def standard_deviation(self):
        return self.result()["std"]

    def quantile(self, q):
        if not self.store_values or self._data is None:
            raise ValueError("quantile requires store_values=True")
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 1)):
            raise ValueError("q must be between 0 and 1")
        if self._is_1d is False:
            return np.asarray([
                np.full(q_arr.shape, np.nan, dtype=float) if len(v) == 0 else np.quantile(np.asarray(v, dtype=float), q_arr)
                for v in self._data
            ])
        values = np.asarray(self._data, dtype=float)
        return np.full(q_arr.shape, np.nan, dtype=float) if values.size == 0 else np.quantile(values, q_arr)

    def quantiles(self, q):
        if not self.store_values or self._data is None:
            raise ValueError("quantiles requires store_values=True")
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0) | (q_arr > 100)):
            raise ValueError("q must be between 0 and 100")
        return self.quantile(q_arr / 100.0)

    def histogram(self):
        if not self.store_values or self._data is None:
            raise ValueError("histogram requires store_values=True")
        if self._is_1d is False:
            values = np.concatenate([np.asarray(v, dtype=float) for v in self._data]) if self._data else np.array([])
        else:
            values = np.asarray(self._data, dtype=float)
        return histogram(values, bins=self.bins, value_range=self.value_range)


class StreamingHistogram:
    """
    Maintain a streaming histogram for numeric data.

    Parameters
    ----------
    bins : int or array-like, default=10
        Histogram bins passed to np.histogram(). Fixed bin edges are recommended
        for true streaming consistency.
    value_range : tuple, optional
        Lower and upper range for bins when bins is an integer.

    Complexity
    ----------
    update_stats: O(n_samples)
    result: O(n_bins)
    """

    def __init__(self, bins=10, value_range=None):
        if isinstance(bins, int) and bins <= 0:
            raise ValueError("bins must be positive")

        self.bins = bins
        self.value_range = value_range
        self.reset()

    def reset(self):
        """
        Reset histogram counts and bin edges.
        """
        self.counts = None
        self.bin_edges = None
        return self

    def update_stats(self, X_chunk):
        """
        Update histogram counts from a numeric chunk.
        """
        X_chunk = _validate_numeric(X_chunk, name="X_chunk").ravel()
        X_chunk = X_chunk[~np.isnan(X_chunk)]

        if X_chunk.size == 0:
            return self

        counts, edges = np.histogram(X_chunk, bins=self.bins, range=self.value_range)

        if self.counts is None:
            self.counts = counts.astype(int)
            self.bin_edges = edges
        else:
            if not np.array_equal(edges, self.bin_edges):
                raise ValueError("Histogram bin edges changed; use fixed bins or value_range")
            self.counts += counts

        return self

    def result(self):
        """
        Return histogram counts and bin edges.
        """
        if self.counts is None:
            return np.array([], dtype=int), np.array([], dtype=float)

        return self.counts.copy(), self.bin_edges.copy()


# Backwards-compatible aliases that are common in student test suites.
minimum = min_value
maximum = max_value
range_value = data_range
standardize = z_score


def update_stats(stat_object, X_chunk=None):
    """
    Convenience wrapper for streaming statistics updates.

    Parameters
    ----------
    stat_object : object or array-like
        If X_chunk is provided, this must be an object implementing
        update_stats(). If X_chunk is omitted, stat_object is treated as a data
        chunk and a temporary StreamingStats object is returned as a dictionary.
    X_chunk : array-like, optional
        Numeric data chunk.

    Returns
    -------
    object or dict
        Updated stat object, or a result dictionary when called as
        update_stats(X_chunk).
    """
    if X_chunk is None:
        stats = StreamingStats()
        stats.update_stats(stat_object)
        return stats.result()

    if not hasattr(stat_object, "update_stats"):
        raise ValueError("stat_object must implement update_stats")

    return stat_object.update_stats(X_chunk)
