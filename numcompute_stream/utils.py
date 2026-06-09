import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    test_size : float, default=0.2
        Proportion of data to use for testing.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != len(y):
        raise ValueError("X and y must have same number of samples")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _validate_numeric_array(X, name="Input"):
    # Converts input into numpy array and checks numeric type
    X = np.asarray(X)

    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"{name} must be numeric")

    return X


def euclidean_distance(x1, x2):
    x1 = _validate_numeric_array(x1, "x1").astype(float)
    x2 = _validate_numeric_array(x2, "x2").astype(float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("euclidean_distance expects 1D arrays")

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")

    # Euclidean distance = square root of sum of squared differences
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    x1 = _validate_numeric_array(x1, "x1").astype(float)
    x2 = _validate_numeric_array(x2, "x2").astype(float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("manhattan_distance expects 1D arrays")

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")

    # Manhattan distance = sum of absolute differences
    return np.sum(np.abs(x1 - x2))


def cosine_distance(x1, x2):
    x1 = _validate_numeric_array(x1, "x1").astype(float)
    x2 = _validate_numeric_array(x2, "x2").astype(float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("cosine_distance expects 1D arrays")

    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")

    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)

    if norm_x1 == 0 or norm_x2 == 0:
        raise ValueError("cosine_distance is undefined for zero vectors")

    # Cosine distance = 1 - cosine similarity
    cosine_similarity = np.dot(x1, x2) / (norm_x1 * norm_x2)

    return 1 - cosine_similarity


def pairwise_distances(X, Y=None, metric="euclidean"):
    if metric not in ["euclidean", "manhattan", "cosine"]:
        raise ValueError("Invalid metric. Supported metrics are: 'euclidean', 'manhattan', 'cosine'.")

    X = _validate_numeric_array(X, "X").astype(float)

    if X.ndim != 2:
        raise ValueError("pairwise_distances expects X to be a 2D array")

    if Y is None:
        Y = X
    else:
        Y = _validate_numeric_array(Y, "Y").astype(float)

    if Y.ndim != 2:
        raise ValueError("pairwise_distances expects Y to be a 2D array")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    if metric == "euclidean":
        # Vectorized Euclidean distance using broadcasting
        diff = X[:, None, :] - Y[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    elif metric == "manhattan":
        # Vectorized Manhattan distance using broadcasting
        diff = X[:, None, :] - Y[None, :, :]
        return np.sum(np.abs(diff), axis=2)

    elif metric == "cosine":
        # Vectorized cosine distance
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)

        if np.any(X_norm == 0) or np.any(Y_norm == 0):
            raise ValueError("cosine distance is undefined for zero vectors")

        similarity = (X @ Y.T) / (X_norm[:, None] * Y_norm[None, :])

        return 1 - similarity


def sigmoid(x):
    x = _validate_numeric_array(x, "x").astype(float)

    # Stable sigmoid to reduce overflow problems
    result = np.empty_like(x, dtype=float)

    positive = x >= 0
    negative = ~positive

    result[positive] = 1 / (1 + np.exp(-x[positive]))

    exp_x = np.exp(x[negative])
    result[negative] = exp_x / (1 + exp_x)

    return result


def logsumexp(x, axis=None):
    x = _validate_numeric_array(x, "x").astype(float)

    if x.size == 0:
        raise ValueError("logsumexp expects a non-empty array")

    # Stable logsumexp using max-shift trick
    max_x = np.max(x, axis=axis, keepdims=True)

    shifted = x - max_x
    result = max_x + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    if axis is None:
        return float(np.squeeze(result))

    return np.squeeze(result, axis=axis)


def softmax(x, axis=-1):
    x = _validate_numeric_array(x, "x").astype(float)

    if x.size == 0:
        raise ValueError("softmax expects a non-empty array")

    # Stable softmax using max-shift trick
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    return exp_x / sum_exp


def batch_iterator(X, batch_size, y=None, shuffle=False, random_state=None):
    X = _validate_numeric_array(X, "X")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if X.ndim == 0:
        raise ValueError("X must have at least one dimension")

    n_samples = X.shape[0]

    if y is not None:
        y = np.asarray(y)

        if len(y) != n_samples:
            raise ValueError("X and y must have the same number of samples")

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Initialize Minibatches
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]

        if y is None:
            yield X[batch_idx]
        else:
            yield X[batch_idx], y[batch_idx]