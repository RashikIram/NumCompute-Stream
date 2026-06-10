import sys
import time
import csv
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np

from numcompute_stream.io import read_csv
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import (
    OnlineBaggingClassifier,
    RandomForestClassifier,
    RandomSubspaceClassifier,
    ExtraTreesClassifier,
    AdaBoostSAMMEClassifier,
)
from numcompute_stream.metrics import (
    accuracy,
    macro_f1,
    macro_precision,
    macro_recall,
    weighted_f1,
    matthews_corrcoef,
    cohen_kappa,
)

DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "benchmark" / "benchmark_results.csv"

VECTORIZATION_OUT_PATH = ROOT / "benchmark" / "vectorization_results.csv"

DATASETS = {
    "Iris": DATA_DIR / "Iris.csv",
    "Sleep": DATA_DIR / "sleep_data.csv",
}

MISSING_TOKENS = {"", "nan", "none", "na", "n/a", "null", "?"}
ID_COLUMN_NAMES = {"id", "idx", "index", "unnamed: 0", "unnamed:_0", "row", "row_id"}

PERFORMANCE_COLUMNS = [
    "dataset",
    "random_state",
    "model",
    "prequential_cumulative_accuracy",
    "prequential_macro_precision",
    "prequential_macro_recall",
    "prequential_macro_f1",
    "prequential_weighted_f1",
    "prequential_balanced_accuracy",
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
    "test_balanced_accuracy",
    "test_mcc",
    "test_cohen_kappa",
]

EFFICIENCY_COLUMNS = [
    "dataset",
    "random_state",
    "model",
    "n_estimators",
    "chunks",
    "evaluated_chunks",
    "fit_time",
    "pred_time",
    "total_time",
    "memory_mb",
    "latency_ms_per_sample",
    "train_samples_per_second",
    "test_samples_per_second",
]


# ---------------- BACKWARDS-COMPATIBILITY HELPERS ---------------- #

def convert_features_to_numeric(X):
    """
    Convert mixed feature data into numeric values.

    Numeric columns are cast to float. Categorical columns are encoded using
    deterministic integer codes. Missing numeric values are replaced by the
    column mean, or 0.0 when a whole column is missing.

    This function is kept for compatibility with earlier benchmark tests. The
    main benchmark path uses train-only preprocessing to avoid data leakage.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    Returns
    -------
    X_out : ndarray of shape (n_samples, n_features)

    Complexity
    ----------
    Time: O(n_samples * n_features log n_samples)
    Space: O(n_samples * n_features)
    """
    X = np.asarray(X, dtype=object)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    columns = []

    for i in range(X.shape[1]):
        col = X[:, i]
        missing = _missing_mask(col)
        numeric_col = np.full(col.shape[0], np.nan, dtype=float)

        try:
            numeric_col[~missing] = col[~missing].astype(float)
            numeric_col = np.where(np.isfinite(numeric_col), numeric_col, np.nan)
        except (ValueError, TypeError):
            text_col = col.astype(str)
            valid_text = text_col[~missing]

            if valid_text.size == 0:
                numeric_col = np.zeros(col.shape[0], dtype=float)
            else:
                categories, encoded_valid = np.unique(valid_text, return_inverse=True)
                numeric_col = np.full(col.shape[0], float(len(categories)), dtype=float)
                numeric_col[~missing] = encoded_valid.astype(float)

        if np.any(np.isnan(numeric_col)):
            valid = numeric_col[~np.isnan(numeric_col)]
            fill_value = float(np.mean(valid)) if valid.size > 0 else 0.0
            numeric_col = np.where(np.isnan(numeric_col), fill_value, numeric_col)

        columns.append(numeric_col)

    return np.column_stack(columns).astype(float)


def load_csv_dataset(path, target_col=-1):
    """
    Load a CSV classification dataset using the custom read_csv() function.

    This compatibility helper follows the original benchmark behaviour by
    returning a fully numeric feature matrix and encoded numeric labels. For
    leakage-safe benchmarking, prefer load_preprocessed_train_test().

    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    target_col : int, default=-1
        Index of the target column.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    """
    raw = read_csv(path, delimiter=",", dtype=object)
    raw = np.asarray(raw, dtype=object)

    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    if raw.shape[0] < 2:
        raise ValueError("dataset must contain a header row and at least one data row")

    data = raw[1:, :]

    if target_col < 0:
        target_col = raw.shape[1] + target_col

    if target_col < 0 or target_col >= raw.shape[1]:
        raise ValueError("target_col is out of bounds")

    X_raw = np.delete(data, target_col, axis=1)
    y_raw = data[:, target_col].astype(str)

    X = convert_features_to_numeric(X_raw)
    _, y = np.unique(y_raw, return_inverse=True)

    return X, y


def train_test_split_simple(X, y, test_size=0.25, random_state=42):
    """
    Split arrays into train and test sets using a shuffled holdout split.

    This function is kept for compatibility with earlier tests. The main
    benchmark uses stratified_train_test_split() for stronger class balance.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    test_size : float, default=0.25
    random_state : int, default=42

    Returns
    -------
    X_train, X_test, y_train, y_test

    Complexity
    ----------
    Time: O(n_samples)
    Space: O(n_samples)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = int(len(X) * test_size)
    test_count = min(max(test_count, 1), len(X) - 1)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ---------------- DATA LOADING AND PREPROCESSING ---------------- #

def _as_2d_object_array(values, name="values"):
    """
    Convert input to a 2D object array.

    Parameters
    ----------
    values : array-like
        Input values.
    name : str, default="values"
        Name used in error messages.

    Returns
    -------
    arr : ndarray of shape (n_samples, n_features)

    Raises
    ------
    ValueError
        If the input is empty.
    """
    arr = np.asarray(values, dtype=object)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")

    return arr


def _missing_mask(values):
    """
    Return a boolean mask for common missing-value tokens.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    mask : ndarray of bool
        True where a value should be treated as missing.
    """
    arr = np.asarray(values, dtype=object)
    text = np.char.lower(np.char.strip(arr.astype(str)))
    return np.isin(text, list(MISSING_TOKENS))


def _try_numeric_column(col):
    """
    Convert a column to float, preserving missing values as NaN.

    Parameters
    ----------
    col : array-like of shape (n_samples,)

    Returns
    -------
    numeric_col : ndarray of float
        Converted numeric column.
    success : bool
        Whether all non-missing values were numeric.
    """
    col = np.asarray(col, dtype=object)
    mask = _missing_mask(col)
    numeric_col = np.full(col.shape[0], np.nan, dtype=float)

    if np.all(mask):
        return numeric_col, True

    try:
        numeric_col[~mask] = col[~mask].astype(float)
        numeric_col = np.where(np.isfinite(numeric_col), numeric_col, np.nan)
        return numeric_col, True
    except (ValueError, TypeError):
        return numeric_col, False


def _looks_like_sequential_id(col):
    """
    Heuristically detect a sequential identifier column.

    This is intentionally conservative and is mainly intended to remove columns
    such as Iris.csv's common `Id` feature.
    """
    numeric_col, success = _try_numeric_column(col)

    if not success or np.any(np.isnan(numeric_col)):
        return False

    if len(np.unique(numeric_col)) != len(numeric_col):
        return False

    sorted_values = np.sort(numeric_col)
    zero_based = np.arange(len(numeric_col), dtype=float)
    one_based = np.arange(1, len(numeric_col) + 1, dtype=float)

    return bool(np.array_equal(sorted_values, zero_based) or np.array_equal(sorted_values, one_based))


def _drop_identifier_columns(X_raw, feature_names=None):
    """
    Drop obvious identifier columns that should not be used as predictors.

    Parameters
    ----------
    X_raw : ndarray of shape (n_samples, n_features)
        Raw feature matrix.
    feature_names : sequence of str, optional
        Feature names from the CSV header.

    Returns
    -------
    X_clean : ndarray
        Feature matrix after dropping identifier columns.
    names_clean : list[str]
        Updated feature names.
    dropped : list[str]
        Names of dropped columns.
    """
    X_raw = _as_2d_object_array(X_raw, name="X_raw")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_raw.shape[1])]
    else:
        feature_names = [str(name) for name in feature_names]

    keep_indices = []
    dropped = []

    for i, name in enumerate(feature_names):
        normalized = name.strip().lower().replace("_", " ")
        normalized_underscore = name.strip().lower().replace(" ", "_")

        is_named_id = (
            normalized in ID_COLUMN_NAMES
            or normalized_underscore in ID_COLUMN_NAMES
            or normalized_underscore.endswith("_id")
        )

        # Only use the sequential-ID heuristic for the first feature column.
        # This avoids accidentally removing genuinely useful continuous features.
        is_sequential_first_column = i == 0 and _looks_like_sequential_id(X_raw[:, i])

        if is_named_id or is_sequential_first_column:
            dropped.append(name)
        else:
            keep_indices.append(i)

    if not keep_indices:
        raise ValueError("all feature columns were dropped as identifiers")

    return X_raw[:, keep_indices], [feature_names[i] for i in keep_indices], dropped


def _parse_slash_numeric_column(col):
    """
    Parse columns containing paired numeric values such as blood pressure "120/80".

    Parameters
    ----------
    col : array-like of shape (n_samples,)

    Returns
    -------
    left : ndarray of float
    right : ndarray of float
    success : bool
        True when all non-missing values are parseable as number/number.
    """
    col = np.asarray(col, dtype=object)
    missing = _missing_mask(col)
    left = np.full(col.shape[0], np.nan, dtype=float)
    right = np.full(col.shape[0], np.nan, dtype=float)

    non_missing = col[~missing].astype(str)

    if non_missing.size == 0:
        return left, right, False

    # Require a slash in every non-missing value so ordinary categorical values
    # are not accidentally split.
    if not np.all(np.char.find(non_missing, "/") >= 0):
        return left, right, False

    try:
        parts = np.char.split(non_missing, "/")
        parsed_left = []
        parsed_right = []

        for item in parts:
            if len(item) != 2:
                return left, right, False

            parsed_left.append(float(item[0].strip()))
            parsed_right.append(float(item[1].strip()))

        left[~missing] = np.asarray(parsed_left, dtype=float)
        right[~missing] = np.asarray(parsed_right, dtype=float)
        return left, right, True

    except (ValueError, TypeError):
        return left, right, False


def _expand_slash_numeric_columns(X_raw, feature_names):
    """
    Expand paired numeric string columns into two numeric columns.

    This is useful for the provided Sleep dataset where Blood Pressure appears
    as values such as "126/83". Treating this as an arbitrary categorical code
    would inject an artificial ordinal relationship, so the benchmark converts
    it into two meaningful numeric predictors.

    Parameters
    ----------
    X_raw : ndarray of shape (n_samples, n_features)
    feature_names : sequence of str

    Returns
    -------
    X_expanded : ndarray of object
    names_expanded : list[str]
    expanded_columns : list[str]
    """
    X_raw = _as_2d_object_array(X_raw, name="X_raw")
    feature_names = [str(name) for name in feature_names]

    columns = []
    names = []
    expanded_columns = []

    for i, name in enumerate(feature_names):
        left, right, success = _parse_slash_numeric_column(X_raw[:, i])

        if success:
            clean_name = name.strip() or f"feature_{i}"
            columns.append(left.astype(object))
            columns.append(right.astype(object))

            if clean_name.lower().replace(" ", "_") in {"blood_pressure", "bp"}:
                names.extend(["Systolic Blood Pressure", "Diastolic Blood Pressure"])
            else:
                names.extend([f"{clean_name}_left", f"{clean_name}_right"])

            expanded_columns.append(name)
        else:
            columns.append(X_raw[:, i])
            names.append(name)

    return np.column_stack(columns).astype(object), names, expanded_columns


def load_csv_raw_dataset(path, target_col=-1, has_header=True, drop_id_columns=True):
    """
    Load raw CSV data without fitting preprocessing on the full dataset.

    This prevents benchmark leakage by allowing the caller to split the raw data
    before fitting numeric/categorical conversions and imputation statistics.

    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    target_col : int, default=-1
        Index of the target column.
    has_header : bool, default=True
        Whether the first CSV row contains column names.
    drop_id_columns : bool, default=True
        Whether to drop obvious identifier columns from features.

    Returns
    -------
    X_raw : ndarray of shape (n_samples, n_features)
    y_raw : ndarray of shape (n_samples,)
    feature_names : list[str]
    dropped_columns : list[str]
    expanded_columns : list[str]
    """
    raw = read_csv(path, delimiter=",", dtype=object)
    raw = _as_2d_object_array(raw, name="raw CSV data")

    if raw.shape[1] < 2:
        raise ValueError("dataset must contain at least one feature column and one target column")

    if has_header:
        header = raw[0, :].astype(str)
        data = raw[1:, :]
    else:
        header = np.array([f"column_{i}" for i in range(raw.shape[1])], dtype=object)
        data = raw

    if data.shape[0] == 0:
        raise ValueError("dataset contains no data rows")

    if target_col < 0:
        target_col = raw.shape[1] + target_col

    if target_col < 0 or target_col >= raw.shape[1]:
        raise ValueError("target_col is out of bounds")

    X_raw = np.delete(data, target_col, axis=1)
    y_raw = data[:, target_col].astype(str)
    feature_names = np.delete(header, target_col).astype(str).tolist()

    X_raw, feature_names, expanded_columns = _expand_slash_numeric_columns(X_raw, feature_names)

    if drop_id_columns:
        X_raw, feature_names, dropped_columns = _drop_identifier_columns(X_raw, feature_names)
    else:
        dropped_columns = []

    return X_raw, y_raw, feature_names, dropped_columns, expanded_columns


def stratified_train_test_split(X, y, test_size=0.25, random_state=42):
    """
    Split arrays into train/test partitions while preserving class proportions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    test_size : float, default=0.25
    random_state : int, default=42

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.asarray(X, dtype=object)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        if len(cls_indices) == 1:
            # A single example cannot be split; keep it in training.
            n_test = 0
        else:
            n_test = int(round(len(cls_indices) * test_size))
            n_test = min(max(n_test, 1), len(cls_indices) - 1)

        test_indices.extend(cls_indices[:n_test].tolist())
        train_indices.extend(cls_indices[n_test:].tolist())

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    if test_indices.size == 0:
        raise ValueError("test split is empty; use a larger dataset or smaller number of classes")

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def fit_feature_preprocessor(X_train_raw):
    """
    Fit numeric/categorical conversion rules using training data only.

    Numeric columns are mean-imputed using training means. Categorical columns
    are encoded using deterministic training-set category codes; unseen test
    categories are mapped to a reserved unknown code.

    Parameters
    ----------
    X_train_raw : ndarray of shape (n_samples, n_features)

    Returns
    -------
    transformers : list[dict]
        Per-column preprocessing metadata.
    """
    X_train_raw = _as_2d_object_array(X_train_raw, name="X_train_raw")
    transformers = []

    for i in range(X_train_raw.shape[1]):
        col = X_train_raw[:, i]
        missing = _missing_mask(col)
        numeric_col, numeric_success = _try_numeric_column(col)

        if numeric_success:
            valid = numeric_col[~np.isnan(numeric_col)]
            fill_value = float(np.mean(valid)) if valid.size > 0 else 0.0
            transformers.append({
                "kind": "numeric",
                "fill_value": fill_value,
            })
            continue

        # Categorical encoding learned from training data only.
        non_missing = col[~missing].astype(str)

        if non_missing.size == 0:
            categories = np.array([], dtype=str)
            mode_value = ""
        else:
            categories, counts = np.unique(non_missing, return_counts=True)
            mode_value = categories[np.argmax(counts)]

        category_to_code = {category: idx for idx, category in enumerate(categories.tolist())}
        mode_code = category_to_code.get(mode_value, 0)
        unknown_code = len(categories)

        transformers.append({
            "kind": "categorical",
            "category_to_code": category_to_code,
            "mode_code": float(mode_code),
            "unknown_code": float(unknown_code),
        })

    return transformers


def transform_features(X_raw, transformers):
    """
    Transform raw features using fitted training-only preprocessing metadata.

    Parameters
    ----------
    X_raw : ndarray of shape (n_samples, n_features)
    transformers : list[dict]
        Preprocessing metadata from fit_feature_preprocessor().

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    """
    X_raw = _as_2d_object_array(X_raw, name="X_raw")

    if X_raw.shape[1] != len(transformers):
        raise ValueError("number of transformers must match number of features")

    columns = []

    for i, transformer in enumerate(transformers):
        col = X_raw[:, i]

        if transformer["kind"] == "numeric":
            numeric_col, success = _try_numeric_column(col)
            if not success:
                raise ValueError(f"feature column {i} was numeric during training but contains non-numeric values")
            numeric_col = np.where(np.isnan(numeric_col), transformer["fill_value"], numeric_col)
            columns.append(numeric_col.astype(float))
            continue

        if transformer["kind"] == "categorical":
            mapping = transformer["category_to_code"]
            unknown_code = transformer["unknown_code"]
            encoded = np.full(col.shape[0], unknown_code, dtype=float)
            missing = _missing_mask(col)
            encoded[missing] = transformer["mode_code"]

            text_col = col.astype(str)
            for category, code in mapping.items():
                encoded[(text_col == category) & (~missing)] = float(code)

            columns.append(encoded)
            continue

        raise ValueError(f"unknown transformer kind: {transformer['kind']}")

    return np.column_stack(columns).astype(float)


def encode_targets(y_train_raw, y_test_raw):
    """
    Encode target labels using labels observed in the training stream.

    Parameters
    ----------
    y_train_raw : ndarray of shape (n_train,)
    y_test_raw : ndarray of shape (n_test,)

    Returns
    -------
    y_train : ndarray of int
    y_test : ndarray of int
    classes : ndarray of int
    label_names : ndarray of str
    """
    y_train_raw = np.asarray(y_train_raw).astype(str)
    y_test_raw = np.asarray(y_test_raw).astype(str)

    label_names = np.unique(y_train_raw)
    label_to_code = {label: idx for idx, label in enumerate(label_names.tolist())}

    unknown_test = np.setdiff1d(np.unique(y_test_raw), label_names)
    if unknown_test.size > 0:
        raise ValueError(f"test set contains labels not present in training data: {unknown_test}")

    y_train = np.array([label_to_code[label] for label in y_train_raw], dtype=int)
    y_test = np.array([label_to_code[label] for label in y_test_raw], dtype=int)
    classes = np.arange(len(label_names), dtype=int)

    return y_train, y_test, classes, label_names


def load_preprocessed_train_test(path, target_col=-1, test_size=0.25, random_state=42):
    """
    Load a CSV, stratify split raw rows, then preprocess using training data only.

    Parameters
    ----------
    path : str or Path
        Dataset path.
    target_col : int, default=-1
        Target column index.
    test_size : float, default=0.25
        Test split fraction.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test, metadata
    """
    X_raw, y_raw, feature_names, dropped_columns, expanded_columns = load_csv_raw_dataset(
        path,
        target_col=target_col,
        has_header=True,
        drop_id_columns=True,
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = stratified_train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
    )

    transformers = fit_feature_preprocessor(X_train_raw)
    X_train = transform_features(X_train_raw, transformers)
    X_test = transform_features(X_test_raw, transformers)
    y_train, y_test, classes, label_names = encode_targets(y_train_raw, y_test_raw)

    metadata = {
        "feature_names": feature_names,
        "dropped_columns": dropped_columns,
        "expanded_columns": expanded_columns,
        "classes": classes,
        "label_names": label_names,
        "transformers": transformers,
    }

    return X_train, X_test, y_train, y_test, metadata


# ---------------- STREAMING BENCHMARK HELPERS ---------------- #

def _make_n_chunks(X, y, n_chunks=5):
    """
    Split training data into streaming chunks.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    n_chunks : int, default=5

    Returns
    -------
    chunks : list[tuple[ndarray, ndarray]]
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    X_chunks = np.array_split(X, n_chunks)
    y_chunks = np.array_split(y, n_chunks)

    return [(Xc, yc) for Xc, yc in zip(X_chunks, y_chunks) if len(yc) > 0]


def make_chunks(X, y, chunk_size=None, n_chunks=None):
    """
    Split arrays into streaming chunks.

    Backwards-compatible API:
    - make_chunks(X, y, chunk_size=2) yields consecutive chunks of size 2.
    - make_chunks(X, y, n_chunks=5) splits into approximately equal chunks.
    - make_chunks(X, y) keeps the newer default of 5 chunks.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return [
            (X[start:start + chunk_size], y[start:start + chunk_size])
            for start in range(0, len(y), chunk_size)
        ]

    if n_chunks is None:
        n_chunks = 5

    return _make_n_chunks(X, y, n_chunks=n_chunks)


def build_models(random_state=42, n_estimators=10):
    """
    Create base and ensemble classifiers for streaming benchmark.

    Parameters
    ----------
    random_state : int, default=42
    n_estimators : int, default=10
        Number of estimators used by ensemble methods.

    Returns
    -------
    models : dict[str, classifier]
    """
    return {
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=2,
            random_state=random_state,
        ),
        "OnlineBagging": OnlineBaggingClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            max_features="sqrt",
            random_state=random_state,
        ),
        "RandomSubspace": RandomSubspaceClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            max_features="sqrt",
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            max_features="sqrt",
            random_state=random_state,
        ),
        "AdaBoostSAMME": AdaBoostSAMMEClassifier(
            n_estimators=n_estimators,
            max_depth=1,
            random_state=random_state,
        ),
    }


def partial_fit_model(model, X_chunk, y_chunk, classes):
    """
    Call partial_fit while supporting both signatures used in the package.
    """
    try:
        model.partial_fit(X_chunk, y_chunk, classes=classes)
    except TypeError:
        model.partial_fit(X_chunk, y_chunk)


def measure_peak_memory(func, *args):
    """
    Measure peak Python-level memory allocation using tracemalloc.

    Parameters
    ----------
    func : callable
    *args
        Arguments passed to callable.

    Returns
    -------
    result : any
    peak_mb : float
    """
    tracemalloc.start()
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 ** 2)

# ---------------- VECTORIZATION BENCHMARK HELPERS ---------------- #

def _time_call(func, *args, repeats=5):
    """
    Time a function and return the best runtime across repeats.
    """
    func(*args)  # warm-up

    times = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)

    return min(times), result


def _results_close(a, b):
    """
    Check that loop and vectorized outputs are numerically equivalent.
    """
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _results_close(x, y) for x, y in zip(a, b)
        )

    try:
        return bool(np.allclose(a, b, equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(a, b))


# ---------- Confusion matrix: loop vs vectorized ---------- #

def _loop_confusion_matrix(y_true, y_pred, classes):
    """
    Loop-based confusion matrix kept only for benchmarking.
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))

    return cm


def _vectorized_confusion_matrix(y_true, y_pred, classes):
    """
    Vectorized confusion matrix using flat bincount indexing.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)

    sort_order = np.argsort(classes)
    sorted_classes = classes[sort_order]

    true_pos_sorted = np.searchsorted(sorted_classes, y_true)
    pred_pos_sorted = np.searchsorted(sorted_classes, y_pred)

    valid = (
        (true_pos_sorted >= 0) &
        (true_pos_sorted < len(sorted_classes)) &
        (pred_pos_sorted >= 0) &
        (pred_pos_sorted < len(sorted_classes))
    )

    valid_indices = np.where(valid)[0]

    valid[valid_indices] &= (
        (sorted_classes[true_pos_sorted[valid_indices]] == y_true[valid_indices]) &
        (sorted_classes[pred_pos_sorted[valid_indices]] == y_pred[valid_indices])
    )

    true_idx = sort_order[true_pos_sorted[valid]]
    pred_idx = sort_order[pred_pos_sorted[valid]]

    flat_idx = true_idx * len(classes) + pred_idx

    return np.bincount(
        flat_idx,
        minlength=len(classes) * len(classes)
    ).reshape(len(classes), len(classes))


# ---------- NaN mean/variance: loop vs NumPy ---------- #

def _loop_nanmean_nanvar(X):
    """
    Mostly loop-based column-wise NaN-aware mean and variance.
    Kept only for benchmarking against vectorized implementation.
    """
    X = np.asarray(X, dtype=float)

    means = np.zeros(X.shape[1], dtype=float)
    variances = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        total = 0.0
        count = 0

        for i in range(X.shape[0]):
            value = X[i, j]

            if not np.isnan(value):
                total += value
                count += 1

        if count == 0:
            means[j] = np.nan
            variances[j] = np.nan
            continue

        mean_value = total / count
        means[j] = mean_value

        squared_error_total = 0.0

        for i in range(X.shape[0]):
            value = X[i, j]

            if not np.isnan(value):
                diff = value - mean_value
                squared_error_total += diff * diff

        variances[j] = squared_error_total / count

    return means, variances


def _vectorized_nanmean_nanvar(X):
    """
    Vectorized column-wise NaN-aware mean and variance using sums and squared sums.
    """
    X = np.asarray(X, dtype=float)

    valid = ~np.isnan(X)
    counts = np.sum(valid, axis=0)

    X_zeroed = np.where(valid, X, 0.0)

    sums = np.sum(X_zeroed, axis=0)
    squared_sums = np.sum(X_zeroed ** 2, axis=0)

    means = np.divide(
        sums,
        counts,
        out=np.full(X.shape[1], np.nan, dtype=float),
        where=counts != 0
    )

    variances = np.divide(
        squared_sums,
        counts,
        out=np.full(X.shape[1], np.nan, dtype=float),
        where=counts != 0
    ) - means ** 2

    variances = np.maximum(variances, 0.0)

    return means, variances


# ---------- Ensemble voting: loop vs vectorized ---------- #

def _loop_hard_vote(votes, classes):
    """
    Loop-based hard voting.

    votes shape: (n_samples, n_estimators)
    """
    predictions = []

    for row in votes:
        counts = np.asarray([np.sum(row == cls) for cls in classes])
        predictions.append(classes[np.argmax(counts)])

    return np.asarray(predictions)


def _vectorized_hard_vote(votes, classes):
    """
    Vectorized hard voting using indexed accumulation.
    """
    votes = np.asarray(votes)
    classes = np.asarray(classes)

    sort_order = np.argsort(classes)
    sorted_classes = classes[sort_order]

    flat_votes = votes.ravel()
    flat_pos_sorted = np.searchsorted(sorted_classes, flat_votes)

    valid = (
        (flat_pos_sorted < len(sorted_classes)) &
        (sorted_classes[flat_pos_sorted] == flat_votes)
    )

    if not np.all(valid):
        raise ValueError("votes contain classes not present in classes")

    flat_pos_original = sort_order[flat_pos_sorted]

    vote_counts = np.zeros((votes.shape[0], len(classes)), dtype=int)
    rows = np.repeat(np.arange(votes.shape[0]), votes.shape[1])

    np.add.at(vote_counts, (rows, flat_pos_original), 1)

    return classes[np.argmax(vote_counts, axis=1)]


# ---------- Tree split scoring: loop vs vectorized ---------- #

def _gini_from_counts(counts):
    """
    Compute Gini impurity from class counts.
    """
    total = np.sum(counts)

    if total <= 0:
        return 0.0

    probs = counts / total
    return 1.0 - np.sum(probs ** 2)


def _loop_best_split_single_feature(feature, y, sample_weight, classes):
    """
    Loop-based split scoring for one feature.

    This is kept only as a baseline for benchmarking the vectorized tree logic.
    """
    feature = np.asarray(feature, dtype=float)
    y = np.asarray(y)
    sample_weight = np.asarray(sample_weight, dtype=float)
    classes = np.asarray(classes)

    unique_values = np.unique(feature)

    if unique_values.size <= 1:
        return np.nan, 0.0

    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

    total_counts = np.asarray([
        np.sum(sample_weight[y == cls]) for cls in classes
    ])

    parent_impurity = _gini_from_counts(total_counts)
    total_weight = np.sum(sample_weight)

    best_threshold = np.nan
    best_gain = -np.inf

    for threshold in thresholds:
        left_mask = feature <= threshold
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            continue

        left_counts = np.asarray([
            np.sum(sample_weight[left_mask & (y == cls)]) for cls in classes
        ])

        right_counts = total_counts - left_counts

        left_weight = np.sum(left_counts)
        right_weight = np.sum(right_counts)

        child_impurity = (
            left_weight * _gini_from_counts(left_counts) +
            right_weight * _gini_from_counts(right_counts)
        ) / max(total_weight, 1e-12)

        gain = parent_impurity - child_impity if False else parent_impurity - child_impurity

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def _vectorized_best_split_single_feature(feature, y, sample_weight, classes):
    """
    Vectorized split scoring for one feature using sorting and cumulative counts.
    """
    feature = np.asarray(feature, dtype=float)
    y = np.asarray(y)
    sample_weight = np.asarray(sample_weight, dtype=float)
    classes = np.asarray(classes)

    order = np.argsort(feature)

    sorted_values = feature[order]
    sorted_y = y[order]
    sorted_weights = sample_weight[order]

    sort_order = np.argsort(classes)
    sorted_classes = classes[sort_order]

    class_idx_sorted = np.searchsorted(sorted_classes, sorted_y)
    class_idx_original = sort_order[class_idx_sorted]

    n_classes = len(classes)

    one_hot_weighted = (
        np.eye(n_classes)[class_idx_original] *
        sorted_weights[:, None]
    )

    total_counts = np.sum(one_hot_weighted, axis=0)
    parent_impurity = _gini_from_counts(total_counts)
    total_weight = np.sum(total_counts)

    if sorted_values.size <= 1:
        return np.nan, 0.0

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
        return np.nan, 0.0

    left_probs = left_counts / np.maximum(left_weight[:, None], 1e-12)
    right_probs = right_counts / np.maximum(right_weight[:, None], 1e-12)

    left_gini = 1.0 - np.sum(left_probs ** 2, axis=1)
    right_gini = 1.0 - np.sum(right_probs ** 2, axis=1)

    child_impurity = (
        left_weight * left_gini +
        right_weight * right_gini
    ) / np.maximum(total_weight, 1e-12)

    gains = parent_impurity - child_impurity
    gains[~valid_splits] = -np.inf

    best_idx = int(np.argmax(gains))
    best_threshold = (sorted_values[best_idx] + sorted_values[best_idx + 1]) / 2.0

    return best_threshold, gains[best_idx]


def benchmark_vectorization(n_samples=50000, n_features=20, n_classes=4,
                            n_estimators=25, repeats=5, random_state=42,
                            save_csv=True):
    """
    Compare loop-based and vectorized implementations for core numerical work.

    This benchmark is separate from model benchmarking. It exists to provide
    evidence for the vectorization design requirement.
    """
    rng = np.random.default_rng(random_state)

    classes = np.arange(n_classes, dtype=int)

    y_true = rng.integers(0, n_classes, size=n_samples)
    y_pred = rng.integers(0, n_classes, size=n_samples)

    X = rng.normal(size=(n_samples, n_features))
    nan_mask = rng.random(size=X.shape) < 0.03
    X[nan_mask] = np.nan

    votes = rng.integers(
        0,
        n_classes,
        size=(n_samples, n_estimators)
    )

    feature = rng.normal(size=n_samples)
    split_y = rng.integers(0, n_classes, size=n_samples)
    sample_weight = np.ones(n_samples, dtype=float)

    benchmarks = [
        {
            "operation": "confusion_matrix",
            "loop_func": _loop_confusion_matrix,
            "vectorized_func": _vectorized_confusion_matrix,
            "args": (y_true, y_pred, classes),
            "n_samples": n_samples,
            "n_features": 0,
        },
        {
            "operation": "nanmean_nanvar",
            "loop_func": _loop_nanmean_nanvar,
            "vectorized_func": _vectorized_nanmean_nanvar,
            "args": (X,),
            "n_samples": n_samples,
            "n_features": n_features,
        },
        {
            "operation": "hard_voting",
            "loop_func": _loop_hard_vote,
            "vectorized_func": _vectorized_hard_vote,
            "args": (votes, classes),
            "n_samples": n_samples,
            "n_features": n_estimators,
        },
        {
            "operation": "tree_split_single_feature",
            "loop_func": _loop_best_split_single_feature,
            "vectorized_func": _vectorized_best_split_single_feature,
            "args": (feature, split_y, sample_weight, classes),
            "n_samples": n_samples,
            "n_features": 1,
        },
    ]

    results = []

    for item in benchmarks:
        loop_time, loop_result = _time_call(
            item["loop_func"],
            *item["args"],
            repeats=repeats
        )

        vectorized_time, vectorized_result = _time_call(
            item["vectorized_func"],
            *item["args"],
            repeats=repeats
        )

        speedup = loop_time / vectorized_time if vectorized_time > 0 else np.inf

        results.append({
            "operation": item["operation"],
            "n_samples": item["n_samples"],
            "n_features": item["n_features"],
            "loop_time": loop_time,
            "vectorized_time": vectorized_time,
            "speedup": speedup,
            "outputs_match": _results_close(loop_result, vectorized_result),
        })

    print_vectorization_results(results)

    if save_csv:
        save_vectorization_results(results)

    return results


def print_vectorization_results(results):
    """
    Print vectorization benchmark results.
    """
    if not results:
        print("No vectorization benchmark results to show.")
        return

    print("\nVECTORIZATION BENCHMARK")
    print("Loop-based baselines are kept only for comparison.")
    print("-" * 100)

    print(
        f"{'Operation':<28}"
        f"{'Samples':<12}"
        f"{'Features':<10}"
        f"{'Loop(s)':<12}"
        f"{'Vector(s)':<12}"
        f"{'Speedup':<12}"
        f"{'Match':<8}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['operation']:<28}"
            f"{r['n_samples']:<12}"
            f"{r['n_features']:<10}"
            f"{r['loop_time']:<12.6f}"
            f"{r['vectorized_time']:<12.6f}"
            f"{r['speedup']:<12.2f}"
            f"{str(r['outputs_match']):<8}"
        )


def save_vectorization_results(results):
    """
    Save vectorization benchmark results to CSV.
    """
    if not results:
        return

    VECTORIZATION_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operation",
        "n_samples",
        "n_features",
        "loop_time",
        "vectorized_time",
        "speedup",
        "outputs_match",
    ]

    with open(VECTORIZATION_OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved vectorization benchmark results to: {VECTORIZATION_OUT_PATH}")

def balanced_accuracy_multiclass(y_true, y_pred):
    """
    Compute multiclass balanced accuracy as the mean per-class recall.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    recalls = []

    for cls in classes:
        mask = y_true == cls
        recalls.append(np.mean(y_pred[mask] == cls) if np.any(mask) else 0.0)

    return float(np.mean(recalls)) if recalls else 0.0


def benchmark_model(model_name, model, X_train, y_train, X_test, y_test,
                    classes=None, n_chunks=5):
    """
    Benchmark one model under a test-then-train streaming protocol.

    The first chunk is used to initialise the model. Each later chunk is scored
    before calling partial_fit(), which gives a prequential streaming estimate
    rather than an optimistic score on data the model has already learned.

    Parameters
    ----------
    model_name : str
    model : classifier with partial_fit() and predict()
    X_train : ndarray
    y_train : ndarray
    X_test : ndarray
    y_test : ndarray
    classes : ndarray
        Known class labels for partial_fit initialisation.
    n_chunks : int, default=5

    Returns
    -------
    result : dict
        Benchmark metrics for one model.
    """
    if classes is None:
        classes = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))

    chunks = make_chunks(X_train, y_train, n_chunks=n_chunks)

    chunk_accuracies = []
    cumulative_true = []
    cumulative_pred = []

    fit_time = 0.0
    pred_time = 0.0
    fitted = False

    def train_stream():
        nonlocal fit_time, pred_time, fitted

        for X_chunk, y_chunk in chunks:
            if fitted:
                start = time.perf_counter()
                y_chunk_pred = model.predict(X_chunk)
                pred_time += time.perf_counter() - start

                chunk_accuracies.append(accuracy(y_chunk, y_chunk_pred))
                cumulative_true.extend(y_chunk.tolist())
                cumulative_pred.extend(y_chunk_pred.tolist())

            start = time.perf_counter()
            partial_fit_model(model, X_chunk, y_chunk, classes=classes)
            fit_time += time.perf_counter() - start
            fitted = True

        return model

    _, peak_memory = measure_peak_memory(train_stream)

    start = time.perf_counter()
    y_test_pred = model.predict(X_test)
    test_pred_time = time.perf_counter() - start
    pred_time += test_pred_time

    # Test metrics
    test_acc = accuracy(y_test, y_test_pred)
    test_macro_f1_val = macro_f1(y_test, y_test_pred)
    test_macro_prec = macro_precision(y_test, y_test_pred)
    test_macro_rec = macro_recall(y_test, y_test_pred)
    test_weighted_f1_val = weighted_f1(y_test, y_test_pred)
    test_bal_acc = balanced_accuracy_multiclass(y_test, y_test_pred)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)
    test_kappa = cohen_kappa(y_test, y_test_pred)

    # Prequential cumulative metrics
    cumulative_acc = (
        accuracy(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )
    prequential_macro_prec = (
        macro_precision(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )
    prequential_macro_rec = (
        macro_recall(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )
    prequential_macro_f1_val = (
        macro_f1(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )
    prequential_weighted_f1_val = (
        weighted_f1(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )
    prequential_bal_acc = (
        balanced_accuracy_multiclass(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )

    latency_ms = (test_pred_time / len(X_test)) * 1000 if len(X_test) > 0 else 0.0
    train_samples_per_sec = len(X_train) / fit_time if fit_time > 0 else 0.0
    test_samples_per_sec = len(X_test) / test_pred_time if test_pred_time > 0 else 0.0

    return {
        "dataset": None,
        "model": model_name,
        "random_state": None,
        "n_estimators": None,
        "chunks": n_chunks,
        "evaluated_chunks": len(chunk_accuracies),
        "fit_time": fit_time,
        "pred_time": pred_time,
        "total_time": fit_time + pred_time,
        "memory_mb": peak_memory,
        "prequential_cumulative_accuracy": cumulative_acc,
        "prequential_macro_precision": prequential_macro_prec,
        "prequential_macro_recall": prequential_macro_rec,
        "prequential_macro_f1": prequential_macro_f1_val,
        "prequential_weighted_f1": prequential_weighted_f1_val,
        "prequential_balanced_accuracy": prequential_bal_acc,
        "test_accuracy": test_acc,
        "test_macro_precision": test_macro_prec,
        "test_macro_recall": test_macro_rec,
        "test_macro_f1": test_macro_f1_val,
        "test_weighted_f1": test_weighted_f1_val,
        "test_balanced_accuracy": test_bal_acc,
        "test_mcc": test_mcc,
        "test_cohen_kappa": test_kappa,
        "latency_ms_per_sample": latency_ms,
        "train_samples_per_second": train_samples_per_sec,
        "test_samples_per_second": test_samples_per_sec,
    }


def save_results(results):
    """
    Save benchmark results to CSV with column selection.
    """
    if not results:
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Combine performance and efficiency columns for output
    all_columns = sorted(set(PERFORMANCE_COLUMNS + EFFICIENCY_COLUMNS))
    
    # Filter to only columns present in results
    fieldnames = [col for col in all_columns if col in results[0].keys()]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved benchmark results to: {OUT_PATH}")


def print_results(results):
    """
    Print benchmark results in a readable table.
    """
    if not results:
        print("No benchmark results to show.")
        return

    print(
        f"{'Dataset':<10}"
        f"{'Seed':<8}"
        f"{'Model':<18}"
        f"{'Chunks':<8}"
        f"{'Eval':<6}"
        f"{'Fit(s)':<10}"
        f"{'Pred(s)':<10}"
        f"{'Total(s)':<10}"
        f"{'Mem(MB)':<10}"
        f"{'PreqAcc':<10}"
        f"{'TestAcc':<10}"
        f"{'MacroF1':<10}"
        f"{'BalAcc':<10}"
        f"{'MCC':<10}"
        f"{'Kappa':<10}"
        f"{'Latency(ms)':<12}"
        f"{'Train/s':<12}"
        f"{'Test/s':<12}"
    )
    print("-" * 180)

    for r in results:
        print(
            f"{str(r.get('dataset', 'N/A')):<10}"
            f"{str(r.get('random_state', 'N/A')):<8}"
            f"{r.get('model', 'N/A'):<18}"
            f"{r.get('chunks', 'N/A'):<8}"
            f"{r.get('evaluated_chunks', 'N/A'):<6}"
            f"{r.get('fit_time', 0.0):<10.4f}"
            f"{r.get('pred_time', 0.0):<10.4f}"
            f"{r.get('total_time', 0.0):<10.4f}"
            f"{r.get('memory_mb', 0.0):<10.4f}"
            f"{r.get('prequential_cumulative_accuracy', 0.0):<10.4f}"
            f"{r.get('test_accuracy', 0.0):<10.4f}"
            f"{r.get('test_macro_f1', 0.0):<10.4f}"
            f"{r.get('test_balanced_accuracy', 0.0):<10.4f}"
            f"{r.get('test_mcc', 0.0):<10.4f}"
            f"{r.get('test_cohen_kappa', 0.0):<10.4f}"
            f"{r.get('latency_ms_per_sample', 0.0):<12.4f}"
            f"{r.get('train_samples_per_second', 0.0):<12.2f}"
            f"{r.get('test_samples_per_second', 0.0):<12.2f}"
        )


# ...existing backwards-compatible functions...

def benchmark_streaming_model(model, X_train, y_train, X_test, y_test,
                              classes=None, chunk_size=100):
    """
    Backwards-compatible benchmark helper expected by older tests.
    """
    if classes is None:
        classes = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))

    chunks = make_chunks(X_train, y_train, chunk_size=chunk_size)
    chunk_accuracies = []
    cumulative_true = []
    cumulative_pred = []
    fit_time = 0.0
    predict_time = 0.0

    def train_stream():
        nonlocal fit_time, predict_time
        for X_chunk, y_chunk in chunks:
            start = time.perf_counter()
            partial_fit_model(model, X_chunk, y_chunk, classes=classes)
            fit_time += time.perf_counter() - start

            start = time.perf_counter()
            y_chunk_pred = model.predict(X_chunk)
            predict_time += time.perf_counter() - start

            chunk_accuracies.append(accuracy(y_chunk, y_chunk_pred))
            cumulative_true.extend(np.asarray(y_chunk).tolist())
            cumulative_pred.extend(np.asarray(y_chunk_pred).tolist())
        return model

    _, peak_memory_mb = measure_peak_memory(train_stream)

    start = time.perf_counter()
    y_test_pred = model.predict(X_test)
    predict_time += time.perf_counter() - start

    cumulative_accuracy = (
        accuracy(np.asarray(cumulative_true), np.asarray(cumulative_pred))
        if cumulative_true else 0.0
    )

    return {
        "fit_time": fit_time,
        "predict_time": predict_time,
        "total_time": fit_time + predict_time,
        "peak_memory_mb": peak_memory_mb,
        "mean_chunk_accuracy": float(np.mean(chunk_accuracies)) if chunk_accuracies else 0.0,
        "cumulative_accuracy": cumulative_accuracy,
        "test_accuracy": accuracy(y_test, y_test_pred),
        "n_chunks": len(chunks),
        "n_train_samples": len(y_train),
        "n_test_samples": len(y_test),
    }


def benchmark_streaming_dataset(dataset_name, path, chunk_size=100, random_state=42,
                                model_factories=None, test_size=0.25):
    """
    Backwards-compatible dataset-level streaming benchmark helper.
    """
    if model_factories is None:
        model_factories = {
            name: (lambda seed, model_name=name: build_models(seed)[model_name])
            for name in build_models(random_state).keys()
        }

    X_train, X_test, y_train, y_test, metadata = load_preprocessed_train_test(
        path,
        target_col=-1,
        test_size=test_size,
        random_state=random_state,
    )

    results = []
    for model_name, factory in model_factories.items():
        model = factory(random_state)
        row = benchmark_streaming_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            classes=metadata["classes"],
            chunk_size=chunk_size,
        )
        row["dataset"] = dataset_name
        row["model"] = model_name
        row["random_state"] = random_state
        results.append(row)

    return results


# ---------------- RUNNER AND OUTPUT ---------------- #

def run_streaming_benchmarks(datasets=DATASETS, n_chunks=5, save_csv=True,
                             random_states=(42,), n_estimators=10):
    """
    Run streaming benchmarks across datasets and models.

    Parameters
    ----------
    datasets : dict[str, Path]
        Dataset name to CSV path.
    n_chunks : int, default=5
        Number of training chunks.
    save_csv : bool, default=True
        Whether to save benchmark_results.csv.
    random_states : iterable[int], default=(42,)
        One or more seeds. Multiple seeds allow stability checks.
    n_estimators : int, default=10
        Number of estimators for ensemble models.

    Returns
    -------
    results : list[dict]
    """
    results = []

    print("\nSTREAMING MODEL BENCHMARK")
    print("Protocol: stratified holdout split + prequential test-then-train chunks")
    print("-" * 160)

    for dataset_name, path in datasets.items():
        path = Path(path)

        if not path.exists():
            print(f"{dataset_name} not found: {path}")
            continue

        for seed in random_states:
            try:
                X_train, X_test, y_train, y_test, metadata = load_preprocessed_train_test(
                    path,
                    target_col=-1,
                    test_size=0.25,
                    random_state=seed,
                )
            except ValueError as exc:
                print(f"{dataset_name} skipped for seed {seed}: {exc}")
                continue

            if len(np.unique(y_train)) < 2:
                print(f"{dataset_name} skipped: classification requires at least two training classes")
                continue

            if metadata["dropped_columns"]:
                print(f"{dataset_name}: dropped identifier columns {metadata['dropped_columns']}")

            if metadata["expanded_columns"]:
                print(f"{dataset_name}: expanded paired numeric columns {metadata['expanded_columns']}")

            models = build_models(random_state=seed, n_estimators=n_estimators)

            for model_name, model in models.items():
                row = benchmark_model(
                    model_name,
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    classes=metadata["classes"],
                    n_chunks=n_chunks,
                )
                row["dataset"] = dataset_name
                row["random_state"] = seed
                row["n_estimators"] = n_estimators if model_name != "DecisionTree" else 1
                results.append(row)

    print_results(results)

    if save_csv:
        save_results(results)

    return results


def main():
    run_streaming_benchmarks(
        DATASETS,
        n_chunks=5,
        save_csv=True,
        random_states=(42,),
        n_estimators=10,
    )

    benchmark_vectorization(
        n_samples=50000,
        n_features=50,
        n_classes=20,
        n_estimators=25,
        repeats=5,
        random_state=42,
        save_csv=True,
    )

if __name__ == "__main__":
    main()