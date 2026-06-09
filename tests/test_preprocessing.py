
import numpy as np
import pytest

from numcompute.preprocessing import (
    Imputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    ColumnTransformer,
)


# ---------------- STANDARD SCALER TESTS ---------------- #

def test_standard_scaler_basic():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), [0, 0])
    assert np.allclose(np.std(X_scaled, axis=0), [1, 1])


def test_standard_scaler_constant_column():
    X = np.array([[5], [5], [5]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_standard_scaler_transform_before_fit():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.transform(np.array([[1, 2]]))


def test_standard_scaler_column_mismatch():
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[1, 2, 3]])

    scaler = StandardScaler()
    scaler.fit(X_train)

    with pytest.raises(ValueError):
        scaler.transform(X_test)


def test_standard_scaler_wrong_dimension_input():
    scaler = StandardScaler()

    with pytest.raises(ValueError):
        scaler.fit(np.array([1, 2, 3]))


def test_standard_scaler_partial_fit_matches_full_fit():
    X1 = np.array([[1, 2], [3, 4]], dtype=float)
    X2 = np.array([[5, 6], [7, 8]], dtype=float)
    X_all = np.vstack([X1, X2])

    stream_scaler = StandardScaler()
    stream_scaler.partial_fit(X1)
    stream_scaler.partial_fit(X2)

    batch_scaler = StandardScaler()
    batch_scaler.fit(X_all)

    assert np.allclose(stream_scaler.mean, batch_scaler.mean)
    assert np.allclose(stream_scaler.var, batch_scaler.var)
    assert np.allclose(stream_scaler.std, batch_scaler.std)


def test_standard_scaler_partial_fit_with_nan():
    X1 = np.array([[1, np.nan], [3, 4]], dtype=float)
    X2 = np.array([[5, 6], [np.nan, 8]], dtype=float)

    scaler = StandardScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert np.all(np.isfinite(scaler.mean))
    assert np.all(np.isfinite(scaler.std))


def test_standard_scaler_reset():
    scaler = StandardScaler()
    scaler.fit(np.array([[1, 2], [3, 4]]))
    scaler.reset()

    assert scaler.mean is None
    assert scaler.std is None


# ---------------- MINMAX SCALER TESTS ---------------- #

def test_minmax_scaler_basic():
    X = np.array([[1], [3], [5]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0.5], [1]])


def test_minmax_scaler_constant_column():
    X = np.array([[5], [5], [5]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled, [[0], [0], [0]])


def test_minmax_scaler_invalid_feature_range():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


def test_minmax_scaler_partial_fit_updates_min_max():
    X1 = np.array([[2], [4]])
    X2 = np.array([[1], [10]])

    scaler = MinMaxScaler()
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)

    assert scaler.min[0] == 1
    assert scaler.max[0] == 10

    X_scaled = scaler.transform(np.array([[1], [10]]))
    assert np.allclose(X_scaled, [[0], [1]])


def test_minmax_scaler_partial_fit_column_mismatch():
    scaler = MinMaxScaler()
    scaler.partial_fit(np.array([[1, 2]]))

    with pytest.raises(ValueError):
        scaler.partial_fit(np.array([[1, 2, 3]]))


def test_minmax_scaler_reset():
    scaler = MinMaxScaler()
    scaler.fit(np.array([[1], [2]]))
    scaler.reset()

    assert scaler.min is None
    assert scaler.max is None
    assert scaler.range is None


# ---------------- IMPUTER TESTS ---------------- #

def test_simple_imputer_mean():
    X = np.array([[1.0], [np.nan], [3.0]])
    imputer = Imputer(strategy="mean")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [2.0], [3.0]])


def test_simple_imputer_median():
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="median")
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [3.0], [5.0]])


def test_simple_imputer_constant():
    X = np.array([[1.0], [np.nan], [5.0]])
    imputer = Imputer(strategy="constant", fill_value=-1)
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out, [[1.0], [-1.0], [5.0]])


def test_simple_imputer_invalid_strategy():
    with pytest.raises(ValueError):
        Imputer(strategy="mode")


def test_simple_imputer_all_nan_column():
    X = np.array([
        [np.nan, 1],
        [np.nan, 3],
        [np.nan, 5]
    ])

    imputer = Imputer(strategy="mean", fill_value=0)
    X_out = imputer.fit_transform(X)

    assert np.allclose(X_out[:, 0], [0, 0, 0])
    assert np.allclose(X_out[:, 1], [1, 3, 5])


def test_imputer_partial_fit_mean_updates_statistics():
    X1 = np.array([[1.0], [np.nan]])
    X2 = np.array([[3.0], [5.0]])

    imputer = Imputer(strategy="mean")
    imputer.partial_fit(X1)
    imputer.partial_fit(X2)

    assert np.allclose(imputer.statistics, [3.0])


def test_imputer_partial_fit_median_updates_statistics():
    X1 = np.array([[1.0], [5.0]])
    X2 = np.array([[9.0]])

    imputer = Imputer(strategy="median")
    imputer.partial_fit(X1)
    imputer.partial_fit(X2)

    assert np.allclose(imputer.statistics, [5.0])


def test_imputer_partial_fit_column_mismatch():
    imputer = Imputer(strategy="mean")
    imputer.partial_fit(np.array([[1, 2]], dtype=float))

    with pytest.raises(ValueError):
        imputer.partial_fit(np.array([[1, 2, 3]], dtype=float))


def test_imputer_reset():
    imputer = Imputer(strategy="mean")
    imputer.fit(np.array([[1.0], [2.0]]))
    imputer.reset()

    assert imputer.statistics is None


# ---------------- ONE HOT ENCODER TESTS ---------------- #

def test_onehot_encoder_basic():
    X = np.array([
        ["Red", "S"],
        ["Blue", "M"],
        ["Red", "M"],
        ["Green", "S"]
    ])

    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape == (4, 5)
    assert np.all((X_encoded == 0) | (X_encoded == 1))


def test_onehot_encoder_transform_before_fit():
    encoder = OneHotEncoder()

    with pytest.raises(ValueError):
        encoder.transform(np.array([["Red"]]))


def test_onehot_encoder_column_mismatch():
    X_train = np.array([["Red", "S"], ["Blue", "M"]])
    X_test = np.array([["Red"]])

    encoder = OneHotEncoder()
    encoder.fit(X_train)

    with pytest.raises(ValueError):
        encoder.transform(X_test)


def test_onehot_encoder_partial_fit_expands_categories():
    X1 = np.array([["Red"], ["Blue"]])
    X2 = np.array([["Green"]])

    encoder = OneHotEncoder()
    encoder.partial_fit(X1)

    assert len(encoder.categories[0]) == 2

    encoder.partial_fit(X2)

    assert set(encoder.categories[0]) == {"Blue", "Green", "Red"}

    X_encoded = encoder.transform(np.array([["Green"], ["Red"]]))
    assert X_encoded.shape == (2, 3)


def test_onehot_encoder_partial_fit_column_mismatch():
    encoder = OneHotEncoder()
    encoder.partial_fit(np.array([["A", "B"]]))

    with pytest.raises(ValueError):
        encoder.partial_fit(np.array([["A"]]))


def test_onehot_encoder_reset():
    encoder = OneHotEncoder()
    encoder.fit(np.array([["A"], ["B"]]))
    encoder.reset()

    assert encoder.categories is None


# ---------------- COLUMN TRANSFORMER TESTS ---------------- #

def test_column_transformer_mixed_data():
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert X_out.shape == (3, 3)


def test_column_transformer_with_nan_numeric():
    X = np.array([
        [25, "Male"],
        [np.nan, "Female"],
        [35, "Female"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    X_out = ct.fit_transform(X)

    assert np.all(np.isfinite(X_out.astype(float)))


def test_column_transformer_no_columns_selected():
    X = np.array([[1, "A"], [2, "B"]], dtype=object)
    ct = ColumnTransformer(num_cols=[], cat_cols=[])

    with pytest.raises(ValueError):
        ct.fit_transform(X)


def test_column_transformer_auto_detect_columns():
    X = np.array([
        [25, "Male"],
        [30, "Female"],
        [22, "Female"]
    ], dtype=object)

    ct = ColumnTransformer()
    ct.fit(X)

    assert ct.num_cols == [0]
    assert ct.cat_cols == [1]


def test_column_transformer_partial_fit_updates_encoder_categories():
    X1 = np.array([
        [25, "Male"],
        [30, "Female"]
    ], dtype=object)

    X2 = np.array([
        [35, "Other"]
    ], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    ct.partial_fit(X1)
    ct.partial_fit(X2)

    assert set(ct.encoder.categories[0]) == {"Female", "Male", "Other"}

    X_out = ct.transform(X2)
    assert X_out.shape == (1, 4)


def test_column_transformer_partial_fit_numeric_updates_scaler():
    X1 = np.array([[1, "A"], [3, "B"]], dtype=object)
    X2 = np.array([[5, "A"], [7, "B"]], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    ct.partial_fit(X1)
    ct.partial_fit(X2)

    assert np.allclose(ct.scaler.mean, [4.0])


def test_column_transformer_transform_before_fit_raises():
    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])

    with pytest.raises(ValueError):
        ct.transform(np.array([[1, "A"]], dtype=object))


def test_column_transformer_reset():
    X = np.array([[1, "A"], [2, "B"]], dtype=object)

    ct = ColumnTransformer(num_cols=[0], cat_cols=[1])
    ct.fit(X)
    ct.reset()

    assert ct.imputer.statistics is None
    assert ct.scaler.mean is None
    assert ct.encoder.categories is None
