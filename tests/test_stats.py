import warnings
import numpy as np
import pytest

from numcompute_stream.stats import (
    mean,
    median,
    std,
    minimum,
    maximum,
    variance,
    histogram,
    quantiles,
    describe,
    StreamingStats,
)


# ---------------- BATCH STATS TESTS ---------------- #

def test_mean_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(mean(X), 3.0)


def test_median_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(median(X), 3.0)


def test_std_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(std(X), np.std(X))


def test_variance_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert np.isclose(variance(X), np.var(X))


def test_minimum_maximum_normal_case():
    X = np.array([1, 2, 3, 4, 5])
    assert minimum(X) == 1
    assert maximum(X) == 5


def test_constant_array_statistics():
    X = np.array([5, 5, 5, 5])

    assert mean(X) == 5
    assert median(X) == 5
    assert variance(X) == 0
    assert std(X) == 0


def test_single_value_statistics():
    X = np.array([42])

    assert mean(X) == 42
    assert median(X) == 42
    assert variance(X) == 0
    assert std(X) == 0


def test_stats_ignore_nan_values():
    X = np.array([1, 2, np.nan, 4, 5])

    assert np.isclose(mean(X), np.nanmean(X))
    assert np.isclose(median(X), np.nanmedian(X))
    assert np.isclose(std(X), np.nanstd(X))
    assert np.isclose(variance(X), np.nanvar(X))


def test_axis_wise_mean_axis_0():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mean(X, axis=0)

    assert np.allclose(result, np.array([2.5, 3.5, 4.5]))


def test_axis_wise_mean_axis_1():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = mean(X, axis=1)

    assert np.allclose(result, np.array([2.0, 5.0]))


def test_axis_wise_std():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    result = std(X, axis=0)

    assert np.allclose(result, np.std(X, axis=0))


def test_quantiles_normal_case():
    X = np.array([1, 2, 3, 4, 5])

    result = quantiles(X, [25, 50, 75])

    assert np.allclose(result, np.array([2, 3, 4]))


def test_quantiles_boundary_values():
    X = np.array([1, 2, 3, 4, 5])

    assert quantiles(X, 0) == 1
    assert quantiles(X, 100) == 5


def test_quantiles_ignore_nan_values():
    X = np.array([1, 2, np.nan, 4, 5])

    result = quantiles(X, [25, 50, 75])
    expected = np.nanpercentile(X, [25, 50, 75])

    assert np.allclose(result, expected)


def test_quantiles_invalid_low_value():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        quantiles(X, -10)


def test_quantiles_invalid_high_value():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        quantiles(X, 110)


def test_histogram_basic():
    X = np.array([1, 2, 3, 4, 5])

    counts, bins = histogram(X, bins=2)

    assert counts.sum() == len(X)
    assert len(bins) == 3


def test_histogram_empty_array_allowed():
    counts, bins = histogram([], bins=5)

    assert counts.sum() == 0
    assert len(counts) == 5
    assert len(bins) == 6


def test_histogram_invalid_bins():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        histogram(X, bins=0)


def test_empty_array_raises_error():
    with pytest.raises(ValueError):
        mean([])


def test_non_numeric_input_raises_error():
    with pytest.raises(ValueError):
        mean(["a", "b", "c"])


def test_none_input_raises_error():
    with pytest.raises(ValueError):
        mean(None)


def test_describe_contains_expected_keys():
    X = np.array([1, 2, 3, 4, 5])

    result = describe(X)

    expected_keys = {"mean", "median", "variance", "std", "min", "max"}

    assert set(result.keys()) == expected_keys


def test_describe_values_are_correct():
    X = np.array([1, 2, 3, 4, 5])

    result = describe(X)

    assert np.isclose(result["mean"], 3.0)
    assert np.isclose(result["median"], 3.0)
    assert np.isclose(result["variance"], np.var(X))
    assert np.isclose(result["std"], np.std(X))
    assert result["min"] == 1
    assert result["max"] == 5


def test_describe_axis_wise():
    X = np.array([
        [1, 2],
        [3, 4]
    ])

    result = describe(X, axis=0)

    assert np.allclose(result["mean"], [2, 3])
    assert np.allclose(result["median"], [2, 3])
    assert np.allclose(result["variance"], [1, 1])


def test_all_nan_array_returns_nan():
    X = np.array([np.nan, np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        assert np.isnan(mean(X))
        assert np.isnan(median(X))
        assert np.isnan(std(X))
        assert np.isnan(variance(X))


# ---------------- STREAMING STATS TESTS ---------------- #

def test_streaming_stats_single_chunk_matches_numpy():
    X = np.array([1, 2, 3, 4, 5])

    stats = StreamingStats()
    stats.update_stats(X)
    result = stats.result()

    assert result["count"] == 5
    assert np.isclose(result["mean"], np.mean(X))
    assert np.isclose(result["variance"], np.var(X))
    assert np.isclose(result["std"], np.std(X))
    assert result["min"] == 1
    assert result["max"] == 5


def test_streaming_stats_multiple_chunks_matches_numpy():
    X1 = np.array([1, 2, 3])
    X2 = np.array([4, 5, 6])

    stats = StreamingStats()
    stats.update_stats(X1)
    stats.update_stats(X2)

    combined = np.concatenate([X1, X2])
    result = stats.result()

    assert result["count"] == 6
    assert np.isclose(result["mean"], np.mean(combined))
    assert np.isclose(result["variance"], np.var(combined))


def test_streaming_stats_ignores_nan_values():
    X1 = np.array([1, np.nan, 3])
    X2 = np.array([np.nan, 5])

    stats = StreamingStats()
    stats.update_stats(X1)
    stats.update_stats(X2)

    combined = np.array([1, 3, 5])
    result = stats.result()

    assert result["count"] == 3
    assert np.isclose(result["mean"], np.mean(combined))
    assert np.isclose(result["variance"], np.var(combined))


def test_streaming_stats_empty_chunk_does_not_change_state():
    stats = StreamingStats()
    stats.update_stats([])

    result = stats.result()

    assert result["count"] == 0
    assert np.isnan(result["mean"])
    assert np.isnan(result["variance"])


def test_streaming_stats_all_nan_chunk_does_not_change_state():
    stats = StreamingStats()
    stats.update_stats([np.nan, np.nan])

    result = stats.result()

    assert result["count"] == 0
    assert np.isnan(result["mean"])
    assert np.isnan(result["variance"])


def test_streaming_stats_reset():
    stats = StreamingStats()
    stats.update_stats([1, 2, 3])
    assert stats.result()["count"] == 3

    stats.reset()
    result = stats.result()

    assert result["count"] == 0
    assert np.isnan(result["mean"])


def test_streaming_stats_histogram():
    X = np.array([1, 2, 3, 4])

    stats = StreamingStats(bins=2)
    stats.update_stats(X)

    counts, bins = stats.histogram()

    assert counts.sum() == len(X)
    assert len(counts) == 2
    assert len(bins) == 3


def test_streaming_stats_quantiles():
    X1 = np.array([1, 2, 3])
    X2 = np.array([4, 5])

    stats = StreamingStats()
    stats.update_stats(X1)
    stats.update_stats(X2)

    result = stats.quantiles([25, 50, 75])

    assert np.allclose(result, np.percentile(np.array([1, 2, 3, 4, 5]), [25, 50, 75]))


def test_streaming_stats_invalid_bins_raises():
    with pytest.raises(ValueError):
        StreamingStats(bins=0)


def test_streaming_stats_non_numeric_chunk_raises():
    stats = StreamingStats()

    with pytest.raises(ValueError):
        stats.update_stats(["a", "b"])


def test_streaming_stats_quantiles_invalid_q_raises():
    stats = StreamingStats()
    stats.update_stats([1, 2, 3])

    with pytest.raises(ValueError):
        stats.quantiles(120)


def test_streaming_stats_no_store_values_histogram_raises():
    stats = StreamingStats(store_values=False)
    stats.update_stats([1, 2, 3])

    with pytest.raises(ValueError):
        stats.histogram()


def test_streaming_stats_no_store_values_quantiles_raises():
    stats = StreamingStats(store_values=False)
    stats.update_stats([1, 2, 3])

    with pytest.raises(ValueError):
        stats.quantiles(50)

# ---------------- STORE_VALUES=False STREAMING STATS TESTS ---------------- #

def test_streaming_stats_store_values_false_matches_numpy_over_chunks():
    X1 = np.array([1.0, 2.0, np.nan])
    X2 = np.array([4.0, 5.0])

    stats = StreamingStats(store_values=False)
    stats.update_stats(X1)
    stats.update_stats(X2)

    combined = np.array([1.0, 2.0, 4.0, 5.0])
    result = stats.result()

    assert result["count"] == len(combined)
    assert np.isclose(result["mean"], np.mean(combined))
    assert np.isclose(result["variance"], np.var(combined))
    assert np.isclose(result["std"], np.std(combined))
    assert result["min"] == np.min(combined)
    assert result["max"] == np.max(combined)


def test_streaming_stats_store_values_false_2d_columnwise_matches_numpy():
    X1 = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
    ])

    X2 = np.array([
        [5.0, 6.0],
        [np.nan, 8.0],
    ])

    stats = StreamingStats(store_values=False)
    stats.update_stats(X1)
    stats.update_stats(X2)

    combined = np.vstack([X1, X2])
    result = stats.result()

    assert np.allclose(result["mean"], np.nanmean(combined, axis=0))
    assert np.allclose(result["variance"], np.nanvar(combined, axis=0))
    assert np.allclose(result["std"], np.nanstd(combined, axis=0))
    assert np.allclose(result["min"], np.nanmin(combined, axis=0))
    assert np.allclose(result["max"], np.nanmax(combined, axis=0))
    assert np.array_equal(result["count"], np.array([3, 3]))


def test_streaming_stats_store_values_false_quantile_raises():
    stats = StreamingStats(store_values=False)
    stats.update_stats([1, 2, 3])

    with pytest.raises(ValueError):
        stats.quantile(0.5)


def test_streaming_stats_store_values_false_histogram_raises():
    stats = StreamingStats(store_values=False)
    stats.update_stats([1, 2, 3])

    with pytest.raises(ValueError):
        stats.histogram()