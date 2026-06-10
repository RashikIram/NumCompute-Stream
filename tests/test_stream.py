import numpy as np
import pytest

from numcompute_stream.stream import StreamTrainer


class DummyStreamingModel:
    def __init__(self):
        self.fitted = False
        self.majority_class = 0
        self.classes_seen = None
        self.partial_fit_calls = 0

    def partial_fit(self, X, y, classes=None):
        X = np.asarray(X)
        y = np.asarray(y)

        self.partial_fit_calls += 1
        self.fitted = True
        self.classes_seen = None if classes is None else np.asarray(classes)

        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X)
        return np.full(X.shape[0], self.majority_class)


class DummyModelNoClasses:
    def __init__(self):
        self.fitted = False

    def partial_fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model has not been fitted yet")
        return np.zeros(np.asarray(X).shape[0], dtype=int)


class BadModelNoPartialFit:
    def predict(self, X):
        return np.zeros(np.asarray(X).shape[0], dtype=int)


class BadModelNoPredict:
    def partial_fit(self, X, y):
        return self


class BadPredictionShapeModel:
    def partial_fit(self, X, y, classes=None):
        return self

    def predict(self, X):
        return np.array([0])


def test_stream_trainer_fit_chunk_updates_logs():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 0])

    trainer.fit_chunk(X, y)

    assert model.fitted is True
    assert model.partial_fit_calls == 1
    assert trainer.n_samples_seen == 3
    assert len(trainer.logs) == 1
    assert trainer.logs[0]["event"] == "fit"
    assert "peak_memory_mb" in trainer.logs[0]


def test_stream_trainer_passes_classes_when_supported():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=np.array([0, 1, 2]))

    trainer.fit_chunk([[1], [2]], [2, 2])

    assert np.array_equal(model.classes_seen, np.array([0, 1, 2]))


def test_stream_trainer_handles_model_without_classes_argument():
    model = DummyModelNoClasses()
    trainer = StreamTrainer(model, classes=[0, 1])

    trainer.fit_chunk([[1], [2]], [0, 1])

    assert model.fitted is True


def test_score_chunk_returns_accuracy_and_confusion_matrix():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    X = np.array([[0], [1], [2], [3]])
    y = np.array([1, 1, 0, 0])

    trainer.fit_chunk(X, y)
    result = trainer.score_chunk(X, y)

    assert "chunk_accuracy" in result
    assert "cumulative_accuracy" in result
    assert result["confusion_matrix"].shape == (2, 2)
    assert 0 <= result["chunk_accuracy"] <= 1


def test_cumulative_accuracy_across_score_chunks():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    trainer.fit_chunk([[0], [1]], [1, 1])

    result1 = trainer.score_chunk([[0], [1]], [1, 0])
    result2 = trainer.score_chunk([[2], [3]], [1, 1])

    assert np.isclose(result1["cumulative_accuracy"], 0.5)
    assert np.isclose(result2["cumulative_accuracy"], 0.75)


def test_fit_score_chunk_train_then_score():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    metrics = trainer.fit_score_chunk([[0], [1]], [1, 1])

    assert metrics["event"] == "score"
    assert trainer.n_samples_seen == 2
    assert len(trainer.logs) == 2


def test_get_metric_history_returns_only_score_metrics():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    trainer.fit_score_chunk([[0], [1]], [1, 1])
    trainer.fit_score_chunk([[2], [3]], [0, 0])

    history = trainer.get_metric_history("chunk_accuracy")

    assert len(history) == 2
    assert all(0 <= value <= 1 for value in history)


def test_reset_clears_logs_and_counts():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    trainer.fit_score_chunk([[0], [1]], [1, 1])
    trainer.reset()

    assert trainer.logs == []
    assert trainer.n_samples_seen == 0
    assert trainer.total_correct == 0
    assert trainer.total_scored == 0
    assert trainer.chunk_index == 0


def test_invalid_model_without_partial_fit_raises():
    with pytest.raises(ValueError):
        StreamTrainer(BadModelNoPartialFit())


def test_invalid_model_without_predict_raises():
    with pytest.raises(ValueError):
        StreamTrainer(BadModelNoPredict())


def test_invalid_X_dimension_raises():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([1, 2, 3]), np.array([0, 1, 0]))


def test_invalid_y_dimension_raises():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1], [2]]), np.array([[0], [1]]))


def test_sample_mismatch_raises():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    with pytest.raises(ValueError):
        trainer.fit_chunk(np.array([[1], [2]]), np.array([0]))


def test_score_before_fit_propagates_model_error():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model)

    with pytest.raises(ValueError):
        trainer.score_chunk(np.array([[1], [2]]), np.array([0, 1]))


def test_prediction_shape_mismatch_raises():
    model = BadPredictionShapeModel()
    trainer = StreamTrainer(model)

    trainer.fit_chunk(np.array([[1], [2]]), np.array([0, 1]))

    with pytest.raises(ValueError):
        trainer.score_chunk(np.array([[1], [2]]), np.array([0, 1]))


def test_track_memory_false_sets_memory_to_zero():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, track_memory=False)

    trainer.fit_chunk(np.array([[1], [2]]), np.array([0, 1]))

    assert trainer.logs[0]["current_memory_mb"] == 0
    assert trainer.logs[0]["peak_memory_mb"] == 0

# ---------------- PREQUENTIAL FIRST-CHUNK TESTS ---------------- #

def test_fit_score_chunk_prequential_first_chunk_skips_score_and_trains():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    result = trainer.fit_score_chunk(
        np.array([[0], [1]]),
        np.array([1, 1]),
        score_before_fit=True
    )

    assert result["event"] == "score_skipped"
    assert result["skipped"] is True
    assert model.fitted is True
    assert trainer.n_samples_seen == 2
    assert trainer.total_scored == 0
    assert trainer.total_correct == 0

    assert len(trainer.logs) == 2
    assert trainer.logs[0]["event"] == "score_skipped"
    assert trainer.logs[1]["event"] == "fit"


def test_fit_score_chunk_prequential_second_chunk_scores_before_training():
    model = DummyStreamingModel()
    trainer = StreamTrainer(model, classes=[0, 1])

    trainer.fit_score_chunk(
        np.array([[0], [1]]),
        np.array([1, 1]),
        score_before_fit=True
    )

    result = trainer.fit_score_chunk(
        np.array([[2], [3]]),
        np.array([0, 0]),
        score_before_fit=True
    )

    assert result["event"] == "score"
    assert result["chunk"] == 2

    # The model was trained only on the first chunk before scoring this chunk,
    # so it predicts class 1 and gets both class-0 samples wrong.
    assert np.isclose(result["chunk_accuracy"], 0.0)

    # After scoring, the second chunk is used for training.
    assert model.majority_class == 0
    assert trainer.n_samples_seen == 4

    assert [log["event"] for log in trainer.logs] == [
        "score_skipped",
        "fit",
        "score",
        "fit",
    ]
