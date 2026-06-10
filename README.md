![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Used-green)
![pytest](https://img.shields.io/badge/pytest-293%2B_tests-brightgreen)

# NumCompute-Stream

NumCompute-Stream is a lightweight scientific computing and streaming machine learning toolkit built from scratch using **pure Python, NumPy, and Matplotlib**.

This project extends the original **NumCompute** framework. The previous toolkit already included core scientific computing utilities such as custom CSV I/O, preprocessing, statistics, sorting/searching, ranking, optimisation helpers, metrics, and pipeline abstraction. This extended version adds **streaming classification**, a custom **decision tree classifier**, **tree-based ensembles**, **incremental preprocessing**, **streaming statistics**, **streaming and rolling metrics**, **prequential evaluation**, **benchmarking**, and **visualisation**.

The package is designed without external machine learning libraries and focuses on **vectorisation, numerical stability, modular software design, automated testing, and reproducible benchmarking**.

---

## Features

### Data I/O
- Custom CSV loading utilities
- Missing-value handling
- Mixed-type CSV support
- Dataset loading for demos and benchmarking

### Preprocessing
- `StandardScaler`
- `MinMaxScaler`
- `Imputer`
- `OneHotEncoder`
- `ColumnTransformer`
- Batch and streaming-style `partial_fit()` support
- Handling for NaNs, constant columns, all-NaN columns, column mismatches, and categorical expansion

### Sorting, Searching, and Ranking
- Stable sorting
- Multi-key sorting
- Top-k selection
- Quickselect
- Binary search
- Ranking with tie handling
- Average, dense, and ordinal ranking modes
- Percentile computation

### Statistics
- Mean, median, variance, and standard deviation
- Minimum and maximum
- Histograms
- Quantiles and percentiles
- Axis-wise computations
- NaN-safe statistical operations
- Streaming statistics with online mean and variance updates
- Optional stored-value support for exact quantiles and histograms

### Tree-Based Classification
- `DecisionTreeClassifier`
- Gini and entropy splitting
- Probability prediction with `predict_proba()`
- Sample-weight support
- Feature importance
- Reset support
- Streaming-style updates using `partial_fit()`

### Ensemble Learning
- `OnlineBaggingClassifier`
- `RandomForestClassifier`
- `RandomSubspaceClassifier`
- `ExtraTreesClassifier`
- `AdaBoostSAMMEClassifier`
- Hard and soft voting
- Multiclass classification support
- Incremental training with `partial_fit()`

### Metrics
- Accuracy
- Precision, recall, specificity
- F1-score and F-beta
- Macro, micro, and weighted classification metrics
- Balanced accuracy
- Confusion matrix
- Matthews correlation coefficient
- Cohen’s kappa
- Top-k accuracy
- Log loss and Brier score
- Binary and multiclass ROC-AUC
- MSE, RMSE, MAD, and MAPE
- Streaming and rolling-window metrics

### Optimisation
- Finite-difference gradients
- Jacobian estimation
- Numerical derivative utilities

### Mathematical Utilities
- Batch iteration helpers
- Euclidean, Manhattan, Cosine, and pairwise distances
- Stable sigmoid, softmax, and logsumexp helpers

### Pipeline API
- Transformer-based design
- Sequential pipelines
- Transformer-only workflows
- Single-estimator workflows
- Streaming pipelines using `partial_fit()`
- Unified `fit()`, `transform()`, `fit_transform()`, `predict()`, `predict_proba()`, and `score()` API

### Stream Training
- `StreamTrainer` for chunk-wise learning
- Prequential test-then-train evaluation
- First-chunk scoring skip when the model is not yet fitted
- Chunk accuracy and cumulative accuracy logging
- Timing and memory tracking

### Visualisation
- Metric-over-time plots
- Model comparison plots
- Class-distribution plots
- Class-distribution-over-time plots
- Missing-value plots
- Confusion matrix heatmaps
- Predictions vs ground truth
- Feature histograms
- Binary ROC curves
- One-vs-rest multiclass ROC curves

### Benchmarking
- Streaming benchmark using prequential test-then-train evaluation
- Stratified train-test split before chunking
- Iris and Sleep dataset support
- Predictive and efficiency metrics
- Runtime, memory, latency, and throughput tracking
- Vectorisation microbenchmarks against loop-based baselines

### Testing
- 293+ pytest cases
- Tests for original NumCompute utilities and the new streaming extension
- Coverage for normal workflows, invalid inputs, edge cases, streaming updates, metrics, preprocessing, stream trainer behaviour, visualisation, and benchmarking

---

## Project Structure

```bash
NumCompute-Stream/
├── numcompute_stream/
│   ├── __init__.py
│   ├── benchmarking.py
│   ├── ensemble.py
│   ├── io.py
│   ├── metrics.py
│   ├── models.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── rank.py
│   ├── sort_search.py
│   ├── stats.py
│   ├── stream.py
│   ├── tree.py
│   ├── utils.py
│   └── visualise.py
├── benchmark/
│   ├── benchmark_results.csv
│   └── vectorization_results.csv
├── data/
├── demo/
│   └── stream_demo.ipynb
├── tests/
│   ├── test_ensemble.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_stats.py
│   ├── test_stream.py
│   ├── test_tree.py
│   └── test_visualise.py
├── README.md
└── pyproject.toml
```

---

## Example Usage

```python
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer

X1 = np.array([[0.0], [1.0], [2.0]])
y1 = np.array([0, 0, 1])

X2 = np.array([[3.0], [4.0], [5.0]])
y2 = np.array([1, 1, 1])

model = DecisionTreeClassifier(max_depth=3)
trainer = StreamTrainer(model, classes=[0, 1])

trainer.fit_score_chunk(X1, y1, score_before_fit=True)
trainer.fit_score_chunk(X2, y2, score_before_fit=True)

print(trainer.get_metric_history("chunk_accuracy"))
```

---

## Running Tests

```bash
pytest
```

---

## Running the Demo

Open the demo notebook:

```bash
jupyter notebook demo/stream_demo.ipynb
```

The notebook demonstrates chunk-wise preprocessing, stream training, decision tree and ensemble comparison, streaming metrics, ROC visualisation, and benchmark outputs.

---

## Benchmarking

The benchmark compares the custom decision tree against Online Bagging, Random Forest, Random Subspace, Extra Trees, and AdaBoost SAMME using a stratified holdout split and prequential test-then-train chunks.

The benchmarking module records:

- prequential accuracy
- held-out test accuracy
- macro-F1 score
- fit time
- prediction time
- total runtime
- memory usage
- latency
- throughput

Vectorisation microbenchmarks are also included to compare selected NumPy implementations against loop-based baselines.

---

## Notes

The decision tree uses an accumulate-and-rebuild strategy for `partial_fit()`. This provides a clear and testable streaming simulation, but it is not a fully online Hoeffding tree. Exact quantiles and some histogram functionality require stored values, while running mean and variance can be updated online without storing all previous chunks.

---

## License

This project was developed for educational purposes as part of a programming for artificial intelligence assignment.
