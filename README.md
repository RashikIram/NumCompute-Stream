````markdown
![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Used-green)
![pytest](https://img.shields.io/badge/pytest-293%2B%20tests-brightgreen)

# NumCompute-Stream

A modular scientific computing and machine learning toolkit built using **pure Python, NumPy, and Matplotlib**.

This project extends the original **NumCompute** framework. The previous toolkit already included core scientific computing and machine learning utilities such as custom CSV I/O, preprocessing, statistics, sorting/searching, ranking, optimisation, metrics, and pipeline abstraction. This extended version adds **streaming classification**, **decision trees**, **tree-based ensembles**, **incremental preprocessing**, **streaming metrics**, **benchmarking**, and **visualisation**.

The project is built from scratch without external machine learning libraries, with a strong focus on **vectorisation, numerical stability, modular software design, testing, and reproducible benchmarking**.

---

## 🚀 Features

### 📥 Data I/O
- Custom CSV reader using `io.py`
- Missing-value handling
- Mixed-type CSV support
- Dataset loading for demos and benchmarking

### 🧼 Preprocessing
- `StandardScaler`
- `MinMaxScaler`
- `Imputer`
- `OneHotEncoder`
- `ColumnTransformer`
- Batch and streaming-style `partial_fit()` support
- Handling for NaNs, constant columns, all-NaN columns, column mismatches, and categorical expansion

### 🔍 Sorting and Searching
- Stable sorting
- Multi-key sorting
- Top-k selection using `argpartition`
- Quickselect for k-th smallest values
- Binary search

### 🏆 Ranking
- Ranking with tie handling
- Average, dense, and ordinal ranking modes
- Percentile computation
- Duplicate and NaN-aware behaviour

### 📊 Statistics
- Mean, median, variance, and standard deviation
- Minimum and maximum
- Histograms
- Quantiles and percentiles
- Axis-wise computations
- NaN-safe statistical operations
- Streaming statistics utilities

### 🌳 Tree-Based Classification
- `DecisionTreeClassifier`
- Gini and entropy splitting
- Probability prediction with `predict_proba()`
- Sample-weight support
- Reset support
- Streaming-style updates using `partial_fit()`

### 🌲 Ensemble Learning
- `OnlineBaggingClassifier`
- `RandomForestClassifier`
- `RandomSubspaceClassifier`
- `ExtraTreesClassifier`
- `AdaBoostSAMMEClassifier`
- Hard and soft voting
- Multiclass classification support
- Incremental training with `partial_fit()`

### 📏 Metrics
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
- ROC/AUC
- MSE, RMSE, MAD, and MAPE
- Rolling and streaming metrics

### ⚡ Optimisation
- Finite-difference gradients
- Jacobian estimation
- Numerical derivative utilities

### 🧮 Mathematical Utilities
- Batch iteration helpers
- Distance metrics
- Euclidean, Manhattan, Cosine, and pairwise distances
- Stable sigmoid, softmax, and logsumexp helpers

### 🔗 Pipeline API
- Transformer-based design
- Sequential pipelines
- Transformer-only workflows
- Single-estimator workflows
- Streaming pipelines using `partial_fit()`
- Unified `fit()`, `transform()`, `fit_transform()`, `predict()`, `predict_proba()`, and `score()` API

### 📈 Visualisation
- Metric-over-time plots
- Model comparison plots
- Class-distribution plots
- Class-distribution-over-time plots
- Missing-value plots
- Confusion matrix heatmaps
- Predictions vs ground truth
- Feature histograms
- ROC curve plots

### ⏱ Benchmarking
- Streaming benchmark using prequential test-then-train evaluation
- Stratified train-test split before chunking
- Iris and Sleep dataset support
- Separate predictive performance and computational efficiency tables
- Runtime, memory, latency, and throughput tracking

### 🧪 Testing
- 293+ pytest cases
- Tests for previous NumCompute modules and the new streaming extension
- Coverage for normal workflows, invalid inputs, edge cases, streaming updates, metrics, preprocessing, visualisation, and benchmarking

---

## 📁 Project Structure

```bash
NumCompute-Stream/
├── numcompute/                  # Core library
│   ├── benchmarking.py           # Streaming benchmark utilities
│   ├── ensemble.py               # Tree-based ensemble classifiers
│   ├── io.py                     # Custom CSV I/O
│   ├── metrics.py                # Batch and streaming metrics
│   ├── models.py                 # Baseline models
│   ├── pipeline.py               # Pipeline abstraction
│   ├── preprocessing.py          # Scalers, imputer, encoder, column transformer
│   ├── ranking.py                # Ranking and percentile utilities
│   ├── search.py                 # Searching algorithms
│   ├── sort.py                   # Sorting algorithms
│   ├── stats.py                  # Statistical utilities
│   ├── stream.py                 # Stream trainer and logging
│   ├── tree.py                   # Decision tree classifier
│   ├── utils.py                  # Mathematical and utility helpers
│   └── visualise.py              # Matplotlib visualisation helpers
├── benchmarking/                 # Benchmark scripts and output CSV files
├── data/                         # CSV datasets
├── demo/                         # Demo notebook
├── tests/                        # Unit tests
├── README.md
└── pyproject.toml
````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd NumCompute-Stream
```

Install the package in editable mode:

```bash
pip install -e .
```

Install testing dependencies:

```bash
pip install pytest
```

---

## ▶️ Running the Demo

Open the streaming demo notebook:

```bash
jupyter notebook demo/stream_demo.ipynb
```

The demo shows:

* Loading a CSV dataset using the custom I/O module
* Dataset inspection and visualisation
* Train-test splitting before stream chunking
* Chunk-wise incremental training using `.partial_fit()`
* Pipeline usage and logging
* Streaming metric visualisation
* Model comparison
* Predictions and confusion matrix visualisation

---

## ⏱ Running Benchmarks

Run the benchmark script:

```bash
python benchmarking/benchmarking.py
```

Or run it from Python:

```python
from numcompute.benchmarking import run_streaming_benchmarks

results = run_streaming_benchmarks(save_csv=True)
```

The benchmark creates:

```bash
benchmarking/benchmark_results.csv
benchmarking/benchmark_performance_results.csv
benchmarking/benchmark_efficiency_results.csv
```

The benchmark reports two views:

### Predictive Performance

* Accuracy
* Macro precision
* Macro recall
* Macro F1
* Weighted F1
* Balanced accuracy
* Matthews correlation coefficient
* Cohen’s kappa

### Computational Efficiency

* Fit time
* Prediction time
* Total time
* Peak memory
* Latency per sample
* Training throughput
* Test throughput

---

## 🧪 Running Tests

Run all tests:

```bash
pytest tests/
```

Run selected test files:

```bash
pytest tests/test_pipeline.py
pytest tests/test_preprocessing.py
pytest tests/test_benchmarking.py
pytest tests/test_metrics.py
```

---

## 🧠 Design Principles

* **Extension-first design**
  The project preserves the original NumCompute toolkit and adds a streaming machine learning layer on top.

* **NumPy-first implementation**
  Core functionality is implemented using NumPy without external machine learning libraries.

* **Vectorisation where appropriate**
  Array operations are used for preprocessing, statistics, metrics, and probability aggregation.

* **Streaming compatibility**
  Models, preprocessing components, metrics, and trainers support chunk-wise workflows using `partial_fit()`.

* **Consistent API**
  Components follow familiar method names such as `fit()`, `transform()`, `partial_fit()`, `predict()`, and `predict_proba()`.

* **Modular architecture**
  Each module has a focused responsibility, making the project easier to test, maintain, and extend.

* **Numerical stability**
  The implementation handles NaNs, zero divisions, constant columns, probability clipping, and invalid numerical inputs.

* **Benchmark transparency**
  Predictive performance and computational efficiency are reported separately to show accuracy-runtime trade-offs.

* **Test-driven robustness**
  The framework is validated with 293+ pytest cases covering both the original toolkit and the new streaming extension.

---

## 📌 Example Streaming Workflow

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler
from numcompute.tree import DecisionTreeClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42))
])

for X_chunk, y_chunk in stream_chunks:
    pipe.partial_fit(X_chunk, y_chunk, classes=classes)

predictions = pipe.predict(X_test)
```

---

## 👥 Author

**Rashik Iram Chowdhury**

```
```
