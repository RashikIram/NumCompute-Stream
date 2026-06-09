![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)
# NumCompute

A modular, production-grade scientific computing toolkit built using **pure Python + NumPy**.

Built from scratch using only NumPy, without external ML libraries. This project implements core components of a machine learning framework, including data handling, preprocessing, sorting/search algorithms, statistics, evaluation metrics, and pipeline abstraction — all from scratch with a strong focus on **vectorisation, numerical stability, and clean software design**.


## 🚀 Features

- 📥 **Data I/O**
  - CSV reader with missing value handling

- 🧼 **Preprocessing**
  - StandardScaler, MinMaxScaler
  - OneHotEncoder for categorical data

- 🔍 **Sorting & Searching**
  - Stable sorting, multi-key sorting
  - Top-k using `argpartition`
  - Quickselect (k-th smallest)
  - Binary search

- 🏆 **Ranking**
  - Ranking with tie handling (average, dense, ordinal)
  - Percentile computation

- 📊 **Statistics**
  - Mean, variance, std (NaN-safe)
  - Quantiles

- 📏 **Metrics**
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix, MSE

- ⚡ **Optimisation**
  - Finite-difference gradients
  - Jacobian estimation

- 🔗 **Pipeline API**
  - Transformer-based design
  - Sequential pipelines and feature unions

- ⏱ **Benchmarking**
  - Compare vectorised NumPy vs Python loops


## 📁 Project Structure
```bash
NumCompute/
├── numcompute/ # Core library
├── tests/ # Unit tests
├── demo/ # Demo scripts / notebook
├── README.md
├── pyproject.toml`
```


## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repo-link>
cd NumCompute
```

Install the package:

```bash
pip install -e .
```
Run the benchmark to get the results

```bash
python -m demo.
```
🧪 Testing

Run tests:

```bash
pip install pytest
pytest tests/
```

## 🧠 Design Principles
- Vectorisation-first: Avoid Python loops where possible
- Numerical Stability: Handles NaNs, overflow/underflow
- Modular API: Consistent `fit`, `transform`, `predict` interface
- Reusability: Components work independently and in pipelines


## 👥 Authors
- Rashik Iram Chowdhury
- Sheikh Muhammad Aarib Azfar
- Syeda Farhat Tarannum
- Sanzana Mahrukh Hassan


