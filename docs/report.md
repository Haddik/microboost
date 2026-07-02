# MicroBoost: reproducing gradient boosting from scratch

A short technical report accompanying the code in this repository.

## 1. Motivation

Gradient boosted decision trees (GBDT) remain the strongest off-the-shelf
method for medium-sized tabular problems. The goal of this project is
educational and reproductive: implement the algorithm from first principles in
NumPy and quantify how close a minimal, readable implementation gets to a
mature library (scikit-learn) under identical hyperparameters.

## 2. Method

### 2.1 Base learner

The base learner is a CART regression tree (`microboost/tree.py`). At each node
we search greedily for the `(feature, threshold)` pair that maximises the
reduction in sum of squared errors (SSE):

```
gain = SSE(parent) − [ SSE(left) + SSE(right) ]
```

Candidate thresholds are the midpoints between consecutive sorted feature
values. Using prefix sums of `y` and `y²`, the SSE of every candidate split is
evaluated in `O(1)`, so a feature costs `O(n log n)` (dominated by the sort).
Depth, minimum samples per split, and minimum samples per leaf act as
regularisers.

### 2.2 Boosting loop

We follow Friedman's stagewise additive procedure (`microboost/boosting.py`):

1. Initialise with the constant that minimises the loss,
   `F₀ = argmin_c Σ L(yᵢ, c)`.
2. For each stage `m = 1 … M`:
   - compute pseudo-residuals `rᵢ = −[∂L/∂F]_{F=Fₘ₋₁}`,
   - fit a regression tree `hₘ` to `r`,
   - update `Fₘ = Fₘ₋₁ + η · hₘ`.

The learning rate `η` (shrinkage) damps each step. Optional stochastic row
subsampling (`subsample < 1`) fits each tree on a random subset, which reduces
variance.

### 2.3 Losses

| loss           | `F₀`                    | negative gradient      | output          |
|----------------|-------------------------|------------------------|-----------------|
| Squared error  | mean(y)                 | `y − F`                | `F`             |
| Logistic       | log-odds of base rate   | `y − sigmoid(F)`       | `sigmoid(F)`    |

The logistic sigmoid is implemented in a numerically stable, branch-by-sign
form to avoid overflow for large `|F|`.

## 3. Experimental setup

- **Datasets.** Diabetes (regression, 442×10) and Breast Cancer (binary
  classification, 569×30). Both ship with scikit-learn, so experiments run
  offline and are fully reproducible.
- **Protocol.** A single 75/25 train/test split with `random_state=0`
  (stratified for classification). Identical hyperparameters for MicroBoost and
  the scikit-learn reference.
- **Metrics.** RMSE, MAE, R² for regression; accuracy and log-loss for
  classification.

## 4. Results

### Regression (Diabetes, `M=200, η=0.05, depth=2`)

| model         |   RMSE |    MAE |    R² |
|---------------|-------:|-------:|------:|
| MicroBoost    | 59.541 | 46.984 | 0.286 |
| scikit-learn  | 59.502 | 46.981 | 0.287 |

The gap is +0.039 RMSE (≈0.07%), i.e. statistically indistinguishable from the
reference at this configuration.

### Classification (Breast Cancer, `M=150, η=0.1, depth=3`)

| model         | accuracy | log-loss |
|---------------|---------:|---------:|
| MicroBoost    |    0.951 |    0.183 |
| scikit-learn  |    0.951 |    0.218 |

Accuracy is identical; MicroBoost's log-loss is slightly lower on this split,
though a single split is noisy and this should not be over-interpreted.

### Learning rate and overfitting

`experiments/plot_learning_curves.py` sweeps the learning rate and tracks test
RMSE against the number of trees (see `docs/learning_curves.png`). The results
reproduce the textbook picture:

- Large `η` (0.2) reaches a low error quickly but then **overfits**: test RMSE
  climbs as more trees are added.
- Small `η` (0.02) converges more slowly but to a slightly better minimum and
  is far more stable.

This is the empirical justification for combining a small learning rate with
early stopping — exactly what `staged_predict` is provided for.

## 5. Limitations and future work

- Leaves store the mean pseudo-residual. Replacing this with a per-leaf Newton
  step (second-order, as in TreeBoost/XGBoost) should tighten the classification
  log-loss.
- No categorical or missing-value handling.
- Single-split evaluation; k-fold cross-validation would give tighter estimates.

## 6. Reproducibility

```bash
pip install -e ".[experiments,dev]"
pytest -q                                   # 16 tests
python experiments/run_regression.py        # Table in §4
python experiments/run_classification.py    # Table in §4
python experiments/plot_learning_curves.py  # Figure
```

## References

1. J. H. Friedman. *Greedy Function Approximation: A Gradient Boosting
   Machine.* Annals of Statistics, 29(5), 2001.
2. T. Hastie, R. Tibshirani, J. Friedman. *The Elements of Statistical
   Learning*, 2nd ed., Springer, 2009 — chapter 10.
