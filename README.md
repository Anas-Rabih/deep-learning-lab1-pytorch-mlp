# Deep Learning – Lab 1: MLPs in PyTorch (Regression & Multi-class Classification)

Master MBD – Deep Learning – FST Tanger (Pr. L. Elaachak)

**Objective:** get familiar with PyTorch by building DNN/MLP architectures for a regression task (NYSE stock prices) and a multi-class classification task (machine predictive maintenance), following the course method: *prepare data → define model → train → evaluate → predict*.

**Tools:** Google Colab, PyTorch, scikit-learn, skorch (GridSearch), imbalanced-learn (SMOTE).

**Notebook:** `lab1.ipynb` (both parts, run top to bottom). Result tables and charts are in `results/`.

---

## Part 1 – Regression (NYSE)

### Data and EDA
- File used: `prices-split-adjusted.csv` (split-adjusted, so no fake jumps from stock splits).
- 851,264 rows, 501 stocks, 2010-01-04 to 2016-12-30. No missing values, no duplicates.
- Prices differ by orders of magnitude between stocks (min ≈ 1.6, max ≈ 1,578) and `close` is strongly right-skewed, whereas `log(close)` is almost bell-shaped. `volume` has extreme outliers, so `log1p` is used.
- `open`, `high`, `low`, `close` are almost perfectly correlated (≈ 1.0); `volume` is weakly correlated (-0.13).

### Problem set-up
- **Target:** next day's log closing price of the same stock.
- **Features (9):** log open/high/low/close/volume of today + log close of the previous 4 days.
- **Split:** chronological, train = 2010–2015 (723,143 rows), test = 2016 (125,616 rows). A random split would leak future days of the same stock into training.
- **Scaling:** `StandardScaler` fitted on the training set only.
- **Naive baseline** ("tomorrow = today"): R² = 0.99934, MAE (log) = 0.0123. Because of this baseline, R² is not very discriminating here (everything is ≈ 0.999), so MAE in log units is used to compare models. R² is also used instead of accuracy for the second curve, since accuracy is not defined for a continuous target.

### Models
- **Model 1:** MLP 9 → 64 → 32 → 1 (ReLU, Kaiming/Xavier init), `MSELoss`, SGD (lr 0.01, momentum 0.9), 20 epochs, batch 512 (course setup).
- **GridSearch** (skorch + `GridSearchCV`, 16 combinations, chronological validation inside the training years, not the test set): best = **Adam, lr 0.001, 20 epochs, 32 hidden units** (validation MSE 0.000683 in scaled units).
  - SGD with lr 0.001 was too slow; Adam with lr 0.01 was unstable. The top configurations differ by less than the expected noise.
- **Model 2:** MLP with the best parameters.

### Curves (Loss/Epochs and R²/Epochs, train and test)
- Model 1: loss falls quickly in the first ~5 epochs then flattens. Train and test curves go down in parallel (no overfitting). The test loss stays above the train loss, probably because 2016 is a different market period from the training years.
- Model 2: same level, but with a loss spike at epoch 4 (Adam instability), and noisier curves afterwards.

### Regularization comparison (test set, 2016)

| Model | R² | MAE (log) |
|---|---|---|
| Naive baseline | 0.99934 | 0.01230 |
| **L2 (wd = 1e-5)** | 0.99932 | **0.01254** |
| Early stopping (patience 7) | 0.99932 | 0.01268 |
| L2 (wd = 1e-4) | 0.99929 | 0.01299 |
| L1 (1e-6) | 0.99929 | 0.01302 |
| Model 2 (tuned, no regularization) | 0.99928 | 0.01326 |
| Early stopping (patience 3) | 0.99921 | 0.01391 |
| Dropout (p = 0.02) | 0.99871 | 0.02027 |
| Dropout (p = 0.1) | 0.98623 | 0.06796 |

Model 1 (untuned) reached R² 0.99927 and MAE 0.0133.

**Interpretation**
- Tuning did not improve on the first model: the model already captures the learnable signal, and the rest is mostly unpredictable daily noise.
- Light L2 gave the best result (≈ 5 % lower MAE than Model 2), but the differences between the top four models are small and each configuration was run with one seed, so the ranking is suggestive, not proven.
- Dropout hurt: p = 0.1 was very bad (MAE 0.068) and even p = 0.02 stayed worse than no regularization (MAE 0.020). The task needs high precision (≈ 1 % error), the network is small, and nothing was overfitting.
- Early stopping with patience 3 stopped at epoch 6 because of the noisy validation curve (MAE 0.0139). With patience 7 it ran to epoch 39 and became the second-best model (MAE 0.01268), so the first result was caused by the patience setting, not by early stopping itself. It also trains on less data, since a validation period is held out.
- No model beat the naive baseline: the models are underfitting rather than overfitting, so regularization has little to offer here.

---

## Part 2 – Multi-class classification (Machine Predictive Maintenance)

### Cleaning and pre-processing
- 10,000 rows, no missing values, no duplicates.
- **27 contradictory rows removed:** 9 rows with `Target = 1` but "No Failure", and 18 rows with `Target = 0` but "Random Failures". This also removes the "Random Failures" class (too few examples and by nature not predictable from sensor readings), leaving **5 classes**: No Failure (9,643), Heat Dissipation (112), Power (95), Overstrain (78), Tool Wear (45).
- Dropped `UDI`, `Product ID` (identifiers) and `Target` (derived from the label, would leak it).
- `Type` (L/M/H) one-hot encoded; the 5 numeric features standardized (scaler fitted on train only).

### EDA findings
- Extreme imbalance: 96.7 % of rows are "No Failure".
- `air_temp`/`proc_temp` correlated at 0.88; `rpm`/`torque` correlated at -0.88 (physical relation).
- Class signatures from the box plots: Heat Dissipation = high air temperature with low, tightly clustered rpm; Tool Wear = very high tool wear; Overstrain = high tool wear and high torque; Power Failure = extreme torque/rpm values.
- Failure rates differ by product type.

### Balancing
- Stratified 80/20 split **first** (train 7,978 / test 1,995). The test set keeps the real imbalance (1,929 No Failure, 22 / 19 / 16 / 9 for the failure classes).
- Training set only: random undersampling of "No Failure" to 2,000, then **SMOTE-NC** oversampling of every failure class to 2,000, giving 10,000 balanced training rows (the rarest class had only 36 real rows, so most of its training samples are synthetic).

### Models
- **Classifier 1:** MLP 8 → 64 → 32 → 5, `CrossEntropyLoss`, SGD (lr 0.01, momentum 0.9), 50 epochs, batch 64. The course example puts a `Softmax` layer before `CrossEntropyLoss`, which applies softmax twice; here the model outputs raw logits.
- **GridSearch** (skorch inside an imbalanced-learn pipeline so that balancing happens only on the training folds; 3-fold stratified CV; 16 combinations; score = macro-F1): best = **Adam, lr 0.001, 50 epochs, 128 hidden units**, CV macro-F1 = 0.435 ± 0.018. The top four configurations are within noise of each other; all best runs used 50 epochs (the edge of the grid).
- **Classifier 2:** MLP with the best parameters.

### Curves
- Train accuracy rises to ≈ 0.98–0.99 on the balanced training data; test accuracy rises to ≈ 0.95 but is jumpy from epoch to epoch (only 66 failure rows in the test set, so a small weight update flips many predictions). The test loss stays above the train loss without increasing, and part of this gap comes from the train/test distribution mismatch (balanced and partly synthetic vs realistic).

### Metrics (test set)

| | Train acc. | Test acc. | Test macro-F1 | Test macro-recall |
|---|---|---|---|---|
| Classifier 1 | 0.987 | 0.948 | 0.619 | 0.799 |
| Classifier 2 (tuned) | 0.991 | 0.939 | 0.606 | 0.865 |

Per class (Classifier 1, test): recall (sensitivity) 1.00 / 0.95 / 0.75 / 0.74 / 0.56 for Heat Dissipation / No Failure / Overstrain / Power / Tool Wear; precision 0.49 / 0.995 / 0.57 / 0.61 / 0.09.

- Failures are detected well (high sensitivity) but with **many false alarms** (4.7 % of healthy machines for Classifier 1, 5.9 % for Classifier 2). The main source is Tool Wear, whose precision is ≈ 0.09. Confusion between different failure types is small.
- **Why the Tool Wear false alarms (check on Classifier 2, test set):** the 62 healthy machines predicted as Tool Wear have a mean tool wear of 212.5 min (minimum 186), almost the same as the 9 real Tool Wear failures (mean 214.1, minimum 203), whereas correctly classified healthy machines average 104.9. So the model learned "very high tool wear → Tool Wear Failure", and healthy machines with very high tool wear look identical to failing ones on that feature. This shows an association; I did not test whether other features could separate them.
- Accuracy alone would be misleading: always predicting "No Failure" gives ≈ 96.7 %.
- Tuning shifted the balance towards recall (fewer missed failures, more false alarms) without improving macro-F1.

### Regularization comparison (test set)

| Model | Test acc. | Macro-F1 | Macro-recall | False alarms* | Missed failures** |
|---|---|---|---|---|---|
| Early stopping (patience 10) | 0.942 | **0.643** | 0.843 | 5.4 % | 12.1 % |
| L1 (1e-5) | 0.943 | 0.624 | 0.878 | 5.6 % | 7.6 % |
| Classifier 1 (baseline) | 0.948 | 0.619 | 0.799 | 4.7 % | 15.2 % |
| Classifier 2 (tuned) | 0.939 | 0.606 | 0.865 | 5.9 % | 7.6 % |
| Dropout (p = 0.2) | 0.918 | 0.600 | 0.882 | 8.1 % | 4.5 % |
| L2 (wd = 1e-4) | 0.934 | 0.598 | 0.842 | 6.4 % | 9.1 % |
| Dropout 0.2 + L2 1e-4 | 0.919 | 0.598 | 0.871 | 8.0 % | 6.1 % |
| L2 (wd = 1e-3) | 0.924 | 0.577 | 0.874 | 7.5 % | 6.1 % |
| Dropout (p = 0.4) | 0.894 | 0.561 | 0.909 | 10.8 % | 1.5 % |

\* healthy machines flagged as a failure  \*\* real failures predicted as "No Failure"

![Part 2 comparison](results/part2_comparison.png)

**Interpretation**
- Early stopping had the best macro-F1 (0.643) while training on 20 % less data, but the gap to L1 and the baseline (≈ 0.02) is within what a single flipped prediction on the small rare classes can cause.
- Stronger regularization (more dropout, larger weight decay) moves the model along a precision/recall trade-off: fewer missed failures, more false alarms, lower accuracy and lower macro-F1.
- All models fit the balanced training set (0.97–0.99) and drop to 0.56–0.64 macro-F1 on the real test set. Regularization narrowed this gap slightly but did not close it; I believe most of it comes from the difference between the balanced/synthetic training distribution and the real test distribution, which regularization cannot fix (not tested directly).
- The "best" model depends on the cost of errors: Dropout 0.2 misses the fewest failures, the baseline raises the fewest false alarms, early stopping/L1 are a balance.

---

## Limitations
- One random seed per run; no confidence intervals. Differences of about 0.02 in macro-F1 (Part 2) or about 0.0005 in MAE (Part 1) should not be over-interpreted.
- The Part 2 test set has only 9–22 samples per failure class, so per-class metrics are very noisy.
- GridSearch grids are small (16 combinations); the best epoch value was at the edge of the Part 2 grid.
- Part 1: a single next-day prediction task with a strong naive baseline; the MLP does not beat it.
- Part 2: removing the "Random Failures" class and the SMOTE-generated samples are design choices that affect the results.

## Personal synthesis

- I learned the PyTorch workflow for MLPs: `Dataset`/`DataLoader`, a `Module` with `Linear` layers and weight initialization, and the training loop (`zero_grad → forward → loss → backward → step`), plus how to choose the loss and output layer for regression (MSE, no activation) and multi-class classification (cross-entropy on raw logits).
- Evaluation design matters as much as the model: a chronological split for time series, scaling and balancing on the training set only, and a naive baseline to put the results in perspective.
- Accuracy and R² can be misleading (imbalanced classes, near-constant prices), so I used macro-F1, sensitivity, the confusion matrix and MAE.
- GridSearch with skorch works, but the top configurations were often within noise of each other; balancing must be done inside the cross-validation folds.
- Regularization is not automatically beneficial: it helped only slightly (light L2, early stopping) and hurt when the model was underfitting (dropout in Part 1).

## Repository structure
```
├── lab1.ipynb
├── README.md
└── results/
    ├── part1_regression_comparison.csv
    ├── part2_classifier_comparison.csv
    └── part2_comparison.png
```
