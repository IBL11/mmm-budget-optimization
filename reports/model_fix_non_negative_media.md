# Model Fix: Non-Negative Media Coefficients

**Change:** media coefficients in the Ridge MMM are now constrained to be greater than or equal to zero.
**Reason:** the previous model gave Radio a negative effect, which would mean that spending more on Radio reduces sales.
**Files changed:** `src/bounded_ridge.py` (new), `notebooks/05_modelling.ipynb`. Notebooks 06 and 07 were re-run without code changes.

---

## 1. Summary

The previous Ridge model estimated a **negative coefficient for Radio (-792,643)**. Advertising cannot reduce sales, so the result was not believable. It made Radio's ROI **-7.30**. It also forced a workaround in the budget optimiser, where Radio had to be held at its current level.

The model now uses **Ridge regression with non-negative media coefficients**. The six media channels are constrained to have an effect of zero or more. External factors (price, competitors, exchange rate, and so on) can still have a positive or negative effect.

After the fix:

- **Radio's coefficient is 0.** The model finds no measurable effect from Radio, rather than a harmful one.
- The other five channels keep **the same ranking**.
- Test accuracy falls slightly (**R² 0.954 to 0.937, MAPE 1.02% to 1.18%**). This is expected when a constraint is added, and the model is now easier to interpret and defend.
- Residuals look healthier: **Durbin-Watson 2.29 to 2.02**, closer to the ideal value of 2.

---

## 2. The Problem

| Item | Previous model |
|---|---|
| Model | Ridge regression (`sklearn.linear_model.Ridge`), alpha = 10 |
| Radio coefficient (standardised) | **-792,643** |
| Radio ROI | **-7.30** (each LKR 1 on Radio "removes" LKR 7.30 of sales) |
| Effect on optimisation | Radio had to be held at its current level manually, because the optimiser would otherwise cut it only because of the wrong sign |

The Radio spend pattern overlaps with other channels and external factors. With only 36 monthly observations, an unconstrained regression can give one channel a negative weight to compensate for the others. This is a known problem in Marketing Mix Modelling. The usual solution is to require media effects to be non-negative.

---

## 3. The Fix

### Method

A small estimator, `NonNegativeMediaRidge`, was added in `src/bounded_ridge.py`. It solves the same Ridge problem as before:

```
minimise  ||y - Xb||^2 + alpha * ||b||^2
subject to  b_media >= 0
```

- The first six coefficients (the media `_hill` features) have a lower bound of 0.
- The external-factor coefficients have no bounds.
- As in sklearn's Ridge, the intercept is not penalised.
- The problem is solved with `scipy.optimize.lsq_linear` (bounded least squares).

The estimator behaves like sklearn's Ridge (`alpha`, `coef_`, `intercept_`, `fit`, `predict`). The rest of the pipeline therefore stays the same:

- Same `StandardScaler` step and the same feature order (6 media + 8 external factors).
- Same `TimeSeriesSplit(n_splits=3)` and `GridSearchCV` over alpha. Alpha = 10 was selected again.
- Same model files in `outputs/models/`, so `MMMPredictor`, notebooks 06 and 07, the CSV outputs and the dashboard need no code changes.

### Code changes

| File | Change |
|---|---|
| `src/bounded_ridge.py` | New: the `NonNegativeMediaRidge` estimator |
| `notebooks/05_modelling.ipynb`, imports cell | Added `from src.bounded_ridge import NonNegativeMediaRidge` |
| `notebooks/05_modelling.ipynb`, training cell | `Ridge()` replaced with `NonNegativeMediaRidge(n_positive=len(MEDIA_FEATURES))` |
| `notebooks/05_modelling.ipynb`, save cell | `model_type` changed to "Ridge Regression (non-negative media coefficients)" |

---

## 4. Results: Before vs After

### 4.1 Model accuracy (6-month holdout, Jul-Dec 2025)

| Metric | Before | After |
|---|---:|---:|
| MAE (LKR) | 891,421 | 1,028,780 |
| RMSE (LKR) | 1,121,793 | 1,307,957 |
| R² | 0.954 | 0.937 |
| MAPE | 1.02% | 1.18% |
| Durbin-Watson (test residuals) | 2.29 | 2.02 |
| Cross-validation RMSE (LKR) | 4,382,372 | 4,377,084 |
| Selected alpha | 10 | 10 |

The holdout error rises slightly, while the cross-validation error stays almost the same (slightly lower). The constraint therefore does not make the model generalise worse. It removes a coefficient that fitted the test months a little better but had no business meaning.

### 4.2 Media coefficients (standardised)

| Channel | Before | After |
|---|---:|---:|
| Google Search | 1,460,642 | 1,287,092 |
| TV | 1,104,297 | 992,336 |
| YouTube | 1,023,262 | 896,714 |
| Outdoor | 699,827 | 599,483 |
| Facebook/Instagram | 413,409 | 351,815 |
| **Radio** | **-792,643** | **0** |

External factors keep their direction. Promotions, seasonality and economic sentiment remain the strongest positive drivers, and competitor pressure remains negative.

### 4.3 Channel ROI (notebook 06)

| Channel | ROI before | ROI after | Marginal ROI before | Marginal ROI after |
|---|---:|---:|---:|---:|
| YouTube | 4.85 | 4.25 | 2.39 | 2.09 |
| Google Search | 4.38 | 3.86 | 2.15 | 1.89 |
| TV | 2.60 | 2.33 | 1.30 | 1.16 |
| Outdoor | 2.38 | 2.04 | 1.16 | 1.00 |
| Facebook/Instagram | 1.25 | 1.07 | 0.62 | 0.53 |
| Radio | -7.30 | 0.00 | -3.53 | 0.00 |

- The ranking of the five channels with an effect is unchanged.
- ROI values are a little lower for every channel. The negative Radio term most likely pushed the other channels' estimates up to compensate.
- **Base sales** (sales that would happen without advertising) move from **57.5% to 50.2%** of predicted sales.

### 4.4 Budget optimisation (notebook 07, 2025 budget LKR 240.3M, limits ±30%)

| Channel | Change before | Change after |
|---|---:|---:|
| Google Search | +30% | +30% |
| YouTube | +30% | +30% |
| TV | -4% | +6% |
| Outdoor | -11% | -7% |
| Facebook/Instagram | -30% | -30% |
| Radio | held (0%) | -30% |
| **Expected uplift (same budget)** | **+LKR 20.1M (+1.9%)** | **+LKR 24.7M (+2.4%)** |

- Radio no longer needs a manual rule. Because it shows no measurable effect, the optimiser moves part of its budget to channels that do.
- The main recommendation is unchanged: move budget from Facebook/Instagram to Google Search and YouTube.

| Budget scenario | Sales, current split (LKR M) | Sales, optimised split (LKR M) |
|---|---:|---:|
| -20% | 989.8 | 1,006.6 |
| -10% | 1,018.6 | 1,042.0 |
| 0% | 1,044.5 | 1,069.2 |
| +10% | 1,067.8 | 1,089.9 |
| +20% | 1,089.0 | 1,103.8 |

With a 10% smaller budget, the optimised split gives about the same sales as today's full budget with the current split (LKR 1,042.0M vs 1,044.5M).

---

## 5. How to Read the Radio Result

A coefficient of zero means **the model cannot detect a sales effect from Radio in this dataset**. It does **not** prove that Radio has no value. With 36 months of data and spend patterns that overlap with other channels, a small Radio effect may be impossible to separate. Before cutting Radio heavily, test it in practice (for example a regional on/off test), or reduce it gradually and watch sales.

---

## 6. Limitations

- **Small sample:** 36 monthly observations, so all estimates carry uncertainty.
- **Multicollinearity:** the price index and inflation proxy move together (VIF about 100), so their individual effects should not be interpreted separately.
- **Observational data:** MMM shows association, not proven cause and effect.
- **Bounded at zero:** the constraint prevents impossible negative effects, but it cannot tell apart "no effect" and "a small effect the data cannot detect".

---

## 7. How to Reproduce

1. Make sure `src/bounded_ridge.py` exists.
2. Run `notebooks/05_modelling.ipynb` (Restart and Run All). This saves the new model to `outputs/models/`.
3. Run `notebooks/06_roi_analysis.ipynb`, then `notebooks/07_budget_optimisation.ipynb`.
4. Check that `radio_spend_lkr_hill` is 0 and all other media coefficients are positive in `outputs/tables/model_coefficients.csv`.

All 19 project tests pass with the new model (`pytest tests`).

**Note:** the metrics in section 1 of `model_evaluation_and_handoff.md` describe the previous model, so they should be updated to the values in section 4.1 above.
