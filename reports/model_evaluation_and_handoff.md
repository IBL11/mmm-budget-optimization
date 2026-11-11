# Marketing Mix Modelling (MMM) – Model Evaluation & Handoff Guide

**Author**: Member 3 (Model Evaluation, Diagnostics, Validation & Downstream Handoff)  
**Branch**: `feature/marketing-mix-modeling`  
**Dataset**: `data/processed/feature_engineered_marketing_mix_data.csv` (36 monthly records: Jan 2023 – Dec 2025)  
**Target Variable**: `sales_revenue_lkr`  

---

## 1. Executive Summary & Evaluation Results

A 30-month training split (**Jan 2023 – Jun 2025**) and an untouched 6-month holdout test split (**Jul 2025 – Dec 2025**) were used to evaluate model generalization performance.

### 6-Month Holdout Test Performance Comparison

| Model | MAE (LKR) | RMSE (LKR) | $R^2$ Score | MAPE (%) | Durbin-Watson ($DW$) | Status / Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Baseline** | 4,000,405.83 | 6,028,313.61 | -0.337 | 4.31% | 1.15 | Naive benchmark fails out-of-sample |
| **OLS Regression** | 2,505,110.44 | 3,219,366.91 | 0.619 | 2.81% | 0.20 | Strong autocorrelation in errors |
| **Ridge Regression (non-negative media)** | **976,829.95** | **1,301,501.29** | **0.938** | **1.13%** | **2.22** | **Production Champion Model** |

### Key Takeaways:
1. **Predictive Accuracy ($R^2 = 93.8\%$, $\text{MAPE} = 1.13\%$)**: The regularized Ridge model (with non-negative media coefficients) accounts for ~93.8% of total sales revenue variance on completely unseen future months, holding average percentage forecasting error to just ~1.13%.
2. **Residual Diagnostic & Autocorrelation ($DW = 2.22$)**: Unlike standard OLS ($DW = 0.20$), which exhibited extreme positive autocorrelation due to unmodeled carryover, the Ridge regression model with Adstock and Hill Saturation yields a Durbin-Watson statistic near $2.0$. This confirms zero residual memory or serial correlation leakage.
3. **Diagnostic Visualizations Generated**:
   - `outputs/figures/model_actual_vs_predicted.png`: Monthly actual vs. predicted revenue timeline with holdout test shading and metrics callout.
   - `outputs/figures/model_residuals_diagnostic.png`: 3-panel residual analysis (time-series with $\pm 1.96\sigma$ bounds, residuals vs. fitted values, and error normality histogram).

> **Update:** Media coefficients are now constrained to be non-negative, which fixed the negative Radio coefficient. See `reports/model_fix_non_negative_media.md` for the before/after comparison.


> **Update 2 (multicollinearity fix):** `inflation_proxy_index` was removed from the model inputs because it moved almost together with `average_price_index` (VIF about 111 and 95). The model now uses **13 features**, the highest VIF is **6.3**, and the cross-validation error fell from LKR 4.38M to LKR 4.33M. The metrics above are for the updated model. See Section 5 for the full validation results.
---

## 2. Architecture & Modules Delivered

```
mmm-budget-optimization/
├── src/
│   ├── evaluation.py        <-- Member 3: R², RMSE, safe MAPE, DW, diagnostic plots
│   └── predict.py           <-- Member 3: Stateful inference, scaler preservation & scenario simulator
├── tests/
│   ├── test_evaluation.py   <-- Member 3: Pure unit tests for all metric functions & edge cases
│   └── test_predict.py      <-- Member 3: Schema preservation, adstock continuity, optimizer tests
├── outputs/
│   ├── figures/
│   │   ├── model_actual_vs_predicted.png
│   │   └── model_residuals_diagnostic.png
│   ├── models/
│   │   ├── ridge_mmm_baseline.joblib
│   │   ├── model_config.json
│   │   └── transformation_params.json
│   └── tables/
│       └── model_metrics.csv
```

---

## 3. Downstream Handoff Guide for ROI & Budget Optimization

The downstream subgroup (`06_roi_analysis.ipynb` and `07_budget_optimisation.ipynb`) should **NOT** manually apply `StandardScaler` or attempt to re-implement adstock carryover loops. Use the dedicated, tested `MMMPredictor` interface in `src/predict.py`.

### A. Initializing the Predictor
The predictor automatically loads the trained pipeline, transformation hyperparameters, and feature orders:
```python
from src.predict import MMMPredictor
import pandas as pd

# Automatically loads outputs/models artifacts
predictor = MMMPredictor()
```

### B. Standard Batch Prediction
Pass any DataFrame containing raw media spend columns and external economic features:
```python
df = pd.read_csv("data/processed/feature_engineered_marketing_mix_data.csv")

# Returns 1D array of sales revenue predictions in LKR
predicted_sales_lkr = predictor.predict(df)
```

### C. Stateful Adstock Carryover Continuation
Because geometric adstock has memory ($A_t = X_t + \lambda A_{t-1}$), when forecasting future months (e.g., Q1 2026), pass the final carryover state from December 2025:
```python
# Step 1: Predict on historical data and capture ending carryover
_, last_carryover = predictor.predict(df, return_carryover=True)

# Step 2: Use ending carryover as initial_carryover for future period
future_df = pd.DataFrame([...]) # Future planned spend & macroeconomic features
future_sales = predictor.predict(future_df, initial_carryover=last_carryover)
```

### D. Counterfactual "What-If" Budget Simulation (For ROI Analysis)
In `06_roi_analysis.ipynb`, easily simulate budget reallocation scenarios:
```python
# Example: What if TV spend increases by 20% and YouTube spend decreases by 10%?
scenario_results = predictor.predict_counterfactual(
    df_baseline=df.iloc[-6:], # Last 6 months
    spend_multipliers={
        "tv_spend_lkr": 1.20,
        "youtube_spend_lkr": 0.90
    }
)

display(scenario_results[[
    "baseline_predicted_sales_lkr",
    "scenario_predicted_sales_lkr",
    "delta_sales_lkr",
    "sales_lift_pct"
]])
```

### E. Single-Period Fast Predictor (For Budget Optimization Solvers)
In `07_budget_optimisation.ipynb`, numerical solvers like `scipy.optimize.minimize` (SLSQP / L-BFGS-B) need to evaluate thousands of candidate budget allocations:
```python
# Objective function to maximize sales (minimize negative sales)
def objective(spend_vector, external_factors, prior_carryover):
    predicted_sales, _ = predictor.predict_single_period(
        spend_vector=spend_vector, # [tv, fb_ig, google, youtube, radio, outdoor]
        external_factors=external_factors,
        initial_carryover=prior_carryover
    )
    return -predicted_sales
```

---

## 4. Feature Schema & Column Order Guarantees

The model was trained on **13 features** in the exact sequence below. `MMMPredictor` automatically guarantees this ordering even if input columns are shuffled:

1. `tv_spend_lkr_hill`
2. `facebook_instagram_spend_lkr_hill`
3. `google_search_spend_lkr_hill`
4. `youtube_spend_lkr_hill`
5. `radio_spend_lkr_hill`
6. `outdoor_spend_lkr_hill`
7. `average_price_index`
8. `promotion_discount_pct`
9. `competitor_pressure_index_0_100`
10. `economic_sentiment_index_0_100`
11. `lkr_usd_exchange_rate`
12. `seasonal_demand_index`
13. `public_holiday_count`

`inflation_proxy_index` is not a model input (removed to fix multicollinearity with `average_price_index`).



---

## 5. Model Validation (multicollinearity, residuals, uncertainty, alternative model)

These checks are in the last cells of `notebooks/05_modelling.ipynb`.

### 5.1 Multicollinearity (VIF)

| | Before (14 features) | After (13 features, inflation removed) |
|---|---:|---:|
| Highest VIF | 111.3 (`inflation_proxy_index`) | **6.3** (`economic_sentiment_index_0_100`) |
| `average_price_index` VIF | 94.9 | 6.1 |
| CV RMSE (TimeSeriesSplit, 3 folds) | LKR 4.38M | **LKR 4.33M** |
| Test R² / MAPE | 0.937 / 1.18% | **0.938 / 1.13%** |
| Channel ranking | YouTube > Google > TV > Outdoor > FB/IG > Radio | Same |

All VIF values are now below 10. Output: `outputs/tables/model_vif.csv`.

### 5.2 Residual diagnostics (training residuals)

| Test | What it checks | Value | Result |
|---|---|---:|---|
| Durbin-Watson | Autocorrelation | 2.18 | Pass (close to 2) |
| Ljung-Box (lag 3) | Autocorrelation over several months | p = 0.21 | Pass |
| Shapiro-Wilk | Normal errors | p = 0.64 | Pass |
| Breusch-Pagan | Constant error size | p = 0.15 | Pass |

The model's errors show no pattern over time, are close to normal and have a constant size, so the regression assumptions hold.
(The DW of 2.22 in Section 1 is for the 6 test months; 2.18 here is for the 30 training months.)
Outputs: `outputs/tables/model_residual_diagnostics.csv`, `outputs/figures/model_residual_diagnostics.png`.

### 5.3 Bootstrap uncertainty for channel ROI

A moving-block bootstrap (3-month blocks, 1,000 samples) re-fitted the model on resampled training months.

| Channel | ROI | 90% interval | Samples with ROI > 1 |
|---|---:|---|---:|
| YouTube | 4.33 | 0.51 – 7.67 | 93% |
| Google Search | 3.93 | 1.69 – 6.56 | 99% |
| TV | 2.39 | 0.00 – 4.42 | 73% |
| Outdoor | 2.04 | 0.00 – 3.90 | 53% |
| Facebook/Instagram | 1.01 | 0.00 – 4.13 | 48% |
| Radio | 0.00 | 0.00 – 0.51 | 3% |

Google Search and YouTube are clearly profitable. The intervals for TV, Outdoor and Facebook/Instagram are wide because the model has only 30 training months, so their exact ROI is uncertain.
Outputs: `outputs/tables/roi_bootstrap_intervals.csv`, `outputs/figures/model_roi_bootstrap_intervals.png`.

### 5.4 Alternative model (log-spend Ridge)

A simpler model using log(1 + spend), with no adstock or saturation curve, was fitted with the same external factors, split and cross-validation.

| Model | CV RMSE (LKR) | Test MAE (LKR) | Test R² | Test MAPE |
|---|---:|---:|---:|---:|
| **Final: adstock + Hill saturation** | **4.33M** | **0.98M** | **0.938** | **1.13%** |
| Alternative: log(1 + spend) | 4.44M | 1.73M | 0.872 | 2.00% |

| Channel | Marginal ROI (final) | Marginal ROI (alternative) | Rank (final) | Rank (alternative) |
|---|---:|---:|---:|---:|
| YouTube | 2.13 | 1.84 | 1 | 2 |
| Google Search | 1.92 | 2.32 | 2 | 1 |
| TV | 1.19 | 0.55 | 3 | 4 |
| Outdoor | 1.00 | 0.80 | 4 | 3 |
| Facebook/Instagram | 0.51 | 0.00 | 5 | 5 |
| Radio | 0.00 | 0.00 | 6 | 5 |

The final model is more accurate, and both models agree on the main message (rank correlation 0.87): Google Search and YouTube give the best return on extra spend, and Facebook/Instagram and Radio the lowest.
Outputs: `outputs/tables/model_alternative_comparison.csv`, `outputs/tables/model_alternative_roi_comparison.csv`.
