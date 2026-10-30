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
| **OLS Regression** | 3,260,295.90 | 4,207,898.34 | 0.348 | 3.66% | 0.19 | Severe multicollinearity & autocorrelation |
| **Ridge Regression** | **891,421.35** | **1,121,793.01** | **0.954** | **1.02%** | **2.29** | **Production Champion Model** |

### Key Takeaways:
1. **Predictive Accuracy ($R^2 = 95.37\%$, $\text{MAPE} = 1.02\%$)**: The regularized Ridge model accounts for ~95.4% of total sales revenue variance on completely unseen future months, holding average percentage forecasting error to just ~1.02%.
2. **Residual Diagnostic & Autocorrelation ($DW = 2.29$)**: Unlike standard OLS ($DW = 0.19$), which exhibited extreme positive autocorrelation due to unmodeled carryover, the Ridge regression model with Adstock and Hill Saturation yields a Durbin-Watson statistic near $2.0$. This confirms zero residual memory or serial correlation leakage.
3. **Diagnostic Visualizations Generated**:
   - `outputs/figures/model_actual_vs_predicted.png`: Monthly actual vs. predicted revenue timeline with holdout test shading and metrics callout.
   - `outputs/figures/model_residuals_diagnostic.png`: 3-panel residual analysis (time-series with $\pm 1.96\sigma$ bounds, residuals vs. fitted values, and error normality histogram).

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

The model was trained on **14 features** in the exact sequence below. `MMMPredictor` automatically guarantees this ordering even if input columns are shuffled:

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
12. `inflation_proxy_index`
13. `seasonal_demand_index`
14. `public_holiday_count`
