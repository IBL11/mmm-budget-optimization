# Marketing Mix Modelling (MMM) & Budget Optimisation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

Measuring the impact of advertising spend across six marketing channels for a Sri Lankan FMCG brand, and recommending how to allocate a fixed budget to maximise return on investment.

---

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Analysis Pipeline](#analysis-pipeline)
- [Outputs](#outputs)
- [Deliverables](#deliverables)
- [Team](#team)
- [Limitations](#limitations)
- [Tech Stack](#tech-stack)

---

## Overview
Marketing teams need to know which channels actually drive sales, how long their effect lasts, and when extra spend stops paying off. This project builds a **Marketing Mix Model** that separates baseline sales from the contribution of each channel and external factor, then uses the fitted model to find the **optimal budget allocation** under a fixed spend.

## Objectives
1. Quantify the incremental sales contribution and ROI of each marketing channel.
2. Capture **time delays** (carryover / adstock) and **diminishing returns** (saturation).
3. Measure the influence of **external factors** such as price, promotions, competition, seasonality and macro-economic conditions.
4. Build an optimisation model that **maximises predicted revenue** for a fixed advertising budget.
5. Present findings through an interactive **Power BI dashboard** and a report with data-driven recommendations.

## Dataset
| Property | Detail |
|---|---|
| File | `data/raw/mmm_dataset.xlsx` (sheets: `MMM_Data`, `Data_Dictionary`) |
| Context | Sri Lanka · FMCG – Packaged Foods · National |
| Granularity | Monthly |
| Period | January 2023 – December 2025 (36 months) |
| Currency | Sri Lankan Rupees (LKR) |
| Nature | Synthetic data for analytical practice |

**Target:** `sales_revenue_lkr`

**Media channels:** TV · Facebook & Instagram · Google Search · YouTube · Radio · Outdoor

**External factors:** price index · promotion discount · competitor pressure · economic sentiment · LKR/USD exchange rate · inflation proxy · seasonal demand · public holidays

Full column definitions are in [`config/data_schema.md`](config/data_schema.md).

## Methodology
```
Raw data ─► Cleaning & feature engineering ─► EDA
        ─► Adstock (carryover) ─► Saturation (diminishing returns)
        ─► Regression MMM (Ridge / OLS) ─► Validation
        ─► Contribution & ROI analysis ─► Budget optimisation ─► Dashboard & report
```

| Stage | Approach |
|---|---|
| **Adstock** | Geometric decay: `a_t = x_t + λ · a_(t-1)`, with decay λ tuned per channel |
| **Saturation** | Hill function: `x^s / (x^s + K^s)` to model diminishing returns |
| **Model** | Ridge regression (scikit-learn) and OLS (statsmodels) with non-negative media coefficients |
| **Validation** | Time-ordered train/test split (last 6 months held out); R², MAPE, RMSE, residual diagnostics |
| **ROI** | Channel contribution ÷ channel spend; marginal ROI from response curves |
| **Optimisation** | `scipy.optimize.minimize` (SLSQP), maximising predicted revenue subject to a fixed total budget and per-channel bounds |

## Project Structure
```
mmm-budget-optimisation/
├── config/
│   ├── params.json          # channels, target, adstock/saturation & budget settings
│   └── data_schema.md       # column definitions
├── data/
│   ├── raw/                 # original dataset (read-only)
│   └── processed/           # cleaned & feature-engineered data
├── notebooks/               # numbered analysis notebooks (run in order)
├── src/
│   ├── utils.py             # paths, config and data loading
│   └── transforms.py        # adstock & saturation functions
├── outputs/
│   ├── figures/             # charts for the report
│   └── tables/              # CSV outputs consumed by Power BI
├── dashboard/               # Power BI (.pbix) file
├── reports/                 # final report
├── tests/                   # unit tests
└── requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.10+
- Power BI Desktop (for the dashboard)
- Git

### Installation
```bash
git clone https://github.com/<your-username>/mmm-budget-optimisation.git
cd mmm-budget-optimisation

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Quick check
```python
from src.utils import load_raw, load_params

df = load_raw()          # 36 rows × 25 columns
params = load_params()   # channels, target, model settings
```

## Analysis Pipeline
Run the notebooks in order. Each reads the previous step's output.

| # | Notebook | Purpose | Output |
|---|---|---|---|
| 1 | `01_data_cleaning.ipynb` | Validate types, check gaps & outliers, drop constant columns | `data/processed/clean_data.csv` |
| 2 | `02_external_features.ipynb` | Trend, seasonality and external factor features | `data/processed/model_data.csv` |
| 3 | `03_eda.ipynb` | Trends, spend mix, correlations, VIF, decomposition | figures |
| 4 | `04_transformations.ipynb` | Tune adstock & saturation per channel | `config/params.json` |
| 5 | `05_modelling.ipynb` | Fit and validate the MMM | `outputs/tables/model_metrics.csv` |
| 6 | `06_roi_analysis.ipynb` | Contributions, ROI, response curves, external impact | `contributions.csv`, `roi_table.csv`, `response_curves.csv`, `external_impact.csv` |
| 7 | `07_budget_optimisation.ipynb` | Optimal allocation and budget scenarios | `optimal_allocation.csv`, `budget_scenarios.csv` |

## Outputs
All tables in `outputs/tables/` feed the Power BI dashboard:

| File | Contents |
|---|---|
| `model_metrics.csv` | R², MAPE, RMSE on train and test |
| `contributions.csv` | Monthly base and per-channel contribution |
| `roi_table.csv` | Spend, contribution, ROI and marginal ROI per channel |
| `response_curves.csv` | Predicted incremental revenue across spend levels |
| `external_impact.csv` | Contribution of each external factor |
| `optimal_allocation.csv` | Current vs recommended spend per channel |
| `budget_scenarios.csv` | Predicted revenue at different total budgets |

## Deliverables
- [ ] Cleaned dataset and exploratory analysis
- [ ] Working Marketing Mix Model
- [ ] Channel effectiveness and ROI analysis
- [ ] Budget optimisation algorithm
- [ ] Interactive Power BI dashboard
- [ ] Final report with data-driven recommendations

## Team
| Member | Responsibility |
|---|---|
| Member 1 | Team Lead – project setup, code review, recommendations, final submission |
| Member 2 | Data cleaning |
| Member 3 | External factor features & impact analysis |
| Member 4 | Exploratory data analysis |
| Member 5 | Adstock & saturation transformations |
| Member 6 | Model building & validation |
| Member 7 | Channel effectiveness & ROI |
| Member 8 | Budget optimisation |
| Member 9 | Power BI dashboard |
| Member 10 | Report & documentation |



## Limitations
- **Small sample:** 36 monthly observations against 6 channels and 8 external factors. Models must stay parsimonious (regularisation, feature selection) to avoid overfitting.
- **Correlation, not causation:** MMM estimates are observational; results should ideally be validated with experiments (e.g. geo or lift tests).
- **Multicollinearity:** channels that move together make individual effects harder to separate.
- **Synthetic data:** findings illustrate the method and are not real business results.

## Tech Stack
**Python:** Pandas · NumPy · Scikit-learn · SciPy · Statsmodels · Matplotlib · Seaborn
**BI:** Power BI
**Collaboration:** Git & GitHub
