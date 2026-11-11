# Data Cleaning & Feature Engineering Report

**Owner:** Data Preparation team 
**Notebooks:** `notebooks/01_data_cleaning.ipynb`, `notebooks/02_external_features.ipynb`
**Input:** `data/raw/mmm_dataset.xlsx` (sheet `MMM_Data`)
**Output:** `data/processed/cleaned_marketing_mix_data.csv` (36 × 25), `data/processed/feature_engineered_marketing_mix_data.csv` (36 × 44)

---

## 1. Objective

Prepare an accurate, consistent dataset for exploratory analysis, modelling, ROI analysis and budget optimisation. This stage checks data quality, fixes inconsistencies, and adds features that describe time, channel mix, engagement and delayed advertising effects.

---

## 2. Dataset Overview

| Item | Detail |
|---|---|
| Source | `data/raw/mmm_dataset.xlsx` (synthetic data for analytical practice) |
| Context | Sri Lanka · FMCG – Packaged Foods · National |
| Granularity | Monthly |
| Period | January 2023 – December 2025 (36 months) |
| Size | 36 rows × 25 columns |
| Currency | Sri Lankan Rupees (LKR) |
| Target | `sales_revenue_lkr` |
| Media channels | TV, Facebook/Instagram, Google Search, YouTube, Radio, Outdoor |
| External factors | Price index, promotion discount, competitor pressure, economic sentiment, LKR/USD exchange rate, inflation proxy, seasonal demand index, public holidays |

---

## 3. Data Quality Checks

| Check | Result | Action |
|---|---|---|
| Missing values | 0 in every column | None needed |
| Duplicate rows | 0 | None needed |
| Date format | `month` stored as text | Converted to datetime |
| Numeric types | Spend and sales columns checked | Converted with `pd.to_numeric` (no invalid values found) |
| Monthly continuity | 36 consecutive months | None needed |
| Total ad spend consistency | `total_ad_spend_lkr` differed from the sum of the six channels in **16 of 36 months**, by at most **LKR 2** (rounding in the source data) | Replaced `total_ad_spend_lkr` with the sum of the six channel columns |

---

## 4. Outlier Treatment

Outliers were checked with the IQR rule (values below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR) and capped at those limits. The rule was applied to all numeric columns, and changed values in three of them:

| Column | Months changed | Original → capped |
|---|---|---|
| `sales_revenue_lkr` | 2023-01, 2023-04, 2024-08 | 68.73M → 70.11M · 100.71M → 100.06M · 69.89M → 70.11M |
| `seasonal_demand_index` | 2023-12, 2024-12, 2025-12 | 1.24 → 1.231 (each December) |
| `conversions` | 2025-03, 2025-06, 2025-08 | 3.10M / 2.91M / 2.42M → 2.41M |

All other columns were unchanged.

**Note for the final report:** the changed values are small, but they include the target (`sales_revenue_lkr`) and the December festive peak (`seasonal_demand_index`), which are real business patterns rather than errors. For this reason the Marketing Mix Model is trained on the original raw data (`config/params.json` → `data_file`), not on the capped values. The capped dataset is used for EDA and as the input table in notebooks 06 and 07, where the differences have no practical effect on model predictions.

---

## 5. Feature Engineering

Notebook 02 adds 19 features to the cleaned dataset (25 → 44 columns):

| Group | Features | Purpose |
|---|---|---|
| Time | `year`, `quarter`, `month_number` | Trend and seasonal patterns |
| Channel mix | `tv_share`, `digital_share`, `radio_share`, `outdoor_share` | Share of the monthly budget per channel group |
| Engagement | `ctr`, `conversion_rate` | Clicks per impression, conversions per click |
| Revenue efficiency | `revenue_per_unit`, `revenue_per_conversion`, `roas`, `cost_per_conversion` | Descriptive efficiency measures |
| Delayed effects | `lag_1_total_spend`, `lag_2_total_spend`, `rolling_3_month_spend` | Spend in previous months |
| Composite | `marketing_pressure`, `competitive_advantage`, `promotion_impact` | Combined business indicators |

Missing values created by the lag and rolling features (first one or two months) were filled with 0.

### Which features are used in the model

| Used as model inputs | Not used as model inputs |
|---|---|
| The six channel spend columns (transformed with adstock and saturation in notebook 05) and seven external factors (all except `inflation_proxy_index`, which was removed because it moves almost together with the price index) | `total_ad_spend_lkr` (sum of the channels), `impressions`, `clicks`, `conversions`, and features calculated from sales such as `roas`, `revenue_per_unit` and `revenue_per_conversion` |

Features built from sales or from advertising outcomes would leak the target into the model, so they are used for EDA only. Delayed effects are handled in the model by adstock rather than by the lag features.

---

## 6. Key Findings

- The dataset is complete: no missing values, no duplicates, and 36 consecutive months.
- Total ad spend in the source had small rounding differences (at most LKR 2) and was recalculated from the six channels.
- Very few values fell outside the IQR limits; the main ones are the December seasonal peaks and three sales months.
- The final feature table has 44 columns. 13 of them (6 channels + 7 external factors) are inputs to the Marketing Mix Model.

---

## 7. Limitations

- **Small sample:** 36 monthly observations limit how many variables the model can reliably use.
- **Outlier capping on the target:** capping sales and the seasonal index removes some real variation. The model avoids this by using the raw data.
- **Lag features filled with 0:** the first months' lag values of 0 are not real spend values; these features are not used in the model.
- **Synthetic data:** results show the method, not real business performance.

---

## 8. Handoff

- **EDA (notebook 03)** uses `cleaned_marketing_mix_data.csv`.
- **Modelling (notebook 05)** reads the raw file through `config/params.json` and applies its own validation checks.
- **ROI and optimisation (notebooks 06 and 07)** use `feature_engineered_marketing_mix_data.csv` as the input table for `MMMPredictor`.
- Column definitions are listed in `config/data_schema.md` and in the `Data_Dictionary` sheet of the raw workbook.

