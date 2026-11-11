# Channel Effectiveness & ROI Analysis Report

**Owner:** ROI, External Factors & Budget Optimisation team
**Notebook:** `notebooks/06_roi_analysis.ipynb`
**Input:** `data/processed/feature_engineered_marketing_mix_data.csv`, trained model in `outputs/models/` (loaded with `MMMPredictor` from `src/predict.py`)
**Output:** `outputs/tables/roi_table.csv`, `contributions.csv`, `response_curves.csv`, `external_impact.csv`; figures `outputs/figures/roi_01` to `roi_04`

---

## 1. Objective

Use the Marketing Mix Model to answer four questions:

1. How much of sales is **base sales** (would happen without advertising), and how much does **each channel** add?
2. What is each channel's **ROI** and **marginal ROI**?
3. Where does each channel start to **saturate** (diminishing returns)?
4. How much do **external factors** (promotions, seasonality, economy, competition, price) move sales?

The dataset records one total sales figure per month, so channel effects cannot be read from the data directly. The model separates them.

---

## 2. Method

| Step | What was done |
|---|---|
| Load and check the model | `MMMPredictor` loads the trained Ridge model (non-negative media coefficients). Predictions are checked against the Modelling team's saved test predictions. |
| Channel contribution | For each channel, the model predicts sales with that channel's spend set to zero in every month. The drop from normal predicted sales is the channel's contribution. Adstock carry-over is included. |
| Base sales | Total predicted sales minus all six channel contributions. |
| ROI | Channel contribution ÷ channel spend (also shown as ROI % = (contribution − spend) ÷ spend). |
| Marginal ROI | Extra predicted sales from a 1% increase in a channel's spend, divided by the extra spend. |
| Response curves | Each channel's 2025 spend scaled from 0× to 2× (other channels unchanged), with 2024 carry-over passed in. The break-even point is where marginal ROI falls below 1. |
| External factors | Each factor set to its 36-month average, and the effect of a one-standard-deviation increase measured. |

---

## 3. Results

### 3.1 Base sales vs advertising (Jan 2023 – Dec 2025)

| Component | Sales (LKR M) | Share |
|---|---:|---:|
| Base sales | 1,520.1 | 49.6% |
| Advertising (all channels) | 1,541.7 | 50.4% |
| **Total predicted sales** | **3,061.8** | 100% |

Total advertising spend over the period was LKR 668.7M, so advertising as a whole returned about **LKR 2.31 in sales per LKR 1 spent**.

Figure: `outputs/figures/roi_01_sales_decomposition.png`

### 3.2 ROI and marginal ROI by channel

| Rank | Channel | Spend (LKR M) | Sales caused (LKR M) | ROI | ROI % | Marginal ROI |
|---:|---|---:|---:|---:|---:|---:|
| 1 | YouTube | 79.9 | 345.8 | 4.33 | 333% | 2.13 |
| 2 | Google Search | 119.4 | 468.6 | 3.93 | 293% | 1.92 |
| 3 | TV | 157.0 | 375.2 | 2.39 | 139% | 1.19 |
| 4 | Outdoor | 94.1 | 191.7 | 2.04 | 104% | 1.00 |
| 5 | Facebook/Instagram | 158.3 | 160.4 | 1.01 | 1% | 0.50 |
| 6 | Radio | 60.1 | 0.0 | 0.00 | – | 0.00 |

Channels are ranked by marginal ROI, the return on the next rupee, which is the measure that should guide budget decisions.

Figure: `outputs/figures/roi_02_roi_by_channel.png`

### 3.3 Share of spend vs share of sales caused

| Channel | Share of spend | Share of advertising-driven sales | Assessment |
|---|---:|---:|---|
| Google Search | 17.8% | 30.4% | Under-funded |
| YouTube | 11.9% | 22.4% | Under-funded |
| TV | 23.5% | 24.3% | Balanced |
| Outdoor | 14.1% | 12.4% | Slightly over-funded |
| Facebook/Instagram | 23.7% | 10.4% | Over-funded |
| Radio | 9.0% | 0.0% | No measurable effect |

### 3.4 Response curves (2025)

| Channel | 2025 spend (LKR M) | Spend where marginal ROI falls below 1 (LKR M) | Status |
|---|---:|---:|---|
| Google Search | 43.2 | 71.3 | Room to grow |
| YouTube | 30.6 | 50.6 | Room to grow |
| TV | 56.5 | 59.4 | Some room to grow |
| Outdoor | 33.4 | 30.1 | At break-even |
| Facebook/Instagram | 54.5 | 21.8 | Well past break-even |
| Radio | 22.0 | – | No measurable effect |

Figure: `outputs/figures/roi_03_response_curves.png`

### 3.5 External factor impact

| Factor | Normal change (+1 SD) | Effect on monthly sales | Direction |
|---|---|---:|---|
| Promotion discount | +6.0 points | +LKR 2.60M (+3.1%) | Increases sales |
| Seasonal demand index | +0.087 | +LKR 2.50M (+2.9%) | Increases sales |
| Competitor pressure | +10.2 points | −LKR 0.84M (−1.0%) | Reduces sales |
| Average price index | +0.063 | +LKR 0.80M (+0.9%) | See note |
| Economic sentiment | +12.3 points | +LKR 0.77M (+0.9%) | Increases sales |
| LKR/USD exchange rate | +6.2 | −LKR 0.16M (−0.2%) | Reduces sales |
| Public holidays | +1.8 days | +LKR 0.13M (+0.2%) | Increases sales |

Examples:

- The **December festive season** adds about **LKR 5.3M** per month compared with an average month.
- A **5-point higher promotion discount** adds about **LKR 2.2M (+2.5%)** per month.
- **10 points more competitor pressure** reduces sales by about **LKR 0.82M (−1.0%)** per month.

**Note:** the inflation proxy moved almost together with the price index (VIF about 100), so it was removed from the model to fix multicollinearity. The price index effect is positive, which is unusual. In this data the price index rises steadily every month (correlation 0.99 with time), so in the model it mostly works as a time-trend term. Do not read it as proof that higher prices raise sales.

Figure: `outputs/figures/roi_04_external_impact.png`

---

## 4. Key Findings

- About half of sales (49.6%) would happen without advertising. Advertising drives the other half.
- **YouTube and Google Search** give the highest return on additional spend (marginal ROI 2.13 and 1.92) and bring in more sales than their share of the budget.
- **Facebook/Instagram** receives about 24% of the budget but causes only about 10% of advertising-driven sales. Its marginal ROI (0.50) is below 1.
- **TV** is balanced, and **Outdoor** is at its break-even point.
- The model finds **no measurable sales effect from Radio**.
- **Promotions and seasonality** move sales more than any single media channel.

---

## 5. Limitations

- **Small sample:** 36 monthly observations, so all estimates carry uncertainty. The bootstrap in notebook 05 gives 90% ROI intervals: Google Search (1.7–6.6) and YouTube (0.5–7.7) are clearly above zero, while the intervals for TV, Outdoor and Facebook/Instagram reach zero. Treat the rankings as directional.
- **Observational data:** MMM shows association, not proven cause and effect. Real-world tests (for example regional on/off tests) would confirm channel effects.
- **Radio:** a value of zero means the model cannot detect an effect in this data. It does not prove that Radio has no value. The EDA found a two-month lagged correlation of 0.35 between Radio spend and sales, which suggests a possible delayed effect that the model's current Radio adstock setting may not fully capture.
- **Collinear factors:** fixed by removing the inflation proxy from the model; all model inputs now have VIF below 10.
- **Model choice:** channel ROI depends on the model's adstock and saturation settings. A simpler log-spend model (notebook 05) gives different exact values but the same main ranking (rank correlation 0.87).

---

## 6. Handoff

- **Budget optimisation (notebook 07)** uses the same model and the marginal ROI pattern above.
- **Dashboard** uses `roi_table.csv` (channel ROI page), `contributions.csv` (decomposition chart), `response_curves.csv` (diminishing returns page) and `external_impact.csv`.
- If the model is re-trained, re-run this notebook to refresh every table and figure.

