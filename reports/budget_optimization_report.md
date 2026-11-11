# Budget Optimisation Report

**Owner:** ROI, External Factors & Budget Optimisation team 
**Notebook:** `notebooks/07_budget_optimisation.ipynb`
**Input:** `data/processed/feature_engineered_marketing_mix_data.csv`, trained model in `outputs/models/` (loaded with `MMMPredictor` from `src/predict.py`)
**Output:** `outputs/tables/optimal_allocation.csv`, `outputs/tables/budget_scenarios.csv`; figures `outputs/figures/opt_01_current_vs_optimal.png`, `opt_02_budget_scenarios.png`

---

## 1. Objective

Find the split of a **fixed advertising budget** across the six channels that gives the **highest predicted sales**, and show how sales change if the budget goes up or down.

---

## 2. Method

### 2.1 Problem set-up

| Setting | Choice | Reason |
|---|---|---|
| Planning period | The 12 months of 2025 | Latest full year, with its real seasonality, promotions and economic conditions |
| Budget | Total 2025 ad spend, **LKR 240.3M**, kept fixed | The brief asks for a fixed budget |
| Decision variables | One multiplier per channel, applied to that channel's 2025 monthly spend | Each channel keeps its month-to-month pattern; only its annual total changes |
| Objective | Maximise total predicted 2025 sales from the MMM | Sales come from the model's adstock and saturation curves, not from fixed ROI numbers |
| Constraint | Total spend = budget | Fixed budget |
| Bounds | Each channel between **70% and 130%** of its current spend | Realistic changes; no channel is switched off or tripled in one year |
| Carry-over | 2025 starts with the advertising effect left over from 2024 | Delayed ad effects continue across the year boundary |
| Solver | `scipy.optimize.minimize`, method **SLSQP**, values scaled to LKR millions | Handles the equality constraint and bounds; scaling keeps the solver numerically stable |

### 2.2 How the optimiser works

The optimiser starts from the current split and asks the model what total 2025 sales would be. It then moves money towards channels where the next rupee earns the most, keeping the total equal to the budget and every channel within its limits. It repeats this until no change improves sales.

Because of diminishing returns, the best split does not put all the money into one channel. As a channel receives more money, its marginal ROI falls, so the remaining money is balanced across channels.

### 2.3 Checks

| Check | Result |
|---|---|
| Solver status | Converged ("Optimization terminated successfully") |
| Total spend equals budget | Yes (LKR 240,285,900) |
| Every channel within 70%–130% | Yes |
| The plan changes the split | Yes |
| Comparison with 2,000 random plans that follow the same rules | Best random plan: LKR 1,060.3M · optimiser: LKR 1,067.4M, so the optimiser is better |

---

## 3. Results

### 3.1 Current vs optimal split (same budget)

| Channel | Current (LKR M) | Optimal (LKR M) | Change (LKR M) | Change | Current share | Optimal share |
|---|---:|---:|---:|---:|---:|---:|
| Google Search | 43.2 | 56.2 | +13.0 | **+30%** (limit) | 18.0% | 23.4% |
| YouTube | 30.6 | 39.8 | +9.2 | **+30%** (limit) | 12.8% | 16.6% |
| TV | 56.5 | 60.2 | +3.7 | +6% | 23.5% | 25.1% |
| Outdoor | 33.4 | 30.5 | −2.9 | −9% | 13.9% | 12.7% |
| Radio | 22.0 | 15.4 | −6.6 | **−30%** (limit) | 9.2% | 6.4% |
| Facebook/Instagram | 54.5 | 38.1 | −16.3 | **−30%** (limit) | 22.7% | 15.9% |
| **Total** | **240.3** | **240.3** | **0.0** | | 100% | 100% |

| Predicted 2025 sales | LKR M |
|---|---:|
| Current split | 1,041.5 |
| Optimal split | 1,067.4 |
| **Expected uplift** | **+25.9 (+2.5%)** |

Four channels reach their ±30% limit, which means the optimiser would move further if the limits were wider.

Figure: `outputs/figures/opt_01_current_vs_optimal.png`

### 3.2 Budget scenarios

| Budget change | Total budget (LKR M) | Sales, current split (LKR M) | Sales, optimal split (LKR M) | Gain from better split (LKR M) |
|---|---:|---:|---:|---:|
| −20% | 192.2 | 986.3 | 1,003.7 | +17.4 |
| −10% | 216.3 | 1,015.4 | 1,039.8 | +24.4 |
| **0%** | **240.3** | **1,041.5** | **1,067.4** | **+25.9** |
| +10% | 264.3 | 1,065.1 | 1,088.4 | +23.3 |
| +20% | 288.3 | 1,086.4 | 1,101.6 | +15.2 |

- At every budget level, the optimal split gives more sales than today's split.
- With a **10% smaller budget**, the optimal split gives about the same sales as today's full budget with the current split (LKR 1,039.8M vs 1,041.5M), a saving of about **LKR 24M**.
- A **10% larger budget** with the optimal split adds about **LKR 46.9M** compared with today.
- The gain from a better split shrinks at higher budgets, because more channels start to saturate.

Figure: `outputs/figures/opt_02_budget_scenarios.png`

---

## 4. Key Findings

- Re-splitting the same LKR 240.3M budget is expected to raise 2025 sales by about **LKR 25.9M (+2.5%)**.
- Money should move **from Facebook/Instagram and Radio to Google Search and YouTube**, with a small increase for TV and a small decrease for Outdoor.
- A better split is about as valuable as a 10% bigger budget with today's split.
- If the budget is increased, the extra money should go first to Google Search and YouTube, then TV.

---

## 5. Recommendations

1. Shift budget towards Google Search and YouTube, and away from Facebook/Instagram, following the optimal split.
2. Make changes gradually (within ±30% per channel per year) and track monthly sales against the model's predictions.
3. Improve the split before increasing the total budget.
4. Before cutting Radio further, confirm with a real-world test (for example a regional on/off test), because the model's "no measurable effect" comes from limited data.
5. Re-run the optimisation each year with new data, as saturation levels and channel performance change.

---

## 6. Limitations

- **Model-based results:** the uplift is a model prediction, not a guaranteed outcome.
- **Small sample:** the model is trained on 36 months, so the response curves carry uncertainty. The bootstrap in notebook 05 shows that the gains from Google Search and YouTube are the most certain; the exact ROI of TV, Outdoor and Facebook/Instagram is less certain.
- **Known external conditions:** 2025 is re-planned with its actual seasonality, promotions and economy. For 2026 planning, forecasts of these factors are needed.
- **Bounds are a business assumption:** the ±30% limits should be checked against contracts, minimum brand presence and campaign capacity.
- **Revenue, not profit:** the optimiser maximises sales revenue. Margins were not available, so profit-based ROI was not calculated.

---

## 7. Handoff

- **Dashboard** uses `optimal_allocation.csv` (current vs optimal page) and `budget_scenarios.csv` (what-if page).
- The old file `optimized_budget_allocation.csv` from an earlier version is no longer used.
- If the model is re-trained, re-run notebook 06 and then this notebook to refresh all results.
