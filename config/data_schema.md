# Data Schema

**Source:** `data/raw/mmm_dataset.xlsx` (sheet `MMM_Data`; full definitions in sheet `Data_Dictionary`)
**Granularity:** monthly, one row per month | **Period:** Jan 2023 – Dec 2025 (36 rows) | **Currency:** LKR
**Context:** Sri Lanka, FMCG – Packaged Foods, national coverage. All records are synthetic.

## Target
| Column | Description |
|---|---|
| `sales_revenue_lkr` | Monthly sales revenue (primary MMM outcome) |
| `units_sold` | Monthly units sold (secondary outcome) |

## Media channels (spend, LKR)
| Column | Channel |
|---|---|
| `tv_spend_lkr` | TV |
| `facebook_instagram_spend_lkr` | Facebook & Instagram |
| `google_search_spend_lkr` | Google Search |
| `youtube_spend_lkr` | YouTube |
| `radio_spend_lkr` | Radio |
| `outdoor_spend_lkr` | Outdoor (OOH) |
| `total_ad_spend_lkr` | Sum of all channel spend (do **not** use as a model input alongside channels) |

## Media activity metrics
`impressions`, `clicks`, `conversions` — useful for EDA and efficiency KPIs, not as model drivers (they are outcomes of spend).

## External factors
| Column | Description |
|---|---|
| `average_price_index` | Product price level |
| `promotion_discount_pct` | Promotional discount (%) |
| `competitor_pressure_index_0_100` | Competitor activity (0–100) |
| `economic_sentiment_index_0_100` | Consumer/economic sentiment (0–100) |
| `lkr_usd_exchange_rate` | LKR/USD exchange rate |
| `inflation_proxy_index` | Inflation proxy |
| `seasonal_demand_index` | Seasonality incl. Sri Lankan festive periods |
| `public_holiday_count` | Public holidays in the month |

## Constant / metadata columns (drop before modelling)
`country`, `brand_category`, `region_scope`, `data_type`
