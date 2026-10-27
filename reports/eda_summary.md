 Exploratory Data Analysis: Summary

 1. Dataset Overview

Number of observations: 36 (monthly)
Number of variables: 44 (25 original + 19 engineered)
Date range: January 2023 to December 2025
Target variable: `sales_revenue_lkr`

 2. Sales Trends

Main trend observed: Sales are broadly flat: LKR 1,029M (2023), 993M (2024, -3.5%), 1,041M (2025, +4.8%).
Highest/lowest sales periods: Highest in April 2023, December 2025 and December 2024 (about LKR 98 to 100M). Lowest in January 2023 and August 2024 (about LKR 70M).
Possible seasonal patterns: Peaks around April and December (festive periods).

 3. Advertising Spend

Highest-spend channel: Facebook/Instagram (23.7%), almost equal to TV (23.5%).
Lowest-spend channel: Radio (9.0%).
Notable spending patterns: Google Search 17.8%, Outdoor 14.1%, YouTube 11.9%.

 4. Advertising Spend vs Sales

Observed relationships: Weak to moderate positive relationships (r = 0.33 to 0.43) for all channels except Radio.
Differences between channels: Outdoor is strongest (0.43), then Google Search (0.38). Radio shows almost none (0.05), but its lag-2 correlation is 0.35 (possible delayed effect). Google Search and Outdoor flatten at high spend (possible diminishing returns).
Important limitations: With 36 observations these are exploratory signals, not causal effects.

 5. Correlation Analysis

Variables most positively correlated with sales: Seasonal demand index (+0.62), promotion discount (+0.58), Outdoor spend (+0.43).
Variables negatively correlated with sales: Competitor pressure index (about -0.33).
Potential multicollinearity concerns: `inflation_proxy_index` (VIF 105) and `average_price_index` (VIF 93) are very high. All media channels have VIF below 3.

 6. External Factors
 
Notable relationships: Seasonality and promotions relate to sales more strongly than any single channel.
Possible seasonal or promotional patterns: High-discount months coincide with sales peaks (April 2023, December 2024).

 7. Conclusion

Overall conclusion: Sales are broadly flat, with peaks around April and December. Seasonality and promotions are related to sales more strongly than any single media channel. Outdoor and Google Search show the strongest channel relationships, while Radio shows almost none in the same month.
