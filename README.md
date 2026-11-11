# Cookie Cats A/B Test: Does Moving the First Gate Hurt Retention?

An end-to-end A/B test analysis of a mobile game experiment, done in **Snowflake SQL** and **Python**.

## Business question
Cookie Cats is a mobile puzzle game. Players hit a "gate" that makes them wait or pay to continue. The team tested moving the first gate from **level 30 (`gate_30`)** to **level 40 (`gate_40`)**.
**Should the gate move?** The deciding metric is **7-day retention**: the share of players who return a week after installing.

## Data
Kaggle: *Mobile Games A/B Testing - Cookie Cats* (90,189 players).

| Column | Meaning |
|---|---|
| `userid` | Unique player ID |
| `version` | Experiment group (`gate_30` or `gate_40`) |
| `sum_gamerounds` | Rounds played in the first 14 days |
| `retention_1` | Player returned 1 day after install (boolean) |
| `retention_7` | Player returned 7 days after install (boolean) |

## Method
1. Loaded the CSV into Snowflake (`ML_PROJECT.PUBLIC.COOKIE_CATS`) using an X-Small warehouse.
2. Cleaned the data: removed one extreme outlier (a player with about 49,854 rounds) with `WHERE sum_gamerounds < 10000`.
3. Checked randomisation with a sample ratio mismatch (SRM) test.
4. Compared retention and engagement by group.
5. Ran a two-proportion z-test with a 95% confidence interval for each retention metric, written directly in SQL.
6. Repeated and extended the analysis in Python (`ab_testing_cookie_cats.ipynb`): power analysis, Holm correction, bootstrap CI and a Bayesian comparison.

## Results (Snowflake SQL, outlier removed)

| | gate_30 | gate_40 |
|---|---|---|
| Players | 44,699 | 45,489 |
| 1-day retention | 44.82% | 44.23% |
| 7-day retention | 19.02% | 18.20% |
| Avg. game rounds | 51.34 | 51.30 |
| Median game rounds | 17 | 16 |

| Metric | Difference (gate_40 - gate_30) | 95% CI | z-score | Significant at 5%? |
|---|---|---|---|---|
| **7-day retention** | **-0.82 pp** | [-1.33, -0.31] | -3.17 | **Yes** |
| 1-day retention | -0.59 pp | [-1.24, +0.06] | -1.79 | No |

## Recommendation
**Keep the first gate at level 30.**
Moving it to level 40 lowered 7-day retention by about 0.82 percentage points (a relative drop of roughly 4%) with a 95% CI of -1.33 to -0.31 pp, and it brought no gain in engagement. The loss looks small per player but adds up across a large player base, and 7-day retention is an early signal of long-term revenue.

## Caveats
- **Sample ratio mismatch:** the groups are slightly uneven (44,699 vs 45,489, SRM p-value about 0.009), so the randomisation may not be perfect.
- Only retention and rounds were measured. There is no revenue or in-app purchase data, so the effect on money is unknown.
- One test period, so novelty effects cannot be ruled out.
- The 1-day result points the same way but is not statistically significant.

## Next steps
- Test an intermediate gate (for example level 35).
- Run a follow-up test using revenue as the metric.

## Repository contents
```
cookie-cats-ab-test/
├── README.md
├── ab_test_cookie_cats.sql        # Snowflake queries (group metrics + z-tests)
├── ab_testing_cookie_cats.ipynb   # Python analysis (power, bootstrap, Bayesian)
└── screenshots/
    ├── group_metrics.png
    └── z_test_results.png
```

## Tools
Snowflake (SQL), Python (pandas, SciPy, statsmodels, matplotlib), Jupyter

## Author
Kumar Harsh | [LinkedIn](https://linkedin.com/in/kumar-harsh-6ab195241) | [GitHub](https://github.com/Kumarharsh1)
