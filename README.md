
# Favorita Store Sales Forecasting

Forecasting daily sales for 1,782 store × product-family series (54 stores × 33 families) from the Kaggle [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition (Corporación Favorita, Ecuador).

The goal isn't a leaderboard score. It's an honest comparison: how much does a global LightGBM model beat a seasonal-naive baseline, and what actually drives the difference?

## Results

16-day holdout: **July 31 – August 15, 2017**. Both models forecast recursively in 7-day chunks, so neither sees real holdout sales.

| Model | RMSLE | MAE |
| --- | --- | --- |
| Seasonal-naive baseline (repeat last known week) | 0.6170 | 96.54 |
| LightGBM v1 (store, family, promotion, lag-7) | 0.4541 | 77.35 |
| **LightGBM final (E6, chosen by CV)** | **0.4894** | **73.64** |

The final model's 7-fold time-series CV average is **0.4684 RMSLE**.

The final model has a worse holdout RMSLE than v1 but a clearly better MAE and CV score. I kept the model the cross-validation chose rather than tuning to the holdout, because picking whichever version happens to score best on one 16-day window is a quiet form of overfitting.

## Approach

**Data preparation**
- Reindexed every series onto a complete daily calendar, filling gaps with zeros (3,008,016 rows: 1,688 days × 1,782 series). The raw data has no December 25 rows, which silently turned 49,896 "lag-7" values into lag-8. With the reindexing, positional lag features are exact.
- Masked closed store-days (all 33 families at zero sales) out of training and validation: 241,890 rows.
- Trained on a `log1p` target.

**Final LightGBM features (E6):** store, product family, promotion count, lag-7 sales, day of week, month, day of month, payday flag, New Year flag.

**Forecasting:** recursive, in 7-day chunks, with each chunk's predictions fed back in as the next chunk's lag-7.

**Evaluation:** 7-fold time-series cross-validation for model selection, plus a fixed holdout for the final comparison.

## Findings

- **The first baseline was leaking.** It scored 0.5694 RMSLE, but its lag-7 used real holdout sales for days 8–16. The recursive baseline (0.6170) is the fair comparison, and the gap between the two is a good reminder of how easily leakage flatters results.
- **Why RMSLE is the headline metric.** Errors are extremely skewed: the median row misses by 8.4 units, while the mean miss is 100.2, and the worst 1% of rows carry 26.7% of all absolute error. MAE is dominated by a few huge-volume series.
- **The two metrics tell different stories.** The worst families by MAE (GROCERY I, BEVERAGES, CLEANING) and by RMSLE (SCHOOL AND OFFICE SUPPLIES, LINGERIE, GROCERY II) don't overlap at all. One is about volume, the other about low, intermittent sales.
- **Day of month was the only clear win in feature experiments,** mostly on the holiday folds. Holiday tables, store metadata and extra flags added nothing once it was included. Calendar features also fixed a "label trap" around the transferred New Year holiday in January 2017.
- **Data quirks:** store 18 was closed from mid-August to November 2016. The mid-2016 CV folds (July–September) score worse and stop early. I ruled out a single store, a single day, August specifically and the April 2016 earthquake, and the cause is still unexplained.

## Repository

- Baseline and LightGBM scripts, plus the feature experiments `exp1`–`exp6`
- `pyproject.toml` / `uv.lock`: environment (managed with uv)

The Kaggle data isn't included. Download it from the competition page, or with:

```bash
kaggle competitions download -c store-sales-time-series-forecasting
```

Then:

```bash
uv sync
uv run python <script>.py
```

## Next

- Retrain the final model on all data through July 30
- Recursive cross-validation, so CV matches how the holdout is forecast
- Regional and local holidays
