# E6: E2 (flags + month + dayofmonth, no holidays) + closed store-days excluded from training/validation (open_mask)
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import pandas as pd
import numpy as np

data = pd.read_csv('store-sales-time-series-forecasting/train.csv',
                   parse_dates=['date'])
data = data.drop('id', axis=1)
full = pd.MultiIndex.from_product(
    [data.store_nbr.unique(),
     data.family.unique(),
     pd.date_range(data.date.min(), data.date.max(), freq='D')],
    names=['store_nbr', 'family', 'date'])

data = (data
        .set_index(['store_nbr', 'family', 'date'])
        .reindex(full, fill_value=0)
        .reset_index())
data['dayofweek'] = data.date.dt.dayofweek

print(len(data))   # should be 3,008,016

data['lag7'] = data.groupby(['store_nbr', 'family'])['sales'].shift(7)
data['family'] = data['family'].astype('category')
data['store_nbr'] = data['store_nbr'].astype('category')

print(data[data.date.dt.strftime('%m-%d') == '01-01'].groupby(data.date.dt.year)['sales'].sum())

data['dayofmonth'] = data.date.dt.day
data['is_payday'] = ((data.date.dt.day == 15) | data.date.dt.is_month_end).astype(int)
data['is_new_year'] = ((data.date.dt.month == 1) & (data.date.dt.day == 1)).astype(int)
data['month'] = data.date.dt.month

data = data.drop(data[(data['dayofmonth'] == 25) & (data['month'] == 12)].index, axis=0)

# store-days where all 33 families sold nothing = store closed; excluded from training/validation only
store_day_total = data.groupby(['store_nbr', 'date'])['sales'].transform('sum')
open_mask = store_day_total > 0
print('closed store-day rows excluded:', (~open_mask).sum())

x_train = data[(data.date < '2017-07-15') & open_mask].dropna(subset=['lag7'])

eval_x = data[(data.date >= '2017-07-15') & (data.date <= '2017-07-30') & open_mask].dropna(subset=['lag7']).drop(['date'], axis=1)

eval_y = np.log1p(eval_x['sales'])

eval_x = eval_x.drop(['sales'], axis=1)

y_train = np.log1p(x_train['sales'])
x_train = x_train.drop(['sales', 'date'], axis=1)


model = LGBMRegressor(n_estimators=5000, learning_rate=0.05).fit(x_train, y_train, callbacks=[log_evaluation(100), early_stopping(50)], eval_set=[(eval_x, eval_y)])
print(model.best_iteration_)

pred_log = model.predict(eval_x)

rmsle = np.sqrt(np.mean((pred_log - eval_y) ** 2))
mae = np.mean(np.abs(np.clip(np.expm1(pred_log), 0, None) - np.expm1(eval_y)))
print(rmsle, mae)

'--------------------------------------------------' #testing actual holdout period via recursive prediction

#crete a copy of the data to use for recursive prediction
data_copy = data.copy()

data_copy['lag7'] = data.groupby(['store_nbr', 'family'])['sales'].shift(7)
data_copy['family'] = data_copy['family'].astype('category')
data_copy['store_nbr'] = data_copy['store_nbr'].astype('category')

h1_x = data_copy[(data_copy.date > '2017-07-30') & (data_copy.date <= '2017-08-6')].dropna(subset=['lag7'])
h1_y = np.log1p(h1_x['sales'])
sales_copy = h1_x['sales'].copy()
h1_x = h1_x.drop(['sales', 'date'], axis=1)


pred1 = model.predict(h1_x)

print(np.expm1(pred1))

h2_x = data_copy[(data_copy.date >= '2017-08-7') & (data_copy.date <= '2017-08-13')].dropna(subset=['lag7'])
h2_x['lag7'] = np.clip(np.expm1(pred1), 0, None)
h2_y = np.log1p(h2_x['sales'])

pred2 = model.predict(h2_x.drop(['sales', 'date'], axis=1))

h3_x = data_copy[(data_copy.date >= '2017-08-14') & (data_copy.date <= '2017-08-15')].dropna(subset=['lag7'])
h3_x['lag7'] = np.clip(np.expm1(pred2.reshape(-1, 7)[:, 0:2].ravel()), 0, None)
h3_y = np.log1p(h3_x['sales'])

pred3 = model.predict(h3_x.drop(['sales', 'date'], axis=1))

all_pred = np.concatenate([pred1, pred2, pred3])
all_y = np.concatenate([h1_y, h2_y, h3_y])


rmsle = np.sqrt(np.mean((all_pred - all_y) ** 2))
mae = np.mean(np.abs(np.clip(np.expm1(all_pred), 0, None) - np.expm1(all_y)))
print(rmsle, mae)
print(len(all_pred))

RUN_CV = True
if RUN_CV:
    starts = ['2017-06-13', '2017-06-29', '2017-07-15', '2016-10-30', '2016-12-18', '2017-02-20', '2016-08-01']
    scores = []
    for s in starts:
        start = pd.Timestamp(s)
        end = start + pd.Timedelta(days=15)
        tr = data[(data.date < start) & open_mask].dropna(subset=['lag7'])
        va = data[(data.date >= start) & (data.date <= end) & open_mask].dropna(subset=['lag7'])

        tr_y, va_y = np.log1p(tr['sales']), np.log1p(va['sales'])
        tr_x, va_x = tr.drop(['sales', 'date'], axis=1), va.drop(['sales', 'date'], axis=1)

        fold_model = LGBMRegressor(n_estimators=5000, learning_rate=0.5, verbose=-1).fit(
            tr_x, tr_y, eval_set=[(va_x, va_y)], callbacks=[early_stopping(50, verbose=False)])
        if s == '2016-12-18':
            err = (fold_model.predict(va_x) - va_y.values) ** 2
            print(pd.Series(err, index=va['date'].values).groupby(level=0).mean().pow(0.5))
        if s == '2016-08-01':
           err = (fold_model.predict(va_x) - va_y.values) ** 2
           print(pd.Series(err, index=va['date'].values).groupby(level=0).mean().pow(0.5))
        fold_rmsle = np.sqrt(np.mean((fold_model.predict(va_x) - va_y) ** 2))
        scores.append(fold_rmsle)
        print(s, fold_model.best_iteration_, round(fold_rmsle, 4))
    print(scores, np.mean(scores))
