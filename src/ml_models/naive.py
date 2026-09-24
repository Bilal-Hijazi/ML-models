import pandas as pd

import pandas as pd

train_data = pd.read_csv('store-sales-time-series-forecasting/train.csv',
                         parse_dates=['date'])
train_data = train_data.drop('id', axis=1)

full = pd.MultiIndex.from_product(
    [train_data.store_nbr.unique(),
     train_data.family.unique(),
     pd.date_range(train_data.date.min(), train_data.date.max(), freq='D')],
    names=['store_nbr', 'family', 'date'])

train_data = (train_data
              .set_index(['store_nbr', 'family', 'date'])
              .reindex(full, fill_value=0)
              .reset_index())

train_data['lag7'] = train_data.groupby(['store_nbr', 'family'])['sales'].shift(7)


print(len(train_data))                 # 3007584
print(train_data.lag7.isna().sum())    # 12474

d = train_data.assign(
    lag7_date=train_data.groupby(['store_nbr','family'])['date'].shift(7))
print((d.date - d.lag7_date).value_counts())


import numpy as np

val = train_data[train_data.date >= '2017-07-31'].dropna(subset=['lag7'])

mae   = (val.sales - val.lag7).abs().mean()
rmsle = np.sqrt(((np.log1p(val.lag7) - np.log1p(val.sales))**2).mean())
print(len(val), mae, rmsle)

err = (val.sales - val.lag7).abs()
print(err.describe())
print(err.quantile([0.5, 0.9, 0.99]))

top1 = err.nlargest(286)
print(top1.sum() / err.sum())

import numpy as np

v = val.assign(
    err=(val.sales - val.lag7).abs(),
    sle=(np.log1p(val.lag7) - np.log1p(val.sales))**2)

print(v.groupby('family').err.mean().sort_values(ascending=False).head(10))
print(v.groupby('family').sle.mean().pow(0.5).sort_values(ascending=False).head(10))