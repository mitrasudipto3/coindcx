from utils.data import read_pq
import pandas as pd
import numpy as np
from utils.smitra import sm_data_path, create_pivot
from datetime import datetime, timedelta, timezone
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 2500)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')


def round_off(number):
    """Round a number to the closest half integer.
    round_off(1.3)
    1.5
    round_off(2.6)
    2.5
    round_off(3.0)
    3.0
    round_off(4.1)
    4.0"""
    if pd.isna(number):
        return 1
    return round(number * 2) / 2


def epoch_to_dt(epoch):
    epoch = round(float(epoch))
    # returns utc time
    return datetime.utcfromtimestamp(epoch).replace(tzinfo=timezone.utc)


def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + timedelta(hours=t.minute // 30))


def min_max_scale_cs(df):
    # wide form df with date in index and symbols in cols. each number replaced by cross sectionally (same date)
    # min max scaled value. (val-min)/(max-min)
    # finally multipled by 100 to get into % form
    df = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)
    return df * 100

"""
Doc about this code and model is 
https://docs.google.com/document/d/1u_ABpyzTrRe2C-f3uorF4hK2u4-IQQlxBeFKraPUAEg/edit?usp=sharing
Owned by Sudipto. Anybody with link can view. 

data queried on superset https://bi.coindcx.com/superset/sqllab
Query for metadata (leverage) on each pair
select * from quant_team.currency_pairs

Query for candlestick data
select * from quant_team.binance_candle_sticks
where duration ='1h'
and open_time >1617667200

Query for cmc liquidity score (data updated hourly) - we need only latest so check if data overshoots 10 lac row limit
select * from market_data.cmc_liquidity_score

Query for dcx volume on binance tickers
SELECT created_hour::date created_date,market,sum(volume_usdt) from quant_team.binance_dcx_hourly_volume 
where created_hour >current_date - 45
group by created_date,market 
order by created_date,market

Note: internal_candle_sticks returns INR pair candles. But huobi and hitbtc return no data. Ignoring all 3. So 
keeping just binance. Hence even in metadata ensure you keep binance only (ecode=B)
We need upto 30+ day, hour candle volume info in USD from db quant_team.binance_candle_sticks_conv
and 60+ hr, hr candle for volatility from quant_team.binance_candle_sticks

Datetime to unix timestamp converter
https://www.epochconverter.com/
"""


def dynamic_leverage():
    # label of a candle shall be its close time (ie right side)
    df1 = pd.read_csv(f'{sm_data_path()}/data/candles_hour_binance_volatility.csv',
                      parse_dates=['close_time', 'open_time'])
    df1[['close']] = df1[['close']].astype(float)
    df1 = df1.sort_values(by=['close_time'])
    df1.close_time = df1.close_time.apply(lambda x: hour_rounder(epoch_to_dt(x)))
    # remove UTC to no tz. It is in UTC though. This is imp as later DCX volume will be in no tz
    df1.close_time = df1.close_time.dt.tz_convert(None)
    df1 = create_pivot(df1[['symbol', 'close_time', 'close']], col1='close_time', col2='symbol')
    df1 = df1.pct_change()  # returns from close
    df1 = df1.rolling(window=60, min_periods=2).std()  # 60 hr volatility (to replace 60 min and stands for 2.5 days)
    df1 = min_max_scale_cs(df1)  # normalize is min max scaling cross sectionally
    df1 = df1.unstack().reset_index(name='volatility')
    df2 = pd.read_csv(f'{sm_data_path()}/data/candles_hour_binance_volume.csv', parse_dates=['close_time', 'open_time'])
    df2[['quote_asset_volume', 'close_usdt_price']] = df2[['quote_asset_volume', 'close_usdt_price']].astype(float)
    df2['volume'] = df2['quote_asset_volume'] * df2['close_usdt_price']
    df2 = df2.sort_values(by=['close_time'])
    # dt will be rounded to an hour so no 23:59:59
    df2.close_time = df2.close_time.apply(lambda x: hour_rounder(epoch_to_dt(x)))
    # remove UTC to no tz. It is in UTC though. This is imp as later DCX volume will be in no tz
    df2.close_time = df2.close_time.dt.tz_convert(None)
    df2 = df2[['symbol', 'close_time', 'volume']].drop_duplicates(['symbol', 'close_time']).reset_index(drop=True)
    df2 = create_pivot(df2[['symbol', 'close_time', 'volume']], col1='close_time', col2='symbol')
    # lot of hourly volume data gaps on each symbol so use last hrly sample volume there else daily sum will
    # not make sense.
    df2 = df2.ffill()
    df2 = df2.resample('D', closed='right', label='right').sum()
    # volume = 0.5*adv7+0.5*adv30
    df2 = 0.5 * df2.rolling(window=7, min_periods=2).mean() + 0.5 * df2.rolling(window=30, min_periods=2).mean()
    adv = df2.copy()  # will be used to scale dcx_volume later on so save it
    df2 = min_max_scale_cs(df2)  # normalize is min max scaling cross sectionally
    df2 = df2.unstack().reset_index(name='volume')
    # inner join on close_time, symbol so only points where day AND hour candlestick meet are left
    df = pd.merge(df1, df2, how='inner', on=['symbol', 'close_time'])
    df = df[df.close_time == df.close_time.max()]
    print(df.symbol.nunique())
    candle_symbols = set(df.symbol.unique())
    dfm = pd.read_csv(f'{sm_data_path()}/data/pairs_meta_data.csv')
    # keep Binance only as candle data for HitBTC and Huobi not avlbl. INR (internal not considered for margin)
    dfm = dfm[dfm.ecode == 'B']
    # only keep relevant col ie max leverages
    dfm[['max_leverage', 'max_leverage_short']] = dfm[['max_leverage', 'max_leverage_short']].astype(float)
    dfm = dfm[['symbol', 'max_leverage', 'max_leverage_short']]
    print(dfm.symbol.nunique())
    metadata_symbols = set(dfm.symbol.unique())
    # symbols in metadata but not in candles = 222
    print(metadata_symbols.difference(candle_symbols))
    # symbols in candle but not in metadata = 2
    print(candle_symbols.difference(metadata_symbols))
    # merge metadata and candledata
    df = pd.merge(df, dfm, how='inner', on=['symbol'])
    df['max_leverage'] = df['max_leverage'].fillna(1)

    # binance margins
    dfb = pd.read_csv(f'{sm_data_path()}/data/binance_margins.csv')[['symbol', 'margin_ratio']]
    dfb['margin_ratio'] = pd.to_numeric(dfb.margin_ratio)
    dfb['symbol'] = dfb['symbol'].astype(str)
    dfb = dfb.rename(columns={'margin_ratio': 'binance_leverage'})
    print(f'Binance margins data has {dfb.symbol.nunique()} symbols')
    print(f'Inner merged volume and volatility data has {df.symbol.nunique()} symbols')
    # keep all symbols that were there in merged candle data
    df = pd.merge(df, dfb, how='left', on='symbol')
    df = df.drop_duplicates(subset=['symbol', 'close_time'])
    df['binance_leverage'] = df['binance_leverage'].fillna(1)  # let nan stay nan

    # add cmc liquidity
    dfc = pd.read_csv(f'{sm_data_path()}/data/cmc_liquidity.csv')
    dfc['close_time'] = pd.to_datetime(dfc['quote_usd_last_updated']).apply(lambda x: hour_rounder(x))
    dfc['symbol'] = dfc['market_pair'].apply(lambda x: x.replace(' ', ''))
    # keep latest row for each pair
    dfc = dfc.sort_values(by=['close_time', 'symbol']).drop_duplicates(subset=['symbol'], keep='last')
    dfc['cmc_liquidity'] = pd.to_numeric(dfc['quote_usd_effective_liquidity'])
    df = pd.merge(df, dfc[['symbol', 'cmc_liquidity']], how='inner',
                  on='symbol')  # 7 symbols drop among 613 ; very less

    # add dcx volume
    dfd = pd.read_csv(f'{sm_data_path()}/data/dcx_volume.csv').rename(columns={'market': 'symbol', 'sum': 'dcx_volume'})
    dfd['close_time'] = pd.to_datetime(dfd['created_date'])
    dfd['dcx_volume'] = pd.to_numeric(dfd['dcx_volume'])
    dfd = create_pivot(dfd[['close_time', 'symbol', 'dcx_volume']], 'close_time', 'symbol')
    # volume = 0.5*adv7+0.5*adv30
    dfd = 0.5 * dfd.rolling(window=7, min_periods=2).mean() + 0.5 * dfd.rolling(window=30, min_periods=2).mean()
    dfd = dfd / adv  # dcx volume / binance volume
    dfd = dfd.unstack().reset_index(name='dcx_volume')
    # if no dcx vol data then we will later fill it with nan
    df = pd.merge(df, dfd, on=['symbol', 'close_time'], how='left')
    df['ratio_dcx_binance_volume'] = df['dcx_volume'] / df['volume']
    # multiply the below factor (specific to pair) to new leverage (common for cluster) to get final new leverage
    # factor is always between 1x and 2x. If dcx volume is 1% of binance then it is 1x
    df['factor'] = 0.01 * 1.5 / df['ratio_dcx_binance_volume']
    df['factor'] = df['factor'].fillna(1)  # nan data in dcx/binance volume must lead to 1 factor multiplier
    df['factor'] = np.maximum(1, df['factor'])  # new leverage to be >= 1x that of binance
    df['factor'] = np.minimum(2, df['factor'])  # new leverage to be <= 3x that of binance

    # clustering
    kmeans = KMeans(n_clusters=25).fit(df[['volume', 'volatility', 'cmc_liquidity']])
    df['label'] = kmeans.labels_
    # get common new leverage for each cluster
    df['new_leverage'] = df.groupby(['label'])['binance_leverage'].transform(
        lambda x: (np.nanmax(x) + np.nanmin(x) + np.nanmedian(x) + np.nanmean(x)) / 4)
    # now multiply cluster's leverage by pair specific factor dependent on dcx/binance volume ratio
    # it is chosen such that factor is between 1x and 2x. If dcx volume is 1% of binance then it is 1.5x
    # round off final leverage to nearest 0.5
    df['final_new_leverage'] = (df['new_leverage'] * df['factor']).apply(lambda x: round_off(x))
    print(df)
    print(df[['label', 'final_new_leverage']].drop_duplicates().sort_values(by=['label']))
    print(df.max_leverage.value_counts().sort_index())
    s_dcx = df.max_leverage.value_counts().sort_index()
    print(df.binance_leverage.value_counts().sort_index())
    s_binance = df.binance_leverage.value_counts().sort_index()
    print(df.final_new_leverage.value_counts().sort_index())
    s_new = df.final_new_leverage.value_counts().sort_index()
    dfs = pd.merge(s_dcx, s_binance, how='outer', left_index=True, right_index=True)
    dfs = pd.merge(dfs, s_new, how='outer', left_index=True, right_index=True)
    dfs = dfs.reset_index().rename(columns={'index': 'leverage', 'max_leverage': 'current_dcx_max_leverage'})
    print(dfs)
    ax = dfs.plot(kind='bar', x='leverage', subplots=True)
    plt.show()
    print(df.label.value_counts())
    print(kmeans.cluster_centers_)
    print(df[df.max_leverage > 1].symbol.nunique())
    print(df[df.final_new_leverage > 1].symbol.nunique())
    # now a full frame with pair symbol and new leverage
    print(df.fillna(None))


"""
X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
>>> kmeans.labels_
array([1, 1, 1, 0, 0, 0], dtype=int32)
>>> kmeans.predict([[0, 0], [12, 3]])
array([1, 0], dtype=int32)
>>> kmeans.cluster_centers_
array([[10.,  2.],
       [ 1.,  2.]])
"""

if __name__ == "__main__":
    dynamic_leverage()
