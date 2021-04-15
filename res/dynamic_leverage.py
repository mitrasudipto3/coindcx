from utils.data import read_pq
import pandas as pd
import numpy as np
from utils.smitra import sm_data_path, create_pivot
from datetime import datetime, timedelta, timezone
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy import stats

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


def get_score_tags(s):
    # cutoffs given in
    # https://coindcx.atlassian.net/wiki/spaces/GB/pages/700449272/Bucketing+of+Markets+for+Dynamic+Leverage
    def volume_cutoff(x):
        if x <= 0.02:
            return 'u'
        elif x <= 0.15:
            return 'l'
        elif x <= 1.07:
            return 'm'
        else:
            return 'h'

    def volatility_cutoff(x):
        if x <= 0.19:
            return 'u'
        elif x <= 1.07:
            return 'l'
        elif x <= 2.89:
            return 'm'
        else:
            return 'h'

    s['volume_tag'] = s.volume.apply(lambda x: volume_cutoff(x))
    s['volatility_tag'] = s.volatility.apply(lambda x: volatility_cutoff(x))
    s['tag'] = s['volume_tag'] + s['volatility_tag']
    # be default leverage 1
    mydict = defaultdict(lambda: 1)
    # those tags that map to other than 1
    for key in ['lu', 'll', 'mh']:
        mydict[key] = 2

    # print(s)
    # exit()
    return s


"""
data queried on superset https://bi.coindcx.com/superset/sqllab
Query for metadata (leverage) on each pair
select * from quant_team.currency_pairs

Query for candlestick data
select * from quant_team.binance_candle_sticks
where duration ='1h'
and open_time >1617667200

Note: internal_candle_sticks returns INR pair candles. But huobi and hitbtc return no data. Ignoring all 3. So 
keeping just binance. Hence even in metadata ensure you keep binance only (ecode=B)
We need upto 90 day, day candle volume info and 60 hr, hr candle volatility

Datetime to unix timestamp converter
https://www.epochconverter.com/
"""


def load_data():
    # label of a candle shall be its close time (ie right side)
    df1 = pd.read_csv(f'{sm_data_path()}/data/candles_hour_binance_volatility.csv',
                      parse_dates=['close_time', 'open_time'])
    df1[['close']] = df1[['close']].astype(float)
    df1 = df1.sort_values(by=['close_time'])
    df1.close_time = df1.close_time.apply(lambda x: hour_rounder(epoch_to_dt(x)))
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
    print(df2[df2.symbol == 'AAVEUSDT'])
    df2 = df2[['symbol', 'close_time', 'volume']].drop_duplicates(['symbol', 'close_time']).reset_index(drop=True)
    df2 = create_pivot(df2[['symbol', 'close_time', 'volume']], col1='close_time', col2='symbol')
    # lot of hourly volume data gaps on each symbol so use last hrly sample volume there else daily sum will
    # not make sense.
    df2 = df2.ffill()
    df2 = df2.resample('D', closed='right', label='right').sum()
    # volume = 0.5*adv7+0.5*adv30
    df2 = 0.5 * df2.rolling(window=7, min_periods=2).mean() + 0.5 * df2.rolling(window=30, min_periods=2).mean()
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
    df['binance_leverage'] = df['binance_leverage'].fillna(1)

    # clustering
    kmeans = KMeans(n_clusters=25).fit(df[['volume', 'volatility', 'binance_leverage']])
    df['label'] = kmeans.labels_
    df['new_leverage'] = df.groupby(['label'])['binance_leverage'].transform(
        lambda x: round_off((np.max(x) + np.min(x) + np.median(x) + np.mean(x)) / 4))
    # index has label and the two columns give coordinates of centroids
    s = pd.DataFrame(kmeans.cluster_centers_, columns=['volume', 'volatility', 'binance_leverage'])
    # get ultra low, low, medium and high tags (cutoffs centrally defined in static manner)
    # s = get_score_tags(s)
    print(df)
    print(df[['label', 'new_leverage']].drop_duplicates().sort_values(by=['label']))
    print(df.max_leverage.value_counts())
    print(df.binance_leverage.value_counts())
    print(df.new_leverage.value_counts())
    print(df.label.value_counts())
    print(kmeans.cluster_centers_)
    print(df[df.max_leverage > 1].symbol.nunique())
    # print(s)


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
    load_data()
    """
    Deliverables:
    1. make drawdown lesser in index without including stablecoin (KPS)
    2. new coins we can provide leverage >1 on.
    3. new model (and working code) of leverage calculation. One which also allows for fractional leverages.
    """
