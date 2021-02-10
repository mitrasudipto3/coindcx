import pandas as pd
from utils.smitra import sm_data_path, plot, normalize, create_pivot, reference_date
import numpy as np
from utils.data import lvals, read_pq, write_pq, exists
from datetime import datetime
import matplotlib.pyplot as plt
from res.usdtinr import usdtinr

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def clean_mcap_data():
    # read mcap data by coin
    mcap = pd.read_csv(f'{sm_data_path()}/data/market_cap_cmc.csv').rename(
        columns={'currency': 'target', 'market_cap': 'mcap'})
    # Confirmed with Nitish: the file 'open_date' col is actually close time of the candle
    # convert Jan 1 23:59:59.9999 to Jan 1 00:00 ; will help in joining with candle data later
    mcap['date'] = pd.to_datetime(pd.to_datetime(mcap['open_date']).dt.date)
    # keep just relevant cols
    mcap = mcap[['date', 'target', 'mcap']]
    # set types
    mcap.target = mcap.target.astype(str)
    mcap.mcap = mcap.mcap.astype(float)
    # drop zero mcap data
    mcap = mcap[mcap.mcap != 0]
    # Note: mcap is in usdt

    # write to parquet before returning
    write_pq(mcap, f'data/cleaned_mcap.pq')
    # write mcap as wide form (in USDT)
    write_pq(create_pivot(mcap[['date', 'target', 'mcap']]), f'data/mcap.pq')


def clean_candle_data():
    # read candle data, set names and restrict cols
    candle = pd.read_csv(f'{sm_data_path()}/data/candles_sticks.csv').rename(
        columns={'close_price': 'close', 'volume': 'volume_target', 'quote_asset_volume': 'volume_base',
                 'market': 'symbol'})
    candle = candle[['date', 'symbol', 'close', 'volume_target', 'volume_base']]
    pairs = pd.read_csv(f'{sm_data_path()}/data/currency_pairs.csv').rename(
        columns={'base_currency_short_name': 'base', 'target_currency_short_name': 'target'})
    pairs = pairs[['symbol', 'base', 'target']]

    # set types
    candle['old_date'] = candle['date']
    # VImp : not passing dayfirst=True will jumble up will misunderstand the date string and jumble up order of
    # historic time series data
    candle['date'] = pd.to_datetime(candle['date'], dayfirst=True)
    candle['symbol'] = candle['symbol'].astype(str)
    pairs[['symbol', 'base', 'target']] = pairs[['symbol', 'base', 'target']].astype(str)
    candle[['close', 'volume_target', 'volume_base']] = candle[['close', 'volume_target', 'volume_base']].astype(
        float)

    # add base and target info to each sumbol row in candle data
    candle = pd.merge(candle, pairs, on='symbol', how='inner')

    # only keep symbols with base as USDT or BTC ---> size goes from 36.8k rows to 22.3k rows which is totally okay
    # as index shall be at coin level and not symbol level. The implementation (execution) would need to be at symbol
    # level and for that BTC or USDT route should suffice.
    # symbols from from 529 to 340 unique ones ;
    # target coins drop from 223 to 214 unique ones which is totally not a big drop as expected.
    candle = candle[candle.base.isin({'BTC', 'USDT'})]

    btcusdt = candle[candle.symbol == 'BTCUSDT'][['date', 'close']].rename(columns={'close': 'btcusdt'})
    candle = pd.merge(candle, btcusdt, on='date',
                      how='inner')  # 'inner' means all will start on/after btcusdt start date

    # get volume, close price of the symbol (coin) in usdt using btcusdt as of that date
    candle['close_usdt'] = np.where(candle.base == 'USDT', candle.close, candle.close * candle.btcusdt)
    candle['volume_usdt'] = np.where(candle.base == 'USDT', candle.volume_base, candle.volume_base * candle.btcusdt)

    # same target coin may have 2 symbols with base BTC or USDT...we rely (for data as well as implementation) on the
    # market which has higher volume and ignore the other
    # Note: it does not guarantee that 1 target is always only gonna remain with 1 base but it does guarantee that
    # 1 target will be mapped to only 1 base as of a prticular date.
    # This is coz over time BTC base market may become more liquid than USDT base market and in that case we just
    # continue following the 'most liquid mkt'. The same can be used in implementation/execution and there should be
    # no issue.
    candle = candle.sort_values(by=['target', 'date', 'volume_usdt'],
                                ascending=False)  # higher volume_usdt comes up
    candle = candle.drop_duplicates(subset=['target', 'date'], keep='first').sort_values(by=['target', 'date'])

    # create daily return (note: already sorted by target,date)
    candle['ret'] = candle.groupby(['target'])['close_usdt'].pct_change()

    """
    the below USDT snippet has been commented out as mcap wtd index without USDT makes slightttly better perf than 
    with USDT which also makes sense. Plus USDT mcap is based on adoption more than price...it is unfair and inapt to
    include that in the index. Other indices by CIX100 / Bloomberg also avoid USDT as a constituent 
    """
    # # candle data will never have USDT as target by we need it (mcap data shall have it)
    # # add USDT data - at least make up date,target,close_usdt col data
    # df_usdt = candle[candle.symbol == 'BTCUSDT'].copy()  # coz that gives proper history and also that is correct mkt to
    # # use
    # df_usdt['close_usdt'] = 1
    # df_usdt['target'] = 'USDT'
    # # append this to candle
    # candle = pd.concat([candle, df_usdt], ignore_index=True)

    # write to parquet before returning
    write_pq(candle, f'data/cleaned_candle.pq')
    # write close_usdt
    write_pq(create_pivot(candle[['date', 'target', 'close_usdt']]), f'data/close_usdt.pq')
    # write volume_usdt
    write_pq(create_pivot(candle[['date', 'target', 'volume_usdt']]), f'data/volume_usdt.pq')


def join_candle_and_mcap(candle, mcap):
    # if cleaned and stored already then read that
    if exists(f'data/cleaned_candle_and_mcap.pq'):
        return read_pq(f'data/cleaned_candle_and_mcap.pq')
    else:
        df = pd.merge(candle, mcap, how='inner', on=['date', 'target'])
        df = df.sort_values(by=['date', 'target'])
        df['mcap_daily_total'] = df.groupby(['date'])['mcap'].transform('sum')
        df['wt_mcapwtd'] = df['mcap'] / df['mcap_daily_total']  # index wt is proportional to mcap
        # index wt is proportional to mcap ; max 0.4, min 0.025 wt
        df['wt_mcapwtd_capped'] = df.groupby(['date'])['mcap'].apply(lambda x: normalize(x, ub=0.4))
        # write to parquet before returning
        write_pq(candle, f'data/cleaned_candle_and_mcap.pq')
        return df

def clean_nifty(frame=False):
    """
    For this first download data from https://uk.investing.com/indices/s-p-cnx-nifty-historical-data
    and save as nifty.csv in data folder. Must have start date as 10th July 2018.
    make a series/frame with date in index (all cal days, data ffilled) and close in the the data
    returns series if frame=False
    """
    df = pd.read_csv(f'{sm_data_path()}/data/nifty.csv')[['Price','Date']].rename(columns={'Date':'date','Price':'nifty'})
    df['date'] = pd.to_datetime(df.date)
    df['nifty'] = pd.to_numeric(df['nifty'].apply(lambda x: x.replace(',','')))
    df = df.set_index('date').sort_index()
    if frame:
        write_pq(df,f'data/nifty.pq')
        return df
    else:
        write_pq(df['nifty'], f'data/nifty.pq')
        return df['nifty']

def index_standardisation(index):
    # index is a series
    # reference of 100 on a reference date
    return index * 100 / index[reference_date]


if __name__ == "__main__":
    """
    yet to 
    1. put min threshold on volume_usdt or its rolling avg (30 day median daily volume > 2 mn $)
    2. no single constituent can exceed 40% wt or less than 1%
    3. proxy for broader market
    4. rebalanced monthly
    5. trades wrt USD 
    """
    clean_nifty()
    nifty = read_pq(f'data/nifty.pq')['nifty']
    clean_mcap_data()
    mcap = read_pq(f'data/mcap.pq')
    clean_candle_data()
    close_usdt = read_pq(f'data/close_usdt.pq')
    volume_usdt = read_pq(f'data/volume_usdt.pq')
    usdtinr = usdtinr()  # series with date in index
    close = close_usdt.mul(usdtinr, axis=0)  # in inr now

    # keep mcap data only for coins with proper candle data and vice versa (nan all others out)
    mcap = mcap * close.mask(pd.notna(close), 1)
    close = close * mcap.mask(pd.notna(mcap), 1)

    # btc only index ; reference of 100 on 1st Oct 2017 when btc was at 4378
    index_btc_only = close['BTC']
    index_btc_only = index_standardisation(index_btc_only)

    # MCAP wtd index
    wt_mcap = normalize(mcap)
    index_mcap = (close * wt_mcap).sum(axis=1)  # summed over cols so 1 val for each date
    index_mcap = index_standardisation(index_mcap)

    # mcap wtd capped index. max wt = 0.4
    wt_mcap_capped = normalize(mcap, ub=0.4)
    index_mcap_capped = (close * wt_mcap_capped).sum(axis=1)  # summed over cols so 1 val for each date
    index_mcap_capped = index_standardisation(index_mcap_capped)

    # nifty index
    nifty = index_standardisation(nifty)

    # df = wt_mcap.unstack().replace(0, np.nan).dropna().reset_index(name='wt').rename(
    #     columns={'level_0': 'target', 'level_1': 'date'})
    # print(df)
    # print(df.target.nunique())
    # print(df[df.date==df.date.max()].target.nunique())
    # fx = pd.read_csv(f'{sm_data_path()}/data/currency_pairs.csv')
    # print(fx)
    # fx = fx[fx.target_currency_short_name.isin(set(df.target.unique()))]
    # fx = fx[['target_currency_short_name','target_currency_name']].drop_duplicates()
    # print(fx)
    # fx.to_csv(f'{sm_data_path()}/data/potential_constituents.csv')
    # exit()

    plot(index_btc_only)
    plot(index_mcap)
    plot(index_mcap_capped)
    plot(nifty)

    def f(s):
        # s = s[s.index >= datetime(2020, 7, 1)]
        return s


    plt.plot(f(index_btc_only), 'r')
    plt.plot(f(index_mcap), 'b')
    plt.plot(f(index_mcap_capped), 'y')
    plt.plot(f(nifty), 'g')
    plt.show()
