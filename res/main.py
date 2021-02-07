import pandas as pd
from utils.smitra import sm_data_path, plot, normalize
import numpy as np
from utils.data import lvals, read_pq, write_pq, exists
from datetime import datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 1300)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

reference_date = datetime(2017, 10, 1)  # index value considered 100 as reference on this date


def clean_mcap_data():
    # if cleaned and stored already then read that
    if exists(f'data/cleaned_mcap.pq'):
        return read_pq(f'data/cleaned_mcap.pq')
    else:
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
        return mcap


def clean_candle_data():
    # if cleaned and stored already then read that
    if exists(f'data/cleaned_candle.pq'):
        return read_pq(f'data/cleaned_candle.pq')
    else:
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

        # candle data will never have USDT as target by we need it (mcap data shall have it)
        # add USDT data - at least make up date,target,close_usdt col data
        df_usdt = candle[candle.symbol=='BTCUSDT'].copy()    # coz that gives proper history and also that is correct mkt to
                                                        # use
        df_usdt['close_usdt'] = 1
        df_usdt['target'] = 'USDT'
        # append this to candle
        candle = pd.concat([candle,df_usdt],ignore_index=True)

        # write to parquet before returning
        write_pq(candle, f'data/cleaned_candle.pq')
        return candle


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
        df['wt_mcapwtd_capped'] = df.groupby(['date'])['mcap'].apply(lambda x: normalize(x,ub=0.4))
        # write to parquet before returning
        write_pq(candle, f'data/cleaned_candle_and_mcap.pq')
        return df


if __name__ == "__main__":
    """
    yet to 
    1. put min threshold on volume_usdt or its rolling avg (30 day median daily volume > 2 mn $)
    2. no single constituent can exceed 40% wt or less than 1%
    3. proxy for broader market
    4. rebalanced monthly
    5. trades wrt USD 
    """
    # read mcap data by coin
    mcap = clean_mcap_data()
    # read candle data
    candle = clean_candle_data()
    # add mcap info to candle
    df = join_candle_and_mcap(candle, mcap)

    # btc only index ; reference of 100 on 1st Oct 2017 when btc was at 4378
    index_btc_only = df[df.target == 'BTC'].set_index('date')['close_usdt']
    index_btc_only = index_btc_only * 100 / index_btc_only[reference_date]

    # mcap weighted index
    df['val_mcapwtd'] = df['close_usdt'] * df['wt_mcapwtd']
    index_mcap_wtd = df.groupby(['date'])['val_mcapwtd'].sum()
    index_mcap_wtd = index_mcap_wtd * 100 / index_mcap_wtd[reference_date]

    # mcap wtd capped index. max wt is 0.4
    df['val_mcapwtd_capped'] = df['close_usdt'] * df['wt_mcapwtd_capped']
    index_mcap_wtd_capped = df.groupby(['date'])['val_mcapwtd_capped'].sum()
    index_mcap_wtd_capped = index_mcap_wtd_capped * 100 / index_mcap_wtd_capped[reference_date]


    # number of coins over time in mcap wtd index
    plot(df.groupby(['date'])['target'].nunique())
    # number of coins over time in mcap wtd capped index
    plot(df[df.wt_mcapwtd_capped != 0].groupby(['date'])['target'].nunique())

    # btc weight over time in mcap wtd index
    plot(df[df.target=='BTC'].set_index('date')['wt_mcapwtd'])
    # plot USDT weight over time in mcap wtd index
    plot(df[df.target == 'USDT'].set_index('date')['wt_mcapwtd'])

    # btc weight over time in mcap wtd capped index
    plot(df[df.target=='BTC'].set_index('date')['wt_mcapwtd_capped'])
    # plot USDT weight over time in mcap wtd capped index
    plot(df[df.target == 'USDT'].set_index('date')['wt_mcapwtd_capped'])

    plot(index_btc_only)
    plot(index_mcap_wtd)
    plot(index_mcap_wtd_capped)


    def f(s):
        # s = s[s.index >= datetime(2020, 7, 1)]
        return s


    plt.plot(f(index_btc_only),'r')
    plt.plot(f(index_mcap_wtd),'b')
    plt.plot(f(index_mcap_wtd_capped),'g')
    plt.show()
