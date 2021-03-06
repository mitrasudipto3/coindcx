import requests
from datetime import timezone
import time
import pandas as pd
from utils.smitra import sm_data_path
from utils.smitra import pmap

from datetime import datetime

url = "https://api.binance.com/api/v1/klines?symbol={}&interval=1d&startTime={}"
market_urls = 'https://api.coindcx.com/'
markets = pd.DataFrame(requests.get(market_urls).json()['currency_pairs'])
markets = markets[markets.ecode == 'B'].coindcx_name

def regen_candles(parallel=True):
    years = list(range(2018,datetime.now().year+1))
    print(years)

    df = pd.DataFrame()

    if not parallel:
        for year in years:
            ts = datetime(year, 1, 1)
            ts = int(time.mktime(ts.timetuple()) * 1000)
            i = 0
            for market in markets:

                df_temp = pd.DataFrame(data=requests.get(url.format(market, ts)).json())
                if len(df_temp) > 0:
                    df_temp.columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore_column']
                    df_temp['date'] = pd.to_datetime(df_temp.open_time, unit='ms')
                    df_temp['market'] = market
                    df = df.append(df_temp)
                i = i + 1

                print(year, market, i)
    else:
        for year in years:
            ts = datetime(year, 1, 1)
            ts = int(time.mktime(ts.timetuple()) * 1000)
            def do_mkt(market):
                print(year, market)
                df_temp = pd.DataFrame(data=requests.get(url.format(market, ts)).json())
                if len(df_temp) > 0:
                    df_temp.columns = ['open_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore_column']
                    df_temp['date'] = pd.to_datetime(df_temp.open_time, unit='ms')
                    df_temp['market'] = market
                    return df_temp
                else:
                    return pd.DataFrame()
            dfy = pmap(do_mkt,markets)  # all markets for 1 year
            dfy = pd.concat(dfy,ignore_index=True)
            df = pd.concat([df,dfy],ignore_index=True)  # append to all year's info

    print(df)
    df.to_csv(f'{sm_data_path()}/data/candles_sticks.csv')
