import requests
import pandas as pd

def wazirx():
    resp = requests.get(
        'https://x.wazirx.com/api/v2/k?market=usdtinr&period=1440&limit=2000&timestamp=1529926492').json()
    df = pd.DataFrame(resp)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df.open_time = pd.to_datetime(df.open_time, unit='s')
    return df.rename(columns={'open_time':'date'})

def coindcx():
    url = 'https://api.coindcx.com/api/v1/chart/history_v2?symbol=USDTINR&resolution=1D&from=1486561175&to=1612791575'
    resp = requests.get(url).json()
    df = pd.DataFrame(resp['data'])
    df.time = pd.to_datetime(df.time, unit='ms')
    return df.rename(columns={'time':'date'})

def usdtinr():
    """
    10 July 2018 till 4th Feb 2019 uses wazirx data
    5th Feb 2019 onwards it uses coindcx data.
    returns a series
    """
    dfw = wazirx()
    dfc = coindcx()
    dfw = dfw[dfw.date<dfc.date.min()]
    df = pd.concat([dfw,dfc],ignore_index=True)
    s = df.set_index('date')['close']
    return s

if __name__ == "__main__":
    print(usdtinr())