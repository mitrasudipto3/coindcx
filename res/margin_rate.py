import pandas as pd
from utils.smitra import sm_data_path

if __name__ == "__main__":
    # df = pd.read_csv(f'{sm_data_path()}/data/margin_rate/lend_assets.csv')
    # df['date'] = pd.to_datetime(df['dates'])
    # df['amount_usd'] = df['amount_usd'].str.replace(',','').astype(float)
    # df = df[['date','amount_usd']]

    btc_usdt = pd.read_csv(f'{sm_data_path()}/data/margin_rate/btc_usdt.csv', parse_dates=['open_date'])
    btc_usdt['date'] = btc_usdt['open_date']
    btc_usdt['btcusdt'] = btc_usdt['close_price'].astype(float)
    btc_usdt = btc_usdt[['date', 'btcusdt']]

    df = pd.read_csv(f'{sm_data_path()}/data/margin_rate/lend_assets_by_token.csv', parse_dates=['dates'])
    df['date'] = df['dates']
    for col in ['amount_token', 'amount_usd', 'amount_btc']:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(',', '').astype(float)
    # add btc usd column
    df = pd.merge(df, btc_usdt, how='left', on='date')
    # add token usd column
    df['token_usd'] = df['amount_usd'] / df['amount_token']
    for coin in df.currency.unique():
        print(coin)
        df1 = df[df.currency == coin]
        print(f"corr with btcusdt = {format(df1['amount_usd'].corr(df1['btcusdt']), '.2f')}")
        print(f"corr with token_usdt = {format(df1['amount_usd'].corr(df1['token_usd']), '.2f')}")
