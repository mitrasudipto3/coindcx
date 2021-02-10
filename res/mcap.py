#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from dcxdatamodule import etl_utilities

service = etl_utilities.Util(config)

# In[2]:


url = 'https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical?convert={}&symbol={}&time_end={}&time_start={}'

# In[3]:


start_date = '2012-01-01'
end_date = '2021-03-01'
df_currencies = pd.read_csv('currencies_binance_dcx.csv')

# In[5]:


for currency in df.currency:
    try:
        _json = requests.get(url.format('USD', currency, end_date, start_date)).json()['data']['quotes']
        df_price_temp = pd.DataFrame([key['quote']['USD'] for key in _json])
        df_price_temp['currency'] = currency
        df_price_temp = df_price_temp.rename(columns={'timestamp': 'open_date',
                                                      'open': 'open_price',
                                                      'close': 'close_price',
                                                      'high': 'high_price',
                                                      'low': 'low_price'})
        service.pandas_to_redshift(df_price_temp, 'market_data.market_cap_cmc_temp', False)
        service.redshift_upserter('market_data.market_cap_cmc_temp', 'market_data.market_cap_cmc',
                                  unique_columns=['open_date', 'currency'])
    except Exception as e:
        print(e)

        _json = requests.get(url.format('USD', currency, end_date, start_date)).json()

        #         missed_currency.append(currency)
        print(currency)

# In[24]:


# In[9]:


# In[8]:


# In[ ]:




