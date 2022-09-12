#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np;
import pandas as pd;

#Binance python integration library
from binance.client import Client
import config


# In[27]:


# when I want to change this, need to do file>download as>.py>save to windows/users/colli

symbol = "BTCUSD"
asset = 'BTC'

Stop = 21500
Target = 100000

AccumulationZoneNO_STOPS = 20000

# value represents decimals required for asset purchase
precision = 4

# NewStop = 21700
# NewTarget = 24500


# In[12]:


apiKey = "yraM0G206IlWudMPhs5OYeDKvBFaQtExG3HvpkXhb6mNZ3tUCY2D32lCIUQDEXyU"
apiSecret = "Ws4M63iwO270Oab9v5ETqmVtnZdEAY6TadqMOnNUbxales2t3kaN6pzBxLtF4Dxs"

client = Client(apiKey, apiSecret, tld='us')
print("logged in")


# In[28]:


def getminutedata(symbol):
    frame = pd.DataFrame(client.get_historical_klines(symbol,
                                                        "1m",
                                                        "1 minute ago UTC"))
    frame = frame.iloc[:,:5]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close']
    frame[['Open', 'High', 'Low', 'Close']] = frame[['Open', 'High', 'Low', 'Close']].astype(float)
    frame.Time = pd.to_datetime(frame.Time, unit='ms')
    return frame

df = getminutedata(symbol)
CurrentPrice = df['Close'].values[0]
df


# In[32]:


account_info = client.get_account()
balance = account_info['balances']

balances = pd.DataFrame(balance)
balances[['free', 'locked']] = balances[['free', 'locked']].astype(float)
symbol_balance = balances.loc[balances['asset'] == asset, 'free'].iloc[0]
symbol_balance = round(symbol_balance, precision)
USD_balance = balances.loc[balances['asset'] == 'USD', 'free'].iloc[0]

print(symbol_balance)
print(asset)
print(USD_balance) 
print("USD")


# In[33]:


# if price > target AND position > 0, sell at market
# if price < stop loss AND position > 0, sell at market
    
if CurrentPrice < Stop and symbol_balance > 0.0003:
    order = client.create_order(
    symbol = symbol,
    side = 'sell',
    type = 'market',
    quantity = symbol_balance,
    #timeInForce = TIME_IN_FORCE_GTC,
    #price = 0,
    )
    print("Stop Loss Executed")
    
if CurrentPrice > Target and symbol_balance > 0.0003:
    order = client.create_order(
    symbol = symbol,
    side = 'sell',
    type = 'market',
    quantity = symbol_balance,
    #timeInForce = TIME_IN_FORCE_GTC,
    #price = 0,
    )
    print("Take Profit Executed")


# In[23]:


#this is essentially a glorified limit order. only really useful if below stop

buy_size = USD_balance*.98
qty = buy_size/CurrentPrice
qty = round(qty, precision)

if CurrentPrice < AccumulationZoneNO_STOPS and symbol_balance < 0.0003:
    order = client.create_order(
    symbol = symbol,
    side = 'buy',
    type = 'market',
    quantity = qty)
    #timeInForce = TIME_IN_FORCE_GTC,
    #price = 0,
    
    print("Asset Accumulated")

