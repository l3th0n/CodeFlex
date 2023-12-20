from websocket import create_connection
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import json, re, pprint, time, asyncio, os, requests
# import pprint
# import time
# import asyncio
import pandas as pd
# import re
import numpy as np
# import os

stocknews_url = 'https://stocknewsapi.com/api/v1?tickers={}&items=50&token=i579pxeya5umabkbyfvasyczrewokx0laezgrbuu'
stocknews_req = requests.get(stocknews_url.format('GOOGL'))
print(stocknews_req.status_code)


tickers = pd.read_csv('supported_tickers.csv')
tickers_gathered = []
# print(tickers.head())

def discord_chat():
    channel_dct = {
        'AtlasTrading': {
            'SC_trading_floor': [458013311846449152, "AT_SC_trading_floor"],
            'SC_swings': [457671525743591425, "AT_SC_swings"],
            #'SC_alerts': [681492147299680336, "AT_SC_alerts"],
            'LC_trading_floor': [751100183655350373, "AT_LC_trading_floor"],
        },
        'Sapphire': {
            'M_pro_trader_channel': [748896255081447506, "S_M_pro_trader_channel"]
        }
    }

    chat_df = {
        'AT_SC_trading_floor': None,
        'AT_SC_swings': None,
        'AT_LC_trading_floor': None,
        'S_M_pro_trader_channel': None
    }

    date_long = datetime.today() - timedelta(days=1)
    date_short = str(date_long)[:10]
    # print(date_short)

    for key in channel_dct.keys():
        for chan in channel_dct[key]:
            channel_lst = channel_dct[key][chan]
            output_file = 'D:\Google Drive\Programmering\TradingBotPython\{}_{}.csv'.format(channel_lst[1], date_short)
            if not os.path.exists(output_file):
                os.system('DiscordChatExporter\DiscordChatExporter.CLI.exe export -t "NTMyNjI1NTY5ODMwMTQxOTUz.YDIZ-w.5VUjzqoruIsN17MLGIQubGFKf0U" \
                -c {} -f Csv -o "{}" --after "{}" --dateformat unixms'.format(channel_lst[0], output_file, date_long))
            try:
                temp_df = pd.read_csv(output_file)
            except FileNotFoundError:
                print('Not found: {}'.format(output_file))

            temp_df.drop(columns=['AuthorID', 'Attachments'], inplace=True)

            temp_df['Reactions'] = temp_df['Reactions'].str.split(',')
            temp_df['Reactions'] = temp_df['Reactions'].apply(lambda x: len(x) if type(x) != type(np.nan) else 0)
            chat_df[channel_lst[1]] = temp_df


    for key in chat_df.keys():
        # print(chat_df[key])
        chat_df[key] = chat_df[key].dropna()
        chat_df[key]['Tickers'] = chat_df[key]['Content'].str.findall(r'([A-Z]{2,5})\s+')
        # print(chat_df[key].head())
        tickers_gathered = tickers_gathered + chat_df[key]['Tickers'].explode().dropna().tolist()

    print(pd.DataFrame(tickers_gathered).value_counts())



def seconds_delay(time_in):
    return round((time.time_ns()-time_in)/1e9, 2)

def stock_ds():
    ws = create_connection("wss://api.tiingo.com/iex")

    subscribe = {
            'eventName':'subscribe',
            'authorization':'741b0578fbf3f680d602eafe28b6d3990caa5181',
            'eventData': {
                'thresholdLevel': 5,
                'tickers': ['AMC', 'GME', 'GPRO']
        }
    }

    ws.send(json.dumps(subscribe))
    while True:
        ws_msg = json.loads(ws.recv())
        # print(ws_msg)

        if ws_msg['messageType'] == "I": print("Initialization: {}. SubscriptionID: {}.".format(ws_msg['response']['message'], ws_msg['data']['subscriptionId']))
        if ws_msg['messageType'] == "H": print("...")
        if ws_msg['messageType'] == "A":
            ws_data = ws_msg['data']
            if ws_data[0] == "T":
                print("T:Ticker: {}. Delay: {}. Latest Price: {}. Volume: {}.".format(\
                ws_data[3], seconds_delay(ws_data[2]), ws_data[9], ws_data[10]))
            else:
                print("Q:Ticker: {}. Delay: {}. Mid Price: {}.".format(\
                ws_data[3], seconds_delay(ws_data[2]), ws_data[6]))



# discord_chat()
# stock_ds()
