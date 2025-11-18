#!/usr/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime, timedelta,timezone
import os
import sys
import threading
import time
import matplotlib.pyplot as plt
import joblib
import ccxt
#import talib
#import backtrader as bt
from tqdm import tqdm
from collections import deque

import common_eng
from common_eng import sleepTime, dict_period, dict_plot

arr_PeriodTick=["15m"]
arr_PeriodS=["15m","30m","1h","4h","1d"]
arr_PeriodI=[15,30,60,240,1440]

work_dir=os.path.dirname(os.path.abspath(__file__))
dataFile="products--transformer.txt"
class dataThread() :
    def readTxt(self,file_name):    
        data = []
        file = open(file_name,'r') 
        file_data = file.readlines()    
        if len(file_data)>0:
            line=file_data[0]
            line=line.strip()    
            tmp_list = line.split(',')
            for obj in tmp_list:
                data.append(obj)                    
        return data

    def getKData(self,symbol,period,type,dataYear):
        thread2 = common_eng.commonThread()
        dataFrame = pd.DataFrame()

        klines_amount =int(dataYear*365*24*60/arr_PeriodI[thread2.refArrayInd(arr_PeriodS,period)])+(int(arr_PeriodI[len(arr_PeriodI)-1]/arr_PeriodI[thread2.refArrayInd(arr_PeriodS,period)]))*200
        if type=="real":
            klines_amount =1000
            
        exchange = ccxt.binance({
            'rateLimit': 1000,
            'enableRateLimit': True,
            # 'verbose': True,
        })
        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=klines_amount *arr_PeriodI[thread2.refArrayInd(arr_PeriodS,period)])
        temp=since.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        from_timestamp= exchange.parse8601(temp)
        data = []
        now = exchange.milliseconds()
        
        while from_timestamp <now:
            try:
                candles = exchange.fetch_ohlcv(symbol,timeframe=period,since=from_timestamp,limit=1000)
                if len(candles)<=0:
                    break
                first = candles[0][0]
                last = candles[-1][0]
               
                from_timestamp = last +arr_PeriodI[thread2.refArrayInd(arr_PeriodS,period)]*1000*60
                data += candles

                dataFrame = pd.DataFrame(data, columns=['Timestamp','Open','High','Low','Close', 'Volume'])                
                dataFrame['Timestamp'] = pd.DataFrame(dataFrame['Timestamp'].apply(exchange.iso8601))
                dataFrame['Timestamp'] = pd.to_datetime(dataFrame['Timestamp'])
                dataFrame['Timestamp'] = dataFrame['Timestamp'].dt.tz_convert(None)
                dataFrame['Timestamp'] = dataFrame['Timestamp'].dt.tz_localize('UTC').dt.tz_convert(
                'Asia/Shanghai').dt.tz_localize(None)

            except Exception as e:
                print(f"Error fetching ticker1: {e}")  
                time.sleep(sleepTime*5)              
        return dataFrame 
              
    def initK(self,type,label,dataYear):
        dict_symbol={}
        if label=="all":    
            arr_symbols=self.readTxt(work_dir +"//"+ dataFile)
        else:
            arr_symbols=[label]

        thread2 = common_eng.commonThread()
        if type=="train":
            print("检查标的列表:")
            print(arr_symbols)
            print("检查周期(分钟):")
            print(arr_PeriodS)   

        for symbol in arr_symbols:
            if type=="train":
                print("初始化标的:("+symbol+")K线行情数据###############")
            dict_Period={}
            
            try:
                for period in arr_PeriodS:
                    df = self.getKData(symbol,period,type,dataYear)
                    #df = df.iloc[::-1]
                    dict_Period[period]=df
                    
                    if symbol not in dict_symbol:
                        dict_symbol[symbol] = {} 
    
                    #dict_symbol[symbol][period]=df
                    df2=self.preprocess_indicators(df,symbol,period)  
                    dict_symbol[symbol][period] = df2
                    
                    dataPlot=df.copy()
                    dataPlot['signal']=0.0
                    dataPlot['asset']=None
                    dataPlot['signal_long']=None
                    dataPlot['signal_short']=None
                    dataPlot['signal_close']=None
                    dataPlot.set_index('Timestamp',inplace=True)
                    dataPlot.index=pd.to_datetime(dataPlot.index)
                    dict_period[arr_PeriodI[thread2.refArrayInd(arr_PeriodS,period)]]=dataPlot
                    dict_plot[symbol]=dict_period
                    
                    if type=="train":    
                        print("Symbol:"+symbol+",period:"+period+",data size:"+str(len(df)))
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                                        
                return dict_symbol
            except Exception as e:
                print(f"Error fetching ticker3: {e}")
                time.sleep(sleepTime*5)                
            #print(self.iClose(symbol,60,0))
            #print(self.iHigh(symbol,60,0))
            #print(self.iLow(symbol,60,0)) 
            #print(f"MODE_MAIN:{self.iMacd(symbol,60,12,26,9,PRICE_CLOSE,MODE_MAIN,0)}")  
            #print(f"MODE_SIGNAL:{self.iMacd(symbol,60,12,26,9,PRICE_CLOSE,MODE_SIGNAL,0)}")                   
            #print(f"iStochastic MAIN:{self.iStochastic(symbol,60,14,3,'SMA','MAIN',0)}")   
            #print(f"iStochastic SIGNAL:{self.iStochastic(symbol,60,14,3,'SMA','SIGNAL',0)}")             

    def initTick(self,type):
        dict_symbol={}

        arr_symbols=self.readTxt(work_dir +"//"+ dataFile)

        for symbol in arr_symbols:
            try:
                # 并行获取所有周期数据（可选）
                period_data = {}
                for period in arr_PeriodS:
                    df = self.getKData(symbol, period, type)
                    if df is not None:
                        period_data[period] = df
                
                # 如果成功获取数据
                if period_data:
                    # 初始化symbol的数据存储
                    if symbol not in dict_symbol:
                        dict_symbol[symbol] = {}
                    
                    # 批量存储并预处理
                    for period, df in period_data.items():
                        #dict_symbol[symbol][period] = df
                        df2=self.preprocess_indicators(df,symbol, period)
                        dict_symbol[symbol][period] = df2
                        
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                time.sleep(sleepTime * 5)
                continue  # 继续处理下一个标的
        return dict_symbol
        
    def preprocess_indicators(self,df, symbol, period):
        """
        预处理技术指标数据（MACD、Stochastic、Bollinger Bands）
        并将结果缓存到字典中
        """
        try:
            #df = dict_symbol[symbol][period].copy()
            #df = df.iloc[::-1]
            # 计算MACD
            df['macd_diff'], df['macd_dea'], df['macd_hist'] = self.calculate_macd_series(df['Close'])
            #print(df['macd_diff'])

            # 计算Stochastic (KDJ)
            df['stoch_k'], df['stoch_d'] = self.calculate_stochastic_series(
                high=df['High'], 
                low=df['Low'], 
                close=df['Close'],
                k_period=14,
                d_period=3
            )

            # 计算布林带
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
                prices=df['Close'],
                window=19,
                num_std=3
            )

            df['rsi'] = self.calculate_rsi_series(
                close=df['Close'],
                period=14
            )

            # 更新到原始数据                       
            return df
        except Exception as e:
            print(f"Error : {e}")

    def calculate_rsi_series(self, close, period=14):
        """计算RSI序列"""
        try:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"Error in calculate_rsi_series: {e}")
            return pd.Series([0] * len(close), pd.Series([0] * len(close)))
    
    def calculate_macd_series(self, close_prices, fast=15, slow=30, signal=12):
        """计算MACD序列"""
        ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_stochastic_series(self, high, low, close, k_period=14, d_period=3):
        """计算随机指标KDJ序列"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper, rolling_mean, lower