#!/usr/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
import time
import math
from datetime import datetime, timedelta, timezone
import os
import sys
import threading
import matplotlib.pyplot as plt
import joblib
import ccxt
from tqdm import tqdm
from collections import deque
import common_eng
from common_eng import sleepTime, dict_period, dict_plot

arr_PeriodTick = ["15m"]
arr_PeriodS = ["15m", "30m", "1h", "4h"]
arr_PeriodI = [15, 30, 60, 240]

work_dir = os.path.dirname(os.path.abspath(__file__))
dataFile = "products--transformer.txt"


class dataThread():
    def readTxt(self, file_name):
        data = []
        file = open(file_name, 'r')
        file_data = file.readlines()
        if len(file_data) > 0:
            line = file_data[0]
            line = line.strip()
            tmp_list = line.split(',')
            for obj in tmp_list:
                data.append(obj)
        return data

    def getKData(self, symbol, period, type, dataYear):
        global arr_PeriodI, arr_PeriodS
        thread2 = common_eng.commonThread()

        # 获取周期索引
        period_idx = thread2.refArrayInd(arr_PeriodS, period)
        period_minutes = arr_PeriodI[period_idx]

        # 计算需要获取的K线数量
        if type == "real":
            klines_amount = 1000
        else:
            klines_amount = int(dataYear * 365 * 24 * 60 / period_minutes) + 200

        exchange = ccxt.binance({
            'rateLimit': 1000,
            'enableRateLimit': True,
        })

        # 计算起始时间
        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=klines_amount * period_minutes)

        # 将时间转换为时间戳（毫秒）
        from_timestamp = int(since.timestamp() * 1000)

        # 获取当前时间戳（毫秒）
        current_timestamp = int(now.timestamp() * 1000)

        all_candles = []

        while from_timestamp < current_timestamp:
            try:
                # 获取K线数据
                candles = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=period,
                    since=from_timestamp,
                    limit=1000
                )

                if len(candles) == 0:
                    break

                # 添加数据到列表
                all_candles.extend(candles)

                # 更新起始时间为最后一条数据的时间戳 + 周期间隔
                last_timestamp = candles[-1][0]
                from_timestamp = last_timestamp + (period_minutes * 60 * 1000)

                # 防止无限循环
                if from_timestamp <= last_timestamp:
                    from_timestamp = last_timestamp + (period_minutes * 60 * 1000)

                # 休息一下避免API限制
                time.sleep(exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                print(f"Rate limit exceeded for {symbol}, waiting...")
                time.sleep(sleepTime * 5)
            except ccxt.RequestTimeout:
                print(f"Request timeout for {symbol}, retrying...")
                time.sleep(sleepTime)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                time.sleep(sleepTime)
                break

        # 如果没有数据，返回空DataFrame
        if not all_candles:
            print(f"No data fetched for {symbol} {period}")
            return pd.DataFrame()

        # 创建DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        )

        # 处理时间戳
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

        # 去重并排序
        df = df.drop_duplicates(subset=['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)

        return df

    def initK(self, type, label, dataYear):
        dict_symbol = {}

        # 读取标的列表
        if label == "all":
            arr_symbols = self.readTxt(os.path.join(work_dir, dataFile))
        else:
            arr_symbols = [label]

        if type == "test":
            print("检查标的列表:")
            print(arr_symbols)
            print("检查周期(分钟):")
            print(arr_PeriodS)

        for symbol in arr_symbols:
            if type == "test":
                print(f"初始化标的:({symbol})K线行情数据###############")

            dict_Period = {}

            try:
                for period in arr_PeriodS:
                    print(f"获取 {symbol} {period} 数据...")

                    # 获取K线数据
                    df = self.getKData(symbol, period, type, dataYear)

                    if df.empty:
                        print(f"警告: {symbol} {period} 没有获取到数据")
                        continue

                    # 预处理技术指标
                    df2 = self.preprocess_indicators(df, symbol, period)

                    # 存储处理后的数据
                    if symbol not in dict_symbol:
                        dict_symbol[symbol] = {}
                    dict_symbol[symbol][period] = df2

                    # 准备绘图数据
                    dataPlot = df.copy()
                    dataPlot['signal'] = 0.0
                    dataPlot['asset'] = None
                    dataPlot['signal_long'] = None
                    dataPlot['signal_short'] = None
                    dataPlot['signal_close'] = None

                    # 设置索引
                    dataPlot.set_index('Timestamp', inplace=True)
                    dataPlot.index = pd.to_datetime(dataPlot.index)

                    # 存储到全局字典
                    thread2 = common_eng.commonThread()
                    period_minutes = arr_PeriodI[thread2.refArrayInd(arr_PeriodS, period)]
                    dict_period[period_minutes] = dataPlot
                    dict_plot[symbol] = dict_period.copy()

                    if type == "train":
                        print(f"Symbol: {symbol}, period: {period}, data size: {len(df)}")
                        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

                print(f"{symbol} 数据初始化完成")

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                time.sleep(sleepTime * 5)

        return dict_symbol

    def preprocess_indicators(self, df, symbol, period):
        """预处理技术指标数据"""
        try:
            df = df.copy()

            # 确保数据按时间排序
            df = df.sort_values('Timestamp').reset_index(drop=True)

            # 计算MACD
            df['macd_diff'], df['macd_dea'], df['macd_hist'] = self.calculate_macd_series(
                df['Close']
            )

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

            # 计算RSI
            df['rsi'] = self.calculate_rsi_series(
                close=df['Close'],
                period=14
            )

            return df

        except Exception as e:
            print(f"Error in preprocess_indicators for {symbol} {period}: {e}")
            return df

    def calculate_rsi_series(self, close, period=14):
        """计算RSI序列"""
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # 避免除零错误
            loss = loss.replace(0, 0.00001)

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # 填充NaN值为50（中性）

        except Exception as e:
            print(f"Error in calculate_rsi_series: {e}")
            return pd.Series([50] * len(close), index=close.index)

    def calculate_macd_series(self, close_prices, fast=15, slow=30, signal=12):
        """计算MACD序列"""
        try:
            ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
            ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        except Exception as e:
            print(f"Error in calculate_macd_series: {e}")
            return (
                pd.Series([0] * len(close_prices), index=close_prices.index),
                pd.Series([0] * len(close_prices), index=close_prices.index),
                pd.Series([0] * len(close_prices), index=close_prices.index)
            )

    def calculate_stochastic_series(self, high, low, close, k_period=14, d_period=3):
        """计算随机指标KDJ序列"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()

            # 避免除零错误
            denominator = highest_high - lowest_low
            denominator = denominator.replace(0, 0.00001)

            k = 100 * ((close - lowest_low) / denominator)
            d = k.rolling(window=d_period).mean()

            # 处理边界值
            k = k.clip(0, 100)
            d = d.clip(0, 100)

            return k.fillna(50), d.fillna(50)
        except Exception as e:
            print(f"Error in calculate_stochastic_series: {e}")
            return (
                pd.Series([50] * len(close), index=close.index),
                pd.Series([50] * len(close), index=close.index)
            )

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()

            # 填充标准差为NaN的值
            rolling_std = rolling_std.fillna(prices.std())

            upper = rolling_mean + (rolling_std * num_std)
            lower = rolling_mean - (rolling_std * num_std)

            return upper, rolling_mean, lower
        except Exception as e:
            print(f"Error in calculate_bollinger_bands: {e}")
            return prices, prices, prices
