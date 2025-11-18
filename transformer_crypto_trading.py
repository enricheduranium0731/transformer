#!/usr/bin/python
# coding=utf-8
import traceback

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
import mplfinance as mpf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#import talib
#import backtrader as bt
from tqdm import tqdm
from collections import deque

import common_eng      #基础模块
import data_eng        #行情数据模块
import features_eng    #特征数据模块
import deal_eng        #交易模块
        
work_dir=os.path.dirname(os.path.abspath(__file__))
arr_symbols=['BTCUSDT','DOGEUSDT','BNBUSDT','XRPUSDT','SOLUSDT','ADAUSDT','UNIUSDT']
arr_PeriodTick=["15m"]
arr_PeriodS=["15m","30m","1h","4h","1d"]
arr_PeriodI=[15,30,60,240,1440]
dataFile=""
arr_Deal_PeriodI=[60]   
dataFile="products--transformer.txt"
label=sys.argv[1]
dataYear=float(sys.argv[2])   
run_type=sys.argv[3]
    
features_size=0

class traderThread (threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        
    def run(self):
        arr_symbols=self.readTxt(work_dir +"//products--transformer.txt")

        if self.name =="initTick":
            while True:
                thread1 = traderThread()
                common_eng.dict_symbol=self.initTick(run_type)
                thread1.sleep(5) 

        for symbol in arr_symbols:            
            if self.name =="initFeature"+symbol:
                thread1 = features_eng.featuresThread
                thread1.genFeatures(symbol,60,run_type)
                while True:
                    thread1.maintainFeatures(symbol,60,run_type)
                time.sleep(5)  
                
        if self.name =="gatheringdata":
            while True:
                thread1 = traderThread()
                dict_symbol=thread1.initK(run_type,label,dataYear)
                break
                
        for symbol in arr_symbols:            
            if self.name =="deal:"+symbol:
                # 替换为你的API密钥
                API_KEY = 'your_api_key'
                API_SECRET = 'your_api_secret'
                
                # 初始化交易系统
                trader = RealTimeTrading(symbol, API_KEY, API_SECRET)
                #trader = RealTimeTrading(symbol, API_KEY, API_SECRET)
                
                # 运行交易系统
                trader.run() 

# ==================== 2. Transformer模型 ====================
class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
    
    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class CryptoTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 多任务输出头
        self.price_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2))
        
    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后时间步
        
        price_out = self.price_head(x)
        dir_out = self.dir_head(x)
        return price_out.squeeze(), dir_out

# ==================== 3. 训练流程 ====================
class CryptoDataset(Dataset):
    def __init__(self, X, y_price, y_dir):
        self.X = torch.FloatTensor(X)
        self.y_price = torch.FloatTensor(y_price)
        self.y_dir = torch.LongTensor(y_dir)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_price[idx], self.y_dir[idx]

def train_model(X_train, y_price_train, y_dir_train, epochs=50, batch_size=32):
    """使用改进模型的训练函数"""
    dataset = CryptoDataset(X_train, y_price_train, y_dir_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 使用改进的模型
    model = ImprovedCryptoTransformer(input_dim=X_train.shape[-1])
    
    # 如果有GPU则使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    price_criterion = nn.MSELoss()
    dir_criterion = nn.CrossEntropyLoss()
    
    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_price_batch, y_dir_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            X_batch = X_batch.to(device)
            y_price_batch = y_price_batch.to(device)
            y_dir_batch = y_dir_batch.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                price_pred, dir_pred = model(X_batch)
                loss_price = price_criterion(price_pred, y_price_batch)
                loss_dir = dir_criterion(dir_pred, y_dir_batch)
                loss = 0.7 * loss_price + 0.3 * loss_dir
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(dataloader)
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    return model

# ==================== 4. 交易策略 ====================
class TradingEngine:
    def __init__(self, model, scaler, threshold=0.01):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.current_position = None  # 'LONG', 'SHORT', None
        self.current_price=0
    
    def generate_signal(self, latest_data):
        """输入最新K线数据，生成交易信号"""
        # 最新60条数据 (需包含close, rsi, macd等特征)
        features = [
            #"timestamp",            
            "openh4",
            "closeh4",            
            "highh4",
            "lowh4",
            "volumeh4",
            "macdMh4",
            "macdSh4",            
            "kdjMh4",
            "kdjSh4",
            "rsih4",
            "band1h4",            
            "band2h4",
            "band3h4",
            "ma5h4",            
            "ma10h4",
            "ma30h4",
            "ma50h4",
            "ma100h4",           

            "opend1",
            "closed1",            
            "highd1",
            "lowd1",
            "volumed1",
            "macdMd1",
            "macdSd1",            
            "kdjMd1",
            "kdjSd1",
            "rsid1",
            "band1d1",            
            "band2d1",
            "band3d1",
            "ma5d1",            
            "ma10d1",
            "ma30d1",
            "ma50d1",
            "ma100d1",
            
            "getRefHighIndh4",
            "getRefLowIndh4",            
            "getRefHighIndd1",
            "getRefLowIndd1",
            
            "iStochasticDown",
            "iStochasticUp",
            
            "isQianDown",
            "iKdjDownInd",            
            "iRsiDownInd",
            "isRsiDown",
            "iMacdDownInd",
            "isTouchedTopBand", 
            "isBandGoDown",
            
            "isPowerMaDown1",
            "isPowerMaDown2",
            "isTopChan",
            "iContinueDownFromTopByHighId",
            "isContinueDown",
            "isPierceAndSwallowDownP",

            "isQianUp",
            "iKdjUpInd",            
            "iRsiUpInd",
            "isRsiUp",
            "iMacdUpInd",
            "isTouchedBottomBand",
            "isBandGoUp",    
            
            "isPowerMaUp1",
            "isPowerMaUp2",
            "isBottomChan",
            "iContinueUpFromBottomByLowId",
            "isContinueUp",
            "isPierceAndSwallowUpP",
            "future_return",
            "direction"
        ]
        scaled_data = self.scaler.transform(latest_data[features].values.reshape(1, -1))
        seq = torch.FloatTensor(scaled_data).unsqueeze(0)  # [1, seq_len, features]
        
        with torch.no_grad():
            price_change, dir_prob = self.model(seq)
            price_change = price_change.item()
            direction = torch.argmax(dir_prob).item()  # 0或1
        
        if price_change > self.threshold and direction == 1:
            return 'LONG'
        elif price_change < -self.threshold and direction == 0:
            return 'SHORT'
        else:
            return 'HOLD'
    
    def execute_trade(self, signal, account_balance, leverage=5, risk_pct=0.01):
        """执行交易 (模拟版)"""
        if signal == self.current_position:
            return  # 无需重复开仓
        
        # 计算仓位大小 (简化版)
        position_size = (account_balance * risk_pct * leverage) / self.current_price
        
        if signal == 'LONG':
            print(f"开多仓: {position_size:.4f} BTC")
        elif signal == 'SHORT':
            print(f"开空仓: {position_size:.4f} BTC")
        elif signal == 'HOLD' and self.current_position is not None:
            print(f"平仓: {self.current_position}")
        
        self.current_position = signal if signal != 'HOLD' else None

# ==================== 实盘交易主程序 ====================
class RealTimeTrading:
    def __init__(self, symbol, api_key, api_secret):
        self.symbol = symbol
        # self.exchange = ccxt.binance({
            # 'apiKey': api_key,
            # 'secret': api_secret,
            # 'enableRateLimit': True,
            # 'options': {
                # 'defaultType': 'future',  # 使用合约交易
            # }
        # })
        
        self.exchange = ccxt.binance({
            'rateLimit': 1000,
            'enableRateLimit': True,
            # 'verbose': True,
        })    
        
        # 加载模型和scaler
        self.model = CryptoTransformer(input_dim=features_size)
        self.model.load_state_dict(torch.load(f'./transformer-models/transformer_model-{self.symbol.lower()}.pth'))
        self.model.eval()
        
        self.scaler = joblib.load(f'./transformer-models/scaler-{self.symbol.lower()}.pkl')
        
        # 交易状态
        self.position = None  # 'LONG', 'SHORT', None
        self.entry_price = None
        self.stop_loss = 0.03  # 5%止损
        self.take_profit = 0.03  # 止盈
        
        # 参数
        self.leverage = 5
        self.risk_pct = 0.01  # 每笔交易风险1%
        self.stop_loss_pct = 0.03  # 5%止损
        self.take_profit_pct = 0.03  # 止盈
        self.threshold = 0.01  # 信号阈值
        
        # 数据缓存
        self.data_window = []
        self.seq_length = 60  # 与模型训练时一致
        
    def fetch_historical_data(self):
        global dict_features
        #thread = traderThread(3,"initTick")
        #df1 = thread.genFeatures(symbol,self.seq_length,run_type)
        df = dict_features[self.symbol.lower()]
        #df2 = dict_symbol[symbol][arr_PeriodS[thread.refArrayInd(arr_PeriodI,arr_Deal_PeriodI[0])]].tail(self.seq_length)
        #df = pd.concat([df1, df2], axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])        
        return df
        
        """获取历史数据初始化窗口"""  
        now = datetime.now(timezone.utc)
        since = now - timedelta(hours=4 * self.seq_length)
        #temp=since.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        #from_timestamp= self.exchange.parse8601(temp)  
        period='4h'    
        ohlcv = self.exchange.fetch_ohlcv(self.symbol,timeframe=period,limit=self.seq_length*2)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def initialize(self):
        """初始化交易系统"""
        # 设置杠杆
        # try:
            # self.exchange.set_leverage(self.leverage, self.symbol)
        # except Exception as e:
            # print(f"设置杠杆失败: {e}")
        
        # 获取历史数据
        df = self.fetch_historical_data()
        #df = add_technical_indicators(df)

        # 填充数据窗口
        # for _, row in df.iterrows():
            # self.update_data_window(row)
        
        print("交易系统初始化完成")
    
    def update_data_window(self, new_bar):
        """更新数据窗口"""
        if len(self.data_window) >= self.seq_length:
            self.data_window.pop(0)
        self.data_window.append(new_bar)
    
    def prepare_features(self):
        """准备特征数据"""
        #print(f"Scaler features: {len(self.scaler.feature_names_in_)}")
        #print(f"Scaler features: {len(self.data_window[0].keys())}")
        # print(f"Scaler features: {self.scaler.feature_names_in_}")
        # print(f"Data window features: {self.data_window[0].keys()}")
        # print(f"Sequence length: {len(self.data_window)} vs expected: {self.seq_length}")
        
        if len(self.data_window) < self.seq_length:
            return None
            
        df_window = pd.DataFrame(self.data_window)
        
        # 确保包含所有需要的特征
        required_features = self.scaler.feature_names_in_
        for feat in required_features:
            if feat not in df_window.columns:
                # 简单处理 - 用0填充缺失特征 (实际应根据特征含义处理)
                df_window[feat] = 0
        
        # 选择并缩放特征
        features = df_window[required_features]
        scaled_data = self.scaler.transform(features)
        
        return torch.FloatTensor(scaled_data).unsqueeze(0)
    
    def generate_signal(self):
        """生成交易信号"""
        seq = self.prepare_features()
        if seq is None:
            return 'HOLD'
        
        with torch.no_grad():
            price_change, dir_prob = self.model(seq)
            price_change = price_change.item()
            direction = torch.argmax(dir_prob).item()
        
        if price_change > self.threshold and direction == 1:
            return 'LONG'
        elif price_change < -self.threshold and direction == 0:
            return 'SHORT'
        else:
            return 'HOLD'
    
    def execute_trade(self, signal):
        """执行交易"""
        # 获取当前价格
        ticker = self.exchange.fetch_ticker(f'{self.symbol}')
        current_price = ticker['last']
        
        # 如果已经有仓位，先检查止损/止盈
        self.check_exit_conditions(current_price)
        
        # if self.position:
            # self.check_exit_conditions(current_price)
            # return
        
        # 计算仓位大小
        balance =15.0 #balance = self.get_account_balance()
        position_size =15.0 #position_size = (balance * self.risk_pct * self.leverage) / current_price
        
        if signal == 'LONG':
            self.open_long(position_size, current_price)
        elif signal == 'SHORT':
            self.open_short(position_size, current_price)
    
    def open_long(self, size, price):
        """开多仓"""
        try:
            deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, price, price, price + price * 0, price - price * 0.1)
            deal_eng.gen_deal(self.symbol, 'BUY', 'transformer', 240, price, 0, price + price * 0.2, price - price * 0.1)
            return
        except Exception as e:
            print(f"开多仓失败: {e}")  
            
        try:
            order = self.exchange.create_market_buy_order(
                f'{self.symbol}', 
                amount=size
            )
            self.position = 'LONG'
            self.entry_price = price
            self.stop_loss = price * (1 - self.stop_loss_pct)
            self.take_profit = price * (1 + self.take_profit_pct)
            print(f"开多仓: {size:.4f} {self.symbol} @ {price:.2f}")
        except Exception as e:
            print(f"开多仓失败: {e}")
    
    def open_short(self, size, price):
        """开空仓"""
        try:
            deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, price, price, price - price * 0.2, price + price * 0.1)
            deal_eng.gen_deal(self.symbol, 'SELL', 'transformer', 240, price, 0, price - price * 0.2, price + price * 0.1)
            return
        except Exception as e:
            print(f"开多仓失败: {e}")     
    
        """开空仓"""
        try:
            order = self.exchange.create_market_sell_order(
                f'{self.symbol}', 
                amount=size
            )
            self.position = 'SHORT'
            self.entry_price = price
            self.stop_loss = price * (1 + self.stop_loss_pct)
            self.take_profit = price * (1 - self.take_profit_pct)
        except Exception as e:
            print(f"开空仓失败: {e}")
    
    def close_position(self):
        """平仓"""
        ticker = self.exchange.fetch_ticker(f'{self.symbol}')
        price = ticker['last']
        current_price = ticker['last']
        try:
            if self.position == 'LONG':
                deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, price, price, price - price * 0.2, price + price * 0.1)
                
                # order = self.exchange.create_market_sell_order(
                    # f'{self.symbol}', 
                    # amount=self.get_position_size()
                # )
                # pnl = (current_price - self.entry_price) / self.entry_price * 100
                #print(f"平多仓 @ {current_price:.2f}, 盈利: {pnl:.2f}%")
            elif self.position == 'SHORT':
                deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, price, price, price + price * 0, price - price * 0.1)
            
                # order = self.exchange.create_market_buy_order(
                    # f'{self.symbol}', 
                    # amount=self.get_position_size()
                # )
                # pnl = (self.entry_price - current_price) / self.entry_price * 100
                #print(f"平空仓 @ {current_price:.2f}, 盈利: {pnl:.2f}%")
            
            self.position = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
        except Exception as e:
            print(f"平仓失败: {e}")
    
    def check_exit_conditions(self, current_price):
        """检查止损/止盈条件"""
        ticker = self.exchange.fetch_ticker(f'{self.symbol}')
        price = ticker['last']
        current_price = ticker['last']
        if self.position == 'LONG':
            if current_price <= self.stop_loss:
                print(f"触发止损 @ {current_price:.2f}")
                #self.close_position()
                deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, price, price, price - price * 0.2, price + price * 0.1)
            elif current_price >= self.take_profit:
                print(f"触发止盈 @ {current_price:.2f}")
                #self.close_position()
                deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, price, price, price - price * 0.2, price + price * 0.1)
        elif self.position == 'SHORT':
            if current_price >= self.stop_loss:
                print(f"触发止损 @ {current_price:.2f}")
                #self.close_position()
                deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, price, price, price + price * 0, price - price * 0.1)
            elif current_price <= self.take_profit:
                print(f"触发止盈 @ {current_price:.2f}")
                #self.close_position()
                deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, price, price, price + price * 0, price - price * 0.1)
    
    def get_account_balance(self):
        """获取账户USDT余额"""
        balance = self.exchange.fetch_balance()
        return balance['USDT']['free']
    
    def get_position_size(self):
        """获取当前持仓大小"""
        positions = self.exchange.fetch_positions([self.symbol])
        for pos in positions:
            if pos['symbol'] == f'{self.symbol}':
                return abs(pos['contracts'])
        return 0
    
    def run(self):
        """运行实盘交易"""
        self.initialize()
        temp='BTC'
        print("开始实盘交易...")
        while True:
            try:
                ohlcv = self.fetch_historical_data()
                
                #确保获取到数据
                if len(ohlcv) == 0:
                    print("未获取到K线数据，等待重试...")
                    time.sleep(30)
                    continue
                    
                # latest_bar = {
                    # 'timestamp': pd.to_datetime(ohlcv[0][0]), 
                    # 'open': ohlcv[0][1],
                    # 'high': ohlcv[0][2],
                    # 'low': ohlcv[0][3],
                    # 'close': ohlcv[0][4],
                    # 'volume': ohlcv[0][5]
                # }
                
                #更新数据窗口
                # if len(self.data_window) > 0 and latest_bar['timestamp'] <= self.data_window[-1]['timestamp']:
                    #同一根K线，更新最新值
                    # self.data_window[-1] = latest_bar
                # else:
                    #新K线
                    # self.update_data_window(latest_bar)
                
                # 确保数据窗口已满
                df_window=pd.DataFrame(ohlcv)
                if len(df_window) < self.seq_length:
                    print(f"数据窗口未满({len(self.data_window)}/{self.seq_length})，等待更多数据...")
                    print(df_window)
                    time.sleep(30)
                    continue
                    
                #df_window = pd.DataFrame(self.df_window)
                
                # 确保DataFrame包含必要的列
                required_cols = ['openh4', 'highh4', 'lowh4', 'closeh4', 'volumeh4']
                for col in required_cols:
                    if col not in df_window.columns:
                        print(f"缺失必要列: {col}")
                        time.sleep(30)
                        continue
                        
                # 添加技术指标
                try:
                    #df_window = add_technical_indicators(ohlcv)  # 使用副本避免修改原数据
                    
                    # 确保转换后的DataFrame有效
                    if not isinstance(df_window, pd.DataFrame):
                        raise ValueError("技术指标函数未返回DataFrame")
                        
                    # 更新数据窗口（只保留数值数据）
                    self.data_window = df_window.to_dict('records')
                    
                    # 生成交易信号
                    signal = self.generate_signal()
                    
                    # 执行交易
                    if signal != 'HOLD':
                        self.execute_trade(signal)
                    
                except Exception as e:
                    print(f"处理技术指标出错: {e}")
                    traceback.print_exc()  # 打印完整错误堆栈
                    
                # 休眠一段时间避免频繁请求
                time.sleep(10)
                          
            
            except ccxt.NetworkError as e:
                print(f"网络错误: {e}, 等待重试...")
                time.sleep(60)
            except ccxt.ExchangeError as e:
                print(f"交易所错误: {e}, 等待重试...")
                time.sleep(60)
            except Exception as e:
                print(f"交易出错: {e}")
                traceback.print_exc()
                time.sleep(60)  # 出错后等待更长时间
    # ==================== 主程序入口 ====================

class ImprovedCryptoTransformer(nn.Module):
    def __init__(self, input_dim=68, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        # 改进的嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 改进的位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 改进的Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 改进的输出头
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 1)
        )
        
        self.dir_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 只取最后时间步
        
        price_out = self.price_head(x)
        dir_out = self.dir_head(x)
        return price_out.squeeze(), dir_out

class OptimizedTradingEngine:
    def __init__(self, model, scaler, symbol, threshold=0.01, seq_length=60):
        self.model = model
        self.scaler = scaler
        self.symbol = symbol
        self.threshold = threshold
        self.seq_length = seq_length
        
        # 交易状态
        self.position = None  # 'LONG', 'SHORT', None
        self.entry_price = None
        self.stop_loss = 0.03       #止损
        self.take_profit = 0.03     # 止盈
        
        # 参数
        self.leverage = 5
        self.risk_pct = 0.01  # 每笔交易风险1%
        self.stop_loss_pct = 0.03       #止损
        self.take_profit_pct = 0.03     # 止盈
        
        # 优化后的数据缓存
        self.data_buffer = deque(maxlen=seq_length)
        self.last_trade_time = None
        
        # 初始化交易所连接
        self.exchange = ccxt.binance({
            'rateLimit': 1000,
            'enableRateLimit': True,
        })
    
    def update_data_buffer(self, new_data):
        """高效更新数据缓冲区"""
        self.data_buffer.append(new_data)
    
    def prepare_features(self):
        """准备输入特征"""
        if len(self.data_buffer) < self.seq_length:
            return None
            
        # 转换为DataFrame
        df_window = pd.DataFrame(list(self.data_buffer))
        
        # 确保包含所有需要的特征
        required_features = self.scaler.feature_names_in_
        for feat in required_features:
            if feat not in df_window.columns:
                df_window[feat] = 0  # 用0填充缺失特征
        
        # 使用预分配的numpy数组提高效率
        features = np.zeros((1, self.seq_length, len(required_features)))
        features[0] = self.scaler.transform(df_window[required_features])
        
        return torch.FloatTensor(features)
    
    def generate_signal(self):
        """生成交易信号"""
        seq = self.prepare_features()
        if seq is None:
            return 'HOLD'
        
        with torch.no_grad():
            price_change, dir_prob = self.model(seq)
            price_change = price_change.item()
            direction = torch.argmax(dir_prob).item()
        
        if price_change > self.threshold and direction == 1:
            return 'LONG'
        elif price_change < -self.threshold and direction == 0:
            return 'SHORT'
        else:
            return 'HOLD'
                
    def execute_trade(self, signal):
        """执行交易"""
        # 获取当前价格
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker['last']        
        # 检查是否应该平仓
        self.check_exit_conditions(current_price)
        
        # if self.position is not None:
            # return
        
        # 计算仓位大小
        balance = 15.0  # 简化示例，实际应从交易所获取
        position_size = (balance * self.risk_pct * self.leverage) / current_price
        thread450 = common_eng.commonThread()

        if (signal == "LONG") and (not isinstance(self.position, str) or self.position != signal):
            if (thread450.isCanDeal(self.symbol,60,"real","LONG","transformer",0)):
                #if not(deal_eng.isNNDealOk("SELL")):
                self.open_long(position_size, current_price)
                
        elif (signal == "SHORT") and (not isinstance(self.position, str) or self.position != signal):
            thread450 = traderThread(450, "trend_detect")
            if (thread450.isCanDeal(self.symbol,60,"real","SHORT","transformer",0)):  
                #if not(deal_eng.isNNDealOk("BUY")):
                self.open_short(position_size, current_price)
    
    def open_long(self, size, price):
        """开多仓"""
        try:
            deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, price, 
                                 price, price + price * 0, price - price * 0.1)
            deal_eng.gen_deal(self.symbol, 'BUY', 'transformer', 240, price, 0, 
                              price + price * 0.2, price - price * 0.1)            
            self.position = 'LONG'
            self.entry_price = price
            self.stop_loss = price * (1 - self.stop_loss_pct)
            self.take_profit = price * (1 + self.take_profit_pct)
            print(f"[{datetime.now()}] 开多仓 {self.symbol} @ {price:.2f}")            
        except Exception as e:
            print(f"开多仓失败: {e}")
    
    def open_short(self, size, price):
        """开空仓"""
        try:
            deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, price, 
                                 price, price - price * 0.2, price + price * 0.1)
            deal_eng.gen_deal(self.symbol, 'SELL', 'transformer', 240, price, 0, 
                              price - price * 0.2, price + price * 0.1)
            
            self.position = 'SHORT'
            self.entry_price = price
            self.stop_loss = price * (1 + self.stop_loss_pct)
            self.take_profit = price * (1 - self.take_profit_pct)
            print(f"[{datetime.now()}] 开空仓 {self.symbol} @ {price:.2f}")
            
        except Exception as e:
            print(f"开空仓失败: {e}")
    
    def check_exit_conditions(self, current_price):
        """检查止损/止盈条件"""
        data=deal_eng.get_close_deal(self.symbol,"BUY","transformer",240)
        thread650 = traderThread(650, "trend_detect3") 
        xCount=6
        ind=0
        
        if (len(data)>0):
            d_open=data.iloc[0,5]
            time=data.iloc[0, 9]
            xCount=thread650.getRecentInd(symbol,15,time,ind)                                       
        
        if self.position == 'LONG':
            if current_price <= self.stop_loss:
                print(f"[{datetime.now()}] 触发止损 @ {current_price:.2f}")
                deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, 
                                     current_price, current_price, 
                                     current_price - current_price * 0.2, 
                                     current_price + current_price * 0.1)
                self.position = None
            elif (((thread650.iHigh(symbol,15,thread650.getHighInd(symbol,xCount,ind,15))-d_open)/d_open)>=0.03): #self.take_profit
                if (thread650.iHigh(symbol,15,thread650.getHighInd(symbol,xCount,ind,15))-thread650.iClose(symbol,15,ind))/(thread650.iHigh(symbol,15,thread650.getHighInd(symbol,xCount,ind,15))-d_open)>=0.3:             
                    print(f"[{datetime.now()}] 触发止盈 @ {current_price:.2f}")
                    deal_eng.close_order(self.symbol, 'BUY', 'transformer', 240, 
                                         current_price, current_price, 
                                         current_price - current_price * 0.2, 
                                         current_price + current_price * 0.1)
                    self.position = None
                
        elif self.position == 'SHORT':
            if current_price >= self.stop_loss:
                print(f"[{datetime.now()}] 触发止损 @ {current_price:.2f}")
                deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, 
                                     current_price, current_price, 
                                     current_price + current_price * 0, 
                                     current_price - current_price * 0.1)
                self.position = None
            elif (((d_open-thread650.iLow(symbol,15,thread650.getLowInd(symbol,xCount,ind,15)))/d_open)>=0.03): #self.take_profit
                if (thread650.iClose(symbol,15,ind)-thread650.iLow(symbol,15,thread650.getLowInd(symbol,xCount,ind,15)))/(d_open-thread650.iLow(symbol,15,thread650.getLowInd(symbol,xCount,ind,15)))>=0.3:
                    print(f"[{datetime.now()}] 触发止盈 @ {current_price:.2f}")
                    deal_eng.close_order(self.symbol, 'SELL', 'transformer', 240, 
                                         current_price, current_price, 
                                         current_price + current_price * 0, 
                                         current_price - current_price * 0.1)
                    self.position = None
    
    def run(self):
        """运行优化后的交易引擎"""
        global dict_features
        print(f"开始 {self.symbol} 交易...")        
        while True:
            try:
                # 获取最新K线数据
                df = dict_features[self.symbol.lower()]
                #latest_data = df.iloc[-1].to_dict()
                #self.update_data_buffer(latest_data)   
                
                new_records = df.to_dict('records')
                self.data_buffer = deque(maxlen=self.seq_length)
                for record in new_records:
                    #if len(self.data_buffer) == 0 or record['timestamp'] > self.data_buffer[-1]['timestamp']:
                    self.data_buffer.append(record)                                                    
                
                # 确保数据窗口已满
                if len(self.data_buffer) < self.seq_length:
                    #print(f"等待数据填充 ({len(self.data_buffer)}/{self.seq_length})...")
                    time.sleep(10)
                    continue
                
                # 生成交易信号
                signal = self.generate_signal()
                
                # 执行交易
                if signal != 'HOLD':
                    self.execute_trade(signal)
                
                # 控制交易频率
                time.sleep(10)
                
            except Exception as e:
                print(f"交易出错: {e}")
                time.sleep(60)
    
def train(symbol):
    if run_type=='train':
        thread1 = data_eng.dataThread()
        common_eng.dict_symbol=thread1.initK(run_type,symbol,dataYear)
        
        thread2 = features_eng.featuresThread()
        df = thread2.genFeatures(symbol, int(dataYear*365*6*4),run_type)
        X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test, scaler,features = thread2.prepare_sequences(df) 
        #print(X_test)       
        # 2. 训练模型
        model = train_model(X_train, y_price_train, y_dir_train, epochs=150)
        torch.save(model.state_dict(), './transformer-models/transformer_model-'+symbol.lower()+'.pth') 
        joblib.dump(scaler, './transformer-models/scaler-' + symbol.lower() + '.pkl')

def deal(arr_symbols):
    if run_type=='real':
        if label != 'all':
            arr_symbols = [label]
            
        for symbol in arr_symbols: 
            # 加载模型和scaler
            model = ImprovedCryptoTransformer(input_dim=features_size)
            model.load_state_dict(torch.load(f'./transformer-models/transformer_model-{symbol.lower()}.pth'))
            model.eval()
            
            scaler = joblib.load(f'./transformer-models/scaler-{symbol.lower()}.pkl')
            
            # 创建优化交易引擎
            trading_engine = OptimizedTradingEngine(
                model=model,
                scaler=scaler,
                symbol=symbol
            )
            
            # 启动交易线程
            def trading_thread():
                trading_engine.run()
                
            thread = threading.Thread(target=trading_thread, name=f"trading_{symbol}")
            thread.start()
            print(f"启动 {symbol} 交易线程...")            
        
def test(symbol):
    thread2 = features_eng.featuresThread()
    df = thread2.genFeatures(symbol, 60,run_type)
    X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test, scaler,features = thread2.prepare_sequences(df)
    
    # 加载 scaler
    scaler = joblib.load('./transformer-models/scaler-' + symbol.lower() + '.pkl')
    
    # 加载模型
    model = CryptoTransformer(input_dim=len(features))
    model.load_state_dict(torch.load('./transformer-models/transformer_model-' + symbol.lower() + '.pth'))
    model.eval()

    # 初始化交易引擎
    trading_engine = TradingEngine(model, scaler)

    print("\n模拟交易开始...")
    for i in range(len(X_test)):
        # 获取正确的窗口数据 (直接使用X_test中已经预处理好的序列)
        seq_data = X_test[i]  # 这已经是正确形状的序列数据 [seq_len, n_features]
        
        # 转换为tensor
        seq_tensor = torch.FloatTensor(seq_data).unsqueeze(0)  # 添加batch维度
        trading_engine.current_price = df.iloc[-len(X_test) + i]['closeh4']
        
        # 生成信号
        with torch.no_grad():
            price_change, dir_prob = model(seq_tensor)
            price_change = price_change.item()
            direction = torch.argmax(dir_prob).item()
            
        if price_change > trading_engine.threshold and direction == 1:
            signal = 'LONG'
        elif price_change < -trading_engine.threshold and direction == 0:
            signal = 'SHORT'
        else:
            signal = 'HOLD'
            
        # 执行交易
        trading_engine.execute_trade(signal, account_balance=10000)
        
# ==================== 主程序 ====================
# ================================================
# ================================================   
if __name__ == "__main__":        
    thread1 = data_eng.dataThread()
    arr_symbols=thread1.readTxt(work_dir +"//products--transformer.txt")
    
    if run_type!='train':
        thread1 = traderThread(1,"initTick")
        thread1.start()
        time.sleep(60*5)

        ind=2
        if label == 'all':                
            for symbol in arr_symbols:
                thread2 = traderThread(ind,"initFeature"+symbol)
                thread2.start()
                ind+=1    
        else:          
            thread2 = traderThread(ind,"initFeature"+label)
            thread2.start()
            ind+=1
        
        time.sleep(60*60) 
            
    # 设置特征维度
    features_size = 68    
    
    if run_type == 'real':
        deal(arr_symbols)

    elif run_type == 'train':
        if label == 'all':
            for symbol in arr_symbols:
                train(symbol)
        else:
            train(label)                    