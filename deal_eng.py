#!/usr/bin/python
# coding=utf-8
import hashlib
import hmac
import base64
import time
import requests
import json
import random
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import pymysql
import ccxt
from binance.client import Client
from binance.exceptions import BinanceAPIException

import gate_api
from gate_api import Configuration, ApiClient, SpotApi, WalletApi, FuturesApi
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import time

User='root'
Password='90148810Cgh$$'
HostName='47.236.248.110'
Database='currency-db'
Port=3306
Database2='anything'
accountType=1
testnet=False

def getPrices(symbol, period, count):
    conn = None
    try: 
        period = int(period)  # Ensure period is numeric
        symbols = symbol.split('USDT')
        symbol = symbols[0] + '-USDT'
        sql = f"""SELECT DATE_FORMAT(t1.time_slot, '%Y-%m-%d %H:%i:%s') as 'Timestamp', 
                 t1.Open as Open, t1.High as High, t1.Low as Low, 
                 t1.Close as Close, t1.vol as Volume 
                 FROM kline t1 
                 WHERE t1.symbol='{symbol}' AND t1.granularity={period}  
                 ORDER BY t1.time_slot DESC LIMIT {count}"""
        
        conn = pymysql.connect(host=HostName, port=Port, database=Database2, 
                              user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute(sql)        
        columns = [desc[0] for desc in cur1.description]
        df = pd.DataFrame(cur1.fetchall(), columns=columns)
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        return df.iloc[::-1]
    except Exception as e1:
        print(f"An error occurred: {e1}")        
        try:
            if conn:
                conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
            
        return False 
    finally:
        if conn:
            conn.close()

def clear_testdata(db_host,strategy,period):
    conn = None     
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        sql = f" delete from deal_command where host_name='{db_host}' and strategy='{strategy}' and period={period}"
        cur1.execute(sql)
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        cur1.close()
        conn.close()
        
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
    
def insert_testdata(db_host, symbol, trade_way, strategy, period,price, close, profit, loss, my_date):
    conn = None
    
    try: 
        if isinstance(my_date, str):
            date = datetime.strptime(my_date, "%Y-%m-%dT%H:%M:%S.%fZ")
            formatted_date = date.strftime("%Y-%m-%d %H:%M:%S")
        elif hasattr(my_date, 'strftime'):  # It's already a datetime object
            formatted_date = my_date.strftime("%Y-%m-%d %H:%M:%S")
        else:  # It might be a pandas Timestamp or other datetime-like object
            formatted_date = pd.to_datetime(my_date).strftime("%Y-%m-%d %H:%M:%S")
    
        conn = pymysql.connect(host=HostName, port=Port, database=Database, 
                              user=User, password=Password, charset='utf8')
        
        cur2 = conn.cursor()
        deal_time = datetime.now()
        deal_date = deal_time.strftime('%Y-%m-%d %H:%M:%S')
        
        sql1 = """INSERT INTO deal_command 
                 (product_code, trade_way, price, close, profit, loss, date, 
                  host_name, comment, strategy, period, order_date) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        
        params = (symbol, trade_way, price, close, profit, loss, formatted_date, 
                 db_host, strategy, strategy, period, formatted_date)
        
        count = cur2.execute(sql1, params)
        conn.commit()
        cur2.close()
        
    except Exception as e1:
        print(f"An error occurred: {e1}")
        if conn is not None:
            try:
                conn.ping(True)
            except Exception as e2:
                print(f"An error occurred: {e2}")
                
        else:
            print("Connection was not established")
    
    finally:
        # Always close connection if it exists
        if conn is not None:
            conn.close()
            
def getTickerPricePrecision(symbol):
    client = Client()
    resp = client.futures_exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['pricePrecision']

def getTickerQtyPrecision(symbol):
    client = Client()
    resp = client.futures_exchange_info()['symbols']
    for elem in resp:
        if elem['symbol'] == symbol:
            return elem['quantityPrecision']
                
def getBid(symbol,apikey,secretkey):
    exchange = ccxt.binance({
        'apiKey': apikey,
        'secret': secretkey
    })
    if exchange.has['fetchTicker']:
        try:
            # 使用fetchTicker方法获取tick数据
            ticker = exchange.fetch_ticker(symbol)
            
            # 打印tick信息，例如：symbol, last, high, low等
            return (ticker["bid"])
        except Exception as e:
            print(f"Error fetching ticker: {e}")        
    else:
        print("Error: Fetching ticker is not supported by the Binance exchange in ccxt version used.")

def getAsk(symbol,apikey,secretkey):
    exchange = ccxt.binance({
        'apiKey': apikey,
        'secret': secretkey
    })
    if exchange.has['fetchTicker']:
        try:
            # 使用fetchTicker方法获取tick数据
            ticker = exchange.fetch_ticker(symbol)
            
            # 打印tick信息，例如：symbol, last, high, low等
            return (ticker["ask"])
        except Exception as e:
            print(f"Error fetching ticker: {e}")        
    else:
        print("Error: Fetching ticker is not supported by the Binance exchange in ccxt version used.")

def getBalance(symbol,apikey,secretkey):
    binance = ccxt.binance(
        {
            'apiKey': apikey,
            'secret': secretkey,            
        }
    )    
    try:
        public_data = \
        [i for i in binance.fapiPublicGetExchangeInfo()['symbols'] if i['symbol'] == symbol][0]['filters']
        public_information = public_data[2]['stepSize']
        minQty = public_data[1]['minQty']
        account = [i for i in binance.fapiPrivateV2GetBalance({"timestamp": str(int(time.time()) - 1) + '000'}) if i['asset'] == 'USDT'][0]['balance']
        size_float = float(account)        
    except:        
        return 0              
    #print(size_float)
    return size_float

def isClosed():
    conn = None
    isOk=False
    try: 
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute(' select 1 from  general_server_status where server_status=\'closed\'')
        df = pd.DataFrame(cur1.fetchall())
        conn.commit()
        cur1 = conn.cursor()
        if len(df)>0:
            isOk=True              
        cur1.close()
        conn.close() 
        
        return isOk
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        return False 

def isSNNDealOk(symbol,trade_way):
    conn = None
    isOk=False
    try: 
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        
        cur1 = conn.cursor()
        cur1.execute(f" SELECT t1.trade_way,count(t1.trade_way) from vw_nn_signal t1 where t1.product_code='{symbol}' GROUP BY t1.trade_way ")
        df = pd.DataFrame(cur1.fetchall())
        conn.commit()
        if len(df)==1:
            if (df.iloc[0,0]==trade_way and df.iloc[0,1]==1):
                isOk=True              
        cur1.close()
        conn.close() 
        
        return isOk
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        return False  
        
def isNNDealOk(trade_way):
    conn = None
    isOk=False
    try: 
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        
        cur1 = conn.cursor()
        cur1.execute(' SELECT t1.trade_way,count(t1.product_code) from vw_nn_signal t1 GROUP BY t1.trade_way ')
        df = pd.DataFrame(cur1.fetchall())
        conn.commit()
        if len(df)==1:
            if (df.iloc[0,0]==trade_way and df.iloc[0,1]>=3):
                isOk=True              
        cur1.close()
        conn.close() 
        
        return isOk
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        return False    
            
def gen_deal(symbol,trade_way,strategy,period,price,close,profit,loss):
    conn = None
    try: 
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        
        cur1 = conn.cursor()
        cur1.execute(' SELECT t1.trade_way,COUNT(t1.product_code) from deal_command_nn t1 where (DATE_ADD(t1.date,INTERVAL -4 hour)<=now()) and (DATE_ADD(t1.date,INTERVAL 4 hour)>=now()) GROUP BY t1.trade_way ')
        df = pd.DataFrame(cur1.fetchall())
        conn.commit()
        if len(df)==1:
            if not (df.iloc[0,0]==trade_way and df.iloc[0,1]>1 ):
                return                
        cur1.close()
        
        cur1 = conn.cursor()
        cur1.execute(' select * from deal_command t1  where t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\' and t1.strategy=\''+strategy+'\' and t1.period='+str(period)+' and ISNULL(t1.close_date)')       
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df = pd.DataFrame(cur1.fetchall())
        conn.commit()
        if len(df)<=0:    
            cur2 = conn.cursor()
            deal_time = datetime.now()
            deal_date = deal_time.strftime('%Y-%m-%d %H:%M:%S')
            # print("----------------",len(deal_command))
            # 执行insert语句，并返回受影响的行数：添加一条学生数据
            # 增加
                
            sql1="insert into deal_command(product_code,trade_way,price,close,profit,loss,date,host_name,comment,strategy,period) values('" + symbol + "',"+"'" + trade_way + "'," + str(price)+"," + str(close) +"," + str(profit) + "," + str(loss) +",'" + deal_date + "', '" + "Power" + "', '" + strategy + "', '" + strategy + "'," +str(period) +")";
            #print(sql1)
            count = cur2.execute(sql1)
            # 关闭Cursor对象            
            conn.commit()
            cur2.close()
            #gen_detail(symbol,trade_way,strategy,period,price,close,profit,loss)
            
        cur1.close()
        conn.close()    
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")

def gen_detail(symbol,trade_way,strategy,period,price,close,profit,loss):
    conn = None
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute('select t1.id from deal_command t1  where t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\' and t1.strategy=\''+strategy+'\' and t1.period='+str(period)+' and ISNULL(t1.close_date)')
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        
        for t in range(0,len(df1)):            
            curX = conn.cursor()
            curX.execute('select t1.accessKey,t1.secretKey,t1.leverage,t1.host from api_key t1  where t1.accountType='+str(accountType))        
            ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
            dfX = pd.DataFrame(curX.fetchall())   
            conn.commit()
            for x in range(0,len(dfX)): 
                apikey=dfX.iloc[x,0]   
                                    
                cur2 = conn.cursor()
                cur2.execute('select t1.id from deal_detail t1  where t1.closed=0 and t1.product_code=\''+symbol+'\' and t1.api_key=\''+apikey+'\' and t1.order_id='+str(df1.iloc[t,0]))
                df2 = pd.DataFrame(cur2.fetchall())

                if len(df2)<=0:
                    positionSide='LONG'
                    if trade_way=='SELL':
                        positionSide='SHORT'                    

                    place_deal(symbol,df1.iloc[t,0],trade_way,strategy,positionSide,price,profit,loss)
                
                cur2.close()
        
            curX.close()
            
        cur1.close()
        conn.close()             
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")

def extra_open(symbol,trade_way,price,accessKey):
    conn = None
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute('select t1.id,t1.price,t1.profit,t1.loss,t1.strategy from deal_command t1 where (t1.date >= NOW() - INTERVAL 5 MINUTE) and t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\' and ISNULL(t1.close_date)')
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        
        for t in range(0,len(df1)): 
            profit=df1.iloc[t,2]
            loss=df1.iloc[t,3]
            strategy=df1.iloc[t,4]
            #print(df1)

            # if (trade_way=="BUY") and (price>df1.iloc[t,1]):
                # return

            # if (trade_way=="SELL") and (price<df1.iloc[t,1]):
                # return
                
            curX = conn.cursor()
            curX.execute('select t1.accessKey,t1.secretKey,t1.leverage,t1.host from api_key t1  where t1.accessKey=\''+accessKey+'\'  and t1.accountType='+str(accountType))        
            ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
            dfX = pd.DataFrame(curX.fetchall())   
            conn.commit()
            for x in range(0,len(dfX)): 
                apikey=dfX.iloc[x,0]   
                                    
                cur2 = conn.cursor()
                cur2.execute('select 1 from deal_detail t1  where t1.closed=0 and t1.product_code=\''+symbol+'\' and t1.api_key=\''+apikey+'\' and t1.order_id='+str(df1.iloc[t,0]))
                df2 = pd.DataFrame(cur2.fetchall())

                if len(df2)<=0:
                    positionSide='LONG'
                    if trade_way=='SELL':
                        positionSide='SHORT'                    
                    
                    place_deal(symbol,df1.iloc[t,0],trade_way,strategy,positionSide,price,profit,loss)
                    
                cur2.close()
        
            curX.close() 
            
        cur1.close()
        conn.close()             
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")

def convert_to_swap_symbol(symbol):
    """
    将交易对转换为欧易合约格式（如 BTCUSDT → BTC-USDT-SWAP）
    
    参数:
        symbol (str): 原始交易对（如 "BTCUSDT"）
    
    返回:
        str: 标准合约格式（如 "BTC-USDT-SWAP"）
    """
    # 分割基础币和报价币（假设格式为 "BTCUSDT"）
    if "USDT" in symbol:
        base_currency = symbol.split("USDT")[0]  # 提取 "BTC"
        quote_currency = "USDT"
    elif "USD" in symbol:
        base_currency = symbol.split("USD")[0]   # 提取 "BTC"
        quote_currency = "USD"
    else:
        raise ValueError(f"不支持的交易对格式: {symbol}")
    
    # 拼接为欧易合约格式
    return f"{base_currency}-{quote_currency}-SWAP"
        
def place_ba_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price):
    try: 
    
        try:
            if (strategy=='dqn') or (strategy=='ppo') or (strategy=='transformer'):
                tradeCount=tradeCount*1
                
            quantity=round(getBalance(symbol,apikey,secretkey)/tradeCount/price,getTickerQtyPrecision(symbol))     
            binance_client = Client(apikey, secretkey, testnet=False)  # 使用testnet=False进行实盘交易
            binance_client.futures_change_leverage(symbol=symbol, leverage=leverage)        
            binance_client.futures_change_position_mode(dualSidePosition=True)
        except Exception as e:
            print(f"An error occurred: {e}")

        try:
            binance_client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
        except Exception as e:
            x=1
        # 下单
        
        clientId='0'    
        try:        
            order = binance_client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET', 
                isIsolated=True,
                quantity=quantity,  # 买入数量，根据你的资金和策略调整
                positionSide=positionSide  # 确保是买单
            )
            clientId=S=str(order['orderId'])
            print(f"Order placed: {order}")
        except Exception as e:
            x=1
            
        # 设置止损单
        stopSide='SELL'
        if side=="SELL":
            stopSide='BUY'
        
        lossClientId='0'    
        try:         
            stop_loss_order = binance_client.futures_create_order(
                symbol=symbol,
                side=stopSide,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_loss_price,
                positionSide=positionSide
            )
            lossClientId=str(stop_loss_order['orderId'])
            print(f"Take loss order placed: {stop_loss_order}")
        except Exception as e:
            x=1        

        # 设置止盈单
        profitClientId='0'
        # try:  
            # take_profit_order = binance_client.futures_create_order(
                # symbol=symbol,
                # side=stopSide,
                # type='TAKE_PROFIT_MARKET',
                # quantity=quantity,
                # stopPrice=take_profit_price,
                # positionSide=positionSide
            # )
            # profitClientId=str(take_profit_order['orderId'])
            # print(f"Take profile order placed: {take_profit_order}")
        # except Exception as e:
            # x=1                      

        return clientId,lossClientId,profitClientId    
    except BinanceAPIException as e:
        print(f"An error occurred: {e}")     

def get_okx_server_time():
    """获取欧易服务器时间（返回13位毫秒级时间戳）"""
    resp = requests.get("https://www.okx.com/api/v5/public/time").json()
    return int(resp['data'][0]['ts'])  # 示例：1716800000000
    
def get_synced_timestamp():
    """获取已同步的时间戳（动态校准）"""
    local_ts = int(time.time() * 1000)
    server_ts = get_okx_server_time()
    time_diff = server_ts - local_ts
    
    # 如果时间差超过5秒，自动补偿
    if abs(time_diff) > 5000:
        return str(server_ts)
    return str(local_ts + time_diff)

def get_contract_info(futures_api, contract):
    """获取合约详细信息 - 调试版本"""
    try:
        contracts = futures_api.list_futures_contracts("usdt")
        for contract_info in contracts:
            if contract_info.name == contract:
                # 先打印所有可用字段来调试
                print(f"合约 {contract} 的所有字段:")
                # for attr in dir(contract_info):
                    # if not attr.startswith('_'):
                        # value = getattr(contract_info, attr, None)
                        # if value is not None:
                            # print(f"  {attr}: {value}")
                
                # 根据常见字段名尝试获取信息
                info = {
                    'name': contract_info.name,
                    'min_size': 1,  # 默认值
                    'size_step': 1,  # 默认值
                    'quanto_multiplier': 1,  # 默认值
                }
                
                # 尝试不同的字段名
                if hasattr(contract_info, 'order_size_min') and contract_info.order_size_min:
                    info['min_size'] = float(contract_info.order_size_min)
                elif hasattr(contract_info, 'min_order_size') and contract_info.min_order_size:
                    info['min_size'] = float(contract_info.min_order_size)
                elif hasattr(contract_info, 'min_size') and contract_info.min_size:
                    info['min_size'] = float(contract_info.min_size)
                
                if hasattr(contract_info, 'order_size_step') and contract_info.order_size_step:
                    info['size_step'] = float(contract_info.order_size_step)
                elif hasattr(contract_info, 'order_step') and contract_info.order_step:
                    info['size_step'] = float(contract_info.order_step)
                elif hasattr(contract_info, 'size_step') and contract_info.size_step:
                    info['size_step'] = float(contract_info.size_step)
                
                if hasattr(contract_info, 'quanto_multiplier') and contract_info.quanto_multiplier:
                    info['quanto_multiplier'] = float(contract_info.quanto_multiplier)
                elif hasattr(contract_info, 'multiplier') and contract_info.multiplier:
                    info['quanto_multiplier'] = float(contract_info.multiplier)
                
                #print(f"解析后的合约信息: {info}")
                return info
        
        print(f"未找到合约: {contract}")
        return {'min_size': 1, 'size_step': 1, 'quanto_multiplier': 1}
        
    except Exception as e:
        print(f"⚠️ 获取合约信息失败: {e}")
        return {'min_size': 1, 'size_step': 1, 'quanto_multiplier': 1}
      
def get_gate_futures_balance(apikey, secretkey, testnet=False):
    """获取期货账户USDT余额 - 详细版本"""
    try:
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=apikey,
            secret=secretkey
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 获取期货账户余额
        accounts = futures_api.list_futures_accounts(settle='usdt')
        
        # 打印所有可用属性用于调试
        # print("账户详细信息:")
        # for attr in dir(accounts):
            # if not attr.startswith('_'):
                # value = getattr(accounts, attr, None)
                # if value is not None:
                    # print(f"  {attr}: {value}")
        
        # 尝试不同的余额属性
        available_balance = 0
        if hasattr(accounts, 'available') and accounts.available:
            available_balance = float(accounts.available)
        elif hasattr(accounts, 'total') and accounts.total:
            available_balance = float(accounts.total)
        elif hasattr(accounts, 'unrealised_pnl'):
            # 如果有未实现盈亏，需要计算可用余额
            total = float(accounts.total) if hasattr(accounts, 'total') else 0
            unrealised_pnl = float(accounts.unrealised_pnl) if hasattr(accounts, 'unrealised_pnl') else 0
            available_balance = total - unrealised_pnl
        
        print(f"最终确定可用余额: {available_balance} USDT")
        return available_balance
            
    except Exception as e:
        print(f"❌ 获取期货余额失败: {e}")
        return 0

def calculate_safe_position_size(available_balance, trade_percent, leverage, current_price, contract_info):
    """安全计算仓位大小 - 考虑保证金要求"""
    try:
        # 计算实际可用资金 (考虑仓位比例)
        max_usable_balance = available_balance /trade_percent
        print(f"最大可用资金: {max_usable_balance:.4f} USDT (余额 {available_balance} × 份数 {trade_percent})")
        
        # 计算所需保证金 = 开仓价值 / 杠杆
        # 但需要确保不超过可用资金
        position_value = max_usable_balance * leverage
        required_margin = position_value / leverage  # 实际上就是 max_usable_balance
        
        print(f"理论开仓价值: {position_value:.4f} USDT")
        print(f"所需保证金: {required_margin:.4f} USDT")
        
        # 检查保证金是否足够
        if required_margin > available_balance:
            print(f"⚠️ 保证金不足! 需要 {required_margin:.4f} USDT, 但只有 {available_balance:.4f} USDT")
            # 调整到最大可用
            max_usable_balance = available_balance * 0.95  # 留5%作为缓冲
            position_value = max_usable_balance * leverage
            print(f"调整后可用资金: {max_usable_balance:.4f} USDT")
            print(f"调整后开仓价值: {position_value:.4f} USDT")
        
        # 计算合约张数
        quanto_multiplier = contract_info.get('quanto_multiplier', 1)
        contract_size = position_value / (current_price * quanto_multiplier)
        
        print(f"合约乘数: {quanto_multiplier}")
        print(f"计算合约张数: {contract_size:.6f}")
        
        # 调整到最小交易单位的整数倍
        size_step = contract_info.get('size_step', 1)
        if size_step > 0:
            contract_size = int(contract_size / size_step) * size_step
        
        # 确保不小于最小交易数量
        min_size = contract_info.get('min_size', 1)
        if contract_size < min_size:
            contract_size = min_size
            print(f"⚠️ 合约张数小于最小值，调整为: {contract_size}")
        
        # 确保是整数 (合约张数必须是整数)
        contract_size = int(contract_size)
        
        # 重新计算实际开仓价值和所需保证金
        actual_position_value = contract_size * current_price * quanto_multiplier
        actual_required_margin = actual_position_value / leverage
        
        print(f"最终合约张数: {contract_size}")
        print(f"实际开仓价值: {actual_position_value:.4f} USDT")
        print(f"实际所需保证金: {actual_required_margin:.4f} USDT")
        
        # 最终保证金检查
        if actual_required_margin > available_balance:
            print(f"❌ 最终检查: 保证金仍然不足! 需要 {actual_required_margin:.4f} USDT")
            return 0, 0
        
        return contract_size, actual_position_value
        
    except Exception as e:
        print(f"❌ 计算合约张数失败: {e}")
        return 0, 0

def place_gate_deal(symbol, apikey, secretkey, strategy, leverage, tradeCount, side, positionSide, price, take_profit_price, stop_loss_price, testnet=False):
    """
    创建逐仓期货订单 - 安全版本
    """
    try:
        # 解析交易对
        if '_' in symbol:
            contract = symbol
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            contract = f"{base}_USDT"
        else:
            contract = symbol.replace('/', '_')
        
        print(f"交易合约: {contract}")
        
        # 配置API客户端
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=apikey,
            secret=secretkey
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)

        # 1. 获取期货账户余额
        print("步骤1: 获取账户余额...")
        account_balance = get_gate_futures_balance(apikey, secretkey, testnet)
        if account_balance <= 10:  # 至少需要10 USDT
            print(f"❌ 期货账户余额不足: {account_balance} USDT")
            return None, 0, 0
        
        print(f"期货账户可用余额: {account_balance} USDT")
        print(f"仓位比例: {tradeCount}, 杠杆: {leverage}x")

        # 2. 获取合约信息和当前价格
        print("步骤2: 获取市场信息...")
        contract_info = get_contract_info(futures_api, contract)
        
        # 获取当前价格
        try:
            tickers = futures_api.list_futures_tickers(settle="usdt", contract=contract)
            if tickers and hasattr(tickers[0], 'last') and tickers[0].last:
                current_price = float(tickers[0].last)
                print(f"当前价格: {current_price}")
            else:
                print("❌ 无法获取当前价格")
                return None, 0, 0
        except Exception as e:
            print(f"❌ 获取价格失败: {e}")
            return None, 0, 0

        # 3. 安全计算合约张数
        print("步骤3: 安全计算合约张数...")
        contract_size, position_value = calculate_safe_position_size(
            account_balance, tradeCount, leverage, current_price, contract_info
        )
        
        if contract_size <= 0:
            print("❌ 计算合约张数失败 - 可能保证金不足")
            return None, 0, 0
        
        # 根据方向确定正负号 (正数=多单, 负数=空单)
        if side.lower() in ["sell", "short"]:
            contract_size = -contract_size
        
        size_str = str(contract_size)
        
        print(f"开仓详情:")
        print(f"  - 账户余额: {account_balance:.4f} USDT")
        print(f"  - 仓位比例: {tradeCount}")
        print(f"  - 杠杆倍数: {leverage}x")
        print(f"  - 合约张数: {size_str}")
        print(f"  - 方向: {side} ({positionSide})")
        print(f"  - 开仓价值: {position_value:.4f} USDT")
        print(f"  - 所需保证金: {position_value/leverage:.4f} USDT")

        # 4. 设置逐仓杠杆
        print("步骤4: 设置逐仓杠杆...")
        try:
            result = futures_api.update_position_leverage(
                settle="usdt", 
                contract=contract, 
                leverage=str(leverage)
            )
            print(f"✅ 设置杠杆 {leverage}x 成功")
        except Exception as e:
            print(f"❌ 设置杠杆失败: {e}")
            return None, 0, 0

        # 5. 创建订单
        print("步骤5: 创建订单...")
        futures_order = gate_api.FuturesOrder(
            contract=contract,
            size=size_str,  # 合约张数，正数多单，负数空单
            price="0",      # 0表示市价单
            tif="ioc"       # 立即成交或取消
        )
        
        # 下单
        response = futures_api.create_futures_order("usdt", futures_order)
        
        print(f"✅ 逐仓订单创建成功!")
        print(f"   订单ID: {response.id}")
        print(f"   合约: {contract}")
        print(f"   合约张数: {size_str}")
        print(f"   杠杆: {leverage}x")
        print(f"   开仓价值: {position_value:.4f} USDT")
        
        return response.id, 0, 0
        
    except gate_api.ApiException as e:
        error_msg = f"❌ API异常: {e}"
        if hasattr(e, 'body'):
            error_msg += f", 响应: {e.body}"
        print(error_msg)
        return None, 0, 0
    except Exception as e:
        error_msg = f"❌ 未知错误: {e}"
        print(error_msg)
        return None, 0, 0

# 更安全的版本 - 使用固定金额而不是比例
def place_gate_isolated_deal_fixed(symbol, apikey, secretkey, usdt_amount, leverage, side, testnet=False):
    """
    使用固定金额开仓
    """
    try:
        if '_' in symbol:
            contract = symbol
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            contract = f"{base}_USDT"
        
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=apikey,
            secret=secretkey
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 获取余额
        account_balance = get_gate_futures_balance(apikey, secretkey, testnet)
        if usdt_amount > account_balance:
            usdt_amount = account_balance * 0.95  # 使用95%的余额
            print(f"⚠️ 调整开仓金额为: {usdt_amount:.2f} USDT")
        
        # 获取价格和合约信息
        tickers = futures_api.list_futures_tickers(settle="usdt", contract=contract)
        current_price = float(tickers[0].last)
        contract_info = get_contract_info(futures_api, contract)
        
        # 计算合约张数
        quanto_multiplier = contract_info.get('quanto_multiplier', 1)
        position_value = usdt_amount * leverage
        contract_size = position_value / (current_price * quanto_multiplier)
        
        # 调整到整数
        contract_size = int(contract_size)
        if contract_size < 1:
            contract_size = 1
        
        if side.lower() in ["sell", "short"]:
            contract_size = -contract_size
        
        # 设置杠杆和下单
        futures_api.update_position_leverage(settle="usdt", contract=contract, leverage=str(leverage))
        
        order = gate_api.FuturesOrder(
            contract=contract,
            size=str(contract_size),
            price="0",
            tif="ioc"
        )
        
        response = futures_api.create_futures_order("usdt", order)
        print(f"✅ 固定金额订单成功: {response.id}")
        return response.id
        
    except Exception as e:
        print(f"❌ 固定金额下单失败: {e}")
        return None
             
def place_ok_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price):
    side=side.lower()
    passphrase="LiangHua500@"
    try:
        symbol2=symbol
        symbol=convert_to_swap_symbol(symbol)
        
        pos_side="long"
        if side=="SELL":
            pos_side="short"
        
        if (strategy=='dqn') or (strategy=='ppo') or (strategy=='transformer'):
            tradeCount=tradeCount*1
                
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        method = "POST"
        path = "/api/v5/account/config/leverage"
        query_params = ""
        message = timestamp + method + path + query_params
        signature = hmac.new(
            secretkey.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }

        data = {
            "instId": symbol,  # 交易对
            "lever": str(leverage),  # 杠杆倍数
            "mgnMode": "isolated"  # 保证金模式，cross 表示全仓，isolated 表示逐仓
        }

        url = "https://www.okx.com" + path
        response = requests.post(url, headers=headers, data=json.dumps(data))

        print("URL:", url)
        print("Headers:", headers)
        print("Data:", data)        
        print(response)
    except Exception as e1:
        print(f"An error occurred: {e1}")  
            
    try:
        # 生成签名
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        method = "GET"
        path = "/api/v5/account/balance"
        query_params = ""
        message = timestamp + method + path + query_params
        signature = hmac.new(
            secretkey.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # 请求头
        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json"
        }

        # 发送请求
        url = "https://www.okx.com" + path
        response = requests.get(url, headers=headers)
        data = response.json()
        print(message)
        print(signature)
        print(data)
        return 
        quantity=float(data["data"][0]["details"][0]["availBal"])  
        
        qty=round(quantity/tradeCount/price,getTickerQtyPrecision(symbol2))
        
        method = "POST"
        path = "/api/v5/trade/order"
        query_params = ""
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        message = timestamp + method + path + query_params
        signature = hmac.new(
            secretkey.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        # 请求头
        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": "",  # 如有密码需填写
            "Content-Type": "application/json"
        }

        # 请求体
        data = {
            "instId": symbol,  # 交易对
            "tdMode": "isolated",  # 交易模式，isolated 表示逐仓
            "side": side,  # 交易方向，buy 表示开多，sell 表示开空
            "ordType": "market",  # 订单类型，market 表示市价单
            "lever": str(leverage)  # 杠杆倍数
        }

        # 发送请求
        url = "https://www.okx.com" + path
        response = requests.post(url, headers=headers, data=json.dumps(data))

        print(response)
        
        # 返回API响应（JSON格式）
        return clOrdId,"0","0"
    except Exception as e1:
        print(f"An error occurred: {e1}") 
        
def place_hplq_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price):
    try:
        dex = ccxt.hyperliquid({
            "walletAddress": apikey,  # 替换为你的钱包地址
            "privateKey": secretkey   # 替换为你的私钥
        })

        if (strategy=='dqn') or (strategy=='ppo') or (strategy=='transformer'):
            tradeCount=tradeCount*1
                
        # 获取账户余额
        balance = dex.fetch_balance()
        account_value = float(balance['info']['marginSummary']['accountValue'])
        #print(account_value) 

        #markets = dex.load_markets()
        #print("支持的交易对：", list(markets.keys()))

        x=symbol.split("USDT")
        symbol = x[0]+"/USDC:USDC"  # 示例交易对
          
        quantity=round(account_value/tradeCount/price,getTickerQtyPrecision(symbol))   
        #print(quantity)
        
        margin_mode = "isolated"  # 逐仓模式
        leverage = leverage       # 设置杠杆
        dex.set_margin_mode(margin_mode, symbol, params={"leverage": leverage})

        # 下单示例：市价单买入
        market_type = "market"
        side = side
        amount = quantity  # 交易数量
        order = dex.create_order(symbol, market_type, side, amount, price=price)
        clientId=str(order['id'])
        lossClientId=str(0)
        profitClientId=str(0)
        print("下单信息：", order)
        # 查询仓位信息
        #positions = dex.fetch_positions([symbol])
        #print("仓位信息：", positions) 
        
        return clientId,lossClientId,profitClientId    
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
def place_deal(symbol,orderId,side,strategy,positionSide,price,take_profit_price,stop_loss_price):
    # 设置杠杆倍数
    leverage = 15  # 例如设置杠杆倍数为15倍
    clientId='0'
    lossClientId='0'
    profitClientId='0'   
    conn = None
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()                        
        cur1.execute('select t1.accessKey,t1.secretKey,t1.leverage,t1.times,t1.host,t1.accountType from api_key t1 where t1.remark=\'ok\' ')
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        for x in range(0,len(df1)):
            try:                            
                apikey=df1.iloc[x,0]
                secretkey=df1.iloc[x,1]
                leverage=df1.iloc[x,2]
                tradeCount=df1.iloc[x,3]
                accountType=df1.iloc[x,5]
                # if strategy=="ex-superma":
                    # leverage=10
                curX = conn.cursor()
                curX.execute('select 1 from deal_detail t1 where t1.closed=0 and t1.order_id='+str(orderId)+ ' and t1.api_key=\''+str(apikey)+'\'')
                ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
                dfX = pd.DataFrame(curX.fetchall())
                conn.commit() 

                if len(dfX)>0: 
                    continue
                
                if accountType==1:
                    clientId,lossClientId,profitClientId=place_ba_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price)

                if accountType==2:
                    clientId,lossClientId,profitClientId=place_hplq_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price)
                    
                if accountType==3:
                    clientId,lossClientId,profitClientId=place_ok_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price)
                    
                if accountType==4:
                    clientId,lossClientId,profitClientId=place_gate_deal(symbol,apikey,secretkey,strategy,leverage,tradeCount,side,positionSide,price,take_profit_price,stop_loss_price)

            except BinanceAPIException as e:
                print(f"An error occurred: {e}")                

            cur2 = conn.cursor()
            sql1="insert into deal_detail(product_code,trade_way,api_key,order_id,client_id,profit_id,loss_id,closed) values('" + symbol + "',"+"'" + side + "'," +"'" + apikey + "'," + str(orderId) +"," + str(clientId) + "," + str(profitClientId) +","+ str(lossClientId) +",0)";
            #print(sql1)
            count = cur2.execute(sql1)
            # 关闭Cursor对象            
            conn.commit()
            cur2.close()
            curX.close()
            
        cur1.close()
        conn.close()
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")

def auto_close():     
    df1 =pd.DataFrame()
    conn = None    
    try:  
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        sql=' SELECT t1.client_id,t1.product_code,t1.trade_way,t2.strategy,t2.period,t2.price,t2.profit,t2.loss,t3.accessKey,t3.secretKey from deal_detail t1,deal_command t2,api_key t3,users t4 '
        sql =sql +' where t1.api_key=t3.accessKey and t3.userId=t4.id and t4.user_name=\'rocky.cai\' and t1.order_id=t2.id and t2.close=0 and t1.closed=0' 
        cur1.execute(sql)
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        for x in range(0,len(df1)): 
            client_id=df1.iloc[x,0]
            symbol=df1.iloc[x,1]
            trade_way=df1.iloc[x,2]
            strategy=df1.iloc[x,3]
            period=df1.iloc[x,4]
            price=df1.iloc[x,5] 
            profit=df1.iloc[x,6]
            loss=df1.iloc[x,7] 
            apikey=df1.iloc[x,8]
            secretkey=df1.iloc[x,9]  
            
            close=getBid(symbol,apikey,secretkey)
            if  trade_way=='SELL':
                close=getAsk(symbol,apikey,secretkey) 
                
            try:   
                exchange = ccxt.binance({
                    'apiKey': apikey,
                    'secret': secretkey,           
                    'options': {
                        'defaultType': 'future'  # 指定交易类型为永续合约
                    }
                })            
                markets = exchange.loadMarkets()
                positions = exchange.fetchPositions()
                
                canClose=True
                for position in positions:
                    print(position['info'])
                    print(position['info']['symbol'])
                    if (symbol==position['info']['symbol']):
                        canClose=False
                        
                if canClose:        
                    close_order(symbol,trade_way,strategy,period,price,close,profit,loss)
                    print("关闭订单:"+str(client_id))
                    
            except Exception as e:   
                print(f"关闭订单时发生错误: {e}")
            
        cur1.close()
        conn.close()
    except Exception as e1:
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        print(f"An error occurred: {e1}") 
        
def get_close_deal(symbol,trade_way,strategy,period):
    df1 =pd.DataFrame()
    conn = None    
    try:  
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute('select * from deal_command t1  where (t1.date <= NOW() - INTERVAL 5 MINUTE) and t1.close<=0 and t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\' and t1.strategy=\''+strategy+'\' and t1.period='+str(period))
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        cur1.close()
        conn.close()
    except Exception as e1:
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        print(f"An error occurred: {e1}")      

    return df1

def get_holding(strategy): 
    conn = None
    cnt=0    
    try:  
        df1 =pd.DataFrame()
        sql="SELECT count(1) from deal_command t1 where t1.close=0 and t1.strategy not in ('dqn','ppo','transformer')"
        
        if (strategy=="dqn") or (strategy=="transformer"):
            sql="SELECT count(1) from deal_command t1 where t1.close=0 and t1.strategy in ('dqn','ppo','transformer')"
            
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute(sql)
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        if len(df1)>0:
            cnt=df1.iloc[0,0]
        cur1.close()
        conn.close()
    except Exception as e1:
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        print(f"An error occurred: {e1}")      

    return cnt
    
def close_order(symbol,trade_way,strategy,period,price,close,profit,loss):
    #print('开始关闭订单')
    conn = None     
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        sql ='select t1.id,t1.product_code,t1.trade_way,t1.strategy,t1.period from deal_command t1  where t1.close<=0  and t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\' and t1.strategy=\''+strategy+'\' and t1.period='+str(period)
        cur1.execute(sql)
        #print(sql)
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        for x in range(0,len(df1)):   
            #print('开始关闭订单3')        
            #cur2 = conn.cursor()
            #cur2.execute('select t1.id,t1.api_key,t1.trade_way,t1.client_id,t1.profit_id,t1.loss_id from deal_detail t1  where t1.closed=0 and t1.product_code=\''+symbol+'\' and t1.order_id='+str(df1.iloc[x,0]))
            #df2 = pd.DataFrame(cur2.fetchall())

            #for z in range(0,len(df2)):
            #    close_deal(symbol,df2.iloc[z,2],df2.iloc[z,3],df2.iloc[z,4],df2.iloc[z,5])

            #cur2.close()
            
            cur3 = conn.cursor()                
            deal_time = datetime.now()
            deal_date = deal_time.strftime('%Y-%m-%d %H:%M:%S')
            sql1=' update deal_command set close='+str(close)+',close_date=\''+deal_date+ '\' where id=' +str(df1.iloc[x,0])
            #print(sql1)
            count = cur3.execute(sql1)
            # 关闭Cursor对象                
            conn.commit()
            cur3.close()
        
        cur1.close()
        conn.close()
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
        
def extra_close(symbol,trade_way):
    #print('开始关闭订单')
    conn = None     
    try:        
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        sql ='select t1.id,t1.product_code,t1.trade_way,t1.strategy,t1.period from deal_command t1  where (DATE_ADD(t1.close_date,INTERVAL 1 hour)>=now()) and t1.close>0 and t1.product_code=\''+symbol+'\' and t1.host_name=\''+'Power'+'\' and t1.trade_way=\''+trade_way+'\''
        cur1.execute(sql)
        #print(sql)
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())
        conn.commit()
        for x in range(0,len(df1)):   
            #print('开始关闭订单3')
            apikey=df1.iloc[x,0]            
            cur2 = conn.cursor()
            cur2.execute('select t1.id,t1.api_key,t1.trade_way,t1.client_id,t1.profit_id,t1.loss_id from deal_detail t1  where t1.closed=0 and t1.product_code=\''+symbol+'\' and t1.order_id='+str(df1.iloc[x,0]))
            df2 = pd.DataFrame(cur2.fetchall())

            for z in range(0,len(df2)):
                close_deal(symbol,df2.iloc[z,2],df2.iloc[z,3],df2.iloc[z,4],df2.iloc[z,5])

            cur2.close()
            
        cur1.close()
        conn.close()
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")
            
def close_ok_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id): 
    side=side.lower()
    try:
        side1="BUY"
        if side=="BUY":
            side1="SELL"
            
        # 欧易API配置
        url = "https://www.okx.com/api/v5/trade/order"
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"        
        # 构造请求参数
        params = {
            "instId": symbol,
            "clOrdId": clOrdId,       # 指定客户端订单ID
            "tdMode": "isolated",      # 逐仓模式（可选 "cross" 全仓）
            "side": side1,              # 平仓方向（buy/sell）
            "sz": "auto",              # 自动平仓全部仓位
            "ordType": "market"
        } 

        # 生成签名
        body = json.dumps(params)
        message = timestamp + "POST" + "/api/v5/trade/order" + body
        signature = hmac.new(
            secretkey.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()        
        # 请求头
        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": "",  # 如有密码需填写
            "Content-Type": "application/json",
        }
        
        # 发送请求
        response = requests.post(url, headers=headers, data=body)
        print(response)
    
    except Exception as e1:
        print(f"An error occurred: {e1}") 
        
def close_ba_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id): 
    stopSide='SELL'
    positionSide='LONG'
    if side=='SELL':
        stopSide='BUY'
        positionSide='SHORT' 
        
    try:
        binance_client = Client(apikey, secretkey, testnet=False)  # 使用testnet=False进行实盘交易 
        try:         
            binance_client.futures_cancel_all_open_orders(symbol=symbol,orderId=profit_id)
        except Exception as e: 
            x=1
        try:     
            binance_client.futures_cancel_all_open_orders(symbol=symbol,orderId=loss_id)
        except Exception as e: 
            x=1            
        # 尝试获取原多单的详细信息 
        #print(apikey)
        #print(secretkey)
        #print(symbol)
        #print(client_id)            
        original_order = binance_client.futures_get_order(symbol=symbol,orderId=client_id)
        #print(original_order)
        # 检查订单是否存在
        if original_order:
            # 计算要平仓的数量，这里我们选择平掉原订单的全部数量
            quantity_to_close = original_order['origQty']
            #print(quantity_to_close)
            # 执行卖出订单来平仓多头仓位
            close_order = binance_client.futures_create_order(
                symbol=symbol,
                side=stopSide,
                type='MARKET',  # 使用市价单快速平仓
                quantity=quantity_to_close,
                positionSide=positionSide # 指定平多头仓位
            )
                
            print(f"Close order placed: {close_order}")
        else:
            print(f"No order found with orderId: {client_id}")   
    except Exception as e1:
        print(f"An error occurred: {e1}") 

def get_order_by_id(order_id: str,api_key: str,api_secret: str, settle: str = "usdt") -> Dict[str, Any]:
    """
    根据订单ID获取订单信息
    
    Args:
        order_id: 订单ID
        settle: 结算货币
        
    Returns:
        Dict: 订单信息
    """
    try:
        # 配置API客户端
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=api_key,
            secret=api_secret
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 方法1: 尝试直接获取订单
        try:
            order = futures_api.get_futures_order(settle,order_id)
        except Exception as e:
            print(f"直接获取订单失败: {e}")
            # 方法2: 通过列表查询找到该订单
            orders = futures_api.list_futures_orders(settle, limit=100)
            order = None
            for o in orders:
                if o.id == order_id:
                    order = o
                    break
            
            if not order:
                raise Exception(f"订单 {order_id} 未找到")
        
        #print(order)
        order_info = {
            'id': order.id,
            'contract': order.contract,
            'size': float(order.size),
            'price': float(order.price) if order.price else 0,
            'close': order.close,
            #'side': order.side,
            #'order_type': order.order_type,
            'status': order.status,
            'left': float(order.left) if order.left else 0,
            'fill_price': float(order.fill_price) if order.fill_price else 0,
            'create_time': order.create_time,
            'finish_time': order.finish_time,
            'text': order.text if order.text else ''
        }
        
        return order_info
        
    except gate_api.ApiException as e:
        print(f"API异常: {e}")
        return {"error": f"API异常: {e}"}
    except Exception as e:
        print(f"获取订单信息时出错: {e}")
        return {"error": f"获取订单信息时出错: {e}"}
        
        
    except gate_api.ApiException as e:
        print(f"❌ 获取订单 {order_id} 失败: {e}")
        return {}

def check_order_and_position_status(api_key: str, api_secret: str, order_id: str, testnet: bool = False) -> Dict[str, Any]:
    """
    详细检查订单和持仓状态
    """
    try:
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=api_key,
            secret=api_secret
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        print(f"=== 详细检查订单 {order_id} ===")
        
        # 1. 获取订单详细信息
        print("1. 获取订单信息...")
        order = futures_api.get_futures_order( "usdt",order_id)
        
        order_info = {
            'id': order.id,
            'contract': order.contract,
            'size': float(order.size),
            'price': float(order.price) if order.price else 0,
            'close': order.close,
            #'side': order.side,
            #'order_type': order.order_type,
            'status': order.status,  # 重要：检查订单状态
            'left': float(order.left) if order.left else 0,
            'fill_price': float(order.fill_price) if order.fill_price else 0,
            'create_time': order.create_time,
            'finish_time': order.finish_time,
            'text': order.text if order.text else ''
        }
        
        print(f"订单状态: {order_info['status']}")
        print(f"剩余数量: {order_info['left']}")
        print(f"是否已成交: {order_info['fill_price'] > 0}")
        
        # 2. 检查所有持仓
        print("\n2. 检查所有持仓...")
        all_positions = futures_api.list_positions("usdt")
        
        active_positions = []
        for position in all_positions:
            if position.size and float(position.size) != 0:
                pos_info = {
                    'contract': position.contract,
                    'size': float(position.size),
                    'value': float(position.value) if position.value else 0,
                    'entry_price': float(position.entry_price) if position.entry_price else 0,
                    'unrealised_pnl': float(position.unrealised_pnl) if position.unrealised_pnl else 0
                }
                active_positions.append(pos_info)
        
        print(f"找到 {len(active_positions)} 个活跃持仓:")
        for pos in active_positions:
            print(f"  - {pos['contract']}: {pos['size']} 张")
               
        return {
            'order_info': order_info,
            'active_positions': active_positions,
            'has_position': len(active_positions) > 0
        }
        
    except Exception as e:
        print(f"检查状态时出错: {e}")
        return {"error": f"检查状态时出错: {e}"}

def cancel_order_by_id(api_key: str, api_secret: str, order_id: str, testnet: bool = False) -> Dict[str, Any]:
    """
    如果订单是挂单状态，则取消它
    """
    try:
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=api_key,
            secret=api_secret
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 1. 首先获取订单信息
        order = futures_api.get_futures_order("usdt", order_id)
        print(f"订单信息: ID={order.id}, 合约={order.contract}, 大小={order.size}, 状态={order.status}")
        
        # 2. 获取当前持仓信息
        hold = get_active_positions_by_contract(order.contract, api_key, api_secret, "usdt")
        print(f"持仓信息: {hold}")
        
        if not hold or 'size' not in hold:
            print(f"❌ 未找到合约 {order.contract} 的持仓信息")
            return {"error": f"No position found for {order.contract}"}
        
        current_size = float(hold['size'])
        actual_close_size = float(order.size)
        
        print(f"当前持仓: {current_size} 张")
        print(f"订单平仓数量: {actual_close_size} 张")
        
        # 3. 详细验证持仓和平仓数量
        if current_size == 0:
            print(f"❌ 当前持仓为0，无法平仓")
            return {"error": "Position size is zero"}
        
        if abs(actual_close_size) > abs(current_size):
            print(f"❌ 平仓数量 {actual_close_size} 大于持仓数量 {current_size}")
            # 调整到最大可平仓数量
            actual_close_size = current_size
            print(f"调整平仓数量为: {actual_close_size}")
        
        # 4. 检查持仓方向和平仓方向是否匹配
        if (current_size > 0 and actual_close_size > 0) or (current_size < 0 and actual_close_size < 0):
            print(f"⚠️ 持仓方向和平仓方向相同，调整平仓数量符号")
            actual_close_size = -actual_close_size if current_size > 0 else abs(actual_close_size)
            print(f"调整后平仓数量: {actual_close_size}")
        
        print(f"=== 部分平仓 {order.contract} ===")
        print(f"当前持仓: {current_size} 张")
        print(f"实际平仓: {actual_close_size} 张")
        print(f"平仓后剩余: {current_size + actual_close_size} 张")
        
        # 5. 确定最终平仓数量（必须是整数）
        final_close_size = int(round(abs(actual_close_size)))
        if final_close_size <= 0:
            print(f"❌ 平仓数量为0或负数: {final_close_size}")
            return {"error": "Invalid close size"}
        
        print(f"整数平仓数量: {final_close_size}")
        
        # 6. 根据持仓方向确定平仓参数 - 修复：添加 tif 参数
        if current_size > 0:  # 多仓平仓
            print("持仓方向: 多仓 → 平仓方向: 卖出")
            futures_order = gate_api.FuturesOrder(
                contract=order.contract,
                size=-final_close_size,  # 负数表示卖出平仓
                price="0",               # 市价单
                tif="ioc",               # 关键修复：添加 IOC 时间条件
                reduce_only=True         # 只减仓
            )
        else:  # 空仓平仓
            print("持仓方向: 空仓 → 平仓方向: 买入")
            futures_order = gate_api.FuturesOrder(
                contract=order.contract,
                size=final_close_size,   # 正数表示买入平仓
                price="0",               # 市价单
                tif="ioc",               # 关键修复：添加 IOC 时间条件
                reduce_only=True         # 只减仓
            )
        
        print(f"最终订单参数: 合约={futures_order.contract}, 数量={futures_order.size}, tif={futures_order.tif}")
        
        # 7. 提交订单
        created_order = futures_api.create_futures_order("usdt", futures_order)
        #print(f"订单创建结果: {created_order}")
        
        order_info = {
            'id': created_order.id,
            'contract': created_order.contract,
            'size': float(created_order.size),
            'status': created_order.status,
            'left': float(created_order.left) if created_order.left else 0,
            'fill_price': float(created_order.fill_price) if created_order.fill_price else 0
        }
        
        print(f"✅ 部分平仓订单提交成功!")
        return order_info
        
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}       
        
def close_gate_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id):         
    try:
        settle= "usdt"
        cancel_order_by_id(apikey,secretkey,str(client_id))
    except Exception as e1:
        print(f"An error occurred: {e1}") 
        
def close_hplq_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id):
    try:
        side=side.lower()
        dex = ccxt.hyperliquid({
            "walletAddress": apikey,  # 替换为你的钱包地址
            "privateKey": secretkey   # 替换为你的私钥
        })

        #markets = dex.load_markets()
        #print("支持的交易对：", list(markets.keys()))

        x=symbol.split("USDT")
        symbol = x[0]+"/USDC:USDC"  # 示例交易对 
        
        # 查询当前仓位
        #symbol = x[0]+"/USDC:USDC"  # 示例交易对
        positions = dex.fetch_positions([symbol])
        #print(positions)
        # 如果有仓位，平仓
        if positions:
            market_type= "market"
            position = positions[0]
            amount = position["contracts"]  # 获取仓位数量
            side = "SELL" if position["side"] == "long" else "BUY"  # 平仓方向与持仓方向相反
            price = dex.load_markets()[symbol]["info"]["midPx"]# 使用市价单平仓
            slippage = 0.02
            order = dex.create_order(symbol, market_type, side, amount, price=price, params={"slippage": slippage})
            print("平仓订单：", order)
        else:
            print("没有仓位需要平仓")  
            
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
def close_deal(symbol,side,client_id,profit_id,loss_id):  
    #print('执行关闭订单')
    stopSide='SELL'
    positionSide='LONG'
    if side=='SELL':
        stopSide='BUY'
        positionSide='SHORT'    
        
    try:
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        cur1 = conn.cursor()
        cur1.execute('select t1.accessKey,t1.secretKey,t2.id,t1.leverage,t1.host,t1.accountType from api_key t1,deal_detail t2  where t2.closed=0 and t1.accessKey=t2.api_key and t2.client_id='+str(client_id))
        ###使用 fetchone() 方法获取单条数据.fetchall()获取所有数据
        df1 = pd.DataFrame(cur1.fetchall())        
        conn.commit()
        for x in range(0,len(df1)):
            apikey=df1.iloc[x,0]
            secretkey=df1.iloc[x,1]
            detailid=df1.iloc[x,2]
            accountType=df1.iloc[x,5]
            #print(accountType)
            if accountType==1:
                close_ba_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id)

            if accountType==2:
                close_hplq_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id)

            if accountType==3:
                close_ok_deal(symbol,apikey,secretkey,side,client_id,profit_id,loss_id)

            if accountType==4:
                close_gate_deal(symbol,apikey,secretkey,positionSide,client_id,profit_id,loss_id)
                
            cur2 = conn.cursor()                
            deal_time = datetime.now()
            deal_date = deal_time.strftime('%Y-%m-%d %H:%M:%S')
            sql1=' update deal_detail set closed=1 where id=' +str(detailid)
            count = cur2.execute(sql1)
            # 关闭Cursor对象                
            conn.commit()
            cur2.close()
                
        cur1.close()
        conn.close()        
    except Exception as e1:
        print(f"An error occurred: {e1}")        
        try:
            conn.ping(True)
        except Exception as e2:
            print(f"An error occurred: {e2}")

def testPlace(symbol,side,price,take_profit_price,stop_loss_price,tradeCount):
    try:
        stopSide='SELL'
        positionSide='LONG'
        if side=='SELL':
            stopSide='BUY'
            positionSide='SHORT'
            
        symbol=symbol
        apikey='MCxSspLpTlrWW63MPxxZ5Aqq4nqORTZgA9bUZFlZSEalf67lMCySvhA5ztR1KARa'
        secretkey='wNywZxMXmTQZ0EQiWDcCOEAsbJXwyp6pHrTWGQndTdrEcppzVzq5jrRMMUX1JYMJ'
        quantity=round(getBalance(symbol,apikey,secretkey)/tradeCount/price,getTickerQtyPrecision(symbol))  
        leverage=10
        
        binance_client = Client(apikey, secretkey, testnet=False)  # 使用testnet=False进行实盘交易
        binance_client.futures_change_leverage(symbol=symbol, leverage=leverage)
        
        try:
            binance_client.futures_change_position_mode(dualSidePosition=True)
        except Exception as e:
            x=1

        try:
            binance_client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
        except Exception as e:
            x=1
                
        
        # 下单              
        order = binance_client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET', 
            isIsolated=True,
            quantity=quantity,  # 买入数量，根据你的资金和策略调整
            positionSide=positionSide  # 确保是买单
        )
        clientId=str(order['orderId'])
        print(f"Order placed: {order}")

        # 设置止损单
        stopSide='SELL'
        if side=="SELL":
            stopSide='BUY'
        stop_loss_order = binance_client.futures_create_order(
            symbol=symbol,
            side=stopSide,
            type='STOP_MARKET',
            quantity=quantity,
            stopPrice=stop_loss_price,
            positionSide=positionSide
        )
        lossClientId=str(stop_loss_order['orderId'])
        print(f"Stop loss order placed: {stop_loss_order}")

        # 设置止盈单
        take_profit_order = binance_client.futures_create_order(
            symbol=symbol,
            side=stopSide,
            type='TAKE_PROFIT_MARKET',
            quantity=quantity,
            stopPrice=take_profit_price,
            positionSide=positionSide
        )
        profitClientId=str(take_profit_order['orderId'])
        print(f"Take profile order placed: {take_profit_order}")
    except Exception as e1:
        print(f"An error occurred: {e1}")  

def testClose(symbol,side,price,client_id,profit_id,loss_id,tradeCount):
    try:
        stopSide='SELL'
        positionSide='LONG'
        if side=='SELL':
            stopSide='BUY'
            positionSide='SHORT'
            
        symbol=symbol
        apikey='MCxSspLpTlrWW63MPxxZ5Aqq4nqORTZgA9bUZFlZSEalf67lMCySvhA5ztR1KARa'
        secretkey='wNywZxMXmTQZ0EQiWDcCOEAsbJXwyp6pHrTWGQndTdrEcppzVzq5jrRMMUX1JYMJ'
        quantity=round(getBalance(symbol,apikey,secretkey)/tradeCount/price,getTickerQtyPrecision(symbol))  
        leverage=10

        # 创建Binance Futures API客户端
        binance_client = Client(apikey, secretkey, testnet=False)  # 使用testnet=False进行实盘交易 
        try:         
            binance_client.futures_cancel_all_open_orders(symbol=symbol,orderId=float(profit_id))
        except Exception as e: 
            x=1
        try:     
            binance_client.futures_cancel_all_open_orders(symbol=symbol,orderId=float(loss_id))
        except Exception as e: 
            x=1  
            
        # 尝试获取原多单的详细信息        
        original_order = binance_client.futures_get_order(symbol=symbol,orderId=float(client_id))
        print(original_order)
        # 检查订单是否存在
        if original_order:
            # 计算要平仓的数量，这里我们选择平掉原订单的全部数量
            quantity_to_close = original_order['origQty']
            #print(quantity_to_close)
            # 执行卖出订单来平仓多头仓位
            close_order = binance_client.futures_create_order(
                symbol=symbol,
                side=stopSide,
                type='MARKET',  # 使用市价单快速平仓
                quantity=quantity_to_close,
                positionSide=positionSide # 指定平多头仓位
            )                
            print(f"Close order placed: {close_order}")
        else:
            print(f"No order found with orderId: {client_id}")
            
    except Exception as e1:
        print(f"An error occurred: {e1}")  
        
def get_ba_accountInfo(apikey, secretkey):
    try:
        binance_client = Client(apikey, secretkey, testnet=False)  # 使用testnet=False进行实盘交易 
        
        #account_info = binance_client.get_account()
        print(getBalance('BTCUSDT',apikey,secretkey))                      
        
        positions = binance_client.futures_position_information()
        holdings = []
        for position in positions:
            symbol = position['symbol']
            position_amt = float(position['positionAmt'])  # Positive = long, Negative = short
            entry_price = float(position['entryPrice'])
            unrealized_pnl = float(position['unRealizedProfit'])
            
            # Include only open positions
            if position_amt != 0:      
                holdings.append(position)
                #print(position)
                
                # symbol_info = binance_client.get_symbol_info(symbol)
                # max_quantity = None
                # lot_size_filter = next((filter_info for filter_info in symbol_info['filters'] if filter_info['filterType'] == 'LOT_SIZE'), None)
                # if lot_size_filter:
                    # max_quantity = float(lot_size_filter['maxQty'])
                # else:
                    # max_quantity = None  # 如果没有找到 LOT_SIZE 过滤器，设置为 None 或者合适                    
                # print(max_quantity)  
                
                #testClose(symbol,side,5.652,29872086666,0,0,3)                      
                
        # Print positions
        for holding in holdings:
            print(holding)
    except Exception as e1:
        print(f"An error occurred: {e1}") 

def list_open_orders_alt(api_key: str, api_secret: str, settle: str = "usdt", contract: str = None, testnet: bool = False) -> List[Dict[str, Any]]:
    """
    备用方法：获取当前未成交订单列表
    """
    try:
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=api_key,
            secret=api_secret
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 方法1: 不使用status参数，然后手动过滤
        all_orders = futures_api.list_futures_orders(settle,status="open", limit=100)
        
        open_orders = []
        for order in all_orders:
            if order.status == 'open':
                # 如果指定了合约，还需要匹配合约
                if contract and order.contract != contract:
                    continue
                    
                order_info = {
                    'id': order.id,
                    'contract': order.contract,
                    'size': float(order.size),
                    'price': float(order.price) if order.price else 0,
                    'close': order.close,
                    #'side': order.side,
                    #'order_type': order.order_type,
                    'status': order.status,
                    'left': float(order.left) if order.left else 0,
                    'fill_price': float(order.fill_price) if order.fill_price else 0,
                    'create_time': order.create_time,
                    'finish_time': order.finish_time,
                    'text': order.text if order.text else ''
                }
                open_orders.append(order_info)
        
        return open_orders
        
    except Exception as e:
        print(f"备用方法获取未成交订单时出错: {e}")
        return []

def get_positions(api_key: str, api_secret: str, settle: str = "usdt", testnet: bool = False) -> List[Dict[str, Any]]:
    """
    获取所有持仓合约
    """
    try:
        configuration = gate_api.Configuration(
            host="https://api.gateio.ws/api/v4" if not testnet 
                 else "https://fx-api-testnet.gateio.ws/api/v4",
            key=api_key,
            secret=api_secret
        )
        
        api_client = gate_api.ApiClient(configuration)
        futures_api = gate_api.FuturesApi(api_client)
        
        # 获取所有持仓
        positions = futures_api.list_positions(settle)
        
        position_list = []
        for position in positions:
            position_info = {
                'contract': position.contract,
                'size': float(position.size),
                'value': float(position.value) if position.value else 0,
                'leverage': float(position.leverage) if position.leverage else 0,
                'leverage_max': float(position.leverage_max) if position.leverage_max else 0,
                'entry_price': float(position.entry_price) if position.entry_price else 0,
                'liq_price': float(position.liq_price) if position.liq_price else 0,
                'mark_price': float(position.mark_price) if position.mark_price else 0,
                'unrealised_pnl': float(position.unrealised_pnl) if position.unrealised_pnl else 0,
                'realised_pnl': float(position.realised_pnl) if position.realised_pnl else 0,
                'history_pnl': float(position.history_pnl) if position.history_pnl else 0,
                'last_close_pnl': float(position.last_close_pnl) if position.last_close_pnl else 0,
                'adl_ranking': position.adl_ranking,
                'pending_orders': position.pending_orders,
                'close_order': position.close_order,
                'mode': position.mode,  # 仓位模式: single-逐仓, dual-双仓
                'cross_leverage_limit': float(position.cross_leverage_limit) if position.cross_leverage_limit else 0,
                'update_time': position.update_time
            }
            position_list.append(position_info)
        
        return position_list
        
    except Exception as e:
        print(f"获取持仓信息时出错: {e}")
        return []
        
def get_active_positions(api_key: str, api_secret: str, settle: str = "usdt", testnet: bool = False) -> List[Dict[str, Any]]:
    """
    获取有实际持仓的合约（size不为0）
    """
    try:
        all_positions = get_positions(api_key, api_secret, settle, testnet)
        
        # 过滤出有实际持仓的合约
        active_positions = []
        for position in all_positions:
            if position.get('size', 0) != 0:
                active_positions.append(position)
        
        return active_positions
        
    except Exception as e:
        print(f"获取活跃持仓时出错: {e}")
        return []

def get_active_positions_by_contract(contract: str,api_key: str, api_secret: str, settle: str = "usdt", testnet: bool = False):
    """
    获取有实际持仓的合约（size不为0）
    """
    try:
        all_positions = get_active_positions(api_key, api_secret, settle, testnet)
        
        # 过滤出有实际持仓的合约
        active_positions = []
        for position in all_positions:
            if position.get('contract', '') == contract:
                active_positions=position
                return position
                
        return None
        
    except Exception as e:
        print(f"获取活跃持仓时出错: {e}")
        return []
        
def get_position_summary(api_key: str, api_secret: str, settle: str = "usdt", testnet: bool = False) -> Dict[str, Any]:
    """
    获取持仓汇总信息
    """
    try:
        active_positions = get_active_positions(api_key, api_secret, settle, testnet)
        
        if not active_positions:
            return {"total_positions": 0, "total_value": 0, "total_pnl": 0}
        
        total_value = sum(position.get('value', 0) for position in active_positions)
        total_pnl = sum(position.get('unrealised_pnl', 0) for position in active_positions)
        
        # 按盈亏排序
        profitable_positions = [p for p in active_positions if p.get('unrealised_pnl', 0) > 0]
        loss_positions = [p for p in active_positions if p.get('unrealised_pnl', 0) < 0]
        
        summary = {
            'total_positions': len(active_positions),
            'total_value': total_value,
            'total_unrealised_pnl': total_pnl,
            'profitable_count': len(profitable_positions),
            'loss_count': len(loss_positions),
            'positions': active_positions
        }
        
        return summary
        
    except Exception as e:
        print(f"获取持仓汇总时出错: {e}")
        return {"error": f"获取持仓汇总时出错: {e}"}

        
def get_gate_accountInfo(apikey, secretkey):
    try:    
        print(get_gate_futures_balance(apikey,secretkey)) 

        # 快速使用
        load_dotenv()

        positions = get_position_summary(apikey, secretkey,"usdt")
        if positions:
            print(positions)
        else:    
            print("当前没有持仓")        
    except Exception as e1:
        print(f"An error occurred: {e1}")     


def get_simple_positions(api_key: str, api_secret: str) -> list:
    """
    快速获取持仓的简化函数
    """
    configuration = gate_api.Configuration(
        host="https://api.gateio.ws/api/v4",
        key=api_key,
        secret=api_secret
    )
    
    api_client = gate_api.ApiClient(configuration)
    futures_api = gate_api.FuturesApi(api_client)
    
    try:
        positions = futures_api.list_positions("usdt")
        active_positions = []
        
        for pos in positions:
            if float(pos.size) != 0:
                active_positions.append({
                    '合约': pos.contract,
                    '方向': '多头' if float(pos.size) > 0 else '空头',
                    '数量': abs(float(pos.size)),
                    '价值': float(pos.value),
                    '杠杆': pos.leverage,
                    '浮动盈亏': float(pos.unrealised_pnl),
                    '入场价': float(pos.entry_price) if pos.entry_price else 0
                })
        
        return active_positions
        
    except Exception as e:
        print(f"错误: {e}")
        return []
        
def get_ok_accountInfo(apikey, secretkey):
    try:
        symbol="usdt"
        url = "https://www.okx.com/api/v5/account/balance"
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"

        #symbol=convert_to_swap_symbol(symbol)
    
        # 请求参数（可指定币种）
        currency="USDT"
        params = {"ccy": currency} if currency else None
        
        # 生成签名
        message = timestamp + "GET" + "/api/v5/account/balance" + ("" if not params else f"?ccy={currency}")
        signature = hmac.new(
            secretkey.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # 请求头
        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": "",  # 如有密码需填写
        }
        
        # 发送请求
        response = requests.get(url, headers=headers, params=params) 
        data = response.json()
        quantity=float(data["data"][0]["details"][0]["availBal"])  
        print(quantity)

        url = "https://www.okx.com/api/v5/account/positions"
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + "Z"
        
        # 请求参数
        params = {"instType": inst_type}  # 可选：过滤合约类型
        
        # 签名和请求头（同余额查询）
        message = timestamp + "GET" + "/api/v5/account/positions" + ("" if not params else f"?instType={inst_type}")
        signature = hmac.new(secretkey.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
        headers = {
            "OK-ACCESS-KEY": apikey,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": passphrase,
        }
        
        # 发送请求
        response = requests.get(url, headers=headers, params=params)
        
        print(response["data"])
        
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
def get_hplq_accountInfo(apikey, secretkey):
    try:
        dex = ccxt.hyperliquid({
            "walletAddress": apikey,  # 替换为你的钱包地址
            "privateKey": secretkey   # 替换为你的私钥
        })

        # 获取账户余额
        balance = dex.fetch_balance()
        account_value = float(balance['info']['marginSummary']['accountValue'])
        print(account_value) 

        # 获取所有支持的交易对
        markets = dex.load_markets()
        symbols = list(markets.keys())  # 获取所有交易对的符号

        # 查询全部持仓仓位
        positions = dex.fetch_positions(symbols)

        for position in positions:
            print("交易对:", position["symbol"])
            print("仓位方向:", position["side"])
            print("仓位数量:", position["contracts"])
            print("入场价格:", position["entryPrice"])
            print("未实现盈亏:", position["unrealizedPnl"])
            print("清算价格:", position["liquidationPrice"])
            print("保证金模式:", position["marginMode"])
            print("------------------------------")
        
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
def testAcctInfo(accountType):
    try:      
        conn = pymysql.connect(host=HostName, port=Port, database=Database, user=User, password=Password, charset='utf8')
        curX = conn.cursor()
        curX.execute('select t1.accessKey,t1.secretKey,t1.leverage,t1.host from api_key t1  where t1.accountType='+str(accountType)) 
        
        dfX = pd.DataFrame(curX.fetchall())   
        conn.commit()  
        
        for x in range(0,len(dfX)): 
            apikey=dfX.iloc[x,0] 
            secretkey=dfX.iloc[x,1]  
            if accountType==1:
                get_ba_accountInfo(apikey, secretkey)
                
            if accountType==2:
                get_hplq_accountInfo(apikey, secretkey)

            if accountType==3:
                get_ok_accountInfo(apikey, secretkey)

            if accountType==4:
                get_gate_accountInfo(apikey, secretkey)
                
        curX.close()
    except Exception as e1:
        print(f"An error occurred: {e1}")
        
if __name__ == "__main__":   
    #********************************************# 
    #********************************************#  
    #********************************************#   
    #********************************************# 
    #********************************************#   
    
    #place_gate_deal("ADAUSDT","55a851ef800eb2bfa9c387584ce9918f","bd2c72d15ae1a6b8ef916f2993229789154ae59ce6602d67afacb9b3a23578bd","trendX",20,30,"BUY","LONG",0.8575,0.8775,0.8475)
    #close_gate_deal("ADAUSDT","55a851ef800eb2bfa9c387584ce9918f","bd2c72d15ae1a6b8ef916f2993229789154ae59ce6602d67afacb9b3a23578bd","long","5066550878089887","0","0")
    #close_gate_deal("ETHUSDT","55a851ef800eb2bfa9c387584ce9918f","bd2c72d15ae1a6b8ef916f2993229789154ae59ce6602d67afacb9b3a23578bd","long","63894828978619854","0","0")    
    testAcctInfo(1)
    testAcctInfo(2)
    testAcctInfo(4)
    print(get_holding("normal"))
    print(get_holding("dqn"))   
    