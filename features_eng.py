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

asset = 0.0
sleepTime = 60
sharpeBase = 0.03
closeRate = 0.03
dHoursCloseRate = 0.15
iCloseZoom = 36
miniWin = 0.03
maxHoldinhg = 6

work_dir = os.path.dirname(os.path.abspath(__file__))
dataFile = "products--transformer.txt"
arr_MaMethod = ["MODE_SMA"]
arr_symbols = ['BTCUSDT', 'DOGEUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'UNIUSDT']
arr_PeriodTick = ["15m"]
arr_PeriodS = ["15m", "30m", "1h", "4h"]
arr_PeriodI = [15, 30, 60, 240]
dataFile = ""
arr_Deal_PeriodI = [60]
#label = sys.argv[1]
#dataYear = float(sys.argv[2])
#run_type = sys.argv[3]
    
iRealDataCounts = 1000
iPosReadyCount = 16
iRefContinueCount = 3
iContinueCount = 60
iContinueCount2 = 48
iContinueLimit = 12

dShadowRate = 0.7
iMacdBrokenRefCount = 12
iMaReverseCount = 8
iMaQuitCount = 5
iMaEntryCount = 3
iPreMacdCount = 0
iPreFixedMacdCount = 9
maClosePeriod = 15
iPeriodSuperRef = 10
iPeriodSuperQuit = 8
iPeriodFrom = 5
iPeriodTo = 10
chanCount = 5
iRefChanCount = 80
neural_count = 100

MODE_MAIN = 0
MODE_UPPER = 1
MODE_LOWER = 2
MODE_SIGNAL = 3
PRICE_CLOSE = 'Close'
PRICE_OPEN = 'Open'

import common_eng      #基础模块
import data_eng        #行情数据模块

class featuresThread() :
    def genFeatures(self,symbol,count,run_type):
        global dict_features,dict_symbol        
        
        if (os.path.exists("./features/features-"+symbol.lower()+".csv")) and (run_type=="train"):
            features = pd.read_csv("./features/features-"+symbol.lower()+".csv")
            return features            
        
        features = {
            "timestamp":[],            
            "openh4":[],
            "closeh4":[],            
            "highh4":[],
            "lowh4":[],
            "volumeh4":[],
            "macdMh4":[],
            "macdSh4":[],            
            "kdjMh4":[],
            "kdjSh4":[],
            "rsih4":[],
            "band1h4":[],            
            "band2h4":[],
            "band3h4":[],
            "ma5h4":[],            
            "ma10h4":[],
            "ma30h4":[],
            "ma50h4":[],
            "ma100h4":[],           

            "opend1":[],
            "closed1":[],            
            "highd1":[],
            "lowd1":[],
            "volumed1":[],
            "macdMd1":[],
            "macdSd1":[],            
            "kdjMd1":[],
            "kdjSd1":[],
            "rsid1":[],
            "band1d1":[],            
            "band2d1":[],
            "band3d1":[],
            "ma5d1":[],            
            "ma10d1":[],
            "ma30d1":[],
            "ma50d1":[],
            "ma100d1":[],
            
            "getRefHighIndh4":[],
            "getRefLowIndh4":[],            
            "getRefHighIndd1":[],
            "getRefLowIndd1":[],
            
            "iStochasticDown":[],
            "iStochasticUp":[],
            
            "isQianDown":[],
            "iKdjDownInd":[],            
            "iRsiDownInd":[],
            "isRsiDown":[],
            "iMacdDownInd":[],
            "isTouchedTopBand":[], 
            "isBandGoDown":[],
            
            "isPowerMaDown1":[],
            "isPowerMaDown2":[],
            "isTopChan":[],
            "iContinueDownFromTopByHighId":[],
            "isContinueDown":[],
            "isPierceAndSwallowDownP":[],

            "isQianUp":[],
            "iKdjUpInd":[],            
            "iRsiUpInd":[],
            "isRsiUp":[],
            "iMacdUpInd":[],
            "isTouchedBottomBand":[],
            "isBandGoUp":[],    
            
            "isPowerMaUp1":[],
            "isPowerMaUp2":[],
            "isBottomChan":[],
            "iContinueUpFromBottomByLowId":[],
            "isContinueUp":[],
            "isPierceAndSwallowUpP":[]             
        }
        
        for x in arr_Deal_PeriodI:
            #refKDataAmount=len(dict_symbol[symbol][arr_PeriodS[thread1.refArrayInd(arr_PeriodI,x)]])
            #refStart=int((240/x+10)*100)
            #print(refKDataAmount)
            #ind =refStart
               
            ind =1                
            period=x
            upMA=0
            upChan=0
            downMA=0
            downChan=0                
            asset=0.0                
            while ind<=count:
                k=count-ind
                features=self.genFeature(symbol,x,k,features,run_type)
                if (run_type=="train"):
                    print(f"Generated Features:{ind}/{count}")
                ind+=1
        
        df = pd.DataFrame(features)
        if (run_type!="real"):
            # 价格变化率 (未来1小时)
            df['future_return'] = df['closeh4'].pct_change().shift(-1)  
            # 方向标签 (1:涨, 0:跌)
            df['direction'] = (df['future_return'] > 0).astype(int)     
            
        df.dropna(inplace=True)
        dict_features[symbol.lower()]=df
        if (run_type=="train") and ( not os.path.exists("./features/features-"+symbol.lower()+".csv")):
            df.to_csv("./features/features-"+symbol.lower()+".csv")

        #print(df)        
        return df
        
    def maintainFeatures(self,symbol,count,run_type):
        global dict_features,dict_symbol
        thread1 = common_eng.commonThread()
        
        features = {
            "timestamp":[],            
            "openh4":[],
            "closeh4":[],            
            "highh4":[],
            "lowh4":[],
            "volumeh4":[],
            "macdMh4":[],
            "macdSh4":[],            
            "kdjMh4":[],
            "kdjSh4":[],
            "rsih4":[],
            "band1h4":[],            
            "band2h4":[],
            "band3h4":[],
            "ma5h4":[],            
            "ma10h4":[],
            "ma30h4":[],
            "ma50h4":[],
            "ma100h4":[],           

            "opend1":[],
            "closed1":[],            
            "highd1":[],
            "lowd1":[],
            "volumed1":[],
            "macdMd1":[],
            "macdSd1":[],            
            "kdjMd1":[],
            "kdjSd1":[],
            "rsid1":[],
            "band1d1":[],            
            "band2d1":[],
            "band3d1":[],
            "ma5d1":[],            
            "ma10d1":[],
            "ma30d1":[],
            "ma50d1":[],
            "ma100d1":[],
            
            "getRefHighIndh4":[],
            "getRefLowIndh4":[],            
            "getRefHighIndd1":[],
            "getRefLowIndd1":[],
            
            "iStochasticDown":[],
            "iStochasticUp":[],
            
            "isQianDown":[],
            "iKdjDownInd":[],            
            "iRsiDownInd":[],
            "isRsiDown":[],
            "iMacdDownInd":[],
            "isTouchedTopBand":[], 
            "isBandGoDown":[],
            
            "isPowerMaDown1":[],
            "isPowerMaDown2":[],
            "isTopChan":[],
            "iContinueDownFromTopByHighId":[],
            "isContinueDown":[],
            "isPierceAndSwallowDownP":[],

            "isQianUp":[],
            "iKdjUpInd":[],            
            "iRsiUpInd":[],
            "isRsiUp":[],
            "iMacdUpInd":[],
            "isTouchedBottomBand":[],
            "isBandGoUp":[],    
            
            "isPowerMaUp1":[],
            "isPowerMaUp2":[],
            "isBottomChan":[],
            "iContinueUpFromBottomByLowId":[],
            "isContinueUp":[],
            "isPierceAndSwallowUpP":[]                        
        }
        
        for x in arr_Deal_PeriodI:
            #refKDataAmount=len(dict_symbol[symbol][arr_PeriodS[thread1.refArrayInd(arr_PeriodI,x)]])
            #refStart=int((240/x+10)*100)
            #print(refKDataAmount)
            #ind =refStart
            icount=count   
            ind =1                
            period=x
            upMA=0
            upChan=0
            downMA=0
            downChan=0                
            asset=0.0                
            while ind<=icount:
                k=icount-ind
                features=self.genFeature(symbol,x,k,features,run_type)
                if (run_type=="train"):
                    print(f"Generated Features:{ind}/{icount}")
                ind+=1
        
        df1 = pd.DataFrame(features)
        if (run_type!="real"):
            # 价格变化率 (未来1小时)
            df1['future_return'] = df1['closeh4'].pct_change().shift(-1)
            # 方向标签 (1:涨, 0:跌)
            df1['direction'] = (df1['future_return'] > 0).astype(int)
            
        df1.dropna(inplace=True)
        
        df2=dict_features[symbol.lower()]
        
        merged_df = pd.concat([df2, df1]).drop_duplicates('timestamp', keep='last')

        # 按时间戳排序（可选）
        merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        df_last = merged_df.iloc[-count:]
        #print(f"df_last:{df_last}")     
        dict_features[symbol.lower()]=df_last
        
    def genFeature(self,symbol,period,ind,features,run_type):
        global refKDataAmount
        thread1 = common_eng.commonThread()

        features["timestamp"].append(thread1.iTime(symbol,period,ind))
        
        features["openh4"].append(thread1.iOpen(symbol,60,period*ind//60))
        features["closeh4"].append(thread1.iClose(symbol,60,period*ind//60))            
        features["highh4"].append(thread1.iHigh(symbol,60,period*ind//60))
        features["lowh4"].append(thread1.iLow(symbol,60,period*ind//60))
        features["volumeh4"].append(thread1.iVoiume(symbol,60,period*ind//60))
        features["macdMh4"].append(thread1.iMacd(symbol,60,12,26,9,PRICE_CLOSE,MODE_MAIN,period*ind//60))
        features["macdSh4"].append(thread1.iMacd(symbol,60,12,26,9,PRICE_CLOSE,MODE_SIGNAL,period*ind//60))            
        features["kdjMh4"].append(thread1.iStochastic(symbol,60,10,3,"SMA","MAIN",period*ind//60))
        features["kdjSh4"].append(thread1.iStochastic(symbol,60,10,3,"SMA","SIGNAL",period*ind//60))
        features["rsih4"].append(thread1.iRSI(symbol, 60, 14, period*ind//60))
        features["band1h4"].append(thread1.iBands(symbol,60,19,3,0,PRICE_CLOSE,MODE_MAIN,period*ind//60))      
        features["band2h4"].append(thread1.iBands(symbol,60,19,3,0,PRICE_CLOSE,MODE_UPPER,period*ind//60)) 
        features["band3h4"].append(thread1.iBands(symbol,60,19,3,0,PRICE_CLOSE,MODE_LOWER,period*ind//60)) 

        features["ma5h4"].append(thread1.iMA(symbol,60,5,period*ind//60,arr_MaMethod[0],PRICE_CLOSE))
        features["ma10h4"].append(thread1.iMA(symbol,60,10,period*ind//60,arr_MaMethod[0],PRICE_CLOSE))
        features["ma30h4"].append(thread1.iMA(symbol,60,30,period*ind//60,arr_MaMethod[0],PRICE_CLOSE))      
        features["ma50h4"].append(thread1.iMA(symbol,60,50,period*ind//60,arr_MaMethod[0],PRICE_CLOSE)) 
        features["ma100h4"].append(thread1.iMA(symbol,60,100,period*ind//60,arr_MaMethod[0],PRICE_CLOSE)) 


        features["opend1"].append(thread1.iOpen(symbol,240,period*ind//240))
        features["closed1"].append(thread1.iClose(symbol,240,period*ind//240))            
        features["highd1"].append(thread1.iHigh(symbol,240,period*ind//240))
        features["lowd1"].append(thread1.iLow(symbol,240,period*ind//240))
        features["volumed1"].append(thread1.iVoiume(symbol,240,period*ind//240))
        features["macdMd1"].append(thread1.iMacd(symbol,240,12,26,9,PRICE_CLOSE,MODE_MAIN,period*ind//240))
        features["macdSd1"].append(thread1.iMacd(symbol,240,12,26,9,PRICE_CLOSE,MODE_SIGNAL,period*ind//240))            
        features["kdjMd1"].append(thread1.iStochastic(symbol,240,10,3,"SMA","MAIN",period*ind//240))
        features["kdjSd1"].append(thread1.iStochastic(symbol,240,10,3,"SMA","SIGNAL",period*ind//240))
        features["rsid1"].append(thread1.iRSI(symbol, 240, 14, period*ind//240))
        features["band1d1"].append(thread1.iBands(symbol,240,19,3,0,PRICE_CLOSE,MODE_MAIN,period*ind//240))      
        features["band2d1"].append(thread1.iBands(symbol,240,19,3,0,PRICE_CLOSE,MODE_UPPER,period*ind//240)) 
        features["band3d1"].append(thread1.iBands(symbol,240,19,3,0,PRICE_CLOSE,MODE_LOWER,period*ind//240)) 

        features["ma5d1"].append(thread1.iMA(symbol,240,5,period*ind//240,arr_MaMethod[0],PRICE_CLOSE))
        features["ma10d1"].append(thread1.iMA(symbol,240,10,period*ind//240,arr_MaMethod[0],PRICE_CLOSE))
        features["ma30d1"].append(thread1.iMA(symbol,240,30,period*ind//240,arr_MaMethod[0],PRICE_CLOSE))      
        features["ma50d1"].append(thread1.iMA(symbol,240,50,period*ind//240,arr_MaMethod[0],PRICE_CLOSE)) 
        features["ma100d1"].append(thread1.iMA(symbol,240,100,period*ind//240,arr_MaMethod[0],PRICE_CLOSE)) 
        
        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x
            if (thread1.iRsiDownInd(symbol, x, xInd ,14,70,60)>0) and (thread1.iRsiDownInd(symbol, x, xInd ,14,70,60)<xInd+9):
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                v=-1
                break
                
        features["iRsiDownInd"].append(v) 

        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x
            if (thread1.isRsiDown(symbol, x, xInd,9,12,70)):
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                v=-1
                break
                
        features["isRsiDown"].append(v) 
        
        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x
            if (thread1.iRsiUpInd(symbol, x, xInd ,14,70,60)>0) and (thread1.iRsiDownInd(symbol, x, xInd ,14,70,60)<xInd+9):
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                v=1
                break
                
        features["iRsiUpInd"].append(v) 

        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x
            if (thread1.isRsiUp(symbol,x,xInd,9,12,30)):
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                v=-1
                break
                
        features["isRsiUp"].append(v)         
        
        v=0 
        for x in [y for y in arr_PeriodI if (y>=240 and y<=240)]:
            xInd=period*ind//x
            if (thread1.iStochastic(symbol,x,10,3,"SMA","MAIN",xInd)>=80) and (thread1.iStochastic(symbol,x,10,3,"SMA","SIGNAL",xInd)>=80):
                v=-1
                break
                
        features["iStochasticDown"].append(v)            

        v=0 
        for x in [y for y in arr_PeriodI if (y>=240 and y<=240)]:
            xInd=period*ind//x
            if (thread1.iStochastic(symbol,x,10,3,"SMA","MAIN",xInd)<=20) and (thread1.iStochastic(symbol,x,10,3,"SMA","SIGNAL",xInd)<=20):
                v=1
                break
                
        features["iStochasticUp"].append(v)  
        
        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x          
            if  (thread1.iMacdDownInd(symbol,x,xInd)>=0) and (thread1.iMacdDownInd(symbol,x,xInd)<xInd+16):
                if (iPreMacdCount>=iPreFixedMacdCount):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟MACD死叉向下!!")
                    v=-1
                    break
                    
        features["iMacdDownInd"].append(v)            

        v=0
        for x in [y for y in arr_PeriodI if y>=240]:
            xInd=period*ind//x          
            if  (thread1.iMacdUpInd(symbol,x,xInd)>=0) and (thread1.iMacdUpInd(symbol,x,xInd)<xInd+16):
                if (iPreMacdCount>=iPreFixedMacdCount):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟MACD死叉向下!!")
                    v=1
                    break
                    
        features["iMacdUpInd"].append(v) 
        
        v=0
        for u in [t for t in arr_PeriodI if (t>=240 and t<=240)]:
            xInd=period*ind//u
            if (thread1.isTouchedTopBand(symbol,u,xInd)):
                v=-1
                break
                
        features["isTouchedTopBand"].append(v)

        v=0
        for u in [t for t in arr_PeriodI if (t>=240 and t<=240)]:
            xInd=period*ind//u
            if (thread1.isBandGoDown(symbol,u,xInd)):
                v=-1
                break
                
        features["isBandGoDown"].append(v)
        
        v=0
        for u in [t for t in arr_PeriodI if (t>=240 and t<=240)]:
            xInd=period*ind//u
            if (thread1.isTouchedBottomBand(symbol,u,xInd)):
                v=1
                break
                
        features["isTouchedBottomBand"].append(v)

        v=0
        for u in [t for t in arr_PeriodI if (t>=240 and t<=240)]:
            xInd=period*ind//u
            if (thread1.isBandGoUp(symbol,u,xInd)):
                v=-1
                break
                
        features["isBandGoUp"].append(v)
        
        v1=v2=0
        for x in [y for y in arr_PeriodI if (y>=240 and y<240)]:
            xInd=period*ind//x          
            for method in arr_MaMethod:
                if (thread1.isPowerMaDown(symbol,30,50,100,x,method,xInd,12)):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟均线趋势向下!!")
                    v1=-1
                    break
                    
                if (thread1.isPowerMaDown(symbol,10,20,60,x,method,xInd,12)):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟均线趋势向下!!")
                    v2=-1
                    break
            if v!=0:
                break
                
        features["isPowerMaDown1"].append(v1) 
        features["isPowerMaDown2"].append(v2)        

        v1=v2=0
        for x in [y for y in arr_PeriodI if (y>=240 and y<240)]:
            xInd=period*ind//x          
            for method in arr_MaMethod:
                if (thread1.isPowerMaUp(symbol,30,50,100,x,method,xInd,12)):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟均线趋势向下!!")
                    v1=1
                    break
                if (thread1.isPowerMaUp(symbol,10,20,60,x,method,xInd,12)):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟均线趋势向下!!")
                    v2=1
                    break
            if v!=0:
                break
                
        features["isPowerMaUp1"].append(v1)           
        features["isPowerMaUp2"].append(v2)
        v=0
        for x in [y for y in arr_PeriodI if (y>=240 and y<=240)]:
            xInd=period*ind//x
            y=thread1.iKdjDownInd(symbol,x,xInd)
            if ((y>0) and ((y-xInd)<9)):
                v=-1
                break
                
        features["iKdjDownInd"].append(v)

        v=0
        for x in [y for y in arr_PeriodI if (y>=240 and y<=240)]:
            xInd=period*ind//x
            y=thread1.iKdjUpInd(symbol,x,xInd)
            if ((y>0) and ((y-xInd)<9)):
                v=1
                break
                
        features["iKdjUpInd"].append(v)                 
        
        v=0
        for x in [y for y in arr_PeriodI if (y>=30 and y<=60)]:
            xInd=period*ind//x
            if thread1.isTopChan(symbol,x,xInd,16)>=0:
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟顶部背驰缠绕趋势向下!!")
                v=-1
                break
                
        features["isTopChan"].append(v)

        v=0
        for x in [y for y in arr_PeriodI if (y>=30 and y<=60)]:
            xInd=period*ind//x
            if thread1.isBottomChan(symbol,x,xInd,16)>=0:
                thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(x)+")分钟顶部背驰缠绕趋势向下!!")
                v=1
                break
                
        features["isBottomChan"].append(v)
        
        v=0           
        for x in arr_PeriodI:
            if x>=60:
                xInd=period*ind//x
                for y in range(5,10):
                    if (thread1.iContinueDownFromTopByHighId(symbol,iContinueCount2,xInd,x,y,iRefContinueCount)>0):
                        v=-1
                        break
                if v!=0:
                    break
                        
        features["iContinueDownFromTopByHighId"].append(v)

        v=0           
        for x in arr_PeriodI:
            if x>=60:
                xInd=period*ind//x
                for y in range(5,10):
                    if (thread1.iContinueUpFromBottomByLowId(symbol,iContinueCount2,xInd,x,y,iRefContinueCount)>0):
                        v=1
                        break
                if v!=0:
                    break
                    
        features["iContinueUpFromBottomByLowId"].append(v)

        v1=v2=0   
        for x in [y for y in arr_PeriodI if y>=60]:
            xInd=period*ind//x        
            if (thread1.isQianDown(symbol,iCloseZoom, xInd,x)):
                v1=-1 
                
            if (thread1.isQianUp(symbol,iCloseZoom, xInd,x)):
                v2=1 
                
        features["isQianDown"].append(v1)
        features["isQianUp"].append(v2)
        
        v=0
        if thread1.isContinueDown(symbol,60,period*ind//60+1,3):
            thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")持续向下!!")
            v=-1 

        if thread1.isContinueDown(symbol,60,period*ind//240+1,3):
            thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")持续向下!!")
            v=-1                                         
        features["isContinueDown"].append(v)

        v=0
        if thread1.isContinueUp(symbol,60,period*ind//60+1,3):
            thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")持续向下!!")
            v=1 

        if thread1.isContinueUp(symbol,60,period*ind//240+1,3):
            thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")持续向下!!")
            v=1                                         
        features["isContinueUp"].append(v)
        
        v=0    
        refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)
        if (True):
            if (thread1.isPierceAndSwallowDownP(symbol,60,refId1,1.5)):            
                refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)    
                if not ((refId2<refId1) and (thread1.isPierceAndSwallowUpP(symbol,60,refId2,1.5))):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向下吞没!!")
                    v=-1  

        refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)
        refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)
        if (True):
            if (refId1<refId2) and(thread1.isPierceAndSwallowDownP(symbol,240,refId1,1.5)):
                refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)
                refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)    
                if not ((refId2<refId1) and (thread1.isPierceAndSwallowUpP(symbol,60,refId2,1.5))):                
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向下吞没!!")
                    v=-1
   
        features["isPierceAndSwallowDownP"].append(v)       

        v=0    
        refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)
        if (True):
            if (thread1.isPierceAndSwallowUpP(symbol,60,refId1,1.5)):            
                refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)    
                if not ((refId2<refId1) and (thread1.isPierceAndSwallowDownP(symbol,60,refId2,1.5))):
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向下吞没!!")
                    v=1                         

        refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)
        refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)
        if (True):
            if (refId1<refId2) and(thread1.isPierceAndSwallowDownP(symbol,240,refId1,1.5)):
                refId1=thread1.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)
                refId2=thread1.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)    
                if not ((refId2<refId1) and (thread1.isPierceAndSwallowDownP(symbol,60,refId2,1.5))):                
                    thread1.print_comment(symbol,run_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向下吞没!!")
                    v=1 
                    
        features["isPierceAndSwallowUpP"].append(v) 
           
        features["getRefHighIndh4"].append(thread1.getRefHighInd(symbol,iContinueCount,period*ind//240,240,16)-period*ind//240)
        features["getRefLowIndh4"].append(thread1.getRefLowInd(symbol,iContinueCount,period*ind//240,240,16)-period*ind//240)

        features["getRefHighIndd1"].append(thread1.getRefHighInd(symbol,iContinueCount,period*ind//240,240,16)-period*ind//240)
        features["getRefLowIndd1"].append(thread1.getRefLowInd(symbol,iContinueCount,period*ind//240,240,16)-period*ind//240)
        
        # thread1.reportDict(features,len(features)-1)
        
        return features
        
    def prepare_sequences(self,df, seq_length=60, test_size=0.2):
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
        
        df = df.copy()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp just in case
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Verify we have enough data
        if len(df) < seq_length + 1:
            raise ValueError(f"Not enough data. Need at least {seq_length + 1} rows, got {len(df)}")
        
        # Exclude non-feature columns
        features = [col for col in df.columns if col not in ['timestamp', 'future_return', 'direction']]
        
        # Verify we have features to use
        if not features:
            raise ValueError("No features found in dataframe")
     
        # 使用更高效的数据标准化方式
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
        
        # 使用numpy的滑动窗口函数提高效率
        X = np.lib.stride_tricks.sliding_window_view(scaled_data, (seq_length, scaled_data.shape[1]))
        X = X.squeeze()  # 移除多余的维度
        
        # 获取对应的目标值
        y_price = df['future_return'].values[seq_length:]
        y_dir = df['direction'].values[seq_length:]
        
        # 分割数据集
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
        y_dir_train, y_dir_test = y_dir[:split_idx], y_dir[split_idx:]

        return X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test, scaler, features        
