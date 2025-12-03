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
import mplfinance as mpf

from tqdm import tqdm
from collections import deque

import deal_eng

dict_features={}
dict_symbol={}
dict_period={}
dict_plot={}
dict_comment={}
dict_order={}
   
asset=0.0
sleepTime=60    
sharpeBase=0.03
closeRate=0.03
dHoursCloseRate=0.15
iCloseZoom=36
miniWin=0.03
maxHoldinhg=6
        
work_dir=os.path.dirname(os.path.abspath(__file__))
dataFile="products--transformer.txt"
arr_MaMethod=["MODE_SMA"]
arr_symbols=['BTCUSDT','DOGEUSDT','BNBUSDT','XRPUSDT','SOLUSDT','ADAUSDT','UNIUSDT']
arr_PeriodTick=["15m"]
arr_PeriodS=["15m","30m","1h","4h","1d"]
arr_PeriodI=[15,30,60,240,1440]
dataFile=""
arr_Deal_PeriodI=[60]   

iRealDataCounts=1000
iPosReadyCount=16
iRefContinueCount=3
iContinueCount=60
iContinueCount2=48
iContinueLimit=12

dShadowRate=0.7
iMacdBrokenRefCount=12
iMaReverseCount=8
iMaQuitCount=5
iMaEntryCount=3
iPreMacdCount=0
iPreFixedMacdCount=9
maClosePeriod=15
iPeriodSuperRef=10
iPeriodSuperQuit=8
iPeriodFrom=5
iPeriodTo=10
chanCount=5
iRefChanCount=80
neural_count = 100

MODE_MAIN = 0
MODE_UPPER = 1
MODE_LOWER = 2
MODE_SIGNAL = 3
PRICE_CLOSE = 'Close'
PRICE_OPEN = 'Open'

features_size=0

class commonThread() :
    def isCanDeal(self,symbol,period,dect_type,type,strategy,ind):
        comment='Transformer'
        self.print_comment(symbol,dect_type,comment)
        isOk=True
        
        if dect_type=="real":
            ind=0
        
        if DealTrader.get_holding("normal")>=maxHoldinhg:
            return False
        
        if type=="LONG":                      
            isRevserseOk=False
            xInd=0
            xTmp=0
            refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//1440,1440,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,240,refId1-xInd)<self.iLow(symbol,240,refId1)):
                    isRevserseOk=True
                    xTmp=refId1
                xInd+=1

            xInd=0
            refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//1440,1440,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,240,refId1-xInd)>self.iHigh(symbol,240,refId1)):
                    if not (refId1>xTmp and isRevserseOk):
                        isRevserseOk=False
                xInd+=1

            xInd=0
            xTmp=0
            refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,60,refId1-xInd)<self.iLow(symbol,60,refId1)):
                    isRevserseOk=True
                    xTmp=refId1
                xInd+=1

            xInd=0
            refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,60,refId1-xInd)>self.iHigh(symbol,60,refId1)):
                    if not (refId1>xTmp and isRevserseOk):
                        isRevserseOk=False
                xInd+=1
                
            if isRevserseOk:
                return False
                        
            # for x in [y for y in arr_PeriodI if (y==period)]:
                # xInd=period*ind//x
                # if (self.iStochastic(symbol,x,14,3,"SMA","MAIN",xInd)>=80) and (self.iStochastic(symbol,x,14,3,"SMA","MAIN",xInd)<=self.iStochastic(symbol,x,14,3,"SMA","SIGNAL",xInd)):
                    # return False
            
            isTrendOk=False    
                                       
            if strategy=="temp":                                                       
                # for x in [y for y in arr_PeriodI if (y==period)]:
                    # xInd=period*ind//x
                    # y=self.iKdjDownInd(symbol,x,xInd)
                    # if ((y>0) and ((y-xInd)<12)):
                        # isOk=False
                                            
                # for x in [y for y in arr_PeriodI if y==period]:
                    # xInd=period*ind//x
                    # for y in range(5,10):
                        # if (self.isContinueTrendDown(symbol,iContinueCount2,xInd,x,y,iRefContinueCount))  :             
                            # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(period)+")分钟趋势持续向下!!")
                            # return False                            
                            
                if self.isContinueDown(symbol,60,period*ind//60+1,3):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")持续向下!!")
                    return False 

                if self.isContinueDown(symbol,60,period*ind//240+1,3):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")持续向下!!")
                    return False       
                                         
                # refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//1440,1440,16)
                # refId2=self.getRefLowInd(symbol, iContinueCount,period*ind//1440,1440,16)
                # if (True):
                    # if (refId1<refId2) and(self.isPierceAndSwallowDownP(symbol,240,refId1,1.5)):
                        # refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)
                        # refId2=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)    
                        # if not ((refId2<refId1) and (self.isPierceAndSwallowUpP(symbol,60,refId2,1.5))):                
                            # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")分钟向下吞没!!")
                            # return False 
                   
            if dect_type=="trend":
                return isOk

            # refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,9)
            # if (True):
                # if (self.isPierceAndSwallowDownP(symbol,60,refId1,1.3)):            
                    # refId2=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,9)    
                    # if not ((refId2<refId1) and (self.isPierceAndSwallowUpP(symbol,60,refId2,1.3))):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向下吞没!!")
                        # return False  
                            
            # xInd=period*ind//240
            # if self.iClose(symbol,period,ind)>self.iMA(symbol,60,8,xInd,arr_MaMethod[0],PRICE_CLOSE):
                # return False
                
            # if (self.getRefLowInd(symbol, iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # return False
                # refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)
                # refId2=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)    
                # if not ((refId2<refId1) and (self.isPierceAndSwallowUpP(symbol,60,refId2,1.5))):              
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")分钟追到顶部!!")
                    # return False                                                         

            # if (self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)<=period*ind//240):
                # if (True):  #if (not (self.isContinueUp(symbol,60,period*ind//60+1,3))):
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟追到顶部!!")
                    # return False  
                    
            if (self.getRefHighInd(symbol, iContinueCount,period*ind//1440,240,9)<=period*ind//1440):
                return False

            if (self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,9)<=period*ind//240+2):
                return False

            # if (self.getRefHighInd("BTCUSDT", iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # if (True):
                    # isOk=False 

            # if (self.getRefLowInd("BTCUSDT", iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # if (True):
                    # isOk=False 

            # for x in [y for y in arr_PeriodI if (y>=60 and y<=240)]:
                # xInd=period*ind//x          
                # if  (self.iMacdDownInd(symbol,x,xInd)>=0) and (self.iMacdDownInd(symbol,x,xInd)<xInd+12):
                    # if (iPreMacdCount>=iPreFixedMacdCount):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(x)+")分钟MACD死叉向下!!")
                        # return False 
                        
            # for x in [y for y in arr_PeriodI if (y>=60 and y<=240)]:
                # xInd=period*ind//x
                # y=self.iKdjDownInd(symbol,x,xInd)
                # if ((y>0) and ((y-xInd)<12)):
                    # isOk=False  
                                            
            # for x in [y for y in arr_PeriodI if (y>=240 and y<=1440)]:
                # xInd=period*ind//x
                # if (self.iRsiDownInd(symbol, x, xInd ,14,70,60)>0) and (self.iRsiDownInd(symbol, x, xInd ,14,70,60)<xInd+6):
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                    # return False 

                # if (self.isRsiDown(symbol, x, xInd,9,14,70)):
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超买!!")
                    # return False 

            for u in [y for y in arr_PeriodI if (y>=240 and y<=1440)]:
                xInd=period*ind//u
                if (self.isTouchedTopBand(symbol,u,xInd)):
                    return False

                if (self.isBandGoDown(symbol,u,xInd)):
                    return False
                   
            for u in [t for t in arr_PeriodI if (t>=15 and t<=240)]:
                xInd=period*ind//u                                    
                if (self.isQianDown(symbol,iCloseZoom,xInd,u)):
                    return False  

            for u in [t for t in arr_PeriodI if (t==period or t==240)]:
                xInd=period*ind//u       
                for method in arr_MaMethod:
                    if (self.isSuperMaDown(symbol,30,60,100,u,method,xInd,12)):
                        self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        return False
                        
                    if (self.isSuperMaDown(symbol,21,55,89,u,method,xInd,12)):
                        self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        return False

                    # if self.isPowerMaDown(symbol,30,60,100,u,method,xInd,12):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        # return False
                        
                    # if self.isPowerMaDown(symbol,21,55,89,u,method,xInd,12):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        # return False
                        
            xInd=period*ind//15
            if (not self.isPosReady(symbol,15,type,dect_type,xInd)) and (dect_type=="real"):
                return False                
                            
            for x in [y for y in arr_PeriodI if (y<240)]:
                xInd=period*ind//x
                if self.isSpecAction(symbol,x,24,xInd):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(period)+")分钟特殊波动!!")
                    return False 
                                  
        ####                  
        if type=="SHORT":
            isRevserseOk=False
            xInd=0
            xTmp=0
            refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//1440,1440,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,240,refId1-xInd)>self.iHigh(symbol,240,refId1)):
                    isRevserseOk=True
                    xTmp=refId1
                xInd+=1

            xInd=0
            refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//1440,1440,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,240,refId1-xInd)<self.iLow(symbol,240,refId1)):
                    if not (refId1>xTmp and isRevserseOk):
                        isRevserseOk=False
                xInd+=1

            xInd=0
            xTmp=0
            refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,60,refId1-xInd)>self.iHigh(symbol,60,refId1)):
                    isRevserseOk=True
                    xTmp=refId1
                xInd+=1

            xInd=0
            refId1=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)
            while (((refId1-xInd)>=0) and (xInd<=2)):
                if (self.iClose(symbol,60,refId1-xInd)<self.iLow(symbol,60,refId1)):
                    if not (refId1>xTmp and isRevserseOk):
                        isRevserseOk=False
                xInd+=1
                    
            if isRevserseOk:
                return False  
                        
            # for x in [y for y in arr_PeriodI if (y==period)]:
                # xInd=period*ind//x
                # if (self.iStochastic(symbol,x,14,3,"SMA","MAIN",xInd)<=80) and (self.iStochastic(symbol,x,14,3,"SMA","MAIN",xInd)>=self.iStochastic(symbol,x,14,3,"SMA","SIGNAL",xInd)):
                    # return False 
                                          
            if strategy=="temp":                                                        
                # for x in [y for y in arr_PeriodI if (y==period)]:
                    # xInd=period*ind//x
                    # y=self.iKdjUpInd(symbol,x,xInd)
                    # if ((y>0) and ((y-xInd)<12)):
                        # isOk=False                       

                # for x in [y for y in arr_PeriodI if y==period]:
                    # xInd=period*ind//x
                    # for y in range(5,10):    
                        # if (self.isContinueTrendUp(symbol,iContinueCount2,xInd,x,y,iRefContinueCount)) :            
                            # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(period)+")分钟趋势持续向上!!")
                            # return False                                          
                               
                if self.isContinueUp(symbol,60,period*ind//60+1,3):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")持续向上!!")
                    return False  

                if self.isContinueUp(symbol,60,period*ind//240+1,3):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")持续向上!!")
                    return False  
                                                                               # refId1=self.getRefLowInd(symbol,iContinueCount,period*ind//1440,1440,12)
                # refId2=self.getRefHighInd(symbol,iContinueCount,period*ind//1440,1440,12)
                # if (True):
                    # if (refId1<refId2) and(self.isPierceAndSwallowUpP(symbol,240,refId1, 1.5)):
                        # refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)
                        # refId2=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)    
                        # if not ((refId2<refId1) and (self.isPierceAndSwallowDownP(symbol,60,refId2,1.5))):                
                            # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向上刺透!!")
                            # return False                      
                        
            if dect_type=="trend":
                return isOk

            # refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)
            # if (True):
                # if (self.isPierceAndSwallowUpP(symbol,60,refId1, 1.3)):
                    # refId2=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,12)    
                    # if not ((refId2<refId1) and (self.isPierceAndSwallowDownP(symbol,60,refId2,1.3))):            
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟向上刺透!!")
                        # return False 
                            
            # xInd=period*ind//240
            # if self.iClose(symbol,period,ind)<self.iMA(symbol,60,8,xInd,arr_MaMethod[0],PRICE_CLOSE):
                # return False       
                
            # if (self.getRefHighInd(symbol, iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # return False
                # refId1=self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,16)
                # refId2=self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)    
                # if not ((refId2<refId1) and (self.isPierceAndSwallowDownP(symbol,60,refId2,1.5))):           
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")分钟追到顶部!!")
                    # return False                                                        
                
            # if (self.getRefHighInd(symbol, iContinueCount,period*ind//240,240,16)<=period*ind//240):
                # return False


            if (self.getRefLowInd(symbol, iContinueCount,period*ind//240,240,12)<=period*ind//240+2):
                if (True):  #if (not (self.isContinueDown(symbol,60,period*ind//60+1,3))):                
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟追到底部!!")
                    return False 
                    
            if (self.getRefLowInd(symbol, iContinueCount,period*ind//1440,1440,12)<=period*ind//1440):
                if (True):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(1440)+")分钟追到底部!!")
                    return False 

            # if (self.getRefHighInd("BTCUSDT", iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # if (True):
                    # isOk=False 

            # if (self.getRefLowInd("BTCUSDT", iContinueCount,period*ind//1440,1440,16)<=period*ind//1440):
                # if (True):
                    # isOk=False 

            # for x in [y for y in arr_PeriodI if (y>=60 and y<=240)]:
                # xInd=period*ind//x         
                # if (self.iMacdUpInd(symbol,x,xInd)>=0) and (self.iMacdUpInd(symbol,x,xInd)<xInd+12):
                    # if (iPreMacdCount>=iPreFixedMacdCount):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(x)+")分钟MACD金叉向上!!")
                        # return False  
                        
            # for x in [y for y in arr_PeriodI if (y>=60 and y<=240)]:
                # xInd=period*ind//x
                # y=self.iKdjUpInd(symbol,x,xInd)
                # if ((y>0) and ((y-xInd)<12)):
                    # isOk=False    
                    
            # for x in [y for y in arr_PeriodI if (y>=240 and y<=1440)]:
                # xInd=period*ind//x                
                # if (self.iRsiUpInd(symbol, x, xInd ,14,30,40)>0) and (self.iRsiUpInd(symbol, x, xInd ,14,30,40)<xInd+6):
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超卖!!")
                    # return False 
                    
                # if (self.isRsiUp(symbol, x, xInd,9,14,30)):
                    # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(240)+")分钟已经超卖!!")
                    # return False 

            for u in [y for y in arr_PeriodI if (y>=240 and y<=1440)]:
                xInd=period*ind//u
                if (self.isTouchedBottomBand(symbol,u,xInd)):
                    return False
                    
                if (self.isBandGoUp(symbol,u,xInd)):
                    return False   
 
            for u in [t for t in arr_PeriodI if (t>=15 and t<=240)]:
                xInd=period*ind//u                                
                if (self.isQianUp(symbol,iCloseZoom,xInd,u)):
                    return False 
                   
            for u in [t for t in arr_PeriodI if (t==period or t==240)]:
                xInd=period*ind//u       
                for method in arr_MaMethod:
                    if (self.isSuperMaUp(symbol,30,60,100,u,method,xInd,12)):
                        self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        return False
                        
                    if (self.isSuperMaUp(symbol,21,55,89,u,method,xInd,12)):
                        self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        return False

                    # if self.isPowerMaUp(symbol,30,60,100,u,method,xInd,12):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        # return False
                        
                    # if self.isPowerMaUp(symbol,21,55,89,u,method,xInd,12):
                        # self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(u)+")分钟均线趋势向下!!")
                        # return False
                        
            xInd=period*ind//15
            if (not self.isPosReady(symbol,15,type,dect_type,xInd)) and (dect_type=="real"):
                self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:15分钟顶部位置没准备好!!")
                return False  

            for x in [y for y in arr_PeriodI if y<240]:
                xInd=period*ind//x
                if self.isSpecAction(symbol,x,24,xInd):
                    self.print_comment(symbol,dect_type,"标的:("+symbol+"),周期:("+str(period)+")分钟特殊波动!!")
                    return False  
                
        return isOk 
                                                
    def refArrayInd(self,arr,value):
        ind=-1
        for i in range(1,len(arr)):
            if value==arr[i-1]:
                ind=i-1
                break
        return ind    
        
    def iTime(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,0]
        
    def iOpen(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,1]

    def iHigh(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,2]
        
    def iLow(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,3]
        
    def iClose(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,4]

    def iVoiume(self,symbol,period,k):
        global dict_symbol

        return dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1].iloc[k,5]

    def isPosReady(self,symbol,period,type,dect_type,ind):            
        rst=True
        if type=="BUY":
            low=self.iLow(symbol,period,self.getLowInd(symbol,iPosReadyCount,ind,period))
            close=self.iClose(symbol,period,ind)
            avg=self.getAvgSpace(symbol,period,iPosReadyCount,ind)/3
            if close>low+avg:
                rst=False

        if type=="SELL":
            high=self.iHigh(symbol,period,self.getHighInd(symbol,iPosReadyCount,ind,period))
            close=self.iClose(symbol,period,ind)
            avg=self.getAvgSpace(symbol,period,iPosReadyCount,ind)/3
            if close<high-avg:
                rst=False

        return rst   
        
    def iMA(self,symbol, period, iPeriodCnt,iIndX,iModel, PRICE_CLOSE):
        total_price = 0
        ma_price = 0

        if iModel == 'MODE_LWMA':
            i_power_ind = 1
            c = 0
            for i in range(iIndX, iIndX + iPeriodCnt):
                total_price += i_power_ind * self.iClose(symbol, period, i)
                c += i_power_ind
                i_power_ind += 1
            ma_price = total_price / c

        elif iModel == 'MODE_EMA':
            i_power_ind = 0
            c = 0
            for i in range(iIndX, iIndX + iPeriodCnt):
                total_price += self.iClose(symbol, period, i) * (iPeriodCnt - i_power_ind)
                c += i_power_ind
                i_power_ind += 1
            ma_price = total_price / c            

        elif iModel == 'MODE_SMA':
            for i in range(iIndX, iIndX + iPeriodCnt):
                total_price += self.iClose(symbol, period, i)
            ma_price = total_price / iPeriodCnt

        elif iModel == 'MODE_SMMA':
            i_power_ind = 0
            for i in range(iIndX, iIndX + iPeriodCnt):
                total_price += self.iClose(symbol, period, i)
                i_power_ind += 1
            ma_price = (total_price - total_price / iPeriodCnt + self.iClose(symbol, period, iIndX)) / iPeriodCnt

        return ma_price

    def getRecentInd(self,symbol,period,time,frmInd):
        ind=0
        while self.iTime(symbol,period,ind+frmInd)>time:
            ind += 1
        return ind
        
    def getHighInd(self,symbol, iZoom, iInd, period):
        iInd1 = 0
        d_high = 0  # 使用负无穷大来确保第一个if条件为真，从而初始化iInd1和d_high

        for i in range(iInd, iZoom + iInd):
            current_high = self.iHigh(symbol, period, i)
            if d_high < current_high:
                iInd1 = i
                d_high = current_high

        # 循环直到iInd1小于iZoom + iInd - 1或者self.iHigh(symbol, period, iInd1 + 1)小于等于self.iHigh(symbol, period, iInd1)
        while iInd1 >= iZoom + iInd - 1 and self.iHigh(symbol, period, iInd1 + 1) > self.iHigh(symbol, period, iInd1):
            iInd1 += 1

        return iInd1
        
    def getRefHighInd(self,symbol, iZoom, iInd, period, irefLimit):
        i_ref_ind = iInd
        x=1
        while x<=irefLimit:
            if self.iHigh(symbol,period,iInd+x) >self.iHigh(symbol,period,i_ref_ind):     
                i_ref_ind=iInd+x
            x=x+1
        if i_ref_ind>=iInd+irefLimit:        
            while self.iHigh(symbol, period, i_ref_ind+1)>self.iHigh(symbol, period, i_ref_ind):
                i_ref_ind=i_ref_ind+1
        return i_ref_ind

    def getRefRsiHighInd(self,symbol, iZoom, iInd, period, irefLimit,iCalCount):
        i_ref_ind = iInd
        x=1
        while x<=irefLimit:
            if self.iRSI(symbol, period, iCalCount, iInd+x)>self.iRSI(symbol, period, iCalCount, i_ref_ind):     
                i_ref_ind=iInd+x
            x=x+1
        if i_ref_ind>=iInd+irefLimit:        
            while self.iRSI(symbol, period, iCalCount, i_ref_ind+1)>self.iRSI(symbol, period, iCalCount, i_ref_ind):
                i_ref_ind=i_ref_ind+1
        return i_ref_ind
        
    def getLowInd(self,symbol, iZoom, iInd, period):
        iInd1 = 0
        d_low = 0  # 使用负无穷大来确保第一个if条件为真，从而初始化iInd1和d_high

        for i in range(iInd, iZoom + iInd):
            current_low = self.iLow(symbol, period, i)
            if d_low > current_low or d_low == 0:
                iInd1 = i
                d_low = current_low

        # 循环直到iInd1小于iZoom + iInd - 1或者self.iHigh(symbol, period, iInd1 + 1)小于等于self.iHigh(symbol, period, iInd1)
        while iInd1 >= iZoom + iInd - 1 and self.iLow(symbol, period, iInd1 + 1) > self.iLow(symbol, period, iInd1):
            iInd1 += 1

        return iInd1
        
    def getRefLowInd(self,symbol, iZoom, iInd, period, irefLimit):
        i_ref_ind = iInd
        x=1
        while x<=irefLimit:
            if self.iLow(symbol,period,iInd+x) <self.iLow(symbol,period,i_ref_ind):     
                i_ref_ind=iInd+x
            x=x+1
        if i_ref_ind>=iInd+irefLimit:        
            while self.iLow(symbol, period, i_ref_ind+1)<self.iLow(symbol, period, i_ref_ind):
                i_ref_ind=i_ref_ind+1
        return i_ref_ind    

    def getRefRsiLowInd(self,symbol, iZoom, iInd, period, irefLimit,iCalCount):
        i_ref_ind = iInd
        x=1
        while x<=irefLimit:
            if self.iRSI(symbol, period, iCalCount, iInd+x)<self.iRSI(symbol, period, iCalCount, i_ref_ind):     
                i_ref_ind=iInd+x
            x=x+1
            
        if i_ref_ind>=iInd+irefLimit:        
            while self.iRSI(symbol, period, iCalCount, i_ref_ind+1)<self.iRSI(symbol, period, iCalCount, i_ref_ind):
                i_ref_ind=i_ref_ind+1
                
        return i_ref_ind
        
    def getCloseHighRate(self,s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate):
        i_count = 0
        d_high = self.iHigh(s_symbol, i_period, self.getRefHighInd(s_symbol, i_zoom, i_ind, i_period,i_zoom))
        
        i_close_height = self.getAvgSpace(s_symbol,i_period,i_zoom, i_ind)
        
        for i in range(i_ind, i_zoom + i_ind):
            if (self.iHigh(s_symbol, i_period, i)+ d_scale * d_rate > d_high) and (self.iHigh(s_symbol, i_period, i) - self.iLow(s_symbol, i_period, i) >= i_close_height):
                i_count += 1
        
        return i_count / i_zoom

    def getCloseHighRateId(self,s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate, d_cmp_rate):
        i_rst_id = -1
        if self.getCloseHighRate(s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate) >= d_cmp_rate:
            d_high = self.iHigh(s_symbol, i_period, self.getRefHighInd(s_symbol, i_zoom, i_ind, i_period,i_zoom))
            i_ind2 = 0
            for i in range(i_ind, i_zoom + i_ind):
                if self.iHigh(s_symbol, i_period, i)+d_scale * d_rate > d_high:
                    if i_ind2 == 1:
                        i_rst_id = i
                        break
                    i_ind2 += 1
        return i_rst_id
    
    def getCloseLowRate(self,s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate):
        i_count = 0
        d_low = self.iLow(s_symbol, i_period, self.getRefLowInd(s_symbol, i_zoom, i_ind, i_period,i_zoom))
        
        i_close_height = self.getAvgSpace(s_symbol,i_period,i_zoom, i_ind)
        
        for i in range(i_ind, i_zoom + i_ind):
            if (self.iLow(s_symbol, i_period, i)- d_scale * d_rate< d_low) and (self.iHigh(s_symbol, i_period, i) - self.iLow(s_symbol, i_period, i) >= i_close_height):
                i_count += 1
        
        return i_count / i_zoom

    def getCloseLowRateId(self,s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate, d_cmp_rate):
        i_rst_id = -1
        if self.getCloseLowRate(s_symbol, i_period, i_zoom, i_ind, d_scale, d_rate) >= d_cmp_rate:
            d_low = self.iLow(s_symbol, i_period, self.getRefLowInd(s_symbol, i_zoom, i_ind, i_period,i_zoom))
            i_ind2 = 0
            for i in range(i_ind, i_zoom + i_ind):
                if self.iLow(s_symbol, i_period, i)-d_scale * d_rate < d_low:
                    if i_ind2 == 1:
                        i_rst_id = i
                        break
                    i_ind2 += 1
        return i_rst_id

    def getAvgSpace(self,symbol,period,refCount,ind):
        scale=0
        x=0
        try:
            while x<refCount:
                scale+=self.iHigh(symbol,period,ind+x)-self.iLow(symbol,period,ind+x)
                x+=1
        except Exception as e:
            print(f"Error fetching ticker2: {symbol},{e}")
                
        return scale/refCount
        
    def isQianUp(self,symbol, iZoom, iInd, period):
        i_close_height = self.getAvgSpace(symbol,period,iZoom, iInd)/3

        lowH1Rate1= self.getCloseLowRate(symbol,period,iCloseZoom,iInd,i_close_height,1.0)   
        lowH1Rate2= self.getCloseLowRate(symbol,period,int(iCloseZoom/2),iInd,i_close_height,1.0) 
        
        bCanDeal=False
        if (lowH1Rate1>=dHoursCloseRate):  
          closeInd=self.getCloseLowRateId(symbol,period,iCloseZoom,iInd,i_close_height,1.0,dHoursCloseRate);
          if (closeInd<=iInd+12):
             if (self.getRefHighInd(symbol,iCloseZoom,iInd,period,int(iCloseZoom/2))<=iInd+6):
                ##if (self.exitContinueUp(symbol,iInd,closeInd+iInd,period)):
                bCanDeal=True

        return bCanDeal
       
    def isQianDown(self,symbol, iZoom, iInd, period):
        i_close_height = self.getAvgSpace(symbol,period,iZoom, iInd)/3

        highH1Rate1= self.getCloseHighRate(symbol,period,iCloseZoom,iInd,i_close_height,1.0)   
        highH1Rate2= self.getCloseHighRate(symbol,period,int(iCloseZoom/2),iInd,i_close_height,1.0) 
        
        bCanDeal=False
        if (highH1Rate1>=dHoursCloseRate):  
          closeInd=self.getCloseHighRateId(symbol,period,iCloseZoom,iInd,i_close_height,1.0,dHoursCloseRate);
          if (closeInd<=iInd+12):
             if (self.getRefLowInd(symbol,iCloseZoom,iInd,period,int(iCloseZoom/2))<=iInd+6):
                ##if (self.exitContinueDown(symbol,iInd,closeInd+iInd,period)):
                bCanDeal=True

        return bCanDeal
    
    def getMaGradient(self,symbol,period,iPeriodInd,iInd,iMaMethod):
        return (self.iMA(symbol,period,iPeriodInd,iInd,iMaMethod,PRICE_CLOSE)-self.iMA(symbol,period,iPeriodInd,iInd+1,iMaMethod,PRICE_CLOSE))

    def isContinueTrendUp(self,s_symbol, iZoom, i_ind,i_period,iRefCount,iRefContinueCount):
        return False
        ind = i_ind
        i_count = 0
        iHigh_ref = self.getRefHighInd(s_symbol, iZoom, i_ind, i_period,iZoom)
        d_high_ref = self.iHigh(s_symbol, i_period, iHigh_ref)
        iLow_ref = self.getRefLowInd(s_symbol, iZoom, i_ind, i_period,iZoom)
        d_low_ref = self.iLow(s_symbol, i_period, iLow_ref)

        while ind < iZoom + i_ind:
            iHigh1 = self.getRefHighInd(s_symbol,iZoom, ind, i_period,iRefCount)
            d_high1 = self.iHigh(s_symbol, i_period, iHigh1)
            iLow1 = self.getRefLowInd(s_symbol,iZoom, iHigh1+1, i_period,iRefCount)
            d_low1 = self.iLow(s_symbol, i_period, iLow1)

            iHigh2 = self.getRefHighInd(s_symbol,iZoom, iLow1+1, i_period,iRefCount)
            d_high2 = self.iHigh(s_symbol, i_period, iHigh2)
            iLow2 = self.getRefLowInd(s_symbol,iZoom, iHigh2+1, i_period,iRefCount)
            d_low2 = self.iLow(s_symbol, i_period, iLow2)

            if (d_low1 >= d_low2) and (d_low1 >= d_low_ref) and (d_high1 >= d_high2) and (d_high1 <= d_high_ref):
                ind = iLow2+1
                i_count += 1
            else:
                return False
        return i_count >= 1
        
    def iContinueUpFromBottomByLowId(self,sSymbol, iZoom, iInd, period, iRefCount,iRefContinueCount):
        iHighInd = iInd
        iTempInd = iInd
        iLowInd = iInd
        isHighUp = False
        isLowUp = False
        iTimes = 0
        iRstId2 = 0
        
        # 由于没有提供iHigh的具体实现，这里假设它是一个可用的函数
        for i in range(iInd, iZoom + iInd):
            if self.iHigh(sSymbol,period,i) <= self.iHigh(sSymbol,period,iHighInd):
                if i < iRefCount:
                    iHighInd = i
                # 假设getHighInd是一个可用的函数
                if self.getRefHighInd(sSymbol,iContinueCount,i,period,iRefCount) == i:
                    iTimes += 1
                    if iTimes >= iRefContinueCount:
                        isHighUp = True
                        break
                    iHighInd = i
                    iTempInd = i
        
        iTimes = 0
        iTempInd = iInd
        for i in range(iInd, iZoom + iInd):
            if self.iLow(sSymbol,period,i) <= self.iLow(sSymbol,period,iLowInd):
                if i < iRefCount:
                    iLowInd = i
                if self.getRefLowInd(sSymbol,iContinueCount,i,period,iRefCount) == i:
                    iTimes += 1
                    if iTimes >= iRefContinueCount:
                        isLowUp = True
                        break
                    if iTimes == 1:
                        iRstId2 = i
                    iLowInd = i
                    iTempInd = i

        # if (self.getRefHighInd(sSymbol,iContinueCount,iInd,period,iRefCount)>=self.getRefLowInd(sSymbol,iContinueCount,iInd,period,iRefCount)):
            # return -1

        if (self.getRefHighInd(sSymbol,iZoom,iInd,period,iRefCount)>self.getRefLowInd(sSymbol,iZoom,iInd,period,iRefCount)):
            return -1
            
        iRstId = -1
        if isHighUp and isLowUp:
            iRstId = iRstId2
        
        return iRstId

    def isContinueTrendDown(self,s_symbol, iZoom, i_ind,i_period,iRefCount,iRefContinueCount):
        return False
        ind = i_ind
        i_count = 0
        iHigh_ref = self.getRefHighInd(s_symbol, iZoom, i_ind, i_period,iZoom)
        d_high_ref = self.iHigh(s_symbol, i_period, iHigh_ref)
        iLow_ref = self.getRefLowInd(s_symbol, iZoom, i_ind, i_period,iZoom)
        d_low_ref = self.iLow(s_symbol, i_period, iLow_ref)

        while ind < iZoom + i_ind:
            iLow1 = self.getRefLowInd(s_symbol,iZoom, ind, i_period,iRefCount)
            d_low1 = self.iLow(s_symbol, i_period, iLow1)  
            
            iHigh1 = self.getRefHighInd(s_symbol,iZoom, iLow1+1, i_period,iRefCount)
            d_high1 = self.iHigh(s_symbol, i_period, iHigh1)

            iLow2 = self.getRefLowInd(s_symbol,iZoom, iHigh1+1, i_period,iRefCount)
            d_low2 = self.iLow(s_symbol, i_period, iLow2)  
            
            iHigh2 = self.getRefHighInd(s_symbol,iZoom, iLow2+1, i_period,iRefCount)
            d_high2 = self.iHigh(s_symbol, i_period, iHigh2)            

            if (d_low1 <= d_low2) and (d_low1 >= d_low_ref) and (d_high1 <= d_high2) and (d_high1 <= d_high_ref):
                ind = iHigh2+1
                i_count += 1
            else:
                return False
        return i_count >= 1
 
    def iContinueDownFromTopByHighId(self,sSymbol, iZoom, iInd, period,iRefCount,iRefContinueCount):
        iHighInd = iInd
        iTempInd = iInd
        iLowInd = iInd
        isHighDown = False
        isLowDown = False
        iTimes = 0
        iRstId2 = 0
        
        # 假设iHigh和getHighInd是已定义的函数
        for i in range(iInd, iZoom + iInd):
            if self.iHigh(sSymbol,period,i) >= self.iHigh(sSymbol,period,iHighInd):
                if i < iRefCount:
                    iHighInd = i

                if self.getRefHighInd(sSymbol,iContinueCount,i,period,iRefCount) == i:
                    iTimes += 1
                    if iTimes >= iRefContinueCount:
                        isHighDown = True
                        break
                    if iTimes == 1:
                        iRstId2 = i
                    iHighInd = i
                    iTempInd = i
        
        iTimes = 0
        iTempInd = iInd
        for i in range(iInd, iZoom + iInd):
            if self.iLow(sSymbol,period,i) >= self.iLow(sSymbol,period,iLowInd):
                if i < iRefCount:
                    iLowInd = i
                
                if self.getRefLowInd(sSymbol,iContinueCount,i,period,iRefCount) == i:
                    iTimes += 1
                    if iTimes >= iRefContinueCount:
                        isLowDown = True
                        break
                    iLowInd = i
                    iTempInd = i

        
        # if (self.getRefHighInd(sSymbol,iContinueCount,iInd,period,iRefCount)<=self.getRefLowInd(sSymbol,iContinueCount,iInd,period,iRefCount)):
            # return -1

        if (self.getRefLowInd(sSymbol,iZoom,iInd,period,iRefCount)>self.getRefHighInd(sSymbol,iZoom,iInd,period,iRefCount)):
            return -1
            
        iRstId = -1
        
        if isHighDown and isLowDown:
            iRstId = iRstId2
        
        return iRstId
        
    def isBottomShandowLine(self,sSymbol,period,iInd):
        bRst = False
        # 第一个条件判断
        if self.iClose(sSymbol,period,iInd) >= (self.iLow(sSymbol,period,iInd) + (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * dShadowRate):
            bRst = True
        # 第二个条件判断
        if (self.iClose(sSymbol,period,iInd) >= (self.iLow(sSymbol,period,iInd) + (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * 0.4)) and (self.iClose(sSymbol,period,iInd) > self.iOpen(sSymbol,period,iInd)):
            if (self.iClose(sSymbol,period,iInd) - self.iOpen(sSymbol,period,iInd)) > (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * 0.2:
                bRst = True
        return bRst
        
    def isTopShandowLine(self,sSymbol,period,iInd):
        bRst = False
        # 第一个条件判断
        if self.iClose(sSymbol,period,iInd) <= (self.iHigh(sSymbol,period,iInd) - (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * dShadowRate):
            bRst = True
        # 第二个条件判断
        if (self.iClose(sSymbol,period,iInd) <= (self.iHigh(sSymbol,period,iInd) - (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * 0.4)) and (self.iClose(sSymbol,period,iInd) < self.iOpen(sSymbol,period,iInd)):
            if (self.iOpen(sSymbol,period,iInd) - self.iClose(sSymbol,period,iInd)) > (self.iHigh(sSymbol,period,iInd) - self.iLow(sSymbol,period,iInd)) * 0.2:
                bRst = True
        return bRst

    def isPierceAndSwallowUpP(self,sSymbol,period,iInd, dPierceAndSwallowRate):
        isPierceAndSwallowUp = False
        iPierceAndSwallowHeight = self.getAvgSpace(sSymbol,period,iContinueCount, iInd)

        if (self.getRefLowInd(sSymbol, iContinueCount, iInd, period,12)-iInd)>=15:
            return False
            
        if self.isBottomShandowLine(sSymbol,period,iInd) and \
            self.iHigh(sSymbol,period,iInd)-self.iLow(sSymbol,period,iInd) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
            isPierceAndSwallowUp = True
            return isPierceAndSwallowUp
        
        if iInd > 0:
            iInd2 = iInd - 1
            if self.isBottomShandowLine(sSymbol,period,iInd2) and \
                self.iHigh(sSymbol,period,iInd2)-self.iLow(sSymbol,period,iInd2) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
                isPierceAndSwallowUp = True
                return isPierceAndSwallowUp

        if iInd > 0:
            iInd2 = iInd - 1
            if self.isBottomShandowLine(sSymbol,period,iInd2) and \
                self.iClose(sSymbol,period,iInd2)>self.iHigh(sSymbol,period,iInd):
                isPierceAndSwallowUp = True
                return isPierceAndSwallowUp
                
        if iInd - 1 > 0:
            iInd2 = iInd - 2
            if self.isBottomShandowLine(sSymbol,period,iInd2) and \
                self.iHigh(sSymbol,period,iInd2)-self.iLow(sSymbol,period,iInd2) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
                isPierceAndSwallowUp = True
                return isPierceAndSwallowUp
    
        return isPierceAndSwallowUp

    def isPierceAndSwallowDownP(self,sSymbol,period,iInd, dPierceAndSwallowRate):
        isPierceAndSwallowDown = False
        iPierceAndSwallowHeight = self.getAvgSpace(sSymbol,period,iContinueCount, iInd)

        if (self.getRefHighInd(sSymbol, iContinueCount, iInd, period,12)-iInd)>=15:
            return False
                                
        if self.isTopShandowLine(sSymbol,period,iInd) and \
            self.iHigh(sSymbol,period,iInd)-self.iLow(sSymbol,period,iInd) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
            isPierceAndSwallowDown = True
            return isPierceAndSwallowDown

        if iInd > 0:
            iInd2 = iInd - 1
            if self.isTopShandowLine(sSymbol,period,iInd2) and \
                self.iClose(sSymbol,period,iInd2)<self.iLow(sSymbol,period,iInd):
                isPierceAndSwallowDown = True
                return isPierceAndSwallowDown
                
        if iInd > 0:
            iInd2 = iInd - 1
            if self.isTopShandowLine(sSymbol,period,iInd2) and \
                self.iHigh(sSymbol,period,iInd2)-self.iLow(sSymbol,period,iInd2) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
                isPierceAndSwallowDown = True
                return isPierceAndSwallowDown
        
        if iInd - 1 > 0:
            iInd2 = iInd - 2
            if self.isTopShandowLine(sSymbol,period,iInd2) and \
                self.iHigh(sSymbol,period,iInd2)- self.iLow(sSymbol,period,iInd2) > iPierceAndSwallowHeight * dPierceAndSwallowRate:
                isPierceAndSwallowDown = True
                return isPierceAndSwallowDown            
        
        return isPierceAndSwallowDown  
        
    def get2LineUpId(self,symbol,period, iFromPeriodInd, iToPeriodInd,iInd, iMaMethod):
        ind=-1
        ind2 = iInd        
        dMAFrom = self.iMA(symbol,period,iFromPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
        dMATo = self.iMA(symbol,period,iToPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
        if (dMAFrom > dMATo):
            while True:                
                dMAFrom = self.iMA(symbol,period,iFromPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
                dMATo = self.iMA(symbol,period,iToPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
                if (dMAFrom < dMATo):
                    ind=ind2
                    break
                ind2 +=1
        return  ind
        
    def getXLineUpId(self,symbol,period,iInd, iMaMethod):
        ind2 = iInd
        ind50 = 0
        ind1020 = 0
        dLow = self.iLow(symbol, period, ind2)
        dMA5 = self.iMA(symbol, period,iPeriodFrom, ind2,iMaMethod, PRICE_CLOSE)
        dMA10 = self.iMA(symbol, period,iPeriodTo, ind2,iMaMethod, PRICE_CLOSE)
        dMA20 = self.iMA(symbol, period,20, ind2,iMaMethod, PRICE_CLOSE)
        isPiercedUp = False

        if (dMA5 >= dMA10) and (dMA10 >= dMA20) and (self.getMaGradient(symbol, period,iPeriodTo, ind2,iMaMethod) >= 0) and (self.getMaGradient(symbol, period,20, ind2,iMaMethod) >= 0):
            while True:
                dLow = self.iClose(symbol, period, ind2)
                dMA5 = self.iMA(symbol, period,iPeriodFrom, ind2,iMaMethod, PRICE_CLOSE)
                dMA10 = self.iMA(symbol, period,iPeriodTo, ind2,iMaMethod, PRICE_CLOSE)
                dMA20 = self.iMA(symbol, period,20, ind2,iMaMethod, PRICE_CLOSE)
                if (dLow >= dMA5) and (dLow >= dMA10) and (dLow >= dMA20):
                    isPiercedUp = True
                if (dMA10 < dMA20) and (ind1020 == 0):
                    ind1020 = ind2
                if (dMA5 < dMA10) and (ind50 == 0):
                    ind50 = ind2
                if ((ind1020 > 0 and ind50 > 0) or (ind2-iInd > 100)):
                    break
                ind2 += 1

        if (not isPiercedUp) or (abs(ind50-ind1020) >= 5):
            return -1

        if ind50 <= ind1020:
            ind = ind50
        else:
            ind = ind1020

        return ind

    def getX2LineUpId(self,symbol,period,iFromPeriodInd,iToPeriodInd,iInd,iMaMethod):
        ind2 = iInd
        ind1020 = 0
        dLow = self.iLow(symbol, period, ind2)
        dMA10 = self.iMA(symbol, period,iFromPeriodInd, ind2,iMaMethod, PRICE_CLOSE)
        dMA20 = self.iMA(symbol, period,iToPeriodInd,ind2,iMaMethod, PRICE_CLOSE)
        isPiercedUp = False

        if (dMA10 >= dMA20) and (self.getMaGradient(symbol, period,iFromPeriodInd, ind2,iMaMethod) >= 0) and (self.getMaGradient(symbol, period,iToPeriodInd,ind2,iMaMethod) >= 0):
            while True:
                dLow = self.iLow(symbol, period, ind2)                
                dMA10 = self.iMA(symbol, period,iFromPeriodInd, ind2,iMaMethod, PRICE_CLOSE)
                dMA20 = self.iMA(symbol, period,iToPeriodInd,ind2,iMaMethod, PRICE_CLOSE)
                if (dLow >= dMA10) and (dLow >= dMA20):
                    isPiercedUp = True
                if (dMA10 < dMA20) and (ind1020 == 0):
                    ind1020 = ind2
                if ((ind1020 >0) or (ind2-iInd > 100)):
                    break
                ind2 += 1

        if (not isPiercedUp):
            return -1
        ind = ind1020          
        return ind
        
    def get2LineDownId(self,symbol,period,iFromPeriodInd,iToPeriodInd,iInd, iMaMethod):
        ind=-1
        ind2 = iInd
        dMAFrom = self.iMA(symbol,period,iFromPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
        dMATo = self.iMA(symbol,period,iToPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
        if (dMAFrom < dMATo):
            while True:    
                dMAFrom = self.iMA(symbol,period,iFromPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
                dMATo = self.iMA(symbol,period,iToPeriodInd, ind2, iMaMethod, PRICE_CLOSE)
                if (dMAFrom > dMATo):
                    ind=ind2
                    break
                ind2 +=1
        return  ind

    def getX2LineDownId(self,symbol,period,iFromPeriodInd,iToPeriodInd,iInd,iMaMethod):
        ind2 = iInd
        ind1020 = 0
        dHigh = self.iHigh(symbol, period, ind2)        
        dMA10 = self.iMA(symbol, period,iFromPeriodInd, ind2,iMaMethod, PRICE_CLOSE)
        dMA20 = self.iMA(symbol, period,iToPeriodInd,ind2,iMaMethod, PRICE_CLOSE)
        isSwallowedDown = False

        if (dMA10 <= dMA20) and (self.getMaGradient(symbol, period,iFromPeriodInd, ind2,iMaMethod) <= 0) and (self.getMaGradient(symbol, period,iToPeriodInd,ind2,iMaMethod) <= 0):
            while True:
                dHigh = self.iClose(symbol, period, ind2)
                dMA10 = self.iMA(symbol, period,iFromPeriodInd, ind2,iMaMethod, PRICE_CLOSE)
                dMA20 = self.iMA(symbol, period,iToPeriodInd,ind2,iMaMethod, PRICE_CLOSE) 
                if  (dHigh <= dMA10) and (dHigh <= dMA20):
                    isSwallowedDown = True           
                if (dMA10 > dMA20) and (ind1020 == 0):
                    ind1020 = ind2
                if ((ind1020 > 0) or (ind2-iInd > 100)):
                    break
                ind2 += 1

        if (not isSwallowedDown):
            return -1
        ind = ind1020
        return ind
        
    def getXLineDownId(self,symbol, period,iInd, iMaMethod):
        ind2 = iInd
        ind50 = 0
        ind1020 = 0

        dHigh = self.iHigh(symbol, period, ind2)
        dMA5 = self.iMA(symbol, period,iPeriodFrom, ind2,iMaMethod, PRICE_CLOSE)
        dMA10 = self.iMA(symbol, period,iPeriodTo, ind2,iMaMethod, PRICE_CLOSE)
        dMA20 = self.iMA(symbol, period,20, ind2,iMaMethod, PRICE_CLOSE)
        isSwallowedDown = False

        if (dMA5 <= dMA10) and (dMA10 <= dMA20) and (self.getMaGradient(symbol, period,iPeriodTo, ind2,iMaMethod) <= 0) and (self.getMaGradient(symbol, period,20, ind2,iMaMethod) <= 0):
            while True:
                dHigh = self.iHigh(symbol, period, ind2)
                dMA5 = self.iMA(symbol, period,iPeriodFrom, ind2,iMaMethod, PRICE_CLOSE)
                dMA10 = self.iMA(symbol, period,iPeriodTo, ind2,iMaMethod, PRICE_CLOSE)
                dMA20 = self.iMA(symbol, period,20, ind2,iMaMethod, PRICE_CLOSE) 
                if (dHigh <= dMA5) and (dHigh <= dMA10) and (dHigh <= dMA20):
                    isSwallowedDown = True           
                if (dMA10 > dMA20) and (ind1020 == 0):
                    ind1020 = ind2
                if (dMA5 > dMA10) and (ind50 == 0):
                    ind50 = ind2
                if ((ind1020 > 0 and ind50 > 0) or (ind2-iInd > 100)):
                    break
                ind2 += 1

        if (not isSwallowedDown) or (abs(ind50-ind1020) >= 5):
            return -1

        if ind50 <= ind1020:
            ind = ind50
        else:
            ind = ind1020

        return ind
     
    def isPowerMaUp(self,symbol,period1,period2,period3,period,iMaMethod,iInd,brokenRefKDataLimit):
        rst = False
        if (self.getMaGradient(symbol, period, period2, iInd, iMaMethod) > 0 and
            self.getMaGradient(symbol, period, period3, iInd, iMaMethod) > 0):
            if (self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE) >
                self.iMA(symbol, period, period3, iInd, iMaMethod, PRICE_CLOSE)):
                if (self.iMA(symbol, period, period1, iInd, iMaMethod, PRICE_CLOSE) >
                    self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE)):
                        if (self.getMaGradient(symbol, period, period1, iInd, iMaMethod) > 0):
                            rst = True
        return rst

    def isSuperMaUp(self,symbol,period1,period2,period3,period,iMaMethod,iInd,brokenRefKDataLimit):
        global iPeriodInd1,iPeriodInd2
        rst = False
        if (self.getMaGradient(symbol, period, period2, iInd, iMaMethod) > 0 or
            self.getMaGradient(symbol, period, period3, iInd, iMaMethod) > 0):
            if (self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE) >
                self.iMA(symbol, period, period3, iInd, iMaMethod, PRICE_CLOSE)):
                if (self.iMA(symbol, period, period1, iInd, iMaMethod, PRICE_CLOSE) >
                    self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE)):
                    if (self.getMaGradient(symbol, period, period1, iInd, iMaMethod) > 0):                           
                        z1=self.get2LineUpId(symbol,period,period1,period2,iInd,iMaMethod) 
                        z2=self.get2LineUpId(symbol,period,period2,period3,iInd,iMaMethod)                        
                        if ((z1>iInd) and (z1<=iInd+brokenRefKDataLimit)) or ((z2>iInd) and (z2<=iInd+brokenRefKDataLimit)):
                            rst = True
                            iPeriodInd1=period1
                            iPeriodInd2=period2
        return rst
        
    def isPowerMaDown(self,symbol,period1,period2,period3,period,iMaMethod,iInd,brokenRefKDataLimit):
        rst = False
        if (self.getMaGradient(symbol, period, period2, iInd, iMaMethod) < 0 and
            self.getMaGradient(symbol, period, period3, iInd, iMaMethod) < 0):
            if (self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE) <
                self.iMA(symbol, period, period3, iInd, iMaMethod, PRICE_CLOSE)):
                if (self.iMA(symbol, period, period1, iInd, iMaMethod, PRICE_CLOSE) <
                    self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE)):
                        if (self.getMaGradient(symbol, period, period1, iInd, iMaMethod) < 0):
                            rst = True
        return rst

    def isSuperMaDown(self,symbol,period1,period2,period3,period,iMaMethod,iInd,brokenRefKDataLimit):
        global iPeriodInd1,iPeriodInd2
        rst = False
        if (self.getMaGradient(symbol, period, period2, iInd, iMaMethod) < 0 or
            self.getMaGradient(symbol, period, period3, iInd, iMaMethod) < 0):
            if (self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE) <
                self.iMA(symbol, period, period3, iInd, iMaMethod, PRICE_CLOSE)):
                if (self.iMA(symbol, period, period1, iInd, iMaMethod, PRICE_CLOSE) <
                    self.iMA(symbol, period, period2, iInd, iMaMethod, PRICE_CLOSE)):
                    if (self.getMaGradient(symbol, period, period1, iInd, iMaMethod) < 0):
                        z1=self.get2LineDownId(symbol,period,period1,period2,iInd,iMaMethod)
                        z2=self.get2LineDownId(symbol,period,period2,period3,iInd,iMaMethod)    
                        if ((z1>iInd) and (z1<=iInd+brokenRefKDataLimit)) or ((z2>iInd) and (z2<=iInd+brokenRefKDataLimit)):
                            rst = True
                            iPeriodInd1=period1
                            iPeriodInd2=period2
                                
        return rst        
        
    def get_moving_average(self,symbol, timeframe, period, current_index):
        sum_ = 0
        for i in range(current_index, current_index - period, -1):
            sum_ += self.iClose(symbol, timeframe, i)
        moving_average = sum_ / period
        return moving_average

    def get_standard_deviation(self,symbol, timeframe, period, current_index):
        sum_ = 0
        moving_average = self.get_moving_average(symbol, timeframe, period, current_index)
        for i in range(current_index, current_index - period, -1):
            sum_ += math.pow(self.iClose(symbol, timeframe, i) - moving_average, 2)
        variance = sum_ / period
        standard_deviation = math.sqrt(variance)
        return standard_deviation
    def print_comment(self,symbol,run_type,comment):
        return 
        
    def iBandsA(self,symbol,timeframe,period, deviation, bands_shift, applied_price, mode, shift):
        """
        计算布林带指标
        参数:
        price: pd.Series - 价格数据（例如收盘价、开盘价等）
        period: int - 布林带的计算周期
        deviation: float - 标准差的倍数，用于计算上下轨
        bands_shift: int - 布林带的偏移量（通常为0）
        applied_price: str - 应用价格类型（例如 'close', 'open', 'high', 'low' 等）
        mode: str - 返回值模式（'upper' 返回上轨，'lower' 返回下轨，'middle' 返回中轨，'all' 返回全部）
        shift: int - 返回值的时间偏移量（例如向前偏移1个周期）
        返回:
        如果 mode='all'，返回一个包含上轨、中轨、下轨的 DataFrame；
        如果 mode='upper'，返回上轨的 Series；
        如果 mode='middle'，返回中轨的 Series；
        如果 mode='lower'，返回下轨的 Series。
        """
        price=dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,timeframe)]].iloc[:,4]
        price_data = price
        # if isinstance(price, pd.Series):
            # price_data = price
        # elif isinstance(price, pd.DataFrame):
            # price_data = price[applied_price]
        # else:
            # raise ValueError("price must be a pandas Series or DataFrame.")

        # 计算中轨（移动平均线）
        middle_band = price_data.rolling(window=period).mean()
        #print(middle_band)
        # 计算标准差
        std_dev = price_data.rolling(window=period).std()
            
        # 计算上下轨
        upper_band = middle_band + (std_dev * deviation)
        lower_band = middle_band - (std_dev * deviation)

        # 应用时间偏移量
        upper_band = upper_band.shift(shift)
        middle_band = middle_band.shift(shift)
        lower_band = lower_band.shift(shift)

        # 根据 mode 返回相应的结果
        if mode == MODE_UPPER:
            return upper_band.iloc[-1]
        elif mode == MODE_MAIN:
            #print(middle_band.iloc[-1])
            return middle_band.iloc[-1]
        elif mode == MODE_LOWER:
            return lower_band.iloc[-1]
        else:
            raise ValueError("Invalid mode. Please choose from 'upper', 'middle', 'lower', or 'all'.")

    def iBands(self, symbol, period, ma_period, deviations, bands_shift, applied_price, mode, shift):
        """优化后的布林带访问方法"""
        df = dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1]
        if mode == MODE_UPPER:
            return df['bb_upper'].iloc[shift]
        elif mode == MODE_MAIN:
            return df['bb_middle'].iloc[shift]
        elif mode == MODE_LOWER:
            return df['bb_lower'].iloc[shift]
        return 0
        
    def iBandsO(self, symbol, timeframe, period, deviation, bands_shift, applied_price, mode, shift):
        # 检查 period 和 shift 是否有效
        if period <= 0 or shift < 0:
            raise ValueError("Invalid period or shift")

        # 获取价格数据（根据 applied_price）
        price_data = []
        if applied_price == PRICE_CLOSE:
            price_data = [self.iClose(symbol, timeframe, i) for i in range(period + shift)]
        elif applied_price == PRICE_OPEN:
            price_data = [self.iOpen(symbol, timeframe, i) for i in range(period + shift)]
        # ... 其他价格类型

        if len(price_data) < period + shift:
            raise ValueError("Not enough data for the given period and shift")

        upper_band = []
        lower_band = []
        main_band = []

        for i in range(shift, shift + period):
            window = price_data[i - period + 1 : i + 1]
            moving_average = sum(window) / period
            std_dev = self.get_standard_deviation(symbol, timeframe, period, i)

            upper_band.append(moving_average + deviation * std_dev)
            lower_band.append(moving_average - deviation * std_dev)
            main_band.append(moving_average)

        # 返回对应 band 的值
        if mode == MODE_MAIN:
            return main_band[-1]  # 最新值
        elif mode == MODE_UPPER:
            return upper_band[-1]
        elif mode == MODE_LOWER:
            return lower_band[-1]
        else:
            raise ValueError("Invalid mode")
        
    def get_expma(self, symbol, period, iInd1, number):
        # 获取所有收盘价（避免多次调用 iClose）
        close_prices = dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[:,4]
        data_length = len(close_prices)
        
        # 边界检查
        if iInd1 < 0 or iInd1 >= data_length:
            iInd1 = data_length - 1
        
        # 计算EMA
        k = 2.0 / (number + 1.0)
        ema = close_prices[iInd1]  # 初始EMA = 当前收盘价
        for i in range(1, number):
            next_index = i + iInd1
            if next_index >= data_length:
                break
            ema = close_prices[next_index] * k + ema * (1 - k)
        
        return ema

    def calculate_macd(self, symbol,period,index, short_window=12, long_window=26, signal_window=9):
        #prices= dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[:,4][::-1]
        prices=self.get_prices(symbol, period, index, short_window+long_window+signal_window)
        #index=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])-index-1
        #if index < 0 or index >= len(prices):
        #    raise IndexError("指定的下标超出价格数据范围")
        #if index < long_window - 1:
        #    raise ValueError("数据不足，无法计算MACD指标")
        #print(prices)
        data = pd.Series(prices['Close'])
        short_ema = data.ewm(span=short_window, adjust=False).mean()
        long_ema = data.ewm(span=long_window, adjust=False).mean()
        
        dif = short_ema.iloc[-1] - long_ema.iloc[-1]
        
        if len(data) >= long_window + signal_window - 1:
            dif_history = []
            for i in range(long_window-1, len(data)):
                s = data[:i+1].ewm(span=short_window, adjust=False).mean().iloc[-1]
                l = data[:i+1].ewm(span=long_window, adjust=False).mean().iloc[-1]
                dif_history.append(s - l)
            
            dea = pd.Series(dif_history).ewm(span=signal_window, adjust=False).mean().iloc[-1]
        else:
            dea = 0
        
        return {
            'DIF': dif,
            'DEA': dea,         
            'MACD': (dif - dea) * 2
        }

    def iMacd(self, symbol, period, shortPeriod, longPeriod, signalPeriod, price_type, type, index):
        """优化后的MACD访问方法"""
        df = dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1]
        if type == MODE_MAIN:
            return df['macd_diff'].iloc[index]
        elif type == MODE_SIGNAL:
            return df['macd_dea'].iloc[index]
        return 0
        
    def iMacdO(self, symbol, period, shortPeriod, longPeriod, signalPeriod,price_type,type, index):
        macd_data = self.calculate_macd(symbol, period, index, shortPeriod, longPeriod, signalPeriod)
        if type == MODE_MAIN:
            return macd_data["DIF"]
        elif type == MODE_SIGNAL:
            return macd_data["DEA"]
        return 0


    def iMacdUpInd(self,symbol, period, iInd):
        global iPreMacdCount
        
        rst =-1
        xInd=iInd
        
        if (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)>self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):    
            while (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)>self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):
                xInd=xInd+1
                
                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break 
                    
            rst =xInd
            
            while (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)<=self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):
                xInd=xInd+1
                
                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break 
                    
            iPreMacdCount =xInd-rst
        return rst

    def iMacdDownInd(self,symbol, period, iInd):
        global iPreMacdCount
        
        rst =-1
        xInd=iInd
        
        if (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)<self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):    
            while (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)<self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):
                xInd=xInd+1

                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break 
                    
            rst =xInd
            
            while (self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_MAIN,xInd)>=self.iMacd(symbol,period,12,26,9,PRICE_CLOSE,MODE_SIGNAL,xInd)):
                xInd=xInd+1

                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break 
                    
            iPreMacdCount =xInd-rst 
        return rst

    def calculate_ema(self,values, period):
        multiplier = 2 / (period + 1)
        ema_values = [values[0]]
        
        for value in values[1:]:
            ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values[-1]

    def get_prices(self,symbol, period, index, total_period):
        """
        参数:
        - symbol: 交易品种代码，例如股票代码或期货合约代码。
        - period: 数据周期，例如 'D' 表示日线，'H1' 表示小时线。
        - index: 当前时间点的索引。
        - total_period: 需要获取的总数据周期长度。
        
        返回:
        - prices: 一个列表，包含指定周期内的价格数据，每条数据是一个字典，包含 'high', 'low', 'close' 等字段。
        """
        
        # 获取周期对应的索引
        period_index = self.refArrayInd(arr_PeriodI, period)
        
        # 从嵌套字典中获取数据
        try:
            prices = dict_symbol[symbol][arr_PeriodS[period_index]][::-1]
            index =len(prices)-index-1
        except KeyError:
            print(f"数据中不存在交易品种 {symbol} 或周期 {period} 的数据。")
            return []
        
        # 获取指定范围内的数据
        start_index = max(0, index - total_period + 1)
        end_index = index + 1
        selected_prices = prices[start_index:end_index]
        
        return selected_prices

    def iAtr(self,symbol,period,window,index):
        """
        手动计算ATR
        :param data: DataFrame，需包含High, Low, Close列
        :param window: ATR周期
        :return: 添加ATR列后的DataFrame        
        """
        data = self.get_prices(symbol, period, index,window*2)
        
        # 计算真实范围（True Range）
        high_low = data['High'] - data['Low']
        high_close_prev = np.abs(data['High'] - data['Close'].shift(1))
        low_close_prev = np.abs(data['Low'] - data['Close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # 计算ATR
        atr = pd.Series(index=data.index, dtype='float64')
        atr.iloc[window-1] = tr.iloc[:window].mean()  # 第一天的ATR是前period天的TR平均值
        
        for i in range(window, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (window-1) + tr.iloc[i]) / window
        
        #print(atr)
        return atr.iloc[-1]

    def iStochastic(self, symbol, period, k_period, d_period, d_sma_type, valueType, index):
        """优化后的随机指标访问方法"""
        df = dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1]
        if valueType == "MAIN":
            return df['stoch_k'].iloc[index]
        elif valueType == "SIGNAL":
            return df['stoch_d'].iloc[index]
        return 0
        
    def iStochasticO(self,symbol,period,k_period, d_period, d_sma_type,valueType,index):
        prices = self.get_prices(symbol, period, index, k_period + d_period)
        #print(prices)    
        #prices = [x for x in prices if isinstance(x, list) and len(x) > 1]    
        #print(prices)
        K = []
        D = []
        
        # 遍历每个时间点
        for i in range(len(prices)):
            # 获取当前周期内的最高价和最低价
            high = prices['High'].max()
            low = prices['Low'].min()
            current_price = prices['Close'].iloc[i]
            #print(current_price)             
            # 计算 %K（main）值
            if high != low:
                K_value = ((current_price - low) / (high - low)) * 100
            else:
                K_value = 0
            
            # 将 %K 值添加到 K 列表
            K.append(K_value)
            # 如果有足够的数据，计算 %D（signal）值
            if len(K) >= d_period:
                if d_sma_type == 'SMA':
                    D_value = sum(K[-d_period:]) / d_period
                elif d_sma_type == 'EMA':
                    D_value = self.calculate_ema(K[-d_period:], d_period)
                D.append(D_value)
                
        #print("K Value: {}, D Value: {}".format(K[-1], D[-1]))
        #print(K[-1])        
        # 返回指定下标处的 main（%K）值和 signal（%D）值
        if valueType=="MAIN":
            return K[-1]
        if valueType=="SIGNAL":
            return D[-1]
            
    def iKdjUpInd(self,symbol, period, iInd):
        rst = -1
        xInd = iInd
        
        if self.iStochastic(symbol,period,10,3,"SMA","MAIN",xInd) > self.iStochastic(symbol,period,10,3,"SMA","SIGNAL",xInd):
            while True:
                main_stoch = self.iStochastic(symbol,period,10,3,"SMA","MAIN",xInd)
                signal_stoch = self.iStochastic(symbol,period,10,3,"SMA","SIGNAL",xInd)
                
                if main_stoch <= 20 and main_stoch <= signal_stoch:
                    break
                    
                xInd += 1
                
                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break  
 
            rst = xInd
        
        return rst   

    def iKdjUpIndX(self,symbol,k,d,period, iInd):
        rst = -1
        xInd = iInd
        
        if self.iStochastic(symbol,period,k,d,"SMA","MAIN",xInd) > self.iStochastic(symbol,period,k,d,"SMA","SIGNAL",xInd):
            while True:
                main_stoch = self.iStochastic(symbol,period,k,d,"SMA","MAIN",xInd)
                signal_stoch = self.iStochastic(symbol,period,k,d,"SMA","SIGNAL",xInd)
                
                if main_stoch <= 20 and main_stoch<= signal_stoch:
                    break
                    
                xInd += 1
                
                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break  
 
            rst = xInd
        
        return rst 
        
    def iKdjDownInd(self,symbol, period, iInd):
        rst = -1
        xInd = iInd
        
        if self.iStochastic(symbol,period,10,3,"SMA","MAIN",xInd) < self.iStochastic(symbol,period,10,3,"SMA","SIGNAL",xInd):
            while True:
                main_stoch = self.iStochastic(symbol,period,10,3,"SMA","MAIN",xInd)
                signal_stoch = self.iStochastic(symbol,period,10,3,"SMA","SIGNAL",xInd)
                
                if main_stoch >= 80 and main_stoch >= signal_stoch:
                    break
                    
                xInd += 1

                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break 
                        
            rst = xInd
        
        return rst  

    def iKdjDownIndX(self,symbol,k,d,period, iInd):
        rst = -1
        xInd = iInd
        
        if self.iStochastic(symbol,period,k,d,"SMA","MAIN",xInd) < self.iStochastic(symbol,period,k,d,"SMA","SIGNAL",xInd):
            while True:
                main_stoch = self.iStochastic(symbol,period,k,d,"SMA","MAIN",xInd)
                signal_stoch = self.iStochastic(symbol,period,k,d,"SMA","SIGNAL",xInd)
                
                if main_stoch >= 80 and main_stoch >= signal_stoch:
                    break
                    
                xInd += 1
                
                if (xInd>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                    break  
 
            rst = xInd
        
        return rst 
        
    def getBandUpInd(self,symbol,period,shift):
        rst =-1
        if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift+1)):
            if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,shift)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,shift+1)):        
                if (self.iClose(symbol,period,shift)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift)):
                    x=shift
                    while (self.iClose(symbol,period,x)>=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)):
                        x+=1
                        #print(self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x))
                        if (x>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                            break                        
                        rst=x
                    #print(shift,rst,sep=",")    
        return rst

    def isBandGoUp(self,symbol,period,x):
        if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x+1)):
            if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x+1)):                
                refId1=self.getRefLowInd(symbol,iContinueCount,x,period,9)    
                if (self.isPierceAndSwallowUpP(symbol,period,refId1,1.5)):
                    refId2=self.getRefHighInd(symbol, iContinueCount,x,period,9)    
                    if not ((refId2<refId1) and (self.isPierceAndSwallowDownP(symbol,period,refId2,1.5))):
                        return True
                        
        if self.isBandClosing(symbol,period,x):
            refId1=self.getRefLowInd(symbol,iContinueCount,x,period,9)    
            if (self.isPierceAndSwallowUpP(symbol,period,refId1,1.5)):
                refId2=self.getRefHighInd(symbol, iContinueCount,x,period,9)    
                if not ((refId2<refId1) and (self.isPierceAndSwallowDownP(symbol,period,refId2,1.5))):
                    return True
                    
        return False         
        
    def isTouchedTopBand(self,symbol,period,shift):
        ind=0
        ref=6
        broken=0
        limitCnt=1
        x=shift
        if ((self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)>=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x+1)) and (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x)>=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x+1))):
            limitCnt=2
            
        while ind<ref:
            if (self.iHigh(symbol,period,shift+ind)>self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,shift+ind)):
                broken+=1 
                
            if (broken>=limitCnt):
                return True
            ind+=1                
  
        return False
        
    def getBandDownInd(self,symbol,period,shift):
        rst =-1
        if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift)<self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift+1)):
            if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,shift)<self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,shift+1)):            
                if (self.iClose(symbol,period,shift)<self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,shift)):
                    x=shift
                    while (self.iClose(symbol,period,x)<=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)):
                        x+=1
                        #print(self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x))
                        if (x>=len(dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]])):
                            break                        
                        rst=x
                    #print(shift,rst,sep=",")   
        return rst

    def isTouchedBottomBand(self,symbol,period,shift):
        ind=0
        ref=6
        broken=0
        limitCnt=1
        x=shift
        if ((self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)<=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x+1)) and (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x)<=self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x+1))):
            limitCnt=2
            
        while ind<ref:
            if (self.iLow(symbol,period,shift+ind)<self.iBands(symbol,period,20,2,0,PRICE_CLOSE,MODE_LOWER,shift+ind)):
                broken+=1 
                
            if (broken>=limitCnt):
                return True
            ind+=1                
  
        return False
        
    def isBandGoDown(self,symbol,period,x):
        if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x)<self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x+1)):
            if (self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x)<self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_MAIN,x+1)):
                refId1=self.getRefHighInd(symbol,iContinueCount,x,period,9)
                if (self.isPierceAndSwallowDownP(symbol,period,refId1,1.5)):
                    refId2=self.getRefLowInd(symbol, iContinueCount,x,period,9)    
                    if not ((refId2<refId1) and (self.isPierceAndSwallowUpP(symbol,period,refId2,1.5))):
                        return True
        
        if self.isBandClosing(symbol,period,x):
            refId1=self.getRefHighInd(symbol,iContinueCount,x,period,9)
            if (self.isPierceAndSwallowDownP(symbol,period,refId1,1.5)):
                refId2=self.getRefLowInd(symbol, iContinueCount,x,period,9)    
                if not ((refId2<refId1) and (self.isPierceAndSwallowUpP(symbol,period,refId2,1.5))):
                    return True    
                    
        return False                
    
    def isBandClosing(self,symbol,period,x):
        ind=5
        space=0
        while ind<30:
            if ((self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x+ind)-self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x+ind))>space):
                space=(self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x+ind)-self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x+ind))
            ind+=1
                
        if (space>(self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_UPPER,x)-self.iBands(symbol,period,19,3,0,PRICE_CLOSE,MODE_LOWER,x)) *2):
            return True
            
        return False                 
                        
    def iRSI(self, symbol, period, iCalCount, iInd):
        """
        优化后的RSI访问方法
        参数:
            symbol: 交易对
            period: 时间周期(如"15m")
            iCalCount: RSI计算周期(应与预处理时一致)
            iInd: K线索引位置
        """
        df = dict_symbol[symbol][arr_PeriodS[self.refArrayInd(arr_PeriodI,period)]].iloc[::-1]
        return df['rsi'].iloc[iInd]
        
    def iRSIO(self,symbol, period, iCalCount, iInd):
        """
        计算RSI指标
        
        参数:
            symbol: 标的代码
            period: 时间周期
            iCalCount: 计算周期(通常为14)
            iInd: 返回的索引位置(0为当前K线)
            
        返回:
            RSI值
        """
        # 获取价格数据，多取1个用于计算差值
        data = self.get_prices(symbol, period, iInd, iCalCount + 1)
        prices = (pd.Series(data['Close'])).values.tolist()
        #print(prices)
        if len(prices) < iCalCount + 1:
            return 0.0  # 或抛出异常
        
        # 计算价格变化
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # 分离上涨和下跌
        gains = [delta if delta > 0 else 0.0 for delta in deltas]
        losses = [-delta if delta < 0 else 0.0 for delta in deltas]
        
        # 计算初始平均值(前iCalCount个周期)
        avg_gain = sum(gains[:iCalCount]) / iCalCount
        avg_loss = sum(losses[:iCalCount]) / iCalCount
        
        # 平滑计算后续值(如果数据长度大于周期)
        for i in range(iCalCount, len(deltas)):
            avg_gain = (avg_gain * (iCalCount - 1) + gains[i]) / iCalCount
            avg_loss = (avg_loss * (iCalCount - 1) + losses[i]) / iCalCount
        
        # 计算RSI
        if avg_loss == 0:
            rsi = 100.0  # 避免除以零
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def iRsiUpInd(self,symbol, period, iInd , iCalCount,refVal1,refVal2):
        rst =-1
        if (self.iRSI(symbol, period, iCalCount, iInd)<refVal1):
            rst =iInd
            
        xInd=iInd
        while (self.iRSI(symbol, period, iCalCount, xInd)<=refVal2):
            xInd+=1
            if (self.iRSI(symbol, period, iCalCount, xInd)<refVal1):
                rst =xInd
                break
                
        return rst
        
    def isRsiUp(self,symbol, period, iInd,irefLimit, iCalCount,refVal1):
        if (self.iRSI(symbol, period, iCalCount, iInd)<refVal1):
            if self.getRefLowInd(symbol,iContinueCount, iInd, period,irefLimit) < self.getRefRsiLowInd(symbol,iContinueCount, iInd, period,irefLimit,iCalCount):
                if (self.getRefRsiLowInd(symbol,iContinueCount, iInd, period,irefLimit,iCalCount)-self.getRefLowInd(symbol,iContinueCount, iInd, period,iCalCount))>=3:
                    return True
                    
        return False      

    def iRsiDownInd(self,symbol, period, iInd,iCalCount,refVal1,refVal2):
        rst =-1
        if (self.iRSI(symbol, period, iCalCount, iInd)>refVal1):
            rst =iInd
            
        xInd=iInd
        while (self.iRSI(symbol, period, iCalCount, xInd)>=refVal2):
            xInd+=1
            if (self.iRSI(symbol, period, iCalCount, xInd)>refVal1):
                rst =xInd
                break
        return rst

    def isRsiDown(self,symbol, period, iInd,irefLimit, iCalCount,refVal1):
        if (self.iRSI(symbol, period, iCalCount, iInd)>refVal1):
            if self.getRefHighInd(symbol, iContinueCount, iInd, period,irefLimit) < self.getRefRsiHighInd(symbol, iContinueCount, iInd, period,irefLimit,iCalCount):
                if (self.getRefRsiHighInd(symbol,iContinueCount, iInd, period,irefLimit,iCalCount)-self.getRefHighInd(symbol,iContinueCount, iInd, period,irefLimit))>=3:
                    return True
                    
        return False  
        
    def get_ma_up_id(self,symbol, period, i_zoom, i_ref_chan_count, iInd, i_ma_method):
        ind = -1
        ind2 = 0
        i_chan_count = 0
        d_ma5 = self.iMA(symbol, period,5,iInd,i_ma_method, PRICE_CLOSE)
        d_ma10 = self.iMA(symbol, period,10,iInd,i_ma_method, PRICE_CLOSE)

        if d_ma5 > d_ma10:
            for x in range(0,i_zoom):
                d_ma5 = self.iMA(symbol, period, 5, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)
                d_ma10 = self.iMA(symbol, period, 10, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)

                d_ma52 = self.iMA(symbol, period, 5, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
                d_ma102 = self.iMA(symbol, period, 10, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
                #if (d_ma5 > d_ma10) and not (d_ma52 > d_ma102) and (x - ind2 > i_ref_chan_count):
                if (d_ma5 > d_ma10) and (d_ma52 <= d_ma102):
                    i_chan_count += 1
                    if ind2 == 0: 
                        ind2 = (i_zoom-x)                  

        if i_chan_count >= i_ref_chan_count:
            ind = ind2

        return ind

    def get_ma_down_id(self,symbol, period, i_zoom, i_ref_chan_count, iInd, i_ma_method):
        ind = -1
        ind2 = 0
        i_chan_count = 0
        d_ma5 = self.iMA(symbol, period,5,iInd,i_ma_method, PRICE_CLOSE)
        d_ma10 = self.iMA(symbol, period,10,iInd,i_ma_method, PRICE_CLOSE)

        if d_ma5 < d_ma10:
            for x in range(0,i_zoom):
                d_ma5 = self.iMA(symbol, period, 5, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)
                d_ma10 = self.iMA(symbol, period, 10, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)

                d_ma52 = self.iMA(symbol, period, 5, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
                d_ma102 = self.iMA(symbol, period, 10, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
                #if (d_ma5 < d_ma10) and not (d_ma52 < d_ma102) and (x - ind2 > i_ref_chan_count):
                if (d_ma5 < d_ma10) and (d_ma52 >= d_ma102):                    
                    i_chan_count += 1
                    if ind2 == 0: 
                        ind2 = (i_zoom-x)

        if i_chan_count >= i_ref_chan_count:
            ind = ind2

        return ind


    def isMacdUp(self,symbol, period, iInd,iMacdBrokenrefKDataLimit):
        iMacd_ind = self.iMacdUpInd(symbol, period, iInd)
        rst = False
        if  (iPreMacdCount>=iPreFixedMacdCount) and (iMacd_ind>0) and (iMacd_ind-iInd<=iMacdBrokenrefKDataLimit):
            # if (iPreMacdCount>=iPreFixedMacdCount) and (iMacd_ind-iInd<=iMacdBrokenrefKDataLimit):
                # for method in arr_MaMethod:    
                    # if self.get2LineDownId(symbol,period, 5, 10,iInd, method)>=iInd+6:                                                                                                
                        # return True  
                        
            if (self.iRsiUpInd(symbol, period, iInd ,14,30,40)>0) and (self.iRsiUpInd(symbol, period, iInd ,14,30,40)<iInd+9):
                return True   

            z=self.getBandUpInd(symbol,period,iInd)    
            if  (z>0) and (z-iInd)<9:
                return True
                
            z=self.iKdjUpIndX(symbol,9,3,period,iInd)    
            if  (z>0) and (z-iInd)<9:
                return True
                
        return rst

    def isMacdDown(self,symbol, period, iInd,iMacdBrokenrefKDataLimit):
        iMacd_ind = self.iMacdDownInd(symbol, period, iInd)
        rst = False
        if  (iPreMacdCount>=iPreFixedMacdCount) and (iMacd_ind>0) and (iMacd_ind-iInd<=iMacdBrokenrefKDataLimit):
            #if (iPreMacdCount>=iPreFixedMacdCount) and (iMacd_ind-iInd<=iMacdBrokenrefKDataLimit):
                # for method in arr_MaMethod:
                    # if self.get2LineUpId(symbol,period, 5, 10,iInd, method)>=iInd+6:                            
                        # return True
                        
            if (self.iRsiDownInd(symbol, period, iInd ,14,70,60)>0) and (self.iRsiDownInd(symbol, period, iInd ,14,70,60)<iInd+9):
                return True

            z=self.getBandDownInd(symbol,period,iInd)    
            if  (z>0) and (z-iInd)<9:
                return True
                
            z=self.iKdjDownIndX(symbol,9,3,period,iInd)    
            if  (z>0) and (z-iInd)<9:
                return True
                        
        return rst
        
    def get_ma_up_arr(self,symbol, period, i_zoom, i_ref_chan_count, iInd, i_ma_method):
        ind = -1
        x = 0
        i_chan_count = 0
        rstArr=[]
        #d_ma5 = self.iMA(symbol, period,5,iInd,i_ma_method, PRICE_CLOSE)
        #d_ma10 = self.iMA(symbol, period,10,iInd,i_ma_method, PRICE_CLOSE)

        while  (i_chan_count<=i_ref_chan_count):
            d_ma5 = self.iMA(symbol, period, 5, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)
            d_ma10 = self.iMA(symbol, period, 10, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)

            d_ma52 = self.iMA(symbol, period, 5, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
            d_ma102 = self.iMA(symbol, period, 10, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
            #if (d_ma5 < d_ma10) and not (d_ma52 < d_ma102) and (x - ind2 > i_ref_chan_count):
            if (d_ma5 > d_ma10) and (d_ma52 <= d_ma102):                    
                i_chan_count += 1
                rstArr.append(iInd + (i_zoom-x))    
            x+=1

        return rstArr

    def get_ma_down_arr(self,symbol, period, i_zoom, i_ref_chan_count, iInd, i_ma_method):
        ind = -1
        x = 0
        i_chan_count = 0
        rstArr=[]
        #d_ma5 = self.iMA(symbol, period,5,iInd,i_ma_method, PRICE_CLOSE)
        #d_ma10 = self.iMA(symbol, period,10,iInd,i_ma_method, PRICE_CLOSE)

        while  (i_chan_count<=i_ref_chan_count):
            d_ma5 = self.iMA(symbol, period, 5, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)
            d_ma10 = self.iMA(symbol, period, 10, iInd + (i_zoom-x), i_ma_method, PRICE_CLOSE)

            d_ma52 = self.iMA(symbol, period, 5, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
            d_ma102 = self.iMA(symbol, period, 10, iInd + (i_zoom-x) + 1, i_ma_method, PRICE_CLOSE)
            #if (d_ma5 < d_ma10) and not (d_ma52 < d_ma102) and (x - ind2 > i_ref_chan_count):
            if (d_ma5 < d_ma10) and (d_ma52 >= d_ma102):                    
                i_chan_count += 1
                rstArr.append(iInd + (i_zoom-x))    
            x+=1

        return rstArr
        
    def isTopChan(self,symbol, period, iInd,iMacdBrokenrefKDataLimit):
        t=self.getHighInd(symbol,iRefChanCount,iInd,period)
        if t>iInd+(iRefChanCount/2):
            return -1

        #iMacd_ind = self.iMacdDownInd(symbol, period, iInd)
        rst = -1
        
        if (True):#if iMacd_ind>iInd and iMacd_ind<iInd+iMacdBrokenrefKDataLimit:
            for method in arr_MaMethod:  
                arr=self.get_ma_up_arr(symbol, period, iRefChanCount,chanCount,iInd, method)
                if (len(arr)>chanCount):
                    xcnt=1
                    xarr=[]
                    frm=iRefChanCount
                    while xcnt<=chanCount:
                        xarr.append(self.iHigh(symbol,period,self.getHighInd(symbol,arr[xcnt-1]-arr[xcnt],arr[xcnt],period)))
                        if xcnt==3:
                            tmp=self.getHighInd(symbol,arr[xcnt-1]-arr[xcnt],arr[xcnt],period) 
                        frm=arr[xcnt]-iInd                            
                        xcnt+=1
                    
                    if (xarr[0]<xarr[1]) and (xarr[1]<xarr[2]):
                        if (xarr[3]<xarr[2]) or (xarr[4]<xarr[2]):         
                            rst = tmp
                            
        return rst
        
    def isBottomChan(self,symbol, period, iInd,iMacdBrokenrefKDataLimit):
        t=self.getLowInd(symbol,iRefChanCount,iInd,period)
        if t>iInd+(iRefChanCount/2):
            return -1

        #iMacd_ind = self.iMacdUpInd(symbol, period, iInd)
        rst = -1
        
        if (True):#if iMacd_ind>iInd and iMacd_ind<iInd+iMacdBrokenrefKDataLimit:
            for method in arr_MaMethod:  
                arr=self.get_ma_down_arr(symbol, period, iRefChanCount,chanCount,iInd, method)
                if (len(arr)>chanCount):
                    xcnt=1
                    xarr=[]
                    frm=iRefChanCount
                    tmp=0
                    while xcnt<=chanCount:
                        xarr.append(self.iLow(symbol,period,self.getLowInd(symbol,arr[xcnt-1]-arr[xcnt],arr[xcnt],period)))
                        if xcnt==3:
                            tmp=self.getLowInd(symbol,arr[xcnt-1]-arr[xcnt],arr[xcnt],period)
                        frm=arr[xcnt]-iInd   
                        xcnt+=1
                    
                    if (xarr[0]>xarr[1]) and (xarr[1]>xarr[2]):
                        if (xarr[3]>xarr[2]) or (xarr[4]>xarr[2]):         
                            rst = tmp
                            
        return rst
        
    def isContinueUp(self,symbol,period,ind,count):
        rst=True
        for x in range(1,count):
            if (not self.isBottomShandowLine(symbol,period,ind+x-1)):
                rst=False
            if x>1:
                if not (self.iClose(symbol,period,ind+x-1)<self.iClose(symbol,period,ind+x-1-1)):
                    rst=False
        return rst            

    def exitContinueUp(self,symbol,iFromInd,iToInd,period):
       ok=False
       ind=0
       while (ind <=iToInd):
          if (self.isContinueUp(symbol,period,ind,3)):
             ok=True
             break
             
          ind +=1
       return ok

   
    def isContinueDown(self,symbol,period,ind,count):
        rst=True
        for x in range(1,count):
            if (not self.isTopShandowLine(symbol,period,ind+x-1)):
                rst=False
            if x>1:
                if not (self.iClose(symbol,period,ind+x-1)>self.iClose(symbol,period,ind+x-1-1)):
                    rst=False
        return rst             

    def exitContinueDown(self,symbol,iFromInd,iToInd,period):
       ok=False
       ind=0
       while (ind <=iToInd):
          if (self.isContinueDown(symbol,period,ind,3)):
             ok=True
             break
             
          ind +=1
       return ok
       
    def reportDict(self,my_dict,count):
        for key, value in my_dict.items():
            if len(value)!=count:
                print(key)
                print(len(value))
                print(count)    

