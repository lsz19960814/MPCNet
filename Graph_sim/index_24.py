# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import diff
import copy
#import talib



def calculateEMA(df,name,period,d = 0): #period 为天数参数
    """计算指数移动平均"""
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    Array=np.array(df[str(name)])
    length = len(Array)
    emaArray = []
    nan_number=0
    for i in range(0,length):
        #if(Array[i].isnan):
        if(math.isnan(Array[i])):
            nan_number+=1
            emaArray.append(np.nan)
        else:
            break
    firstema = Array[nan_number]#2 * Array[nan_number] / (period + 1)
    emaArray.append(firstema)
    for i in range(nan_number+1, length):
        ema = (2 * Array[i] + (period - 1) * emaArray[-1]) / (period + 1)
        emaArray.append(ema)
    _df[str(name) + '_EMA_' + str(period)] = emaArray

    return _df

def calculateMA(df,name,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    Array = np.array(df[str(name)])
    length = len(Array)
    maArray = []
    nan_number = 0
    for i in range(0, length):
        if(math.isnan(Array[i])):
            nan_number += 1
            maArray.append(np.nan)
        else:
            break
    for i in  range(0,period-1):
        maArray.append(np.nan)
        nan_number += 1
    #print(Array.shape)
    for i in range(nan_number, length):
        ma = np.mean(Array[i-period+1:i+1])
        maArray.append(ma)
    _df[str(name) + '_MA_' + str(period)] = maArray

    return _df

def calculateSMA(df,name,period,M,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    Array = np.array(df[str(name)])
    length = len(Array)
    smaArray = []
    nan_number = 0
    for i in range(0, length):
        if(math.isnan(Array[i])):
            nan_number += 1
            smaArray.append(np.nan)
        else:
            break
    #firstema = 2 * Array[nan_number] / (period + 1)
    firstsma = Array[nan_number]
    smaArray.append(firstsma)
    for i in range(nan_number+1, length):
        sma = (M * Array[i] + (period - M) * smaArray[-1]) / (period)
        smaArray.append(sma)
    _df[str(name) + '_SMA_' + str(period)+'_' + str(M)] = smaArray

    return _df
def calculateTD(df,name,period,MA_type,percent,M=12,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    if(MA_type=='EMA'):
        df = calculateEMA(df, name, period)
        array_EMA = np.array(df[str(name) + '_EMA_' + str(period)])
        for i in range(0,len(array_EMA)):
            array_EMA[i] = array_EMA[i]*(percent/100+1)
        _df[str(name)+'_EMA_'+str(period)+'_'+str(percent)]=array_EMA
    elif(MA_type=='MA'):
        df = calculateMA(df, name, period)
        array_MA = np.array(df[str(name) + '_MA_' + str(period)])
        for i in range(0, len(array_MA)):
            array_MA[i] = array_MA[i]*(percent/100+1)
        _df[str(name)+'_MA_'+str(period)+'_'+str(percent)] = array_MA
    elif(MA_type=='SMA'):
        df = calculateSMA(df, name, period,M)
        array_SMA = np.array(
            df[str(name) + '_SMA_' + str(period)+'_' + str(M)])
        for i in range(0, len(array_SMA)):
            array_SMA[i] = array_SMA[i]*(percent/100+1)
        _df[str(name)+'_SMA_'+str(period)+'_'+str(M)+'_'+str(percent)] = array_SMA
    return _df

def calculatehistory(df,index_name = 'close',Period = 4,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    index_data = df[index_name]
    new_data = index_data.shift(Period)
    _df[index_name+'_'+'shift'+'_'+str(Period)] = np.array(new_data)
    return _df

def calculateMACD(df, shortPeriod=12, longPeriod=26, signalPeriod=9,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    #emashort = calculateEMA( closeArray, shortPeriod,[])
    df = calculateEMA(df,'close',shortPeriod)
    emashort=np.array(df['close_EMA_'+str(shortPeriod)])
    #print (len(emashort))
    #emalong = calculateEMA(closeArray,longPeriod,  [])
    df = calculateEMA(df, 'close', longPeriod)
    emalong=np.array(df['close_EMA_'+str(longPeriod)])
    #print (len(emalong))
    Dif = emashort - emalong
    df['Dif'] = Dif
    df= calculateEMA(df, 'Dif', signalPeriod)
    Dea=np.array(df['Dif_EMA_'+str(signalPeriod)])
    MACD = 2 * (Dif - Dea)
    _df['Dea'] = Dea
    _df['MACD'] = MACD
    _df['close_EMA_'+str(longPeriod)] = emalong
    _df['close_EMA_'+str(shortPeriod)] = emashort
    return _df

def calculateKDJ(df,period=9,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    lowArray = np.array(df['low'])
    highArray = np.array(df['high'])
    len_data=len(closeArray)
    RSV=[]
    K=[]
    D=[]
    J=[]
    for i in range(0,period-1):
        RSV.append(np.nan)
        K.append(np.nan)
        D.append(np.nan)
        J.append(np.nan)
    K[-1]=50
    D[-1]=50
    J[-1]=3*K[-1]-2*D[-1]
    for i in range(period-1,len_data):
        Lmin=min(lowArray[i-period+1:i+1])
        Hmax=max(highArray[i-period+1:i+1])
        if((Hmax-Lmin)==0):
            today_RSV = 0
        else:
            today_RSV = (closeArray[i]-Lmin)/(Hmax-Lmin)*100
        RSV.append(today_RSV)
        K.append(K[-1]*2/3+RSV[-1]/3)
        D.append(D[-1]*2/3+K[-1]/3)
        J.append(3*K[-1]-2*D[-1])
    _df['K']=K
    _df['D']=D
    _df['J']=J
    return _df

def calculateWR(df,period=14,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    lowArray = np.array(df['low'])
    highArray = np.array(df['high'])

    len_data = len(closeArray)
    WR=[]
    for i in range(0,period-1):
        WR.append(np.nan)

    for i in range(period-1,len_data):
        Lmin=min(lowArray[i-period+1:i+1])
        Hmax=max(highArray[i-period+1:i+1])
        today_WR=(Hmax-closeArray[i])/(Hmax-Lmin)*100
        WR.append(today_WR)
    _df['WR']=WR
    return _df

def calculateRSI(df,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    RSI = []
    for i in range(0, period-1):
        RSI.append(np.nan)
    for i in range(period-1,len(closeArray)):
        #print(closeArray[i-period+1:i+1])
        #print(closeArray[i-period:i])
        delete=closeArray[i-period+2:i+1]-closeArray[i-period+1:i]
        positive=0
        negative=0
        for j in range(0,len(delete)):
            if delete[j]>0:
                positive+=delete[j]
            else:
                negative+=-delete[j]
        RSI.append(positive/(positive+negative)*100)
    _df['RSI_'+str(period)]=RSI
    return _df

def calculateCR(df,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    lowArray = np.array(df['low'])
    highArray = np.array(df['high'])
    CR=[]
    for i in range(0, period):
        CR.append(np.nan)
    for i in range(period, len(closeArray)):
        YM=[]
        for j in range(i-period+1,i+1):
            '''
            1、M=（2C+H+L）÷4 
            2、M=（C+H+L+O）÷4 
            3、M=（C+H+L）÷3 
            4、M=（H+L）÷2 
            '''
            YM.append((2*closeArray[j-1]+highArray[j-1]+lowArray[j-1])/4)
        P1=sum(highArray[i+1-period:i+1]-YM)
        P2=sum(YM-lowArray[i+1-period:i+1])
        CR.append(P1/P2*100)
    _df['CR']=CR
    return _df

def calculateCCI(df,period=14,d = 0):#word算法错误
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    lowArray = np.array(df['low'])
    highArray = np.array(df['high'])
    TYP=(closeArray+highArray+lowArray)/3
    MA=[]
    MD=[]

    for i in range(0,period-1):
        MA.append(np.nan)
    for i in range(0,2*period-2):
        MD.append(np.nan)
    CCI=copy.copy(MD)
    for i in range(period-1,len(closeArray)):
        MA.append(np.mean(TYP[i+1-period:i+1]))
    for i in range(2*period-2,len(closeArray)):
        MD.append(np.mean(MA[i+1-period:i+1]-TYP[i+1-period:i+1]))
        CCI.append((TYP[i]-MA[i])/(MD[i]*0.015))
    #print(CCI)
    _df['CCI']=CCI
    return _df

def calculateTOWER():
    #画线观察
    pass

def calculateMTM_1(df,period=6,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    MTM=[]
    for i in range(0,period):
        MTM.append(np.nan)
    for i in range(period,len(closeArray)):
        MTM.append(closeArray[i]-closeArray[i-period])
    _df['MTM']=MTM
    return df

def calculateBOLL(df,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    MA,MD=[],[]
    MB, UP, DN = [], [], []
    for i in range(0,period-1):
        MA.append(np.nan)
    for i in range(0,2*period-2):
        MD.append(np.nan)
        UP.append(np.nan)
        DN.append(np.nan)
    for i in range(period-1,len(closeArray)):
        MA.append(np.mean(closeArray[i+1-period:i+1]))
    for i in range(2*period-2,len(closeArray)):
        sum_MA_C=0
        for j in range(i+1-period,i+1):
            sum_MA_C+=(closeArray[j]-MA[i])**2
        MD.append(np.sqrt(sum_MA_C/period))

    for i in range(0,period):
        MB.append(np.nan)

    for i in range(period,len(closeArray)):
        MB.append(MA[i-1])
    for i in range(2*period-2,len(closeArray)):
        UP.append(MB[i]+2*MD[i])
        DN.append(MB[i]-2*MD[i])
    _df['MB']=MB
    _df['UP'] = UP
    _df['DN'] = DN
    return _df

def calculateTRIX(df,period,m,d = 0):#word有错误
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])


    df=calculateEMA(df,'close',period)
    AX=np.array(df['close_EMA_'+str(period)])
    df['AX']=AX
    df=calculateEMA(df,'AX',period)
    BX=np.array(df['AX_EMA_'+str(period)])
    df['BX'] = BX
    df=calculateEMA(df,'BX',period)
    TR=np.array(df['BX_EMA_'+str(period)])

    TRIX=[np.nan]
    for i in range(1,len(closeArray)):
        TRIX.append((TR[i]-TR[i-1])/TR[i-1]*100)
    TRMA=[]
    for i in range(0,m):
        TRMA.append(np.nan)
    for i in range(m,len(closeArray)):
        TRMA.append(np.mean(TRIX[i+1-m:i+1]))
    _df['TRIX']=TRIX
    _df['TRMA']=TRMA
    return _df

def calculateOBV(df,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    volArray = np.array(df['vol'])
    OBV=[volArray[0]]
    for i in range(1,len(volArray)):
        if(volArray[i]>volArray[i-1]):
            OBV.append(OBV[-1]+volArray[i])
        elif(volArray[i]<volArray[i-1]):
            OBV.append(OBV[-1] - volArray[i])
        else:
            OBV.append(OBV[-1])
    _df['OBV']=OBV
    return _df

def calculateMIKE():
    #复杂
    pass

def calculateDMA():
    #研究指数
    pass
def calculateTAPI():
    #加权指数
    pass

def calculatePSY(df,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    PSY=[]
    for i in range(0,period):
        PSY.append(np.nan)
    for i in range(period,len(closeArray)):
        up_number=0
        for j in range(i+1-period,i+1):
            if(closeArray[j]>closeArray[j-1]):
                up_number+=1
        PSY.append(up_number/period*100)
    _df['PSY']=PSY
    return _df

def calculateAR_BR(df,period=26,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    lowArray = np.array(df['low'])
    highArray = np.array(df['high'])
    openArray=np.array(df['open'])
    AR,BR=[],[]
    for i in range(0,period-1):
        AR.append(np.nan)
        BR.append(np.nan)
    BR.append(np.nan)
    for i in range(period-1,len(closeArray)):
        AR.append(sum(highArray[i+1-period:i+1]-openArray[i+1-period:i+1])/
                  sum(openArray[i+1-period:i+1]-lowArray[i+1-period:i+1]))
    for i in range(period,len(closeArray)):
        BR.append(sum(highArray[i+1-period:i+1]-closeArray[i-period:i])/
                  sum(closeArray[i-period:i]-lowArray[i+1-period:i+1]))
    _df['AR']=AR
    _df['BR']=BR
    return _df


def calculateVR(df,period=12,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    openArray=np.array(df['open'])
    volArray = np.array(df['vol'])
    VR=[]
    for i in range(0,period-1):
        VR.append(np.nan)
    for i in range(period-1,len(closeArray)):
        UV,DV,PV=0,0,0
        for j in range(i+1-period,i+1):
            if(closeArray[j]>openArray[j]):
                UV+=volArray[j]
            elif(closeArray[j]<openArray[j]):
                DV+=volArray[j]
            else:
                PV += volArray[j]
        VR.append((UV+0.5*PV)/(DV+0.5*PV))
    _df['VR']=VR
    return _df

def calculateBIAS(df,period,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    EA,BIAS=[],[]
    for i in range(0,period-1):
        EA.append(np.nan)
    BIAS=copy.copy(EA)
    for i in range(period-1,len(closeArray)):
        EA.append(sum(closeArray[i+1-period:i+1])/period)
        BIAS.append((closeArray[i]-EA[-1])/EA[-1]*100)
    #print(len(BIAS),len(_df))
    _df['BIAS_'+str(period)]=BIAS
    return _df

def calculateROC(df, period ,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    ROC=[]
    for i in range(0,period):
        ROC.append(np.nan)
    for i in range(period,len(closeArray)):
        ROC.append((closeArray[i]-closeArray[i-period])/closeArray[i-period]*100)
    _df['ROC_'+str(period)]=ROC
    #print(_df)
    return _df

def add_nan(sar,period):
    for i in range(0,period-1):
        sar.append(np.nan)
    return sar

def calculateSAR(df, period):#4月6号新写
    closeArray = np.array(df['close'])
    highArray = np.array(df['high'])
    lowArray = np.array(df['low'])
    SAR=[]

    len_data=len(closeArray)
    num_round=(len_data-len_data%period)/period
    num_round=int(num_round)

    #T0周期SAR确定
    SAR=add_nan(SAR,period)
    min_T_old = min(lowArray[0:period])
    max_T_old = max(highArray[0:period])
    if(closeArray[2*period-1]>=closeArray[period]):
        sar_old = min_T_old
        SAR.append(sar_old)
    else:
        sar_old = max_T_old
        SAR.append(sar_old)

    if(closeArray[period-1] >=closeArray[0]):
        UP_old = 1
    else:
        UP_old = 0

    AF=0.02

    for i in range(1,num_round):
        min_T_new = min(lowArray[i*period:(i+1)*period])
        max_T_new = max(highArray[i*period:(i+1)*period])
        if(closeArray[(i+1)*period-1]>=closeArray[i*period]):
            UP_new=1
            EP = max_T_old

        else:
            UP_new=0
            EP = min_T_old

        #计算AF
        if((UP_old==1)&(UP_new==1)):
            if(max_T_new>max_T_old):
                AF+=0.02
        elif((UP_old == 0) & (UP_new == 0)):
            if(min_T_new<min_T_old):
                AF+=0.02
        else:#行情转变重新从0.02起
            AF=0.02
        if(AF>0.2):
            AF=0.02
        sar_now=sar_old+AF*(EP-sar_old)

        if(UP_new==1):
            SAR = add_nan(SAR, period)
            if((sar_now>min_T_new)|(sar_now>min_T_old)):
                SAR.append(min(min_T_new,min_T_old))
            else:
                SAR.append(sar_now)
        elif(UP_new==0):
            SAR = add_nan(SAR, period)
            if((sar_now<max_T_new)|(sar_now<max_T_old)):
                SAR.append(max(max_T_new, max_T_old))
            else:
                SAR.append(sar_now)
        sar_old=sar_now
        UP_old=UP_new
        max_T_old=max_T_new
        min_T_old=min_T_new
    SAR=add_nan(SAR,len_data%period+1)
    df['SAR']=SAR
    return df

def calculateDMI(df,period1,period2):#添加+DI -DI ADX ADXR 指标
    closeArray = np.array(df['close'])
    highArray = np.array(df['high'])
    lowArray = np.array(df['low'])
    len_data=len(closeArray)
    DM_up=[]
    DM_down=[]
    TR=[]
    DI_up=[]
    di_up = []
    DI_down=[]
    di_down=[]
    DX=[]
    DM_up.append(np.nan)
    DM_down.append(np.nan)
    TR.append(np.nan)
    di_up.append(np.nan)
    di_down.append(np.nan)

    
    for i in range(1,len_data):
        #DM_up DM_down
        DM_up.append((highArray[i]-highArray[i-1])
                     if((highArray[i]-highArray[i-1]) >= 0) else 0)
        DM_down.append((lowArray[i-1]-lowArray[i])
                       if((lowArray[i-1]-lowArray[i]) >= 0) else 0)
        if (DM_down[-1]>DM_up[-1]):
            DM_up[-1]=0
        elif(DM_up[-1] > DM_down[-1]):
            DM_down[-1]=0
        else:
            DM_up[-1] = 0
            DM_down[-1] = 0

        #TR
        a1 = abs(highArray[i]-lowArray[i])
        b1=abs(highArray[i]-closeArray[i-1])
        c1=abs(lowArray[i]-closeArray[i-1])
        TR.append(max(a1,b1,c1))

        di_up.append(DM_up[-1]/TR[-1]*100)
        di_down.append(DM_down[-1]/TR[-1]*100)
    di_up[period1]=np.mean(di_up[1:period1+1])
    di_down[period1] = np.mean(di_down[1:period1+1])
    for i in range(0,period1):
        DX.append(np.nan)
        di_up[i]=np.nan
        di_down[i] = np.nan
    df['di_up']=di_up
    df['di_down']=di_down
    df = calculateEMA(df,'di_up',period1)
    df = calculateEMA(df, 'di_down', period1)
    DI_up = np.array(df['di_up_EMA_'+str(period1)])
    DI_down = np.array(df['di_down_EMA_'+str(period1)])
    df['+DI']=DI_up
    df['-DI']=DI_down
    #DX
    for i in range(period1,len_data):
        if (abs(DI_up[i]-DI_down[i])==0):
            DX.append(0)
        else:
            DX.append(abs(DI_up[i]-DI_down[i])/(DI_up[i]+DI_down[i]))
    DX[period1+period2-1]=np.mean(DX[period1:period1+period2])
    for i in range(period1,period1+period2-1):
        DX[i]=np.nan
    df['DX']=DX
    df=calculateEMA(df,'DX',period2)
    ADX=np.array(df['DX_EMA_'+str(period2)])
    df['ADX']=ADX
    adxr=copy.copy(ADX)
    adxr[period1+2*(period2-1)] = np.mean(adxr[period1 +
                                               period2-1:period1+2*(period2-1)+1])
    for i in range(period1+period2-1,period1+2*(period2-1)):
        adxr[i] = np.nan
    df['adxr']=adxr
    df=calculateEMA(df,'adxr',period2)
    ADXR=df['adxr_EMA_'+str(period2)]
    df['ADXR']=ADXR

    return df

def calculateOBOS(df,period):
    #多只股票上涨只数
    pass

def calculateADR(df,period):
    #多只股票上涨只数
    pass

def calculateADL(df,period):
    #多只股票上涨只数
    pass

def calculateKlength(df,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    openArray = np.array(df['open'])
    _df['Klength'] = closeArray - openArray
    return _df

def calculateKUpperLength(df,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    openArray = np.array(df['open'])
    highArray = np.array(df['high'])
    KUL = []
    for _high,_close,_open in zip(highArray,closeArray,openArray):
        KUL.append(_high-np.max([_close,_open]))
    _df['KUpperlength'] = np.array(KUL)
    return _df

def calculateKLowerLength(df,d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    openArray = np.array(df['open'])
    lowArray = np.array(df['low'])
    KLL = []
    for _low,_close,_open in zip(lowArray,closeArray,openArray):
        KLL.append(np.min([_close,_open])-_low)
    _df['KLowerlength'] = np.array(KLL)
    return _df

def calculateMeanAmplitude(df, period, d = 0):
    _df = copy.copy(df)
    if(d != 0 ):
        df = diff.data_frame_diff(df,d)
    closeArray = np.array(df['close'])
    highArray = np.array(df['high'])
    lowArray = np.array(df['low'])
    MALArray = (highArray-lowArray)/closeArray
    MAL = []
    for i in range(0,period-1):
        MAL.append(np.nan)
    for i in range(period-1,len(closeArray)):
        MAL.append(MALArray[i-period+1:i+1].mean())
    _df['MeanAmplitude'] = np.array(MAL)
    return _df