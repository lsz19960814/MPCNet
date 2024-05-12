# -*- coding: utf-8 -*-
# __author__ = 'Li sz'

# 本脚本功能为特征计算
# 输入数据为pandas里的数据框 基础为日期，高开低收，股票代码（待定），输出为日期，特征，股票代码的数据框

# 特征有simple 10-days moving average，weighted 10-days moving average，momentum (n = 10)，
# stochastic K (n=10)，stochastic D (n=10)，relative strength index (RSI, n=10)，
# moving average convergence divergence (macd, ema12, ema26)，Larry William's R (n = 10)，
# A/D(Accumulation/Distribution) Oscillator，Commodity Channel Index (CCI, n=10)


import pandas as pd 
import numpy as np 
import stockstats
import tushare as ts
#from sklearn.preprocessing import Imputer

back_day = 10

def stats_init(stock):
    stockStat = stockstats.StockDataFrame.retype(stock)
    #print("init finish .")
    return stockStat

# 计算简单移动平均，可能要添加长度小于span的报错
def get_sma(data,span = 10):
    if(len(data) < span):
        print("长度小于移动窗口长度")
        return data
    sma = pd.rolling_mean(data,span = span)
    return sma

# 计算指数加权移动平均
def get_wma(data,span = 10):
    if(len(data) < span):
        print("长度小于移动窗口长度")
        return data
    wma = pd.ewma(data,span = span)
    return wma

# 还有动量指标和A/D(Accumulation/Distribution) Oscillator指标未计算，macd指标的ewma的滑动天数无法自定
def get_adline(features,stock):
    fund_data = stock[['close','high','low','open','vol']]
    adline = fund_data['vol']*(2*fund_data['close']-fund_data['low']-fund_data['high'])/(fund_data['high']-fund_data['low'])
    ad_line = adline
    for i in range(1,len(adline)):
        ad_line[i-1] = sum(adline[0:i])
    features['ad_line'] = ad_line
    return features

def get_features(stock):
    stockStat = stats_init(stock)
    features = stockStat[['close_10_sma','close_10_ema','kdjk_10','kdjd_10','rsi_10','macd','wr_10'
                        ,'cci_10']]
    features['mtm_10']=stockStat[['close']]-stockStat[['close']].shift(back_day)
    #features = get_adline(features,stock)
    features = features.replace([np.inf, -np.inf], np.nan)
    #print(np.isnan(features).any(),np.isinf(features).any())
    for col in features.columns.tolist():
        #features[col] = features[col].fillna(features[col].median)
        features[col] = features[col].fillna(0)
    features['close'] = stockStat['close']
    features['open'] = stockStat['open']
    return features

if __name__ == "__main__":
    begin_time = '2017-02-01'
    end_time = '2017-11-01'
    code = "000001"
    stock = ts.get_hist_data(code, start=begin_time, end=end_time)
    stock["date"] = stock.index.values #增加日期列。
    stock = stock.sort_index(0) # 将数据按照日期排序下。
    features = get_features(stock)
    print(features) 