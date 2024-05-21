#!/usr/bin/env python
# encoding: utf-8
import networkx as nx
from pscn import PSCN
from sklearn.model_selection import train_test_split
import numpy as np
from graph import Graph
import matplotlib.pyplot as plt
import copy
import pyunicorn.timeseries.visibility_graph as ptv
from sklearn.metrics import precision_score,accuracy_score,f1_score
#import sig_data
#import index_24
import os
import tushare as ts
from rqdata import up_file,now_file
import pandas as pd
import warnings
from Feature import get_features
import joblib
import math
import pickle
from multiprocessing import Pool
import Feature

#PWD = '/root/meta_stra_framwork/'
PWD = '/gs/home/by1809107/lsz/tamp_trick/trick/'#os.path.dirname(os.path.realpath(__file__))

warnings.filterwarnings("ignore")

def get_labels(ydata,com_price):
    if(ydata['close']>com_price):
        return 1,ydata['close']/com_price-1
    else:
        return -1,ydata['close']/com_price-1
    
def get_adj(xdata):
    G = ptv.VisibilityGraph(xdata,silence_level =2)
    adj = G.visibility_relations()
    return adj

def get_graph(adj,features):
    G = Graph() # 初始化图
    fea_matri = features.reset_index(drop=True).to_numpy()
    for i in range(0,x_w):
        G.add_vertex(i)
        G.add_one_attribute(i,list(fea_matri[i,:]))
        for j in range(0,i):
            if(adj[i,j] == 1):
                G.add_edge((i,j))
    return G

def stand(features):
    for colu_name in features:
        # 只针对输入数据进行标准化，标准化算法为: (原始数据 - 平均值) / 标准差
        # 这里每一次for循环，都拿出了1列数据，针对这一列进行标准化并覆盖原数据
        features[colu_name] = (
                (features[colu_name] - features[colu_name].mean()) /
                features[colu_name].std())
    return features

def make_one_graph(i,price_data,features,x_w):
    if(x_w>=10):
        xdata = stand(features.iloc[i-x_w:i])
    else:
        xdata = features.iloc[i-x_w:i]
    #print(len(xdata))
    x_price = price_data.iloc[i-x_w:i]
    try:
        ydata = price_data.iloc[i+y_w-1]
    except Exception as e:
        print(len(price_data),i+y_w-1)
    adj = get_adj(np.array(x_price[index_name].tolist()))
    x_G = get_graph(adj,xdata)
    if(y_k>0):
        _label = get_labels(ydata,price_data.iloc[i-y_k-1][index_name])
    else:
        _label,_t = get_labels(ydata,price_data.iloc[i-y_k-1]['close'])
    return x_G,_label,x_price[index_name].tolist(),_t 

def make_price_data(i,price_data,features,x_w):
    if(x_w>=10):
        xdata = stand(features.iloc[i-x_w:i])
    else:
        xdata = features.iloc[i-x_w:i]
    #print(len(xdata))
    try:
        ydata = price_data.iloc[i+y_w-1]
    except Exception as e:
        print(len(price_data),i+y_w-1)
    
    if(y_k>0):
        _label = get_labels(ydata,price_data.iloc[i-y_k-1][index_name])
    else:
        _label,_t = get_labels(ydata,price_data.iloc[i-y_k-1]['close'])
    return xdata,_label,_t

def load_stock(s,x_column):

    #df = pd.read_csv(os.path.join(PWD, 'new_day_data_csv', s), index_col=0)
    df = pd.read_csv(os.path.join(PWD, 'us_stock_30', s), index_col=0)
    fea = Feature.get_features(df)
    df[x_column] = fea.loc[df.index,x_column]
    df.set_index(df.index.astype('str'), inplace=True)
    df = df.fillna(method='pad')

    #计算是否为转折点
    close = df['close']
    zz_list = [0]
    for i in range(1,len(close)-1):
        if((close.iloc[i-1] > close.iloc[i]) * (close.iloc[i+1] > close.iloc[i])):
            zz_list.append(1)
        else:
            zz_list.append(0)
    zz_list.append(0)

    zz2_list = [0]


    ### 这里是为了扩充转折点，使得转折点更多
    #print()
    for i in range(1,len(zz_list)-1):
        if(zz_list[i+1] == 1 or zz_list[i] == 1):
        #if(zz_list[i-1] == 1 or zz_list[i+1] == 1 or zz_list[i] == 1):
        #if(zz_list[i-1] == 1 or zz_list[i+1] == 1 or zz_list[i] == 1 or zz_list[i-2] == 1 or zz_list[i+2] == 1):
            zz2_list.append(1)
        else:
            zz2_list.append(0)
    zz2_list.append(0)
    df['turn_point'] = zz2_list

    return df


def z_score(df):
    #return df
    return (df - df.mean()) / df.std()

def stock_sample(s,_w, _k, x_w, i):
    #print(s,i)
    df = global_df[s]
    if i not in df.index:
        return
    iloc = list(df.index).index(i) + 1  # df.iloc是前闭后开的，所有要+1才能取到这个点
    if iloc < x_w:  # 数据量不够
        return
    price_data = copy.copy(df)
    features = df.loc[:,x_column]
    pscn=PSCN(w=_w,k=_k,epochs=50,batch_size=32,verbose=2,attr_dim=atrr_len,dummy_value=np.repeat(0,atrr_len),labeling_procedure_name = 'ATS',step_max = 20)
    x,y,ATS,t= make_one_graph(iloc,price_data,features,x_w)
    tp = df.iloc[iloc-1,:]['turn_point'] #扩散转折点
    X,Y = pscn.process_data([x],[tp],[ATS])
    if np.isnan(X[0]).any():
        return


    return X[0],Y[0],s,i

def batch_by_date(input_):
    _w, _k, x_w, i = input_
    xyts = [stock_sample(f,_w, _k, x_w, i) for f in files]
    xyts = filter(lambda xyt: xyt is not None, xyts)
    try:
        xs, ys, ts,_is= zip(*xyts)
        return {'x': np.array(xs), 'y': np.array(ys),'t': np.array(ts),'i':np.array(_is)}
    except Exception as e:
        #print(xyts,e)
        return {'x': np.array([]), 'y': np.array([]), 't': np.array([]),'i':np.array([])}
def time_series_gen(_w,_k,x_w):
    #df = global_df['999999.XSHG.csv']
    df = global_df['AAPL.csv']

    ti = df['20211229':'20211231'].index  # train
    vi = df['20211229':'20211231'].index  # validation
    ti_ = df['20220101':'20221231'].index   # test

    # train
    pool = Pool(22)
    train = pool.map(batch_by_date, [(_w,_k,x_w, i) for i in ti])
    pool.close()
    pool.join()
    
    # validation
    pool = Pool(22)
    validation = pool.map(batch_by_date, [(_w,_k,x_w, i) for i in vi])
    pool.close()
    pool.join()
    # test
    pool = Pool(22)
    test = pool.map(batch_by_date, [(_w,_k,x_w, i) for i in ti_])
    pool.close()
    pool.join()
    return train, validation, test

def generate_data(_w,_k,x_w):
    train, validation, test = time_series_gen(_w,_k,x_w)
    dataset = {'train': train,
               'validation': validation,
               'test': test
               }
    with open(os.path.join(up_file, 'daydataset', 'raw_xyt_w%s_k%s_xw%s_yb1.pickle' %(_w,_k,x_w)), 'wb') as fp:
        pickle.dump(dataset, fp, protocol=2)


if __name__ == '__main__':
    x_column = ['close_10_sma','close_10_ema','kdjk_10','kdjd_10','rsi_10','macd','wr_10'
                        ,'cci_10','mtm_10']
    #global_df = {f: load_stock(f,x_column) for f in os.listdir(os.path.join(PWD, 'new_day_data_csv'))}
    global_df = {f: load_stock(f,x_column) for f in os.listdir(os.path.join(PWD, 'us_stock_30'))}
    y_w = 1
    y_k = 0
    atrr_len = 9
    index_name = 'close'
    #files = os.listdir(os.path.join(PWD, 'new_day_data_csv'))
    files = os.listdir(os.path.join(PWD, 'us_stock_30'))
    #files = set(files) - set(['999999.XSHG.csv']) #us data should delete
    w_list = [20]
    k_list = [1]
    xw_list = [40]
    for _w in w_list:
        for _k in k_list:#, 30, 40, 60]:
            for x_w in xw_list:
                generate_data(_w,_k,x_w)