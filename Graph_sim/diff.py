# -*- coding: utf-8 -*-
# __author__ = 'Li sz'

# 本脚本的功能为对时间序列进行任意阶数（包括分数阶）的差分
from scipy.special import gamma
import numpy as np
import pandas as pd
#from rpy2 import robjects
#from rpy2.robjects import r
#from rpy2.robjects.packages import importr
#import statsmodels.tsa.stattools as ts
#%matplotlib inline
#import matplotlib.pyplot as plt

#fracdiff=importr('fracdiff')


def sita(k,d):#计算系数
    return(gamma(k-d)/(gamma(-d)*gamma(k+1)))

def get_sitas(d):#得到系数序列
    N = 10000 #分数阶差分中求和的项数
    sitas = np.zeros(N+1)
    for i in range(N):
        sitas[i+1] = sita(i,d)
    return sitas

def fra_diff(x,d):
    n=len(x)
    y = np.zeros(n) #存放自定义的分数差分结果
    temp = 0
    sitas = get_sitas(d)
    for i in range(n):
        temp = i
        if i >= n:
            temp = n
        for j in range(temp):
            y[i] += sitas[j]*x[i-j]
    return y

def R_fra_diff(x,d):#R语言中的fracdiff包里的分数阶差分，用python的rpy2包借调的R中的函数
    #fracdiff=importr('fracdiff')
    rvector = robjects.FloatVector(x)#先要将python中的list转成R中的vector
    return(list(fracdiff.diffseries(rvector,d)))

def get_max_like_d(x):#得到最大似然的分数阶差分
    rvector = robjects.FloatVector(x)#先要将python中的list转成R中的vector
    return(list(r['coef'](fracdiff.fracdiff(rvector)))[0])#得到其中的d

#def 

def diffseris(x,d):
    if(d<0):
        print("差分阶数不能为负数")
        return x
    elif(isinstance(d,int)):
        x_seris = pd.Series(x)
        while(d!=0):
            x_seris = x_seris.diff()
            d-=1
        return(x_seris.tolist())#这里要将seris转成list，不然会出现dataframe里面全是NAN
    else:
        #return(fra_diff(x,d))
        return(R_fra_diff(x,d))
    
def data_frame_diff(f,d=0.5):
    #print 'd in diff',d
    data_frame = f
    row_name_list = data_frame.columns.values.tolist()
    for row_name in row_name_list:
        row_data = data_frame[row_name].tolist()
        #print row_data
        data_frame[row_name] = diffseris(row_data,d)
        #print diffseris(row_data,d)
    return data_frame
'''
def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas import plotting

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_style'), '_get_standard_colors')(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)])
        ax_new.set_ylabel(ylabel=cols[n])

        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    return ax
if __name__ == "__main__":
    
    data = pd.read_csv('/Users/wode/Documents/my_project/framework/trade/data/000001.SZ.csv')
    ltm = 60
    for j in range(1,len(data.columns)):
        index_name = [data.columns[j]]
        #index_name = data.columns[j]
        d_fra={}
        for _index in index_name:
            d_fra[_index] = []
        for i in range(len(data)-ltm):
            data_sup = data[i:i+ltm]
            for _index in index_name:
                #print(d_fra)
                #print(get_max_like_d(data_sup[_index]))
                d_fra[_index].append(get_max_like_d(data_sup[_index]))
        a = pd.DataFrame(data=d_fra,columns=index_name)
        plot_multi(a)
        plt.show()
        plt.close()
'''