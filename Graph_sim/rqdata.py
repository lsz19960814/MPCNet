# -*- coding: utf-8 -*-
import requests
import pandas as pd
import json
import sys,time,os

now_file = os.path.abspath(os.path.dirname(__file__))
#up_file = os.path.abspath(os.path.join(os.getcwd(), ".."))
up_file = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
result_save_path = up_file+'/result/test/'

def get_finance(order_book_id, ftype=None):
    ret = {'components':None, 'error':'no'}

    if ftype == None:
        raise Exception('finance type is MUST, not OPTIONAL')

    r = requests.get('http://123.196.116.51:48705/finance/query?order_book_id=%s&type=%s' % (order_book_id, ftype))
    #r = requests.get('http://123.196.116.182:48706/finance/query?order_book_id=%s&type=%s' % (order_book_id, ftype))
    result = json.loads(r.text)
    if result['status'] == -1:
        print('result_mess',result['message'])
        return pd.DataFrame()
    else:
        return pd.DataFrame(result['components'])


'''
date open close high low vol amount settle percent flow total
'''
def history_bars(code, start=None, end=None, level='1d', fq='qfq', fmt='df', suspend=0):
    if('XSHG' not in code and 'XSHE' not in code):
        code = code.replace('SH','XSHG').replace('SZ','XSHE')
    ret = {'data':None, 'error':'no'}
    if fmt == 'df':
        ret['data'] = pd.DataFrame()
    else:
        ret['data'] = []

    if start == None:
        start = '20050101'
    if end   == None:
        end   = '30000000'

    if level not in ['1d', '1m']:
        raise Exception('level should be 1d or 1m')

    if fq not in ['bfq', 'qfq']:
        raise Exception('fq should be bfq or qfq')

    if fmt not in ['df', 'list']:
        raise Exception('fmt should be df or list')
    #r = requests.get('http://123.196.116.182:48706/history_bars/query?code=%s&start=%s&end=%s&level=%s&fq=%s&suspend=%r' % \
    r = requests.get('http://123.196.116.51:48705/history_bars/query?code=%s&start=%s&end=%s&level=%s&fq=%s&suspend=%r' % \
                     (code,
                      start,
                      end,
                      level,
                      fq,
                      suspend
                     ))
    #print(r.text)
    result = json.loads(r.text)
    if result['status'] == -1:
        ret['error'] = result['message']
    else:
        if fmt == 'df':
            bars = pd.DataFrame(result['data'])

            #bars.set_index('date', inplace=True)
            ret['data'] = bars
        else:
            ret['data'] = result['data']
    return ret

# 获取股票代码为code的date之前长度为limit的股票数据 日期边界为bound
def get_1d_data(code, date, limit = 240, bound = '20140101', fixed = True, offline = True):
    if offline:
        f = open(up_file+'/wmdata/'+code,'r',encoding='utf-8')
        data = f.readlines()[0]
        f.close()
        data = json.loads(data)
        for i in range(len(data)):
            if data[i]['date'] > bound:
                data = data[i:]
                break

        ret = pd.DataFrame()
        pos = 0 
        while (pos < len(data) and data[pos]['date'] <= date):
            pos += 1
        if pos > 0 and (data[pos-1]['date'] == date or (not fixed)):
            ret = pd.DataFrame(data[max(0, pos-limit): pos])
        return ret

    else:
        result = history_bars(code, start=bound)['data'] 
        ret = pd.DataFrame()
        pos = 0 
        while(pos < result.shape[0] and result['date'][pos] <= date):
            pos += 1
        if pos > 0 and (result['date'][pos-1] == date or (not fixed)):
            ret = result[max(0, pos-limit): pos].reset_index(drop=True)
        return ret 

if __name__ == '__main__':
    print(history_bars('000004.SZ', '20050301', '20190421')['data'])


