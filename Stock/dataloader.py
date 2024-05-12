import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon, MPG_Window_Horizon
from load_dataset import load_topo_dataset
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import uea

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def triple_data_loader(X, PI, Y, L, D, T = [], I = [], batch_size = 36, shuffle=False, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    D = np.array(D)
    x2 = torch.from_numpy(D.transpose(0,2,1))
    x2 = x2.type(torch.FloatTensor)

    if(len(I)>0):
        X, PI, Y, L, T, I = TensorFloat(X), TensorFloat(PI), TensorFloat(Y), TensorFloat(L), TensorFloat(T), TensorFloat(I)
        data = torch.utils.data.TensorDataset(X, PI, Y, L, x2, T, I)
    else:
        X, PI, Y, L = TensorFloat(X), TensorFloat(PI), TensorFloat(Y), TensorFloat(L) 

        data = torch.utils.data.TensorDataset(X, PI, Y, L, x2)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_topo_dataloader(args, data_type, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw dataset
    data,label,test_X,test_y,test_t,test_i,all_t,all_i = uea.get_stock_dataset()
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    topo_data = load_topo_dataset(data_type,data.shape[0])[0:data.shape[0],:,:]
    topo_test_data = topo_data[0:test_X.shape[0],:,:]

    single = False
    x_tra, y_tra, l_tra = Add_Window_Horizon(data, label, args.lag, args.horizon, single)
    topo_tra = MPG_Window_Horizon(topo_data, args.lag, args.horizon, single)
    x_test, y_test, l_test = Add_Window_Horizon(test_X, test_y, args.lag, args.horizon, single)
    topo_test = MPG_Window_Horizon(topo_test_data, args.lag, args.horizon, single)
    
    print('Train: x, MPG, y , label , ori_data ->', x_tra.shape, topo_tra.shape, y_tra.shape, l_tra.shape, data[args.lag+args.horizon-1:].shape)
    print('Val: x, MPG, y, label , ori_data ->', x_test.shape, topo_test.shape, y_test.shape, l_test.shape, test_X[args.lag+args.horizon-1:].shape)
    
    ##############get triple dataloader######################
    train_dataloader = triple_data_loader(x_tra, topo_tra, y_tra, l_tra, data[args.lag+args.horizon-1:], batch_size = args.batch_size, shuffle=False, drop_last=True)
    if len(x_test) == 0:
        test_dataloader = None
    else:
        test_dataloader = triple_data_loader(x_test, topo_test, y_test, l_test, test_X[args.lag+args.horizon-1:], test_t[args.lag+args.horizon-1:], test_i[args.lag+args.horizon-1:], batch_size = args.batch_size, shuffle=False, drop_last=True)
   

    return train_dataloader, test_dataloader, all_t,all_i,scaler
