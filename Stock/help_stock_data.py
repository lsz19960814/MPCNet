import os
import sys
file_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(file_dir)
import argparse
import configparser
from datetime import datetime
from MPCNETs import MPCGCRNCNN as Network
from Trainer import Trainer
from TrainInits import init_seed
from TrainInits import print_model_parameters
from metrics import MAE_torch
import torch
import torch.nn as nn
from dataloader import get_topo_dataloader
import copy
import numpy as np

def get_stock_loader(data_type):
    Mode = 'train'
    DEBUG = 'True'
    DATASET = 'Stock' 
    DEVICE = 'cuda:0'
    MODEL = 'MPCNETs'
    #data_type = '2021' # 2020 or 2021 or 2022 or us for Stock dataset

    #get configuration
    config_file = '{}_{}.conf'.format(DATASET, MODEL)
    print(config_file)
    config = configparser.ConfigParser()
    config.read(file_dir+'/'+config_file)
    print(config.sections())

    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default=DATASET, type=str)
    args.add_argument('--mode', default=Mode, type=str)
    args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
    args.add_argument('--debug', default=DEBUG, type=eval)
    args.add_argument('--model', default=MODEL, type=str)
    args.add_argument('--cuda', default=True, type=bool)
    # from data, these below information could be found in .conf file
    #data
    args.add_argument('--lag', default=config['data']['lag'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--tod', default=config['data']['tod'], type=eval)
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
    #model
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--link_len', default=config['model']['link_len'], type=int)
    args.add_argument('--window_len', default=config['model']['window_len'], type=int)
    args.add_argument('--fixed_adj', default=config['model']['fixed_adj'], type=int)
    args.add_argument('--stay_cost', default=config['model']['stay_cost'], type=float)
    args.add_argument('--jump_cost', default=config['model']['jump_cost'], type=float)
    #train
    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--teacher_forcing', default=False, type=bool)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
    #test
    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    #log
    args.add_argument('--log_dir', default='./', type=str)
    args.add_argument('--log_step', default=config['log']['log_step'], type=int)
    args.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = args.parse_known_args()[0]
    init_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.device[5]))
    else:
        args.device = 'cpu'
    args.debug = False


    model = Network(args)
    model = model.to(args.device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    model.load_state_dict(torch.load(file_dir+'/experiments/Stock/'+data_type+'/best_model_2.pth'))
    #torch.save(model.state_dict(), file_dir+'/experiments/Stock/'+data_type+'/best_model_2.pth')

    train_loader, test_loader, all_t, all_i, scalar = get_topo_dataloader(args, data_type,
                                                                normalizer=args.normalizer,
                                                                tod=args.tod, dow=False,
                                                                weather=False, single=False)

    train_X = []
    train_y = []
    for batch_idx, (ori_data, MPG, target, label, use_data) in enumerate(train_loader):
        data = ori_data[..., :args.input_dim]
        
        teacher_forcing_ratio = 1.
        output = model(data, target, MPG, teacher_forcing_ratio=teacher_forcing_ratio)
        numpy_output = output.cpu().detach().numpy()
        numpy_output = np.array(numpy_output.tolist()).reshape(numpy_output.shape[0],numpy_output.shape[1],-1)
        numpy_label = label.cpu().detach().numpy()
        if(len(train_X) <= 0):
            train_X = copy.copy(numpy_output + use_data.cpu().detach().numpy())
            train_y = copy.copy(numpy_label)
        else:
            train_X = np.concatenate((train_X,numpy_output + use_data.cpu().detach().numpy()),axis = 0)
            train_y = np.concatenate((train_y,numpy_label),axis = 0)

    test_X = []
    test_y = []
    test_t = []
    test_i = []
    for batch_idx, (ori_data, MPG, target, label, use_data, use_t, use_i) in enumerate(test_loader):
        data = ori_data[..., :args.input_dim]
        
        teacher_forcing_ratio = 1.
        output = model(data, target, MPG, teacher_forcing_ratio=teacher_forcing_ratio)
        numpy_output = output.cpu().detach().numpy()
        numpy_output = np.array(numpy_output.tolist()).reshape(numpy_output.shape[0],numpy_output.shape[1],-1)
        numpy_label = label.cpu().detach().numpy()
        numpy_t = use_t.cpu().detach().numpy()
        numpy_i = use_i.cpu().detach().numpy()
        if(len(test_X) <= 0):
            test_X = copy.copy(numpy_output + use_data.cpu().detach().numpy())
            test_y = copy.copy(numpy_label)
            test_t = copy.copy(numpy_t)
            test_i = copy.copy(numpy_i)
        else:
            test_X = np.concatenate((test_X,numpy_output+use_data.cpu().detach().numpy()),axis = 0)
            test_y = np.concatenate((test_y,numpy_label),axis = 0)
            test_t = np.concatenate((test_t,numpy_t),axis = 0)
            test_i = np.concatenate((test_i,numpy_i),axis = 0)

    real_t = []
    real_i = []
    for t in test_t:
        real_t.append(all_t[int(t)])
    real_t = np.array(real_t)

    for i in test_i:
        real_i.append(all_i[int(i)])
    real_i = np.array(real_i)

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, test_t.shape, test_i.shape)
    return train_X, train_y, test_X, test_y, real_t, real_i

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = get_stock_loader()