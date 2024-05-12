import os
import numpy as np
import copy

path = os.path.abspath(os.path.dirname(__file__))


def load_topo_dataset(data_type, need_length = 1):
    topo_data = np.load(path + '/tda_data/Stock' + data_type + '_01_MPGrid_Euler_characteristic_betweenness_sublevel_attribute_power.npz')['arr_0']
    need_topo_data = copy.copy(topo_data)
    while(need_topo_data.shape[0] < need_length):
        need_topo_data = np.concatenate((need_topo_data,topo_data))

    return need_topo_data
    

