import os
import sys
import json
import math
import numpy
import argparse
import timeit
import pickle
import torch

PWD = os.path.dirname(os.path.realpath(__file__))

def load_dataset(f):
    print('path',os.path.join(PWD, 'dataset/', '%s.pickle' % f))
    with open(os.path.join(PWD, 'dataset/', '%s.pickle' % f), 'rb') as fp:
        dataset = pickle.load(fp)
    return dataset

def get_stock_dataset():
    dataset = load_dataset('raw_xyt_T%s_yb1' % 20)
    train_X = dataset['train'][0]['x']
    train_y = dataset['train'][0]['y']
    for i in range(len(dataset['train'])):
        train_X = numpy.concatenate((train_X,dataset['train'][i]['x']),axis = 0)
        train_y = numpy.concatenate((train_y,dataset['train'][i]['y']),axis = 0)
    
    test_X = dataset['test'][0]['x']
    test_y = dataset['test'][0]['y']
    test_t = dataset['test'][0]['t']
    test_i = dataset['test'][0]['i']


    for i in range(len(dataset['test'])):
        if(len(dataset['test'][i]['x'])>0):
            test_X = numpy.concatenate((test_X,dataset['test'][i]['x']),axis = 0)
            test_y = numpy.concatenate((test_y,dataset['test'][i]['y']),axis = 0)
            test_t = numpy.concatenate((test_t,dataset['test'][i]['t']),axis = 0)
            test_i = numpy.concatenate((test_i,dataset['test'][i]['i']),axis = 0)
    
    new_t = []
    all_t = numpy.array(list(set(test_t.tolist())))
    for i in test_t:
        new_t.append(numpy.where(all_t == i)[0])
    new_t = numpy.array(new_t).reshape(-1)

    new_i = []
    all_i = numpy.array(list(set(test_i.tolist())))
    for i in test_i:
        new_i.append(numpy.where(all_i == i)[0])
    new_i = numpy.array(new_i).reshape(-1)
    
    return train_X,train_y,test_X,test_y,new_t,new_i,all_t,all_i


def fit_parameters(file, train, train_labels, test, test_labels, cuda, gpu, save_path, cluster_num,
                        save_memory=False):
    """
    Creates a classifier from the given set of parameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of parameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['in_channels'] = 1
    params['cuda'] = cuda
    params['gpu'] = gpu
    classifier.set_params(**params)
    return classifier.get_data(
        train, train_labels, test, test_labels, save_path, cluster_num, save_memory=save_memory, verbose=True
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UEA repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', required=True,
                        help='dataset name')
    parser.add_argument('--path', type=str, metavar='PATH', required=True,
                        help='path where the dataset is located')
    parser.add_argument('--save_path', type=str, metavar='PATH', required=True,
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', action='store_true',
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', required=True,
                        help='path of the file of parameters to use ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=False,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    print('parse arguments succeed !!!')
    return parser.parse_args()


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_arguments()
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    train, train_labels, test, test_labels = get_stock_dataset()#load_UEA_dataset(
        #args.path, args.dataset
    #)
    if not args.load and not args.fit_classifier:
        print('start new network training')
        classifier = fit_parameters(
            args.hyper, train, train_labels, test, test_labels, args.cuda, args.gpu, args.save_path, cluster_num,
            save_memory=False
        )
    else:
        classifier = wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        classifier.set_params(**hp_dict)
        classifier.load(os.path.join(args.save_path, args.dataset))

    if not args.load:
        if args.fit_classifier:
            classifier.fit_classifier(classifier.encode(train), train_labels)
        classifier.save(
            os.path.join(args.save_path, args.dataset)
        )
        with open(
            os.path.join(
                args.save_path, args.dataset + '_parameters.json'
            ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)

    end = timeit.default_timer()
    print("All time: ", (end- start)/60)
