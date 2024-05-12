# -*-coding:utf-8-*-
import logging
import math
import os
import shutil
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from rqdata import up_file,now_file
import pickle
import copy
import torch.utils.data as Data
import random
import sys
#print(now_file)
sys.path.append(now_file+'/Stock/')
from Stock.help_stock_data import get_stock_loader

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] - [%(filename)s line:%(lineno)3d] : %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def data_augmentation(config, is_train=True):
    aug = []
    if is_train:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding=4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == "cifar10":
            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
        else:
            aug.append(
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            )

    if is_train and config.augmentation.cutout:
        # cutout
        aug.append(
            Cutout(n_holes=config.augmentation.holes, length=config.augmentation.length)
        )
    return aug


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + ".pth.tar")
    if is_best:
        shutil.copyfile(filename + ".pth.tar", filename + "_best.pth.tar")


def load_checkpoint(path, model, optimizer=None):
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            best_prec = checkpoint["best_prec"]
            last_epoch = checkpoint["last_epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=== done. also loaded optimizer from "
                + "checkpoint '{}' (epoch {}) ===".format(path, last_epoch + 1)
            )
            return best_prec, last_epoch


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset == "cifar10" or config.dataset == "cifar100"
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR10(
            root=config.data_path, train=False, download=True, transform=transform_test
        )
    else:
        trainset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=True, download=True, transform=transform_train
        )

        testset = torchvision.datasets.CIFAR100(
            root=config.data_path, train=False, download=True, transform=transform_test
        )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers
    )
    return train_loader, test_loader


def get_PEM_data(_w,_k,xw_list,config,Dict_need = False):
    
    train_X, train_y, test_X, test_y, test_t, test_i = get_stock_loader()

    train_y[train_y == -1] = 0
    for i in range(len(train_X)):
        train_X[i] = train_X[i][::-1]
    print(train_X.shape)

    train = np.array(train_X)
    #train = np.delete(train, 4, axis=2)
    y_train = np.array(train_y)
    print(np.sum(y_train == 1),len(y_train))
    print('pos train ratio',len(train),np.sum(y_train == 1)/len(y_train))
    print('neg train ratio',len(train),np.sum(y_train == 0)/len(y_train))
    
    x1 = torch.from_numpy(train)
    y1 = torch.from_numpy(y_train)

    x1 = x1.type(torch.FloatTensor)  # 转Float
    #x1 = x1.cuda()  # 转cuda

    y1 = y1.type(torch.LongTensor)  # 转Float
    #y1 = y1.cuda()  # 转cuda


    train_dataset = Data.TensorDataset(x1, y1)
    train_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers
    )

    test_y[test_y == -1] = 0
    for i in range(len(test_X)):
        test_X[i] = test_X[i][::-1]

    test = np.array(test_X)
    #test = np.delete(test, 4, axis=2)

    y_test = np.array(test_y)
    print('pos sample ratio',len(test),np.sum(y_test == 1)/len(y_test))
    print('neg sample ratio',len(test),np.sum(y_test == 0)/len(y_test))
    
    x2 = torch.from_numpy(test)
    y2 = torch.from_numpy(y_test)

    x2 = x2.type(torch.FloatTensor)  # 转Float
    #x2 = x2.cuda()  # 转cuda

    y2 = y2.type(torch.LongTensor)  # 转Float
    #y2 = y2.cuda()  # 转cuda

    test_dataset = Data.TensorDataset(x2, y2)
    test_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test_dataset,
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers
    )
    if(Dict_need):
        return train_loader,test_loader,test_X,test_y,test_t,test_i
    else:
        return train_loader,test_loader

def mixup_data(x, y, alpha, device):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def adjust_learning_rate(optimizer, epoch, config):
    lr = get_current_lr(optimizer)
    if config.lr_scheduler.type == "STEP":
        if epoch in config.lr_scheduler.lr_epochs:
            lr *= config.lr_scheduler.lr_mults
    elif config.lr_scheduler.type == "COSINE":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (1.0 + math.cos(math.pi * ratio))
            / 2.0
        )
    elif config.lr_scheduler.type == "HTD":
        ratio = epoch / config.epochs
        lr = (
            config.lr_scheduler.min_lr
            + (config.lr_scheduler.base_lr - config.lr_scheduler.min_lr)
            * (
                1.0
                - math.tanh(
                    config.lr_scheduler.lower_bound
                    + (
                        config.lr_scheduler.upper_bound
                        - config.lr_scheduler.lower_bound
                    )
                    * ratio
                )
            )
            / 2.0
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr



