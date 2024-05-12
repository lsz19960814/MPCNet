# -*-coding:utf-8-*-
import argparse
import logging
import time
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import precision_score
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from rqdata import up_file,now_file
import os
from models import get_model
from utils import (
    Logger,
    adjust_learning_rate,
    count_parameters,
    get_current_lr,
    load_checkpoint,
    mixup_criterion,
    mixup_data,
    save_checkpoint,
    get_PEM_data,
)

parser = argparse.ArgumentParser(description="PyTorch CIFAR Dataset Training")
parser.add_argument("--work-path", required=True, type=str)
parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
parser.add_argument("--need_train", action="store_true", help="need train")

args = parser.parse_args()
logger = Logger(
    log_file_name=args.work_path + "/log.txt",
    log_level=logging.DEBUG,
    logger_name="CIFAR",
).get_log()
config = None


def train(train_loader, net, criterion, optimizer, epoch, device):
    global writer

    start = time.time()
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    logger.info(" === Epoch: [{}/{}] === ".format(epoch + 1, config.epochs))

    for batch_index, (inputs, targets) in enumerate(train_loader):
        # move tensor to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        if config.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, config.mixup_alpha, device
            )
            #print('input',inputs.size())
            outputs = net(inputs)
            #print('output',output.size)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            #print('input',inputs.size())
            outputs = net(inputs)
            #print('output',outputs.size())
            #print('targets',outputs,targets)
            loss = criterion(outputs, targets)

        # zero the gradient buffers
        optimizer.zero_grad()
        # backward
        loss.backward()
        # update weight
        optimizer.step()

        # count the loss and acc
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if config.mixup:
            correct += (
                lam * predicted.eq(targets_a).sum().item()
                + (1 - lam) * predicted.eq(targets_b).sum().item()
            )
        else:
            correct += predicted.eq(targets).sum().item()

        if (batch_index + 1) % 100 == 0:
            logger.info(
                "   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
                    batch_index + 1,
                    len(train_loader),
                    train_loss / (batch_index + 1),
                    100.0 * correct / total,
                    get_current_lr(optimizer),
                )
            )

    logger.info(
        "   == step: [{:3}/{}], train loss: {:.3f} | train acc: {:6.3f}% | lr: {:.6f}".format(
            batch_index + 1,
            len(train_loader),
            train_loss / (batch_index + 1),
            100.0 * correct / total,
            get_current_lr(optimizer),
        )
    )

    end = time.time()
    logger.info("   == cost time: {:.4f}s".format(end - start))
    train_loss = train_loss / (batch_index + 1)
    train_acc = correct / total

    writer.add_scalar("train_loss", train_loss, global_step=epoch)
    writer.add_scalar("train_acc", train_acc, global_step=epoch)

    return train_loss, train_acc


def test(test_loader, net, criterion, optimizer, epoch, device):
    global best_prec, writer

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    logger.info(" === Validate ===")
    
    test_result = []
    y_true = []
    y_pre = []
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            test_result += predicted.eq(targets).cpu().numpy().tolist()
            correct += predicted.eq(targets).sum().item()
            y_true += targets.cpu().numpy().tolist()
            y_pre += predicted.cpu().numpy().tolist()
        #print('tar',targets)
        #print('pre',predicted)
    logger.info(
        "   == test loss: {:.3f} | test acc: {:6.3f}% | test 0 pred {:6.3f}% | test 1 pred {:6.3f}% | test 2 pred {:6.3f}%".format(
            test_loss / (batch_index + 1), 100.0 * correct / total, precision_score(y_true, y_pre, labels=[0], average=None)[0], precision_score(y_true, y_pre, labels=[1], average=None)[0],precision_score(y_true, y_pre, labels=[2], average=None)[0]
        )
    )
    '''
    logger.info(
        "   == test loss: {:.3f} | test acc: {:6.3f}%".format(
            test_loss / (batch_index + 1), 100.0 * correct / total
        )
    )
    '''
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar("test_loss", test_loss, global_step=epoch)
    writer.add_scalar("test_acc", test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100.0 * correct / total
    state = {
        "state_dict": net.state_dict(),
        "best_prec": best_prec,
        "last_epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }
    is_best = acc > best_prec
    save_checkpoint(state, is_best, args.work_path + "/" + config.ckpt_name)
    if is_best:
        best_prec = acc
    return test_result,y_pre

def main():
    global args, config, last_epoch, best_prec, writer
    writer = SummaryWriter(log_dir=args.work_path + "/event")

    # read config from yaml file
    with open(args.work_path + "/config.yaml") as f:
        config = yaml.load(f)
    # convert to dict
    config = EasyDict(config)
    logger.info(config)

    # define netowrk
    net = get_model(config)
    logger.info(net)
    logger.info(" == total parameters: " + str(count_parameters(net)))

    # CPU or GPU
    device = "cuda" if config.use_gpu else "cpu"
    # data parallel for multiple-GPU
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    net.to(device)

    # define loss and optimizer
    #criterion = nn.BCELoss#
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        config.lr_scheduler.base_lr,
        momentum=config.optimize.momentum,
        weight_decay=config.optimize.weight_decay,
        nesterov=config.optimize.nesterov,
    )

    # resume from a checkpoint
    last_epoch = -1
    best_prec = 0
    if args.work_path:
        ckpt_file_name = args.work_path + "/" + config.ckpt_name + ".pth.tar"
        if args.resume:
            best_prec, last_epoch = load_checkpoint(
                ckpt_file_name, net, optimizer=optimizer
            )

    # load training data, do data augmentation and get data loader
    
    if(args.need_train):
        train_loader, test_loader = get_PEM_data(_w = 20,_k = 1,xw_list =[1],config = config,Dict_need=False )#
        logger.info("            =======  Training  =======\n")
        for epoch in range(last_epoch + 1, config.epochs):
            lr = adjust_learning_rate(optimizer, epoch, config)
            writer.add_scalar("learning_rate", lr, epoch)
            train(train_loader, net, criterion, optimizer, epoch, device)
            if (
                epoch == 0
                or (epoch + 1) % config.eval_freq == 0
                or epoch == config.epochs - 1
            ):
                test_pre_right,test_pre = test(test_loader, net, criterion, optimizer, epoch, device)
                
        writer.close()
        torch.save(net.state_dict(), 'parameter.pkl')
        logger.info(
            "======== Training Finished.   best_test_acc: {:.3f}% ========".format(
                best_prec
            )
        )
    else:
        _w = 20
        _k = 1
        train_loader,test_loader, val_X,val_y,val_t,val_i = get_PEM_data(_w = _w, _k = _k, xw_list =[1], config = config, Dict_need=True)#
        net.load_state_dict(torch.load('parameter.pkl'))
        test(test_loader, net, criterion, optimizer, 0, device)
        test_pre_right,test_pre = test(test_loader, net, criterion, optimizer, 0, device)
    

if __name__ == "__main__":
    main()
