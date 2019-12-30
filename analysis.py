from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models

import time
import sys
import matplotlib
from matplotlib import pyplot as plt
import pdb
import seaborn as sns


#rom utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Analysis')
parser.add_argument('--model', default="ResNet34", type=str,
                    help='model type (default: ResNet34)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--weight_path',default="./normal_weights/200.pth")
parser.add_argument('--Bin',default=15,type=int)


def main(args):
    use_cuda = torch.cuda.is_available()
    sns.set()
    if args.seed != 0:
        torch.manual_seed(args.seed)

    # Data
    print('==> Preparing data..')


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                             shuffle=False, num_workers=8)


    print('==> Building model..')
    net = models.__dict__[args.model]()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    state = torch.load(args.weight_path)
    param = state["param"]
    net.load_state_dict(param)
    analysis(net,testloader,use_cuda)

def analysis(net,testloader,use_cuda):
    #print('\nEpoch: %d' % epoch)
    net.eval()
    B = np.array([i/args.Bin for i in range(args.Bin+1)])
    correct = np.array([0. for i in range(args.Bin)])
    total = np.array([0. for i in range(args.Bin)])
    left = np.array([i/args.Bin for i in range(args.Bin)])
    conf_ = np.array([i for i in range(args.Bin)])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = nn.functional.softmax(net(inputs),dim=1)
            conf, predicted = torch.max(outputs.data, 1)
            #pdb.set_trace()
            for b in range(inputs.size(0)):
                for n in range(args.Bin):

                    if(conf[b].data.item()>B[n] and conf[b].data.item()<=B[n+1]):
                        total[n] += 1
                        correct[n] += predicted[b].eq(targets[b].data).cpu().sum().item()
                        conf_[n] += conf[b].data.item()

    #print(total,correct)
    total[total==0] = 1.
    conf_ = conf_/total
    ece = ECE(conf_,correct/total,total,B)
    plt.bar(left,correct/total,alpha=0.7,label="Acc",color="b",width=1/args.Bin,align="edge")
    plt.bar(left,B[1:],alpha=0.3,label="Gap",color="r",width=1/args.Bin,align="edge")
    plt.title("ECE={:.2f}".format(ece))
    plt.legend()
    plt.savefig("normal_conf_B{}.jpg".format(args.Bin))

    plt.bar(left,total,width=1/args.Bin,align="edge")
    plt.title("Histgram of Confidence")
    plt.savefig("conf.jpg")
    plt.close()

def ECE(conf,acc,total,B):
    return float((np.abs((conf-acc)/total)*B[1:]/total).sum())


def test(epoch,net,criterion,testloader,use_cuda):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    return (test_loss/batch_idx, 100.*correct/total)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
