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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default="ResNet34", type=str,
                    help='model type (default: ResNet34)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=0.3, type=float,
                    help='mixup interpolation coefficient (default: 1)')

parser.add_argument('--save_state_dir',default="./normal_weights/")
parser.add_argument('--milestones',default=[60,120,160])



def main(args):
    use_cuda = torch.cuda.is_available()
    print("use_cuda={}".format(use_cuda))
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    os.makedirs(args.save_state_dir, exist_ok=True)

    if args.seed != 0:
        torch.manual_seed(args.seed)

    # Data
    print('==> Preparing data..')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)

    testset = datasets.CIFAR10(root='~/data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=8)

    net = models.__dict__[args.model]()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    start = time.time()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch,net,optimizer,criterion,trainloader,use_cuda)
        test_loss, test_acc = test(epoch,net,criterion,testloader,use_cuda)
        h,m,s = time_(start,time.time(),flush=True)
        print("{}h {}m {}s".format(h,m,s))
        print("Train Loss:{:.2f} Train_Acc:{:.3f} Test Loss:{:.2f} Test Acc:{:.3f}".format(train_loss,train_acc,test_loss,test_acc),flush=True)
        train_acc_.append(train_acc)
        test_acc_.append(test_acc)
        train_loss_.append(train_loss)
        test_loss_.append(test_loss)
        plot_(train_acc_,test_acc_,train_loss_,test_loss_)

        scheduler.step()

        if((epoch+1)%100==0):
            path = os.path.join(args.save_state_dir,"{}.pth".format(epoch+1))
            state = {"param":net.state_dict(),"train_acc":train_acc,"test_acc":test_acc}
            torch.save(state,path)

def plot_(train_acc_,test_acc_,train_loss_,test_loss_):
    plt.plot(train_acc_,label="train")
    plt.plot(test_acc_,label="test")
    plt.legend()
    plt.savefig("./normal_acc.jpg")
    plt.close()

    plt.plot(train_loss_,label="train")
    plt.plot(test_loss_,label="test")
    plt.legend()
    plt.savefig("./normal_loss.jpg")
    plt.close()




def time_(start,end):
    s = int(end-start)
    h = s//3600
    m = (s//60)%60
    sec = s%60

    return h,m,sec


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_(path,state):
    torch.save(path,state)


def train(epoch,net,optimizer,criterion,trainloader,use_cuda):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


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
