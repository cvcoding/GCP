# -*- coding: utf-8 -*-
'''
Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import pandas as pd
import csv
from models import *
from models.vit import ViT
from utils import progress_bar

import os
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

print(torch.__version__)  # 1.1.0
print(torchvision.__version__)  # 0.3.0

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.set_num_threads(1)

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='32')
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args() #, default='1'

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler
if args.aug:
    import albumentations
bs = int(args.bs)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.Scale(34),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize(mean=[0.1307], std=[0.3081]),
])

##############kaishi

# testset = torchvision.datasets.ImageFolder(root='./data/fashion_mnist_png/test', transform=transform_test)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 自定义Dataset类，__getitem__(self,index)每次返回(img1, img2, 0/1)
class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # 37个类别中任选一个
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0]).convert('RGB')
        img1 = Image.open(img1_tuple[0]).convert('RGB')

        # if img0.mode == 'RGBA':
        #     r, g, b, a = img0.split()
        #     img0 = Image.merge('RGB', (r, g, b))
        # if img1.mode == 'RGBA':
        #     r, g, b, a = img1.split()
        #     img1 = Image.merge('RGB', (r, g, b))

        # img0 = img0.convert("L")
        # img1 = img1.convert("L")

        # if self.should_invert:
        #     img0 = PIL.ImageOps.invert(img0)
        #     img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            # a = 1

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 定义文件dataset
# training_dir = "./data/fashion_mnist_png/trainval/"  # 训练集地址
# folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)

# # 定义图像dataset
# transform = transforms.Compose([
#     transforms.Scale(128),
#     # transforms.RandomCrop(128, padding=4),
#     transforms.RandomHorizontalFlip(),
#     # transforms.Scale(32),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# # transform = transforms.Compose([transforms.Resize((100, 100)),  # 有坑，传入int和tuple有区别
# #                                 transforms.ToTensor()])
# # trainset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_train128', transform=transform_train)
# folder_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)


transform = transforms.Compose([transforms.Scale(36),
                                transforms.RandomCrop(32, padding=2),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                transforms.Normalize(mean=[0.1307], std=[0.3081]),
                                ])

# trainset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_train128', transform=transform_train)
train_after_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_after_dataset = torchvision.datasets.ImageFolder(root='./data/fashion_mnist_png/trainval/', transform=transform)
trainafterloader = torch.utils.data.DataLoader(train_after_dataset, batch_size=bs, shuffle=True, num_workers=0)

####################jieshu

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res34':
    net = ResNet34()
elif args.net == 'res50':
    net = ResNet50()
elif args.net == 'res101':
    net = ResNet101()
elif args.net == "vit":
    # ViT for cifar10
    net = ViT(
        image_size=32,
        patch_size=args.patch,
        kernel_size=5,
        levels4tcn=6,
        batch_size=bs,
        num_classes=10,
        dim=32,
        depth=2,
        heads=8,
        mlp_dim=32,
        patch_stride=1,
        dropout=0.1,  # 0.1
        emb_dropout=0.1,  # 0.1
        expansion_factor=2
    )

if device == 'cuda':
    net = torch.nn.DataParallel(net)  # make parallel
    cudnn.benchmark = True
net = net.to(device)

# net = torch.load('model4cifarnores.pkl')

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in net.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in net.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CL
criterion_ce = nn.CrossEntropyLoss()

# reduce LR on Plateau
if args.opt == "adam":
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
if not args.cos:
    from torch.optim import lr_scheduler

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-3 * 1e-5,
                                               factor=0.5)
else:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=2.0, total_epoch=5, after_scheduler=scheduler_cosine)

counter = []
loss_history = []
iteration_number = 0

import time
##### Training
def train_after(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainafterloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion_ce(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.5f}'
    print(content)

    # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if not args.cos:
        scheduler.step(train_loss)
    else:
        scheduler.step(epoch - 1)
    return train_loss / (batch_idx + 1)

##### Validation
import time

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion_ce(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.net + '-{}-ckpt.t7'.format(args.patch))
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


list_loss = []
list_acc = []

val_loss_best = 1e5
train_loss_best = 1e5

# net = torch.load('model4cifarnores.pkl')

for epoch in range(start_epoch, args.n_epochs):
    trainloss = train_after(epoch)
    val_loss, acc = test(epoch)

    if val_loss < val_loss_best:
        val_loss_best = val_loss
        torch.save(net, 'model4imagenet.pkl')

    if args.cos:
        scheduler.step(epoch - 1)

    list_loss.append(val_loss)
    list_acc.append(acc)

    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss)
        writer.writerow(list_acc)
    print(list_loss)


