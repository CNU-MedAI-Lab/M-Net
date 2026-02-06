# -*- coding: utf-8 -*-
# 删除了验证部分
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torchvision import utils

from run.dataset import Dataset2
from utils import count_params, losses
from utils.metrics import dice_coef, batch_iou, mean_iou, iou_score
from model.M_Net_Mamba import VSSM as mnet
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

arch_names = list(mnet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

DEVICE = "cuda"
device = DEVICE


def parse_args():
    parser = argparse.ArgumentParser(description="M-Net Training Script")

    # ===================== Experiment =====================
    parser.add_argument('--name', default="M_Net_Mamba",
                        help='experiment name (used for logging)')
    parser.add_argument('--seed', default=41, type=int)

    # ===================== Dataset =====================
    parser.add_argument('--dataset_path', default="/mnt/sdn/data/BraTS2023_new/",
                        help='dataset identifier')
    parser.add_argument('--input-channels', default=4, type=int)

    # ===================== Model =====================
    parser.add_argument('--deepsupervision', action='store_true')

    # ===================== Training =====================
    parser.add_argument('--save_dir', default='checkpoint/')
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--early-stop', default=50, type=int)
    parser.add_argument('-b', '--batch-size', default=15, type=int,
                        help='if use 2d slices(shuffled), please use the sequence length, default is 15')

    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', action='store_true')

    # ===================== Device =====================
    parser.add_argument('--gpu_device', default='0', type=str,
                        help='CUDA_VISIBLE_DEVICES')

    return parser.parse_args()


# 计算平均值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def joint_loss(pred, mask):  # criterion==>joint_loss
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')#reduction='mean'
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  # reduction='mean'
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



# 训练函数
def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print(input.shape)
        # if input.shape[1] < 15:
        #     continue
        input = input.cuda()
        target = target.cuda()
        # print(input.shape, target.shape)
        # input = torch.squeeze(input, dim=0)
        # input = input.view(input.shape[1], input.shape[2], input.shape[3], input.shape[4])
        # 自适应调整学习率
        adjust_learning_rate(optimizer, epoch, args.epochs, args.lr)
        # compute output 将数据送入网络中计算输出
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += joint_loss(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            # output = torch.unsqueeze(output, dim=0)
            # output = output.view(1, output.shape[0], output.shape[1], output.shape[2], output.shape[3])
            loss = joint_loss(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()  # 清除上一次所求出的梯度
        loss.backward()  # 误差反向传播
        optimizer.step()  # 优化器开始工作
        # torch.cuda.empty_cache()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])
    # 保存模型
    if (epoch + 1) % 50 == 0:
        torch.save(model.state_dict(), 'checkpoint/unet/%s/' % args.dataset + '2unet_%d.pth' % (epoch + 1))
        print('[Saving Snapshot]', 'checkpoint/unet/%s/' % args.dataset + '2unet_%d.pth' % (epoch + 1))

    return log


# 验证
def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    val_dices = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            # if input.shape[1] < 15:
            #     continue
            input = input.cuda()
            # input = torch.squeeze(input, dim=0)
            # input = input.view(input.shape[1], input.shape[2], input.shape[3], input.shape[4])
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += joint_loss(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                # output = torch.unsqueeze(output, dim=0)
                # output = output.view(1, output.shape[0], output.shape[1], output.shape[2], output.shape[3])
                loss = joint_loss(output, target)
                iou = iou_score(output, target)
                output = torch.sigmoid(output).data.cpu().numpy()
                output[output > 0.5] = 1
                output[output <= 0.5] = 0
                for j in range(output.shape[0]):
                    dice_1 = dice_coef(output[j, 0, :, :], target[j, 0, :, :])
                    dice_2 = dice_coef(output[j, 1, :, :], target[j, 1, :, :])
                    dice_3 = dice_coef(output[j, 2, :, :], target[j, 2, :, :])
                    val_dices.append(np.mean([dice_1, dice_2, dice_3]))

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice', np.mean(val_dices))
    ])

    return log


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def main():
    # 先定义一些常用变量
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    # 确保 checkpoint 路径存在
    os.makedirs(f'{args.save_dir}{args.name}', exist_ok=True)

    # 打印出所有的参数
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('----------------')

    with open(f'{args.save_dir}{args.name}/args.txt', 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, f'{args.save_dir}{args.name}/args.pkl')

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True

    # Data loading code
    train_img_paths = glob(rf'{args.dataset_path}/train/Image/*')  # 原始切片路径
    train_mask_paths = glob(rf'{args.dataset_path}/train/Mask/*')


    train_img_paths.sort()
    train_mask_paths.sort()

    val_img_paths = glob(rf'{args.dataset_path}/val/Image/*')
    val_mask_paths = glob(rf'{args.dataset_path}/val/Mask/*')
    val_img_paths.sort()
    val_mask_paths.sort()

    print("train_num:%s" % str(len(train_img_paths)))
    #    print(train_img_paths)
    print("val_num:%s" % str(len(val_img_paths)))


    # create model
    print(f"=> creating model {args.name}")
    model = mnet(15) #(input sequence, depend on data preprocessing, default is 15)

    model.to(device)

    # 选择设置优化器
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # 给训练集、验证集赋值
    train_dataset = Dataset2(args, train_img_paths, train_mask_paths)
    val_dataset = Dataset2(args, val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=4)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou', 'dice'
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou', 'val_dice', ])
        log = log._append(tmp, ignore_index=True)
        log.to_csv(f'{args.dataset_path}log.csv', index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), f'{args.dataset_path}/modelbest_15f.pth')
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
