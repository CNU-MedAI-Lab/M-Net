# -*- coding: utf-8 -*-
# unet
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
from tqdm import tqdm
import cv2

DEVICE = "cuda"
device = DEVICE

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

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

from run.dataset import Dataset2
from utils.metrics import dice_coef, batch_iou, mean_iou, iou_score, ppv, sensitivity
from utils import str2bool, count_params, losses
import joblib
from hausdorff import hausdorff_distance
import imageio
import pandas as pd
from model.M_Net_Mamba import VSSM as mnet

wt_dices = []
tc_dices = []
et_dices = []
wt_sensitivities = []
tc_sensitivities = []
et_sensitivities = []
wt_ppvs = []
tc_ppvs = []
et_ppvs = []
wt_Hausdorf = []
tc_Hausdorf = []
et_Hausdorf = []
wt_jc = []
tc_jc = []
et_jc = []

log = pd.DataFrame(index=[], columns=[
    'Label', 'wt_dices', 'tc_dices', 'et_dices', 'wt_sensitivities', 'tc_sensitivities', 'et_sensitivities',
    'wt_ppvs', 'tc_ppvs', 'et_ppvs', 'wt_Hausdorf', 'tc_Hausdorf', 'et_Hausdorf'
])

temp_wt_dice = []


def jac(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))

    intersection = np.count_nonzero(predict & target)
    union = np.count_nonzero(predict | target)

    jac = float(intersection) / float(union)

    return jac



def parse_args():
    parser = argparse.ArgumentParser(description="M-Net Testing Script")

    # ===================== Experiment =====================
    parser.add_argument('--name', default="M_Net_Mamba",
                        help='experiment name (used for logging)')

    # ===================== Dataset =====================
    parser.add_argument('--dataset_path', default="/mnt/sdn/data/BraTS23_15f/",
                        help='dataset identifier')
    parser.add_argument('--input-channels', default=4, type=int)


    # ===================== Training =====================
    parser.add_argument('--save_dir', default='checkpoint/')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='if use 2d slices(shuffled), please use the sequence length, default is 15')

    # ===================== Device =====================
    parser.add_argument('--gpu_device', default='0', type=str,
                        help='CUDA_VISIBLE_DEVICES')

    return parser.parse_args()


def CalculateWTTCET(args, wtpbregion, wtmaskregion, tcpbregion, tcmaskregion, etpbregion, etmaskregion, labelname):
    # pb = out, mask = tar
    # WT
    wtdice = dice_coef(wtpbregion, wtmaskregion)
    wt_dices.append(wtdice)
    wtppv_n = ppv(wtpbregion, wtmaskregion)
    wt_ppvs.append(wtppv_n)
    wtHausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
    wt_Hausdorf.append(wtHausdorff)
    wtsensitivity_n = sensitivity(wtpbregion, wtmaskregion)
    wt_sensitivities.append(wtsensitivity_n)
    wtjc_n = iou_score(wtpbregion, wtmaskregion)
    wt_jc.append(wtjc_n)
    # TC
    tcdice = dice_coef(tcpbregion, tcmaskregion)
    tc_dices.append(tcdice)
    tcppv_n = ppv(tcpbregion, tcmaskregion)
    tc_ppvs.append(tcppv_n)
    tcHausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
    tc_Hausdorf.append(tcHausdorff)
    tcsensitivity_n = sensitivity(tcpbregion, tcmaskregion)
    tc_sensitivities.append(tcsensitivity_n)
    tcjc_n = iou_score(tcpbregion, tcmaskregion)
    tc_jc.append(tcjc_n)
    # ET
    etdice = dice_coef(etpbregion, etmaskregion)
    et_dices.append(etdice)
    etppv_n = ppv(etpbregion, etmaskregion)
    et_ppvs.append(etppv_n)
    etHausdorff = hausdorff_distance(etmaskregion, etpbregion)
    et_Hausdorf.append(etHausdorff)
    etsensitivity_n = sensitivity(etpbregion, etmaskregion)
    et_sensitivities.append(etsensitivity_n)
    etjcn = iou_score(etpbregion, etmaskregion)
    et_jc.append(etjcn)

    tmp = pd.Series([
        labelname,
        wtdice, tcdice, etdice,
        wtsensitivity_n, tcsensitivity_n, etsensitivity_n,
        wtppv_n, tcppv_n, etppv_n,
        wtHausdorff, tcHausdorff, etHausdorff,
        wtjc_n, tcjc_n, etjcn
    ], index=['Label', 'wt_dices', 'tc_dices', 'et_dices',
              'wt_sensitivities', 'tc_sensitivities', 'et_sensitivities',
              'wt_ppvs', 'tc_ppvs', 'et_ppvs',
              'wt_Hausdorf', 'tc_Hausdorf', 'et_Hausdorf',
              'wt_iou', 'tc_iou', 'et_iou',])

    global log
    log = log._append(tmp, ignore_index=True)
    # log.to_csv(f'checkpoint/unet/{args.name}/testlogall_0.csv', index=False)
    # checkpoint/%s/testlog.csv  2021 11 14cjw


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


def clean_data():
    args = parse_args()


def main():
    val_args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = val_args.gpu_device

    args = joblib.load(f"{val_args.save_dir if hasattr(val_args, 'save_dir') else ''}{val_args.name}/args.pkl")

    # 结果输出路径

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')
    joblib.dump(args, f"{args.save_dir}{args.name}/args.pkl")

    # create model
    print("=> creating model %s" % args.name)
    # model = Unet.__dict__[args.arch](args)
    model = mnet(15)

    model = model.to(device)
    # model = torch.nn.DataParallel(model).cuda()
    # Data loading code
    img_paths = glob(
        fr'{val_args.dataset_path}/test/Image/*')  # /home/cnu_cjw/cjwn/Data/BraTS20_Data/val3br20ImageAll/  /home/cnu_cjw/cjwn/Data/BraTS20_Data/val3br20Image/
    mask_paths = glob(fr'{val_args.dataset_path}/test/Mask/*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    print("val_img_paths:%s" % str(len(val_img_paths)))
    print("val_mask_paths:%s" % str(len(val_mask_paths)))

    # os.makedirs(f"{args.save_dir}{args.name}", exist_ok=True)
    model.load_state_dict(torch.load(f'{args.save_dir}{args.name}/modelbest_15f.pth'))  # checkpoint/%s/model_best.pth
    # model.load_state_dict(torch.load(f'checkpoint/unet/{val_args.dataset}/2unet_50.pth'))
    model.eval()

    val_dataset = Dataset2(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(  # num_workers=16,
        val_dataset,
        batch_size=15,
        # batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)

    # # if val_args.mode == "GetPicture":

    tc_num = 0
    # save_path = "/mnt/sdb_newdisk/ljc_cnu/circle_cut/Tasks/MICCAI/unet2d-MICCAI/temp_data_bad/"

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                # for i, (input, target) in enumerate(val_loader):
                # print("len(val_loader):",len(val_loader),i)
                input = input.to(device)
                input = input.view(input.shape[1], input.shape[2], input.shape[3], input.shape[4])
                # input = torch.squeeze(input, dim=0)
                # target = target.cuda()
                if input.shape[0] != 15:
                    break
                # compute output
                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)

                    # output = torch.unsqueeze(output, dim=0)
                # print("img_paths[i]:%s" % img_paths[i])
                output = torch.sigmoid(output).data.cpu().numpy()
                # print(output.shape)

                for j in range(output.shape[0]):

                        tar_1 = np.array(target[j, :, :, :])
                        out_1 = np.array(output[j, :, :, :])
                        out_1[out_1 > 0.5] = 1
                        out_1[out_1 <= 0.5] = 0

                        wtmaskregion = tar_1[0, :, :]
                        wtpbregion = out_1[0, :, :]

                        tcmaskregion = tar_1[1, :, :]
                        tcpbregion = out_1[1, :, :]

                        if val_args.kernel != 0:
                            tcmaskregion = np.array(tcmaskregion)
                            kernel = np.ones((val_args.kernel, val_args.kernel), np.uint8)
                            tcmaskregion = cv2.erode(tcmaskregion, kernel, iterations=1)

                        etmaskregion = tar_1[2, :, :]
                        etpbregion = out_1[2, :, :]
                        name = f'{i} + {j}'

                        wt_dice = dice_coef(wtpbregion, wtmaskregion)
                        tc_dice = dice_coef(tcpbregion, tcmaskregion)
                        et_dice = dice_coef(etpbregion, etmaskregion)
                        res_rgb_img = input[j, 0, :, :] * 255
                        res_rgb_img = res_rgb_img.cpu().numpy()
                        # print(np.max(res_rgb_img))
                        cv2.imwrite(f'{args.save_dir}{args.name}/image/' + str(i) + '.png', res_rgb_img)

                        res_rgb_fake = np.zeros([wtpbregion.shape[0], wtpbregion.shape[1], 3], dtype=np.uint8)
                        res_rgb_fake[:, :, 0][wtpbregion > 0.5] = 255
                        res_rgb_fake[:, :, 1][etpbregion > 0.5] = 255
                        res_rgb_fake[:, :, 2][tcpbregion > 0.5] = 255
                        # print(name)
                        cv2.imwrite(f'{args.save_dir}{args.name}/result/' + str(i) + '.png', res_rgb_fake)

                        res_rgb_mask = np.zeros([wtmaskregion.shape[0], wtmaskregion.shape[1], 3], dtype=np.uint8)
                        res_rgb_mask[:, :, 0][wtmaskregion > 0.5] = 255
                        res_rgb_mask[:, :, 1][etmaskregion > 0.5] = 255
                        res_rgb_mask[:, :, 2][tcmaskregion > 0.5] = 255
                        # print(name)
                        cv2.imwrite(f'{args.save_dir}{args.name}/mask/' + str(i) + '.png', res_rgb_mask)
                        temp_wt_dice.append(np.mean([wt_dice, tc_dice, et_dice]))
                        # cv2.imwrite(f"./checkpoint/unet/{args.dataset}/pred/{i}_{j}.png", res_rgb_fake)

                        # if np.max(np.array(tcmaskregion)) > 0 and tc_dice < 0.88:
                        #     if tc_num % 100 == 0:
                        #         np.save(save_path + f"Image/{tc_num}.npy", np.array(tcpbregion))
                        #         np.save(save_path + f"Mask/{tc_num}.npy", np.array(tcmaskregion))
                        #
                        #     tc_num += 1

                        CalculateWTTCET(val_args, wtpbregion, wtmaskregion, tcpbregion, tcmaskregion, etpbregion,
                                        etmaskregion,
                                        name)
        torch.cuda.empty_cache()
    """
    make the GT numpy data of Val dataset change to 'png' and save
    """
    print(f'{np.mean(temp_wt_dice)}')
    print('WT Dice: %.4f' % np.mean(wt_dices))
    print('TC Dice: %.4f' % np.mean(tc_dices))
    print('ET Dice: %.4f' % np.mean(et_dices))
    print("=============")
    print('WT PPV: %.4f' % np.mean(wt_ppvs))
    print('TC PPV: %.4f' % np.mean(tc_ppvs))
    print('ET PPV: %.4f' % np.mean(et_ppvs))
    print("=============")
    print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
    print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
    print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
    print("=============")
    print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
    print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
    print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
    print("=============")
    print('WT iou: %.4f' % np.mean(wt_jc))
    print('TC iou: %.4f' % np.mean(tc_jc))
    print('ET iou: %.4f' % np.mean(et_jc))
    print("=============")
    print('tc_num:', tc_num)

    mean_core = pd.Series({
        'Label': "mean_core",
        'wt_dices': np.mean(wt_dices), 'tc_dices': np.mean(tc_dices), 'et_dices': np.mean(et_dices),
        'wt_sensitivities': np.mean(wt_sensitivities), 'tc_sensitivities': np.mean(tc_sensitivities),
        'et_sensitivities': np.mean(et_sensitivities),
        'wt_ppvs': np.mean(wt_ppvs), 'tc_ppvs': np.mean(tc_ppvs), 'et_ppvs': np.mean(et_ppvs),
        'wt_Hausdorf': np.mean(wt_Hausdorf), 'tc_Hausdorf': np.mean(tc_Hausdorf), 'et_Hausdorf': np.mean(et_Hausdorf),
        'wt_iou': np.mean(wt_jc), 'tc_iou': np.mean(tc_jc), 'et_iou': np.mean(et_jc),
    },

        name="mean")
    global log
    log = log._append(mean_core, ignore_index=True)

    log.to_csv(f'{args.save_dir}{args.name}/testlogall_new.csv',
               index=False)  # checkpoint/%s/testlog.csv


if __name__ == '__main__':
    #    file = open('val_img_path.txt', 'r')
    #    val_img_path = []
    #    for lines in file.readlines():
    #        lines = lines.replace("\n", "").split(",")
    #        val_img_path.append(lines)
    #    file.close()
    #    print(val_img_path)
    main()
