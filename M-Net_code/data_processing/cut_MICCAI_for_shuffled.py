import argparse
import imageio
from cut import find_best_area

parser = argparse.ArgumentParser()
parser.add_argument("--potion", default=1, type=int)
parser.add_argument("--num_p_start", default=0, type=float)
parser.add_argument("--size", default=512, type=int)
args = parser.parse_args()


#!/usr/bin/env python
# coding: utf-8
# pip install SimpleITK

import os
import SimpleITK as sitk
from test import seg_conv, r_cut, conv_fuzzy_moving
import time
import nibabel as nib
from expend_3d import expend_3d
from utils_cut.images_process import cv_show, gray_scale, gray_scale_3d
from utils_cut.nii_process import nii_to_numpy
import tqdm
from PIL import Image

##读路径
#def text_read(filename):
#    #imagepath = outputImg_path + "/" + str(pathlgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
#    file = open(filename, 'r')
#    arr = []
#    for lines in file.readlines():
#        lines = lines.replace("\n", "")    #.split(",")
#        arr.append(lines)
#    file.close()
#    return arr


import numpy as np
# import cupy as np
#18

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cut_axis = 0

bratshgg_path = "/mnt/sdn/data/BraTS2023/"
#bratslgg_path = "/home/cnu_cjw/cjwn/Data/BraTS20_Data/LGG/"
outputImg_path = f"/mnt/sdn/data/BraTS2023_all/Image/"
outputMask_path = f"/mnt/sdn/data/BraTS2023_all/Mask/"
if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

flair_name = "_flair.nii" # t2f is flair
t1_name = "_t1.nii" # t1n is t1
t1ce_name = "_t1ce.nii" # t1c is t1ce
t2_name = "_t2.nii" # t2w is t2
mask_name = "_seg.nii"



def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            #print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            #print("files:", files)
            return files


pathhgg_list = file_name_path(bratshgg_path)

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """

    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9 #黑色背景区域
        return tmp



def crop_ceter(img,croph,cropw):
    #for n_slice in range(img.shape[0]):
    height,width = img[0].shape
    starth = height//2-(croph//2)
    startw = width//2-(cropw//2)
    return img[:,starth:starth+croph,startw:startw+cropw]


num = 0
for subsetindex in range(
                         int(len(pathhgg_list) * 0.8)):


    start = time.time()
    print(str(pathhgg_list[subsetindex]))
    brats_subset_path = bratshgg_path + str(pathhgg_list[subsetindex]) + "/"
    flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
    mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
    # 获取每个病例的四个模态及Mask数据
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
    # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)
    print(flair_array.shape, t1_array.shape, mask_array.shape)

    # flair_array = flair_array.transpose(1, 0, 2)
    # t1_array = t1_array.transpose(1, 0, 2)
    # t1ce_array = t1ce_array.transpose(1, 0, 2)
    # t2_array = t2_array.transpose(1, 0, 2)
    # mask_array = mask_array.transpose(1, 0, 2)

    flair_array = np.reshape(flair_array, [155, 240, 240])
    t1_array = np.reshape(t1_array, [155, 240, 240])
    t1ce_array = np.reshape(t1ce_array, [155, 240, 240])
    t2_array = np.reshape(t2_array, [155, 240, 240])
    mask_array = np.reshape(mask_array, [155, 240, 240])
    # 对四个模态分别进行标准化,由于它们对比度不同
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)
    # 裁剪(偶数才行)
    flair_crop = crop_ceter(flair_array_nor, 160, 160)
    t1_crop = crop_ceter(t1_array_nor, 160, 160)
    t1ce_crop = crop_ceter(t1ce_array_nor, 160, 160)
    t2_crop = crop_ceter(t2_array_nor, 160, 160)
    mask_crop = crop_ceter(mask_array, 160, 160)

    for n_slice in range(flair_crop.shape[0]):
        # if np.max(mask_crop[n_slice, :, :]) > 0:
            num += 1
            # if np.max(mask_crop[n_slice,:,:]) != 0:
            maskImg = mask_crop[n_slice, :, :]
            FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float32)
            flairImg = flair_crop[n_slice, :, :]
            flairImg = flairImg.astype(np.float32)
            FourModelImageArray[:, :, 0] = flairImg
            t1Img = t1_crop[n_slice, :, :]
            t1Img = t1Img.astype(np.float32)
            FourModelImageArray[:, :, 1] = t1Img
            t1ceImg = t1ce_crop[n_slice, :, :]
            t1ceImg = t1ceImg.astype(np.float32)
            FourModelImageArray[:, :, 2] = t1ceImg
            t2Img = t2_crop[n_slice, :, :]
            t2Img = t2Img.astype(np.float32)
            FourModelImageArray[:, :, 3] = t2Img

            imagepath = outputImg_path + "/" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path + "/" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(imagepath, FourModelImageArray)  # (160,160,4) np.float dtype('float64')
            np.save(maskpath, maskImg)  # (160, 160) dtype('uint8') 值为0 1 2 4

    print(f"gen img nums:{num}, data nums:{subsetindex}")
    print("用时：{0}".format(time.time() - start))

print("hgg_Done！")


