import argparse
import random


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
import time

import tqdm


import numpy as np


bratshgg_path = "/mnt/sdn/data/BraTS2023_2/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"

outputImg_path = f"/mnt/sdn/data/BraTS2023_15f/"
outputMask_path = f"/mnt/sdn/data/BraTS2023_15f/"

os.makedirs(outputImg_path + 'train/Image',exist_ok=True)
os.makedirs(outputImg_path + 'val/Image',exist_ok=True)
os.makedirs(outputImg_path + 'test/Image',exist_ok=True)
# if not os.path.exists(outputMask_path):
os.makedirs(outputMask_path + 'train/Mask',exist_ok=True)
os.makedirs(outputMask_path + 'val/Mask',exist_ok=True)
os.makedirs(outputMask_path + 'test/Mask',exist_ok=True)


flair_name = "-t2f.nii.gz"
t1_name = "-t1n.nii.gz"
t1ce_name = "-t1c.nii.gz"
t2_name = "-t2w.nii.gz"
mask_name = "-seg.nii.gz"


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
random.shuffle(pathhgg_list)

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

# print(pathhgg_list[0:int(len(pathhgg_list) * 0.8)])
num = 0
start = time.time()
for subsetindex in tqdm.tqdm(range(int(len(pathhgg_list) * 0.6))):
    # print(str(pathhgg_list[subsetindex]))
    brats_subset_path = bratshgg_path + str(pathhgg_list[subsetindex]) + "/"
    flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + flair_name
    t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1_name
    t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t1ce_name
    t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + t2_name
    mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + mask_name
    # 获取每个病例的四个模态及Mask数据
    # .nii.gz
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
    # print(flair_array.shape)
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
    npmask = crop_ceter(mask_array, 160, 160)
    # 155, 160, 160

    Image = np.zeros([4, 155, 160, 160])

    Mask = np.zeros([3, 155, 160, 160])

    # input 4, 160, 160
    # output 3, 160, 160


    Image[0, :, :, :] = flair_crop
    Image[1, :, :, :] = t1_crop
    Image[2, :, :, :] = t1ce_crop
    Image[3, :, :, :] = t2_crop
    # print(type(Image[0, 0, 0, 0]))

    FourModelImageArray = Image.transpose(1, 0, 2, 3)

    WT_Label = npmask.copy()
    WT_Label[npmask == 1] = 1.
    WT_Label[npmask == 2] = 1.
    WT_Label[npmask == 3] = 1.
    TC_Label = npmask.copy()
    TC_Label[npmask == 1] = 1.
    TC_Label[npmask == 2] = 0.
    TC_Label[npmask == 3] = 1.
    ET_Label = npmask.copy()
    ET_Label[npmask == 1] = 0.
    ET_Label[npmask == 2] = 0.
    ET_Label[npmask == 3] = 1.

    Mask[0, :, :, :] = WT_Label
    Mask[1, :, :, :] = TC_Label
    Mask[2, :, :, :] = ET_Label

    maskImg = Mask.transpose(1, 0, 2, 3)
    # Train: Val:Test
    # Train: Real_Train: VAL: Test
            # 6:2:2
    # UNet
    for i in range(155):
        img = Image[:, i, :, :] # 4, 160, 160
        mask = Mask[:, i, :, :]
        np.save(f'/Image/{i}.npy', img)
        np.save(f'/Mask/{i}.npy', mask)

    # UNet MRI Segmentation

    # itk-snap

    # print(FourModelImageArray.shape)
    # print(maskImg.shape)

    for j in range(int(155/15)):
        img = FourModelImageArray[15*j:15*j+15, :, :, :]
        mask = maskImg[15*j:15*j+15, :, :, :]
        imagepath = outputImg_path + "train/Image/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        maskpath = outputMask_path + "train/Mask/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        # print(img.shape, mask.shape)
        np.save(imagepath, img)  # (160,160,4) np.float dtype('float64')
        np.save(maskpath, mask)  # (160, 160) dtype('uint8') 值为0 1 2 4

    # print(f"gen img nums:{num}, data nums:{subsetindex}")
for subsetindex in tqdm.tqdm(range(int(len(pathhgg_list) * 0.6),int(len(pathhgg_list) * 0.8))):
    # print(str(pathhgg_list[subsetindex]))
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
    # print(flair_array.shape)
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
    npmask = crop_ceter(mask_array, 160, 160)

    Image = np.zeros([4, 155, 160, 160])
    Mask = np.zeros([3, 155, 160, 160])
    Image[0, :, :, :] = flair_crop
    Image[1, :, :, :] = t1_crop
    Image[2, :, :, :] = t1ce_crop
    Image[3, :, :, :] = t2_crop

    FourModelImageArray = Image.transpose(1, 0, 2, 3)

    WT_Label = npmask.copy()
    WT_Label[npmask == 1] = 1.
    WT_Label[npmask == 2] = 1.
    WT_Label[npmask == 3] = 1.
    TC_Label = npmask.copy()
    TC_Label[npmask == 1] = 1.
    TC_Label[npmask == 2] = 0.
    TC_Label[npmask == 3] = 1.
    ET_Label = npmask.copy()
    ET_Label[npmask == 1] = 0.
    ET_Label[npmask == 2] = 0.
    ET_Label[npmask == 3] = 1.

    Mask[0, :, :, :] = WT_Label
    Mask[1, :, :, :] = TC_Label
    Mask[2, :, :, :] = ET_Label

    maskImg = Mask.transpose(1, 0, 2, 3)

    # print(FourModelImageArray.shape)
    # print(maskImg.shape)

    for j in range(int(155 / 15)):
        img = FourModelImageArray[15 * j:15 * j + 15, :, :, :]
        mask = maskImg[15 * j:15 * j + 15, :, :, :]
        imagepath = outputImg_path + "val/Image/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        maskpath = outputMask_path + "val/Mask/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        # print(img.shape, mask.shape)
        np.save(imagepath, img)  # (160,160,4) np.float dtype('float64')
        np.save(maskpath, mask)  # (160, 160) dtype('uint8') 值为0 1 2 4

    # print(f"gen img nums:{num}, data nums:{subsetindex}")
for subsetindex in tqdm.tqdm(range(int(len(pathhgg_list) * 0.8),len(pathhgg_list))):
    # print(str(pathhgg_list[subsetindex]))
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
    # print(flair_array.shape)
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
    npmask = crop_ceter(mask_array, 160, 160)

    Image = np.zeros([4, 155, 160, 160])
    Mask = np.zeros([3, 155, 160, 160])
    Image[0, :, :, :] = flair_crop
    Image[1, :, :, :] = t1_crop
    Image[2, :, :, :] = t1ce_crop
    Image[3, :, :, :] = t2_crop

    FourModelImageArray = Image.transpose(1, 0, 2, 3)

    WT_Label = npmask.copy()
    WT_Label[npmask == 1] = 1.
    WT_Label[npmask == 2] = 1.
    WT_Label[npmask == 3] = 1.
    TC_Label = npmask.copy()
    TC_Label[npmask == 1] = 1.
    TC_Label[npmask == 2] = 0.
    TC_Label[npmask == 3] = 1.
    ET_Label = npmask.copy()
    ET_Label[npmask == 1] = 0.
    ET_Label[npmask == 2] = 0.
    ET_Label[npmask == 3] = 1.

    Mask[0, :, :, :] = WT_Label
    Mask[1, :, :, :] = TC_Label
    Mask[2, :, :, :] = ET_Label

    maskImg = Mask.transpose(1, 0, 2, 3)

    # print(FourModelImageArray.shape)
    # print(maskImg.shape)

    for j in range(int(155 / 15)):
        img = FourModelImageArray[15 * j:15 * j + 15, :, :, :]
        mask = maskImg[15 * j:15 * j + 15, :, :, :]
        imagepath = outputImg_path + "test/Image/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        maskpath = outputMask_path + "test/Mask/" + str(pathhgg_list[subsetindex]) + f"_{j}.npy"
        # print(img.shape, mask.shape)
        np.save(imagepath, img)  # (160,160,4) np.float dtype('float64')
        np.save(maskpath, mask)  # (160, 160) dtype('uint8') 值为0 1 2 4

    # print(f"gen img nums:{num}, data nums:{subsetindex}")
print("用时：{0}".format(time.time() - start))

print("Done！")


