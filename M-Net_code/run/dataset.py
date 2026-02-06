import numpy as np
import cv2 #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms
import os
from PIL import Image

class Dataset2(torch.utils.data.Dataset):
    # for ordered
    def __init__(self, args, img_paths, mask_paths, aug=False, train=True):
        self.args = args
        # self.imgfrq_paths = imgfrq_paths###################
        self.img_paths = img_paths
        self.mask_paths = mask_paths

        # self.img_paths = sorted(self.img_paths)
        # self.mask_paths = sorted(self.mask_paths)
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        # print(mask_path)
        # 读numpy数据(npy)的代码
        npimage = np.load(self.img_paths[idx])
        # npimage = npimage.transpose(2, 0, 1)
        npmask = np.load(self.mask_paths[idx])

        # npimage[npimage == -9.0]=0.
        if np.max(npimage) == -9.0:
            npimage[npimage == -9.0] = 0.
        else:
            npimage = (npimage - np.amin(npimage) + 0.00001) / (np.amax(npimage) - np.amin(npimage) + 0.00001)
        npimage = npimage.astype('float32')
        npmask = npmask.astype('float32')

        return npimage, npmask

