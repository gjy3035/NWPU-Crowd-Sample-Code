# -*- coding: utf-8 -*-

import os
import scipy.io as scio
import numpy as np
from PIL import Image
import random
import torch

def get_gt_dots(mat_path, img_height, img_width):
    """
    Load Matlab file with ground truth labels and save it to numpy array.
    ** cliping is needed to prevent going out of the array
    """
    mat = scio.loadmat(mat_path)
    if "image_info" in mat:
        gt = mat["image_info"][0,0][0,0][0].astype(np.float32).round().astype(int)
    elif "annPoints" in mat:
        gt = mat['annPoints'].astype(int)
    gt[:,0] = gt[:,0].clip(0, img_width - 1)
    gt[:,1] = gt[:,1].clip(0, img_height - 1)
    return gt

def mat2png(matf, dotf, w, h):
    gtlist = get_gt_dots(matf, h, w)
    dotmap = np.zeros((h, w))
    for (i, j) in gtlist:
        dotmap[j, i] += 1.0
    dotimg = Image.fromarray(dotmap.astype('uint8'))
    #print(dotmap.max(), dotmap.min(), dotmap.sum())
    dotimg.save(dotf)

def mkmdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)