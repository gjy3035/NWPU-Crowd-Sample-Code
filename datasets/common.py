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

class SHHA_collate(object):
    def __init__(self, train_size, label_factor):
        self.train_size = train_size
        self.label_factor = label_factor
    
    def get_min_size(self, batch):

        min_ht = self.train_size[0]
        min_wd = self.train_size[1]

        for i_sample in batch:
            
            _,ht,wd = i_sample.shape
            if ht<min_ht:
                min_ht = ht
            if wd<min_wd:
                min_wd = wd
        return min_ht,min_wd
    
    def random_crop(self, img, den, dst_size):
        # dst_size: ht, wd
        label_factor = self.label_factor
        _,ts_hd,ts_wd = img.shape

        x1 = random.randint(0, ts_wd - dst_size[1]) // label_factor * label_factor
        y1 = random.randint(0, ts_hd - dst_size[0]) // label_factor * label_factor
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        label_x1 = x1 // label_factor
        label_y1 = y1 // label_factor
        label_x2 = x2 // label_factor
        label_y2 = y2 // label_factor

        return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]

    @staticmethod
    def share_memory(batch):
        out = None
        if False:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return out


    def __call__(self, batch):
        # @GJY 
        r"""Puts each data field into a tensor with outer dimension batch size"""

        transposed = list(zip(*batch)) # imgs and dens
        imgs, dens = [transposed[0], transposed[1]]


        error_msg = "batch must contain tensors; found {}"
        if isinstance(imgs[0], torch.Tensor) and isinstance(dens[0], torch.Tensor):
            
            min_ht, min_wd = self. get_min_size(imgs)

            # print min_ht, min_wd

            # pdb.set_trace()
            
            cropped_imgs = []
            cropped_dens = []
            for i_sample in range(len(batch)):
                _img, _den = self.random_crop(imgs[i_sample],dens[i_sample],[min_ht,min_wd])
                cropped_imgs.append(_img)
                cropped_dens.append(_den)


            cropped_imgs = torch.stack(cropped_imgs, 0, out=self.share_memory(cropped_imgs))
            cropped_dens = torch.stack(cropped_dens, 0, out=self.share_memory(cropped_dens))

            return [cropped_imgs,cropped_dens]

        raise TypeError((error_msg.format(type(batch[0]))))

def gccvalmode2list(mode, is_train):
    return {
        'rd': ('test_list.txt', 'train_list.txt'),
        'cc': ('cross_camera_test_list.txt', 'cross_camera_train_list.txt'),
        'cl': ('cross_location_test_list.txt', 'cross_location_train_list.txt')
    }[mode][is_train]