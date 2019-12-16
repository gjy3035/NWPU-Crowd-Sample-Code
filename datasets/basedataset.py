# -*- coding: utf-8 -*-

import torch.utils.data as data
from . import common
import os
from PIL import Image
import numpy as np
import torch

import pdb


class NWPUDataset(data.Dataset):
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.data_path = data_path
        self.datasetname = datasetname

        self.file_name = []
        self.info = []
        
        with open(argv['list_file']) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()
            self.file_name.append(splited[0])
            self.info.append(splited[1:3]) # crowd level, illum


        self.imgfileTemp = os.path.join(data_path, 'img', '{}.jpg')
        self.matfileTemp = os.path.join(data_path, 'mat', '{}.mat')
        self.dotfileTemp = os.path.join(data_path, 'dot', '{}.png')
        # self.gen_dot()
        # pdb.set_trace()

        self.num_samples = len(self.file_name)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.dot_transform = None
        if 'dot_transform' in argv.keys():
            self.dot_transform = argv['dot_transform']
        
        if self.mode is 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode is 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} validation images.')   

    
    def __getitem__(self, index):
        img, dot = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, dot = self.main_transform(img, dot) 
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dot_transform is not None:
            dot = self.dot_transform(dot)

        if self.mode == 'train':    
            return img, dot  
        elif self.mode == 'val':
            attributes_pt = torch.from_numpy(np.array(
                list(map(int, self.info[index]))
            ))
            return img, dot, attributes_pt
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples


    def read_image_and_gt(self,index):

        # pdb.set_trace()

        img_path = self.imgfileTemp.format(self.file_name[index])
        dot_path = self.dotfileTemp.format(self.file_name[index])
       
        img = Image.open(img_path)
        dot = Image.open(dot_path)

        return img, dot

    def get_num_samples(self):
        return self.num_samples

    def gen_dot(self):
        for filename in zip(self.file_name):
            dotfile = self.dotfileTemp.format(filename)
            if not os.path.exists(dotfile):
                imgfile = self.imgfileTemp.format(filename)
                matfile = self.matfileTemp.format(filename)
                w, h = Image.open(imgfile).size
                common.mat2png(matfile, dotfile, w, h)
                print(f'create dot map : {dotfile}')


class BaseDataset(data.Dataset): #for SHHA, QNRF
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.datasetname = datasetname
        self.img_path = os.path.join(data_path, 'img')
        self.mat_path = os.path.join(data_path, 'mat')
        self.dot_path = os.path.join(data_path, 'dot')
        
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        # self.gen_dot()

        self.num_samples = len(self.data_files)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.dot_transform = None
        if 'dot_transform' in argv.keys():
            self.dot_transform = argv['dot_transform']
        
        if self.mode is 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode is 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} testing images.')   

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, dot = self.read_image_and_gt(fname)      
        if self.main_transform is not None:
            img, dot = self.main_transform(img, dot)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dot_transform is not None:
            dot = self.dot_transform(dot)
        return img, dot

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        imgfile = os.path.join(self.img_path,fname)
        img = Image.open(imgfile)
        if img.mode == 'L':
            img = img.convert('RGB')
        # open dot
        idx = os.path.splitext(fname)[0]
        dotfile = os.path.join(self.dot_path, idx + '.png')
        dot = Image.open(dotfile)
        return img, dot    

    def get_num_samples(self):
        return self.num_samples
    
    def gen_dot(self):
        if not os.path.exists(self.dot_path):
            os.mkdir(self.dot_path)
        for fname in self.data_files:
            idx = os.path.splitext(fname)[0]
            dotfile = os.path.join(self.dot_path, idx + '.png')
            if not os.path.exists(dotfile):
                imgfile = os.path.join(self.img_path, fname)
                matfile = os.path.join(self.mat_path, idx + '.mat')
                w, h = Image.open(imgfile).size
                common.mat2png(matfile, dotfile, w, h)
                print(f'create dot map : {dotfile}')

class GCCDataset(data.Dataset):
    def __init__(self, data_path, datasetname, mode, **argv):
        self.mode = mode
        self.data_path = data_path

        self.info = []
        self.file_folder = []
        self.file_name = []
        self.gt_cnt = []
        
        with open(argv['list_file']) as f:
            lines = f.readlines()
        
        for line in lines:
            splited = line.strip().split()
            self.info.append(splited[:3]) # crowd_level, time, weather
            self.file_folder.append(splited[3])
            self.file_name.append(splited[4])
            self.gt_cnt.append(int(splited[5]))

        self.imgfileTemp = os.path.join(data_path, '{}', 'pngs_544_960', '{}.png')
        self.matfileTemp = os.path.join(data_path, '{}', 'mats_544_960', '{}.mat')
        self.dotfileTemp = os.path.join(data_path, '{}', 'dots_544_960', '{}.png')
        self.gen_dot()

        self.num_samples = len(self.data_files)
        self.main_transform = None
        if 'main_transform' in argv.keys():
            self.main_transform = argv['main_transform']
        self.img_transform = None
        if 'img_transform' in argv.keys():
            self.img_transform = argv['img_transform']
        self.dot_transform = None
        if 'dot_transform' in argv.keys():
            self.dot_transform = argv['dot_transform']
        
        if self.mode is 'train':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} training images.')
        if self.mode is 'val':
            print(f'[{self.datasetname} DATASET]: {self.num_samples} testing images.')   

    
    def __getitem__(self, index):
        img, dot = self.read_image_and_gt(index)
      
        if self.main_transform is not None:
            img, dot = self.main_transform(img, dot) 
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dot_transform is not None:
            dot = self.dot_transform(dot)

        if self.mode == 'train':    
            return img, dot,  
        elif self.mode == 'val':
            attributes_pt = torch.from_numpy(np.array(
                list(map(int, self.info[index]))
            ))
            return img, den, attributes_pt
        else:
            print('invalid data mode!!!')

    def __len__(self):
        return self.num_samples


    def read_image_and_gt(self,index):

        img_path = self.imgfileTemp.format(self.file_folder[index], self.file_name[index])
        dot_path = self.dotfileTemp.format(self.file_folder[index], self.file_name[index])
       
        img = Image.open(img_path)
        dot = Image.open(dot_path)

        return img, dot

    def get_num_samples(self):
        return self.num_samples

    def gen_dot(self):
        for folder, filename in zip(self.file_folder, self.file_name):
            dotfile = self.dotfileTemp.format(folder, filename)
            if not os.path.exists(dotfile):
                imgfile = self.imgfileTemp.format(folder, filename)
                matfile = self.matfileTemp.format(folder, filename)
                w, h = Image.open(imgfile).size
                common.mat2png(matfile, dotfile, w, h)
                print(f'create dot map : {dotfile}')


