from .common import *
from easydict import EasyDict as edict
import .basedataset.GCCDataset as GCC
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
import torch

# init
__C_GCC = edict()

cfg_data = __C_GCC

__C_GCC.STD_SIZE = (544,960)
__C_GCC.TRAIN_SIZE = (272,480)
__C_GCC.DATA_PATH = '../ProcessedData/GCC'

__C_GCC.VAL_MODE = 'cc' # rd: radomn splitting; cc: cross camera; cl: cross location

__C_GCC.DATA_GT = 'k15_s4'            

__C_GCC.MEAN_STD = ([0.302234709263, 0.291243076324, 0.269087553024], [0.227743327618, 0.211051672697, 0.184846073389])

__C_GCC.LABEL_FACTOR = 1
__C_GCC.LOG_PARA = 1000.

__C_GCC.RESUME_MODEL = ''#model path
__C_GCC.TRAIN_BATCH_SIZE = 16 #imgs

__C_GCC.VAL_BATCH_SIZE = 16 #


cfg_data = __C_GCC

def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para = cfg_data.LOG_PARA
    train_main_transform = own_transforms.Compose([
        # own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])


    test_list = 'test_list.txt'
    train_list = 'train_list.txt'
  


    train_set = GCC(cfg_data.DATA_PATH, cfg_data.DATA_PATH+'/txt_list/' + train_list, 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)

    val_set = GCC(cfg_data.DATA_PATH, cfg_data.DATA_PATH+'/txt_list/'+ test_list, 'test', main_transform=None, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_loader, val_loader, restore_transform
