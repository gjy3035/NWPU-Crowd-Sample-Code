# -*- coding: utf-8 -*-

import os
from importlib import import_module
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms
from . import basedataset
from . import common
from . import setting
from torch.utils.data import DataLoader
import pdb

def createTrainData(datasetname, Dataset, cfg_data):
    cfg_data.DATA_PATH + '/train'
    folder, list_file = None, None
    if datasetname == 'GCC':
        train_list = common.gccvalmode2list(cfg_data.VAL_MODE, True)
        list_file = os.path.join(cfg_data.DATA_PATH, 'txt_list', train_list)
        train_path = cfg_data.DATA_PATH
    if datasetname == 'NWPU':
        list_file = os.path.join(cfg_data.DATA_PATH, 'txt_list/train.txt')
        train_path = cfg_data.DATA_PATH

    # pdb.set_trace()

    main_transform = own_transforms.Compose([
    	own_transforms.RandomCrop(cfg_data.TRAIN_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    img_transform = standard_transforms.Compose([
        own_transforms.RGB2Gray(0.1),
        own_transforms.GammaCorrection([0.4,2]),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    dot_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        own_transforms.tensormul(255.0),
        own_transforms.LabelNormalize(cfg_data.LOG_PARA),
    ])

    train_set = Dataset(train_path, datasetname, 'train',
        main_transform = main_transform,
        img_transform = img_transform,
        dot_transform = dot_transform,
        list_file = list_file,
        folder = folder    
    )
    return DataLoader(train_set, batch_size=cfg_data.TRAIN_BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)

def createValData(datasetname, Dataset, cfg_data):
    val_path = cfg_data.DATA_PATH + '/test'
    folder, list_file = None, None
    if datasetname == 'GCC':
        test_list = common.gccvalmode2list(cfg_data.VAL_MODE, True)
        list_file = os.path.join(cfg_data.DATA_PATH, '/txt_list/', test_list)
        val_path = cfg_data.DATA_PATH
    if datasetname == 'NWPU':
        list_file = os.path.join(cfg_data.DATA_PATH, 'txt_list/val.txt')
        val_path = cfg_data.DATA_PATH        
    
    #main_transform = None#cfg_data['main_transform']
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])
    dot_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        own_transforms.tensormul(255.0),
        own_transforms.LabelNormalize(cfg_data.LOG_PARA),
    ])

    test_set = Dataset(val_path, datasetname, 'val',
        img_transform = img_transform,
        dot_transform = dot_transform,
        list_file = list_file,
        folder = folder    
    )
    train_loader = DataLoader(test_set, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    return train_loader

def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def loading_data(datasetname):
    datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    if datasetname == 'GCC':
        Dataset = basedataset.GCCDataset
    elif datasetname == 'NWPU':
        Dataset = basedataset.NWPUDataset        
    else:
        Dataset = basedataset.BaseDataset

    # pdb.set_trace()
    
    train_loader = createTrainData(datasetname, Dataset, cfg_data)
    val_loader = createValData(datasetname, Dataset, cfg_data)
    restore_transform = createRestore(cfg_data.MEAN_STD)
    return train_loader, val_loader, restore_transform

