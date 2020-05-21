import os
import numpy as np
import torch
import datasets
from config import cfg
from importlib import import_module
#------------prepare enviroment------------
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True


#------------prepare data loader------------
data_mode = cfg.DATASET
datasetting = import_module(f'datasets.setting.{data_mode}')
cfg_data = datasetting.cfg_data


#------------Prepare Trainer------------
from trainer import Trainer

#------------Start Training------------
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(cfg_data, pwd)
cc_trainer.forward()
