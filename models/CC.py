import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from misc import layer
from . import counters

import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()    
        # pdb.set_trace()    
        ccnet =  getattr(getattr(counters, model_name), model_name)

        gs_layer = getattr(layer, 'Gaussianlayer')

        self.CCN = ccnet()
        self.gs = gs_layer()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            self.gs = torch.nn.DataParallel(self.gs, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
            self.gs = self.gs.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, dot_map):
        density_map = self.CCN(img)
        gt_map = self.gs(dot_map)
        self.loss_mse= self.build_loss(density_map.squeeze(), gt_map.squeeze())               
        return density_map, gt_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

