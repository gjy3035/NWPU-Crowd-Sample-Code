from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0

dataRoot = '../ProcessedData/NWPU/XXX/'

model_path = 'xxx.pth'

def main():

    with open(dataRoot+'test.txt') as f:
        lines = f.readlines()                            

    test(lines, model_path)
   

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID,'VGG')
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    gts = []
    preds = []
    for infos in lines:
        filename = info[0]
    	print( filename )

        imgname = dataRoot + '/img/' + filename + '.jpg'

        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)

        with torch.no_grad():
            img = Variable(img).cuda()
            dot_map = Variable(dot_map).cuda()

            crop_imgs, crop_dots, crop_masks = [], [], []
            b, c, h, w = img.shape
            rh, rw = self.cfg_data.TRAIN_SIZE
            for i in range(0, h, rh):
                gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                    crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                    crop_dots.append(dot_map[:, :, gis:gie, gjs:gje])
                    mask = torch.zeros_like(dot_map).cuda()
                    mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                    crop_masks.append(mask)
            crop_imgs, crop_dots, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_dots, crop_masks))

            # forward may need repeatng
            crop_preds, crop_dens = [], []
            nz, bz = crop_imgs.size(0), self.cfg_data.TRAIN_BATCH_SIZE
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i+bz)
                crop_pred, crop_den = self.net.forward(crop_imgs[gs:gt], crop_dots[gs:gt])
                crop_preds.append(crop_pred)
                crop_dens.append(crop_den)
            crop_preds = torch.cat(crop_preds, dim=0)
            crop_dens = torch.cat(crop_dens, dim=0)

            # splice them to the original size
            idx = 0
            pred_map = torch.zeros_like(dot_map).cuda()
            den_map = torch.zeros_like(dot_map).cuda()
            for i in range(0, h, rh):
                gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                for j in range(0, w, rw):
                    gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                    pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                    den_map[:, :, gis:gie, gjs:gje] += crop_dens[idx]
                    idx += 1

            # for the overlapping area, compute average value
            mask = crop_masks.sum(dim=0).unsqueeze(0)
            pred_map = pred_map / mask
            den_map = den_map / mask

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]


        pred = np.sum(pred_map) / LOG_PARA

        with open('submmited.txt', 'a') as f:
            f.write(filename_no_ext + ' %.4f\n' % (pred))
                

if __name__ == '__main__':
    main()




