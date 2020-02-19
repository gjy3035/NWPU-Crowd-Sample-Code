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
from quailitycc import calc_psnr, calc_ssim
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

dot_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        own_transforms.tensormul(255.0)
    ])

restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0

dataRoot = '../ProcessedData/Data.2019.11/NWPU/1204_min_576x768_mod16_2048'

model_path = 'exp/12-06_15-03_NWPU_Res101_SFCN_1e-05/latest_state.pth'

os.makedirs('pred', exist_ok=True)

def main():

    txtpath = os.path.join(dataRoot, 'txt_list', 'val.txt')
    with open(txtpath) as f:
        lines = f.readlines()                            

    test(lines, model_path)
   

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID, 'Res101_SFCN')
    net.cuda()
    lastest_state = torch.load(model_path)
    net.load_state_dict(lastest_state['net'])
    #net.load_state_dict(torch.load(model_path))
    net.eval()

    #f = open('submmited.txt', 'w+')
    for infos in file_list:
        filename = infos.split()[0]
        #print(filename)

        imgname = os.path.join(dataRoot, 'img', filename + '.jpg')
        img = Image.open(imgname)

        dotname = imgname.replace('img', 'dot').replace('jpg', 'png')
        dot_map = Image.open(dotname)
        dot_map = dot_transform(dot_map)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)[None, :, :, :]
        dot_map = dot_map[None, :, :, :]
        with torch.no_grad():
            img = Variable(img).cuda()
            dot_map = Variable(dot_map).cuda()
            algt = torch.sum(dot_map).item()
            crop_imgs, crop_dots, crop_masks = [], [], []
            b, c, h, w = img.shape
            rh, rw = 576, 768
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
            nz, bz = crop_imgs.size(0), 1
            for i in range(0, nz, bz):
                gs, gt = i, min(nz, i+bz)
                crop_pred, crop_den = net.forward(crop_imgs[gs:gt], crop_dots[gs:gt])
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

            pred_map /= LOG_PARA
            pred = torch.sum(pred_map).item()

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        den_map = den_map.cpu().data.numpy()[0,0,:,:]
        print(pred_map.sum(), den_map.sum())
        psnr = calc_psnr(den_map, pred_map)
        ssim = calc_ssim(den_map, pred_map)
        if psnr == 'NaN':
            plt.imsave(os.path.join('pred', f'[{filename}]_[{pred:.2f}|{algt:.2f}]_[{psnr}]_[{ssim:.4f}].png'), pred_map, cmap='jet')
        else:
            plt.imsave(os.path.join('pred', f'[{filename}]_[{pred:.2f}|{algt:.2f}]_[{psnr:.2f}]_[{ssim:.4f}].png'), pred_map, cmap='jet')

        # print(f'{filename} {pred:.4f}', file=f)
        # print(f'{filename} {pred:.4f}')
    #f.close()
                

if __name__ == '__main__':
    main()
