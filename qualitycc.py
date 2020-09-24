# -*- coding: utf-8 -*-

# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
# from math import exp
# import math

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim(img1, img2, window, window_size, channel, L=1):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
#     return ssim_map.mean()

# def calc_ssim(gtmap, pdmap):
#     window = create_window(11, 1)
#     if gtmap.is_cuda:
#         window = window.cuda(gtmap.get_device())
#     window = window.type_as(gtmap)
#     return _ssim(gtmap, pdmap, window, 11, 1)


# def calc_psnr(gtmap, pdmap):
#     #print(gtmap.sum(), pdmap.sum())
#     gtmax = gtmap.max()
#     mse = torch.sqrt(torch.mean((gtmap - pdmap) ** 2))
#     if gtmax == 0 or mse == 0:
#         return "NaN"
#     psnr = 20 * torch.log10(gtmax / mse)
#     print(mse.item(), gtmax.item(), psnr.item())
#     return psnr


import numpy
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import numpy as np
import math
"""
Hat tip: http://stackoverflow.com/a/5078155/1828289
"""
def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def calc_ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (11,11))
    bimg2 = block_view(img2, (11,11))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)



# def psnr(img1, img2):
#     mse = numpy.mean( (img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_psnr(img1, img2):
    '''
    if np.max(img1) != 0:
        img1 = img1/np.max(img1)*255
    if np.max(img2) != 0:    
        img2 = img2/np.max(img2)*255
    '''

    mse = np.mean( (img1 - img2) ** 2 )
    img1max = np.max(img1)
    if mse == 0 or img1max == 0:
        return "NaN"
    return 20 * math.log10(np.max(img1) / math.sqrt(mse))