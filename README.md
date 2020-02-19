# NWPU-Crowd Sample Code

---

This repo is the official implementation of [paper](https://arxiv.org/abs/2001.03360): **NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting**. The code is developed based on [C^3 Framework](https://github.com/gjy3035/C-3-Framework). 

Compared with the original C^3 Framework, 
- the python3.x's new features are utilized;
- the density map is generated online by a conv layer for saving io time on the disk;
- improve the visualization in the Tensorboard.

These features will be merged into C^3 Framework as soon as possible.


# Getting Started

## Preparation
- Prerequisites
  - Python 3.x
  - Pytorch 1.x: http://pytorch.org .
  - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```.

- Installation
  - Clone this repo:
    ```
    git clone https://github.com/gjy3035/NWPU-Crowd-Sample-Code.git
    ```
    
- Data Preparation
  - Download NWPU-Crowd dataset from this [link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FNWPU%2DCrowd&originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fc3ViTXA0OHd3SkRpSDBZbFQ4Mk5ZWUJtWTlMMHMtRnByckJjb2FBSmtJMXJ3P3J0aW1lPWdxTkxjV0dTMTBn). 
  - Unzip ```*zip``` files in turns and place ```images_part*``` into a folder. Finally, the folder tree is below:
  ```
    -- NWPU-Crowd
        |-- images
        |   |-- 0001.jpg
        |   |-- 0002.jpg
        |   |-- ...
        |   |-- 5109.jpg
        |-- jsons
        |   |-- 0001.json
        |   |-- 0002.json
        |   |-- ...
        |   |-- 3609.json
        |-- mats
        |   |-- 0001.mat
        |   |-- 0002.mat
        |   |-- ...
        |   |-- 3609.mat
        |-- train.txt
        |-- val.txt
        |-- test.txt
        |-- readme.md
    ```
  - Run ```./datasets/prepare_NWPU.m``` using [Matlab](https://www.mathworks.com/). 
  - Modify ```__C_NWPU.DATA_PATH``` in ```./datasets/setting/NWPU.py``` with the path of your processed data.


## Training

- Set the parameters in ```config.py``` and ```./datasets/setting/NWPU.py``` (if you want to reproduce our results, you are recommended to use our parameters in ```./saved_exp_para```).
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

## Testing

We only provide an example to forward the model on the test set. You may need to modify it to test your models.

- Modify some key parameters in ```test.py```: 
  - Line 32: ```LOG_PARA```, the same as ```__C_NWPU.LOG_PARA``` in ```./datasets/setting/NWPU.py```.
  - Line 34: ```dataRoot```, the same as ```__C_NWPU.DATA_PATH``` in ```./datasets/setting/NWPU.py```.
  - Line 36: ```model_path```.  
  - Line 48: GPU Id and Model Name. 
- Run ```python test.py```. 

## Pre-trained Models

We provide the pre-trained models in this [link](http://share.crowdbenchmark.com:2443/home/Pre-trained_Models_NWPU-Crowd), which is a temporary share point of OneDrive. We will provide a permanent website ASAP. 

# Performance on the validation set

The overall results on val set:

|   Method   |  MAE  |  MSE  |  PSNR  |  SSIM  | 
|------------|-------|-------|--------|--------|
| MCNN [1]   | 218.53| 700.61| 28.558 |  0.875 |
| C3F-VGG [2]| 105.79| 504.39| 29.977 |  0.918 |
| CSRNet [3] | 104.89| 433.48| 29.901 |  0.883 |
| CANNet [4] |  93.58| 489.90| 30.428 |  0.870 |
| SCAR [5]   |  **81.57**| **397.92**| 30.356 |  0.920 |
| SFCN+ [6]  |  95.46| 608.32| **30.591** | **0.952**|


About the leaderboard on the test set, please visit [Crowd benchmark](https://crowdbenchmark.com/nwpucrowd.html).  

## References

1. Single-Image Crowd Counting via Multi-Column Convolutional Neural Network, CPVR, 2016.
2. C^3 Framework: An Open-source PyTorch Code for Crowd Counting, arXiv, 2019.
3. CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes, CVPR, 2018. 
4. Context-Aware Crowd Counting, CVPR, 2019.
5. SCAR: Spatial-/Channel-wise Attention Regression Networks for Crowd Counting, Neurocomputing, 2019.
6. Learning from Synthetic Data for Crowd Counting in the Wild, CVPR, 2019.


# Evaluation Scheme 

The Evaluation Python Code of the ```crowdbenchmark.com``` is shown in ```./misc/evaluation_code.py```, which is similar to our validation code in ```trainer.py```. 

# Citation
If you find this project is useful for your research, please cite:
```
@article{gao2020nwpu,
  title={NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting},
  author={Wang, Qi and Gao, Junyu and Lin, Wei and Li, Xuelong},
  journal={arXiv preprint arXiv:2001.03360},
  year={2020}
}
```

Our code borrows a lot from the C^3 Framework, you may cite:
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```
If you use crowd counting models in this repo (MCNN, C3F-VGG, CSRNet, CANNet, SCAR, and SFCN+), please cite them. 


