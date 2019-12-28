# NWPU-Crowd Sample Code



---

This repo is the ofiicial implementation of [paper](). It is developed based on [C^3 Framework](). 

Compared with the original C^3 Framework, the python3.x's new features are utilized, and the density map is generated online by a conv layer for saving io time on the disk. These features will be mergerd to C^3 Framework as soon as possible.


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
  - In ```./datasets/XXX/readme.md```, download our processed dataset or run the ```prepare_XXX.m/.py``` to generate the desity maps. If you want to directly download all processeed data (including Shanghai Tech, UCF-QNRF, UCF_CC_50 and WorldExpo'10), please visit the [**link**](https://mailnwpueducn-my.sharepoint.com/:f:/g/personal/gjy3035_mail_nwpu_edu_cn/EkxvOVJBVuxPsu75YfYhv9UBKRFNP7WgLdxXFMSeHGhXjQ?e=IdyAzA).
  - Place the processed data to ```../ProcessedData```.

- Pretrained Model
  - Some Counting Networks (such as VGG, CSRNet and so on) adopt the pre-trained models on ImageNet. You can download them from [TorchVision](https://github.com/pytorch/vision/tree/master/torchvision/models)
  - Place the processed model to ```~/.cache/torch/checkpoints/``` (only for linux OS). 


### Training

- set the parameters in ```config.py``` and ```./datasets/XXX/setting.py``` (if you want to reproduce our results, you are recommonded to use our parameters in ```./results_reports```).
- run ```python train.py```.
- run ```tensorboard --logdir=exp --port=6006```.

### Testing

We only provide an example to test the model on the test set. You may need to modify it to test your own models.

# Performance on the validation set

The overall results on val set:

|   Method   |  MAE  |  MSE  |  PSNR  |  SSIM  | 
|------------|-------|-------|--------|--------|
| MCNN       | 218.53| 700.61| 28.558 |  0.875 |
| C3F-VGG    | 105.79| 504.39| 29.977 |  0.918 |
| CSRNet     | 104.89| 433.48| 29.901 |  0.883 |
| CANNet     |  93.58| 489.90| 30.428 |  0.870 |
| SCAR       |  **81.57**| **397.92**| 30.356 |  0.920 |
| SFCN+      |  95.46| 608.32| **30.591** | **0.952**|


 About the leaderboard on the test set, please visit [Crowd benchmark](https://www.crowdbenchmark.com/crowdresult.html).  




## Citation
If you find this project is useful for your research, please cite:
```

```

Our code borrows a lot from the C^3 Framework, you may cite it:
```
@article{gao2019c,
  title={C$^3$ Framework: An Open-source PyTorch Code for Crowd Counting},
  author={Gao, Junyu and Lin, Wei and Zhao, Bin and Wang, Dong and Gao, Chenyu and Wen, Jun},
  journal={arXiv preprint arXiv:1907.02724},
  year={2019}
}
```
If you use a specific crowd couning models, please cite it. 
