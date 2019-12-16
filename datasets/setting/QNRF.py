from easydict import EasyDict as edict

# init
__C_QNRF = edict()

cfg_data = __C_QNRF

__C_QNRF.TRAIN_SIZE = (576, 768)
__C_QNRF.DATA_PATH = '../ProcessedData/Data.2019.11/QNRF/min_'+ str(__C_QNRF.TRAIN_SIZE[0]) + 'x' + str(__C_QNRF.TRAIN_SIZE[1]) + '_mod16'          

__C_QNRF.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449]) 

__C_QNRF.LABEL_FACTOR = 1
__C_QNRF.LOG_PARA = 100.

__C_QNRF.RESUME_MODEL = ''#model path
__C_QNRF.TRAIN_BATCH_SIZE = 6 #imgs

__C_QNRF.VAL_BATCH_SIZE = 1 # must be 1
