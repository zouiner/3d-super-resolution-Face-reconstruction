import os, sys
import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from .vggface import VGGFace2Dataset
from .ethnicity import EthnicityDataset
from .aflw2000 import AFLW2000
from .now import NoWDataset
from .vox import VoxelDataset

from .mocktest import mockDataset
from .mocktest_old import mockDataset_old

def build_train(config, is_train=True):
    data_list = []
    print("Train:")
    if 'vox2' in config.train.name:
        data_list.append(VoxelDataset(dataname='vox2', K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'vggface2' in config.train.name:
        data_list.append(VGGFace2Dataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'vggface2hq' in config.train.name:
        data_list.append(VGGFace2HQDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'ethnicity' in config.train.name:
        data_list.append(EthnicityDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'coco' in config.train.name:
        data_list.append(COCODataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    if 'celebahq' in config.train.name:
        data_list.append(CelebAHQDataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    if 'mocktest' in config.train.name:
        data_list.append(mockDataset(K=config.K, image_size=config.image_size, 
                                         l_resolution = config.train.l_resolution, r_resolution = config.train.r_resolution, 
                                         scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, data_len = config.train.data_len))
    if 'mocktest_old' in config.train.name:
        data_list.append(mockDataset_old(K=config.K, image_size=config.image_size, 
                                         l_resolution = config.train.l_resolution, r_resolution = config.train.r_resolution, 
                                         scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    dataset = ConcatDataset(data_list)
    
    
    return dataset

def build_val(config, is_train=True):
    data_list = []
    print("Val:")
    if 'vggface2' in config.val.name:
        data_list.append(VGGFace2Dataset(isEval=True, K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'now' in config.val.name:
        data_list.append(NoWDataset())
    if 'aflw2000' in config.val.name:
        data_list.append(AFLW2000())
    if 'mocktest' in config.val.name:
        data_list.append(mockDataset(K=config.K, image_size=config.image_size, 
                                         l_resolution = config.train.l_resolution, r_resolution = config.train.r_resolution, 
                                         scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, data_len = config.val.data_len, need_LR = True, split = 'val'))
    if 'mocktest_old' in config.val.name:
        data_list.append(mockDataset_old(K=config.K, image_size=config.image_size, 
                                         l_resolution = config.train.l_resolution, r_resolution = config.train.r_resolution, 
                                         scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    dataset = ConcatDataset(data_list)

    return dataset
    