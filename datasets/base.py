# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import re
from abc import ABC
from functools import reduce
from pathlib import Path
import glob

import loguru
import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms

import lmdb
import random
import datasets.util as Util
from insightface.utils import face_align
import cv2
from PIL import Image


lmk_folder = '/shared/storage/cs/staffstore/ps1510/Work/TAP2-2/results/lmk_LYHM/arcface_mymodel/lmk'
ori_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/datasets/arcface/LYHM/arcface_input'

class BaseDataset(Dataset, ABC):
    def __init__(self, name, config, device, isEval, need_LR=False, split='train'):
        self.config = config
        self.K = config.mica.datasets.K
        self.isEval = isEval
        self.n_train = np.Inf
        self.imagepaths = []
        self.face_dict = {}
        self.name = name
        self.device = device
        self.min_max_K = 0
        self.cluster = False
        self.dataset_root = config.mica.datasets.root
        self.total_images = 0
        self.image_folder = 'arcface_input'
        self.flame_folder = 'FLAME_parameters'
        self.datatype = config.mica.datasets.datatype
        
        
        # sr
        if self.isEval:
            config_sr = config.sr.datasets.val
        else:
            config_sr = config.sr.datasets.train
        self.datatype = config_sr.datatype
        self.l_res = config_sr.l_resolution
        self.r_res = config_sr.r_resolution
        self.name_res = '_' + str(self.l_res) + '_' + str(self.r_res)
        self.data_len = config_sr.data_len
        self.need_LR = need_LR
        self.split = split
        self.dataroot_sr()
        
        self.initialize()

    def dataroot_sr(self):
        train_dataname = self.config.mica.datasets.training_data[0]
        self.dataroot = os.path.join(self.config.mica.datasets.root, train_dataname + '_' + str(self.l_res) + '_' + str(self.r_res), 'arcface_input')
    
    def scan_img_arcface(self, path, folder):
        train_dataname = self.config.mica.datasets.training_data[0]
        new_paths = []

        # Create full paths for the images
        for i in path:
            name = i.split('/')[-1]
            _path = os.path.join(self.config.mica.datasets.dataset_path, 
                                f"{train_dataname}_{self.l_res}_{self.r_res}", 
                                folder, name)
            new_paths.append(_path)
        
        valid_paths = []
        checked_basenames = set()

        # Check for image pairs and filter
        for i in new_paths:
            name = i.split('/')[-1]
            # Extract the base name by splitting the file name and extracting the middle part
            parts = name.split('_')
            base_name = parts[1]  # Assuming the base name is the second part
            
            if base_name not in checked_basenames:
                checked_basenames.add(base_name)
                # Construct the expected paths for the image pairs
                pair_1 = os.path.join(os.path.dirname(os.path.dirname(path[0])), 'sr' + self.name_res, f"{parts[0]}_{base_name}_1C.png")
                pair_2 = os.path.join(os.path.dirname(os.path.dirname(path[0])), 'sr' + self.name_res, f"{str(int(parts[0])+1).zfill(len(parts[0]))}_{base_name}_2C.png")
                
                # Check if both files exist
                if os.path.exists(pair_1) and os.path.exists(pair_2):
                    valid_paths.append(os.path.join(os.path.dirname(i), f"{parts[0]}_{base_name}_1C.png"))
                    valid_paths.append(os.path.join(os.path.dirname(i), f"{str(int(parts[0])+1).zfill(len(parts[0]))}_{base_name}_2C.png"))
        
        return valid_paths

    def initialize(self):
        image_list = f'{os.path.abspath(os.path.dirname(__file__))}/image_paths/{self.name}.npy'
        
        if self.config.rank == 0:
            logger.info(f'[{self.name}] Initialization')
            logger.info(f'[{self.name}] Load cached file list: ' + image_list)
        
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        
        # sr
        if self.datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif self.datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(self.dataroot, self.l_res, self.r_res))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(self.dataroot, self.r_res))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(self.dataroot, self.l_res))
                self.lr_path = self.scan_img_arcface(self.lr_path, 'lr_' + str(self.l_res)) # !!edit aceface
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(self.datatype))
        
        self.sr_path = self.scan_img_arcface(self.sr_path, 'sr' + self.name_res) # !!edit aceface
        self.hr_path = self.scan_img_arcface(self.hr_path, 'hr_' + str(self.r_res)) # !!edit aceface
        
        self.create_new_face_dict()
        self.imagepaths = list(self.face_dict.keys())
        if self.config.rank == 0:
            logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')
        self.set_smallest_k()
    
    # Function to get the subject ID from the path
    def get_subject_id(self, path):
        match = re.search(r'_(\d{5})_', path)
        return match.group(1) if match else None    
    
    def create_new_face_dict(self):
        import re
        # Create the new dictionary
        new_dict = {}
        for subject_id, (images, npz_path) in self.face_dict.items():
            # Find corresponding sr and hr paths
            sr_images = [p for p in self.sr_path if self.get_subject_id(p) == subject_id]
            hr_images = [p for p in self.hr_path if self.get_subject_id(p) == subject_id]
            
            if self.need_LR:
                lr_images = [p for p in self.lr_path if self.get_subject_id(p) == subject_id]
                # Combine into the new dictionary format
                new_dict[subject_id] = (sr_images, hr_images, lr_images, npz_path)
            else:
                if sr_images != []:
                    # Combine into the new dictionary format
                    new_dict[subject_id] = (sr_images, hr_images, npz_path)
        self.face_dict = new_dict

    def set_smallest_k(self):
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][0]), self.imagepaths))
        loguru.logger.info(f'Dataset {self.name} with min K = {self.min_max_K} max K = {max_min_k} length = {len(self.face_dict)} total images = {self.total_images}')
        return self.min_max_K

    def __len__(self):
        return len(self.imagepaths)
    
    def name_sr(self, path, prefix=''):
        # e.g. 'contents/LYHM_32_128/sr_32_128/0001_00001_1C.png'
        # we want 'sr_32_128/0001_00001_1C.png' with an optional prefix
        parts = Path(path).parts
        # find the index of 'sr_32_128'
        sr_index = parts.index('sr' + self.name_res)
        # construct the new path with prefix
        new_path = Path(*parts[sr_index:])
        return new_path

    def __getitem__(self, index):
        actor = self.imagepaths[index]
        if self.need_LR:
            sr_images, gt_images, lr_images, params_path = self.face_dict[actor]
        else:
            sr_images, gt_images, params_path = self.face_dict[actor]
        sr_images = [Path(self.dataset_root, self.name + self.name_res, self.image_folder, self.name_sr(path, 'sr' + self.name_res)) for path in sr_images]        
        sample_list = np.array(np.random.choice(range(len(sr_images)), size=self.K, replace=False))
        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(sr_images))[:K])

        params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
        pose = torch.tensor(params['pose']).float()
        betas = torch.tensor(params['betas']).float()

        flame = {
            'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
            'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
            'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        }

        images_list = []
        arcface_list = []

        for i in sample_list:
            image_path = sr_images[i]
            image = np.array(Image.open(image_path))
            image = image / 255.
            image = image.transpose(2, 0, 1)
            
            subject = image_path.name.split('/')[-1].split('_')[-2]
            idx = image_path.name.split('/')[-1].split('_')[-1][:-4] # normal _sr should use [-2]
        
            aimg_path = os.path.join(ori_path, subject, idx + '.npy')

            arcface_image = np.load(aimg_path)

            images_list.append(image)
            arcface_list.append(torch.tensor(arcface_image))

        images_array = torch.from_numpy(np.array(images_list)).float()
        arcface_array = torch.stack(arcface_list).float()
        
        # sr
        img_HR = None
        img_LR = None
        
        if self.datatype == 'lmdb': 
            pass # didn't change the index problem yet !!take a look!!
            # with self.env.begin(write=False) as txn:
            #     hr_img_bytes = txn.get(
            #         'hr_{}_{}'.format(
            #             self.r_res, str(index).zfill(5)).encode('utf-8')
            #     )
            #     sr_img_bytes = txn.get(
            #         'sr_{}_{}_{}'.format(
            #             self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
            #     )
            #     if self.need_LR:
            #         lr_img_bytes = txn.get(
            #             'lr_{}_{}'.format(
            #                 self.l_res, str(index).zfill(5)).encode('utf-8')
            #         )
            #     # skip the invalid index
            #     while (hr_img_bytes is None) or (sr_img_bytes is None):
            #         new_index = random.randint(0, self.data_len-1)
            #         hr_img_bytes = txn.get(
            #             'hr_{}_{}'.format(
            #                 self.r_res, str(new_index).zfill(5)).encode('utf-8')
            #         )
            #         sr_img_bytes = txn.get(
            #             'sr_{}_{}_{}'.format(
            #                 self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
            #         )
            #         if self.need_LR:
            #             lr_img_bytes = txn.get(
            #                 'lr_{}_{}'.format(
            #                     self.l_res, str(new_index).zfill(5)).encode('utf-8')
            #             )
            #     img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
            #     img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
            #     if self.need_LR:
            #         img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = []
            img_SR = []
            img_LR = []
            for i in sample_list:
                img_HR.append(Image.open(self.hr_path[index*2 + i]).convert("RGB"))
                img_SR.append(Image.open(self.sr_path[index*2 + i]).convert("RGB"))
                if self.need_LR:
                    img_LR.append(Image.open(self.lr_path[index*2 + i]).convert("RGB"))
        if self.need_LR:
            for i in sample_list:
                [img_LR[i], img_SR[i], img_HR[i], ] = Util.transform_augment(
                    [img_LR[i], img_SR[i], img_HR[i]], split=self.split, min_max=(-1, 1))
            return {
                'image': images_array,
                'arcface': arcface_array,
                'imagename': actor,
                'dataset': self.name,
                'flame': flame,
                'LR': img_LR, 
                'HR': img_HR, 
                'SR': img_SR, 
                'Index': index
            }
        else:
            for i in sample_list:
                [img_SR[i], img_HR[i]] = Util.transform_augment(
                    [img_SR[i], img_HR[i]], split=self.split, min_max=(-1, 1))
            return {
                'image': images_array,
                'arcface': arcface_array,
                'imagename': actor,
                'dataset': self.name,
                'flame': flame,
                'HR': img_HR, # (2,4,3,128,128)
                'SR': img_SR, # (2,4,3,128,128)
                'Index': index
            }

        # return {
        #     'image': images_array, -> torch.Size([2, 3, 224, 224])
        #     'arcface': arcface_array, -> torch.Size([2, 3, 112, 112])
        #     'imagename': actor,
        #     'dataset': self.name,
        #     'flame': flame,
        # }
