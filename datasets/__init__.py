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


import numpy as np
from torch.utils.data import ConcatDataset

from datasets.base import BaseDataset

import logging
from re import split
import torch.utils.data


def build_train(config, device):
    data_list = []
    total_images = 0
    for dataset in config.mica.datasets.training_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=False)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images


def build_val(config, device):
    data_list = []
    total_images = 0
    for dataset in config.mica.datasets.eval_data:
        dataset_name = dataset.upper()
        config.n_train = np.Inf
        if type(dataset) is list:
            dataset_name, n_train = dataset
            config.n_train = n_train

        dataset = BaseDataset(name=dataset_name, config=config, device=device, isEval=True)
        data_list.append(dataset)
        total_images += dataset.total_images

    return ConcatDataset(data_list), total_images

# sr

def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from datasets.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset



