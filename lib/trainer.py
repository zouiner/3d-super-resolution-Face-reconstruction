import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .datasets import build_datasets

from tensorboardX import SummaryWriter
import logging

import torch
import model.sr as Model


# add by Patipol
# import matplotlib.pyplot as plt
# from skimage.io import imsave
# from PIL import Image, ImageDraw
# import torchvision.transforms as transforms




class Trainer(object):
    def __init__(self, model, config=None, device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.datasets.train.batch_size
        self.l_image_size = self.cfg.datasets.train.l_resolution
        self.r_image_size = self.cfg.datasets.train.r_resolution
        
        self.tb_logger = SummaryWriter(log_dir=self.cfg.path.tb_logger)
        
        # SR model
        # self.sr = model.to(self.device)
        # self.configure_optimizers()
        # self.load_checkpoint()

        # initialize loss  
    
    
    def configure_optimizers(self):
        
        self.opt = torch.optim.Adam(
                                lr=self.cfg.sr.train.optimizer.lr,
                                amsgrad=False)
            
    
    def load_checkpoint(self):
        pass
            
    def training_step(self):
        
        # SR #
        
        # model
        diffusion = Model.create_model(self.cfg)
        logger.info('Initial Model Finished')
        
        # Train
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_iter = self.cfg.sr.train.n_iter
        
        if self.cfg.path.resume_state:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                current_epoch, current_step))
        
        diffusion.set_new_noise_schedule(
            self.cfg['model']['beta_schedule'][self.cfg['phase']], schedule_phase=self.cfg['phase'])
        
        if self.cfg['phase'] == 'train':
            while current_step < n_iter:
                current_epoch += 1
                for _, train_data in tqdm(enumerate(self.train_iter), total=len(self.train_iter), desc="Processing training data"):
                    current_step += 1
                    if current_step > n_iter:
                        break
                    diffusion.feed_data(train_data)
                    diffusion.optimize_parameters()
                    
                    # log
                    if current_step % self.cfg.sr.train.print_freq == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)

                        if wandb_logger:
                            wandb_logger.log_metrics(logs)

                    # validation
                    if current_step % self.cfg.sr.train.val_freq == 0:
                        avg_psnr = 0.0
                        idx = 0
                        result_path = '{}/{}'.format(self.cfg.path.results, current_epoch)
                        os.makedirs(result_path, exist_ok=True)

                        diffusion.set_new_noise_schedule(
                            self.cfg.model.beta_schedule.val, schedule_phase='val')
                        for _,  val_data in enumerate(self.val_iter):
                            idx += 1
                            diffusion.feed_data(val_data)
                            diffusion.test(continous=False)
                            visuals = diffusion.get_current_visuals()
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                            self.tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                            avg_psnr += Metrics.calculate_psnr(
                                sr_img, hr_img)

                            if wandb_logger:
                                wandb_logger.log_image(
                                    f'validation_{idx}', 
                                    np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                )

                        avg_psnr = avg_psnr / idx
                        diffusion.set_new_noise_schedule(
                            self.cfg.model.beta_schedule.train, schedule_phase='train')
                        # log
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        self.tb_logger.add_scalar('psnr', avg_psnr, current_step)

                        if wandb_logger:
                            wandb_logger.log_metrics({
                                'validation/val_psnr': avg_psnr,
                                'validation/val_step': val_step
                            })
                            val_step += 1

                    if current_step % self.cfg.sr.train.save_checkpoint_freq == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)

                        if wandb_logger and opt['log_wandb_ckpt']:
                            wandb_logger.log_checkpoint(current_epoch, current_step)

                if wandb_logger:
                    wandb_logger.log_metrics({'epoch': current_epoch-1})

            # save model
            logger.info('End of training.')
        else:
            logger.info('Begin Model Evaluation.')
            avg_psnr = 0.0
            avg_ssim = 0.0
            idx = 0
            result_path = '{}'.format(cfg.path.results)
            os.makedirs(result_path, exist_ok=True)
            for _,  val_data in enumerate(self.val_iter):
                idx += 1
                diffusion.feed_data(val_data)
                diffusion.test(continous=True)
                visuals = diffusion.get_current_visuals()

                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                sr_img_mode = 'grid'
                if sr_img_mode == 'single':
                    # single img series
                    sr_img = visuals['SR']  # uint8
                    sample_num = sr_img.shape[0]
                    for iter in range(0, sample_num):
                        Metrics.save_img(
                            Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
                else:
                    # grid img
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    Metrics.save_img(
                        sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                    Metrics.save_img(
                        Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

                Metrics.save_img(
                    hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

                # generation
                eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
                eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim

                if wandb_logger and self.cfg['log_eval']:
                    wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx

            # log
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim))

            if wandb_logger:
                if self.cfg['log_eval']:
                    wandb_logger.log_eval_table()
                wandb_logger.log_metrics({
                    'PSNR': float(avg_psnr),
                    'SSIM': float(avg_ssim)
                })
            
        pass
    
    def validation_step(self):
        pass
    
    def evaluate(self):
        # NOW Benchmark
        pass
    
    def prepare_data(self):
        self.train_dataset = build_datasets.build_train(self.cfg.datasets)
        logger.info("Training data number: ", len(self.train_dataset))
        self.val_dataset = build_datasets.build_val(self.cfg.datasets)
        logger.info("Valuating data number: ", len(self.val_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.datasets.train.num_workers,
                            pin_memory=False,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)
        self.val_iter = iter(self.val_dataloader)
        
        logger.info('Initial Dataset Finished')
    
    def fit(self):
        self.prepare_data()
        
        self.training_step()