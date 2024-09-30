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
import glob
from insightface.utils import face_align
from pathlib import Path
import trimesh


from tensorboardX import SummaryWriter
import logging

import torch
# import torch.distributed as dist  # with the code in load_checkpoint
import model.sr as Model
import core.metrics as Metrics
from lib.MICA.utils import util

import datasets
import random

sys.path.append("./lib/MICA/micalib")
from validator import Validator

# Intial Setting #
        
from lib.MICA.utils.landmark_detector import LandmarksDetector, detectors
from insightface.app.common import Face
from datasets.creation.util import get_arcface_input

input_std = 127.5
input_mean = 127.5

# add by Patipol
# import matplotlib.pyplot as plt
# from skimage.io import imsave
# from PIL import Image, ImageDraw
# import torchvision.transforms as transforms

def print_info(rank):
    props = torch.cuda.get_device_properties(rank)

    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def rescale_tensor(tensor):
    # Rescale tensor from [-1, 1] to [0, 1]
    return (tensor + 1) / 2


class Trainer(object):
    def __init__(self, model, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = [i for i in range(len(self.cfg.device_id))]
        self.batch_size_sr = self.cfg.sr.datasets.train.batch_size
        self.l_image_size = self.cfg.sr.datasets.train.l_resolution
        self.r_image_size = self.cfg.sr.datasets.train.r_resolution
        self.batch_size_mica = self.cfg.mica.datasets.batch_size
        
        self.tb_logger = SummaryWriter(log_dir=self.cfg.path.tb_logger)
        if self.cfg.enable_wandb:
            import wandb
            self.wandb_logger = WandbLogger(opt)
            wandb.define_metric('validation/val_step')
            wandb.define_metric('epoch')
            wandb.define_metric("validation/*", step_metric="val_step")
            val_step = 0
        else:
            self.wandb_logger = None
        
        # SR model
        # model
        self.diffusion = Model.create_model(self.cfg) # -> load_model inside funct.
        logger.info('Initial Model Finished')
        
        # MICA model 
        # self.nfc = model.to(self.device)
        self.nfc = model #!!take a look!! 
        # If there are multiple devices, wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.nfc = nn.DataParallel(self.nfc, device_ids=self.device)
            self.diffusion = nn.DataParallel(self.diffusion, device_ids=self.device)
            # Move model to the primary device
            self.nfc = self.nfc.to(self.device[0]).module
            self.diffusion = self.diffusion.to(self.device[0]).module
        else:
            self.nfc = self.nfc.to(self.device)
            self.diffusion = self.diffusion.to(self.device)

        
        '''
        Exception has occurred: TypeError
        to() received an invalid combination of arguments - got (list), but expected one of:
        * (torch.device device, torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        * (torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        * (Tensor tensor, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        File "/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/lib/trainer.py", line 81, in __init__
            self.nfc = model.to(self.device)
        File "/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/main.py", line 57, in main
            trainer = Trainer(model=nfc, config=cfg)
        File "/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/main.py", line 71, in <module>
            main(cfg)
        TypeError: to() received an invalid combination of arguments - got (list), but expected one of:
        * (torch.device device, torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        * (torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        * (Tensor tensor, bool non_blocking, bool copy, *, torch.memory_format memory_format)
        
        self.device
        [device(type='cuda', index=0), device(type='cuda', index=2)]
        
        --- search in chatgpt
        '''

        self.validator = Validator(self)
        self.configure_optimizers()
        self.load_checkpoint()
        
        # # reset optimizer if loaded from pretrained model
        # if self.cfg.train.reset_optimizer:
        #     self.configure_optimizers()  # reset optimizer
        #     logger.info(f"[TRAINER] Optimizer was reset")

        # if self.cfg.train.write_summary and self.device == 0:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        # print_info(device)

        # initialize loss  
        
    def get_device(self, device, device_id):
        if device == 'cuda' and torch.cuda.is_available():
            return [torch.device(f'cuda:{i}') for i in device_id]
        else:
            return torch.device('cpu')
    
    
    def configure_optimizers(self):
        # sr
        # self.opt_sr = torch.optim.Adam(
        #                         lr=self.cfg.sr.train.optimizer.lr,
        #                         amsgrad=False)
        # mica
        
        params = self.nfc.parameters_to_optimize()
        
        self.opt = torch.optim.AdamW(
            lr=self.cfg.mica.train.lr,
            weight_decay=self.cfg.mica.train.weight_decay,
            params=params,
            amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.1)
            
    
    def load_checkpoint(self):
        
        # SR
        # Already in the model.py for SR
        
        # mica
        self.epoch = 0
        self.global_step = 0
        # dist.barrier() # !!take a look!! for multi gpu
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device} # !!take a look!! for multi gpu
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device[0]}
        model_path = os.path.join(self.cfg.mica.output_dir, 'model_mica.tar')
        if os.path.exists(self.cfg.mica.pretrained_model_path):
            model_path = self.cfg.mica.pretrained_model_path
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location)
            if 'opt' in checkpoint:
                self.opt.load_state_dict(checkpoint['opt'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'epoch' in checkpoint:
                self.current_epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                self.global_step = checkpoint['global_step']
            logger.info(f"[TRAINER] Resume training from {model_path}")
            logger.info(f"[TRAINER] Start from step {self.global_step}")
            logger.info(f"[TRAINER] Start from epoch {self.epoch}")
        else:
            logger.info('[TRAINER] Model path not found, start training from scratch')
            
    def save_checkpoint(self, filename):
        
        model_dict = self.nfc.model_dict()

        model_dict['opt'] = self.opt.state_dict()
        model_dict['scheduler'] = self.scheduler.state_dict()
        model_dict['validator'] = self.validator.state_dict()
        model_dict['epoch'] = self.current_epoch
        model_dict['global_step'] = self.global_step
        model_dict['batch_size'] = self.batch_size_mica

        torch.save(model_dict, filename)
            
    def filter_and_slice_train_data(self, train_data, order, keys_to_keep_and_slice = {'HR', 'SR', 'LR'}, keys_to_keep = {'Index', 'imagename'}):
    
        # Create a new dictionary with the specified keys
        dict_a = {}
        for key in train_data:
            if key in keys_to_keep_and_slice:
                dict_a[key] = train_data[key][order]  # Get the order sublist
            elif key in keys_to_keep:
                dict_a[key] = train_data[key]
        
        return dict_a
    
            
    def training_step(self):
        
        # app = LandmarksDetector(model=detectors.RETINAFACE)
        
        # SR #

        # Train
        self.global_step = self.diffusion.begin_step
        self.current_epoch = self.diffusion.begin_epoch # !!take a look!! for use the same parameter
        n_iter = self.cfg.sr.train.n_iter
        if self.cfg.sr.pretrained_model_path:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                self.current_epoch, self.global_step))
        
        self.diffusion.set_new_noise_schedule(
            self.cfg['sr']['model']['beta_schedule'][self.cfg['phase']], schedule_phase=self.cfg['phase'])
        
        if self.cfg['phase'] == 'train':
            iters_every_epoch = int(len(self.train_dataset) / self.batch_size_mica)+1
            while self.global_step < n_iter + self.cfg.mica.train.max_steps: # !!take a look!! if train only SR -> set mica steps to be 0
                self.current_epoch += 1
                for _, train_data in tqdm(enumerate(self.train_iter), total=len(self.train_iter), desc="Processing training data"):
                    
                    # SR # -------------------------------------------
                    train_data_sub = self.filter_and_slice_train_data(train_data, order = 0)

                    for i in range(1,len(train_data['SR'])): # make k x b, c, h, w
                        _train_data_sub = self.filter_and_slice_train_data(train_data, order = i)
                        for key in train_data_sub:
                            if isinstance(train_data_sub[key], torch.Tensor):
                                train_data_sub[key] = torch.cat([train_data_sub[key], _train_data_sub[key]], dim=0)
                            elif isinstance(train_data_sub[key], list):
                                train_data_sub[key].extend(_train_data_sub[key])
                    
                    if self.global_step > n_iter + self.cfg.mica.train.max_steps:
                        break
                    
                    self.diffusion.feed_data(train_data_sub)
                    loss_sr = self.diffusion.optimize_parameters()
                
                
                    # log
                    if self.global_step % self.cfg.sr.train.print_freq == 0:
                        logs = self.diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}, sub-iter:{:2,d}> '.format(
                            self.current_epoch, self.global_step, i)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            self.tb_logger.add_scalar(k, v, self.global_step)
                        logger.info(message)

                        if self.wandb_logger:
                            self.wandb_logger.log_metrics(logs)

                    # validation
                    if self.global_step % self.cfg.sr.train.val_freq == 0:
                        avg_psnr = 0.0
                        idx = 0
                        result_path = '{}/{}'.format(self.cfg.path.results, self.current_epoch)
                        os.makedirs(result_path, exist_ok=True)

                        self.diffusion.set_new_noise_schedule(
                            self.cfg.sr.model.beta_schedule.val, schedule_phase='val')
                        for _,  val_data in enumerate(self.val_iter):
                            idx += 1
                            self.diffusion.feed_data(val_data)
                            self.diffusion.test(continous=False)
                            visuals = self.diffusion.get_current_visuals()
                            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                            # generation
                            Metrics.save_img(
                                hr_img, '{}/{}_{}_hr.png'.format(result_path, self.global_step, idx))
                            Metrics.save_img(
                                sr_img, '{}/{}_{}_sr.png'.format(result_path, self.global_step, idx))
                            Metrics.save_img(
                                lr_img, '{}/{}_{}_lr.png'.format(result_path, self.global_step, idx))
                            Metrics.save_img(
                                fake_img, '{}/{}_{}_inf.png'.format(result_path, self.global_step, idx))
                            self.tb_logger.add_image(
                                'Iter_{}'.format(self.global_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                            avg_psnr += Metrics.calculate_psnr(
                                sr_img, hr_img)

                            if self.wandb_logger:
                                self.wandb_logger.log_image(
                                    f'validation_{idx}', 
                                    np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                )
                                
                                

                        avg_psnr = avg_psnr / idx
                        # reset dataloader    
                        self.val_iter = iter(self.val_dataloader)
                        
                        # log
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            self.current_epoch, self.global_step, avg_psnr))
                        # tensorboard logger
                        self.tb_logger.add_scalar('psnr', avg_psnr, self.global_step)

                        if self.wandb_logger:
                            self.wandb_logger.log_metrics({
                                'validation/val_psnr': avg_psnr,
                                'validation/val_step': val_step
                            })
                            val_step += 1

                    if self.global_step % self.cfg.sr.train.save_checkpoint_freq == 0:
                        logger.info('Saving models and training states.')
                        self.diffusion.save_network(self.current_epoch, self.global_step)

                        if self.wandb_logger and opt['log_wandb_ckpt']:
                            self.wandb_logger.log_checkpoint(self.current_epoch, self.global_step)
                    
                    # MICA # ------------------------------------------- 
                    images_list = []
                    arcface_list = []
                    self.diffusion.set_new_noise_schedule(
                                self.cfg.sr.model.beta_schedule.val, schedule_phase='train') # !! Train or Val
                    for i in range(len(train_data['SR'])):
                        for j in range(len(self.filter_and_slice_train_data(train_data, order = i)['SR'])):
                            train_data_sub['SR'] = self.filter_and_slice_train_data(train_data, order = i)['SR'][j].unsqueeze(0)
                            train_data_sub['HR'] = self.filter_and_slice_train_data(train_data, order = i)['HR'][j].unsqueeze(0)
                            train_data_sub['Index'] = self.filter_and_slice_train_data(train_data, order = i)['Index'][j]
                            self.diffusion.feed_data(train_data_sub)
                            self.diffusion.test(continous=False)
                            visuals = self.diffusion.get_current_visuals()
                            
                            sr_img = Metrics.tensor2img(visuals['SR'])
                            sr_up_img = (cv2.resize(sr_img, (224, 224)))
                            
                            if self.cfg.mica.train.arcface_new:
                                # temp_arcface = self.create_arcface_MICA(sr_up_img, train_data['imagename'][j], i) #!!! if the result good, can delete it
                                temp_arcface = self.create_arcface_MICA(sr_up_img)
                                arcface_list.append(torch.tensor(temp_arcface))
                            
                            ### test image ####
                            if self.global_step % 500 == 0:
                                # sr_img = Metrics.tensor2img(visuals['SR'])
                                os.makedirs(os.path.join(self.cfg.output_dir, 'test_train'), exist_ok=True)
                                Metrics.save_img(sr_up_img, '{}/test_train/{}_{}_{}_test_sr.png'.format(self.cfg.output_dir, self.global_step, i, j))
                            ### test image ####
                            
                            images_list.append(sr_up_img)
                    
                    images_list = [torch.from_numpy(image) if isinstance(image, np.ndarray) else image for image in images_list]
                    images_array = torch.stack(images_list).view(train_data['image'].shape)
                    arcface_array = torch.stack(arcface_list).view(train_data['arcface'].shape)
                    
                    # Use the output from the SR feed to MICA
                    batch = self.filter_and_slice_train_data(train_data, 0, keys_to_keep_and_slice = {}, keys_to_keep = {'image', 'arcface', 'imagename', 'dataset', 'flame'})
                    batch['image'] = images_array
                    batch['arcface'] = arcface_array
                    
                    # reset setting of training diffusion
                    self.diffusion.set_new_noise_schedule(
                        self.cfg.sr.model.beta_schedule.train, schedule_phase='train')
                    
                    visualizeTraining = self.global_step % self.cfg.mica.train.vis_steps == 0
                    
                    self.opt.zero_grad()
                    losses, opdict = self.training_MICA(batch)

                    loss_mica = losses['all_loss']
                    losses['L1'] = loss_sr
                    
                    loss_mica = loss_mica.to(self.device[0])
                    loss_sr = loss_sr.to(self.device[0])
                    all_loss = loss_mica + loss_sr
                    gradient = torch.ones_like(all_loss)
                    all_loss.backward(gradient)
                    self.opt.step()
                    self.diffusion.optG.step()
                    

                    if self.global_step % self.cfg.mica.train.log_steps == 0 and self.device[0] == 0:
                        loss_info = f"\n" \
                                    f"  Epoch: {self.current_epoch}\n" \
                                    f"  Step: {self.global_step}\n" \
                                    f"  Iter: {self.global_step}/{iters_every_epoch}\n" \
                                    f"  LR: {self.opt.param_groups[0]['lr']}\n" \
                                    f"  Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}\n"
                        text = '[MICA] '
                        for k, v in losses.items():
                            if k != 'all_loss':
                                loss_info = loss_info + f'  {text + k}: {v:.4f}\n'
                            else:
                                text = '[SR] '
                            if self.cfg.mica.train.write_summary:
                                self.tb_logger.add_scalar('train_loss/' + k, v, global_step=self.global_step)
                        logger.info(loss_info)

                    if visualizeTraining and self.device[0] == 0:
                        visdict = {
                            'input_images': opdict['images'],
                        }
                        # add images to tensorboard
                        for k, v in visdict.items():
                            self.tb_logger.add_images(k, np.clip(v.detach().cpu(), 0.0, 1.0), self.global_step)

                        pred_canonical_shape_vertices = torch.empty(0, 3, 512, 512).cuda()
                        flame_verts_shape = torch.empty(0, 3, 512, 512).cuda()
                        deca_images = torch.empty(0, 3, 512, 512).cuda()
                        input_images = torch.empty(0, 3, 224, 224).cuda()
                        L = opdict['pred_canonical_shape_vertices'].shape[0]
                        S = 4 if L > 4 else L                    
                        for n in np.random.choice(range(L), size=S, replace=False):
                            
                            rendering = self.nfc.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                            pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                            rendering = self.nfc.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                            flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                            input_images = torch.cat([input_images, opdict['images'][n:n + 1, ...]])
                            if 'deca' in opdict:
                                deca = self.nfc.render.render_mesh(opdict['deca'][n:n + 1, ...])
                                deca_images = torch.cat([deca_images, deca])
                            


                        visdict = {}

                        if 'deca' in opdict:
                            visdict['deca'] = deca_images

                        visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices
                        visdict["flame_verts_shape"] = flame_verts_shape
                        visdict["images"] = input_images

                        savepath = os.path.join(self.cfg.output_dir, 'train_images_mica/train_{}_{}.jpg'.format(self.current_epoch, self.global_step))
                        util.visualize_grid(visdict, savepath, size=512)

                    if self.global_step % self.cfg.mica.train.val_steps == 0:
                        self.validation_MICA()

                    if self.global_step % self.cfg.mica.train.lr_update_step == 0:
                        self.scheduler.step()

                    if self.global_step % self.cfg.mica.train.eval_steps == 0:
                        self.evaluate_MICA()

                    if self.global_step % self.cfg.mica.train.checkpoint_steps == 0:
                        self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_mica' + '.tar'))

                    if self.global_step % self.cfg.mica.train.checkpoint_epochs_steps == 0:
                        self.save_checkpoint(os.path.join(self.cfg.path.checkpoint_mica, 'model_' + str(self.global_step) + '.tar'))
                    
                    self.global_step += 1

                if self.wandb_logger:
                    self.wandb_logger.log_metrics({'epoch': self.current_epoch-1})
                # reset dataloader    
                self.train_iter = iter(self.train_dataloader)

             
            # save model
            self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model_mica' + '.tar'))  
            logger.info('End of training.')
        else:
            logger.info('Begin Model Evaluation.')
            avg_psnr = 0.0
            avg_ssim = 0.0
            idx = 0
            result_path = '{}'.format(self.cfg.path.results_val)
            os.makedirs(result_path, exist_ok=True)
            faces = self.nfc.flameModel.generator.faces_tensor.cpu()
            self.nfc.testing = True
            for _,  val_data in tqdm(enumerate(self.val_iter), total=len(self.val_iter), desc="Processing training data"):
                idx += 1
                self.diffusion.feed_data(val_data)
                self.diffusion.test(continous=False)
                visuals = self.diffusion.get_current_visuals()

                hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                
                name = os.path.basename(val_data['path_sr'][0])[:-4]

                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(sr_img, '{}/{}_sr.png'.format(result_path, name))

                Metrics.save_img(
                    hr_img, '{}/{}_hr.png'.format(result_path, name))
                Metrics.save_img(
                    fake_img, '{}/{}_inf.png'.format(result_path, name))
                Metrics.save_img(
                    lr_img, '{}/{}_lr.png'.format(result_path, name))

                # MICA 
                
                sr_up_img = cv2.resize(sr_img, (224, 224))
                temp_arcface = self.create_arcface_MICA(sr_up_img)
                temp_arcface = torch.tensor(temp_arcface).cuda()[None]
                
                sr_up_img = sr_up_img / 255.
                sr_up_img = sr_up_img.transpose(2, 0, 1)
                sr_up_img = torch.tensor(sr_up_img).cuda()[None]
                
                ### test image ####
                if idx % 10 == 0:
                    # sr_img = Metrics.tensor2img(visuals['SR'])
                    os.makedirs(os.path.join(self.cfg.output_dir, 'test_val'), exist_ok=True)
                    Metrics.save_img(sr_up_img, '{}/test_val/{}_{}_test_arcface_sr.png'.format(self.cfg.output_dir, self.global_step, idx))
                ### test image ####
                
                self.nfc.eval()
                encoder_output = self.nfc.encode(sr_up_img, temp_arcface)
                opdict = self.nfc.decode(encoder_output)
                meshes = opdict['pred_canonical_shape_vertices']
                code = opdict['pred_shape_code']
                lmk = self.nfc.flame.compute_landmarks(meshes)

                mesh = meshes[0]
                landmark_51 = lmk[0, 17:]
                landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

                
                savepath = os.path.join(self.cfg.output_dir, 'val_images_mica')
                
                dst = Path(savepath, name)
                dst.mkdir(parents=True, exist_ok=True)
                trimesh.Trimesh(vertices=mesh.detach().cpu().numpy() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
                trimesh.Trimesh(vertices=mesh.detach().cpu().numpy() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
                np.save(f'{dst}/identity', code[0].detach().cpu().numpy())
                np.save(f'{dst}/kpt7', landmark_7.detach().cpu().numpy() * 1000.0)
                np.save(f'{dst}/kpt68', lmk.detach().cpu().numpy() * 1000.0)



                # generation
                eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR']), hr_img)
                eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR']), hr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim
                # compute_loss mica
                if self.wandb_logger and self.cfg['log_eval']:
                    self.wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx

            # log
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
                self.current_epoch, self.global_step, avg_psnr, avg_ssim))

            if self.wandb_logger:
                if self.cfg['log_eval']:
                    self.wandb_logger.log_eval_table()
                self.wandb_logger.log_metrics({
                    'PSNR': float(avg_psnr),
                    'SSIM': float(avg_ssim)
                })

    
    def training_MICA(self, batch):
        self.nfc.train() # loop here!!

        images = batch['image'].to(self.device[0]) # !!take a look!! for multi gpu
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device[0]) # !!take a look!! for multi gpu

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }
        encoder_output = self.nfc.encode(images, arcface)
        encoder_output['flame'] = flame
        decoder_output = self.nfc.decode(encoder_output, self.current_epoch)
        losses = self.nfc.compute_losses(inputs, encoder_output, decoder_output)

        all_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        losses['all_loss'] = all_loss

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return losses, opdict
    
    def validation_MICA(self):
        self.validator.run()
    
    def evaluate_MICA(self):
        # NOW Benchmark
        pass
    
    def create_arcface_MICA(self, array_image):

        
        blob = cv2.dnn.blobFromImages([array_image], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True) # cv2.resize(img, (224, 224))
        
        
        return blob[0]
    
    def prepare_data(self):
        generator = torch.Generator()
        generator.manual_seed(int(self.cfg.gpu_ids[0]))
        if self.cfg.phase != 'val':
            self.train_dataset, total_images = datasets.build_train(self.cfg, self.device)
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size =self.batch_size_mica,
                num_workers=self.cfg.mica.datasets.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=generator)

            self.train_iter = iter(self.train_dataloader)
        # logger.info(f'[TRAINER] Training dataset is ready with {len(self.train_dataset)} actors and {total_images} images.')
        # self.val_dataset, total_images = datasets.build_val(self.cfg, self.device)
        # self.val_dataloader = DataLoader(
        #     self.val_dataset, batch_size =self.batch_size_mica,
        #     num_workers=self.cfg.mica.datasets.num_workers,
        #     shuffle=True,
        #     pin_memory=True,
        #     drop_last=False,
        #     worker_init_fn=seed_worker,
        #     generator=generator)

        # self.val_iter = iter(self.val_dataloader)
        
        # sr - val
        for phase, dataset_opt in self.cfg['sr']['datasets'].items(): # make a real val for sr in one system !!take a look!!
            # if phase == 'train' and self.cfg.phase != 'val':
            #     train_set = datasets.create_dataset(dataset_opt, phase)
            #     self.train_dataloader = datasets.create_dataloader(
            #         train_set, dataset_opt, phase)
            #     self.train_iter = iter(self.train_dataloader)
            if phase == 'val':
                val_set = datasets.create_dataset(dataset_opt, phase)
                self.val_dataloader = datasets.create_dataloader(
                    val_set, dataset_opt, phase)
                self.val_iter = iter(self.val_dataloader)
        
        logger.info('Initial Dataset Finished')
    
    def fit(self):
        self.prepare_data()
        
        self.training_step()