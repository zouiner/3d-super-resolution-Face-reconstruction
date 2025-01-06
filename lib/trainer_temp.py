import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from time import time
from skimage.io import imread
import cv2
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import glob
from pathlib import Path
import trimesh


from tensorboardX import SummaryWriter
import logging

import torch
# import torch.distributed as dist  # with the code in load_checkpoint
import core.metrics as Metrics
from lib.MICA.utils import util

import datasets
import random

# Model
from model.sr3d.model import ThreeDSuperResolutionModel

# Intial Setting #
        

from datasets.creation.util import get_arcface_input

input_std = 127.5
input_mean = 127.5

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
    def __init__(self, config=None, device=None):
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
            self.wandb_logger = WandbLogger(cfg)
            wandb.define_metric('validation/val_step')
            wandb.define_metric('epoch')
            wandb.define_metric("validation/*", step_metric="val_step")
            val_step = 0
        else:
            self.wandb_logger = None
        
        
        # model
        self.model = ThreeDSuperResolutionModel(self.cfg, self.device)
        logger.info('Initial Model Finished')
        
        # If there are multiple devices, wrap the model with DataParallel
        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.device)
        #     self.model = self.model.module
        self.model = self.model.cuda()

        # self.validator = Validator(self)
        self.configure_optimizers()
        self.load_checkpoint()
        
        # reset optimizer if loaded from pretrained model
        if self.cfg.train.reset_optimizer:
            self.configure_optimizers()  # reset optimizer
            logger.info(f"[TRAINER] Optimizer was reset")

        # if self.cfg.train.write_summary and self.device == 0:
        #     from torch.utils.tensorboard import SummaryWriter
        #     self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))

        # print_info(device)

        # initialize loss
    
    
    def configure_optimizers(self):
        # sr
        self.model.train()
        # find the parameters to optimize
        if self.cfg.sr['model']['finetune_norm']:
            optim_params = []
            for k, v in self.model.sr_model.named_parameters():
                v.requires_grad = False
                if k.find('transformer') >= 0:
                    v.requires_grad = True
                    v.data.zero_()
                    optim_params.append(v)
                    logger.info(
                        'Params [{:s}] initialized to 0 and will optimize.'.format(k))
        else:
            optim_params = list(self.model.sr_model.parameters())
    
        self.opt_sr = torch.optim.Adam(optim_params, lr=self.cfg.sr['train']["optimizer"]["lr"])
    # mica
    
        params = self.model.parameters_to_optimize() # !! can add into the model
        
        self.opt_mica = torch.optim.AdamW(
            lr=self.cfg.mica.train.lr,
            weight_decay=self.cfg.mica.train.weight_decay,
            params=params,
            amsgrad=False)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt_mica, step_size=1, gamma=0.1)
            
    
    def load_checkpoint(self):
        """
        Load both SR and MICA model checkpoints from one file.
        """
        self.current_epoch = 0
        self.global_step = 0

        # Path to the combined checkpoint file
        checkpoint_dir = self.cfg['path']['checkpoint']
        load_path = sorted(glob.glob(f"{checkpoint_dir}/*"))
        pretrained_model = self.cfg['sr']['pretrained_model_path']

        if len(load_path) > 0:
            load_path = load_path[-1]  # Get the latest checkpoint file (the last file in the sorted list)
        else:
            load_path = None  # No files found, set to None
            
        if self.cfg['phase'] == 'val' and os.path.exists(os.path.join(self.cfg.output_dir,'model.tar')):
            
            load_path = os.path.join(self.cfg.output_dir, 'model.tar')
        
        if self.cfg.checkpoint:
            
            load_path = self.cfg.checkpoint


        if load_path and os.path.exists(load_path):
            logger.info(f'Loading combined checkpoint from [{load_path}]')
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'), weights_only=True)

            # Add `module.` prefix if the model is distributed
            sr_state_dict = checkpoint['sr_model_state']
            if isinstance(self.model.sr_model, torch.nn.parallel.DistributedDataParallel):
                sr_state_dict = {f"module.{k}" if not k.startswith("module.") else k: v for k, v in sr_state_dict.items()}
            unexpected_keys = sr_state_dict.keys() - self.model.sr_model.state_dict().keys()
            self.model.sr_model.load_state_dict(sr_state_dict, strict=False)

            mica_state_dict = checkpoint['mica_model_state']
            if isinstance(self.model.mica_model, torch.nn.parallel.DistributedDataParallel):
                mica_state_dict = {f"module.{k}" if not k.startswith("module.") else k: v for k, v in mica_state_dict.items()}
            self.model.mica_model.load_state_dict(mica_state_dict, strict=False)

            # Load other states
            self.opt_sr.load_state_dict(checkpoint['sr_optimizer_state'])
            self.opt_mica.load_state_dict(checkpoint['mica_optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.batch_size_mica = checkpoint['batch_size_mica']
            
            logger.info(f'Resumed training from epoch {self.current_epoch}, step {self.global_step}')
        elif pretrained_model:
            logger.info(f'[SR] Load pretrained model for G [{pretrained_model}]')
            
            # Define paths for generator and optimizer files
            gen_path = '{}_gen.pth'.format(pretrained_model)
            opt_path = '{}_opt.pth'.format(pretrained_model)
            
            # Check if the generator model file exists
            if os.path.exists(gen_path):
                # Load the SR model (generator)
                sr_network = self.model.sr_model
                
                # Load the generator state with strict=False in case there are extra keys
                sr_network.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu'), weights_only=True), strict=False) # , weights_only=True ??
                logger.info(f'[SR] Loaded pretrained SR model from [{gen_path}]')
                
                # If the optimizer path exists, load the optimizer state (optional)
                if os.path.exists(opt_path) and self.cfg['phase'] == 'train':
                    opt = torch.load(opt_path, map_location='cpu')
                    self.opt_sr.load_state_dict(opt['optimizer'])
                    self.global_step = opt.get('iter', 0)  # default to 0 if not in checkpoint
                    self.current_epoch = opt.get('epoch', 0)  # default to 0 if not in checkpoint
                    logger.info(f'[SR] Loaded optimizer state from [{opt_path}]')
            else:
                logger.info(f'[SR] No optimizer state fooud on ', pretrained_model, ' path')
                logger.info(f'[SR] No optimizer state found for pretrained SR model, continuing from scratch.')
                
        else:
            logger.info('No checkpoint found, starting from scratch.')

            
    def save_checkpoint(self, checkpoint_dir = None):
        """
        Save both SR and MICA model checkpoints into one file.
        """
        
        if not checkpoint_dir:
            checkpoint_dir = os.path.join(self.cfg.output_dir, self.cfg['path']['checkpoint'])
            
            # Define a single file to save the checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'I{self.global_step}_E{self.current_epoch}_checkpoint.pth')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = checkpoint_dir
            
            
        
        
        
        
        # Prepare dictionary to hold all information
        checkpoint = {
            # SR Model
            'sr_model_state': self.model.sr_model.state_dict() if not isinstance(self.model.sr_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.model.sr_model.module.state_dict(),
            'sr_optimizer_state': self.opt_sr.state_dict(),
            
            # MICA Model
            'mica_model_state': self.model.mica_model.state_dict() if not isinstance(self.model.mica_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.model.mica_model.module.state_dict(),
            'mica_optimizer_state': self.opt_mica.state_dict(),
            
            # Scheduler states
            'scheduler_state': self.scheduler.state_dict(),
            
            # Other states
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'batch_size_mica': self.batch_size_mica
        }
        
        # Save the combined dictionary to one file
        torch.save(checkpoint, checkpoint_path)
        
        # if self.rank == 0:
        #     logger.info(f'Saved a checkpoint in [{checkpoint_path}]')
        logger.info(f'Saved a checkpoint in [{checkpoint_path}]')


            
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
        
        n_iter = self.cfg.sr.train.n_iter
        
        if self.cfg.sr.pretrained_model_path:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                self.current_epoch, self.global_step))
        
        self.model.set_new_noise_schedule(
            self.cfg['sr']['model']['beta_schedule'][self.cfg['phase']], schedule_phase=self.cfg['phase'])
        
        
        
        if self.cfg['phase'] == 'train':
            iters_every_epoch = int(len(self.train_dataset) / self.batch_size_mica)+1
            while self.global_step < n_iter + self.cfg.mica.train.max_steps: # !!take a look!! if train only SR -> set mica steps to be 0
                self.current_epoch += 1
                for _, train_data in tqdm(enumerate(self.train_iter), total=len(self.train_iter), desc="Processing training data"):
                    
                    visualizeTraining = self.global_step % self.cfg.train.vis_steps == 0
                    
                    if self.global_step > n_iter + self.cfg.mica.train.max_steps:
                        break
                    
                    sr_train_data = self.model.preprocess_sr_data(train_data)
                    
                    # MICA # ------------------------------------------- 
                    images_list = []
                    arcface_list = []
                    _sr_train_data = sr_train_data
                    self.model.set_new_noise_schedule(self.cfg.sr.model.beta_schedule[self.cfg['phase']], schedule_phase= self.cfg['phase']) # for visual SR to fedd into mica
                    for i in range(len(train_data['SR'])):
                        for j in range(len(self.filter_and_slice_train_data(train_data, order = i)['SR'])):
                            _sr_train_data['SR'] = self.filter_and_slice_train_data(train_data, order = i)['SR'][j].unsqueeze(0)
                            _sr_train_data['HR'] = self.filter_and_slice_train_data(train_data, order = i)['HR'][j].unsqueeze(0)
                            _sr_train_data['Index'] = self.filter_and_slice_train_data(train_data, order = i)['Index'][j].unsqueeze(0)
                            # self.model.feed_data(_sr_train_data)
                            # tensor_sr, visuals = self.model.get_tensor_sr_img(t_output = True, v_output = True)
                            # visuals = self.model.get_current_visuals()
                            _sr_train_data = self.model.feed_data(_sr_train_data)
                            tensor_sr, visuals = self.model(_sr_train_data, t_output = True, v_output = True) # -> (3,16,16)
                            
                            
                            sr_img = Metrics.tensor2img(visuals['SR'])
                            temp_sr_up_img = Metrics.tensor2tensor_img(tensor_sr) * 255.0
                            
                            
                            if self.cfg.mica.train.arcface_new:
                                temp_arcface = self.model.create_tensor_blob(temp_sr_up_img)
                                arcface_list.append(temp_arcface.clone().detach().requires_grad_(True))
                            
                            
                            if visualizeTraining:
                                savepath = os.path.join(self.cfg.output_dir, 'train_images/{}_{}'.format(self.current_epoch, self.global_step))
                                os.makedirs(savepath, exist_ok=True)
                                Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(savepath, i, j))
                                hr_img = Metrics.tensor2img(visuals['HR'])
                                inf_img = Metrics.tensor2img(visuals['INF'])
                                Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(savepath, i, j))
                                Metrics.save_img(inf_img, '{}/{}_{}_inf.png'.format(savepath, i, j))
                            
                            sr_up_img = (cv2.resize(sr_img, (224, 224)))
                            images_list.append(sr_up_img.transpose(2,0,1)/255) # image no need to be tensor becasue mica use only arcface
                    
                    images_list = [torch.from_numpy(image) if isinstance(image, np.ndarray) else image for image in images_list]
                    images_array = torch.stack(images_list).view(train_data['image'].shape)
                    arcface_array = torch.stack(arcface_list).view(train_data['arcface'].shape)
                    
                    # Use the output from the SR feed to MICA
                    batch = self.filter_and_slice_train_data(train_data, 0, keys_to_keep_and_slice = {}, keys_to_keep = {'image', 'arcface', 'imagename', 'dataset', 'flame'})
                    batch['image'] = images_array
                    batch['arcface'] = arcface_array
                    
                    input_mica, opdict, encoder_output, decoder_output = self.model.training_MICA(batch, self.current_epoch)
                    
                    self.opt_sr.zero_grad()
                    self.opt_mica.zero_grad()
                    sr_train_data = self.model.feed_data(sr_train_data)
                    l_sr, l_mica, losses = self.model.compute_loss(sr_train_data, input_mica, encoder_output, decoder_output)
                    
                    
                    l_mica = l_mica.cuda()
                    l_sr = l_sr.cuda()

                    # Combine losses
                    alpha = 0.5 # !!! weight for loss contorlization
                    beta = 0.5 # !!! weight for loss contorlization
                    combined_loss = alpha * l_sr +  beta * l_mica
                    
                    losses['L1'] = l_sr
                    losses['pred_verts_shape_canonical_diff'] = l_mica
                    
                    # Backward pass
                    combined_loss.backward()

                    # Update optimizers
                    self.opt_sr.step()
                    self.opt_mica.step()
                    
                    
                    # Dynamic weighting based on gradients
                    '''
                    Summary
                    If the ranges of the losses are different:

                    Normalize or scale the losses to a similar range.
                    Use weights to balance their contributions to the total loss.
                    If left unaddressed, the imbalance can lead to poor or unstable training outcomes. Proper normalization ensures both losses contribute effectively to the optimization process.
                    '''
                    # grad_sr = torch.autograd.grad(l_sr, sr_model.parameters(), retain_graph=True)
                    # grad_mica = torch.autograd.grad(l_mica, mica_model.parameters(), retain_graph=True)

                    # norm_sr = sum(g.norm() for g in grad_sr)
                    # norm_mica = sum(g.norm() for g in grad_mica)

                    # alpha = norm_sr / (norm_sr + norm_mica)
                    # combined_loss = alpha * l_sr + (1 - alpha) * l_mica
                    
                    
                    if self.global_step % self.cfg.train.log_steps == 0:
                        loss_info = f"\n" \
                                    f"  Epoch: {self.current_epoch}\n" \
                                    f"  Step: {self.global_step}\n" \
                                    f"  Iter: {self.global_step}/{iters_every_epoch}\n" \
                                    f"  LR_sr: {self.opt_sr.param_groups[0]['lr']}\n" \
                                    f"  LR_mica: {self.opt_mica.param_groups[0]['lr']}\n" \
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
                    
                    if visualizeTraining:
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
                            
                            rendering = self.model.render.render_mesh(opdict['pred_canonical_shape_vertices'][n:n + 1, ...])
                            pred_canonical_shape_vertices = torch.cat([pred_canonical_shape_vertices, rendering])
                            rendering = self.model.render.render_mesh(opdict['flame_verts_shape'][n:n + 1, ...])
                            flame_verts_shape = torch.cat([flame_verts_shape, rendering])
                            input_images = torch.cat([input_images, opdict['images'].cuda()[n:n + 1, ...]])
                            if 'deca' in opdict:
                                deca = self.model.render.render_mesh(opdict['deca'][n:n + 1, ...])
                                deca_images = torch.cat([deca_images, deca])
                            


                        visdict = {}

                        if 'deca' in opdict:
                            visdict['deca'] = deca_images

                        visdict["pred_canonical_shape_vertices"] = pred_canonical_shape_vertices
                        visdict["flame_verts_shape"] = flame_verts_shape
                        visdict["images"] = input_images

                        savepath = os.path.join(savepath, 'train_3d.jpg')
                        util.visualize_grid(visdict, savepath, size=512, return_gird = False)
                    
                    # log
                    if self.global_step % self.cfg.train.print_freq == 0:
                        logs = self.model.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}, sub-iter:{:2,d}> '.format(
                            self.current_epoch, self.global_step, i)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            self.tb_logger.add_scalar(k, v, self.global_step)
                        logger.info(message)

                        if self.wandb_logger:
                            self.wandb_logger.log_metrics(logs)

                    

                    if self.global_step % self.cfg.mica.train.lr_update_step == 0:
                        self.scheduler.step() #!! ???

                    if self.global_step % self.cfg.mica.train.eval_steps == 0:
                        self.evaluate_MICA() #!!

                    if self.global_step % self.cfg.train.checkpoint_steps == 0:
                        logger.info('Saving models and training states.')
                        self.save_checkpoint()
                        
                        if self.wandb_logger and self.cfg['log_wandb_ckpt']:
                            self.wandb_logger.log_checkpoint(self.current_epoch, self.global_step)
                    
                    self.global_step += 1
                
                self.train_iter = iter(self.train_dataloader)
            
            # save model
            self.save_checkpoint(os.path.join(self.cfg.output_dir, 'model' + '.tar'))  
            logger.info('End of training.')
            
        else:
            logger.info('Begin Model Evaluation.')
            avg_psnr = 0.0
            avg_ssim = 0.0
            idx = 0
            faces = self.model.mica_model.generator.faces_tensor.cpu()
            self.model.testing = True
            self.model.eval()
            with torch.no_grad():
                for _,  val_data in tqdm(enumerate(self.val_iter), total=len(self.val_iter), desc="Processing training data"):
                    idx += 1
                    for k in range(self.cfg.sample):
                        val_data = self.model.feed_data(val_data)
                        visuals = self.model(val_data, v_output = True)

                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        

                        # MICA 
                        
                        sr_up_img = cv2.resize(sr_img, (224, 224))
                        temp_arcface = self.model.create_arcface_embeddings(sr_up_img)
                        temp_arcface = torch.tensor(temp_arcface).cuda()[None]
                        
                        sr_up_img = sr_up_img / 255.
                        sr_up_img = sr_up_img.transpose(2, 0, 1)
                        sr_up_img = torch.tensor(sr_up_img).cuda()[None]
                        
                        self.model.eval()
                        encoder_output = self.model.encode_mica(sr_up_img, temp_arcface)
                        opdict = self.model.decode_mica(encoder_output)
                        meshes = opdict['pred_canonical_shape_vertices']
                        code = opdict['pred_shape_code']
                        lmk = self.model.flame.compute_landmarks(meshes)

                        mesh = meshes[0]
                        landmark_51 = lmk[0, 17:]
                        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

                        
                        if self.cfg.sample == 1:
                            name = os.path.basename(val_data['path_sr'][0])[:-4]
                            savepath = os.path.join(self.cfg.output_dir, 'val_images', '{}_{}'.format(self.current_epoch, self.global_step))
                        else:
                            name = os.path.basename(val_data['path_sr'][0])[:-4] + '_' + str(k).zfill(len(str(self.cfg.sample)))
                            savepath = os.path.join(self.cfg.output_dir, 'val_images', '{}_{}_s{}'.format(self.current_epoch, self.global_step,self.cfg.sample))
                        
                        dst = Path(savepath, name)
                        dst.mkdir(parents=True, exist_ok=True)
                        trimesh.Trimesh(vertices=mesh.detach().cpu().numpy() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.ply')  # save in millimeters
                        trimesh.Trimesh(vertices=mesh.detach().cpu().numpy() * 1000.0, faces=faces, process=False).export(f'{dst}/mesh.obj')
                        np.save(f'{dst}/identity', code[0].detach().cpu().numpy())
                        np.save(f'{dst}/kpt7', landmark_7.detach().cpu().numpy() * 1000.0)
                        np.save(f'{dst}/kpt68', lmk.detach().cpu().numpy() * 1000.0)

                        
                        Metrics.save_img(sr_img, '{}/{}_sr.png'.format(dst, name))

                        Metrics.save_img(
                            hr_img, '{}/{}_hr.png'.format(dst, name))
                        Metrics.save_img(
                            fake_img, '{}/{}_inf.png'.format(dst, name))
                        Metrics.save_img(
                            lr_img, '{}/{}_lr.png'.format(dst, name))


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
        
    
    def evaluate_MICA(self):
        # NOW Benchmark
        pass
    
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