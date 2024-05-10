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
import core.metrics as Metrics


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
                    
                    # SR # -------------------------------------------
                    
                    if current_step > n_iter:
                        break
                    diffusion.feed_data(train_data)
                    diffusion.optimize_parameters()
                    
                    # DECA # ------------------------------------------- 1 step
                    
                    visuals = diffusion.get_current_visuals() 
                    # sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                    # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
                    
                    
                    
                    # log
                    if current_step % self.cfg.sr.train.print_freq == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            self.tb_logger.add_scalar(k, v, current_step)
                        logger.info(message)

                        if self.wandb_logger:
                            self.wandb_logger.log_metrics(logs)

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

                            if self.wandb_logger:
                                self.wandb_logger.log_image(
                                    f'validation_{idx}', 
                                    np.concatenate((fake_img, sr_img, hr_img), axis=1)
                                )
                                
                                

                        avg_psnr = avg_psnr / idx
                        # reset dataloader    
                        self.val_iter = iter(self.val_dataloader)
                        
                        diffusion.set_new_noise_schedule(
                            self.cfg.model.beta_schedule.train, schedule_phase='train')
                        # log
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        self.tb_logger.add_scalar('psnr', avg_psnr, current_step)

                        if self.wandb_logger:
                            self.wandb_logger.log_metrics({
                                'validation/val_psnr': avg_psnr,
                                'validation/val_step': val_step
                            })
                            val_step += 1

                    if current_step % self.cfg.sr.train.save_checkpoint_freq == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)

                        if self.wandb_logger and opt['log_wandb_ckpt']:
                            self.wandb_logger.log_checkpoint(current_epoch, current_step)
                    
                    

                if self.wandb_logger:
                    self.wandb_logger.log_metrics({'epoch': current_epoch-1})
                # reset dataloader    
                self.train_iter = iter(self.train_dataloader)
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
                    for iter_ in range(0, sample_num):
                        Metrics.save_img(
                            Metrics.tensor2img(sr_img[iter_]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter_))
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

                if self.wandb_logger and self.cfg['log_eval']:
                    self.wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx

            # log
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim))

            if self.wandb_logger:
                if self.cfg['log_eval']:
                    self.wandb_logger.log_eval_table()
                self.wandb_logger.log_metrics({
                    'PSNR': float(avg_psnr),
                    'SSIM': float(avg_ssim)
                })
            
        # DECA #
    
    def training_deca(self, batch, batch_nb, training_type='coarse'):
        self.deca.train()
        if self.train_detail:
            self.deca.E_flame.eval()
            
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])  #([B*K, 3, 224, 224])
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1]) # ([B*K, 68, 3])
        masks = batch['mask'].to(self.device); masks = masks.view(-1, images.shape[-2], images.shape[-1]) # ([B*K, 224, 224])

        
        #-- encoder
        codedict = self.deca.encode(images, use_detail=self.train_detail)
        
        # add by patipol
        if self.deca_pretrain:
            codedict_pretrain = self.deca_pretrain.encode(images, use_detail=self.train_detail)
        # add by patipol
        
        
        ### shape constraints for coarse model
        ### detail consistency for detail model
        # import ipdb; ipdb.set_trace()
        if self.cfg.loss.shape_consistency or self.cfg.loss.detail_consistency:
            '''
            make sure s0, s1 is something to make shape close
            the difference from ||so - s1|| is 
            the later encourage s0, s1 is cloase in l2 space, but not really ensure shape will be close
            '''
            new_order = np.array([np.random.permutation(self.K) + i*self.K for i in range(self.batch_size)])
            new_order = new_order.flatten()
            shapecode = codedict['shape']
            if self.train_detail:
                detailcode = codedict['detail']
                detailcode_new = detailcode[new_order]
                codedict['detail'] = torch.cat([detailcode, detailcode_new], dim=0)
                codedict['shape'] = torch.cat([shapecode, shapecode], dim=0)
                
                # add by patipol
                if self.deca_pretrain:
                    detailcode_p = codedict_pretrain['detail']
                    detailcode_new_p = detailcode_p[new_order]
                    codedict_pretrain['detail'] = torch.cat([detailcode_p, detailcode_new_p], dim=0)
                    
                    for key in ['tex', 'exp', 'pose', 'cam', 'light', 'images']:
                        code = codedict_pretrain[key]
                        codedict_pretrain[key] = torch.cat([code, code], dim=0)
                # add by patipol
                
            else:
                shapecode_new = shapecode[new_order]
                codedict['shape'] = torch.cat([shapecode, shapecode_new], dim=0)
            for key in ['tex', 'exp', 'pose', 'cam', 'light', 'images']:
                code = codedict[key]
                codedict[key] = torch.cat([code, code], dim=0)
            ## append gt
            images = torch.cat([images, images], dim=0)# images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1]) 
            lmk = torch.cat([lmk, lmk], dim=0) #lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
            masks = torch.cat([masks, masks], dim=0)

        batch_size = images.shape[0]

        ###--------------- training coarse model
        if not self.train_detail:
            #-- decoder
            rendering = True if self.cfg.loss.photo>0 else False
            opdict = self.deca.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False, step_global = self.global_step, step_procress = 'train')
            opdict['images'] = images
            opdict['lmk'] = lmk

            if self.cfg.loss.photo > 0.:
                #------ rendering
                # mask
                mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
                # images
                predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
                opdict['predicted_images'] = predicted_images

            #### ----------------------- Losses
            losses = {}
            
            ############################# base shape
            predicted_landmarks = opdict['landmarks2d']
            if self.cfg.loss.useWlmk:
                losses['landmark'] = lossfunc.weighted_landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            else:    
                losses['landmark'] = lossfunc.landmark_loss(predicted_landmarks, lmk)*self.cfg.loss.lmk
            if self.cfg.loss.eyed > 0.:
                losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks, lmk)*self.cfg.loss.eyed
            if self.cfg.loss.lipd > 0.:
                losses['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks, lmk)*self.cfg.loss.lipd
            
            if self.cfg.loss.photo > 0.:
                if self.cfg.loss.useSeg:
                    masks = masks[:,None,:,:]
                else:
                    masks = mask_face_eye*opdict['alpha_images']
                losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()*self.cfg.loss.photo

            if self.cfg.loss.id > 0.:
                shading_images = self.deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
                albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
                overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
                losses['identity'] = self.id_loss(overlay, images) * self.cfg.loss.id
            
            losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.loss.reg_shape
            losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
            losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
            losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
            if self.cfg.model.jaw_type == 'euler':
                # import ipdb; ipdb.set_trace()
                # reg on jaw pose
                losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:,-1]**2)/2)*100.
                losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:,0])**2)/2)*10.
            
            if self.global_step % self.cfg.train.vis_steps == 0:
                output_path = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:08}')
                os.makedirs(output_path, exist_ok=True)
                self.grid_save(images, output_path, 'images')
                try:
                    self.grid_save(predicted_images, output_path, 'predicted_images')
                except:
                    pass
        
        ###--------------- training detail model
        else:
            #-- decoder -> code like in decoder
            shapecode = codedict['shape']
            expcode = codedict['exp']
            posecode = codedict['pose']
            texcode = codedict['tex']
            lightcode = codedict['light']
            detailcode = codedict['detail']
            cam = codedict['cam']

            # FLAME - world space
            verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
            landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:] #; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
            # world to camera
            trans_verts = util.batch_orth_proj(verts, cam)
            predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]
            # camera to image space
            trans_verts[:,:,1:] = -trans_verts[:,:,1:]
            predicted_landmarks[:,:,1:] = - predicted_landmarks[:,:,1:]
            
            albedo = self.deca.flametex(texcode) 

            #------ rendering
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)
            # mask
            mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), ops['grid'].detach(), align_corners=False)
            # images
            predicted_images = ops['images']*mask_face_eye*ops['alpha_images']

            masks = masks[:,None,:,:]

            uv_z = self.deca.D_detail(torch.cat([posecode[:,3:], expcode, detailcode], dim=1))  
            
            # add by patipol
            if self.deca_pretrain:
                detailcode_p = codedict_pretrain['detail']
                expcode_p = codedict_pretrain['exp']
                posecode_p = codedict_pretrain['pose']
                
                uv_z_1 = self.deca_pretrain.D_detail(torch.cat([posecode_p[:,3:], expcode_p, detailcode_p], dim=1))  
            
            # add by patipol
            
            # render detail
            uv_detail_normals = self.deca.displacement2normal(uv_z, verts, ops['normals']) # point
            uv_shading = self.deca.render.add_SHlight(uv_detail_normals, lightcode.detach())
            uv_texture = albedo.detach()*uv_shading
            
            predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)
            
            # add by patipol
            if self.global_step % self.cfg.train.vis_steps == 0:
                output_path = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:08}')
                os.makedirs(output_path, exist_ok=True)
                name = 'ops'
                savepath = os.path.join(output_path, name)
                
                self.save_img(ops['images'], savepath + '_image' )
                self.save_img(ops['normal_images'], savepath + '_normal' )
                
                faces = self.deca.flame.faces_tensor.cpu().numpy()
                # save mesh
                for k in range(ops['images'].shape[0]):
                    util.write_obj(savepath + f'_obj_image{k + 1}.obj', vertices=verts[k], faces=faces)
            
            # add by patipol

            #--- extract texture
            uv_pverts = self.deca.render.world2uv(trans_verts).detach()
            uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)
            uv_texture_gt = uv_gt[:,:3,:,:].detach(); uv_mask_gt = uv_gt[:,3:,:,:].detach()
            # self-occlusion
            normals = util.vertex_normals(trans_verts, self.deca.render.faces.expand(batch_size, -1, -1))
            uv_pnorm = self.deca.render.world2uv(normals)
            uv_mask = (uv_pnorm[:,[-1],:,:] < -0.05).float().detach()
            ## combine masks
            uv_vis_mask = uv_mask_gt*uv_mask*self.deca.uv_face_eye_mask
            
            
            
            #### ----------------------- Losses
            losses = {}
            ############################### details
            # if self.cfg.loss.old_mrf: 
            #     if self.cfg.loss.old_mrf_face_mask:
            #         masks = masks*mask_face_eye*ops['alpha_images']
            #     losses['photo_detail'] = (masks*(predicted_detailed_image - images).abs()).mean()*100
            #     losses['photo_detail_mrf'] = self.mrf_loss(masks*predicted_detailed_image, masks*images)*0.1
            # else:
            pi = 0
            new_size = 256
            uv_texture_patch = F.interpolate(uv_texture[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_texture_gt_patch = F.interpolate(uv_texture_gt[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
            
            # add by patipol
            
            if self.global_step % self.cfg.train.vis_steps == 0:
                # for i in range(len(uv_pverts)):
                #     savepath = os.path.join(output_path, 'uv_pverts')
                #     temp =  (uv_pverts[i].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_texture_gt')
                #     temp =  (uv_texture_gt[i].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_pnorm')
                #     temp =  (uv_pnorm[i].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_mask')
                #     temp =  (uv_mask[i].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_vis_mask')
                #     temp =  (uv_vis_mask[i].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                    
                #     savepath = os.path.join(output_path, 'uv_texture_patch')
                #     temp =  (uv_texture_patch[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_texture_gt_patch')
                #     temp =  (uv_texture_gt_patch[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_vis_mask_patch')
                #     temp =  (uv_vis_mask_patch[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                    
                #     savepath = os.path.join(output_path, 'albedo')
                #     temp =  (albedo[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'mask_face_eye')
                #     temp =  (mask_face_eye[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'predicted_images')
                #     temp =  (predicted_images[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'masks')
                #     temp =  (masks[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_z')
                #     temp =  (uv_z[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_detail_normals')
                #     temp =  (uv_detail_normals[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                    
                #     savepath = os.path.join(output_path, 'uv_shading')
                #     temp =  (uv_shading[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'uv_texture')
                #     temp =  (uv_texture[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                #     savepath = os.path.join(output_path, 'predicted_detail_images')
                #     temp =  (predicted_detail_images[i].cpu().detach().numpy().transpose(1, 2, 0)* 255).astype(np.uint8)
                #     transforms.ToPILImage()(temp).save(savepath + '_' + str(i) + '.png')
                
                self.grid_save(uv_pverts, output_path, 'uv_pverts')
                self.grid_save(uv_texture_gt, output_path, 'uv_texture_gt')
                self.grid_save(uv_pnorm, output_path, 'uv_pnorm')
                self.grid_save(uv_mask, output_path, 'uv_mask')
                self.grid_save(uv_vis_mask, output_path, 'uv_vis_mask')
                self.grid_save(uv_texture_patch, output_path, 'uv_texture_patch')
                self.grid_save(uv_texture_gt_patch, output_path, 'uv_texture_gt_patch')
                self.grid_save(uv_vis_mask_patch, output_path, 'uv_vis_mask_patch')
                self.grid_save(albedo, output_path, 'albedo')
                self.grid_save(mask_face_eye, output_path, 'mask_face_eye')
                self.grid_save(predicted_images, output_path, 'predicted_images')
                self.grid_save(masks, output_path, 'masks')
                self.grid_save(uv_z, output_path, 'uv_z')
                self.grid_save(uv_detail_normals, output_path, 'uv_detail_normals')
                self.grid_save(uv_shading, output_path, 'uv_shading')
                self.grid_save(uv_texture, output_path, 'uv_texture')
                self.grid_save(predicted_detail_images, output_path, 'predicted_detail_images')
                
                if self.deca_pretrain:
                    self.grid_save(uv_z_1, output_path, 'uv_z_1')
            
            if self.deca_pretrain:
                losses['uv_z'] = torch.mean((uv_z - uv_z_1).abs())*0.5
                    
            # add by patipol
            
            losses['photo_detail'] = (uv_texture_patch*uv_vis_mask_patch - uv_texture_gt_patch*uv_vis_mask_patch).abs().mean()*self.cfg.loss.photo_D
            losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch)*self.cfg.loss.photo_D*self.cfg.loss.mrf
            
            losses['z_reg'] = torch.mean(uv_z.abs())*self.cfg.loss.reg_z
            losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading)*self.cfg.loss.reg_diff
            if self.cfg.loss.reg_sym > 0.:
                nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
                losses['z_sym'] = (nonvis_mask*(uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum()*self.cfg.loss.reg_sym
            opdict = {
                'verts': verts,
                'trans_verts': trans_verts,
                'landmarks2d': landmarks2d,
                'predicted_images': predicted_images,
                'predicted_detail_images': predicted_detail_images,
                'images': images,
                'lmk': lmk
            }
            
        #########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key].to(self.device)
        losses['all_loss'] = all_loss
        return losses, opdict
    
    def validation_deca(self):
        pass
    
    def evaluate_deca(self):
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
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=1, shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)
        self.val_iter = iter(self.val_dataloader)
        
        logger.info('Initial Dataset Finished')
    
    def fit(self):
        self.prepare_data()
        
        self.training_step()