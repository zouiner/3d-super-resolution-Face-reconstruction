import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from .base_model import BaseModel  # Import the shared base model class
from model.sr.networks import define_G  # Import your super-resolution model from network.py
from model.mica.generator import Generator  # Import the MICA model (FLAME-based generator)
from model.mica.arcface import Arcface
from lib.MICA.micalib.renderer import MeshShapeRenderer

from collections import OrderedDict
from loguru import logger

# for SR3
import core.metrics as Metrics

input_std = 127.5
input_mean = 127.5


class ThreeDSuperResolutionModel(BaseModel):
    def __init__(self, cfg, device='cuda', freeze_sr=False, tag='MICA'):
        super(ThreeDSuperResolutionModel, self).__init__( cfg, tag)
        
        self.cfg = cfg
        self.device = device
        
        # Initialize the super-resolution model
        self.load_super_resolution_model(self.cfg)
        # Initialize the MICA model (FLAME-based generator)
        
        self.load_mica_model(self.cfg)
        
        self.initialize()
        self.render = MeshShapeRenderer(obj_filename=self.cfg.mica.model.topology_path).cuda()

        # Optionally freeze the super-resolution model's parameters
        if freeze_sr:
            self.freeze_model(self.sr_model)
            
        self.set_loss()
        self.log_dict = OrderedDict()

    def load_super_resolution_model(self, sr_model_config):
        """
        Load the super-resolution model using the configuration provided.
        This will use the define_G function from network.py.
        """
        self.sr_model = define_G(sr_model_config)
        self.schedule_phase = None
        
        self.sr_model = self.sr_model.cuda()
        

    def load_mica_model(self, mica_model_config):
        """
        Load the MICA model for 3D face reconstruction.
        """
        mica_model_config = mica_model_config.mica.model
        pretrained_path = None

        if not mica_model_config.use_pretrained:
            pretrained_path = mica_model_config.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path)
            
        self.mica_model = Generator(
            z_dim = 512, 
            map_hidden_dim = 300, 
            map_output_dim = mica_model_config['n_shape'], 
            hidden = mica_model_config['mapping_layers'], 
            model_cfg = mica_model_config, 
            device = self.device
        )
            
        self.arcface = Arcface(pretrained_path=pretrained_path).cuda()
        self.mica_model = self.mica_model.cuda()
        
        
    # --------------------------- computing fn ----------------------------
    
    def feed_data(self, data):
        x = self.set_device(data)
        
        return x 
    
    def enhance_images_with_super_resolution(self, train_data): # !! move to training.py
        """
        Enhance the resolution of a batch of input images using the super-resolution model.
        """
        # Preprocess images to tensors and send to GPU if available
        batch_set = self.preprocess_data(train_data)

        # Forward pass through the super-resolution model for the entire batch
        self.sr_model.set_new_noise_schedule(self.cfg.sr.model.beta_schedule.val, schedule_phase='val')
    
        self.sr_model.feed_data(batch_set)
        self.sr_model.test(continous = False)

        # Convert back to image format
        enhanced_images = self.postprocess_data(enhanced_images)
        return enhanced_images
    
    def create_tensor_blob(self, images, input_mean=127.5, input_std=127.5, size=(112, 112), swapRB=True):
        """
        images: tensor of shape (3, H, W), assumed to be in range [0, 1]
        input_mean: mean value for normalization
        input_std: standard deviation for normalization
        size: target size for resizing (width, height)
        swapRB: swap the Red and Blue channels (if True, swap channels)
        """

        # Normalize: (image - mean) / std
        images = (images - input_mean) / input_std
        
        # Resize the image using interpolation
        resized_images = F.interpolate(images.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

        # Swap the channels from RGB to BGR if necessary
        if swapRB:
            resized_images = resized_images[[2, 1, 0], :, :]  # Swap channels if needed
        
        return resized_images


    def create_arcface_embeddings(self, images):
        
        blob = cv2.dnn.blobFromImages([images], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True) # cv2.resize(img, (224, 224))
        
        return blob[0]

    def decode_mica(self, codedict, epoch = 0):
        """
        Reconstruct the 3D faces for a batch of enhanced images using the MICA model.
        """
        
        # Use the MICA model to predict the 3D faces
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            shapecode = shapecode[:, :self.cfg.mica.model.n_shape].cuda()
            with torch.no_grad():
                flame_verts_shape, _, _ = self.flame(shape_params=shapecode)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.mica_model(identity_code)

        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface']
        }

        return output

    def encode_mica(self, images, arcface_imgs):
        codedict = {}

        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images

        return codedict

    def preprocess_sr_data(self, train_data):
        """
        Preprocess a batch of input images by resizing them and converting them to tensors.
        :param images: A batch of images (numpy arrays) with shape (B, k, 3, H, W).
        :return: Tensor with shape (B, 3, H, W).
        """
        train_data_sub = self.filter_and_slice_train_data(train_data, order = 0)

        for i in range(1,len(train_data['SR'])): # make k x b, c, h, w
            _train_data_sub = self.filter_and_slice_train_data(train_data, order = i)
            for key in train_data_sub:
                if isinstance(train_data_sub[key], torch.Tensor):
                    train_data_sub[key] = torch.cat([train_data_sub[key], _train_data_sub[key]], dim=0)
                elif isinstance(train_data_sub[key], list):
                    train_data_sub[key].extend(_train_data_sub[key])
        
        return train_data_sub
    
    def training_MICA(self, batch, current_epoch = 0):
        self.mica_model.train() # loop here!!

        images = batch['image']
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface'].cuda()
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1])

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }
        encoder_output = self.encode_mica(images, arcface)
        encoder_output['flame'] = flame
        decoder_output = self.decode_mica(encoder_output, current_epoch)

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return inputs, opdict, encoder_output, decoder_output
    
    def get_current_visuals(self, x_in, x, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = x.detach().float().cpu()
        else:
            out_dict['SR'] = x.detach().float().cpu()
            out_dict['INF'] = x_in['SR'].detach().float().cpu()
            if 'HR' in x_in:                        # !!take a look!! -> because just want only SR
                out_dict['HR'] = x_in['HR'].detach().float().cpu()
            if need_LR and 'LR' in x_in:
                out_dict['LR'] = x_in['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def test_sr(self, x_in, continous=False):
        
        if isinstance(self.sr_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            SR = self.sr_model.module.super_resolution(
                x_in['SR'], continous)
        else:
            SR = self.sr_model.super_resolution(
                x_in['SR'], continous)
        
        return SR
            
    def get_tensor_sr_img(self, x):
        
        wrapped_data = DictTensor(x)
        
        self.SR = self.sr_model(wrapped_data, sr_out = True)
        
        return self.SR

    def sample(self, batch_size=1, continous=False):
        self.sr_model.eval()
        with torch.no_grad():
            if isinstance(self.sr_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                self.SR = self.sr_model.module.sample(batch_size, continous)
            else:
                self.SR = self.sr_model.sample(batch_size, continous)
        self.sr_model.train()

    def set_loss(self):
        if isinstance(self.sr_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            self.sr_model.module.set_loss(self.device)
        else:
            self.sr_model.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.sr_model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                self.sr_model.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.sr_model.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict
    
    def compute_loss(self, input_sr, input_mica, encoder_output, decoder_output):
        
        self.train()
        if 'imagename' in input_sr:
            del input_sr['imagename']
        # sr loss
        
        l_sr = self.sr_model(input_sr)
        # need to average in multi-gpu
        b, c, h, w = input_sr['HR'].shape
        l_sr = l_sr.sum()/int(b*c*h*w) 
        
        # mica loss
        losses = self.compute_losses(input_mica, encoder_output, decoder_output)

        all_loss = 0.
        losses_key = losses.keys()

        for key in losses_key:
            all_loss = all_loss + losses[key]

        losses['all_loss'] = all_loss
        
        l_mica = all_loss
        
        # set log
        self.log_dict['l_sr'] = l_sr.item()
        self.log_dict['l_mica'] = l_mica.item()
        

        return l_sr, l_mica, losses
    
    def compute_losses(self, input, encoder_output, decoder_output):
        losses = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask

        losses['pred_verts_shape_canonical_diff'] = torch.mean(pred_verts_shape_canonical_diff) * 1000.0

        return losses
    
    def model_dict(self):
        return {
            'flameModel': self.mica_model.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        
        # mica
        return [
            {'params': self.mica_model.parameters(), 'lr': self.cfg.mica.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.mica.train.arcface_lr},
        ]
    
    def train(self):
        self.sr_model.train()
        self.mica_model.train()
        self.arcface.train()
    
    def eval(self):
        self.sr_model.eval()
        self.mica_model.eval()
        self.arcface.eval()
    
    def SR3_training(self, x, t_output = None, v_output = None):
        # Change name in other function(eg. training function)
        tensor_sr = None
        visuals = None
        if t_output:
            tensor_sr = self.get_tensor_sr_img(x)
            if v_output:
                visuals = self.get_current_visuals(x, tensor_sr)
                return tensor_sr, visuals
            else:
                return tensor_sr
        elif v_output:
            x_sr = self.test_sr(x)
            visuals = self.get_current_visuals(x, x_sr)
            
            return visuals
    
    def test_val(self, x, epoch, global_step, faces, avg_psnr, avg_ssim, k):
        val_data = self.feed_data(x)
        visuals = self.SR3_training(x, v_output = True)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        

        # MICA 
        
        sr_up_img = cv2.resize(sr_img, (224, 224))
        temp_arcface = self.create_arcface_embeddings(sr_up_img)
        temp_arcface = torch.tensor(temp_arcface).cuda()[None]
        
        sr_up_img = sr_up_img / 255.
        sr_up_img = sr_up_img.transpose(2, 0, 1)
        sr_up_img = torch.tensor(sr_up_img).cuda()[None]
        
        self.eval()
        encoder_output = self.encode_mica(sr_up_img, temp_arcface)
        opdict = self.decode_mica(encoder_output)
        meshes = opdict['pred_canonical_shape_vertices']
        code = opdict['pred_shape_code']
        lmk = self.flame.compute_landmarks(meshes)

        mesh = meshes[0]
        landmark_51 = lmk[0, 17:]
        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

        
        if self.cfg.sample == 1:
            name = os.path.basename(val_data['path_sr'][0])[:-4]
            savepath = os.path.join(self.cfg.output_dir, 'val_images', '{}_{}'.format(epoch, global_step))
        else:
            name = os.path.basename(val_data['path_sr'][0])[:-4] + '_' + str(k).zfill(len(str(self.cfg.sample)))
            savepath = os.path.join(self.cfg.output_dir, 'val_images', '{}_{}_s{}'.format(epoch, global_step,self.cfg.sample))
        
        from pathlib import Path
        import trimesh
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

        # # compute_loss mica
        # if self.wandb_logger and self.cfg['log_eval']:
        #     self.wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        return avg_psnr, avg_ssim
    
    def forward(self, x, epoch, global_step):
        visualizeTraining = global_step % self.cfg.train.vis_steps == 0
        
        sr_train_data = self.preprocess_sr_data(x)
        
        # MICA # ------------------------------------------- 
        images_list = []
        arcface_list = []
        new_sr = []
        _sr_train_data = sr_train_data
        self.set_new_noise_schedule(self.cfg.sr.model.beta_schedule[self.cfg['phase']], schedule_phase= self.cfg['phase']) # for visual SR to fedd into mica
        for i in range(len(x['SR'])):
            for j in range(len(self.filter_and_slice_train_data(x, order = i)['SR'])):
                _sr_train_data['SR'] = self.filter_and_slice_train_data(x, order = i)['SR'][j].unsqueeze(0)
                _sr_train_data['HR'] = self.filter_and_slice_train_data(x, order = i)['HR'][j].unsqueeze(0)
                _sr_train_data['Index'] = self.filter_and_slice_train_data(x, order = i)['Index'][j].unsqueeze(0)
                _sr_train_data = self.feed_data(_sr_train_data)
                if self.cfg.model == 'model2':
                    visuals = self.SR3_training(_sr_train_data, t_output = False, v_output = True) # -> (3,16,16)
                            
                            
                    sr_img = Metrics.tensor2img(visuals['SR'])
                    sr_up_img = (cv2.resize(sr_img, (224, 224)))
                    
                    if self.cfg.mica.train.arcface_new:
                        temp_arcface = self.create_arcface_embeddings(sr_up_img)
                        arcface_list.append(temp_arcface)
                    

                    images_list.append(sr_up_img.transpose(2,0,1)/255) # image no need to be tensor becasue mica use only arcface
                
                else:
                    tensor_sr, visuals = self.SR3_training(_sr_train_data, t_output = True, v_output = True) # -> (3,16,16)
                    
                    
                    sr_img = Metrics.tensor2img(visuals['SR'])
                    temp_sr_up_img = Metrics.tensor2tensor_img(tensor_sr) * 255.0
                    
                    
                    if self.cfg.mica.train.arcface_new:
                        temp_arcface = self.create_tensor_blob(temp_sr_up_img)
                        arcface_list.append(temp_arcface.clone().detach().requires_grad_(True))
                    

                    sr_up_img = (cv2.resize(sr_img, (224, 224)))
                    images_list.append(sr_up_img.transpose(2,0,1)/255) # image no need to be tensor becasue mica use only arcface
                    tensor_sr = tensor_sr.unsqueeze(0) * (2) + -1 # -> change te range to be (-1,1)
                    new_sr.append(tensor_sr.clone().detach().requires_grad_(True))

                
                    
                    
                if visualizeTraining:
                    savepath = os.path.join(self.cfg.output_dir, 'train_images/{}_{}'.format(epoch, global_step))
                    os.makedirs(savepath, exist_ok=True)
                    Metrics.save_img(sr_img, '{}/{}_{}_sr.png'.format(savepath, i, j))
                    hr_img = Metrics.tensor2img(visuals['HR'])
                    inf_img = Metrics.tensor2img(visuals['INF'])
                    Metrics.save_img(hr_img, '{}/{}_{}_hr.png'.format(savepath, i, j))
                    Metrics.save_img(inf_img, '{}/{}_{}_inf.png'.format(savepath, i, j))
                        
        
        # Prepare train data for SR3 training -> Can change the train_data to be the SR3 dataloader 
        input_sr = {}
        input_sr['HR'] = torch.cat(x['HR'], dim=0)
        # input_sr['SR'] = torch.cat(train_data['SR'], dim=0)
        if self.cfg.model == 'model2':
            input_sr['SR'] = torch.cat(x['SR'], dim=0)
            # Convert each element in arcface_list to a tensor
            arcface_list = [torch.from_numpy(item) if isinstance(item, np.ndarray) else item for item in arcface_list]
        
        else:
            input_sr['SR'] = torch.cat(new_sr, dim=0)
        
        images_list = [torch.from_numpy(image) if isinstance(image, np.ndarray) else image for image in images_list]
        images_array = torch.stack(images_list).view(x['image'].shape)
        arcface_array = torch.stack(arcface_list).view(x['arcface'].shape)

        input_sr = self.feed_data(input_sr)

        # Use the output from the SR feed to MICA
        batch = self.filter_and_slice_train_data(x, 0, keys_to_keep_and_slice = {}, keys_to_keep = {'image', 'arcface', 'imagename', 'dataset', 'flame'})
        batch['image'] = images_array
        batch['arcface'] = arcface_array
        
        input_mica, opdict, encoder_output, decoder_output = self.training_MICA(batch, epoch)

        sr_train_data = self.feed_data(sr_train_data)
        l_sr, l_mica, losses = self.compute_loss(sr_train_data, input_mica, encoder_output, decoder_output)
        
        return l_sr, l_mica, losses, opdict

    
    # -------------------- preprocessing fn --------------------------

    
    def filter_and_slice_train_data(self, train_data, order, keys_to_keep_and_slice = {'HR', 'SR', 'LR'}, keys_to_keep = {'Index', 'imagename'}):
    
        # Create a new dictionary with the specified keys
        dict_a = {}
        for key in train_data:
            if key in keys_to_keep_and_slice:
                dict_a[key] = train_data[key][order]  # Get the order sublist
            elif key in keys_to_keep:
                dict_a[key] = train_data[key]
        
        return dict_a
    
    def move_to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            # Move tensor to the specified device
            return data.to(device)
        elif isinstance(data, list):
            # Convert list to tensor only if it contains numeric data
            try:
                return torch.tensor(data, device=device)
            except ValueError:
                return data  # Return the list unchanged if it can't be converted
        elif isinstance(data, dict):
            # Recursively process dictionaries
            return {key: self.move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, str):
            # Leave strings unchanged
            return data
        else:
            # Return other data types unchanged (e.g., None, int, float)
            return data
        

###########

class DictTensor:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        self.data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.data.items()
        }
        return self

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def __repr__(self):
        return str(self.data)
