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

input_std = 127.5
input_mean = 127.5

class ThreeDSuperResolutionModel(BaseModel):
    def __init__(self, cfg, device='cuda', freeze_sr=False, tag='MICA'):
        super(ThreeDSuperResolutionModel, self).__init__(cfg, tag)
        
        self.cfg = cfg
        self.device = device
        
        # Initialize the super-resolution model
        self.load_super_resolution_model(self.cfg)
        # Initialize the MICA model (FLAME-based generator)
        self.load_mica_model(self.cfg)
        
        self.initialize()
        self.render = MeshShapeRenderer(obj_filename=self.cfg.mica.model.topology_path)

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
        
        if len(self.device) > 1:
            self.sr_model = nn.DataParallel(self.sr_model, device_ids=self.device)
            self.sr_model = self.sr_model.cuda().module
        

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
        
        if len(self.device) > 1:
            self.arcface = torch.nn.DataParallel(self.arcface, device_ids=self.device)
            self.arcface = self.arcface.cuda().module
            self.mica_model = torch.nn.DataParallel(self.mica_model, device_ids=self.device)
            self.mica_model = self.mica_model.cuda().module
        else:
            self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device[0])
        
        
        
            
    def save_model(self):
        # sr
        checkpoint_dir = self.cfg['path']['checkpoint_sr']
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        gen_path = os.path.join(
            self.cfg['path']['checkpoint_sr'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        cfg_path = os.path.join(
            self.cfg['path']['checkpoint_sr'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            '[SR] Saved model in [{:s}] ...'.format(gen_path))
        
        # mica
        
    # --------------------------- computing fn ----------------------------
    
    def feed_data(self, data):
        self.data = self.set_device(data)
    
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
            if len(self.device) > 1:
                shapecode = shapecode.to(self.device[0])[:, :self.cfg.mica.model.n_shape]
            else:
                shapecode = shapecode.to(self.device)[:, :self.cfg.mica.model.n_shape]
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

    def forward(self, images):
        """
        Process a batch of images. Pass the batch of images through the super-resolution
        model first, then the MICA model.
        """
        # Step 1: Enhance the resolution of the input images (batch)
        enhanced_images = self.enhance_images_with_super_resolution(images)

        # Step 2: Reconstruct the 3D faces for the batch
        reconstructed_faces, shape_params = self.reconstruct_3d_faces(enhanced_images)

        return reconstructed_faces, shape_params

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

        images = batch['image'].to(self.device[0]) # !!take a look!! for multi gpu
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface']
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1]).to(self.device[0]) # !!take a look!! for multi gpu

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
    
    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            if 'HR' in self.data:                        # !!take a look!! -> because just want only SR
                out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def test_sr(self, continous=False):
        # sr
        self.sr_model.eval()
        with torch.no_grad():
            if isinstance(self.sr_model, nn.DataParallel):
                self.SR = self.sr_model.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.sr_model.super_resolution(
                    self.data['SR'], continous)
        self.sr_model.train()

    def sample(self, batch_size=1, continous=False):
        self.sr_model.eval()
        with torch.no_grad():
            if isinstance(self.sr_model, nn.DataParallel):
                self.SR = self.sr_model.module.sample(batch_size, continous)
            else:
                self.SR = self.sr_model.sample(batch_size, continous)
        self.sr_model.train()

    def set_loss(self):
        if isinstance(self.sr_model, nn.DataParallel):
            self.sr_model.module.set_loss(self.device[0])
        else:
            self.sr_model.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.sr_model, nn.DataParallel):
                self.sr_model.module.set_new_noise_schedule(
                    schedule_opt, self.device[0])
            else:
                self.sr_model.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict
    
    def compute_loss(self, inputs, encoder_output, decoder_output):
        
        self.train()
        if 'imagename' in self.data:
            del self.data['imagename']
        # sr loss
        
        l_sr = self.sr_model(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_sr = l_sr.sum()/int(b*c*h*w) 
        
        # mica loss
        losses = self.compute_losses(inputs, encoder_output, decoder_output)

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