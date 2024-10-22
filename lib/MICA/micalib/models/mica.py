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
import sys

sys.path.append("./nfclib")

import torch
import torch.nn.functional as F

from model.mica.arcface import Arcface
from model.mica.generator import Generator
from lib.MICA.micalib.base_model import BaseModel

from loguru import logger

# add by patipol
from lib.MICA.micalib.renderer import MeshShapeRenderer
# add by patipol


class MICA(BaseModel):
    def __init__(self, config=None, device=None, tag='MICA'):
        super(MICA, self).__init__(config, device, tag)

        self.initialize()
        
        # add by patipol
        self.render = MeshShapeRenderer(obj_filename=config.model.topology_path)

    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path)
        if len(self.device) > 1:
            self.arcface = torch.nn.DataParallel(self.arcface, device_ids=self.device)
            self.arcface = self.arcface.module
        else:
            self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device)
        self.flameModel = Generator(512, 300, self.cfg.model.n_shape, mapping_layers, model_cfg, self.device)

    def load_model(self):
        model_path = os.path.join(self.cfg.output_dir, 'model_mica.tar')
        if os.path.exists(self.cfg.pretrained_model_path) and self.cfg.model.use_pretrained:
            model_path = self.cfg.pretrained_model_path
        if os.path.exists(model_path):
            logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
            checkpoint = torch.load(model_path)
            if 'arcface' in checkpoint:
                self.arcface.load_state_dict(checkpoint['arcface'])
            if 'flameModel' in checkpoint:
                self.flameModel.load_state_dict(checkpoint['flameModel'])
        else:
            logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}

        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images

        return codedict

    def decode(self, codedict, epoch=0):
        self.epoch = epoch

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            if len(self.device) > 1:
                shapecode = shapecode.to(self.device[0])[:, :self.cfg.model.n_shape]
            else:
                shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            with torch.no_grad():
                flame_verts_shape, _, _ = self.flame(shape_params=shapecode)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code)
        
        # add by patipol
        
        # flame = {
        #     'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
        #     'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
        #     'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        # }
        
        # zero_tensor = torch.zeros_like(codedict['arcface'])
        # pred_canonical_vertices, pred_shape_code = self.flameModel(zero_tensor)
        # add by patipol
        

        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface']
        }

        return output

    def compute_losses(self, input, encoder_output, decoder_output):
        losses = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask

        losses['pred_verts_shape_canonical_diff'] = torch.mean(pred_verts_shape_canonical_diff) * 1000.0

        return losses