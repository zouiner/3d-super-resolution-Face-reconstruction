from abc import abstractmethod

import torch
import torch.nn as nn

from config.default.config import cfg
from model.mica.flame import FLAME
from lib.MICA.utils.masking import Masking

class BaseModel(nn.Module):
    def __init__(self, config=None, tag=''):
        super(BaseModel, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        
        self.testing = self.cfg.mica.model.testing
        self.tag = tag
        self.use_mask = self.cfg.mica.train.use_mask
        self.masking = Masking(self.cfg.mica)
        
    
    def initialize(self):
        self.create_flame(self.cfg.mica.model)
        self.setup_renderer(self.cfg.mica.model)

        self.create_weights()
    
    def freeze_model(self, model):
        """
        Freeze the model's parameters by setting requires_grad = False.
        """
        for param in model.parameters():
            param.requires_grad = False

    def load_pretrained_weights(self, model, weights_path): # !!! check again may not use
        """
        Load pretrained weights into the model.
        :param model: The model instance.
        :param weights_path: Path to the pretrained weights file.
        """
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded pretrained weights from {weights_path}")

    def set_device(self, x):
        if len(self.device) > 1:
            device = self.device[0]
        else:
            device = self.device
        if isinstance(x, dict):
            for key, item in x.items():
                try:
                    if item is not None:
                        x[key] = item.to(device)
                except:
                    pass
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(device)
        else:
            x = x.to(device)
        return x
    
    # MICA
    
    def create_weights(self):
        if len(self.device) > 1:
            self.vertices_mask = self.masking.get_weights_per_vertex().to(self.device[0])
            self.triangle_mask = self.masking.get_weights_per_triangle().to(self.device[0])
        else:
            self.vertices_mask = self.masking.get_weights_per_vertex().to(self.device)
            self.triangle_mask = self.masking.get_weights_per_triangle().to(self.device)

    def create_flame(self, model_cfg):
        self.flame = FLAME(model_cfg)
        if len(self.device) > 1:
            self.flame = torch.nn.DataParallel(self.flame, device_ids=self.device)
            self.flame = self.flame.to(self.device[0])
            self.flame = self.flame.module
        else:
            self.flame.to(self.device[0])
        self.average_face = self.flame.v_template.clone()[None]

        self.flame.eval()
        
    def setup_renderer(self, model_cfg):
        self.verts_template_neutral = self.flame.v_template[None]
        self.verts_template = None
        self.verts_template_uv = None

    def create_weights(self):
        self.vertices_mask = self.masking.get_weights_per_vertex().to(self.device[0])
        self.triangle_mask = self.masking.get_weights_per_triangle().to(self.device[0])

    def create_template(self, B):
        with torch.no_grad():
            if self.verts_template is None:
                self.verts_template_neutral = self.flame.v_template[None]
                pose = torch.zeros(B, self.cfg.model.n_pose, device=self.device)
                pose[:, 3] = 10.0 * np.pi / 180.0  # 48
                self.verts_template, _, _ = self.flame(shape_params=torch.zeros(B, cfg.model.n_shape, device=self.device), expression_params=torch.zeros(B, self.cfg.model.n_exp, device=self.device), pose_params=pose)  # use template mesh with open mouth

            if self.verts_template.shape[0] != B:
                self.verts_template_neutral = self.verts_template_neutral[0:1].repeat(B, 1, 1)
                self.verts_template = self.verts_template[0:1].repeat(B, 1, 1)
    


    @abstractmethod
    def create_model(self):
        return

    @abstractmethod
    def load_model(self):
        return

    @abstractmethod
    def model_dict(self):
        return

    @abstractmethod
    def parameters_to_optimize(self):
        return

    @abstractmethod
    def encode(self, images, arcface_images):
        return

    @abstractmethod
    def decode(self, codedict, epoch):
        pass

    @abstractmethod
    def compute_losses(self, input, encoder_output, decoder_output):
        pass