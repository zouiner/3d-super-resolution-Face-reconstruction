'''
Default configuration
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_sr3d_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.sr3d_dir = abs_sr3d_dir

cfg.device = 'cuda'
cfg.device_id = [0,1]

cfg.name = ""
cfg.phase = ""
cfg.debug = None
cfg.enable_wandb = None
cfg.log_wandb_ckpt = None
cfg.log_eval = None

cfg.output_dir = "/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/Output"

# ---------------------------------------------------------------------------- #
# Options for path
# ---------------------------------------------------------------------------- #
cfg.path = CN()
cfg.path.log = "logs"
cfg.path.tb_logger = "tb_logger"
cfg.path.results = "results"
cfg.path.checkpoint_sr = "checkpoint_sr"
cfg.path.checkpoint_mica = "checkpoint_mica"

# SR
cfg.sr = CN()
cfg.sr.pretrained_model_path = None
# MICA
cfg.mica = CN()
cfg.mica.pretrained_model_path = os.path.join(cfg.sr3d_dir, 'data/pretrained', 'mica.tar')


# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #

# SR
cfg.sr.datasets = CN()
# Train
cfg.sr.datasets.train = CN()
cfg.sr.datasets.train.name = "mocktest"
cfg.sr.datasets.train.mode = "HR"
cfg.sr.datasets.train.dataroot = "contents/vgg_face2_train_32_128"
cfg.sr.datasets.train.datatype = "img"
cfg.sr.datasets.train.l_resolution = 32
cfg.sr.datasets.train.r_resolution = 128
cfg.sr.datasets.train.batch_size = 4
cfg.sr.datasets.train.num_workers = 8
cfg.sr.datasets.train.use_shuffle = True
cfg.sr.datasets.train.data_len = 10000
cfg.sr.datasets.K = 4 # temp from mica
# Val
cfg.sr.datasets.val = CN()
cfg.sr.datasets.val.name = "mocktest"
cfg.sr.datasets.val.mode = "LRHR"
cfg.sr.datasets.val.dataroot = "contents/vgg_face2_eval_32_128"
cfg.sr.datasets.val.datatype = "img"
cfg.sr.datasets.val.l_resolution = 32
cfg.sr.datasets.val.r_resolution = 128
cfg.sr.datasets.val.data_len = 3

# MICA
cfg.mica.datasets = CN()
cfg.mica.datasets.training_data = ['LYHM']
cfg.mica.datasets.eval_data = ['FLORENCE']
cfg.mica.datasets.datatype = "img"
cfg.mica.datasets.batch_size = 2
cfg.mica.datasets.K = 4
cfg.mica.datasets.n_train = 100000
cfg.mica.datasets.num_workers = 4
cfg.mica.datasets.root = '/datasets/arcface/'
cfg.mica.datasets.dataset_path = 'contents'

# ---------------------------------------------------------------------------- #
# Options for model
# ---------------------------------------------------------------------------- #

# SR
cfg.sr.model = CN()
cfg.sr.model.which_model_G = "sr3"
cfg.sr.model.finetune_norm = False

cfg.sr.model.unet = CN()
cfg.sr.model.unet.in_channel = 6
cfg.sr.model.unet.out_channel = 3
cfg.sr.model.unet.inner_channel = 64
cfg.sr.model.unet.channel_multiplier = [1, 2, 4, 8, 8]
cfg.sr.model.unet.attn_res = [16]
cfg.sr.model.unet.res_blocks = 2
cfg.sr.model.unet.dropout = 0.2

cfg.sr.model.beta_schedule = CN()
cfg.sr.model.beta_schedule.train = CN()
cfg.sr.model.beta_schedule.train.schedule = "linear"
cfg.sr.model.beta_schedule.train.n_timestep = 2000
cfg.sr.model.beta_schedule.train.linear_start = 0.000001
cfg.sr.model.beta_schedule.train.linear_end = 0.01

cfg.sr.model.beta_schedule.val = CN()
cfg.sr.model.beta_schedule.val.schedule = "linear"
cfg.sr.model.beta_schedule.val.n_timestep = 2000
cfg.sr.model.beta_schedule.val.linear_start = 0.000001
cfg.sr.model.beta_schedule.val.linear_end = 0.01

cfg.sr.model.diffusion = CN()
cfg.sr.model.diffusion.image_size = 128
cfg.sr.model.diffusion.channels = 3
cfg.sr.model.diffusion.conditional = True

# MICA
cfg.mica.model = CN()
cfg.mica.model.testing = False
cfg.mica.model.name = ""
cfg.mica.model.topology_path = os.path.join(cfg.sr3d_dir, 'data/FLAME2020', 'head_template.obj')
cfg.mica.model.flame_model_path = os.path.join(cfg.sr3d_dir, 'data/FLAME2020', 'generic_model.pkl')
cfg.mica.model.flame_lmk_embedding_path = os.path.join(cfg.sr3d_dir, 'data/FLAME2020', 'landmark_embedding.npy')
cfg.mica.model.n_shape = 300
cfg.mica.model.layers = 8
cfg.mica.model.hidden_layers_size = 256
cfg.mica.model.mapping_layers = 3
cfg.mica.model.use_pretrained = True
cfg.mica.model.arcface_pretrained_model = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/data/pretrained/backbone.pth'
# got from https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215582&cid=4A83B6B633B029CC
# cfg.model.arcface_pretrained_model = '/scratch/is-rg-ncs/models_weights/arcface-torch/backbone100.pth'
cfg.mica.model.n_pose = 6 # add by patipol
cfg.mica.model.n_exp = 50 # add by patipol

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #

cfg.train = CN()
cfg.train.log_dir = 'logs'

# SR
cfg.sr.train = CN()
cfg.sr.train.n_iter = 1000000
cfg.sr.train.val_freq = 10000
cfg.sr.train.save_checkpoint_freq = 10000
cfg.sr.train.print_freq = 200

cfg.sr.train.optimizer = CN()
cfg.sr.train.optimizer.type = "adam"
cfg.sr.train.optimizer.lr = 0.0001

cfg.sr.train.ema_scheduler = CN()
cfg.sr.train.ema_scheduler.step_start_ema = 5000
cfg.sr.train.ema_scheduler.update_ema_every = 1
cfg.sr.train.ema_scheduler.ema_decay = 0.9999

# MICA
cfg.mica.train = CN()

cfg.mica.train.use_mask = False
cfg.mica.train.max_epochs = 50
cfg.mica.train.max_steps = 100000
cfg.mica.train.lr = 1e-4
cfg.mica.train.arcface_lr = 1e-3
cfg.mica.train.weight_decay = 0.0
cfg.mica.train.lr_update_step = 100000000
cfg.mica.train.log_dir = 'logs'
cfg.mica.train.log_steps = 10
cfg.mica.train.vis_dir = 'train_images_mica'
cfg.mica.train.vis_steps = 200
cfg.mica.train.write_summary = True
cfg.mica.train.checkpoint_steps = 1000
cfg.mica.train.checkpoint_epochs_steps = 2
cfg.mica.train.val_steps = 1000
cfg.mica.train.val_vis_dir = 'val_images_mica'
cfg.mica.train.eval_steps = 5000
cfg.mica.train.reset_optimizer = False
cfg.mica.train.val_save_img = 5000
cfg.mica.test_dataset = 'now'

# ---------------------------------------------------------------------------- #
# Mask weights
# ---------------------------------------------------------------------------- #
cfg.mica.mask_weights = CN()
cfg.mica.mask_weights.face = 150.0
cfg.mica.mask_weights.nose = 50.0
cfg.mica.mask_weights.lips = 50.0
cfg.mica.mask_weights.forehead = 50.0
cfg.mica.mask_weights.lr_eye_region = 50.0
cfg.mica.mask_weights.eye_region = 50.0

cfg.mica.mask_weights.whole = 1.0
cfg.mica.mask_weights.ears = 0.01
cfg.mica.mask_weights.eyes = 0.01

cfg.mica.running_average = 7


# ---------------------------------------------------------------------------- #
# Options for log
# ---------------------------------------------------------------------------- #

cfg.wandb = CN()
cfg.wandb.project = "sr_vggf2"

# ---------------------------------------------------------------------------- #
# ----------------------------------  Main  ---------------------------------- #
# ---------------------------------------------------------------------------- #


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()


    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.phase
    
    if args.config is not None:
        cfg_file = args.config
        cfg = update_cfg(cfg, args.config)
        cfg.cfg_file = cfg_file

    cfg.output_dir = os.path.join(cfg.output_dir, cfg.name)
    
    print(cfg, end='\n\n')

    return cfg
