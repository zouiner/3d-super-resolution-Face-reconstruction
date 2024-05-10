'''
Default configuration
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir

cfg.gpu_ids = None

cfg.name = ""
cfg.phase = ""
cfg.debug = None
cfg.enable_wandb = None
cfg.log_wandb_ckpt = None
cfg.log_eval = None


cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
cfg.output_dir = "/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/Output"

# ---------------------------------------------------------------------------- #
# Options for path
# ---------------------------------------------------------------------------- #
cfg.path = CN()
cfg.path.pretrained_modelpath = ""
cfg.path.log = "logs"
cfg.path.tb_logger = "tb_logger"
cfg.path.results = "results"
cfg.path.checkpoint = "checkpoint"
cfg.path.resume_state = None

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #

cfg.datasets = CN()
# Train
cfg.datasets.train = CN()
cfg.datasets.train.name = "mocktest"
cfg.datasets.train.mode = "HR"
cfg.datasets.train.dataroot = "contents/vgg_face2_train_32_128"
cfg.datasets.train.datatype = "img"
cfg.datasets.train.l_resolution = 32
cfg.datasets.train.r_resolution = 128
cfg.datasets.train.batch_size = 4
cfg.datasets.train.num_workers = 8
cfg.datasets.train.use_shuffle = True
cfg.datasets.train.data_len = 10000
# Val
cfg.datasets.val = CN()
cfg.datasets.val.name = "mocktest"
cfg.datasets.val.mode = "LRHR"
cfg.datasets.val.dataroot = "contents/vgg_face2_eval_32_128"
cfg.datasets.val.datatype = "img"
cfg.datasets.val.l_resolution = 32
cfg.datasets.val.r_resolution = 128
cfg.datasets.val.data_len = 3

# TEMP for DECA
cfg.datasets.K = 0 # use in deca 
cfg.datasets.isSingle = False
cfg.datasets.num_workers = 2
cfg.datasets.image_size = 224
cfg.datasets.scale_min = 1.4
cfg.datasets.scale_max = 1.8
cfg.datasets.trans_scale = 0.

# ---------------------------------------------------------------------------- #
# Options for model
# ---------------------------------------------------------------------------- #

# SR
cfg.model = CN()
cfg.model.which_model_G = "sr3"
cfg.model.finetune_norm = False

cfg.model.unet = CN()
cfg.model.unet.in_channel = 6
cfg.model.unet.out_channel = 3
cfg.model.unet.inner_channel = 64
cfg.model.unet.channel_multiplier = [1, 2, 4, 8, 8]
cfg.model.unet.attn_res = [16]
cfg.model.unet.res_blocks = 2
cfg.model.unet.dropout = 0.2

cfg.model.beta_schedule = CN()
cfg.model.beta_schedule.train = CN()
cfg.model.beta_schedule.train.schedule = "linear"
cfg.model.beta_schedule.train.n_timestep = 2000
cfg.model.beta_schedule.train.linear_start = 0.000001
cfg.model.beta_schedule.train.linear_end = 0.01

cfg.model.beta_schedule.val = CN()
cfg.model.beta_schedule.val.schedule = "linear"
cfg.model.beta_schedule.val.n_timestep = 2000
cfg.model.beta_schedule.val.linear_start = 0.000001
cfg.model.beta_schedule.val.linear_end = 0.01

cfg.model.diffusion = CN()
cfg.model.diffusion.image_size = 128
cfg.model.diffusion.channels = 3
cfg.model.diffusion.conditional = True

#DECA

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #

cfg.train = CN()
cfg.train.log_dir = 'logs'

# SR
cfg.sr = CN()
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

# Deca
cfg.deca = CN()
cfg.deca.train = CN()
cfg.deca.train.train_detail = False
cfg.deca.train.max_epochs = 500
cfg.deca.train.max_steps = 1000000
cfg.deca.train.lr = 1e-4
cfg.deca.train.log_dir = 'logs'
cfg.deca.train.log_steps = 10
cfg.deca.train.vis_dir = 'train_images'
cfg.deca.train.vis_steps = 200
cfg.deca.train.write_summary = True
cfg.deca.train.checkpoint_steps = 500
cfg.deca.train.val_steps = 500
cfg.deca.train.val_vis_dir = 'val_images'
cfg.deca.train.eval_steps = 5000
cfg.deca.train.resume = True
cfg.deca.train.save_obj = False

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
