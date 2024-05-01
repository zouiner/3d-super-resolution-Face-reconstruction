import os, sys
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

# import core.logger as Logger


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)


def main(cfg):
    
    # export CUDA_VISIBLE_DEVICES
    if cfg.gpu_ids is not None:
        gpu_list = ','.join(str(id) for id in cfg.gpu_ids)

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # Save config
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
    
    # Set a path for log
    cfg.path.tb_logger = os.path.join(cfg.output_dir, cfg.train.log_dir, cfg.path.tb_logger)
    
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # # start training -- DECA
    # from decalib.deca import DECA
    # from lib.trainer import Trainer
    # cfg.rasterizer_type = 'pytorch3d'
    
    # device = torch.device(cfg.device)
    # deca = DECA(cfg, device = device)
    # trainer = Trainer(model=deca, config=cfg, device = device)

    # # start train
    # if cfg.dataset.test_mode:
    #     trainer.test_img(cfg.dataset.test_data)
    # else:
    #     os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    #     os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    #     trainer.fit()
    
    # test dataloader
    from lib.trainer import Trainer
    trainer = Trainer(model=None, config=cfg)
    
    trainer.fit()
    
def config():
    from config.default.config import parse_args
    
    # parse configs
    args = parse_args()
    
    return args

if __name__ == '__main__':
    cfg = config()
    main(cfg)

# run:
# python main.py -p train -c config/sr_sr3_VGGF2_32_128.yml