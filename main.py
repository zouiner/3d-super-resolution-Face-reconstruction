import os, sys
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

import core.logger as Logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))
    
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
    
def config():
    from config.default.config import parse_args
    
    # parse configs
    args = parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    # opt = Logger.dict_to_nonedict(opt)
    
    return opt

if __name__ == '__main__':
    cfg = config()
    if cfg.cfg_file is not None:
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main.py -p train -c config/sr_sr3_VGGF2_32_128.yml