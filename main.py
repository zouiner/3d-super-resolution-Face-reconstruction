import os, sys
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

# mica
from lib.MICA.utils import util

# import core.logger as Logger


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))



def main(cfg):
    
    # export CUDA_VISIBLE_DEVICES
    if cfg.device is not None:
        gpu_list = ','.join(str(id) for id in cfg.device_id)

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
        cfg.gpu_ids = [i for i in range(len(cfg.device_id))]
        if len(gpu_list) > 1:
            cfg.distributed = True
        else:
            cfg.distributed = False
    
    
    
    # Setting a path for log
    cfg.path.tb_logger = os.path.join(cfg.output_dir, cfg.train.log_dir, cfg.path.tb_logger)
    cfg.path.results = os.path.join(cfg.output_dir, cfg.path.results)
    cfg.path.checkpoint_sr = os.path.join(cfg.output_dir, cfg.path.checkpoint_sr)
    cfg.path.checkpoint_mica = os.path.join(cfg.output_dir, cfg.path.checkpoint_mica)
    cfg.path.results_mica = os.path.join(cfg.output_dir, 'train_images_mica')
    cfg.mica.output_dir = cfg.output_dir # !!take a look!! it uses duplicate form
    #  creat folders
    for i in cfg.path:
        os.makedirs(os.path.join(cfg.output_dir, cfg.path[i]), exist_ok=True)
    
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # cudnn related setting - SR
    
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    # torch.cuda.empty_cache()
    # deterministic(rank) - MICA

    from lib.trainer import Trainer
    nfc = util.find_model_using_name(model_dir='lib.MICA.micalib.models', model_name=cfg.mica.model.name)(cfg.mica, cfg.gpu_ids) # int(gpu_list[0])
    trainer = Trainer(model=nfc, config=cfg)
    
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