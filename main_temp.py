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

import torch.multiprocessing as mp

# import core.logger as Logger


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch.distributed as dist

def setup(rank, world_size):

    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',  # Use NCCL for GPUs
        init_method='env://',  # Initialize using environment variables
        rank=rank,  # Unique ID for the process
        world_size=world_size  # Total number of processes
    )
    torch.cuda.set_device(rank)  # Set the GPU device for the current process

def cleanup():
    dist.destroy_process_group()

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
    cfg.mica.output_dir = cfg.output_dir # !!take a look!! it uses duplicate form
    #  creat folders
    for i in cfg.path:
        os.makedirs(os.path.join(cfg.output_dir, cfg.path[i]), exist_ok=True)
    
    cfg.path.checkpoint = os.path.join(cfg.output_dir, cfg.path.checkpoint)
    
    # Copy the file
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, cfg.train.log_dir, 'config.yml'))
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # cudnn related setting - SR
    
    # !!! Check it again
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True
    torch.cuda.empty_cache()
    # deterministic(rank) - MICA

    world_size = torch.cuda.device_count()

    # Launch training
    mp.spawn(main_worker, args=(world_size, cfg), nprocs=world_size, join=True)
    
def main_worker(rank, world_size, cfg):
    # Setup distributed environment
    setup(rank, world_size)

    # Initialize Trainer
    from lib.trainer_temp import Trainer
    trainer = Trainer(config=cfg, rank=rank, world_size=world_size)

    # Start training (you can implement this in `Trainer`)
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
# python main_temp.py -p train -c config/sr_sr3_VGGF2_test_code.yml