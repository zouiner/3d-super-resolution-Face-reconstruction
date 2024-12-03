import os, sys
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
# mica
from lib.MICA.utils import util

# import core.logger as Logger


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))



def main(rank, world_size, cfg):
    
    # Set the rank-specific device
    setup(rank, world_size)
    

    # Update config with distributed settings
    cfg.gpu_ids = [i for i in range(len(cfg.device_id))]
    cfg.rank = rank
    cfg.world_size = world_size
    cfg.distributed = True
    
    
    
    # Create directories and copy config files (only rank 0 handles this)
    cfg.path.tb_logger = os.path.join(cfg.output_dir, cfg.train.log_dir, cfg.path.tb_logger)
    cfg.mica.output_dir = cfg.output_dir
    for i in cfg.path:
        os.makedirs(os.path.join(cfg.output_dir, cfg.path[i]), exist_ok=True)

    cfg.path.checkpoint = os.path.join(cfg.output_dir, cfg.path.checkpoint)

    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, cfg.train.log_dir, 'config.yml'))
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # cudnn related setting - SR
    
    # !!! Check it again
    cudnn.benchmark = True
    cudnn.deterministic = True
    cudnn.enabled = True

    from lib.trainer_temp import Trainer
    trainer = Trainer(config=cfg, rank=rank, world_size=world_size)

    trainer.fit()

    dist.destroy_process_group()

def setup(rank, world_size):
    # Set environment variables for master address and port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ["LOCAL_RANK"] = "0"

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # Set device for each process
    torch.cuda.set_device(rank)

    # Your training logic here
    print(f"Rank {rank}/{world_size} initialized.")
    
def config():
    from config.default.config import parse_args
    
    # parse configs
    args = parse_args()
    
    return args

# if __name__ == '__main__':
#     cfg = config()

#     world_size = len(cfg.device_id)
#     rank = int(os.environ['LOCAL_RANK'])  # Automatically set by torchrun

#     main(rank, world_size, cfg)
    
if __name__ == '__main__':
    cfg = config()
    
    torch.cuda.empty_cache()
    
    gpu_list = ','.join(str(id) for id in cfg.device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # world_size = len(cfg.device_id)  # Number of GPUs
    world_size = torch.cuda.device_count()
    
    mp.spawn(main, args=(world_size, cfg), nprocs=world_size, join=True)


# run:
# python main_temp.py -p train -c config/sr_sr3_VGGF2_test_code.yml