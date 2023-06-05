import os
import torch
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.utils as vutils
from torch.distributed import init_process_group


def load_config(path):
    with open(path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs
    return configs


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


def save_generated_image(generator, nz, device):
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(datetime.now()) + ".png")

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    with torch.no_grad():
        generated = generator(fixed_noise).detach().cpu()
    generated = vutils.make_grid(generated, padding=2, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(generated, (1, 2, 0)))
    plt.savefig(save_path)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
