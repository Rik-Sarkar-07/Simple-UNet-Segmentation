import torch
import os
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, optimizer, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))