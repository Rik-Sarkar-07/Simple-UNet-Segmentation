import torch.nn as nn

def get_loss(loss_name):
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(ignore_index=255)  # 255 is often used for void classes
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")