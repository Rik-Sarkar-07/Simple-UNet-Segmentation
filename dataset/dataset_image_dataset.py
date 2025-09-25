import torch
from torchvision import datasets, transforms
from .image_transforms import get_transforms
import os

def get_dataset(dataset_name, data_dir, img_size):
    os.makedirs(data_dir, exist_ok=True)
    train_transform, val_transform = get_transforms(img_size)
    
    if dataset_name == 'cityscapes':
        train_dataset = datasets.Cityscapes(data_dir, mode='fine', target_type='semantic', transform=train_transform, target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)))
        val_dataset = datasets.Cityscapes(data_dir, mode='fine', split='val', target_type='semantic', transform=val_transform, target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)))
        num_classes = 19  # Cityscapes has 19 classes for fine annotations
    elif dataset_name == 'voc':
        train_dataset = datasets.VOCSegmentation(data_dir, year='2012', image_set='train', download=True, transform=train_transform, target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)))
        val_dataset = datasets.VOCSegmentation(data_dir, year='2012', image_set='val', download=True, transform=val_transform, target_transform=transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long)))
        num_classes = 21  # VOC has 21 classes (including background)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, val_dataset, num_classes