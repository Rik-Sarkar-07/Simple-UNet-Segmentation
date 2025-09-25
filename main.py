import argparse
import os
import torch
from dataset.image_dataset import get_dataset
from models.unet import UNet
from tools.utils import setup_seed, save_checkpoint
from tools.losses import get_loss
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Simple U-Net Image Segmentation (Single GPU)')
    parser.add_argument('--dataset', default='cityscapes', help='dataset name: cityscapes, voc')
    parser.add_argument('--data_dir', default='./data', help='directory to store/download dataset')
    parser.add_argument('--model', default='unet', help='model name')
    parser.add_argument('--depth', type=int, default=4, help='U-Net depth')
    parser.add_argument('--num_channels', type=int, default=32, help='U-Net base channels')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--opt', default='adamw', help='optimizer')
    parser.add_argument('--sched', default='cosine', help='scheduler')
    parser.add_argument('--output_dir', default='./outputs', help='output directory')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('--initial_checkpoint', default='', help='path to checkpoint')
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, epoch, device):
    model.train()
    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch}"):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluation"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")

def main():
    args = parse_args()
    setup_seed(42)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available, using CPU. Performance may be slow.")

    # Dataset
    train_dataset, val_dataset, num_classes = get_dataset(args.dataset, args.data_dir, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = UNet(num_classes=num_classes, depth=args.depth, num_channels=args.num_channels).to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = get_loss('cross_entropy')

    # Load checkpoint if provided
    if args.initial_checkpoint:
        checkpoint = torch.load(args.initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])

    # Training/Evaluation Loop
    if args.eval:
        evaluate(model, val_loader, criterion, device)
    else:
        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, device)
            evaluate(model, val_loader, criterion, device)
            save_checkpoint(model, optimizer, epoch, args.output_dir)

if __name__ == '__main__':
    main()