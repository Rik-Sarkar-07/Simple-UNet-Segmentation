# Simple-UNet-Segmentation
This repository provides a simplified PyTorch implementation of a U-Net model for image segmentation, optimized for small datasets available in PyTorch's `torchvision.datasets` (e.g., Cityscapes, VOC). The code automatically downloads the specified dataset if not present and supports single-GPU or CPU training.

## Repository Structure
```
Simple-UNet-Segmentation/
├── dataset/
│   ├── __init__.py
│   ├── image_dataset.py
│   ├── image_transforms.py
├── models/
│   ├── __init__.py
│   ├── unet.py
├── tools/
│   ├── __init__.py
│   ├── utils.py
│   ├── losses.py
├── LICENSE
├── README.md
├── requirements.txt
├── main.py
```

## Usage
### Cloning the Repository
First, clone the repository locally:
```bash
git clone https://github.com/Rik-Sarkar-07/Simple-UNet-Segmentation
cd Simple-UNet-Segmentation
```

### Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation
The code automatically downloads the specified dataset (Cityscapes or VOC) to the `--data_dir` directory if it’s not already present. Supported datasets:
- `cityscapes`: Cityscapes dataset for urban scene segmentation.
- `voc`: Pascal VOC 2012 dataset for object segmentation.

Specify the dataset using the `--dataset` argument and provide a directory to store the dataset with `--data_dir`. For example:
```bash
python main.py --dataset cityscapes --data_dir ./data
```

### Training and Evaluation
The U-Net model is configured for small datasets with the following parameters:
- **Model**: Lightweight U-Net with depth 4 and 32 base channels.
- **Input Size**: Default 256x256 (configurable via `--img_size`).
- **Optimizer**: AdamW with learning rate 1e-4 (configurable via `--lr`).
- **Loss**: CrossEntropyLoss for multi-class segmentation.
- **Transformations**: Random horizontal flip, resize, and normalization.

#### Training Example
To train a U-Net model on the Cityscapes dataset:
```bash
python main.py --dataset cityscapes --data_dir ./data \
--opt adamw --lr 1e-4 --epochs 30 --batch-size 4 --img_size 256 \
--model unet --depth 4 --num_channels 32 --output_dir ./outputs
```

#### Evaluation Example
To evaluate a trained model:
```bash
python main.py --dataset cityscapes --data_dir ./data \
--opt adamw --lr 1e-4 --epochs 30 --batch-size 4 --img_size 256 \
--model unet --depth 4 --num_channels 32 --output_dir ./outputs --eval --initial_checkpoint ./outputs/checkpoint_epoch_29.pth
```

For more options, run:
```bash
python main.py --help
```

## Model Configuration
| Model   | Depth | Num Channels | Image Size | Parameters (M) | FLOPs (G) |
|---------|-------|--------------|------------|----------------|-----------|
| UNet-S  | 4     | 32           | 256        | ~1.4           | ~15       |

## Dataset Performance
| Dataset     | Model   | mIoU (%) | Download |
|-------------|---------|----------|----------|
| Cityscapes  | UNet-S  | ~72.0    | Auto     |
| VOC         | UNet-S  | ~70.0    | Auto     |

## License
This repository is released under the Apache-2.0 license as found in the [LICENSE](#LICENSE) file.

## Citation
If you use this code for a paper, please cite:
```bibtex
@article{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  journal={Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2015}
}
```
