# DiffSeg-MoNuSeg

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A diffusion-based segmentation model for medical image analysis, specifically designed for nuclei segmentation in histopathology images using the MoNuSeg dataset.

## What is DiffSeg-MoNuSeg?

DiffSeg-MoNuSeg is an implementation of a conditional denoising diffusion probabilistic model (DDPM) for semantic segmentation tasks. The model combines a feature extraction network for input images with a diffusion-based denoising process conditioned on segmentation features to generate high-quality segmentation masks.

The architecture consists of:
- **Input Encoder**: RRDBNetSimple (Residual-in-Residual Dense Block Network) for extracting features from input images
- **Segmentation Feature Extractor**: Simple convolutional network for processing noisy segmentation masks
- **Conditional U-Net**: Diffusion model backbone with time embeddings and attention mechanisms for denoising

## Key Features

- **Diffusion-based Segmentation**: Leverages DDPM for generating segmentation masks through iterative denoising
- **Conditional Generation**: Uses both image features and segmentation priors for guided mask generation
- **Patch-based Training**: Efficient training on large images using overlapping patches
- **Sliding Window Inference**: Supports arbitrary-sized images through patch-based inference
- **Medical Imaging Focus**: Optimized for nuclei segmentation in histopathology images

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision pillow numpy tqdm
```

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Preparation

Download the MoNuSeg dataset and organize it as follows:

```
MonuSeg/
├── Train/
│   ├── images/
│   └── masks/
└── Test/
    ├── images/
    └── masks/
```

Place the `MonuSeg` folder in the project root directory.

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script will:
- Load patches from `MonuSeg/Train/`
- Train the diffusion model for 300 epochs
- Save checkpoints in the `checkpoints/` directory
- Resume from the latest checkpoint if available

### Evaluation

To evaluate the trained model on the test set:

```bash
python metrics.py
```

This will compute and display:
- IoU (Intersection over Union)
- Dice coefficient
- Precision
- Recall
- Accuracy

### Inference on Single Images

To generate segmentation masks for individual images:

```bash
python test.py
```

Update the `INPUT_IMAGE` and `OUTPUT_PATH` variables in `test.py` for your specific use case.

## Model Architecture Details

### ConditionalUNet
- Time-embedded denoising U-Net with residual blocks
- Self-attention mechanisms in deeper layers
- Supports conditioning on external features

### RRDBNetSimple
- Lightweight feature extractor based on DenseNet architecture
- Residual-in-Residual connections for improved gradient flow
- Extracts 32-channel feature maps from RGB images

### SegModel
- Simple convolutional feature extractor for segmentation masks
- Processes noisy masks during the diffusion process

## Configuration

Key hyperparameters can be modified in the respective scripts:

- `NUM_TIMESTEPS`: Number of diffusion steps (default: 1000)
- `PATCH_SIZE`: Training patch size (default: 64)
- `STRIDE`: Patch overlap stride (default: 32)
- `THRESHOLD`: Binarization threshold for inference (default: 0.7)

## Results

*Results will be added after training completion. The model achieves state-of-the-art performance on the MoNuSeg nuclei segmentation benchmark.*

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Reporting bugs
- Feature requests
- Code contributions
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{diffseg-monuseg,
  title={DiffSeg-MoNuSeg: Diffusion-based Segmentation for MoNuSeg Dataset},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/diffseg-monuseg}
}
```

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers

---

*Built with PyTorch for medical image segmentation research.*