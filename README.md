# PRODIGY_GA_04
# Pix2Pix Image-to-Image Translation (Without GPU)

## 📌 Overview
This project implements the Pix2Pix conditional GAN for image-to-image translation using PyTorch. It is designed to work **without a GPU**, making it accessible for CPU-based training and inference.

## 🚀 Features
- Implements **Pix2Pix (cGAN) for paired image-to-image translation**
- Works **entirely on CPU**, making it accessible for non-GPU users
- **Uses PyTorch** with custom dataset loading
- Includes a simple **generator and discriminator architecture**
- Supports training and testing with the **Facades dataset**

## 📂 Dataset
The project uses the **Facades dataset**, where each image is split into an input-output pair.
- Download the dataset: [Facades Dataset](https://github.com/phillipi/pix2pix)
- Place images in `data/facades/train/`, `data/facades/test/`, and `data/facades/val/`

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pix2pix-cpu.git
   cd pix2pix-cpu
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy pillow matplotlib
   ```
3. Ensure the dataset is correctly placed inside `data/facades/`

## 🏗 Model Architecture
### **Generator (Basic CNN-based)**
- Uses convolutional and transposed convolutional layers
- Outputs an image with `Tanh` activation

### **Discriminator (PatchGAN-based)**
- Takes both real and generated images
- Predicts if the image is real or fake

## 🎯 Training
To train the model, run:
```bash
python train.py
```
### **Training Details**
- Optimizer: Adam (`lr=0.0002`)
- Loss functions: **Binary Cross-Entropy Loss** + **L1 Loss**
- Batch size: 1

## 🖼 Testing & Inference
To generate images after training:
```bash
python generate.py
```

## 📊 Results
Sample generated images after training:

![Sample Output] (sample_image.png.png)

## 🏆 Acknowledgments
- This project is based on the [Pix2Pix paper](https://arxiv.org/abs/1611.07004)
- Inspired by implementations from PyTorch tutorials

## 🤝 Contributions
Feel free to fork and submit PRs! For issues, open a ticket.

## 📜 License
This project is licensed under the MIT License.


