# Industrial Defect Detection using Autoencoders

An unsupervised deep learning system for industrial anomaly detection trained only on defect-free images.

## 🚀 Overview
This project uses a U-Net based convolutional autoencoder to learn the distribution of normal images from the MVTec dataset. Defects are detected via reconstruction errors.

## 🧠 Key Ideas
- Train only on normal images
- Reconstruction error highlights anomalies
- Hybrid loss improves reconstruction quality

## ⚙️ Methodology
- Architecture: U-Net Autoencoder
- Loss: MSE + SSIM
- Post-processing:
  - Gaussian smoothing
  - Error normalization
- Scoring:
  - Top 5% pixel errors
  - Threshold = μ + 3σ

## 📊 Results
- Pixel-level AUROC: ~0.83
- Good defect localization performance

## 🖥️ Demo
Gradio interface showing:
- Input image
- Reconstructed image
- Anomaly heatmap

## 📦 Tech Stack
- PyTorch
- OpenCV
- NumPy
- Gradio