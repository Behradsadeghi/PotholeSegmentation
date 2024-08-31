# 🚧 Pothole Detection and Segmentation with YOLOv9

This project leverages the YOLOv9 model for detecting and segmenting potholes in images and videos. The model is trained on a custom dataset using advanced techniques like mixed precision training, gradient accumulation, and optimized hyperparameters. This README outlines the project's setup, features, and key results.

## 📚 Table of Contents

- [🌟 Introduction](#-introduction)
- [✨ Features](#-features)
- [⚙️ Installation](#%EF%B8%8F-installation)
- [📂 Dataset](#-dataset)
- [🛠️ Training](#%EF%B8%8F-training)
- [📊 Results](#-results)
- [🎥 Inference on Video](#-inference-on-video)
- [📝 License](#-license)

## 🌟 Introduction

The goal of this project is to accurately detect and segment potholes from images and videos using a custom-trained YOLOv9 model. The model is fine-tuned on a pothole segmentation dataset, achieving robust performance metrics. This README covers how to set up the environment, train the model, and perform inference on images and videos.

## ✨ Features

- **YOLOv9c-seg Model:** Utilizes a custom-trained YOLOv9 model specifically for pothole detection and segmentation.
- **Data Augmentation:** Includes random horizontal flips, resizing, and color jitter to improve model generalization.
- **Optimized Training:** Hyperparameters like learning rate, batch size, and dropout rate are optimized for better performance.
- **Mixed Precision Training:** Speeds up training and reduces memory consumption.
- **Inference on Video:** The model is capable of making real-time inferences on video streams.

## ⚙️ Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- PyTorch
- YOLOv9 (ultralytics package)
- OpenCV
- Other dependencies: `numpy`, `matplotlib`, `seaborn`, `Pillow`, `ffmpeg`

### Setup

1. Clone the repository:
   - `git clone <repository-url>`
   - `cd <repository-directory>`

2. Install the required Python packages:
   - `pip install ultralytics numpy matplotlib seaborn opencv-python pillow`

3. Download the dataset:
   - Use the Kaggle CLI or download manually from the Kaggle website.

## 📂 Dataset

The dataset used for training and validation contains images of roads with labeled potholes. The images are divided into `train`, `valid`, and `test` directories:

- **Train Directory:** Contains the images used for training.
- **Validation Directory:** Used to validate the model's performance during training.
- **Test Directory:** Used to evaluate the final model's performance.

You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/farzadnekouei/pothole-image-segmentation-dataset).

## 🛠️ Training

The model is trained using the following setup:

- **Model:** YOLOv9 with segmentation head (`yolov9c-seg.pt`)
- **Epochs:** 30
- **Image Size:** 512x512 pixels
- **Batch Size:** 30
- **Optimizer:** AdamW
- **Learning Rate:** Initial `lr0=0.00001`, Final `lrf=0.01`
- **Dropout Rate:** 0.15
- **Device:** CUDA (GPU)

### Training Overview

- **Data Augmentation:** Techniques such as random flipping, resizing, and color jitter are applied to the images to enhance the model's ability to generalize.
- **Model Training:** The model is trained over multiple epochs, and its performance is evaluated on a validation set after each epoch.
- **Checkpointing:** The best model weights based on validation performance are saved during training.
- **Hyperparameter Tuning:** Key parameters such as the learning rate, batch size, and dropout rate are optimized for best performance.

## 📊 Results

After training, the model achieved the following performance metrics:

- **Precision (P):** 0.757
- **Recall (R):** 0.726
- **mAP@50:** 0.807
- **mAP@50-95:** 0.519
- **Mask Precision (P):** 0.763
- **Mask Recall (R):** 0.746
- **Mask mAP@50:** 0.825
- **Mask mAP@50-95:** 0.498

These metrics indicate strong performance in detecting and segmenting potholes across various test images.

## 🎥 Inference on Video

The model can perform real-time inference on video streams to detect and segment potholes:

- **Video Processing:** The trained model is used to process each frame of the video and apply the segmentation mask.
- **Video Output:** The processed video is saved in a specified format and can be viewed to assess the model's performance in real-world scenarios.

To convert the video to a suitable format (e.g., MP4) and view the results, use tools like FFmpeg.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
