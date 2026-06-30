# 🚗 Real-Time License Plate Detector 🤖

A custom-trained computer vision project using YOLOv8 and OpenCV to detect and identify license plates in live traffic video.

## ✨ Features
- 🎯 **High Accuracy:** Custom-trained YOLOv8 model specialized for plate detection.
- ⚡ **Real-time:** Optimized for live video processing using OpenCV.
- 🌦️ **Robust:** Trained on a diverse dataset to handle various lighting and angles.

## 🛠️ Tech Stack
- **AI/ML:** [Ultralytics YOLOv8](https://ultralytics.com/)
- **Language:** Python
- **Libraries:** OpenCV, PyTorch, NumPy

## 🐛 Challenges & Debugging
- **Data Transfer:** Overcame network-related file corruption during 900MB+ dataset uploads using `gdown` and Google Colab.
- **Environment:** Resolved directory mounting errors and `Transport endpoint` issues by cleaning local file systems.
- **Development:** Fixed YAML syntax and configuration logic with AI-assisted debugging to optimize model performance.

## 🎥 Demo
![Detection Result](Screenshot%202026-07-01%20004818.png)

## 🚀 How to Run
1. **Install requirements:** 
   ```bash
   pip install -r requirements.txt
   python main.py
