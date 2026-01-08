Hand Gesture & Digit Recognition using CNN: This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras for image classification. The model is trained on the MNIST dataset and extended to support real-time webcam-based gesture recognition using OpenCV.

Tech Stack:
-Python
-TensorFlow / Keras
-NumPy
-OpenCV
-Matplotlib
-Scikit-learn

Dataset:
-MNIST handwritten digits
-Image size: `28 × 28` (grayscale)
-Classes: `0–9`

Workflow:
-Data loading and normalization
-Data augmentation
-CNN model training and evaluation
-Performance analysis using accuracy, classification report, and confusion matrix
-Model saving and inference
-Real-time webcam prediction using OpenCV

Run:
```bash
python main.py
`````

Results:
-Test Accuracy: ~98.9%
-Strong precision, recall, and F1-score across all classes

Notes: This project demonstrates a complete **deep learning pipeline**, from model training to real-time inference. It can be extended for custom hand gesture datasets and real-world applications.

Author: Sumit Senapati
