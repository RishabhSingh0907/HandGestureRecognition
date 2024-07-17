# Hand Gesture Recognition Model

This project utilizes a deep learning-based approach to recognize hand gestures. The system is divided into four main parts: Data Creation, Data Annotation, Model Training, and Gesture Recognition. The project leverages the power of GPU acceleration using CUDA toolkit to significantly reduce training time.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Creation](#data-creation)
- [Data Annotation](#data-annotation)
- [Model Training](#model-training)
- [Results](#results)

## Overview
This project aims to develop a hand gesture recognition model using deep learning techniques. The gestures are captured, annotated, and used to train a neural network to accurately recognize different hand gestures. The model achieves an accuracy of over 94%.

## Installation
To get started with this project, you need to have the following dependencies installed:
- Python 3.x
- Pytorch
- Mediapipe
- CUDA Toolkit (for GPU acceleration)
- OpenCV
- Pandas
- NumPy

You can install the required Python packages using pip:
```sh
pip install mediapipe opencv-python pandas numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu112
```

## Data Creation
The first step involves creating the dataset for the gestures you want to recognize. Capture images of hand gestures and store them in a directory. Ensure that you have sufficient images for each gesture to train the model effectively.

## Data Annotation
In this step, we perform the following tasks:
1. **Image Filtering**: Keep the most appropriate images from the dataset.
2. **Hand Region Extraction**: Use Mediapipe to detect hand landmarks and extract the hand region from each image.
3. **CSV File Creation**: Create a CSV file that stores the path of the gesture images along with the gesture name. The CSV file should have two columns: `location` (image path) and `gesture` (gesture name).

## Model Training
To train the deep learning model, follow these steps:
1. Load the annotated data from the CSV file.
2. Preprocess the images and labels.
3. Define the neural network architecture using Pytorch.
4. Train the model 
5. Evaluate the model on the test dataset.

If your system is equipped with GPU, train the model using GPU acceleration. This reduces a lot of time when compared to the training through CPU (ensure CUDA Toolkit is properly installed and configured).

## Results
The model achieves an accuracy of over 94% on the test dataset. The real-time recognition system is capable of accurately recognizing hand gestures with minimal latency.
