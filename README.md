# Brain Tumor MRI Classification Using CNN

A deep learning project that classifies brain MRI images into four categories using a custom Convolutional Neural Network built from scratch with PyTorch.

## Project Overview

This project uses a custom CNN model to classify brain MRI scans into:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

The trained model is included in the `checkpoints` folder, allowing users to perform predictions directly without retraining the model.

## Features

- Custom CNN architecture built using PyTorch
- Four-class brain tumor MRI classification
- Trained model checkpoint included
- GPU support through CUDA when available
- Command-line image prediction
- Image visualization with predicted class and confidence

## Dataset

The dataset used in this project is the Brain Tumor MRI Dataset.

Dataset source:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The classes used in this project are:

1. Glioma
2. Meningioma
3. No Tumor
4. Pituitary

## Model Architecture

The CNN contains four convolutional blocks.

Each convolutional block consists of:

- Two convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling

The model then uses:

- Adaptive Average Pooling
- Flatten layer
- Dropout
- Fully Connected layers
- Final classification layer

The final layer produces four output values, one for each class.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- tqdm

## Project Structure

```text
Brain-CNN-Model/
│
├── checkpoints/
│   └── best_model.pth
│
├── models/
│   └── cnn.py
│
├── utils/
│
├── config.py
├── predict.py
├── train.py
├── test.py
├── requirements.txt
└── README.md

## How to Run the Project
There are 2 ways : a. download and train model again - else 
b. Use directly trained model and predict . 

1. Download the repository as a ZIP file and extract it.
2. Open the extracted project folder in VS Code.
3. Install the required dependencies:

```bash
pip install -r requirements.txt

Run predict.py with the path to an MRI image: py predict.py "C:\Path\To\Your\MRI\Image.jpg"

Example: py predict.py "C:\Users\YourName\Pictures\brain_scan.jpg"

The trained model is already included in:
checkpoints/best_model.pth
