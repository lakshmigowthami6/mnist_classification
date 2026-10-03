# MNIST Digit Classification using TensorFlow and Keras

This repository contains a simple feedforward neural network built with TensorFlow and Keras to classify handwritten digits from the classic MNIST dataset. It includes preprocessing, training with early stopping, model evaluation, and loss/accuracy visualization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training & Optimization](#training--optimization)
- [Performance & Results](#performance--results)
- [How to Run](#how-to-run)

---

## Project Overview
The goal of this project is to build an artificial neural network (ANN) capable of recognizing handwritten digits (0-9). The model processes 28x28 grayscale images and outputs the predicted digit class with high accuracy.

## Dataset
The project uses the **MNIST dataset**, which consists of:
- **60,000** training images
- **10,000** testing images
- Grayscale pixels valued from `0` to `255` representing handwritten digits from `0` to `9`.

### Preprocessing
- Pixel values are normalized by dividing by `255.0` to scale features between `0` and `1`. This helps the neural network converge faster during gradient descent.

## Model Architecture
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```
- **Flatten Layer:** Reshapes the 2D 28x28 image arrays into a 1D vector of size 784.
- **Hidden Layer:** A dense layer with 128 units and **ReLU** (Rectified Linear Unit) activation.
- **Output Layer:** A dense layer with 10 units representing class probabilities using **Softmax** activation.

## Training & Optimization
- **Loss Function:** `sparse_categorical_crossentropy` (suitable for integer-labeled target classes).
- **Optimizer:** `Adam`.
- **Metrics:** `Accuracy`.
- **Early Stopping:** Implemented to prevent overfitting by monitoring `val_loss` with a patience of 20 epochs.

## Performance & Results
- **Test Accuracy:** ~`97.82%`
- The training stopped early around **Epoch 20** due to the `EarlyStopping` callback because validation loss stopped improving, preserving the generalization capacity of the network.

### Loss & Accuracy Plots
The model records history to plot learning curves:
- **Loss Curves:** Comparing `loss` vs. `val_loss`.
- **Accuracy Curves:** Comparing `Accuracy` vs. `val_Accuracy`.

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow matplotlib scikit-learn numpy
   ```
2. Run the training script or Jupyter Notebook cells sequentially.
