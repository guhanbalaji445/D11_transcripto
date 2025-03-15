# Assignment 3: Deep Learning

This assignment is designed to give an introduction to some of the fundamental concepts in deep learning. Specifically, you will be learning to implement basic neural networks and then progress to convolutional neural networks (CNNs) using one of TensorFlow/PyTorch. 

## Why do we need deep learning?

Deep learning has shown remarkable performance in image and speech recognition, natural language processing, and many other domains. It leverages large neural networks to learn complex patterns and representations from data.

### Objectives

- Gain a good intuition on the working of neural networks.
- Learn about convolutional neural networks.
- Implement deep learning using TensorFlow/PyTorch.

## Dataset

Use the Flowers Recognition dataset (\~4317 images of 5 different species) from: [Kaggle - Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

## Learning Resources

- [Neural networks - YouTube (3Blue1Brown)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Neural Networks / Deep Learning - YouTube (StatQuest)](https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [But what is a convolution?](https://youtu.be/KuXjwB4LzSA)
- [Convolutional Neural Network - YouTube](https://www.youtube.com/playlist?list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu)
- [Learn PyTorch in a day](https://youtu.be/Z_ikDlimN6A?si=diw9-rJVI_GBHspF)
- [Tutorials | TensorFlow Core](https://www.tensorflow.org/tutorials)
- [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/)
- [Welcome to PyTorch Tutorials — PyTorch Tutorials 2.6.0+cu124 documentation](https://pytorch.org/tutorials/)
- Additional resources: Google, ChatGPT, DeepSeek, etc.
- Refer to [D11 Timeline.md](https://github.com/f3ltz/D11_transcripto/blob/main/D11%20Timeline.md) for further recommended resources.

## Task 1: Implement a basic neural network model to classify flowers

### Data Pipeline

- Load Dataset: Download from Kaggle.
- Preprocessing:
  1. Resize images to 128x128.
  2. Normalize pixel values to [0, 1].
  3. One-hot encode labels.
  4. Split into training and validation sets (80:20).
  5. Flatten the images to feed to the network.
- Data Augmentation: Use random flips and rotations to enhance generalization.

### Train a Basic Neural Network

- **Architecture:**
  - Input: Flattened image vectors.
  - Hidden Layers: 1-2 dense layers with ReLU activation.
  - Output: Softmax activation.
- **Compile:**
  - Loss: Categorical cross-entropy.
  - Optimizer: Adam or any other optimizer.
- Train the model on the training data.
- Predict flower species.

### Evaluate the Model

- Plot training/test accuracy and loss curves.
- Generate a confusion matrix to analyze performance.

## Task 2: Improve accuracy using a Convolutional Neural Network

### Data Pipeline

- Use the same preprocessed data as in Task 1. Don’t flatten the images, pass them directly to the model.
- Perform data augmentation (e.g., random rotations, flips, zoom).

### Train the Model

- **Architecture:**
  - Input: Convolutional layers with ReLU activation and max pooling.
  - Hidden Layers: 2-3 convolutional layers followed by a flatten layer.
  - Dense Layers: 1-2 fully connected layers with ReLU activation.
  - Output: Softmax activation.
- **Compile:**
  - Loss: Categorical cross-entropy.
  - Optimizer: Adam or any suitable optimizer.

### Evaluate the Model

- Plot training/test accuracy and loss curves.
- Generate a confusion matrix to analyze performance.

## Submission Guidelines

### Notebook/Scripts

- Submit your work in a Jupyter Notebook (.ipynb) or a Python script.
- Ensure that all code is well-structured and outputs (plots, tables, etc.) are included.

### Documentation

- Add inline comments to explain key steps and logic in your code.

### Report

- Submit a brief report (2-4 pages) summarizing:
  - Evaluation Reports (F1-score, Accuracy, Loss, etc.)
  - Challenges faced.
  - Why is a CNN better at classifying images than a standard neural network?

### Deadline

- Submit your assignment by Friday, March 21.

