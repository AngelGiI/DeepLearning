# Deep Learning Projects Overview

This repository contains projects and assignments completed as part of the **Deep Learning** course. The projects focus on both foundational and advanced concepts, including manual implementation of neural networks, automatic differentiation, and convolutional neural networks (CNNs).

---

## Assignment 1: Implementing MLP Using NumPy
- **Objective**: Build a Multi-Layer Perceptron (MLP) from scratch using NumPy, including forward and backward propagation.
- **Topics**: 
  - Backpropagation  
  - Cross-entropy loss with softmax activation  
  - Stochastic Gradient Descent (SGD)  
- **Technologies**: Python, NumPy  
- **Key Outcomes**:  
  - Implemented and trained a simple neural network on MNIST with manual backpropagation.  
  - Dynamic learning rate adjustment for improved convergence.  
  - Achieved **96.3% accuracy** on the MNIST validation set.  

---

## Assignment 2: Automatic Differentiation
- **Objective**: Understand and extend a mini-library for automatic differentiation.
- **Topics**:
  - Derivation of gradients for scalar, vector, and matrix operations.  
  - Building a computation graph and implementing backpropagation.  
  - Implementation of backward passes for custom operations.  
- **Technologies**: Python, NumPy  
- **Key Outcomes**:  
  - Developed backward functions for element-wise and matrix operations.  
  - Created a custom ReLU operation and compared its performance against Sigmoid.  
  - Explored network optimizations and different gradient strategies.  

---

## Assignment 3A: Convolutional Neural Networks
- **Objective**: Implement and optimize CNNs for image classification tasks.  
- **Topics**:
  - Convolution operations (manual and PyTorch)  
  - Hyperparameter tuning (learning rate, batch size)  
  - Data augmentation techniques  
- **Technologies**: Python, PyTorch  
- **Key Outcomes**:  
  - Designed a CNN achieving **99.2% accuracy** on the baseline and **99.4% with augmentation**.  
  - Implemented custom forward and backward passes for convolution layers.  
  - Compared global max pooling and global mean pooling, highlighting performance differences.  

---

## Assignment 4: Multi-Resolution CNNs
- **Objective**: Train CNNs to handle variable-resolution images and evaluate performance.  
- **Topics**:
  - Multi-resolution input handling  
  - Global pooling strategies  
  - Parameter efficiency and model comparison  
- **Technologies**: Python, PyTorch  
- **Key Outcomes**:  
  - Implemented multi-resolution training loops.  
  - Compared performance of fixed-resolution vs. variable-resolution networks.  
  - Tuned hyperparameters to achieve **97.69% accuracy** on a variable-resolution network.  

---

## Key Skills and Tools Used
- **Deep Learning Concepts**: Backpropagation, gradient computation, CNNs, pooling layers, and data augmentation.  
- **Technologies**: Python, NumPy, PyTorch  
- **Libraries**:  
  - PyTorch for model training and deployment.  
  - NumPy for low-level operations and vectorization.  

---

## How to Navigate the Projects
Each assignment is organized into its own folder. You can open the respective Jupyter Notebooks or Python scripts for code, explanations, and results.  

---

