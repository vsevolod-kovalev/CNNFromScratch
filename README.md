# CNNFromScratch
<img width="1010" alt="Screenshot 2024-11-25 at 4 50 49 PM" src="https://github.com/user-attachments/assets/ca4ce8eb-7e9a-4d3c-8c56-9fd6fff96cf4">
  <p>Feature visualization of convolutional net trained on ImageNet from [Zeiler & Fergus 2013].</p>

**CNNFromScratch** is a custom implementation of a **Convolutional Neural Network (CNN)** for recognizing and classifying images from the CIFAR-10 dataset. This project recreates functionalities similar to **TensorFlow's Sequential**, allowing the creation of models with an arbitrary number of layers, depth, and parameters.

---

## Features

- **Layer Implementations**:
  - **Convolutional 2D Layer**: For feature extraction.
  - **Fully Connected Layer**: For classification.
  - **Flatten Layer**: Converts tensors into 1D vectors.
  - **Average Pooling Layer**: Reduces spatial dimensions by averaging.
  - **Max Pooling Layer**: Reduces spatial dimensions by selecting maximum values.

**Custom Models**: Design and train custom architectures by stacking layers sequentially.

---

## **LeNet-5 Implementation**:

The following example demonstrates the construction of **LeNet-5** using the `Sequential` class:
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/6f00f336-91bc-48e1-a75e-293140909a56">


  <p>LeNet-5 model visualization to recognize gray-scale 28x28 images.</p>

  ```python
    from Sequential import *

    model = Sequential([
      # Layer 1: Convolutional layer
      Conv2D(filters=6, kernel_size=5, strides=1, padding='valid', activation='relu'),
      AveragePooling2D(pool_size=2, strides=2),
  
      # Layer 2: Convolutional layer
      Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'),
      AveragePooling2D(pool_size=2, strides=2),
  
      # Flatten the 2D feature maps into a 1D vector
      Flatten(),
  
      # Fully connected layer 1
      Dense(120, activation='relu'),
  
      # Fully connected layer 2
      Dense(84, activation='relu'),
  
      # Output layer: 10 classes for classification
      Dense(10, activation='softmax')
    ])
    
    # Adjust input shape and compilation
    input_shape = [1, 32, 32]
    model.compile(optimizer='sgd', loss='cce', learning_rate=0.01, input_shape=input_shape)
    
    # Train the model
    model.fit(data.X_train, data.Y_train, epochs=30)
  ```

# Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 classes, with 6,000 images per class. Below are the dataset classes along with example images:

<div align="center">
## Example Images by Class

  | Class Name   | Example Images                                              |
  |--------------|-------------------------------------------------------------|
  | Airplane     | <img src="https://github.com/user-attachments/assets/44aae37f-81eb-4499-8969-614d51a6f483" width="550"> |
  | Automobile   | <img src="https://github.com/user-attachments/assets/649f34b0-8fd7-4de1-9b17-13e65021804a" width="550"> |
  | Bird         | <img src="https://github.com/user-attachments/assets/d12aaf91-209f-45a9-8158-d35ff5a1b499" width="550"> |
  | Cat          | <img src="https://github.com/user-attachments/assets/36b33ebf-9ee8-48aa-965b-972924b078f8" width="550"> |
  | Deer         | <img src="https://github.com/user-attachments/assets/ba8b7fae-5a86-4575-b4be-10ee0190f5fc" width="550"> |
  | Dog          | <img src="https://github.com/user-attachments/assets/42f82d02-3912-49a5-9b9a-9040dff792d0" width="550"> |
  | Frog         | <img src="https://github.com/user-attachments/assets/527c4c6e-5272-4900-b87d-81fd30e3e802" width="550"> |
  | Horse        | <img src="https://github.com/user-attachments/assets/7dd4e0a7-e9e3-4dfa-ae85-dc5dda0a9b7f" width="550"> |
  | Ship         | <img src="https://github.com/user-attachments/assets/9578c78c-64c8-4e84-b4c3-0a5c353affa7" width="550"> |
  | Truck        | <img src="https://github.com/user-attachments/assets/e1544531-ea88-43c8-98de-c87d60ff6028" width="550"> |

</div>


# Performance

For demonstration purposes, the model was overfitted on a small subset of images (5000):

<table align="center">
  <tr>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/07592407-ac9b-43f5-834e-fa4be987a199"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/0d8a08a2-f10c-44e7-b439-02c612e9550a"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/0f121bd8-afef-4a07-9e2c-e927d9c1d011"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/cc165e47-b531-4ba2-9165-aef9cacdf9b0"></td>
  </tr>
  <tr>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/a764ff6f-951c-4397-84a1-a321067f3fce"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/6beb4a7d-7614-4c1c-80ba-ecf717bbd149"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/e4196c7f-791b-4205-8263-21c9bb3f6557"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/f3e95c61-7f3a-48ce-9a44-2ac5aeda2712"></td>
  </tr>
  <tr>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/11cfb218-7fc0-4585-b6d0-f06e61b4301c"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/42a7916b-0f95-43cf-8a40-106392440f64"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/0901d5c3-cb6c-45f5-b4d2-e3b2711a7011"></td>
    <td><img width="235" alt="image" src="https://github.com/user-attachments/assets/0edc0850-da5e-4b43-ab2f-73dd7e3e2b0e"></td>
  </tr>
</table>

<img width="550" alt="389715031-22c4bc8e-ecf5-4ae0-9b51-f2819bf137ac" src="https://github.com/user-attachments/assets/14d8f4c3-f56e-4199-b402-a7c0c0aae130">

As a result, an average classification accuracy of 95% is achieved. 



---

## Challenges and Optimization
**Training Speed:**
Vanilla Python's interpreter is significantly slower compared to libraries like NumPy, which leverage vectorization and GPU acceleration. Below is a comparison of training times for a small set of images using pure Python and NumPy optimizations:

- Vanilla Python: Backpropagation: 27.47s, Forward Pass: 14.00s
  <img width="1200" alt="raw" src="https://github.com/user-attachments/assets/4383ee7d-0449-4e1b-aa77-9c9f8fed9080">

- NumPy Optimization: Backpropagation: 0.12s, Forward Pass: 0.03s
  <img width="1200" alt="numpy" src="https://github.com/user-attachments/assets/370cd6ff-db76-44e1-85d2-64d6d75d1cc0">


  
**Performance Boost:**

- **~230x** faster in backpropagation.
- **~460x** faster in forward propagation.

While it doesn't match TensorFlow's speed, the implementation unlocks creative ways to experiment with custom image dimensions and different architectures.

---
## Activation Functions
- ReLU
- Leaky ReLU
- Sigmoid
- Softmax
## Loss Functions
- Sparse Categorical Cross-Entropy
- Categorical Cross-Entropy
- Mean Squared Error
- Binary Categorical Entropy
## Optimizers
- Stochastic Gradient Descent (SGD)
--- 

# File Structure

```
CNNFromScratch/
  ├── AveragePooling2D.py      # Implements Average Pooling layer
  ├── CIFAR10.py               # CIFAR-10 dataset loader and preprocessor
  ├── CIFAR10_display_utils.py # Visualization utilities for CIFAR-10
  ├── Conv2D.py                # Implements Convolutional 2D layer
  ├── Dense.py                 # Implements Dense (Fully Connected) layer
  ├── Flatten.py               # Implements Flatten layer
  ├── Layer.py                 # Base class for all layers
  ├── Sequential.py            # Implements the Sequential model
  ├── datasets/                # Directory containing datasets
  │   └── CIFAR10/             # CIFAR-10 dataset files
  ├── demo.py                  # Demo script comparing functionality with TensorFlow
  ├── main.py                  # Main script for loading models and making predictions
  └── trained_model.pkl        # Pickled trained model
```
---

# Core Libraries
This project deliberately avoids high-level libraries like TensorFlow, PyTorch, and NumPy. However, some native Python libraries are crucial:

- **random**: For weight initialization and dataset shuffling.
- **math**: For mathematical functions (e.g., exp, log).
- **sys** & **os**: For file and dataset management.
- **pickle**: For saving and loading models.
- **time**: For measuring training durations.
  

