



-----
# Introduction to PyTorch

PyTorch is an open-source machine learning library that emphasizes flexibility and ease-of-use. It provides a strong GPU acceleration backend and supports dynamic computation graphs. We'll explore the basics of PyTorch, starting with tensors, performing simple operations, selecting a device, and finally an introduction to building neural networks with both fully connected (linear) and convolutional (CNN) layers.

----------

# 1. Tensors in PyTorch

Tensors are the fundamental data structures in PyTorch—multidimensional arrays similar to NumPy arrays but with additional capabilities such as GPU support.

### Creating Tensors

You can create tensors in several ways:

```python
import torch

# Create a tensor from a Python list
tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("Tensor from list:\n", tensor_from_list)

# Create a tensor with random values (3x3)
random_tensor = torch.rand(3, 3)
print("Random tensor:\n", random_tensor)

```

### Tensor Attributes

Each tensor has attributes like its shape, data type, and device:

```python
print("Shape:", tensor_from_list.shape)
print("Data type:", tensor_from_list.dtype)
print("Device:", tensor_from_list.device)

```

### Basic Tensor Operations

Perform elementwise operations, reshaping, and more:

```python
# Elementwise arithmetic
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Addition:", a + b)
print("Multiplication:", a * b)

# Reshape tensor (flatten)
x = torch.arange(9)
x_reshaped = x.view(3, 3)
print("Reshaped tensor (3x3):\n", x_reshaped)

```

For more operations (indexing, concatenation, etc.), PyTorch offers methods like `torch.cat()`, `unsqueeze()`, and `squeeze()`.

----------

# 2. Selecting a Device: CPU vs GPU

PyTorch allows you to run your computations on a CPU or GPU. You can select a device based on CUDA availability:

```python
# Select the device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create a tensor and move it to the selected device
tensor_on_device = torch.tensor([1, 2, 3], dtype=torch.float32).to(device)
print("Tensor on device:", tensor_on_device)

```

By moving both your model and data to the same device, you ensure that all operations are performed in one place.

----------

# 3. Neural Networks in PyTorch

PyTorch's `torch.nn` module provides tools to build and train neural networks. There are two common types of layers we'll touch on here:

-   **Linear (Fully Connected) Layers**: These perform a matrix multiplication between input features and weights.
    
-   **Convolutional (CNN) Layers**: Ideal for processing images, these layers learn spatial hierarchies of features using filters.
    

### Building a Simple Neural Network

Below is an example of a fully connected network (using `nn.Module`):

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # For MNIST images (28x28)
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Instantiate the model
model = SimpleNN().to(device)
print(model)

```

### Building a Simple Convolutional Neural Network (CNN)

Here’s a basic CNN example for image classification:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # A convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # A pooling layer: halves the spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 32 in, 64 out channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer: input features calculated from flattened feature map
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # assuming input images of size 28x28
        self.fc2 = nn.Linear(128, 10)          # 10 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> ReLU -> pool
        x = x.view(x.size(0), -1)             # flatten the tensor
        x = F.relu(self.fc1(x))               # first fully connected layer
        x = self.fc2(x)                       # output layer
        return x

# Instantiate the CNN and move it to the device
cnn_model = SimpleCNN().to(device)
print(cnn_model)

```

In this CNN:

-   The first convolutional layer processes the input image (e.g., a grayscale MNIST image with 1 channel).
    
-   The max pooling layers reduce the spatial dimensions.
    
-   The fully connected layers convert the learned features into class scores.
    

----------


