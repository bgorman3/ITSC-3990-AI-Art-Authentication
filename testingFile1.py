import torch
import torchvision
import numpy
from PIL import Image
import matplotlib
import sklearn

print("Dependencies are installed and working correctly!")

# Optional: Test each library with a simple function call

# PyTorch test
print(f"PyTorch version: {torch.__version__}")

# Torchvision test
print(f"Torchvision version: {torchvision.__version__}")

# Numpy test
print(f"NumPy version: {numpy.__version__}")

# Pillow test
img = Image.new('RGB', (60, 30), color = (73, 109, 137))
print("Pillow is working.")

# Matplotlib test
print(f"Matplotlib version: {matplotlib.__version__}")

# Scikit-learn test
print(f"Scikit-learn version: {sklearn.__version__}")


x = torch.tensor([1.0, 2.0, 3.0])
print(x + 1)


from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
print(f"Number of images in dataset: {len(dataset)}")
