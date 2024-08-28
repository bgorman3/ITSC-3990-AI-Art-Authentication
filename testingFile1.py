import torch
import torchvision.models as models

# Load the pre-trained ResNet-101 model
resnet101 = models.resnet101(pretrained=True)
