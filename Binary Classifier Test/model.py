import torchvision.models as models # type: ignore
import torch.nn as nn # type: ignore

def create_model():
    # Load the pre-trained ResNet-101 model
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # Modify the first convolutional layer to accept 1-channel input
    model.conv1 = nn.Conv2d(
        in_channels=3,  # Change from 3 to 1
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=model.conv1.bias
    )

    # Update the number of output features to match the binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

    return model