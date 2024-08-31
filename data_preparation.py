import torch # type: ignore
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader, Subset # type: ignore

def load_data(batch_size=32):
    # Define the transformation to convert grayscale images to RGB and then resize and convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Load the MNIST dataset with the defined transformations
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    def filter_data(dataset, target_classes):
        indices = [i for i, target in enumerate(dataset.targets) if target in target_classes]
        return Subset(dataset, indices)

    # Filter for binary classification
    binary_dataset = filter_data(full_dataset, [0, 1])

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(binary_dataset))
    val_size = len(binary_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(binary_dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
