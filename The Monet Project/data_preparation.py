# data_preparation.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config

def load_data(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, val_split=0.2, test_split=0.2):
    """
    Load and split the dataset into training, validation, and testing sets.

    Parameters:
    - data_dir (str): Path to the dataset directory.
    - batch_size (int): Number of samples per batch.
    - val_split (float): Proportion of the dataset to include in the validation split.
    - test_split (float): Proportion of the dataset to include in the test split.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - test_loader (DataLoader): DataLoader for the test set.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size - test_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader