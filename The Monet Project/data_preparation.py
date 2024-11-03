import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from PIL import Image
import json
import torch
import config

class LabeledImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label = label  # 1 for Monet, 0 for non-Monet
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                          if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

def load_data(batch_size, val_split=0.2, test_split=0.1):
    """
    Load and split the Monet and non-Monet datasets into training, validation, and testing sets.

    Parameters:
    - batch_size (int): Number of samples per batch.
    - val_split (float): Proportion of the dataset to include in the validation split.
    - test_split (float): Proportion of the dataset to include in the test split.

    Returns:
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - monet_test_loader (DataLoader): DataLoader for the Monet test set.
    - non_monet_test_loader (DataLoader): DataLoader for the non-Monet test set.
    """
    # Define the transformations with data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.RandomRotation(30),  # Randomly rotate the image
        transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the Monet dataset using the custom dataset class
    monet_dataset = LabeledImageDataset(image_dir=config.MONET_DATA_DIR, label=1, transform=transform)
    non_monet_dataset = LabeledImageDataset(image_dir=config.NON_MONET_DATA_DIR, label=0, transform=transform)

    # Calculate the split sizes for Monet dataset
    total_size_monet = len(monet_dataset)
    test_size_monet = int(test_split * total_size_monet)
    val_size_monet = int(val_split * total_size_monet)
    train_size_monet = total_size_monet - test_size_monet - val_size_monet

    # Split the Monet dataset
    monet_train_val_dataset, monet_test_dataset = random_split(
        monet_dataset, 
        [train_size_monet + val_size_monet, test_size_monet],
        generator=torch.Generator().manual_seed(42)
    )

    monet_train_dataset, monet_val_dataset = random_split(
        monet_train_val_dataset,
        [train_size_monet, val_size_monet],
        generator=torch.Generator().manual_seed(42)
    )

    # Calculate the split sizes for non-Monet dataset
    total_size_non_monet = len(non_monet_dataset)
    test_size_non_monet = int(test_split * total_size_non_monet)
    val_size_non_monet = int(val_split * total_size_non_monet)
    train_size_non_monet = total_size_non_monet - test_size_non_monet - val_size_non_monet

    # Split the non-Monet dataset
    non_monet_train_val_dataset, non_monet_test_dataset = random_split(
        non_monet_dataset, 
        [train_size_non_monet + val_size_non_monet, test_size_non_monet],
        generator=torch.Generator().manual_seed(42)
    )

    non_monet_train_dataset, non_monet_val_dataset = random_split(
        non_monet_train_val_dataset,
        [train_size_non_monet, val_size_non_monet],
        generator=torch.Generator().manual_seed(42)
    )

    # Combine the training datasets
    train_dataset = monet_train_dataset + non_monet_train_dataset
    val_dataset = monet_val_dataset + non_monet_val_dataset

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    monet_test_loader = DataLoader(monet_test_dataset, batch_size=batch_size, shuffle=False)
    non_monet_test_loader = DataLoader(non_monet_test_dataset, batch_size=batch_size, shuffle=False)

    # Save indices for Monet and non-Monet test datasets
    monet_test_indices = monet_test_dataset.indices
    non_monet_test_indices = non_monet_test_dataset.indices

    test_indices_path = os.path.join(config.TEST_DATA_DIR, 'test_indices.json')
    os.makedirs(config.TEST_DATA_DIR, exist_ok=True)
    with open(test_indices_path, 'w') as f:
        json.dump({
            'monet_test_indices': monet_test_indices,
            'non_monet_test_indices': non_monet_test_indices
        }, f)

    # Print dataset sizes and sample labels for verification
    print(f"Monet dataset size: {total_size_monet}")
    print(f"Train size (Monet): {train_size_monet}")
    print(f"Validation size (Monet): {val_size_monet}")
    print(f"Test size (Monet): {test_size_monet}")

    print(f"Non-Monet dataset size: {total_size_non_monet}")
    print(f"Train size (Non-Monet): {train_size_non_monet}")
    print(f"Validation size (Non-Monet): {val_size_non_monet}")
    print(f"Test size (Non-Monet): {test_size_non_monet}")

    # Verify labels in the training dataset
    print("Sample labels from the training dataset:")
    for i in range(10):
        _, label = train_dataset[i]
        print(f"Sample {i}: Label {label}")

    # Verify labels in the validation dataset
    print("Sample labels from the validation dataset:")
    for i in range(10):
        _, label = val_dataset[i]
        print(f"Sample {i}: Label {label}")

    # Verify labels in the Monet test dataset
    print("Sample labels from the Monet test dataset:")
    for i in range(10):
        _, label = monet_test_dataset[i]
        print(f"Sample {i}: Label {label}")

    # Verify labels in the Non-Monet test dataset
    print("Sample labels from the Non-Monet test dataset:")
    for i in range(10):
        _, label = non_monet_test_dataset[i]
        print(f"Sample {i}: Label {label}")

    return train_loader, val_loader, monet_test_loader, non_monet_test_loader

if __name__ == "__main__":
    train_loader, val_loader, monet_test_loader, non_monet_test_loader = load_data(
        batch_size=config.BATCH_SIZE
    )
    print("Data loaders created successfully.")