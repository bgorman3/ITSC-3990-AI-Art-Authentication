import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Assuming all images are of the same class

def load_data(monet_dir, non_monet_dir, batch_size, val_split=0.1, test_split=0.2):
    """
    Load and split the Monet dataset into training, validation, and testing sets.
    Load the non-Monet dataset for testing.

    Parameters:
    - monet_dir (str): Path to the Monet dataset directory.
    - non_monet_dir (str): Path to the non-Monet dataset directory.
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
    monet_dataset = CustomImageDataset(image_dir=monet_dir, transform=transform)

    # Calculate the split sizes
    total_size = len(monet_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - test_size - val_size

    # Split the Monet dataset
    train_dataset, val_dataset, monet_test_dataset = random_split(monet_dataset, [train_size, val_size, test_size])

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    monet_test_loader = DataLoader(monet_test_dataset, batch_size=batch_size, shuffle=False)
    non_monet_test_loader = DataLoader(non_monet_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, monet_test_loader, non_monet_test_loader