import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import json
import random
import config

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
        return image, img_path

def load_test_data(monet_dir, non_monet_dir, batch_size, test_indices_file='test_indices.json'):
    """
    Load the Monet and non-Monet test datasets using the saved test indices.

    Parameters:
    - monet_dir (str): Path to the Monet dataset directory.
    - non_monet_dir (str): Path to the non-Monet dataset directory.
    - batch_size (int): Number of samples per batch.
    - test_indices_file (str): Name of the JSON file containing the test indices.

    Returns:
    - monet_test_loader (DataLoader): DataLoader for the Monet test set.
    - non_monet_test_loader (DataLoader): DataLoader for the non-Monet test set.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the Monet dataset using the custom dataset class
    monet_dataset = CustomImageDataset(image_dir=monet_dir, transform=transform)

    # Load the test indices from the JSON file in the specified directory
    test_indices_path = os.path.join(config.TEST_DATA_DIR, test_indices_file)
    with open(test_indices_path, 'r') as f:
        test_indices = json.load(f)

    # Create a subset for the Monet test dataset using the loaded indices
    monet_test_subset = Subset(monet_dataset, test_indices)

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, transform=transform)

    # Randomly select the same number of non-Monet images as the Monet test set
    selected_non_monet_indices = random.sample(range(len(non_monet_dataset)), len(test_indices))
    non_monet_test_subset = Subset(non_monet_dataset, selected_non_monet_indices)

    # Create DataLoaders
    monet_test_loader = DataLoader(monet_test_subset, batch_size=batch_size, shuffle=False)
    non_monet_test_loader = DataLoader(non_monet_test_subset, batch_size=batch_size, shuffle=False)

    return monet_test_loader, non_monet_test_loader

if __name__ == "__main__":
    monet_dir = config.MONET_DATA_DIR
    non_monet_dir = config.NON_MONET_DATA_DIR
    batch_size = config.BATCH_SIZE

    monet_test_loader, non_monet_test_loader = load_test_data(monet_dir, non_monet_dir, batch_size)

    print("Monet test set size:", len(monet_test_loader.dataset))
    print("Non-Monet test set size:", len(non_monet_test_loader.dataset))