import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from PIL import Image
import json
import random
import numpy as np
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

class ShuffledConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)
        
        # Create a shuffled index mapping
        all_indices = []
        for dataset_idx, length in enumerate(self.lengths):
            all_indices.extend([(dataset_idx, i) for i in range(length)])
        random.shuffle(all_indices)
        self.shuffled_indices = all_indices

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx, item_idx = self.shuffled_indices[idx]
        return self.datasets[dataset_idx][item_idx]

def load_balanced_data(monet_dir, non_monet_dir, batch_size, val_split=0.2, test_split=0.1, outlier_proportion=0.3):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    monet_dataset = LabeledImageDataset(monet_dir, label=1, transform=transform)
    non_monet_dataset = LabeledImageDataset(non_monet_dir, label=0, transform=transform)

    monet_size = len(monet_dataset)
    non_monet_size = len(non_monet_dataset)

    print(f"Monet dataset size: {monet_size}")
    print(f"Non-Monet dataset size: {non_monet_size}")

    balanced_size = min(int(outlier_proportion * monet_size), non_monet_size)
    non_monet_indices = random.sample(range(non_monet_size), balanced_size)
    non_monet_train_indices = non_monet_indices[:int(0.8 * balanced_size)]
    non_monet_val_indices = non_monet_indices[int(0.8 * balanced_size):]

    non_monet_train_dataset = Subset(non_monet_dataset, non_monet_train_indices)
    non_monet_val_dataset = Subset(non_monet_dataset, non_monet_val_indices)

    test_size = int(test_split * monet_size)
    remaining_size = monet_size - test_size
    val_size = int(val_split * remaining_size)
    train_size = remaining_size - val_size

    print(f"Total Monet size: {monet_size}")
    print(f"Train size: {train_size}")
    print(f"Validation size: {val_size}")
    print(f"Test size: {test_size}")

    monet_train_val_dataset, monet_test_dataset = random_split(
        monet_dataset, 
        [train_size + val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    monet_train_dataset, monet_val_dataset = random_split(
        monet_train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    combined_train_dataset = ShuffledConcatDataset(monet_train_dataset, non_monet_train_dataset)
    combined_val_dataset = ShuffledConcatDataset(monet_val_dataset, non_monet_val_dataset)
    
    print(f"Training dataset size (Monet + Non-Monet): {len(combined_train_dataset)}")
    print(f"Validation dataset size (Monet + Non-Monet): {len(combined_val_dataset)}")
    print(f"Test dataset size (Monet): {len(monet_test_dataset)}")

    # Verify labels in the combined training dataset
    print("Sample labels from the combined training dataset:")
    for i in range(10):
        _, label = combined_train_dataset[i]
        print(f"Sample {i}: Label {label}")

    # Verify labels in the combined validation dataset
    print("Sample labels from the combined validation dataset:")
    for i in range(10):
        _, label = combined_val_dataset[i]
        print(f"Sample {i}: Label {label}")

    # Verify labels in the test dataset
    print("Sample labels from the test dataset:")
    for i in range(10):
        _, label = monet_test_dataset[i]
        print(f"Sample {i}: Label {label}")

    test_indices = monet_test_dataset.indices
    test_indices_path = os.path.join(config.TEST_DATA_DIR, 'test_indices.json')
    os.makedirs(config.TEST_DATA_DIR, exist_ok=True)
    with open(test_indices_path, 'w') as f:
        json.dump(test_indices, f)

    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(monet_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_balanced_data(
        monet_dir=config.MONET_DATA_DIR,
        non_monet_dir=config.NON_MONET_DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    print("Data loaders created successfully.")