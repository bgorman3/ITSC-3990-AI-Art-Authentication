import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import os
from PIL import Image
from model import create_model
import config
import random

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

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

# New test datasets
monet_test_path = 'data/large_monet_dataset'
non_monet_test_path = 'data/non-monet_test'


# Create datasets for the new test datasets
monet_test_dataset = CustomImageDataset(image_dir=monet_test_path, transform=transform)
non_monet_test_dataset = CustomImageDataset(image_dir=non_monet_test_path, transform=transform)

# Verify the contents of the datasets


# Randomly select 20 images from each test dataset
selected_monet_indices = random.sample(range(len(monet_test_dataset)), 20)
selected_non_monet_indices = random.sample(range(len(non_monet_test_dataset)), 20)

# Create subsets for the selected images
monet_test_subset = Subset(monet_test_dataset, selected_monet_indices)
non_monet_test_subset = Subset(non_monet_test_dataset, selected_non_monet_indices)

# Create DataLoaders for the new test subsets
monet_test_loader = DataLoader(monet_test_subset, batch_size=1, shuffle=False)
non_monet_test_loader = DataLoader(non_monet_test_subset, batch_size=1, shuffle=False)

def evaluate_images(model, data_loader, label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    num_images = len(data_loader)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten()

    with torch.no_grad():
        for i, (images, img_paths) in enumerate(data_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_monet = probabilities[0][0].item()
            prob_non_monet = probabilities[0][1].item()

            # Determine the predicted label
            predicted_label = 1 if prob_non_monet > prob_monet else 0

            # Display the image with the probability score
            ax = axes[i]
            img = images[0].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] for display
            ax.imshow(img)
            ax.set_title(f'{label} Non-Monet: {prob_non_monet:.2f}, Monet: {prob_monet:.2f}\n{img_paths[0]}')
            ax.axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load(config.MODEL_PATH))

    # Print dataset paths for debugging
    print(f"Monet test dataset path: {monet_test_path}")
    print(f"Non-Monet test dataset path: {non_monet_test_path}")

    # Print selected indices for debugging
    print(f"Selected Monet indices: {selected_monet_indices}")
    print(f"Selected Non-Monet indices: {selected_non_monet_indices}")

    # Print image paths for selected indices for debugging
    print("Selected Monet image paths:")
    for idx in selected_monet_indices:
        print(monet_test_dataset.image_paths[idx])

    print("Selected Non-Monet image paths:")
    for idx in selected_non_monet_indices:
        print(non_monet_test_dataset.image_paths[idx])

    # Evaluate Monet test images first
    print("Evaluating Monet test images...")
    evaluate_images(model, monet_test_loader, "Monet Test Paintings")

    # Evaluate Non-Monet test images next
    print("Evaluating Non-Monet test images...")
    evaluate_images(model, non_monet_test_loader, "Non-Monet Test Paintings")