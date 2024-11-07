import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import random
from PIL import Image
from modelTraining import MonetNonMonetClassifier  # Import the classifier from modelTraining.py
import config

class CustomImageDataset(Dataset):
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

# Define the transformations for evaluation (without augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Load the dataset
monet_dataset = CustomImageDataset(image_dir=config.MONET_TEST_DIR, label=1, transform=transform)
non_monet_dataset = CustomImageDataset(image_dir=config.NON_MONET_TEST_DIR, label=0, transform=transform)

# Randomly select 5 images from each category
selected_monet_indices = random.sample(range(len(monet_dataset)), 5)
selected_non_monet_indices = random.sample(range(len(non_monet_dataset)), 5)

# Create subsets for the selected images
monet_subset = Subset(monet_dataset, selected_monet_indices)
non_monet_subset = Subset(non_monet_dataset, selected_non_monet_indices)

# Create DataLoaders for the subsets
monet_loader = DataLoader(monet_subset, batch_size=1, shuffle=False)
non_monet_loader = DataLoader(non_monet_subset, batch_size=1, shuffle=False)

def evaluate_images(model, data_loader, label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    fig, axes = plt.subplots(1, 5, figsize=(20, 8))  # Adjusted figsize for better fit
    axes = axes.flatten()

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            prob_monet = probabilities[0][0]
            prob_non_monet = 1 - prob_monet

            # Determine the predicted label
            predicted_label = 1 if prob_monet > 0.5 else 0

            # Display the image with the probability score
            ax = axes[i]
            img = images[0].permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1] for display
            ax.imshow(img)
            title_text = f'{label}\nNM: {prob_non_monet:.2f}, M: {prob_monet:.2f}'
            ax.set_title(title_text, fontsize=30, fontweight='bold') 
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize the model
    model = MonetNonMonetClassifier()
    model.load_model(config.MODEL_PATH)

    print("Evaluating Monet images from the original evaluation dataset")
    evaluate_images(model.model, monet_loader, "Monet Paintings")

    print("Evaluating Non-Monet images from the original evaluation dataset")
    evaluate_images(model.model, non_monet_loader, "AI Paintings")