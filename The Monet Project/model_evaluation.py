import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
from model import create_model
import config

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Load the dataset
dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=transform)

# Separate Monet and Non-Monet images
monet_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
non_monet_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]

# Randomly select 20 images from each category
selected_monet_indices = random.sample(monet_indices, 20)
selected_non_monet_indices = random.sample(non_monet_indices, 20)

# Create subsets for the selected images
monet_subset = Subset(dataset, selected_monet_indices)
non_monet_subset = Subset(dataset, selected_non_monet_indices)

# Create DataLoaders for the subsets
monet_loader = DataLoader(monet_subset, batch_size=1, shuffle=False)
non_monet_loader = DataLoader(non_monet_subset, batch_size=1, shuffle=False)

def evaluate_images(model, data_loader, label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_monet = probabilities[0][1].item()
            prob_non_monet = probabilities[0][0].item()

            # Display the image with the probability score
            ax = axes[i]
            ax.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            ax.set_title(f'{label} Non-Monet: {prob_monet:.2f}, Monet: {prob_non_monet:.2f}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model = create_model()
    model.load_state_dict(torch.load(config.MODEL_PATH))

    print("Evaluating Non-Monet images...")
    evaluate_images(model, monet_loader, ":")  # Correct loader for Non-Monet

    print("Evaluating Monet images...")
    evaluate_images(model, non_monet_loader, ":")  # Correct loader for Monet