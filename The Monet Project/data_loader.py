import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Subset
import random
from data_preparation import load_data
from torchvision import transforms
from PIL import Image

# Define the normalization values used in the data preparation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def show_image(data_loader, transform):
    """
    Display a single image from the data loader and print the transformations applied.

    Parameters:
    - data_loader (DataLoader): The data loader to get the image from.
    - transform (transforms.Compose): The transformations applied to the image.
    """
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Select the first image
    image = images[0]

    # Display the image
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.show()

    # Print the transformations applied
    print("Transformations applied to the image:")
    for t in transform.transforms:
        print(f"- {t}")

if __name__ == "__main__":
    # Define the transformations for evaluation (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load data
    train_loader, val_loader, monet_test_loader, non_monet_test_loader = load_data(batch_size=32)

    # Randomly select 1 image from each category
    monet_indices = random.sample(range(len(monet_test_loader.dataset)), 1)
    non_monet_indices = random.sample(range(len(non_monet_test_loader.dataset)), 1)

    # Create subsets for the selected images
    monet_subset = Subset(monet_test_loader.dataset, monet_indices)
    non_monet_subset = Subset(non_monet_test_loader.dataset, non_monet_indices)

    # Create DataLoaders for the subsets
    monet_loader = DataLoader(monet_subset, batch_size=1, shuffle=False)
    non_monet_loader = DataLoader(non_monet_subset, batch_size=1, shuffle=False)

    # Show a single image from the Monet test data loader
    print("Displaying 1 Monet image:")
    show_image(monet_loader, transform)

    # Show a single image from the Non-Monet test data loader
    print("Displaying 1 Non-Monet image:")
    show_image(non_monet_loader, transform)