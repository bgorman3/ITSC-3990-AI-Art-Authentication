import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Subset
import random
from data_preparation import load_data

# Define the normalization values used in the data preparation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def show_batch(data_loader, num_images, title):
    """
    Display a specified number of images from the data loader.

    Parameters:
    - data_loader (DataLoader): The data loader to get the batch from.
    - num_images (int): The number of images to display.
    - title (str): The title for the plot.
    """
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Select the specified number of images
    images = images[:num_images]
    labels = labels[:num_images]

    # Create a grid of images with white padding
    grid_img = torchvision.utils.make_grid(images, nrow=num_images, padding=50, pad_value=1)  # Set pad_value to 1 for white space

    # Display the images
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.title(title, fontdict={'fontsize': 36, 'fontweight': 'bold'})  # Set title font size and weight
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, val_loader, monet_test_loader, non_monet_test_loader = load_data(batch_size=32)

    # Randomly select 5 images from each category
    monet_indices = random.sample(range(len(monet_test_loader.dataset)), 5)
    non_monet_indices = random.sample(range(len(non_monet_test_loader.dataset)), 5)

    # Create subsets for the selected images
    monet_subset = Subset(monet_test_loader.dataset, monet_indices)
    non_monet_subset = Subset(non_monet_test_loader.dataset, non_monet_indices)

    # Create DataLoaders for the subsets
    monet_loader = DataLoader(monet_subset, batch_size=5, shuffle=False)
    non_monet_loader = DataLoader(non_monet_subset, batch_size=5, shuffle=False)

    # Show a batch from the Monet test data loader
    print("Displaying 5 Monet images:")
    show_batch(monet_loader, num_images=5, title="Monet Images")

    # Show a batch from the Non-Monet test data loader
    print("Displaying 5 Non-Monet images:")
    show_batch(non_monet_loader, num_images=5, title="Non-Monet Images")