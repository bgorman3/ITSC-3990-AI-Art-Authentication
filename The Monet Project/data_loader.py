import matplotlib.pyplot as plt
import torchvision
from data_preparation import load_data

# Define the normalization values used in the data preparation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def show_batch(data_loader):
    """
    Display a batch of images from the data loader.

    Parameters:
    - data_loader (DataLoader): The data loader to get the batch from.
    """
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images)

    # Display the images
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.title('Batch of Images')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size=32)
    
    # Show a batch from the training data loader
    show_batch(train_loader)