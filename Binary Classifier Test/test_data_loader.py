import matplotlib.pyplot as plt # type: ignore
import torchvision # type: ignore
from data_preparation import load_data

def show_batch(data_loader):
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images)

    # Display the images
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.title('Batch of Images')
    plt.show()

if __name__ == "__main__":
    # Load data
    train_loader, _ = load_data(batch_size=32)
    
    # Show a batch from the training data loader
    show_batch(train_loader)
