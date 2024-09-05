import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import datasets, transforms # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torchvision # type: ignore
from matplotlib.widgets import Button # type: ignore
from utils import load_model
from model import create_model
import config

# Enable interactive mode
plt.ion()

# Global variable to control the stopping of the testing process
stop_testing = False

def load_test_data(batch_size=32):
    # Define the transformation to convert grayscale images to RGB and then resize and convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Load the MNIST dataset with the defined transformations
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter for binary classification (0 and 1)
    indices = [i for i, target in enumerate(test_dataset.targets) if target in [0, 1]]
    binary_test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # Create data loader for the test set
    test_loader = DataLoader(binary_test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def show_image(image, title, ax):
    """Helper function to display an image with a title."""
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.set_title(title)
    ax.axis('off')
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

def stop(event):
    global stop_testing
    stop_testing = True

def test_model(model, test_loader):
    global stop_testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    incorrect = 0
    
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2)
    ax_image = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_button = fig.add_subplot(gs[1, 1])
    
    btn_stop = Button(ax_button, 'Stop')
    btn_stop.on_clicked(stop)
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            if stop_testing:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            incorrect = total - correct
            
            # Display each image with the prediction and whether it is correct
            for i in range(images.size(0)):
                if stop_testing:
                    break
                image = images[i]
                label = labels[i].item()
                prediction = predicted[i].item()
                title = f'Prediction: {prediction}, Actual: {label}, {"Correct" if prediction == label else "Incorrect"}'
                show_image(image, title, ax_image)
                
                # Update the accuracy and counts
                ax_stats.clear()
                ax_stats.text(0.5, 0.5, f'Accuracy: {correct / total:.4f}\nCorrect: {correct}\nIncorrect: {incorrect}', 
                              horizontalalignment='center', verticalalignment='center', transform=ax_stats.transAxes, fontsize=12)
                ax_stats.axis('off')
                plt.draw()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    # Load the test data
    test_loader = load_test_data(batch_size=config.BATCH_SIZE)
    
    # Create the model and load the trained weights
    model = create_model()
    model = load_model(model, config.MODEL_PATH)
    
    # Test the model
    test_model(model, test_loader)