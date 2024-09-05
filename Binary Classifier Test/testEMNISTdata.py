import torch # type: ignore
from torch.utils.data import DataLoader, Subset # type: ignore
from torchvision import datasets, transforms # type: ignore
from tqdm import tqdm  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import torchvision # type: ignore
from matplotlib.widgets import Button # type: ignore
from utils import load_model
from model import create_model
import config
import csv
import os

# Enable interactive mode
plt.ion()

# Global variable to control the stopping of the testing process
stop_testing = False

def load_emnist_data(batch_size=32, num_samples=200):
    # Define the transformation to convert grayscale images to RGB and then resize and convert to tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Load the EMNIST dataset with the defined transformations
    emnist_dataset = datasets.EMNIST(root='./data', split='digits', train=False, download=True, transform=transform)

    # Filter the dataset to only include images with labels '1' and '0'
    indices = [i for i, (_, label) in enumerate(emnist_dataset) if label in [0, 1]]
    subset_emnist_dataset = Subset(emnist_dataset, indices[:num_samples])

    # Create data loader for the test set
    test_loader = DataLoader(subset_emnist_dataset, batch_size=batch_size, shuffle=False)

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

def test_model(model, test_loader, csv_filename=config.TESTING_PATH):
    global stop_testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    incorrect = 0
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    ax_image = fig.add_subplot(gs[:, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_button = fig.add_subplot(gs[1, 1])
    ax_incorrect = fig.add_subplot(gs[:, 2])
    
    btn_stop = Button(ax_button, 'Stop')
    btn_stop.on_clicked(stop)
    
    incorrect_count = 0
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Image Index', 'Prediction', 'Actual', 'Correct'])
        
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
                    
                    # Save and display incorrect images immediately
                    if prediction != label:
                        ax_incorrect.clear()
                        show_image(image, title, ax_incorrect)
                        incorrect_count += 1
                    
                    # Write the result to the CSV file
                    writer.writerow([total, prediction, label, prediction == label])
                    
                    # Update the accuracy and counts
                    ax_stats.clear()
                    ax_stats.text(0.5, 0.5, f'Accuracy: {correct / total:.4f}\nCorrect: {correct}\nIncorrect: {incorrect}', 
                                  horizontalalignment='center', verticalalignment='center', transform=ax_stats.transAxes, fontsize=12)
                    ax_stats.axis('off')
                    plt.draw()
        
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.4f}')
        # Write the final accuracy to the CSV file
        writer.writerow(['Final Accuracy', accuracy])
        writer.writerow(['Total Correct', correct])
        writer.writerow(['Total Incorrect', incorrect])

if __name__ == "__main__":
    # Load the EMNIST test data
    test_loader = load_emnist_data(batch_size=config.BATCH_SIZE, num_samples=200)
    
    # Create the model and load the trained weights
    model = create_model()
    model = load_model(model, config.MODEL_PATH)
    
    # Test the model and save results to CSV
    test_model(model, test_loader, config.TESTING_PATH)