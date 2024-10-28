import torch  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn as nn  # type: ignore
from tqdm import tqdm  # type: ignore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import create_model
from utils import ensure_dir_exists, save_to_csv, save_model
import config
import os
import matplotlib.pyplot as plt  # type: ignore

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

# Load the dataset
dataset = datasets.ImageFolder(root=config.DATA_DIR, transform=transform)

# Calculate the split sizes
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS, csv_filepath=config.RESULTS_PATH):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    results = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Wrap the train_loader with tqdm for a progress bar
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        
        # Wrap the val_loader with tqdm for a progress bar
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                print(f"Batch Labels: {labels.cpu().numpy()}")
                print(f"Batch Predictions: {predicted.cpu().numpy()}")
                print(f"Batch Loss: {loss.item()}")

        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Collect results for CSV
        results.append([epoch+1, epoch_loss, train_accuracy, val_loss, val_accuracy])

    # Save results to CSV
    save_to_csv(results, csv_filepath, headers=['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])

    # Ensure the directory exists and save the model
    ensure_dir_exists(os.path.dirname(config.MODEL_PATH))
    save_model(model, config.MODEL_PATH)

def show_image(image, title, ax):
    """Helper function to display an image with a title."""
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    ax.set_title(title)
    ax.axis('off')
    plt.draw()
    plt.pause(0.001)  # Pause to allow the plot to update

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    # Create a figure with subplots for image display and stats
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Display each image with the prediction and whether it is correct
            for i in range(images.size(0)):
                image = images[i]
                label = labels[i].item()
                prediction = predicted[i].item()
                title = f'Prediction: {"Monet" if prediction == 1 else "Non-Monet"}, Actual: {"Monet" if label == 1 else "Non-Monet"}, {"Correct" if prediction == label else "Incorrect"}'
                show_image(image, title, ax)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    model = create_model()
    train_model(model, train_loader, val_loader)
    test_model(model, test_loader)