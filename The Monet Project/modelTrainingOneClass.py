import torch  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn as nn  # type: ignore
from tqdm import tqdm  # type: ignore
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from model import create_model
from utils import ensure_dir_exists, save_to_csv, save_model
import config
import os
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import csv
from PIL import Image
from data_preparation import load_data

# Define the paths
monet_path = 'C:/Users/getin/Documents/networking/ITSC-3990-AI-Art-Authentication/The Monet Project/data/large_monet_dataset'
non_monet_path = 'C:/Users/getin/Documents/networking/ITSC-3990-AI-Art-Authentication/The Monet Project/data/non-monet_test'

# Load data
train_loader, val_loader, monet_test_loader, non_monet_test_loader = load_data(monet_path, non_monet_path, batch_size=config.BATCH_SIZE)

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

def test_model(model, monet_test_loader, non_monet_test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []

    # Create a figure with subplots for image display and stats
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    with torch.no_grad():
        for images, labels in tqdm(monet_test_loader, desc="Testing Monet Paintings", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Display each image with the prediction and whether it is correct
            for i in range(images.size(0)):
                image = images[i]
                label = labels[i].item()
                prediction = predicted[i].item()
                title = f'Prediction: {"Monet" if prediction == 1 else "Non-Monet"}, Actual: {"Monet" if label == 1 else "Non-Monet"}, {"Correct" if prediction == label else "Incorrect"}'
                show_image(image, title, ax)
        
        for images, labels in tqdm(non_monet_test_loader, desc="Testing Non-Monet Paintings", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Display each image with the prediction and whether it is correct
            for i in range(images.size(0)):
                image = images[i]
                label = labels[i].item()
                prediction = predicted[i].item()
                title = f'Prediction: {"Monet" if prediction == 1 else "Non-Monet"}, Actual: {"Monet" if label == 1 else "Non-Monet"}, {"Correct" if prediction == label else "Incorrect"}'
                show_image(image, title, ax)
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Ensure the confusion matrix has the correct shape
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions, labels=[0, 1]).ravel()
    
    # Debug prints for confusion matrix
    print(f"Confusion Matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    # Prepare results
    results = ['Overall', precision, recall, f1, accuracy, tn, fp, fn, tp]

    # Write results to CSV
    write_results_to_csv(results)

    # Print results to console
    print("Overall Results:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    print("Results written to results.csv")

def write_results_to_csv(results, filename='results.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'TN', 'FP', 'FN', 'TP'])
        writer.writerow(results)

if __name__ == "__main__":
    model = create_model()
    train_model(model, train_loader, val_loader)
    test_model(model, monet_test_loader, non_monet_test_loader)