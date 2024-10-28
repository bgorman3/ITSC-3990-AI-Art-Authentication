import torch  # type: ignore
import torch.optim as optim  # type: ignore
import torch.nn as nn  # type: ignore
from tqdm import tqdm  # type: ignore
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from model import create_model
from utils import ensure_dir_exists, save_to_csv, save_model
import config
import os
import matplotlib.pyplot as plt  # type: ignore
import csv
from data_preparation import load_data
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# Load data using the data_preparation code
train_loader, val_loader, monet_test_loader, non_monet_test_loader, test_indices = load_data(
    config.MONET_DATA_DIR, config.NON_MONET_DATA_DIR, batch_size=config.BATCH_SIZE
)

def plot_metrics(metrics, csv_filepath):
    epochs = [x[0] for x in metrics]
    train_losses = [x[1] for x in metrics]
    train_accuracies = [x[2] for x in metrics]
    val_losses = [x[3] for x in metrics]
    val_accuracies = [x[4] for x in metrics]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(csv_filepath), 'training_validation_metrics.png'))
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs=config.NUM_EPOCHS, csv_filepath=config.RESULTS_PATH, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    results = []
    best_val_loss = float('inf')
    patience_counter = 0

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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            save_model(model, config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Collect results for CSV
        results.append([epoch+1, epoch_loss, train_accuracy, val_loss, val_accuracy])

    # Save results to CSV
    save_to_csv(results, csv_filepath, headers=['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy'])

    # Plot metrics
    plot_metrics(results, csv_filepath)

def evaluate_images(model, data_loader, true_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_monet = probabilities[0][1].item()
            prob_non_monet = probabilities[0][0].item()

            # Determine the predicted label
            predicted_label = 1 if prob_monet > prob_non_monet else 0

            # Collect all labels and predictions for metric calculation
            all_labels.append(true_label)
            all_predictions.append(predicted_label)

            # Debug prints
            print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Prob Monet: {prob_monet:.2f}, Prob Non-Monet: {prob_non_monet:.2f}")

    return all_labels, all_predictions

def calculate_metrics(true_labels, predictions):
    precision = precision_score(true_labels, predictions, pos_label=1, zero_division=1)
    recall = recall_score(true_labels, predictions, pos_label=1, zero_division=1)
    f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=1)
    accuracy = accuracy_score(true_labels, predictions)
    
    # Ensure the confusion matrix has the correct shape
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[0, 1]).ravel()
    
    # Debug prints for confusion matrix
    print(f"Confusion Matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    return precision, recall, f1, accuracy, tn, fp, fn, tp

def write_results_to_csv(results, filename='results.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Category', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'TN', 'FP', 'FN', 'TP'])
        writer.writerow(results)

if __name__ == "__main__":
    # Load the pre-trained ResNet-101 model
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    # Modify the last fully connected layer to match the number of output classes (1 class: "Monet")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (Monet vs. Non-Monet)

    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Train the model on Monet images
    train_model(model, train_loader, val_loader)

    # Evaluate Monet images
    print("Evaluating Monet test images...")
    monet_labels, monet_predictions = evaluate_images(model, monet_test_loader, 1)

    # Evaluate Non-Monet images
    print("Evaluating Non-Monet test images...")
    non_monet_labels, non_monet_predictions = evaluate_images(model, non_monet_test_loader, 0)

    # Combine labels and predictions
    all_labels = monet_labels + non_monet_labels
    all_predictions = monet_predictions + non_monet_predictions

    # Calculate metrics
    precision, recall, f1, accuracy, tn, fp, fn, tp = calculate_metrics(all_labels, all_predictions)

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