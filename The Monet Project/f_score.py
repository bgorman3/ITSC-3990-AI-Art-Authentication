import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import random
from model import create_model
import config
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import csv
import os
from PIL import Image
import json

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
])

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def load_test_data(monet_dir, non_monet_dir, batch_size, test_indices_file='test_indices.json'):
    """
    Load the Monet and non-Monet test datasets using the saved test indices.

    Parameters:
    - monet_dir (str): Path to the Monet dataset directory.
    - non_monet_dir (str): Path to the non-Monet dataset directory.
    - batch_size (int): Number of samples per batch.
    - test_indices_file (str): Name of the JSON file containing the test indices.

    Returns:
    - monet_test_loader (DataLoader): DataLoader for the Monet test set.
    - non_monet_test_loader (DataLoader): DataLoader for the non-Monet test set.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the Monet dataset using the custom dataset class
    monet_dataset = CustomImageDataset(image_dir=monet_dir, transform=transform)

    # Load the test indices from the JSON file in the specified directory
    test_indices_path = os.path.join(config.TEST_DATA_DIR, test_indices_file)
    with open(test_indices_path, 'r') as f:
        test_indices = json.load(f)

    # Create a subset for the Monet test dataset using the loaded indices
    monet_test_subset = Subset(monet_dataset, test_indices)

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, transform=transform)

    # Create DataLoaders
    monet_test_loader = DataLoader(monet_test_subset, batch_size=batch_size, shuffle=False)
    non_monet_test_loader = DataLoader(non_monet_dataset, batch_size=batch_size, shuffle=False)

    return monet_test_loader, non_monet_test_loader

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
            predicted_label = 1 if prob_non_monet > prob_monet else 0

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
    model = create_model()
    model.load_state_dict(torch.load(config.MODEL_PATH))

    # Load test data
    monet_test_loader, non_monet_test_loader = load_test_data(config.MONET_DATA_DIR, config.NON_MONET_DATA_DIR, config.BATCH_SIZE)

    # Evaluate Monet images
    monet_labels, monet_predictions = evaluate_images(model, monet_test_loader, 1)

    # Evaluate Non-Monet images
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