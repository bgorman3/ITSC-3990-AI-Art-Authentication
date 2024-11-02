import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from PIL import Image
import json
import torch
import config
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import csv

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label = label  # 1 for Monet, 0 for non-Monet
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) 
                          if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

def load_combined_test_data(monet_dir, non_monet_dir, batch_size, test_indices_file='test_indices.json'):
    """
    Load the combined Monet and non-Monet test datasets using the saved test indices.

    Parameters:
    - monet_dir (str): Path to the Monet dataset directory.
    - non_monet_dir (str): Path to the non-Monet dataset directory.
    - batch_size (int): Number of samples per batch.
    - test_indices_file (str): Name of the JSON file containing the test indices.

    Returns:
    - combined_test_loader (DataLoader): DataLoader for the combined test set.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the Monet dataset using the custom dataset class
    monet_dataset = CustomImageDataset(image_dir=monet_dir, label=1, transform=transform)

    # Load the test indices from the JSON file in the specified directory
    test_indices_path = os.path.join(config.TEST_DATA_DIR, test_indices_file)
    with open(test_indices_path, 'r') as f:
        test_indices = json.load(f)

    # Create a subset for the Monet test dataset using the loaded indices
    monet_test_subset = Subset(monet_dataset, test_indices)

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, label=0, transform=transform)

    # Combine the Monet and non-Monet test datasets
    combined_dataset = ConcatDataset([monet_test_subset, non_monet_dataset])

    # Create DataLoader
    combined_test_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

    return combined_test_loader

def evaluate_model(model, test_loader, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to_device(device)
    model.model.eval()

    all_predictions = []
    all_labels = []
    prob_monet = []
    prob_non_monet = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model.model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probabilities > threshold).astype(int)  # Convert probabilities to binary predictions
            all_predictions.extend(predicted)
            all_labels.extend(labels.numpy())
            prob_monet.extend(probabilities[:, 0])
            prob_non_monet.extend(1 - probabilities[:, 0])

    # Calculate confusion matrix with specified labels
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'prob_monet': prob_monet,
        'prob_non_monet': prob_non_monet
    }

def save_results_to_csv(results, filepath):
    headers = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix']
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(results)

if __name__ == "__main__":
    from modelTrainingOneClass import MonetOneClassClassifier

    # Load the combined test data
    combined_test_loader = load_combined_test_data(
        monet_dir=config.MONET_DATA_DIR,
        non_monet_dir=config.NON_MONET_TEST_DIR,  # Use the new directory for non-Monet test data
        batch_size=config.BATCH_SIZE,
        test_indices_file='test_indices.json'
    )

    # Initialize the model
    model = MonetOneClassClassifier()
    model.load_model(config.MODEL_PATH)

    # Evaluate the model on the combined test set with adjustable threshold
    combined_metrics = evaluate_model(model, combined_test_loader, threshold=0.5)
    print("\nCombined Test Set Performance:")
    print(f"Accuracy: {combined_metrics['accuracy']:.4f}")
    print(f"Precision: {combined_metrics['precision']:.4f}")
    print(f"Recall: {combined_metrics['recall']:.4f}")
    print(f"F1 Score: {combined_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(combined_metrics['confusion_matrix'])

    # Save results to CSV
    results = [
        combined_metrics['accuracy'],
        combined_metrics['precision'],
        combined_metrics['recall'],
        combined_metrics['f1'],
        combined_metrics['confusion_matrix'].tolist()  # Convert numpy array to list for CSV
    ]
    save_results_to_csv(results, config.F_SCORE_RESULTS_PATH)