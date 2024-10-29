import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import json
import torch
import config
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

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
    monet_dataset = CustomImageDataset(image_dir=monet_dir, label=1, transform=transform)

    # Load the test indices from the JSON file in the specified directory
    test_indices_path = os.path.join(config.TEST_DATA_DIR, test_indices_file)
    with open(test_indices_path, 'r') as f:
        test_indices = json.load(f)

    # Create a subset for the Monet test dataset using the loaded indices
    monet_test_subset = Subset(monet_dataset, test_indices)

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, label=0, transform=transform)

    # Create DataLoaders
    monet_test_loader = DataLoader(monet_test_subset, batch_size=batch_size, shuffle=False)
    non_monet_test_loader = DataLoader(non_monet_dataset, batch_size=batch_size, shuffle=False)

    return monet_test_loader, non_monet_test_loader

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=1)
    cm = confusion_matrix(all_labels, all_predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    from modelTrainingOneClass import MonetOneClassClassifier

    # Load the test data
    monet_test_loader, non_monet_test_loader = load_test_data(
        monet_dir=config.MONET_DATA_DIR,
        non_monet_dir=config.NON_MONET_DATA_DIR,
        batch_size=config.BATCH_SIZE
    )

    # Initialize the model
    model = MonetOneClassClassifier()
    model.load_state_dict(torch.load(config.MODEL_PATH))

    # Evaluate the model on Monet test set
    monet_metrics = evaluate_model(model, monet_test_loader)
    print("\nMonet Test Set Performance:")
    print(f"Accuracy: {monet_metrics['accuracy']:.4f}")
    print(f"Precision: {monet_metrics['precision']:.4f}")
    print(f"Recall: {monet_metrics['recall']:.4f}")
    print(f"F1 Score: {monet_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(monet_metrics['confusion_matrix'])

    # Evaluate the model on non-Monet test set
    non_monet_metrics = evaluate_model(model, non_monet_test_loader)
    print("\nNon-Monet Test Set Performance:")
    print(f"Accuracy: {non_monet_metrics['accuracy']:.4f}")
    print(f"Precision: {non_monet_metrics['precision']:.4f}")
    print(f"Recall: {non_monet_metrics['recall']:.4f}")
    print(f"F1 Score: {non_monet_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(non_monet_metrics['confusion_matrix'])