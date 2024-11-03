import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import config
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
import csv

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label = label  # 0 for non-Monet
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

def load_ai_test_data(non_monet_dir, batch_size):
    """
    Load the non-Monet test dataset.

    Parameters:
    - non_monet_dir (str): Path to the non-Monet dataset directory.
    - batch_size (int): Number of samples per batch.

    Returns:
    - ai_test_loader (DataLoader): DataLoader for the non-Monet test set.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.RandomRotation(30),  # Randomly rotate the image
        transforms.RandomResizedCrop(224),  # Randomly crop and resize the image
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize
    ])

    # Load the non-Monet dataset using the custom dataset class
    non_monet_dataset = CustomImageDataset(image_dir=non_monet_dir, label=0, transform=transform)

    # Print dataset length for debugging
    print(f"Non-Monet dataset length: {len(non_monet_dataset)}")

    # Create DataLoader
    ai_test_loader = DataLoader(non_monet_dataset, batch_size=batch_size, shuffle=False)

    return ai_test_loader

def evaluate_model(model, test_loader, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to_device(device)
    model.model.eval()

    all_predictions = []
    all_labels = []
    prob_non_monet = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model.model(images)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predicted = (probabilities > threshold).astype(int)  # Convert probabilities to binary predictions
            all_predictions.extend(predicted.flatten())
            all_labels.extend(labels.numpy())
            prob_non_monet.extend(probabilities[:, 0])

    # Print number of samples evaluated
    print(f"Number of samples evaluated: {len(all_labels)}")

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
        'prob_non_monet': prob_non_monet
    }

def save_results_to_csv(results, filepath):
    headers = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Confusion Matrix']
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerow(results)

if __name__ == "__main__":
    from modelTraining import MonetNonMonetClassifier

    # Load the AI test data
    ai_test_loader = load_ai_test_data(
        non_monet_dir=config.NON_MONET_TEST_DIR,
        batch_size=config.BATCH_SIZE
    )

    # Initialize the model
    model = MonetNonMonetClassifier()
    model.load_model(config.MODEL_PATH)

    # Evaluate the model on the AI test set with adjustable threshold
    ai_metrics = evaluate_model(model, ai_test_loader, threshold=0.5)
    print("\nAI Test Set Performance:")
    print(f"Accuracy: {ai_metrics['accuracy']:.4f}")
    print(f"Precision: {ai_metrics['precision']:.4f}")
    print(f"Recall: {ai_metrics['recall']:.4f}")
    print(f"F1 Score: {ai_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(ai_metrics['confusion_matrix'])

    # Save results to CSV
    results = [
        ai_metrics['accuracy'],
        ai_metrics['precision'],
        ai_metrics['recall'],
        ai_metrics['f1'],
        ai_metrics['confusion_matrix'].tolist()  # Convert numpy array to list for CSV
    ]
    save_results_to_csv(results, config.F_SCORE_RESULTS_PATH)